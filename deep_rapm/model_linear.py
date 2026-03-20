"""
model_linear.py — Direct neural RAPM baseline.

mu = bias + sum(off_rapm[i] for i on offense)
           + sum(def_rapm[j] for j on defense)

No head projection — gradients flow directly to the embeddings, exactly
like ridge RAPM solved with gradient descent instead of normal equations.

Separate scalar embeddings for offense and defense:
  off_rapm[i] > 0  → player i helps his team score
  def_rapm[j] < 0  → player j suppresses opponent scoring (good defender)

The bias term learns the league-average points per possession.
An optional gamestate linear term absorbs score-diff effects.

All embeddings are initialized to zero so the model starts at the
global-mean solution; the optimizer then moves individual players away
from zero as the data demands.  L2 regularization (embedding_reg) is the
direct analog of the ridge penalty in traditional RAPM.

Keeps the same external API as DeepRAPM:
  forward(...) -> mu  (B,)
  compute_loss(...)   -> scalar
  embedding_penalty() -> scalar
  embed_parameters()  -> list[Parameter]   (for optimizer param groups)
  center_embeddings() -> None              (no-op; L2 handles identifiability)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRAPM(nn.Module):
    """
    Direct RAPM formulation trained with gradient descent.

    Parameters
    ----------
    num_players   : Vocabulary size (including index 0 = UNKNOWN).
    gamestate_dim : Dimension of the gamestate feature (0 = unused).
    embedding_reg : L2 penalty on off/def embeddings — the ridge analog.

    All other keyword arguments are accepted but ignored so that TrainConfig
    can be passed through unchanged.
    """

    def __init__(
        self,
        num_players:   int,
        gamestate_dim: int   = 0,
        embedding_reg: float = 1e-4,
        # Accepted for API compatibility with DeepRAPM; ignored here
        d:           int   = 64,
        num_heads:   int   = 4,
        num_layers:  int   = 2,
        head_hidden: int   = 128,
        ffn_ratio:   int   = 4,
        dropout:     float = 0.1,
        loss:        str   = "mse",
        **kwargs,
    ):
        super().__init__()
        self.embedding_reg = embedding_reg
        self.loss_type     = "mse"

        # Scalar RAPM embeddings — one value per player per role
        self.off_embed = nn.Embedding(num_players, 1)
        self.def_embed = nn.Embedding(num_players, 1)

        # Global mean (intercept)
        self.bias = nn.Parameter(torch.zeros(1))

        # Optional linear gamestate effect (e.g. score_diff influence)
        self.gs_proj = nn.Linear(gamestate_dim, 1, bias=False) if gamestate_dim > 0 else None

        # Zero init: start at the global-mean solution, no random saddle
        nn.init.zeros_(self.off_embed.weight)
        nn.init.zeros_(self.def_embed.weight)

    # ------------------------------------------------------------------
    # Shared interface with DeepRAPM (used by train.py)
    # ------------------------------------------------------------------

    def embed_parameters(self) -> list:
        """Embedding parameters for the no-weight-decay optimizer group."""
        return list(self.off_embed.parameters()) + list(self.def_embed.parameters())

    @torch.no_grad()
    def center_embeddings(self) -> None:
        """No-op: L2 regularization provides identifiability for this model."""
        pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        offense_ids: torch.Tensor,            # (B, 5)
        defense_ids: torch.Tensor,            # (B, 5)
        offense_pos: torch.Tensor,            # (B, 5) — unused, API compat
        defense_pos: torch.Tensor,            # (B, 5) — unused, API compat
        gamestate:   Optional[torch.Tensor] = None,   # (B, gamestate_dim)
        offense_features=None,               # ignored — API compat with CrossRAPM
        defense_features=None,
    ) -> torch.Tensor:
        """Returns mu of shape (B,)."""
        off_val = self.off_embed(offense_ids).sum(dim=1).squeeze(-1)  # (B,)
        def_val = self.def_embed(defense_ids).sum(dim=1).squeeze(-1)  # (B,)

        mu = self.bias.squeeze() + off_val + def_val

        if self.gs_proj is not None and gamestate is not None:
            mu = mu + self.gs_proj(gamestate).squeeze(-1)

        return mu

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def embedding_penalty(self) -> torch.Tensor:
        """L2 penalty on both offense and defense embeddings."""
        return self.embedding_reg * (
            self.off_embed.weight.pow(2).sum()
            + self.def_embed.weight.pow(2).sum()
        )

    def compute_loss(
        self,
        offense_ids: torch.Tensor,
        defense_ids: torch.Tensor,
        offense_pos: torch.Tensor,
        defense_pos: torch.Tensor,
        targets:     torch.Tensor,            # (B,) float
        gamestate:   Optional[torch.Tensor] = None,
        offense_features=None,               # ignored — API compat with CrossRAPM
        defense_features=None,
    ) -> torch.Tensor:
        """MSE + embedding regularization."""
        mu = self(offense_ids, defense_ids, offense_pos, defense_pos, gamestate)
        return F.mse_loss(mu, targets) + self.embedding_penalty()
