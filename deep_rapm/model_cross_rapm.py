"""
model_cross_rapm.py — Cross-Attention RAPM with optional player-feature enrichment.

Why the plain cross-attention plateau
--------------------------------------
Without player features all five offensive embeddings start at nearly
identical small-random values, so the attention weights are approximately
uniform → the model reduces to mean-pooling of the original embeddings,
which is identical to LinearRAPM.  Metadata breaks this symmetry from
epoch 1: a PG and a C carry different feature vectors, so the attention
can immediately distinguish guard-vs-centre matchups.

Architecture
------------
1. Enrich player embeddings with observable features:
       E_i^o = off_embed[i] + meta_proj_o(f_i)    ← additive, separate O/D projections
       E_j^d = def_embed[j] + meta_proj_d(f_j)
   where f_i ∈ ℝ^feature_dim is a fixed feature vector (e.g. one-hot position).
   If no features are provided the enrichment step is skipped and the model
   falls back to pure learned embeddings.

2. Cross-attention (offense ↔ defense):
       H_o = LayerNorm(E_o + CrossAttn(Q=E_o, K=E_d, V=E_d))   (B, 5, d)
       H_d = LayerNorm(E_d + CrossAttn(Q=E_d, K=E_o, V=E_o))   (B, 5, d)

3. Mean pool each side:
       h_o = mean(H_o),  h_d = mean(H_d)                        (B, d)

4. Linear prediction head:
       ŷ = head([h_o ; h_d ; g]) + bias

Current player features (from players.parquet)
-----------------------------------------------
    f_i = one-hot(position_idx) ∈ {0,1}^5   (PG, SG, SF, PF, C)

Additional features (height, weight, age, minutes) can be appended later
without changing the architecture — just extend the feature tensor.

Warm-start from analytical RAPM
---------------------------------
off_embed[idx, 0] ← orapm  / 100
def_embed[idx, 0] ← −drapm / 100
bias              ← intercept
The feature projection is initialised to zero so the warm-started RAPM
values dominate at epoch 0.  The attention then learns to use positional
structure to improve on the additive baseline.

Parameter count (d=32, 1500 players, position features, gamestate_dim=1):
  Embeddings      : 2 × 1500 × 32 = 96 000
  Meta projections: 2 × 5 × 32   =    320
  Cross-attn (×2) : 2 × 4 × 32²  ≈  8 192
  Head + bias     : 65 + 1        =     66
  Total           : ~104 k
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossRAPM(nn.Module):
    """
    RAPM with cross-attention between offensive and defensive lineups,
    enriched by fixed per-player feature vectors.

    Parameters
    ----------
    num_players   : Vocabulary size (including index 0 = UNKNOWN).
    d             : Embedding / attention dimension.
    num_heads     : Number of attention heads (must divide d).
    dropout       : Dropout inside attention projections.
    gamestate_dim : Dimension of the per-possession gamestate vector (0 = none).
    embedding_reg : L2 penalty coefficient on off/def embedding tables.
    player_features : FloatTensor of shape (num_players, feature_dim) — fixed
                      per-player feature vectors (e.g. one-hot position).
                      Pass None to disable feature enrichment.

    Other keyword arguments are accepted and silently ignored so that
    TrainConfig can be forwarded unchanged (same convention as LinearRAPM).
    """

    def __init__(
        self,
        num_players:     int,
        d:               int                      = 32,
        num_heads:       int                       = 4,
        dropout:         float                     = 0.0,
        gamestate_dim:   int                       = 1,
        embedding_reg:   float                     = 1e-4,
        player_features: Optional[torch.Tensor]   = None,
        **kwargs,
    ):
        super().__init__()
        if d % num_heads != 0:
            raise ValueError(f"d ({d}) must be divisible by num_heads ({num_heads})")

        self.d             = d
        self.embedding_reg = embedding_reg

        # ── Player embedding tables (separate for offense and defense) ─────────
        self.off_embed = nn.Embedding(num_players, d)
        self.def_embed = nn.Embedding(num_players, d)

        # ── Optional player-feature enrichment ───────────────────────────────
        if player_features is not None:
            feature_dim = player_features.shape[1]
            # Register as a non-trainable buffer so it moves with the model
            # (to GPU etc.) but is not updated by the optimizer.
            self.register_buffer("player_features", player_features.float())
            # Separate projections for offense and defense: a PG matters
            # differently when attacking vs. when guarding.
            self.meta_proj_o = nn.Linear(feature_dim, d, bias=False)
            self.meta_proj_d = nn.Linear(feature_dim, d, bias=False)
        else:
            self.player_features = None
            self.meta_proj_o     = None
            self.meta_proj_d     = None

        # ── Cross-attention ───────────────────────────────────────────────────
        # Two separate modules — the O→D and D→O directions are asymmetric
        # (different embedding spaces, different learned structure).
        self.cross_attn_od = nn.MultiheadAttention(
            d, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_do = nn.MultiheadAttention(
            d, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_o = nn.LayerNorm(d)
        self.norm_d = nn.LayerNorm(d)

        # ── Prediction head ───────────────────────────────────────────────────
        head_in = 2 * d + gamestate_dim
        self.head = nn.Linear(head_in, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        # Small random init for embeddings — first dim overwritten by warm-start.
        nn.init.normal_(self.off_embed.weight, std=0.01)
        nn.init.normal_(self.def_embed.weight, std=0.01)
        # Zero-init meta projections so that at epoch 0 the feature signal is
        # zero and the RAPM warm-start values (dim 0 of embeddings) dominate.
        if self.meta_proj_o is not None:
            nn.init.zeros_(self.meta_proj_o.weight)
            nn.init.zeros_(self.meta_proj_d.weight)
        # Zero-init head: model predicts ~bias at epoch 0.
        nn.init.zeros_(self.head.weight)

    # ── Shared interface with LinearRAPM / DeepRAPM ───────────────────────────

    def embed_parameters(self) -> list:
        """Embedding parameters — no-weight-decay optimizer group."""
        return list(self.off_embed.parameters()) + list(self.def_embed.parameters())

    @torch.no_grad()
    def center_embeddings(self) -> None:
        """Subtract per-dimension mean from each table to keep values interpretable."""
        self.off_embed.weight.sub_(self.off_embed.weight.mean(dim=0, keepdim=True))
        self.def_embed.weight.sub_(self.def_embed.weight.mean(dim=0, keepdim=True))

    # ── Forward ───────────────────────────────────────────────────────────────

    def _enrich(
        self,
        ids:        torch.Tensor,            # (B, 5)
        embed_tbl:  nn.Embedding,
        meta_proj:  Optional[nn.Linear],
        per_sample_feats: Optional[torch.Tensor] = None,  # (B, 5, feature_dim)
    ) -> torch.Tensor:
        """
        Look up embeddings and add the feature projection.

        Feature source priority:
          1. per_sample_feats — EWMA features computed per possession (date-varying)
          2. self.player_features buffer — static features (position only)
          3. No enrichment if neither is available
        """
        E = embed_tbl(ids)   # (B, 5, d)
        if meta_proj is None:
            return E
        if per_sample_feats is not None:
            E = E + meta_proj(per_sample_feats)               # date-varying EWMA features
        elif self.player_features is not None:
            E = E + meta_proj(self.player_features[ids])      # static buffer fallback
        return E

    def forward(
        self,
        offense_ids:      torch.Tensor,            # (B, 5)
        defense_ids:      torch.Tensor,            # (B, 5)
        offense_pos:      torch.Tensor,            # (B, 5) — unused, API compat
        defense_pos:      torch.Tensor,            # (B, 5) — unused, API compat
        gamestate:        Optional[torch.Tensor] = None,  # (B, gamestate_dim)
        offense_features: Optional[torch.Tensor] = None,  # (B, 5, feature_dim)
        defense_features: Optional[torch.Tensor] = None,  # (B, 5, feature_dim)
    ) -> torch.Tensor:
        """Returns ŷ of shape (B,)."""
        # Step 1: enrich embeddings (date-varying EWMA features if provided)
        E_o = self._enrich(offense_ids, self.off_embed, self.meta_proj_o, offense_features)
        E_d = self._enrich(defense_ids, self.def_embed, self.meta_proj_d, defense_features)

        # Step 2: cross-attention — each side attends to the other
        H_o, _ = self.cross_attn_od(E_o, E_d, E_d)   # offense queries defense
        H_d, _ = self.cross_attn_do(E_d, E_o, E_o)   # defense queries offense

        # Step 3: residual + layer norm
        H_o = self.norm_o(E_o + H_o)   # (B, 5, d)
        H_d = self.norm_d(E_d + H_d)   # (B, 5, d)

        # Step 4: permutation-invariant mean pool
        h_o = H_o.mean(dim=1)   # (B, d)
        h_d = H_d.mean(dim=1)   # (B, d)

        # Step 5: predict
        z = torch.cat([h_o, h_d], dim=-1)   # (B, 2d)
        if gamestate is not None:
            z = torch.cat([z, gamestate], dim=-1)

        return self.head(z).squeeze(-1) + self.bias.squeeze()

    # ── Loss ──────────────────────────────────────────────────────────────────

    def embedding_penalty(self) -> torch.Tensor:
        return self.embedding_reg * (
            self.off_embed.weight.pow(2).sum()
            + self.def_embed.weight.pow(2).sum()
        )

    def compute_loss(
        self,
        offense_ids:      torch.Tensor,
        defense_ids:      torch.Tensor,
        offense_pos:      torch.Tensor,
        defense_pos:      torch.Tensor,
        targets:          torch.Tensor,
        gamestate:        Optional[torch.Tensor] = None,
        offense_features: Optional[torch.Tensor] = None,
        defense_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mu = self(offense_ids, defense_ids, offense_pos, defense_pos,
                  gamestate, offense_features, defense_features)
        return F.mse_loss(mu, targets) + self.embedding_penalty()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, P = 32, 500

    # Simulate one-hot position features (5 positions)
    pos_feats = torch.zeros(P, 5)
    pos_feats[torch.arange(P), torch.randint(0, 5, (P,))] = 1.0

    model = CrossRAPM(
        num_players=P, d=32, num_heads=4, gamestate_dim=1,
        player_features=pos_feats,
    )
    off_ids = torch.randint(0, P, (B, 5))
    def_ids = torch.randint(0, P, (B, 5))
    pos     = torch.randint(0, 5, (B, 5))
    gs      = torch.randn(B, 1)
    targets = torch.randn(B)

    mu   = model(off_ids, def_ids, pos, pos, gs)
    loss = model.compute_loss(off_ids, def_ids, pos, pos, targets, gs)
    print(f"mu shape : {mu.shape}")
    print(f"loss     : {loss.item():.4f}")
    print(f"params   : {sum(p.numel() for p in model.parameters()):,}")

    # Permutation invariance
    model.eval()
    with torch.no_grad():
        po = torch.randperm(5)
        pd = torch.randperm(5)
        mu1 = model(off_ids, def_ids, pos, pos, gs)
        mu2 = model(off_ids[:, po], def_ids[:, pd], pos[:, po], pos[:, pd], gs)
    print(f"Perm-invariance max|diff| : {(mu1 - mu2).abs().max().item():.2e}")

    # Without features (fallback path)
    model_nf = CrossRAPM(num_players=P, d=32, num_heads=4, gamestate_dim=1)
    mu_nf = model_nf(off_ids, def_ids, pos, pos, gs)
    print(f"No-feature path shape    : {mu_nf.shape}")
