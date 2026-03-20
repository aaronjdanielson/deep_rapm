"""
deep_rapm_model.py

Deep RAPM: A Set Transformer for OnCourt Contribution.
Implements the architecture from the paper exactly (Section 4).

Architecture summary
--------------------
For each possession with 5 offensive and 5 defensive players:

  1. PlayerEmbed(p) + RoleEmbed(r) -> MLP projection -> z_i^(0) in R^d
     Role: 0 = offense, 1 = defense
     Position: s in {0,...,4} (PG=0, SG=1, SF=2, PF=3, C=4)

  2. Role-position index: c_i = r_i * 5 + s_i in {0,...,9}
     Defines 10 structural categories (5 offensive positions, 5 defensive)

  3. L transformer layers with biased self-attention:
       alpha_{ij}^{(h)} = (q_i . k_j) / sqrt(d_h) + B[h, c_i, c_j]
     B in R^{H x 10 x 10} is a learned bias over role-position pairs

  4. Mean pool over all 10 player tokens -> z_lineup in R^d

  5. Prediction head: MLP([z_lineup; g]) -> outcome
     CE mode:  logits (B, num_classes), cross-entropy over 0--4 point outcomes
     NLL mode: (mu, log_sigma^2), Gaussian negative log-likelihood

  6. Embedding L2 regularization on PlayerEmbed (the ridge analog)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer MLP with ReLU: in_dim -> hidden_dim -> out_dim."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RolePositionBiasedAttention(nn.Module):
    """
    Multi-head self-attention with a learned role-position bias tensor.

    For head h and players i (query), j (key):
        alpha_{ij}^{(h)} = (q_i . k_j) / sqrt(d_h) + B[h, c_i, c_j]

    B in R^{H x 10 x 10} is indexed by the 10 role-position categories.
    This adds a structural prior over matchup types (OO, OD, DO, DD) that
    is learned from data rather than hand-specified.
    """

    def __init__(
        self,
        d: int,
        num_heads: int,
        d_prime: Optional[int] = None,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        d_prime = d_prime or d
        if d_prime % num_heads != 0:
            raise ValueError(f"d_prime ({d_prime}) must be divisible by num_heads ({num_heads})")

        self.H      = num_heads
        self.d_h    = d_prime // num_heads
        self.d_prime = d_prime

        self.W_Q = nn.Linear(d, d_prime, bias=False)
        self.W_K = nn.Linear(d, d_prime, bias=False)
        self.W_V = nn.Linear(d, d_prime, bias=False)
        self.W_O = nn.Linear(d_prime, d, bias=False)

        self.attn_drop = nn.Dropout(attn_dropout)

        # Learned role-position bias: B[h, c_i, c_j]
        # 10 x 10 = all pairs of role-position categories
        self.B = nn.Parameter(torch.zeros(num_heads, 10, 10))

    def _gather_bias(self, role_pos_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute bias[b, h, i, j] = B[h, c_i[b, i], c_j[b, j]] for all
        batch elements, heads, and player pairs.

        role_pos_ids : (B, N)   -- c_i values in {0,...,9}
        Returns      : (B, H, N, N)
        """
        B_size, N = role_pos_ids.shape
        H = self.H

        # Expand B along batch: (B, H, 10, 10)
        B_exp = self.B.unsqueeze(0).expand(B_size, H, 10, 10)

        # --- Step 1: gather rows indexed by query role-pos (ci) ---
        # ci_exp : (B, H, N, 10) — query index broadcast across all key columns
        ci_exp  = role_pos_ids.unsqueeze(1).unsqueeze(-1).expand(B_size, H, N, 10)
        B_rows  = B_exp.gather(2, ci_exp)
        # B_rows[b, h, i, k] = B[h, ci[b, i], k]  for k in {0,...,9}

        # --- Step 2: gather cols indexed by key role-pos (cj) ---
        # cj_exp : (B, H, N, N) — key index broadcast across all query rows
        cj_exp = role_pos_ids.unsqueeze(1).unsqueeze(2).expand(B_size, H, N, N)
        bias   = B_rows.gather(3, cj_exp)
        # bias[b, h, i, j] = B[h, ci[b, i], cj[b, j]]  ✓

        return bias

    def forward(self, Z: torch.Tensor, role_pos_ids: torch.Tensor) -> torch.Tensor:
        """
        Z            : (B, N, d)   stacked player tokens, N = 10
        role_pos_ids : (B, N)      role-position category indices c_i in {0,...,9}
        Returns      : (B, N, d)
        """
        B_size, N, _ = Z.shape
        H, d_h = self.H, self.d_h

        # Project to queries, keys, values and reshape into heads
        Q = self.W_Q(Z).view(B_size, N, H, d_h).transpose(1, 2)  # (B, H, N, d_h)
        K = self.W_K(Z).view(B_size, N, H, d_h).transpose(1, 2)
        V = self.W_V(Z).view(B_size, N, H, d_h).transpose(1, 2)

        # Scaled dot-product scores: (B, H, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)

        # Inject role-position bias
        scores = scores + self._gather_bias(role_pos_ids)

        # Row-wise softmax over keys, then weighted sum of values
        attn = self.attn_drop(F.softmax(scores, dim=-1))       # (B, H, N, N)
        out  = torch.matmul(attn, V)                            # (B, H, N, d_h)

        # Concatenate heads and project back to d
        out = out.transpose(1, 2).contiguous().view(B_size, N, self.d_prime)
        return self.W_O(out)                                    # (B, N, d)


class TransformerLayer(nn.Module):
    """
    Pre-norm transformer layer (Section 4.5):
        Z1 = LayerNorm(Z)
        Z2 = Z  + MultiHead(Z1)
        Z3 = LayerNorm(Z2)
        Z4 = Z2 + FFN(Z3)
    """

    def __init__(
        self,
        d: int,
        num_heads: int,
        d_prime: Optional[int] = None,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = RolePositionBiasedAttention(d, num_heads, d_prime, attn_dropout=0.0)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(
            nn.Linear(d, ffn_ratio * d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * d, d),
            nn.Dropout(dropout),
        )

    def forward(self, Z: torch.Tensor, role_pos_ids: torch.Tensor) -> torch.Tensor:
        Z = Z + self.attn(self.norm1(Z), role_pos_ids)
        Z = Z + self.ffn(self.norm2(Z))
        return Z


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DeepRAPM(nn.Module):
    """
    Deep RAPM: Set Transformer for possession-level player contribution.

    Parameters
    ----------
    num_players   : Total number of players in the dataset (N).
    d             : Model (embedding) dimension after projection.
    num_heads     : Number of attention heads (H).
    num_layers    : Number of transformer layers (L).
    gamestate_dim : Dimension of the gamestate feature vector (p). 0 = no gamestate.
    d_prime       : Internal attention dimension. Defaults to d.
    head_hidden   : Hidden dim of the prediction head MLP.
    ffn_ratio     : FFN hidden = ffn_ratio * d (standard: 4).
    dropout       : Dropout rate in FFN and prediction head.
    loss          : 'ce' for cross-entropy (ordinal scoring) or 'nll' for Gaussian NLL.
    num_classes   : Number of CE classes (default 5: 0,1,2,3,4 points per possession).
    embedding_reg : L2 penalty strength on PlayerEmbed weights (lambda). Ridge analog.
    """

    def __init__(
        self,
        num_players:   int,
        d:             int            = 64,
        num_heads:     int            = 4,
        num_layers:    int            = 2,
        gamestate_dim: int            = 0,
        d_prime:       Optional[int]  = None,
        head_hidden:   int            = 128,
        ffn_ratio:     int            = 4,
        dropout:       float          = 0.1,
        loss:          str            = 'ce',
        num_classes:   int            = 5,
        embedding_reg: float          = 1e-4,
    ):
        super().__init__()
        if loss not in ('ce', 'nll', 'mse'):
            raise ValueError("loss must be 'ce', 'nll', or 'mse'")

        self.d             = d
        self.num_layers    = num_layers
        self.loss_type     = loss
        self.embedding_reg = embedding_reg

        # --- Embedding tables ---
        self.player_embed = nn.Embedding(num_players, d)
        self.role_embed   = nn.Embedding(2, d)   # 0 = offense, 1 = defense

        # --- Projection MLP: [PlayerEmbed; RoleEmbed] in R^{2d} -> z^(0) in R^d ---
        self.proj_mlp = MLP(in_dim=2 * d, out_dim=d, hidden_dim=2 * d)

        # --- Transformer stack ---
        self.layers = nn.ModuleList([
            TransformerLayer(d, num_heads, d_prime, ffn_ratio, dropout)
            for _ in range(num_layers)
        ])

        # --- Prediction head ---
        head_in  = d + gamestate_dim
        head_out = num_classes if loss == 'ce' else 1   # CE: class logits; NLL/MSE: scalar mu
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_out),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.player_embed.weight, std=0.01)
        nn.init.zeros_(self.role_embed.weight)
        # B is already initialised to zero (neutral prior)

    # ------------------------------------------------------------------
    # Token construction
    # ------------------------------------------------------------------

    def _build_tokens(
        self,
        offense_ids: torch.Tensor,   # (B, 5)
        defense_ids: torch.Tensor,   # (B, 5)
        offense_pos: torch.Tensor,   # (B, 5)  position in {0,...,4}
        defense_pos: torch.Tensor,   # (B, 5)  position in {0,...,4}
    ):
        """
        Construct the initial token matrix Z^(0) in R^{B x 10 x d} and the
        role-position index vector c in {0,...,9}^{B x 10}.

        Offensive players occupy slots 0--4; defensive players occupy slots 5--9.
        """
        B      = offense_ids.shape[0]
        device = offense_ids.device

        player_ids = torch.cat([offense_ids, defense_ids], dim=1)   # (B, 10)

        role_ids = torch.cat([
            torch.zeros(B, 5, dtype=torch.long, device=device),     # offense = 0
            torch.ones( B, 5, dtype=torch.long, device=device),     # defense = 1
        ], dim=1)                                                    # (B, 10)

        pos_ids      = torch.cat([offense_pos, defense_pos], dim=1) # (B, 10)
        role_pos_ids = role_ids * 5 + pos_ids                       # (B, 10), in {0,...,9}

        e = self.player_embed(player_ids)   # (B, 10, d)
        r = self.role_embed(role_ids)       # (B, 10, d)
        Z = self.proj_mlp(torch.cat([e, r], dim=-1))  # (B, 10, d)

        return Z, role_pos_ids

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        offense_ids: torch.Tensor,
        defense_ids: torch.Tensor,
        offense_pos: torch.Tensor,
        defense_pos: torch.Tensor,
        gamestate:   Optional[torch.Tensor] = None,
        offense_features=None,               # ignored — API compat with CrossRAPM
        defense_features=None,
    ):
        """
        Returns
        -------
        CE mode       : logits of shape (B, num_classes)
        NLL/MSE mode  : mu of shape (B,)
        """
        Z, role_pos_ids = self._build_tokens(offense_ids, defense_ids, offense_pos, defense_pos)

        for layer in self.layers:
            Z = layer(Z, role_pos_ids)

        # Permutation-invariant mean pool over all 10 players
        z_lineup = Z.mean(dim=1)                          # (B, d)

        if gamestate is not None:
            z_lineup = torch.cat([z_lineup, gamestate], dim=-1)

        out = self.head(z_lineup)

        if self.loss_type == 'ce':
            return out           # (B, num_classes)
        else:
            return out[:, 0]    # mu (B,)

    # ------------------------------------------------------------------
    # Shared interface with LinearRAPM (used by train.py)
    # ------------------------------------------------------------------

    def embed_parameters(self) -> list:
        """Embedding parameters for the no-weight-decay optimizer group."""
        return list(self.player_embed.parameters())

    @torch.no_grad()
    def center_embeddings(self) -> None:
        """Subtract embedding mean to pin the translation-invariant direction."""
        mean = self.player_embed.weight.mean(dim=0, keepdim=True)
        self.player_embed.weight.sub_(mean)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def embedding_penalty(self) -> torch.Tensor:
        """
        L2 penalty on player embeddings -- the ridge analog (Proposition 5.4).
        Only PlayerEmbed is regularized; role embeddings and MLP weights are not.
        """
        return self.embedding_reg * self.player_embed.weight.pow(2).sum()

    def compute_loss(
        self,
        offense_ids: torch.Tensor,
        defense_ids: torch.Tensor,
        offense_pos: torch.Tensor,
        defense_pos: torch.Tensor,
        targets:     torch.Tensor,
        gamestate:   Optional[torch.Tensor] = None,
        offense_features=None,               # ignored — API compat with CrossRAPM
        defense_features=None,
    ) -> torch.Tensor:
        """
        Task loss + embedding regularization.

        targets : (B,) long  for CE  (class index 0--4)
                  (B,) float for MSE/NLL (point total as real)
        """
        if self.loss_type == 'ce':
            logits    = self(offense_ids, defense_ids, offense_pos, defense_pos, gamestate)
            task_loss = F.cross_entropy(logits, targets)
        else:
            mu        = self(offense_ids, defense_ids, offense_pos, defense_pos, gamestate)
            task_loss = F.mse_loss(mu, targets)

        return task_loss + self.embedding_penalty()

    # ------------------------------------------------------------------
    # Player-value estimand (Section 7.2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def player_value(
        self,
        player_id:     int,
        ref_player_id: int,
        contexts:      list,   # list of (off_ids, def_ids, off_pos, def_pos, gamestate) tuples
        slot:          int = 0,
    ) -> float:
        """
        Plug-in estimate of V_ell^o: average substitution counterfactual.

        Substitutes player_id and ref_player_id into offensive slot `slot`
        across all evaluation contexts and returns the mean contrast in
        expected points per possession.

        All tensors in contexts should be on the same device as the model.
        Each element is a tuple (off_ids, def_ids, off_pos, def_pos, gamestate)
        with off_ids of shape (1, 5).
        """
        self.eval()
        contrasts = []

        for (off_ids, def_ids, off_pos, def_pos, gs) in contexts:
            off_ell = off_ids.clone(); off_ell[0, slot] = player_id
            off_ref = off_ids.clone(); off_ref[0, slot] = ref_player_id

            if self.loss_type == 'ce':
                logits_ell = self(off_ell, def_ids, off_pos, def_pos, gs)
                logits_ref = self(off_ref, def_ids, off_pos, def_pos, gs)
                classes    = torch.arange(logits_ell.shape[-1], dtype=torch.float,
                                          device=logits_ell.device)
                ep_ell = (F.softmax(logits_ell, -1) * classes).sum(-1)
                ep_ref = (F.softmax(logits_ref, -1) * classes).sum(-1)
                contrasts.append((ep_ell - ep_ref).mean().item())
            else:
                mu_ell, _ = self(off_ell, def_ids, off_pos, def_pos, gs)
                mu_ref, _ = self(off_ref, def_ids, off_pos, def_pos, gs)
                contrasts.append((mu_ell - mu_ref).mean().item())

        return sum(contrasts) / len(contrasts)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)

    B           = 32    # batch size
    N_players   = 500   # roster size
    p_gamestate = 4     # e.g. [score_diff, time_remaining, home, period]

    model = DeepRAPM(
        num_players   = N_players,
        d             = 64,
        num_heads     = 4,
        num_layers    = 2,
        gamestate_dim = p_gamestate,
        head_hidden   = 128,
        loss          = 'ce',
        embedding_reg = 1e-4,
    )

    offense_ids = torch.randint(0, N_players, (B, 5))
    defense_ids = torch.randint(0, N_players, (B, 5))
    offense_pos = torch.randint(0, 5, (B, 5))
    defense_pos = torch.randint(0, 5, (B, 5))
    gamestate   = torch.randn(B, p_gamestate)
    targets     = torch.randint(0, 5, (B,))

    logits = model(offense_ids, defense_ids, offense_pos, defense_pos, gamestate)
    loss   = model.compute_loss(offense_ids, defense_ids, offense_pos, defense_pos,
                                targets, gamestate)

    print(f"Output shape : {logits.shape}")    # (32, 5)
    print(f"Loss         : {loss.item():.4f}")
    print(f"Params       : {sum(p.numel() for p in model.parameters()):,}")

    # Verify permutation invariance (eval mode disables Dropout)
    model.eval()
    with torch.no_grad():
        perm_o = torch.randperm(5)
        perm_d = torch.randperm(5)
        logits_base = model(offense_ids, defense_ids, offense_pos, defense_pos, gamestate)
        logits_perm = model(
            offense_ids[:, perm_o], defense_ids[:, perm_d],
            offense_pos[:, perm_o], defense_pos[:, perm_d],
            gamestate,
        )
    max_diff = (logits_base - logits_perm).abs().max().item()
    print(f"Permutation invariance check (max |diff|): {max_diff:.2e}")
