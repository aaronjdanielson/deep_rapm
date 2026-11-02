"""RAPM with a neural density prior over each player's joint (ORAPM, DRAPM).

The phase-1 prior (deep_rapm.torch_rapm) gives each player one Gaussian per
side with a fixed spread and no link between offense and defense. Here a
network maps a player's as-of-origin features to a mixture of K bivariate
Gaussians over z_i = (ORAPM_i, DRAPM_i) / 100:

    p(z_i | x_i) = Σ_k π_ik(x_i) · N(z_i; μ_ik(x_i), s² P_ik(x_i)⁻¹)

with P_ik a full 2×2 precision in pseudo-possession units (λ = 2000 at
initialisation, as in v1), so each component carries its own offense/defense
correlation. K = 1 is a joint Gaussian prior; K > 1 is a non-Gaussian joint
density (player types, skew, a long tail of stars).

At each origin the coefficients are the posterior mode, found by EM: given
responsibilities r_ik the prior term is quadratic, so the M-step is the same
ridge solve as v1 with a penalty that couples each player's offense and
defense coefficients,

    (G + Λ̃) β = X'W(y − μ) + h,   Λ̃_i = Σ_k r_ik P_ik,   h_i = Σ_k r_ik P_ik μ_ik,

and the E-step recomputes r_ik from the current β. A few iterations are
unrolled per origin, warm-started from the previous origin, and the network
is trained end to end on the forward objective (weighted squared error of
the next window, as in phase 1). The evaluation pass reports the forward
scores, the predictive density from the Laplace posterior (G + Λ̃)⁻¹, and per
player the posterior joint (ORAPM, DRAPM): SDs and their correlation, the
prior mean and the dominant mixture component.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn

from .forward import KINDS_MAIN, WindowScorer, eval_weights, predictive_terms
from .priors import FEATURE_SETS
from .production import RapmConfig
from .rolling import IncrementalGramState
from .torch_rapm import HOME_PENALTY, LAG_FEATURES, Context, FitSpec, _lag_features, _window, prepare


@dataclass
class DensitySpec(FitSpec):
    family: str = "mixture"      # "mixture" (bivariate Gaussian mixture, EM) or "maf" (conditional MAF, Newton)
    components: int = 3
    precision: str = "full"      # mixture: "fixed" (λI per component), "diag" or "full" (learned O/D correlation)
    em_iters: int = 3
    mean_cap: float = 0.08       # mixture: |component mean| ≤ 8 per 100 per side
    base_precision: float = 2000.0
    flow_blocks: int = 4         # maf: stacked MADE transformations, coordinate order swapped between blocks
    ctx_dim: int = 32            # maf: player-context embedding fed to every MADE
    newton_iters: int = 3        # maf: Newton steps per origin (warm-started)
    ref_s2: float = 1.41         # maf: noise variance that sets the initial prior scale √(s²/λ)
    hess_floor: float = 50.0     # maf: minimum eigenvalue of each player's 2×2 prior curvature (penalty units)


class MixturePriorNet(nn.Module):
    """features → mixture logits (K), means (K × 2, z units), precision factors (K × 3)."""

    def __init__(self, n_in: int, hidden: int, K: int, precision: str, mean_cap: float) -> None:
        super().__init__()
        self.K, self.precision, self.mean_cap = K, precision, mean_cap
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.logits = nn.Linear(hidden, K)
        self.means = nn.Linear(hidden, 2 * K)
        self.prec = nn.Linear(hidden, 3 * K)
        for head in (self.logits, self.means, self.prec):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if K > 1:      # break the symmetry between components: spread their starting means
            with torch.no_grad():
                offs = torch.linspace(-0.5, 0.5, K)
                self.means.bias.copy_(torch.stack([offs, offs], 1).reshape(-1))
            nn.init.normal_(self.means.weight, std=1e-3)
            nn.init.normal_(self.logits.weight, std=1e-3)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        n = x.shape[0]
        logit = self.logits(h)
        mu = self.mean_cap * torch.tanh(self.means(h)).view(n, self.K, 2)
        q = self.prec(h).view(n, self.K, 3)
        if self.precision == "fixed":
            q = q * 0.0
        elif self.precision == "diag":
            q = torch.cat([q[..., :2], q[..., 2:] * 0.0], dim=-1)
        return logit, mu, q


def precision_matrices(q: torch.Tensor, lam: float) -> torch.Tensor:
    """(n, K, 3) factors → (n, K, 2, 2) precisions λ·L Lᵀ in z coordinates (ORAPM, DRAPM)."""
    l11 = torch.exp(torch.clamp(q[..., 0], -3, 3))
    l22 = torch.exp(torch.clamp(q[..., 1], -3, 3))
    l21 = torch.clamp(q[..., 2], -3, 3)
    p11, p12, p22 = l11 * l11, l11 * l21, l21 * l21 + l22 * l22
    return lam * torch.stack([torch.stack([p11, p12], -1), torch.stack([p12, p22], -1)], -2)


class DensityModel(nn.Module):
    def __init__(self, spec: DensitySpec, n_features: int, feat_mean: np.ndarray, feat_std: np.ndarray) -> None:
        super().__init__()
        self.spec = spec
        self.net = MixturePriorNet(n_features, spec.hidden, spec.components, spec.precision, spec.mean_cap)
        self.register_buffer("feat_mean", torch.tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_std", torch.tensor(feat_std, dtype=torch.float32))

    def prior(self, feats: torch.Tensor):
        """π (P, K), μ (P, K, 2), P (P, K, 2, 2), all float64, z coordinates."""
        logit, mu, q = self.net((feats - self.feat_mean) / self.feat_std)
        return (torch.softmax(logit.double(), -1), mu.double(),
                precision_matrices(q.double(), self.spec.base_precision), logit.double())

    def responsibilities(self, z: torch.Tensor, s2: float, pi, mu, Pk, logit):
        """r_ik ∝ π_ik N(z_i; m_ik, s² P_ik⁻¹) — the s⁻² in the normalising constants cancels."""
        logpi = torch.log_softmax(logit, -1)
        logdet = torch.logdet(Pk)
        dz = z[:, None, :] - mu
        quad = torch.einsum("pki,pkij,pkj->pk", dz, Pk, dz) / s2
        return torch.softmax(logpi + 0.5 * logdet - 0.5 * quad, -1)

    def neg_log_prior(self, z: torch.Tensor, s2: float, pi, mu, Pk, logit) -> torch.Tensor:
        """−s² Σ_i log q(z_i | x_i), up to a constant free of z (the term the weekly objective adds)."""
        logpi = torch.log_softmax(logit, -1)
        logdet = torch.logdet(Pk)
        dz = z[:, None, :] - mu
        quad = torch.einsum("pki,pkij,pkj->pk", dz, Pk, dz) / s2
        return -s2 * torch.logsumexp(logpi + 0.5 * logdet - 0.5 * quad, -1).sum()

    def prior_curvature(self, z: torch.Tensor, s2: float, pi, mu, Pk, logit):
        """(EM surrogate, true Hessian) of −s² log q at z, per player, (P, 2, 2), z coordinates.

        True Hessian = Σ_k r_k P_k − Cov_r[v_k] / s² with v_k = P_k (z − m_k); the
        second term is the missing-information correction (Louis, 1982) that
        the EM surrogate omits. For K = 1 the two coincide.
        """
        r = self.responsibilities(z, s2, pi, mu, Pk, logit)
        surrogate = torch.einsum("pk,pkij->pij", r, Pk)
        v = torch.einsum("pkij,pkj->pki", Pk, z[:, None, :] - mu)                 # (P, K, 2)
        vbar = torch.einsum("pk,pki->pi", r, v)
        cov = torch.einsum("pk,pki,pkj->pij", r, v, v) - vbar[:, :, None] * vbar[:, None, :]
        return surrogate, surrogate - cov / s2, r

    def map_solve(self, G: torch.Tensor, xty: torch.Tensor, feats: torch.Tensor, beta_init: torch.Tensor,
                  s2: float, P: int):
        """Posterior mode: one coupled ridge solve for K = 1, EM for K > 1.

        Returns β, the EM-surrogate system matrix A (for the next warm start
        and the surrogate covariance) and a dict with the prior pieces.
        """
        pi, mu, Pk, logit = self.prior(feats)
        K = pi.shape[1]
        beta = beta_init
        n_iter = 1 if K == 1 else self.spec.em_iters
        for _ in range(n_iter):
            z = torch.stack([beta[:P], -beta[P:2 * P]], -1)              # (P, 2) current (ORAPM, DRAPM)/100
            r = self.responsibilities(z, s2, pi, mu, Pk, logit) if K > 1 else torch.ones_like(pi)
            lam_z = torch.einsum("pk,pkij->pij", r, Pk)                  # (P, 2, 2)
            h_z = torch.einsum("pk,pkij,pkj->pi", r, Pk, mu)             # (P, 2)
            A = assemble_system(G, lam_z, P)
            rhs = xty + torch.cat([h_z[:, 0], -h_z[:, 1], torch.zeros(G.shape[0] - 2 * P, dtype=G.dtype)])
            beta = torch.linalg.solve(A, rhs)
        return beta, A, {"resp": r, "pi": pi, "mu": mu, "Pk": Pk, "logit": logit, "s2": s2}

    def objective(self, beta: torch.Tensor, G: torch.Tensor, xty: torch.Tensor, info: dict, P: int) -> float:
        """J(β) = ½β'Gβ − β'b + ½λ_h h² − s² Σ log q, up to a β-free constant."""
        z = torch.stack([beta[:P], -beta[P:2 * P]], -1)
        quad = 0.5 * beta @ (G @ beta) - beta @ xty + 0.5 * HOME_PENALTY * (beta[2 * P:] ** 2).sum()
        return float(quad + self.neg_log_prior(z, info["s2"], info["pi"], info["mu"], info["Pk"], info["logit"]))

    @torch.no_grad()
    def covariance_system(self, G: torch.Tensor, beta: torch.Tensor, info: dict, P: int):
        """System matrix whose inverse (× s²) is the Laplace covariance at β: G plus the
        *true* prior Hessian. Falls back to the EM surrogate for a player whose corrected
        2×2 block is not positive definite (returns how many)."""
        z = torch.stack([beta[:P], -beta[P:2 * P]], -1)
        surrogate, true_h, _ = self.prior_curvature(z, info["s2"], info["pi"], info["mu"], info["Pk"], info["logit"])
        tr = true_h[:, 0, 0] + true_h[:, 1, 1]
        det = true_h[:, 0, 0] * true_h[:, 1, 1] - true_h[:, 0, 1] ** 2
        pd_ok = (tr > 0) & (det > 0)
        blocks = torch.where(pd_ok[:, None, None], true_h, surrogate)
        return assemble_system(G, blocks, P), int((~pd_ok).sum())

    @torch.no_grad()
    def summary(self, feats: torch.Tensor, info: dict) -> dict:
        """Per-player prior mean (per 100) and dominant component, for the cache."""
        pi, mu, r = info["pi"].numpy(), info["mu"].numpy(), info["resp"].numpy()
        return {"prior_mean": np.einsum("pk,pki->pi", pi, mu) * 100,
                "component": r.argmax(1), "resp_max": r.max(1)}


def assemble_system(G: torch.Tensor, blocks_z: torch.Tensor, P: int) -> torch.Tensor:
    """G plus per-player 2×2 prior-curvature blocks given in z = (α, −δ) coordinates,
    mapped to coefficient coordinates (the off-diagonal flips sign), plus the home penalty."""
    n_cols = G.shape[0]
    idx = torch.arange(P)
    rows, cols = torch.cat([idx, P + idx]), torch.cat([P + idx, idx])
    diag = torch.cat([blocks_z[:, 0, 0], blocks_z[:, 1, 1],
                      torch.full((n_cols - 2 * P,), HOME_PENALTY, dtype=G.dtype)])
    off = -blocks_z[:, 0, 1]
    return G + torch.diag(diag) + torch.zeros_like(G).index_put((rows, cols), torch.cat([off, off]))


# ---------------------------------------------------------------------------
# Conditional masked autoregressive flow prior
# ---------------------------------------------------------------------------

def _mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, n_out))
    nn.init.zeros_(net[-1].weight)          # every block starts as the identity map
    nn.init.zeros_(net[-1].bias)
    return net


class ConditionalMADE2(nn.Module):
    """MADE for d = 2 with a context vector: (μ₁, α₁) depend only on the context,
    (μ₂, α₂) on the context and u₁ — the masked-network dependency structure,
    written out directly because two coordinates need no mask matrices."""

    def __init__(self, ctx_dim: int, hidden: int) -> None:
        super().__init__()
        self.first = _mlp(ctx_dim, hidden, 2)
        self.second = _mlp(ctx_dim + 1, hidden, 2)

    def forward(self, u: torch.Tensor, c: torch.Tensor):
        p1 = self.first(c)
        p2 = self.second(torch.cat([c, u[:, :1]], 1))
        mu = torch.stack([p1[:, 0], p2[:, 0]], 1)
        alpha = torch.clamp(torch.stack([p1[:, 1], p2[:, 1]], 1), -5, 5)
        return mu, alpha


class MAFBlock(nn.Module):
    def __init__(self, ctx_dim: int, hidden: int, swap: bool) -> None:
        super().__init__()
        self.made = ConditionalMADE2(ctx_dim, hidden)
        self.register_buffer("perm", torch.tensor([1, 0] if swap else [0, 1]))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """density direction x → y, with log|det ∂y/∂x|."""
        u = x[:, self.perm]
        mu, alpha = self.made(u, c)
        y = (u - mu) * torch.exp(-alpha)
        return y[:, self.perm], -alpha.sum(1)            # the swap is its own inverse

    @torch.no_grad()
    def inverse(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        v = y[:, self.perm]
        u = torch.zeros_like(v)
        for i in range(2):                                # autoregressive: coordinates in order
            mu, alpha = self.made(u, c)
            u[:, i] = v[:, i] * torch.exp(alpha[:, i]) + mu[:, i]
        return u[:, self.perm]


class FlowModel(nn.Module):
    """Conditional MAF prior over z = (ORAPM, DRAPM)/100 given player features.

    z is divided by s0 = √(s² / λ) — the current origin's noise scale over the
    base penalty — before the flow, so with identity blocks the prior is
    N(0, s²/λ I) at *every* origin, exactly v1's ridge prior. The posterior
    mode is found by Newton steps on ½β'Gβ − β'X'W(y−μ) + s² Σ_i −log q(z_i | x_i);
    each step solves (G + s² H̃) Δ = gradient with H̃ the per-player 2×2
    Hessian of −log q (curvature floored, so every step is a descent step).
    """

    def __init__(self, spec: DensitySpec, n_features: int, feat_mean: np.ndarray, feat_std: np.ndarray) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = nn.Sequential(nn.Linear(n_features, spec.hidden), nn.GELU(),
                                     nn.Linear(spec.hidden, spec.ctx_dim), nn.GELU())
        self.blocks = nn.ModuleList([MAFBlock(spec.ctx_dim, spec.hidden, swap=(b % 2 == 1))
                                     for b in range(spec.flow_blocks)]).double()   # the solve runs in float64
        self.register_buffer("feat_mean", torch.tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_std", torch.tensor(feat_std, dtype=torch.float32))

    def context(self, feats: torch.Tensor) -> torch.Tensor:
        return self.encoder((feats - self.feat_mean) / self.feat_std).double()

    def _s0(self, s2, n: int) -> torch.Tensor:
        s2 = torch.as_tensor(s2, dtype=torch.float64)
        return torch.sqrt(s2 / self.spec.base_precision).expand(n) if s2.dim() == 0 else torch.sqrt(s2 / self.spec.base_precision)

    def log_prob(self, z: torch.Tensor, c: torch.Tensor, s2) -> torch.Tensor:
        """log q(z | c) with base scale s0 = √(s²/λ); s2 a scalar or one value per row."""
        s0 = self._s0(s2, z.shape[0])
        y = z / s0[:, None]
        logdet = torch.zeros(z.shape[0], dtype=z.dtype)
        for blk in self.blocks:
            y, ld = blk(y, c)
            logdet = logdet + ld
        return -0.5 * (2 * np.log(2 * np.pi) + (y ** 2).sum(1)) + logdet - 2 * torch.log(s0)

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n: int, s2) -> torch.Tensor:
        """n samples of z per player: (n, P, 2)."""
        P = c.shape[0]
        cc = c.repeat(n, 1)
        y = torch.randn(n * P, 2, dtype=c.dtype)
        for blk in reversed(self.blocks):
            y = blk.inverse(y, cc)
        return (y * self._s0(s2, P).repeat(n)[:, None]).view(n, P, 2)

    def map_solve(self, G: torch.Tensor, xty: torch.Tensor, feats: torch.Tensor, beta_init: torch.Tensor,
                  s2: float, P: int):
        c = self.context(feats)
        n_cols = G.shape[0]
        idx = torch.arange(P)
        rows, cols = torch.cat([idx, P + idx]), torch.cat([P + idx, idx])
        home = torch.full((n_cols - 2 * P,), HOME_PENALTY, dtype=G.dtype)
        beta = beta_init
        for _ in range(self.spec.newton_iters):
            z = torch.stack([beta[:P], -beta[P:2 * P]], -1)
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)
            nlq = -self.log_prob(z, c, s2).sum()
            g_z = torch.autograd.grad(nlq, z, create_graph=True)[0]                          # (P, 2)
            H_z = torch.stack([torch.autograd.grad(g_z[:, k].sum(), z, create_graph=True)[0]
                               for k in range(2)], 1)                                         # (P, 2, 2)
            Hs = s2 * 0.5 * (H_z + H_z.transpose(1, 2))
            # Keep each 2×2 block's smallest eigenvalue ≥ hess_floor by shifting its
            # diagonal (Levenberg–Marquardt). Closed form, with a smoothed square root:
            # an eigen-decomposition has no gradient at the isotropic starting point.
            a, b, d = Hs[:, 0, 0], Hs[:, 0, 1], Hs[:, 1, 1]
            half_gap = torch.sqrt(0.25 * (a - d) ** 2 + b ** 2 + 1e-12 * (a + d) ** 2)
            shift = torch.relu(self.spec.hess_floor - (0.5 * (a + d) - half_gap))
            Hs = Hs + shift[:, None, None] * torch.eye(2, dtype=Hs.dtype)
            # z = (α, −δ): flip signs of every term that touches δ
            grad = G @ beta - xty + torch.cat([s2 * g_z[:, 0], -s2 * g_z[:, 1],
                                               HOME_PENALTY * beta[2 * P:]])
            A = (G + torch.diag(torch.cat([Hs[:, 0, 0], Hs[:, 1, 1], home]))
                 + torch.zeros_like(G).index_put((rows, cols), torch.cat([-Hs[:, 0, 1], -Hs[:, 0, 1]])))
            beta = beta - torch.linalg.solve(A, grad)
        return beta, A, {"context": c, "s2": s2}

    def objective(self, beta: torch.Tensor, G: torch.Tensor, xty: torch.Tensor, info: dict, P: int) -> float:
        z = torch.stack([beta[:P], -beta[P:2 * P]], -1)
        quad = 0.5 * beta @ (G @ beta) - beta @ xty + 0.5 * HOME_PENALTY * (beta[2 * P:] ** 2).sum()
        with torch.no_grad():
            return float(quad - info["s2"] * self.log_prob(z, info["context"], info["s2"]).sum())

    def covariance_system(self, G: torch.Tensor, beta: torch.Tensor, info: dict, P: int):
        """G plus the true per-player prior Hessian at β (no curvature floor); surrogate
        fallback for blocks that are not positive definite."""
        z = torch.stack([beta[:P], -beta[P:2 * P]], -1).detach().clone().requires_grad_(True)
        nlq = -info["s2"] * self.log_prob(z, info["context"].detach(), info["s2"]).sum()
        g = torch.autograd.grad(nlq, z, create_graph=True)[0]
        H = torch.stack([torch.autograd.grad(g[:, k].sum(), z, retain_graph=True)[0] for k in range(2)], 1).detach()
        H = 0.5 * (H + H.transpose(1, 2))
        tr, det = H[:, 0, 0] + H[:, 1, 1], H[:, 0, 0] * H[:, 1, 1] - H[:, 0, 1] ** 2
        pd_ok = (tr > 0) & (det > 0)
        floor = torch.eye(2, dtype=H.dtype) * self.spec.hess_floor
        blocks = torch.where(pd_ok[:, None, None], H, floor.expand_as(H))
        return assemble_system(G, blocks, P), int((~pd_ok).sum())

    @torch.no_grad()
    def summary(self, feats: torch.Tensor, info: dict, n: int = 128) -> dict:
        """Per-player prior mean, SDs and O/D correlation (per 100) from n flow samples."""
        s = self.sample(info["context"].detach(), n, info["s2"]).numpy() * 100   # (n, P, 2)
        m = s.mean(0)
        sd = s.std(0)
        cov = ((s - m) ** 2).mean(0)
        corr = ((s[..., 0] - m[:, 0]) * (s[..., 1] - m[:, 1])).mean(0) / np.maximum(sd[:, 0] * sd[:, 1], 1e-12)
        return {"prior_mean": m, "prior_sd": sd, "prior_corr": corr}


def _inverse(A: np.ndarray) -> np.ndarray:
    c = scipy.linalg.cho_factor(A, check_finite=False)
    return scipy.linalg.cho_solve(c, np.eye(len(A)), check_finite=False)


def run_epoch(ctx: Context, model: DensityModel, opt=None, scorer: WindowScorer | None = None,
              cache_rows: list | None = None):
    spec, cfg, P = ctx.spec, ctx.cfg, ctx.n_players
    horizon = pd.Timedelta(days=spec.horizon_days)
    state = IncrementalGramState(ctx.n_cols, cfg.alpha, cfg.half_life_days, n_players=P,
                                 offseason_discount=cfg.offseason_discount)
    history: list[tuple] = []
    sums = {s: np.zeros(2) for s in ("train", "val", "holdout")}
    diagnostics = {"n_origins": 0, "fallback_blocks": 0, "surrogate_origins": 0, "conv_checks": []}
    run_epoch.last_diagnostics = diagnostics
    beta_prev = np.zeros(ctx.n_cols)
    j, n_batch = 0, 0
    if opt is not None:
        opt.zero_grad()
    for t in ctx.origins:
        while j < len(ctx.dates) and ctx.dates[j].date <= t:
            d = ctx.dates[j]
            state.advance_to(d.date)
            state.ingest(d.X, d.y, d.n_off, d.n_def, comp_sigma=ctx.comp_sigma,
                         score_diff=d.score_diff, context_weight=d.context)
            j += 1
        if state.total_w == 0:
            continue
        split = ctx.split_of(t)
        xty_np, mu0, _ = state.centered_rhs()
        s2 = state.noise_variance(beta_prev) if np.any(beta_prev) else max(state.weighted_sum_y2 / state.total_w
                                                                          - mu0 * mu0, 1e-3)
        x = torch.from_numpy(np.concatenate([ctx.feats[t], _lag_features(t, history, P, spec.lag_days)], axis=1))
        grad = opt is not None and split == "train"
        score_window = opt is None or split != "holdout"
        test = _window(ctx.dates, j, t, horizon)
        with torch.set_grad_enabled(grad or model.spec.family == "maf"):   # Newton needs ∇ log q even when scoring
            beta, A, info = model.map_solve(torch.from_numpy(state.G).clone(), torch.from_numpy(xty_np),
                                            x, torch.from_numpy(beta_prev), s2, P)
            if not grad:
                beta, A = beta.detach(), A.detach()
            num = torch.zeros((), dtype=torch.float64)
            den = 0.0
            for d in (test if score_window else []):
                if d.kind not in KINDS_MAIN:
                    continue
                w = torch.from_numpy(eval_weights(d.score_diff))
                num = num + (w * (torch.from_numpy(d.y) - (mu0 + d.Xt @ beta)) ** 2).sum()
                den += float(w.sum())
            if grad and den > 0:
                (num / den / spec.batch_origins).backward()
                n_batch += 1
                if n_batch == spec.batch_origins:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    opt.zero_grad()
                    n_batch = 0
        beta_np = beta.detach().numpy()
        beta_prev = beta_np
        history.append((t, beta_np.copy(), np.diag(state.G).copy()))
        sums[split] += (float(num.detach()), den)
        if scorer is not None or cache_rows is not None:
            # Laplace covariance from the true prior Hessian at the mode, not the
            # EM surrogate (which omits the responsibility-covariance term).
            A_cov, n_fallback = model.covariance_system(torch.from_numpy(state.G), beta.detach(), info, P)
            diagnostics["fallback_blocks"] += n_fallback
            try:
                A_inv = _inverse(A_cov.detach().numpy())
            except np.linalg.LinAlgError:
                A_inv = _inverse(A.detach().numpy())
                diagnostics["surrogate_origins"] += 1
            if diagnostics["n_origins"] % 25 == 0:          # convergence check: more iterations from the mode
                extra = max(spec.em_iters, spec.newton_iters) + 7
                saved = (spec.em_iters, spec.newton_iters)
                spec.em_iters = spec.newton_iters = extra
                with torch.set_grad_enabled(model.spec.family == "maf"):
                    b_more, _, info_more = model.map_solve(torch.from_numpy(state.G).clone(), torch.from_numpy(xty_np),
                                                           x, beta.detach(), s2, P)
                spec.em_iters, spec.newton_iters = saved
                j0 = model.objective(beta.detach(), torch.from_numpy(state.G), torch.from_numpy(xty_np), info, P)
                j1 = model.objective(b_more.detach(), torch.from_numpy(state.G), torch.from_numpy(xty_np), info_more, P)
                diagnostics["conv_checks"].append({"origin": str(t.date()), "max_dbeta_per100":
                                                   float(100 * np.abs(b_more.detach().numpy() - beta_np).max()),
                                                   "rel_obj_change": float((j0 - j1) / max(abs(j0), 1e-12))})
            diagnostics["n_origins"] += 1
            s2_post = state.noise_variance(beta_np)
            if scorer is not None:
                for d in test:
                    pv, mv, bv = predictive_terms(A_inv, s2_post, d)
                    scorer.add(0, t, d, mu0 + d.X @ beta_np, mu0, pred_var=pv, s2=s2_post, margin_var=mv, base_var=bv)
            if cache_rows is not None:
                ar = np.arange(P)
                v_o, v_d, c_od = (s2_post * np.diag(A_inv)[:P], s2_post * np.diag(A_inv)[P:2 * P],
                                  s2_post * A_inv[ar, P + ar])
                corr = -c_od / np.sqrt(np.maximum(v_o * v_d, 1e-30))          # corr(ORAPM, DRAPM), DRAPM = −δ
                summ = model.summary(x, info)
                pm = summ["prior_mean"]
                for pid, idx in zip(ctx.vocab["player_id"].astype(int).values, ctx.vocab["player_idx"].astype(int).values):
                    if state.n_off[idx] < cfg.min_poss and state.n_def[idx] < cfg.min_poss:
                        continue
                    row = {
                        "date": t, "player_id": pid, "player_name": ctx.name_lookup.get(pid, "Unknown"),
                        "orapm": float(beta_np[idx] * 100), "drapm": float(-beta_np[P + idx] * 100),
                        "rapm": float((beta_np[idx] - beta_np[P + idx]) * 100),
                        "sd_orapm": float(100 * np.sqrt(v_o[idx])), "sd_drapm": float(100 * np.sqrt(v_d[idx])),
                        "sd_rapm": float(100 * np.sqrt(max(v_o[idx] + v_d[idx] - 2 * c_od[idx], 0))),
                        "corr_od": float(corr[idx]),
                        "prior_orapm": float(pm[idx, 0]), "prior_drapm": float(pm[idx, 1]),
                        "n_off": int(state.n_off[idx]), "n_def": int(state.n_def[idx]),
                    }
                    if "component" in summ:
                        row.update(component=int(summ["component"][idx]), resp_max=float(summ["resp_max"][idx]))
                    if "prior_sd" in summ:
                        row.update(prior_sd_orapm=float(summ["prior_sd"][idx, 0]),
                                   prior_sd_drapm=float(summ["prior_sd"][idx, 1]),
                                   prior_corr_od=float(summ["prior_corr"][idx]))
                    cache_rows.append(row)
    if opt is not None and n_batch:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad()
    return {s: (v[0] / v[1] if v[1] > 0 else float("nan")) for s, v in sums.items()}


def fit_density(spec: DensitySpec, base_cfg: RapmConfig, out_dir: Path, forward_dir: Path = Path("outputs/forward"),
                until=None, verbose: bool = True) -> dict:
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = prepare(spec, base_cfg, verbose=verbose, until=until)
    n_features = len(FEATURE_SETS[spec.feature_set]) + len(LAG_FEATURES)
    model_cls = FlowModel if spec.family == "maf" else DensityModel
    model = model_cls(spec, n_features, ctx.feat_mean, ctx.feat_std)
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr)
    best_val, best_state, best_epoch, log = float("inf"), None, -1, []
    for epoch in range(spec.epochs):
        te = time.time()
        wmse = run_epoch(ctx, model, opt)
        log.append({"epoch": epoch, "train": wmse["train"], "val": wmse["val"], "seconds": time.time() - te})
        if verbose:
            print(f"  epoch {epoch:2d}  train {wmse['train']:.5f}  val {wmse['val']:.5f}  ({time.time() - te:.0f}s)",
                  flush=True)
        if wmse["val"] < best_val - 1e-7:
            best_val, best_epoch = wmse["val"], epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "spec": asdict(spec), "base": asdict(base_cfg),
                        "features": FEATURE_SETS[spec.feature_set] + LAG_FEATURES, "epoch": epoch},
                       out_dir / "model.pt")
        elif epoch - best_epoch >= spec.patience:
            if verbose:
                print(f"  no validation improvement for {spec.patience} epochs — stopping", flush=True)
            break
    model.load_state_dict(best_state)
    scorer = WindowScorer(spec.name, [(spec.base_precision, spec.base_precision)])
    cache_rows: list[dict] = []
    wmse = run_epoch(ctx, model, None, scorer=scorer, cache_rows=cache_rows)
    poss, margins, origins = scorer.frames()
    forward_dir.mkdir(parents=True, exist_ok=True)
    poss.to_parquet(forward_dir / f"{spec.name}_poss.parquet", index=False)
    margins.to_parquet(forward_dir / f"{spec.name}_margins.parquet", index=False)
    origins.to_parquet(forward_dir / f"{spec.name}_origins.parquet", index=False)
    pd.DataFrame(cache_rows).to_parquet(out_dir / "rolling_cache_density.parquet", index=False)
    summary = {"name": spec.name, "spec": asdict(spec), "best_epoch": best_epoch, "wmse": wmse, "epochs": log,
               "diagnostics": run_epoch.last_diagnostics, "seconds": time.time() - t0}
    (out_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if verbose:
        print(f"[{spec.name}] best epoch {best_epoch}; eval wmse {{train {wmse['train']:.5f}, val {wmse['val']:.5f}, "
              f"holdout {wmse['holdout']:.5f}}}; {time.time() - t0:.0f}s", flush=True)
    return summary
