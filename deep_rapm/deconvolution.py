"""Train density priors by deconvolving season-level evidence about each player.

The forward objective (next week's possessions) is a weak signal for learning a
prior. Here the prior q(z | x) over a player's z = (ORAPM, DRAPM)/100 is fit to
what each *later season* says about him:

Targets. For season s, build the season's normal equations (regular season +
playoffs, non-garbage-time weights as in the harness, no recency decay) and
solve the ridge with the reference prior N(0, s²/λ) on every coefficient
(home court nearly unpenalised). For player i let Σ_i be the 2×2 block of
(G + Λ)⁻¹ for his offense and defense coefficients and β̂_i his posterior
mean. Removing his own prior from his posterior gives his *marginal
likelihood*, with every teammate and opponent integrated out under the
reference prior:

    ℓ_i(c) = −(c'F_i c − 2c'h_i) / (2 s²),   F_i = Σ_i⁻¹ − λI,   h_i = Σ_i⁻¹ β̂_i

(in coefficient coordinates c = (α, δ); stored in z = (α, −δ) coordinates).
Collinearity with teammates is handled exactly; a player with little data
gets F_i ≈ 0 and contributes almost nothing.

Objective. The deconvolution log-likelihood Σ log ∫ e^{ℓ_i(z)} q(z | x_i) dz,
with x_i the player's features at the season's first origin (games before
it only). For a Gaussian mixture prior the integral is closed form; for the
conditional MAF it is importance-sampled with the reference-prior posterior
N((F+λI)⁻¹h, s²(F+λI)⁻¹) as proposal.

Splits: seasons whose start precedes train_until train, then validation
seasons choose the epoch by held-out log-likelihood, later seasons are
reported. The trained model plugs into deep_rapm.density_prior.run_epoch as
the prior of the weekly solve, for the forward harness.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import torch

from .density_prior import DensityModel, DensitySpec, FlowModel, run_epoch
from .forward import KINDS_MAIN, WindowScorer, eval_weights
from .priors import FEATURE_SETS
from .torch_rapm import HOME_PENALTY, LAG_FEATURES, prepare


def build_targets(ctx, lam: float = 2000.0, min_poss: int = 50, verbose: bool = True) -> dict:
    """Per player-season: features at the season's first origin, F (2×2) and h (2) in z coordinates, s²."""
    P, n_cols = ctx.n_players, ctx.n_cols
    seasons = sorted({d.season for d in ctx.dates if d.kind in KINDS_MAIN})
    origins = pd.DatetimeIndex(ctx.origins)
    rows = []
    t0 = time.time()
    for s in seasons:
        ds = [d for d in ctx.dates if d.season == s and d.kind in KINDS_MAIN]
        if not ds:
            continue
        start = min(d.date for d in ds)
        # Features at origin t use games dated strictly before t, so the last
        # origin at or before the first target game keeps every feature date
        # strictly earlier than every target game.
        k = origins.searchsorted(start, side="right") - 1
        if k < 0 or start - origins[k] > pd.Timedelta(days=14):
            continue                                      # season starts before the first origin
        t_feat = origins[k]
        assert t_feat <= start
        G = np.zeros((n_cols, n_cols))
        b_y = np.zeros(n_cols)
        b_1 = np.zeros(n_cols)
        sw = swy = swy2 = 0.0
        n_off = np.zeros(P)
        n_def = np.zeros(P)
        for d in ds:
            w = eval_weights(d.score_diff) * d.context
            Xw = d.X.multiply(np.sqrt(w)[:, None])
            G += (Xw.T @ Xw).toarray()
            b_y += np.asarray(d.X.T @ (w * d.y)).ravel()
            b_1 += np.asarray(d.X.T @ w).ravel()
            sw += w.sum(); swy += (w * d.y).sum(); swy2 += (w * d.y * d.y).sum()
            n_off += d.n_off; n_def += d.n_def
        mu = swy / sw
        xty = b_y - mu * b_1
        pen = np.full(n_cols, lam)
        pen[2 * P:] = HOME_PENALTY
        A = G.copy()
        A.flat[::n_cols + 1] += pen
        Ainv = scipy.linalg.cho_solve(scipy.linalg.cho_factor(A, check_finite=False), np.eye(n_cols),
                                      check_finite=False)
        beta = Ainv @ xty
        s2 = max((swy2 - mu * mu * sw - 2 * beta @ xty + beta @ G @ beta) / sw, 1e-6)
        ar = np.arange(P)
        keep = np.flatnonzero((n_off >= min_poss) | (n_def >= min_poss))
        keep = keep[keep > 0]                               # not the UNKNOWN slot
        S = np.stack([np.stack([Ainv[keep, keep], Ainv[keep, P + keep]], -1),
                      np.stack([Ainv[P + keep, keep], Ainv[P + keep, P + keep]], -1)], -2)   # (n, 2, 2) in c coords
        Sinv = np.linalg.inv(S)
        bc = np.stack([beta[keep], beta[P + keep]], -1)
        F_c = Sinv - lam * np.eye(2)
        h_c = np.einsum("nij,nj->ni", Sinv, bc)
        T = np.diag([1.0, -1.0])
        F_z = T @ F_c @ T                                    # broadcast: (n, 2, 2)
        h_z = h_c * np.array([1.0, -1.0])
        for j, i in enumerate(keep):
            rows.append({"season": s, "t_feat": t_feat, "player_idx": int(i), "n_off": int(n_off[i]),
                         "n_def": int(n_def[i]), "s2": s2, "F00": F_z[j, 0, 0], "F01": F_z[j, 0, 1],
                         "F11": F_z[j, 1, 1], "h0": h_z[j, 0], "h1": h_z[j, 1],
                         "post_orapm": 100 * bc[j, 0], "post_drapm": -100 * bc[j, 1]})
        if verbose:
            print(f"  targets {s}: {len(keep)} players, s² = {s2:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    tg = pd.DataFrame(rows)
    X = np.stack([np.concatenate([ctx.feats[t][i], np.zeros(len(LAG_FEATURES), dtype=np.float32)])
                  for t, i in zip(tg["t_feat"], tg["player_idx"])])
    return {"targets": tg, "X": X}


def build_window_targets(ctx, lam: float = 2000.0, min_poss: int = 20, verbose: bool = True) -> dict:
    """Game-level measurements on the RAPM scale, summed over each weekly window.

    At origin t the reference fit (v1 + home court, penalty λ) gives every
    coefficient. For each possession in (t, t + 7] the residual r = y − μ − xβ̂
    plus the player's own estimate is a one-possession measurement of his
    effect with every other player on the floor held at his estimate. Summed
    over his possessions in the window (all his games that week):

        offense  F_o = Σ w,  h_o = Σ w (r + α̂_i)       defense  F_d = Σ w,  h_d = Σ w (r + δ̂_i)

    which is exactly the product of the per-game Gaussian likelihoods. Stored
    in z = (α, −δ) coordinates with the same natural-parameter convention as
    the season targets, so the deconvolution loss is unchanged.
    """
    import scipy.sparse as sp

    cfg, P, n_cols = ctx.cfg, ctx.n_players, ctx.n_cols
    horizon = pd.Timedelta(days=ctx.spec.horizon_days)
    pen = np.full(n_cols, lam)
    pen[2 * P:] = HOME_PENALTY
    from .rolling import IncrementalGramState
    from .torch_rapm import _window
    state = IncrementalGramState(n_cols, cfg.alpha, cfg.half_life_days, n_players=P,
                                 offseason_discount=cfg.offseason_discount)
    rows, j, t0 = [], 0, time.time()
    for k, t in enumerate(ctx.origins):
        while j < len(ctx.dates) and ctx.dates[j].date <= t:
            d = ctx.dates[j]
            state.advance_to(d.date)
            state.ingest(d.X, d.y, d.n_off, d.n_def, comp_sigma=ctx.comp_sigma,
                         score_diff=d.score_diff, context_weight=d.context)
            j += 1
        if state.total_w == 0:
            continue
        test = [d for d in _window(ctx.dates, j, t, horizon) if d.kind in KINDS_MAIN]
        if not test:
            continue
        beta, mu = state.solve(pen)
        s2 = state.noise_variance(beta)
        F_o = np.zeros(P); F_d = np.zeros(P); H_o = np.zeros(P); H_d = np.zeros(P)
        for d in test:
            w = eval_weights(d.score_diff) * d.context
            r = d.y - mu - d.X @ beta
            Xo, Xd = d.X[:, :P], d.X[:, P:2 * P]
            F_o += np.asarray(Xo.T @ w).ravel()
            F_d += np.asarray(Xd.T @ w).ravel()
            H_o += np.asarray(Xo.T @ (w * r)).ravel()
            H_d += np.asarray(Xd.T @ (w * r)).ravel()
        H_o += beta[:P] * F_o                       # add back each player's own estimate
        H_d += beta[P:2 * P] * F_d
        keep = np.flatnonzero((F_o >= min_poss * 0.3) | (F_d >= min_poss * 0.3))
        keep = keep[keep > 0]
        season = test[0].season
        for i in keep:
            rows.append({"season": season, "t_feat": t, "player_idx": int(i), "n_off": float(F_o[i]),
                         "n_def": float(F_d[i]), "s2": s2, "F00": F_o[i], "F01": 0.0, "F11": F_d[i],
                         "h0": H_o[i], "h1": -H_d[i]})
        if verbose and ((k + 1) % max(1, len(ctx.origins) // 10) == 0):
            print(f"  window targets [{100 * (k + 1) / len(ctx.origins):5.1f}%] {t.date()}  rows {len(rows):,}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    tg = pd.DataFrame(rows)
    X = np.stack([np.concatenate([ctx.feats[t][i], np.zeros(len(LAG_FEATURES), dtype=np.float32)])
                  for t, i in zip(tg["t_feat"], tg["player_idx"])])
    return {"targets": tg, "X": X}


def _tensors(tg: pd.DataFrame, X: np.ndarray, mask: np.ndarray):
    t = tg[mask]
    F = torch.tensor(np.stack([np.stack([t.F00, t.F01], -1), np.stack([t.F01, t.F11], -1)], -2))
    h = torch.tensor(np.stack([t.h0, t.h1], -1))
    s2 = torch.tensor(t.s2.values)
    return torch.tensor(X[mask]), F, h, s2


def loglik_mixture(model: DensityModel, x, F, h, s2) -> torch.Tensor:
    """Exact log ∫ e^{ℓ(z)} q(z|x) dz for a Gaussian-mixture prior (up to a q-free constant)."""
    pi, mu, Pk, logit = model.prior(x)                               # Pk: penalty units, prior cov = s² Pk⁻¹
    Fk, hk, s2k = F[:, None], h[:, None], s2[:, None]
    PF = Pk + Fk
    v = torch.einsum("nkij,nkj->nki", Pk, mu) + hk                   # Pμ + h
    sol = torch.linalg.solve(PF, v.unsqueeze(-1)).squeeze(-1)
    quad = (v * sol).sum(-1) / s2k - torch.einsum("nki,nkij,nkj->nk", mu, Pk, mu) / s2k
    logdet = torch.logdet(PF) - torch.logdet(Pk)                     # log |I + P⁻¹F|
    comp = -0.5 * logdet + 0.5 * quad
    return torch.logsumexp(torch.log_softmax(logit, -1) + comp, -1)


def loglik_flow(model: FlowModel, x, F, h, s2, eps: torch.Tensor, lam: float) -> torch.Tensor:
    """Importance-sampled log ∫ e^{ℓ(z)} q(z|x) dz for the conditional MAF prior."""
    n, S = x.shape[0], eps.shape[0]
    R = F + lam * torch.eye(2, dtype=F.dtype)                       # reference-posterior precision (penalty units)
    m = torch.linalg.solve(R, h.unsqueeze(-1)).squeeze(-1)
    L = torch.linalg.cholesky(torch.linalg.inv(R) * s2[:, None, None])
    z = m[None] + torch.einsum("nij,sj->sni", L, eps)                # (S, n, 2); the same S draws for every player
    ell = -0.5 * (torch.einsum("sni,nij,snj->sn", z, F, z) - 2 * (z * h[None]).sum(-1)) / s2[None]
    d = z - m[None]
    Rs = R / s2[:, None, None]
    log_r = (-np.log(2 * np.pi) + 0.5 * torch.logdet(Rs)[None]
             - 0.5 * torch.einsum("sni,nij,snj->sn", d, Rs, d))
    c = model.context(x)
    log_q = model.log_prob(z.reshape(S * n, 2), c.repeat(S, 1), s2.repeat(S)).view(S, n)
    return torch.logsumexp(log_q + ell - log_r, 0) - np.log(S)


def train_deconv(model, spec: DensitySpec, data: dict, train_until: str, val_until: str,
                 epochs: int = 300, lr: float = 3e-3, batch: int = 512, n_samples: int = 64,
                 patience: int = 25, verbose: bool = True) -> dict:
    tg, X = data["targets"], data["X"]
    t_feat = pd.to_datetime(tg["t_feat"])
    split = np.where(t_feat < pd.Timestamp(train_until), "train",
                     np.where(t_feat < pd.Timestamp(val_until), "val", "holdout"))
    sets = {k: _tensors(tg, X, split == k) for k in ("train", "val", "holdout")}
    flow = spec.family == "maf"
    gen = torch.Generator().manual_seed(spec.seed)
    eval_eps = torch.randn(256, 2, dtype=torch.float64, generator=gen)

    def ll(sub, eps=None):
        x, F, h, s2 = sub
        if flow:
            return loglik_flow(model, x, F, h, s2, eps if eps is not None else eval_eps, spec.base_precision)
        return loglik_mixture(model, x, F, h, s2)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, best_epoch, log = -np.inf, None, -1, []
    n_train = sets["train"][0].shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n_train, generator=gen)
        model.train()
        for k in range(0, n_train, batch):
            idx = perm[k:k + batch]
            sub = tuple(a[idx] for a in sets["train"])
            eps = torch.randn(n_samples, 2, dtype=torch.float64, generator=gen) if flow else None
            loss = -ll(sub, eps).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            score = {k: float(ll(v).mean()) for k, v in sets.items() if v[0].shape[0]}
        log.append({"epoch": epoch, **score})
        if score["val"] > best + 1e-6:
            best, best_epoch = score["val"], epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
        if verbose and (epoch % 10 == 0):
            print(f"    epoch {epoch:3d}  train {score['train']:+.4f}  val {score['val']:+.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final = {k: float(ll(v).mean()) for k, v in sets.items() if v[0].shape[0]}
    return {"best_epoch": best_epoch, "loglik": final, "n": {k: int(v[0].shape[0]) for k, v in sets.items()},
            "log": log}


def score_plugin(model, spec: DensitySpec, ctx, out_dir: Path, forward_dir: Path = Path("outputs/forward"),
                 verbose: bool = True) -> dict:
    """Forward scores of the weekly RAPM solve with the trained density as its (fixed) prior."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = WindowScorer(spec.name, [(spec.base_precision, spec.base_precision)])
    cache_rows: list[dict] = []
    with torch.no_grad():
        pass
    wmse = run_epoch(ctx, model, None, scorer=scorer, cache_rows=cache_rows)
    poss, margins, origins = scorer.frames()
    forward_dir.mkdir(parents=True, exist_ok=True)
    poss.to_parquet(forward_dir / f"{spec.name}_poss.parquet", index=False)
    margins.to_parquet(forward_dir / f"{spec.name}_margins.parquet", index=False)
    origins.to_parquet(forward_dir / f"{spec.name}_origins.parquet", index=False)
    pd.DataFrame(cache_rows).to_parquet(out_dir / "rolling_cache_density.parquet", index=False)
    torch.save({"state_dict": model.state_dict(), "spec": asdict(spec),
                "features": FEATURE_SETS[spec.feature_set] + LAG_FEATURES}, out_dir / "model.pt")
    (out_dir / "plugin_diagnostics.json").write_text(json.dumps(run_epoch.last_diagnostics, indent=2))
    if verbose:
        print(f"  [{spec.name}] plug-in forward wmse {{train {wmse['train']:.5f}, val {wmse['val']:.5f}, "
              f"holdout {wmse['holdout']:.5f}}}", flush=True)
    return wmse
