"""RAPM as a linear-Gaussian state-space model with an explicit transition.

State β_k ∈ ℝ^{2P+1}: every player's offensive and defensive coefficient plus
home court, on a weekly grid t_0 < t_1 < … . With m_k the feature-based
expected ability known at t_k (zero, or a frozen learned prior mean),

    transition    β_{k+1} = m_k + D (β_k − m_k) + η_{k+1},   η ~ N(0, Q),
                  D = diag(φ on the 2P player coefficients, 1 on home),
                  Q = diag((1 − φ²) τ² on players, 0 on home),   τ² = s²/λ,
    observation   y = μ 1 + X β + ε,   ε ~ N(0, s² W⁻¹)   for the possessions in (t_k, t_{k+1}],
                  W = competition × context weights (no recency: that is the transition's job).

Interval bookkeeping (the review's timeline), at each grid date t_k:
    1. assimilate the possessions in (t_{k−1}, t_k] into the state predicted for that interval
       → filtered (a_k, C_k);
    2. compute m_k from the feature snapshot at t_k (features use games before t_k only);
    3. forecast the next interval: a_{k+1|k} = m_k + D (a_k − m_k),  C_{k+1|k} = D C_k Dᵀ + Q
       (crossing July 1 applies D and Q ``summer_steps`` times);
    4. score (t_k, t_{k+1}] with the *forecast* mean and covariance; the same forecast is
       the prior of step 1 at t_{k+1}.
Only the new interval's possessions enter each update. The information form
Ω = C⁻¹, ξ = Ω a is kept; three dense inverses per week.

Intercept μ_k and noise s_k² follow v1's conventions exactly (decayed running
weighted mean; weighted residual variance at the filtered state, from the
same accumulators v1 uses), so a comparison with v1 / v1_home isolates the
transition. With φ = 1, Q = 0, fixed μ and fixed s² the sequential updates
reproduce the batch ridge on the same data (the static-limit check); the
usual run lets μ and s² evolve, so τ² = s_k²/λ is adaptive rather than one
fixed stationary variance — the same adaptivity v1's prior has.

Warm-up: the filter runs chronologically from the first game on the weekly
grid with zero mean; scoring (and, for a learned mean, the mean itself)
starts at the first harness origin, where features exist. The learned mean
enters ξ at that first origin's forecast; the initial information state is
Ω₀ = Λ/s₀², ξ₀ = Ω₀ m₀ with m₀ = 0.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg

from .forward import KINDS_MAIN, WindowScorer, eval_weights, predictive_terms
from .rolling import IncrementalGramState
from .torch_rapm import HOME_PENALTY, Context, _window


def _inv(A: np.ndarray) -> np.ndarray:
    c = scipy.linalg.cho_factor(A, check_finite=False)
    return scipy.linalg.cho_solve(c, np.eye(len(A)), check_finite=False)


@dataclass
class SSMSpec:
    name: str
    phi: float = 0.99               # weekly persistence of the deviation from the prior mean
    summer_steps: int = 1           # transitions applied when the next interval crosses July 1
    lam: float = 2000.0             # prior precision (penalty units) → τ² = s²/λ
    prior: str = "zero"             # "zero" or a path to a v2 checkpoint whose prior mean is used
    fixed_mu: float | None = None   # testing: hold the intercept fixed
    fixed_s2: float | None = None   # testing: hold the noise variance fixed


class PriorMean:
    """m_k from a frozen v2 checkpoint (learned mean), or zero. Defined only at
    harness origins (where feature snapshots exist); zero elsewhere."""

    def __init__(self, spec: SSMSpec, ctx: Context) -> None:
        self.ctx, self.model = ctx, None
        if spec.prior != "zero":
            import torch
            from .priors import FEATURE_SETS
            from .torch_rapm import LAG_FEATURES, FitSpec, _lag_features, new_model
            ck = torch.load(spec.prior, weights_only=False)
            self.fs = FitSpec(**ck["spec"])
            # The context's feature arrays are already built; they must be the checkpoint's pipeline.
            n_in = ck["state_dict"]["prior.net.0.weight"].shape[1]
            n_feat = next(iter(ctx.feats.values())).shape[1]
            if n_feat + len(LAG_FEATURES) != n_in or FEATURE_SETS[ctx.spec.feature_set] + LAG_FEATURES != list(ck["features"]):
                raise ValueError(f"prior checkpoint expects features {ck['features'][:3]}… ({n_in} inputs); the context "
                                 f"was built with feature_set={ctx.spec.feature_set!r} ({n_feat + len(LAG_FEATURES)} inputs)")
            saved, ctx.spec = ctx.spec, self.fs
            self.model = new_model(ctx)
            ctx.spec = saved
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval()
            self.lag, self.torch = _lag_features, torch

    def __call__(self, t: pd.Timestamp, history: list, n_cols: int) -> np.ndarray:
        if self.model is None or t not in self.ctx.feats:
            return np.zeros(n_cols)
        P = self.ctx.n_players
        x = self.torch.from_numpy(np.concatenate([self.ctx.feats[t], self.lag(t, history, P, self.fs.lag_days)], axis=1))
        with self.torch.no_grad():
            m, _ = self.model.prior_terms(x)
        return m.numpy()                        # coefficient coordinates (δ-component already sign-flipped)


def weekly_grid(ctx: Context) -> list[pd.Timestamp]:
    """Warm-up weeks back from the first harness origin to the first game, then the origins."""
    first_game = min(d.date for d in ctx.dates)
    n_warm = int((ctx.origins[0] - first_game).days // 7) + 1
    warm = [ctx.origins[0] - pd.Timedelta(days=7 * k) for k in range(n_warm, 0, -1)]
    return [t for t in warm if t > first_game - pd.Timedelta(days=7)] + list(ctx.origins)


def run_filter(spec: SSMSpec, ctx: Context, scorer: WindowScorer | None = None, cache_rows: list | None = None,
               verbose: bool = True, grid: list[pd.Timestamp] | None = None, return_states: bool = False) -> dict:
    P, n_cols = ctx.n_players, ctx.n_cols
    horizon = pd.Timedelta(days=ctx.spec.horizon_days)
    prior_mean = PriorMean(spec, ctx)
    player = np.ones(n_cols, bool)
    player[2 * P:] = False
    lam_vec = np.where(player, spec.lam, HOME_PENALTY)
    grid = weekly_grid(ctx) if grid is None else list(grid)
    score_origins = set(ctx.origins)

    # v1's accumulators, used only for μ_k, s_k² and the exposure history (not for the state)
    side = IncrementalGramState(n_cols, spec.lam, ctx.cfg.half_life_days, n_players=P,
                                offseason_discount=ctx.cfg.offseason_discount)
    s2 = spec.fixed_s2 if spec.fixed_s2 is not None else 1.4
    Omega = np.diag(lam_vec / s2)                      # Ω₀ = Λ/s₀², ξ₀ = Ω₀ m₀, m₀ = 0
    xi = np.zeros(n_cols)
    m_prev = np.zeros(n_cols)                          # the mean the current predicted state is centred on
    a_pred, C_pred = np.zeros(n_cols), None            # forecast for the interval about to be assimilated
    history: list[tuple] = []
    sums = {k: np.zeros(2) for k in ("train", "val", "holdout")}
    states = {}
    t0, j = time.time(), 0
    for k, t in enumerate(grid):
        # ---- 1. assimilate (t_{k−1}, t_k] into the predicted state ----
        prior_state = (a_pred.copy(), None if C_pred is None else C_pred.copy()) if return_states else None
        week = []
        while j < len(ctx.dates) and ctx.dates[j].date <= t:
            week.append(ctx.dates[j])
            j += 1
        _, mu, _ = side.centered_rhs() if side.total_w > 0 else (None, None, None)
        for d in week:
            w = np.exp(-(np.abs(d.score_diff) / ctx.comp_sigma) ** 2) * d.context
            mu_d = spec.fixed_mu if spec.fixed_mu is not None else (mu if mu is not None else float(np.average(d.y, weights=w)))
            Xw = d.X.multiply(np.sqrt(w)[:, None])
            Omega += (Xw.T @ Xw).toarray() / s2
            xi += np.asarray(d.X.T @ (w * (d.y - mu_d))).ravel() / s2
            side.advance_to(d.date)
            side.ingest(d.X, d.y, d.n_off, d.n_def, comp_sigma=ctx.comp_sigma, score_diff=d.score_diff,
                        context_weight=d.context)
        if side.total_w == 0:
            continue
        C = _inv(Omega)
        a = C @ xi
        _, mu, _ = side.centered_rhs()
        if spec.fixed_mu is not None:
            mu = spec.fixed_mu
        s2 = spec.fixed_s2 if spec.fixed_s2 is not None else side.noise_variance(a)
        history.append((t, a.copy(), np.diag(side.G).copy()))
        # ---- 2. mean known at t_k; 3. forecast (t_k, t_{k+1}] ----
        m_k = prior_mean(t, history, n_cols)
        t_next = grid[k + 1] if k + 1 < len(grid) else t + pd.Timedelta(days=7)
        july1 = pd.Timestamp(year=t_next.year if t_next.month >= 7 else t_next.year - 1, month=7, day=1)
        steps = spec.summer_steps if t < july1 <= t_next else 1
        phi_k = spec.phi ** steps
        d_diag = np.where(player, phi_k, 1.0)
        a_pred = m_k + d_diag * (a - m_k)
        C_pred = C * np.outer(d_diag, d_diag)
        C_pred[np.diag_indices(n_cols)] += np.where(player, (1 - phi_k ** 2) * s2 / spec.lam, 0.0)
        Omega = _inv(C_pred)
        xi = Omega @ a_pred
        m_prev = m_k
        if return_states:
            states[t] = {"a": a.copy(), "C": C, "a_pred": a_pred.copy(), "C_pred": C_pred.copy(), "mu": mu, "s2": s2,
                         "a_prior": prior_state[0], "C_prior": prior_state[1], "m": m_k.copy()}
        if t not in score_origins:
            continue
        # ---- 4. score the next interval with the forecast ----
        test = _window(ctx.dates, j, t, horizon)
        split = ctx.split_of(t)
        for d in test:
            yhat = mu + d.X @ a_pred
            if d.kind in KINDS_MAIN:
                w = eval_weights(d.score_diff)
                e = d.y - yhat
                sums[split] += ((w * e * e).sum(), w.sum())
            if scorer is not None:
                pv, mv, bv = predictive_terms(C_pred / s2, s2, d)
                scorer.add(0, t, d, yhat, mu, pred_var=pv, s2=s2, margin_var=mv, base_var=bv)
        if cache_rows is not None and t == ctx.origins[-1]:
            sd = 100 * np.sqrt(np.diag(C))
            ar = np.arange(P)
            sd_r = 100 * np.sqrt(np.maximum(np.diag(C)[:P] + np.diag(C)[P:2 * P] - 2 * C[ar, P + ar], 0))
            for pid, idx in zip(ctx.vocab["player_id"].astype(int).values, ctx.vocab["player_idx"].astype(int).values):
                cache_rows.append({"date": t, "player_id": pid, "player_name": ctx.name_lookup.get(pid, "Unknown"),
                                   "orapm": 100 * a[idx], "drapm": -100 * a[P + idx], "rapm": 100 * (a[idx] - a[P + idx]),
                                   "sd_orapm": sd[idx], "sd_drapm": sd[P + idx], "sd_rapm": sd_r[idx],
                                   "prior_orapm": 100 * m_k[idx], "prior_drapm": -100 * m_k[P + idx],
                                   "n_off": int(side.n_off[idx]), "n_def": int(side.n_def[idx])})
        if verbose and ((k + 1) % max(1, len(grid) // 10) == 0):
            print(f"  [{spec.name}] [{100 * (k + 1) / len(grid):5.1f}%] {t.date()}  s²={s2:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    out = {k: (v[0] / v[1] if v[1] > 0 else float("nan")) for k, v in sums.items()}
    if return_states:
        out["states"] = states
    return out
