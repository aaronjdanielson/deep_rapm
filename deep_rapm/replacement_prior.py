"""v1 with a replacement-level prior: the ridge's prior mean set by playing time.

The production ridge shrinks every player toward the same value. Coaches give
minutes to players who help and withhold them from players who hurt, so a
player with little playing time is, on that evidence alone, below average.
This model keeps v1 (penalty 2000 on every coefficient) and replaces the zero
prior mean with

    ORAPM prior_i = a_o + b_o·log(1 + min_i) [+ c_o·mpg_i]
    DRAPM prior_i = a_d + b_d·log(1 + min_i) [+ c_d·mpg_i]

where min_i and mpg_i are the player's minutes and minutes per game played in
the 365 days before the origin (PlayerGameLogs: regular season, play-in and
playoffs). Players with no recent games — rookies, returning players, the
UNKNOWN slot — get the intercepts.

With the precision fixed, the coefficients at origin t are
β(θ) = β0 + C θ with β0 = (G + Λ)⁻¹ X'W(y − μ) and C = (G + Λ)⁻¹ Λ M, where M
maps θ to per-coefficient prior means. Every forward prediction is therefore
linear in θ and the forward objective — weighted squared error of the next
window, non-garbage-time weighted, regular season + playoffs — is a weighted
least-squares problem in θ, solved exactly on the training origins.

Shifting every player's ORAPM and DRAPM prior up by the same amount changes no
prediction (both teams field five players), so only a_o − a_d, the slopes and
the curvature terms are identified; the overall level is a convention. Ratings
are reported relative to the possession-weighted average player (weights:
offensive possessions in the past 365 days), which is what "points per 100
relative to an average player" means.

The predictive variance does not depend on θ beyond the noise scale s²(θ),
which is quadratic in θ and stored per origin, so every variant is scored —
including the predictive density — from one walk over the origins.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .forward import KINDS_MAIN, WindowScorer, eval_weights, posterior_inverse, predictive_terms
from .priors import FEATURES
from .production import V1, RapmConfig
from .rolling import IncrementalGramState

TERMS = ("one", "logmin", "mpg")          # prior-mean basis per side
HOME_PENALTY = {"nohome": 1e12, "home": 1.0}


def prior_basis(feats: np.ndarray) -> np.ndarray:
    """(n_players + 1, 3) basis [1, log(1 + minutes), minutes per game] from the prior features."""
    i_min, i_games = FEATURES.index("log_min_last365"), FEATURES.index("games_last365")
    logmin = feats[:, i_min].astype(np.float64)
    games = feats[:, i_games].astype(np.float64)
    mpg = np.where(games > 0, np.expm1(logmin) / np.maximum(games, 1), 0.0)
    return np.column_stack([np.ones(len(feats)), logmin, mpg])


def design_M(basis: np.ndarray, P: int, n_cols: int) -> np.ndarray:
    """Map θ = [a_o, b_o, c_o, a_d, b_d, c_d] to coefficient prior means (δ = −DRAPM)."""
    K = basis.shape[1]
    M = np.zeros((n_cols, 2 * K))
    M[:P, :K] = basis
    M[P:2 * P, K:] = -basis
    return M


@dataclass
class Walk:
    """Everything the variants need, from one walk over the origins (one home setting)."""
    home: str
    origins: list = field(default_factory=list)          # per origin: dict of scalars / small arrays
    tests: list = field(default_factory=list)            # per (origin, game date): arrays
    final: dict = field(default_factory=dict)


def walk(ctx, home_settings=("nohome", "home"), verbose: bool = True) -> dict[str, Walk]:
    """One pass over ctx.origins (a deep_rapm.torch_rapm Context) for every home setting."""
    cfg, P, n_cols = ctx.cfg, ctx.n_players, ctx.n_cols
    horizon = pd.Timedelta(days=ctx.spec.horizon_days)
    state = IncrementalGramState(n_cols, cfg.alpha, cfg.half_life_days, n_players=P,
                                 offseason_discount=cfg.offseason_discount)
    out = {h: Walk(h) for h in home_settings}
    pens = {}
    for h in home_settings:
        pen = cfg.penalty(P)
        pen[-1] = HOME_PENALTY[h]
        pens[h] = pen
    snaps: list[tuple[pd.Timestamp, np.ndarray]] = []
    t0, j = time.time(), 0
    for k, t in enumerate(ctx.origins):
        while j < len(ctx.dates) and ctx.dates[j].date <= t:
            d = ctx.dates[j]
            state.advance_to(d.date)
            state.ingest(d.X, d.y, d.n_off, d.n_def, comp_sigma=ctx.comp_sigma,
                         score_diff=d.score_diff, context_weight=d.context)
            j += 1
        if state.total_w == 0:
            continue
        snaps.append((t, state.n_off.copy()))
        xty, mu, sy2 = state.centered_rhs()
        test = []
        m = j
        while m < len(ctx.dates) and ctx.dates[m].date <= t + horizon:
            test.append(ctx.dates[m])
            m += 1
        basis = prior_basis(ctx.feats[t])
        M = design_M(basis, P, n_cols)
        for h in home_settings:
            pen = pens[h]
            A_inv = posterior_inverse(state.G, pen)
            beta0 = A_inv @ xty
            C = A_inv @ (pen[:, None] * M)
            Gb0 = state.G @ beta0
            GC = state.G @ C
            rss0 = sy2 - 2 * beta0 @ xty + beta0 @ Gb0
            out[h].origins.append({"origin": t, "mu": mu, "total_w": state.total_w, "rss0": rss0,
                                   "g1": C.T @ (xty - Gb0), "H": C.T @ GC})
            for d in test:
                pv1, mv1, bv1 = predictive_terms(A_inv, 1.0, d)
                out[h].tests.append({"oi": len(out[h].origins) - 1, "d": d,
                                     "yhat0": mu + d.X @ beta0, "F": np.asarray(d.X @ C),
                                     "pv1": pv1, "mv1": mv1, "bv1": bv1})
            if t == ctx.origins[-1]:
                past = [n for (tt, n) in snaps if tt <= t - pd.Timedelta(days=365)]
                out[h].final = {"origin": t, "beta0": beta0, "C": C, "A_inv_diag": np.diag(A_inv).copy(),
                                "A_inv_od": A_inv[np.arange(P), P + np.arange(P)].copy(),
                                "n_off": state.n_off.copy(), "n_def": state.n_def.copy(),
                                "recent_off": state.n_off - (past[-1] if past else 0),
                                "basis": basis}
        if verbose and ((k + 1) % max(1, len(ctx.origins) // 10) == 0 or k == len(ctx.origins) - 1):
            print(f"  [{100 * (k + 1) / len(ctx.origins):5.1f}%] {t.date()}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def _cols(variant_terms: tuple[str, ...]) -> list[int]:
    K = len(TERMS)
    idx = [TERMS.index(x) for x in variant_terms]
    return idx + [K + i for i in idx]


def fit_theta(w: Walk, terms: tuple[str, ...], train_until: pd.Timestamp) -> dict:
    """Exact WLS for θ on the training origins; min-norm on the unidentified level direction."""
    cols = _cols(terms)
    rows_F, rows_r, rows_w, rows_g = [], [], [], []
    for tt in w.tests:
        o = w.origins[tt["oi"]]
        d = tt["d"]
        if o["origin"] >= train_until or d.kind not in KINDS_MAIN:
            continue
        rows_F.append(tt["F"][:, cols])
        rows_r.append(d.y - tt["yhat0"])
        rows_w.append(eval_weights(d.score_diff))
        rows_g.append(np.full(len(d.y), tt["oi"]))
    F, r, wt, g = (np.concatenate(x) for x in (rows_F, rows_r, rows_w, rows_g))
    Fw = F * wt[:, None]
    XtX = F.T @ Fw
    theta_sub, *_ = np.linalg.lstsq(XtX, Fw.T @ r, rcond=None)
    e = r - F @ theta_sub
    pinv = np.linalg.pinv(XtX)
    meat = np.zeros_like(XtX)
    for k in np.unique(g):                                   # cluster by origin week
        s = (Fw[g == k] * e[g == k, None]).sum(0)
        meat += np.outer(s, s)
    cov = pinv @ meat @ pinv
    theta = np.zeros(2 * len(TERMS))
    theta[cols] = theta_sub
    cov_full = np.zeros((len(theta), len(theta)))
    cov_full[np.ix_(cols, cols)] = cov
    return {"terms": list(terms), "theta": theta, "cov": cov_full, "n": int(len(r)),
            "origins": int(len(np.unique(g)))}


def score(w: Walk, theta: np.ndarray, name: str, penalty_label: tuple[float, float]) -> WindowScorer:
    """Forward scores (MSE, density, margins) for the variant θ, in the harness format."""
    sc = WindowScorer(name, [penalty_label])
    s2 = []
    for o in w.origins:
        rss = o["rss0"] - 2 * theta @ o["g1"] + theta @ o["H"] @ theta
        s2.append(max(rss / o["total_w"], 1e-9))
    for tt in w.tests:
        o, d = w.origins[tt["oi"]], tt["d"]
        s = s2[tt["oi"]]
        yhat = tt["yhat0"] + tt["F"] @ theta
        sc.add(0, o["origin"], d, yhat, o["mu"], pred_var=s * tt["pv1"], s2=s,
               margin_var=s * tt["mv1"], base_var=s * tt["bv1"])
    return sc


def final_ratings(w: Walk, theta: np.ndarray, ctx, min_poss: int = 100) -> pd.DataFrame:
    """Ratings at the last origin, re-levelled so the possession-weighted average player is 0."""
    f, P = w.final, ctx.n_players
    beta = f["beta0"] + f["C"] @ theta
    orapm, drapm = beta[:P] * 100, -beta[P:2 * P] * 100
    K = len(TERMS)
    vocab = ctx.vocab
    idx = vocab["player_idx"].astype(int).values
    keep = (f["n_off"][idx] >= min_poss) | (f["n_def"][idx] >= min_poss)
    basis = f["basis"][idx]
    df = pd.DataFrame({"player_id": vocab["player_id"].astype(int).values, "player_idx": idx,
                       "orapm": orapm[idx], "drapm": drapm[idx],
                       "prior_orapm": 100 * basis @ theta[:K], "prior_drapm": 100 * basis @ theta[K:],
                       "recent_off": f["recent_off"][idx],
                       "minutes_365": np.expm1(basis[:, 1]), "mpg_365": basis[:, 2]})[keep].copy()
    # Equal shifts of every ORAPM and DRAPM change no prediction; anchor the
    # level so the possession-weighted average player has RAPM 0.
    w_ = df["recent_off"].clip(lower=0)
    shift = np.average(df["orapm"] + df["drapm"], weights=w_) / 2 if w_.sum() > 0 else 0.0
    for col in ("orapm", "drapm", "prior_orapm", "prior_drapm"):
        df[col] -= shift
    df["rapm"] = df["orapm"] + df["drapm"]
    df["prior_rapm"] = df["prior_orapm"] + df["prior_drapm"]
    df["player_name"] = df["player_id"].map(ctx.name_lookup).fillna("Unknown")
    df["date"] = f["origin"]
    return df
