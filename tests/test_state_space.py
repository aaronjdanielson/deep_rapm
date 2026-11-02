"""Validation of the state-space RAPM filter (deep_rapm.state_space).

1. Static limit: φ = 1, Q = 0, fixed μ and s² → the sequential updates reproduce
   the batch ridge (G + Λ)⁻¹ X'W(y − μ) on the same possessions.
2. Dynamic reference: on a small synthetic system the filter's means,
   covariances and one-step forecasts match a direct covariance-form Kalman
   filter, with non-zero player–home covariance.
3. Timing: the forecast used to score (t_k, t_{k+1}] depends only on possessions
   dated ≤ t_k and on the feature snapshot at t_k.

All offline; the synthetic tests build their own Context-like objects.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from deep_rapm.state_space import SSMSpec, run_filter
from deep_rapm.torch_rapm import HOME_PENALTY


def _synthetic_ctx(P=6, weeks=12, poss_per_week=60, seed=0):
    rng = np.random.default_rng(seed)
    n_cols = 2 * P + 1
    dates, origins = [], []
    t0 = pd.Timestamp("2020-10-05")
    for k in range(weeks):
        t = t0 + pd.Timedelta(days=7 * k)
        origins.append(t)
        for g in range(2):                                 # two game dates per week
            n = poss_per_week // 2
            rows, cols, vals = [], [], []
            sign = rng.choice([-1.0, 1.0], size=n)
            off = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
            de = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
            for i in range(n):
                rows += [i] * 11
                cols += list(off[i]) + list(P + de[i]) + [2 * P]
                vals += [1.0] * 10 + [sign[i]]
            X = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(n, n_cols))
            dates.append(SimpleNamespace(date=t - pd.Timedelta(days=5 - 3 * g), season="2020-21", kind="regular_season",
                                         X=X, y=rng.choice([0.0, 2.0, 3.0], size=n, p=[0.5, 0.38, 0.12]),
                                         score_diff=rng.integers(-15, 16, size=n).astype(float),
                                         context=np.ones(n), n_off=np.bincount(off.ravel(), minlength=P),
                                         n_def=np.bincount(de.ravel(), minlength=P),
                                         game_codes=np.zeros(n, int), game_ids=np.array(["g"]), sign=sign))
    cfg = SimpleNamespace(half_life_days=365.0, offseason_discount=1.0, alpha=2000.0)
    spec = SimpleNamespace(horizon_days=7)
    vocab = pd.DataFrame({"player_id": np.arange(1, P), "player_idx": np.arange(1, P)})
    ctx = SimpleNamespace(n_players=P, n_cols=n_cols, dates=dates, origins=pd.DatetimeIndex(origins),
                          comp_sigma=13.0, cfg=cfg, spec=spec, feats={}, vocab=vocab,
                          name_lookup=pd.Series(dtype=object), split_of=lambda t: "train")
    return ctx


def test_static_limit_matches_batch_ridge():
    ctx = _synthetic_ctx()
    mu, s2, lam = 1.1, 1.4, 2000.0
    spec = SSMSpec(name="static", phi=1.0, lam=lam, fixed_mu=mu, fixed_s2=s2)
    out = run_filter(spec, ctx, verbose=False, grid=list(ctx.origins), return_states=True)
    last = ctx.origins[-1]
    a_seq, C_seq = out["states"][last]["a"], out["states"][last]["C"]
    P, n = ctx.n_players, ctx.n_cols
    G, b = np.zeros((n, n)), np.zeros(n)
    for d in ctx.dates:
        w = np.exp(-(np.abs(d.score_diff) / ctx.comp_sigma) ** 2)
        Xw = d.X.multiply(np.sqrt(w)[:, None])
        G += (Xw.T @ Xw).toarray()
        b += np.asarray(d.X.T @ (w * (d.y - mu))).ravel()
    pen = np.full(n, lam); pen[2 * P:] = HOME_PENALTY
    A = G + np.diag(pen)
    assert np.allclose(a_seq, np.linalg.solve(A, b), atol=1e-10)
    assert np.allclose(C_seq, s2 * np.linalg.inv(A), atol=1e-10)


def test_matches_direct_kalman_filter():
    ctx = _synthetic_ctx(seed=1)
    mu, s2, lam, phi = 1.1, 1.4, 2000.0, 0.9
    spec = SSMSpec(name="dyn", phi=phi, lam=lam, fixed_mu=mu, fixed_s2=s2)
    out = run_filter(spec, ctx, verbose=False, grid=list(ctx.origins), return_states=True)
    P, n = ctx.n_players, ctx.n_cols
    player = np.ones(n, bool); player[2 * P:] = False
    D = np.diag(np.where(player, phi, 1.0))
    Q = np.diag(np.where(player, (1 - phi ** 2) * s2 / lam, 0.0))
    pen = np.full(n, lam); pen[2 * P:] = HOME_PENALTY
    a, C = np.zeros(n), s2 * np.diag(1 / pen)                   # prior at the start
    by_week = {}
    for d in ctx.dates:
        by_week.setdefault(min(t for t in ctx.origins if t >= d.date), []).append(d)
    for t in ctx.origins:
        H, r, Rinv = [], [], []
        for d in by_week.get(t, []):
            w = np.exp(-(np.abs(d.score_diff) / ctx.comp_sigma) ** 2)
            H.append(d.X.toarray()); r.append(d.y - mu); Rinv.append(w / s2)
        H, r, Rinv = np.vstack(H), np.concatenate(r), np.concatenate(Rinv)
        # covariance-form update: C⁺ = (C⁻¹ + HᵀR⁻¹H)⁻¹, a⁺ = C⁺(C⁻¹a + HᵀR⁻¹r)
        Cinv = np.linalg.inv(C)
        C_up = np.linalg.inv(Cinv + H.T @ (Rinv[:, None] * H))
        a_up = C_up @ (Cinv @ a + H.T @ (Rinv * r))
        st = out["states"][t]
        assert np.allclose(st["a"], a_up, atol=1e-9), f"filtered mean differs at {t.date()}"
        assert np.allclose(st["C"], C_up, atol=1e-9), f"filtered covariance differs at {t.date()}"
        a_pred, C_pred = D @ a_up, D @ C_up @ D.T + Q           # zero mean → forecast = D a
        assert np.allclose(st["a_pred"], a_pred, atol=1e-9)
        assert np.allclose(st["C_pred"], C_pred, atol=1e-9)
        assert abs(C_pred[0, 2 * P]) > 1e-8                       # player–home covariance is non-zero and tested
        assert np.all(np.linalg.eigvalsh(C_pred) > 0)
        a, C = a_pred, C_pred


def test_forecast_uses_only_past_information():
    ctx = _synthetic_ctx(seed=2)
    spec = SSMSpec(name="timing", phi=0.95, fixed_mu=1.1, fixed_s2=1.4)
    full = run_filter(spec, ctx, verbose=False, grid=list(ctx.origins), return_states=True)["states"]
    t_cut = ctx.origins[5]
    ctx2 = _synthetic_ctx(seed=2)
    ctx2.dates = [d for d in ctx2.dates if d.date <= t_cut]        # drop everything after t_cut
    part = run_filter(spec, ctx2, verbose=False, grid=list(ctx2.origins[:6]), return_states=True)["states"]
    assert np.allclose(full[t_cut]["a_pred"], part[t_cut]["a_pred"])
    assert np.allclose(full[t_cut]["C_pred"], part[t_cut]["C_pred"])


def test_forecast_is_the_next_prior():
    """The state used to score (t_k, t_{k+1}] is exactly the state the filter assimilates
    that window into at t_{k+1}, before seeing its outcomes."""
    ctx = _synthetic_ctx(seed=3)
    spec = SSMSpec(name="consistency", phi=0.93, fixed_mu=1.1, fixed_s2=1.4)
    st = run_filter(spec, ctx, verbose=False, grid=list(ctx.origins), return_states=True)["states"]
    for k in range(len(ctx.origins) - 1):
        t, t_next = ctx.origins[k], ctx.origins[k + 1]
        assert np.array_equal(st[t]["a_pred"], st[t_next]["a_prior"])
        assert np.array_equal(st[t]["C_pred"], st[t_next]["C_prior"])


def test_prior_mean_requires_the_checkpoints_feature_pipeline(tmp_path):
    """A prior checkpoint loads only with the feature set it was trained on."""
    import torch
    from deep_rapm.priors import FEATURE_SETS
    from deep_rapm.torch_rapm import LAG_FEATURES
    from deep_rapm.state_space import PriorMean
    ctx = _synthetic_ctx(seed=4)
    n_v1 = len(FEATURE_SETS["v1"])
    ctx.feats = {t: np.zeros((ctx.n_players, n_v1), np.float32) for t in ctx.origins}
    ctx.spec = SimpleNamespace(horizon_days=7, feature_set="v1")
    n_in = n_v1 + len(LAG_FEATURES)
    ck = {"spec": {"name": "x", "feature_set": "v2", "shrink_minutes": 500.0}, "features": FEATURE_SETS["v2"] + LAG_FEATURES,
          "state_dict": {"prior.net.0.weight": torch.zeros(64, len(FEATURE_SETS["v2"]) + len(LAG_FEATURES))}}
    torch.save(ck, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="prior checkpoint expects features"):
        PriorMean(SSMSpec(name="p", prior=str(tmp_path / "wrong.pt")), ctx)
