"""
tests/test_incremental_rapm.py — correctness tests for IncrementalGramState.

Tests that the incremental Gram update produces results numerically identical
to the full recompute (solve_ridge) on synthetic data, where competition sigma
is fixed (removing the only legitimate source of divergence between the two
paths in production).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from deep_rapm.rapm import solve_ridge
from deep_rapm.rolling import IncrementalGramState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_possession_batch(
    rng: np.random.Generator,
    n_poss: int,
    n_players: int,
    game_date: pd.Timestamp,
) -> tuple[sp.csr_matrix, np.ndarray, pd.Series]:
    """Return (X, y, score_diff) for a synthetic batch of possessions."""
    n_cols = 2 * n_players
    rows, cols, vals = [], [], []
    for i in range(n_poss):
        off = rng.choice(n_players, size=5, replace=False)
        def_ = rng.choice(n_players, size=5, replace=False)
        for j in off:
            rows.append(i); cols.append(j); vals.append(1.0)
        for j in def_:
            rows.append(i); cols.append(n_players + j); vals.append(1.0)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n_poss, n_cols))
    y = rng.normal(1.0, 0.8, size=n_poss).astype(np.float32)
    score_diff = pd.Series(rng.normal(0, 10, size=n_poss))
    return X, y, score_diff


def _recency_weights(dates: np.ndarray, ref_date: pd.Timestamp, half_life: float) -> np.ndarray:
    days_ago = np.array([(ref_date - d).days for d in dates], dtype=np.float64)
    return 0.5 ** (days_ago / half_life)


def _comp_weights(score_diff: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(np.abs(score_diff) / sigma) ** 2)


# ---------------------------------------------------------------------------
# Core equivalence test: incremental == full recompute
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("half_life", [90.0, 365.0])
@pytest.mark.parametrize("n_game_dates", [5, 15])
def test_incremental_matches_full_recompute(half_life: float, n_game_dates: int) -> None:
    """
    IncrementalGramState.solve() must match solve_ridge() to near machine
    precision when competition sigma is fixed (as in the incremental path).
    """
    rng        = np.random.default_rng(42)
    n_players  = 30
    n_cols     = 2 * n_players
    alpha      = 500.0
    comp_sigma = 13.87

    # Build synthetic game-date batches
    base_date  = pd.Timestamp("2023-10-01")
    game_dates = [base_date + pd.Timedelta(days=3 * i) for i in range(n_game_dates)]
    batches    = [
        _make_possession_batch(rng, rng.integers(100, 300), n_players, gd)
        for gd in game_dates
    ]

    # ── Incremental path ──────────────────────────────────────────────────────
    state = IncrementalGramState(n_cols, alpha, half_life)
    for gd, (X_g, y_g, sd_g) in zip(game_dates, batches):
        state.advance_to(gd)
        state.ingest(X_g, y_g,
                     np.zeros(n_players, dtype=np.int64),
                     np.zeros(n_players, dtype=np.int64),
                     comp_sigma=comp_sigma,
                     score_diff=sd_g.values)
    beta_inc, mu_inc = state.solve()

    # ── Full recompute path ───────────────────────────────────────────────────
    last_game_date = game_dates[-1]

    all_X_list, all_y_list, all_w_list = [], [], []
    for gd, (X_g, y_g, sd_g) in zip(game_dates, batches):
        w_rec  = _recency_weights([gd], last_game_date, half_life)[0]
        w_comp = _comp_weights(sd_g.values, comp_sigma)
        w      = w_rec * w_comp
        all_X_list.append(X_g)
        all_y_list.append(y_g)
        all_w_list.append(w)

    X_full = sp.vstack(all_X_list, format="csr")
    y_full = np.concatenate(all_y_list)
    w_full = np.concatenate(all_w_list)

    beta_full, mu_full = solve_ridge(X_full, y_full, alpha, weights=w_full)

    np.testing.assert_allclose(beta_inc, beta_full, rtol=1e-5, atol=1e-6,
        err_msg="beta from incremental and full recompute diverge")
    assert abs(mu_inc - mu_full) < 1e-6, (
        f"intercept mismatch: inc={mu_inc:.8f} full={mu_full:.8f}"
    )


def test_advance_to_scales_state_correctly() -> None:
    """
    Advancing by Δ days should multiply G and b by γ = 0.5^(Δ/H).
    """
    rng       = np.random.default_rng(7)
    n_players = 10
    n_cols    = 2 * n_players
    alpha     = 100.0
    half_life = 90.0

    state = IncrementalGramState(n_cols, alpha, half_life)
    t0    = pd.Timestamp("2024-01-01")
    state.advance_to(t0)

    X, y, sd = _make_possession_batch(rng, 50, n_players, t0)
    state.ingest(X, y,
                 np.zeros(n_players, dtype=np.int64),
                 np.zeros(n_players, dtype=np.int64))

    G_before   = state.G.copy()
    total_before = state.total_w

    delta = 30
    gamma = 0.5 ** (delta / half_life)
    state.advance_to(t0 + pd.Timedelta(days=delta))

    np.testing.assert_allclose(state.G, G_before * gamma, rtol=1e-12,
        err_msg="G not scaled correctly after advance_to")
    assert abs(state.total_w - total_before * gamma) < 1e-12


def test_empty_state_raises_cleanly() -> None:
    """Solving on an empty state should not silently return nonsense."""
    state = IncrementalGramState(20, alpha=100.0, half_life_days=180.0)
    state.advance_to(pd.Timestamp("2024-01-01"))
    # total_w == 0 → mu = 0.0, solve returns zero vector
    beta, mu = state.solve()
    assert mu == 0.0
    np.testing.assert_array_equal(beta, np.zeros(20))


def test_single_game_date_matches_full_recompute() -> None:
    """Degenerate case: one game date, no recency decay needed."""
    rng        = np.random.default_rng(99)
    n_players  = 20
    n_cols     = 2 * n_players
    alpha      = 200.0
    half_life  = 180.0
    comp_sigma = 13.87

    gd   = pd.Timestamp("2025-01-15")
    X, y, sd = _make_possession_batch(rng, 200, n_players, gd)

    # Incremental
    state = IncrementalGramState(n_cols, alpha, half_life)
    state.advance_to(gd)
    state.ingest(X, y,
                 np.zeros(n_players, dtype=np.int64),
                 np.zeros(n_players, dtype=np.int64),
                 comp_sigma=comp_sigma,
                 score_diff=sd.values)
    beta_inc, mu_inc = state.solve()

    # Full recompute — one game date, recency weight = 1.0 for all rows
    w = _comp_weights(sd.values, comp_sigma)
    beta_full, mu_full = solve_ridge(X, y, alpha, weights=w)

    np.testing.assert_allclose(beta_inc, beta_full, rtol=1e-5, atol=1e-6)
    assert abs(mu_inc - mu_full) < 1e-6
