"""
rapm.py — Analytical Ridge RAPM (public API).

Fits traditional RAPM as a ridge regression on possession data:

    y ≈ intercept
        + Σ α[i]  for each offensive player i
        + Σ δ[j]  for each defensive player j

Sign convention:
  α[i] > 0  →  player i improves offensive scoring
  δ[j] < 0  →  player j suppresses opponent scoring (good defender)

Reported values (per 100 possessions):
  ORAPM = α × 100
  DRAPM = −δ × 100   (positive = good defender)
  RAPM  = ORAPM + DRAPM

Recency weighting:
  When half_life_days is provided (and a reference date is available from the
  data), each possession is weighted by:

      w_i = 0.5 ^ (days_ago_i / half_life_days)

  The ridge solve then minimises  Σ w_i (y_i − X_i β)² + alpha ‖β‖²

Public API
----------
    fit_rapm(data_dir, seasons, player_vocab_path, player_table_path,
             alpha, output_dir) -> pd.DataFrame

    fit_rapm(data_dir, from_date, to_date, player_vocab_path,
             player_table_path, alpha, half_life_days, output_dir) -> pd.DataFrame

    load_rapm(rapm_dir) -> pd.DataFrame

Example
-------
    from pathlib import Path
    from deep_rapm import fit_rapm

    results = fit_rapm(
        data_dir=Path("data"),
        seasons=["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
        player_vocab_path=Path("data/player_vocab.parquet"),
        player_table_path=Path("data/players.parquet"),
        alpha=2000,
        output_dir=Path("checkpoints/rapm"),
    )
    print(results.nlargest(10, "rapm")[["player_name", "orapm", "drapm", "rapm"]])
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Core: matrix construction (from a pre-loaded DataFrame)
# ---------------------------------------------------------------------------

def _build_matrix_from_df(
    df: pd.DataFrame,
    player_vocab: pd.DataFrame,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the sparse RAPM indicator matrix from a possessions DataFrame.

    Returns
    -------
    X       : csr_matrix (n_poss, 2 * n_players)
                cols 0 … n_players-1        — offense indicators
                cols n_players … 2*n-1      — defense indicators
    y       : ndarray (n_poss,) float32     — points scored
    n_off   : ndarray (n_players,) int      — offensive possession count per idx
    n_def   : ndarray (n_players,) int      — defensive possession count per idx
    """
    pid_to_idx = pd.Series(
        player_vocab["player_idx"].values,
        index=player_vocab["player_id"].values,
        dtype="int32",
    )
    n_players = len(player_vocab) + 1   # index 0 = UNKNOWN
    n = len(df)

    off_ids = np.array(df["offense_player_ids"].tolist(), dtype=object)
    def_ids = np.array(df["defense_player_ids"].tolist(), dtype=object)

    def _map(ids_matrix: np.ndarray) -> np.ndarray:
        flat = pd.Series(ids_matrix.ravel().astype(int))
        return flat.map(pid_to_idx).fillna(0).astype("int32").values.reshape(-1, 5)

    off_idx = _map(off_ids)   # (n, 5)
    def_idx = _map(def_ids)   # (n, 5)

    row_rep = np.repeat(np.arange(n, dtype=np.int32), 5)
    col_all = np.concatenate([off_idx.ravel(), n_players + def_idx.ravel()])
    row_all = np.concatenate([row_rep, row_rep])

    X = sp.csr_matrix(
        (np.ones(10 * n, dtype=np.float32), (row_all, col_all)),
        shape=(n, 2 * n_players),
    )
    y = df["points"].values.astype(np.float32)
    n_off = np.bincount(off_idx.ravel(), minlength=n_players)
    n_def = np.bincount(def_idx.ravel(), minlength=n_players)

    return X, y, n_off, n_def


# ---------------------------------------------------------------------------
# Core: matrix construction (loading from season parquets)
# ---------------------------------------------------------------------------

def build_rapm_matrix(
    data_dir: Path,
    seasons: list[str],
    player_vocab: pd.DataFrame,
    include_playoffs: bool = False,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the sparse RAPM indicator matrix from season possession parquets.

    Parameters
    ----------
    include_playoffs : If True, also load ``possessions_*_playoffs.parquet``
                       files alongside the regular season parquets.

    Returns
    -------
    X       : csr_matrix (n_poss, 2 * n_players)
    y       : ndarray (n_poss,) float32
    n_off   : ndarray (n_players,) int
    n_def   : ndarray (n_players,) int
    """
    frames: list[pd.DataFrame] = []
    for season in seasons:
        season_dir = Path(data_dir) / season
        all_parquets = sorted(season_dir.glob("possessions_*.parquet"))
        if include_playoffs:
            parquets = all_parquets
        else:
            parquets = [p for p in all_parquets if "playoffs" not in p.name]
        if not parquets:
            raise FileNotFoundError(
                f"No possession parquet found in {season_dir}. "
                f"Run collect-possessions --season {season} first."
            )
        for p in parquets:
            frames.append(pd.read_parquet(p))

    df = pd.concat(frames, ignore_index=True)
    return _build_matrix_from_df(df, player_vocab)


# ---------------------------------------------------------------------------
# Core: recency weights
# ---------------------------------------------------------------------------

def _recency_weights(df: pd.DataFrame, half_life_days: float) -> np.ndarray:
    """
    Compute per-possession recency weights.

    w_i = 0.5 ^ (days_ago_i / half_life_days)

    The most recent possession in df gets weight 1.0.

    Parameters
    ----------
    df             : DataFrame with a ``game_date`` column (str or datetime).
    half_life_days : Half-life in calendar days.

    Returns
    -------
    ndarray (n_poss,) float64 — weights in (0, 1].
    """
    dates = pd.to_datetime(df["game_date"])
    latest = dates.max()
    days_ago = (latest - dates).dt.total_seconds() / 86_400
    return np.power(0.5, days_ago.values / half_life_days)


def _competition_weights(
    df: pd.DataFrame,
    sigma: Optional[float] = None,
    p_target: float = 95.0,
    w_target: float = 0.05,
) -> np.ndarray:
    """
    Compute per-possession competition weights.

    Possessions played in blowouts carry less information about true player
    quality.  We down-weight by a Gaussian kernel on the absolute score
    differential:

        w_comp(d) = exp(−(d / σ)²)

    σ is calibrated so that w_comp(d_p) = w_target where d_p is the
    p_target-th percentile of |score_diff| in df:

        σ = d_p / √(−ln w_target)

    Defaults (p_target=95, w_target=0.05) set σ ≈ 13.9 given the empirical
    p95 ≈ 24 pts, yielding:
        d=0   → 1.00
        d=7   → 0.78  (median possession)
        d=12  → 0.47  (p75)
        d=19  → 0.16  (p90)
        d=24  → 0.05  (p95, ≈ "basically zero")

    Parameters
    ----------
    df       : DataFrame with a ``score_diff`` column (signed, offense − defense).
    sigma    : Explicit scale parameter (pts).  If None, auto-calibrated from df.
    p_target : Percentile at which w_comp = w_target (default 95).
    w_target : Target weight at d_p (default 0.05).

    Returns
    -------
    ndarray (n_poss,) float64 — weights in (0, 1].
    """
    d = np.abs(df["score_diff"].values).astype(np.float64)
    if sigma is None:
        d_p = float(np.percentile(d, p_target))
        sigma = d_p / np.sqrt(-np.log(w_target))
    return np.exp(-(d / sigma) ** 2)


# ---------------------------------------------------------------------------
# Core: solver
# ---------------------------------------------------------------------------

def solve_ridge(
    X: sp.csr_matrix,
    y: np.ndarray,
    alpha: float,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """
    Solve (optionally weighted) ridge regression via the normal equations.

    Minimises  Σ w_i (y_i − X_i β)² + alpha * ‖β‖²

    When ``weights`` is None every possession is weighted equally.

    Returns
    -------
    beta      : ndarray (2 * n_players,)
    intercept : float   (weighted mean points per possession)
    """
    if weights is None:
        intercept = float(y.mean())
        y_c = y - intercept

        t0 = time.time()
        XtX = (X.T @ X).toarray().astype(np.float64)
        Xty = np.asarray(X.T @ y_c).ravel().astype(np.float64)
        print(f"  X'X  {XtX.shape}  computed in {time.time() - t0:.1f}s")
    else:
        w = weights.astype(np.float64)
        intercept = float(np.average(y, weights=w))
        y_c = (y - intercept).astype(np.float64)

        t0 = time.time()
        # Weighted X'WX: scale each row of X by sqrt(w), then (XW)'(XW)
        w_sqrt = np.sqrt(w)
        Xw = X.multiply(w_sqrt[:, None])          # (n, 2p) sparse
        XtX = (Xw.T @ Xw).toarray().astype(np.float64)
        Xty = np.asarray(X.T @ (w * y_c)).ravel().astype(np.float64)
        print(f"  X'WX {XtX.shape}  computed in {time.time() - t0:.1f}s")

    beta = np.linalg.solve(XtX + alpha * np.eye(XtX.shape[0]), Xty)
    return beta, intercept


# ---------------------------------------------------------------------------
# Public: fit
# ---------------------------------------------------------------------------

def fit_rapm(
    data_dir: Path,
    seasons: Optional[list[str]] = None,
    player_vocab_path: Path = Path("data/player_vocab.parquet"),
    player_table_path: Path = Path("data/players.parquet"),
    alpha: float = 2000.0,
    min_poss: int = 100,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
    # Date-range alternative to seasons:
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    season_type: str = "Regular Season",
    pbp_cache_dir: Optional[Path] = None,
    max_workers: int = 4,
    # Recency weighting:
    half_life_days: Optional[float] = None,
    # Playoffs:
    include_playoffs: bool = False,
) -> pd.DataFrame:
    """
    Fit ridge RAPM and return a DataFrame of player estimates.

    Data source — specify exactly one of:
      • ``seasons`` — load from pre-collected season parquets in data_dir.
      • ``from_date`` + ``to_date`` — auto-fetch and cache individual games.

    Parameters
    ----------
    data_dir          : Root data directory.
    seasons           : Season labels to include, e.g. ["2022-23", "2021-22"].
    player_vocab_path : Path to player_vocab.parquet.
    player_table_path : Path to players.parquet (for player names).
    alpha             : Ridge penalty.  Higher = more shrinkage.  Default 2000.
    min_poss          : Minimum offensive AND defensive possessions to be
                        flagged as qualified (does not filter output rows).
    output_dir        : If given, saves rapm.parquet and rapm_summary.json.
    verbose           : Print progress.
    from_date         : Start of date range (ISO 8601, e.g. "2024-01-01").
    to_date           : End of date range (ISO 8601, e.g. "2024-03-31").
    season_type       : "Regular Season", "Playoffs", etc. (date-range mode).
    pbp_cache_dir     : Cache dir for raw pbpstats JSON (date-range mode).
    max_workers       : Parallel workers for fetching games (date-range mode).
    half_life_days    : Half-life for exponential recency weighting (days).
                        None = equal weighting.  E.g. 365 = 1-year half-life.
    include_playoffs  : If True, include playoff possessions alongside regular
                        season data.  Default False.

    Returns
    -------
    DataFrame with columns:
        player_id, player_idx, player_name,
        n_off, n_def, qualified,
        orapm, drapm, rapm          (all per 100 possessions)
    """
    using_seasons   = seasons is not None
    using_daterange = from_date is not None or to_date is not None

    if using_seasons and using_daterange:
        raise ValueError("Specify either 'seasons' or ('from_date' and 'to_date'), not both.")
    if not using_seasons and not using_daterange:
        raise ValueError("Specify either 'seasons' or ('from_date' and 'to_date').")
    if using_daterange and (from_date is None or to_date is None):
        raise ValueError("Both 'from_date' and 'to_date' must be provided together.")

    player_vocab = pd.read_parquet(player_vocab_path)
    player_table = pd.read_parquet(player_table_path)

    name_lookup = (
        player_table.sort_values("season")
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")["player_name"]
    )

    # ── Load / fetch possessions ──────────────────────────────────────────────
    if using_seasons:
        if verbose:
            print(f"Loading {len(seasons)} season(s)…")
        t0 = time.time()
        X, y, n_off, n_def = build_rapm_matrix(data_dir, seasons, player_vocab, include_playoffs=include_playoffs)
        df_for_weights = None   # no game_date column handy in this path
        if verbose:
            print(f"  {len(y):,} possessions  mean={y.mean():.3f}  std={y.std():.3f}"
                  f"  ({time.time()-t0:.1f}s)")
    else:
        from .data.season import collect_games_for_dates
        t0 = time.time()
        df_poss = collect_games_for_dates(
            from_date=from_date,
            to_date=to_date,
            data_dir=Path(data_dir),
            season_type=season_type,
            pbp_cache_dir=Path(pbp_cache_dir) if pbp_cache_dir else None,
            max_workers=max_workers,
            verbose=verbose,
        )
        if verbose:
            n_games = df_poss["game_id"].nunique() if len(df_poss) else 0
            print(f"  {len(df_poss):,} possessions from {n_games} games"
                  f"  ({time.time()-t0:.1f}s)")
        X, y, n_off, n_def = _build_matrix_from_df(df_poss, player_vocab)
        df_for_weights = df_poss if half_life_days is not None else None

    # ── Recency weights ───────────────────────────────────────────────────────
    weights = None
    if half_life_days is not None:
        if df_for_weights is None:
            # For season mode, we need game_date — load it
            if verbose:
                print(f"Computing recency weights (half_life={half_life_days} days)…")
            frames_for_dates: list[pd.DataFrame] = []
            for season in seasons:  # type: ignore[union-attr]
                season_dir = Path(data_dir) / season
                all_pq = sorted(season_dir.glob("possessions_*.parquet"))
                pq_list = all_pq if include_playoffs else [p for p in all_pq if "playoffs" not in p.name]
                for p in pq_list:
                    frames_for_dates.append(pd.read_parquet(p, columns=["game_date"]))
            df_dates = pd.concat(frames_for_dates, ignore_index=True)
            weights = _recency_weights(df_dates, half_life_days)
        else:
            if verbose:
                print(f"Computing recency weights (half_life={half_life_days} days)…")
            weights = _recency_weights(df_for_weights, half_life_days)

        if verbose:
            print(f"  weight range: {weights.min():.4f} – {weights.max():.4f}"
                  f"  (effective n ≈ {weights.sum():.0f})")

    # ── Solve ─────────────────────────────────────────────────────────────────
    if verbose:
        print(f"Solving ridge regression  alpha={alpha}…")
    beta, intercept = solve_ridge(X, y, alpha, weights=weights)
    if verbose:
        print(f"  intercept={intercept:.4f}")

    n_players = len(player_vocab) + 1
    alpha_off = beta[:n_players]
    delta_def = beta[n_players:]

    results = player_vocab.copy()
    results["player_name"] = results["player_id"].map(name_lookup).fillna("Unknown")
    results["n_off"]    = n_off[results["player_idx"].values]
    results["n_def"]    = n_def[results["player_idx"].values]
    results["qualified"] = (results["n_off"] >= min_poss) & (results["n_def"] >= min_poss)
    results["orapm"]    = alpha_off[results["player_idx"].values] * 100
    results["drapm"]    = -delta_def[results["player_idx"].values] * 100
    results["rapm"]     = results["orapm"] + results["drapm"]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_parquet(output_dir / "rapm.parquet", index=False)
        summary = {
            "alpha":               alpha,
            "intercept":           intercept,
            "n_possessions":       int(len(y)),
            "n_qualified_players": int(results["qualified"].sum()),
            "rapm_std":            float(results.loc[results["qualified"], "rapm"].std()),
            "half_life_days":      half_life_days,
        }
        (output_dir / "rapm_summary.json").write_text(json.dumps(summary, indent=2))
        if verbose:
            print(f"  Saved → {output_dir}/rapm.parquet")

    return results


# ---------------------------------------------------------------------------
# Public: load
# ---------------------------------------------------------------------------

def load_rapm(rapm_dir: Path) -> pd.DataFrame:
    """
    Load pre-computed RAPM results saved by fit_rapm.

    Parameters
    ----------
    rapm_dir : Directory containing rapm.parquet (and optionally rapm_summary.json).

    Returns
    -------
    DataFrame with columns player_id, player_name, orapm, drapm, rapm, …
    """
    path = Path(rapm_dir) / "rapm.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fit_rapm(..., output_dir='{rapm_dir}') "
            "or the solve-rapm CLI first."
        )
    return pd.read_parquet(path)
