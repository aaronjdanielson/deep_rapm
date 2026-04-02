"""
rolling.py — Incremental rolling RAPM (public API).

Fits RAPM at a sequence of evaluation dates using incremental Gram matrix
updates.  Instead of rebuilding X'WX from all possessions at every date,
IncrementalGramState maintains G and b as running state, advancing them
forward in time at O(k·P²) per date step (where k ≈ possessions/day) vs
the O(n·P²) cost of a full recompute.

See docs/incremental_rapm.md for design rationale and algorithmic details.

Public API
----------
    fit_rolling_rapm(
        data_dir, seasons, player_vocab_path, player_table_path,
        step_days, half_life_days, alpha, min_poss,
    ) -> pd.DataFrame

    IncrementalGramState  (also exported for testing / advanced use)

Example
-------
    from pathlib import Path
    from deep_rapm import fit_rolling_rapm

    df = fit_rolling_rapm(
        data_dir=Path("data"),
        seasons=["2022-23", "2023-24", "2024-25"],
        player_vocab_path=Path("data/player_vocab.parquet"),
        player_table_path=Path("data/players.parquet"),
        step_days=7,
        half_life_days=365,
        alpha=2000,
    )
    # df columns: date, player_id, player_name, orapm, drapm, rapm, n_off, n_def
    print(df[df["date"] == df["date"].max()].nlargest(10, "rapm"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .rapm import _build_matrix_from_df, _competition_weights, solve_ridge


# ---------------------------------------------------------------------------
# Incremental Gram matrix state
# ---------------------------------------------------------------------------

class IncrementalGramState:
    """
    Incrementally maintains the weighted normal equations for ridge RAPM.

    Under exponential recency weighting

        w_i(t) = comp_i * 0.5^(days_ago_i / H)

    advancing the evaluation date by Δ days multiplies all existing weights
    by γ = 0.5^(Δ/H).  The Gram matrix G = X'WX and right-hand side
    b_y = X'Wy therefore update as:

        G(t+Δ)   = γ · G(t)   + X_new' W_new X_new
        b_y(t+Δ) = γ · b_y(t) + X_new' W_new y_new

    Each date step costs O(k·P²) where k is possessions in new games —
    roughly 650× fewer ops than rebuilding from the full history.

    Usage pattern::

        state = IncrementalGramState(2 * n_players, alpha, half_life_days)
        for game_date, X, y, n_off, n_def, score_diff in game_batches:
            state.advance_to(game_date)          # scale existing state
            state.ingest(X, y, n_off, n_def,     # add new possessions
                         comp_sigma=sigma, score_diff=score_diff)
        beta, intercept = state.solve()
    """

    def __init__(self, n_cols: int, alpha: float, half_life_days: float) -> None:
        self.alpha          = alpha
        self.half_life_days = half_life_days
        self.G              = np.zeros((n_cols, n_cols), dtype=np.float64)
        self.b_y            = np.zeros(n_cols, dtype=np.float64)
        self.b_1            = np.zeros(n_cols, dtype=np.float64)
        self.total_w        = 0.0
        self.weighted_sum_y = 0.0
        # Raw (unweighted) possession counts — used only for qualified filter
        self.n_off          = np.zeros(n_cols // 2, dtype=np.int64)
        self.n_def          = np.zeros(n_cols // 2, dtype=np.int64)
        self.current_date: pd.Timestamp | None = None

    def advance_to(self, new_date: pd.Timestamp) -> None:
        """
        Scale all accumulated state by γ = 0.5^(Δ/H) for a date advance.

        Call this before ingesting possessions for a new game date so that
        existing state is decayed to the correct recency before new data
        is added at weight 1.0.
        """
        if self.current_date is None:
            self.current_date = new_date
            return
        delta = (new_date - self.current_date).days
        if delta <= 0:
            return
        gamma                = 0.5 ** (delta / self.half_life_days)
        self.G              *= gamma
        self.b_y            *= gamma
        self.b_1            *= gamma
        self.total_w        *= gamma
        self.weighted_sum_y *= gamma
        self.current_date    = new_date

    def ingest(
        self,
        X: sp.csr_matrix,
        y: np.ndarray,
        n_off_counts: np.ndarray,
        n_def_counts: np.ndarray,
        comp_sigma: Optional[float] = None,
        score_diff: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add one batch of possessions at the current date.

        All possessions in this batch receive recency weight 1.0 at ingestion;
        future calls to advance_to() will decay them correctly.  Call
        advance_to(game_date) before calling this method.

        Parameters
        ----------
        X            : (n_poss, 2·P) sparse indicator matrix.
        y            : (n_poss,) points scored.
        n_off_counts : (P,) raw offensive possession counts for this batch.
        n_def_counts : (P,) raw defensive possession counts for this batch.
        comp_sigma   : Competition-weight scale (pts).  If None, all weights = 1.
        score_diff   : (n_poss,) signed score differential for competition weights.
        """
        if comp_sigma is not None and score_diff is not None:
            w = np.exp(-(np.abs(score_diff) / comp_sigma) ** 2).astype(np.float64)
        else:
            w = np.ones(len(y), dtype=np.float64)

        w_sqrt   = np.sqrt(w)
        Xw       = X.multiply(w_sqrt[:, None])
        self.G   += (Xw.T @ Xw).toarray()
        self.b_y += np.asarray(X.T @ (w * y)).ravel()
        self.b_1 += np.asarray(X.T @ w).ravel()
        self.total_w        += float(w.sum())
        self.weighted_sum_y += float((w * y).sum())

        self.n_off += n_off_counts
        self.n_def += n_def_counts

    def solve(self) -> tuple[np.ndarray, float]:
        """
        Solve the current ridge system and return (beta, intercept).

        Minimises  Σ w_i (y_i − μ − X_i β)² + α ‖β‖²

        Returns
        -------
        beta      : ndarray (2·P,)
        intercept : float
        """
        mu   = self.weighted_sum_y / self.total_w if self.total_w > 0 else 0.0
        Xty  = self.b_y - mu * self.b_1
        beta = np.linalg.solve(self.G + self.alpha * np.eye(len(self.G)), Xty)
        return beta, mu


# ---------------------------------------------------------------------------
# Public: fit_rolling_rapm
# ---------------------------------------------------------------------------

def fit_rolling_rapm(
    data_dir: Path,
    seasons: list[str],
    player_vocab_path: Path = Path("data/player_vocab.parquet"),
    player_table_path: Path = Path("data/players.parquet"),
    step_days: int = 7,
    half_life_days: float = 365.0,
    alpha: float = 2000.0,
    min_poss: int = 100,
    warmup_days: int = 180,
    include_playoffs: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fit RAPM at a sequence of evaluation dates using incremental Gram updates.

    Parameters
    ----------
    data_dir          : Root data directory containing per-season subdirectories.
    seasons           : Season labels to include, e.g. ["2022-23", "2023-24"].
    player_vocab_path : Path to player_vocab.parquet.
    player_table_path : Path to players.parquet (for player names).
    step_days         : Days between consecutive evaluation dates.
    half_life_days    : Exponential recency weighting half-life (days).
    alpha             : Ridge penalty.  Higher = more shrinkage.
    min_poss          : Minimum offensive AND defensive possessions to include
                        a player in the output at a given date.
    warmup_days       : Skip the first N days (too few possessions for stable
                        estimates).
    include_playoffs  : If True, include playoff possessions.
    verbose           : Print progress.

    Returns
    -------
    Long-format DataFrame with columns:
        date, player_id, player_name, orapm, drapm, rapm, n_off, n_def

    All RAPM values are per 100 possessions.
    """
    data_dir          = Path(data_dir)
    player_vocab_path = Path(player_vocab_path)
    player_table_path = Path(player_table_path)

    player_vocab = pd.read_parquet(player_vocab_path)
    player_table = pd.read_parquet(player_table_path)
    name_lookup  = (
        player_table.sort_values("season")
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")["player_name"]
    )
    n_players = len(player_vocab) + 1

    # ── Load all possessions ──────────────────────────────────────────────────
    if verbose:
        print("Loading possessions…")
    frames: list[pd.DataFrame] = []
    for season in seasons:
        season_dir   = data_dir / season
        all_parquets = sorted(season_dir.glob("possessions_*.parquet"))
        parquets     = all_parquets if include_playoffs else [
            p for p in all_parquets if "playoffs" not in p.name
        ]
        if not parquets:
            ckpt = season_dir / "checkpoints" / "checkpoint.parquet"
            if ckpt.exists():
                if verbose:
                    print(f"  {season}: using in-progress checkpoint")
                parquets = [ckpt]
            else:
                if verbose:
                    print(f"  Warning: no possessions found for {season} — skipping")
                continue
        for pq in parquets:
            frames.append(pd.read_parquet(pq))

    all_poss = pd.concat(frames, ignore_index=True)
    all_poss["game_date"] = pd.to_datetime(all_poss["game_date"])
    all_poss = all_poss.sort_values("game_date").reset_index(drop=True)

    first_date = all_poss["game_date"].min()
    last_date  = all_poss["game_date"].max()
    if verbose:
        print(f"  {len(all_poss):,} possessions  {first_date.date()} → {last_date.date()}")

    # ── Competition-weight sigma (fixed once from all data) ───────────────────
    has_score_diff = "score_diff" in all_poss.columns
    comp_sigma: Optional[float] = None
    if has_score_diff:
        d_p95      = float(np.percentile(np.abs(all_poss["score_diff"].values), 95))
        comp_sigma = d_p95 / np.sqrt(-np.log(0.05))
        if verbose:
            print(f"  competition σ = {comp_sigma:.2f} pts  (p95 score diff = {d_p95:.1f})")

    # ── Pre-build per-game-date sparse matrices ───────────────────────────────
    if verbose:
        print("Building per-game-date matrices…")
    game_date_data: list[tuple] = []
    for gd, grp in all_poss.groupby("game_date"):
        X_g, y_g, n_off_g, n_def_g = _build_matrix_from_df(grp, player_vocab)
        sd_g = grp["score_diff"].values.astype(np.float64) if has_score_diff else None
        game_date_data.append((pd.Timestamp(gd), X_g, y_g, n_off_g, n_def_g, sd_g))
    game_date_data.sort(key=lambda t: t[0])
    if verbose:
        print(f"  {len(game_date_data)} unique game dates")

    # ── Evaluation dates ──────────────────────────────────────────────────────
    eval_dates = pd.date_range(
        start=first_date + pd.Timedelta(days=warmup_days),
        end=last_date,
        freq=f"{step_days}D",
    )
    if verbose:
        print(f"\nFitting RAPM at {len(eval_dates)} dates "
              f"(step={step_days}d, half-life={half_life_days:.0f}d, incremental)…")

    # ── Incremental solve ─────────────────────────────────────────────────────
    state       = IncrementalGramState(2 * n_players, alpha, half_life_days)
    next_gd_idx = 0
    records: list[dict] = []

    for i, eval_date in enumerate(eval_dates):
        while (next_gd_idx < len(game_date_data)
               and game_date_data[next_gd_idx][0] <= eval_date):
            gd, X_g, y_g, n_off_g, n_def_g, sd_g = game_date_data[next_gd_idx]
            state.advance_to(gd)
            state.ingest(X_g, y_g, n_off_g, n_def_g,
                         comp_sigma=comp_sigma, score_diff=sd_g)
            next_gd_idx += 1

        if state.total_w == 0:
            continue

        beta, _   = state.solve()
        alpha_off = beta[:n_players]
        delta_def = beta[n_players:]

        for _, row in player_vocab.iterrows():
            idx = int(row["player_idx"])
            if state.n_off[idx] < min_poss and state.n_def[idx] < min_poss:
                continue
            records.append({
                "date":        eval_date,
                "player_id":   int(row["player_id"]),
                "player_name": name_lookup.get(int(row["player_id"]), "Unknown"),
                "orapm":       float(alpha_off[idx] * 100),
                "drapm":       float(-delta_def[idx] * 100),
                "rapm":        float((alpha_off[idx] - delta_def[idx]) * 100),
                "n_off":       int(state.n_off[idx]),
                "n_def":       int(state.n_def[idx]),
            })

        if verbose and (
            (i + 1) % max(1, len(eval_dates) // 20) == 0
            or i == len(eval_dates) - 1
        ):
            pct = 100 * (i + 1) / len(eval_dates)
            print(f"  [{pct:5.1f}%]  {eval_date.date()}")

    df = pd.DataFrame(records)
    if verbose:
        print(f"\nDone. {df['player_id'].nunique()} players tracked across "
              f"{df['date'].nunique()} dates.")
    return df
