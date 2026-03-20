"""
tests/test_sample_games.py — random game sampling across all collected seasons.

For each season in data/, we randomly sample SAMPLE_SIZE games and run every
structural invariant against them.  All checks read directly from the
pre-built season parquets — no network calls, no re-extraction, fully offline.

Why parquets and not re-extraction?
------------------------------------
The sampling test's job is to validate DATA QUALITY across the full corpus.
Re-extracting 120 games would require 120 calls to the NBA Stats API (for
game metadata), causing rate-limit failures.  The unit tests in
test_possessions.py already exercise the extraction pipeline itself on 6
representative games.  Here we trust the pipeline and check its output.

Usage
-----
Run the full sample (120 games, ~3 s):
    pytest tests/test_sample_games.py -v

Smoke test (5 games per season):
    pytest tests/test_sample_games.py -v --sample-size=5

Specific seed:
    pytest tests/test_sample_games.py -v --sample-seed=123

Adding invariants
-----------------
Add a new check_ function below and call it from _run_all_checks().
Each function receives a single-game DataFrame and the game_id string
and returns a list of error strings (empty = pass).
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd
import pytest

from conftest import PBP_CACHE


# ---------------------------------------------------------------------------
# CLI options (pytest plugin hooks in conftest or here via pytest_addoption)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Build the sample at collection time
# ---------------------------------------------------------------------------

def _build_sample(sample_size: int, seed: int) -> list[tuple[str, str]]:
    """Return list of (game_id, season_label) pairs, sorted for stable output."""
    rng = random.Random(seed)
    cases: list[tuple[str, str]] = []
    data_root = Path("data")
    if not data_root.exists():
        return cases
    for season_dir in sorted(data_root.iterdir()):
        parquets = sorted(season_dir.glob("possessions_*.parquet"))
        if not parquets:
            continue
        df = pd.read_parquet(parquets[0], columns=["game_id"])
        game_ids = df["game_id"].unique().tolist()
        n = min(sample_size, len(game_ids))
        sample = rng.sample(game_ids, n)
        season_label = season_dir.name
        cases.extend((gid, season_label) for gid in sample)
    return sorted(cases)


# Season parquet cache — loaded once per session, filtered per game
_parquet_store: dict[str, pd.DataFrame] = {}


def _get_parquet(season: str) -> pd.DataFrame:
    """Return (cached) full-season possession DataFrame."""
    if season not in _parquet_store:
        parquets = sorted(Path("data", season).glob("possessions_*.parquet"))
        if not parquets:
            raise FileNotFoundError(f"No parquet found for season {season}")
        _parquet_store[season] = pd.read_parquet(parquets[0])
    return _parquet_store[season]


def _get_game_df(game_id: str, season: str) -> pd.DataFrame:
    """Return possession DataFrame for a single game from the season parquet."""
    df = _get_parquet(season)
    return df[df["game_id"] == game_id].reset_index(drop=True)


# Evaluate lazily at pytest collection time via the fixture mechanism below.
# This avoids importing pandas at module load when the test file is merely
# being discovered by IDEs or import checkers.
_SAMPLE_CACHE: list[tuple[str, str]] | None = None


def _get_sample(config) -> list[tuple[str, str]]:
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE is None:
        size = config.getoption("--sample-size", default=20)
        seed = config.getoption("--sample-seed", default=42)
        _SAMPLE_CACHE = _build_sample(size, seed)
    return _SAMPLE_CACHE


# ---------------------------------------------------------------------------
# Per-invariant check functions
# ---------------------------------------------------------------------------

_PERIOD_BOUNDS = {
    1: (0, 720), 2: (720, 1440), 3: (1440, 2160), 4: (2160, 2880),
    5: (2880, 3180), 6: (3180, 3480), 7: (3480, 3780), 8: (3780, 4080),
}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _check_schema(df: pd.DataFrame, game_id: str) -> list[str]:
    errs = []
    required = {
        "game_id", "game_date", "period", "seconds_into_game",
        "offense_team_id", "defense_team_id",
        "offense_player_ids", "defense_player_ids",
        "points", "score_diff", "home_team_id",
    }
    missing = required - set(df.columns)
    if missing:
        errs.append(f"Missing columns: {missing}")
    if len(df) == 0:
        errs.append("DataFrame is empty")
    return errs


def _check_lineups(df: pd.DataFrame, game_id: str) -> list[str]:
    errs = []
    bad_off = (df["offense_player_ids"].apply(len) != 5).sum()
    bad_def = (df["defense_player_ids"].apply(len) != 5).sum()
    if bad_off:
        errs.append(f"{bad_off} possessions with != 5 offensive players")
    if bad_def:
        errs.append(f"{bad_def} possessions with != 5 defensive players")

    overlaps = df.apply(
        lambda r: len(set(r["offense_player_ids"]) & set(r["defense_player_ids"])) > 0,
        axis=1,
    ).sum()
    if overlaps:
        errs.append(f"{overlaps} possessions with player on both teams")

    dup_off = df["offense_player_ids"].apply(lambda ids: len(ids) != len(set(ids))).sum()
    dup_def = df["defense_player_ids"].apply(lambda ids: len(ids) != len(set(ids))).sum()
    if dup_off:
        errs.append(f"{dup_off} possessions with duplicate offensive player IDs")
    if dup_def:
        errs.append(f"{dup_def} possessions with duplicate defensive player IDs")
    return errs


def _check_teams(df: pd.DataFrame, game_id: str) -> list[str]:
    errs = []
    same = (df["offense_team_id"] == df["defense_team_id"]).sum()
    if same:
        errs.append(f"{same} possessions where offense_team_id == defense_team_id")
    n_teams = df["offense_team_id"].nunique()
    if n_teams != 2:
        errs.append(f"Expected 2 offensive teams, found {n_teams}: {df['offense_team_id'].unique()}")
    unique_home = df["home_team_id"].nunique()
    if unique_home != 1:
        errs.append(f"home_team_id takes {unique_home} distinct values")
    counts = df["offense_team_id"].value_counts(normalize=True)
    for tid, frac in counts.items():
        if not (0.30 <= frac <= 0.70):
            errs.append(f"Team {tid} has {frac:.1%} of possessions (expected 30–70%)")
    return errs


def _check_points(df: pd.DataFrame, game_id: str) -> list[str]:
    errs = []
    bad = ((df["points"] < 0) | (df["points"] > 8)).sum()
    if bad:
        errs.append(f"{bad} possessions with points outside [0, 8]")
    return errs


def _check_score_diff(df: pd.DataFrame, game_id: str) -> list[str]:
    errs = []
    first = df.iloc[0]["score_diff"]
    if abs(first) > 4:
        # Allow up to a 4-point pre-game offset (pre-game technical fouls can
        # award 1-2 free throws before the first tracked lineup event).
        errs.append(f"First possession score_diff={first}, expected near 0")

    home_id = df["home_team_id"].iloc[0]
    home_pts = 0
    away_pts = 0
    for _, row in df.iterrows():
        off = row["offense_team_id"]
        expected = (home_pts - away_pts) if off == home_id else (away_pts - home_pts)
        actual = int(row["score_diff"])
        if abs(actual - expected) > 8:
            errs.append(
                f"score_diff={actual} but reconstruction gives {expected} "
                f"(diff={actual - expected}) at period={row['period']} t={row['seconds_into_game']}"
            )
            break  # report first violation only
        if off == home_id:
            home_pts += row["points"]
        else:
            away_pts += row["points"]

    if df["score_diff"].abs().max() > 60:
        errs.append(f"score_diff exceeds ±60: max={df['score_diff'].abs().max()}")
    return errs


def _check_time(df: pd.DataFrame, game_id: str) -> list[str]:
    errs = []
    if not (df["seconds_into_game"] >= 0).all():
        errs.append("Negative seconds_into_game found")

    # Period / seconds consistency
    for _, row in df.iterrows():
        p = int(row["period"])
        s = row["seconds_into_game"]
        lo, hi = _PERIOD_BOUNDS.get(p, (None, None))
        if lo is None:
            continue
        if not (lo <= s <= hi):
            errs.append(f"period={p} but seconds_into_game={s:.1f} not in [{lo},{hi}]")
            break

    # All four regulation periods present
    periods = set(df["period"].unique())
    for p in (1, 2, 3, 4):
        if p not in periods:
            errs.append(f"Period {p} has no possessions")

    # No large within-period gaps
    MAX_GAP = 90.0
    df_s = df.sort_values(["period", "seconds_into_game"])
    for period, grp in df_s.groupby("period"):
        times = grp["seconds_into_game"].values
        for i in range(len(times) - 1):
            gap = times[i + 1] - times[i]
            if gap > MAX_GAP:
                errs.append(f"period={period} gap={gap:.1f}s at t={times[i]:.1f}")
                break

    return errs


def _check_duplicates(df: pd.DataFrame, game_id: str) -> list[str]:
    errs = []
    check = df.copy()
    check["offense_player_ids"] = check["offense_player_ids"].apply(tuple)
    check["defense_player_ids"] = check["defense_player_ids"].apply(tuple)
    # Only flag consecutively adjacent identical rows — those indicate the
    # extraction pipeline emitted the same possession twice in sequence.
    # Non-adjacent identical rows (same team, same lineup, same points at the
    # same clock tick) can be legitimate rapid-turnover sequences where the
    # same team regains possession within a single clock resolution window.
    shifted = check.shift(1)
    consecutive_dupes = (check == shifted).all(axis=1).sum()
    if consecutive_dupes:
        errs.append(f"{consecutive_dupes} consecutively adjacent identical possession rows")
    return errs


def _check_metadata(df: pd.DataFrame, game_id: str) -> list[str]:
    errs = []
    bad_dates = (~df["game_date"].astype(str).str.match(_DATE_RE)).sum()
    if bad_dates:
        errs.append(f"{bad_dates} rows with malformed game_date")
    bad_game_id = (df["game_id"] != game_id).sum()
    if bad_game_id:
        errs.append(f"{bad_game_id} rows with unexpected game_id value")
    return errs


def _run_all_checks(df: pd.DataFrame, game_id: str) -> list[str]:
    """Collect all invariant violations for one game. Returns list of error strings."""
    violations: list[str] = []
    for check_fn in (
        _check_schema,
        _check_lineups,
        _check_teams,
        _check_points,
        _check_score_diff,
        _check_time,
        _check_duplicates,
        _check_metadata,
    ):
        violations.extend(check_fn(df, game_id))
    return violations


# ---------------------------------------------------------------------------
# Parametrised test — one case per sampled game
# ---------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    """Dynamically parametrize test_sampled_game with the season sample."""
    if "game_id" in metafunc.fixturenames and "season" in metafunc.fixturenames:
        sample = _get_sample(metafunc.config)
        metafunc.parametrize(
            "game_id,season",
            sample,
            ids=[f"{season}/{gid}" for gid, season in sample],
        )


def test_sampled_game(game_id: str, season: str) -> None:
    """Run all structural invariants on a randomly sampled game.

    Reads from the pre-built season parquet — fully offline, no extraction,
    no NBA Stats API calls.  Every violation is reported together so a single
    re-run surfaces the full picture.
    """
    df = _get_game_df(game_id, season)
    violations = _run_all_checks(df, game_id)
    assert not violations, (
        f"Game {game_id} ({season}) — {len(violations)} violation(s):\n"
        + "\n".join(f"  • {v}" for v in violations)
    )
