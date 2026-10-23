"""
box_scores.py — per-game, per-player box score collection and caching.

Each game is cached as data/box/<game_id>.parquet.  The schema matches
the raw columns from boxscoretraditionalv3 plus derived per-minute rates
needed by the feature EWMA.

Per-player columns written to cache
------------------------------------
    player_id       int
    team_id         int
    min_played      float   — minutes parsed from "MM:SS"
    fgm, fga        int
    fg3m, fg3a      int
    ftm, fta        int
    reb, ast, stl, blk, tov   int
    pts             int

Derived at feature-build time (not stored):
    fg2m = fgm - fg3m
    fg2a = fga - fg3a

Usage
-----
    from deep_rapm.data.box_scores import collect_box_scores_for_seasons

    collect_box_scores_for_seasons(
        seasons=["2022-23", "2023-24"],
        data_dir=Path("data"),
        box_dir=Path("data/box"),
    )
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# MIN string parser:  "30:15"  →  30.25   ("PT30M15.00S" also handled)
# ---------------------------------------------------------------------------

import re
_MIN_RE = re.compile(r"(?:PT)?(\d+)M([\d.]+)S")


def _parse_min(s: str) -> float:
    """Parse NBA minute string to float minutes."""
    if not s or s in ("", "None", "null"):
        return 0.0
    s = str(s).strip()
    m = _MIN_RE.match(s)
    if m:
        return int(m.group(1)) + float(m.group(2)) / 60
    # Fallback: MM:SS
    parts = s.split(":")
    if len(parts) == 2:
        return float(parts[0]) + float(parts[1]) / 60
    return 0.0


# ---------------------------------------------------------------------------
# Column rename map for boxscoretraditionalv3 (v3 API uses camelCase names)
# ---------------------------------------------------------------------------

# v3 API column → our canonical column name
_V3_RENAME = {
    "personId":              "player_id",
    "teamId":                "team_id",
    "minutes":               "min_played",
    "fieldGoalsMade":        "fgm",
    "fieldGoalsAttempted":   "fga",
    "threePointersMade":     "fg3m",
    "threePointersAttempted":"fg3a",
    "freeThrowsMade":        "ftm",
    "freeThrowsAttempted":   "fta",
    "reboundsTotal":         "reb",
    "assists":               "ast",
    "steals":                "stl",
    "blocks":                "blk",
    "turnovers":             "tov",
    "points":                "pts",
}
_INT_OUT = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
            "reb", "ast", "stl", "blk", "tov", "pts"]


# ---------------------------------------------------------------------------
# Single-game fetch (main process — not subprocess safe unless imports are ok)
# ---------------------------------------------------------------------------

def fetch_game_box_score(
    game_id: str,
    box_dir: Optional[Path] = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch per-player box score stats for one game.

    Returns a DataFrame with columns:
        player_id, team_id, min_played,
        fgm, fga, fg3m, fg3a, ftm, fta,
        reb, ast, stl, blk, tov, pts

    Empty DataFrame if the game has no usable rows.

    Results are cached to ``box_dir/<game_id>.parquet`` when *box_dir* is set.
    """
    if box_dir is not None:
        cache = Path(box_dir) / f"{game_id}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)

    from nba_api.stats.endpoints import boxscoretraditionalv3

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            ep = boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=game_id, timeout=60
            )
            raw = ep.player_stats.get_data_frame()
            break
        except Exception:
            if attempt == max_retries - 1:
                raise

    # Keep only columns we have a rename for
    keep = [c for c in _V3_RENAME if c in raw.columns]
    df = raw[keep].rename(columns=_V3_RENAME).copy()

    df["min_played"] = df["min_played"].map(_parse_min)

    for col in _INT_OUT:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df[df["min_played"] > 0].reset_index(drop=True)   # drop DNP rows

    if box_dir is not None and len(df) > 0:
        Path(box_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)

    return df


# ---------------------------------------------------------------------------
# Subprocess worker (mirrors _process_game in season.py)
# ---------------------------------------------------------------------------

def _worker(game_id: str, box_dir_str: str, delay: float = 0.0) -> tuple[str, Optional[pd.DataFrame], Optional[str]]:
    try:
        if delay > 0:
            time.sleep(delay)
        df = fetch_game_box_score(game_id, box_dir=Path(box_dir_str))
        return (game_id, df, None)
    except Exception:
        return (game_id, None, traceback.format_exc())


# ---------------------------------------------------------------------------
# Bulk collection
# ---------------------------------------------------------------------------

def collect_box_scores_for_games(
    game_ids: list[str],
    box_dir: Path,
    max_workers: int = 1,
    request_delay: float = 0.6,
    verbose: bool = True,
) -> None:
    """Fetch and cache box scores for all *game_ids* not already in *box_dir*.

    Parameters
    ----------
    max_workers:
        Parallel workers.  Keep at 1 (default) to avoid NBA API rate limits;
        increase only on a fast connection with low timeouts.
    request_delay:
        Seconds to sleep before each API call (per worker).  The default 0.6s
        keeps the single-worker request rate under ~100/min.
    """
    box_dir = Path(box_dir)
    box_dir.mkdir(parents=True, exist_ok=True)

    todo = [g for g in game_ids if not (box_dir / f"{g}.parquet").exists()]
    if verbose:
        cached = len(game_ids) - len(todo)
        print(f"  Box scores: {cached} cached, fetching {len(todo)} new")

    if not todo:
        return

    failed: list[str] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker, g, str(box_dir), request_delay): g
            for g in todo
        }
        with tqdm(total=len(todo), desc="Box scores", disable=not verbose) as pbar:
            for future in as_completed(futures):
                gid = futures[future]
                try:
                    _, df, error = future.result()
                    if error:
                        tqdm.write(f"  ✗ {gid}: {error.splitlines()[-1]}")
                        failed.append(gid)
                except Exception as exc:
                    tqdm.write(f"  ✗ {gid}: {exc}")
                    failed.append(gid)
                pbar.update(1)

    if failed and verbose:
        print(f"  {len(failed)} games failed: {failed[:5]}")


def collect_box_scores_for_seasons(
    seasons: list[str],
    data_dir: Path,
    box_dir: Path,
    season_type: str = "Regular Season",
    max_workers: int = 1,
    request_delay: float = 0.6,
    verbose: bool = True,
) -> None:
    """
    Collect and cache box scores for all games in *seasons*.

    Reuses the possession parquets in *data_dir* to get game IDs (no extra
    API call for the game list if possessions are already collected).
    Falls back to LeagueGameLog if possession parquets are not found.
    """
    game_ids: list[str] = []

    for season in seasons:
        season_dir = Path(data_dir) / season
        parquets = sorted(season_dir.glob("possessions_*.parquet"))
        if parquets:
            df = pd.read_parquet(parquets[0], columns=["game_id"])
            game_ids.extend(df["game_id"].unique().tolist())
        else:
            if verbose:
                print(f"  No possessions found for {season} — fetching game list from API")
            from .season import get_season_game_ids
            game_ids.extend(get_season_game_ids(season, season_type))

    game_ids = list(dict.fromkeys(game_ids))   # deduplicate, preserve order
    if verbose:
        print(f"Collecting box scores for {len(game_ids)} games across {len(seasons)} seasons")
    collect_box_scores_for_games(
        game_ids, box_dir,
        max_workers=max_workers,
        request_delay=request_delay,
        verbose=verbose,
    )
