"""
stats.py — per-season per-player box and bio stats from the NBA Stats API.

Fetches two complementary datasets and merges them into a single parquet:

  LeagueDashPlayerStats (PerGame)
      MIN, PTS, REB, AST, STL, BLK, TOV, FG3A, FGA, FT_PCT

  LeagueDashPlayerBioStats
      PLAYER_HEIGHT_INCHES, PLAYER_WEIGHT, AGE

Usage
-----
    from deep_rapm.data.stats import build_player_stats_table

    df = build_player_stats_table(
        seasons=["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
        stats_cache_dir=Path("data/stats_cache"),
        output_path=Path("data/player_stats.parquet"),
    )

CLI:
    collect-player-stats --seasons 2018-19 2019-20 2020-21 2021-22 2022-23
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Columns we keep from each endpoint
# ---------------------------------------------------------------------------

# LeagueDashPlayerStats (PerGame) — rate stats
_BOX_COLS = ["PLAYER_ID", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
             "FG3A", "FGA", "FT_PCT"]

# LeagueDashPlayerBioStats — physical and demographic attributes
_BIO_COLS = ["PLAYER_ID", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "AGE"]


# ---------------------------------------------------------------------------
# Single-season fetchers
# ---------------------------------------------------------------------------

def fetch_box_stats(
    season: str,
    stats_cache_dir: Optional[Path] = None,
    max_retries: int = 3,
    sleep_s: float = 0.6,
) -> pd.DataFrame:
    """Fetch per-game box stats for *season* via LeagueDashPlayerStats."""
    if stats_cache_dir is not None:
        cache = Path(stats_cache_dir) / f"box_{season}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)

    from nba_api.stats.endpoints import leaguedashplayerstats

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            ep = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_simple="PerGame",
                timeout=60,
            )
            df = ep.get_data_frames()[0]
            time.sleep(sleep_s)
            break
        except Exception:
            if attempt == max_retries - 1:
                raise

    keep = [c for c in _BOX_COLS if c in df.columns]
    result = df[keep].copy()
    result["season"] = season

    if stats_cache_dir is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache, index=False)

    return result


def fetch_bio_stats(
    season: str,
    stats_cache_dir: Optional[Path] = None,
    max_retries: int = 3,
    sleep_s: float = 0.6,
) -> pd.DataFrame:
    """Fetch player height/weight/age for *season* via LeagueDashPlayerBioStats."""
    if stats_cache_dir is not None:
        cache = Path(stats_cache_dir) / f"bio_{season}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)

    from nba_api.stats.endpoints import leaguedashplayerbiostats

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            ep = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                season=season,
                per_mode_simple="PerGame",
                timeout=60,
            )
            df = ep.get_data_frames()[0]
            time.sleep(sleep_s)
            break
        except Exception:
            if attempt == max_retries - 1:
                raise

    keep = [c for c in _BIO_COLS if c in df.columns]
    result = df[keep].copy()
    result["season"] = season

    if stats_cache_dir is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache, index=False)

    return result


# ---------------------------------------------------------------------------
# Multi-season table builder
# ---------------------------------------------------------------------------

def build_player_stats_table(
    seasons: list[str],
    stats_cache_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Fetch box + bio stats for all *seasons* and merge into a single table.

    Output columns
    --------------
    player_id, season,
    min, pts, reb, ast, stl, blk, tov,      (PerGame)
    fg3a_rate,                               (FG3A / FGA — floor-spacing proxy)
    ft_pct,
    height_inches, weight_lbs, age          (from bio endpoint)

    All column names are lowercased for consistency.
    Missing values are left as NaN; callers should fill them when building
    the feature tensor.

    Parameters
    ----------
    seasons          : List of NBA season strings, e.g. ["2022-23", "2023-24"].
    stats_cache_dir  : Directory for per-season raw parquet caches.
    output_path      : If provided, write the merged table here.
    overwrite        : If False and output_path exists, load and return it.
    """
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            return pd.read_parquet(output_path)

    box_frames: list[pd.DataFrame] = []
    bio_frames: list[pd.DataFrame] = []

    for season in seasons:
        print(f"  Fetching stats for {season}…", flush=True)
        box_frames.append(fetch_box_stats(season, stats_cache_dir))
        bio_frames.append(fetch_bio_stats(season, stats_cache_dir))

    box = pd.concat(box_frames, ignore_index=True)
    bio = pd.concat(bio_frames, ignore_index=True)

    # Merge on (PLAYER_ID, season)
    merged = box.merge(bio, on=["PLAYER_ID", "season"], how="left")

    # Derived feature: 3-point attempt rate (floor-spacing proxy)
    if "FG3A" in merged.columns and "FGA" in merged.columns:
        merged["FG3A_RATE"] = (merged["FG3A"] / merged["FGA"].replace(0, float("nan")))
        merged.drop(columns=["FG3A", "FGA"], inplace=True)

    # Lowercase column names, rename for clarity
    rename = {
        "PLAYER_ID":             "player_id",
        "MIN":                   "min",
        "PTS":                   "pts",
        "REB":                   "reb",
        "AST":                   "ast",
        "STL":                   "stl",
        "BLK":                   "blk",
        "TOV":                   "tov",
        "FG3A_RATE":             "fg3a_rate",
        "FT_PCT":                "ft_pct",
        "PLAYER_HEIGHT_INCHES":  "height_inches",
        "PLAYER_WEIGHT":         "weight_lbs",
        "AGE":                   "age",
    }
    merged.rename(columns=rename, inplace=True)
    merged["player_id"] = merged["player_id"].astype(int)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_path, index=False)
        print(f"Saved {len(merged):,} rows → {output_path}")

    return merged
