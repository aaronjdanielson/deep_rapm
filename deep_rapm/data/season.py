"""
season.py — parallel, checkpointed season-level possession collection.

Usage
-----
From Python:
    from deep_rapm.data import collect_season
    df = collect_season("2023-24", output_dir=Path("data"))

    # Date-range variant (auto-fetches and caches per game):
    from deep_rapm.data.season import collect_games_for_dates
    df = collect_games_for_dates("2024-01-01", "2024-03-31", data_dir=Path("data"))

From the CLI:
    collect-possessions --season 2023-24
"""

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from tqdm.auto import tqdm

from .game import get_game_possessions


# ---------------------------------------------------------------------------
# Game list — by season
# ---------------------------------------------------------------------------

def get_season_game_ids(
    season: str,
    season_type: str = "Regular Season",
    max_retries: int = 3,
) -> list[str]:
    """Return deduplicated game IDs for a season in chronological order."""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            gl = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star=season_type
            )
            games = gl.get_data_frames()[0]
            return list(games["GAME_ID"].drop_duplicates())
        except Exception:
            if attempt == max_retries - 1:
                raise
    return []


# ---------------------------------------------------------------------------
# Game list — by date range
# ---------------------------------------------------------------------------

def _date_to_nba_season(date_str: str) -> str:
    """Convert YYYY-MM-DD to NBA season string like '2023-24'.

    NBA seasons start in October: Oct–Dec of year Y belongs to season Y-(Y+1).
    Jan–Sep of year Y belongs to season (Y-1)-Y.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start_year = dt.year if dt.month >= 10 else dt.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _seasons_in_range(from_date: str, to_date: str) -> list[str]:
    """Return all NBA season strings that overlap [from_date, to_date]."""
    from_season = _date_to_nba_season(from_date)
    to_season   = _date_to_nba_season(to_date)
    seasons = []
    season = from_season
    while True:
        seasons.append(season)
        if season == to_season:
            break
        start_year = int(season.split("-")[0]) + 1
        season = f"{start_year}-{str(start_year + 1)[-2:]}"
    return seasons


def get_game_ids_for_dates(
    from_date: str,
    to_date: str,
    season_type: str = "Regular Season",
    max_retries: int = 3,
) -> list[str]:
    """Return deduplicated game IDs for games played in [from_date, to_date].

    Parameters
    ----------
    from_date, to_date : str
        ISO 8601 date strings, e.g. ``"2024-01-01"``.
    season_type : str
        ``"Regular Season"``, ``"Playoffs"``, etc.

    Returns
    -------
    list[str]
        Deduplicated game IDs.
    """
    nba_from = datetime.strptime(from_date, "%Y-%m-%d").strftime("%m/%d/%Y")
    nba_to   = datetime.strptime(to_date,   "%Y-%m-%d").strftime("%m/%d/%Y")

    all_ids: list[str] = []
    for season in _seasons_in_range(from_date, to_date):
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                gl = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star=season_type,
                    date_from_nullable=nba_from,
                    date_to_nullable=nba_to,
                )
                games = gl.get_data_frames()[0]
                all_ids.extend(games["GAME_ID"].drop_duplicates().tolist())
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise

    # Deduplicate while preserving order
    seen: set[str] = set()
    return [g for g in all_ids if not (g in seen or seen.add(g))]  # type: ignore[func-returns-value]


# ---------------------------------------------------------------------------
# Single-game worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _process_game(
    game_id: str,
    pbp_cache_dir: Optional[str],
) -> tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    Worker function for ProcessPoolExecutor.

    Returns (game_id, df_or_None, error_str_or_None).
    """
    try:
        records = get_game_possessions(
            game_id,
            pbp_cache_dir=Path(pbp_cache_dir) if pbp_cache_dir else None,
        )
        if not records:
            return (game_id, pd.DataFrame(), None)
        return (game_id, pd.DataFrame(records), None)
    except Exception:
        return (game_id, None, traceback.format_exc())


# ---------------------------------------------------------------------------
# Date-range collection (game-level cache)
# ---------------------------------------------------------------------------

def collect_games_for_dates(
    from_date: str,
    to_date: str,
    data_dir: Path = Path("data"),
    season_type: str = "Regular Season",
    pbp_cache_dir: Optional[Path] = None,
    max_workers: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Collect possession data for games played in [from_date, to_date].

    Each game is cached individually at ``data_dir/games/<game_id>.parquet``
    so subsequent calls reuse already-fetched games without re-fetching.

    Parameters
    ----------
    from_date, to_date : str
        ISO 8601 date strings, e.g. ``"2024-01-01"``.
    data_dir : Path
        Root data directory.  Game cache goes in ``data_dir/games/``.
    season_type : str
        ``"Regular Season"``, ``"Playoffs"``, etc.
    pbp_cache_dir : Path, optional
        Cache directory for raw pbpstats JSON files.
    max_workers : int
        Parallel workers for fetching new games.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Combined possession DataFrame sorted by (game_date, game_id,
        seconds_into_game).
    """
    games_cache_dir = Path(data_dir) / "games"
    games_cache_dir.mkdir(parents=True, exist_ok=True)
    if pbp_cache_dir is not None:
        Path(pbp_cache_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Fetching game list for {from_date} → {to_date}…")
    game_ids = get_game_ids_for_dates(from_date, to_date, season_type=season_type)
    if verbose:
        print(f"  {len(game_ids)} games in date range")

    # Partition: cached vs. todo
    cached_ids = [g for g in game_ids if (games_cache_dir / f"{g}.parquet").exists()]
    todo_ids   = [g for g in game_ids if g not in set(cached_ids)]
    if verbose and cached_ids:
        print(f"  {len(cached_ids)} already cached, fetching {len(todo_ids)} new")

    frames: list[pd.DataFrame] = []

    # Load cached games
    for gid in cached_ids:
        frames.append(pd.read_parquet(games_cache_dir / f"{gid}.parquet"))

    # Fetch new games in parallel
    if todo_ids:
        cache_str = str(pbp_cache_dir) if pbp_cache_dir else None
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_game, gid, cache_str): gid
                for gid in todo_ids
            }
            with tqdm(total=len(todo_ids), desc="Fetching games", disable=not verbose) as pbar:
                for future in as_completed(futures):
                    gid = futures[future]
                    try:
                        _, df_game, error = future.result()
                        if error is None and df_game is not None and len(df_game) > 0:
                            df_game.to_parquet(
                                games_cache_dir / f"{gid}.parquet", index=False
                            )
                            frames.append(df_game)
                            if verbose:
                                tqdm.write(f"  {gid}: {len(df_game)} possessions")
                        elif error:
                            tqdm.write(f"  ✗ {gid}: {error.splitlines()[-1]}")
                    except Exception as exc:
                        tqdm.write(f"  ✗ {gid}: {exc}")
                    pbar.update(1)

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["game_date", "game_id", "seconds_into_game"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Season collection
# ---------------------------------------------------------------------------

def collect_season(
    season: str,
    season_type: str = "Regular Season",
    output_dir: Path = Path("data"),
    pbp_cache_dir: Optional[Path] = None,
    checkpoint_interval: int = 50,
    max_workers: int = 4,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Collect possession-level data for an entire NBA season.

    Writes two files to ``output_dir / season``:
      - ``possessions_{slug}.parquet``
      - ``possessions_{slug}.csv``

    Checkpoints are saved every ``checkpoint_interval`` games to
    ``output_dir / season / checkpoints / checkpoint.parquet`` so
    interrupted runs can resume automatically.

    Parameters
    ----------
    season : str
        Season string, e.g. ``"2023-24"``.
    season_type : str
        ``"Regular Season"``, ``"Playoffs"``, etc.
    output_dir : Path
        Root output directory.  Season data goes in a subdirectory.
    pbp_cache_dir : Path, optional
        Cache directory for pbpstats raw JSON files.
    checkpoint_interval : int
        Save a checkpoint after every N completed games.
    max_workers : int
        Number of parallel worker processes.
    overwrite : bool
        If False (default), return the existing file when it already exists.

    Returns
    -------
    pd.DataFrame
        Full possession dataset.
    """
    season_slug = season.replace("-", "_")
    type_slug = season_type.lower().replace(" ", "_")
    slug = f"{season_slug}_{type_slug}"

    season_dir = Path(output_dir) / season
    season_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = season_dir / f"possessions_{slug}.parquet"
    out_csv = season_dir / f"possessions_{slug}.csv"
    ckpt_dir = season_dir / "checkpoints"
    ckpt_file = ckpt_dir / "checkpoint.parquet"

    # ── Return cached result if available ───────────────────────────────────
    if not overwrite and out_parquet.exists():
        print(f"Found existing file: {out_parquet}")
        df = pd.read_parquet(out_parquet)
        print(f"Loaded {len(df):,} possessions from {df['game_id'].nunique()} games")
        return df

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if pbp_cache_dir is not None:
        pbp_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Load game list ───────────────────────────────────────────────────────
    print(f"Fetching game list for {season} {season_type}…")
    all_game_ids = get_season_game_ids(season, season_type)
    print(f"Total games: {len(all_game_ids)}")

    # ── Resume from checkpoint ───────────────────────────────────────────────
    existing_frames: list[pd.DataFrame] = []
    processed: set[str] = set()

    if ckpt_file.exists():
        ckpt_df = pd.read_parquet(ckpt_file)
        existing_frames.append(ckpt_df)
        processed = set(ckpt_df["game_id"].unique())
        print(f"Resuming from checkpoint: {len(processed)} games already done")

    todo = [g for g in all_game_ids if g not in processed]
    print(f"Games remaining: {len(todo)}")

    failed: list[tuple[str, str]] = []
    completed = 0

    # ── Parallel processing ──────────────────────────────────────────────────
    cache_str = str(pbp_cache_dir) if pbp_cache_dir else None

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_game, gid, cache_str): gid for gid in todo
        }
        with tqdm(total=len(todo), desc=f"Collecting {season}") as pbar:
            for future in as_completed(futures):
                game_id = futures[future]
                try:
                    gid, df_game, error = future.result()
                    if error is None and df_game is not None and len(df_game) > 0:
                        existing_frames.append(df_game)
                        tqdm.write(f"  {gid}: {len(df_game)} possessions")
                    elif error:
                        tqdm.write(f"  ✗ {gid}: {error.splitlines()[-1]}")
                        failed.append((gid, error))
                except Exception as exc:
                    tqdm.write(f"  ✗ {game_id}: {exc}")
                    failed.append((game_id, str(exc)))

                completed += 1
                pbar.update(1)

                # Save checkpoint
                if completed % checkpoint_interval == 0 and existing_frames:
                    _save_checkpoint(existing_frames, ckpt_file)
                    tqdm.write(
                        f"  Checkpoint: "
                        f"{sum(len(f) for f in existing_frames):,} possessions"
                    )

    # ── Final merge and save ─────────────────────────────────────────────────
    if not existing_frames:
        print("No data collected.")
        return pd.DataFrame()

    df = pd.concat(existing_frames, ignore_index=True)
    df = df.sort_values(["game_date", "game_id", "seconds_into_game"]).reset_index(
        drop=True
    )

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    n_games = df["game_id"].nunique()
    print(f"\nSaved {len(df):,} possessions from {n_games} / {len(all_game_ids)} games")
    print(f"  Parquet : {out_parquet}")
    print(f"  CSV     : {out_csv}")

    if failed:
        print(f"\n{len(failed)} games failed:")
        for gid, err in failed[:10]:
            print(f"  {gid}: {err.splitlines()[-1]}")
        if len(failed) > 10:
            print(f"  … and {len(failed) - 10} more")

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(frames: list[pd.DataFrame], path: Path) -> None:
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(path, index=False)
