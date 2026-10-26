"""
collect_nightly.py — Collect yesterday's NBA games and merge into season parquets.

Designed to run as a daily cron job at ~4am local time, after all games finish.

Usage
-----
    python3.11 /Users/aarondanielson/Dropbox/deep_rapm/scripts/collect_nightly.py

Cron entry (4am daily):
    0 4 * * * cd /Users/aarondanielson/Dropbox/deep_rapm && python3.11 scripts/collect_nightly.py >> /tmp/collect_nightly.log 2>&1
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Always run from project root regardless of cron working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)

from deep_rapm.data.game import get_game_meta, get_game_possessions
from deep_rapm.data.game_nba_api import get_game_possessions_nba_api
from nba_api.stats.endpoints import leaguegamefinder

DATA_DIR  = Path("data")
NBA_TEAM_IDS = set(range(1610612737, 1610612767))   # the 30 franchises
GAMES_DIR = DATA_DIR / "games"
GAMES_DIR.mkdir(parents=True, exist_ok=True)


def get_game_ids_for_date(target_date: date) -> list[str]:
    """Return all NBA game IDs for a given date."""
    date_str = target_date.strftime("%m/%d/%Y")
    finder = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=date_str,
        date_to_nullable=date_str,
        league_id_nullable="00",
    )
    df = finder.get_data_frames()[0]
    if df.empty:
        return []
    # Only include games with a final result (WL not null)
    final = df[df["WL"].notna()]
    # Preseason exhibitions against non-NBA clubs are not collected.
    all_nba = final.groupby("GAME_ID")["TEAM_ID"].apply(
        lambda s: s.astype(int).isin(NBA_TEAM_IDS).all())
    return sorted(g for g in final["GAME_ID"].unique().tolist() if all_nba.get(g, False))


def collect_game(game_id: str) -> int:
    """
    Collect possessions for one game. Returns possession count, or 0 on failure.
    Tries pbpstats first, falls back to nba_api PlayByPlayV3.
    """
    out = GAMES_DIR / f"possessions_{game_id}.parquet"
    if out.exists():
        return -1  # already collected

    try:
        possessions = get_game_possessions(game_id)
        if not possessions:
            raise ValueError("empty pbpstats result")
    except Exception as e1:
        print(f"    pbpstats failed ({e1}), trying nba_api…")
        try:
            # BoxScoreSummaryV2 returns an empty body for 2026-27 games; V3 works.
            meta = {"game_id": game_id, **get_game_meta(game_id)}
            possessions = get_game_possessions_nba_api(game_id, meta)
        except Exception as e2:
            print(f"    nba_api also failed ({e2}) — skipping {game_id}")
            return 0

    if not possessions:
        print(f"    no possessions returned for {game_id}")
        return 0

    df = pd.DataFrame(possessions)
    df.to_parquet(out, index=False)
    return len(df)


_GAME_TYPE_BY_CHAR = {"1": "preseason", "2": "regular_season", "4": "playoffs"}


def _season_from_game_id(game_id: str) -> tuple[str, str] | None:
    """Return (season_label, game_type) from NBA game ID string, or None for unsupported types."""
    t  = game_id[2]   # '1'=preseason, '2'=regular, '4'=playoffs
    gtype = _GAME_TYPE_BY_CHAR.get(t)
    if gtype is None:
        return None
    yy = int(game_id[3:5])
    season = f"20{yy:02d}-{(yy+1):02d}"
    return season, gtype


def merge_into_season_parquets() -> None:
    """Merge all individual game parquets into per-season parquets."""
    games_dir = GAMES_DIR
    game_files = sorted(games_dir.glob("possessions_00*.parquet"))
    if not game_files:
        print("  No individual game parquets found.")
        return

    # Group by (season, game_type)
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for f in game_files:
        gid = f.stem.replace("possessions_", "")
        info = _season_from_game_id(gid)
        if info is None:
            continue
        season, gtype = info
        groups[(season, gtype)].append(f)

    for (season, gtype), files in sorted(groups.items()):
        season_dir = DATA_DIR / season.replace("-", "-")
        season_dir.mkdir(parents=True, exist_ok=True)
        out_path = season_dir / f"possessions_{season.replace('-','_')}_{gtype}.parquet"

        new_dfs = [pd.read_parquet(f) for f in files]
        new_df  = pd.concat(new_dfs, ignore_index=True)

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            existing_ids = set(existing["game_id"].unique())
            new_df = new_df[~new_df["game_id"].isin(existing_ids)]
            if new_df.empty:
                continue
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined = combined.sort_values(["game_id", "seconds_into_game"]).reset_index(drop=True)
        combined.to_parquet(out_path, index=False)
        n_games = combined["game_id"].nunique()
        print(f"  {season} {gtype}: +{len(new_df['game_id'].unique())} games → {n_games} total")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    if args.date:
        target = date.fromisoformat(args.date)
    else:
        target = date.today() - timedelta(days=1)
    print(f"=== collect_nightly  target={target}  ===")

    game_ids = get_game_ids_for_date(target)
    if not game_ids:
        print(f"No final games found for {target}")
        return

    print(f"Found {len(game_ids)} final game(s): {game_ids}")

    new_games = 0
    for gid in game_ids:
        n = collect_game(gid)
        if n == -1:
            print(f"  {gid}  already collected — skip")
        elif n == 0:
            print(f"  {gid}  FAILED")
        else:
            print(f"  {gid}  {n} possessions")
            new_games += 1

    print(f"\nCollected {new_games} new game(s)")

    if new_games > 0:
        print("\nMerging into season parquets…")
        merge_into_season_parquets()

    print("Done.")


if __name__ == "__main__":
    main()
