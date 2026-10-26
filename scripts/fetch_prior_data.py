"""Collect the raw inputs for RAPM priors (docs/rapm_v2_design.md §4).

    python3.11 scripts/fetch_prior_data.py            # both parts, resumable
    python3.11 scripts/fetch_prior_data.py --info     # CommonPlayerInfo only
    python3.11 scripts/fetch_prior_data.py --logs     # PlayerGameLogs only

1. data/player_info.parquet — one row per vocab player from CommonPlayerInfo:
   birthdate, draft year / round / number, height, weight, position, country,
   school, first/last NBA season. Players the API has no record for get a
   row of nulls (so they are not retried) with ``fetched = False``.
2. data/raw/player_game_logs/<season>_<kind>.parquet — PlayerGameLogs for
   every season × kind (Regular Season, Playoffs, PlayIn, Pre Season): one
   row per player-game with the box-score line. ~1 call per file; the
   in-progress season's files are refetched on every run.

Both are cached and paced (~1 call/s, pooled connection reset on failure).
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, playergamelogs

from deep_rapm.data.game_nba_api import reset_http_session
from deep_rapm.production import SEASONS_ALL

INFO_PATH = Path("data/player_info.parquet")
LOGS_DIR = Path("data/raw/player_game_logs")
KINDS = {"regular_season": "Regular Season", "playoffs": "Playoffs",
         "playin": "PlayIn", "preseason": "Pre Season"}
INFO_COLS = {
    "PERSON_ID": "player_id", "DISPLAY_FIRST_LAST": "player_name", "BIRTHDATE": "birthdate",
    "COUNTRY": "country", "SCHOOL": "school", "HEIGHT": "height", "WEIGHT": "weight",
    "POSITION": "position", "DRAFT_YEAR": "draft_year", "DRAFT_ROUND": "draft_round",
    "DRAFT_NUMBER": "draft_number", "FROM_YEAR": "from_year", "TO_YEAR": "to_year",
}


def _call(fn, retries=3):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            reset_http_session()
            if attempt == retries - 1:
                raise e
            time.sleep(5 + 15 * attempt)


def fetch_info() -> None:
    vocab = pd.read_parquet("data/player_vocab.parquet")
    have = pd.read_parquet(INFO_PATH) if INFO_PATH.exists() else pd.DataFrame(columns=["player_id"])
    missing = sorted(set(vocab["player_id"].astype(int)) - set(have["player_id"].astype(int)))
    print(f"player info: {len(have)} cached, {len(missing)} to fetch", flush=True)
    if not missing:
        return
    rows = []
    t0 = time.time()
    for i, pid in enumerate(missing, 1):
        row = {"player_id": pid, "fetched": False}
        try:
            time.sleep(1.0)
            df = _call(lambda: commonplayerinfo.CommonPlayerInfo(player_id=pid, timeout=20)
                       .get_data_frames()[0])
            if len(df):
                r = df.iloc[0]
                row.update({v: r.get(k) for k, v in INFO_COLS.items() if k != "PERSON_ID"})
                row["fetched"] = True
        except Exception as e:
            if "resultSet" not in str(e):      # no record → a null row; anything else → retry next run
                print(f"  {pid}: {type(e).__name__} {str(e)[:60]} — will retry next run", flush=True)
                continue
        rows.append(row)
        if i % 100 == 0 or i == len(missing):
            out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
            out.to_parquet(INFO_PATH, index=False)
            print(f"  [{i}/{len(missing)}] {time.time() - t0:.0f}s", flush=True)
    print(f"player info done: {sum(r['fetched'] for r in rows)} fetched, "
          f"{sum(not r['fetched'] for r in rows)} without a record", flush=True)


def fetch_logs(seasons=SEASONS_ALL) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    current = max(seasons)
    for season in seasons:
        for kind, api_kind in KINDS.items():
            if kind == "playin" and season < "2019-20":
                continue
            path = LOGS_DIR / f"{season}_{kind}.parquet"
            if path.exists() and season != current:
                continue
            try:
                time.sleep(1.5)
                df = _call(lambda: playergamelogs.PlayerGameLogs(
                    season_nullable=season, season_type_nullable=api_kind, timeout=60)
                    .get_data_frames()[0])
            except Exception as e:
                print(f"  {season} {kind}: FAILED {type(e).__name__} {str(e)[:60]}", flush=True)
                continue
            df.to_parquet(path, index=False)
            print(f"  {season} {kind}: {len(df):,} player-games", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--info", action="store_true")
    p.add_argument("--logs", action="store_true")
    args = p.parse_args()
    both = not (args.info or args.logs)
    print(f"=== fetch_prior_data {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC ===", flush=True)
    if args.logs or both:
        fetch_logs()
    if args.info or both:
        fetch_info()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
