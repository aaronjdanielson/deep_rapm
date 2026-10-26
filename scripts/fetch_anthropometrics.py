"""Add height/weight for vocab players missing from data/player_anthropometrics.parquet.

    python3.11 scripts/fetch_anthropometrics.py

One CommonPlayerInfo call per missing player (~1 s each, paced, with the
pooled connection reset on failure). Players the API cannot describe keep
height/weight 0, which the feature builder treats as an average player.
Saves every 50 players, so it can be interrupted and rerun.
"""
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo

from deep_rapm.data.game_nba_api import reset_http_session

CACHE = Path("data/player_anthropometrics.parquet")


def main() -> None:
    vocab = pd.read_parquet("data/player_vocab.parquet")
    cache = pd.read_parquet(CACHE)
    # Rows with height 0 are earlier failures (throttling) — retry them too.
    cache = cache[cache["height_in"] > 0]
    missing = sorted(set(vocab["player_id"].astype(int)) - set(cache["player_id"].astype(int)))
    print(f"{len(cache)} cached with a height, {len(missing)} missing", flush=True)
    if not missing:
        return
    shutil.copy2(CACHE, CACHE.with_name(
        CACHE.name + f".bak.{datetime.now(timezone.utc):%Y-%m-%dT%H%M%S}"))

    rows, failed = [], []
    for i, pid in enumerate(missing, 1):
        h = w = 0.0
        for attempt in range(3):
            try:
                time.sleep(1.0 + 10 * attempt)
                info = commonplayerinfo.CommonPlayerInfo(
                    player_id=pid, timeout=20).get_data_frames()[0]
                hs, ws = str(info["HEIGHT"].iloc[0]), str(info["WEIGHT"].iloc[0])
                if "-" in hs:
                    ft, inch = hs.split("-")
                    h = float(ft) * 12 + float(inch)
                w = float(ws) if ws.isdigit() else 0.0
                break
            except Exception:
                reset_http_session()
                if attempt == 2:
                    failed.append(pid)
        rows.append({"player_id": pid, "height_in": h, "weight_lb": w})
        if i % 50 == 0 or i == len(missing):
            print(f"  [{i}/{len(missing)}] failed so far: {len(failed)}", flush=True)
            pd.concat([cache, pd.DataFrame(rows)], ignore_index=True).to_parquet(CACHE, index=False)

    got = sum(1 for r in rows if r["height_in"] > 0)
    print(f"DONE: {len(cache) + len(rows)} rows; {got}/{len(rows)} new players with a height; "
          f"failed={failed}", flush=True)


if __name__ == "__main__":
    main()
