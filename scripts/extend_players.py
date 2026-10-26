"""Extend data/players.parquet and data/player_vocab.parquet for new seasons.

    python3.11 scripts/extend_players.py 2013-14 2014-15 2015-16

Additive only: existing rows and existing vocab indices are never changed
(saved torch models index embeddings by player_idx, so indices must be
stable). For each season: team rosters (cached under data/rosters/), then
players who appear in that season's possession parquets but on no roster
(camp bodies, 10-day contracts) via CommonPlayerInfo. New player ids are
appended to the vocab after the current maximum index. Both files are
backed up with a timestamp first.
"""
import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from deep_rapm.data.players import fetch_player_info_positions, fetch_season_rosters

PLAYERS = Path("data/players.parquet")
VOCAB = Path("data/player_vocab.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seasons", nargs="+")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    for p in (PLAYERS, VOCAB):
        shutil.copy2(p, p.with_name(p.name + f".bak.{stamp}"))

    table = pd.read_parquet(PLAYERS)
    have = set(zip(table["player_id"], table["season"]))
    print(f"players.parquet: {len(table):,} rows, seasons "
          f"{min(table.season)} … {max(table.season)}", flush=True)

    frames = [table]
    for season in args.seasons:
        print(f"{season}: fetching rosters…", flush=True)
        r = fetch_season_rosters(season, roster_cache_dir=Path("data/rosters"))
        r = r[[(p, s) not in have for p, s in zip(r["player_id"], r["season"])]]
        print(f"  +{len(r)} roster rows", flush=True)
        frames.append(r)
    table = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["player_id", "season"], keep="first")

    have = set(zip(table["player_id"], table["season"]))
    for season in args.seasons:
        missing: set[int] = set()
        for pq in sorted(Path(f"data/{season}").glob("possessions_*.parquet")):
            df = pd.read_parquet(pq, columns=["offense_player_ids", "defense_player_ids"])
            for col in df.columns:
                for ids in df[col]:
                    for pid in ids:
                        if (int(pid), season) not in have:
                            missing.add(int(pid))
        print(f"{season}: {len(missing)} players in possessions but on no roster", flush=True)
        if missing:
            supp = fetch_player_info_positions(sorted(missing), season=season)
            table = pd.concat([table, supp], ignore_index=True)
            have |= set(zip(supp["player_id"], supp["season"]))
            time.sleep(1.0)

    table = table.drop_duplicates(subset=["player_id", "season"], keep="first")
    table.to_parquet(PLAYERS, index=False)
    print(f"players.parquet → {len(table):,} rows", flush=True)

    vocab = pd.read_parquet(VOCAB)
    new_ids = sorted(set(table["player_id"].astype(int)) - set(vocab["player_id"]))
    start = int(vocab["player_idx"].max()) + 1
    vocab = pd.concat([vocab, pd.DataFrame({
        "player_id": new_ids, "player_idx": range(start, start + len(new_ids))})],
        ignore_index=True)
    vocab.to_parquet(VOCAB, index=False)
    print(f"player_vocab.parquet: +{len(new_ids)} players → {len(vocab):,} total "
          f"(indices {start}…{start + len(new_ids) - 1} appended)", flush=True)


if __name__ == "__main__":
    main()
