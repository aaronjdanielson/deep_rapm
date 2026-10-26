"""Remove preseason exhibitions against non-NBA clubs from the season parquets.

    python3.11 scripts/remove_exhibitions.py            # report only
    python3.11 scripts/remove_exhibitions.py --apply    # remove, with backups

Games where either team is not one of the 30 franchises (Real Madrid,
Shanghai Sharks, Maccabi Haifa, …) are not NBA competition and their
opponents' rosters are absent from the box scores. Each affected season
parquet is backed up (*.bak.<timestamp>) and rewritten without those games;
the per-game cache files move to data/games_excluded/ so the nightly merge
cannot re-add them.
"""
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

NBA_TEAM_IDS = set(range(1610612737, 1610612767))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    excluded_dir = Path("data/games_excluded")
    total_games = total_poss = 0
    for p in sorted(Path("data").glob("20*/possessions_*.parquet")):
        df = pd.read_parquet(p)
        bad = ~df["offense_team_id"].isin(NBA_TEAM_IDS) | ~df["defense_team_id"].isin(NBA_TEAM_IDS)
        if not bad.any():
            continue
        gids = sorted(df.loc[bad, "game_id"].unique())
        total_games += len(gids)
        total_poss += int(bad.sum())
        print(f"{p}: {len(gids)} exhibition games, {int(bad.sum()):,} possessions")
        if not args.apply:
            continue
        shutil.copy2(p, p.with_name(p.name + f".bak.{stamp}"))
        df[~df["game_id"].isin(gids)].to_parquet(p, index=False)
        csv = p.with_suffix(".csv")
        if csv.exists():
            shutil.copy2(csv, csv.with_name(csv.name + f".bak.{stamp}"))
            df[~df["game_id"].isin(gids)].to_csv(csv, index=False)
        excluded_dir.mkdir(exist_ok=True)
        for gid in gids:
            cached = Path("data/games") / f"possessions_{gid}.parquet"
            if cached.exists():
                shutil.move(str(cached), excluded_dir / cached.name)
    print(f"{'removed' if args.apply else 'would remove'} {total_games} games, {total_poss:,} possessions")


if __name__ == "__main__":
    main()
