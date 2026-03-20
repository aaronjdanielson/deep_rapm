"""
collect.py — CLI for possession data collection.

Single game:
    collect-possessions --game 0022300154

Full season (default Regular Season):
    collect-possessions --season 2023-24

Multiple seasons:
    collect-possessions --season 2022-23 --season 2023-24

With PBP file cache (speeds up reruns):
    collect-possessions --season 2023-24 --pbp-cache pbp_cache/

Restart fresh (ignore existing output):
    collect-possessions --season 2023-24 --overwrite
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from deep_rapm.data.game import get_game_possessions
from deep_rapm.data.season import collect_season


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="collect-possessions",
        description="Collect NBA possession data for Deep RAPM training.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--game",
        metavar="GAME_ID",
        help="Collect a single game (e.g. 0022300154).",
    )
    mode.add_argument(
        "--season",
        metavar="SEASON",
        action="append",
        dest="seasons",
        help="Collect a full season (e.g. 2023-24). Repeatable.",
    )
    p.add_argument(
        "--season-type",
        default="Regular Season",
        choices=["Regular Season", "Playoffs", "Pre Season", "All Star"],
        help="Season type (default: Regular Season).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Root output directory (default: data/).",
    )
    p.add_argument(
        "--pbp-cache",
        type=Path,
        default=None,
        metavar="DIR",
        help="Cache directory for raw pbpstats JSON files.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes (default: 4).",
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N games (default: 50).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-collect even if output files exist.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # ── Single game ──────────────────────────────────────────────────────────
    if args.game:
        print(f"Collecting game {args.game}…")
        records = get_game_possessions(
            args.game,
            pbp_cache_dir=args.pbp_cache,
        )
        df = pd.DataFrame(records)
        print(f"  {len(df)} possessions extracted")

        out_dir = args.output_dir / "games"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"possessions_{args.game}.parquet"
        df.to_parquet(out, index=False)
        print(f"  Saved to {out}")
        return

    # ── One or more seasons ──────────────────────────────────────────────────
    for season in args.seasons:
        print(f"\n{'='*60}")
        print(f"Season: {season}  |  {args.season_type}")
        print(f"{'='*60}")
        collect_season(
            season=season,
            season_type=args.season_type,
            output_dir=args.output_dir,
            pbp_cache_dir=args.pbp_cache,
            checkpoint_interval=args.checkpoint_interval,
            max_workers=args.workers,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
