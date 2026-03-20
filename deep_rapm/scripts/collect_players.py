"""
collect_players.py — CLI for player position metadata collection.

Fetches per-season NBA roster data from the NBA Stats API and writes a
consolidated player-position lookup table used by the training dataset.

Usage
-----
All six training seasons (recommended first run):
    collect-players

Specific seasons only:
    collect-players --seasons 2022-23 2023-24

Force re-fetch even if output already exists:
    collect-players --overwrite

Custom output path:
    collect-players --output data/players.parquet

What it writes
--------------
data/rosters/<season>.parquet   — per-season raw roster cache (one file per season)
data/players.parquet            — consolidated (player_id, season, position_idx, ...)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_rapm.data.players import build_player_table, build_player_vocab, supplement_player_table


_DEFAULT_SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="collect-players",
        description="Fetch NBA player position metadata for Deep RAPM training.",
    )
    p.add_argument(
        "--seasons",
        nargs="+",
        default=_DEFAULT_SEASONS,
        metavar="SEASON",
        help=(
            "Seasons to collect (default: all six training seasons). "
            "Example: --seasons 2022-23 2023-24"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/players.parquet"),
        metavar="PATH",
        help="Output parquet path (default: data/players.parquet).",
    )
    p.add_argument(
        "--roster-cache",
        type=Path,
        default=Path("data/rosters"),
        metavar="DIR",
        help="Directory for per-season roster caches (default: data/rosters/).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-fetch even if output already exists.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        metavar="DIR",
        help=(
            "Root data directory containing season parquets.  "
            "Used to check coverage and supplement missing players "
            "(default: data/)."
        ),
    )
    p.add_argument(
        "--no-supplement",
        action="store_true",
        help="Skip the supplemental commonplayerinfo fetch for missing players.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    print(f"Collecting player positions for {len(args.seasons)} season(s):")
    for s in args.seasons:
        print(f"  {s}")
    print()

    table = build_player_table(
        seasons=args.seasons,
        output_path=args.output,
        roster_cache_dir=args.roster_cache,
        overwrite=args.overwrite,
    )

    # Supplement with commonplayerinfo for players in parquets but not rosters
    if not args.no_supplement and args.data_dir.exists():
        print()
        table = supplement_player_table(
            table=table,
            parquet_dir=args.data_dir,
            output_path=args.output,
        )

    # Build player vocabulary (player_id → embedding index)
    vocab_path = args.output.parent / "player_vocab.parquet"
    vocab = build_player_vocab(
        player_table=table,
        output_path=vocab_path,
        overwrite=args.overwrite,
    )
    num_players = len(vocab) + 1  # +1 for index 0 (UNKNOWN)

    # Summary
    print()
    print(f"{'Season':<12}  {'Players':>8}  {'Positions'}")
    print("-" * 50)
    for season, grp in table.groupby("season"):
        pos_counts = grp["position_str"].value_counts().to_dict()
        pos_summary = "  ".join(f"{k}:{v}" for k, v in sorted(pos_counts.items()))
        print(f"{season:<12}  {len(grp):>8}  {pos_summary}")
    print()
    print(f"Total unique players: {table['player_id'].nunique()}")
    print(f"Player table   → {args.output}")
    print(f"Player vocab   → {vocab_path}  (num_players={num_players})")


if __name__ == "__main__":
    main()
