"""
collect_box_scores.py — CLI entry point for per-game box score collection.

Usage
-----
Collect box scores for specific seasons (reuses game IDs from possession parquets):
    collect-box-scores --seasons 2022-23 2023-24 --box-dir data/box

Collect for all default training seasons:
    collect-box-scores

Adjust parallelism (careful with NBA API rate limits):
    collect-box-scores --seasons 2023-24 --max-workers 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_rapm.data.box_scores import collect_box_scores_for_seasons


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="collect-box-scores",
        description="Fetch and cache per-game box score stats for NBA seasons.",
    )
    p.add_argument(
        "--seasons", nargs="+",
        default=["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"],
        metavar="SEASON",
        help="NBA season labels to collect (default: 6 seasons).",
    )
    p.add_argument(
        "--data-dir", type=Path, default=Path("data"), metavar="DIR",
        help="Root data directory containing season subdirs (default: data/).",
    )
    p.add_argument(
        "--box-dir", type=Path, default=Path("data/box"), metavar="DIR",
        help="Output directory for per-game box score parquets (default: data/box/).",
    )
    p.add_argument(
        "--season-type", default="Regular Season",
        choices=["Regular Season", "Playoffs", "All Star"],
        help="Season type (default: Regular Season).",
    )
    p.add_argument(
        "--max-workers", type=int, default=1, metavar="N",
        help="Parallel workers (default: 1).  Increase carefully — "
             "the NBA API rate-limits aggressively.",
    )
    p.add_argument(
        "--delay", type=float, default=0.6, metavar="SECS",
        help="Sleep between API calls per worker (default: 0.6s).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    collect_box_scores_for_seasons(
        seasons=args.seasons,
        data_dir=args.data_dir,
        box_dir=args.box_dir,
        season_type=args.season_type,
        max_workers=args.max_workers,
        request_delay=args.delay,
        verbose=True,
    )


if __name__ == "__main__":
    main()
