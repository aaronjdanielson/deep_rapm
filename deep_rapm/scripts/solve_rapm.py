"""
solve_rapm.py — CLI entry point for analytical Ridge RAPM.

Usage
-----
Season mode (uses pre-collected parquets):
    solve-rapm
    solve-rapm --seasons 2021-22 2022-23 --alpha 2000
    solve-rapm --top 20 --output-dir checkpoints/rapm

Date-range mode (auto-fetches & caches per game):
    solve-rapm --from-date 2024-01-01 --to-date 2024-03-31
    solve-rapm --from-date 2023-10-01 --to-date 2024-06-30 --half-life 365
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_rapm.rapm import fit_rapm, load_rapm


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="solve-rapm",
        description="Fit ridge RAPM analytically from possession data.",
    )
    p.add_argument("--data-dir",     type=Path, default=Path("data"))
    p.add_argument("--player-vocab", type=Path, default=Path("data/player_vocab.parquet"))
    p.add_argument("--player-table", type=Path, default=Path("data/players.parquet"))
    p.add_argument("--alpha",    type=float, default=2000.0,
                   help="Ridge penalty (default 2000).")
    p.add_argument("--min-poss", type=int, default=100,
                   help="Min possessions each role for leaderboard (default 100).")
    p.add_argument("--top",      type=int, default=15,
                   help="Top/bottom N players to print (default 15).")
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/rapm"))

    # ── Data source: season mode ──────────────────────────────────────────────
    season_grp = p.add_argument_group("season mode (pre-collected parquets)")
    season_grp.add_argument(
        "--seasons", nargs="+",
        default=None,
        metavar="SEASON",
        help="Season labels to load, e.g. 2021-22 2022-23. "
             "Defaults to 5 seasons (2018-19 through 2022-23) when neither "
             "--from-date nor --seasons is given.",
    )

    # ── Data source: date-range mode ─────────────────────────────────────────
    date_grp = p.add_argument_group("date-range mode (auto-fetch & cache)")
    date_grp.add_argument("--from-date", metavar="YYYY-MM-DD",
                          help="Start date (inclusive).")
    date_grp.add_argument("--to-date",   metavar="YYYY-MM-DD",
                          help="End date (inclusive).")
    date_grp.add_argument("--season-type", default="Regular Season",
                          help='NBA season type (default "Regular Season").')
    date_grp.add_argument("--pbp-cache-dir", type=Path, default=None,
                          metavar="DIR",
                          help="Cache directory for raw pbpstats JSON files.")
    date_grp.add_argument("--max-workers", type=int, default=4,
                          help="Parallel workers for fetching games (default 4).")

    # ── Recency weighting ─────────────────────────────────────────────────────
    p.add_argument(
        "--half-life", type=float, default=None, metavar="DAYS",
        help="Half-life in days for exponential recency weighting. "
             "E.g. 365 = recent games count twice as much as games 1 year ago. "
             "Default: no weighting (all possessions equal).",
    )

    return p


def _print_leaderboard(results, top_n: int, min_poss: int) -> None:
    qualified = results[results["qualified"]].copy()
    n = len(qualified)

    def _table(df, title):
        print(f"\n{'─'*62}")
        print(f"  {title}  (min {min_poss} poss each role, n={n})")
        print(f"{'─'*62}")
        print(f"  {'Player':<28}  {'ORAPM':>6}  {'DRAPM':>6}  {'RAPM':>6}  {'N':>7}")
        print(f"  {'─'*28}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}")
        for _, r in df.iterrows():
            print(
                f"  {r.player_name:<28}  "
                f"{r.orapm:>+6.2f}  "
                f"{r.drapm:>+6.2f}  "
                f"{r.rapm:>+6.2f}  "
                f"{int(min(r.n_off, r.n_def)):>7,}"
            )

    _table(qualified.nlargest(top_n, "rapm"),  f"Top {top_n} by RAPM")
    _table(qualified.nsmallest(top_n, "rapm"), f"Bottom {top_n} by RAPM")
    print(f"\n  RAPM std: {qualified['rapm'].std():.2f} pts/100 possessions")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    using_daterange = args.from_date is not None or args.to_date is not None
    using_seasons   = args.seasons is not None

    if using_daterange and using_seasons:
        raise SystemExit("error: --seasons and --from-date/--to-date are mutually exclusive.")
    if using_daterange and (args.from_date is None or args.to_date is None):
        raise SystemExit("error: --from-date and --to-date must be used together.")

    # Default: season mode with 5 seasons
    if not using_daterange and not using_seasons:
        args.seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]

    results = fit_rapm(
        data_dir=args.data_dir,
        seasons=args.seasons if not using_daterange else None,
        player_vocab_path=args.player_vocab,
        player_table_path=args.player_table,
        alpha=args.alpha,
        min_poss=args.min_poss,
        output_dir=args.output_dir,
        from_date=args.from_date,
        to_date=args.to_date,
        season_type=args.season_type,
        pbp_cache_dir=args.pbp_cache_dir,
        max_workers=args.max_workers,
        half_life_days=args.half_life,
    )

    _print_leaderboard(results, args.top, args.min_poss)


if __name__ == "__main__":
    main()
