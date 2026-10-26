"""Validate every game in the possession parquets and write a quality table.

    python3.11 scripts/validate_data.py                 # structural checks only (offline)
    python3.11 scripts/validate_data.py --reconcile     # + official box scores (API, cached)
    python3.11 scripts/validate_data.py --seasons 2025-26 2026-27

Writes data/quality/game_quality.parquet (one row per game: metrics, errors,
warnings, severity) and prints a per-season / per-kind summary plus the most
common problems. Read-only with respect to the possession data.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from deep_rapm.data.validate import validate_corpus


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--seasons", nargs="+", default=None)
    p.add_argument("--reconcile", action="store_true",
                   help="Compare against official box scores (points, minutes).")
    p.add_argument("--out", type=Path, default=Path("data/quality/game_quality.parquet"))
    args = p.parse_args()

    box_dir = args.data_dir / "quality" / "boxscores" if args.reconcile else None
    q = validate_corpus(args.data_dir, args.seasons, box_dir=box_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.seasons and args.out.exists():
        # Update only the requested seasons' rows in the existing table.
        old = pd.read_parquet(args.out)
        q = pd.concat([old[~old["season"].isin(set(q["season"]))], q], ignore_index=True)
        q = q.sort_values(["season", "kind", "game_id"]).reset_index(drop=True)
    q.to_parquet(args.out, index=False)

    summary = q.groupby(["season", "kind"]).agg(
        games=("game_id", "size"),
        error=("severity", lambda s: int((s == "error").sum())),
        warn=("severity", lambda s: int((s == "warn").sum())),
        poss=("n_poss", "sum"),
        poss_in_error=("n_poss", lambda s: int(s[q.loc[s.index, "severity"] == "error"].sum())),
    )
    summary["% poss in error games"] = (100 * summary["poss_in_error"] / summary["poss"]).round(1)
    pd.set_option("display.width", 160)
    print(summary.drop(columns="poss_in_error").to_string())

    def kind(msg: str) -> str:
        return " ".join(w for w in msg.split() if not any(c.isdigit() for c in w))

    for col in ("errors", "warnings"):
        c = Counter(kind(m) for msgs in q[col] for m in msgs)
        print(f"\nMost common {col}:")
        for msg, n in c.most_common(8):
            print(f"  {n:>5} games  {msg}")
    print(f"\nWrote {len(q)} games → {args.out}")


if __name__ == "__main__":
    main()
