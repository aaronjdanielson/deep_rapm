"""Score rolling RAPM configurations on future possessions (forward chaining).

    python3.11 scripts/score_forward.py --variants v1 v1_home hl180     # one pass per variant
    python3.11 scripts/score_forward.py --summarize                     # table from saved results
    python3.11 scripts/score_forward.py --list                          # the known variants

Each pass writes outputs/forward/<variant>_poss.parquet and
<variant>_margins.parquet and prints the tune / holdout comparison. See
deep_rapm/forward.py for the scoring design; variants are defined there.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from deep_rapm.forward import (HOLDOUT_FROM, PENALTY_GRID, VAL_FROM, VARIANTS, comparison_table,
                               paired_table, score_forward, summarize)


def load_results(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def cat(suffix):
        return pd.concat([pd.read_parquet(p) for p in sorted(out.glob(f"*_{suffix}.parquet"))],
                         ignore_index=True)
    return cat("poss"), cat("margins"), cat("origins")


def print_summary(out: Path, selection: bool = False) -> None:
    """Default: tune / holdout tables. ``selection``: train / val tables only —
    the view to choose models on, with the holdout kept out of sight."""
    poss, margins, origins = load_results(out)
    s = summarize(poss, margins, three_way=selection)
    splits = ("train", "val") if selection else ("tune", "holdout")
    labels = {"tune": f"seasons < {HOLDOUT_FROM}", "holdout": f"seasons ≥ {HOLDOUT_FROM}",
              "train": f"seasons < {VAL_FROM}", "val": f"{VAL_FROM} ≤ seasons < {HOLDOUT_FROM}"}
    for split in splits:
        print(f"\n=== {split} ({labels[split]}); regular season + playoffs; weighted for non-garbage time ===")
        print(comparison_table(s, split))
        for metric, label in (("wmse", "Δ weighted MSE"), ("nll", "Δ predictive NLL")):
            pt = paired_table(origins, split=split, metric=metric, three_way=selection)
            if len(pt):
                print(f"\n--- paired vs v1 (2000/2000), {split}: {label} per origin, negative = better ---")
                print(pt.to_string(index=False, float_format=lambda v: f"{v:.6f}" if abs(v) < 1 else f"{v:,.1f}"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variants", nargs="+", default=["v1"])
    p.add_argument("--penalties", nargs="+", default=None,
                   help="alpha_off:alpha_def pairs, e.g. 2000:2000 2000:4000 (default: the grid)")
    p.add_argument("--origin-step", type=int, default=7)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--first-origin", default=None, help="YYYY-MM-DD (default: start of the third season)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("outputs/forward"))
    p.add_argument("--no-nll", action="store_true", help="skip the predictive density (one inverse per origin)")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--selection", action="store_true",
                   help="with --summarize: train / val tables only (choose models here; holdout stays hidden)")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        for name, cfg in VARIANTS.items():
            print(f"{name:<14} {cfg}")
        return
    if args.summarize:
        print_summary(args.out, selection=args.selection)
        return

    penalties = None
    if args.penalties:
        penalties = [tuple(float(x) for x in s.split(":")) for s in args.penalties]
    else:
        penalties = list(PENALTY_GRID)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.variants:
        cfg = VARIANTS[name]
        t0 = time.time()
        res = score_forward(cfg, data_dir=args.data_dir, origin_step_days=args.origin_step,
                            horizon_days=args.horizon, penalties=penalties,
                            first_origin=args.first_origin, nll=not args.no_nll)
        res.poss.to_parquet(args.out / f"{name}_poss.parquet", index=False)
        res.margins.to_parquet(args.out / f"{name}_margins.parquet", index=False)
        res.origins.to_parquet(args.out / f"{name}_origins.parquet", index=False)
        print(f"[{name}] done in {time.time() - t0:.0f}s: {res.n_origins} origins, σ={res.comp_sigma:.2f}")
    print_summary(args.out)


if __name__ == "__main__":
    main()
