"""Fit RAPM v2 (learned prior, home court, learned penalties) and score it like v1.

    caffeinate -ims .venv/bin/python scripts/fit_v2.py                            # forward objective, ~30 min
    caffeinate -ims .venv/bin/python scripts/fit_v2.py --objective evidence --name v2eb --out outputs/v2eb
    .venv/bin/python scripts/fit_v2.py --score-only outputs/v2/model.pt           # re-score a saved model
    .venv/bin/python scripts/fit_v2.py --quick                                    # 4 seasons, 2 epochs
    python3.11 scripts/score_forward.py --summarize                               # next to v1 and the variants

Writes <out>/{model.pt, rolling_cache_v2.parquet, fit_summary.json} and
outputs/forward/<name>_{poss,margins,origins}.parquet. See
deep_rapm/torch_rapm.py and docs/rapm_v2_design.md.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_rapm.forward import VARIANTS
from deep_rapm.torch_rapm import FitSpec, fit_v2, score_saved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", default="v2")
    p.add_argument("--base", default="v1", choices=sorted(VARIANTS))
    p.add_argument("--objective", default="forward", choices=["forward", "evidence"])
    p.add_argument("--shrink", type=float, default=0.0, metavar="MINUTES",
                   help="shrink prior-feature rates toward league averages by this many minutes (v2.1: 500)")
    p.add_argument("--no-learn-mean", action="store_true", help="ablation: prior mean fixed at 0")
    p.add_argument("--no-learn-precision", action="store_true", help="ablation: prior precision fixed at the base penalty")
    p.add_argument("--no-learn-alpha", action="store_true", help="ablation: base penalties fixed")
    p.add_argument("--evidence-noise", default="model", choices=["model", "reference"],
                   help="s² in the evidence objective: current model's residuals, or the base ridge's (fixed)")
    p.add_argument("--score-only", type=Path, default=None, metavar="MODEL_PT",
                   help="re-run the evaluation pass of a saved model instead of fitting")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-origins", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-alpha", type=float, default=0.02)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--origin-step", type=int, default=7)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--train-until", default="2023-07-01")
    p.add_argument("--val-until", default="2025-07-01")
    p.add_argument("--seasons", nargs="+", default=None)
    p.add_argument("--out", type=Path, default=Path("outputs/v2"))
    p.add_argument("--quick", action="store_true", help="small configuration for a smoke test")
    args = p.parse_args()

    if args.score_only:
        score_saved(args.score_only, forward_dir=Path("outputs/forward"),
                    name=None if args.name == "v2" else args.name)
        return

    spec = FitSpec(name=args.name, base=args.base, objective=args.objective, epochs=args.epochs,
                   patience=args.patience, batch_origins=args.batch_origins, lr=args.lr,
                   lr_alpha=args.lr_alpha, hidden=args.hidden, origin_step_days=args.origin_step,
                   horizon_days=args.horizon, train_until=args.train_until, val_until=args.val_until,
                   shrink_minutes=args.shrink, learn_mean=not args.no_learn_mean,
                   learn_precision=not args.no_learn_precision, learn_alpha=not args.no_learn_alpha,
                   evidence_noise=args.evidence_noise,
                   seasons=tuple(args.seasons) if args.seasons else None)
    if args.quick:
        spec = FitSpec(name=args.name + "_quick", base=args.base, objective=args.objective, epochs=2,
                       origin_step_days=28, horizon_days=28, shrink_minutes=args.shrink,
                       learn_mean=not args.no_learn_mean, learn_precision=not args.no_learn_precision,
                       learn_alpha=not args.no_learn_alpha, evidence_noise=args.evidence_noise,
                       seasons=("2013-14", "2014-15", "2015-16", "2016-17"),
                       train_until="2016-07-01", val_until="2017-01-01")
        args.out = Path("outputs/v2_quick")
    fit_v2(spec, VARIANTS[args.base], out_dir=args.out,
           forward_dir=Path("outputs/forward") if not args.quick else Path("outputs/v2_quick/forward"))


if __name__ == "__main__":
    main()
