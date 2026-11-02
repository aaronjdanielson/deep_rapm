"""Fit RAPM with a neural density prior over the joint (ORAPM, DRAPM).

    caffeinate -ims .venv/bin/python scripts/fit_density.py --name d1 --components 1 --precision full
    caffeinate -ims .venv/bin/python scripts/fit_density.py --name d3 --components 3 --precision full
    caffeinate -ims .venv/bin/python scripts/fit_density.py --name maf4 --family maf --flow-blocks 4
    .venv/bin/python scripts/fit_density.py --quick --components 3          # 4-season smoke test

Same snapshot, origins, weights and splits as phase 1 (default --until
2026-10-24). Writes <out>/{model.pt, rolling_cache_density.parquet,
fit_summary.json} and outputs/forward/<name>_{poss,margins,origins}.parquet.
See deep_rapm/density_prior.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_rapm.density_prior import DensitySpec, fit_density
from deep_rapm.production import V1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", default="d3")
    p.add_argument("--family", default="mixture", choices=["mixture", "maf"])
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--flow-blocks", type=int, default=4)
    p.add_argument("--newton-iters", type=int, default=3)
    p.add_argument("--precision", default="full", choices=["fixed", "diag", "full"])
    p.add_argument("--em-iters", type=int, default=3)
    p.add_argument("--mean-cap", type=float, default=0.08)
    p.add_argument("--features", default="v2", choices=["v1", "v2"])
    p.add_argument("--shrink", type=float, default=500.0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--until", default="2026-10-24")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    common = dict(family=args.family, flow_blocks=args.flow_blocks, newton_iters=args.newton_iters,
                  components=args.components, precision=args.precision, em_iters=args.em_iters,
                  mean_cap=args.mean_cap, feature_set=args.features, shrink_minutes=args.shrink,
                  hidden=args.hidden, lr=args.lr, learn_precision=args.precision != "fixed")
    if args.quick:
        spec = DensitySpec(name=args.name + "_quick", epochs=2, origin_step_days=28, horizon_days=28,
                           seasons=("2013-14", "2014-15", "2015-16", "2016-17"),
                           train_until="2016-07-01", val_until="2017-01-01", **common)
        out, fwd = Path("outputs/density_quick") / args.name, Path("outputs/density_quick/forward")
        until = None
    else:
        spec = DensitySpec(name=args.name, epochs=args.epochs, patience=args.patience, **common)
        out, fwd = args.out or Path("outputs/density") / args.name, Path("outputs/forward")
        until = args.until
    fit_density(spec, V1, out, forward_dir=fwd, until=until)


if __name__ == "__main__":
    main()
