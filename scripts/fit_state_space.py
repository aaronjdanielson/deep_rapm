"""Run the state-space RAPM filter over a grid of transitions and score it like v1.

    caffeinate -ims .venv/bin/python scripts/fit_state_space.py --phi 0.98 0.99 0.995 1.0 --summer 1
    caffeinate -ims .venv/bin/python scripts/fit_state_space.py --phi 0.99 --summer 1 4 8 --prior outputs/v2_1A_r/model.pt
    .venv/bin/python scripts/fit_state_space.py --quick

Each (φ, summer_steps, prior) is one filter pass (~35 min on the full
snapshot) and writes outputs/forward/ssm-<label>_{poss,margins,origins}.parquet
and outputs/ssm/<label>/{ratings.parquet, summary.json}. Compare with
scripts/compare_priors.py (rows "ssm-…"); choose on validation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from deep_rapm.forward import WindowScorer
from deep_rapm.production import V1
from deep_rapm.state_space import SSMSpec, run_filter
from deep_rapm.torch_rapm import FitSpec, prepare


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phi", nargs="+", type=float, default=[0.99])
    p.add_argument("--summer", nargs="+", type=int, default=[1])
    p.add_argument("--prior", default="zero", help='"zero" or a v2 checkpoint (learned prior mean)')
    p.add_argument("--lam", type=float, default=2000.0)
    p.add_argument("--until", default="2026-10-24")
    p.add_argument("--out", type=Path, default=Path("outputs/ssm"))
    p.add_argument("--forward-dir", type=Path, default=Path("outputs/forward"))
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    # The feature pipeline must be the one the prior checkpoint was trained with:
    # its feature set, shrinkage and lag definitions, in its column order.
    feat_kw = dict(feature_set="v2", shrink_minutes=500.0)
    ckpt_features = None
    if args.prior != "zero":
        import torch
        ck = torch.load(args.prior, weights_only=False)
        feat_kw = {k: ck["spec"][k] for k in ("feature_set", "shrink_minutes", "lag_days") if k in ck["spec"]}
        ckpt_features = list(ck["features"])
    if args.quick:
        fs = FitSpec(name="ssm", seasons=("2013-14", "2014-15", "2015-16", "2016-17"), origin_step_days=7,
                     horizon_days=7, train_until="2016-07-01", val_until="2017-01-01", **feat_kw)
        ctx = prepare(fs, V1, verbose=False)
        args.out, args.forward_dir = Path("outputs/ssm_quick"), Path("outputs/ssm_quick/forward")
    else:
        fs = FitSpec(name="ssm", **feat_kw)
        ctx = prepare(fs, V1, until=args.until)
    if ckpt_features is not None:
        from deep_rapm.priors import FEATURE_SETS
        from deep_rapm.torch_rapm import LAG_FEATURES
        runtime = FEATURE_SETS[fs.feature_set] + LAG_FEATURES
        assert runtime == ckpt_features, f"feature pipeline {runtime[:3]}…({len(runtime)}) != checkpoint {ckpt_features[:3]}…({len(ckpt_features)})"
        assert next(iter(ctx.feats.values())).shape[1] + len(LAG_FEATURES) == ck["state_dict"]["prior.net.0.weight"].shape[1]
    prior_tag = "zero" if args.prior == "zero" else Path(args.prior).parent.name
    for phi in args.phi:
        for summer in args.summer:
            label = f"phi{phi:g}-s{summer}-{prior_tag}"
            spec = SSMSpec(name=f"ssm-{label}", phi=phi, summer_steps=summer, lam=args.lam, prior=args.prior)
            t0 = time.time()
            scorer = WindowScorer(spec.name, [(args.lam, args.lam)])
            rows: list[dict] = []
            wmse = run_filter(spec, ctx, scorer=scorer, cache_rows=rows)
            poss, margins, origins = scorer.frames()
            args.forward_dir.mkdir(parents=True, exist_ok=True)
            poss.to_parquet(args.forward_dir / f"{spec.name}_poss.parquet", index=False)
            margins.to_parquet(args.forward_dir / f"{spec.name}_margins.parquet", index=False)
            origins.to_parquet(args.forward_dir / f"{spec.name}_origins.parquet", index=False)
            out = args.out / label
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(out / "ratings.parquet", index=False)
            (out / "summary.json").write_text(json.dumps({"spec": spec.__dict__, "wmse": wmse,
                                                          "seconds": time.time() - t0}, indent=2))
            print(f"[{spec.name}] wmse train {wmse['train']:.5f}  val {wmse['val']:.5f}  holdout {wmse['holdout']:.5f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
