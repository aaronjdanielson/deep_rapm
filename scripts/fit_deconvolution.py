"""Fit density priors by deconvolving season-level evidence, then score them in the RAPM solve.

    caffeinate -ims .venv/bin/python scripts/fit_deconvolution.py                 # all models, ~1–2 h
    caffeinate -ims .venv/bin/python scripts/fit_deconvolution.py --models maf4 --no-plugin

Models (prior over each player's joint (ORAPM, DRAPM); all use the v2 features
— phase-1 inputs plus on-court share statistics — and no lag features):
  ref        v1's prior N(0, s²/λ I), nothing learned (baseline log-likelihood)
  mean       one Gaussian, learned mean, fixed covariance (the deconvolution analogue of v2.1-A)
  joint      one Gaussian, learned mean and full 2×2 covariance (O/D correlation)
  mix3       mixture of 3 bivariate Gaussians
  maf4       conditional MAF, 4 MADE blocks
Held-out log-likelihood (validation seasons) chooses among them; --plugin
(default) then runs each through the weekly forward harness as a fixed prior.
Writes outputs/deconv/{targets.parquet, results.json, <model>/…} and
outputs/forward/dc-<model>_*.parquet. See deep_rapm/deconvolution.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from deep_rapm.deconvolution import (build_targets, build_window_targets, loglik_mixture, score_plugin,
                                     train_deconv)
from deep_rapm.density_prior import DensityModel, DensitySpec, FlowModel
from deep_rapm.priors import FEATURE_SETS
from deep_rapm.production import V1
from deep_rapm.torch_rapm import LAG_FEATURES, prepare

MODELS = {
    "ref": dict(family="mixture", components=1, precision="fixed"),
    "mean": dict(family="mixture", components=1, precision="fixed"),
    "joint": dict(family="mixture", components=1, precision="full"),
    "mix3": dict(family="mixture", components=3, precision="full"),
    "maf4": dict(family="maf", flow_blocks=4),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--until", default="2026-10-24")
    p.add_argument("--no-plugin", action="store_true")
    p.add_argument("--targets", default="season", choices=["season", "game"],
                   help="season: exact per-season marginal likelihoods; game: per-game residual measurements "
                        "on the RAPM scale, summed by weekly window")
    p.add_argument("--out", type=Path, default=Path("outputs/deconv"))
    args = p.parse_args()
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    base = DensitySpec(name="dc", feature_set="v2", shrink_minutes=500.0, lag_days=10 ** 6)
    ctx = prepare(base, V1, until=args.until)
    data = build_targets(ctx) if args.targets == "season" else build_window_targets(ctx)
    tag = "" if args.targets == "season" else "g"
    data["targets"].to_parquet(args.out / f"targets_{args.targets}.parquet", index=False)
    n_features = len(FEATURE_SETS["v2"]) + len(LAG_FEATURES)
    results = {}
    for name in args.models:
        spec = DensitySpec(name=f"dc{tag}-{name}", feature_set="v2", shrink_minutes=500.0, lag_days=10 ** 6,
                           mean_cap=0.08, **MODELS[name])
        ctx.spec = spec
        cls = FlowModel if spec.family == "maf" else DensityModel
        torch.manual_seed(spec.seed)
        model = cls(spec, n_features, ctx.feat_mean, ctx.feat_std)
        print(f"\n[{spec.name}] {MODELS[name]}", flush=True)
        if name == "ref":
            model.eval()
            tg, X = data["targets"], data["X"]
            from deep_rapm.deconvolution import _tensors
            t_feat = pd.to_datetime(tg["t_feat"])
            split = np.where(t_feat < pd.Timestamp(spec.train_until), "train",
                             np.where(t_feat < pd.Timestamp(spec.val_until), "val", "holdout"))
            with torch.no_grad():
                res = {"best_epoch": None, "loglik": {k: float(loglik_mixture(model, *_tensors(tg, X, split == k)).mean())
                                                      for k in ("train", "val", "holdout")}}
        else:
            res = train_deconv(model, spec, data, spec.train_until, spec.val_until)
        print(f"  held-out log-likelihood per player-season: {json.dumps({k: round(v, 4) for k, v in res['loglik'].items()})}"
              f"  (best epoch {res['best_epoch']})", flush=True)
        if not args.no_plugin:
            res["forward_wmse"] = score_plugin(model, spec, ctx, args.out / f"{tag}{name}")
        res.pop("log", None)
        results[name] = res
        (args.out / f"results_{args.targets}.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
