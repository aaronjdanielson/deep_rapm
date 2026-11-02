"""Fit and score v1 with a replacement-level (playing-time) prior mean.

    caffeinate -ims .venv/bin/python scripts/fit_replacement_prior.py          # ~30 min
    python3.11 scripts/score_forward.py --summarize --selection                 # next to v1, v2.1-A, …

Same snapshot (games ≤ 2026-10-24), origin grid, competition σ, test weights
and splits as the phase-1 experiments, so every number pairs with v1 and
v2.1-A. θ is fitted by exact weighted least squares on the training origins
(< 2023-07-01); the variant is chosen on validation (2023-24, 2024-25).

Variants (all keep v1's penalty 2000 on every player coefficient):
  v1-R            prior mean a + b·log(1 + minutes in the past 365 days), per side, no home term
  v1-R-mpg        … + c·minutes per game played in the past 365 days
  v1-R+home       v1-R with the home-court column
  v1-R-mpg+home   v1-R-mpg with the home-court column
and θ = 0 with no home term, which must reproduce v1 (a check of the walk).

Writes outputs/forward/<variant>_{poss,margins,origins}.parquet,
outputs/replacement_prior/{fit.json, ratings_<variant>.parquet} and prints the
selection view (train / validation) against v1 and v2.1-A.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from deep_rapm.forward import paired_table, summarize
from deep_rapm.production import V1
from deep_rapm.replacement_prior import TERMS, final_ratings, fit_theta, score, walk
from deep_rapm.torch_rapm import FitSpec, prepare

VARIANTS = {
    "v1-R": (("one", "logmin"), "nohome"),
    "v1-R-mpg": (("one", "logmin", "mpg"), "nohome"),
    "v1-R+home": (("one", "logmin"), "home"),
    "v1-R-mpg+home": (("one", "logmin", "mpg"), "home"),
}


def describe_prior(theta: np.ndarray, ratings: pd.DataFrame) -> pd.DataFrame:
    """Prior total RAPM by minutes in the past year (at a typical 20 mpg for the mpg term),
    relative to the possession-weighted average player."""
    K = len(TERMS)
    shift = (ratings["prior_orapm"] - 100 * (np.column_stack([
        np.ones(len(ratings)), np.log1p(ratings["minutes_365"]), ratings["mpg_365"]]) @ theta[:K])).iloc[0]
    rows = []
    for minutes in (0, 100, 500, 1000, 2000, 3000):
        b = np.array([1.0, np.log1p(minutes), 20.0 if minutes > 0 else 0.0])
        o, d = 100 * b @ theta[:K] + shift, 100 * b @ theta[K:] + shift
        rows.append({"minutes_past_year": minutes, "prior_orapm": o, "prior_drapm": d, "prior_rapm": o + d})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--until", default="2026-10-24")
    p.add_argument("--out", type=Path, default=Path("outputs/replacement_prior"))
    p.add_argument("--forward-dir", type=Path, default=Path("outputs/forward"))
    args = p.parse_args()
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = FitSpec(name="v1-R")
    ctx = prepare(spec, V1, until=args.until)
    walks = walk(ctx)
    train_until = pd.Timestamp(spec.train_until)

    check = score(walks["nohome"], np.zeros(2 * len(TERMS)), "v1-R-check-theta0", (2000.0, 2000.0))
    cp, _, _ = check.frames()
    ref = pd.read_parquet(args.forward_dir / "v1_poss.parquet")
    ref = ref[(ref.alpha_off == 2000) & (ref.alpha_def == 2000)]
    m = cp.merge(ref, on=["season", "kind"], suffixes=("", "_ref"))
    print(f"\ncheck θ = 0 reproduces v1: max |Δ wMSE| over {len(m)} season×kind cells = "
          f"{np.abs(m.wmse - m.wmse_ref).max():.2e}", flush=True)

    fits = {}
    for name, (terms, home) in VARIANTS.items():
        f = fit_theta(walks[home], terms, train_until)
        sc = score(walks[home], f["theta"], name, (2000.0, 2000.0))
        poss, margins, origins = sc.frames()
        poss.to_parquet(args.forward_dir / f"{name}_poss.parquet", index=False)
        margins.to_parquet(args.forward_dir / f"{name}_margins.parquet", index=False)
        origins.to_parquet(args.forward_dir / f"{name}_origins.parquet", index=False)
        ratings = final_ratings(walks[home], f["theta"], ctx)
        ratings.to_parquet(args.out / f"ratings_{name}.parquet", index=False)
        se = np.sqrt(np.clip(np.diag(f["cov"]), 0, None))
        fits[name] = {"terms": list(terms), "home": home, "theta": f["theta"].tolist(), "se": se.tolist(),
                      "theta_labels": [f"{s}_{t}" for s in ("off", "def") for t in TERMS],
                      "n_train_possessions": f["n"], "train_origins": f["origins"],
                      "prior_curve": describe_prior(f["theta"], ratings).round(3).to_dict("records")}
        print(f"\n[{name}] θ (per possession; level not identified, see module docs):")
        for lab, th, s in zip(fits[name]["theta_labels"], f["theta"], se):
            if th != 0:
                print(f"   {lab:12s} {th:+.5f} ± {s:.5f}")
        print("   prior by minutes in the past year, relative to the possession-weighted average player (per 100):")
        print(describe_prior(f["theta"], ratings).round(2).to_string(index=False))
    (args.out / "fit.json").write_text(json.dumps(
        {"until": args.until, "train_until": spec.train_until, "val_until": spec.val_until,
         "comp_sigma": ctx.comp_sigma, "variants": fits}, indent=2, default=str))

    names = ["v1", "v1_home", "v2.1-A"] + list(VARIANTS)
    o = pd.concat([pd.read_parquet(args.forward_dir / f"{n}_origins.parquet") for n in names])
    o = o[~o.config.isin(["v1", "v1_home"]) | ((o.alpha_off == 2000) & (o.alpha_def == 2000))]
    pd.set_option("display.width", 200)
    for split in ("train", "val"):
        for metric in ("wmse", "nll"):
            pt = paired_table(o, split=split, three_way=True, metric=metric)
            print(f"\n=== {split}: Δ {metric} vs v1, regular season + playoffs (negative = better) ===")
            print(pt[["config", "origins", "delta", "t", "t_block"]].to_string(
                index=False, float_format=lambda v: f"{v:+.6f}" if abs(v) < 0.01 else f"{v:+.1f}"))
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
