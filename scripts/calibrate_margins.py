"""Calibrate game-margin predictive variances with one multiplier per model.

    python3.11 scripts/calibrate_margins.py                      # default model list
    python3.11 scripts/calibrate_margins.py --models v1 v2.1-A

For each model the competitive-time game margin is predicted as
N(μ_g, v_g) (deep_rapm/forward.py). The variance multiplier is fitted on the
validation games only (seasons VAL_FROM … HOLDOUT_FROM, regular season +
playoffs) as the maximum-likelihood value with the means fixed,

    ĉ = mean_g (Y_g − μ_g)² / v_g,

and carried unchanged to the holdout. Reported per model and split, before
and after the multiplier: 50% / 90% interval coverage, margin NLL, and the
distribution of standardized residuals z = (Y − μ) / √(ĉ v) (mean, SD,
excess kurtosis, |z| quantiles against the normal's), plus coverage by game
kind.

The residuals are lighter-tailed than normal, so one multiplier cannot fix
the 50% and 90% levels together. The script therefore also reports
level-specific (split-conformal) intervals: half-width k_q·√v with k_q the
q-quantile of |z| on the validation games, carried unchanged to the holdout.

The margins are retrospective: μ_g sums the model's predictions over the
possessions and lineups that actually occurred in the game, so this
calibrates the RAPM model conditional on realized participation, not a
pre-tipoff forecast of the margin.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from deep_rapm.forward import HOLDOUT_FROM, KINDS_MAIN, VAL_FROM

Z50, Z90 = stats.norm.ppf(0.75), stats.norm.ppf(0.95)
NORMAL_ABS_Q = {q: stats.norm.ppf(0.5 + q / 2) for q in (0.5, 0.9, 0.99)}


def load_margins(names, forward_dir: Path) -> pd.DataFrame:
    frames = []
    for n in names:
        m = pd.read_parquet(forward_dir / f"{n}_margins.parquet")
        m = m[m["var"].notna()]                       # the penalty the density was scored at
        frames.append(m)
    m = pd.concat(frames, ignore_index=True)
    m = m[m["kind"].isin(KINDS_MAIN)].copy()
    s = m["season"].values
    m["split"] = np.where(s >= HOLDOUT_FROM, "holdout", np.where(s >= VAL_FROM, "val", "train"))
    m["z0"] = (m["actual"] - m["pred"]) / np.sqrt(m["var"])
    return m


def describe(z: np.ndarray, var: np.ndarray, err: np.ndarray) -> dict:
    a = np.abs(z)
    out = {"games": len(z), "cover50": float((a < Z50).mean()), "cover90": float((a < Z90).mean()),
           "nll": float(np.mean(0.5 * np.log(2 * np.pi * var) + err ** 2 / (2 * var))),
           "z_mean": float(z.mean()), "z_sd": float(z.std(ddof=1)), "z_exkurt": float(stats.kurtosis(z))}
    for q, nq in NORMAL_ABS_Q.items():
        out[f"|z|_q{int(q * 100)}"] = float(np.quantile(a, q))
        out[f"normal_q{int(q * 100)}"] = float(nq)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+",
                   default=["v1", "v1_home", "v2.1-A", "v2.1-C", "v2.1", "v2", "v2.1eb"])
    p.add_argument("--forward-dir", type=Path, default=Path("outputs/forward"))
    p.add_argument("--out", type=Path, default=Path("outputs/forward/margin_calibration.parquet"))
    args = p.parse_args()

    m = load_margins(args.models, args.forward_dir)
    rows, kind_rows, conf_rows = [], [], []
    for name, g in m.groupby("config", sort=False):
        val = g[g["split"] == "val"]
        c = float(np.mean(val["z0"] ** 2))
        k50, k90 = (float(np.quantile(np.abs(val["z0"]), q)) for q in (0.5, 0.9))
        for split, x in g.groupby("split"):
            a = np.abs(x["z0"].values)
            conf_rows.append({"config": name, "split": split, "k50": k50, "k90": k90,
                              "k50/z50": k50 / Z50, "k90/z90": k90 / Z90,
                              "cover50": float((a < k50).mean()), "cover90": float((a < k90).mean())})
        for split, x in g.groupby("split"):
            err = (x["actual"] - x["pred"]).values
            for label, mult in (("raw", 1.0), ("calibrated", c)):
                v = x["var"].values * mult
                rows.append({"config": name, "c_val": c, "split": split, "variance": label,
                             **describe(err / np.sqrt(v), v, err)})
            if split == "holdout":
                for kind, k in x.groupby("kind"):
                    zc = (k["actual"] - k["pred"]).values / np.sqrt(k["var"].values * c)
                    kind_rows.append({"config": name, "kind": kind, "games": len(k),
                                      "cover50": float((np.abs(zc) < Z50).mean()),
                                      "cover90": float((np.abs(zc) < Z90).mean()),
                                      "z_sd": float(zc.std(ddof=1))})
    r = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    r.to_parquet(args.out, index=False)

    pd.set_option("display.width", 240)
    fmt = lambda v: f"{v:.3f}"
    print("Variance multiplier ĉ fitted on validation games, carried unchanged to the holdout:")
    print(r.drop_duplicates("config")[["config", "c_val"]].to_string(index=False, float_format=fmt))
    for split in ("val", "holdout"):
        note = " (in-sample for ĉ)" if split == "val" else " (ĉ from validation, unchanged)"
        print(f"\n=== {split}{note}: coverage, margin NLL, standardized residuals ===")
        cols = ["config", "variance", "games", "cover50", "cover90", "nll", "z_mean", "z_sd", "z_exkurt",
                "|z|_q50", "|z|_q90", "|z|_q99"]
        print(r[r["split"] == split][cols].sort_values(["variance", "nll"]).to_string(index=False, float_format=fmt))
    print(f"\nnormal |z| quantiles: q50 {NORMAL_ABS_Q[0.5]:.3f}, q90 {NORMAL_ABS_Q[0.9]:.3f}, q99 {NORMAL_ABS_Q[0.99]:.3f}")
    print("\n=== holdout coverage after calibration, by game kind ===")
    print(pd.DataFrame(kind_rows).to_string(index=False, float_format=fmt))
    conf = pd.DataFrame(conf_rows)
    conf.to_parquet(args.out.with_name(args.out.stem + "_conformal.parquet"), index=False)
    print("\n=== level-specific (split-conformal) intervals: k_q = validation quantile of |z|, carried to the holdout ===")
    print("    (k/z = half-width relative to the uncalibrated normal interval)")
    print(conf[conf["split"].isin(["val", "holdout"])].sort_values(["split", "config"])
          .to_string(index=False, float_format=fmt))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
