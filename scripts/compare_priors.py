"""One table for every prior family: forward scores on identical origins, weights and splits.

    .venv/bin/python scripts/compare_priors.py                 # selection view: train + validation
    .venv/bin/python scripts/compare_priors.py --holdout       # adds the holdout (look once, after choosing)

Rows (whatever exists in outputs/forward/): v1, v1_home, v2.1-A (frozen candidate),
v2.1-A-r / v2.2-A (paired refit without / with share statistics), the
forward-objective density priors (maf4, d1-joint, d3-mix), the deconvolution
priors trained on season targets (dc-*) and on game targets (dcg-*). For each:
weighted MSE and predictive NLL, paired Δ against v1 and against v2.1-A with
the month-blocked t, and the deconvolution log-likelihoods where available.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from deep_rapm.forward import KINDS_MAIN, paired_table, summarize

GROUPS = [
    ("ridge", ["v1", "v1_home"]),
    ("playing-time prior (exact WLS)", ["v1-R", "v1-R-mpg", "v1-R+home", "v1-R-mpg+home"]),
    ("learned mean, forward objective", ["v2.1-A", "v2.1-A-r", "v2.2-A"]),
    ("density prior, forward objective", ["d1-joint", "d3-mix", "maf4"]),
    ("density prior, season-target deconvolution", ["dc-mean", "dc-joint", "dc-mix3", "dc-maf4"]),
    ("density prior, game-target deconvolution", ["dcg-mean", "dcg-joint", "dcg-mix3", "dcg-maf4"]),
    ("state space (explicit transition)", ["ssm-phi0.98-s1-zero", "ssm-phi0.99-s1-zero", "ssm-phi0.995-s1-zero",
                                           "ssm-phi1-s1-zero", "ssm-phi0.99-s1-v2_1A_r", "ssm-phi1-s1-v2_1A_r"]),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--holdout", action="store_true")
    p.add_argument("--forward-dir", type=Path, default=Path("outputs/forward"))
    args = p.parse_args()
    fd = args.forward_dir

    names = [n for _, ns in GROUPS for n in ns if (fd / f"{n}_origins.parquet").exists()]
    group_of = {n: g for g, ns in GROUPS for n in ns}
    o = pd.concat([pd.read_parquet(fd / f"{n}_origins.parquet") for n in names], ignore_index=True)
    poss = pd.concat([pd.read_parquet(fd / f"{n}_poss.parquet") for n in names], ignore_index=True)
    mg = pd.concat([pd.read_parquet(fd / f"{n}_margins.parquet") for n in names], ignore_index=True)
    ridge = o["config"].isin(["v1", "v1_home"])
    keep = ~ridge | ((o["alpha_off"] == 2000) & (o["alpha_def"] == 2000))
    o = o[keep]
    poss = poss[~poss["config"].isin(["v1", "v1_home"]) | ((poss["alpha_off"] == 2000) & (poss["alpha_def"] == 2000))]
    mg = mg[~mg["config"].isin(["v1", "v1_home"]) | ((mg["alpha_off"] == 2000) & (mg["alpha_def"] == 2000))]
    lab = o.groupby("config")[["alpha_off", "alpha_def"]].first()
    ref = lambda n: (n, float(lab.loc[n, "alpha_off"]), float(lab.loc[n, "alpha_def"]))

    # every config's last origin must agree, or the comparison is not paired
    last = o.groupby("config")["origin"].max()
    if last.nunique() > 1:
        print("WARNING: configs scored on different snapshots:", last.dt.date.astype(str).to_dict())

    ll = {}
    for tag, fname in (("dc-", "results_season.json"), ("dcg-", "results_game.json")):
        f = Path("outputs/deconv") / fname
        if f.exists():
            for k, v in json.load(open(f)).items():
                ll[f"{tag}{k}"] = v.get("loglik", {})

    s = summarize(poss, mg, three_way=True).set_index(["config", "split"])
    splits = ["train", "val"] + (["holdout"] if args.holdout else [])
    pd.set_option("display.width", 250)
    for split in splits:
        pv1 = paired_table(o, ref=ref("v1"), split=split, three_way=True).set_index("config")
        pv1n = paired_table(o, ref=ref("v1"), split=split, three_way=True, metric="nll").set_index("config")
        # the frozen v2.1-A files are on the pre-2026-10-26 vocab; its matched refit is the paired reference
        refA = "v2.1-A-r" if "v2.1-A-r" in names else ("v2.1-A" if "v2.1-A" in names else None)
        pcA = paired_table(o, ref=ref(refA), split=split, three_way=True).set_index("config") if refA else None
        rows = []
        for n in names:
            if (n, split) not in s.index:
                continue
            r = {"group": group_of[n], "model": n, "wmse": s.loc[(n, split), "wmse"],
                 "r2_pct": s.loc[(n, split), "r2_pct"], "nll": s.loc[(n, split), "nll"],
                 "Δwmse_vs_v1": pv1.loc[n, "delta"] if n in pv1.index else np.nan,
                 "t_blk": pv1.loc[n, "t_block"] if n in pv1.index else np.nan,
                 "Δnll_vs_v1": pv1n.loc[n, "delta"] if n in pv1n.index else np.nan,
                 "Δwmse_vs_A-r": pcA.loc[n, "delta"] if pcA is not None and n in pcA.index else np.nan,
                 "t_blk_A": pcA.loc[n, "t_block"] if pcA is not None and n in pcA.index else np.nan,
                 "deconv_ll": ll.get(n, {}).get(split, np.nan)}
            rows.append(r)
        t = pd.DataFrame(rows)
        fmt = {"wmse": "{:.5f}".format, "r2_pct": "{:.3f}".format, "nll": "{:.5f}".format,
               "Δwmse_vs_v1": "{:+.6f}".format, "t_blk": "{:+.1f}".format, "Δnll_vs_v1": "{:+.6f}".format,
               "Δwmse_vs_A-r": "{:+.6f}".format, "t_blk_A": "{:+.1f}".format, "deconv_ll": "{:+.4f}".format}
        print(f"\n=== {split}: regular season + playoffs, non-garbage-time weighted; paired Δ (negative = better), "
              f"month-blocked t ===")
        print(t.to_string(index=False, formatters=fmt, na_rep="—"))
    if not args.holdout:
        print("\n(holdout hidden; pass --holdout once a model has been chosen on validation)")


if __name__ == "__main__":
    main()
