"""Prospective evaluation of the frozen models on games after the freeze.

    .venv/bin/python scripts/prospective_eval.py              # score everything since the freeze
    .venv/bin/python scripts/prospective_eval.py --report     # print the latest report again

The development phase closed on 2026-10-26 with v1 (production) and v2.1-A
(research candidate) frozen and every comparison made on games through
FREEZE_DATE. This script applies both, unchanged, to games after that date:

* the frozen v2.1-A checkpoint and the V1 configuration, walked over the same
  weekly origin grid as the experiments (each origin fits on games ≤ t and
  predicts (t, t + 7]); only test possessions dated after FREEZE_DATE count;
* calibration constants frozen with the release (the validation-fitted
  variance multiplier ĉ and the split-conformal |z| quantiles k50 / k90);
* metrics: weighted MSE, possession NLL, margin RMSE, margin NLL with ĉ,
  50% / 90% margin coverage with ĉ and with k50 / k90, per-origin paired
  differences (v2.1-A − v1), and rating drift since the freeze.

Outputs go to outputs/prospective/<last game date>/ and one summary row per
run is appended to outputs/prospective/history.parquet. Installed weekly
cron entry (Mondays, after the 04:00 nightly collection):

    30 6 * * 1 cd /Users/aarondanielson/Dropbox/deep_rapm && \\
      .venv/bin/python scripts/prospective_eval.py >> outputs/logs/prospective.log 2>&1

The models see the current vocab and player tables (which grow additively);
the release's data/ copy is the frozen snapshot for exact reproduction.
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
from scipy import stats

from deep_rapm.forward import (KINDS_MAIN, WindowScorer, posterior_inverse, predictive_terms,
                               summarize)
from deep_rapm.production import RapmConfig
from deep_rapm.rolling import IncrementalGramState
from deep_rapm.torch_rapm import FitSpec, _lag_features, _window, new_model, prepare

FREEZE_DATE = pd.Timestamp("2026-10-24")
RELEASE = Path("outputs/releases/rapm-phase1_2026-10-26")
MODEL = RELEASE / "models/v2.1-A/model.pt"
CALIBRATION = RELEASE / "calibration/calibration_constants.json"
OUT = Path("outputs/prospective")
Z50, Z90 = stats.norm.ppf(0.75), stats.norm.ppf(0.95)


def run() -> Path:
    t0 = time.time()
    ckpt = torch.load(MODEL, weights_only=False)
    spec = FitSpec(**ckpt["spec"])
    base = RapmConfig(**ckpt["base"])
    ctx = prepare(spec, base, verbose=False)
    model = new_model(ctx)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    cal = json.loads(CALIBRATION.read_text())
    cfg, P, n_cols = ctx.cfg, ctx.n_players, ctx.n_cols
    last_date = max(d.date for d in ctx.dates)
    if last_date <= FREEZE_DATE:
        print(f"no games after the freeze ({FREEZE_DATE.date()}) yet")
        return None
    horizon = pd.Timedelta(days=spec.horizon_days)

    pen_1 = cfg.penalty(P)
    pen_1[-1] = 1e12                          # V1 has no home term
    state = IncrementalGramState(n_cols, cfg.alpha, cfg.half_life_days, n_players=P,
                                 offseason_discount=cfg.offseason_discount)
    scorers = {"v1": WindowScorer("v1", [(2000.0, 2000.0)]), "v2.1-A": WindowScorer("v2.1-A", [(2000.0, 2000.0)])}
    history, ratings = [], {}
    j = 0
    for t in ctx.origins:
        while j < len(ctx.dates) and ctx.dates[j].date <= t:
            d = ctx.dates[j]
            state.advance_to(d.date)
            state.ingest(d.X, d.y, d.n_off, d.n_def, comp_sigma=ctx.comp_sigma,
                         score_diff=d.score_diff, context_weight=d.context)
            j += 1
        if state.total_w == 0:
            continue
        xty, mu, _ = state.centered_rhs()
        x = torch.from_numpy(np.concatenate([ctx.feats[t], _lag_features(t, history, P, spec.lag_days)], axis=1))
        with torch.no_grad():
            beta, m, lam = model.solve(torch.from_numpy(state.G).clone(), torch.from_numpy(xty), x)
        b_c, lam_c = beta.numpy(), lam.numpy()
        history.append((t, b_c.copy(), np.diag(state.G).copy()))
        test = [d for d in _window(ctx.dates, j, t, horizon) if d.date > FREEZE_DATE and d.kind in KINDS_MAIN]
        b_1, _ = state.solve(pen_1)
        if t <= FREEZE_DATE < t + horizon or t == ctx.origins[-1]:
            qualified = (state.n_off >= cfg.min_poss) | (state.n_def >= cfg.min_poss)
            ratings[t] = {"v1": (b_1[:P] - b_1[P:2 * P]) * 100, "v2.1-A": (b_c[:P] - b_c[P:2 * P]) * 100,
                          "qualified": qualified.copy()}
        if not test:
            continue
        for name, b, pen in (("v1", b_1, pen_1), ("v2.1-A", b_c, lam_c)):
            A_inv = posterior_inverse(state.G, pen)
            s2 = state.noise_variance(b)
            for d in test:
                pv, mv, bv = predictive_terms(A_inv, s2, d)
                scorers[name].add(0, t, d, mu + d.X @ b, mu, pred_var=pv, s2=s2, margin_var=mv, base_var=bv)

    frames = [s.frames() for s in scorers.values()]
    poss = pd.concat([f[0] for f in frames], ignore_index=True)
    margins = pd.concat([f[1] for f in frames], ignore_index=True)
    origins = pd.concat([f[2] for f in frames], ignore_index=True)
    out = OUT / f"{last_date.date()}"
    out.mkdir(parents=True, exist_ok=True)
    poss.to_parquet(out / "poss.parquet", index=False)
    margins.to_parquet(out / "margins.parquet", index=False)
    origins.to_parquet(out / "origins.parquet", index=False)

    # ── metrics ──────────────────────────────────────────────────────────────
    s = summarize(poss, margins)
    s = s.groupby("config").agg(possessions=("n", "sum"), games=("games", "sum")).join(
        summarize(poss.assign(season="2026-27"), margins.assign(season="2026-27"))
        .set_index("config")[["wmse", "r2_pct", "nll", "margin_rmse"]])
    rows = []
    for name, g in margins.groupby("config"):
        g = g[g["kind"].isin(KINDS_MAIN)]
        err = (g["actual"] - g["pred"]).values
        v = g["var"].values
        c, k50, k90 = (cal[name][k] for k in ("c", "k50", "k90"))
        z = err / np.sqrt(v)
        rows.append({"config": name,
                     "margin_nll_c": float(np.mean(0.5 * np.log(2 * np.pi * c * v) + err ** 2 / (2 * c * v))),
                     "cover50_c": float((np.abs(z) / np.sqrt(c) < Z50).mean()),
                     "cover90_c": float((np.abs(z) / np.sqrt(c) < Z90).mean()),
                     "cover50_conformal": float((np.abs(z) < k50).mean()),
                     "cover90_conformal": float((np.abs(z) < k90).mean())})
    s = s.join(pd.DataFrame(rows).set_index("config"))

    o = origins[origins["kind"].isin(KINDS_MAIN)].groupby(["config", "origin"]).agg(
        sw=("sum_w", "sum"), se=("se", "sum"), sn=("snll", "sum")).reset_index()
    o["wmse"], o["nll"] = o["se"] / o["sw"], o["sn"] / o["sw"]
    w = o.pivot(index="origin", columns="config", values=["wmse", "nll"]).dropna()
    paired = {}
    for metric in ("wmse", "nll"):
        dlt = (w[(metric, "v2.1-A")] - w[(metric, "v1")]).values
        se = dlt.std(ddof=1) / np.sqrt(len(dlt)) if len(dlt) > 1 else np.nan
        paired[metric] = {"origins": len(dlt), "delta": float(dlt.mean()), "t": float(dlt.mean() / se) if se and se > 0 else np.nan}

    t_freeze = min(ratings)
    t_now = max(ratings)
    drift = {}
    for name in ("v1", "v2.1-A"):
        q = ratings[t_freeze]["qualified"] & ratings[t_now]["qualified"]
        dd = ratings[t_now][name][q] - ratings[t_freeze][name][q]
        drift[name] = {"players": int(q.sum()), "mean_abs_change": float(np.abs(dd).mean()),
                       "max_abs_change": float(np.abs(dd).max())}
    vocab_ids = np.zeros(P, dtype=np.int64)
    vocab_ids[ctx.vocab["player_idx"].astype(int).values] = ctx.vocab["player_id"].astype(int).values
    tops = {}
    for name in ("v1", "v2.1-A"):
        r_now, q = ratings[t_now][name], ratings[t_now]["qualified"].copy()
        q[0] = False
        idx = np.flatnonzero(q)[np.argsort(-r_now[q])][:15]
        tops[name] = pd.DataFrame({"player": [ctx.name_lookup.get(int(vocab_ids[i]), str(i)) for i in idx],
                                   "rapm_now": r_now[idx], "rapm_at_freeze": ratings[t_freeze][name][idx]})

    summary = {"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "freeze_date": str(FREEZE_DATE.date()),
               "last_game_date": str(last_date.date()), "origins_freeze_now": [str(t_freeze.date()), str(t_now.date())],
               "paired_v2.1-A_minus_v1": paired, "drift": drift}
    report = [f"Prospective evaluation — games after {FREEZE_DATE.date()} through {last_date.date()} "
              f"(regular season + playoffs, non-garbage-time weighted)", "",
              s.round(5).to_string(), "",
              f"paired v2.1-A − v1 per origin: {json.dumps(paired)}",
              f"rating drift since the freeze ({t_freeze.date()} → {t_now.date()}): {json.dumps(drift)}", ""]
    for name, tb in tops.items():
        report += [f"{name} top 15 now vs at the freeze:", tb.round(2).to_string(index=False), ""]
    report.append(f"({time.time() - t0:.0f}s)")
    (out / "report.txt").write_text("\n".join(report))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    hist_row = s.reset_index().assign(last_game_date=str(last_date.date()), run_at=summary["run_at"])
    hist_path = OUT / "history.parquet"
    hist = pd.concat([pd.read_parquet(hist_path), hist_row], ignore_index=True) if hist_path.exists() else hist_row
    hist.drop_duplicates(["config", "last_game_date"], keep="last").to_parquet(hist_path, index=False)
    print("\n".join(report))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--report", action="store_true", help="print the latest report instead of scoring")
    args = p.parse_args()
    if args.report:
        runs = sorted(d for d in OUT.glob("20*") if (d / "report.txt").exists())
        print((runs[-1] / "report.txt").read_text() if runs else "no prospective runs yet")
        return
    run()


if __name__ == "__main__":
    main()
