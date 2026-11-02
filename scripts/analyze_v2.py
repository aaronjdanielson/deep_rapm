"""Calibration checks of a v2 candidate against v1 (docs/rapm_v2_design.md §6).

    .venv/bin/python scripts/analyze_v2.py --model outputs/candidates/v2.1-A/model.pt

Walks every origin with the candidate's saved prior network (no training) and,
from the same Gram state at each origin, the v1 solution (prior mean 0,
penalty 2000, no home term) and v1 + home. On every out-of-sample origin
(validation + holdout) it records the per-possession predictions of all
three for the next window, plus how many of each model's top-10 players are
on the floor. Then:

1. Star calibration — weighted mean residual (actual − predicted) on
   possessions by the number of top-10 players on offense / on defense. A
   model that exaggerates its stars predicts too much when they are on
   offense (negative residual) and too little when they defend.
2. Prior-cap saturation — share of players whose prior mean is within 10%
   of the ±5-per-100 cap, over time and for the top 10.
3. Ranking uncertainty at the last origin — posterior covariance of total
   RAPM (α_off − δ_def) for each model's top 20, Monte Carlo rank
   distribution (P(#1), median rank, 90% rank interval) and P(ahead) for
   adjacent pairs. Conditional on the prior network; network-parameter
   uncertainty is not included.

Writes <model dir>/analysis/{predictions,cap,ranks_v1,ranks_cand}.parquet.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from deep_rapm.forward import KINDS_MAIN, eval_weights, posterior_inverse
from deep_rapm.production import RapmConfig
from deep_rapm.rolling import IncrementalGramState
from deep_rapm.torch_rapm import FitSpec, _lag_features, _window, new_model, prepare

CAP = 5.0
TOP = 10


def rank_table(A_inv: np.ndarray, s2: float, beta: np.ndarray, P: int, qualified: np.ndarray,
               names: pd.Series, vocab_ids: np.ndarray, n_top: int = 20, draws: int = 20000,
               seed: int = 0) -> pd.DataFrame:
    rapm = (beta[:P] - beta[P:2 * P]) * 100
    cand = np.flatnonzero(qualified)
    top = cand[np.argsort(-rapm[cand])][:n_top]
    L = np.zeros((len(top), len(A_inv)))
    L[np.arange(len(top)), top] = 1.0
    L[np.arange(len(top)), P + top] = -1.0
    cov = 1e4 * s2 * (L @ A_inv @ L.T)
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(rapm[top], cov, size=draws, method="cholesky")
    ranks = (-sims).argsort(axis=1).argsort(axis=1) + 1
    sd = np.sqrt(np.diag(cov))
    out = pd.DataFrame({
        "player": [names.get(int(vocab_ids[i]), str(i)) for i in top],
        "rapm": rapm[top], "sd": sd,
        "p_first": (ranks == 1).mean(0), "rank_med": np.median(ranks, 0),
        "rank_lo90": np.percentile(ranks, 5, axis=0), "rank_hi90": np.percentile(ranks, 95, axis=0),
    })
    # P(this player ahead of the next one down), from the covariance of the difference
    p_next = [np.nan] * len(top)
    for k in range(len(top) - 1):
        d = rapm[top[k]] - rapm[top[k + 1]]
        v = cov[k, k] + cov[k + 1, k + 1] - 2 * cov[k, k + 1]
        p_next[k] = float((sims[:, k] > sims[:, k + 1]).mean())
        out.loc[k, "sd_diff_next"] = np.sqrt(v)
        out.loc[k, "corr_next"] = cov[k, k + 1] / np.sqrt(cov[k, k] * cov[k + 1, k + 1])
    out["p_ahead_of_next"] = p_next
    out.insert(0, "rank", np.arange(1, len(top) + 1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--from", dest="start", default=None,
                   help="first origin to record (default: the spec's train_until, i.e. validation + holdout)")
    args = p.parse_args()
    t0 = time.time()

    ckpt = torch.load(args.model, weights_only=False)
    spec = FitSpec(**ckpt["spec"])
    base = RapmConfig(**ckpt["base"])
    ctx = prepare(spec, base)
    model = new_model(ctx)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    cfg, P, n_cols = ctx.cfg, ctx.n_players, ctx.n_cols
    out_dir = args.model.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start or spec.train_until)
    horizon = pd.Timedelta(days=spec.horizon_days)
    vocab_ids = np.zeros(P, dtype=np.int64)
    vocab_ids[ctx.vocab["player_idx"].astype(int).values] = ctx.vocab["player_id"].astype(int).values

    pen_h = cfg.penalty(P)                     # v1 + home: 2000 on players, home unpenalised
    pen_1 = pen_h.copy()
    pen_1[-1] = 1e12                           # v1: home coefficient forced to 0

    state = IncrementalGramState(n_cols, cfg.alpha, cfg.half_life_days, n_players=P,
                                 offseason_discount=cfg.offseason_discount)
    history, rows, cap_rows = [], [], []
    j = 0
    last = None
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
        b_c, m_np, lam_np = beta.numpy(), m.numpy(), lam.numpy()
        history.append((t, b_c.copy(), np.diag(state.G).copy()))
        if t < start:
            continue
        b_h, _ = state.solve(pen_h)
        b_1, _ = state.solve(pen_1)
        qualified = (state.n_off >= cfg.min_poss) | (state.n_def >= cfg.min_poss)
        qualified[0] = False
        tops = {}
        for name, b in (("v1", b_1), ("cand", b_c)):
            r = (b[:P] - b[P:2 * P]) * 100
            c = np.flatnonzero(qualified)
            ind = np.zeros(P)
            ind[c[np.argsort(-r[c])][:TOP]] = 1.0
            tops[name] = ind
        prior_o, prior_d = m_np[:P] * 100, -m_np[P:2 * P] * 100
        estab = qualified & (state.n_off >= 2000)
        top_c = tops["cand"] > 0
        cap_rows.append({"origin": t,
                         "share_estab_near_cap_off": float((np.abs(prior_o[estab]) > 0.9 * CAP).mean()),
                         "share_estab_near_cap_def": float((np.abs(prior_d[estab]) > 0.9 * CAP).mean()),
                         "top10_near_cap_off": int((np.abs(prior_o[top_c]) > 0.9 * CAP).sum()),
                         "top10_near_cap_def": int((np.abs(prior_d[top_c]) > 0.9 * CAP).sum()),
                         "top10_prior_rapm": float((prior_o[top_c] + prior_d[top_c]).mean()),
                         "top10_rapm_cand": float(((b_c[:P] - b_c[P:2 * P]) * 100)[top_c].mean()),
                         "top10_rapm_v1": float(((b_1[:P] - b_1[P:2 * P]) * 100)[tops["v1"] > 0].mean())})
        for d in _window(ctx.dates, j, t, horizon):
            if d.kind not in KINDS_MAIN:
                continue
            Xo, Xd = d.X[:, :P], d.X[:, P:2 * P]
            rows.append(pd.DataFrame({
                "origin": t, "date": d.date, "season": d.season, "kind": d.kind,
                "w": eval_weights(d.score_diff), "y": d.y,
                "pred_v1": mu + d.X @ b_1, "pred_v1h": mu + d.X @ b_h, "pred_cand": mu + d.X @ b_c,
                "off_top_v1": Xo @ tops["v1"], "def_top_v1": Xd @ tops["v1"],
                "off_top_cand": Xo @ tops["cand"], "def_top_cand": Xd @ tops["cand"],
            }))
        last = (t, b_1, b_c, lam_np, qualified.copy())
        print(f"  {t.date()}  ({time.time() - t0:.0f}s)", flush=True) if t.month == 1 and t.day <= 7 else None

    pred = pd.concat(rows, ignore_index=True)
    pred.to_parquet(out_dir / "predictions.parquet", index=False)
    cap = pd.DataFrame(cap_rows)
    cap.to_parquet(out_dir / "cap.parquet", index=False)

    t, b_1, b_c, lam_c, qualified = last
    names = ctx.name_lookup
    A1 = posterior_inverse(state.G, pen_1)
    Ac = posterior_inverse(state.G, lam_c)
    r1 = rank_table(A1, state.noise_variance(b_1), b_1, P, qualified, names, vocab_ids)
    rc = rank_table(Ac, state.noise_variance(b_c), b_c, P, qualified, names, vocab_ids)
    r1.to_parquet(out_dir / "ranks_v1.parquet", index=False)
    rc.to_parquet(out_dir / "ranks_cand.parquet", index=False)

    # ── report ────────────────────────────────────────────────────────────────
    pd.set_option("display.width", 220)
    pred["period"] = np.where(pred["origin"] >= pd.Timestamp(spec.val_until), "holdout", "val")
    print(f"\n=== 1. Residuals (actual − predicted, per 100 possessions, non-garbage-time weighted) "
          f"on {pred.origin.nunique()} out-of-sample origins ===")
    for who in ("cand", "v1"):
        for side in ("off", "def"):
            col = f"{side}_top_{who}"
            g = pred.assign(n=np.minimum(pred[col].round().astype(int), 2)).groupby("n")
            tab = []
            for n, x in g:
                w = x["w"].values
                row = {"top10 on " + side: f"{n}{'+' if n == 2 else ''}", "poss": len(x)}
                for mdl in ("v1", "v1h", "cand"):
                    res = x["y"].values - x[f"pred_{mdl}"].values
                    mres = (w * res).sum() / w.sum()
                    se = np.sqrt((w * w * (res - mres) ** 2).sum()) / w.sum()
                    row[f"resid_{mdl}"] = 100 * mres
                    row[f"se_{mdl}"] = 100 * se
                tab.append(row)
            print(f"\n-- possessions by number of {'candidate' if who == 'cand' else 'v1'} top-10 players on {side}ense --")
            print(pd.DataFrame(tab).to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    print("\n=== 2. Prior-mean cap (±5 per 100) ===")
    print(cap.assign(year=cap.origin.dt.year).groupby("year")[[
        "share_estab_near_cap_off", "share_estab_near_cap_def", "top10_near_cap_off", "top10_near_cap_def",
        "top10_prior_rapm", "top10_rapm_cand", "top10_rapm_v1"]].mean().round(3).to_string())

    for label, r in (("v1", r1), ("candidate", rc)):
        print(f"\n=== 3. Ranking uncertainty at {t.date()}, {label} top 20 (posterior covariance, 20,000 draws) ===")
        print(r.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n({time.time() - t0:.0f}s; written to {out_dir})")


if __name__ == "__main__":
    main()
