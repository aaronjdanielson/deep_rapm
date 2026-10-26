"""Forward-chaining scorer for rolling RAPM configurations.

At each origin date t the model is fit on every possession dated ≤ t — the
same incremental state the production rolling fit carries — and predicts the
points of every possession in (t, t + horizon]. Test possessions are
weighted for non-garbage time with a fixed Gaussian in the score margin
(``EVAL_SIGMA``: a 23-point margin counts 5%), so the score measures the
model where there is signal, the same emphasis the fit itself uses. The
weight is a constant, not a config field, so every candidate is scored on
the same test set. Preseason possessions are scored but reported
separately; the headline numbers are regular season + playoffs.

Per (config, penalty, season, kind):
    wmse        weighted mean squared error of points per possession
    wmse_base   the same for the intercept-only prediction (league mean at t)
    r2          1 − wmse / wmse_base
and per game, the competitive-time margin (home − away, both sides weighted
the same way) predicted vs actual, against the zero-margin baseline.

The origin step and the horizon default to 7 days, so the windows tile the
timeline and every possession is scored exactly once, 0–7 days ahead — the
production cadence. The ridge penalty only enters at solve time, so one
pass scores a whole grid of (alpha_off, alpha_def) pairs. Everything else
(half-life, σ, context weights, home court, off-season discount) needs its
own pass.

The competition σ is calibrated on the possessions before the first origin
(two seasons by default) and held fixed, so nothing about the test period
leaks into the training weights. On the production rule this differs from
the all-data calibration by a few tenths of a point.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp

from .production import V1, RapmConfig
from .rapm import _build_matrix_from_df
from .rolling import IncrementalGramState

EVAL_SIGMA   = 13.3        # test weight exp(−(score diff / σ)²)
HOLDOUT_FROM = "2025-26"   # seasons from here on are never used to choose a model
VAL_FROM     = "2023-24"   # with three_way=True: seasons in [VAL_FROM, HOLDOUT_FROM) are "val"
KINDS_MAIN   = ("regular_season", "playoffs")

VARIANTS: dict[str, RapmConfig] = {
    "v1":            V1,
    "v1_home":       V1.with_(name="v1_home", home_court=True),
    "hl180":         V1.with_(name="hl180", half_life_days=180.0),
    "hl730":         V1.with_(name="hl730", half_life_days=730.0),
    "offseason_0.5": V1.with_(name="offseason_0.5", offseason_discount=0.5),
    "no_preseason":  V1.with_(name="no_preseason", include_preseason=False),
    "comp_off":      V1.with_(name="comp_off", comp_sigma=float("inf")),
    "comp_p90":      V1.with_(name="comp_p90", comp_target_pct=90.0),
}

PENALTY_GRID: tuple[tuple[float, float], ...] = (
    (1000.0, 1000.0), (2000.0, 2000.0), (4000.0, 4000.0), (8000.0, 8000.0),
    (2000.0, 4000.0), (1000.0, 2000.0),
)


def eval_weights(score_diff: np.ndarray) -> np.ndarray:
    return np.exp(-(np.abs(score_diff) / EVAL_SIGMA) ** 2)


# ---------------------------------------------------------------------------
# Posterior predictive density
# ---------------------------------------------------------------------------
#
# Under y = μ + Xβ + ε, ε ~ N(0, s²/w) with the training weights read as
# precisions and β ~ N(m, s² Λ⁻¹), the posterior is N(β̂, s²(G + Λ)⁻¹). A
# test possession then has predictive variance s² (1 + xᵀ(G + Λ)⁻¹x), and a
# game's competitive-time margin Σ sign·w·y has variance
# s² Σ w² + aᵀ s²(G + Λ)⁻¹ a with a = Σ sign·w·x. The recency weights are an
# approximation of sampling precision, so these are pseudo-posterior
# quantities; the scores compare models, they are not absolute calibration.

def posterior_inverse(G: np.ndarray, penalty: float | np.ndarray) -> np.ndarray:
    A = G.copy()
    A.flat[::len(A) + 1] += penalty
    c = scipy.linalg.cho_factor(A, overwrite_a=True, check_finite=False)
    return scipy.linalg.cho_solve(c, np.eye(len(G)), check_finite=False)


def predictive_terms(A_inv: np.ndarray, s2: float, d: "GameDate") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(per-possession predictive variance, per-game margin variance, baseline margin variance)."""
    XA = d.X @ A_inv                                                   # (n, P) dense
    q = np.asarray(d.X.multiply(XA).sum(axis=1)).ravel()
    pred_var = s2 * (1.0 + q)
    w = eval_weights(d.score_diff)
    n_games = len(d.game_ids)
    S = sp.csr_matrix((d.sign * w, (d.game_codes, np.arange(len(w)))), shape=(n_games, len(w)))
    a = S @ d.X                                                        # (n_games, P) sparse
    aA = a @ A_inv
    base_var = s2 * np.bincount(d.game_codes, weights=w * w, minlength=n_games)
    margin_var = base_var + np.asarray(a.multiply(aA).sum(axis=1)).ravel()
    return pred_var, margin_var, base_var


def _gauss_nll(err: np.ndarray, var: np.ndarray | float) -> np.ndarray:
    return 0.5 * np.log(2 * np.pi * var) + err * err / (2 * var)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_possessions(
    cfg: RapmConfig,
    data_dir: Path = Path("data"),
    exclude_game_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Every possession the config trains on, with season / kind / context_weight."""
    frames = []
    for season in cfg.seasons:
        for pq in sorted((Path(data_dir) / season).glob("possessions_*.parquet")):
            kind = pq.stem.split("_", 3)[-1]           # possessions_2024_25_regular_season
            if kind == "playoffs" and not cfg.include_playoffs:
                continue
            if kind == "preseason" and not cfg.include_preseason:
                continue
            df = pd.read_parquet(pq)
            df["season"] = season
            df["kind"] = kind
            df["context_weight"] = {"preseason": cfg.preseason_weight,
                                    "playoffs": cfg.playoff_weight}.get(kind, 1.0)
            frames.append(df)
    poss = pd.concat(frames, ignore_index=True)
    if exclude_game_ids:
        poss = poss[~poss["game_id"].isin(set(exclude_game_ids))]
    poss["game_date"] = pd.to_datetime(poss["game_date"])
    return poss.sort_values(["game_date", "game_id", "seconds_into_game"]).reset_index(drop=True)


def default_first_origin(data_dir: Path, seasons: Sequence[str]) -> pd.Timestamp:
    """First game date of the third season, over every kind of game.

    Independent of which kinds a config trains on, so every candidate shares
    the same origin grid and per-origin differences stay paired.
    """
    season = seasons[2] if len(seasons) > 2 else seasons[-1]
    dates = [pd.read_parquet(pq, columns=["game_date"])["game_date"].min()
             for pq in sorted((Path(data_dir) / season).glob("possessions_*.parquet"))]
    return pd.Timestamp(min(dates))


@dataclass
class GameDate:
    date: pd.Timestamp
    season: str
    kind: str
    X: sp.csr_matrix
    y: np.ndarray
    score_diff: np.ndarray
    context: np.ndarray
    n_off: np.ndarray
    n_def: np.ndarray
    game_codes: np.ndarray     # possession → game index within this batch
    game_ids: np.ndarray       # game index → game id
    sign: np.ndarray           # +1 home offense, −1 away offense


def build_game_dates(poss: pd.DataFrame, vocab: pd.DataFrame, home_court: bool) -> list[GameDate]:
    out = []
    for (gd, season, kind), grp in poss.groupby(["game_date", "season", "kind"], sort=True):
        X, y, n_off, n_def = _build_matrix_from_df(grp, vocab, home_court=home_court)
        codes, ids = pd.factorize(grp["game_id"].values)
        out.append(GameDate(
            date=pd.Timestamp(gd), season=season, kind=kind, X=X, y=y.astype(np.float64),
            score_diff=grp["score_diff"].values.astype(np.float64),
            context=grp["context_weight"].values.astype(np.float64),
            n_off=n_off, n_def=n_def, game_codes=codes, game_ids=np.asarray(ids),
            sign=np.where(grp["offense_team_id"].values == grp["home_team_id"].values, 1.0, -1.0),
        ))
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class WindowScorer:
    """Accumulates the harness metrics for predictions on test windows.

    ``penalties[p]`` labels prediction set ``p`` (an (alpha_off, alpha_def)
    pair, or any pair of numbers that identifies a model). ``frames()`` returns
    the per-(penalty, season, kind), per-game and per-origin tables that
    ``summarize`` / ``paired_table`` read.
    """

    def __init__(self, name: str, penalties: Sequence[tuple[float, float]]) -> None:
        self.name = name
        self.penalties = list(penalties)
        # sum_w, Σw·err², Σw·base_err², n, Σw·nll, Σw·base_nll, Σw where nll known
        self.acc: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(7))
        self.acc_o: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(6))
        self.margin_rows: list[tuple] = []

    def add(self, p: int, t: pd.Timestamp, d: "GameDate", yhat: np.ndarray, mu: float,
            pred_var: np.ndarray | None = None, s2: float | None = None,
            margin_var: np.ndarray | None = None, base_var: np.ndarray | None = None) -> None:
        """Score one window. With ``pred_var`` (and ``s2`` for the intercept-only
        baseline) the Gaussian predictive log density is scored as well."""
        w = eval_weights(d.score_diff)
        err = d.y - yhat
        base = d.y - mu
        if pred_var is not None:
            nll, nll_b, w_nll = (w * _gauss_nll(err, pred_var)).sum(), (w * _gauss_nll(base, s2)).sum(), w.sum()
        else:
            nll = nll_b = w_nll = 0.0
        self.acc[(p, d.season, d.kind)] += (w.sum(), (w * err * err).sum(), (w * base * base).sum(),
                                            len(w), nll, nll_b, w_nll)
        self.acc_o[(p, t, d.kind)] += (w.sum(), (w * err * err).sum(), (w * base * base).sum(), nll, nll_b, w_nll)
        n_games = len(d.game_ids)
        pred   = np.bincount(d.game_codes, weights=d.sign * w * yhat, minlength=n_games)
        actual = np.bincount(d.game_codes, weights=d.sign * w * d.y, minlength=n_games)
        basem  = np.bincount(d.game_codes, weights=d.sign * w * mu, minlength=n_games)
        mv = margin_var if margin_var is not None else np.full(n_games, np.nan)
        bv = base_var if base_var is not None else np.full(n_games, np.nan)
        for g in range(n_games):
            self.margin_rows.append((p, d.game_ids[g], d.season, d.kind, d.date,
                                     pred[g], actual[g], basem[g], mv[g], bv[g]))

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pen = self.penalties
        rows = [{"config": self.name, "alpha_off": pen[p][0], "alpha_def": pen[p][1],
                 "season": season, "kind": kind, "n": int(n), "sum_w": sw,
                 "wmse": se / sw, "wmse_base": sb / sw, "r2": 1 - se / sb,
                 "snll": nll, "snll_base": nll_b, "w_nll": wn}
                for (p, season, kind), (sw, se, sb, n, nll, nll_b, wn) in self.acc.items()]
        poss = pd.DataFrame(rows, columns=["config", "alpha_off", "alpha_def", "season", "kind", "n",
                                           "sum_w", "wmse", "wmse_base", "r2", "snll", "snll_base", "w_nll"])
        poss = poss.sort_values(["alpha_off", "alpha_def", "season", "kind"]).reset_index(drop=True)
        margins = pd.DataFrame(self.margin_rows, columns=["p", "game_id", "season", "kind", "date",
                                                          "pred", "actual", "base", "var", "base_var"])
        margins.insert(0, "config", self.name)
        margins["alpha_off"] = [pen[p][0] for p in margins["p"]]
        margins["alpha_def"] = [pen[p][1] for p in margins["p"]]
        margins = margins.drop(columns="p")
        origins = pd.DataFrame(
            [(self.name, pen[p][0], pen[p][1], t, kind, sw, se, sb, nll, nll_b, wn)
             for (p, t, kind), (sw, se, sb, nll, nll_b, wn) in self.acc_o.items()],
            columns=["config", "alpha_off", "alpha_def", "origin", "kind", "sum_w", "se", "sb",
                     "snll", "snll_base", "w_nll"])
        return poss, margins, origins


@dataclass
class ForwardResult:
    config: RapmConfig
    penalties: list[tuple[float, float]]
    comp_sigma: float
    n_origins: int
    poss: pd.DataFrame          # per (penalty, season, kind) aggregates
    margins: pd.DataFrame       # per (penalty, game) competitive-time margins
    final_beta: np.ndarray      # solution at the last origin, first penalty
    final_mu: float
    final_date: pd.Timestamp
    origins: pd.DataFrame = field(default_factory=pd.DataFrame)   # per (penalty, origin, kind) sums


def score_forward(
    cfg: RapmConfig,
    data_dir: Path = Path("data"),
    vocab_path: Path = Path("data/player_vocab.parquet"),
    origin_step_days: int = 7,
    horizon_days: int = 7,
    penalties: Sequence[tuple[float, float]] | None = None,
    first_origin: pd.Timestamp | str | None = None,
    exclude_game_ids: Optional[Iterable[str]] = None,
    nll: bool = True,
    verbose: bool = True,
) -> ForwardResult:
    t0 = time.time()
    vocab = pd.read_parquet(vocab_path)
    n_players = len(vocab) + 1
    n_cols = 2 * n_players + cfg.n_extra_cols
    penalties = list(penalties or [(cfg.alpha, cfg.alpha_def_eff)])
    pen_vecs = [cfg.penalty(n_players, a_off, a_def) for a_off, a_def in penalties]

    poss = load_possessions(cfg, data_dir, exclude_game_ids)
    if first_origin is None:
        first_origin = default_first_origin(data_dir, cfg.seasons)
    first_origin = pd.Timestamp(first_origin)
    last_date = poss["game_date"].max()
    train_sd = np.abs(poss.loc[poss["game_date"] < first_origin, "score_diff"].values)
    comp_sigma = cfg.calibrate_sigma(train_sd)
    if verbose:
        print(f"[{cfg.name}] {len(poss):,} possessions {poss['game_date'].min().date()} → "
              f"{last_date.date()}; σ = {comp_sigma:.2f} (from {len(train_sd):,} possessions "
              f"before {first_origin.date()}); {len(penalties)} penalties")

    dates = build_game_dates(poss, vocab, cfg.home_court)
    del poss
    origins = pd.date_range(first_origin, last_date, freq=f"{origin_step_days}D")
    if origins[-1] < last_date:
        origins = origins.append(pd.DatetimeIndex([last_date]))
    horizon = pd.Timedelta(days=horizon_days)

    state = IncrementalGramState(n_cols, cfg.alpha, cfg.half_life_days,
                                 n_players=n_players, offseason_discount=cfg.offseason_discount)
    scorer = WindowScorer(cfg.name, penalties)
    beta = np.zeros(n_cols)
    mu = 0.0
    j = 0
    for i, t in enumerate(origins):
        while j < len(dates) and dates[j].date <= t:
            d = dates[j]
            state.advance_to(d.date)
            state.ingest(d.X, d.y, d.n_off, d.n_def, comp_sigma=comp_sigma,
                         score_diff=d.score_diff, context_weight=d.context)
            j += 1
        if state.total_w == 0:
            continue
        test = []
        k = j
        while k < len(dates) and dates[k].date <= t + horizon:
            test.append(dates[k])
            k += 1
        solutions = [state.solve(pen) for pen in pen_vecs]
        beta, mu = solutions[0]
        # The predictive density needs (G + Λ)⁻¹ — one inverse per origin, so
        # only the first penalty gets it.
        A_inv = posterior_inverse(state.G, pen_vecs[0]) if (nll and test) else None
        s2 = state.noise_variance(beta) if A_inv is not None else None
        for d in test:
            for p, (b, m) in enumerate(solutions):
                if p == 0 and A_inv is not None:
                    pv, mv, bv = predictive_terms(A_inv, s2, d)
                    scorer.add(p, t, d, m + d.X @ b, m, pred_var=pv, s2=s2, margin_var=mv, base_var=bv)
                else:
                    scorer.add(p, t, d, m + d.X @ b, m)
        if verbose and ((i + 1) % max(1, len(origins) // 10) == 0 or i == len(origins) - 1):
            print(f"  [{100 * (i + 1) / len(origins):5.1f}%] {t.date()}  ({time.time() - t0:.0f}s)", flush=True)

    poss_df, margins, origins_df = scorer.frames()
    return ForwardResult(cfg, penalties, comp_sigma, len(origins), poss_df,
                         margins, beta, mu, state.current_date, origins=origins_df)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _split(season: pd.Series, three_way: bool = False) -> np.ndarray:
    """"tune" / "holdout", or with ``three_way`` "train" / "val" / "holdout" —
    the split to select models on is "val"; "holdout" is reported, never chosen on."""
    s = np.asarray(season)
    if three_way:
        return np.where(s >= HOLDOUT_FROM, "holdout", np.where(s >= VAL_FROM, "val", "train"))
    return np.where(s >= HOLDOUT_FROM, "holdout", "tune")


def summarize(poss: pd.DataFrame, margins: pd.DataFrame, kinds: Sequence[str] = KINDS_MAIN,
              three_way: bool = False) -> pd.DataFrame:
    """One row per (config, penalty, split): weighted MSE, R², predictive NLL,
    margin RMSE and margin interval coverage on the main kinds."""
    p = poss[poss["kind"].isin(kinds)].copy()
    p["split"] = _split(p["season"], three_way)
    keys = ["config", "alpha_off", "alpha_def", "split"]
    p["se"] = p["wmse"] * p["sum_w"]
    p["sb"] = p["wmse_base"] * p["sum_w"]
    for c in ("snll", "snll_base", "w_nll"):
        if c not in p:
            p[c] = 0.0
    agg = p.groupby(keys).agg(n=("n", "sum"), sum_w=("sum_w", "sum"), se=("se", "sum"), sb=("sb", "sum"),
                              snll=("snll", "sum"), snll_base=("snll_base", "sum"), w_nll=("w_nll", "sum"))
    agg["wmse"] = agg["se"] / agg["sum_w"]
    agg["r2_pct"] = 100 * (1 - agg["se"] / agg["sb"])
    has = agg["w_nll"] > 0
    agg["nll"] = np.where(has, agg["snll"] / agg["w_nll"].where(has, 1), np.nan)
    agg["nll_base"] = np.where(has, agg["snll_base"] / agg["w_nll"].where(has, 1), np.nan)

    m = margins[margins["kind"].isin(kinds)].copy()
    m["split"] = _split(m["season"], three_way)
    m["e2"] = (m["actual"] - m["pred"]) ** 2
    m["b2"] = (m["actual"] - m["base"]) ** 2
    if "var" not in m:
        m["var"] = np.nan
        m["base_var"] = np.nan
    z = (m["actual"] - m["pred"]) / np.sqrt(m["var"])
    m["in50"] = (z.abs() < 0.6745).astype(float).where(m["var"].notna())
    m["in90"] = (z.abs() < 1.6449).astype(float).where(m["var"].notna())
    m["mnll"] = 0.5 * np.log(2 * np.pi * m["var"]) + z * z / 2
    zb = (m["actual"] - m["base"]) / np.sqrt(m["base_var"])
    m["mnll_base"] = 0.5 * np.log(2 * np.pi * m["base_var"]) + zb * zb / 2
    mg = m.groupby(keys).agg(games=("game_id", "size"), e2=("e2", "mean"), b2=("b2", "mean"),
                             cover50=("in50", "mean"), cover90=("in90", "mean"),
                             margin_nll=("mnll", "mean"), margin_nll_base=("mnll_base", "mean"))
    agg["margin_rmse"] = np.sqrt(mg["e2"])
    agg["margin_rmse_base"] = np.sqrt(mg["b2"])
    for c in ("games", "cover50", "cover90", "margin_nll", "margin_nll_base"):
        agg[c] = mg[c]
    return agg[["n", "games", "wmse", "r2_pct", "nll", "nll_base", "margin_rmse", "margin_rmse_base",
                "cover50", "cover90", "margin_nll", "margin_nll_base"]].reset_index()


def comparison_table(summary: pd.DataFrame, split: str = "tune") -> str:
    s = summary[summary["split"] == split].sort_values("wmse")
    cols = ["config", "alpha_off", "alpha_def", "wmse", "r2_pct", "nll", "margin_rmse", "margin_rmse_base",
            "cover50", "cover90", "margin_nll", "n", "games"]
    return s[cols].to_string(index=False, na_rep="—",
                             float_format=lambda v: f"{v:.5f}" if abs(v) < 10 else f"{v:,.1f}")


def paired_table(
    origins: pd.DataFrame,
    ref: tuple[str, float, float] = ("v1", 2000.0, 2000.0),
    split: str = "tune",
    kinds: Sequence[str] = KINDS_MAIN,
    metric: str = "wmse",
    three_way: bool = False,
) -> pd.DataFrame:
    """Every candidate's weighted MSE (or predictive NLL, ``metric="nll"``)
    minus the reference's, origin by origin.

    Both are scored on the same possessions at every origin, so the per-origin
    differences are paired; ``se`` is their standard error over origins and
    ``t`` = mean / se. Negative ``delta`` is an improvement on the reference.
    """
    o = origins[origins["kind"].isin(kinds)].copy()
    season = o["origin"].apply(lambda d: f"{d.year}-{str(d.year + 1)[2:]}" if d.month >= 7
                               else f"{d.year - 1}-{str(d.year)[2:]}")
    o = o[_split(season, three_way) == split]
    key = ["config", "alpha_off", "alpha_def"]
    if metric == "nll":
        if "snll" not in o:
            return pd.DataFrame(columns=key + ["origins", "delta", "se", "t"])
        o = o[o["w_nll"] > 0]
        per = o.groupby(key + ["origin"]).agg(sum_w=("w_nll", "sum"), se=("snll", "sum")).reset_index()
    else:
        per = o.groupby(key + ["origin"]).agg(sum_w=("sum_w", "sum"), se=("se", "sum")).reset_index()
    per["wmse"] = per["se"] / per["sum_w"]
    r = per[(per["config"] == ref[0]) & (per["alpha_off"] == ref[1]) & (per["alpha_def"] == ref[2])]
    r = r.set_index("origin")["wmse"]
    rows = []
    for k, g in per.groupby(key):
        g = g.set_index("origin")
        common = g.index.intersection(r.index)
        delta = (g.loc[common, "wmse"] - r.loc[common])
        if len(delta) < 2:
            continue
        se = delta.std(ddof=1) / np.sqrt(len(delta))
        # Weekly origins share most of their training history, so the
        # per-origin standard error understates the uncertainty; the block
        # version averages the differences by calendar month first.
        blocks = delta.groupby(common.to_period("M")).mean()
        se_block = blocks.std(ddof=1) / np.sqrt(len(blocks)) if len(blocks) > 1 else np.nan
        rows.append({"config": k[0], "alpha_off": k[1], "alpha_def": k[2], "origins": len(delta),
                     "delta": delta.mean(), "se": se, "t": delta.mean() / se if se > 0 else 0.0,
                     "blocks": len(blocks), "se_block": se_block,
                     "t_block": delta.mean() / se_block if se_block and se_block > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("delta")
