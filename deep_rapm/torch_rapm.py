"""RAPM v2 — the ridge solve with a learned prior (docs/rapm_v2_design.md).

At every origin t the coefficients are the exact solution of

    (G_t + Λ) β = X'W(y − μ_t) + Λ m

where G_t and μ_t come from the production recursion (IncrementalGramState,
with the weight hyperparameters of a base config plus a home-court column),
and the prior mean m and precision Λ are functions of as-of-origin player
features (deep_rapm.priors) and of the model's own estimate a year earlier.
A small network maps features to (m_off, m_def, log-precision multipliers)
per player; it and the base penalties log λ_off / log λ_def are trained by
gradient through the solve. β therefore keeps its meaning: it is the
posterior mode of a stated model, and the leaderboard reads as before.

Two training objectives (``FitSpec.objective``):

``forward``   the weighted squared error of the next window's regular-season
              and playoff possessions, weighted for non-garbage time exactly
              as the scorer does — the criterion v1 is judged on;
``evidence``  empirical Bayes: the negative log marginal likelihood of the
              training possessions at each origin, with β integrated out,

                −log p(y | θ) = ½ [ Q/s² + log det(G + Λ) − log det Λ ] + const,
                Q = Σw(y−μ)² − 2mᵀX'W(y−μ) + mᵀGm − rᵀ(G + Λ)⁻¹r,  r = X'W(y−μ) − Gm,

              per effective possession (divided by Σw). The noise variance
              s² is the weighted residual variance at the current β̂ and is
              not a learned parameter. The recency weights are read as
              precisions, so this is a pseudo-likelihood.

Whichever objective trains, the epoch is selected on the forward weighted
MSE of the validation origins, so the two objectives are compared on the
same footing. Origins before ``train_until`` are trained on, origins in
[train_until, val_until) select the epoch, origins from ``val_until`` on are
the holdout reported next to v1.

The evaluation pass also scores the posterior predictive density (see
deep_rapm.forward) and writes the posterior standard deviation of every
player's ORAPM and DRAPM into the v2 cache.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

from .forward import (KINDS_MAIN, GameDate, WindowScorer, build_game_dates, default_first_origin,
                      eval_weights, load_possessions, posterior_inverse, predictive_terms)
from .priors import FEATURES, build_prior_features, load_game_logs, static_features
from .production import RapmConfig
from .rolling import IncrementalGramState

LAG_FEATURES = ["orapm_lag", "drapm_lag", "ev_off_lag", "ev_def_lag", "has_lag"]
HOME_PENALTY = 1.0


@dataclass
class FitSpec:
    name: str = "v2"
    base: str = "v1"                 # weight hyperparameters (half-life, σ, context) come from this variant
    objective: str = "forward"       # "forward" | "evidence"
    origin_step_days: int = 7
    horizon_days: int = 7
    train_until: str = "2023-07-01"  # origins before this are trained on
    val_until: str = "2025-07-01"    # [train_until, val_until) selects the epoch; later = holdout
    epochs: int = 12
    patience: int = 3
    batch_origins: int = 8
    lr: float = 1e-3
    lr_alpha: float = 0.02
    hidden: int = 64
    max_prior_mean: float = 0.05     # |m| ≤ 5 points per 100
    lag_days: int = 365
    shrink_minutes: float = 0.0      # prior features: minutes of league-average play added before rates (0 = raw, as v2)
    learn_mean: bool = True          # ablations: prior mean from features (else 0)
    learn_precision: bool = True     #            prior precision from features (else the base penalty)
    learn_alpha: bool = True         #            base penalties log λ_off / log λ_def trained (else fixed)
    evidence_noise: str = "model"    # s² in the evidence: "model" (current β̂, as in v2eb) | "reference" (base ridge, fixed)
    seed: int = 0
    seasons: tuple[str, ...] | None = None   # subset for quick tests


class PriorNet(nn.Module):
    """features → (m_off, m_def in RAPM/100 units, log-precision multipliers off/def)."""

    def __init__(self, n_in: int, hidden: int, max_mean: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 4))
        self.max_mean = max_mean
        nn.init.zeros_(self.net[-1].weight)      # start exactly at the ridge: m = 0, Λ = λ
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return (self.max_mean * torch.tanh(h[:, 0]), self.max_mean * torch.tanh(h[:, 1]),
                torch.clamp(h[:, 2], -3.0, 3.0), torch.clamp(h[:, 3], -3.0, 3.0))


class V2Model(nn.Module):
    def __init__(self, n_features: int, hidden: int, max_prior_mean: float,
                 alpha_off: float, alpha_def: float, feat_mean: np.ndarray, feat_std: np.ndarray,
                 learn_mean: bool = True, learn_precision: bool = True) -> None:
        super().__init__()
        self.prior = PriorNet(n_features, hidden, max_prior_mean)
        self.log_alpha_off = nn.Parameter(torch.tensor(float(np.log(alpha_off))))
        self.log_alpha_def = nn.Parameter(torch.tensor(float(np.log(alpha_def))))
        self.register_buffer("feat_mean", torch.tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_std", torch.tensor(feat_std, dtype=torch.float32))
        self.learn_mean = learn_mean
        self.learn_precision = learn_precision

    @property
    def alphas(self) -> tuple[float, float]:
        return float(torch.exp(self.log_alpha_off.detach())), float(torch.exp(self.log_alpha_def.detach()))

    def prior_terms(self, feats: torch.Tensor):
        """Per-coefficient prior mean and precision for [offense | defense | home]."""
        x = (feats - self.feat_mean) / self.feat_std
        m_off, m_def_rapm, lp_off, lp_def = self.prior(x)
        if not self.learn_mean:
            m_off, m_def_rapm = 0.0 * m_off, 0.0 * m_def_rapm
        if not self.learn_precision:
            lp_off, lp_def = 0.0 * lp_off, 0.0 * lp_def
        lam_off = torch.exp(self.log_alpha_off + lp_off)
        lam_def = torch.exp(self.log_alpha_def + lp_def)
        one = torch.ones(1, dtype=torch.float32)
        m = torch.cat([m_off, -m_def_rapm, 0.0 * one]).double()       # δ coefficient = −DRAPM
        lam = torch.cat([lam_off, lam_def, HOME_PENALTY * one]).double()
        return m, lam

    def solve(self, G: torch.Tensor, xty: torch.Tensor, feats: torch.Tensor):
        m, lam = self.prior_terms(feats)
        A = G + torch.diag(lam)
        beta = torch.linalg.solve(A, xty + lam * m)
        return beta, m, lam

    def neg_log_evidence(self, G: torch.Tensor, xty: torch.Tensor, sy2: float, s2: float,
                         m: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """−log p(y | θ) up to terms free of θ (see module docstring); not normalised."""
        A = G + torch.diag(lam)
        L = torch.linalg.cholesky(A)
        r = xty - G @ m
        Ainv_r = torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
        Q = sy2 - 2.0 * (m @ xty) + m @ (G @ m) - r @ Ainv_r
        logdet_A = 2.0 * torch.log(torch.diagonal(L)).sum()
        return 0.5 * (Q / s2 + logdet_A - torch.log(lam).sum())


def _csr_to_torch(X: sp.csr_matrix) -> torch.Tensor:
    return torch.sparse_csr_tensor(torch.from_numpy(X.indptr.astype(np.int64)),
                                   torch.from_numpy(X.indices.astype(np.int64)),
                                   torch.from_numpy(X.data.astype(np.float64)), size=X.shape)


def _lag_features(t: pd.Timestamp, history: list[tuple], n_players: int, lag_days: int) -> np.ndarray:
    out = np.zeros((n_players, len(LAG_FEATURES)), dtype=np.float32)
    cutoff = t - pd.Timedelta(days=lag_days)
    for origin, beta, diag in reversed(history):
        if origin <= cutoff:
            out[:, 0] = beta[:n_players] * 20.0            # ±0.05 → ±1
            out[:, 1] = -beta[n_players:2 * n_players] * 20.0
            out[:, 2] = np.log1p(diag[:n_players]) / 10.0
            out[:, 3] = np.log1p(diag[n_players:2 * n_players]) / 10.0
            out[:, 4] = 1.0
            break
    return out


def _window(dates: list[GameDate], j: int, t: pd.Timestamp, horizon: pd.Timedelta) -> list[GameDate]:
    test, k = [], j
    while k < len(dates) and dates[k].date <= t + horizon:
        test.append(dates[k])
        k += 1
    return test


# ---------------------------------------------------------------------------
# Data context shared by training, evaluation and re-scoring
# ---------------------------------------------------------------------------

@dataclass
class Context:
    spec: FitSpec
    base_cfg: RapmConfig
    cfg: RapmConfig
    vocab: pd.DataFrame
    n_players: int
    dates: list[GameDate]
    origins: pd.DatetimeIndex
    comp_sigma: float
    feats: dict[pd.Timestamp, np.ndarray]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    name_lookup: pd.Series

    @property
    def n_cols(self) -> int:
        return 2 * self.n_players + 1

    def split_of(self, t: pd.Timestamp) -> str:
        if t < pd.Timestamp(self.spec.train_until):
            return "train"
        return "val" if t < pd.Timestamp(self.spec.val_until) else "holdout"


def prepare(spec: FitSpec, base_cfg: RapmConfig, data_dir: Path = Path("data"),
            vocab_path: Path = Path("data/player_vocab.parquet"),
            player_table_path: Path = Path("data/players.parquet"), verbose: bool = True) -> Context:
    cfg = base_cfg.with_(name=spec.name, home_court=True,
                         **({"seasons": tuple(spec.seasons)} if spec.seasons else {}))
    vocab = pd.read_parquet(vocab_path)
    n_players = len(vocab) + 1
    poss = load_possessions(cfg, data_dir)
    first_origin = default_first_origin(data_dir, cfg.seasons)
    last_date = poss["game_date"].max()
    comp_sigma = cfg.calibrate_sigma(np.abs(poss.loc[poss["game_date"] < first_origin, "score_diff"].values))
    dates = build_game_dates(poss, vocab, home_court=True)
    del poss
    for d in dates:
        d.Xt = _csr_to_torch(d.X)
    origins = pd.date_range(first_origin, last_date, freq=f"{spec.origin_step_days}D")
    if origins[-1] < last_date:
        origins = origins.append(pd.DatetimeIndex([last_date]))
    if verbose:
        print(f"[{spec.name}] base {base_cfg.name}, objective {spec.objective}; {len(dates)} game dates; "
              f"{len(origins)} origins {origins[0].date()} → {origins[-1].date()}; σ = {comp_sigma:.2f}; "
              f"train < {spec.train_until}, val < {spec.val_until}", flush=True)
    feats = build_prior_features(origins, vocab, load_game_logs(), static_features(vocab),
                                 shrink_minutes=spec.shrink_minutes, verbose=verbose)
    ctx = Context(spec, base_cfg, cfg, vocab, n_players, dates, origins, comp_sigma, feats,
                  np.zeros(1), np.ones(1),
                  pd.read_parquet(player_table_path).sort_values("season")
                  .drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"])
    train_stack = np.concatenate([feats[t][1:] for t in origins if ctx.split_of(t) == "train"])
    ctx.feat_mean = np.concatenate([train_stack.mean(0), np.zeros(len(LAG_FEATURES))]).astype(np.float32)
    ctx.feat_std = np.concatenate([np.maximum(train_stack.std(0), 1e-6), np.ones(len(LAG_FEATURES))]).astype(np.float32)
    return ctx


def new_model(ctx: Context) -> V2Model:
    return V2Model(len(FEATURES) + len(LAG_FEATURES), ctx.spec.hidden, ctx.spec.max_prior_mean,
                   ctx.cfg.alpha, ctx.cfg.alpha_def_eff, ctx.feat_mean, ctx.feat_std,
                   learn_mean=ctx.spec.learn_mean, learn_precision=ctx.spec.learn_precision)


# ---------------------------------------------------------------------------
# One walk over the origins
# ---------------------------------------------------------------------------

def run_epoch(ctx: Context, model: V2Model, opt: torch.optim.Optimizer | None,
              scorer: WindowScorer | None = None, cache_rows: list | None = None):
    """Walk every origin once. With ``opt`` the training origins contribute
    gradient steps; with ``scorer`` / ``cache_rows`` the pass also scores the
    windows (including the predictive density) and records the cache.
    Returns per-split forward weighted MSE, the final β, μ and date."""
    spec, cfg, n_players = ctx.spec, ctx.cfg, ctx.n_players
    horizon = pd.Timedelta(days=spec.horizon_days)
    state = IncrementalGramState(ctx.n_cols, cfg.alpha, cfg.half_life_days,
                                 n_players=n_players, offseason_discount=cfg.offseason_discount)
    history: list[tuple] = []
    sums = {s: np.zeros(2) for s in ("train", "val", "holdout")}
    j, n_batch = 0, 0
    beta_np, mu = np.zeros(ctx.n_cols), 0.0
    if opt is not None:
        opt.zero_grad()
    for t in ctx.origins:
        while j < len(ctx.dates) and ctx.dates[j].date <= t:
            d = ctx.dates[j]
            state.advance_to(d.date)
            state.ingest(d.X, d.y, d.n_off, d.n_def, comp_sigma=ctx.comp_sigma,
                         score_diff=d.score_diff, context_weight=d.context)
            j += 1
        if state.total_w == 0:
            continue
        split = ctx.split_of(t)
        xty_np, mu, sy2 = state.centered_rhs()
        G = torch.from_numpy(state.G).clone()
        xty = torch.from_numpy(xty_np)
        x = torch.from_numpy(np.concatenate(
            [ctx.feats[t], _lag_features(t, history, n_players, spec.lag_days)], axis=1))
        grad = opt is not None and split == "train"
        test = _window(ctx.dates, j, t, horizon)
        # While training, holdout origins are neither scored nor logged; they
        # are looked at once, in the evaluation pass after model selection.
        score_window = opt is None or split != "holdout"
        with torch.set_grad_enabled(grad):
            beta, m, lam = model.solve(G, xty, x)
            beta_np = beta.detach().numpy()
            num = torch.zeros((), dtype=torch.float64)
            den = 0.0
            for d in (test if score_window else []):
                if d.kind not in KINDS_MAIN:
                    continue
                w = torch.from_numpy(eval_weights(d.score_diff))
                yhat = mu + d.Xt @ beta
                num = num + (w * (torch.from_numpy(d.y) - yhat) ** 2).sum()
                den += float(w.sum())
            if grad:
                if spec.objective == "evidence":
                    # s² is held fixed within the step either way; "reference" takes it
                    # from the base ridge (m = 0, Λ = λ_base) so it does not move with the
                    # prior being learned, and the dropped n·log s² term is then a constant.
                    if spec.evidence_noise == "reference":
                        s2 = state.noise_variance(state.solve(cfg.penalty(n_players))[0])
                    else:
                        s2 = state.noise_variance(beta_np)
                    loss = model.neg_log_evidence(G, xty, sy2, s2, m, lam) / state.total_w
                else:
                    loss = num / den if den > 0 else None
                if loss is not None:
                    (loss / spec.batch_origins).backward()
                    n_batch += 1
                    if n_batch == spec.batch_origins:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                        opt.zero_grad()
                        n_batch = 0
        history.append((t, beta_np.copy(), np.diag(state.G).copy()))
        sums[split] += (float(num.detach()), den)
        if scorer is not None or cache_rows is not None:
            lam_np = lam.detach().numpy()
            A_inv = posterior_inverse(state.G, lam_np)
            s2 = state.noise_variance(beta_np)
            if scorer is not None:
                for d in test:
                    pv, mv, bv = predictive_terms(A_inv, s2, d)
                    scorer.add(0, t, d, mu + d.X @ beta_np, mu, pred_var=pv, s2=s2, margin_var=mv, base_var=bv)
            if cache_rows is not None:
                m_np = m.detach().numpy()
                diag = np.diag(A_inv)
                sd = 100.0 * np.sqrt(s2 * diag)
                # RAPM = α_off − δ_def, so its variance needs the off/def covariance with a minus sign.
                ar = np.arange(n_players)
                cov_od = A_inv[ar, n_players + ar]
                sd_rapm = 100.0 * np.sqrt(np.maximum(s2 * (diag[:n_players] + diag[n_players:2 * n_players] - 2 * cov_od), 0.0))
                for pid, idx in zip(ctx.vocab["player_id"].astype(int).values, ctx.vocab["player_idx"].astype(int).values):
                    if state.n_off[idx] < cfg.min_poss and state.n_def[idx] < cfg.min_poss:
                        continue
                    cache_rows.append({
                        "date": t, "player_id": pid, "player_name": ctx.name_lookup.get(pid, "Unknown"),
                        "orapm": float(beta_np[idx] * 100), "drapm": float(-beta_np[n_players + idx] * 100),
                        "rapm": float((beta_np[idx] - beta_np[n_players + idx]) * 100),
                        "sd_orapm": float(sd[idx]), "sd_drapm": float(sd[n_players + idx]),
                        "sd_rapm": float(sd_rapm[idx]),
                        "n_off": int(state.n_off[idx]), "n_def": int(state.n_def[idx]),
                        "prior_orapm": float(m_np[idx] * 100), "prior_drapm": float(-m_np[n_players + idx] * 100),
                        "prec_off": float(lam_np[idx]), "prec_def": float(lam_np[n_players + idx]),
                    })
    if opt is not None and n_batch:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad()
    return ({s: (v[0] / v[1] if v[1] > 0 else float("nan")) for s, v in sums.items()},
            beta_np, mu, state.current_date)


# ---------------------------------------------------------------------------
# Train / evaluate / re-score
# ---------------------------------------------------------------------------

def train(ctx: Context, model: V2Model, out_dir: Path, verbose: bool = True) -> tuple[int, list[dict]]:
    spec = ctx.spec
    groups = [{"params": model.prior.parameters(), "lr": spec.lr}]
    if spec.learn_alpha:
        groups.append({"params": [model.log_alpha_off, model.log_alpha_def], "lr": spec.lr_alpha})
    opt = torch.optim.Adam(groups)
    best_val, best_state, best_epoch, log = float("inf"), None, -1, []
    for epoch in range(spec.epochs):
        te = time.time()
        wmse, _, _, _ = run_epoch(ctx, model, opt)
        a_off, a_def = model.alphas
        log.append({"epoch": epoch, "train": wmse["train"], "val": wmse["val"],
                    "alpha_off": a_off, "alpha_def": a_def, "seconds": time.time() - te})
        if verbose:
            print(f"  epoch {epoch:2d}  train {wmse['train']:.5f}  val {wmse['val']:.5f}  "
                  f"α {a_off:.0f}/{a_def:.0f}  ({time.time() - te:.0f}s)", flush=True)
        if wmse["val"] < best_val - 1e-7:
            best_val, best_epoch = wmse["val"], epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "spec": asdict(spec), "base": asdict(ctx.base_cfg),
                        "features": FEATURES + LAG_FEATURES, "epoch": epoch}, out_dir / "model.pt")
        elif epoch - best_epoch >= spec.patience:
            if verbose:
                print(f"  no validation improvement for {spec.patience} epochs — stopping", flush=True)
            break
    model.load_state_dict(best_state)
    return best_epoch, log


def evaluate(ctx: Context, model: V2Model, out_dir: Path, forward_dir: Path, verbose: bool = True) -> dict:
    """Score the model on every origin (forward MSE, predictive NLL, margins) and write the cache."""
    spec = ctx.spec
    a_off, a_def = model.alphas
    scorer = WindowScorer(spec.name, [(a_off, a_def)])
    cache_rows: list[dict] = []
    wmse, beta_np, mu, final_date = run_epoch(ctx, model, None, scorer=scorer, cache_rows=cache_rows)
    poss_df, margins, origins_df = scorer.frames()
    forward_dir.mkdir(parents=True, exist_ok=True)
    poss_df.to_parquet(forward_dir / f"{spec.name}_poss.parquet", index=False)
    margins.to_parquet(forward_dir / f"{spec.name}_margins.parquet", index=False)
    origins_df.to_parquet(forward_dir / f"{spec.name}_origins.parquet", index=False)
    pd.DataFrame(cache_rows).to_parquet(out_dir / "rolling_cache_v2.parquet", index=False)
    summary = {"name": spec.name, "base": ctx.base_cfg.name, "objective": spec.objective, "wmse": wmse,
               "alpha_off": a_off, "alpha_def": a_def, "home_court": float(beta_np[-1] * 100),
               "comp_sigma": ctx.comp_sigma, "n_origins": len(ctx.origins), "final_date": str(final_date.date())}
    if verbose:
        print(f"[{spec.name}] eval wmse {{train {wmse['train']:.5f}, val {wmse['val']:.5f}, holdout "
              f"{wmse['holdout']:.5f}}}; α {a_off:.0f}/{a_def:.0f}; home ±{beta_np[-1] * 100:.2f}/100", flush=True)
    return summary


def fit_v2(spec: FitSpec, base_cfg: RapmConfig, data_dir: Path = Path("data"),
           out_dir: Path = Path("outputs/v2"), forward_dir: Path = Path("outputs/forward"),
           verbose: bool = True) -> dict:
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = prepare(spec, base_cfg, data_dir, verbose=verbose)
    model = new_model(ctx)
    best_epoch, log = train(ctx, model, out_dir, verbose)
    summary = evaluate(ctx, model, out_dir, forward_dir, verbose)
    summary.update({"best_epoch": best_epoch, "epochs": log, "seconds": time.time() - t0})
    (out_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def score_saved(model_path: Path, data_dir: Path = Path("data"), forward_dir: Path = Path("outputs/forward"),
                name: str | None = None, verbose: bool = True) -> dict:
    """Re-run the evaluation pass of a saved model (e.g. to add the predictive density)."""
    ckpt = torch.load(model_path, weights_only=False)
    spec = FitSpec(**ckpt["spec"])
    if name:
        spec.name = name
    base_cfg = RapmConfig(**ckpt["base"])
    ctx = prepare(spec, base_cfg, data_dir, verbose=verbose)
    model = new_model(ctx)
    model.load_state_dict(ckpt["state_dict"])
    out_dir = Path(model_path).parent
    summary = evaluate(ctx, model, out_dir, forward_dir, verbose)
    summary["best_epoch"] = ckpt.get("epoch")
    (out_dir / "fit_summary_rescored.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
