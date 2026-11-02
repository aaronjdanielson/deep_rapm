# RAPM v2 — learning the model from forward prediction

Status: design, 2026-10-25. v1 is frozen (`deep_rapm/production.py::V1`) and
is the reference every candidate is scored against.

## Decisions taken

1. **Score on future possessions, weighted for non-garbage time.** A model's
   score is its weighted squared error on the possessions it had not seen,
   with weight `exp(−(score diff / 13.3)²)` — a 23-point margin counts 5%.
   The weight is a constant of the harness, not a tunable, so every candidate
   is scored on the same test set. Rationale: we want the estimates to be
   right where there is signal; garbage-time possessions are noise whatever
   the model.
2. **v1 stays in production as is.** The forward harness
   (`deep_rapm/forward.py`, `scripts/score_forward.py`) puts a number on it
   and on alternatives; nothing replaces it until something beats it on the
   held-out seasons.
3. **Hyperparameters become learnable in PyTorch, interpretability kept.**
   The player coefficients stay the exact ridge (MAP) solution of the same
   weighted normal equations; what gets learned by gradient is the layer
   above them — the weights, the penalties and the prior.
4. **Priors from a snapshot of career-level information** — games and minutes
   played, position, age, size, draft slot, shooting percentages — evaluated
   *as of the origin date*, never from the future.

## 1. Objective

For origins `t` every 7 days (the production cadence), fit on possessions
dated ≤ t and predict the window `(t, t + 7]`:

```
L(θ) = Σ_t Σ_{i ∈ window(t)} w_eval,i · (y_i − ŷ_i(t; θ))²   /   Σ w_eval,i
```

reported as weighted MSE, as R² against the intercept-only prediction, and as
the RMSE of the competitive-time game margin. Tune on origins in 2015-16 …
2024-25; 2025-26 and 2026-27 are held out and looked at once per candidate.
Paired per-origin differences against v1 give a standard error
(`paired_table`); a candidate has to clear it on the holdout to be promoted.

## 2. The model, and what is learnable in it

For possession `i` scored from the state at origin `t`:

```
ŷ_i = μ_t + h·s_i + Σ_{p ∈ offense} α_p,t + Σ_{p ∈ defense} δ_p,t
```

`s_i = ±1` for home/away offense (v1 has no home term; `v1_home` adds it).
The coefficients `β_t = (α, δ, h)` are the ridge solution

```
β_t = (G_t + Λ)⁻¹ (b_t + Λ m)
G_t = Σ_{i ≤ t} w_i(θ) x_i x_iᵀ,   b_t = Σ_{i ≤ t} w_i(θ) x_i (y_i − μ_t)
w_i(θ) = 0.5^{(t − d_i)/H} · ρ^{summers crossed} · exp(−(diff_i/σ)²) · c_kind(i)
```

Everything inside `w_i`, plus `Λ` and `m`, is a hyperparameter:

| Group | Parameters | How it enters | v1 value |
|---|---|---|---|
| Recency | half-life `H`, off-season discount `ρ` | scalar decay of `G`, `b` per step (`advance_to`) | 365 d, 1.0 |
| Competition | `σ` (and optionally a clock term) | per-possession weight at ingest | p95 → 0.05, σ ≈ 13.3 |
| Context | `c_preseason`, `c_playoffs` | per-possession weight at ingest | 0.20, 1.0 |
| Penalty | `λ_off`, `λ_def`, `λ_home` | diagonal of `Λ` at solve time | 2000, 2000, — |
| Prior mean | `m_p = f_φ(x_p,t)` | shrinkage target at solve time | 0 |
| Prior precision | `Λ_p = λ · exp(g_ψ(x_p,t))` | per-player diagonal of `Λ` | constant |
| Intercept | its own half-life (scoring environment drifts) | separate decay of `total_w`, `Σwy` | shared `H` |

Note what is *not* expressible: a separate half-life per side. The decay is a
property of a possession, not of a column; `G` couples offense and defense
through its off-diagonal blocks, so "decay offense faster" has no weight
interpretation. One `H`, plus the summer discount.

## 3. Why bilevel rather than SGD on the coefficients

Training `β` directly by SGD (treating the 4,324 coefficients as network
weights) is simple but gives up the thing that makes RAPM readable: the
coefficients would no longer be the posterior mode of a stated model, they
would depend on the optimizer's path, and the leaderboard would change
meaning. Keeping the inner problem closed-form and learning only `θ` keeps
every number interpretable, and the gradient of `L` with respect to `θ` is
cheap through the solve:

- **Prior parameters (`φ`, `ψ`, `λ`) — one adjoint solve per origin.**
  `G_t` does not depend on them. With `r = w_eval ⊙ (ŷ − y)` on the window,
  `v = (G_t + Λ)⁻¹ X_windowᵀ r` gives `∂L/∂m = Λ v` and
  `∂L/∂Λ_p = v_p (m_p − β_p)`, and the rest is ordinary backprop into
  `f_φ`, `g_ψ`. Cost per origin: the forward solve plus one adjoint solve
  (≈1 s each on a 4,324-dimensional SPD system in float64 on CPU; float32 on
  MPS is fine at this conditioning). An epoch over ~575 origins is ~20 min;
  mini-batching origins makes steps cheaper.
- **Weight parameters (`H`, `ρ`, `σ`, `c`) — one extra dense accumulator each.**
  `∂G_t/∂θ` follows the same recursion as `G_t`
  (`∂G_{t} = γ ∂G_{t−1} + (∂γ) G_{t−1} + ∂G_new`), so each weight
  hyperparameter costs one more 4,324² float64 matrix (150 MB) carried
  through the pass, and `∂L/∂θ = −vᵀ(∂G_t β − ∂b_t)` reuses the adjoint
  `v`. A full gradient pass is therefore ~2 solves per origin plus a few
  matrix-vector products: 10–15 min on the 14 seasons.

There are only ~6 weight hyperparameters, so for *them* the numpy harness
with Bayesian optimisation or coordinate search (one 15–20 min evaluation
per point) is as fast as gradients and far simpler. The PyTorch machinery
earns its keep on the prior, where the parameters are many.

## 4. The prior

`m_p,t` is what we expect a player's impact to be before seeing lineup data
at origin `t`; `Λ_p,t` is how sure we are. Both are functions of features
known at `t`:

| Feature | Source | Status |
|---|---|---|
| Seasons of experience, games, minutes to date | `PlayerGameLogs` per season (one call per season, all players) | to collect |
| Age at origin, draft year and pick | `CommonPlayerInfo` (`BIRTHDATE`, `DRAFT_YEAR`, `DRAFT_NUMBER`) — ~2,200 calls, ~1 h paced | to collect |
| Position | `data/players.parquet` | have |
| Height, weight | `data/player_anthropometrics.parquet` (343 of 2,161 missing, mostly non-NBA players) | have |
| Career-to-date box rates: per-36 pts/reb/ast/stl/blk/tov, TS%, 3PA rate, FT%, usage | cumulated from the game logs | to collect |
| The model's own estimate 365 days earlier ("self-prior") | rolling cache | have |

Leakage rule: every feature is a cumulative or snapshot value computed from
games dated < t. "Career" means career *to date*, never the career total as
of today.

Expected shape of the learned prior, as a sanity check once it is fit:
rookies well below zero, bigs with stronger defensive than offensive
priors, prior precision rising with minutes played, and the self-prior
dominating for established players.

## 5. Evaluation guardrails

- Every table carries v1 (2000/2000) as the reference row.
- The α grid × variants is ~50 candidates; a weighted-MSE difference below
  the paired standard error is noise. Use `paired_table`, not the raw
  ordering.
- Possession-level R² against the league mean is ~0.2–0.3% on competitive
  possessions — that is the nature of a 0/2/3 outcome, not a model failure.
  The margin RMSE (≈10.4 vs 11.5 baseline on 2015-16) is the number that
  moves visibly.
- Tuning for a 7-day horizon favours whatever predicts the near term; it is
  the right criterion for "who is good now" but it is a choice. Re-scoring
  the chosen configuration with a 90-day horizon tells us if it generalises.

## 0. Status: phase 1 closed 2026-10-26

v1 stays in production; v2.1-A is frozen as the research candidate. The
release `outputs/releases/rapm-phase1_2026-10-26/` (read-only, sha256
manifest) holds both models, the data snapshot, every evaluation, the
frozen calibration constants, the selection record and the code; the
frozen v2.1-A reproduces from its checkpoint to 4e-7. The full test suite
passes (487 passed, 3 expected failures for games the validator already
flags). `scripts/prospective_eval.py` scores both frozen models on games
after 2026-10-24 every Monday (cron, 06:30). Not to be revisited until the
prospective record says so: the network architecture, the prior cap, the
feature shrinkage κ, the empirical-Bayes objective.

Standing qualifications:
- Margin NLL is from the variance-scaled Gaussian (ĉ); the level-specific
  (conformal) intervals calibrate coverage at 50% and 90% only and are not
  a predictive density.
- Split-conformal coverage guarantees assume exchangeability; time-ordered
  NBA games need not satisfy it. The holdout coverage is practical
  evidence, and the prospective runs monitor it.
- Player-level posterior intervals are conditional on the likelihood, the
  prior and the weighting scheme; they do not include uncertainty in those
  choices, and teammates who share the floor can trade attributed credit.
- Three questions are kept apart: prediction uncertainty (game margins —
  calibrated), parameter uncertainty (player effects — conditional only)
  and ranking uncertainty (differences between players — conditional
  only).

## 0½. Replacement-level prior experiment (2026-11-01)

Question (user): the ridge's zero prior treats a player coaches rarely use
as average; should the prior for most players be negative, and does that
improve prediction? Model `deep_rapm/replacement_prior.py`
(`scripts/fit_replacement_prior.py`): v1 with prior mean a + b·log(1 +
minutes in the past 365 days) [+ c·minutes per game] per side, precision
fixed at 2000, θ by exact weighted least squares on the forward objective
(predictions are linear in θ), same snapshot (≤ 2026-10-24), origins,
weights and splits as phase 1.

- Out-of-sample motivation (v2.1-A analysis, 108 val + holdout weeks): v1
  over-predicts by 1.80 ± 0.25 per 100 for each low-minute (< 1,000
  offensive possessions in the past year) player on offense and
  under-predicts by 0.51 ± 0.26 for each on defense; v2.1-A cuts these to
  0.87 and ~0. Relative to the possession-weighted average player, 70% of
  active players get a negative learned prior.
- Fitted playing-time prior (relative to the possession-weighted average):
  ORAPM −6.0 at 0 minutes, −1.8 at 100, −0.4 at 500, +0.3 at 1,000, +1.3
  at 3,000; DRAPM flat (slope not significant). Minutes per game adds
  offensive information (t ≈ 7).
- Prediction (Δ wMSE vs v1, month-blocked t): train −0.000104 (−2.6) /
  −0.000166 (−3.1) for minutes / minutes + mpg; **validation +0.000011
  (+0.1) / −0.000058 (−0.7)**; holdout +0.000008 / −0.000050. Over v1 +
  home the prior adds the same (validation −0.000059, t −0.6). By season
  phase, out of sample: helps in Oct–Nov (−0.00028, t −2.9) and Dec–Jan
  (−0.00025, t −3.8), hurts in Feb–Jun (+0.00012 to +0.00018, t +1 to +1.9);
  cause of the late-season cost not isolated.
- Against v2.1-A: v2.1-A beats the best playing-time prior (v1-R-mpg+home)
  by 0.00017 on validation (t −1.3) and 0.00029 on the holdout (t −1.6).
- Verdict under the selection rule: no playing-time variant beats the
  simpler v1 + home on validation, so none becomes a candidate or joins the
  prospective evaluation. The idea is right in direction — the low-minute
  bias is real and the prior helps when current-season data are thin — but
  playing time alone is not enough; the feature prior (career box rates,
  draft slot, age, history), which already learns negative priors for
  low-minute players, carries the predictive gain.
- Caveat: the θ = 0 check reproduces the stored v1 scores to 3e-5 per
  season×kind cell rather than 1e-7, because the vocab gained 33 players
  (2018-19 fix) after the freeze; this is well below the differences above.
  Ratings in `outputs/replacement_prior/ratings_<variant>.parquet` are
  levelled so the possession-weighted average player is 0.

## 0¾. On-court share statistics and density priors (2026-11-01, in progress)

**Share statistics** (user: per-36 rates miss how much of the team's work a
player does on the floor). `deep_rapm/priors.py` feature set `"v2"` adds
usage, assist, turnover, offensive and defensive rebound, steal and block
percentages (team and opponent game totals scaled by the player's share of
the minutes, Basketball-Reference estimates; career to date plus usage over
the past year; same κ shrinkage). Paired refit of v2.1-A's configuration on
the same snapshot and vocab (`v2.1-A-r` vs `v2.2-A`), Δ wMSE (month-blocked
t): train −0.000083 (−3.3), validation −0.000055 (−1.4), holdout −0.000075
(−0.8) — consistent in direction, ~10–20% more gain over v1, not
established on validation alone. Both fits peak at epoch 1–2: the weekly
forward objective is a weak, noisy training signal and stops the prior
network close to its starting point, which limits what any richer prior can
learn from it.

**Density priors** (user: neural density estimation of the joint (ORAPM,
DRAPM); MADE / MAF). `deep_rapm/density_prior.py`, `scripts/fit_density.py`:
- `family="mixture"`: K bivariate Gaussians per player with full 2×2
  precisions (learned O/D correlation); posterior mode by EM — each M-step
  is the ridge solve with a penalty coupling a player's offense and
  defense coefficients.
- `family="maf"`: conditional MAF — a feature encoder gives a context
  vector; each block is a d = 2 MADE ((μ₁, α₁) from the context, (μ₂, α₂)
  from the context and u₁), coordinate order swapped between blocks; z is
  scaled by s0 = √(s²/λ) so identity blocks give exactly v1's N(0, s0²I)
  prior. Posterior mode by Newton steps using each player's gradient and
  2×2 Hessian of −log q (Levenberg–Marquardt floor on the curvature; an
  eigen-decomposition has no gradient at the isotropic start). Checks:
  identity flow = ridge; Newton = BFGS on a perturbed flow to 2e-8.
- Both are trained end to end on the forward objective; there are no
  observed player effects to fit a density to by maximum likelihood.
- Running: maf4 (4 blocks), d1-joint (one joint Gaussian), d3-mix (3
  components), all with v2 features (logs `outputs/logs/density/`).

**Deconvolution training** (`deep_rapm/deconvolution.py`,
`scripts/fit_deconvolution.py`). The weekly forward objective is too weak
to train a density; instead the prior q(z | x) is fit to later evidence
about each player, with the evidence's own noise integrated out:
- `--targets season`: for each season, the season's ridge with the
  reference prior; removing a player's own prior from his posterior gives
  his exact marginal likelihood (2×2 precision F, natural parameter h) with
  teammates integrated out under the reference prior. One row per
  player-season (~600/season).
- `--targets game` (user's suggestion — per-game offensive and defensive
  measurements on the RAPM scale): at each weekly origin the reference fit
  gives β̂; for each possession in the next window the residual plus the
  player's own estimate is a one-possession measurement of his effect with
  everyone else held at their estimates; summed over his games in the
  window it is the product of the per-game likelihoods (F = Σ w, h = Σ w(r +
  β̂_i)), in points per possession — the RAPM scale by construction. Out of
  sample for the reference fit; teammates not integrated out. ~400 rows per
  week.
- Loss: Σ log ∫ e^{ℓ_i(z)} q(z | x_i) dz — closed form for Gaussian mixtures,
  importance-sampled (reference-posterior proposal) for the MAF. Features
  at the window's / season's first origin. Models: ref (v1's prior), mean
  (learned mean only), joint (full 2×2), mix3, maf4. Selection by held-out
  log-likelihood on the validation seasons; the chosen density then plugs
  into the weekly RAPM solve and is scored by the forward harness against
  v2.1-A, which stays the deciding test.

Season-target results (held-out log-likelihood per player-season, relative
to a q-free constant; train < 2023-07, val 2023-24 … 2024-25, holdout
2025-26 →; v2 features, κ = 500):

| prior | train | val | holdout |
|---|---|---|---|
| ref — v1's N(0, s²/λ I) | −0.024 | −0.009 | +0.000 |
| mean — learned mean, fixed cov (v2.1-A's form) | +0.111 | +0.099 | +0.083 |
| joint — learned mean and full 2×2 cov | +0.234 | +0.172 | +0.138 |
| mix3 — three joint Gaussians | +0.199 | +0.173 | +0.126 |

Learning where each player's prior sits is worth ~0.11 nats per
player-season over v1; letting the prior's spread and offense/defense
correlation depend on the player doubles that; a mixture adds nothing over
one joint Gaussian on validation. Trains in minutes (≈ 7,000 player-seasons)
against ~30 min per epoch for the forward objective.

**Superseded (feature-timing leak).** The table above was computed with the
feature origin chosen as the first origin *at or after* the season's first
game, so for most seasons the features included up to a week of the target
season's games (material for rookies: `no_history`, first rates). Fixed
2026-11-01: the feature origin is the last origin at or before the first
target game, so every feature date is strictly earlier than every target
game. The run was stopped and relaunched; its outputs are in
`outputs/deconv_leaky_superseded/`. Numbers to be replaced.

**Corrections from review (2026-11-01), all applied:**
- *Mixture covariance.* The Laplace covariance must use the true Hessian of
  −s² log q, which is Σ_k r_k P_k − Cov_r[v_k]/s² with v_k = P_k(z − m_k);
  the EM surrogate keeps only the first term (Louis's missing-information
  correction) and overstates precision. `covariance_system` now evaluates
  the true Hessian at the mode (surrogate fallback for a block that is not
  positive definite, counted in `diagnostics`). Checked against finite
  differences; for K = 1 the two coincide and the solve is a single coupled
  ridge, no EM.
- *EM convergence.* Every 25th origin the solve is continued for 7–10 more
  iterations and the change in β and in the objective recorded
  (`diagnostics.conv_checks`); the synthetic check (3 iterations ≈ BFGS to
  1e-8) is evidence for that case only.
- *Flow scale.* The MAF's base scale is now s0 = √(s_t²/λ) at each origin
  (was a fixed reference s²), so its identity initialization equals v1's
  prior everywhere, not at one noise level.
- *Objective, correctly described.* The season-target objective is a
  **reference-dependent composite likelihood**: each player's likelihood
  integrates the others out under the *reference* prior and overlapping
  games enter several players' terms. It is not the joint evidence
  ∫ p(D | z) Π q(z_i | x_i) dz. Likewise the game-level objective treats
  each player-week as an independent latent draw, which is a different
  model from one shared effect per season (integrating after multiplying ≠
  multiplying after integrating), so the +0.006 vs +0.10 comparison is not
  just a noise-ratio story. Both objectives are *methods for learning a
  prior*; their worth is settled only by the matched forward test.
- *Vocabulary.* The 3e-5 zero-prior discrepancy came from the vocab growing
  by 33 players after the freeze; v1 and v1 + home are re-scored on the
  current vocab (`outputs/forward/`, old files in `outputs/forward_oldvocab/`)
  so every row of the comparison shares it. The frozen v2.1-A files stay on
  the old vocab; its matched refit `v2.1-A-r` is the in-table reference.

**Results so far on the matched forward test (validation, Δ wMSE vs v1,
month-blocked t; all on the 2026-10-24 snapshot):**

| model | Δ vs v1 | t | Δ vs v2.1-A-r | t |
|---|---|---|---|---|
| v2.1-A-r (learned mean, forward objective, matched refit) | −0.000583 | −3.2 | 0 | — |
| v2.2-A (+ share statistics) | −0.000637 | −3.1 | −0.000055 | −1.4 |
| d1-joint (joint Gaussian, forward objective) | −0.000592 | −2.7 | −0.000009 | −0.1 |
| d3-mix (3-component mixture, forward objective) | −0.000700 | −3.1 | −0.000117 | −1.3 |
| maf4 (conditional MAF, forward objective) | −0.000659 | −3.6 | −0.000076 | −0.7 |
| dcg-mean (game-target deconvolution, mean only) | −0.000375 | −3.7 | +0.000208 | +1.6 |
| dc-mean (season-target deconvolution, mean only, clean cutoff) | −0.000568 | −7.3 | +0.000014 | +0.1 |
| dcg-joint / dcg-mix3 (game-target deconvolution) | +0.0073 / +0.0064 | +9.3 | — | — |

Trained on the forward objective, the joint, mixture and flow priors are all
within noise of the learned-mean model (and cost 3–4 h each against 30
min). The season-deconvolved *mean* with v1's covariance (dc-mean, clean
cutoff, held-out log-likelihood +0.097 on validation) plugs in at parity
with the forward-trained mean (Δ +0.000014, t +0.1) and beats v1 more
consistently across months (t −7.3 vs −3.2); it trains in minutes. So
deconvolution on season targets is a competitive way to learn the prior
mean; what it must not be used for, as implemented, is the covariance. Trained by deconvolution on weekly targets, any model with a learned
covariance is far worse than v1 as a plug-in: its prior spread collapsed to
≈ 0.3 per 100 (v1: 2.66), so the prior overrides the data. Diagnosis: with
a single global mean removed, the season targets imply a prior spread of
≈ 2.0 (offense) / 1.7 (defense) per 100 with information-scale κ ≈ 0.93 /
0.8 — not collapsed — whereas the weekly targets (no row with F > 1,000)
give τ → 0 whatever κ under this fitted working model (no row has F >
1,000) — a problem with this objective on weekly rows, not a proof that
the spread is unidentifiable from many noisy measurements (finite-sample
behaviour, mean fitting, dependence between a player's weeks and the
measurement-variance model can each produce a zero estimate; the
diagnostics do not single one out). The season-target joint model
collapsed too (0.26 per 100). Exploratory residual check on well-measured
players (F > 1,500), using the corrected targets but checkpoints trained
in the *leaky* run: in-sample, var(y − m) is below the modeled noise
(excess −0.7 to −1.4 per 100²) for both the fixed-covariance and the joint
model — consistent with mean overfitting (train R² 0.28–0.32 vs validation
0.12–0.20) and/or noise overstated on training rows; on validation the
excess is +0.9 to +1.2, i.e. a conditional spread of roughly 1 per 100 on
each side. That number is indicative only (checkpoints from the superseded
run). A smaller feature-conditional spread than the ≈ 2.0 unconditional
one is no contradiction — Var(Z) = Var(E[Z|F]) + E[Var(Z|F)] — and the
open question is whether a fitted conditional uncertainty is appropriate
for the weekly update, which is asymmetric: too confident is catastrophic,
too diffuse is mild. Rule adopted: a covariance scale used in the weekly
solve is set on forward validation, not by this likelihood.

## 0⅞. State-space view (corrected 2026-11-01)

What the weekly solve *is*: an information-form Gaussian measurement update,
Ω ← Ω + X'WX/s², ξ ← ξ + X'W(y − μ)/s², with posterior covariance
s²(G + Λ̃)⁻¹; the decay step discounts the data information while the prior
information stays fixed, Ω⁻_t = γ Ω_{t−1} + (1 − γ) Ω₀ (and likewise ξ). That
is a coherent discounted-likelihood (power-prior) recursion. It is **not**
equivalent to a fixed AR(1)/OU transition toward the prior mean, and an
earlier note claiming near-equivalence with φ² = γ was wrong: with previous
posterior variance C = 0.01, stationary variance V = 1 and γ = φ² = 0.5, the
discount rule predicts variance (γ/C + (1 − γ)/V)⁻¹ ≈ 0.020 while AR(1) gives
φ²C + (1 − φ²)V = 0.505 — process noise keeps injecting uncertainty that
information discounting does not. The two are related only through
adaptive, covariance-dependent process-noise models.

Three things that must stay distinct: the learned prior describes plausible
effects given features; a transition model describes how effects change
over time; the possession likelihood describes what games reveal. The
explicit dynamic model is the next experiment:

    β_t = β̄_t + A (β_{t−1} − β̄_{t−1}) + η_t,   η_t ~ N(0, Q)
    predict:  a⁻_t = β̄_t + A(a_{t−1} − β̄_{t−1}),   C⁻_t = A C_{t−1} Aᵀ + Q
    update:   C_t = [ (C⁻_t)⁻¹ + X_tᵀW_tX_t / s_t² ]⁻¹,
              a_t = C_t [ (C⁻_t)⁻¹ a⁻_t + X_tᵀW_t(y_t − μ_t) / s_t² ]

with β̄_t the feature-based expected ability computed before the window,
A and Q kept simple at first. The update must add **only the new week's**
possessions — propagating last week's posterior and adding the cumulative
G_t again would count old observations twice. The across-player spread of
the learned prior is not the within-player process noise; Q has to be
estimated (the season-to-season targets are the natural data). A nonlinear
feature network for β̄_t needs no nonlinear filtering while transition and
observation stay linear in the state.

**Status (2026-11-01): implemented and validated, comparison running.**
`deep_rapm/state_space.py` (`scripts/fit_state_space.py`,
`tests/test_state_space.py`). Interval bookkeeping as reviewed: assimilate
(t_{k−1}, t_k] into the predicted state; compute m_k from the snapshot at
t_k; forecast a_{k+1|k} = m_k + D(a_k − m_k), C_{k+1|k} = D C_k Dᵀ + Q
(D = diag(φ on players, 1 on home), Q = diag((1 − φ²) s²/λ on players, 0 on
home); summer = ``summer_steps`` transitions); score (t_k, t_{k+1}] with the
forecast; that forecast is the next prior. Only the new interval's
possessions enter; μ_k and s_k² come from v1's own accumulators so the
comparison isolates the transition (τ² = s_k²/λ is therefore adaptive, as
v1's prior variance is, not one fixed stationary variance); warm-up runs
chronologically from the first game with zero mean, the learned mean
switching on at the first harness origin; the lag history carries decayed
exposure (what the frozen network was trained on). The first draft's
defects named in review — player–home covariance not scaled by φ, scoring
with the filtered rather than the forecast state, features from the end of
the assimilated interval, zero initial mean, single-batch warm-up,
posterior-precision in the lag history — are all corrected. Checks passed:
static limit (φ = 1, Q = 0, fixed μ, s²) equals the batch ridge to 1e-10;
means, covariances and forecasts equal a direct covariance-form Kalman
filter on a synthetic system (non-zero player–home covariance, positive
definite) to 1e-9; the forecast for a window is unchanged when all later
possessions are removed; the forecast scored for (t_k, t_{k+1}] is
bit-identical to the prior state assimilated at t_{k+1}; a prior
checkpoint is refused unless the context was built with its exact feature
pipeline (`fit_state_space.py` reads the checkpoint's spec before
`prepare`, asserts feature names and input width). The second review round
(of the second draft) listed forecast timing, pre-window features and
checkpoint compatibility as open: the first two were already corrected in
the third draft; the third was real — the learned-mean lane launched
against a v1-feature checkpoint with v2-feature arrays and failed on load —
and is now fixed. Two review points that remain modeling choices, stated
rather than hidden: the warm-up is zero-mean with the learned mean
switching on at the first harness origin (features exist only there), and
τ² = s_k²/λ is adaptive like v1's prior variance rather than a fixed
stationary law. Running: zero-mean SSM, φ ∈ {0.98, 0.99, 0.995, 1}
(against v1_home — the matched discounted baseline with home court), and
learned-mean SSM, φ ∈ {0.99, 1} with v2.1-A-r's mean (against v2.1-A-r);
selection on validation.

## 6. Steps

**2026-10-25: every result below steps 1–3 was computed on biased data and
is being recomputed.** `score_diff` in pbpstats-parsed games (2018-19 →
2024-25) was the margin after the possession's first event, which skewed
the competition weights of exactly the training and validation seasons.
The data is repaired (`scripts/fix_score_diff.py`, operator guide §1.4) and
the plan reruns in full with `scripts/run_v2_experiments.sh`. Also adopted
from review: holdout origins are neither scored nor logged during training;
`paired_table` adds a month-blocked standard error (weekly origins share
their training history); `sd_rapm` uses the off/def posterior covariance;
the evidence objective can take its noise variance from the base ridge
(`--evidence-noise reference`); κ and every other choice are made on the
validation seasons only.

**Selection, 2026-10-26 (recorded before the holdout was looked at).**
Rerun on the repaired data; train (origins < 2023-07) and validation
(2023-24, 2024-25) only, regular season + playoffs, Δ weighted MSE with
month-blocked t:

| Question (validation) | Δ wMSE | t_block |
|---|---|---|
| learned prior means vs v1 + home (v2.1-A) | −0.000231 | −2.9 |
| learned precisions vs v1 + home (v2.1-B) | −0.000183 | −1.8 |
| adding precisions to means (C − A) | −0.000051 | −0.5 |
| learning base penalties (v2.1 − C) | −0.000009 | −1.0 |
| κ = 500 vs raw features (v2.1 − v2) | +0.000091 | +1.4 |
| κ = 200 / 1000 vs 500 | ≈ 0 | −0.6 / 0.0 |
| evidence vs forward objective (v2.1eb − v2.1) | +0.000418 | +3.2 |
| reference vs model noise in the evidence | ≈ 0 | +0.2 |
| home court alone (v1_home − v1) | −0.000350 | −2.2 |

Rule: the simplest candidate within noise of the best on validation.
Chosen: **v2.1-A** — v1 + home-court column + learned prior mean from
as-of-origin features (κ = 500), prior precision and penalties fixed at
v1's 2000. v2.1-A does *not* have the lowest validation error (v2 does:
1.41558 vs 1.41570); it is the simplest model whose validation error is
within noise of the best. Means carry the gain; learned precisions are
weaker alone and their incremental value over means is not demonstrated
(which is not the same as shown to be zero); learned penalties add
nothing; the evidence objective is worse than forward training and the
noise correction does not change it — under this Gaussian
pseudo-likelihood, which does not rule out empirical Bayes with a
better-specified likelihood. Frozen as the research candidate in
`outputs/candidates/v2.1-A/` (manifest with spec and hashes); v1 stays in
production.

Holdout (looked at once, after the choice above): v2.1-A vs v1 Δ wMSE
−0.00034 (per-origin t −2.3, month-blocked t −1.9), vs v1 + home blocked
t −3.4; R² 0.208% → 0.243%, margin RMSE 11.345 → 11.275. Two limits on
reading it: the blocked t rests on ten monthly blocks of one season, and
2025-26 results were seen during development before the data repair, so
it is not a fully untouched test. The next clean test is 2026-27 after
this point.

**Density, calibration and star checks (2026-10-26, all models on the
same data snapshot through 2026-10-24).**

| Holdout (2025-26 →) | v1 | v1 + home | v2.1-A |
|---|---|---|---|
| weighted MSE | 1.41703 | 1.41693 | 1.41653 |
| predictive NLL per possession | 1.59322 | 1.59319 | 1.59305 |
| margin RMSE | 11.345 | 11.303 | 11.275 |
| 50% / 90% margin-interval coverage | 41.1% / 84.6% | 42.2% / 84.7% | 43.5% / 85.4% |
| SD of margin z-scores (1 = calibrated) | 1.132 | 1.133 | 1.119 |

- NLL: v2.1-A vs v1 −0.000119 (blocked t −1.8), vs v1 + home −0.000118
  (blocked t −3.4); validation −0.000206 (−3.2). At possession level the
  predictive variance is almost all irreducible noise and nearly equal
  across models, so NLL is close to a rescaled MSE and adds little
  independent evidence; the calibration information is in the margins.
- Margin intervals under-cover for every model (z-score SD 1.09 on
  validation, 1.12–1.13 on holdout). Plausible explanations — variation
  shared across a game's possessions that the possession-level model
  treats as independent, such as shooting luck or effort — have not been
  isolated. v2.1-A is marginally the best calibrated.
- Calibration (`scripts/calibrate_margins.py`; fitted on validation games,
  carried unchanged to the holdout). One variance multiplier per model,
  ĉ = mean z² on validation: v1 1.22, v1 + home 1.20, v2.1-A 1.18 (the
  smallest), v2.1eb 1.24. On the holdout it fixes the 90% level (v2.1-A
  89.7%, v1 89.6%) but not the 50% level (46.1% / 45.3%), and the same
  happens in-sample on validation: the standardized residuals are
  lighter-tailed than normal (excess kurtosis −0.36 to −0.5; median |z|
  0.74 against 0.67, 99th percentile 2.3–2.5 against 2.58), so a single
  scale cannot match both levels. One possible contributor, not
  established: the competition weighting limits how large a weighted
  margin can grow. Level-specific (split-conformal) half-widths, k_q =
  validation quantile of |z|, give holdout coverage 50.3% / 88.9% for
  v2.1-A and for v1; relative to the uncalibrated normal interval the 50%
  half-width needs ×1.19 and the 90% half-width ×1.07. After the
  multiplier, margin NLL on the holdout is 4.009 for v2.1-A against 4.027
  for v1. Playoff games stay under-covered (90% ≈ 86%, z SD ≈ 1.07), on
  91 games only. v1's margins are biased toward the road team (mean z
  +0.11 to +0.14) because it has no home term; v1 + home removes it.
- What the margin numbers mean: μ_g sums the model's predictions over the
  possessions and lineups that actually occurred, so they evaluate the
  RAPM model conditional on realized participation. They are not
  pre-tipoff forecasts of the margin, which would also have to predict
  who plays and for how long.
- Game-margin calibration says nothing about the calibration of individual
  player effects; that is a separate, still open problem.
- Star calibration (`scripts/analyze_v2.py`, 108 out-of-sample origins):
  on possessions with one of v2.1-A's top-10 players on offense, realized
  minus predicted is +1.53 ± 0.38 per 100 for v1 (v1 under-predicts its
  stars) and +0.59 ± 0.38 for v2.1-A — no sign of exaggeration; if
  anything still slightly under. On defense neither model is
  significantly off (+0.26 / +0.38 ± 0.38). Two or more top-10 players:
  all within ±1.2 standard errors.
- Prior-mean cap: within 10% of ±5 for < 1% of established players;
  on offense for 1.3–2.8 of the top 10 on average per season, never on
  defense. Since the stars are still slightly under-predicted, the cap may
  be mildly restrictive — a validation-only question.
- Ranking uncertainty at 2026-10-25 (posterior covariance, conditional on
  the prior network): P(#1) at most 26% (Wembanyama, v2.1-A) / 17%
  (Gilgeous-Alexander, v1); 90% rank intervals for the top players span
  ranks 1–13 to 1–17; P(ahead of the next player down) 0.50–0.64. Player
  effects are nearly uncorrelated except teammates (≈ −0.15). The order
  within the top 20 is not resolved by the data under either model.
- Not yet validated: the player-level posterior SDs themselves (the
  margin check mostly tests the noise term).

1. ~~Harness, v1 baseline, ridge-family grid~~ (done 2026-10-25; results in
   `outputs/forward/`). Findings: α = 2000 is the best grid point; home
   court is the only ridge-family change that helps (t = −5 on tune);
   shorter memory (half-life 180 d, off-season discount) hurts clearly,
   longer (730 d) is a wash.
2. ~~Prior data and the as-of-origin feature builder~~ (done;
   `scripts/fetch_prior_data.py`, `deep_rapm/priors.py`).
3. ~~`deep_rapm/torch_rapm.py`~~ (done; forward and evidence objectives).
   v2 (raw features, forward objective): holdout R² +12.9% relative to v1,
   t = −1.9; validation +21%, t = −4.4; gain concentrated early in the
   season. Known defect: rate features from a handful of minutes
   (Vincent Hunter) — fixed by `--shrink 500` in v2.1.
4. In progress: posterior predictive NLL and margin-interval coverage for
   v1, v2, v2eb (evidence objective, raw features), v2.1 and v2.1eb
   (shrunk features). The decisive comparison is v1 → v2.1 → v2.1eb on
   identical held-out possessions: weighted MSE, predictive NLL, coverage.
5. Weight hyperparameters by search on the numpy harness (the first grid
   says the v1 values are close to optimal); structured off/def prior
   covariance and heavy-tailed priors only if overshrinkage of
   high-impact players shows up.
6. Promote to v2 only on a holdout win beyond the paired standard error;
   keep v1 as the comparison in the README.
