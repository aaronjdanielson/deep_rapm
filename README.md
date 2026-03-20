# Deep RAPM

Regularized Adjusted Plus-Minus (RAPM) for NBA possession data, with both
an analytical ridge regression solver and a Set Transformer neural model.

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.1.

---

## Data pipeline

**Step 1 — collect possessions** (calls the NBA Stats API; takes ~10 min per season):

```bash
collect-possessions --season 2022-23 --output-dir data/2022-23
```

**Step 2 — build player vocab and position table**:

```bash
collect-players --seasons 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24
```

Produces `data/player_vocab.parquet` and `data/players.parquet`.

---

## Analytical RAPM

Fits ridge regression via the normal equations — exact, fast (~1 s), and
noise-immune.  This is the recommended starting point.

### CLI

**Season mode** (uses pre-collected parquets):

```bash
solve-rapm                          # default: 5 training seasons, alpha=2000
solve-rapm --alpha 1000 --top 20    # tune regularisation, show more players
solve-rapm --output-dir runs/rapm   # custom output directory
solve-rapm --half-life 365          # down-weight older games (1-year half-life)
```

**Date-range mode** (auto-fetches and caches games from the NBA API):

```bash
# Fit on a specific date window; games cached to data/games/<game_id>.parquet
solve-rapm --from-date 2024-10-01 --to-date 2025-04-15

# With recency weighting — games from 180 days ago count half as much
solve-rapm --from-date 2023-10-01 --to-date 2025-04-15 --half-life 180
```

Output: `checkpoints/rapm/rapm.parquet` and `rapm_summary.json`.

### Python API

```python
from pathlib import Path
from deep_rapm import fit_rapm, load_rapm

# Season mode — load from pre-collected parquets
results = fit_rapm(
    data_dir=Path("data"),
    seasons=["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
    player_vocab_path=Path("data/player_vocab.parquet"),
    player_table_path=Path("data/players.parquet"),
    alpha=2000,
    output_dir=Path("checkpoints/rapm"),
)

# Date-range mode — auto-fetch from NBA API, cache per game
results = fit_rapm(
    data_dir=Path("data"),
    from_date="2024-10-01",
    to_date="2025-04-15",
    player_vocab_path=Path("data/player_vocab.parquet"),
    player_table_path=Path("data/players.parquet"),
    alpha=2000,
    half_life_days=180,   # optional: down-weight older games
    output_dir=Path("checkpoints/rapm"),
)

# Load pre-computed results
results = load_rapm(Path("checkpoints/rapm"))

# Work with the DataFrame
qualified = results[results["qualified"]]   # min 100 poss each role
print(qualified.nlargest(10, "rapm")[["player_name", "orapm", "drapm", "rapm"]])
```

Result columns (all per 100 possessions):

| Column | Description |
|--------|-------------|
| `orapm` | Offensive RAPM — points added per 100 offensive possessions |
| `drapm` | Defensive RAPM — points prevented per 100 defensive possessions (positive = good defender) |
| `rapm`  | Total RAPM = `orapm + drapm` |
| `n_off` / `n_def` | Offensive / defensive possession counts |
| `qualified` | `True` if ≥ 100 possessions in each role |

### Model

Each possession $i$ is labelled by which players are on the court. Define the
indicator matrix $X \in \{0,1\}^{n \times 2p}$ where $p$ is the number of
players: the first $p$ columns are offense indicators and the last $p$ columns
are defense indicators. Each row has exactly 10 ones — one per player on the
court.

The predicted points scored on possession $i$ is

$$\hat{y}_i = \mu + \sum_{j \in \text{off}(i)} \alpha_j + \sum_{k \in \text{def}(i)} \delta_k = \mu + X_i \beta$$

where $\beta = [\alpha_1, \ldots, \alpha_p, \delta_1, \ldots, \delta_p]^\top$
collects the offensive and defensive parameters.

**Unweighted ridge.** Fit by minimising

$$\mathcal{L}(\beta) = \|y_c - X\beta\|^2 + \alpha \|\beta\|^2$$

where $y_c = y - \mu$ is mean-centred. Setting the gradient to zero gives the
normal equations

$$\bigl(X^\top X + \alpha I\bigr)\,\beta = X^\top y_c$$

which are solved exactly via Cholesky decomposition. The matrix $X^\top X$ is
$2p \times 2p$ (typically $3000 \times 3000$) and dense after forming, making
the direct solve fast and numerically stable.

**Recency-weighted ridge.** When a half-life $\tau$ (days) is specified, each
possession is down-weighted exponentially by its age:

$$w_i = 0.5^{\,d_i / \tau}$$

where $d_i$ is the number of days between possession $i$ and the most recent
possession in the dataset. The weighted objective becomes

$$\mathcal{L}_W(\beta) = \|W^{1/2}(y_c - X\beta)\|^2 + \alpha\|\beta\|^2$$

with $W = \operatorname{diag}(w)$. The weighted normal equations are

$$\bigl(X^\top W X + \alpha I\bigr)\,\beta = X^\top W y_c$$

$X^\top W X$ is computed efficiently as $(X \odot \sqrt{w})^\top (X \odot \sqrt{w})$, keeping $X$ sparse throughout.

**Intercept.** The intercept $\mu$ is the (weighted) mean points per
possession and is removed before solving, then added back at prediction time.
This decouples the mean from the ridge penalty.

**Reported values** (per 100 possessions):

$$\text{ORAPM}_j = 100 \cdot \alpha_j \qquad \text{DRAPM}_k = -100 \cdot \delta_k \qquad \text{RAPM} = \text{ORAPM} + \text{DRAPM}$$

The sign flip on DRAPM makes positive values mean *good defender* (a defender
who suppresses scoring has $\delta_k < 0$, so $\text{DRAPM}_k > 0$).

### Sample output (2018-19 through 2022-23, alpha=2000)

```
Player                  ORAPM   DRAPM    RAPM
Nikola Jokić            +7.74   +1.94   +9.68
Joel Embiid             +4.44   +4.56   +9.00
Stephen Curry           +6.09   +2.29   +8.38
Giannis Antetokounmpo   +4.33   +4.03   +8.35
LeBron James            +6.01   +2.03   +8.04
Alex Caruso             +0.96   +6.24   +7.20   ← elite defender
Rudy Gobert             +0.20   +6.28   +6.48   ← elite defender
Damian Lillard          +7.44   -0.52   +6.93   ← scorer, defensive liability
```

---

## Neural model (experimental)

Trains a Set Transformer on the possession data, warm-started from the
analytical RAPM solution.

```bash
# Fit analytical RAPM first (required for warm-start)
solve-rapm --output-dir checkpoints/rapm

# Train neural model warm-started from RAPM
train-deep-rapm --model linear \
                --rapm-dir checkpoints/rapm \
                --output-dir checkpoints/neural

# Train full Set Transformer (DeepRAPM)
train-deep-rapm --model deep \
                --rapm-dir checkpoints/rapm \
                --output-dir checkpoints/deep
```

Key hyperparameters:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `deep` | `deep` (Set Transformer) or `linear` (ridge analog) |
| `--d` | 64 | Embedding dimension |
| `--num-layers` | 2 | Transformer layers |
| `--epochs` | 30 | Training epochs |
| `--embedding-reg` | 1e-4 | L2 penalty on player embeddings |
| `--rapm-dir` | None | Warm-start from analytical RAPM |

---

## Design notes

The general RAPM prediction can be written as

$$\hat{y}_i = f_{\theta}(\mathbf{o}, \mathbf{x}, \mathbf{g})$$

where $\mathbf{o}$ are the indices of the 5 offensive players, $\mathbf{x}$ are the indices of the 5 defensive players, $\mathbf{g}$ is a gamestate vector, and $\theta$ parameterises $f$. The target $y$ is an outcome of interest — points scored, possession length, assist probability, etc.

The analytical model uses a design matrix with $2p$ columns ($p$ = number of players) so each player is represented by two scalars: one offensive, one defensive. This does not capture player-player interactions or lineup synergies.

The neural models replace the two scalars with two latent vectors $\mathbf{u}_i^o, \mathbf{u}_i^d \in \mathbb{R}^d$ per player, enabling richer lineup representations.

### What does NOT increase expressiveness

A natural first idea is to give each player a higher-dimensional embedding $\mathbf{u}_i^o \in \mathbb{R}^d$ and project to a scalar with a shared weight vector $\mathbf{w}_o \in \mathbb{R}^d$:

$$\hat{y} = \text{bias} + \sum_i \mathbf{w}_o^\top \mathbf{u}_i^o + \sum_j \mathbf{w}_d^\top \mathbf{u}_j^d$$

This looks richer, but it is not. The composition $\mathbf{w}_o^\top \mathbf{u}_i^o$ is a linear map $\mathbb{R}^d \to \mathbb{R}$, which spans the same function class as a single scalar $\alpha_i$ per player. Any assignment of real numbers to players can be represented with $d=1$. Under joint optimization the higher-dimensional vectors collapse to rank-1 — equivalent to standard RAPM, just overparameterized.

**The root constraint:** whenever the lineup score decomposes as a *sum of independent player terms*, the model is equivalent to RAPM regardless of the embedding dimension.

### What does increase expressiveness

Expressiveness requires that the lineup encoding cannot be decomposed additively. The key tools:

1. **Cross-player attention before aggregation.** Allow each player's representation to attend to teammates and opponents before being summed:

   $$\mathbf{h}_i^o = \text{Attention}\!\left(\mathbf{u}_i^o;\, \{\mathbf{u}_1^o, \ldots, \mathbf{u}_5^o, \mathbf{u}_1^d, \ldots, \mathbf{u}_5^d\}\right)$$

   $$\hat{y} = \text{MLP}\!\left(\textstyle\sum_i \mathbf{h}_i^o,\; \sum_j \mathbf{h}_j^d\right)$$

   After attention, $\mathbf{h}_i^o$ encodes matchup and lineup context — the final sum is no longer a sum of pre-fixed scalars.

2. **Nonlinear pooling (Deep Sets).** $\rho\!\left(\sum_i \varphi(\mathbf{u}_i^o)\right)$ where $\varphi$ and $\rho$ are nonlinear MLPs. By the universal approximation theorem for set functions, this can represent any permutation-invariant function of the lineup.

3. **Bilinear cross-team interactions.** $\sum_i \sum_j (\mathbf{u}_i^o)^\top M\, \mathbf{u}_j^d$ captures matchup-level terms at the cost of $O(25d^2)$ parameters per possession.

### CrossRAPM architecture (implemented)

Each player $i$ is enriched with a feature projection before cross-attention:

$$E_i^o = \mathbf{u}_i^o + W_o f_i, \qquad E_j^d = \mathbf{u}_j^d + W_d f_j$$

where $f_i \in \mathbb{R}^{14}$ is a per-player feature vector (one-hot position + EWMA rate stats). Offense and defense then attend to each other:

$$H^o = \text{LayerNorm}\!\left(E^o + \text{CrossAttn}(Q{=}E^o,\, K{=}E^d,\, V{=}E^d)\right)$$

$$H^d = \text{LayerNorm}\!\left(E^d + \text{CrossAttn}(Q{=}E^d,\, K{=}E^o,\, V{=}E^o)\right)$$

The attention kernel is the standard scaled dot-product:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

The lineup representations are mean-pooled and concatenated with the gamestate for prediction:

$$\hat{y} = \mathbf{w}^\top \bigl[\bar{H}^o \;\|\; \bar{H}^d \;\|\; \mathbf{g}\bigr] + b$$
