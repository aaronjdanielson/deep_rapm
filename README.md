# deep-rapm

Regularized Adjusted Plus-Minus (RAPM) for NBA possession data — analytical ridge regression with optional recency weighting.

```bash
pip install deep-rapm
```

---

## Overview

RAPM estimates each player's contribution per 100 possessions, controlling for the other nine players on the court. This package provides:

- **Possession collection** — pull and cache play-by-play data from the NBA Stats API
- **Player metadata** — build a player vocabulary with position information
- **Analytical RAPM** — exact ridge regression via the weighted normal equations

---

## Data pipeline

**Step 1 — collect possessions** (~10 min per season):

```bash
collect-possessions --season 2022-23 --output-dir data/2022-23
collect-possessions --season 2023-24 --output-dir data/2023-24
```

**Step 2 — build player vocab and position table**:

```bash
collect-players --seasons 2021-22 2022-23 2023-24
```

Produces `data/player_vocab.parquet` and `data/players.parquet`.

---

## Fitting RAPM

### CLI

**Season mode** (uses pre-collected parquets):

```bash
solve-rapm                                    # 5 training seasons, alpha=2000
solve-rapm --seasons 2021-22 2022-23 2023-24  # specific seasons
solve-rapm --alpha 1000                       # tune regularisation
solve-rapm --half-life 365                    # recency weighting (1-year half-life)
solve-rapm --output-dir runs/rapm             # custom output directory
```

**Date-range mode** (auto-fetches and caches games from the NBA API):

```bash
solve-rapm --from-date 2024-10-01 --to-date 2025-04-15
solve-rapm --from-date 2023-10-01 --to-date 2025-04-15 --half-life 180
```

Output is saved to `checkpoints/rapm/rapm.parquet` and `rapm_summary.json`.

### Python API

```python
from pathlib import Path
from deep_rapm import fit_rapm, load_rapm

# Season mode
results = fit_rapm(
    data_dir=Path("data"),
    seasons=["2021-22", "2022-23", "2023-24"],
    player_vocab_path=Path("data/player_vocab.parquet"),
    player_table_path=Path("data/players.parquet"),
    alpha=2000,
    output_dir=Path("checkpoints/rapm"),
)

# Season mode with recency weighting (1-year half-life)
results = fit_rapm(
    data_dir=Path("data"),
    seasons=["2021-22", "2022-23", "2023-24"],
    player_vocab_path=Path("data/player_vocab.parquet"),
    player_table_path=Path("data/players.parquet"),
    alpha=2000,
    half_life_days=365,
    output_dir=Path("checkpoints/rapm"),
)

# Date-range mode
results = fit_rapm(
    data_dir=Path("data"),
    from_date="2024-10-01",
    to_date="2025-04-15",
    player_vocab_path=Path("data/player_vocab.parquet"),
    player_table_path=Path("data/players.parquet"),
    alpha=2000,
    half_life_days=180,
    output_dir=Path("checkpoints/rapm"),
)

# Load previously saved results
results = load_rapm(Path("checkpoints/rapm"))

qualified = results[results["qualified"]]
print(qualified.nlargest(10, "rapm")[["player_name", "orapm", "drapm", "rapm"]])
```

### Output columns

All values are per 100 possessions.

| Column | Description |
|--------|-------------|
| `orapm` | Offensive RAPM — points added per 100 offensive possessions |
| `drapm` | Defensive RAPM — points prevented per 100 defensive possessions (positive = good defender) |
| `rapm`  | Total RAPM = `orapm + drapm` |
| `n_off` / `n_def` | Offensive / defensive possession counts |
| `qualified` | `True` if ≥ 100 possessions in each role |

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 2000 | Ridge penalty — higher shrinks estimates toward zero |
| `half_life_days` | None | Half-life for recency weighting (days). None = equal weights |
| `min_poss` | 100 | Minimum possessions each role to be flagged as qualified |

### Sample output (2018-19 through 2022-23, alpha=2000)

```
Player                  ORAPM   DRAPM    RAPM
Nikola Jokić            +7.74   +1.94   +9.68
Joel Embiid             +4.44   +4.56   +9.00
Stephen Curry           +6.09   +2.29   +8.38
Giannis Antetokounmpo   +4.33   +4.03   +8.35
LeBron James            +6.01   +2.03   +8.04
Alex Caruso             +0.96   +6.24   +7.20
Rudy Gobert             +0.20   +6.28   +6.48
Damian Lillard          +7.44   -0.52   +6.93
```

---

## Model

We model possession-level outcomes using a ridge-regularized linear model over player participation.

Let:
- $y_i \in \mathbb{R}$ denote the outcome for possession $i$ (e.g., points scored),
- $X \in \mathbb{R}^{n \times 2P}$ be the design matrix,
- $\beta \in \mathbb{R}^{2P}$ be player coefficients.

Each row $X_i$ encodes the 10 players on the court:
- 5 offensive players (indicator = 1 in offense block),
- 5 defensive players (indicator = 1 in defense block).

Thus, each row of $X$ contains exactly 10 ones.

We partition coefficients as:

$$\beta = \begin{bmatrix} \theta^{\text{off}} \\ \theta^{\text{def}} \end{bmatrix}, \quad \theta^{\text{off}}, \theta^{\text{def}} \in \mathbb{R}^P.$$

---

### Linear Model

We model:

$$y = \mu \mathbf{1} + X \beta + \varepsilon,$$

where:
- $\mu \in \mathbb{R}$ is an intercept (not penalized),
- $\varepsilon \sim (0, \sigma^2 I)$.

Equivalently, at the possession level:

$$\hat{y}_i = \mu + \sum_{j \in \text{off}(i)} \theta^{\text{off}}_j + \sum_{k \in \text{def}(i)} \theta^{\text{def}}_k.$$

---

### Weighted Ridge Estimation

To incorporate recency or importance weighting, let:

$$W = \mathrm{diag}(w_1, \dots, w_n), \quad w_i > 0.$$

We estimate parameters via:

$$\min_{\mu, \beta} \; (y - \mu \mathbf{1} - X \beta)^\top W (y - \mu \mathbf{1} - X \beta) + \lambda \|\beta\|_2^2,$$

where:
- $\lambda > 0$ is the ridge penalty,
- the intercept $\mu$ is **not penalized**.

---

### Closed-Form Solution

Define the weighted mean:

$$\bar{y}_w = \frac{\sum_i w_i y_i}{\sum_i w_i}, \quad \bar{X}_w = \frac{\sum_i w_i X_i}{\sum_i w_i}.$$

Center the data:

$$\tilde{y} = y - \bar{y}_w, \quad \tilde{X} = X - \bar{X}_w.$$

Then:

$$\hat{\beta} = (\tilde{X}^\top W \tilde{X} + \lambda I)^{-1} \tilde{X}^\top W \tilde{y},$$

$$\hat{\mu} = \bar{y}_w - \bar{X}_w^\top \hat{\beta}.$$

---

### Interpretation (RAPM)

We report player impacts scaled per 100 possessions:

$$\text{ORAPM}_j = 100 \cdot \theta^{\text{off}}_j, \quad \text{DRAPM}_j = -100 \cdot \theta^{\text{def}}_j.$$

The negative sign in DRAPM arises because:
- $\theta^{\text{def}}_j$ represents contribution to **opponent scoring**,
- strong defenders have $\theta^{\text{def}}_j < 0$.

Thus, higher DRAPM corresponds to better defense.

---

### Summary

| Component | Role |
|-----------|------|
| Offense coefficients $\theta^{\text{off}}$ | Increase scoring |
| Defense coefficients $\theta^{\text{def}}$ | Decrease opponent scoring |
| Ridge penalty $\lambda$ | Stabilizes estimates under collinearity |
| Weights $w_i$ | Time-decay or importance weighting |

---

## License

MIT
