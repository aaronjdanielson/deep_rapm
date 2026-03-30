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

where $y_c = y - \mu$ is mean-centred. The normal equations are

$$\bigl(X^\top X + \alpha I\bigr)\beta = X^\top y_c$$

**Recency-weighted ridge.** With half-life $\tau$ (days), each possession is weighted by its age:

$$w_i = 0.5^{\,d_i / \tau}$$

where $d_i$ is days before the most recent possession. The weighted normal equations are

$$\bigl(X^\top W X + \alpha I\bigr)\beta = X^\top W y_c$$

with $W = \mathrm{diag}(w)$, computed efficiently as $(X \odot \sqrt{w})^\top (X \odot \sqrt{w})$.

**Reported values** (per 100 possessions):

$$\text{ORAPM}_j = 100 \cdot \alpha_j \qquad \text{DRAPM}_k = -100 \cdot \delta_k \qquad \text{RAPM} = \text{ORAPM} + \text{DRAPM}$$

The sign flip on DRAPM makes positive values mean good defender — a defender
who suppresses scoring has $\delta_k < 0$, so $\text{DRAPM}_k > 0$.

---

## License

MIT
