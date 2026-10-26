<p align="center">
  <img src="assets/rolling_rapm.png" alt="Rolling RAPM for today's top 10 players: every week since 2019 on the left, the last six months zoomed in on the right, with playoff, preseason and off-season windows shaded" width="920">
</p>

<h1 align="center">deep-rapm</h1>
<p align="center"><b>Who is actually helping their team win, right now?</b><br>
Regularized Adjusted Plus-Minus over 3.5 million NBA possessions,<br>
refit every week of basketball since 2014 in a few minutes.</p>

<p align="center">
  <a href="https://pypi.org/project/deep-rapm/"><img src="https://img.shields.io/badge/pypi-deep--rapm-blue" alt="PyPI: deep-rapm"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/possessions-3.5M-8A2BE2" alt="3.5M possessions">
  <img src="https://img.shields.io/badge/seasons-2013--14%20%E2%86%92%202026--27-orange" alt="Seasons 2013-14 to 2026-27">
</p>

<p align="center">
  <a href="#the-idea-in-thirty-seconds"><b>The idea</b></a> &nbsp;·&nbsp;
  <a href="#the-trick-incremental-gram-updates"><b>The trick</b></a> &nbsp;·&nbsp;
  <a href="#quickstart"><b>Quickstart</b></a> &nbsp;·&nbsp;
  <a href="#the-data-pipeline"><b>The data</b></a> &nbsp;·&nbsp;
  <a href="#the-model"><b>The model</b></a>
</p>

*Today's top 10, one ridge fit per week. Left: every week since 2014. Right: the last six months on their own scale — the 2026 playoffs (gold), a summer off-season (grey) in which the estimates hold nearly still, and the 2026–27 preseason (blue, weighted at 0.20) moving them again. The dashed box on the left marks the zoomed window.*

---

## The idea in thirty seconds

Plus-minus tells you what happened while a player was on the floor — and credits him for everything his four teammates did. RAPM fixes that with a regression: every possession is a row, the ten players on the court are the columns, and ridge regression untangles who was responsible for the points. The result is each player's offensive and defensive impact **per 100 possessions, controlling for the other nine players**.

`deep-rapm` does this at three speeds:

- **One season, one number** — `solve-rapm`, an exact weighted ridge solve.
- **A date range** — fetch, cache, and fit any window of games.
- **Every week since 2014** — `rolling-rapm`, which refits the entire league at 657 evaluation dates in a few minutes, using the incremental update below.

Possessions are weighted by three things at once: **recency** (exponential decay, one-year half-life), **competitiveness** (garbage time at a 24-point lead counts ~5%), and **context** (preseason possessions enter at 0.20 weight — enough to steady October estimates, too little to let camp scrimmages move a star's rank).

```
pip install "deep-rapm[viz]"
```

## Current leaderboard

<!-- leaderboard:start -->
The most recent weekly fit, using every regular-season, playoff and preseason possession through **October 24, 2026**. Values are points per 100 possessions relative to an average player; positive DRAPM is good defense. Ranked: players who have appeared in a game in the past year.

### Top 25 overall

| # | Player | RAPM | ORAPM | DRAPM |
|--:|---|--:|--:|--:|
| 1 | Victor Wembanyama | **+6.5** | +1.5 | +5.1 |
| 2 | Shai Gilgeous-Alexander | **+6.3** | +3.6 | +2.7 |
| 3 | Chet Holmgren | **+6.3** | +3.1 | +3.2 |
| 4 | Nikola Jokić | **+5.9** | +4.7 | +1.2 |
| 5 | Alex Caruso | **+5.2** | +0.5 | +4.6 |
| 6 | Ajay Mitchell | **+4.8** | +1.9 | +2.9 |
| 7 | Kawhi Leonard | **+4.3** | +3.7 | +0.7 |
| 8 | Giannis Antetokounmpo | **+4.3** | +3.2 | +1.2 |
| 9 | Stephen Curry | **+4.1** | +4.7 | −0.6 |
| 10 | Julian Champagnie | **+3.9** | +3.0 | +0.9 |
| 11 | Aaron Gordon | **+3.9** | +1.7 | +2.2 |
| 12 | Ausar Thompson | **+3.8** | +1.2 | +2.5 |
| 13 | Dylan Harper | **+3.8** | +1.6 | +2.1 |
| 14 | Zach Edey | **+3.7** | +1.4 | +2.3 |
| 15 | Joel Embiid | **+3.5** | +1.5 | +2.0 |
| 16 | Jimmy Butler III | **+3.5** | +1.8 | +1.7 |
| 17 | Luka Dončić | **+3.5** | +3.6 | −0.2 |
| 18 | Donovan Mitchell | **+3.5** | +2.2 | +1.3 |
| 19 | Jarrett Allen | **+3.4** | +2.1 | +1.3 |
| 20 | Payton Pritchard | **+3.4** | +2.6 | +0.8 |
| 21 | Pascal Siakam | **+3.3** | +1.6 | +1.7 |
| 22 | Karl-Anthony Towns | **+3.3** | +2.5 | +0.8 |
| 23 | LaMelo Ball | **+3.2** | +4.0 | −0.8 |
| 24 | Marcus Smart | **+3.2** | +0.8 | +2.4 |
| 25 | Amen Thompson | **+3.2** | +1.2 | +2.0 |

<table>
<tr>
<td valign="top" width="50%">

### Top 10 offense

| # | Player | ORAPM | RAPM |
|--:|---|--:|--:|
| 1 | Stephen Curry | **+4.7** | +4.1 |
| 2 | Nikola Jokić | **+4.7** | +5.9 |
| 3 | LaMelo Ball | **+4.0** | +3.2 |
| 4 | Kawhi Leonard | **+3.7** | +4.3 |
| 5 | Luka Dončić | **+3.6** | +3.5 |
| 6 | Shai Gilgeous-Alexander | **+3.6** | +6.3 |
| 7 | Jamal Murray | **+3.4** | +2.0 |
| 8 | Giannis Antetokounmpo | **+3.2** | +4.3 |
| 9 | Chet Holmgren | **+3.1** | +6.3 |
| 10 | Julian Champagnie | **+3.0** | +3.9 |

</td>
<td valign="top" width="50%">

### Top 10 defense

| # | Player | DRAPM | RAPM |
|--:|---|--:|--:|
| 1 | Victor Wembanyama | **+5.1** | +6.5 |
| 2 | Alex Caruso | **+4.6** | +5.2 |
| 3 | Rudy Gobert | **+4.0** | +3.1 |
| 4 | Chet Holmgren | **+3.2** | +6.3 |
| 5 | Jalen Suggs | **+2.9** | +2.3 |
| 6 | Ajay Mitchell | **+2.9** | +4.8 |
| 7 | Daniss Jenkins | **+2.9** | +3.1 |
| 8 | Shai Gilgeous-Alexander | **+2.7** | +6.3 |
| 9 | Ausar Thompson | **+2.5** | +3.8 |
| 10 | Landry Shamet | **+2.5** | +1.9 |

</td>
</tr>
</table>
<!-- leaderboard:end -->

## The trick: incremental Gram updates

Refitting ridge regression from scratch at every evaluation date means rebuilding $X^\top W X$ from the full possession history — $O(n \cdot P^2)$ per date, with $n$ in the millions. The weights make that unnecessary.

Each possession $i$ carries weight

$$w_i(t) = 0.5^{\Delta_i / H} \cdot \exp\!\left(-d_i^2 / \sigma^2\right) \cdot c_i$$

where $\Delta_i$ is the possession's age in days at evaluation date $t$, $H$ the recency half-life, $d_i$ the score differential when the possession started ($\sigma$ calibrated so a 95th-percentile blowout gets weight 0.05), and $c_i$ a context weight (1.0 for regular season and playoffs, 0.20 for preseason).

Only the recency factor depends on $t$ — and it depends on it the **same way for every possession**. Advance the evaluation date by $\Delta$ days and every existing weight decays by the same scalar $\gamma = 0.5^{\Delta/H}$. So the sufficient statistics decay by $\gamma$ too, and new game-dates just add on:

$$G(t+\Delta) = \gamma \, G(t) + X_{\text{new}}^\top W_{\text{new}} X_{\text{new}}, \qquad b(t+\Delta) = \gamma \, b(t) + X_{\text{new}}^\top W_{\text{new}}\, y_{\text{new}}$$

`IncrementalGramState` holds $G$, $b$, and the intercept accumulators as running state; each weekly step costs $O(k \cdot P^2)$ for the $k$ possessions of the newly ingested dates — roughly **650× fewer floating-point operations** than a rebuild, with numerically identical results (validated against the full recompute via `--no-incremental`).

<details>
<summary><b>The full solve at each date</b></summary>

With state $G = X^\top W X$, $\;b_y = X^\top W y$, $\;b_1 = X^\top W \mathbf{1}$:

$$\hat{\mu} = \frac{\sum_i w_i y_i}{\sum_i w_i}, \qquad \hat{\beta} = (G + \alpha I)^{-1}\left(b_y - \hat{\mu}\, b_1\right)$$

The intercept is the weighted mean of the response and is not penalized; the response is centered, the columns of $X$ are not — the standard RAPM approximation, negligible under strong ridge regularization. Coefficients split into offense and defense blocks, $\beta = [\theta^{\text{off}}, \theta^{\text{def}}]$, and are reported per 100 possessions:

$$\text{ORAPM}_j = 100\,\theta^{\text{off}}_j, \qquad \text{DRAPM}_j = -100\,\theta^{\text{def}}_j$$

(the sign flip makes "higher = better defense"). Players need 100 possessions in a role to appear in the output at a given date.

</details>

## Quickstart

```bash
pip install "deep-rapm[viz]"

# Rolling RAPM: fit every week since 2018, cache the results, draw the chart
rolling-rapm --include-playoffs --include-preseason \
  --cache outputs/rolling_cache.parquet --output outputs/rolling_rapm.png \
  --step-days 7 --half-life 365 --top 20

# One-shot RAPM over chosen seasons
solve-rapm --seasons 2024-25 2025-26 --alpha 2000

# Or any date window, fetched and cached on demand
solve-rapm --from-date 2026-10-01 --to-date 2026-11-15 --half-life 180
```

> [!TIP]
> `--cache` **loads** the cache if the file exists — it does not extend it. To pick up new games, move the cache file aside and rerun.

The same things from Python:

```python
from pathlib import Path
from deep_rapm import fit_rolling_rapm

df = fit_rolling_rapm(
    data_dir=Path("data"),
    seasons=["2024-25", "2025-26", "2026-27"],
    step_days=7, half_life_days=365, alpha=2000,
    include_playoffs=True, include_preseason=True,
)
# columns: date, player_id, player_name, orapm, drapm, rapm, n_off, n_def
latest = df[df["date"] == df["date"].max()]
print(latest.nlargest(10, "rapm")[["player_name", "orapm", "drapm", "rapm"]])
```

<details>
<summary><b>Key parameters</b></summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 2000 | Ridge penalty — higher shrinks estimates toward zero |
| `half_life_days` | 365 | Recency-weighting half-life in days (`None` = equal weights in one-shot mode) |
| `step_days` | 7 | Days between rolling evaluation dates |
| `warmup_days` | 180 | Skip the first N days (too little data for stable estimates) |
| `min_poss` | 100 | Minimum possessions per role to be reported |
| `include_playoffs` | False | Include playoff possessions at full weight |
| `include_preseason` | False | Include preseason possessions at `preseason_weight` |
| `preseason_weight` | 0.20 | Context weight multiplier for preseason possessions |

</details>

## Any players, side by side

<p align="center">
  <img src="assets/players_sample.png" alt="Rolling offensive and defensive RAPM since 2017 for Nikola Jokić, Joel Embiid, Giannis Antetokounmpo, Stephen Curry and Rudy Gobert" width="920">
</p>

Every weekly fit since 2014 is cached, so comparing any set of players over any span is one command — names are matched loosely ("Jokic", "Giannis", "Embiid"), one panel per metric:

```bash
plot-players --players "Nikola Jokić" Embiid Giannis "Stephen Curry" Gobert \
  --cache outputs/rolling_cache.parquet --output players.png          # offense + defense, full history
plot-players --players Curry Lillard Haliburton --metrics orapm --since 2021-10-01
plot-players --players Caruso Suggs Wembanyama --metrics drapm rapm --no-ribbon
```

*Above: Jokić's offense surpassing Curry's in 2025, Gobert's defense back above +4, Embiid's two-way peak in 2021–22.*

## The data pipeline

Everything comes from public NBA sources and lands in per-season parquet files — one row per possession, with the ten players on court, the points scored, and the score differential when the possession began.

| Source | Role |
|---|---|
| `pbpstats` (enhanced play-by-play via `data.nba.com`) | Primary per-game possession extraction |
| `nba_api` `PlayByPlayV3` + `BoxScoreTraditionalV3` | Automatic fallback when pbpstats returns nothing (2025-26 on, Akamai blocks). Lineups are inferred from substitutions and box-score rosters — identical to the NBA's own rotation data on 100% of possessions in a 48-game check — and team points match official finals |
| `nba_api` `LeagueGameLog` / `LeagueGameFinder` | Season schedules and nightly game discovery |
| `nba_api` `CommonTeamRoster` / `CommonPlayerInfo` | Player names, positions, and the 1,457-player vocabulary |

```bash
collect-possessions --season 2024-25        # one season, cached per game
collect-players --seasons 2024-25 2025-26   # player vocab + position table
python scripts/collect_nightly.py           # cron target: yesterday's games → season parquets
```

Collection is **idempotent**: each game is written once to `data/games/`, and the nightly merge appends only game IDs the season parquets don't already have. Current coverage: fourteen seasons, 2013-14 through 2026-27 — regular season, playoffs (play-in included), and 1,088 preseason games back to 2013. `scripts/validate_data.py` checks every game against official box scores, and `--exclude-flagged` on the rolling command leaves out any game it marks.

## The model

A possession is a row of $X \in \mathbb{R}^{n \times 2P}$ with exactly ten ones: five in the offense block, five in the defense block. The outcome is the points scored on the possession,

$$y = \mu \mathbf{1} + X\beta + \varepsilon,$$

fit by weighted ridge regression as described above. The ridge penalty is what makes this *adjusted*: teammates who always play together are statistically inseparable, and $\alpha$ resolves that collinearity by shrinking toward zero rather than exploding in opposite directions.

| Component | Role |
|-----------|------|
| Offense coefficients $\theta^{\text{off}}$ | Points a player adds per offensive possession |
| Defense coefficients $\theta^{\text{def}}$ | Points a player allows per defensive possession (reported negated) |
| Ridge penalty $\alpha$ | Stabilizes estimates under lineup collinearity |
| Weights $w_i$ | Recency × competitiveness × context |

## What is in this repository

```
deep_rapm/            the package
  rolling.py            IncrementalGramState + fit_rolling_rapm — the incremental solver
  rapm.py               design matrix, recency/competition weights, exact ridge solve
  data/                 possession extraction (pbpstats + nba_api fallback), seasons, players,
                        and validate.py (data-quality checks)
  scripts/              console entry points: rolling-rapm (the chart above), solve-rapm, collect-*
scripts/              operator tools
  collect_nightly.py    nightly cron: new games → per-game cache → season parquets
  validate_data.py      per-game quality table: structural checks + box-score reconciliation
  update_readme_leaderboard.py   regenerates the leaderboard tables on this page
  legacy/               one-off backfill scripts, kept for reference
tests/                possession extraction, solver, incremental-vs-full equivalence
docs/                 derivation notes (incremental_rapm.md); papers/ holds LaTeX drafts
experiments/          neural lineup-model research (not part of the package)
assets/               images on this page
```

Generated and local-only (gitignored): `data/` (possessions, player tables), `outputs/` (rolling cache, charts, archived backups and logs), `pbp_cache/`, `checkpoints/`.

## License

MIT — see [LICENSE](LICENSE).
