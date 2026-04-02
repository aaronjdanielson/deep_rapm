"""
rolling_rapm.py — Rolling RAPM time series plot (CLI entry point).

Delegates computation to deep_rapm.rolling.fit_rolling_rapm (incremental)
or the legacy full-recompute path via deep_rapm.rapm.solve_ridge.

Usage
-----
    python rolling_rapm.py
    python rolling_rapm.py --step-days 7 --half-life 365 --top 20
    python rolling_rapm.py --step-days 7 --half-life 365 --top 20 --ci-window 12 --output rolling_rapm.png --cache rolling_cache.parquet
    python rolling_rapm.py --metric orapm --output orapm_rolling.png
    python rolling_rapm.py --no-incremental   # slow full recompute (for validation)

Requirements (beyond deep-rapm)
---------------------------------
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from deep_rapm import fit_rolling_rapm
from deep_rapm.rapm import _build_matrix_from_df, _recency_weights, _competition_weights, solve_ridge


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR     = Path("data")
PLAYER_VOCAB = Path("data/player_vocab.parquet")
PLAYER_TABLE = Path("data/players.parquet")
SEASONS      = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
WARMUP_DAYS  = 180


# ---------------------------------------------------------------------------
# Legacy full-recompute (kept for numerical validation against incremental)
# ---------------------------------------------------------------------------

def _load_all_possessions(
    data_dir: Path,
    seasons: list[str],
    include_playoffs: bool = False,
) -> pd.DataFrame:
    frames = []
    for season in seasons:
        season_dir   = data_dir / season
        all_parquets = sorted(season_dir.glob("possessions_*.parquet"))
        parquets     = all_parquets if include_playoffs else [
            p for p in all_parquets if "playoffs" not in p.name
        ]
        if not parquets:
            ckpt = season_dir / "checkpoints" / "checkpoint.parquet"
            if ckpt.exists():
                print(f"  {season}: using in-progress checkpoint ({ckpt})")
                parquets = [ckpt]
            else:
                print(f"  Warning: no possessions found for {season} — skipping")
                continue
        for pq in parquets:
            frames.append(pd.read_parquet(pq))
    all_df = pd.concat(frames, ignore_index=True)
    all_df["game_date"] = pd.to_datetime(all_df["game_date"])
    return all_df.sort_values("game_date").reset_index(drop=True)


def _compute_rolling_rapm_full(
    data_dir: Path,
    player_vocab_path: Path,
    player_table_path: Path,
    seasons: list[str],
    step_days: int = 1,
    half_life_days: float = 365.0,
    alpha: float = 2000.0,
    min_poss: int = 100,
    include_playoffs: bool = False,
) -> pd.DataFrame:
    """Full recompute at each date — slow, used only for validation."""
    player_vocab = pd.read_parquet(player_vocab_path)
    player_table = pd.read_parquet(player_table_path)
    name_lookup  = (
        player_table.sort_values("season")
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")["player_name"]
    )

    print("Loading possessions…")
    all_poss   = _load_all_possessions(data_dir, seasons, include_playoffs=include_playoffs)
    first_date = all_poss["game_date"].min()
    last_date  = all_poss["game_date"].max()
    print(f"  {len(all_poss):,} possessions  {first_date.date()} → {last_date.date()}")

    eval_dates = pd.date_range(
        start=first_date + pd.Timedelta(days=WARMUP_DAYS),
        end=last_date,
        freq=f"{step_days}D",
    )
    print(f"\nFitting RAPM at {len(eval_dates)} dates "
          f"(step={step_days}d, half-life={half_life_days:.0f}d)…")

    n_players = len(player_vocab) + 1
    records: list[dict] = []

    for i, eval_date in enumerate(eval_dates):
        df_window = all_poss[all_poss["game_date"] <= eval_date]

        with contextlib.redirect_stdout(io.StringIO()):
            X, y, n_off, n_def = _build_matrix_from_df(df_window, player_vocab)
            w_rec   = _recency_weights(df_window, half_life_days)
            w_comp  = _competition_weights(df_window)
            weights = w_rec * w_comp
            beta, _ = solve_ridge(X, y, alpha, weights=weights)

        alpha_off = beta[:n_players]
        delta_def = beta[n_players:]

        for _, row in player_vocab.iterrows():
            idx = int(row["player_idx"])
            if n_off[idx] < min_poss and n_def[idx] < min_poss:
                continue
            records.append({
                "date":        eval_date,
                "player_id":   int(row["player_id"]),
                "player_name": name_lookup.get(int(row["player_id"]), "Unknown"),
                "orapm":       float(alpha_off[idx] * 100),
                "drapm":       float(-delta_def[idx] * 100),
                "rapm":        float((alpha_off[idx] - delta_def[idx]) * 100),
                "n_off":       int(n_off[idx]),
                "n_def":       int(n_def[idx]),
            })

        if (i + 1) % max(1, len(eval_dates) // 20) == 0 or i == len(eval_dates) - 1:
            print(f"  [{100*(i+1)/len(eval_dates):5.1f}%]  {eval_date.date()}")

    df = pd.DataFrame(records)
    print(f"\nDone. {df['player_id'].nunique()} players tracked across "
          f"{df['date'].nunique()} dates.")
    return df


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_rolling(
    df: pd.DataFrame,
    metric: str = "rapm",
    top_n: int = 10,
    ci_window: int = 12,
    output: Path | None = None,
) -> None:
    """
    ci_window : number of evaluation dates used to compute the local ±1σ band.
                At step_days=7 the default of 12 ≈ a 12-week (3-month) window.
    """
    from scipy.ndimage import gaussian_filter1d

    last_date = df["date"].max()
    final     = df[df["date"] == last_date].nlargest(top_n, metric)
    top_ids   = final["player_id"].tolist()
    name_map  = final.set_index("player_id")["player_name"].to_dict()

    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")
    colors = plt.cm.tab10.colors

    for i, pid in enumerate(top_ids):
        pdata  = df[df["player_id"] == pid].sort_values("date")
        dates  = pdata["date"].values
        vals   = pdata[metric].values
        color  = colors[i % len(colors)]

        smooth = gaussian_filter1d(vals.astype(float), sigma=1.5)

        series = pd.Series(vals)
        roll   = series.rolling(ci_window, center=True, min_periods=3)
        lo     = (smooth - roll.std().fillna(roll.std().median())).values
        hi     = (smooth + roll.std().fillna(roll.std().median())).values

        ax.fill_between(dates, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(dates, smooth, color=color, linewidth=2.0, alpha=0.95, label=name_map[pid])
        ax.scatter([dates[-1]], [smooth[-1]], color=color, s=40, zorder=5)

    ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.45)

    years = range(df["date"].dt.year.min(), df["date"].dt.year.max() + 1)
    for yr in years:
        ax.axvspan(pd.Timestamp(f"{yr}-04-20"), pd.Timestamp(f"{yr}-10-15"),
                   color="grey", alpha=0.06, linewidth=0)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=9)

    metric_label = {"rapm": "RAPM", "orapm": "ORAPM", "drapm": "DRAPM"}.get(metric, metric.upper())
    ax.set_ylabel(f"{metric_label}  (pts / 100 possessions)", fontsize=12)
    ax.set_title(f"Rolling {metric_label} — top {top_n} by current estimate  "
                 f"(ribbon = ±1σ local variability)", fontsize=13, pad=14)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, edgecolor="#cccccc", ncol=2)
    ax.grid(True, alpha=0.2, color="#aaaaaa")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved → {output}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rolling RAPM time series plot.")
    p.add_argument("--data-dir",     type=Path,  default=DATA_DIR)
    p.add_argument("--player-vocab", type=Path,  default=PLAYER_VOCAB)
    p.add_argument("--player-table", type=Path,  default=PLAYER_TABLE)
    p.add_argument("--seasons",      nargs="+",  default=SEASONS)
    p.add_argument("--step-days",    type=int,   default=7)
    p.add_argument("--half-life",    type=float, default=365.0,
                   help="Recency weighting half-life in days (default 365).")
    p.add_argument("--alpha",        type=float, default=2000.0,
                   help="Ridge penalty (default 2000).")
    p.add_argument("--top",          type=int,   default=10)
    p.add_argument("--metric",       default="rapm", choices=["rapm", "orapm", "drapm"])
    p.add_argument("--ci-window",    type=int,   default=12,
                   help="Rolling window (data points) for ±1σ band (default 12).")
    p.add_argument("--output",       type=Path,  default=None)
    p.add_argument("--cache",        type=Path,  default=None,
                   help="Parquet file to cache/load rolling results.")
    p.add_argument("--include-playoffs", action="store_true", default=False)
    p.add_argument("--no-incremental",   action="store_true", default=False,
                   help="Full recompute at each date (slow — for validation only).")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.cache and args.cache.exists():
        print(f"Loading cached results from {args.cache}…")
        df = pd.read_parquet(args.cache)
    elif args.no_incremental:
        df = _compute_rolling_rapm_full(
            data_dir=args.data_dir,
            player_vocab_path=args.player_vocab,
            player_table_path=args.player_table,
            seasons=args.seasons,
            step_days=args.step_days,
            half_life_days=args.half_life,
            alpha=args.alpha,
            include_playoffs=args.include_playoffs,
        )
    else:
        df = fit_rolling_rapm(
            data_dir=args.data_dir,
            seasons=args.seasons,
            player_vocab_path=args.player_vocab,
            player_table_path=args.player_table,
            step_days=args.step_days,
            half_life_days=args.half_life,
            alpha=args.alpha,
            include_playoffs=args.include_playoffs,
        )

    if args.cache and not args.cache.exists():
        df.to_parquet(args.cache, index=False)
        print(f"Cached → {args.cache}")

    plot_rolling(df, metric=args.metric, top_n=args.top,
                 ci_window=args.ci_window, output=args.output)


if __name__ == "__main__":
    main()
