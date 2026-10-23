"""
rolling_rapm.py — Rolling RAPM time series plot (CLI entry point).

Delegates computation to deep_rapm.rolling.fit_rolling_rapm (incremental)
or the legacy full-recompute path via deep_rapm.rapm.solve_ridge.

Usage (from the repo root; `rolling-rapm` is the installed equivalent)
-----
    python -m deep_rapm.scripts.rolling_rapm
    python -m deep_rapm.scripts.rolling_rapm --step-days 7 --half-life 365 --top 20
    python -m deep_rapm.scripts.rolling_rapm --top 20 --cache outputs/rolling_cache.parquet --output outputs/rolling_rapm.png
    python -m deep_rapm.scripts.rolling_rapm --metric orapm --output outputs/orapm_rolling.png
    python -m deep_rapm.scripts.rolling_rapm --no-incremental   # slow full recompute (for validation)

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
SEASONS      = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
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

#: Background shading per season phase: (color, band alpha, label).
#: Hues are CVD-distinguishable (gold / blue / grey); bands stay recessive.
PHASE_STYLE = {
    "playoffs":  ("#c9a227", 0.15, "Playoffs"),
    "preseason": ("#4878a8", 0.15, "Preseason"),
    "offseason": ("grey",    0.06, "Off-season"),
}


def _phase_windows(
    data_dir: Path,
    seasons: list[str],
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Date windows per season phase, read from the possession parquets.

    Returns {"preseason": [(start, end), …], "playoffs": […], "offseason": […]}.
    A phase with disjoint clusters of game dates (e.g. the 2019-20 preseason
    parquet also holds the July 2020 bubble scrimmages) yields one window per
    contiguous run, split wherever consecutive game dates sit > 21 days apart.
    Off-season windows are the gaps between one season's last game (any phase)
    and the next season's first. Seasons without a phase parquet are skipped.
    """
    def _runs(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        days = pd.to_datetime(dates.unique()).sort_values()
        out, start = [], days[0]
        for prev, cur in zip(days, days[1:]):
            if (cur - prev).days > 21:
                out.append((start, prev))
                start = cur
        out.append((start, days[-1]))
        return out

    windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {
        "preseason": [], "playoffs": [],
    }
    season_bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for season in seasons:
        slug = season.replace("-", "_")
        lo = hi = None
        for phase, suffix in (("preseason", "preseason"),
                              ("regular",   "regular_season"),
                              ("playoffs",  "playoffs")):
            p = Path(data_dir) / season / f"possessions_{slug}_{suffix}.parquet"
            if not p.exists():
                continue
            runs = _runs(pd.read_parquet(p, columns=["game_date"])["game_date"])
            if phase in windows:
                windows[phase].extend(runs)
            lo = runs[0][0]  if lo is None else min(lo, runs[0][0])
            hi = runs[-1][1] if hi is None else max(hi, runs[-1][1])
        if lo is not None:
            season_bounds.append((lo, hi))
    season_bounds.sort()
    windows["offseason"] = [
        (end, nxt_start)
        for (_, end), (nxt_start, _) in zip(season_bounds, season_bounds[1:])
    ]
    return windows


#: Line styles by rank block: ranks 1–10 solid, 11–20 dashed, 21–30 dotted.
#: Colors repeat across blocks, so style keeps every series distinguishable.
_RANK_LINESTYLES = ["-", (0, (5, 2)), (0, (1, 1.6))]

_INK, _INK_MUTED, _GRID = "#1f2328", "#656d76", "#e6e8eb"


def _spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """Nudge label y-positions apart by at least *min_gap*, keeping their order.

    Only colliding labels move: each run of overlapping labels is spaced
    evenly around the mean of its members' true values, and runs that grow
    into each other merge, until nothing overlaps. Touching labels count as
    one run, so a spaced run never splits back apart (which would oscillate).
    """
    import numpy as np

    ys = np.asarray(ys, dtype=float)
    order = np.argsort(ys)
    true = ys[order]
    pos = true.copy()
    for _ in range(len(pos) + 1):
        clusters = [[0]]
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < min_gap + 1e-9:
                clusters[-1].append(i)
            else:
                clusters.append([i])
        changed = False
        for c in clusters:
            if len(c) > 1:
                start = true[c].mean() - (len(c) - 1) * min_gap / 2
                new = start + np.arange(len(c)) * min_gap
                if not np.allclose(pos[c], new):
                    pos[c] = new
                    changed = True
        if not changed:
            break
    out = np.empty_like(pos)
    out[order] = pos
    return out.tolist()


def plot_rolling(
    df: pd.DataFrame,
    metric: str = "rapm",
    top_n: int = 10,
    ci_window: int = 12,
    output: Path | None = None,
    phase_windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] | None = None,
    recent_days: int = 183,
) -> None:
    """
    Two panels: the full multi-year trajectory (left) and the last
    *recent_days* zoomed on their own y-scale (right, with direct labels).

    ci_window : number of evaluation dates used to compute the local ±1σ band.
                At step_days=7 the default of 12 ≈ a 12-week (3-month) window.
    """
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.transforms import blended_transform_factory
    from scipy.ndimage import gaussian_filter1d

    last_date = df["date"].max()
    final     = df[df["date"] == last_date].nlargest(top_n, metric)
    top_ids   = final["player_id"].tolist()
    name_map  = final.set_index("player_id")["player_name"].to_dict()
    colors    = plt.cm.tab10.colors
    metric_label = {"rapm": "RAPM", "orapm": "ORAPM", "drapm": "DRAPM"}.get(
        metric, metric.upper())

    # Smoothing and ribbons are computed once over the full history, so the
    # zoom panel shows exactly the same curves as the long view.
    series = []
    for i, pid in enumerate(top_ids):
        pdata  = df[df["player_id"] == pid].sort_values("date")
        vals   = pdata[metric].values.astype(float)
        smooth = gaussian_filter1d(vals, sigma=1.5)
        sd     = pd.Series(vals).rolling(ci_window, center=True, min_periods=3).std()
        sd     = sd.fillna(sd.median()).values
        series.append({
            "dates":   pdata["date"].values,
            "smooth":  smooth,
            "lo":      smooth - sd,
            "hi":      smooth + sd,
            "current": float(vals[-1]),
            "name":    name_map[pid],
            "color":   colors[i % len(colors)],
            "style":   _RANK_LINESTYLES[min(i // len(colors), len(_RANK_LINESTYLES) - 1)],
        })

    xlo     = df["date"].min()
    zoom_lo = last_date - pd.Timedelta(days=recent_days)

    band_spans: list[tuple[pd.Timestamp, pd.Timestamp, str, float]] = []
    band_handles = []
    if phase_windows:
        for phase, (color, alpha, label) in PHASE_STYLE.items():
            spans = [(max(s, xlo), min(e, last_date))
                     for s, e in phase_windows.get(phase, [])]
            spans = [(s, e) for s, e in spans if s < e]
            band_spans += [(s, e, color, alpha) for s, e in spans]
            if spans:
                # Legend chips use a stronger alpha so the hue stays readable.
                band_handles.append(
                    Patch(facecolor=color, alpha=min(3 * alpha, 1.0), label=label))
    else:
        # Fallback without data access: fixed approximate off-season spans.
        for yr in range(xlo.year, last_date.year + 1):
            band_spans.append((pd.Timestamp(f"{yr}-04-20"), pd.Timestamp(f"{yr}-10-15"),
                               "grey", 0.06))

    def _draw(ax, linewidth: float) -> None:
        for s, e, color, alpha in band_spans:
            ax.axvspan(s, e, color=color, alpha=alpha, linewidth=0, zorder=0)
        for s in series:
            ax.fill_between(s["dates"], s["lo"], s["hi"], color=s["color"],
                            alpha=0.07, linewidth=0)
            ax.plot(s["dates"], s["smooth"], color=s["color"], linewidth=linewidth,
                    linestyle=s["style"], alpha=0.95,
                    solid_capstyle="round", dash_capstyle="round")
        ax.axhline(0, color="#8c959f", linewidth=0.9, zorder=1)
        ax.tick_params(axis="x", which="major", labelsize=10.5, length=5)
        ax.tick_params(axis="x", which="minor", length=3)
        ax.tick_params(axis="y", labelsize=10.5, length=0)
        ax.grid(True, axis="y", color=_GRID, linewidth=0.9)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#c9ced4",
        "xtick.color": _INK_MUTED,
        "ytick.color": _INK_MUTED,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(16, 8.6))
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1], wspace=0.10,
                              left=0.05, right=0.80, top=0.82, bottom=0.08)
        ax_all, ax_zoom = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

        # Left: the multi-year trajectory.
        _draw(ax_all, linewidth=1.5)
        ax_all.set_xlim(xlo - pd.Timedelta(days=20), last_date + pd.Timedelta(days=12))
        ax_all.xaxis.set_major_locator(mdates.YearLocator())
        ax_all.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_all.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax_all.set_ylabel(f"{metric_label}, points per 100 possessions",
                          fontsize=11, color=_INK_MUTED, labelpad=10)
        ax_all.set_title(f"Since {xlo.year}", loc="left", fontsize=12,
                         color=_INK, pad=10)

        # Right: the last few months, on their own y-scale.
        _draw(ax_zoom, linewidth=2.3)
        ax_zoom.set_xlim(zoom_lo, last_date + pd.Timedelta(days=4))
        ax_zoom.xaxis.set_major_locator(mdates.MonthLocator())
        ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        in_win = [(s["dates"] >= zoom_lo.to_datetime64()) for s in series]
        lo = min(min(s["lo"][m].min(), s["current"])
                 for s, m in zip(series, in_win) if m.any())
        hi = max(max(s["hi"][m].max(), s["current"])
                 for s, m in zip(series, in_win) if m.any())
        pad = 0.06 * (hi - lo)
        ax_zoom.set_ylim(lo - pad, hi + pad)
        months = round(recent_days / 30.5)
        ax_zoom.set_title(f"Last {months} months, zoomed", loc="left", fontsize=12,
                          color=_INK, pad=10)
        # End dots mark the current (unsmoothed) estimate, matching the labels.
        for s in series:
            ax_zoom.scatter([s["dates"][-1]], [s["current"]], color=s["color"],
                            s=38, zorder=5, edgecolor="white", linewidth=1.2)

        # Outline the zoomed window on the long view.
        ax_all.add_patch(Rectangle(
            (mdates.date2num(zoom_lo), lo - pad),
            mdates.date2num(last_date + pd.Timedelta(days=4)) - mdates.date2num(zoom_lo),
            hi - lo + 2 * pad,
            fill=False, edgecolor="#57606a", linewidth=1.0,
            linestyle=(0, (3, 2)), zorder=6))

        # Direct labels beside the zoom panel, nudged apart, with leader lines.
        fig.canvas.draw()
        y0, y1 = ax_zoom.get_ylim()
        px_per_unit = ax_zoom.get_window_extent().height / (y1 - y0)
        min_gap = 9.0 * fig.dpi / 72 * 1.3 / px_per_unit
        label_ys = _spread_labels([s["current"] for s in series], min_gap)
        to_label = blended_transform_factory(ax_zoom.transAxes, ax_zoom.transData)
        for s, ly in zip(series, label_ys):
            ax_zoom.annotate(
                f"{s['name']}  {s['current']:+.1f}".replace("-", "−"),
                xy=(s["dates"][-1], s["current"]), xycoords="data",
                xytext=(1.05, ly), textcoords=to_label,
                fontsize=9, color=_INK, va="center", ha="left",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=s["color"], linewidth=1.0,
                                shrinkA=3, shrinkB=2, alpha=0.9),
            )

        fig.text(0.05, 0.945, f"Rolling {metric_label}: the top {top_n} today, "
                 f"and how they got there",
                 fontsize=18, fontweight="bold", color=_INK, ha="left")
        fig.text(0.05, 0.905,
                 f"Weekly ridge fits through {last_date:%b} {last_date.day}, "
                 f"{last_date:%Y}  ·  relative to an average player  ·  "
                 f"ribbon = ±1σ local variability",
                 fontsize=11, color=_INK_MUTED, ha="left")
        if band_handles:
            fig.legend(handles=band_handles, loc="upper right",
                       bbox_to_anchor=(0.80, 0.925), ncol=len(band_handles),
                       frameon=False, fontsize=10, handlelength=1.4,
                       handleheight=1.0, columnspacing=1.4, labelcolor=_INK_MUTED)

        if output:
            fig.savefig(output, dpi=150, facecolor="white")
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
    p.add_argument("--include-preseason", action="store_true", default=False,
                   help="Include preseason possessions, weighted by --preseason-weight.")
    p.add_argument("--preseason-weight", type=float, default=0.20,
                   help="Per-possession weight for preseason (default 0.20).")
    p.add_argument("--exclude-flagged", type=Path, nargs="?", default=None,
                   const=Path("data/quality/game_quality.parquet"), metavar="QUALITY",
                   help="Leave out games marked severity=error in a quality table "
                        "(scripts/validate_data.py; default path if no value given).")
    p.add_argument("--no-incremental",   action="store_true", default=False,
                   help="Full recompute at each date (slow — for validation only).")
    return p


def _flagged_games(quality_path: Path | None) -> set[str] | None:
    if quality_path is None:
        return None
    q = pd.read_parquet(quality_path, columns=["game_id", "severity"])
    return set(q.loc[q["severity"] == "error", "game_id"])


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
            include_preseason=args.include_preseason,
            preseason_weight=args.preseason_weight,
            exclude_game_ids=_flagged_games(args.exclude_flagged),
        )

    if args.cache and not args.cache.exists():
        df.to_parquet(args.cache, index=False)
        print(f"Cached → {args.cache}")

    plot_rolling(df, metric=args.metric, top_n=args.top,
                 ci_window=args.ci_window, output=args.output,
                 phase_windows=_phase_windows(args.data_dir, args.seasons))


if __name__ == "__main__":
    main()
