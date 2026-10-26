"""
plot_players.py — long-term offensive and defensive RAPM for chosen players.

Reads the rolling cache written by rolling-rapm (one row per evaluation date
and player) and draws one panel per metric with one line per player, using
the same smoothing, ±1σ ribbons and season-phase shading as the headline
chart. Players are matched by name, case- and accent-insensitively, exactly
or as a substring ("Jokic", "Giannis"); a player id works too.

Usage (from the repo root; `plot-players` is the installed equivalent)
-----
    python -m deep_rapm.scripts.plot_players --players "Nikola Jokić" "Joel Embiid" "Rudy Gobert" \\
        --cache outputs/rolling_cache.parquet --output outputs/players.png
    python -m deep_rapm.scripts.plot_players --players Curry Lillard Haliburton --metrics orapm --since 2021-10-01
    python -m deep_rapm.scripts.plot_players --players Caruso "Jalen Suggs" Wembanyama --metrics drapm rapm
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from deep_rapm.scripts.rolling_rapm import (
    DATA_DIR, PHASE_STYLE, SEASONS, _GRID, _INK, _INK_MUTED, _RANK_LINESTYLES,
    _phase_windows, _spread_labels,
)

METRIC_LABEL = {"rapm": "RAPM", "orapm": "Offensive RAPM", "drapm": "Defensive RAPM"}


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def resolve_players(df: pd.DataFrame, queries: list[str]) -> list[tuple[int, str]]:
    """(player_id, player_name) for each query: an id, the full name, a whole
    word of it ("Giannis", "Jokic"), a word prefix ("Antetok"), or a unique
    substring — the first of those tiers with a unique hit wins. Ambiguous or
    unknown queries stop with the candidates."""
    names = df.drop_duplicates("player_id").set_index("player_id")["player_name"]
    folded = {pid: _fold(n) for pid, n in names.items()}
    out: list[tuple[int, str]] = []
    for q in queries:
        if q.isdigit() and int(q) in names.index:
            out.append((int(q), names[int(q)]))
            continue
        fq = _fold(q)
        tiers = (
            lambda n: n == fq,
            lambda n: fq in n.split(),
            lambda n: any(w.startswith(fq) for w in n.split()),
            lambda n: fq in n,
        )
        hits: list[int] = []
        for match in tiers:
            hits = [pid for pid, n in folded.items() if match(n)]
            if hits:
                break
        if len(hits) == 1:
            out.append((hits[0], names[hits[0]]))
        elif not hits:
            raise SystemExit(f"no player in the cache matches {q!r}")
        else:
            raise SystemExit(f"{q!r} is ambiguous: " + ", ".join(sorted(names[p] for p in hits)))
    return out


def _runs(mask: np.ndarray):
    """Start/stop index pairs of the contiguous True runs in *mask*."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1]] + 1
    return list(zip(starts, stops))


def plot_players(
    df: pd.DataFrame,
    players: list[tuple[int, str]],
    metrics: list[str],
    output: Path | None = None,
    phase_windows: dict | None = None,
    ci_window: int = 12,
    ribbon: bool = True,
    inactive_grace_days: int = 240,   # longer than an off-season (~6 months)
) -> None:
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    dates = np.array(sorted(df["date"].unique()))
    colors = plt.cm.tab10.colors
    xlo, xhi = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])

    # Per player and metric: values on the common date grid (NaN where the
    # player is below the possession filter), smoothed within each run of
    # consecutive dates so gaps stay gaps.
    series: dict[tuple[int, str], dict] = {}
    for pid, _ in players:
        p = df[df["player_id"] == pid].set_index("date").sort_index()
        # The fit keeps estimating a player while he isn't playing (the ridge
        # just shrinks stale data toward zero) — a season abroad, a lost year
        # to injury, retirement. Blank any stretch longer than the grace period
        # in which his cumulative possession counts did not grow; the grace
        # exceeds an off-season, so summers carry through.
        counts = p["n_off"] + p["n_def"]
        grew = counts.diff().fillna(1) > 0
        last_growth = pd.Series(np.where(grew, p.index, np.datetime64("NaT")), index=p.index).ffill()
        idle_days = (p.index.to_series() - last_growth).dt.days
        p = p[idle_days <= inactive_grace_days]
        for m in metrics:
            vals = p[m].reindex(dates).to_numpy(dtype=float)
            smooth = np.full_like(vals, np.nan)
            sd = np.full_like(vals, np.nan)
            for a, b in _runs(~np.isnan(vals)):
                seg = vals[a:b]
                smooth[a:b] = gaussian_filter1d(seg, sigma=1.5) if b - a > 2 else seg
                s = pd.Series(seg).rolling(ci_window, center=True, min_periods=3).std()
                sd[a:b] = s.fillna(s.median() if s.notna().any() else 0.0).to_numpy()
            series[(pid, m)] = {"smooth": smooth, "sd": sd, "last": int(np.flatnonzero(~np.isnan(vals))[-1])
                                if (~np.isnan(vals)).any() else None,
                                "current": vals[int(np.flatnonzero(~np.isnan(vals))[-1])]
                                if (~np.isnan(vals)).any() else np.nan}

    band_spans, band_handles = [], []
    if phase_windows:
        for phase, (color, alpha, label) in PHASE_STYLE.items():
            spans = [(max(s, xlo), min(e, xhi)) for s, e in phase_windows.get(phase, [])]
            spans = [(s, e) for s, e in spans if s < e]
            band_spans += [(s, e, color, alpha) for s, e in spans]
            if spans:
                band_handles.append(Patch(facecolor=color, alpha=min(3 * alpha, 1.0), label=label))

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#c9ced4",
        "xtick.color": _INK_MUTED,
        "ytick.color": _INK_MUTED,
    }
    n = len(metrics)
    with plt.rc_context(rc):
        fig, axes = plt.subplots(n, 1, figsize=(15, 3.9 * n + 1.3), sharex=True)
        axes = np.atleast_1d(axes)
        fig.patch.set_facecolor("white")
        top = 1 - 1.2 / (3.9 * n + 1.3)
        fig.subplots_adjust(left=0.05, right=0.80, top=top, bottom=0.6 / (3.9 * n + 1.3), hspace=0.18)

        for ax, m in zip(axes, metrics):
            ax.set_facecolor("white")
            for s, e, color, alpha in band_spans:
                ax.axvspan(s, e, color=color, alpha=alpha, linewidth=0, zorder=0)
            ends = []
            for i, (pid, name) in enumerate(players):
                sr = series[(pid, m)]
                color = colors[i % len(colors)]
                style = _RANK_LINESTYLES[min(i // len(colors), len(_RANK_LINESTYLES) - 1)]
                if ribbon:
                    ax.fill_between(dates, sr["smooth"] - sr["sd"], sr["smooth"] + sr["sd"],
                                    color=color, alpha=0.07, linewidth=0)
                ax.plot(dates, sr["smooth"], color=color, linewidth=1.9, linestyle=style,
                        alpha=0.95, solid_capstyle="round", dash_capstyle="round")
                if sr["last"] is not None:
                    ax.scatter([dates[sr["last"]]], [sr["current"]], color=color, s=34,
                               zorder=5, edgecolor="white", linewidth=1.2)
                    ends.append((sr["current"], dates[sr["last"]], color, name))
            ax.axhline(0, color="#8c959f", linewidth=0.9, zorder=1)
            ax.grid(True, axis="y", color=_GRID, linewidth=0.9)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=10.5, length=0)
            ax.set_ylabel(f"{METRIC_LABEL[m]}, pts / 100 poss.", fontsize=11, color=_INK_MUTED, labelpad=10)

            # Direct labels at the right edge, spread apart, with leader lines.
            fig.canvas.draw()
            y0, y1 = ax.get_ylim()
            px_per_unit = ax.get_window_extent().height / (y1 - y0)
            min_gap = 9.0 * fig.dpi / 72 * 1.3 / px_per_unit
            label_ys = _spread_labels([e[0] for e in ends], min_gap) if ends else []
            to_label = blended_transform_factory(ax.transAxes, ax.transData)
            for (y, x, color, name), ly in zip(ends, label_ys):
                ax.annotate(f"{name}  {y:+.1f}".replace("-", "−"),
                            xy=(x, y), xycoords="data", xytext=(1.02, ly), textcoords=to_label,
                            fontsize=9, color=_INK, va="center", ha="left", annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color=color, linewidth=1.0,
                                            shrinkA=3, shrinkB=2, alpha=0.9))

        ax = axes[-1]
        ax.set_xlim(xlo - pd.Timedelta(days=20), xhi + pd.Timedelta(days=12))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax.tick_params(axis="x", which="major", labelsize=10.5, length=5)
        ax.tick_params(axis="x", which="minor", length=3)

        fig.text(0.05, 1 - 0.38 / (3.9 * n + 1.3),
                 f"Rolling {' and '.join(METRIC_LABEL[m] for m in metrics)}, {len(players)} players",
                 fontsize=18, fontweight="bold", color=_INK, ha="left")
        fig.text(0.05, 1 - 0.72 / (3.9 * n + 1.3),
                 f"Weekly ridge fits through {xhi:%b} {xhi.day}, {xhi:%Y}  ·  relative to an average "
                 f"player" + ("  ·  ribbon = ±1σ local variability" if ribbon else ""),
                 fontsize=11, color=_INK_MUTED, ha="left")
        if band_handles:
            fig.legend(handles=band_handles, loc="upper right", bbox_to_anchor=(0.80, top + 0.045),
                       ncol=len(band_handles), frameon=False, fontsize=10, handlelength=1.4,
                       handleheight=1.0, columnspacing=1.4, labelcolor=_INK_MUTED)

        if output:
            fig.savefig(output, dpi=150, facecolor="white")
            print(f"Saved → {output}")
        else:
            plt.show()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Long-term RAPM chart for chosen players.")
    p.add_argument("--players", nargs="+", required=True, metavar="NAME",
                   help="Player names (accent/case-insensitive, substrings allowed) or ids.")
    p.add_argument("--metrics", nargs="+", default=["orapm", "drapm"],
                   choices=["rapm", "orapm", "drapm"], help="One panel per metric (default: orapm drapm).")
    p.add_argument("--cache", type=Path, default=Path("outputs/rolling_cache.parquet"),
                   help="Rolling cache written by rolling-rapm --cache.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, help="For the season-phase shading.")
    p.add_argument("--since", type=pd.Timestamp, default=None, help="First date to show (YYYY-MM-DD).")
    p.add_argument("--ci-window", type=int, default=12, help="Rolling window (eval dates) for the ±1σ ribbon.")
    p.add_argument("--no-ribbon", action="store_true")
    p.add_argument("--output", type=Path, default=None, help="PNG path; shows a window if omitted.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    df = pd.read_parquet(args.cache)
    players = resolve_players(df, args.players)
    if args.since is not None:
        df = df[df["date"] >= args.since]
    plot_players(df, players, args.metrics, output=args.output,
                 phase_windows=_phase_windows(args.data_dir, SEASONS),
                 ci_window=args.ci_window, ribbon=not args.no_ribbon)


if __name__ == "__main__":
    main()
