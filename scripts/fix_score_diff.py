"""Repair score_diff so it is the margin at the *start* of every possession.

    python3.11 scripts/fix_score_diff.py            # check: per-file report, nothing written
    python3.11 scripts/fix_score_diff.py --apply    # rewrite season parquets, CSV mirrors and data/games cache (backs up)

The pbpstats parser (deep_rapm/data/game.py, fixed 2026-10-25) snapshotted
the score after the possession's first event, so a possession opened by a
made shot stored the margin including its own points. Every pbpstats-parsed
game (2018-19 … 2024-25 except the games re-collected through nba_api) is
affected; the competition weight then down-weighted scores by the team
ahead and up-weighted scores by the team behind, shrinking competitive-time
margins toward zero in those seasons (sd 7 instead of 14 per game).

Per game:
  * games whose stored margins already follow the start rule (nba_api
    extractor, all seasons outside 2018-25) are left untouched;
  * affected games with a cached pbpstats play-by-play (pbp_cache/pbp/) are
    re-parsed offline with the corrected parser — exact, and every other
    column is checked to be identical before the new margins are taken;
  * affected games without a cache are rebuilt from the possessions' own
    points, re-anchored at non-scoring possessions (whose stored value is
    exact) and constrained by the pbpstats bounds start ≤ stored ≤ start +
    points. Against the exact re-parse on the 2022-23 regular season this
    agrees on 98.8% of possessions and is never off by more than 3 points
    (the stored values agreed on 69%, biased by +0.71).

The per-game cache (data/games/) is rewritten from the repaired season rows
of each changed game, so both copies agree.
"""
import argparse
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from deep_rapm.data.game import get_game_possessions

DATA = Path("data")
PBP_CACHE = Path("pbp_cache")
KEY_COLS = ["period", "seconds_into_game", "offense_team_id", "defense_team_id", "points"]


def _sorted(df: pd.DataFrame) -> np.ndarray:
    return np.lexsort((np.arange(len(df)), df["seconds_into_game"].values, df["game_id"].values))


def classify_games(df: pd.DataFrame) -> pd.Series:
    """'START' (margins at possession start), 'END'/'tie' (affected) per game."""
    d = df.iloc[_sorted(df)]
    same = d["game_id"].values[1:] == d["game_id"].values[:-1]
    dt = d["offense_team_id"].values[1:] != d["offense_team_id"].values[:-1]
    ok = same & dt
    sd0, sd1 = d["score_diff"].values[:-1][ok], d["score_diff"].values[1:][ok]
    p0, p1 = d["points"].values[:-1][ok], d["points"].values[1:][ok]
    sc = (p0 > 0) | (p1 > 0)
    t = pd.DataFrame({"g": d["game_id"].values[1:][ok], "start": (sd1 == -(sd0 + p0)) & sc,
                      "end": (sd1 == -sd0 + p1) & sc}).groupby("g").sum()
    cls = pd.Series(np.where(t["start"] > t["end"], "START", np.where(t["end"] > t["start"], "END", "tie")),
                    index=t.index)
    return cls.reindex(df["game_id"].unique()).fillna("tie")


def rebuilt_score_diff(df: pd.DataFrame) -> np.ndarray:
    """Start-of-possession margin for pbpstats-parsed games, from the possessions' own points."""
    order = _sorted(df)
    d = df.iloc[order]
    home = (d["offense_team_id"].values == d["home_team_id"].values)
    pts = d["points"].values.astype(np.int64)
    hp = np.where(home, pts, 0)
    ap = np.where(~home, pts, 0)
    keys = pd.factorize(d["game_id"].values)[0]
    ch = pd.Series(hp).groupby(keys).cumsum().values - hp          # home points before this possession
    ca = pd.Series(ap).groupby(keys).cumsum().values - ap
    rebuilt = np.where(home, ch - ca, ca - ch)
    stored = d["score_diff"].values.astype(np.int64)
    # Points no possession carries (defensive technical free throws) shift the
    # rebuilt margin; the shift is exact on non-scoring possessions. Between
    # two anchors the earlier offset holds while it respects the pbpstats
    # bounds; the later one applies from the first possession where it does not.
    anchor = np.where(pts == 0, stored - rebuilt, np.nan)
    before = pd.Series(anchor).groupby(keys).ffill().values
    after = pd.Series(anchor).groupby(keys).bfill().values
    before = np.nan_to_num(np.where(np.isnan(before), after, before))
    after = np.nan_to_num(np.where(np.isnan(after), before, after))
    lo, hi = stored - pts, stored
    cand1, cand2 = rebuilt + before, rebuilt + after
    ok1 = (cand1 >= lo) & (cand1 <= hi)
    ok2 = (cand2 >= lo) & (cand2 <= hi)
    run = pd.factorize(pd.Series(keys).astype(str) + "_" + pd.Series(np.cumsum(pts == 0)).astype(str))[0]
    broke = pd.Series(~ok1).groupby(run).cummax().values
    fixed = np.where(broke & ok2, cand2, np.where(ok1, cand1, np.clip(cand1, lo, hi)))
    out = np.empty(len(df), dtype=np.int64)
    out[order] = fixed.astype(np.int64)
    return out


def reparsed_score_diff(gid: str, rows: pd.DataFrame) -> np.ndarray | None:
    """Exact margins from an offline re-parse of the cached pbpstats play-by-play,
    aligned to ``rows`` (one game, any order); None if no cache or the rows differ."""
    if not (PBP_CACHE / "pbp" / f"data_{gid}.json").exists():
        return None
    teams = set(int(t) for t in rows["offense_team_id"].unique())
    home = int(rows["home_team_id"].iloc[0])
    away = next((t for t in teams if t != home), None)
    if away is None:
        return None
    meta = {"game_date": str(rows["game_date"].iloc[0]), "home_team_id": home, "away_team_id": away}
    try:
        new = pd.DataFrame(get_game_possessions(gid, pbp_cache_dir=PBP_CACHE, meta=meta))
    except Exception:
        return None
    if len(new) != len(rows):
        return None
    o_old = np.lexsort((np.arange(len(rows)), rows["seconds_into_game"].values))
    o_new = np.lexsort((np.arange(len(new)), new["seconds_into_game"].values))
    old_s, new_s = rows.iloc[o_old], new.iloc[o_new]
    for c in KEY_COLS:
        if not np.array_equal(old_s[c].values.astype(float), new_s[c].values.astype(float)):
            return None
    for c in ("offense_player_ids", "defense_player_ids"):
        if any(list(a) != list(b) for a, b in zip(old_s[c], new_s[c])):
            return None
    out = np.empty(len(rows), dtype=np.int64)
    out[o_old] = new_s["score_diff"].values.astype(np.int64)
    return out


def repair_frame(df: pd.DataFrame, verbose_name: str = "") -> tuple[np.ndarray, dict]:
    """New score_diff for every row plus a report; untouched games keep their values."""
    cls = classify_games(df)
    new = df["score_diff"].values.astype(np.int64).copy()
    affected = [g for g, c in cls.items() if c != "START"]
    rebuilt_all = rebuilt_score_diff(df) if affected else None
    n_reparsed = n_rebuilt = 0
    pos = {g: np.flatnonzero(df["game_id"].values == g) for g in affected}
    for g in affected:
        idx = pos[g]
        rp = reparsed_score_diff(g, df.iloc[idx])
        if rp is not None:
            new[idx] = rp
            n_reparsed += 1
        else:
            new[idx] = rebuilt_all[idx]
            n_rebuilt += 1
    diff = new - df["score_diff"].values
    return new, {"games": len(cls), "untouched": int((cls == "START").sum()), "reparsed": n_reparsed,
                 "rebuilt": n_rebuilt, "poss_changed_pct": round(100 * float((diff != 0).mean()), 2),
                 "mean_shift": round(float(diff.mean()), 3)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--apply", action="store_true")
    p.add_argument("--seasons", nargs="+", default=None)
    args = p.parse_args()
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%dT%H%M%S}"
    backup = Path("outputs/archive") / f"score_diff_fix_{stamp}"
    t0 = time.time()

    rows, n_cache, n_cache_skipped = [], 0, 0
    for pq in sorted(DATA.glob("20*/possessions_*.parquet")):
        if args.seasons and pq.parent.name not in args.seasons:
            continue
        df = pd.read_parquet(pq)
        new, rep = repair_frame(df)
        rows.append({"file": f"{pq.parent.name}/{pq.stem.split('_', 3)[-1]}", **rep})
        changed = new != df["score_diff"].values
        if args.apply and changed.any():
            (backup / pq.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(pq, backup / pq.parent.name / pq.name)
            df["score_diff"] = new
            df.to_parquet(pq, index=False)
            csv = pq.with_suffix(".csv")
            if csv.exists():
                df.to_csv(csv, index=False)
            games = sorted(df.loc[changed, "game_id"].astype(str).unique())
            files = [DATA / "games" / f"possessions_{g}.parquet" for g in games]
            files = [f for f in files if f.exists()]
            if files:
                with tarfile.open(backup / pq.parent.name / f"{pq.stem}_games_cache.tar.gz", "w:gz") as tar:
                    for f in files:
                        tar.add(f, arcname=f.name)
            by_game = df.groupby(df["game_id"].astype(str)).indices
            for f in files:
                gid = f.stem.replace("possessions_", "")
                cache = pd.read_parquet(f)
                fresh = df.iloc[by_game[gid]]
                if len(cache) != len(fresh) or not set(cache.columns) <= set(fresh.columns):
                    n_cache_skipped += 1       # cache and season rows disagree; leave the cache alone
                    continue
                fresh[list(cache.columns)].to_parquet(f, index=False)
                n_cache += 1
        print(f"  {rows[-1]['file']:<40} {rep}  ({time.time() - t0:.0f}s)", flush=True)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))
    if args.apply:
        print(f"\ndata/games: {n_cache} cache files rewritten from the repaired season rows, "
              f"{n_cache_skipped} skipped (row count or columns differ); backups in {backup}  "
              f"({time.time() - t0:.0f}s)")
    else:
        print("\n(check only — rerun with --apply to write)")


if __name__ == "__main__":
    main()
