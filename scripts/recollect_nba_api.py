"""Re-collect seasons through the nba_api extractor, in three stages.

    python3.11 scripts/recollect_nba_api.py fetch --seasons 2025-26 2026-27
    python3.11 scripts/recollect_nba_api.py parse --seasons 2025-26 2026-27
    python3.11 scripts/recollect_nba_api.py swap  --seasons 2025-26 2026-27

    python3.11 scripts/recollect_nba_api.py fetch --mode repair --seasons 2022-23 2023-24

--mode full    (default) every final game in the official schedule.
--mode repair  only games missing from the season parquets or flagged
               severity=error in data/quality/game_quality.parquet
               (run scripts/validate_data.py first); clean games are kept.

fetch  Enumerates final games from the official schedule (LeagueGameLog:
       regular season, playoffs, play-in, preseason) and caches raw
       PlayByPlayV3 + box score (+ GameRotation with --with-rotation) under
       data/raw/nba_api/. The only stage that calls stats.nba.com; resumable,
       paced, backs off when throttled.
parse  Offline. Parses the target games into data/games_v2/ and writes
       data/games_v2/manifest_<season>.parquet (per game: status ok /
       partial / incomplete / no_lineups / no_raw, lineup source,
       possessions kept and dropped, points vs the official final, inferred
       minutes vs box-score minutes). "partial": > 5% of possessions dropped
       for an undeterminable lineup but ≥ 50% kept.
swap   Rebuilds each season's possession parquets: staged games replace
       their old rows; target games that could not be re-collected keep their
       old rows minus possessions with a duplicated player (the fallback);
       everything else is untouched. Backs up every file it replaces
       (*.bak.<timestamp>) and refreshes data/games/.
"""
import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from deep_rapm.data.game_nba_api import (MIN_KEPT_SHARE, RAW_DIR, fetch_raw, game_stints,
                                         parse_possessions, rotation_ok)

STAGE = Path("data/games_v2")
KIND_BY_CHAR = {"1": "preseason", "2": "regular_season", "4": "playoffs", "5": "playoffs"}
SEASON_TYPES = ["Regular Season", "Playoffs", "PlayIn", "Pre Season"]


def schedule(season: str, refresh: bool = False) -> pd.DataFrame:
    """Final games for *season*: game_id, game_date, home_team_id, away_team_id, kind."""
    from nba_api.stats.endpoints import leaguegamelog

    path = RAW_DIR / f"schedule_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    frames = []
    for st in SEASON_TYPES:
        for attempt in range(3):
            try:
                time.sleep(2 + 20 * attempt)
                frames.append(leaguegamelog.LeagueGameLog(
                    season=season, season_type_all_star=st, timeout=60).get_data_frames()[0])
                break
            except Exception:
                if attempt == 2:
                    print(f"  could not fetch {season} {st} schedule")
    logs = pd.concat([f for f in frames if len(f)], ignore_index=True)
    logs = logs[logs["WL"].notna()]
    logs["GAME_ID"] = logs["GAME_ID"].astype(str)
    # Neutral-site games list both teams as "@" (or both "vs."), so this is
    # only a guess; parse_stage confirms home/away from the play-by-play.
    logs["is_home"] = logs["MATCHUP"].str.contains(" vs. ")
    rows = []
    for gid, g in logs.groupby("GAME_ID"):
        if len(g) != 2:
            continue
        g = g.sort_values("is_home", ascending=False)
        rows.append({"game_id": gid, "game_date": str(g["GAME_DATE"].iloc[0])[:10],
                     "home_team_id": int(g["TEAM_ID"].iloc[0]),
                     "away_team_id": int(g["TEAM_ID"].iloc[1])})
    sched = pd.DataFrame(rows)
    sched["kind"] = sched["game_id"].str[2].map(KIND_BY_CHAR)
    sched["season"] = season
    path.parent.mkdir(parents=True, exist_ok=True)
    sched.to_parquet(path, index=False)
    return sched


def _home_from_actions(actions: list[dict], teams: set[int]) -> int | None:
    """The team whose made field goal first moves scoreHome."""
    prev_home = 0
    for a in actions:
        sh = a.get("scoreHome")
        if not sh:
            continue
        try:
            sh = int(sh)
        except ValueError:
            continue
        team = int(a.get("teamId") or 0)
        if sh != prev_home and a.get("actionType") == "Made Shot" and team in teams:
            return team
        if a.get("actionType") == "Made Shot" and team in teams and sh == prev_home:
            return (teams - {team}).pop()
        prev_home = sh
    return None


def targets(season: str, mode: str) -> pd.DataFrame:
    """Schedule rows to (re-)collect for *season* under *mode*."""
    sched = schedule(season)
    if mode == "full":
        return sched
    slug = season.replace("-", "_")
    have: set[str] = set()
    for kind in ("regular_season", "playoffs", "preseason"):
        p = Path("data") / season / f"possessions_{slug}_{kind}.parquet"
        if p.exists():
            have |= set(pd.read_parquet(p, columns=["game_id"])["game_id"])
    q = pd.read_parquet(Path("data/quality/game_quality.parquet"),
                        columns=["game_id", "season", "severity"])
    flagged = set(q.loc[(q.season == season) & (q.severity == "error"), "game_id"])
    return sched[~sched["game_id"].isin(have) | sched["game_id"].isin(flagged)]


def _raw_complete(gid: str, with_rotation: bool) -> bool:
    import gzip
    import json
    p = RAW_DIR / f"{gid}.json.gz"
    if not p.exists():
        return False
    try:
        with gzip.open(p, "rt") as fh:
            rec = json.load(fh)
    except (OSError, EOFError, json.JSONDecodeError):
        return False
    return bool(rec.get("box")) and (rotation_ok(rec) or not with_rotation)


def cmd_fetch(seasons: list[str], refresh_schedule: bool, mode: str,
              with_rotation: bool, passes: int = 3) -> None:
    for season in seasons:
        schedule(season, refresh=refresh_schedule)
        sched = targets(season, mode)
        order = {"regular_season": 0, "playoffs": 1, "preseason": 2}
        sched = sched.sort_values(["kind", "game_id"], key=lambda s: s.map(order)
                                  if s.name == "kind" else s)
        print(f"{season}: {len(sched)} target games ({mode})", flush=True)
        for sweep in range(1, passes + 1):
            todo = [g for g in sched["game_id"] if not _raw_complete(g, with_rotation)]
            if not todo:
                break
            print(f"  sweep {sweep}: {len(todo)} games to fetch", flush=True)
            fails = consecutive = 0
            for i, gid in enumerate(todo, 1):
                # When throttled, stats.nba.com hangs rather than refusing, so a
                # short timeout plus an early pause beats retrying in place.
                try:
                    fetch_raw(gid, max_retries=2, base_sleep=1.5, timeout=20,
                              rotation_retries=1 if with_rotation else 0, rotation_timeout=10)
                    ok = _raw_complete(gid, with_rotation)
                except Exception:
                    ok = False
                fails += not ok
                consecutive = 0 if ok else consecutive + 1
                if i % 100 == 0 or i == len(todo):
                    print(f"    [{i}/{len(todo)}] {fails} incomplete so far "
                          f"({time.strftime('%H:%M')})", flush=True)
                if consecutive >= 3:
                    print(f"    3 straight failures — throttled; pausing 5 min "
                          f"({time.strftime('%H:%M')})", flush=True)
                    time.sleep(300)
                    consecutive = 0
            if sweep < passes:
                time.sleep(60)
        left = sum(not _raw_complete(g, with_rotation) for g in sched["game_id"])
        print(f"{season}: done, {left} games incomplete", flush=True)


def cmd_parse(seasons: list[str], mode: str) -> None:
    import gzip
    import json

    STAGE.mkdir(parents=True, exist_ok=True)
    manifests = []
    for season in seasons:
        rows = []
        sched = targets(season, mode)
        for g in sched.itertuples():
            raw = RAW_DIR / f"{g.game_id}.json.gz"
            row = {"game_id": g.game_id, "season": season, "kind": g.kind, "status": "no_raw"}
            if raw.exists():
                try:
                    with gzip.open(raw, "rt") as fh:
                        rec = json.load(fh)
                except (OSError, EOFError, json.JSONDecodeError):
                    row["status"] = "unreadable"
                    rows.append(row)
                    continue
                stints, source, inf = game_stints(rec)
                home = _home_from_actions(rec["actions"], {g.home_team_id, g.away_team_id})
                home = home or g.home_team_id
                away = g.away_team_id if home == g.home_team_id else g.home_team_id
                meta = {"game_date": g.game_date, "home_team_id": home, "away_team_id": away}
                if not stints:
                    row["status"] = "no_lineups"
                else:
                    poss, info = parse_possessions(g.game_id, meta, rec["actions"], stints)
                    final = next((a for a in reversed(rec["actions"]) if a.get("scoreHome")), None)
                    df = pd.DataFrame(poss)
                    pts = df.groupby("offense_team_id")["points"].sum() if len(df) else pd.Series()
                    off = (abs(pts.get(home, 0) - int(final["scoreHome"]))
                           + abs(pts.get(away, 0) - int(final["scoreAway"]))) if final else None
                    skipped = info["skipped_lineup"]
                    total = len(poss) + skipped
                    if len(poss) < 100 or len(poss) < MIN_KEPT_SHARE * total:
                        status = "incomplete"
                    elif skipped > max(5, 0.05 * total):
                        status = "partial"
                    else:
                        status = "ok"
                    row.update(status=status, lineup_source=source, n_poss=len(poss),
                               skipped=skipped, points_off=off,
                               self_corrections=info["self_corrections"],
                               minutes_max_err=inf.get("minutes_max_err"),
                               unresolved_subs=inf.get("unresolved_subs"),
                               uncertain_periods=inf.get("uncertain_periods"))
                    if status != "incomplete":
                        df.to_parquet(STAGE / f"possessions_{g.game_id}.parquet", index=False)
            rows.append(row)
        season_man = pd.DataFrame(rows).assign(mode=mode)
        season_man.to_parquet(STAGE / f"manifest_{season}.parquet", index=False)
        manifests.append(season_man)
    man = pd.concat(manifests, ignore_index=True)
    print(man.groupby(["season", "kind", "status"]).size().unstack(fill_value=0).to_string())
    ok = man[man["status"].isin(["ok", "partial"])]
    if len(ok):
        kept = ok.n_poss.sum() / (ok.n_poss.sum() + ok.skipped.sum())
        print(f"\nparsed games: median {ok.n_poss.median():.0f} possessions, {kept:.1%} of "
              f"possessions kept; lineups from {ok.lineup_source.value_counts().to_dict()}; "
              f"points off official final: 0 in {(ok.points_off == 0).mean():.0%}, "
              f"≤2 in {(ok.points_off <= 2).mean():.0%}")


def cmd_swap(seasons: list[str]) -> None:
    man = pd.concat([pd.read_parquet(STAGE / f"manifest_{s}.parquet") for s in seasons],
                    ignore_index=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    old_cache = Path("data/games_fallback_old")
    old_cache.mkdir(exist_ok=True)
    for season in seasons:
        slug = season.replace("-", "_")
        for kind in ("regular_season", "playoffs", "preseason"):
            m = man[(man.season == season) & (man.kind == kind)]
            if m.empty:
                continue
            path = Path("data") / season / f"possessions_{slug}_{kind}.parquet"
            old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            new_ids = set(m.loc[m.status.isin(["ok", "partial"]), "game_id"])
            parts = [pd.read_parquet(STAGE / f"possessions_{g}.parquet") for g in sorted(new_ids)]
            kept_old = 0
            if len(old):
                rest = old[~old["game_id"].isin(new_ids)]
                if len(rest):
                    lu = lambda c: np.sort(np.vstack(rest[c].values), axis=1)
                    dup = ((np.diff(lu("offense_player_ids"), axis=1) == 0).any(axis=1)
                           | (np.diff(lu("defense_player_ids"), axis=1) == 0).any(axis=1))
                    parts.append(rest[~dup])
                    kept_old = rest["game_id"].nunique()
                shutil.copy2(path, path.with_name(path.name + f".bak.{stamp}"))
            combined = pd.concat(parts, ignore_index=True)
            combined = combined.sort_values(["game_id", "seconds_into_game"],
                                            kind="stable").reset_index(drop=True)
            combined.to_parquet(path, index=False)
            csv = path.with_suffix(".csv")
            if csv.exists():
                shutil.copy2(csv, csv.with_name(csv.name + f".bak.{stamp}"))
                combined.to_csv(csv, index=False)
            print(f"{path}: {len(new_ids)} re-collected games + {kept_old} old games "
                  f"(flagged possessions dropped) → {len(combined):,} possessions")
            for gid in new_ids:
                cached = Path("data/games") / f"possessions_{gid}.parquet"
                if cached.exists():
                    shutil.move(str(cached), old_cache / cached.name)
                shutil.copy2(STAGE / f"possessions_{gid}.parquet", cached)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=["fetch", "parse", "swap"])
    p.add_argument("--seasons", nargs="+", required=True)
    p.add_argument("--mode", choices=["full", "repair"], default="full",
                   help="full: every final game; repair: only missing or flagged games.")
    p.add_argument("--refresh-schedule", action="store_true",
                   help="Refetch the schedule (picks up newly finished games).")
    p.add_argument("--with-rotation", action="store_true",
                   help="Also request GameRotation (slow and flaky; lineups are otherwise "
                        "inferred from play-by-play + box score).")
    args = p.parse_args()
    {"fetch": lambda: cmd_fetch(args.seasons, args.refresh_schedule, args.mode,
                                args.with_rotation),
     "parse": lambda: cmd_parse(args.seasons, args.mode),
     "swap":  lambda: cmd_swap(args.seasons)}[args.stage]()


if __name__ == "__main__":
    main()
