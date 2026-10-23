"""
validate.py — corpus-wide data-quality checks for possession parquets.

Every game in every season parquet gets one row in a quality table with
metrics, a list of errors, a list of warnings and a severity
("ok" / "warn" / "error"). Nothing here modifies data.

Three layers of checks:

* **Structural** (offline, per possession and per game): lineup size,
  duplicated or overlapping player ids, a player appearing for both teams,
  team/period/time consistency, points range, score_diff reconstruction,
  possession alternation, possession counts.
* **Provenance heuristics** (offline): the pre-2026-10-21 nba_api fallback
  never changed lineups at period breaks, so a game with zero lineup changes
  across every period boundary is flagged as a likely fallback game.
* **Reconciliation** (needs official box scores; see ``fetch_box_logs``):
  team points per game vs the official final, and each player's implied
  minutes (share of the game's possessions on court × game length) vs the
  official minutes. The minutes check is the strongest lineup test available
  without event-level data: a player wrongly left on the floor shows up as
  tens of extra minutes.

Usage
-----
    from deep_rapm.data.validate import validate_corpus
    quality = validate_corpus(Path("data"), box_dir=Path("data/quality/boxscores"))
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

#: Per-game thresholds. Errors make a game unusable as-is; warnings are
#: worth a look but are common in clean data (e.g. odd endings, rare OTs).
THRESHOLDS = {
    "team_share": (0.40, 0.60),      # share of possessions per team (warn)
    "poss_per_48": (170, 260),       # possessions per 48 minutes (warn)
    "max_gap_s": 120.0,              # longest within-period gap (warn)
    "non_alternation": 0.08,         # share of same-team consecutive poss (warn)
    "score_recon_tol": 8,            # score_diff vs running points (warn)
    "box_points_err": 6,             # |team points − official| (error)
    "box_points_warn": 2,            # (warn)
    "minutes_max_err": 10.0,         # worst player |implied − official| min (error)
    "minutes_mae": 3.0,              # mean abs minutes error (error)
    "minutes_max_warn": 6.0,         # (warn)
}

_KINDS = ("regular_season", "playoffs", "preseason")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_corpus(data_dir: Path, seasons: Optional[list[str]] = None) -> pd.DataFrame:
    """All season parquets, tagged with ``season`` and ``kind``, sorted by
    (game_id, seconds_into_game)."""
    frames = []
    for season_dir in sorted(Path(data_dir).iterdir()):
        if not season_dir.is_dir() or (seasons and season_dir.name not in seasons):
            continue
        slug = season_dir.name.replace("-", "_")
        for kind in _KINDS:
            p = season_dir / f"possessions_{slug}_{kind}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df["season"] = season_dir.name
                df["kind"] = kind
                frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["game_id", "seconds_into_game"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lineup_matrix(col: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(n, 5) int64 matrix of sorted ids and a bool mask of rows that had
    exactly five entries (other rows are zero-padded / truncated)."""
    lens = col.map(len).to_numpy()
    ok = lens == 5
    mat = np.zeros((len(col), 5), dtype=np.int64)
    if ok.all():
        mat[:] = np.vstack(col.to_numpy())
    else:
        for i, ids in enumerate(col.to_numpy()):
            k = min(len(ids), 5)
            mat[i, :k] = np.asarray(ids[:k], dtype=np.int64)
    return np.sort(mat, axis=1), ok


def _lineup_hash(mat: np.ndarray) -> np.ndarray:
    """Order-free 64-bit hash of each (sorted) lineup row, for equality tests."""
    weights = np.array([1, 1_000_003, 998_244_353, 1_000_000_007, 2_147_483_647],
                       dtype=np.uint64)
    return (mat.astype(np.uint64) * weights).sum(axis=1)


def _game_length_min(max_period: pd.Series) -> pd.Series:
    return 48 + 5 * (max_period - 4).clip(lower=0)


# ---------------------------------------------------------------------------
# Structural + provenance checks
# ---------------------------------------------------------------------------

def structural_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game of offline metrics."""
    gid = df["game_id"].to_numpy()
    codes, uniq = pd.factorize(gid, sort=False)

    off, off_ok = _lineup_matrix(df["offense_player_ids"])
    dfn, def_ok = _lineup_matrix(df["defense_player_ids"])
    dup = (np.diff(off, axis=1) == 0).any(axis=1) | (np.diff(dfn, axis=1) == 0).any(axis=1)
    overlap = (off[:, :, None] == dfn[:, None, :]).any(axis=(1, 2))

    is_home_off = (df["offense_team_id"] == df["home_team_id"]).to_numpy()
    home_lu = np.where(is_home_off[:, None], off, dfn)
    away_lu = np.where(is_home_off[:, None], dfn, off)

    row = pd.DataFrame({
        "game_id": gid,
        "bad_size": ~(off_ok & def_ok),
        "dup": dup,
        "overlap": overlap,
        "same_team": (df["offense_team_id"] == df["defense_team_id"]).to_numpy(),
        "bad_points": ((df["points"] < 0) | (df["points"] > 8)).to_numpy(),
        "home_off": is_home_off,
        "points": df["points"].to_numpy(),
        "period": df["period"].to_numpy(),
        "t": df["seconds_into_game"].to_numpy(),
        "score_diff": df["score_diff"].to_numpy(),
        "home_hash": _lineup_hash(home_lu),
        "away_hash": _lineup_hash(away_lu),
    })

    # score_diff reconstruction from running points
    home_pts = np.where(is_home_off, row["points"], 0)
    away_pts = np.where(is_home_off, 0, row["points"])
    row["home_before"] = pd.Series(home_pts).groupby(codes).cumsum().to_numpy() - home_pts
    row["away_before"] = pd.Series(away_pts).groupby(codes).cumsum().to_numpy() - away_pts
    expected = np.where(is_home_off, row["home_before"] - row["away_before"],
                        row["away_before"] - row["home_before"])
    row["recon_err"] = np.abs(row["score_diff"].to_numpy() - expected)

    # same team on offense twice in a row within a period
    prev_same = (
        (codes[1:] == codes[:-1])
        & (row["period"].to_numpy()[1:] == row["period"].to_numpy()[:-1])
        & (is_home_off[1:] == is_home_off[:-1])
    )
    row["repeat_off"] = np.concatenate([[False], prev_same])

    # within-period gaps
    t = row["t"].to_numpy()
    same_period = np.concatenate([[False], (codes[1:] == codes[:-1])
                                  & (row["period"].to_numpy()[1:] == row["period"].to_numpy()[:-1])])
    gap = np.where(same_period, np.concatenate([[0.0], np.diff(t)]), 0.0)
    row["gap"] = gap

    g = row.groupby("game_id", sort=False)
    out = pd.DataFrame({
        "n_poss": g.size(),
        "bad_size_rows": g["bad_size"].sum(),
        "dup_rows": g["dup"].sum(),
        "overlap_rows": g["overlap"].sum(),
        "same_team_rows": g["same_team"].sum(),
        "bad_points_rows": g["bad_points"].sum(),
        "home_share": g["home_off"].mean(),
        "home_points": g["points"].sum() - row.assign(p=away_pts).groupby("game_id", sort=False)["p"].sum(),
        "away_points": row.assign(p=away_pts).groupby("game_id", sort=False)["p"].sum(),
        "max_period": g["period"].max(),
        "n_reg_periods": g["period"].agg(lambda s: len({1, 2, 3, 4} & set(s.unique()))),
        "first_score_diff": g["score_diff"].first(),
        "max_recon_err": g["recon_err"].max(),
        "non_alternation": g["repeat_off"].mean(),
        "max_gap_s": g["gap"].max(),
    })

    # lineup changes across period boundaries (fallback signature)
    per = row.groupby(["game_id", "period"], sort=False).agg(
        h_first=("home_hash", "first"), h_last=("home_hash", "last"),
        a_first=("away_hash", "first"), a_last=("away_hash", "last"))
    per = per.reset_index().sort_values(["game_id", "period"])
    nxt = per.groupby("game_id")[["h_first", "a_first"]].shift(-1)
    has_next = nxt["h_first"].notna()
    changes = ((per["h_last"] != nxt["h_first"]) | (per["a_last"] != nxt["a_first"])) & has_next
    bnd = pd.DataFrame({"game_id": per["game_id"], "chg": changes, "bnd": has_next})
    bnd = bnd.groupby("game_id").agg(boundaries=("bnd", "sum"), boundary_changes=("chg", "sum"))
    out = out.join(bnd)

    # a player listed for both teams in the same game
    keys = []
    for lu in (home_lu, away_lu):
        k = (codes.astype(np.int64)[:, None] << 32) | (lu & 0xFFFFFFFF)
        keys.append(np.unique(k[lu > 0]))
    both = np.intersect1d(keys[0], keys[1])
    both_counts = pd.Series(uniq[(both >> 32).astype(np.int64)]).value_counts()
    out["players_both_teams"] = out.index.map(both_counts).fillna(0).astype(int)

    meta = df.groupby("game_id", sort=False).agg(
        season=("season", "first"), kind=("kind", "first"), game_date=("game_date", "first"),
        home_team_id=("home_team_id", "first"))
    return meta.join(out).reset_index()


# ---------------------------------------------------------------------------
# Reconciliation against official box scores
# ---------------------------------------------------------------------------

_SEASON_TYPES = {"regular_season": "Regular Season", "playoffs": "Playoffs",
                 "preseason": "Pre Season"}


def fetch_box_logs(
    seasons_kinds: list[tuple[str, str]],
    box_dir: Path,
    needed_game_ids: Optional[set[str]] = None,
    sleep_s: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Official team and player game logs, cached per (season, kind).

    A cached file is refetched only if it lacks some of *needed_game_ids*
    (e.g. new games in the current season). Play-in games are fetched with
    the playoffs. Returns (team_logs, player_logs) with columns
    game_id, team_id, pts  /  game_id, team_id, player_id, minutes.
    """
    from nba_api.stats.endpoints import leaguegamelog

    box_dir = Path(box_dir)
    box_dir.mkdir(parents=True, exist_ok=True)
    teams, players = [], []
    for season, kind in seasons_kinds:
        types = [_SEASON_TYPES[kind]] + (["PlayIn"] if kind == "playoffs" else [])
        for side in ("T", "P"):
            path = box_dir / f"{season}_{kind}_{side}.parquet"
            cached = pd.read_parquet(path) if path.exists() else None
            if cached is not None and needed_game_ids is not None:
                want = {g for g in needed_game_ids if g[3:5] == season[2:4]
                        and _kind_of(g) == kind}
                if not want - set(cached["GAME_ID"]):
                    (teams if side == "T" else players).append(cached)
                    continue
            elif cached is not None:
                (teams if side == "T" else players).append(cached)
                continue
            parts = []
            for st in types:
                for attempt in range(3):
                    try:
                        time.sleep(sleep_s * (1 + 5 * attempt))
                        lg = leaguegamelog.LeagueGameLog(
                            season=season, season_type_all_star=st,
                            player_or_team_abbreviation=side, timeout=60)
                        parts.append(lg.get_data_frames()[0])
                        break
                    except Exception:
                        if attempt == 2 and st != "PlayIn":
                            print(f"  could not fetch {season} {st} {side}; skipping")
            if parts:
                logs = pd.concat(parts, ignore_index=True)
                logs["GAME_ID"] = logs["GAME_ID"].astype(str)
                logs.to_parquet(path, index=False)
                (teams if side == "T" else players).append(logs)

    team = pd.concat(teams, ignore_index=True) if teams else pd.DataFrame()
    player = pd.concat(players, ignore_index=True) if players else pd.DataFrame()
    team = team.rename(columns={"GAME_ID": "game_id", "TEAM_ID": "team_id", "PTS": "pts"})
    player = player.rename(columns={"GAME_ID": "game_id", "TEAM_ID": "team_id",
                                    "PLAYER_ID": "player_id", "MIN": "minutes"})
    team = team[["game_id", "team_id", "pts"]] if len(team) else team
    player = player[["game_id", "team_id", "player_id", "minutes"]] if len(player) else player
    return team, player


def _kind_of(game_id: str) -> str:
    return {"1": "preseason", "2": "regular_season", "4": "playoffs",
            "5": "playoffs"}.get(game_id[2], "other")


def reconciliation_metrics(
    df: pd.DataFrame,
    team_logs: pd.DataFrame,
    player_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Per-game points error vs official finals and implied-minutes error vs
    official player minutes."""
    # points
    pts = df.groupby(["game_id", "offense_team_id"])["points"].sum().rename("our_pts")
    pts = pts.reset_index().rename(columns={"offense_team_id": "team_id"})
    pts = pts.merge(team_logs, on=["game_id", "team_id"], how="left")
    pts["err"] = (pts["our_pts"] - pts["pts"]).abs()
    p_game = pts.groupby("game_id").agg(box_points_err=("err", "max"),
                                        box_found=("pts", lambda s: s.notna().all()))

    # minutes: share of the game's possessions on court × game length
    n_game = df.groupby("game_id").size().rename("n_game")
    length = _game_length_min(df.groupby("game_id")["period"].max()).rename("length")
    long = []
    for ids_col, team_col in (("offense_player_ids", "offense_team_id"),
                              ("defense_player_ids", "defense_team_id")):
        e = df[["game_id", team_col, ids_col]].explode(ids_col)
        e.columns = ["game_id", "team_id", "player_id"]
        long.append(e)
    on = pd.concat(long, ignore_index=True)
    on["player_id"] = on["player_id"].astype(np.int64)
    on = on.groupby(["game_id", "team_id", "player_id"]).size().rename("poss").reset_index()
    on = on.join(n_game, on="game_id").join(length, on="game_id")
    on["implied_min"] = on["poss"] / on["n_game"] * on["length"]

    box = player_logs.copy()
    box["minutes"] = pd.to_numeric(box["minutes"], errors="coerce").fillna(0.0)
    m = on.merge(box, on=["game_id", "team_id", "player_id"], how="outer")
    m["implied_min"] = m["implied_min"].fillna(0.0)
    m["minutes"] = m["minutes"].fillna(0.0)
    m = m[m["game_id"].isin(set(df["game_id"]))]
    m["err"] = (m["implied_min"] - m["minutes"]).abs()
    have_box = set(player_logs["game_id"])
    m = m[m["game_id"].isin(have_box)]
    m_game = m.groupby("game_id").agg(minutes_max_err=("err", "max"),
                                      minutes_mae=("err", "mean"))
    return p_game.join(m_game, how="outer").reset_index()


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def _verdict(r: pd.Series) -> tuple[list[str], list[str]]:
    T = THRESHOLDS
    err, warn = [], []
    if r.bad_size_rows: err.append(f"{r.bad_size_rows} lineups without 5 players")
    if r.dup_rows: err.append(f"{r.dup_rows} lineups with a duplicated player")
    if r.overlap_rows: err.append(f"{r.overlap_rows} possessions with a player on both sides")
    if r.players_both_teams: err.append(f"{r.players_both_teams} players listed for both teams")
    if r.same_team_rows: err.append(f"{r.same_team_rows} possessions with offense == defense")
    if r.bad_points_rows: err.append(f"{r.bad_points_rows} possessions with points outside 0–8")
    if r.n_reg_periods < 4: err.append(f"only {r.n_reg_periods} of 4 regulation periods")

    lo, hi = T["team_share"]
    if not lo <= r.home_share <= hi: warn.append(f"home share of possessions {r.home_share:.0%}")
    per48 = r.n_poss / _game_length_min(pd.Series([r.max_period])).iloc[0] * 48
    lo, hi = T["poss_per_48"]
    if not lo <= per48 <= hi: warn.append(f"{per48:.0f} possessions per 48 min")
    if r.max_gap_s > T["max_gap_s"]: warn.append(f"{r.max_gap_s:.0f}s gap within a period")
    if r.non_alternation > T["non_alternation"]:
        warn.append(f"{r.non_alternation:.0%} same-team consecutive possessions")
    if r.max_recon_err > T["score_recon_tol"]:
        warn.append(f"score_diff off running points by {r.max_recon_err}")
    if r.boundaries >= 3 and r.boundary_changes == 0:
        warn.append("no lineup change at any period break (old-fallback signature)")

    if pd.notna(r.get("box_points_err")):
        if r.box_points_err > T["box_points_err"]:
            err.append(f"team points off official final by {r.box_points_err:.0f}")
        elif r.box_points_err > T["box_points_warn"]:
            warn.append(f"team points off official final by {r.box_points_err:.0f}")
    if pd.notna(r.get("minutes_max_err")):
        if r.minutes_max_err > T["minutes_max_err"] or r.minutes_mae > T["minutes_mae"]:
            err.append(f"implied minutes off box score (worst {r.minutes_max_err:.0f} min, "
                       f"mean {r.minutes_mae:.1f})")
        elif r.minutes_max_err > T["minutes_max_warn"]:
            warn.append(f"implied minutes off box score by up to {r.minutes_max_err:.0f} min")
    return err, warn


def validate_corpus(
    data_dir: Path,
    seasons: Optional[list[str]] = None,
    box_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Quality table: one row per game with metrics, errors, warnings, severity.
    Pass *box_dir* to reconcile against official box scores (API calls, cached)."""
    df = load_corpus(data_dir, seasons)
    q = structural_metrics(df)
    if box_dir is not None:
        pairs = sorted(set(zip(q["season"], q["kind"])))
        team_logs, player_logs = fetch_box_logs(pairs, box_dir, set(q["game_id"]))
        if len(team_logs) and len(player_logs):
            q = q.merge(reconciliation_metrics(df, team_logs, player_logs),
                        on="game_id", how="left")
    verdicts = q.apply(_verdict, axis=1)
    q["errors"] = verdicts.map(lambda v: v[0])
    q["warnings"] = verdicts.map(lambda v: v[1])
    q["severity"] = np.where(q["errors"].map(len) > 0, "error",
                             np.where(q["warnings"].map(len) > 0, "warn", "ok"))
    return q
