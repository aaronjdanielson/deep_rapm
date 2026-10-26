"""As-of-origin player features for RAPM priors (docs/rapm_v2_design.md §4).

Every feature is computed from information dated *before* the origin: career
box-score totals and rates from the player game logs through the previous
day, minutes in the trailing year, age, draft slot, seasons of experience,
position, size. Nothing from the origin date onward enters.

Inputs (collected by scripts/fetch_prior_data.py):
    data/raw/player_game_logs/<season>_<kind>.parquet   PlayerGameLogs
    data/player_info.parquet                            CommonPlayerInfo snapshot
    data/players.parquet, data/player_anthropometrics.parquet
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LOGS_DIR = Path("data/raw/player_game_logs")
LOG_COLS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB",
            "AST", "TOV", "STL", "BLK", "PF", "PTS", "PLUS_MINUS"]
CAREER_KINDS = ("regular_season", "playoffs", "playin")   # preseason box lines are not career stats

FEATURES = [
    "log_games", "log_minutes", "min_per_game", "log_min_last365", "games_last365",
    "pts36", "oreb36", "dreb36", "ast36", "stl36", "blk36", "tov36", "pf36", "pm36",
    "ts_pct", "fg3a_rate", "ft_pct", "fg3_pct", "no_history",
    "age", "age_sq", "age_missing", "experience", "draft_pick", "undrafted", "draft_missing",
    "pos_g", "pos_f", "pos_c", "height_in", "weight_lb", "size_missing",
]


def load_game_logs(logs_dir: Path = LOGS_DIR, kinds: Sequence[str] = CAREER_KINDS) -> pd.DataFrame:
    frames = []
    for pq in sorted(Path(logs_dir).glob("*.parquet")):
        season, kind = pq.stem.split("_", 1)
        if kind not in kinds:
            continue
        df = pd.read_parquet(pq, columns=["PLAYER_ID", "GAME_DATE"] + LOG_COLS)
        df["kind"] = kind
        frames.append(df)
    logs = pd.concat(frames, ignore_index=True)
    logs = logs.rename(columns={"PLAYER_ID": "player_id"})
    logs["date"] = pd.to_datetime(logs["GAME_DATE"]).dt.normalize()
    logs["player_id"] = logs["player_id"].astype(int)
    for c in LOG_COLS:
        logs[c] = pd.to_numeric(logs[c], errors="coerce").fillna(0.0).astype(np.float64)
    return logs.drop(columns="GAME_DATE").sort_values("date").reset_index(drop=True)


def _position_shares(pos) -> tuple[float, float, float]:
    s = str(pos or "").upper()           # 'G', 'G-F', 'Guard-Forward', 'F-C', 'Center', 'unknown', ''
    g = s.startswith("G") or "-G" in s or "GUARD" in s
    f = "F" in s
    c = "C" in s
    flags = [g, f, c]
    n = sum(flags)
    return tuple(float(x) / n if n else 1 / 3 for x in flags)


def static_features(
    vocab: pd.DataFrame,
    players_path: Path = Path("data/players.parquet"),
    info_path: Path = Path("data/player_info.parquet"),
    anthro_path: Path = Path("data/player_anthropometrics.parquet"),
) -> pd.DataFrame:
    """Per-player time-invariant inputs: birthdate, draft, first season, position, size."""
    ids = vocab["player_id"].astype(int).values
    out = pd.DataFrame(index=pd.Index(ids, name="player_id"))

    players = pd.read_parquet(players_path)
    pos = players.groupby("player_id")["position_str"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
    info = pd.read_parquet(info_path).set_index("player_id") if Path(info_path).exists() else pd.DataFrame()
    info = info[~info.index.duplicated()] if len(info) else info

    pos_full = pos.reindex(ids)
    if len(info) and "position" in info:
        unknown = pos_full.fillna("").isin(["", "unknown"])
        pos_full = pos_full.where(~unknown, info["position"].reindex(ids))
    shares = np.array([_position_shares(p) for p in pos_full.values])
    out["pos_g"], out["pos_f"], out["pos_c"] = shares.T

    if len(info):
        bd = pd.to_datetime(info["birthdate"].reindex(ids), errors="coerce")
        out["birthdate"] = bd.values
        draft = pd.to_numeric(info["draft_number"].reindex(ids), errors="coerce")
        out["draft_pick"] = draft.values
        out["from_year"] = pd.to_numeric(info["from_year"].reindex(ids), errors="coerce").values
        out["info_fetched"] = info["fetched"].reindex(ids).astype("boolean").fillna(False).astype(bool).values
    else:
        out["birthdate"] = pd.NaT
        out["draft_pick"] = np.nan
        out["from_year"] = np.nan
        out["info_fetched"] = False

    anthro = pd.read_parquet(anthro_path).set_index("player_id")
    out["height_in"] = anthro["height_in"].reindex(ids).fillna(0.0).values
    out["weight_lb"] = anthro["weight_lb"].reindex(ids).fillna(0.0).values
    return out


def build_prior_features(
    origins: Sequence[pd.Timestamp],
    vocab: pd.DataFrame,
    logs: pd.DataFrame | None = None,
    static: pd.DataFrame | None = None,
    shrink_minutes: float = 0.0,
    verbose: bool = True,
) -> dict[pd.Timestamp, np.ndarray]:
    """Feature matrix (n_players + 1, len(FEATURES)) per origin; row 0 (UNKNOWN) is zeros.

    Rows follow ``player_idx`` so the matrix lines up with the ridge coefficients.

    ``shrink_minutes`` > 0 shrinks every rate and percentage toward the
    league value as of the origin by adding that many minutes of
    league-average play to the player's totals — a player with 7 career
    minutes then looks like the league average, not like a 29-points-per-36
    scorer. 0 reproduces the raw rates of the first v2 fit.
    """
    logs = load_game_logs() if logs is None else logs
    static = static_features(vocab) if static is None else static
    ids = vocab["player_id"].astype(int).values
    idxs = vocab["player_idx"].astype(int).values
    n_players = len(vocab) + 1
    dates = logs["date"].values
    year = pd.Timedelta(days=365)

    out: dict[pd.Timestamp, np.ndarray] = {}
    for k, t in enumerate(origins):
        t = pd.Timestamp(t)
        hist = logs.iloc[:np.searchsorted(dates, np.datetime64(t))]
        cum = hist.groupby("player_id")[LOG_COLS].sum()
        cum["games"] = hist.groupby("player_id").size()
        recent = hist.iloc[np.searchsorted(hist["date"].values, np.datetime64(t - year)):]
        rec = recent.groupby("player_id")["MIN"].agg(min_last365="sum", games_last365="size")
        c = cum.reindex(ids).fillna(0.0)
        r = rec.reindex(ids).fillna(0.0)

        minutes = c["MIN"].values
        games = c["games"].values
        has = minutes > 0
        k = float(shrink_minutes)
        if k > 0 and len(hist):
            # League totals as of the origin; k minutes of league-average play
            # are added to every player's totals before any rate is formed.
            L = hist[LOG_COLS].sum()
            L_tsa = L["FGA"] + 0.44 * L["FTA"]
            lg = {col: L[col] / max(L["MIN"], 1e-9) for col in LOG_COLS}      # per-minute league rates
            tot = {col: c[col].values + k * lg[col] for col in LOG_COLS}
            denom = minutes + k
        else:
            tot = {col: c[col].values for col in LOG_COLS}
            denom = np.maximum(minutes, 1e-9)
        per36 = lambda col: np.where(has | (k > 0), 36 * tot[col] / denom, 0.0)
        fga, fta, fg3a = tot["FGA"], tot["FTA"], tot["FG3A"]
        ts_den = 2 * (fga + 0.44 * fta)
        f = pd.DataFrame(index=ids)
        f["log_games"] = np.log1p(games)
        f["log_minutes"] = np.log1p(minutes)
        f["min_per_game"] = np.where(games > 0, minutes / np.maximum(games, 1), 0.0)
        f["log_min_last365"] = np.log1p(r["min_last365"].values)
        f["games_last365"] = r["games_last365"].values
        for name, col in [("pts36", "PTS"), ("oreb36", "OREB"), ("dreb36", "DREB"), ("ast36", "AST"),
                          ("stl36", "STL"), ("blk36", "BLK"), ("tov36", "TOV"), ("pf36", "PF"),
                          ("pm36", "PLUS_MINUS")]:
            f[name] = per36(col)
        f["ts_pct"] = np.where(ts_den > 0, tot["PTS"] / np.maximum(ts_den, 1e-9), 0.0)
        f["fg3a_rate"] = np.where(fga > 0, fg3a / np.maximum(fga, 1e-9), 0.0)
        f["ft_pct"] = np.where(fta > 0, tot["FTM"] / np.maximum(fta, 1e-9), 0.0)
        f["fg3_pct"] = np.where(fg3a > 0, tot["FG3M"] / np.maximum(fg3a, 1e-9), 0.0)
        f["no_history"] = (~has).astype(float)

        bd = pd.to_datetime(static["birthdate"].reindex(ids))
        age = ((t - bd).dt.days / 365.25).values
        f["age_missing"] = np.isnan(age).astype(float)
        age = np.where(np.isnan(age), 27.0, age)
        f["age"] = age
        f["age_sq"] = (age - 27.0) ** 2
        # Logs start in 2013-14; CommonPlayerInfo's first season extends careers that began earlier.
        first_game = hist.groupby("player_id")["date"].min().reindex(ids)
        exp_logs = ((t - first_game).dt.days / 365.25).fillna(0.0).values
        season_start_year = t.year if t.month >= 7 else t.year - 1
        exp_info = (season_start_year - static["from_year"].reindex(ids).values.astype(float))
        exp_info = np.where(np.isnan(exp_info) | (first_game.isna().values), 0.0, exp_info)
        f["experience"] = np.maximum(np.maximum(exp_logs, exp_info), 0.0)
        pick = static["draft_pick"].reindex(ids).values.astype(float)
        f["draft_missing"] = (np.isnan(pick) & ~static["info_fetched"].reindex(ids).values).astype(float)
        f["undrafted"] = (np.isnan(pick) & static["info_fetched"].reindex(ids).values).astype(float)
        f["draft_pick"] = np.where(np.isnan(pick), 61.0, pick)
        for col in ("pos_g", "pos_f", "pos_c", "height_in", "weight_lb"):
            f[col] = static[col].reindex(ids).values
        f["size_missing"] = (f["height_in"].values <= 0).astype(float)

        mat = np.zeros((n_players, len(FEATURES)), dtype=np.float32)
        mat[idxs] = f[FEATURES].values.astype(np.float32)
        out[t] = mat
        if verbose and ((k + 1) % max(1, len(origins) // 5) == 0 or k == len(origins) - 1):
            print(f"  prior features [{k + 1}/{len(origins)}] {t.date()}", flush=True)
    return out
