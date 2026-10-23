"""
feature_lookup.py — EWMA player feature computation from per-game box scores.

For each possession at date T, each player's feature vector is the
exponentially weighted average of their per-minute stats from all games
strictly before T:

    w(t) = exp(-γ (T − t))    where γ = ln(2) / half_life_days

Weight is proportional to both time decay AND minutes played in that game.

Feature vector layout (14 dims)
---------------------------------
    [0:5]  one-hot position           (PG, SG, SF, PF, C) — static per player
    [5]    2-point FG %               (recency-weighted EWMA)
    [6]    3-point FG %               (recency-weighted EWMA)
    [7]    2-point FGA per minute     (recency-weighted EWMA)
    [8]    3-point FGA per minute     (recency-weighted EWMA)
    [9]    rebounds per minute
    [10]   assists per minute
    [11]   steals per minute
    [12]   blocks per minute
    [13]   turnovers per minute

Stat features [5:] are z-score normalised across all players who have
prior data in the training set.  Players with no prior games receive a
zero vector for the stat features (i.e. they sit at the league mean after
normalisation).

Usage
-----
    from deep_rapm.data.feature_lookup import build_feature_arrays

    offense_feats, defense_feats = build_feature_arrays(
        possession_df=df,
        box_dir=Path("data/box"),
        player_vocab=vocab_df,
        player_table=players_df,
        half_life_days=180.0,
    )
    # offense_feats: (n_possessions, 5, 14)  float32
    # defense_feats: (n_possessions, 5, 14)  float32
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_POSITIONS  = 5
N_STAT_FEATS = 9    # 2P%, 3P%, 2PA/min, 3PA/min, REB/min, AST/min, STL/min, BLK/min, TOV/min
FEATURE_DIM  = N_POSITIONS + N_STAT_FEATS   # 14


# ---------------------------------------------------------------------------
# Per-player EWMA accumulator
# ---------------------------------------------------------------------------

class _EWMA:
    """Running exponentially weighted accumulator for one player."""

    __slots__ = ("sw", "s2pa", "s3pa", "s2m", "s3m",
                 "sreb", "sast", "sstl", "sblk", "stov", "last_date")

    def __init__(self) -> None:
        self.sw = self.s2pa = self.s3pa = self.s2m = self.s3m = 0.0
        self.sreb = self.sast = self.sstl = self.sblk = self.stov = 0.0
        self.last_date = None   # pd.Timestamp of last update

    def decay(self, to_date: pd.Timestamp, gamma: float) -> None:
        if self.last_date is None:
            return
        days = (to_date - self.last_date).days
        if days <= 0:
            return
        f = math.exp(-gamma * days)
        self.sw    *= f;  self.s2pa *= f;  self.s3pa *= f
        self.s2m   *= f;  self.s3m  *= f;  self.sreb  *= f
        self.sast  *= f;  self.sstl *= f;  self.sblk  *= f
        self.stov  *= f

    def add(self, game_date: pd.Timestamp, row_dict: dict, gamma: float) -> None:
        self.decay(game_date, gamma)
        mn = float(row_dict.get("min_played", 0))
        if mn <= 0:
            return
        w    = mn   # weight ∝ minutes in this game (time decay applied above)
        fg2a = int(row_dict.get("fga", 0)) - int(row_dict.get("fg3a", 0))
        fg3a = int(row_dict.get("fg3a", 0))
        fg2m = int(row_dict.get("fgm", 0)) - int(row_dict.get("fg3m", 0))
        fg3m = int(row_dict.get("fg3m", 0))

        self.sw    += w;  self.s2pa += w * fg2a;  self.s3pa += w * fg3a
        self.s2m   += w * fg2m;  self.s3m  += w * fg3m
        self.sreb  += w * int(row_dict.get("reb", 0))
        self.sast  += w * int(row_dict.get("ast", 0))
        self.sstl  += w * int(row_dict.get("stl", 0))
        self.sblk  += w * int(row_dict.get("blk", 0))
        self.stov  += w * int(row_dict.get("tov", 0))
        self.last_date = game_date

    def stats(self) -> np.ndarray:
        """9-dim rate vector; zeros if no data."""
        v = np.zeros(N_STAT_FEATS, dtype=np.float64)
        sw = self.sw
        if sw < 1e-9:
            return v
        v[0] = self.s2m  / max(self.s2pa, 1e-9)   # 2P%
        v[1] = self.s3m  / max(self.s3pa, 1e-9)   # 3P%
        v[2] = self.s2pa / sw                       # 2PA per minute
        v[3] = self.s3pa / sw                       # 3PA per minute
        v[4] = self.sreb / sw
        v[5] = self.sast / sw
        v[6] = self.sstl / sw
        v[7] = self.sblk / sw
        v[8] = self.stov / sw
        return v


# ---------------------------------------------------------------------------
# Position one-hot
# ---------------------------------------------------------------------------

def _pos_onehot(player_table: pd.DataFrame, player_vocab: pd.DataFrame) -> np.ndarray:
    """(n_players, 5) float32 one-hot; row 0 = UNKNOWN = all zeros."""
    n = len(player_vocab) + 1
    arr = np.zeros((n, N_POSITIONS), dtype=np.float32)
    idx_map = dict(zip(player_vocab["player_id"].astype(int),
                       player_vocab["player_idx"].astype(int)))
    latest = (player_table.sort_values("season")
                          .drop_duplicates("player_id", keep="last"))
    for _, row in latest.iterrows():
        p = idx_map.get(int(row["player_id"]))
        pos = int(row.get("position_idx", 2))
        if p and 0 <= pos < N_POSITIONS:
            arr[p, pos] = 1.0
    return arr


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_feature_arrays(
    possession_df: pd.DataFrame,
    box_dir: Path,
    player_vocab: pd.DataFrame,
    player_table: pd.DataFrame,
    half_life_days: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-possession feature arrays with EWMA recency weighting.

    Parameters
    ----------
    possession_df   : DataFrame with game_id, game_date,
                      offense_player_ids, defense_player_ids.
    box_dir         : Directory of per-game box score parquets.
    player_vocab    : player_id → player_idx mapping.
    player_table    : player metadata (for position).
    half_life_days  : Half-life γ = ln(2) / half_life_days.

    Returns
    -------
    (offense_feats, defense_feats) each (n, 5, FEATURE_DIM) float32.
    """
    gamma   = math.log(2) / half_life_days
    box_dir = Path(box_dir)

    pid_to_idx = dict(zip(player_vocab["player_id"].astype(int),
                          player_vocab["player_idx"].astype(int)))

    # Static position one-hot for every player index
    pos_arr = _pos_onehot(player_table, player_vocab)   # (n_players, 5)

    # ── Prepare possession DataFrame ──────────────────────────────────────
    df = possession_df.copy().reset_index(drop=True)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Map game_id → game_date (one date per game)
    gid_to_date: dict[str, pd.Timestamp] = (
        df.groupby("game_id")["game_date"].first().to_dict()
    )

    # Box score parquets that exist
    cached_games: set[str] = {
        p.stem for p in box_dir.glob("*.parquet")
    } if box_dir.exists() else set()

    # Chronologically sorted games that have a box score
    dated_games = sorted(
        (date, gid)
        for gid, date in gid_to_date.items()
        if gid in cached_games
    )

    # Possession indices per game (for fast snapshot lookup)
    game_to_poss: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(df.itertuples(index=False)):
        game_to_poss[row.game_id].append(i)

    # ── Allocate output ───────────────────────────────────────────────────
    n = len(df)
    off_feat = np.zeros((n, 5, FEATURE_DIM), dtype=np.float32)
    def_feat = np.zeros((n, 5, FEATURE_DIM), dtype=np.float32)

    # Pre-fill static position features from player_vocab lookup
    off_ids_arr = np.array(df["offense_player_ids"].tolist(), dtype=object)
    def_ids_arr = np.array(df["defense_player_ids"].tolist(), dtype=object)
    for slot in range(5):
        o_idx = np.array([pid_to_idx.get(int(p), 0)
                          for p in off_ids_arr[:, slot]], dtype=int)
        d_idx = np.array([pid_to_idx.get(int(p), 0)
                          for p in def_ids_arr[:, slot]], dtype=int)
        off_feat[:, slot, :N_POSITIONS] = pos_arr[o_idx]
        def_feat[:, slot, :N_POSITIONS] = pos_arr[d_idx]

    # ── Chronological EWMA loop ───────────────────────────────────────────
    ewma: dict[int, _EWMA] = {}   # player_idx → state

    print(f"  EWMA features (half_life={half_life_days:.0f}d, "
          f"{len(dated_games)} games with box scores)…")

    for game_date, game_id in dated_games:
        poss_indices = game_to_poss.get(game_id, [])

        # ── Snapshot: record current EWMA (before this game) ─────────────
        for pi in poss_indices:
            row = df.iloc[pi]
            for slot, pid in enumerate(row["offense_player_ids"]):
                p = pid_to_idx.get(int(pid), 0)
                s = ewma.get(p)
                if s is not None and s.sw > 1e-9:
                    off_feat[pi, slot, N_POSITIONS:] = s.stats()

            for slot, pid in enumerate(row["defense_player_ids"]):
                p = pid_to_idx.get(int(pid), 0)
                s = ewma.get(p)
                if s is not None and s.sw > 1e-9:
                    def_feat[pi, slot, N_POSITIONS:] = s.stats()

        # ── Update: add this game's stats to EWMA ────────────────────────
        box_path = box_dir / f"{game_id}.parquet"
        if not box_path.exists():
            continue
        box = pd.read_parquet(box_path)
        for brow in box.itertuples(index=False):
            p = pid_to_idx.get(int(brow.player_id), 0)
            if p == 0:
                continue
            if p not in ewma:
                ewma[p] = _EWMA()
            ewma[p].add(game_date, brow._asdict(), gamma)

    # ── Z-score normalise stat features [N_POSITIONS:] ───────────────────
    # Identify which (poss, slot) entries have data (stat != all-zero).
    off_stat = off_feat[:, :, N_POSITIONS:]   # (n, 5, 9), view
    def_stat = def_feat[:, :, N_POSITIONS:]

    # Flatten to (n*5, 9) to compute global stats
    both = np.concatenate([off_stat.reshape(-1, N_STAT_FEATS),
                           def_stat.reshape(-1, N_STAT_FEATS)], axis=0)
    has_data = both.any(axis=1)   # rows where at least one stat is non-zero

    if has_data.sum() > 0:
        mu  = both[has_data].mean(axis=0)    # (9,)
        std = both[has_data].std(axis=0) + 1e-8

        # Mask of which entries have stat data (before normalising)
        off_has = off_stat.any(axis=-1, keepdims=True)   # (n, 5, 1) bool
        def_has = def_stat.any(axis=-1, keepdims=True)

        # Normalise all
        off_feat[:, :, N_POSITIONS:] = (off_stat - mu) / std
        def_feat[:, :, N_POSITIONS:] = (def_stat - mu) / std

        # Reset no-data entries to 0 (league average = mean player after z-score)
        off_feat[:, :, N_POSITIONS:] *= off_has
        def_feat[:, :, N_POSITIONS:] *= def_has

    print(f"  Done — feature shape {off_feat.shape}  "
          f"({has_data.sum():,}/{len(has_data):,} slots have stat data)")

    return off_feat, def_feat
