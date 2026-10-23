"""
dataset.py — PyTorch Dataset for Deep RAPM training.

Each item represents one possession and contains:

  offense_ids  (5,)  long   — embedding indices for offensive players
  defense_ids  (5,)  long   — embedding indices for defensive players
  offense_pos  (5,)  long   — position indices {0,...,4} for offensive players
  defense_pos  (5,)  long   — position indices {0,...,4} for defensive players
  gamestate    (1,)  float  — score_diff / score_diff_scale  (offense - defense)
  target        ()   float  — raw points scored (continuous, for NLL regression)

Players within each lineup are sorted in ascending player_id order so the
representation is canonical (the model is permutation-invariant, but a fixed
ordering makes debugging and unit tests easier).

All vocab/position lookups are resolved at construction time so __getitem__
is a pure array slice — no Python dicts or conditionals in the hot path.

Usage
-----
    from deep_rapm.data.dataset import make_possession_splits

    train_ds, val_ds = make_possession_splits(
        data_dir=Path("data"),
        seasons=["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"],
        player_vocab_path=Path("data/player_vocab.parquet"),
        player_table_path=Path("data/players.parquet"),
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    for batch in loader:
        logits = model(
            batch["offense_ids"],
            batch["defense_ids"],
            batch["offense_pos"],
            batch["defense_pos"],
            gamestate=batch["gamestate"],
        )
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .players import (
    DEFAULT_POSITION_IDX,
    UNKNOWN_PLAYER_IDX,
    load_player_table,
    load_player_vocab,
    make_player_index_lookup,
    make_position_lookup,
)


class PossessionDataset(Dataset):
    """Possession-level dataset for Deep RAPM training.

    Parameters
    ----------
    parquet_paths:
        List of (path, season_label) pairs.  Each parquet must be a
        season-level possession file produced by collect-possessions.
    player_vocab:
        DataFrame with columns player_id and player_idx.
    player_table:
        DataFrame with columns player_id, season, position_idx.
    score_diff_scale:
        Divisor applied to score_diff before storing as gamestate.
    game_ids:
        If provided, only include possessions from these game_ids.
    offense_features / defense_features:
        Optional pre-computed per-player feature arrays with shape
        (n_possessions_total, 5, feature_dim) float32.  When provided
        they are sliced to match game_ids and returned as
        ``batch["offense_features"]`` / ``batch["defense_features"]``.
    """

    def __init__(
        self,
        parquet_paths: list[tuple[Path, str]],
        player_vocab: pd.DataFrame,
        player_table: pd.DataFrame,
        score_diff_scale: float = 10.0,
        game_ids: Optional[set[str]] = None,
        offense_features: Optional["np.ndarray"] = None,
        defense_features: Optional["np.ndarray"] = None,
    ) -> None:
        self.score_diff_scale = score_diff_scale

        idx_lookup = make_player_index_lookup(player_vocab)
        pos_lookup = make_position_lookup(player_table)

        # ── Load and resolve all possessions ─────────────────────────────────
        frames: list[pd.DataFrame] = []
        for path, season in parquet_paths:
            df = pd.read_parquet(Path(path))
            df["season"] = season
            frames.append(df)

        all_df = pd.concat(frames, ignore_index=True)
        all_df["_orig_idx"] = np.arange(len(all_df))   # track for feature slicing

        if game_ids is not None:
            all_df = all_df[all_df["game_id"].isin(game_ids)].reset_index(drop=True)

        n = len(all_df)

        # Pre-allocate numpy arrays — resolved at construction, __getitem__ is O(1)
        offense_ids = np.zeros((n, 5), dtype=np.int64)
        defense_ids = np.zeros((n, 5), dtype=np.int64)
        offense_pos = np.full((n, 5), DEFAULT_POSITION_IDX, dtype=np.int64)
        defense_pos = np.full((n, 5), DEFAULT_POSITION_IDX, dtype=np.int64)
        gamestate   = np.zeros((n, 1), dtype=np.float32)
        # Target is raw points as float — consistent with the RAPM tradition of
        # treating possession points as a continuous outcome and regressing
        # E[points | lineup] directly.
        target      = np.zeros(n, dtype=np.float32)

        for i, row in enumerate(all_df.itertuples(index=False)):
            season = row.season

            off_ids = sorted(int(p) for p in row.offense_player_ids)
            def_ids = sorted(int(p) for p in row.defense_player_ids)

            for j, pid in enumerate(off_ids):
                offense_ids[i, j] = idx_lookup.get(pid, UNKNOWN_PLAYER_IDX)
                offense_pos[i, j] = pos_lookup.get((pid, season), DEFAULT_POSITION_IDX)

            for j, pid in enumerate(def_ids):
                defense_ids[i, j] = idx_lookup.get(pid, UNKNOWN_PLAYER_IDX)
                defense_pos[i, j] = pos_lookup.get((pid, season), DEFAULT_POSITION_IDX)

            gamestate[i, 0] = row.score_diff / score_diff_scale
            target[i]       = float(row.points)

        # Convert to tensors once at construction — avoids torch.from_numpy
        # bridge issues and makes __getitem__ a plain tensor index (O(1)).
        self._offense_ids = torch.tensor(offense_ids, dtype=torch.long)
        self._defense_ids = torch.tensor(defense_ids, dtype=torch.long)
        self._offense_pos = torch.tensor(offense_pos, dtype=torch.long)
        self._defense_pos = torch.tensor(defense_pos, dtype=torch.long)
        self._gamestate   = torch.tensor(gamestate,   dtype=torch.float32)
        self._target      = torch.tensor(target,      dtype=torch.float32)

        # Optional pre-computed player feature arrays (n, 5, feature_dim)
        if offense_features is not None and defense_features is not None:
            orig_idx = all_df["_orig_idx"].values
            self._offense_features = torch.tensor(
                offense_features[orig_idx], dtype=torch.float32
            )
            self._defense_features = torch.tensor(
                defense_features[orig_idx], dtype=torch.float32
            )
        else:
            self._offense_features = None
            self._defense_features = None

    def __len__(self) -> int:
        return len(self._target)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = {
            "offense_ids": self._offense_ids[idx],
            "defense_ids": self._defense_ids[idx],
            "offense_pos": self._offense_pos[idx],
            "defense_pos": self._defense_pos[idx],
            "gamestate":   self._gamestate[idx],
            "target":      self._target[idx],
        }
        if self._offense_features is not None:
            batch["offense_features"] = self._offense_features[idx]
            batch["defense_features"] = self._defense_features[idx]
        return batch


# ---------------------------------------------------------------------------
# Train / val split factory
# ---------------------------------------------------------------------------

def make_possession_splits(
    data_dir: Path,
    seasons: list[str],
    player_vocab_path: Path,
    player_table_path: Path,
    val_fraction: float = 0.15,
    seed: int = 42,
    score_diff_scale: float = 10.0,
    offense_features: Optional["np.ndarray"] = None,
    defense_features: Optional["np.ndarray"] = None,
) -> tuple[PossessionDataset, PossessionDataset]:
    """Build train and validation PossessionDatasets from season parquets.

    The split is performed at the **game level** so no game straddles the
    train/val boundary.  Game IDs are shuffled with *seed* before splitting,
    giving a reproducible random partition.

    Parameters
    ----------
    data_dir:
        Root directory containing season subdirectories
        (e.g. Path("data")).
    seasons:
        Season labels to include, e.g. ["2022-23", "2023-24"].
    player_vocab_path:
        Path to player_vocab.parquet (output of collect-players).
    player_table_path:
        Path to players.parquet (output of collect-players).
    val_fraction:
        Fraction of games to hold out for validation (default 0.15 ≈ 15%).
    seed:
        Random seed for the game-level shuffle.
    score_diff_scale:
        Passed through to PossessionDataset.

    Returns
    -------
    (train_dataset, val_dataset)
    """
    data_dir = Path(data_dir)
    player_vocab = load_player_vocab(player_vocab_path)
    player_table = load_player_table(player_table_path)

    # Collect (parquet_path, season) pairs and all game_ids
    parquet_paths: list[tuple[Path, str]] = []
    all_game_ids: list[str] = []

    for season in seasons:
        season_dir = data_dir / season
        parquets = sorted(season_dir.glob("possessions_*.parquet"))
        if not parquets:
            raise FileNotFoundError(
                f"No possession parquet found in {season_dir}. "
                f"Run collect-possessions --season {season} first."
            )
        path = parquets[0]
        parquet_paths.append((path, season))
        df = pd.read_parquet(path, columns=["game_id"])
        all_game_ids.extend(df["game_id"].unique().tolist())

    # Deduplicate (a game_id should only appear in one season, but be safe)
    unique_game_ids = sorted(set(all_game_ids))

    # Reproducible game-level shuffle and split
    rng = random.Random(seed)
    shuffled = unique_game_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    val_ids   = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])

    kwargs = dict(
        parquet_paths=parquet_paths,
        player_vocab=player_vocab,
        player_table=player_table,
        score_diff_scale=score_diff_scale,
        offense_features=offense_features,
        defense_features=defense_features,
    )

    train_ds = PossessionDataset(**kwargs, game_ids=train_ids)
    val_ds   = PossessionDataset(**kwargs, game_ids=val_ids)

    return train_ds, val_ds
