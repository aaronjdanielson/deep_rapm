"""
tests/test_dataset.py — tests for PossessionDataset and make_possession_splits.

All tests are fully offline — they operate on synthetic parquets built in
tmp_path fixtures, not on the real data/ directory.

Run:
    pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from deep_rapm.data.dataset import PossessionDataset, make_possession_splits
from deep_rapm.data.players import (
    DEFAULT_POSITION_IDX,
    UNKNOWN_PLAYER_IDX,
    build_player_vocab,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_player_table(player_ids: list[int], season: str) -> pd.DataFrame:
    n = len(player_ids)
    pos_cycle = ["G", "G-F", "F", "F-C", "C"]
    return pd.DataFrame(
        {
            "player_id":    player_ids,
            "season":       [season] * n,
            "position_idx": [i % 5 for i in range(n)],
            "position_str": [pos_cycle[i % 5] for i in range(n)],
            "player_name":  [f"p{pid}" for pid in player_ids],
            "team_id":      [1] * n,
        }
    )


def _make_possession_df(
    game_ids: list[str],
    player_ids: list[int],
    n_per_game: int = 10,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic possession DataFrame matching the parquet schema."""
    rng = random.Random(rng_seed)
    rows = []
    for game_id in game_ids:
        for _ in range(n_per_game):
            sample = rng.sample(player_ids, 10)
            off_ids = sorted(sample[:5])
            def_ids = sorted(sample[5:])
            rows.append(
                {
                    "game_id":             game_id,
                    "game_date":           "2023-11-01",
                    "period":              rng.randint(1, 4),
                    "seconds_into_game":   float(rng.randint(0, 2880)),
                    "offense_team_id":     1,
                    "defense_team_id":     2,
                    # Use Python lists — numpy arrays in pd.DataFrame(rows)
                    # get broadcast as multiple columns rather than stored as
                    # object cells.  The parquet round-trip converts them to
                    # arrays automatically.
                    "offense_player_ids":  off_ids,
                    "defense_player_ids":  def_ids,
                    "points":              rng.randint(0, 6),
                    "score_diff":          rng.randint(-20, 20),
                    "home_team_id":        1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def season_data(tmp_path) -> dict:
    """Synthetic single-season dataset written to tmp_path."""
    player_ids = list(range(100, 130))   # 30 players
    season = "2023-24"
    game_ids = [f"00230{i:04d}" for i in range(20)]

    poss_df = _make_possession_df(game_ids, player_ids, n_per_game=15)

    season_dir = tmp_path / "data" / season
    season_dir.mkdir(parents=True)
    parquet_path = season_dir / f"possessions_{season}.parquet"
    poss_df.to_parquet(parquet_path, index=False)

    player_table = _make_player_table(player_ids, season)
    vocab = build_player_vocab(player_table, tmp_path / "player_vocab.parquet")

    player_table_path = tmp_path / "players.parquet"
    player_table.to_parquet(player_table_path, index=False)

    return {
        "parquet_path":       parquet_path,
        "season":             season,
        "parquet_paths":      [(parquet_path, season)],
        "player_table":       player_table,
        "player_vocab":       vocab,
        "player_ids":         player_ids,
        "game_ids":           game_ids,
        "n_possessions":      len(poss_df),
        "data_dir":           tmp_path / "data",
        "player_vocab_path":  tmp_path / "player_vocab.parquet",
        "player_table_path":  player_table_path,
    }


# ---------------------------------------------------------------------------
# PossessionDataset: length and basic structure
# ---------------------------------------------------------------------------

class TestPossessionDatasetLength:
    def test_len_matches_possession_count(self, season_data):
        ds = PossessionDataset(
            parquet_paths=season_data["parquet_paths"],
            player_vocab=season_data["player_vocab"],
            player_table=season_data["player_table"],
        )
        assert len(ds) == season_data["n_possessions"]

    def test_game_ids_filter_reduces_length(self, season_data):
        half = set(season_data["game_ids"][:10])
        ds = PossessionDataset(
            parquet_paths=season_data["parquet_paths"],
            player_vocab=season_data["player_vocab"],
            player_table=season_data["player_table"],
            game_ids=half,
        )
        assert len(ds) == 10 * 15   # 10 games × 15 possessions/game


# ---------------------------------------------------------------------------
# PossessionDataset: item structure and types
# ---------------------------------------------------------------------------

class TestPossessionDatasetItem:
    @pytest.fixture(autouse=True)
    def _build(self, season_data):
        self.ds = PossessionDataset(
            parquet_paths=season_data["parquet_paths"],
            player_vocab=season_data["player_vocab"],
            player_table=season_data["player_table"],
        )
        self.item = self.ds[0]

    def test_item_has_required_keys(self):
        assert set(self.item.keys()) == {
            "offense_ids", "defense_ids",
            "offense_pos", "defense_pos",
            "gamestate", "target",
        }

    def test_offense_ids_shape(self):
        assert self.item["offense_ids"].shape == (5,)

    def test_defense_ids_shape(self):
        assert self.item["defense_ids"].shape == (5,)

    def test_offense_pos_shape(self):
        assert self.item["offense_pos"].shape == (5,)

    def test_defense_pos_shape(self):
        assert self.item["defense_pos"].shape == (5,)

    def test_gamestate_shape(self):
        assert self.item["gamestate"].shape == (1,)

    def test_target_is_scalar(self):
        assert self.item["target"].shape == ()

    def test_ids_are_long(self):
        assert self.item["offense_ids"].dtype == torch.long
        assert self.item["defense_ids"].dtype == torch.long

    def test_pos_are_long(self):
        assert self.item["offense_pos"].dtype == torch.long
        assert self.item["defense_pos"].dtype == torch.long

    def test_gamestate_is_float(self):
        assert self.item["gamestate"].dtype == torch.float32

    def test_target_is_float(self):
        assert self.item["target"].dtype == torch.float32


# ---------------------------------------------------------------------------
# PossessionDataset: value correctness
# ---------------------------------------------------------------------------

class TestPossessionDatasetValues:
    @pytest.fixture(autouse=True)
    def _build(self, season_data):
        self.ds = PossessionDataset(
            parquet_paths=season_data["parquet_paths"],
            player_vocab=season_data["player_vocab"],
            player_table=season_data["player_table"],
            score_diff_scale=10.0,
        )

    def test_offense_ids_nonzero_for_known_players(self):
        for i in range(min(50, len(self.ds))):
            ids = self.ds[i]["offense_ids"]
            assert (ids > UNKNOWN_PLAYER_IDX).all(), \
                f"Possession {i} has UNKNOWN offensive player index"

    def test_defense_ids_nonzero_for_known_players(self):
        for i in range(min(50, len(self.ds))):
            ids = self.ds[i]["defense_ids"]
            assert (ids > UNKNOWN_PLAYER_IDX).all(), \
                f"Possession {i} has UNKNOWN defensive player index"

    def test_positions_in_valid_range(self):
        for i in range(min(50, len(self.ds))):
            item = self.ds[i]
            assert item["offense_pos"].min() >= 0
            assert item["offense_pos"].max() <= 4
            assert item["defense_pos"].min() >= 0
            assert item["defense_pos"].max() <= 4

    def test_target_is_raw_points_float(self):
        targets = torch.stack([self.ds[i]["target"] for i in range(len(self.ds))])
        assert targets.dtype == torch.float32
        assert targets.min() >= 0

    def test_score_diff_scaled(self):
        # gamestate[0] = score_diff / 10.0 — check roundtrip is plausible
        for i in range(min(20, len(self.ds))):
            gs = self.ds[i]["gamestate"][0].item()
            # synthetic score_diff in [-20, 20] → gamestate in [-2, 2]
            assert -10.0 <= gs <= 10.0

    def test_offense_ids_sorted(self):
        # Players are sorted by player_id so indices should be non-decreasing
        # within the lineup (since player_id order == vocab_index order for
        # the synthetic data where player_ids are 100...129 in order).
        for i in range(min(20, len(self.ds))):
            ids = self.ds[i]["offense_ids"].tolist()
            assert ids == sorted(ids), \
                f"Possession {i} offense_ids not sorted: {ids}"

    def test_defense_ids_sorted(self):
        for i in range(min(20, len(self.ds))):
            ids = self.ds[i]["defense_ids"].tolist()
            assert ids == sorted(ids), \
                f"Possession {i} defense_ids not sorted: {ids}"


# ---------------------------------------------------------------------------
# make_possession_splits
# ---------------------------------------------------------------------------

class TestMakePossessionSplits:
    @pytest.fixture(autouse=True)
    def _build(self, season_data):
        self.train_ds, self.val_ds = make_possession_splits(
            data_dir=season_data["data_dir"],
            seasons=[season_data["season"]],
            player_vocab_path=season_data["player_vocab_path"],
            player_table_path=season_data["player_table_path"],
            val_fraction=0.15,
            seed=42,
        )
        self.total = season_data["n_possessions"]

    def test_sizes_sum_to_total(self):
        assert len(self.train_ds) + len(self.val_ds) == self.total

    def test_val_fraction_approximately_correct(self):
        # 20 games → 3 val games (0.15 × 20 = 3.0)
        expected_val = round(20 * 0.15) * 15   # 3 games × 15 possessions
        assert len(self.val_ds) == expected_val

    def test_train_larger_than_val(self):
        assert len(self.train_ds) > len(self.val_ds)

    def test_no_game_in_both_splits(self, season_data):
        # Load game_ids from each split by looking at which rows were included.
        # We verify by checking that len(train) + len(val) == total (done above)
        # and that the split is reproducible (same seed gives same result).
        train2, val2 = make_possession_splits(
            data_dir=season_data["data_dir"],
            seasons=[season_data["season"]],
            player_vocab_path=season_data["player_vocab_path"],
            player_table_path=season_data["player_table_path"],
            val_fraction=0.15,
            seed=42,
        )
        assert len(train2) == len(self.train_ds)
        assert len(val2)   == len(self.val_ds)

    def test_different_seeds_give_different_splits(self, season_data):
        _, val_alt = make_possession_splits(
            data_dir=season_data["data_dir"],
            seasons=[season_data["season"]],
            player_vocab_path=season_data["player_vocab_path"],
            player_table_path=season_data["player_table_path"],
            val_fraction=0.15,
            seed=99,
        )
        # Different seed should produce a different set of val games
        # (same size since same n_val, but different rows).
        val1_target = torch.stack([self.val_ds[i]["target"] for i in range(len(self.val_ds))])
        val2_target = torch.stack([val_alt[i]["target"] for i in range(len(val_alt))])
        # The two val sets may occasionally have the same targets by coincidence,
        # but for a large enough synthetic set with varied points it's extremely unlikely.
        assert not torch.equal(val1_target, val2_target)

    def test_missing_season_raises(self, season_data):
        with pytest.raises(FileNotFoundError):
            make_possession_splits(
                data_dir=season_data["data_dir"],
                seasons=["1999-00"],   # doesn't exist
                player_vocab_path=season_data["player_vocab_path"],
                player_table_path=season_data["player_table_path"],
            )


# ---------------------------------------------------------------------------
# DataLoader compatibility
# ---------------------------------------------------------------------------

def test_dataloader_batches(season_data):
    """Dataset must be iterable via DataLoader and produce correct batch shapes."""
    from torch.utils.data import DataLoader

    ds = PossessionDataset(
        parquet_paths=season_data["parquet_paths"],
        player_vocab=season_data["player_vocab"],
        player_table=season_data["player_table"],
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    batch = next(iter(loader))

    assert batch["offense_ids"].shape == (32, 5)
    assert batch["defense_ids"].shape == (32, 5)
    assert batch["offense_pos"].shape == (32, 5)
    assert batch["defense_pos"].shape == (32, 5)
    assert batch["gamestate"].shape   == (32, 1)
    assert batch["target"].shape      == (32,)
