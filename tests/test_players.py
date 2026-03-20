"""
tests/test_players.py — tests for player position metadata.

Offline tests (no NBA API calls):
    - Position string mapping is exhaustive and correct
    - make_position_lookup returns the right structure
    - Default fallback for unknown position strings

Network tests (require NBA API):
    - fetch_season_rosters returns valid data for a single season
    - Coverage: every player_id in the season parquets has a position entry
      in the consolidated player table

Run offline only:
    pytest tests/test_players.py -m "not network"

Run with network checks (slow — one API call per 30 teams):
    pytest tests/test_players.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from deep_rapm.data.players import (
    DEFAULT_POSITION_IDX,
    NBA_POSITION_MAP,
    POSITION_NAMES,
    UNKNOWN_PLAYER_IDX,
    build_player_vocab,
    make_player_index_lookup,
    make_position_lookup,
    position_str_to_idx,
)


# ---------------------------------------------------------------------------
# Offline: position mapping
# ---------------------------------------------------------------------------

class TestPositionMapping:
    def test_all_map_keys_are_valid_indices(self):
        for pos_str, idx in NBA_POSITION_MAP.items():
            assert 0 <= idx <= 4, f"{pos_str!r} → {idx} is out of range"

    def test_known_positions_map_correctly(self):
        assert position_str_to_idx("G")   == 0
        assert position_str_to_idx("G-F") == 1
        assert position_str_to_idx("F-G") == 1
        assert position_str_to_idx("F")   == 2
        assert position_str_to_idx("F-C") == 3
        assert position_str_to_idx("C-F") == 3
        assert position_str_to_idx("C")   == 4

    def test_unknown_string_returns_default(self):
        assert position_str_to_idx("") == DEFAULT_POSITION_IDX
        assert position_str_to_idx("X") == DEFAULT_POSITION_IDX
        assert position_str_to_idx("nan") == DEFAULT_POSITION_IDX

    def test_leading_trailing_whitespace_stripped(self):
        assert position_str_to_idx("  G  ") == 0
        assert position_str_to_idx(" C ") == 4

    def test_position_names_covers_all_indices(self):
        assert set(POSITION_NAMES.keys()) == {0, 1, 2, 3, 4}

    def test_all_map_values_have_names(self):
        for idx in NBA_POSITION_MAP.values():
            assert idx in POSITION_NAMES, f"Index {idx} missing from POSITION_NAMES"


# ---------------------------------------------------------------------------
# Offline: make_position_lookup
# ---------------------------------------------------------------------------

class TestMakePositionLookup:
    def _make_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "player_id":    [1, 2, 3],
                "season":       ["2023-24", "2023-24", "2022-23"],
                "position_idx": [0, 4, 2],
                "position_str": ["G", "C", "F"],
                "player_name":  ["A", "B", "C"],
                "team_id":      [1, 2, 3],
            }
        )

    def test_returns_dict(self):
        lookup = make_position_lookup(self._make_table())
        assert isinstance(lookup, dict)

    def test_keys_are_player_season_tuples(self):
        lookup = make_position_lookup(self._make_table())
        assert (1, "2023-24") in lookup
        assert (2, "2023-24") in lookup
        assert (3, "2022-23") in lookup

    def test_values_are_correct(self):
        lookup = make_position_lookup(self._make_table())
        assert lookup[(1, "2023-24")] == 0
        assert lookup[(2, "2023-24")] == 4
        assert lookup[(3, "2022-23")] == 2

    def test_missing_key_returns_keyerror(self):
        lookup = make_position_lookup(self._make_table())
        with pytest.raises(KeyError):
            _ = lookup[(999, "2023-24")]


# ---------------------------------------------------------------------------
# Offline: player vocabulary
# ---------------------------------------------------------------------------

def _make_player_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id":    [10, 30, 20],   # intentionally unsorted
            "season":       ["2023-24", "2023-24", "2022-23"],
            "position_idx": [0, 4, 2],
            "position_str": ["G", "C", "F"],
            "player_name":  ["A", "B", "C"],
            "team_id":      [1, 2, 3],
        }
    )


class TestBuildPlayerVocab:
    def test_indices_start_at_one(self, tmp_path):
        vocab = build_player_vocab(_make_player_table(), tmp_path / "vocab.parquet")
        assert vocab["player_idx"].min() == 1

    def test_unknown_index_zero_not_in_vocab(self, tmp_path):
        vocab = build_player_vocab(_make_player_table(), tmp_path / "vocab.parquet")
        assert UNKNOWN_PLAYER_IDX not in vocab["player_idx"].values

    def test_player_ids_sorted(self, tmp_path):
        vocab = build_player_vocab(_make_player_table(), tmp_path / "vocab.parquet")
        assert list(vocab["player_id"]) == sorted(vocab["player_id"])

    def test_unique_player_ids_only(self, tmp_path):
        # player_id=10 appears twice (two seasons) — vocab should deduplicate
        vocab = build_player_vocab(_make_player_table(), tmp_path / "vocab.parquet")
        assert vocab["player_id"].nunique() == len(vocab)

    def test_contiguous_indices(self, tmp_path):
        vocab = build_player_vocab(_make_player_table(), tmp_path / "vocab.parquet")
        expected = list(range(1, len(vocab) + 1))
        assert list(vocab["player_idx"]) == expected

    def test_saved_and_reloaded(self, tmp_path):
        path = tmp_path / "vocab.parquet"
        vocab1 = build_player_vocab(_make_player_table(), path)
        vocab2 = build_player_vocab(_make_player_table(), path)  # hits cache
        pd.testing.assert_frame_equal(vocab1, vocab2)


class TestMakePlayerIndexLookup:
    def test_returns_dict(self, tmp_path):
        vocab = build_player_vocab(_make_player_table(), tmp_path / "vocab.parquet")
        lookup = make_player_index_lookup(vocab)
        assert isinstance(lookup, dict)

    def test_known_player_maps_to_nonzero(self, tmp_path):
        vocab = build_player_vocab(_make_player_table(), tmp_path / "vocab.parquet")
        lookup = make_player_index_lookup(vocab)
        for pid in [10, 20, 30]:
            assert lookup[pid] != UNKNOWN_PLAYER_IDX

    def test_unknown_player_falls_back_to_zero(self, tmp_path):
        vocab = build_player_vocab(_make_player_table(), tmp_path / "vocab.parquet")
        lookup = make_player_index_lookup(vocab)
        assert lookup.get(999999, UNKNOWN_PLAYER_IDX) == UNKNOWN_PLAYER_IDX

    def test_num_players_formula(self, tmp_path):
        vocab = build_player_vocab(_make_player_table(), tmp_path / "vocab.parquet")
        # num_players passed to DeepRAPM must accommodate index 0 + all known players
        num_players = len(vocab) + 1
        lookup = make_player_index_lookup(vocab)
        assert all(0 <= idx < num_players for idx in lookup.values())


# ---------------------------------------------------------------------------
# Network: roster fetch and coverage
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_fetch_season_rosters_returns_nonempty():
    """One season of rosters should have 400+ player rows."""
    from deep_rapm.data.players import fetch_season_rosters
    df = fetch_season_rosters("2023-24")
    assert len(df) > 400, f"Expected 400+ players, got {len(df)}"
    assert set(df.columns) >= {"player_id", "player_name", "position_str",
                                "position_idx", "team_id", "season"}


@pytest.mark.network
def test_fetch_season_rosters_no_duplicate_player_season():
    """Each (player_id, season) pair must appear exactly once."""
    from deep_rapm.data.players import fetch_season_rosters
    df = fetch_season_rosters("2023-24")
    dupes = df.duplicated(subset=["player_id", "season"]).sum()
    assert dupes == 0, f"{dupes} duplicate (player_id, season) pairs"


@pytest.mark.network
def test_fetch_season_rosters_all_positions_valid():
    """Every position_idx in a fetched roster must be in 0-4."""
    from deep_rapm.data.players import fetch_season_rosters
    df = fetch_season_rosters("2023-24")
    bad = df[~df["position_idx"].isin(range(5))]
    assert len(bad) == 0, f"Invalid position_idx values:\n{bad}"


@pytest.mark.network
def test_player_table_covers_all_parquet_players(tmp_path):
    """Every player_id in the season parquets has a position entry."""
    from deep_rapm.data.players import build_player_table

    seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    table = build_player_table(
        seasons=seasons,
        output_path=tmp_path / "players.parquet",
        roster_cache_dir=tmp_path / "rosters",
    )
    lookup = set(zip(table["player_id"], table["season"]))

    data_root = Path("data")
    missing: list[tuple[int, str]] = []
    for season_dir in sorted(data_root.iterdir()):
        parquets = sorted(season_dir.glob("possessions_*.parquet"))
        if not parquets:
            continue
        season = season_dir.name
        df = pd.read_parquet(parquets[0])
        for col in ("offense_player_ids", "defense_player_ids"):
            for ids in df[col]:
                for pid in ids:
                    if (int(pid), season) not in lookup:
                        missing.append((int(pid), season))

    missing = list(set(missing))
    assert len(missing) == 0, (
        f"{len(missing)} player-season pairs in parquets have no position entry.\n"
        f"Examples: {missing[:10]}"
    )
