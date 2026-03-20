"""
tests/conftest.py — shared fixtures and configuration.

Cache layout
------------
PBP JSON files live in  pbp_cache/pbp/data_<game_id>.json  (written by
pbpstats on first fetch; read on every subsequent run).  The file cache
makes all structural tests network-free.

Markers
-------
@pytest.mark.network   — test calls the NBA Stats API (stats.nba.com).
                         Skip offline with:  pytest -m "not network"
"""

import pytest
import pandas as pd
from pathlib import Path

from deep_rapm.data.game import get_game_possessions


# ---------------------------------------------------------------------------
# Shared cache path — override with --pbp-cache CLI option if needed
# ---------------------------------------------------------------------------

PBP_CACHE = Path("pbp_cache")


def pytest_addoption(parser):
    parser.addoption(
        "--sample-size",
        type=int,
        default=20,
        help="Games to sample per season for test_sample_games.py (default: 20)",
    )
    parser.addoption(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for game sampling (default: 42)",
    )


# ---------------------------------------------------------------------------
# Session-scoped possession store (avoids re-running extraction per test)
# ---------------------------------------------------------------------------

_possession_store: dict[str, pd.DataFrame] = {}


def get_possessions(game_id: str) -> pd.DataFrame:
    """Return (cached) possession DataFrame for *game_id*."""
    if game_id not in _possession_store:
        records = get_game_possessions(game_id, pbp_cache_dir=PBP_CACHE)
        _possession_store[game_id] = pd.DataFrame(records)
    return _possession_store[game_id]
