"""
players.py — player position metadata for Deep RAPM.

Fetches per-season roster data from the NBA Stats API and builds a stable
player-position lookup table used by the training dataset.

Position scheme
---------------
The model uses s ∈ {0, 1, 2, 3, 4} to index five positional roles:

    0 = PG   1 = SG   2 = SF   3 = PF   4 = C

The NBA Stats API returns compound position strings from commonteamroster.
We apply the following deterministic mapping:

    "G"   → 0  (PG)   pure guard — PG/SG ambiguity resolved to PG
    "G-F" → 1  (SG)   combo guard-wing
    "F-G" → 1  (SG)   wing with guard skills
    "F"   → 2  (SF)   generic forward — SF/PF ambiguity resolved to SF
    "F-C" → 3  (PF)   power forward / stretch big
    "C-F" → 3  (PF)   big with forward skills
    "C"   → 4  (C)    center

Known limitation: "G" collapses PG and SG; "F" collapses SF and PF.
The position index is an inductive bias for the attention mechanism, not
a ground-truth label.  Imperfect assignment is acceptable and consistent
with the paper (Section 4.1 implementation note).

Usage
-----
    from deep_rapm.data.players import build_player_table, load_player_table

    # One-time collection (writes data/rosters/<season>.parquet + data/players.parquet)
    build_player_table(
        seasons=["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"],
        output_path=Path("data/players.parquet"),
        roster_cache_dir=Path("data/rosters"),
    )

    # Fast load in training code
    df = load_player_table(Path("data/players.parquet"))
    pos_map = df.set_index(["player_id", "season"])["position_idx"].to_dict()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, commonteamroster
from nba_api.stats.static import teams as nba_teams


# ---------------------------------------------------------------------------
# Position string → index
# ---------------------------------------------------------------------------

#: Deterministic mapping from NBA roster position strings to 0-4 indices.
#: Source: commonteamroster endpoint (abbreviated format).
NBA_POSITION_MAP: dict[str, int] = {
    "G":   0,  # PG (pure guard; PG/SG ambiguity resolved to PG)
    "G-F": 1,  # SG (combo guard-wing)
    "F-G": 1,  # SG (wing with guard skills)
    "F":   2,  # SF (generic forward; SF/PF ambiguity resolved to SF)
    "F-C": 3,  # PF (power forward / stretch big)
    "C-F": 3,  # PF (big with forward skills)
    "C":   4,  # C  (center)
}

#: Deterministic mapping from commonplayerinfo position strings to 0-4 indices.
#: Used as fallback for players not found on any end-of-season team roster
#: (10-day contracts, waived players, injury replacements, etc.).
PLAYER_INFO_POSITION_MAP: dict[str, int] = {
    "Guard":          0,  # PG
    "Guard-Forward":  1,  # SG
    "Forward-Guard":  1,  # SG
    "Forward":        2,  # SF
    "Forward-Center": 3,  # PF
    "Center-Forward": 3,  # PF
    "Center":         4,  # C
}

POSITION_NAMES = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C"}

#: Fallback position index for players with missing/unknown position strings.
DEFAULT_POSITION_IDX: int = 2  # SF — most common "generic" NBA player role


def position_str_to_idx(pos_str: str) -> int:
    """Map an NBA position string to a 0-4 index.

    Unknown strings return DEFAULT_POSITION_IDX (2 = SF) rather than raising,
    so novel position strings in future seasons degrade gracefully.
    """
    return NBA_POSITION_MAP.get(str(pos_str).strip(), DEFAULT_POSITION_IDX)


# ---------------------------------------------------------------------------
# Roster fetching
# ---------------------------------------------------------------------------

_ALL_TEAM_IDS: list[int] = [t["id"] for t in nba_teams.get_teams()]


def fetch_season_rosters(
    season: str,
    roster_cache_dir: Optional[Path] = None,
    max_retries: int = 3,
    sleep_between_calls: float = 0.6,
) -> pd.DataFrame:
    """Return a DataFrame of all players on NBA rosters for *season*.

    Columns: player_id (int), player_name (str), position_str (str),
             position_idx (int), team_id (int), season (str).

    Results are cached to ``roster_cache_dir/<season>.parquet`` if provided.
    On subsequent calls the cache is returned immediately without any API calls.

    Parameters
    ----------
    season:
        NBA season string, e.g. "2023-24".
    roster_cache_dir:
        Directory for per-season parquet caches.  Skips API calls if the
        cache file already exists.
    max_retries:
        Number of attempts per team before raising.
    sleep_between_calls:
        Seconds to sleep between successive commonteamroster requests to
        respect the NBA Stats rate limit (~20 req/s).
    """
    if roster_cache_dir is not None:
        cache_path = Path(roster_cache_dir) / f"{season}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    records: list[dict] = []
    for team_id in _ALL_TEAM_IDS:
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                ep = commonteamroster.CommonTeamRoster(
                    team_id=team_id,
                    season=season,
                    timeout=60,
                )
                df = ep.get_data_frames()[0]
                for _, row in df.iterrows():
                    records.append(
                        {
                            "player_id":    int(row["PLAYER_ID"]),
                            "player_name":  str(row["PLAYER"]),
                            "position_str": str(row["POSITION"]).strip(),
                            "position_idx": position_str_to_idx(row["POSITION"]),
                            "team_id":      int(team_id),
                            "season":       season,
                        }
                    )
                time.sleep(sleep_between_calls)
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise

    result = pd.DataFrame(records)

    # A player traded mid-season appears on multiple team rosters.
    # Keep the last entry (most recent team) so each (player_id, season)
    # pair is unique.
    result = (
        result
        .sort_values(["player_id", "team_id"])
        .drop_duplicates(subset=["player_id", "season"], keep="last")
        .reset_index(drop=True)
    )

    if roster_cache_dir is not None:
        cache_path = Path(roster_cache_dir) / f"{season}.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)

    return result


# ---------------------------------------------------------------------------
# Full player table
# ---------------------------------------------------------------------------

def build_player_table(
    seasons: list[str],
    output_path: Path,
    roster_cache_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Fetch rosters for all *seasons* and write a consolidated player table.

    Output columns:
        player_id (int), player_name (str), position_str (str),
        position_idx (int), team_id (int), season (str).

    One row per (player_id, season) pair.  Players with multiple team
    appearances in the same season are deduplicated to their final team.

    Parameters
    ----------
    seasons:
        List of NBA season strings, e.g. ["2022-23", "2023-24"].
    output_path:
        Destination parquet file (e.g. Path("data/players.parquet")).
    roster_cache_dir:
        Per-season roster cache directory.  Passed to fetch_season_rosters.
    overwrite:
        If False and *output_path* already exists, load and return the
        existing file without re-fetching.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return pd.read_parquet(output_path)

    frames: list[pd.DataFrame] = []
    for season in seasons:
        print(f"  Fetching rosters for {season}…", flush=True)
        df = fetch_season_rosters(season, roster_cache_dir=roster_cache_dir)
        frames.append(df)
        print(f"    {len(df)} players", flush=True)

    table = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    print(f"Saved {len(table)} rows → {output_path}")
    return table


# ---------------------------------------------------------------------------
# Supplemental fetch for players missing from team rosters
# ---------------------------------------------------------------------------

def fetch_player_info_positions(
    player_ids: list[int],
    season: str,
    max_retries: int = 3,
    sleep_between_calls: float = 0.6,
) -> pd.DataFrame:
    """Fetch position data via commonplayerinfo for a list of player IDs.

    Used as a fallback for players not found on any team's end-of-season
    roster (10-day contracts, waived players, injury replacements, etc.).

    Returns a DataFrame with the same columns as fetch_season_rosters.
    team_id is set to 0 (unknown) since these players may not be on any
    current roster.
    """
    records: list[dict] = []
    for player_id in player_ids:
        fetched = False
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                ep = commonplayerinfo.CommonPlayerInfo(
                    player_id=player_id,
                    timeout=60,
                )
                df = ep.get_data_frames()[0]
                pos_str = str(df["POSITION"].iloc[0]).strip()
                pos_idx = PLAYER_INFO_POSITION_MAP.get(pos_str, DEFAULT_POSITION_IDX)
                records.append(
                    {
                        "player_id":    int(player_id),
                        "player_name":  str(df["DISPLAY_FIRST_LAST"].iloc[0]),
                        "position_str": pos_str,
                        "position_idx": pos_idx,
                        "team_id":      0,
                        "season":       season,
                    }
                )
                time.sleep(sleep_between_calls)
                fetched = True
                break
            except Exception:
                if attempt == max_retries - 1:
                    # Some historical player IDs return a malformed response
                    # (e.g., retired players no longer indexed by the API).
                    # Fall back to DEFAULT_POSITION_IDX rather than crashing.
                    print(
                        f"    WARNING: could not fetch position for player_id={player_id}"
                        f" — defaulting to {POSITION_NAMES[DEFAULT_POSITION_IDX]}",
                        flush=True,
                    )
                    records.append(
                        {
                            "player_id":    int(player_id),
                            "player_name":  f"unknown_{player_id}",
                            "position_str": "unknown",
                            "position_idx": DEFAULT_POSITION_IDX,
                            "team_id":      0,
                            "season":       season,
                        }
                    )

    return pd.DataFrame(records)


def supplement_player_table(
    table: pd.DataFrame,
    parquet_dir: Path,
    output_path: Path,
    max_retries: int = 3,
    sleep_between_calls: float = 0.6,
) -> pd.DataFrame:
    """Fetch positions for players in the parquets but not in *table*.

    Scans all season parquets under *parquet_dir*, finds player-season pairs
    without a position entry, calls commonplayerinfo for each unique missing
    player_id, and appends the results to *table*.  Saves the extended table
    to *output_path*.

    Parameters
    ----------
    table:
        Existing player table (output of build_player_table).
    parquet_dir:
        Root data directory containing season subdirectories
        (e.g. Path("data")).
    output_path:
        Where to save the updated player table.
    """
    existing = set(zip(table["player_id"], table["season"]))

    # Find missing (player_id, season) pairs
    missing_by_season: dict[str, set[int]] = {}
    for season_dir in sorted(Path(parquet_dir).iterdir()):
        parquets = sorted(season_dir.glob("possessions_*.parquet"))
        if not parquets:
            continue
        season = season_dir.name
        df = pd.read_parquet(parquets[0])
        for col in ("offense_player_ids", "defense_player_ids"):
            for ids in df[col]:
                for pid in ids:
                    pid = int(pid)
                    if (pid, season) not in existing:
                        missing_by_season.setdefault(season, set()).add(pid)

    if not missing_by_season:
        print("No missing players — player table is complete.")
        return table

    total_missing = sum(len(v) for v in missing_by_season.values())
    print(f"Fetching positions for {total_missing} missing player-season pairs…")

    frames = [table]
    for season, player_ids in sorted(missing_by_season.items()):
        print(f"  {season}: {len(player_ids)} missing players", flush=True)
        supplement = fetch_player_info_positions(
            sorted(player_ids),
            season=season,
            max_retries=max_retries,
            sleep_between_calls=sleep_between_calls,
        )
        frames.append(supplement)

    extended = pd.concat(frames, ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extended.to_parquet(output_path, index=False)
    print(f"Saved {len(extended)} rows → {output_path}")
    return extended


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Player vocabulary
# ---------------------------------------------------------------------------

#: Index reserved for unknown / unseen players in the embedding table.
UNKNOWN_PLAYER_IDX: int = 0


def build_player_vocab(
    player_table: pd.DataFrame,
    output_path: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build a stable player_id → embedding index mapping.

    Index 0 is reserved for UNKNOWN_PLAYER_IDX so that unseen players can
    be represented without crashing the model.  All known players are assigned
    indices 1 … N in ascending player_id order, giving a deterministic mapping
    that is identical across machines and reruns.

    Parameters
    ----------
    player_table:
        Output of build_player_table / load_player_table.
    output_path:
        Destination parquet (e.g. Path("data/player_vocab.parquet")).
    overwrite:
        If False and output_path already exists, load and return it.

    Returns
    -------
    DataFrame with columns player_id (int) and player_idx (int).
    Pass ``len(vocab) + 1`` as num_players to DeepRAPM so the embedding
    table has room for index 0 (unknown) plus all known players.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return pd.read_parquet(output_path)

    unique_ids = sorted(player_table["player_id"].unique())
    vocab = pd.DataFrame(
        {
            "player_id":  unique_ids,
            # 0 is reserved → known players start at 1
            "player_idx": range(1, len(unique_ids) + 1),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.to_parquet(output_path, index=False)
    return vocab


def load_player_vocab(path: Path) -> pd.DataFrame:
    """Load the player vocabulary from *path*."""
    return pd.read_parquet(Path(path))


def make_player_index_lookup(vocab: pd.DataFrame) -> dict[int, int]:
    """Return a player_id → player_idx dict for fast embedding-index lookups.

    Unlisted player_ids should resolve to UNKNOWN_PLAYER_IDX at query time:
        idx = lookup.get(player_id, UNKNOWN_PLAYER_IDX)
    """
    return dict(zip(vocab["player_id"], vocab["player_idx"]))


# ---------------------------------------------------------------------------
# Player feature tensor
# ---------------------------------------------------------------------------

def build_player_features(
    player_table: pd.DataFrame,
    player_vocab: pd.DataFrame,
) -> "torch.Tensor":
    """Build a (num_players, feature_dim) float tensor of per-player features.

    Index 0 is reserved for UNKNOWN and receives an all-zeros feature vector.

    Current features (columns of the output tensor):
        [0:5]  — one-hot position (PG, SG, SF, PF, C)

    Parameters
    ----------
    player_table  : Output of build_player_table / load_player_table.
    player_vocab  : Output of build_player_vocab (player_id → player_idx).

    Returns
    -------
    FloatTensor of shape (len(player_vocab) + 1, 5).
    Pass this directly to CrossRAPM as ``player_features=...``.
    """
    import torch

    n_players = len(player_vocab) + 1   # +1 for UNKNOWN at index 0
    n_positions = 5
    features = torch.zeros(n_players, n_positions)

    idx_map = dict(zip(
        player_vocab["player_id"].astype(int),
        player_vocab["player_idx"].astype(int),
    ))

    # Use each player's most recent season position so the feature is stable.
    latest = (
        player_table
        .sort_values("season")
        .drop_duplicates("player_id", keep="last")
    )
    for _, row in latest.iterrows():
        player_idx = idx_map.get(int(row["player_id"]))
        if player_idx is not None:
            pos_idx = int(row["position_idx"])
            if 0 <= pos_idx < n_positions:
                features[player_idx, pos_idx] = 1.0

    return features


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_player_table(path: Path) -> pd.DataFrame:
    """Load the consolidated player table from *path*."""
    return pd.read_parquet(Path(path))


def make_position_lookup(
    player_table: pd.DataFrame,
) -> dict[tuple[int, str], int]:
    """Return a (player_id, season) → position_idx dict for fast lookups.

    Missing entries resolve to DEFAULT_POSITION_IDX at query time:
        pos = pos_lookup.get((player_id, season), DEFAULT_POSITION_IDX)
    """
    return (
        player_table
        .set_index(["player_id", "season"])["position_idx"]
        .to_dict()
    )
