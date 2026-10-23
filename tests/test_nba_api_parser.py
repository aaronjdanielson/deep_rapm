"""
Regression tests for the nba_api fallback parser (deep_rapm.data.game_nba_api).

Offline: parses cached raw PlayByPlayV3 data (data/raw/nba_api/<gid>.json.gz,
written by fetch_raw) for 2024-25 games that pbpstats also extracted, and
compares against the pbpstats possessions in the season parquet and the
official final score. Skips games whose raw cache is absent.
"""

import glob
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from deep_rapm.data.game_nba_api import (_stints_by_team, infer_stints, parse_possessions,
                                         rotation_ok)

RAW = Path("data/raw/nba_api")
REF = Path("data/2024-25/possessions_2024_25_regular_season.parquet")
GAMES = ["0022400061", "0022400500", "0022401000"]


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    if not REF.exists():
        pytest.skip("2024-25 season parquet not available")
    return pd.read_parquet(REF, columns=["game_id", "game_date", "period", "offense_team_id",
                                         "home_team_id", "points"])


@pytest.mark.parametrize("game_id", GAMES)
def test_matches_pbpstats_and_official_score(game_id: str, reference: pd.DataFrame) -> None:
    raw = RAW / f"{game_id}.json.gz"
    if not raw.exists():
        pytest.skip(f"no cached raw data for {game_id}")
    with gzip.open(raw, "rt") as fh:
        actions = json.load(fh)["actions"]
    ref = reference[reference["game_id"] == game_id]
    home = int(ref["home_team_id"].iloc[0])
    away = int((set(ref["offense_team_id"]) - {home}).pop())
    meta = {"game_date": ref["game_date"].iloc[0], "home_team_id": home, "away_team_id": away}

    poss, info = parse_possessions(game_id, meta, actions, None)
    df = pd.DataFrame(poss)

    final = next(a for a in reversed(actions) if a.get("scoreHome"))
    pts = df.groupby("offense_team_id")["points"].sum()
    assert pts[home] == int(final["scoreHome"])
    assert pts[away] == int(final["scoreAway"])

    assert abs(len(df) - len(ref)) <= 3
    assert info["self_corrections"] <= 3

    teams = df["offense_team_id"].to_numpy()
    periods = df["period"].to_numpy()
    same_team_twice = ((teams[1:] == teams[:-1]) & (periods[1:] == periods[:-1])).sum()
    assert same_team_twice == 0


def _games_with_both_sources(limit: int = 15) -> list[str]:
    out = []
    for f in sorted(glob.glob(str(RAW / "00*.json.gz"))):
        with gzip.open(f, "rt") as fh:
            rec = json.load(fh)
        if rec.get("box") and rotation_ok(rec):
            out.append(Path(f).name[:10])
        if len(out) == limit:
            break
    return out


@pytest.mark.parametrize("game_id", _games_with_both_sources() or ["none-cached"])
def test_inferred_lineups_match_rotation(game_id: str) -> None:
    """Lineups inferred from play-by-play + box score equal GameRotation's on
    every possession the inference covers, and cover nearly all of them."""
    if game_id == "none-cached":
        pytest.skip("no cached game with both box score and rotation data")
    with gzip.open(RAW / f"{game_id}.json.gz", "rt") as fh:
        rec = json.load(fh)
    teams = sorted({int(r[0]) for r in rec["box"]})
    meta = {"game_date": "", "home_team_id": teams[0], "away_team_id": teams[1]}

    truth, _ = parse_possessions(game_id, meta, rec["actions"], _stints_by_team(rec["rotation"]))
    stints, info = infer_stints(rec["actions"], rec["box"])
    inferred, pinfo = parse_possessions(game_id, meta, rec["actions"], stints)

    by_start = {(p["seconds_into_game"], p["offense_team_id"]): p for p in truth}
    for p in inferred:
        t = by_start[(p["seconds_into_game"], p["offense_team_id"])]
        assert set(p["offense_player_ids"]) == set(t["offense_player_ids"])
        assert set(p["defense_player_ids"]) == set(t["defense_player_ids"])
    assert len(inferred) >= 0.6 * len(truth)
    if pinfo["skipped_lineup"] == 0:
        assert info["minutes_max_err"] < 0.1
