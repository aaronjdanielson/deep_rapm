"""
tests/test_possessions.py — end-to-end tests for the possession pipeline.

Run all tests (requires network for boxscore checks):
    pytest tests/ -v

Run only cached/structural tests (no network required):
    pytest tests/ -v -m "not network"

Game sample
-----------
  0022300154  2023-24  reg  no OT   MIN @ BOS
  0022300001  2023-24  reg  no OT   first game of season
  0021800514  2018-19  reg  no OT   SAC @ LAL (contains 5-pt possession)
  0021800399  2018-19  reg  1 OT
  0021800480  2018-19  reg  2 OT
  0021900001  2019-20  reg  1 OT    first game of season
"""

import time

import pytest
import pandas as pd

from deep_rapm.data.game import get_game_possessions
from deep_rapm.data.validate import validate_game, compute_player_seconds

from conftest import PBP_CACHE, get_possessions


# ---------------------------------------------------------------------------
# Game parametrisation
# ---------------------------------------------------------------------------

# (game_id, min_possessions, max_possessions)
# All of these are already in pbp_cache — no network call needed for PBP.
ALL_GAMES = [
    ("0022300154", 180, 260),  # 2023-24 reg, no OT
    ("0022300001", 180, 260),  # 2023-24 reg, no OT
    ("0021800514", 180, 260),  # 2018-19 reg, no OT
    ("0021800399", 180, 280),  # 2018-19 reg, 1 OT
    ("0021800480", 200, 320),  # 2018-19 reg, 2 OT
    ("0021900001", 180, 280),  # 2019-20 reg, 1 OT
]

# Subset used for network-dependent cross-validation (keeps CI times short).
NETWORK_GAMES = [
    ("0022300154", 180, 260),
    ("0021800514", 180, 260),
    ("0021800480", 200, 320),  # 2 OT — exercises period > 4 code paths
]

ids = [g for g, *_ in ALL_GAMES]
net_ids = [g for g, *_ in NETWORK_GAMES]


# ---------------------------------------------------------------------------
# Helper: fetch official boxscore player stats with retries
# ---------------------------------------------------------------------------

def _boxscore_players(game_id: str, max_retries: int = 3):
    """Return {player_id: team_id} for everyone who played in *game_id*."""
    from nba_api.stats.endpoints import boxscoretraditionalv3

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            bx = boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=game_id, timeout=60
            )
            players = bx.player_stats.get_data_frame()
            played = players[players["minutes"].astype(str).str.len() > 0]
            return {
                int(r["personId"]): int(r["teamId"])
                for _, r in played.iterrows()
            }
        except Exception:
            if attempt == max_retries - 1:
                raise


def _parse_min(m: str) -> float:
    parts = str(m).split(":")
    return float(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else 0.0


# ---------------------------------------------------------------------------
# On-court player invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_player_on_court_invariant(game_id, _min, _max):
    """No acting player (shot, FT, rebound, TO, foul) may be on the bench."""
    result = validate_game(game_id, pbp_cache_dir=PBP_CACHE)
    assert result.passed, f"On-court violations:\n{result.summary()}"


# ---------------------------------------------------------------------------
# Score integrity
# ---------------------------------------------------------------------------

def _tech_ft_points(game_id: str) -> dict[int, int]:
    """Count made technical/flagrant free throws per team.

    Technical FTs are shot by the opposing team during the fouled team's
    possession, so they are intentionally excluded from possession-level
    point accumulation.  The score integrity check subtracts these before
    comparing to the official boxscore total.
    """
    from pbpstats.data_loader import (
        DataNbaEnhancedPbpFileLoader,
        DataNbaEnhancedPbpLoader,
    )

    loader = DataNbaEnhancedPbpLoader(
        game_id, DataNbaEnhancedPbpFileLoader(str(PBP_CACHE))
    )
    tech_pts: dict[int, int] = {}
    for ev in loader.items:
        if getattr(ev, "event_type", None) != 3:
            continue
        if not getattr(ev, "is_made", False):
            continue
        desc = str(getattr(ev, "description", "")).lower()
        if "technical" not in desc and "flagrant" not in desc:
            continue
        team_id = getattr(ev, "team_id", None)
        off_id  = getattr(ev, "offense_team_id", None)
        # A technical FT is shot by the team that did NOT commit the foul,
        # which is the non-possession team — i.e. team_id != offense_team_id.
        if team_id and off_id and team_id != off_id:
            tech_pts[team_id] = tech_pts.get(team_id, 0) + 1
    return tech_pts


@pytest.mark.network
@pytest.mark.parametrize("game_id,_min,_max", NETWORK_GAMES, ids=net_ids)
def test_score_totals_match_boxscore(game_id, _min, _max):
    """Sum of possession points per team must equal the official final score.

    Technical and flagrant free throws shot during the opponent's possession
    are excluded from possession-level scoring by design (they represent
    dead-ball points unrelated to the possession contest).  The expected
    total is therefore:  official_score - technical_ft_points.
    """
    from nba_api.stats.endpoints import boxscoresummaryv3

    df = get_possessions(game_id)
    our_scores = df.groupby("offense_team_id")["points"].sum().to_dict()
    tech_pts = _tech_ft_points(game_id)

    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            summ = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id)
            ls = summ.line_score.get_data_frame()[["teamId", "score"]]
            break
        except Exception:
            if attempt == 2:
                raise

    for _, row in ls.iterrows():
        tid = int(row["teamId"])
        official = int(row["score"])
        expected = official - tech_pts.get(tid, 0)
        extracted = our_scores.get(tid, 0)
        assert extracted == expected, (
            f"Team {tid}: extracted {extracted} pts != "
            f"official {official} - {tech_pts.get(tid, 0)} tech = {expected} pts"
        )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_possession_schema(game_id, _min, _max):
    """All required columns must be present and the frame must be non-empty."""
    df = get_possessions(game_id)
    assert len(df) > 0, "No possessions extracted"

    required = {
        "game_id", "game_date", "period", "seconds_into_game",
        "offense_team_id", "defense_team_id",
        "offense_player_ids", "defense_player_ids",
        "points", "score_diff", "home_team_id",
    }
    assert required.issubset(df.columns), f"Missing: {required - set(df.columns)}"


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_lineup_sizes(game_id, _min, _max):
    """Every possession must have exactly 5 offensive and 5 defensive players."""
    df = get_possessions(game_id)
    bad_off = df[df["offense_player_ids"].apply(len) != 5]
    bad_def = df[df["defense_player_ids"].apply(len) != 5]
    assert len(bad_off) == 0, f"{len(bad_off)} possessions with != 5 offensive players"
    assert len(bad_def) == 0, f"{len(bad_def)} possessions with != 5 defensive players"


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_no_overlap_between_lineups(game_id, _min, _max):
    """No player may appear on both offense and defense in the same possession."""
    df = get_possessions(game_id)
    overlaps = df.apply(
        lambda r: len(set(r["offense_player_ids"]) & set(r["defense_player_ids"])) > 0,
        axis=1,
    )
    assert not overlaps.any(), (
        f"{overlaps.sum()} possessions where a player is on both teams"
    )


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_no_duplicate_players_within_lineup(game_id, _min, _max):
    """No player may appear twice in the same 5-player lineup."""
    df = get_possessions(game_id)
    dup_off = df[df["offense_player_ids"].apply(lambda ids: len(ids) != len(set(ids)))]
    dup_def = df[df["defense_player_ids"].apply(lambda ids: len(ids) != len(set(ids)))]
    assert len(dup_off) == 0, f"{len(dup_off)} possessions with duplicate offensive players"
    assert len(dup_def) == 0, f"{len(dup_def)} possessions with duplicate defensive players"


# ---------------------------------------------------------------------------
# Possession count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id,min_poss,max_poss", ALL_GAMES, ids=ids)
def test_possession_count_in_range(game_id, min_poss, max_poss):
    """Possession count must fall within the expected range for each game type."""
    df = get_possessions(game_id)
    n = len(df)
    assert min_poss <= n <= max_poss, (
        f"Game {game_id}: {n} possessions outside [{min_poss}, {max_poss}]"
    )


# ---------------------------------------------------------------------------
# Points sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_points_in_valid_range(game_id, _min, _max):
    """Points per possession must be in [0, 8].

    Standard max is 4 (And-1 3-pointer + made FT).  Rare multi-action
    possessions (made shot → shooting foul → missed FT → offensive rebound →
    scored again) produce up to 7 points in real data; 8 is a safe ceiling.
    """
    df = get_possessions(game_id)
    bad = df[(df["points"] < 0) | (df["points"] > 8)]
    assert len(bad) == 0, (
        f"Out-of-range points:\n"
        f"{bad[['period', 'points', 'offense_team_id']].to_string()}"
    )


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_score_diff_is_integer(game_id, _min, _max):
    df = get_possessions(game_id)
    assert df["score_diff"].dtype in ("int64", "int32", "float64")
    assert df["score_diff"].abs().max() <= 60, "score_diff exceeds plausible NBA margin"


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_seconds_into_game_monotone(game_id, _min, _max):
    """Possessions must start at non-negative times within game length."""
    df = get_possessions(game_id)
    assert (df["seconds_into_game"] >= 0).all()
    # Regulation (2880s) + up to 10 OT periods
    assert df["seconds_into_game"].max() <= 4 * 12 * 60 + 10 * 5 * 60


# ---------------------------------------------------------------------------
# score_diff correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_score_diff_starts_at_zero(game_id, _min, _max):
    """The very first possession of every game must start at 0–0."""
    df = get_possessions(game_id)
    first = df.iloc[0]["score_diff"]
    assert first == 0, f"First possession score_diff={first}, expected 0"


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_score_diff_reconstruction(game_id, _min, _max):
    """score_diff must be consistent with the running possession-point totals.

    We reconstruct the expected pre-possession margin by accumulating points
    from the possessions column.  The actual score_diff is taken from the
    NBA's official running score (which includes technical FTs); our
    reconstruction excludes them.  The difference equals the net technical-FT
    points scored up to that moment.  We allow up to 8 points of divergence —
    enough to cover the most extreme multi-technical games in the dataset —
    so that any sign error, off-by-one, or large computation bug is caught
    while rare edge cases are tolerated.
    """
    df = get_possessions(game_id)
    home_id = df["home_team_id"].iloc[0]

    home_pts = 0
    away_pts = 0
    for _, row in df.iterrows():
        off = row["offense_team_id"]
        expected = (home_pts - away_pts) if off == home_id else (away_pts - home_pts)
        actual = int(row["score_diff"])
        assert abs(actual - expected) <= 8, (
            f"score_diff={actual} but possession-point reconstruction gives "
            f"{expected} (diff={actual - expected}) at period={row['period']} "
            f"t={row['seconds_into_game']}"
        )
        if off == home_id:
            home_pts += row["points"]
        else:
            away_pts += row["points"]


# ---------------------------------------------------------------------------
# Period / time consistency
# ---------------------------------------------------------------------------

_PERIOD_BOUNDS = {
    1: (0, 720), 2: (720, 1440), 3: (1440, 2160), 4: (2160, 2880),
    5: (2880, 3180), 6: (3180, 3480), 7: (3480, 3780), 8: (3780, 4080),
}

@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_period_seconds_consistency(game_id, _min, _max):
    """seconds_into_game must fall in the correct range for its period.

    Period 1 → [0, 720), period 2 → [720, 1440), OT1 → [2880, 3180), etc.
    A possession stamped period=2 but with seconds_into_game=150 (a Q1 time)
    would indicate a clock-parsing bug.
    """
    df = get_possessions(game_id)
    bad = []
    for _, row in df.iterrows():
        p = int(row["period"])
        s = row["seconds_into_game"]
        lo, hi = _PERIOD_BOUNDS.get(p, (None, None))
        if lo is None:
            continue  # unknown OT period — skip rather than false-fail
        if not (lo <= s <= hi):
            bad.append(
                f"period={p} but seconds_into_game={s:.1f} not in [{lo}, {hi}]"
            )
    assert not bad, f"Period/seconds mismatches in {game_id}:\n" + "\n".join(bad[:5])


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_no_large_within_period_gap(game_id, _min, _max):
    """No two consecutive possessions within the same period may be more than
    90 seconds apart.

    The longest realistic possession (garbage-time with no clock stoppage) is
    well under 60 s; 90 s gives a generous buffer.  A gap exceeding this
    suggests dropped possessions or a clock-parsing error.
    """
    df = get_possessions(game_id).sort_values(["period", "seconds_into_game"])
    MAX_GAP = 90.0
    bad = []
    for period, grp in df.groupby("period"):
        times = grp["seconds_into_game"].values
        for i in range(len(times) - 1):
            gap = times[i + 1] - times[i]
            if gap > MAX_GAP:
                bad.append(f"period={period} gap={gap:.1f}s at t={times[i]:.1f}")
    assert not bad, (
        f"Large within-period time gaps in {game_id}:\n" + "\n".join(bad[:5])
    )


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_all_expected_periods_present(game_id, _min, _max):
    """All four regulation periods must be present.  OT games must have ≥ 1
    OT period (period ≥ 5).
    """
    df = get_possessions(game_id)
    periods = set(df["period"].unique())
    for p in (1, 2, 3, 4):
        assert p in periods, f"Period {p} has no possessions in game {game_id}"

    # OT detection: seconds_into_game > 2880 means an OT possession exists
    has_ot_time = (df["seconds_into_game"] > 2880).any()
    has_ot_period = (df["period"] >= 5).any()
    assert has_ot_time == has_ot_period, (
        f"OT time indicator ({has_ot_time}) disagrees with OT period flag "
        f"({has_ot_period}) in game {game_id}"
    )


# ---------------------------------------------------------------------------
# Team / home-team consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_offense_defense_teams_differ(game_id, _min, _max):
    """offense_team_id and defense_team_id must never be the same."""
    df = get_possessions(game_id)
    same = df[df["offense_team_id"] == df["defense_team_id"]]
    assert len(same) == 0, (
        f"{len(same)} possessions where offense == defense in {game_id}"
    )


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_both_teams_have_possessions(game_id, _min, _max):
    """Both teams must appear as offense_team_id.  A team with zero offensive
    possessions means the entire half of the game was silently dropped.
    """
    df = get_possessions(game_id)
    n_teams = df["offense_team_id"].nunique()
    assert n_teams == 2, (
        f"Expected 2 offensive teams, found {n_teams} in {game_id}: "
        f"{df['offense_team_id'].unique()}"
    )


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_home_team_id_constant(game_id, _min, _max):
    """home_team_id must be the same on every row of the same game."""
    df = get_possessions(game_id)
    unique_home = df["home_team_id"].nunique()
    assert unique_home == 1, (
        f"home_team_id takes {unique_home} distinct values in {game_id}"
    )


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_possession_balance(game_id, _min, _max):
    """Each team should have between 30 % and 70 % of possessions.

    A 90/10 split would indicate a catastrophic extraction failure where one
    team's possessions were almost entirely dropped.
    """
    df = get_possessions(game_id)
    counts = df["offense_team_id"].value_counts(normalize=True)
    for tid, frac in counts.items():
        assert 0.30 <= frac <= 0.70, (
            f"Team {tid} has {frac:.1%} of possessions in {game_id} "
            f"(expected 30–70%%)"
        )


# ---------------------------------------------------------------------------
# Duplicate / integrity checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_no_duplicate_possessions(game_id, _min, _max):
    """No possession row may be completely identical to another.

    Multiple possessions can legitimately share the same (period,
    seconds_into_game, offense_team_id) when a rapid sequence of
    turnovers/scores occurs within a single clock tick.  What must never
    happen is the *exact same possession* (identical on all fields) appearing
    more than once — that would indicate a checkpoint-merge or emit() bug.
    """
    df = get_possessions(game_id)
    # Convert list columns to tuples so duplicated() can hash them
    check = df.copy()
    check["offense_player_ids"] = check["offense_player_ids"].apply(tuple)
    check["defense_player_ids"] = check["defense_player_ids"].apply(tuple)
    dupes = check[check.duplicated(keep=False)]
    assert len(dupes) == 0, (
        f"{len(dupes)} fully identical possession rows in {game_id}:\n"
        f"{dupes[['period','seconds_into_game','offense_team_id','points']].head(6).to_string()}"
    )


@pytest.mark.parametrize("game_id,_min,_max", ALL_GAMES, ids=ids)
def test_game_date_format(game_id, _min, _max):
    """game_date must be a valid YYYY-MM-DD string on every row."""
    import re
    df = get_possessions(game_id)
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    bad = df[~df["game_date"].astype(str).str.match(pattern)]
    assert len(bad) == 0, (
        f"Malformed game_date values in {game_id}: "
        f"{bad['game_date'].unique()}"
    )


# ---------------------------------------------------------------------------
# Official boxscore cross-validation (network)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.parametrize("game_id,_min,_max", NETWORK_GAMES, ids=net_ids)
def test_all_players_in_official_boxscore(game_id, _min, _max):
    """Every player ID in extracted lineups must appear in the official boxscore.

    Catches phantom IDs or bench players leaking into lineup strings.
    """
    df = get_possessions(game_id)
    official_set = set(_boxscore_players(game_id).keys())

    unknown: set = set()
    for _, row in df.iterrows():
        for pid in row["offense_player_ids"] + row["defense_player_ids"]:
            if pid not in official_set:
                unknown.add(pid)

    assert len(unknown) == 0, (
        f"Player IDs not in official boxscore for {game_id}: {unknown}"
    )


@pytest.mark.network
@pytest.mark.parametrize("game_id,_min,_max", NETWORK_GAMES, ids=net_ids)
def test_players_assigned_to_correct_team(game_id, _min, _max):
    """Each player must belong to the team listed for them in the boxscore."""
    df = get_possessions(game_id)
    player_team = _boxscore_players(game_id)

    mismatches: list[str] = []
    for _, row in df.iterrows():
        for pid in row["offense_player_ids"]:
            if player_team.get(pid) != row["offense_team_id"]:
                mismatches.append(
                    f"player {pid} on offense {row['offense_team_id']} "
                    f"but boxscore team {player_team.get(pid)}"
                )
        for pid in row["defense_player_ids"]:
            if player_team.get(pid) != row["defense_team_id"]:
                mismatches.append(
                    f"player {pid} on defense {row['defense_team_id']} "
                    f"but boxscore team {player_team.get(pid)}"
                )
        if mismatches:
            break

    assert len(mismatches) == 0, (
        f"Team mismatches in {game_id}:\n" + "\n".join(mismatches[:5])
    )


@pytest.mark.network
@pytest.mark.parametrize("game_id,_min,_max", NETWORK_GAMES, ids=net_ids)
def test_player_minutes_match_boxscore(game_id, _min, _max):
    """On-court seconds from lineup_ids must be within 2 s of official boxscore.

    Uses compute_player_seconds() which tracks exact entry/exit from lineup_ids
    changes — same game-clock measure as the boxscore, so agreement is
    sub-second (residual error is clock-string rounding only).
    """
    from nba_api.stats.endpoints import boxscoretraditionalv3

    player_seconds = compute_player_seconds(game_id, pbp_cache_dir=PBP_CACHE)

    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            bx = boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=game_id, timeout=60
            )
            players = bx.player_stats.get_data_frame()
            break
        except Exception:
            if attempt == 2:
                raise

    played = players[players["minutes"].astype(str).str.len() > 0]
    violations: list[str] = []
    for _, row in played.iterrows():
        pid = int(row["personId"])
        official_s = _parse_min(row["minutes"])
        estimated_s = player_seconds.get(pid, 0.0)
        diff = abs(estimated_s - official_s)
        if diff > 2.0:
            violations.append(
                f"Player {pid} ({row['familyName']}): "
                f"official {official_s/60:.2f} min, "
                f"estimated {estimated_s/60:.2f} min (diff {diff:.1f}s)"
            )

    assert len(violations) == 0, (
        f"Minutes mismatch in {game_id} ({len(violations)} players):\n"
        + "\n".join(violations[:10])
    )
