"""
game.py — possession-level extraction for a single NBA game.

Uses pbpstats Enhanced PBP which provides:
  - event.lineup_ids: Dict[team_id, "pid1-pid2-pid3-pid4-pid5"] at every event
  - event.offense_team_id: which team has possession
  - event.is_possession_ending_event: True when offense_team_id changes
  - event.shot_value: 2 or 3 for field goals
  - event.is_made: True for made shots / free throws
  - event.home_score / event.away_score: cumulative score after the event

Output: one dict per possession with fields
  game_id, game_date, period, seconds_into_game,
  offense_team_id, defense_team_id,
  offense_player_ids (list[int], len=5),
  defense_player_ids (list[int], len=5),
  points (int, 0–4+),
  score_diff (int, offense − defense at possession start),
  home_team_id
"""

import re
import time
from pathlib import Path
from typing import Optional

import requests
from nba_api.stats.endpoints import boxscoresummaryv3
from pbpstats.data_loader import (
    DataNbaEnhancedPbpFileLoader,
    DataNbaEnhancedPbpLoader,
    DataNbaEnhancedPbpWebLoader,
)
from pbpstats.resources.enhanced_pbp.start_of_period import InvalidNumberOfStartersException

# Lazy import to avoid circular dependency; loaded only on 403 fallback
_nba_api_fallback = None

# ---------------------------------------------------------------------------
# Patch: data.nba.com requires browser-like headers (Akamai Bot Manager).
# Subclass the web loader to inject a User-Agent before every request.
# ---------------------------------------------------------------------------

_NBA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
}


class _PatchedEnhancedPbpWebLoader(DataNbaEnhancedPbpWebLoader):
    """DataNbaEnhancedPbpWebLoader with browser headers to bypass bot detection."""

    def _load_request_data(self):
        response = requests.get(self.url, headers=_NBA_HEADERS, timeout=60)
        if response.status_code == 200:
            self.source_data = response.json()
            self._save_data_to_file()
            return self.source_data
        response.raise_for_status()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


def _clock_to_seconds_remaining(clock: str) -> float:
    m = _CLOCK_RE.match(clock)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    # Fallback: MM:SS
    parts = clock.split(":")
    return float(parts[0]) * 60 + float(parts[1])


def _seconds_into_game(period: int, clock: str) -> float:
    sr = _clock_to_seconds_remaining(clock)
    if period <= 4:
        period_start = (period - 1) * 12 * 60
        period_len = 12 * 60
    else:
        period_start = 4 * 12 * 60 + (period - 5) * 5 * 60
        period_len = 5 * 60
    return period_start + (period_len - sr)


# ---------------------------------------------------------------------------
# Game metadata
# ---------------------------------------------------------------------------

def get_game_meta(game_id: str, max_retries: int = 3) -> dict:
    """Return {game_date, home_team_id, away_team_id}."""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            summ = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id)
            gs = summ.game_summary.get_data_frame().iloc[0]
            return {
                "game_date": gs["gameEt"].split("T")[0],
                "home_team_id": int(gs["homeTeamId"]),
                "away_team_id": int(gs["awayTeamId"]),
            }
        except Exception:
            if attempt == max_retries - 1:
                raise
    raise RuntimeError(f"get_game_meta failed for {game_id}")


# ---------------------------------------------------------------------------
# Possession extraction
# ---------------------------------------------------------------------------

def get_game_possessions(
    game_id: str,
    pbp_cache_dir: Optional[Path] = None,
    max_retries: int = 3,
    meta: Optional[dict] = None,
) -> list[dict]:
    """
    Extract possession-level records for one game.

    Parameters
    ----------
    game_id : str
        NBA game ID (e.g. "0022300154").
    pbp_cache_dir : Path, optional
        Directory for pbpstats file cache.  Pass None to disable caching.
    max_retries : int
        Retry attempts for API calls.
    meta : dict, optional
        {game_date, home_team_id, away_team_id} if already known; skips the
        BoxScoreSummaryV3 call, so a cached game can be re-parsed offline.

    Returns
    -------
    list[dict]
        One dict per possession.  Fields:

        game_id, game_date, period, seconds_into_game,
        offense_team_id, defense_team_id,
        offense_player_ids, defense_player_ids,
        points, score_diff, home_team_id

    Notes
    -----
    Score tracking
        home_score / away_score are updated whenever an event carries
        those fields (scoring events).  At possession start we snapshot
        the running totals before any points are added.

    Points accumulation
        Made field goals (+shot_value) and made free throws (+1) by the
        offensive team are summed.  Technical free throws shot by the
        *defensive* team are automatically excluded because we gate on
        event.team_id == offense_team_id.

    Lineup used
        We record the lineup at the *start* of the possession (first event
        after the previous possession ended) so substitutions that happen
        during a live ball are captured correctly.
    """
    if meta is None:
        meta = get_game_meta(game_id, max_retries=max_retries)

    pbp = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            # If a cache directory is provided and the file already exists,
            # use the file loader (no network call).  Otherwise fetch via web
            # and let pbpstats save the file to cache for next time.
            cache_str = str(pbp_cache_dir) if pbp_cache_dir is not None else None
            cached_path = None
            if pbp_cache_dir is not None:
                cached_path = Path(pbp_cache_dir) / "pbp" / f"data_{game_id}.json"

            if cached_path is not None and cached_path.exists():
                source_loader = DataNbaEnhancedPbpFileLoader(cache_str)
            else:
                # Ensure the pbp sub-directory exists before pbpstats tries
                # to write the downloaded JSON there.
                if pbp_cache_dir is not None:
                    (Path(pbp_cache_dir) / "pbp").mkdir(parents=True, exist_ok=True)
                source_loader = _PatchedEnhancedPbpWebLoader(cache_str)

            pbp = DataNbaEnhancedPbpLoader(game_id, source_loader)
            # data.nba.com silently returns empty data for recent seasons
            if not pbp.items:
                from deep_rapm.data.game_nba_api import get_game_possessions_nba_api
                return get_game_possessions_nba_api(game_id, meta)
            break
        except InvalidNumberOfStartersException:
            # pbpstats can't resolve lineups for unusual OT periods (ejections,
            # injuries, etc.) — fall back to our nba_api extractor which tracks
            # lineups via substitutions and doesn't rely on pbpstats' starter logic.
            from deep_rapm.data.game_nba_api import get_game_possessions_nba_api
            return get_game_possessions_nba_api(game_id, meta)
        except requests.exceptions.HTTPError as exc:
            # cdn.nba.com returns 403 for recent seasons — use nba_api fallback
            if exc.response is not None and exc.response.status_code == 403:
                from deep_rapm.data.game_nba_api import get_game_possessions_nba_api
                return get_game_possessions_nba_api(game_id, meta)
            if attempt == max_retries - 1:
                raise
        except Exception:
            if attempt == max_retries - 1:
                raise

    if pbp is None:
        return []

    possessions: list[dict] = []

    # Running cumulative score (updated whenever event carries hs/vs)
    running_home = 0
    running_away = 0

    # Possession state
    poss_offense_id: Optional[int] = None
    poss_lineup: Optional[dict] = None       # lineup_ids snapshot at start
    poss_start_sig: float = 0.0              # seconds_into_game at start
    poss_start_period: int = 1
    poss_points: int = 0
    poss_score_diff: int = 0                 # offense_score − defense_score at start

    def _emit() -> None:
        """Append current possession to results if lineup is valid."""
        if poss_offense_id is None or poss_lineup is None:
            return
        team_ids = [t for t in poss_lineup if poss_lineup[t]]
        if len(team_ids) < 2:
            return
        defense_id = next((t for t in team_ids if t != poss_offense_id), None)
        if defense_id is None:
            return
        o_str = poss_lineup.get(poss_offense_id, "")
        d_str = poss_lineup.get(defense_id, "")
        if not o_str or not d_str:
            return
        o_players = [int(p) for p in o_str.split("-")]
        d_players = [int(p) for p in d_str.split("-")]
        if len(o_players) != 5 or len(d_players) != 5:
            return
        possessions.append(
            {
                "game_id": game_id,
                "game_date": meta["game_date"],
                "period": poss_start_period,
                "seconds_into_game": poss_start_sig,
                "offense_team_id": poss_offense_id,
                "defense_team_id": defense_id,
                "offense_player_ids": o_players,
                "defense_player_ids": d_players,
                "points": poss_points,
                "score_diff": poss_score_diff,
                "home_team_id": meta["home_team_id"],
            }
        )

    def _reset() -> None:
        nonlocal poss_offense_id, poss_lineup, poss_start_sig
        nonlocal poss_start_period, poss_points, poss_score_diff
        poss_offense_id = None
        poss_lineup = None
        poss_start_sig = 0.0
        poss_start_period = 1
        poss_points = 0
        poss_score_diff = 0

    for event in pbp.items:
        etype = getattr(event, "event_type", None)
        offense_id = getattr(event, "offense_team_id", 0)
        lineup = getattr(event, "lineup_ids", {}) or {}
        team_id = getattr(event, "team_id", None)

        # Update running score whenever the event carries score fields. The
        # scores *before* this event are kept for the possession-start
        # snapshot below: event.home_score is the score after the event, so a
        # possession opened by a made shot would otherwise record the margin
        # including its own points (this bug affected every pbpstats-parsed
        # game until 2026-10-25; scripts/fix_score_diff.py repaired the data).
        hs = getattr(event, "home_score", None)
        vs = getattr(event, "away_score", None)
        prev_home, prev_away = running_home, running_away
        if hs is not None:
            running_home = int(hs)
        if vs is not None:
            running_away = int(vs)

        # Skip events with no valid offensive team or no lineup data
        if not offense_id or len(lineup) < 2:
            continue

        # ── Possession start ────────────────────────────────────────────────
        if poss_offense_id is None:
            poss_offense_id = offense_id
            poss_lineup = dict(lineup)
            poss_start_period = getattr(event, "period", 1)
            poss_start_sig = _seconds_into_game(
                poss_start_period,
                getattr(event, "clock", "0:00"),
            )
            # Score diff from the offense's perspective before this event's
            # points are added
            if poss_offense_id == meta["home_team_id"]:
                poss_score_diff = prev_home - prev_away
            else:
                poss_score_diff = prev_away - prev_home

        # ── Points accumulation ──────────────────────────────────────────────
        if team_id == poss_offense_id:
            if etype == 1:  # made field goal
                poss_points += getattr(event, "shot_value", 2)
            elif etype == 3:  # free throw
                if getattr(event, "is_made", False):
                    poss_points += 1

        # ── Possession end ───────────────────────────────────────────────────
        if getattr(event, "is_possession_ending_event", False):
            _emit()
            _reset()

    # Emit any trailing possession at end of game
    _emit()

    return possessions
