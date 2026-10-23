"""
game_nba_api.py — Fallback possession extractor using nba_api PlayByPlayV3.

Used when pbpstats cannot extract a game (403s, empty data for 2025-26+,
unresolvable period starters). Two stages:

* **Fetch** (``fetch_raw``): PlayByPlayV3 actions, the BoxScoreTraditionalV3
  roster and (optionally) GameRotation stints, cached as gzip JSON under
  ``data/raw/nba_api/<game_id>.json.gz``. This is the only stage that touches
  stats.nba.com, which throttles aggressively.
* **Parse** (``parse_possessions``): a pure function of the cached data, so
  the possession logic can be fixed and re-run offline.

Lineups are per-player stint intervals; the five on court at a possession's
start are the stints covering that instant. They come from GameRotation when
it returned valid data, else from ``infer_stints`` (play-by-play
substitutions + box-score rosters). On 48 games with both, the inferred
lineups were identical on 100% of possessions and covered 99.3% of them.

Possession rules (each fixes a bug in the pre-2026-10-22 parser):

* A rebound is offensive iff the rebounding team is the team that shot.
  PlayByPlayV3 rebounds carry no type, and the ``(Off:x Def:y)`` in their
  description is the rebounder's running total, not this rebound's type.
* A made field goal switches possession only once any and-one free throw has
  been taken, so the and-one point stays with the shooter's possession.
* Technical free throws never change possession; after flagrant or
  clear-path free throws the fouled team keeps the ball.
* A shot, turnover or free throw by the team not currently on offense means
  a change of possession was missed (jump ball, unlogged event): the parser
  switches to that team and counts a ``self_correction``.

Points per possession are the offensive team's score delta over the
possession, so free throws and and-ones are captured. The running score is
tallied from the scoring events (made field goals' ``shotValue``, made free
throws), not the ``scoreHome``/``scoreAway`` fields, which occasionally
glitch; a free throw is missed iff its description starts with the word
"MISS" (player Yves Missi's makes start "Missi Free Throw").
"""

from __future__ import annotations

import gzip
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

from nba_api.stats.endpoints import boxscoretraditionalv3, gamerotation, playbyplayv3

RAW_DIR = Path("data/raw/nba_api")

_FT_OF_RE = re.compile(r"(\d+)\s+of\s+(\d+)")
_MISS_RE  = re.compile(r"MISS\b", re.IGNORECASE)
_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")

#: Actions that never resolve a pending post-basket possession change.
_NEUTRAL = {"Foul", "Substitution", "Timeout", "Instant Replay", "Violation",
            "Stoppage", ""}


# ---------------------------------------------------------------------------
# Time helpers (inlined to avoid circular import with game.py)
# ---------------------------------------------------------------------------

def _clock_to_sr(clock: str) -> float:
    m = _CLOCK_RE.match(clock)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    parts = clock.split(":")
    return float(parts[0]) * 60 + float(parts[1])


def _seconds_into_game(period: int, clock: str) -> float:
    sr = _clock_to_sr(clock)
    if period <= 4:
        period_start = (period - 1) * 12 * 60
        period_len   = 12 * 60
    else:
        period_start = 4 * 12 * 60 + (period - 5) * 5 * 60
        period_len   = 5 * 60
    return period_start + (period_len - sr)


# ---------------------------------------------------------------------------
# Fetch (network, cached)
# ---------------------------------------------------------------------------

def reset_http_session() -> None:
    """Drop nba_api's pooled connection. After a throttling stall, requests on
    the old keep-alive connection keep hanging while a fresh one succeeds."""
    from nba_api.stats.library.http import NBAStatsHTTP

    old = NBAStatsHTTP._session
    NBAStatsHTTP.set_session(None)
    if old is not None:
        old.close()


def _with_retries(fn, max_retries: int, base_sleep: float):
    for attempt in range(max_retries):
        try:
            time.sleep(base_sleep * (1 + 10 * attempt))
            return fn()
        except Exception:
            reset_http_session()
            if attempt == max_retries - 1:
                raise


def rotation_ok(rec: dict) -> bool:
    """True if a raw record's rotation covers the whole game: two teams, no
    negative stints, and each team's stints summing to 5 × game length
    (GameRotation sometimes returns partial or garbled data)."""
    rotation, actions = rec.get("rotation"), rec.get("actions") or []
    if not rotation or not actions:
        return False
    periods = max(int(a.get("period", 1)) for a in actions)
    length = (4 * 7200 + max(0, periods - 4) * 3000)  # tenths of seconds
    per_team: dict[int, float] = {}
    for team_id, _, in_t, out_t in rotation:
        if out_t < in_t:
            return False
        per_team[team_id] = per_team.get(team_id, 0.0) + (out_t - in_t)
    return len(per_team) == 2 and all(abs(v - 5 * length) <= 0.01 * 5 * length
                                      for v in per_team.values())


def fetch_raw(
    game_id: str,
    raw_dir: Path = RAW_DIR,
    max_retries: int = 3,
    base_sleep: float = 1.5,
    rotation_retries: Optional[int] = None,
    rotation_timeout: float = 60,
    timeout: float = 60,
) -> dict:
    """PlayByPlayV3 actions, box-score roster and GameRotation stints for one
    game, cached.

    Returns {"game_id", "actions": [...], "box": [[team_id, player_id, first,
    family, name_i, starter, minutes], ...] or None, "rotation": [[team_id,
    player_id, in_tenths, out_tenths], ...] or None}. Missing pieces of a
    cached record are filled on later calls; nothing valid is refetched.
    Raises if play-by-play cannot be fetched.

    GameRotation is optional (lineups can be inferred from play-by-play +
    box score) and unreliable: a cold request hangs ~30 s server-side, then
    returns HTTP 500, and the error is cached for a while. Bulk callers pass
    ``rotation_retries=1`` with a short ``rotation_timeout``, or 0 to skip it.
    """
    rotation_retries = max_retries if rotation_retries is None else rotation_retries
    path = Path(raw_dir) / f"{game_id}.json.gz"
    rec = None
    if path.exists():
        with gzip.open(path, "rt") as fh:
            rec = json.load(fh)
        if rec.get("box") and (rotation_ok(rec) or rotation_retries == 0):
            return rec
    changed = False
    if rec is None:
        actions = _with_retries(
            lambda: playbyplayv3.PlayByPlayV3(
                game_id=game_id, start_period=1, end_period=10, timeout=timeout,
            ).get_dict()["game"]["actions"],
            max_retries, base_sleep)
        rec = {"game_id": game_id, "actions": actions, "box": None, "rotation": None}
        changed = True
    if not rec.get("box"):
        try:
            df = _with_retries(
                lambda: boxscoretraditionalv3.BoxScoreTraditionalV3(
                    game_id=game_id, timeout=min(timeout, 30)).get_data_frames()[0],
                max_retries, base_sleep)
            rec["box"] = [
                [int(r.teamId), int(r.personId), str(r.firstName or ""), str(r.familyName or ""),
                 str(r.nameI or ""), bool(str(r.position or "").strip()), str(r.minutes or "")]
                for r in df.itertuples()
            ] or None
            changed = True
        except Exception:
            pass
    if not rotation_ok(rec) and rotation_retries > 0:
        try:
            frames = _with_retries(
                lambda: gamerotation.GameRotation(
                    game_id=game_id, timeout=rotation_timeout).get_data_frames(),
                rotation_retries, base_sleep)
            fetched = [
                [int(r.TEAM_ID), int(r.PERSON_ID), float(r.IN_TIME_REAL), float(r.OUT_TIME_REAL)]
                for f in frames for r in f.itertuples()
            ]
            if rotation_ok({"rotation": fetched, "actions": rec["actions"]}):
                rec["rotation"] = fetched
                changed = True
        except Exception:
            pass
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as fh:
            json.dump(rec, fh)
    return rec


def _stints_by_team(rotation: Optional[list]) -> Optional[dict[int, list[tuple[float, float, int]]]]:
    if not rotation:
        return None
    stints: dict[int, list[tuple[float, float, int]]] = {}
    for team_id, player_id, in_t, out_t in rotation:
        stints.setdefault(int(team_id), []).append((float(in_t), float(out_t), int(player_id)))
    return stints


def _lineup_at(stints: list[tuple[float, float, int]], t_tenths: float) -> list[int]:
    """Player ids on court at game time *t_tenths* (tenths of seconds)."""
    return [pid for in_t, out_t, pid in stints if in_t <= t_tenths < out_t]


# ---------------------------------------------------------------------------
# Lineups inferred from play-by-play (when GameRotation is unavailable)
# ---------------------------------------------------------------------------

_SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$", re.IGNORECASE)


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

#: Letters NFKD does not decompose to ASCII (Đurišić would become "urisic").
_TRANSLIT = str.maketrans({"Đ": "D", "đ": "d", "Ø": "O", "ø": "o", "Ł": "L", "ł": "l",
                           "ß": "ss", "Æ": "AE", "æ": "ae", "Œ": "OE", "œ": "oe",
                           "Þ": "Th", "þ": "th", "ı": "i"})


def _norm_name(s: str) -> str:
    s = (s or "").translate(_TRANSLIT)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[.,'’\-]", " ", s.lower())
    return " ".join(s.split())


def _strip_suffix(key: str) -> str:
    """Box scores and play-by-play disagree on suffixes ("Martin" vs "Martin Jr.")."""
    words = key.split()
    while len(words) > 1 and words[-1] in _SUFFIXES:
        words.pop()
    return " ".join(words)


def _period_bounds(period: int) -> tuple[float, float]:
    """(start, end) of *period* in tenths of seconds of game time."""
    if period <= 4:
        start = (period - 1) * 7200
        return start, start + 7200
    start = 4 * 7200 + (period - 5) * 3000
    return start, start + 3000


def _minutes(s: str) -> float:
    if not s:
        return 0.0
    if ":" in s:
        m, sec = s.split(":")[:2]
        return float(m) + float(sec) / 60
    return float(s)


def infer_stints(
    actions: list[dict],
    box: list[list],
) -> tuple[dict[int, list[tuple[float, float, int]]], dict]:
    """Rotation-style stints {team_id: [(in, out, player_id), ...]} inferred
    from PlayByPlayV3 substitutions and box-score rosters.

    *box* rows are [team_id, player_id, first_name, family_name, name_i,
    starter, minutes]. Period starters: the box-score starters for period 1;
    afterwards, every player who appears in an event or is subbed out before
    being subbed in, topped up from the previous period's closing lineup when
    that is unambiguous. A team-period whose five starters cannot be
    determined gets no stints (its possessions are skipped downstream).
    Incoming substitutes are matched by normalized family name, then by
    initial + family name, ruling out players already on court.

    info: uncertain_periods, unresolved_subs, inconsistencies, and
    minutes_max_err / minutes_mae (stint minutes vs box-score minutes).
    """
    teams = sorted({int(r[0]) for r in box})
    team_of = {int(r[1]): int(r[0]) for r in box}
    by_key: dict[tuple[int, str], set[int]] = {}
    first_family: dict[int, list[tuple[int, str, str]]] = {t: [] for t in teams}
    # Exact names first; suffix-stripped names only as a fallback, so that
    # "Carter" (Jevon) and "Carter Jr." (Wendell) stay distinct on one team.
    by_stripped: dict[tuple[int, str], set[int]] = {}

    def _add(team: int, key: str, pid: int) -> None:
        if key:
            by_key.setdefault((team, key), set()).add(pid)
            by_stripped.setdefault((team, _strip_suffix(key)), set()).add(pid)

    # Some players are listed given-name-last (Yang Hansen: first "Hansen",
    # family "Yang"), and substitutions then use the box score's first name.
    by_first: dict[tuple[int, str], set[int]] = {}
    for team, pid, first, family, name_i, _, _ in box:
        for key in {_norm_name(family), _norm_name(name_i), _norm_name(f"{first} {family}")}:
            _add(int(team), key, int(pid))
        first_family[int(team)].append(
            (int(pid), _norm_name(first), _strip_suffix(_norm_name(family))))
        by_first.setdefault((int(team), _norm_name(first)), set()).add(int(pid))
    # Players missing from the box score still appear in the play-by-play.
    for a in actions:
        pid, team = int(a.get("personId") or 0), int(a.get("teamId") or 0)
        if team in first_family and pid and pid not in teams and a.get("playerName"):
            team_of.setdefault(pid, team)
            for key in {_norm_name(a["playerName"]), _norm_name(a.get("playerNameI") or "")}:
                _add(team, key, pid)
    box_starters = {t: [int(r[1]) for r in box if int(r[0]) == t and r[5]] for t in teams}

    info = {"uncertain_periods": 0, "unresolved_subs": 0, "inconsistencies": 0}
    stints: dict[int, list[tuple[float, float, int]]] = {t: [] for t in teams}

    def _resolve(team: int, name: str, exclude: set[int]) -> Optional[int]:
        key = _norm_name(name)
        for index, k in ((by_key, key), (by_stripped, _strip_suffix(key))):
            cands = index.get((team, k), set()) - exclude
            if len(cands) == 1:
                return next(iter(cands))
        # "Jal. Williams": a first-name prefix disambiguates shared family names.
        prefix, _, family = _strip_suffix(key).partition(" ")
        if family:
            cands = {pid for pid, first, fam in first_family.get(team, [])
                     if fam == family and first.startswith(prefix)} - exclude
            if len(cands) == 1:
                return next(iter(cands))
        # Last resort: the box score's first name, if no family name matches it.
        if not by_stripped.get((team, _strip_suffix(key))):
            cands = by_first.get((team, key), set()) - exclude
            if len(cands) == 1:
                return next(iter(cands))
        return None

    def _skip_for_starters(a: dict) -> bool:
        atype, sub = a.get("actionType", "") or "", a.get("subType", "") or ""
        return (atype in ("Ejection", "Timeout", "Instant Replay", "period")
                or "Technical" in sub or "Technical" in atype)

    periods = sorted({int(a.get("period", 1)) for a in actions})
    prev_end: dict[int, set[int]] = {t: set() for t in teams}
    for period in periods:
        acts = [a for a in actions if int(a.get("period", 1)) == period]
        p_start, p_end = _period_bounds(period)

        # 1. Who started the period?
        starters: dict[int, list[int]] = {t: [] for t in teams}
        subbed_in: dict[int, set[int]] = {t: set() for t in teams}
        for a in acts:
            pid = int(a.get("personId") or 0)
            if a.get("actionType") == "Substitution":
                team = int(a.get("teamId") or 0)
                if team not in starters:
                    continue
                if pid in team_of and pid not in subbed_in[team] and pid not in starters[team]:
                    starters[team].append(pid)
                m = _SUB_RE.match(a.get("description", "") or "")
                if m:
                    inn = _resolve(team, m.group(1), set())
                    subbed_in[team] |= ({inn} if inn else by_stripped.get(
                        (team, _strip_suffix(_norm_name(m.group(1)))), set())
                        or by_first.get((team, _norm_name(m.group(1))), set()))
            elif pid in team_of and not _skip_for_starters(a):
                team = team_of[pid]
                if pid not in subbed_in[team] and pid not in starters[team]:
                    starters[team].append(pid)

        lineup: dict[int, Optional[dict[int, float]]] = {}
        for t in teams:
            s = starters[t]
            if period == 1 and len(box_starters[t]) == 5:
                s = box_starters[t]
            elif len(s) < 5:
                cands = prev_end[t] - set(s) - subbed_in[t]
                if len(s) + len(cands) == 5:
                    s = s + sorted(cands)
            if len(s) == 5:
                lineup[t] = {pid: p_start for pid in s}
            else:
                lineup[t] = None
                info["uncertain_periods"] += 1

        # 2. Replay the substitutions.
        for a in acts:
            if a.get("actionType") != "Substitution":
                continue
            team = int(a.get("teamId") or 0)
            on = lineup.get(team)
            if on is None:
                continue
            t = _seconds_into_game(period, a.get("clock", "PT00M00.00S")) * 10.0
            out = int(a.get("personId") or 0)
            m = _SUB_RE.match(a.get("description", "") or "")
            inn = _resolve(team, m.group(1), set(on)) if m else None
            if out in on:
                stints[team].append((on.pop(out), t, out))
            else:
                info["inconsistencies"] += 1
            if inn is None:
                info["unresolved_subs"] += 1
            elif inn in on:
                info["inconsistencies"] += 1
            else:
                on[inn] = t
        for t in teams:
            on = lineup[t]
            if on is None:
                prev_end[t] = set()
                continue
            for pid, t0 in on.items():
                stints[t].append((t0, p_end, pid))
            prev_end[t] = set(on)

    # Stint minutes vs box-score minutes.
    played: dict[int, float] = {}
    for t in teams:
        for in_t, out_t, pid in stints[t]:
            played[pid] = played.get(pid, 0.0) + (out_t - in_t) / 600
    errs = [abs(played.get(int(r[1]), 0.0) - _minutes(r[6])) for r in box]
    info["minutes_max_err"] = max(errs) if errs else None
    info["minutes_mae"] = sum(errs) / len(errs) if errs else None
    return stints, info


# ---------------------------------------------------------------------------
# Parse (pure)
# ---------------------------------------------------------------------------

def parse_possessions(
    game_id: str,
    meta: dict,
    actions: list[dict],
    stints: Optional[dict[int, list[tuple[float, float, int]]]],
) -> tuple[list[dict], dict]:
    """Possession records from PlayByPlayV3 actions (see module docstring).

    *stints* maps team_id → rotation stints; with ``None`` the possessions are
    segmented but lineups are left empty (for diagnostics only).
    Returns (possessions, info) where info counts ``skipped_lineup`` and
    ``self_corrections``.
    """
    home_id    = meta["home_team_id"]
    away_id    = meta["away_team_id"]
    both_teams = {home_id, away_id}

    def _other(t: int) -> int:
        return away_id if t == home_id else home_id

    possessions: list[dict] = []
    info = {"skipped_lineup": 0, "self_corrections": 0}

    poss_team: Optional[int] = None
    start_period, start_sig, start_diff = 1, 0.0, 0
    start_home = start_away = 0
    home = away = 0
    awaiting_rebound = False
    pending_to: Optional[int] = None      # team due the ball after a made FG
    pending_shooter: Optional[int] = None

    def _emit() -> None:
        if poss_team is None:
            return
        d_team = _other(poss_team)
        if stints is not None:
            t = start_sig * 10.0
            o_players = _lineup_at(stints.get(poss_team, []), t)
            d_players = _lineup_at(stints.get(d_team, []), t)
            if (len(o_players) != 5 or len(d_players) != 5
                    or len(set(o_players)) != 5 or len(set(d_players)) != 5):
                info["skipped_lineup"] += 1
                return
        else:
            o_players, d_players = [], []
        pts = (home - start_home) if poss_team == home_id else (away - start_away)
        possessions.append({
            "game_id":            game_id,
            "game_date":          meta["game_date"],
            "period":             start_period,
            "seconds_into_game":  start_sig,
            "offense_team_id":    poss_team,
            "defense_team_id":    d_team,
            "offense_player_ids": list(o_players),
            "defense_player_ids": list(d_players),
            "points":             max(0, pts),
            "score_diff":         start_diff,
            "home_team_id":       home_id,
        })

    def _start(team: int, period: int, clock: str) -> None:
        nonlocal poss_team, start_period, start_sig, start_diff, start_home, start_away
        nonlocal awaiting_rebound, pending_to, pending_shooter
        poss_team    = team
        start_period = period
        start_sig    = _seconds_into_game(period, clock)
        start_home, start_away = home, away
        start_diff   = (home - away) if team == home_id else (away - home)
        awaiting_rebound = False
        pending_to = pending_shooter = None

    def _switch(team: int, period: int, clock: str) -> None:
        _emit()
        _start(team, period, clock)

    for a in actions:
        period   = int(a.get("period", 1))
        clock    = a.get("clock", "PT00M00.00S")
        atype    = a.get("actionType", "") or ""
        sub      = a.get("subType", "") or ""
        team     = int(a.get("teamId") or 0)
        person   = int(a.get("personId") or 0)
        desc     = a.get("description", "") or ""
        is_fg    = bool(a.get("isFieldGoal"))
        is_ft    = atype == "Free Throw"
        is_tech  = is_ft and "Technical" in sub
        # "MISS" as a word: player Yves Missi's makes read "Missi Free Throw ...".
        ft_made  = is_ft and not _MISS_RE.match(desc)

        # Score after this action, tallied from the scoring events themselves
        # (made FG shotValue, made FT = 1). The scoreHome/scoreAway fields
        # occasionally glitch (a wrong value, then a correction), which a
        # delta-based count would turn into phantom points. Applied only
        # after any possession change this action implies, so its points land
        # in the possession it belongs to.
        new_home, new_away = home, away
        scored = 0
        if atype == "Made Shot" and is_fg:
            scored = int(a.get("shotValue") or (3 if "3PT" in desc else 2))
        elif ft_made:
            scored = 1
        if scored and team in both_teams:
            if team == home_id:
                new_home += scored
            else:
                new_away += scored

        if atype == "period":
            home, away = new_home, new_away
            if sub == "end":
                _emit()
                poss_team = pending_to = pending_shooter = None
                awaiting_rebound = False
            continue
        if is_tech:
            home, away = new_home, new_away
            continue

        # A made FG hands the ball over once any and-one is done.
        if pending_to is not None:
            andone = is_ft and team == pending_shooter
            if not andone and atype not in _NEUTRAL:
                _switch(pending_to, period, clock)

        offensive_act = (atype in ("Made Shot", "Missed Shot") and is_fg) \
            or atype == "Turnover" or is_ft
        if team in both_teams and offensive_act:
            if poss_team is None:
                _start(team, period, clock)
            elif team != poss_team and pending_to is None:
                info["self_corrections"] += 1
                _switch(team, period, clock)

        home, away = new_home, new_away

        if atype == "Made Shot" and is_fg:
            pending_to, pending_shooter = _other(poss_team), poss_team
            awaiting_rebound = False

        elif atype == "Missed Shot" and is_fg:
            awaiting_rebound = True

        elif atype == "Rebound" and awaiting_rebound and poss_team is not None:
            reb_team = team if team in both_teams else person if person in both_teams else None
            if reb_team is None:
                reb_team = _other(poss_team)
            if reb_team != poss_team:
                _switch(reb_team, period, clock)
            awaiting_rebound = False

        elif atype == "Turnover" and poss_team is not None:
            _switch(_other(poss_team), period, clock)

        elif is_ft and poss_team is not None:
            m = _FT_OF_RE.search(sub)
            is_last = (m is None) or (m.group(1) == m.group(2))
            if not is_last:
                continue
            if pending_to is not None and team == pending_shooter:
                if ft_made:
                    _switch(pending_to, period, clock)
                else:
                    pending_to = pending_shooter = None
                    awaiting_rebound = True
            elif "Flagrant" in sub or "Clear Path" in sub:
                pass
            elif ft_made:
                _switch(_other(poss_team), period, clock)
            else:
                awaiting_rebound = True

    _emit()
    return possessions, info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def game_stints(rec: dict) -> tuple[Optional[dict], str, dict]:
    """Lineup stints for a raw record: GameRotation when valid, else inferred
    from play-by-play + box score. Returns (stints, source, inference_info)
    with source "rotation", "pbp" or "none"."""
    if rotation_ok(rec):
        return _stints_by_team(rec["rotation"]), "rotation", {}
    if rec.get("box"):
        stints, info = infer_stints(rec["actions"], rec["box"])
        return stints, "pbp", info
    return None, "none", {}


#: A game keeps its parsed possessions if at least this share has a lineup;
#: the rest (an unresolvable substitution, say) are dropped, not guessed.
MIN_KEPT_SHARE = 0.5


def get_game_possessions_nba_api(
    game_id: str,
    meta: dict,
    raw_dir: Path = RAW_DIR,
) -> list[dict]:
    """
    Possession records for one game via the nba_api fallback.

    Parameters
    ----------
    game_id : str
    meta    : dict with keys game_date, home_team_id, away_team_id
    raw_dir : cache for raw play-by-play, box score and rotation data

    Returns
    -------
    list[dict] — same schema as get_game_possessions() in game.py; empty when
    no lineup source is available or fewer than MIN_KEPT_SHARE of the
    possessions have a determinable lineup.
    """
    rec = fetch_raw(game_id, raw_dir, rotation_retries=1, rotation_timeout=10)
    stints, source, _ = game_stints(rec)
    if not stints:
        print(f"    {game_id}: no box score or rotation data — skipping (play-by-play cached)")
        return []
    possessions, info = parse_possessions(game_id, meta, rec["actions"], stints)
    skipped = info["skipped_lineup"]
    if skipped:
        print(f"    {game_id}: dropped {skipped} possession(s) without a determinable "
              f"lineup ({source} lineups)")
    if len(possessions) < MIN_KEPT_SHARE * (len(possessions) + skipped):
        print(f"    {game_id}: rejecting game — too few possessions with a lineup")
        return []
    return possessions
