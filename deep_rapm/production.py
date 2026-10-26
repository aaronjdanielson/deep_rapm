"""Frozen production configurations for the rolling RAPM.

``V1`` is the model behind the leaderboard and charts as of 2026-10-25. Its
values are frozen: changing any field makes a *new* configuration with a new
name, so that a score or a ranking reported for "v1" always means the same
model. Candidate models are built with ``V1.with_(name=..., ...)``.

Fields that v1 does not use (``alpha_def``, ``offseason_discount``,
``playoff_weight``, ``home_court``, ``comp_sigma``) default to "same as v1",
so a config that sets none of them *is* v1.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

SEASONS_ALL = tuple(f"20{y:02d}-{y + 1:02d}" for y in range(13, 27))   # 2013-14 … 2026-27


@dataclass(frozen=True)
class RapmConfig:
    name: str
    seasons: tuple[str, ...] = SEASONS_ALL
    alpha: float = 2000.0                 # ridge penalty on offensive coefficients
    alpha_def: float | None = None        # defensive penalty; None → same as alpha
    half_life_days: float = 365.0         # recency weight 0.5^(days / half_life)
    offseason_discount: float = 1.0       # extra multiplier on all weights per July 1 crossed
    comp_target_pct: float = 95.0         # the |score diff| percentile that receives …
    comp_target_weight: float = 0.05      # … this competition weight (sets σ)
    comp_sigma: float | None = None       # pin σ instead of calibrating it (inf = no competition weight)
    include_playoffs: bool = True
    include_preseason: bool = True
    preseason_weight: float = 0.20        # context weight on preseason possessions
    playoff_weight: float = 1.0           # context weight on playoff (and play-in) possessions
    home_court: bool = False              # add a ±1 home-offense column (penalty HOME_PENALTY)
    step_days: int = 7                    # eval-date spacing (reporting only)
    warmup_days: int = 180                # first eval date after the first possession (reporting only)
    min_poss: int = 100                   # players reported once n_off or n_def reaches this (reporting only)

    HOME_PENALTY = 1.0   # effectively unpenalized next to ~1M weighted possessions

    @property
    def alpha_def_eff(self) -> float:
        return self.alpha if self.alpha_def is None else self.alpha_def

    @property
    def n_extra_cols(self) -> int:
        return 1 if self.home_court else 0

    def penalty(self, n_players: int,
                alpha: float | None = None, alpha_def: float | None = None) -> np.ndarray:
        """Diagonal ridge penalty for [offense | defense | extra] coefficients."""
        a_off = self.alpha if alpha is None else alpha
        if alpha_def is not None:
            a_def = alpha_def
        elif alpha is not None:
            a_def = a_off
        else:
            a_def = self.alpha_def_eff
        pen = np.concatenate([np.full(n_players, a_off), np.full(n_players, a_def)])
        if self.home_court:
            pen = np.append(pen, self.HOME_PENALTY)
        return pen

    def calibrate_sigma(self, abs_score_diff: np.ndarray) -> float:
        """σ such that the comp_target_pct percentile of |score diff| gets comp_target_weight."""
        if self.comp_sigma is not None:
            return float(self.comp_sigma)
        d = float(np.percentile(abs_score_diff, self.comp_target_pct))
        return d / np.sqrt(-np.log(self.comp_target_weight))

    def with_(self, **changes) -> "RapmConfig":
        if "name" not in changes:
            raise ValueError("a modified config needs its own name")
        return replace(self, **changes)


V1 = RapmConfig(name="v1")
