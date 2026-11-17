"""Open-match support used by George's carrier/dead-ball training path.

This module is part of the observation model only.  It creates degraded open
match examples for fitting the missing-ball carrier/dead-ball models; it does
not read hackathon possession ground truth.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.data import BallData, runs  # noqa: E402
from ballfill.evaluate import Gap  # noqa: E402
from possession.load_data import Match  # noqa: E402


IDSSE_IDS = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")


@dataclass
class OpenBall:
    d: BallData
    truth: np.ndarray
    dead: np.ndarray
    match_id: str


_PROFILE = None


def _profile():
    global _PROFILE
    if _PROFILE is None:
        from possession.degrade import noise_profile
        from possession.load_data import load_match
        _PROFILE = noise_profile(load_match())
    return _PROFILE


def load_open_ball(match_id, seed=0):
    from possession.degrade import degrade
    from possession.open_data import load_open_match

    original = load_open_match(match_id)
    degraded = degrade(original, _profile(), seed=seed)
    n = original.n_frames
    match = Match(
        ball_xy=degraded.ball_xy,
        ball_speed=np.full(n, np.nan, np.float32),
        players_xy=degraded.players_xy,
        players_speed=np.full(degraded.players_xy.shape[:2], np.nan),
        track_team=degraded.track_team,
        half=original.half.astype(np.int8),
        graded=np.ones(n, dtype=bool),
        tags=pd.DataFrame(columns=["start_frame"]),
        homography=None,
        homography_valid=None,
        gt_public=None,
        pitch=original.pitch,
    )
    nan2 = np.full((n, 2), np.nan, dtype=np.float32)
    data = BallData(
        match=match,
        tac=degraded.ball_xy.astype(np.float32),
        bc=nan2,
        bc_raw=nan2,
        bc_live=np.zeros(n, bool),
        bc_players=None,
        bc_team=None,
        offset=None,
    )
    return OpenBall(
        d=data,
        truth=original.ball_xy.astype(np.float64),
        dead=original.labels == "D",
        match_id=match_id,
    )


def open_gaps(ob, live_only=True):
    """Return two-sided gaps, optionally restricted to live frames."""
    data, match = ob.d, ob.d.match
    seen = ~np.isnan(data.tac[:, 0])
    gaps = []
    for half in (1, 2):
        in_half = match.half == half
        for start, end in runs(~seen & in_half):
            if start == 0 or end >= data.n or not (in_half[start - 1] and in_half[end]):
                continue
            frames = np.arange(start, end)
            if live_only:
                frames = frames[~ob.dead[frames]]
            if len(frames):
                gaps.append(Gap(start, end, half, frames, ob.truth[frames]))
    return gaps
