"""Make clean open-data matches look like our automatic tracking, and give them our label semantics.

Two things stand between the IDSSE matches and a fair test of our method:

1. Their tracking is clean (ball always present, every player always tracked). Ours loses the ball
   in long blocks and drops players in and out. `degrade()` imposes our noise, with run lengths
   resampled from our own match (only noise statistics are taken from it -- never labels).

2. Their possession flag flips on any touch; our ground truth flips only on *controlled*
   possession. `controlled_labels()` converts one into the other with a rule fixed from the
   challenge text (CHALLENGE.md, "What does not count as a change of possession"): a spell of the
   other team that is too short to be control -- a touch, a deflection -- is given back to the
   team that had the ball before it. The threshold is a design choice stated up front, not
   something fitted to the public five minutes.
"""

import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from possession.open_data import OpenMatch  # noqa: E402

FPS = 25
# A spell this short or shorter, of the other team, is "a simple touch without control".
# 1 s = 25 frames: long enough to cover a touch or deflection, short enough to keep quick
# interceptions that are then lost again. Changeable, but choose it from football, not from the
# public ground truth.
MIN_CONTROL_S = 1.0
# Ball-track smoothing that makes DFL's ball look like ours, fitted to label-free statistics
# (ball speed and velocity-change percentiles overall and near players) of our match.
BALL_NOISE_M = 0.5
BALL_SMOOTH_FRAMES = 6


# -- noise statistics of our match --------------------------------------------------------------

def _runs(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def noise_profile(match):
    """Run-length samples (in frames) describing how our tracking drops out. No labels used."""
    graded = match.graded
    ball_seen = ~np.isnan(match.ball_xy[:, 0])
    ball_vis, ball_gap = [], []
    for h in (1, 2):
        in_half = match.half == h
        ball_vis.append(_runs(ball_seen & in_half))
        ball_gap.append(_runs(~ball_seen & in_half))

    # Players: a slot's visible run is how long the tracker holds someone. A slot's *gap* is not a
    # player's gap -- with 46 slots for 22 players the player is usually re-acquired in another
    # slot at once -- so gaps are instead sized to reproduce our per-frame coverage
    # (players tracked / 22), as exponential gaps with the matching mean.
    seen = ~np.isnan(match.players_xy[:, :, 0])
    player_vis = np.concatenate([_runs(seen[:, j] & graded) for j in range(seen.shape[1])])
    coverage = min(seen[graded].sum(axis=1).mean() / 22.0, 0.99)
    mean_gap = player_vis.mean() * (1 - coverage) / coverage
    rng = np.random.default_rng(0)
    player_gap = np.maximum(1, rng.exponential(mean_gap, 5000)).astype(np.int64)
    return {"ball_visible": np.concatenate(ball_vis), "ball_gap": np.concatenate(ball_gap),
            "player_visible": player_vis, "player_gap": player_gap, "player_coverage": coverage,
            "ball_missing_rate": float((~ball_seen[graded]).mean()),
            "gap_hazard": fit_gap_hazard(match)}


# -- where the ball goes missing ----------------------------------------------------------------
# On our match the ball does not vanish at random: gaps start more often when the ball has
# slowed, is near a line, and players are slow -- typically stoppages. A logistic model of
# "a gap starts at t" on label-free, backward-looking features of t-1, fitted on our match,
# places the open matches' gaps in the same kind of moments.

HAZARD_COLUMNS = ("last_line_m", "back_speed", "player_speed_1s", "d_last_nearest", "spread_x", "tracked")


def _hazard_inputs(match, base=None):
    from possession import features as F
    base = F.compute(match) if base is None else base
    ball = match.ball_xy.astype(np.float64)
    back = np.r_[np.full(3, np.nan), np.linalg.norm(ball[3:] - ball[:-3], axis=1) * FPS / 3]
    X = np.c_[np.clip(base["last_line_m"].values, -5, 20), np.clip(back, 0, 40),
              base["player_speed_1s"].values, np.clip(base["d_last_nearest"].values, 0, 20),
              base["spread_x"].values, base["tracked"].values]
    return np.vstack([np.full((1, X.shape[1]), np.nan), X[:-1]])   # features of t-1


def fit_gap_hazard(match):
    from sklearn.linear_model import LogisticRegression
    X = _hazard_inputs(match)
    seen = ~np.isnan(match.ball_xy[:, 0])
    start = np.r_[False, seen[:-1] & ~seen[1:]]
    sel = match.graded & np.r_[False, seen[:-1]] & ~np.isnan(X).any(axis=1)
    model = LogisticRegression(C=1.0, max_iter=1000).fit(X[sel], start[sel])
    return model


def _hazard_mask(match_with_players, ball_xy, profile, rng):
    """Visibility mask with gap starts drawn from the fitted hazard, lengths from our gaps."""
    from dataclasses import replace as _replace
    probe = _replace(match_with_players, ball_xy=ball_xy)
    X = _hazard_inputs(probe)
    ok = ~np.isnan(X).any(axis=1)
    p = np.zeros(len(X))
    p[ok] = profile["gap_hazard"].predict_proba(X[ok])[:, 1]
    gaps = profile["ball_gap"]
    target = profile["ball_missing_rate"]
    scale = 1.0
    for _ in range(4):                                  # calibrate the overall missing rate
        draws = rng.random(len(p))
        mask = np.ones(len(p), dtype=bool)
        t = 0
        while t < len(p):
            if draws[t] < p[t] * scale:
                length = int(rng.choice(gaps))
                mask[t:t + length] = False
                t += length
            else:
                t += 1
        missing = 1 - mask.mean()
        scale *= target / max(missing, 1e-3) * (1 - missing) / max(1 - target, 1e-3)
    return mask


def _renewal_mask(n, visible_runs, gap_runs, rng):
    """Boolean visibility over n frames: alternate visible / gap runs drawn from the samples."""
    mask = np.zeros(n, dtype=bool)
    pos = -int(rng.integers(0, max(1, int(np.median(visible_runs)))))   # random phase
    visible = bool(rng.random() < 0.6)
    while pos < n:
        length = int(rng.choice(visible_runs if visible else gap_runs))
        if visible:
            mask[max(pos, 0):max(pos + length, 0)] = True
        pos += max(length, 1)
        visible = not visible
    return mask


BALL_GAPS = "hazard"      # "hazard": gaps placed like ours; "random": renewal process


def degrade(open_match, profile, seed=0, ball=True, players=True, jitter_m=0.0,
            ball_noise_m=BALL_NOISE_M, ball_smooth_frames=BALL_SMOOTH_FRAMES, ball_gaps=None):
    """A copy of an OpenMatch with our match's dropout pattern imposed. Labels are untouched."""
    from scipy.ndimage import gaussian_filter1d

    ball_gaps = ball_gaps or BALL_GAPS
    rng = np.random.default_rng(seed)
    n = open_match.n_frames
    ball_xy = open_match.ball_xy.copy()
    players_xy = open_match.players_xy.copy()

    if ball_noise_m or ball_smooth_frames:
        # Our ball track is far smoother than DFL's (velocity change at a touch ~2 m/s vs ~8).
        # Noise then Gaussian smoothing reproduces our ball speed / velocity-change percentiles.
        noisy = ball_xy.astype(np.float64) + rng.normal(0, ball_noise_m, ball_xy.shape)
        if ball_smooth_frames:
            noisy = gaussian_filter1d(noisy, ball_smooth_frames, axis=0, mode="nearest")
        ball_xy = noisy.astype(np.float32)

    if players:
        on_pitch = ~np.isnan(players_xy[:, :, 0])
        for j in range(players_xy.shape[1]):
            keep = _renewal_mask(n, profile["player_visible"], profile["player_gap"], rng)
            players_xy[on_pitch[:, j] & ~keep, j] = np.nan

    if jitter_m:
        players_xy = players_xy + rng.normal(0, jitter_m, players_xy.shape).astype(np.float32)

    if ball:
        if ball_gaps == "hazard":
            keep = _hazard_mask(replace(open_match, players_xy=players_xy), ball_xy, profile, rng)
        else:
            keep = _renewal_mask(n, profile["ball_visible"], profile["ball_gap"], rng)
        ball_xy[~keep] = np.nan

    return replace(open_match, ball_xy=ball_xy, players_xy=players_xy)


# -- label semantics ----------------------------------------------------------------------------

def controlled_labels(labels, min_control_s=MIN_CONTROL_S):
    """DFL any-touch possession -> controlled possession, by the rule in the module docstring.

    A spell of team X that lasts <= min_control_s and comes right after a spell of the other team
    is given back to that team, whatever follows it (the other team again, or a dead ball). Dead
    ball is never changed. Applied left to right, so a run of short touches all revert.
    """
    labels = np.asarray(labels).copy()
    limit = int(round(min_control_s * FPS))
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    stops = np.r_[change, len(labels)]

    owner = None                      # team in controlled possession before the current spell
    for start, stop in zip(starts, stops):
        label = labels[start]
        if label == "D":
            owner = None              # restarts begin with a clear possessor
            continue
        if owner is not None and label != owner and stop - start <= limit:
            labels[start:stop] = owner
            continue
        owner = label
    return labels


def as_our_match(open_match, profile, seed=0, min_control_s=MIN_CONTROL_S):
    """Degraded tracking + controlled-possession labels: the stand-in for our match."""
    degraded = degrade(open_match, profile, seed=seed)
    return replace(degraded, labels=controlled_labels(open_match.labels, min_control_s))
