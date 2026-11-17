"""One observed ball track from the two cameras, with a source code and an uncertainty per frame.

What the two tracks showed (BAR-ATH, live-verified broadcast frames):
  * Where both see the ball they agree to a median 0.75 m. Player positions from the two cameras
    agree to 0.28 m with no bias anywhere on the pitch, so the homographies are not the problem.
  * Disagreements over 3 m come in smooth runs (median 1.2 s) that peak exactly mid-run: balls in
    the air. 81% point along the ray from the cameras, which both sit at about (53, 116) -- behind
    the y = 68 touchline at halfway -- at heights 30 m (tactical) and 21.5 m (broadcast).
  * The sign of the disagreement is random, and each track stays within ~1.3 m of a straight line
    mid-flight: both trackers already correct for height, imperfectly. A naive ground projection
    of a 5 m-high ball would be 10-20 m off. So the disagreement is height-estimation error,
    which lives on the camera ray; the across-ray coordinate is reliable in both.

Rules, in order:
  agree      both seen, within AGREE_M: the mean.
  flight     both seen, in a run of disagreement bounded by agreement: across-ray coordinate from
             the mean of the two; along-ray coordinate from the straight constant-velocity path
             between the agreed frames that bound the run (a ball in flight moves in a straight
             line over the ground).
  disagree   both seen, disagreeing, but the run is not bounded by agreement: the mean, flagged.
  tac / bc   one camera only: that camera.
Single-frame teleports (a step over JUMP_M that is undone on the next frame) are interpolated over
(none on BAR-ATH: its 119 tactical and 80 broadcast >2 m steps are lasting re-acquisitions, almost
all where the other camera does not see the ball, so they are kept and covered by the edge sigma).

    python -m ballfill.fuse
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.data import runs  # noqa: E402

CAMERA_XY = np.array([53.2, 116.0])   # both cameras, horizontal position, metres (pitch frame)
AGREE_M = 3.0
JUMP_M = 2.0                          # one-frame step = 50 m/s: not a football
SOURCES = ("none", "agree", "flight", "disagree", "tac", "bc")
# 1-sigma position uncertainty per source, metres. agree: half the typical disagreement;
# single camera: the typical disagreement (each tracker's own error is not separable from the
# other's without a third reference); flight / disagree: the along-ray spread.
SIGMA = {"agree": 0.4, "flight": 1.5, "disagree": 3.0, "tac": 0.8, "bc": 0.8}
# Trackers are worse next to their own losses: within a second of a tactical gap the tracks
# disagree by 1.1-1.7 m (0.74 m elsewhere), within a second of a broadcast gap by up to 2.3 m.
# Single-camera sigma is inflated by EDGE_EXTRA * exp(-frames to that camera's gap / EDGE_DECAY).
EDGE_EXTRA = {"tac": 0.8, "bc": 1.5}
EDGE_DECAY = 10.0
# A step faster than JUMP_M per frame means one side of it is wrong (usually a re-acquisition
# after the tracker coasted on a prediction); when only one camera sees the ball nothing can say
# which side. Frames within JUMP_WINDOW of such a step get JUMP_EXTRA added to their sigma.
JUMP_WINDOW = 12
JUMP_EXTRA = 3.0


def frames_to_gap(seen):
    """(N,) frames from each seen frame to the nearest unseen one (inf if none)."""
    n = len(seen)
    idx = np.arange(n, dtype=np.float64)
    last = np.where(~seen, idx, -np.inf)
    last = np.maximum.accumulate(last)
    nxt = np.where(~seen, idx, np.inf)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    return np.minimum(idx - last, nxt - idx)


def remove_blips(xy):
    """Interpolate over single frames that jump away by > JUMP_M and come straight back."""
    xy = xy.astype(np.float64).copy()
    step_in = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    step_out = np.r_[step_in[1:], np.nan]
    bridge = np.r_[np.nan, np.linalg.norm(xy[2:] - xy[:-2], axis=1), np.nan]
    blip = (step_in > JUMP_M) & (step_out > JUMP_M) & (bridge < JUMP_M)
    i = np.flatnonzero(blip)
    xy[i] = (xy[i - 1] + xy[i + 1]) / 2
    return xy, blip


def fuse(d):
    """(ball (N, 2), source (N,) index into SOURCES, sigma (N,), report dict)."""
    tac, tac_blip = remove_blips(d.tac)
    bc, bc_blip = remove_blips(d.bc)
    t_ok, b_ok = ~np.isnan(tac[:, 0]), ~np.isnan(bc[:, 0])
    both = t_ok & b_ok
    sep = np.where(both, np.linalg.norm(tac - bc, axis=1), np.nan)
    agree = both & (sep <= AGREE_M)

    ball = np.full_like(tac, np.nan)
    source = np.zeros(len(tac), dtype=np.int8)
    ball[t_ok & ~b_ok], source[t_ok & ~b_ok] = tac[t_ok & ~b_ok], SOURCES.index("tac")
    ball[b_ok & ~t_ok], source[b_ok & ~t_ok] = bc[b_ok & ~t_ok], SOURCES.index("bc")
    mean = (tac + bc) / 2
    ball[both] = mean[both]
    source[agree] = SOURCES.index("agree")
    source[both & ~agree] = SOURCES.index("disagree")

    n_flight = 0
    for a, z in runs(both & ~agree):
        if a == 0 or z >= len(tac) or not (agree[a - 1] and agree[z]):
            continue
        p0, p1 = mean[a - 1], mean[z]
        s = (np.arange(a, z) - (a - 1)) / (z - (a - 1))
        line = p0 + (p1 - p0) * s[:, None]
        # split into along-ray / across-ray at each frame; keep the cameras' across-ray part
        ray = mean[a:z] - CAMERA_XY
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        across = np.c_[-ray[:, 1], ray[:, 0]]
        along_c = ((line - CAMERA_XY) * ray).sum(1)
        across_c = ((mean[a:z] - CAMERA_XY) * across).sum(1)
        ball[a:z] = CAMERA_XY + ray * along_c[:, None] + across * across_c[:, None]
        source[a:z] = SOURCES.index("flight")
        n_flight += 1

    sigma = np.full(len(tac), np.nan)
    for name, s in SIGMA.items():
        sigma[source == SOURCES.index(name)] = s
    for name, ok in (("tac", t_ok), ("bc", b_ok)):
        k = source == SOURCES.index(name)
        sigma[k] += EDGE_EXTRA[name] * np.exp(-frames_to_gap(ok)[k] / EDGE_DECAY)
    step = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1)]
    near_jump = np.zeros(len(ball), dtype=bool)
    for t in np.flatnonzero(step > JUMP_M):
        near_jump[max(t - JUMP_WINDOW, 0):t + JUMP_WINDOW] = True
    sigma[near_jump & (source > 0)] += JUMP_EXTRA
    report = dict(near_jump=int((near_jump & (source > 0)).sum()), tac_blips=int(tac_blip.sum()), bc_blips=int(bc_blip.sum()), flights=n_flight)
    return ball, source, sigma, report


if __name__ == "__main__":
    from ballfill.data import load_ball_data
    d = load_ball_data()
    ball, source, sigma, rep = fuse(d)
    g = d.match.graded
    print("graded frames {:,}".format(g.sum()))
    for i, name in enumerate(SOURCES):
        k = g & (source == i)
        print("  {:<9} {:>7,}  {:>6.1%}".format(name, k.sum(), k.mean() if False else k.sum() / g.sum()))
    print("observed frames within {} frames of an impossible step (sigma +{} m): {:,}".format(
        JUMP_WINDOW, JUMP_EXTRA, rep["near_jump"]))
    print("single-frame blips removed: tactical {tac_blips}, broadcast {bc_blips}; "
          "flights reconstructed: {flights}".format(**rep))

