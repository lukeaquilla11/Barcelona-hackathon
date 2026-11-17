"""Is play stopped, frame by frame inside a gap? And where is a dead ball?

p_dead: a classifier on player movement (everyone slows down at a stoppage, the shape loosens)
and gap context (sightings at the lines, a next sighting that looks like a restart spot). Trained
on the open matches, whose DFL ball status gives dead/alive on every frame; our match has no such
labels outside the public 5 minutes, which may not be used for training.

Where the ball is while dead follows the laws of the game, not a fit:
  exit     where play stopped: the last sighting carried on with its velocity until it crosses a
           line (at most EXIT_S seconds), or the last sighting if it does not cross one. When the
           next sighting is the centre spot (a kick-off: the gap held a goal), the goal mouth the
           ball was heading for.
  restart  where play restarts: the next sighting (a throw-in, corner, goal kick or free kick is
           taken where the ball is next seen moving).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

FPS = 25
EXIT_S = 3.0
DEAD_FEATURES = ("speed_now", "speed_1s", "speed_3s", "slow_frac", "slow_frac_3s", "spread_x",
                 "spread_y", "n_tracked", "since_a", "until_b", "gap_len", "line0", "line1",
                 "out0", "out1", "centre1", "corner1", "v0", "v1", "frac_in_gap")


def _rolling(x, half):
    from possession.features import rolling_mean
    return rolling_mean(x, half)


def frame_stats(match):
    """Per-frame player statistics, (N, 7)."""
    P = match.players_xy
    v = np.full(P.shape, np.nan, dtype=np.float32)
    v[2:-2] = (P[4:] - P[:-4]) * (FPS / 4)
    spd = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(spd, axis=1)
        slow = np.nanmean(np.where(np.isnan(spd), np.nan, spd < 1.5), axis=1)
        sx, sy = np.nanstd(P[:, :, 0], axis=1), np.nanstd(P[:, :, 1], axis=1)
    n = (~np.isnan(P[:, :, 0])).sum(1).astype(np.float32)
    return np.c_[mean, _rolling(mean, FPS), _rolling(mean, 3 * FPS), slow,
                 _rolling(slow, 3 * FPS), sx, sy, n]


def _line(p, pitch):
    L, W = pitch
    return min(p[0], L - p[0], p[1], W - p[1])


def exit_point(ball, gap, pitch):
    """Where play stopped (see module docstring)."""
    from ballfill.carrier import edge_velocity
    L, W = pitch
    p0, p1 = ball[gap.a - 1], ball[gap.b]
    if np.hypot(p1[0] - L / 2, p1[1] - W / 2) < 3 and gap.length > 20 * FPS:
        return np.array([0.0 if p0[0] < L / 2 else L, W / 2])        # goal, then kick-off
    if _line(p0, pitch) < 0:
        return p0.copy()
    v0 = edge_velocity(ball, gap.a - 1, -1)
    for tau in np.arange(0, EXIT_S, 0.04):
        q = p0 + v0 * tau
        if _line(q, pitch) <= 0:
            return q
    return p0.copy()


def gap_features(stats, ball, gap, grid, pitch):
    """(len(grid), F) dead-ball features for frames `grid` of `gap`."""
    from ballfill.carrier import edge_velocity
    L, W = pitch
    p0, p1 = ball[gap.a - 1], ball[gap.b]
    corners = np.array([[0, 0], [0, W], [L, 0], [L, W]])
    ctx = np.array([_line(p0, pitch), _line(p1, pitch), float(_line(p0, pitch) < 0),
                    float(_line(p1, pitch) < 0), np.hypot(p1[0] - L / 2, p1[1] - W / 2),
                    np.min(np.linalg.norm(corners - p1, axis=1)),
                    np.linalg.norm(edge_velocity(ball, gap.a - 1, -1)),
                    np.linalg.norm(edge_velocity(ball, gap.b, +1))])
    since = (grid - (gap.a - 1)) / FPS
    until = (gap.b - grid) / FPS
    n = len(grid)
    return np.c_[stats[grid], since, until, np.full(n, gap.length / FPS),
                 np.repeat(ctx[None, :6], n, 0), np.repeat(ctx[None, 6:], n, 0),
                 since / (gap.length / FPS)].astype(np.float32)


def open_training(match_ids, step=5):
    """(X, y, match index) from gap frames of the degraded open matches."""
    from ballfill.opendata import load_open_ball, open_gaps
    Xs, ys, gs = [], [], []
    for i, mid in enumerate(match_ids):
        ob = load_open_ball(mid)
        stats = frame_stats(ob.d.match)
        ball = ob.d.tac.astype(np.float64)
        for g in open_gaps(ob, live_only=False):
            grid = np.arange(g.a, g.b, step)
            Xs.append(gap_features(stats, ball, g, grid, ob.d.match.pitch))
            ys.append(ob.dead[grid])
            gs.append(np.full(len(grid), i))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0).fit(X, y)


_MODELS = {}


def model_excluding(source):
    """Dead-ball classifier trained on every open match except `source`."""
    from ballfill.opendata import IDSSE_IDS
    if source not in _MODELS:
        ids = [m for m in IDSSE_IDS if m != source]
        X, y, _ = open_training(ids)
        _MODELS[source] = fit(X, y)
    return _MODELS[source]


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from ballfill.opendata import IDSSE_IDS
    X, y, g = open_training(IDSSE_IDS)
    print("gap frames {:,}, dead share {:.0%}".format(len(y), y.mean()))
    aucs, accs = [], []
    for i, mid in enumerate(IDSSE_IDS):
        model = fit(X[g != i], y[g != i])
        p = model.predict_proba(X[g == i])[:, 1]
        aucs.append(roc_auc_score(y[g == i], p))
        accs.append(((p > 0.5) == y[g == i]).mean())
        print("  held out {}: AUC {:.3f}, accuracy {:.1%}".format(mid, aucs[-1], accs[-1]))
    print("leave-one-match-out mean AUC {:.3f}, accuracy {:.1%}".format(np.mean(aucs), np.mean(accs)))

