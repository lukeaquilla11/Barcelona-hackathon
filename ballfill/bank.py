"""Carrier-classifier training rows for every match we have, computed once and cached.

Sources
    "BAR-ATH"     our match: frames where the tactical ball is seen (its own tracker as label)
    IDSSE ids     the 7 open matches, players degraded like ours, label from the clean DFL ball
                  on live frames only (DFL freezes the ball while dead)

Rows are (frame, half, X, y) with one row per present track slot. Pooling rule used everywhere:
a model that will score match M, half h is trained on every other match, plus -- for our match --
its other half. Nothing ever trains on the frames it scores.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill import carrier as C  # noqa: E402
from ballfill.data import CACHE_DIR  # noqa: E402

OUR = "BAR-ATH"
STRIDE = {OUR: C.STEP, "open": 10}
_MEM = {}


def _rows(match, ball, frames, usable):
    feats = C.compute_features_v2(match, frames)
    lab = C.carrier_labels(match, ball, frames)
    keep = usable & (lab >= -1)
    X = feats[keep]
    y = np.arange(X.shape[1])[None, :] == lab[keep][:, None]
    present = ~np.isnan(X[:, :, 0])
    fr = np.repeat(frames[keep][:, None], X.shape[1], 1)[present]
    return dict(frame=fr, half=match.half[fr], X=X[present].astype(np.float32), y=y[present])


def rows_for(source):
    if source in _MEM:
        return _MEM[source]
    path = os.path.join(CACHE_DIR, "ballfill_rows_{}.npz".format(source))
    if os.path.exists(path):
        _MEM[source] = dict(np.load(path))
        return _MEM[source]
    if source == OUR:
        from ballfill.data import load_ball_data
        d = load_ball_data()
        m = d.match
        frames = np.flatnonzero(m.graded)[::STRIDE[OUR]]
        r = _rows(m, d.tac.astype(np.float64), frames, ~np.isnan(d.tac[frames, 0]))
    else:
        from ballfill.opendata import load_open_ball
        ob = load_open_ball(source)
        m = ob.d.match
        frames = np.arange(0, m.n_frames, STRIDE["open"])
        r = _rows(m, ob.truth, frames, ~ob.dead[frames])
    np.savez(path, **r)
    _MEM[source] = r
    return r


def pooled(exclude_source, exclude_half=None, sources=None):
    """(X, y) from every source except `exclude_source`; for our match keep the other half."""
    from ballfill.opendata import IDSSE_IDS
    sources = sources or (OUR,) + IDSSE_IDS
    Xs, ys = [], []
    for s in sources:
        r = rows_for(s)
        if s == exclude_source:
            if exclude_half is None:
                continue
            k = r["half"] != exclude_half
            Xs.append(r["X"][k]), ys.append(r["y"][k])
        else:
            Xs.append(r["X"]), ys.append(r["y"])
    return np.concatenate(Xs), np.concatenate(ys)


if __name__ == "__main__":
    from ballfill.opendata import IDSSE_IDS
    for s in (OUR,) + IDSSE_IDS:
        r = rows_for(s)
        print("{:<8} rows {:>9,}  positives {:>7,}".format(s, len(r["y"]), int(r["y"].sum())))

