"""Turn raw video peaks into ball sightings, and fill around them.

1. Static look-alikes. A penalty spot, a spare ball by the touchline or a broadcast overlay gives a
   peak in the same place all half. Cells (1 m on the pitch, 16 px on the image) with peaks in many
   distinct 10 s windows are masked. A real resting ball stays a minute or two at most.
2. Tracklets. Peaks are linked frame to frame on the pitch (so camera pans don't matter), with a
   constant-velocity prediction and up to MAX_SKIP missed frames.
3. A tracklet becomes a sighting if it is long enough and confident enough (two thresholds, chosen
   two-fold by half on test set A).
4. VideoFill: inside each gap, frames with a sighting take it; the rest are sub-gaps for the HMM,
   now anchored by the video.

    python -m ballfill.vidfuse      # two-fold thresholds; A and B against the HMM alone
"""

import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.data import runs  # noqa: E402
from ballfill.video import IMG_H, IMG_W, load_detections, to_pitch  # noqa: E402

FPS = 25
MIN_SCORE = 0.05
STATIC_WINDOW = 10 * FPS
STATIC_PITCH_WINDOWS = 40
STATIC_PIXEL_WINDOWS = 60
LINK_M = 1.5          # per frame, beyond the constant-velocity prediction
MAX_SKIP = 3
EDGE_PX = 4           # peaks this close to the image border are cut-off blobs


def candidates(match, det):
    """Pitch xy (n, K, 2), score (n, K) with low, static and border peaks removed."""
    H = match.homography.astype(np.float64)
    uv = det[..., :2].astype(np.float64)
    s = det[..., 2].astype(np.float64).copy()
    xy = to_pitch(H[:, None], uv)
    ok = np.isfinite(s) & (s >= MIN_SCORE)
    ok &= (uv[..., 0] > EDGE_PX) & (uv[..., 0] < IMG_W - EDGE_PX)
    ok &= (uv[..., 1] > EDGE_PX) & (uv[..., 1] < IMG_H - EDGE_PX)
    frame = np.broadcast_to(np.arange(len(det))[:, None], s.shape)
    win = frame // STATIC_WINDOW
    static = np.zeros_like(ok)
    for half in (1, 2):
        inh = (match.half == half)[:, None] & ok
        for coord, cell, need in ((xy, 1.0, STATIC_PITCH_WINDOWS),
                                  (uv, 16.0, STATIC_PIXEL_WINDOWS)):
            c = np.floor(coord / cell).astype(np.int64)
            key = c[..., 0] * 100000 + c[..., 1]
            k, w = key[inh], win[inh]
            pairs = np.unique(np.c_[k, w], axis=0)
            cells, counts = np.unique(pairs[:, 0], return_counts=True)
            bad = cells[counts >= need]
            static |= inh & np.isin(key, bad)
    ok &= ~static
    s[~ok] = np.nan
    xy[~ok] = np.nan
    return xy, s, static


def tracklets(xy, s, frames):
    """Greedy constant-velocity linking over `frames` (sorted). Returns list of dicts."""
    active, done = [], []
    for f in frames:
        cand = np.flatnonzero(np.isfinite(s[f]))
        cand = cand[np.argsort(-s[f, cand])]
        used = set()
        for t in sorted(active, key=lambda t: -t["sum"] / len(t["f"])):
            dt = f - t["f"][-1]
            pred = t["xy"][-1] + t["v"] * dt
            best, bd = None, LINK_M * dt
            for j in cand:
                if j in used:
                    continue
                dd = np.linalg.norm(xy[f, j] - pred)
                if dd <= bd:
                    best, bd = j, dd
            if best is not None:
                used.add(best)
                p = xy[f, best]
                t["v"] = 0.5 * t["v"] + 0.5 * (p - t["xy"][-1]) / dt
                t["f"].append(f)
                t["xy"].append(p)
                t["s"].append(s[f, best])
                t["sum"] += s[f, best]
        for j in cand:
            if j not in used:
                active.append(dict(f=[f], xy=[xy[f, j]], s=[s[f, j]], v=np.zeros(2),
                                   sum=s[f, j]))
        keep = []
        for t in active:
            (keep if f - t["f"][-1] < MAX_SKIP + 1 else done).append(t)
        active = keep
    done += active
    out = []
    for t in done:
        out.append(dict(f=np.array(t["f"]), xy=np.array(t["xy"]), s=np.array(t["s"])))
    return out


def build_tracklets(d, det=None):
    """All tracklets over the graded halves."""
    det = load_detections(d.n) if det is None else det
    xy, s, _ = candidates(d.match, det)
    out = []
    for half in (1, 2):
        fr = np.flatnonzero(d.match.half == half)
        fr = fr[np.isfinite(s[fr]).any(1)]
        for t in tracklets(xy, s, fr):
            t["half"] = half
            out.append(t)
    return out


def sightings(tracks, n, min_len, min_mean):
    """(n, 2) ball from accepted tracklets (NaN elsewhere), and (n,) mean score of the tracklet.
    Frames skipped inside a tracklet are linearly interpolated."""
    ball = np.full((n, 2), np.nan)
    conf = np.zeros(n)
    for t in tracks:
        L = t["f"][-1] - t["f"][0] + 1
        ms = t["s"].mean()
        if L < min_len or ms < min_mean:
            continue
        fr = np.arange(t["f"][0], t["f"][-1] + 1)
        better = ms > conf[fr]
        if not better.any():
            continue
        x = np.interp(fr, t["f"], t["xy"][:, 0])
        y = np.interp(fr, t["f"], t["xy"][:, 1])
        ball[fr[better]] = np.c_[x, y][better]
        conf[fr[better]] = ms
    return ball, conf


class VideoFill:
    """Video sightings where present; the HMM (or any method) on what remains of each gap."""

    def __init__(self, method, video):
        self.method, self.video = method, video

    def __call__(self, d, ball, gap):
        ball = ball.copy()
        a, b = gap.a, gap.b
        v = self.video[a:b]
        seen = np.isfinite(v[:, 0])
        ball[a:b][seen] = v[seen]
        est = np.where(seen[:, None], v, np.nan)
        for s, e in runs(~seen):
            sub = replace(gap, a=a + s, b=a + e, truth_frames=np.array([], dtype=int),
                          truth=np.zeros((0, 2)))
            est[s:e] = self.method(d, ball, sub)
        self.last_seen = seen
        return est


GRID = dict(min_len=[3, 5, 10, 20], min_mean=[0.1, 0.2, 0.3, 0.4, 0.5])


def main():
    from ballfill.carrier import CarrierHMM
    from ballfill.data import load_ball_data
    from ballfill.evaluate import real_gaps, synthetic_gaps
    from ballfill.run import CONFIG
    from ballfill.tune import errors, summary, loss
    d = load_ball_data()
    tracks = build_tracklets(d)
    print("{} tracklets".format(len(tracks)))
    h = CarrierHMM(**CONFIG)
    h.set_source("BAR-ATH")
    h.prepare(d, "A", None)
    A = real_gaps(d)
    base = errors(d, A, "A", h)
    print("A  HMM alone          ", summary(base))
    by_half = {k: [i for i, g in enumerate(A) if g.half == k] for k in (1, 2)}
    table = {}
    for L in GRID["min_len"]:
        for m in GRID["min_mean"]:
            vid, _ = sightings(tracks, d.n, L, m)
            errs = errors(d, A, "A", VideoFill(h, vid))
            table[(L, m)] = errs
            f = np.concatenate([g.truth_frames for g in A])
            cov = np.isfinite(vid[f, 0]).mean()
            print("A  len>={:<3} mean>={:.1f}  video on {:5.1%} of truth frames  {}".format(
                L, m, cov, summary(errs)), flush=True)
    # two-fold: params chosen on one half's gaps, applied to the other's
    oof = [None] * len(A)
    for k in (1, 2):
        other = by_half[3 - k]
        best = min(table, key=lambda c: loss([table[c][i] for i in other]))
        print("half {} uses {}".format(k, best))
        for i in by_half[k]:
            oof[i] = table[best][i]
    print("A  video + HMM, two-fold ", summary(oof))
    B = synthetic_gaps(d)
    print("B  HMM alone          ", summary(errors(d, B, "B", h)))
    for c in sorted(set([min(table, key=lambda c: loss([table[c][i] for i in by_half[k]]))
                         for k in (1, 2)])):
        vid, _ = sightings(tracks, d.n, *c)
        print("B  video + HMM {}  {}".format(c, summary(errors(d, B, "B", VideoFill(h, vid)))))


if __name__ == "__main__":
    main()

