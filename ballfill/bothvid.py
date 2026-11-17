"""Fuse both videos inside each gap: tactical-video tracklets (verified) + broadcast-video
tracklets, with the HMM bridging whatever neither camera sees.

Test sets:
- A: real tactical gaps, truth = broadcast tracking ball. The broadcast detector is not run on
  these frames (the broadcast tracking already has the ball there), so A measures the tactical
  video only.
- B: synthetic gaps over known frames.
- D: real stretches where the broadcast tracker lost the ball but one tactical tracker kept it
  (source "tac"); the tactical ball is hidden and is the truth. This is where the broadcast camera
  is in the same situation as in our real gaps, so D measures what the broadcast video adds.

Broadcast settings (score threshold, minimum tracklet length, which camera wins where both see a
ball) are chosen two-fold by half on D; the tactical threshold is the one chosen on B in
ballfill.verify (0.6).

    python -m ballfill.bothvid
"""

import os
import pickle
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.data import runs  # noqa: E402

FPS = 25
TAC_THRESHOLD = 0.6
B_MIN_SCORE = 0.05
B_STATIC_WINDOWS = 40          # 1 m pitch cells with peaks in this many distinct 10 s windows
D_MIN_LEN = 10
GRID = [(th, ml, prio) for th in (0.2, 0.3, 0.5, 0.7) for ml in (1, 5) for prio in ("tac", "bc")]
MODELS = os.path.join(os.path.dirname(HERE), "cache", "ballvid_verify_models.pkl")


def broadcast_tracklets(d):
    from ballfill import bvideo
    from ballfill.vidfuse import tracklets
    xy, s = bvideo.load(d.n)
    xy, s = xy.astype(np.float64), s.astype(np.float64)
    L, W = d.match.pitch
    ok = np.isfinite(s) & (s >= B_MIN_SCORE)
    ok &= (xy[..., 0] > -3) & (xy[..., 0] < L + 3) & (xy[..., 1] > -3) & (xy[..., 1] < W + 3)
    # static look-alikes (penalty spots, overlays mapped onto the pitch)
    frame = np.broadcast_to(np.arange(d.n)[:, None], s.shape)
    c = np.floor(np.nan_to_num(xy)).astype(np.int64)
    key = c[..., 0] * 1000 + c[..., 1]
    pairs = np.unique(np.c_[key[ok], (frame // (10 * FPS))[ok]], axis=0)
    cells, counts = np.unique(pairs[:, 0], return_counts=True)
    ok &= ~np.isin(key, cells[counts >= B_STATIC_WINDOWS])
    s[~ok], xy[~ok] = np.nan, np.nan
    out = []
    for half in (1, 2):
        fr = np.flatnonzero((d.match.half == half) & np.isfinite(s).any(1))
        for t in tracklets(xy, s, fr):
            t["half"] = half
            out.append(t)
    return out


def broadcast_sightings(btracks, n):
    """Per frame: best broadcast tracklet's (xy, mean score, length), for any threshold later."""
    xy = np.full((n, 2), np.nan)
    ms = np.zeros(n)
    ln = np.zeros(n, dtype=int)
    for t in btracks:
        fr = np.arange(t["f"][0], t["f"][-1] + 1)
        m = t["s"].mean()
        better = m > ms[fr]
        if better.any():
            p = np.c_[np.interp(fr, t["f"], t["xy"][:, 0]), np.interp(fr, t["f"], t["xy"][:, 1])]
            xy[fr[better]], ms[fr[better]], ln[fr[better]] = p[better], m, len(fr)
    return xy, ms, ln


def verifier_models(d, tracks):
    from ballfill.verify import ROW_SEEDS, fit, training_rows
    if os.path.exists(MODELS):
        with open(MODELS, "rb") as f:
            return pickle.load(f)
    rows = {}
    for h in (1, 2):
        parts = [training_rows(d, tracks, h, seed) for seed in ROW_SEEDS]
        rows[h] = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    models = {h: fit(*rows[3 - h]) for h in (1, 2)}
    with open(MODELS, "wb") as f:
        pickle.dump(models, f)
    return models


class BothFill:
    """Tactical verified sightings + broadcast sightings; HMM in between."""

    def __init__(self, method, vf, bsight, th_b, min_len, prio, use_tac=True, use_bc=True):
        self.method, self.vf, self.bsight = method, vf, bsight
        self.th_b, self.min_len, self.prio = th_b, min_len, prio
        self.use_tac, self.use_bc = use_tac, use_bc
        self.tac_cache = {}

    def __call__(self, d, ball, gap):
        n = gap.length
        key = (gap.a, gap.b)
        if self.use_tac:
            if key not in self.tac_cache:
                self.tac_cache[key] = self.vf.video_for_gap(d, ball, gap)[0]
            vt = self.tac_cache[key]
        else:
            vt = np.full((n, 2), np.nan)
        vb = np.full((n, 2), np.nan)
        if self.use_bc:
            bxy, bms, bln = self.bsight
            fr = slice(gap.a, gap.b)
            ok = (bms[fr] >= self.th_b) & (bln[fr] >= self.min_len)
            vb[ok] = bxy[fr][ok]
        first, second = (vt, vb) if self.prio == "tac" else (vb, vt)
        v = np.where(np.isfinite(first[:, :1]), first, second)
        seen = np.isfinite(v[:, 0])
        ball = ball.copy()
        ball[gap.a:gap.b][seen] = v[seen]
        est = v.copy()
        for s, e in runs(~seen):
            sub = replace(gap, a=gap.a + s, b=gap.a + e, truth_frames=np.array([], dtype=int),
                          truth=np.zeros((0, 2)))
            est[s:e] = self.method(d, ball, sub)
        self.last_seen = seen
        return est


def d_gaps(d):
    """Test set D: runs of tactical-only frames (broadcast lost the ball), seen on both sides."""
    from ballfill.evaluate import Gap
    from ballfill.fuse import fuse
    ball, src, _, _ = fuse(d)
    gaps = []
    for half in (1, 2):
        inh = d.match.half == half
        for a, b in runs((src == 4) & inh):
            if b - a < D_MIN_LEN or not (inh[a - 1] and inh[b]):
                continue
            if not (np.isfinite(ball[a - 1, 0]) and np.isfinite(ball[b, 0])):
                continue
            fr = np.arange(a, b)
            gaps.append(Gap(a, b, half, fr, ball[fr].astype(np.float64)))
    return gaps


def main():
    from ballfill.evaluate import real_gaps, synthetic_gaps
    from ballfill.run import make_hmm
    from ballfill.tune import errors, loss, summary
    from ballfill.verify import VerifiedVideoFill
    from ballfill.vidfuse import build_tracklets
    from ballfill.data import load_ball_data
    d = load_ball_data()
    hmm = make_hmm()
    hmm.prepare(d, "A", None)
    tracks = build_tracklets(d)
    models = verifier_models(d, tracks)
    vf = VerifiedVideoFill(hmm, d, tracks, models, TAC_THRESHOLD)
    btracks = broadcast_tracklets(d)
    bsight = broadcast_sightings(btracks, d.n)
    print("broadcast tracklets: {}".format(len(btracks)), flush=True)
    D = d_gaps(d)
    print("D: {} gaps, {} frames ({:.0f} s)".format(len(D), sum(g.length for g in D),
                                                    sum(g.length for g in D) / FPS), flush=True)
    by_half = {k: [i for i, g in enumerate(D) if g.half == k] for k in (1, 2)}
    print("D  HMM alone              ", summary(errors(d, D, "B", hmm)), flush=True)
    tac_only = BothFill(hmm, vf, bsight, 1.0, 1, "tac", use_bc=False)
    print("D  tactical video only    ", summary(errors(d, D, "B", tac_only)), flush=True)
    table = {}
    for th, ml, prio in GRID:
        f = BothFill(hmm, vf, bsight, th, ml, prio)
        f.tac_cache = tac_only.tac_cache
        table[(th, ml, prio)] = errors(d, D, "B", f)
        print("D  both: bc>={:.1f} len>={} first={:<3} {}".format(
            th, ml, prio, summary(table[(th, ml, prio)])), flush=True)
    oof, chosen = [None] * len(D), {}
    for k in (1, 2):
        chosen[k] = min(table, key=lambda c: loss([table[c][i] for i in by_half[3 - k]]))
        for i in by_half[k]:
            oof[i] = table[chosen[k]][i]
    print("D  both, two-fold {}  {}".format(chosen, summary(oof)), flush=True)
    best = min(table, key=lambda c: loss(table[c]))
    bo = BothFill(hmm, vf, bsight, best[0], best[1], best[2], use_tac=False)
    print("D  broadcast video only {} {}".format(best, summary(errors(d, D, "B", bo))), flush=True)
    for name, gaps, which in (("A", real_gaps(d), "A"), ("B", synthetic_gaps(d), "B")):
        print("{}  HMM alone              ".format(name), summary(errors(d, gaps, which, hmm)),
              flush=True)
        f = BothFill(hmm, vf, bsight, best[0], best[1], best[2])
        print("{}  both {}  {}".format(name, best, summary(errors(d, gaps, which, f))), flush=True)


if __name__ == "__main__":
    main()

