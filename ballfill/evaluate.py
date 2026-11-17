"""Score gap-filling methods against held-out truth. Every number a method claims comes from here.

Two test sets, both on BAR-ATH itself:

A "real gaps"   Every tactical gap inside a half with an observed frame on both sides. Truth is
                the broadcast ball on the frames inside the gap where the broadcast saw it. The
                method sees the tactical ball outside the gap only. These are genuine losses, in
                genuinely hard moments: the test that matters. Caveat: the truth is itself a
                tracker (median 0.8 m from tactical where both see the ball, worse in the air).

B "synthetic"   Stretches of the observed tactical track hidden on purpose, lengths drawn from
                the real gap-length distribution. Many more samples, but placed where the ball
                happened to be visible, so they are easier than real gaps. Truth is the hidden
                tactical ball.

Methods with fitted parameters are scored two-fold by half (fit on one half, score the other),
so no frame scores a model that saw it.

    python -m ballfill.evaluate                # all registered methods, both sets
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.data import runs  # noqa: E402

FPS = 25
LENGTH_BUCKETS = ((1, 125, "<=5 s"), (126, 375, "5-15 s"), (376, 750, "15-30 s"),
                  (751, 10 ** 9, ">30 s"))


@dataclass
class Gap:
    a: int            # first hidden frame
    b: int            # one past the last hidden frame
    half: int
    truth_frames: np.ndarray   # frames inside [a, b) with truth
    truth: np.ndarray          # (k, 2) metres

    @property
    def length(self):
        return self.b - self.a


def real_gaps(d):
    """Test set A. Context ball = tactical only."""
    m = d.match
    seen = ~np.isnan(d.tac[:, 0])
    gaps = []
    for half in (1, 2):
        in_half = m.half == half
        for a, b in runs(~seen & in_half):
            if a == 0 or b >= d.n or not (in_half[a - 1] and in_half[b]):
                continue
            if not (seen[a - 1] and seen[b]):
                continue
            frames = np.arange(a, b)
            k = ~np.isnan(d.bc[frames, 0])
            if k.any():
                gaps.append(Gap(a, b, half, frames[k], d.bc[frames[k]].astype(np.float64)))
    return gaps


def synthetic_gaps(d, n_per_half=250, seed=0, min_len=25):
    """Test set B. Lengths resampled from the real gaps (>= min_len frames)."""
    m = d.match
    seen = ~np.isnan(d.tac[:, 0])
    real = np.array([b - a for h in (1, 2) for a, b in runs(~seen & (m.half == h))])
    real = real[real >= min_len]
    rng = np.random.default_rng(seed)
    # cumulative count of unseen frames, to test "fully observed span" in O(1)
    unseen_c = np.r_[0, np.cumsum(~seen)]
    gaps, taken = [], np.zeros(d.n, dtype=bool)
    for half in (1, 2):
        idx = np.flatnonzero(m.half == half)
        lo, hi = idx[0] + 1, idx[-1] - 1
        made, tries = 0, 0
        while made < n_per_half and tries < 200000:
            tries += 1
            length = int(rng.choice(real))
            a = int(rng.integers(lo, hi - length))
            b = a + length
            if unseen_c[b + 1] - unseen_c[a - 1] or taken[a - 1:b + 1].any():
                continue
            taken[a - 1:b + 1] = True
            frames = np.arange(a, b)
            gaps.append(Gap(a, b, half, frames, d.tac[frames].astype(np.float64)))
            made += 1
    return gaps


def context_ball(d, gap, which):
    """The ball a method is allowed to see while filling `gap`."""
    ball = d.tac.astype(np.float64).copy()
    if which == "B":
        ball[gap.a:gap.b] = np.nan
    return ball


def score(errors_by_gap, gaps):
    """Summary rows: overall (frame-weighted), and by gap length."""
    rows = []

    def summarise(name, sel):
        e = np.concatenate([errors_by_gap[i] for i in sel]) if sel else np.array([])
        if not len(e):
            return
        per_gap = np.array([np.mean(errors_by_gap[i]) for i in sel])
        rows.append(dict(bucket=name, gaps=len(sel), frames=len(e), median=np.median(e),
                         mean=e.mean(), within2=(e <= 2).mean(), within5=(e <= 5).mean(),
                         gap_mean_median=np.median(per_gap)))

    summarise("all", list(range(len(gaps))))
    for lo, hi, name in LENGTH_BUCKETS:
        summarise(name, [i for i, g in enumerate(gaps) if lo <= g.length <= hi])
    return rows


def run_method(d, gaps, which, method):
    """method(d, ball, gap, fold_half) -> (gap.length, 2). fold_half = the half being scored."""
    errors = []
    for g in gaps:
        ball = context_ball(d, g, which)
        est = method(d, ball, g)
        rel = g.truth_frames - g.a
        errors.append(np.linalg.norm(est[rel] - g.truth, axis=1))
    return errors


def print_table(title, results):
    print("\n" + title)
    print("{:<26}{:<9}{:>6}{:>8}{:>9}{:>8}{:>8}{:>8}".format(
        "method", "gaps", "n", "frames", "median", "mean", "<=2m", "<=5m"))
    for name, rows in results:
        for r in rows:
            print("{:<26}{:<9}{:>6}{:>8}{:>8.2f}m{:>7.2f}m{:>8.0%}{:>8.0%}".format(
                name if r["bucket"] == "all" else "", r["bucket"], r["gaps"], r["frames"],
                r["median"], r["mean"], r["within2"], r["within5"]))


def evaluate(methods, d=None, sets=("A", "B")):
    from ballfill.data import load_ball_data
    d = d if d is not None else load_ball_data()
    out = {}
    for which in sets:
        gaps = real_gaps(d) if which == "A" else synthetic_gaps(d)
        results = []
        for name, method in methods.items():
            if hasattr(method, "prepare"):
                method.prepare(d, which, gaps)
            results.append((name, score(run_method(d, gaps, which, method), gaps)))
        out[which] = results
        print_table("Test set {} -- {}: {} gaps, {} truth frames".format(
            which, "real gaps, broadcast truth" if which == "A" else "synthetic gaps",
            len(gaps), sum(len(g.truth) for g in gaps)), results)
    return out


if __name__ == "__main__":
    from ballfill import fill
    evaluate(fill.METHODS)

