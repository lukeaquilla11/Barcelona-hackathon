"""Is this video tracklet the ball? A classifier trained where the tracker knows the answer.

A false tracklet (flag, glove, line marking) is rare but tens of metres wrong, so accepting video
sightings on score alone made real gaps worse on average. This model judges each tracklet from:
detector scores, its motion, its players (a ball is usually near one), whether it is on the pitch,
where it is in the image, and how fast the ball would have had to travel from the last sighting
before the gap and to the next one after.

Training: tracklets on frames where the tracker has the ball, in the *other* half (so test set B's
hidden truth is never seen). Label: follows the tracked ball within 2 m. To make the edge features
look like they do inside a real gap, the tracker's ball is hidden over a random window around each
training tracklet, with lengths drawn from the real gaps.

    python -m ballfill.verify      # held-out AUC; A and B with verified sightings, two-fold
"""

import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.data import runs  # noqa: E402
from ballfill.video import to_pixel  # noqa: E402

FPS = 25
MIN_SPAN = 3
LABEL_M = 2.0
L, W = 105.3, 68.0
FEATURES = ("span_s", "fill", "s_mean", "s_max", "s_min", "s_std", "speed_mean", "speed_max",
            "disp", "straight", "pl_mean", "pl_min", "pl_max", "near_frac", "crowd",
            "outside", "y_mid", "v_px", "dt_before", "dt_after", "v_before", "v_after",
            "d_before", "d_after", "rivals")


def track_features(t, P, Hs, before, after, rivals):
    """One tracklet -> feature vector. before/after = (dt seconds, xy) of the gap's edge sightings
    (dt >= 1 frame), P (span, 46, 2) players over the tracklet's detected frames."""
    f, xy, s = t["f"], t["xy"], t["s"]
    span = f[-1] - f[0] + 1
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(f), 1) * FPS \
        if len(f) > 1 else np.zeros(1)
    path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() if len(f) > 1 else 0.0
    disp = np.linalg.norm(xy[-1] - xy[0])
    dp = np.linalg.norm(P - xy[:, None], axis=2)
    near = np.nanmin(np.where(np.isfinite(dp), dp, np.inf), axis=1)
    near = np.where(np.isfinite(near), near, 50.0)
    crowd = (np.nan_to_num(dp, nan=1e9) < 5).sum(1).mean()
    out_x = np.maximum(np.maximum(-xy[:, 0], xy[:, 0] - L), 0)
    out_y = np.maximum(np.maximum(-xy[:, 1], xy[:, 1] - W), 0)
    outside = np.maximum(out_x, out_y).max()
    uv = to_pixel(Hs, xy)
    (dtb, pb), (dta, pa) = before, after
    db = np.linalg.norm(xy[0] - pb) if np.all(np.isfinite(pb)) else 50.0
    da = np.linalg.norm(xy[-1] - pa) if np.all(np.isfinite(pa)) else 50.0
    return np.array([span / FPS, len(f) / span, s.mean(), s.max(), s.min(), s.std(),
                     step.mean(), step.max(), disp, disp / max(path, 1e-6), near.mean(),
                     near.min(), near.max(), (near < 1.5).mean(), crowd, outside,
                     xy[:, 1].mean(), uv[:, 1].mean(), dtb, dta, db / dtb, da / dta, db, da,
                     rivals])


def _clip(t, a, b):
    k = (t["f"] >= a) & (t["f"] < b)
    if k.sum() < 1:
        return None
    return dict(f=t["f"][k], xy=t["xy"][k], s=t["s"][k], half=t.get("half"))


def _edges(ball, a, b, hidden=None):
    """Last sighting before frame a and first at/after b: ((dt_frames, xy), (dt_frames, xy)).
    `hidden` = (ha, hb): frames in [ha, hb) count as unseen too."""
    ha, hb = hidden if hidden is not None else (0, 0)

    def seen(k):
        return np.isfinite(ball[k, 0]) and not (ha <= k < hb)
    i = a - 1
    while i >= 0 and not seen(i):
        i -= 1
    j = b
    while j < len(ball) and not seen(j):
        j += 1
    pb = ball[i] if i >= 0 else np.array([np.nan, np.nan])
    pa = ball[j] if j < len(ball) else np.array([np.nan, np.nan])
    return (a - i, pb), (j - b + 1, pa)


def gap_tracklets(tracks, starts, a, b, max_span=None):
    """Tracklets overlapping [a, b), clipped to it, each with its rival count."""
    if max_span is None:
        max_span = max(t["f"][-1] - t["f"][0] + 1 for t in tracks) if tracks else 0
    lo = np.searchsorted(starts, a - max_span)
    out = []
    for t in tracks[lo:]:
        if t["f"][0] >= b:
            break
        if t["f"][-1] < a:
            continue
        c = _clip(t, a, b)
        if c is not None and c["f"][-1] - c["f"][0] + 1 >= MIN_SPAN:
            out.append(c)
    for c in out:
        c["rivals"] = sum(1 for o in out if o is not c and o["f"][0] <= c["f"][-1]
                          and o["f"][-1] >= c["f"][0])
    return out


def features_in_window(d, ball, tracks_w, a, b, hidden=None):
    """Feature rows for tracklets inside a window [a, b) with context `ball` (plus `hidden`)."""
    m = d.match
    H = m.homography.astype(np.float64)
    rows = []
    for t in tracks_w:
        before, after = _edges(ball, t["f"][0], t["f"][-1] + 1, hidden)
        # the edge sightings are outside the window by construction
        (dtb, pb), (dta, pa) = before, after
        before = (dtb / FPS, pb)
        after = (dta / FPS, pa)
        rows.append(track_features(t, m.players_xy[t["f"]].astype(np.float64), H[t["f"]],
                                   before, after, t["rivals"]))
    return np.array(rows).reshape(-1, len(FEATURES))


def sort_tracks(tracks):
    tracks = sorted(tracks, key=lambda t: t["f"][0])
    return tracks, np.array([t["f"][0] for t in tracks])


def training_rows(d, tracks, half, seed=0):
    """(X, y) from tracklets in `half` on frames where the fused ball is known."""
    from ballfill.fuse import fuse
    ball, src, sig, _ = fuse(d)
    known = np.isfinite(ball[:, 0]) & (sig <= 1.0)
    tac = d.tac.astype(np.float64)
    seen_tac = np.isfinite(tac[:, 0])
    real = np.array([b - a for h in (1, 2) for a, b in runs(~seen_tac & (d.match.half == h))])
    real = real[real >= 10]
    rng = np.random.default_rng(seed + half)
    X, y = [], []
    tracks, starts = sort_tracks([t for t in tracks if t["half"] == half])
    max_span = max(t["f"][-1] - t["f"][0] + 1 for t in tracks)
    hidx = np.flatnonzero(d.match.half == half)
    lo_h, hi_h = hidx[0] + 1, hidx[-1]
    for t in tracks:
        f0, f1 = t["f"][0], t["f"][-1] + 1
        if f1 - f0 < MIN_SPAN or known[f0:f1].mean() < 0.8:
            continue
        err = np.linalg.norm(t["xy"] - ball[t["f"]], axis=1)
        err = err[np.isfinite(err)]
        if not len(err):
            continue
        # a hidden window around the tracklet, as long as a real gap (or the tracklet)
        length = max(int(rng.choice(real)), f1 - f0)
        a = int(np.clip(f0 - rng.integers(0, length - (f1 - f0) + 1), lo_h, hi_h))
        b = min(a + length, hi_h)
        a, b = min(a, f0), max(b, f1)
        tw = gap_tracklets(tracks, starts, a, b, max_span)
        me = [c for c in tw if c["f"][0] == f0 and len(c["f"]) == len(t["f"])
              and np.allclose(c["xy"][0], t["xy"][0])]
        if not me:
            continue
        X.append(features_in_window(d, tac, me, a, b, hidden=(a, b))[0])
        y.append(np.median(err) <= LABEL_M)
    return np.array(X), np.array(y)


N_BAG = 5
ROW_SEEDS = (0, 10, 20)


class _Bag:
    """Average of seeded gradient-boosting models (deterministic: no random early stopping)."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(0)
    out = []
    for k in range(N_BAG):
        i = rng.integers(0, len(y), len(y))           # bootstrap rows
        out.append(HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=False, random_state=k).fit(X[i], y[i]))
    return _Bag(out)


class VerifiedVideoFill:
    """In each gap: verified tracklets (p >= threshold) are sightings, best p wins per frame;
    the HMM fills the rest of the gap between them."""

    def __init__(self, method, d, tracks, models, threshold):
        self.method, self.models, self.threshold = method, models, threshold
        self.tracks, self.starts = sort_tracks(tracks)
        self.max_span = max(t["f"][-1] - t["f"][0] + 1 for t in tracks)
        self.d = d

    def video_for_gap(self, d, ball, gap):
        tw = gap_tracklets(self.tracks, self.starts, gap.a, gap.b, self.max_span)
        v = np.full((gap.length, 2), np.nan)
        best = np.zeros(gap.length)
        if not tw:
            return v, best
        X = features_in_window(d, ball, tw, gap.a, gap.b)
        p = self.models[gap.half].predict_proba(X)[:, 1]
        for t, pi in zip(tw, p):
            if pi < self.threshold:
                continue
            fr = np.arange(t["f"][0], t["f"][-1] + 1)
            k = fr - gap.a
            better = pi > best[k]
            x = np.interp(fr, t["f"], t["xy"][:, 0])
            yy = np.interp(fr, t["f"], t["xy"][:, 1])
            v[k[better]] = np.c_[x, yy][better]
            best[k[better]] = pi
        return v, best

    def __call__(self, d, ball, gap):
        v, best = self.video_for_gap(d, ball, gap)
        seen = np.isfinite(v[:, 0])
        ball = ball.copy()
        ball[gap.a:gap.b][seen] = v[seen]
        est = v.copy()
        for s, e in runs(~seen):
            sub = replace(gap, a=gap.a + s, b=gap.a + e, truth_frames=np.array([], dtype=int),
                          truth=np.zeros((0, 2)))
            est[s:e] = self.method(d, ball, sub)
        self.last_seen, self.last_p = seen, best
        return est


THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def main():
    from sklearn.metrics import roc_auc_score
    from ballfill.carrier import CarrierHMM
    from ballfill.data import load_ball_data
    from ballfill.evaluate import real_gaps, synthetic_gaps
    from ballfill.run import CONFIG
    from ballfill.tune import errors, loss, summary
    from ballfill.vidfuse import build_tracklets
    d = load_ball_data()
    tracks = build_tracklets(d)
    rows = {}
    for hh in (1, 2):
        parts = [training_rows(d, tracks, hh, seed) for seed in ROW_SEEDS]
        rows[hh] = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    models = {}
    for h in (1, 2):
        Xtr, ytr = rows[3 - h]                       # model used on half h: trained on the other
        models[h] = fit(Xtr, ytr)
        Xte, yte = rows[h]
        p = models[h].predict_proba(Xte)[:, 1]
        print("half {}: train {} tracklets ({:.0%} ball), held-out AUC {:.3f}".format(
            h, len(ytr), ytr.mean(), roc_auc_score(yte, p)), flush=True)
    h = CarrierHMM(**CONFIG)
    h.set_source("BAR-ATH")
    h.prepare(d, "A", None)
    A = real_gaps(d)
    print("A  HMM alone            ", summary(errors(d, A, "A", h)), flush=True)
    by_half = {k: [i for i, g in enumerate(A) if g.half == k] for k in (1, 2)}
    table = {}
    for th in THRESHOLDS:
        vf = VerifiedVideoFill(h, d, tracks, models, th)
        table[th] = errors(d, A, "A", vf)
        print("A  verified p>={:.1f}       {}".format(th, summary(table[th])), flush=True)
    oof, chosen = [None] * len(A), {}
    for k in (1, 2):
        chosen[k] = min(table, key=lambda c: loss([table[c][i] for i in by_half[3 - k]]))
        for i in by_half[k]:
            oof[i] = table[chosen[k]][i]
    print("A  verified, two-fold {}  {}".format(chosen, summary(oof)), flush=True)
    B = synthetic_gaps(d)
    print("B  HMM alone            ", summary(errors(d, B, "B", h)), flush=True)
    tb = {}
    for th in THRESHOLDS:
        vf = VerifiedVideoFill(h, d, tracks, models, th)
        tb[th] = errors(d, B, "B", vf)
        print("B  verified p>={:.1f}       {}".format(th, summary(tb[th])), flush=True)
    th = min(tb, key=lambda c: loss(tb[c]))      # threshold chosen on B only, then scored on A
    print("A  verified, threshold from B ({:.1f})  {}".format(th, summary(table[th])), flush=True)

if __name__ == "__main__":
    main()

