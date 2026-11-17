"""Produce the ball for every graded frame of the tactical video.

    python -m ballfill.run          # writes cache/BAR-ATH_ball.csv and prints a summary

Columns
    frame       tactical video frame (= submission row)
    x, y        metres, corner origin, x along the length (0..105.3), y across (0..68)
    source      agree | flight | disagree | tac | bc     observed (see ballfill.fuse)
                kickoff                                 the centre spot at a half's first frame
                video                                   inside a gap: the ball seen in the tactical
                                                        video (ballfill.video, verified tracklets,
                                                        ballfill.verify)
                filled                                  inside a gap, between sightings (see
                                                        ballfill.carrier)
    sigma_m     1-sigma uncertainty in metres. Observed: from the camera comparison. Video: the
                median out-of-sample error of video sightings on real gaps (test set A).
                Filled: the median out-of-sample error at that distance from the nearest
                sighting, the larger of real gaps (test set A) and the open matches (test set C).
    gap_s       for video and filled frames, the length of the gap in seconds
    p_video     for video frames, the verifier's probability the tracklet is the ball
    p_carrier   for filled frames, the HMM posterior of the most likely state (how sure it is)
    p_dead      for filled frames, the probability play is stopped (ballfill.dead)

Gaps run between observed frames. A gap that touches the start of a half is anchored at the
centre spot (every half starts with a kick-off there); one that touches the end of a half is
held at the last sighting. The output is derived from the licensed match data: it lives in
cache/, which is gitignored. Do not commit or share it.
"""

import os
import sys

import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.carrier import CarrierHMM  # noqa: E402
from ballfill.data import CACHE_DIR, load_ball_data, runs  # noqa: E402
from ballfill.evaluate import Gap, real_gaps  # noqa: E402
from ballfill.fuse import SOURCES, fuse  # noqa: E402

FPS = 25
# The configuration chosen by ballfill.ablate (selected on test set C, read off A and B).
# Rule: best on C, and no worse than the plain pooled HMM on A. Pass segments lost everywhere;
# the stoppage model only helps on our real gaps when it does not take weight off the players.
CONFIG = dict(flight_s=2.0, decode="mean", dead=True, switch=0.1, beta=0.1, static_e=0.03,
              pass_scale=5.0, dead_e=1.0, dead_players=False)
# Bins of seconds-to-nearest-sighting for sigma_m of filled frames.
SIGMA_EDGES_S = np.array([0, 0.5, 1, 2, 4, 8, 15, 30, 1e9])
SIGMA_C_PATH = os.path.join(CACHE_DIR, "ballfill_sigma_C.json")


def make_hmm():
    return CarrierHMM(**CONFIG)


def binned_median(dist, err):
    out = []
    for lo, hi in zip(SIGMA_EDGES_S[:-1], SIGMA_EDGES_S[1:]):
        k = (dist >= lo) & (dist < hi)
        out.append(float(np.median(err[k])) if k.sum() >= 200 else np.nan)
    return np.array(out)


def edge_errors(d, gaps, hmm, which):
    from ballfill.evaluate import context_ball
    dist, err = [], []
    for g in gaps:
        est = hmm(d, context_ball(d, g, which), g)
        err.append(np.linalg.norm(est[g.truth_frames - g.a] - g.truth, axis=1))
        dist.append(np.minimum(g.truth_frames - (g.a - 1), g.b - g.truth_frames) / FPS)
    return np.concatenate(dist), np.concatenate(err)


def sigma_from_open():
    """Out-of-sample error table from test set C (deep-gap live truth). Cached: slow."""
    if os.path.exists(SIGMA_C_PATH):
        return np.array(json.load(open(SIGMA_C_PATH)), dtype=float)
    from ballfill.opendata import IDSSE_IDS, load_open_ball, open_gaps
    hmm = make_hmm()
    dist, err = [], []
    for mid in IDSSE_IDS:
        ob = load_open_ball(mid)
        hmm.set_source(mid)
        hmm.prepare(ob.d, "C", None)
        di, er = edge_errors(ob.d, open_gaps(ob), hmm, "C")
        dist.append(di), err.append(er)
    table = binned_median(np.concatenate(dist), np.concatenate(err))
    json.dump([None if np.isnan(v) else v for v in table], open(SIGMA_C_PATH, "w"))
    return table


def sigma_table(d):
    """Per-bin sigma: the larger of the real-gap (A) and open-match (C) out-of-sample medians,
    each where it has >= 200 truth frames, made non-decreasing with distance from a sighting."""
    hmm = make_hmm()
    hmm.prepare(d, "A", None)
    a = binned_median(*edge_errors(d, real_gaps(d), hmm, "A"))
    c = sigma_from_open()
    table = np.fmax(a, c)
    table = pd.Series(table).ffill().bfill().values
    return np.maximum.accumulate(table), a, c


VIDEO_THRESHOLD = 0.6     # verifier probability; chosen on test set B (ballfill.verify)


def video_filler(d, hmm):
    """Verified tactical-video sightings + the HMM between them, as scored in ballfill.verify:
    each half uses the verifier trained on the other half."""
    from ballfill.bothvid import verifier_models
    from ballfill.verify import VerifiedVideoFill
    from ballfill.vidfuse import build_tracklets
    tracks = build_tracklets(d)
    return VerifiedVideoFill(hmm, d, tracks, verifier_models(d, tracks), VIDEO_THRESHOLD)


def video_sigma(d, vf):
    """Median out-of-sample error of video-sighted frames on test set A."""
    from ballfill.evaluate import context_ball
    err = []
    for g in real_gaps(d):
        vf.method.prepare(d, "A", None)
        v, _ = vf.video_for_gap(d, context_ball(d, g, "A"), g)
        k = np.isfinite(v[g.truth_frames - g.a, 0])
        err.append(np.linalg.norm(v[g.truth_frames - g.a][k] - g.truth[k], axis=1))
    err = np.concatenate(err)
    return float(np.median(err)), len(err)


def main():
    d = load_ball_data()
    m = d.match
    ball, source, sigma, _ = fuse(d)
    src = np.array(SOURCES, dtype=object)[source]
    table, table_a, table_c = sigma_table(d)

    hmm_a = make_hmm()
    sig_video, n_video = video_sigma(d, video_filler(d, hmm_a))
    hmm = make_hmm()
    hmm.use_production_models(d)
    vf = video_filler(d, hmm)
    p_video = np.full(d.n, np.nan)

    out_xy = ball.copy()
    p_carrier = np.full(d.n, np.nan)
    p_dead = np.full(d.n, np.nan)
    gap_s = np.full(d.n, np.nan)
    L, W = m.pitch
    for half in (1, 2):
        idx = np.flatnonzero(m.half == half)
        first, last = idx[0], idx[-1]
        if np.isnan(out_xy[first, 0]):                # every half starts with a kick-off
            out_xy[first] = (L / 2, W / 2)
            src[first], sigma[first] = "kickoff", 1.0
        seen = ~np.isnan(out_xy[:, 0])
        for a, b in runs(~seen & (m.half == half)):
            if b > last:                              # runs to the end of the half: hold
                out_xy[a:b] = out_xy[a - 1]
                src[a:b], gap_s[a:b], sigma[a:b] = "filled", (b - a) / FPS, table[-1]
                continue
            gap = Gap(a, b, half, np.arange(a, b), np.zeros((b - a, 2)))
            v, pv = vf.video_for_gap(d, out_xy, gap)
            seen = np.isfinite(v[:, 0])
            fr = np.arange(a, b)
            out_xy[fr[seen]] = v[seen]
            src[fr[seen]], sigma[fr[seen]], p_video[fr[seen]] = "video", sig_video, pv[seen]
            gap_s[a:b] = (b - a) / FPS
            for s0, e0 in runs(~seen):               # the HMM between sightings
                sa, sb = a + s0, a + e0
                sub = Gap(sa, sb, half, np.arange(sa, sb), np.zeros((sb - sa, 2)))
                out_xy[sa:sb] = hmm(d, out_xy, sub)
                p_carrier[sa:sb] = hmm.last_conf
                if hmm.last_dead is not None:
                    p_dead[sa:sb] = hmm.last_dead
                frames = np.arange(sa, sb)
                edge_s = np.minimum(frames - (sa - 1), sb - frames) / FPS
                sigma[sa:sb] = table[np.searchsorted(SIGMA_EDGES_S, edge_s, side="right") - 1]
                src[sa:sb] = "filled"

    g = m.graded
    df = pd.DataFrame(dict(frame=np.flatnonzero(g), x=out_xy[g, 0].round(3),
                           y=out_xy[g, 1].round(3), source=src[g], sigma_m=np.round(sigma[g], 2),
                           gap_s=np.round(gap_s[g], 2), p_video=np.round(p_video[g], 3),
                           p_carrier=np.round(p_carrier[g], 3),
                           p_dead=np.round(p_dead[g], 3)))
    path = os.path.join(CACHE_DIR, "BAR-ATH_ball.csv")
    df.to_csv(path, index=False)

    print("wrote {} ({:,} graded frames)".format(path, len(df)))
    print(df.source.value_counts().to_string())
    print("video sigma_m: {:.2f} m (median error of {} video-sighted frames on test set A)".format(
        sig_video, n_video))
    print("missing after fill: {}".format(int(df.x.isna().sum())))
    print("filled-frame sigma_m by seconds to the nearest sighting (median error, out of sample):")
    print("   {:<12}{:>12}{:>14}{:>10}".format("bin", "A real gaps", "C open live", "used"))
    for lo, hi, sa, sc, s in zip(SIGMA_EDGES_S[:-1], SIGMA_EDGES_S[1:], table_a, table_c, table):
        print("   {:<12}{:>11}{:>14}{:>9.1f}m".format(
            "{:g}-{}".format(lo, "{:g}".format(hi) if hi < 1e8 else "inf"),
            "-" if np.isnan(sa) else "{:.1f}m".format(sa),
            "-" if np.isnan(sc) else "{:.1f}m".format(sc), s))


if __name__ == "__main__":
    main()

