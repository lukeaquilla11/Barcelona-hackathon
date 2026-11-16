"""Make clean tracking look like the hackathon's video-derived tracking (noise, ball dropouts, players lost).
The training data (Metrica samples, IDSSE) is much cleaner than a broadcast/tactical-camera ATD, so a model trained
on it clean would trust signals that will not be there. Every call with a different seed gives a different draw."""
import numpy as np
from .schema import Match

DEFAULTS = dict(player_noise=0.35, ball_noise=0.25, ball_burst_rate=0.003, ball_burst_len=20,
                ball_out_drop=0.9, ball_fast_drop=0.4, fast_mps=22.0,
                player_burst_rate=0.0012, player_burst_len=60, player_frame_drop=0.01)


def _bursts(n, rate, mean_len, rng):
    m = np.zeros(n, bool)
    for s in np.flatnonzero(rng.random(n) < rate):
        m[s:s + 1 + rng.geometric(1.0 / max(mean_len, 1))] = True
    return m


def degrade(m: Match, seed=0, **kw) -> Match:
    c = dict(DEFAULTS, **kw)
    rng = np.random.default_rng(seed)
    P, B = m.P.copy(), m.B.copy()
    n, S, _ = P.shape
    P += rng.normal(0, c["player_noise"], P.shape).astype(np.float32)
    B += rng.normal(0, c["ball_noise"], B.shape).astype(np.float32)
    # ball: random bursts + off-pitch + very fast flight are often not seen by the camera pipeline
    drop = _bursts(n, c["ball_burst_rate"], c["ball_burst_len"], rng)
    bx, by = m.B[:, 0], m.B[:, 1]
    out = (bx < -0.5) | (bx > m.L + 0.5) | (by < -0.5) | (by > m.W + 0.5)
    drop |= out & (rng.random(n) < c["ball_out_drop"])
    sp = np.r_[0, np.hypot(np.diff(bx), np.diff(by))] * m.fps
    drop |= (sp > c["fast_mps"]) & (rng.random(n) < c["ball_fast_drop"])
    B[drop] = np.nan
    # players: per-slot bursts + a little per-frame loss
    for s in range(S):
        P[_bursts(n, c["player_burst_rate"], c["player_burst_len"], rng), s] = np.nan
    P[rng.random((n, S)) < c["player_frame_drop"]] = np.nan
    return m.with_tracking(P, B)
