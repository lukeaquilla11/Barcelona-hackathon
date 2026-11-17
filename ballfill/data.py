"""Both ball tracks on the tactical timeline.

The broadcast recording is not frame-aligned to the tactical one, but it does not need a mapping
table: matching the broadcast's visible players to the tactical players shows the offset is a
constant per half (replays and close-ups do not insert time, the broadcast just loses the pitch)
plus a slow clock drift: -1 frame over each 53-minute half (~13 ppm). The offset is therefore
modelled per half as a + drift * (fraction of the half), fitted to sub-frame precision by making
the two ball tracks agree, and the broadcast is interpolated at the resulting fractional frames.

    python -m ballfill.data         # parse (once), align, print a summary
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402
from possession.load_data import CACHE_DIR, Match, _parse_atd, _track_team, load_match  # noqa: E402

FPS = 25
# Coarse offsets (tactical frame = broadcast frame + offset), found by a blind search of player
# configurations over the whole tactical timeline. Only used as the centre of the fine search.
COARSE_OFFSET = {1: 4015, 2: 26920}
OFFSET_GRID = np.arange(-1.5, 1.51, 0.1)    # frames around the coarse offset
DRIFT_GRID = np.arange(-3.0, 1.01, 0.1)     # total drift over a half, frames
TRUNCATE_M = 3.0   # aerial balls disagree by metres whatever the offset; cap them in the cost
# A broadcast frame is "live" -- showing this moment of the match, not a replay or another feed
# that kept a valid homography -- when its players sit on the tactical players of the same frame.
LIVE_MEDIAN_M = 1.5    # median distance, broadcast player -> nearest same-team tactical player
LIVE_MIN_MATCHED = 3   # broadcast players within 2 m of a tactical player


@dataclass
class BallData:
    match: Match
    tac: np.ndarray          # (N, 2) tactical ball, metres
    bc: np.ndarray           # (N, 2) broadcast ball on the tactical timeline, metres, live frames only
    bc_raw: np.ndarray       # (N, 2) the same before the live check
    bc_live: np.ndarray      # (N,) bool, broadcast frame verified to show this moment
    bc_players: np.ndarray   # (N, 51, 2) broadcast players on the tactical timeline, metres
    bc_team: np.ndarray      # (51,) HOME / AWAY
    offset: np.ndarray       # (N,) fractional; broadcast frame = tactical frame - offset; NaN unmapped

    @property
    def n(self):
        return len(self.tac)


def load_broadcast(game=GAME, data_dir=None, refresh=False):
    """(table, track_team) of the broadcast ATD, cached like the tactical one."""
    meta, raw = require(["atd_meta", "atd_raw"], game, data_dir, source="broadcast")
    team = _track_team(meta)
    cache = os.path.join(CACHE_DIR, "{}_broadcast_atd.npz".format(game))
    if os.path.exists(cache) and not refresh:
        return np.load(cache)["table"], team
    print("parsing {} (first run only) ...".format(os.path.basename(raw)))
    table = _parse_atd(raw, len(team), source_frames("broadcast", game))
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, table=table)
    return table, team


def _interp(arr, x):
    """arr sampled at fractional indices x, linear between neighbours; NaN if either is NaN."""
    out = np.full((len(x),) + arr.shape[1:], np.nan, dtype=np.float32)
    ok = ~np.isnan(x)
    i = np.floor(np.where(ok, x, -1)).astype(np.int64)
    ok &= (i >= 0) & (i + 1 < len(arr))
    f = (x[ok] - i[ok]).reshape((-1,) + (1,) * (arr.ndim - 1))
    out[ok] = arr[i[ok]] * (1 - f) + arr[i[ok] + 1] * f
    return out


def align(tac, bc, game=GAME):
    """Fractional per-frame offset: offset = a + drift * (position within the half).

    Fitted per half by grid search on the truncated mean distance between the two ball tracks.
    On BAR-ATH both halves drift by -1 frame over 53 minutes, a constant clock-rate difference.
    Returns the (N,) offset (NaN outside the graded halves) and {half: (a, drift, cost, n)}.
    """
    offset = np.full(len(tac), np.nan)
    fits = {}
    for half, first, last in half_video_blocks(game):
        frames = np.arange(first, last + 1)
        s = (frames - first) / (last - first)
        best = None
        for a in COARSE_OFFSET[half] + OFFSET_GRID:
            for drift in DRIFT_GRID:
                b = _interp(bc, frames - (a + drift * s))
                k = ~np.isnan(b[:, 0]) & ~np.isnan(tac[frames, 0])
                cost = np.mean(np.minimum(np.linalg.norm(b[k] - tac[frames[k]], axis=1), TRUNCATE_M))
                if best is None or cost < best[2]:
                    best = (a, drift, cost, int(k.sum()))
        offset[frames] = best[0] + best[1] * s
        fits[half] = best
    return offset, fits


def _on_tactical(arr, offset):
    return _interp(arr, np.arange(len(offset)) - offset)


def broadcast_live(match, bc_players, bc_team):
    """(N,) bool: the broadcast frame's players match the tactical players of the same frame.

    Without this, replays with a valid homography put a ball from an earlier moment onto the
    timeline: on BAR-ATH 8% of broadcast ball frames fail it, and those disagree with the
    tactical ball by a median 4.9 m against 0.75 m for frames that pass.
    """
    from scipy.spatial import cKDTree
    live = np.zeros(match.n_frames, dtype=bool)
    has = ~np.isnan(bc_players[:, :, 0]).all(axis=1)
    for t in np.flatnonzero(has):
        dists = []
        for team in (0, 1):
            p = bc_players[t][bc_team == team]
            p = p[~np.isnan(p[:, 0])]
            q = match.players_xy[t][match.track_team == team]
            q = q[~np.isnan(q[:, 0])]
            if len(p) and len(q):
                dists.append(cKDTree(q).query(p)[0])
        if dists:
            dd = np.concatenate(dists)
            live[t] = np.median(dd) < LIVE_MEDIAN_M and (dd < 2).sum() >= LIVE_MIN_MATCHED
    return live


def load_ball_data(game=GAME, data_dir=None, match=None):
    m = match if match is not None else load_match(game, data_dir)
    length, width = m.pitch
    scale = np.array([length, width], dtype=np.float32)
    table, team = load_broadcast(game, data_dir)
    n_slots = len(team)
    bc_players = table[:, :3 * n_slots].reshape(len(table), n_slots, 3)[:, :, :2] * scale
    bc_ball = table[:, 3 * n_slots:3 * n_slots + 2] * scale

    cache = os.path.join(CACHE_DIR, "{}_broadcast_on_tactical.npz".format(game))
    if os.path.exists(cache):
        z = np.load(cache)
        offset, bc_raw, bc_pl, live = z["offset"], z["bc_raw"], z["bc_players"], z["live"]
    else:
        offset, _ = align(m.ball_xy, bc_ball, game)
        bc_raw, bc_pl = _on_tactical(bc_ball, offset), _on_tactical(bc_players, offset)
        live = broadcast_live(m, bc_pl, team)
        np.savez(cache, offset=offset, bc_raw=bc_raw, bc_players=bc_pl, live=live)
    bc = np.where(live[:, None], bc_raw, np.nan).astype(np.float32)
    return BallData(match=m, tac=m.ball_xy.astype(np.float32), bc=bc, bc_raw=bc_raw,
                    bc_live=live, bc_players=bc_pl, bc_team=team, offset=offset)


def runs(mask):
    """(start, stop) of every run of True, stop exclusive."""
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


if __name__ == "__main__":
    m = load_match()
    d = load_ball_data(match=m)
    _, fits = align(d.tac, load_broadcast()[0][:, 153:155] * np.array(m.pitch, np.float32))
    for half, (a, drift, cost, n) in fits.items():
        print("half {}: tactical = broadcast + {:.1f} {:+.1f}*(fraction of half)  "
              "[truncated mean ball distance {:.2f} m over {} frames]".format(half, a, drift, cost, n))
    g = m.graded
    has_ball = ~np.isnan(d.bc_raw[:, 0]) & g
    print("broadcast ball frames failing the live check: {:.1%}".format(
        (has_ball & ~d.bc_live).sum() / has_ball.sum()))
    t_seen, b_seen = ~np.isnan(d.tac[:, 0]), ~np.isnan(d.bc[:, 0])
    print("graded ball coverage: tactical {:.1%}, broadcast {:.1%}, either {:.1%}".format(
        t_seen[g].mean(), b_seen[g].mean(), (t_seen | b_seen)[g].mean()))

