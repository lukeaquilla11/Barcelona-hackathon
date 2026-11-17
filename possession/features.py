"""Per-frame features, computed identically for our match and for (degraded) open matches.

Inputs are only what both sources share: ball_xy (NaN when missing), players_xy (NaN when not
tracked), track_team and the pitch size. No labels, no tags, no player identities.
"""

import numpy as np
import pandas as pd

FPS = 25
HOME, AWAY = 0, 1


def rolling_mean(x, half_width):
    """Centred NaN-ignoring mean over +/- half_width frames (cumulative sums, O(n))."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    csum = np.r_[0, np.cumsum(np.where(valid, x, 0))]
    ccnt = np.r_[0, np.cumsum(valid)]
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half_width)
    hi = np.minimum(len(x), idx + half_width + 1)
    cnt = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (csum[hi] - csum[lo]) / cnt, np.nan)


def speed(xy, lag=5):
    """Speed in m/s from positions `lag` frames apart (centred). lag > 1 damps position jitter."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    h = lag // 2
    step = np.linalg.norm(xy[lag:] - xy[:-lag], axis=-1) * FPS / lag
    out[h:h + len(step)] = step
    return out


def velocity(xy, lag=5):
    """Velocity vectors in m/s from positions `lag` frames apart (centred), same shape as xy."""
    out = np.full(xy.shape, np.nan, dtype=np.float64)
    h = lag // 2
    v = (xy[lag:] - xy[:-lag]) * FPS / lag
    out[h:h + len(v)] = v
    return out


def velocity_change(xy, k=3):
    """|v_after - v_before| in m/s: ball velocity over the k frames after t minus the k before.

    A touch redirects or slows the ball (large change); a pass rolling past a player keeps its
    velocity (small change), however close it passes. That is the difference between control
    and a pass-by, and ball speed alone cannot see it: a first-time pass leaves as fast as it came.
    """
    out = np.full(len(xy), np.nan, dtype=np.float32)
    v_before = (xy[k:-k] - xy[:-2 * k]) * FPS / k
    v_after = (xy[2 * k:] - xy[k:-k]) * FPS / k
    out[k:-k] = np.linalg.norm(v_after - v_before, axis=1)
    return out


def outside_distance(xy, pitch):
    """Metres the point lies outside the pitch (0 inside, NaN where xy is NaN)."""
    length, width = pitch
    dx = np.maximum(0, np.maximum(-xy[:, 0], xy[:, 0] - length))
    dy = np.maximum(0, np.maximum(-xy[:, 1], xy[:, 1] - width))
    return np.where(np.isnan(xy[:, 0]), np.nan, np.maximum(dx, dy))


def line_distance(xy, pitch):
    """Metres to the nearest touchline or goal line (negative outside)."""
    length, width = pitch
    return np.minimum.reduce([xy[:, 0], length - xy[:, 0], xy[:, 1], width - xy[:, 1]])


def compute(match):
    """DataFrame of per-frame features, one row per frame of `match`."""
    ball = match.ball_xy.astype(np.float64)
    players = match.players_xy.astype(np.float64)
    team = match.track_team
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])

    # last known ball position and time since it was seen
    last = pd.DataFrame(ball).ffill().values
    idx = np.where(seen, np.arange(n), -1)
    last_seen = np.maximum.accumulate(idx)
    since = np.where(last_seen >= 0, (np.arange(n) - last_seen) / FPS, 60.0)

    # distances from the (visible) ball to the nearest player of each team
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home = dist[:, team == HOME].min(axis=1)
    d_away = dist[:, team == AWAY].min(axis=1)
    d_home[~seen] = np.nan
    d_away[~seen] = np.nan

    # the same from the last known position, for gaps
    dist_last = np.linalg.norm(players - last[:, None, :], axis=2)
    dist_last = np.where(np.isnan(dist_last), np.inf, dist_last)
    d_last = np.where(np.isfinite(dist_last.min(axis=1)), dist_last.min(axis=1), np.nan)

    ball_speed = speed(ball, lag=3)
    ball_dv = velocity_change(ball, k=3)
    ball_dv6 = velocity_change(ball, k=6)
    ball_dv12 = velocity_change(ball, k=12)
    p_speed = speed(players, lag=5)

    # ball moving with the nearest player of each team: |v_ball - v_player| (m/s). Small while a
    # player carries or controls it; large for a ball rolling past. Robust to a smoothed ball.
    v_ball = velocity(ball, lag=5)
    v_players = velocity(players, lag=5)
    rel = {}
    for code, name in ((HOME, "home"), (AWAY, "away")):
        cols = np.flatnonzero(team == code)
        nearest = cols[np.argmin(dist[:, cols], axis=1)]
        v_near = v_players[np.arange(n), nearest]
        r = np.linalg.norm(v_ball - v_near, axis=1)
        r[~seen | ~np.isfinite(dist[np.arange(n), nearest])] = np.nan
        rel[name] = r
    tracked = (~np.isnan(players[:, :, 0])).sum(axis=1)
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_player_speed = np.nanmean(p_speed, axis=1)
        spread_x = np.nanstd(players[:, :, 0], axis=1)
        spread_y = np.nanstd(players[:, :, 1], axis=1)
        centroid_x = np.nanmean(players[:, :, 0], axis=1)

    f = pd.DataFrame({
        "ball_seen": seen.astype(np.float32),
        "ball_seen_1s": rolling_mean(seen, FPS),
        "ball_seen_5s": rolling_mean(seen, 5 * FPS),
        "since_seen_s": np.minimum(since, 60.0),
        "ball_out_m": outside_distance(ball, match.pitch),
        "last_out_m": outside_distance(last, match.pitch),
        "last_line_m": line_distance(last, match.pitch),
        "out_recent_3s": rolling_mean(outside_distance(ball, match.pitch) > 0.5, 3 * FPS),
        "ball_speed": ball_speed,
        "ball_dv": ball_dv,
        "ball_dv6": ball_dv6,
        "ball_dv12": ball_dv12,
        "rel_v_home": rel["home"],
        "rel_v_away": rel["away"],
        "ball_speed_1s": rolling_mean(ball_speed, FPS),
        "ball_still_2s": rolling_mean(ball_speed < 0.5, 2 * FPS),
        "d_home": d_home,
        "d_away": d_away,
        "d_nearest": np.fmin(d_home, d_away),
        "d_last_nearest": d_last,
        "player_speed": mean_player_speed,
        "player_speed_1s": rolling_mean(mean_player_speed, FPS),
        "player_speed_3s": rolling_mean(mean_player_speed, 3 * FPS),
        "player_speed_10s": rolling_mean(mean_player_speed, 10 * FPS),
        "spread_x": spread_x,
        "spread_y": spread_y,
        "centroid_off_centre": np.abs(centroid_x - match.pitch[0] / 2),
        "tracked": tracked.astype(np.float32),
    })
    # change in collective speed: play stopping (players decelerate) or restarting
    f["player_speed_trend"] = f["player_speed_3s"].shift(-3 * FPS) - f["player_speed_3s"].shift(3 * FPS)
    return f


DEAD_FEATURES = [
    "ball_seen", "ball_seen_1s", "ball_seen_5s", "since_seen_s", "ball_out_m", "last_out_m",
    "last_line_m", "out_recent_3s", "ball_speed_1s", "ball_still_2s", "d_nearest",
    "d_last_nearest", "player_speed", "player_speed_1s", "player_speed_3s", "player_speed_10s",
    "spread_x", "spread_y", "centroid_off_centre", "tracked", "player_speed_trend",
]
