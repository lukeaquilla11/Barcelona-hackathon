"""Gap-filling methods. Each is method(d, ball, gap) -> (gap.length, 2) metres.

`ball` is what the method may see (NaN inside the gap); players, tags and everything else in
`d` are always visible. Methods with fitted parts expose prepare(d, which, gaps) and are scored
two-fold by half by ballfill.evaluate.
"""

import numpy as np


def _endpoints(ball, gap):
    return ball[gap.a - 1], ball[gap.b]


def linear(d, ball, gap):
    """Straight line, constant speed, from the last sighting to the next."""
    p0, p1 = _endpoints(ball, gap)
    s = (np.arange(gap.a, gap.b) - (gap.a - 1)) / (gap.b - (gap.a - 1))
    return p0 + (p1 - p0) * s[:, None]


def hold(d, ball, gap):
    """Last sighting for the first half of the gap, next sighting for the second."""
    p0, p1 = _endpoints(ball, gap)
    out = np.empty((gap.length, 2))
    mid = gap.length // 2
    out[:mid], out[mid:] = p0, p1
    return out


def follow_track(players, team, slot, a, b, max_jump=3.0):
    """Position of one player from a to b, surviving slot drop-outs and identity swaps.

    When the slot goes empty the player is handed to the nearest same-team slot within
    max_jump metres of where they were last seen (the tracker re-acquiring them).
    """
    out = np.full((b - a, 2), np.nan)
    last = None
    for i, t in enumerate(range(a, b)):
        p = players[t, slot]
        if np.isnan(p[0]) and last is not None:
            cand = players[t].copy()
            cand[team != team[slot]] = np.nan
            dist = np.linalg.norm(cand - last, axis=1)
            if np.isfinite(dist).any() and np.nanmin(dist) < max_jump:
                slot = int(np.nanargmin(dist))
                p = players[t, slot]
        if not np.isnan(p[0]):
            last = p
        out[i] = last if last is not None else np.nan
    return out


def nearest_slot(players, t, point):
    dist = np.linalg.norm(players[t] - point, axis=1)
    return int(np.nanargmin(dist)) if np.isfinite(dist).any() else None


def follow_players(d, ball, gap):
    """The player nearest the ball at the last sighting carries it for the first half of the gap;
    the player nearest it at the next sighting for the second half (followed backwards in time)."""
    m = d.match
    p0, p1 = _endpoints(ball, gap)
    out = linear(d, ball, gap)
    mid = gap.length // 2
    s0 = nearest_slot(m.players_xy, gap.a - 1, p0)
    s1 = nearest_slot(m.players_xy, gap.b, p1)
    if s0 is not None:
        tr = follow_track(m.players_xy, m.track_team, s0, gap.a - 1, gap.a + mid)[1:]
        ok = ~np.isnan(tr[:, 0])
        out[:mid][ok] = tr[ok]
    if s1 is not None:
        rev = m.players_xy[::-1]
        n = len(rev)
        tr = follow_track(rev, m.track_team, s1, n - 1 - gap.b, n - 1 - (gap.a + mid) + 1)[1:][::-1]
        ok = ~np.isnan(tr[:, 0])
        out[mid:][ok] = tr[ok]
    return out


def oracle_nearest_player(d, ball, gap):
    """NOT A METHOD -- reads the truth. The nearest tracked player to the true ball on every
    truth frame: the ceiling for any approach that puts the ball at a player's feet."""
    out = linear(d, ball, gap)
    for t, truth in zip(gap.truth_frames, gap.truth):
        s = nearest_slot(d.match.players_xy, t, truth)
        if s is not None:
            out[t - gap.a] = d.match.players_xy[t, s]
    return out


from ballfill.carrier import CarrierHMM  # noqa: E402

METHODS = {
    "linear": linear,
    "follow_players": follow_players,
    "carrier_hmm v1": CarrierHMM(flight_s=2.0, switch=0.1, beta=0.1, static_e=0.03, pass_scale=5.0,
                                 pooled=False),
    "carrier_hmm v2": CarrierHMM(flight_s=2.0, switch=0.1, beta=0.1, static_e=0.03, pass_scale=5.0),
    "ORACLE nearest player": oracle_nearest_player,
}

