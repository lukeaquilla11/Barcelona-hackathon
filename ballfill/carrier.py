"""Who has the ball, from the players alone -- and the ball path through a gap that follows from it.

1. A per-frame, per-track-slot classifier: "is this slot the player nearest the ball (within
   CARRIER_M)?". Features are player-only (positions, velocities, pressure from opponents, team
   shape, Smart Tagging's team) so it can run inside a gap. Trained on frames where the tactical
   ball is seen, from the *other* half only, so no scored frame was ever a training frame.

2. A hidden Markov model over slots on a STEP-frame grid through the gap. Emissions are the
   classifier's probabilities; transitions keep the ball with its carrier, hand it to the slot
   that re-acquires a player when the tracker drops them, and allow passes (more likely to near
   players). Both sightings that bound the gap pin the start and end states. Forward-backward
   gives each slot's posterior; the ball estimate is the posterior-weighted position.
"""

import numpy as np

FPS = 25
STEP = 5             # HMM / feature grid, frames (0.2 s)
CARRIER_M = 2.5      # a slot is "the carrier" if it is the nearest to the ball and within this
FEATURE_NAMES = ("x_att", "y_centre", "speed", "v_att", "d_opp1", "d_opp2", "n_opp3", "n_opp6",
                 "d_mate1", "pressure", "converge", "d_centroid", "team_v_att", "tag_team",
                 "deepest", "n_tracked", "speed_rank_team")


# -- features -----------------------------------------------------------------------------------

def _velocity(players, lag=5):
    v = np.full(players.shape, np.nan, dtype=np.float32)
    h = lag // 2
    v[h:h + len(players) - lag] = (players[lag:] - players[:-lag]) * (FPS / lag)
    return v


def attack_sign(match):
    """(N, 2) +1 if team attacks towards +x on that frame, -1 if towards 0 (from team mean x)."""
    sign = np.zeros((match.n_frames, 2), dtype=np.float32)
    L = match.pitch[0]
    for half in (1, 2):
        k = match.half == half
        for team in (0, 1):
            x = match.players_xy[k][:, match.track_team == team, 0]
            sign[k, team] = 1.0 if np.nanmean(x) < L / 2 else -1.0   # defends x=0 -> attacks +x
    return sign


def tag_team(match):
    """(N,) 0 home / 1 away / -1 unknown, from Smart Tagging intervals that carry a team."""
    out = np.full(match.n_frames, -1, dtype=np.int8)
    t = match.tags.dropna(subset=["team_side"]) if "team_side" in match.tags else match.tags[:0]
    for _, r in t.iterrows():
        out[int(r.start_frame):int(r.end_frame)] = 0 if r.team_side == "H" else 1
    return out


def compute_features(match, frames, vel=None, sign=None, tags=None, chunk=3000):
    """(len(frames), slots, F) float32; NaN rows for empty slots."""
    P_all, team = match.players_xy, match.track_team
    vel = _velocity(P_all) if vel is None else vel
    sign = attack_sign(match) if sign is None else sign
    tags = tag_team(match) if tags is None else tags
    L, W = match.pitch
    S = len(team)
    same = team[:, None] == team[None, :]
    out = np.full((len(frames), S, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    for c0 in range(0, len(frames), chunk):
        fr = frames[c0:c0 + chunk]
        P, V = P_all[fr].astype(np.float32), vel[fr]
        n = len(fr)
        sg = sign[fr][:, team]                                     # (n, S)
        present = ~np.isnan(P[:, :, 0])
        x_att = np.where(sg > 0, P[:, :, 0], L - P[:, :, 0])
        diff = P[:, None, :, :] - P[:, :, None, :]                 # (n, i, j, 2): j - i
        dist = np.linalg.norm(diff, axis=-1)
        dist[:, np.arange(S), np.arange(S)] = np.nan
        d_opp = np.where(same[None], np.nan, dist)
        d_mate = np.where(same[None], dist, np.nan)
        d_opp_sorted = np.sort(np.where(np.isnan(d_opp), np.inf, d_opp), axis=2)
        # closing speed of j towards i: -(v_j . (p_j - p_i)/|..|)
        unit = diff / np.where(dist > 0, dist, np.nan)[..., None]
        closing = -(V[:, None, :, :] * unit).sum(-1)               # (n, i, j)
        w = np.where(np.isnan(d_opp) | (d_opp > 15), 0, 1 / np.maximum(d_opp, 1))
        pressure = np.nansum(np.nan_to_num(closing) * w, axis=2)
        spd_j = np.linalg.norm(V, axis=-1)
        cosang = closing / np.where(spd_j[:, None, :] > 1, spd_j[:, None, :], np.nan)
        converge = np.nansum((cosang > 0.85) & (dist < 20), axis=2)
        centroid = np.nanmean(P, axis=1, keepdims=True)
        spd = np.linalg.norm(V, axis=-1)
        team_v = np.zeros((n, S), dtype=np.float32)
        spd_rank = np.full((n, S), np.nan, dtype=np.float32)
        for tm in (0, 1):
            k = team == tm
            vx = np.nanmean(V[:, k, 0], axis=1) * sign[fr, tm]
            team_v[:, k] = vx[:, None]
            s = np.where(np.isnan(spd[:, k]), -1, spd[:, k])
            spd_rank[:, k] = np.argsort(np.argsort(-s, axis=1), axis=1)
        deepest = np.zeros((n, S), dtype=np.float32)
        for tm in (0, 1):
            k = np.flatnonzero(team == tm)
            xa = np.where(present[:, k], x_att[:, k], np.inf)
            deepest[np.arange(n), k[np.argmin(xa, axis=1)]] = 1
        tg = tags[fr][:, None]
        tag_feat = np.where(tg < 0, 0, np.where(tg == team[None, :], 1, -1)).astype(np.float32)
        f = np.stack([
            x_att, np.abs(P[:, :, 1] - W / 2), spd, V[:, :, 0] * sg,
            d_opp_sorted[:, :, 0], d_opp_sorted[:, :, 1],
            np.nansum(d_opp < 3, axis=2), np.nansum(d_opp < 6, axis=2),
            np.nanmin(np.where(np.isnan(d_mate), np.inf, d_mate), axis=2),
            pressure, converge, np.linalg.norm(P - centroid, axis=-1), team_v, tag_feat,
            deepest, np.repeat(present.sum(1, keepdims=True), S, 1), spd_rank], axis=-1)
        f[~np.isfinite(f)] = 50.0                   # "no opponent / mate in range"
        f[~present] = np.nan
        out[c0:c0 + chunk] = f
    return out


TEMPORAL_LAG = 10                       # frames (0.4 s) before / after
TEMPORAL_COLS = ("speed", "d_opp1", "n_opp3", "pressure", "converge", "v_att")
RANK_COLS = (("pressure", -1), ("converge", -1), ("d_opp1", 1), ("speed", -1), ("n_opp3", -1))
FEATURE_NAMES_V2 = (FEATURE_NAMES + tuple("rank_" + c for c, _ in RANK_COLS)
                    + tuple("{}_{}".format(c, w) for w in ("before", "after") for c in TEMPORAL_COLS))


def _ranks(f):
    """Within-frame rank (0 = most) of selected columns among present slots, scaled to 0..1."""
    out = []
    present = ~np.isnan(f[:, :, 0])
    n = np.maximum(present.sum(1, keepdims=True) - 1, 1)
    for col, direction in RANK_COLS:
        v = f[:, :, FEATURE_NAMES.index(col)] * direction
        v = np.where(present, v, np.inf)
        r = np.argsort(np.argsort(v, axis=1), axis=1).astype(np.float32) / n
        out.append(np.where(present, r, np.nan))
    return np.stack(out, axis=-1)


def compute_features_v2(match, frames, vel=None, sign=None, tags=None):
    """v1 features + within-frame ranks + selected features 0.4 s before and after."""
    vel = _velocity(match.players_xy) if vel is None else vel
    sign = attack_sign(match) if sign is None else sign
    tags = tag_team(match) if tags is None else tags
    n = match.n_frames
    f0 = compute_features(match, frames, vel, sign, tags)
    cols = [FEATURE_NAMES.index(c) for c in TEMPORAL_COLS]
    parts = [f0, _ranks(f0)]
    for lag in (-TEMPORAL_LAG, TEMPORAL_LAG):
        fl = compute_features(match, np.clip(frames + lag, 0, n - 1), vel, sign, tags)
        parts.append(fl[:, :, cols])
    return np.concatenate(parts, axis=-1)


# -- classifier ---------------------------------------------------------------------------------

def carrier_labels(match, ball, frames):
    """(len(frames),) slot index of the carrier, -1 if no slot within CARRIER_M, -2 if no ball."""
    P = match.players_xy[frames]
    b = ball[frames]
    dist = np.linalg.norm(P - b[:, None, :], axis=-1)
    dist = np.where(np.isnan(dist), np.inf, dist)
    lab = np.argmin(dist, axis=1)
    lab = np.where(dist[np.arange(len(frames)), lab] <= CARRIER_M, lab, -1)
    lab[np.isnan(b[:, 0])] = -2
    return lab


def training_rows(match, ball, frames, feats):
    """(X, y) rows, one per present slot on frames where the ball is known."""
    lab = carrier_labels(match, ball, frames)
    keep = lab >= -1
    X = feats[keep]                                 # (n, S, F)
    y = (np.arange(X.shape[1])[None, :] == lab[keep][:, None])
    present = ~np.isnan(X[:, :, 0])
    return X[present], y[present]


def fit_rows(X, y, max_iter=300):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08,
                                           max_leaf_nodes=63, l2_regularization=1.0,
                                           random_state=0)
    model.fit(X, y)
    return model


def train(match, ball, frames, feats):
    return fit_rows(*training_rows(match, ball, frames, feats))


def predict(model, feats):
    """(n, S) carrier probability, 0 for empty slots."""
    present = ~np.isnan(feats[:, :, 0])
    p = np.zeros(feats.shape[:2], dtype=np.float64)
    p[present] = model.predict_proba(feats[present])[:, 1]
    return p


# -- HMM through a gap --------------------------------------------------------------------------

def gap_posterior(P, team, emis, p_start, p_end, switch=0.05, pass_scale=20.0, beta=1.0,
                  sigma_end=2.0, handover_m=3.0, static_e=0.0, extra=None, dead_p=None,
                  exit_pt=None, dead_e=1.0, dead_players=True, return_model=False):
    """Forward-backward on the grid. P (T, S, 2) slot positions (first and last rows are the
    sighting frames), emis (T, S) carrier probabilities.

    With static_e > 0, ball states are appended: resting where last seen, resting where next seen
    (a dead ball at its restart spot), and `extra` (T, k, 2) moving ones (flights from the edges).
    Without dead_p every ball state has the constant emission static_e. With dead_p (T,) -- the
    probability play is stopped -- player and flight states are weighted by (1 - dead_p), and the
    resting states plus `exit_pt` (where play stopped) by dead_e * dead_p.
    Returns (T, n) posteriors and the (T, n, 2) state positions."""
    T, S = emis.shape
    ball_e = static_e ** beta                      # same scale as carrier probabilities ** beta
    live_w = np.ones(T) if dead_p is None else 1 - dead_p
    dead_w = np.full(T, ball_e) if dead_p is None else dead_e * ball_e * dead_p
    # dead_players=False: a stoppage does not take the ball off the players (they pick it up and
    # carry it to the restart); it only makes the resting states more plausible.
    player_w = live_w if dead_players else np.ones(T)
    emis = np.maximum(emis, 1e-6) ** beta * player_w[:, None]
    if static_e > 0:
        rest = [p_start, p_end] + ([exit_pt] if exit_pt is not None else [])
        P = np.concatenate([P] + [np.broadcast_to(r, (T, 1, 2)) for r in rest], axis=1)
        team = np.r_[team, -1 - np.arange(len(rest))]
        emis = np.c_[emis, np.repeat(dead_w[:, None], len(rest), 1)]
        if extra is not None:                      # (T, k, 2) moving ball states, e.g. flights
            P = np.concatenate([P, extra], axis=1)
            team = np.r_[team, -10 - np.arange(extra.shape[1])]
            emis = np.c_[emis, np.repeat((ball_e * live_w)[:, None], extra.shape[1], 1)]
    n = P.shape[1]
    present = ~np.isnan(P[:, :, 0])
    e = np.where(present, np.maximum(emis, 1e-12), 0.0)

    def end_lik(t, point):
        d = np.linalg.norm(P[t] - point, axis=1)
        return np.where(present[t], np.exp(-0.5 * (d / sigma_end) ** 2), 0.0) + 1e-12

    e[0] = end_lik(0, p_start)
    e[-1] = end_lik(T - 1, p_end)

    # all transitions t-1 -> t at once: (T-1, n, n)
    prev, cur = P[:-1], P[1:]
    d_hand = np.linalg.norm(prev[:, :, None, :] - cur[:, None, :, :], axis=-1)
    valid = ~np.isnan(d_hand)
    d0 = np.where(valid, d_hand, 0.0)
    # identity continuity: same slot, or a same-team slot near where a vanished one was
    keep = np.where(valid, np.exp(-0.5 * (d0 / handover_m) ** 2), 0.0)
    keep *= (team[:, None] == team[None, :])[None]
    stay = np.eye(n)[None] * present[1:, None, :]
    lost = (~present[1:] | ~present[:-1])[:, :, None]
    keep = np.where(lost, keep, stay)
    keep /= np.maximum(keep.sum(2, keepdims=True), 1e-12)
    # passes / losses / ball put down or picked up: nearer more likely
    pas = np.where(valid, np.exp(-d0 / pass_scale), 0.0)
    pas[:, np.arange(n), np.arange(n)] = 0
    pas /= np.maximum(pas.sum(2, keepdims=True), 1e-12)
    Ts = (1 - switch) * keep + switch * pas

    alpha = np.zeros((T, n))
    alpha[0] = e[0] / e[0].sum()
    for t in range(1, T):
        a = (alpha[t - 1] @ Ts[t - 1]) * e[t]
        s = a.sum()
        alpha[t] = a / s if s > 0 else e[t] / max(e[t].sum(), 1e-12)
    beta_ = np.ones((T, n))
    for t in range(T - 2, -1, -1):
        b = Ts[t] @ (e[t + 1] * beta_[t + 1])
        s = b.sum()
        beta_[t] = b / s if s > 0 else 1.0
    post = alpha * beta_
    post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
    if return_model:
        return post, P, Ts, e
    return post, P


def viterbi(Ts, e):
    """Most likely state sequence, log domain. Ts (T-1, n, n), e (T, n)."""
    T, n = e.shape
    le = np.log(np.maximum(e, 1e-300))
    lt = np.log(np.maximum(Ts, 1e-300))
    score = le[0] - np.log(max(e[0].sum(), 1e-300))
    back = np.zeros((T, n), dtype=np.int64)
    for t in range(1, T):
        cand = score[:, None] + lt[t - 1]
        back[t] = cand.argmax(0)
        score = cand.max(0) + le[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def segment_track(P, path, grid, pass_speed):
    """Ball along the carrier path; each change of state drawn as a straight pass at constant
    speed, centred on the change, never overlapping the neighbouring passes."""
    Pz = np.nan_to_num(P)
    est = Pz[np.arange(len(path)), path].copy()
    changes = np.flatnonzero(path[1:] != path[:-1]) + 1
    bounds = np.r_[0, changes, len(path) - 1]
    for i, k in enumerate(changes):
        dist = np.linalg.norm(est[k] - est[k - 1])
        half = max(1, int(np.ceil(dist / pass_speed * FPS / STEP / 2)))
        lo = max(k - half, (bounds[i] + k) // 2)
        hi = min(k + half, (k + bounds[i + 2]) // 2)
        if hi > lo:
            s = (grid[lo:hi + 1] - grid[lo]) / max(grid[hi] - grid[lo], 1)
            est[lo:hi + 1] = est[lo] + (est[hi] - est[lo]) * s[:, None]
    return est


def edge_velocity(ball, t, direction, k=5):
    """Ball velocity (m/s) over the k seen frames ending (direction=-1) or starting (+1) at t."""
    idx = np.arange(t - k, t + 1) if direction < 0 else np.arange(t, t + k + 1)
    idx = idx[(idx >= 0) & (idx < len(ball))]
    pts = ball[idx]
    ok = ~np.isnan(pts[:, 0])
    if ok.sum() < 3:
        return np.zeros(2)
    i = idx[ok]
    return np.polyfit(i, pts[ok], 1)[0] * FPS          # least-squares slope, per frame -> per s


def flight_states(ball, gap, grid, flight_s):
    """(T, 2, 2): the ball carrying on from the last sighting with its velocity, and arriving at
    the next sighting with its velocity, each for at most flight_s seconds then at rest."""
    p0, p1 = ball[gap.a - 1], ball[gap.b]
    v0 = edge_velocity(ball, gap.a - 1, -1)
    v1 = edge_velocity(ball, gap.b, +1)
    tau0 = np.minimum((grid - (gap.a - 1)) / FPS, flight_s)[:, None]
    tau1 = np.minimum((gap.b - grid) / FPS, flight_s)[:, None]
    return np.stack([p0 + v0 * tau0, p1 - v1 * tau1], axis=1)


_POOLED_MODELS = {}


def pooled_model(exclude_source, exclude_half=None):
    """v2 classifier trained on every match except the one scored (and its scored half)."""
    from ballfill import bank
    key = (exclude_source, exclude_half)
    if key not in _POOLED_MODELS:
        _POOLED_MODELS[key] = fit_rows(*bank.pooled(exclude_source, exclude_half))
    return _POOLED_MODELS[key]


class CarrierHMM:
    """fill method. prepare() gets one classifier per scored half; __call__ fills a gap.

    pooled=True (default): v2 features, trained on all other matches (ballfill.bank) plus, on
    our match, its other half. pooled=False: v1 features, the scored match's other half only.
    """

    def __init__(self, decode="mean", flight_s=0.0, pooled=True, dead=False, pass_speed=15.0,
                 **hmm):
        self.decode, self.hmm, self.flight_s, self.pooled = decode, hmm, flight_s, pooled
        self.pass_speed = pass_speed
        self.use_exit = True
        self.dead = dead
        self.source = "BAR-ATH"
        self._ready = None

    def set_source(self, source):
        self.source = source

    def use_production_models(self, d):
        """Classifier and dead model trained on everything (no frame is scored in production)."""
        self.source = "BAR-ATH"
        self.prepare(d, "prod", None)
        model = pooled_model("__none__")
        self.models = {1: model, 2: model}
        if self.dead:
            from ballfill import dead as D
            self.dead_model = D.model_excluding("__none__")
        self.emis_cache = {}

    def prepare(self, d, which, gaps):
        m = d.match
        if self._ready is m:
            return
        self.vel = _velocity(m.players_xy)
        self.sign, self.tags = attack_sign(m), tag_team(m)
        self.models = {}
        if self.pooled:
            for half in (1, 2):
                self.models[half] = (pooled_model(self.source, half) if self.source == "BAR-ATH"
                                     else pooled_model(self.source))
        else:
            grid = np.flatnonzero(m.graded)[::STEP]
            feats = compute_features(m, grid, self.vel, self.sign, self.tags)
            for half in (1, 2):
                tr = m.half[grid] != half
                self.models[half] = train(m, d.tac.astype(np.float64), grid[tr], feats[tr])
        if self.dead:
            from ballfill import dead as D
            self.stats = D.frame_stats(m)
            self.dead_model = D.model_excluding(self.source)
        self._ready = m
        self.emis_cache = {}

    def _dead(self, d, ball, gap, grid):
        from ballfill import dead as D
        X = D.gap_features(self.stats, ball, gap, grid, d.match.pitch)
        return self.dead_model.predict_proba(X)[:, 1], D.exit_point(ball, gap, d.match.pitch)

    def _emissions(self, d, frames, half):
        key = (half, frames[0], frames[-1], len(frames))
        if key not in self.emis_cache:
            fn = compute_features_v2 if self.pooled else compute_features
            f = fn(d.match, frames, self.vel, self.sign, self.tags)
            self.emis_cache[key] = predict(self.models[half], f)
        return self.emis_cache[key]

    def __call__(self, d, ball, gap):
        m = d.match
        grid = np.unique(np.r_[np.arange(gap.a - 1, gap.b, STEP), gap.b])
        P = m.players_xy[grid].astype(np.float64)
        emis = self._emissions(d, grid, gap.half)
        extra = flight_states(ball, gap, grid, self.flight_s) if self.flight_s > 0 else None
        dead_p, exit_pt = self._dead(d, ball, gap, grid) if self.dead else (None, None)
        if not self.use_exit:
            exit_pt = None
        post, P, Ts, e = gap_posterior(P, m.track_team, emis, ball[gap.a - 1], ball[gap.b],
                                       extra=extra, dead_p=dead_p, exit_pt=exit_pt,
                                       return_model=True, **self.hmm)
        Pz = np.nan_to_num(P)
        if self.decode == "map":
            est = Pz[np.arange(len(grid)), post.argmax(1)]
        elif self.decode == "segments":
            est = segment_track(P, viterbi(Ts, e), grid, self.pass_speed)
        else:
            est = (post[:, :, None] * Pz).sum(1)
        est[0], est[-1] = ball[gap.a - 1], ball[gap.b]
        frames = np.arange(gap.a, gap.b)
        # per-frame diagnostics of the last call, for the production output
        self.last_conf = np.interp(frames, grid, post.max(1))
        self.last_dead = np.interp(frames, grid, dead_p) if dead_p is not None else None
        return np.c_[np.interp(frames, grid, est[:, 0]), np.interp(frames, grid, est[:, 1])]

