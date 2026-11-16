"""Per-frame feature table from a Match. NaN-tolerant (HistGradientBoosting handles NaN natively).

Everything is computed per half (period block) so nothing leaks across the break. Windows look both backwards and
forwards -- fine here because the task labels a whole recorded match offline.
Team-specific columns are written for home (H) and away (A); training adds the home/away-swapped copy of each match so
the model cannot learn "home is special".
"""
import warnings
import numpy as np
import pandas as pd
from .schema import Match


def _interp(a, limit):
    return pd.DataFrame(a).interpolate(limit=limit, limit_area="inside", axis=0).to_numpy()


def _vel(x, fps, k=2):
    v = np.full(x.shape, np.nan, np.float32)
    if len(x) > 2 * k:
        v[k:-k] = (x[2 * k:] - x[:-2 * k]) * fps / (2 * k)
    return v


def _roll(x, w, center=True):
    return pd.Series(x).rolling(w, center=center, min_periods=1).mean().to_numpy()


def _past_future(x, w):
    past = pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()
    fut = pd.Series(x[::-1]).rolling(w, min_periods=1).mean().to_numpy()[::-1]
    return past, fut


def _gap_len(seen):
    """Frames since the signal was last seen / until it is next seen (0 where seen)."""
    n = len(seen); idx = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, idx, -10**9))
    nxt = np.minimum.accumulate(np.where(seen, idx, 10**9)[::-1])[::-1]
    return np.where(seen, 0, np.minimum(idx - last, nxt - idx))


def block_features(P, B, team, fps, L, W):
    n, S, _ = P.shape
    seen_ball = np.isfinite(B[:, 0])
    Pf = _interp(P.reshape(n, -1), int(0.4 * fps)).reshape(P.shape).astype(np.float32)
    Bf = _interp(B, int(0.6 * fps)).astype(np.float32)
    bx, by = Bf[:, 0], Bf[:, 1]
    ball_ok = np.isfinite(bx)
    idx = np.arange(n)

    f = {}
    f["ball_seen"] = seen_ball.astype(np.float32)
    f["ball_ok"] = ball_ok.astype(np.float32)
    f["ball_gap_s"] = np.minimum(_gap_len(seen_ball), 20 * fps) / fps
    f["n_players"] = np.isfinite(Pf[:, :, 0]).sum(1).astype(np.float32)

    # --- ball kinematics / geometry
    vb = _vel(Bf, fps)
    bspeed = np.hypot(vb[:, 0], vb[:, 1])
    bacc = np.full(n, np.nan, np.float32)
    if n > 12:
        bacc[6:-6] = (bspeed[12:] - bspeed[:-12]) * fps / 12
    edge_x = np.minimum(bx, L - bx); edge_y = np.minimum(by, W - by)
    edge = np.minimum(edge_x, edge_y)
    f.update(ball_speed=bspeed, ball_acc=bacc, ball_edge=edge, ball_edge_x=edge_x, ball_edge_y=edge_y,
             ball_out=(edge < 0).astype(np.float32))

    # --- player kinematics
    pv = _vel(Pf, fps)
    pspeed = np.hypot(pv[..., 0], pv[..., 1]); pspeed[pspeed > 12] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f["med_speed"] = np.nanmedian(pspeed, axis=1)
        f["frac_still"] = np.nanmean(pspeed < 0.6, axis=1)

    # --- distances to ball
    dist = np.hypot(Pf[:, :, 0] - bx[:, None], Pf[:, :, 1] - by[:, None])
    d1 = {}
    for t, tag in ((0, "H"), (1, "A")):
        sel = team == t
        dm = np.where(sel[None, :] & np.isfinite(dist), dist, np.inf)
        srt = np.sort(dm, axis=1)[:, :3]
        if srt.shape[1] < 3:
            srt = np.pad(srt, ((0, 0), (0, 3 - srt.shape[1])), constant_values=np.inf)
        srt = np.where(np.isfinite(srt), srt, np.nan)
        for k in range(3):
            f[f"d{k + 1}_{tag}"] = srt[:, k]
        d1[tag] = srt[:, 0]
        for r in (3, 6, 12):
            f[f"cnt{r}_{tag}"] = (dm < r).sum(1).astype(np.float32)
        near_i = dm.argmin(1)
        ok = np.isfinite(srt[:, 0])
        vn = pv[idx, near_i]                                    # velocity of the nearest player
        f[f"near_speed_{tag}"] = np.where(ok, np.hypot(vn[:, 0], vn[:, 1]), np.nan)
        f[f"near_rel_{tag}"] = np.where(ok, np.hypot(vb[:, 0] - vn[:, 0], vb[:, 1] - vn[:, 1]), np.nan)
        px, py = Pf[:, sel, 0], Pf[:, sel, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f[f"spread_x_{tag}"] = np.nanstd(px, axis=1); f[f"spread_y_{tag}"] = np.nanstd(py, axis=1)
            f[f"med_speed_{tag}"] = np.nanmedian(pspeed[:, sel], axis=1)
        f[f"n_{tag}"] = np.isfinite(px).sum(1).astype(np.float32)

    diff = np.clip(d1["A"] - d1["H"], -20, 20)                  # >0: home closer
    near = np.fmin(d1["H"], d1["A"])
    f["diff"] = diff; f["near"] = near

    # --- temporal context
    for name, x in (("diff", diff), ("near", near), ("bspeed", bspeed), ("medspd", f["med_speed"]),
                    ("seen", f["ball_seen"]), ("edge", edge)):
        for w in (5, 25, 75):
            f[f"{name}_c{w}"] = _roll(x, w)
    for name, x in (("diff", diff), ("near", near)):
        past, fut = _past_future(x, 25)
        f[f"{name}_past25"], f[f"{name}_fut25"] = past, fut
        past, fut = _past_future(x, 75)
        f[f"{name}_past75"], f[f"{name}_fut75"] = past, fut
    f["near_min25"] = pd.Series(near).rolling(25, center=True, min_periods=1).min().to_numpy()
    f["bspeed_max25"] = pd.Series(bspeed).rolling(25, center=True, min_periods=1).max().to_numpy()

    # --- who last / next had the ball within reach (key for passes in flight and loose balls)
    for r in (1.5, 3.0):
        close = np.isfinite(near) & (near < r)
        last = np.maximum.accumulate(np.where(close, idx, -1))
        nxt = np.minimum.accumulate(np.where(close, idx, n + 10**6)[::-1])[::-1]
        sgn = np.sign(np.nan_to_num(diff))
        has_l, has_n = last >= 0, nxt < n
        f[f"last_team_{r}"] = np.where(has_l, sgn[np.clip(last, 0, n - 1)], np.nan)
        f[f"last_dt_{r}"] = np.where(has_l, np.minimum(idx - last, 30 * fps) / fps, np.nan)
        f[f"next_team_{r}"] = np.where(has_n, sgn[np.clip(nxt, 0, n - 1)], np.nan)
        f[f"next_dt_{r}"] = np.where(has_n, np.minimum(nxt - idx, 30 * fps) / fps, np.nan)
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


def build_features(m: Match):
    """-> (X DataFrame over the rows that belong to a half, row_index array into the match)."""
    parts, rows = [], []
    for _, a, b in m.blocks():
        parts.append(block_features(m.P[a:b + 1], m.B[a:b + 1], m.team, m.fps, m.L, m.W))
        rows.append(np.arange(a, b + 1))
    return pd.concat(parts, ignore_index=True), np.concatenate(rows)
