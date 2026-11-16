"""Per-frame signals derived from tracking, kinematic interpolation, and homography confidence."""
import os
import numpy as np
import pandas as pd
from load import load

L, W = 105.3, 68.0


def rolling_mean(x, half):
    """Centred rolling mean ignoring NaN."""
    v = np.isfinite(x); xs = np.where(v, x, 0.0)
    c = np.r_[0, np.cumsum(xs)]; n = np.r_[0, np.cumsum(v)]
    i = np.arange(len(x)); lo = np.maximum(0, i - half); hi = np.minimum(len(x), i + half + 1)
    cnt = n[hi] - n[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def interpolate_ball_gaps(bx, by, max_gap=25):
    """Linearly interpolate interior ball tracking gaps up to max_gap frames (1s at 25fps),
    clamping out-of-bounds values back to NaN."""
    s_x = pd.Series(bx).interpolate(method="linear", limit=max_gap, limit_area="inside")
    s_y = pd.Series(by).interpolate(method="linear", limit=max_gap, limit_area="inside")
    # Added .copy() to ensure array is writable and avoid read-only assignment error
    ix = s_x.to_numpy(np.float32).copy()
    iy = s_y.to_numpy(np.float32).copy()
    oob = (ix < -1.0) | (ix > L + 1.0) | (iy < -1.0) | (iy > W + 1.0)
    ix[oob] = np.nan
    iy[oob] = np.nan
    return ix, iy


def load_homography(n_frames, scale_factor=1.0):
    """Load homography validity and Gaussian confidence weights from reprojection error."""
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_homography.csv"),
        os.path.join(ROOT, "BAR-ATH_tactical_homography.csv"),
    ]
    homo_path = next((p for p in candidates if os.path.exists(p)), None)
    if homo_path is None:
        return np.ones(n_frames, dtype=np.float32), np.ones(n_frames, dtype=np.float32)

    df = pd.read_csv(homo_path)
    frame_col = "video_frame" if "video_frame" in df.columns else "frame"
    valid = np.ones(n_frames, dtype=np.float32)
    conf = np.ones(n_frames, dtype=np.float32)

    if frame_col in df.columns:
        idx = df[frame_col].astype(int).to_numpy()
        in_range = (idx >= 0) & (idx < n_frames)
        idx = idx[in_range]
        if "valid" in df.columns:
            v_vals = df["valid"].fillna(0).to_numpy(np.float32)[in_range]
            valid[idx] = v_vals
        if "reprojection_error" in df.columns:
            err = df["reprojection_error"].fillna(scale_factor * 2.0).to_numpy(np.float32)[in_range]
            conf[idx] = valid[idx] * np.exp(-(err ** 2) / (2.0 * (scale_factor ** 2)))
        else:
            conf[idx] = valid[idx]
    return valid, conf


def features():
    P, B, team = load()
    n = len(B)
    raw_bx, raw_by = B[:, 0].copy(), B[:, 1].copy()
    bx, by = interpolate_ball_gaps(raw_bx, raw_by, max_gap=25)
    vis = np.isfinite(bx) & np.isfinite(by)

    dx = P[:, :, 0] - bx[:, None]; dy = P[:, :, 1] - by[:, None]
    d = np.sqrt(dx * dx + dy * dy)
    with np.errstate(all="ignore"):
        dH = np.nanmin(np.where(team[None, :] == 0, d, np.nan), axis=1)
        dA = np.nanmin(np.where(team[None, :] == 1, d, np.nan), axis=1)

    # Player speeds (m/s)
    v = np.full(P.shape[:2], np.nan, np.float32)
    v[1:] = np.sqrt(((P[1:, :, :2] - P[:-1, :, :2]) ** 2).sum(-1)) * 25
    v[v > 12] = np.nan
    with np.errstate(all="ignore"):
        pspeed = np.nanmedian(v, axis=1)
    nplayers = np.isfinite(P[:, :, 0]).sum(1)

    # Ball speed (m/s) and 1-second displacement (m)
    bs = np.full(n, np.nan, np.float32)
    bs[1:] = np.hypot(np.diff(bx), np.diff(by)) * 25
    disp_1s = np.full(n, np.nan, np.float32)
    disp_1s[25:] = np.hypot(bx[25:] - bx[:-25], by[25:] - by[:-25])

    out = (bx < -0.5) | (bx > L + 0.5) | (by < -0.5) | (by > W + 0.5)
    valid_homo, homo_conf = load_homography(n, scale_factor=1.0)

    return dict(
        P=P, B=B, team=team, vis=vis, bx=bx, by=by, dH=dH, dA=dA,
        pspeed=pspeed, nplayers=nplayers, bs=bs, bspeed=bs, disp_1s=disp_1s,
        out=out, valid_homo=valid_homo, homo_conf=homo_conf
    )