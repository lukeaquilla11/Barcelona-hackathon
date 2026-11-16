"""Parse the tactical ATD into numpy arrays once and cache them."""
import os, sys, io
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from starter.games import require, GAMES, GAME
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import team_of_each_track


def load():
    """Positions in metres: P[frame, slot, (x, y, speed)], B[frame, (x, y, speed)], team[slot].
    The parsed arrays are cached next to the data (never inside the repository)."""
    meta, raw = require(["atd_meta", "atd_raw"])
    CACHE = os.path.join(os.path.dirname(raw), "cache_tactical.npz")
    if os.path.exists(CACHE) and os.path.getmtime(CACHE) > os.path.getmtime(raw):
        d = np.load(CACHE); return d["P"], d["B"], d["team"]
    with open(raw) as f:
        txt = f.read().replace(":", ",").replace(";", ",")
    arr = pd.read_csv(io.StringIO(txt), header=None, na_values="NaN").to_numpy(np.float32)
    n = GAMES[GAME]["n_video_frames"]
    frames = arr[:, 0].astype(int) - 1
    P = np.full((n, 46, 3), np.nan, np.float32); B = np.full((n, 3), np.nan, np.float32)
    P[frames] = arr[:, 1:139].reshape(-1, 46, 3); B[frames] = arr[:, 139:142]
    L, W = GAMES[GAME]["pitch"]
    P[..., 0] *= L; P[..., 1] *= W; B[:, 0] *= L; B[:, 1] *= W
    team = team_of_each_track(meta)
    np.savez(CACHE, P=P, B=B, team=team)
    return P, B, team

if __name__ == "__main__":
    P, B, team = load(); print(P.shape, B.shape, np.isfinite(B[:,0]).mean(), team)
