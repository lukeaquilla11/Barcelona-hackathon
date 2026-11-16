"""Two external per-frame inputs, read from the data folder (next to the tracking file):

  kino_events.csv         start_frame,end_frame,code   Smart Tagging events; only SET PIECES is used
  reprojection_error.csv  frame,reprojection_error     camera-to-pitch fit error of the tracking, lower = better
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def data_dir():
    from starter.games import require
    (raw,) = require(["atd_raw"])               # same folder load.py caches in
    return os.path.dirname(raw)


def _path(name, folder=None):
    return os.path.join(folder or data_dir(), name)


def set_piece_mask(n, folder=None, code="SET PIECE"):
    """True on every frame inside an event whose code starts with SET PIECE (SET PIECES in the file).
    end_frame is exclusive, as in tag_check.tag_prior."""
    k = pd.read_csv(_path("kino_events.csv", folder))
    k = k[k["code"].astype(str).str.strip().str.upper().str.startswith(code)]
    m = np.zeros(n, bool)
    for a, b in zip(k.start_frame.astype(int), k.end_frame.astype(int)):
        m[max(a, 0):min(b, n)] = True
    return m


def reliability(n, p, folder=None):
    """Weight in (0, 1] per frame from the reprojection error: 1 while the error is at or below the
    match's own median, then falling exponentially. Frames without a value get the median error
    (typical). The reference is the match's median, not a fixed number."""
    r = pd.read_csv(_path("reprojection_error.csv", folder))
    e = np.full(n, np.nan)
    f = r["frame"].to_numpy(int); ok = (f >= 0) & (f < n)
    e[f[ok]] = r["reprojection_error"].to_numpy(float)[ok]
    ref = float(np.nanmedian(e))
    e = np.where(np.isfinite(e), e, ref)
    return np.exp(-np.maximum(e - ref, 0.0) / p["reproj_scale"])


if __name__ == "__main__":
    # Debug helper: python3 solution/external.py  -> shows where the two files are expected
    for name in ("kino_events.csv", "reprojection_error.csv"):
        p = _path(name)
        print("Looking for:", p)
        print("  absolute path:", os.path.abspath(p))
        print("  file exists:  ", os.path.exists(p))