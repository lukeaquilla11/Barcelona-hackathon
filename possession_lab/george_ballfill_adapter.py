"""Optional adapter for George's fused ball observations.

George's ``ballfill`` branch produces a frame-indexed ball track with source
and uncertainty metadata.  This module deliberately treats that track as an
observation source, not as a possession answer:

    tactical ball -> optional George observation -> derived motion/scenarios
    -> anchors -> transition hypotheses -> interval resolver -> labels

The adapter has no dependency on George's package.  That keeps the main
pipeline runnable when the optional branch or its video dependencies are not
installed.  A generated ``BAR-ATH_ball.csv`` is the integration contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ALLOWED_SOURCES = (
    "agree",
    "flight",
    "tac",
    "bc",
    "video",
)


def load_george_ballfill(path: str | Path, n_frames: int) -> pd.DataFrame:
    """Load and validate a George ball-fill CSV on the tactical frame index.

    George stores coordinates in metres.  Validation is intentionally strict:
    a frame shift would manufacture motion and could look like a useful
    transition signal while actually corrupting every downstream feature.
    """
    source_path = Path(path)
    table = pd.read_csv(source_path)
    required = {"frame", "x", "y", "source"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{source_path}: missing George columns {missing}")
    frames = pd.to_numeric(table["frame"], errors="coerce")
    if frames.isna().any() or not np.equal(frames, frames.astype(int)).all():
        raise ValueError(f"{source_path}: frame must contain integer values")
    if table["frame"].duplicated().any():
        raise ValueError(f"{source_path}: duplicate frame rows")
    if ((frames < 0) | (frames >= n_frames)).any():
        raise ValueError(f"{source_path}: frame outside 0..{n_frames - 1}")
    out = table.copy()
    out["frame"] = frames.astype(int)
    for column in ("x", "y", "sigma_m", "gap_s", "p_video", "p_carrier", "p_dead"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.sort_values("frame").reset_index(drop=True)
    return out


def add_george_ballfill_observations(
    frame_table: pd.DataFrame,
    ballfill_path: str | Path,
    *,
    pitch_length_m: float = 105.3,
    pitch_width_m: float = 68.0,
    allowed_sources: Iterable[str] = DEFAULT_ALLOWED_SOURCES,
    maximum_sigma_m: float | None = 4.0,
    allow_model_filled: bool = False,
) -> pd.DataFrame:
    """Attach George observations and fill only tactical ball gaps when safe.

    Original tactical coordinates remain available as ``ball_x_tactical`` and
    ``ball_y_tactical``.  ``ball_filled_from_george`` records exactly which
    rows changed.  By default, only fused/observed sources are accepted;
    George's carrier-HMM ``filled`` source must be requested explicitly so it
    cannot silently become a second possession decoder.
    """
    if "frame" not in frame_table or "ball_x" not in frame_table or "ball_y" not in frame_table:
        raise ValueError("frame_table must contain frame, ball_x, and ball_y")
    out = frame_table.copy()
    out["ball_x_tactical"] = pd.to_numeric(out["ball_x"], errors="coerce")
    out["ball_y_tactical"] = pd.to_numeric(out["ball_y"], errors="coerce")
    # The supplied ATD parser uses float32.  Promote the working coordinates
    # before inserting George's float64 observations so the optional adapter
    # is warning-free on current and future pandas versions.
    out["ball_x"] = out["ball_x_tactical"].astype(float)
    out["ball_y"] = out["ball_y_tactical"].astype(float)
    fill = load_george_ballfill(ballfill_path, len(out)).set_index("frame")
    index = pd.Index(pd.to_numeric(out["frame"], errors="raise").astype(int), name="frame")

    # Reindexing leaves frames outside George's output as pandas ``<NA>``.
    # Convert those to an ordinary empty string before NumPy membership tests;
    # ``np.isin`` cannot compare ``pd.NA`` safely and would abort the optional
    # integration before it reaches feature construction.
    source = fill["source"].reindex(index).astype("string").fillna("").to_numpy(dtype=str)
    out["ball_fill_source"] = source
    for column in ("sigma_m", "gap_s", "p_video", "p_carrier", "p_dead"):
        name = f"ball_fill_{column}"
        out[name] = pd.to_numeric(fill[column].reindex(index), errors="coerce").to_numpy(float) if column in fill else np.nan

    fill_x_m = pd.to_numeric(fill["x"].reindex(index), errors="coerce").to_numpy(float)
    fill_y_m = pd.to_numeric(fill["y"].reindex(index), errors="coerce").to_numpy(float)
    out["ball_fill_x"] = fill_x_m / float(pitch_length_m)
    out["ball_fill_y"] = fill_y_m / float(pitch_width_m)
    available = np.isfinite(fill_x_m) & np.isfinite(fill_y_m)
    allowed = set(str(value) for value in allowed_sources)
    if allow_model_filled:
        allowed.add("filled")
    allowed_mask = np.isin(source, list(allowed))
    sigma = out["ball_fill_sigma_m"].to_numpy(float)
    if maximum_sigma_m is not None:
        allowed_mask &= (~np.isfinite(sigma)) | (sigma <= float(maximum_sigma_m))
    out["ball_fill_available"] = available
    out["ball_fill_eligible"] = available & allowed_mask

    tactical_missing = out["ball_x_tactical"].isna().to_numpy() | out["ball_y_tactical"].isna().to_numpy()
    use_fill = tactical_missing & out["ball_fill_eligible"].to_numpy(bool)
    out["ball_filled_from_george"] = use_fill
    out.loc[use_fill, "ball_x"] = out.loc[use_fill, "ball_fill_x"]
    out.loc[use_fill, "ball_y"] = out.loc[use_fill, "ball_fill_y"]
    out["ball_observation_source"] = np.where(use_fill, "george_" + pd.Series(source, index=out.index).fillna("unknown"), "tactical")
    out["ball_observation_sigma_m"] = np.where(use_fill, sigma, 0.0)
    return out


__all__ = ["DEFAULT_ALLOWED_SOURCES", "add_george_ballfill_observations", "load_george_ballfill"]
