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

# These labels are provenance, not possession classes.  ``flight`` is built
# from observations at the gap edges; ``filled`` is produced by George's
# carrier/dead-ball model.  Neither should be described as a direct sighting.
DIRECT_OBSERVATION_SOURCES = ("agree", "tac", "bc", "video")
RECONSTRUCTED_SOURCES = ("flight", "filled")
SOURCE_CLASS_DIRECT = "direct_observation"
SOURCE_CLASS_GEOMETRIC = "geometric_reconstruction"
SOURCE_CLASS_MODEL = "model_imputation"
SOURCE_CLASS_UNTRUSTED = "untrusted_or_metadata"


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
    fill_tactical_gaps: bool = True,
    minimum_observation_confidence: float | None = None,
) -> pd.DataFrame:
    """Attach George observations and optionally fill tactical ball gaps.

    Original tactical coordinates remain available as ``ball_x_tactical`` and
    ``ball_y_tactical``.  ``ball_filled_from_george`` records exactly which
    rows changed.  Direct observations, edge-based flight reconstruction, and
    carrier/dead-ball model imputations are kept in separate provenance flags.
    George's carrier-HMM ``filled`` source must be requested explicitly so it
    cannot silently become a second possession decoder.  ``ball_fill_source_class``
    makes the contract explicit for downstream feature/scenario code:
    direct observation, geometric reconstruction, model imputation, or
    untrusted/metadata.

    ``minimum_observation_confidence`` is an optional, predeclared gate for
    coordinate replacement.  It is deliberately separate from the evidence
    columns: a George observation can still inform missing-ball and transition
    reasoning even when it is not trusted enough to become a ball coordinate.

    ``fill_tactical_gaps=False`` is the source-aware integration mode.  It
    keeps the tactical ball path untouched while exposing George's source,
    uncertainty, and missing-gap observations to later scenario and
    transition layers.  This is the preferred mode when George is being used
    as auxiliary evidence rather than as a replacement trajectory.
    """
    if minimum_observation_confidence is not None and not 0.0 <= float(minimum_observation_confidence) <= 1.0:
        raise ValueError("minimum_observation_confidence must be between 0 and 1")
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
    frame_values = index.to_numpy(dtype=int)
    previous_available = np.concatenate(([False], available[:-1]))
    consecutive_or_known_gap = np.concatenate(([False], frame_values[1:] > frame_values[:-1]))
    fill_dx_m = np.full(len(out), np.nan, dtype=float)
    fill_dy_m = np.full(len(out), np.nan, dtype=float)
    fill_dx_m[1:] = fill_x_m[1:] - fill_x_m[:-1]
    fill_dy_m[1:] = fill_y_m[1:] - fill_y_m[:-1]
    fill_displacement_m = np.hypot(fill_dx_m, fill_dy_m)
    fill_speed_mps = np.full(len(out), np.nan, dtype=float)
    elapsed_s = np.full(len(out), np.nan, dtype=float)
    elapsed_s[1:] = (frame_values[1:] - frame_values[:-1]) / 25.0
    speed_valid = available & previous_available & consecutive_or_known_gap & (elapsed_s > 0.0)
    fill_speed_mps[speed_valid] = fill_displacement_m[speed_valid] / elapsed_s[speed_valid]
    out["ball_fill_displacement_m"] = fill_displacement_m
    out["ball_fill_speed_mps"] = fill_speed_mps
    tactical_x_m = out["ball_x_tactical"].to_numpy(float) * float(pitch_length_m)
    tactical_y_m = out["ball_y_tactical"].to_numpy(float) * float(pitch_width_m)
    tactical_distance_valid = available & np.isfinite(tactical_x_m) & np.isfinite(tactical_y_m)
    out["ball_fill_distance_from_tactical_m"] = np.where(
        tactical_distance_valid,
        np.hypot(fill_x_m - tactical_x_m, fill_y_m - tactical_y_m),
        np.nan,
    )
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
    george_eligible = out["ball_fill_eligible"].to_numpy(bool)
    source_is_observed = np.isin(source, DIRECT_OBSERVATION_SOURCES)
    source_is_model_imputed = source == "filled"
    source_is_geometric = source == "flight"
    source_is_reconstructed = np.isin(source, RECONSTRUCTED_SOURCES)
    source_class = np.full(len(out), SOURCE_CLASS_UNTRUSTED, dtype=object)
    source_class[source_is_observed] = SOURCE_CLASS_DIRECT
    source_class[source_is_geometric] = SOURCE_CLASS_GEOMETRIC
    source_class[source_is_model_imputed] = SOURCE_CLASS_MODEL
    out["ball_fill_source_class"] = source_class
    out["ball_fill_observed"] = george_eligible & source_is_observed
    out["ball_fill_reconstructed"] = george_eligible & source_is_reconstructed
    out["ball_fill_model_imputed"] = available & source_is_model_imputed
    out["ball_fill_during_tactical_gap"] = tactical_missing & available
    out["ball_fill_observed_during_tactical_gap"] = (
        tactical_missing & george_eligible & source_is_observed
    )
    out["ball_fill_flight_observation"] = source == "flight"
    out["ball_fill_low_uncertainty"] = (
        np.isfinite(sigma) & (sigma <= 2.0) & source_is_observed
    )
    if maximum_sigma_m is None or float(maximum_sigma_m) <= 0.0:
        sigma_quality = np.where(np.isfinite(sigma), 1.0 / (1.0 + np.maximum(sigma, 0.0)), 1.0)
    else:
        sigma_quality = np.where(
            np.isfinite(sigma),
            np.exp(-np.maximum(sigma, 0.0) / float(maximum_sigma_m)),
            1.0,
        )
    video_probability = np.clip(
        np.nan_to_num(out["ball_fill_p_video"].to_numpy(float), nan=1.0),
        0.0,
        1.0,
    )
    source_quality = np.where(source == "video", sigma_quality * video_probability, sigma_quality)
    out["ball_fill_sigma_quality"] = np.where(
        source_is_observed | source_is_reconstructed,
        sigma_quality,
        0.0,
    )
    out["ball_fill_observation_confidence"] = np.where(
        george_eligible & (source_is_observed | source_is_reconstructed),
        np.clip(source_quality, 0.0, 1.0),
        0.0,
    )
    observation_confidence = out["ball_fill_observation_confidence"].to_numpy(float)
    previous_confidence = np.concatenate(([0.0], observation_confidence[:-1]))
    out["ball_fill_speed_confidence"] = np.where(
        speed_valid,
        np.minimum(observation_confidence, previous_confidence),
        0.0,
    )
    out["ball_fill_dead_probability"] = np.where(
        source_is_observed | source_is_reconstructed,
        np.clip(np.nan_to_num(out["ball_fill_p_dead"].to_numpy(float), nan=0.0), 0.0, 1.0),
        0.0,
    )
    confidence_gate = np.ones(len(out), dtype=bool)
    if minimum_observation_confidence is not None:
        confidence_gate = out["ball_fill_observation_confidence"].to_numpy(float) >= float(
            minimum_observation_confidence
        )
    out["ball_fill_confidence_gate_pass"] = george_eligible & confidence_gate
    use_fill = tactical_missing & george_eligible & confidence_gate & bool(fill_tactical_gaps)
    out["ball_filled_from_george"] = use_fill
    out.loc[use_fill, "ball_x"] = out.loc[use_fill, "ball_fill_x"]
    out.loc[use_fill, "ball_y"] = out.loc[use_fill, "ball_fill_y"]
    out["ball_observation_source"] = np.where(
        use_fill,
        "george_" + pd.Series(source, index=out.index).fillna("unknown"),
        "tactical",
    )
    out["ball_observation_sigma_m"] = np.where(use_fill, sigma, 0.0)
    return out


__all__ = [
    "DEFAULT_ALLOWED_SOURCES",
    "DIRECT_OBSERVATION_SOURCES",
    "RECONSTRUCTED_SOURCES",
    "SOURCE_CLASS_DIRECT",
    "SOURCE_CLASS_GEOMETRIC",
    "SOURCE_CLASS_MODEL",
    "SOURCE_CLASS_UNTRUSTED",
    "add_george_ballfill_observations",
    "load_george_ballfill",
]
