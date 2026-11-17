"""End-to-end George ball-observation stage.

This wrapper starts from the raw match bundle and returns George's generated
ball observations as a dataframe.  The generated CSV is an optional audit
artifact, never a required input to the possession pipeline.

George's model is an observation model: video detection/fusion plus a
carrier/dead-ball HMM for unresolved gaps.  Its output is deliberately passed
to our feature/scenario/anchor/transition layers rather than copied as
possession labels.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _run_module(
    module: str,
    args: Sequence[str],
    *,
    data_dir: str | Path,
) -> None:
    """Run a George module in a clean subprocess with the raw-data location set."""
    environment = os.environ.copy()
    environment["MTAH_DATA_DIR"] = str(Path(data_dir).resolve())
    subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def train_george_video_models(
    *,
    data_dir: str | Path,
    step: str = "all",
) -> None:
    """Train/detect George's tactical-video ball model.

    ``step='all'`` runs the branch's extraction, cross-half training and
    detection sequence.  Existing cache files are reused by the underlying
    branch code where possible, so the organiser-facing notebook can expose a
    rerun toggle without retraining on every execution.
    """
    if step not in {"extract", "train", "detect", "all"}:
        raise ValueError("step must be extract, train, detect, or all")
    _run_module("ballfill.video", [step], data_dir=data_dir)


def build_george_observations_from_raw(
    *,
    data_dir: str | Path,
    game: str = "BAR-ATH",
) -> pd.DataFrame:
    """Run George's full raw-data observation pipeline and return its dataframe.

    The branch writes ``cache/{game}_ball.csv`` as a reproducible audit artifact;
    this function immediately loads that generated result so downstream code
    can use an in-memory dataframe rather than requiring a precomputed CSV.
    """
    _run_module("ballfill.run", [], data_dir=data_dir)
    generated = ROOT / "cache" / f"{game}_ball.csv"
    if not generated.exists():
        raise FileNotFoundError(f"George pipeline did not create {generated}")
    return pd.read_csv(generated)


def build_george_hmm_observations_from_raw(
    *,
    data_dir: str | Path,
    game: str = "BAR-ATH",
) -> pd.DataFrame:
    """Build George-style fused/HMM observations without video detector caches.

    The alternate tactical/broadcast observations are fused first.  Remaining
    two-sided gaps are filled by George's pooled carrier/dead-ball HMM, whose
    production models are trained from the seven IDSSE matches and the
    non-scored portions of the challenge match.  The result is still an
    observation table: it exposes ``source``, ``sigma_m``, ``p_carrier`` and
    ``p_dead`` but never creates possession labels.

    This mode is useful for a reproducible path/receiver experiment when the
    optional video-detector cache is unavailable.  It keeps the same raw-data
    contract as George's full runner and leaves video sightings for the full
    ``build_george_observations_from_raw`` path.
    """
    from ballfill.carrier import CarrierHMM
    from ballfill.data import load_ball_data, runs
    from ballfill.evaluate import Gap
    from ballfill.fuse import SOURCES, fuse
    from ballfill.run import CONFIG

    data = load_ball_data(game=game, data_dir=data_dir)
    match = data.match
    ball, source_codes, sigma, _ = fuse(data)
    sources = np.array(SOURCES, dtype=object)[source_codes]

    output_ball = ball.copy()
    carrier_probability = np.full(data.n, np.nan, dtype=float)
    dead_probability = np.full(data.n, np.nan, dtype=float)
    gap_seconds = np.full(data.n, np.nan, dtype=float)
    hmm = CarrierHMM(**CONFIG)
    hmm.use_production_models(data)
    for half in (1, 2):
        half_frames = np.flatnonzero(match.half == half)
        if not half_frames.size:
            continue
        first, last = int(half_frames[0]), int(half_frames[-1])
        if not np.isfinite(output_ball[first]).all():
            output_ball[first] = (match.pitch[0] / 2.0, match.pitch[1] / 2.0)
            sources[first], sigma[first] = "kickoff", 1.0
        seen = np.isfinite(output_ball[:, 0])
        for start, end in runs(~seen & (match.half == half)):
            if end > last:
                output_ball[start:end] = output_ball[start - 1]
                sources[start:end] = "filled"
                gap_seconds[start:end] = (end - start) / 25.0
                # Coordinate uncertainty for a trailing hold is deliberately
                # conservative.  The downstream adapter also receives the
                # HMM confidence separately and can cap path support with it.
                sigma[start:end] = 8.0
                continue
            gap = Gap(start, end, half, np.arange(start, end), np.zeros((end - start, 2)))
            filled = hmm(data, output_ball, gap)
            frames = np.arange(start, end)
            output_ball[frames] = filled
            sources[frames] = "filled"
            gap_seconds[frames] = (end - start) / 25.0
            carrier_probability[frames] = hmm.last_conf
            if hmm.last_dead is not None:
                dead_probability[frames] = hmm.last_dead
            # Do not claim a calibrated metre error for model-imputed points.
            # ``p_carrier`` is the usable model-confidence channel here;
            # direct/fused observations retain their geometric sigma values.
            sigma[start:end] = 8.0

    graded = match.graded
    return pd.DataFrame(
        {
            "frame": np.flatnonzero(graded),
            "x": np.round(output_ball[graded, 0], 3),
            "y": np.round(output_ball[graded, 1], 3),
            "source": sources[graded],
            "sigma_m": np.round(sigma[graded], 2),
            "gap_s": np.round(gap_seconds[graded], 2),
            "p_video": np.nan,
            "p_carrier": np.round(carrier_probability[graded], 3),
            "p_dead": np.round(dead_probability[graded], 3),
        }
    )


def build_our_observations_with_george(
    *,
    data_dir: str | Path,
    game: str = "BAR-ATH",
    evidence_only: bool = True,
    allow_model_filled: bool = False,
    minimum_confidence: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run George, then pass its observations into our feature layer.

    Returns ``(our_features, george_table)``.  George's HMM output remains
    source/uncertainty evidence; our downstream resolver still owns H/A/D/X.
    """
    from possession_lab.baseline_pipeline import ClosestTeamRuleModel
    from possession_lab.calculations.features import build_engineered_features

    george_table = build_george_observations_from_raw(data_dir=data_dir, game=game)
    model = ClosestTeamRuleModel(game=game, data_dir=data_dir)
    features = build_engineered_features(
        model=model,
        george_ballfill_table=george_table,
        george_evidence_only=evidence_only,
        george_allow_model_filled=allow_model_filled,
        george_minimum_observation_confidence=minimum_confidence,
    )
    return features, george_table


__all__ = [
    "build_george_observations_from_raw",
    "build_george_hmm_observations_from_raw",
    "build_our_observations_with_george",
    "train_george_video_models",
]
