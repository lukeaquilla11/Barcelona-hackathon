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
    "build_our_observations_with_george",
    "train_george_video_models",
]
