"""Stable orchestration wrappers for the modular possession pipeline.

The lower-level modules remain independently testable. This facade gives
notebooks, organiser scripts, and future models one consistent boundary:

    load data -> source-aware ball observations -> feature engineering ->
    base scenarios -> restart/dead-ball evidence -> anchors ->
    transition hypotheses -> interval resolver -> optional sequence smoother ->
    H/A/D/X labels

The feature builder keeps a base scenario pass separate from its
anchor-informed refinement. This is intentional: scenarios describe observed
football context, while anchors and resolvers decide ownership.

No function in this module reads public ground truth during construction.
Ground truth belongs only to the evaluator and error-audit notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from possession_lab.baseline_pipeline import ClosestTeamRuleModel, load_supplied_frame_table
from possession_lab.calculations.features import build_engineered_features
from possession_lab.decoder_experiments import (
    add_observation_probability_features,
    add_scenario_episode_features,
    add_transition_channel_profile,
    add_transition_hypothesis_features,
    build_transition_hypothesis_ledger,
    scenario_confidence_scores,
)
from possession_lab.haiqal_baseline import add_evidence_features as add_haiqal_evidence_features
from possession_lab.models.pipelines import (
    PipelineSpec,
    apply_pipeline_spec,
    available_pipeline_specs,
)
from possession_lab.temporal_rules import load_rules


def load_data(
    *,
    game: str = "BAR-ATH",
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the raw tactical frame table and its audit report."""
    return load_supplied_frame_table(game=game, data_dir=data_dir)


def build_observations(
    *,
    model: ClosestTeamRuleModel | None = None,
    rules: dict[str, Any] | None = None,
    game: str = "BAR-ATH",
    data_dir: str | Path | None = None,
    george_ballfill_path: str | Path | None = None,
    george_allow_model_filled: bool = False,
    george_evidence_only: bool = False,
    george_minimum_observation_confidence: float | None = None,
    include_haiqal_observations: bool = False,
) -> pd.DataFrame:
    """Build the complete shared observation/calculation table.

    ``george_evidence_only=True`` preserves the tactical ball trajectory and
    exposes George's source/uncertainty observations to downstream scenarios
    and transition hypotheses.  ``include_haiqal_observations=True`` attaches
    Haiqal's raw closeness/speed/dead-ball clues to the same observation table;
    they remain bounded evidence and never become labels. The feature layer
    performs a base scenario pass before anchors and an anchor-refined pass
    afterward. It is the safe integration mode for the reference model;
    coordinate filling remains an explicit ablation.
    """
    resolved_model = model or ClosestTeamRuleModel(game=game, data_dir=data_dir)
    observations = build_engineered_features(
        model=resolved_model,
        rules=rules,
        george_ballfill_path=george_ballfill_path,
        george_allow_model_filled=george_allow_model_filled,
        george_evidence_only=george_evidence_only,
        george_minimum_observation_confidence=george_minimum_observation_confidence,
    )
    if include_haiqal_observations:
        observations = add_haiqal_observations(
            observations,
            game=resolved_model.game,
            data_dir=resolved_model.data_dir,
        )
    return observations


def detect_scenario_evidence(
    observations: pd.DataFrame,
    *,
    include_episode_evidence: bool = True,
) -> pd.DataFrame:
    """Attach scenario confidence and episode evidence without labels."""
    out = scenario_confidence_scores(observations.copy())
    if include_episode_evidence:
        out = add_scenario_episode_features(out)
    return out


def build_probability_evidence(
    observations: pd.DataFrame,
    *,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Attach the transparent, currently uncalibrated evidence vector."""
    return add_observation_probability_features(
        detect_scenario_evidence(observations),
        temperature=temperature,
    )


def build_transition_evidence(observations: pd.DataFrame) -> pd.DataFrame:
    """Build interval-level transition/no-switch hypotheses."""
    return build_transition_hypothesis_ledger(build_probability_evidence(observations))


def build_transition_channel_evidence(
    observations: pd.DataFrame,
    *,
    profile: str = "balanced",
) -> pd.DataFrame:
    """Build explicit channel-weighted transition hypotheses.

    This is an experiment boundary: it composes anchor, ball-motion,
    player-arrival, restart-continuity, and bounded Haiqal observations, then
    rebuilds the interval hypotheses. It never produces final labels or reads
    ground truth; a resolver remains the next stage.
    """
    profiled = add_transition_channel_profile(observations, profile=profile)
    return add_transition_hypothesis_features(profiled)


def add_haiqal_observations(
    observations: pd.DataFrame,
    *,
    game: str = "BAR-ATH",
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Add Haiqal's raw distance/speed/dead-ball clues as observations only."""
    return add_haiqal_evidence_features(
        observations,
        game=game,
        data_dir=data_dir,
    )


def run_model(
    observations: pd.DataFrame,
    *,
    model_name: str = "protected_aligned_reception",
    spec: PipelineSpec | None = None,
    rules: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply one registered decision model to a shared evidence table."""
    resolved_rules = load_rules() if rules is None else rules
    if spec is None:
        specs = available_pipeline_specs()
        if model_name not in specs:
            raise ValueError(f"unknown model {model_name!r}; choose from {sorted(specs)}")
        resolved_spec = specs[model_name]
    else:
        resolved_spec = spec
    return apply_pipeline_spec(observations, resolved_spec, resolved_rules)


__all__ = [
    "add_haiqal_observations",
    "build_observations",
    "build_probability_evidence",
    "build_transition_channel_evidence",
    "build_transition_evidence",
    "detect_scenario_evidence",
    "load_data",
    "run_model",
]
