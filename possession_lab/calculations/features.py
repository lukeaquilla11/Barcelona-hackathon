"""One canonical composition of the supplied and engineered frame features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from games import asset_paths
from starter.load_smart_tagging import load as load_smart_tagging

from possession_lab.baseline_pipeline import ClosestTeamRuleModel, graded_mask, slot_team_map
from possession_lab.temporal_rules import (
    add_anchor_features,
    add_anchor_forecast_features,
    add_ball_kick_episode_features,
    add_boundary_anchor_features,
    add_first_time_pass_features,
    add_reachability_features,
    add_reception_episode_features,
    add_reception_geometry_features,
    add_reception_outcome_features,
    add_reception_signal_features,
    add_special_anchor_features,
    apply_anchor_hierarchy,
    add_switch_confirmation_features,
    add_missing_ball_trajectory_features,
    add_kalman_ball_trajectory_features,
    add_gap_player_path_features,
    add_pipeline_evidence_features,
    add_temporal_features,
    add_transition_forecast_features,
    load_rules,
)
from possession_lab.decoder_experiments import (
    add_contact_arrival_coherence_features,
    add_scenario_episode_features,
    add_missing_ball_cause_features,
    add_transition_hypothesis_features,
    add_transition_trust_features,
    classify_scenarios,
)
from possession_lab.calculations.scenario_evidence import add_scenario_evidence_features
from possession_lab.george_ballfill_adapter import add_george_ballfill_observations


def build_engineered_features(
    model: ClosestTeamRuleModel | None = None,
    rules: dict[str, Any] | None = None,
    george_ballfill_path: str | Path | None = None,
    george_allow_model_filled: bool = False,
    george_evidence_only: bool = False,
    george_minimum_observation_confidence: float | None = None,
) -> pd.DataFrame:
    """Build the canonical frame table used by every model experiment.

    This function is deliberately model-agnostic: it calculates all supplied
    and engineered observations once. Model code should consume this table,
    not recreate a partial feature subset.
    """
    rules = load_rules() if rules is None else rules
    model = ClosestTeamRuleModel() if model is None else model
    features, _ = model.build_features()
    # Keep the selected game/data environment attached to the model.  This is
    # important for reusable wrappers: a caller may point the model at a
    # private organiser data directory rather than relying on the process-wide
    # MTAH_DATA_DIR default.
    paths = asset_paths(game=model.game, data_dir=model.data_dir)
    if george_ballfill_path is not None:
        features = add_george_ballfill_observations(
            features,
            george_ballfill_path,
            pitch_length_m=float(rules["pitch"]["length_m"]),
            pitch_width_m=float(rules["pitch"]["width_m"]),
            allow_model_filled=george_allow_model_filled,
            fill_tactical_gaps=not george_evidence_only,
            minimum_observation_confidence=george_minimum_observation_confidence,
        )
    temporal = add_temporal_features(features, rules)
    temporal = add_reachability_features(temporal, slot_team_map(paths["atd_meta"]), rules)
    temporal = add_missing_ball_trajectory_features(temporal, rules)
    temporal = add_kalman_ball_trajectory_features(temporal, rules)
    temporal = add_gap_player_path_features(temporal, slot_team_map(paths["atd_meta"]), rules)
    temporal = add_transition_forecast_features(temporal, rules)
    temporal = add_switch_confirmation_features(temporal, rules)
    temporal = add_anchor_features(temporal, rules)
    temporal = add_special_anchor_features(
        temporal,
        load_smart_tagging(game=model.game, data_dir=model.data_dir),
        rules,
    )
    temporal = add_boundary_anchor_features(temporal, rules)
    temporal = add_pipeline_evidence_features(temporal, rules, slot_team_map(paths["atd_meta"]))
    temporal = add_reception_outcome_features(temporal)
    temporal = add_reception_geometry_features(temporal)
    temporal = add_reception_signal_features(temporal, rules=rules)
    temporal = add_first_time_pass_features(temporal)
    temporal = add_reception_episode_features(temporal)
    temporal = add_ball_kick_episode_features(temporal, rules)
    temporal = add_missing_ball_cause_features(temporal)
    temporal = add_anchor_forecast_features(temporal, rules)
    temporal["graded"] = graded_mask(len(temporal), game=model.game)
    temporal["scenario_context"] = classify_scenarios(temporal)
    temporal["scenario_context_causal"] = classify_scenarios(temporal, causal=True)
    temporal = apply_anchor_hierarchy(temporal, rules)
    temporal = add_contact_arrival_coherence_features(temporal)
    temporal = add_transition_trust_features(
        temporal,
        max_anchor_gap_frames=int(rules["transition_forecast"]["anchor_lookahead_frames"]),
        minimum_anchor_confidence=0.85,
    )
    temporal = add_transition_hypothesis_features(temporal)
    temporal = add_scenario_episode_features(temporal)
    return add_scenario_evidence_features(
        temporal,
        slot_team_map(paths["atd_meta"]),
    )
