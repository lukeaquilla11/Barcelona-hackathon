"""One canonical composition of the supplied and engineered frame features."""

from __future__ import annotations

from typing import Any

import pandas as pd

from games import asset_paths
from starter.load_smart_tagging import load as load_smart_tagging

from possession.baseline_pipeline import (
    ClosestTeamRuleModel,
    add_match_time_columns,
    graded_mask,
    slot_team_map,
)
from possession.temporal_rules import (
    add_anchor_features,
    add_anchor_forecast_features,
    add_boundary_anchor_features,
    add_first_time_pass_features,
    add_foul_stoppage_evidence,
    add_goal_kick_features,
    add_kick_departure_features,
    add_reachability_features,
    add_reception_episode_features,
    add_reception_geometry_features,
    add_reception_outcome_features,
    add_reception_signal_features,
    add_special_anchor_features,
    apply_anchor_hierarchy,
    add_switch_confirmation_features,
    add_missing_ball_trajectory_features,
    add_missing_ball_case_features,
    add_gap_player_path_features,
    add_pipeline_evidence_features,
    add_temporal_features,
    add_transition_forecast_features,
    load_rules,
)
from possession.decoder_experiments import add_scenario_episode_features, classify_scenarios, scenario_confidence_scores
from possession.calculations.scenario_evidence import add_scenario_evidence_features


def build_engineered_features(
    model: ClosestTeamRuleModel | None = None,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the canonical frame table used by every model experiment.

    This function is deliberately model-agnostic: it calculates all supplied
    and engineered observations once. Model code should consume this table,
    not recreate a partial feature subset.
    """
    rules = load_rules() if rules is None else rules
    model = ClosestTeamRuleModel() if model is None else model
    features, _ = model.build_features()
    # Keep the kickoff-anchored analytical clock available to every downstream
    # model and diagnostic.  This is the match clock, not the video clock.
    features = add_match_time_columns(features, game=model.game)
    paths = asset_paths()
    temporal = add_temporal_features(features, rules)
    temporal = add_kick_departure_features(temporal, rules)
    temporal = add_reachability_features(temporal, slot_team_map(paths["atd_meta"]), rules)
    temporal = add_missing_ball_trajectory_features(temporal, rules)
    temporal = add_gap_player_path_features(temporal, slot_team_map(paths["atd_meta"]), rules)
    temporal = add_transition_forecast_features(temporal, rules)
    temporal = add_switch_confirmation_features(temporal, rules)
    temporal = add_anchor_features(temporal, rules)
    temporal = add_special_anchor_features(temporal, load_smart_tagging(), rules)
    temporal = add_boundary_anchor_features(temporal, rules)
    temporal = add_goal_kick_features(temporal, slot_team_map(paths["atd_meta"]))
    temporal = add_pipeline_evidence_features(temporal, rules, slot_team_map(paths["atd_meta"]))
    temporal = add_reception_outcome_features(temporal)
    temporal = add_reception_geometry_features(temporal)
    temporal = add_reception_signal_features(temporal, rules=rules)
    temporal = add_first_time_pass_features(temporal)
    temporal = add_reception_episode_features(temporal)
    temporal = add_anchor_forecast_features(temporal, rules)
    temporal = add_missing_ball_case_features(temporal, rules)
    temporal = add_foul_stoppage_evidence(temporal)
    temporal["graded"] = graded_mask(len(temporal))
    temporal["scenario_context"] = classify_scenarios(temporal)
    temporal["scenario_context_causal"] = classify_scenarios(temporal, causal=True)
    temporal = apply_anchor_hierarchy(temporal, rules)
    temporal = add_scenario_episode_features(temporal)
    # Keep linked player-path evidence model-agnostic.  Every decoder sees the
    # same scenario observations; it decides how strongly to use them.
    temporal = add_scenario_evidence_features(
        temporal, slot_team_map(paths["atd_meta"]),
        radius_m=float(rules.get("pitch", {}).get("scenario_player_radius_m", 12.0)),
    )
    # Expose the same H/A/D evidence vector to every decoder and notebook.
    return scenario_confidence_scores(temporal)
