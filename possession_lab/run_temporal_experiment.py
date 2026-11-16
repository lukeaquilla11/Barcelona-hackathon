"""Run the first anchor/temporal-rule experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from possession_lab.baseline_pipeline import ClosestTeamRuleModel, graded_mask, slot_team_map
from games import asset_paths
from starter.load_smart_tagging import load as load_smart_tagging
from possession_lab.calculations.features import build_engineered_features
from possession_lab.temporal_rules import (
    add_anchor_features,
    add_reachability_features,
    add_special_anchor_features,
    add_transition_forecast_features,
    apply_confirmed_transition_switches,
    build_scenario_episodes,
    add_temporal_features,
    bridge_same_team_anchor_gaps,
    load_rules,
)
from possession_lab.scenario_decoder import decode_scenario_labels
from possession_lab.sequence_models import (
    AnchorInitializedHMM,
    AnchorInitializedHSMM,
    BayesianAnchorHMM,
    RuleConstrainedViterbi,
    SelectiveRuleViterbi,
    ScenarioHSMM,
)
from possession_lab.decoder_experiments import (
    classify_scenarios,
    confidence_multiscale_labels,
    refine_dead_ball_labels,
    restart_boundary_dead_ball_labels,
    scenario_gated_labels,
    stop_dead_support_at_restart_control,
    protect_stable_control_anchors,
    _remove_short_dead_runs,
)


ROOT = Path(__file__).resolve().parents[1]


def write_anchor_audit(features: pd.DataFrame, path: Path) -> None:
    """Audit algorithmic anchor precision on the public labels only.

    The public labels are a diagnostic check, never an input to feature
    construction or threshold fitting.
    """
    gt_path = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    ground_truth = pd.read_csv(gt_path)
    gt = pd.Series(ground_truth["label"].to_numpy(), index=ground_truth["frame"].astype(int))
    rows = []
    graded = features["graded"].to_numpy(bool)
    for name, column in (
        ("stable_control", "anchor_confident"),
        ("reception", "reception_anchor"),
        ("genuine_reception", "genuine_reception_anchor"),
        ("transition", "transition_anchor"),
        ("restart_confirmed", "restart_confirmed_anchor"),
    ):
        mask = graded & features[column].to_numpy(bool)
        frames = features.loc[mask, "frame"].astype(int)
        teams = features.loc[mask, "anchor_team"].astype(object)
        observed = gt.reindex(frames).to_numpy()
        expected = teams.to_numpy()
        comparable = pd.notna(observed) & pd.notna(expected) & np.isin(observed, ["H", "A"])
        rows.append({
            "anchor_type": name,
            "frames": int(mask.sum()),
            "graded_coverage": float(mask.sum() / max(1, graded.sum())),
            "comparable_frames": int(comparable.sum()),
            "public_precision": float((observed[comparable] == expected[comparable]).mean()) if comparable.any() else None,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    rules = load_rules()
    model = ClosestTeamRuleModel()
    _, report = model.build_features()
    temporal = build_engineered_features(model=model, rules=rules)
    episodes = build_scenario_episodes(temporal, rules)
    # Upgrade experiment: recover short, internal ball-flight evidence from
    # interpolated ball coordinates. The original temporal table remains the
    # control; this copy is the only one allowed to use the new signal.
    interpolated_flight_temporal = temporal.copy()
    interpolated_flight_temporal["recent_flight_evidence"] = temporal[
        "recent_flight_evidence_interpolated"
    ]
    interpolated_flight_temporal["ball_flight_candidate"] = temporal[
        "ball_flight_candidate_interpolated"
    ]
    # Anchor-first experiment: retain D for tagged set pieces, use H/A only
    # where the algorithmic confidence rule fires, then bridge justified gaps.
    # Unresolved frames remain X instead of silently inheriting nearest-player
    # noise. This makes the experiment materially different from the baseline.
    in_play = graded_mask(len(temporal))
    labels = pd.Series("X", index=temporal.index, dtype="object")
    labels.loc[in_play & temporal["smart_set_piece"].to_numpy(bool)] = "D"
    labels.loc[in_play & temporal["anchor_team"].notna().to_numpy()] = temporal.loc[
        in_play & temporal["anchor_team"].notna(), "anchor_team"
    ]
    labels = labels.to_numpy(dtype=object)
    labels, bridge_report = bridge_same_team_anchor_gaps(labels, temporal, rules)
    labels, transition_report = apply_confirmed_transition_switches(labels, temporal, rules)
    pressure_features = temporal.copy()
    pressure_features["anchor_team"] = pressure_features["pressure_anchor_team"]
    pressure_labels = pd.Series("X", index=temporal.index, dtype="object")
    pressure_labels.loc[in_play & temporal["smart_set_piece"].to_numpy(bool)] = "D"
    pressure_labels.loc[in_play & pressure_features["anchor_team"].notna().to_numpy()] = pressure_features.loc[
        in_play & pressure_features["anchor_team"].notna(), "anchor_team"
    ]
    pressure_labels, pressure_bridge_report = bridge_same_team_anchor_gaps(
        pressure_labels.to_numpy(dtype=object), pressure_features, rules
    )
    adaptive_labels = confidence_multiscale_labels(
        temporal,
        short_window=50,
        long_window=125,
        disagreement_margin_m=1.5,
        minimum_switch_run=12,
        dead_ball_mode="tagged_slow",
    )
    adaptive_no_dead_labels = confidence_multiscale_labels(
        temporal,
        short_window=50,
        long_window=125,
        disagreement_margin_m=1.5,
        minimum_switch_run=25,
        maximum_internal_spike=25,
        dead_ball_mode="none",
    )
    adaptive_long_hysteresis_labels = confidence_multiscale_labels(
        temporal,
        short_window=50,
        long_window=125,
        disagreement_margin_m=2.5,
        minimum_switch_run=50,
        maximum_internal_spike=50,
        dead_ball_mode="tagged_slow",
    )
    adaptive_transition_gate_labels = confidence_multiscale_labels(
        temporal,
        short_window=50,
        long_window=125,
        disagreement_margin_m=1.5,
        minimum_switch_run=25,
        maximum_internal_spike=25,
        dead_ball_mode="tagged_slow",
        require_transition_support=True,
    )
    adaptive_reachability_gate_labels = confidence_multiscale_labels(
        temporal,
        short_window=50,
        long_window=125,
        disagreement_margin_m=1.5,
        minimum_switch_run=25,
        maximum_internal_spike=25,
        dead_ball_mode="tagged_slow",
        require_reachability_support=True,
    )
    adaptive_dead_trim_labels = confidence_multiscale_labels(
        temporal,
        short_window=50,
        long_window=125,
        disagreement_margin_m=2.5,
        minimum_switch_run=50,
        maximum_internal_spike=50,
        dead_ball_mode="tagged_edges",
    )
    adaptive_longer_labels = confidence_multiscale_labels(
        temporal,
        short_window=50,
        long_window=125,
        disagreement_margin_m=2.5,
        minimum_switch_run=75,
        maximum_internal_spike=75,
        dead_ball_mode="tagged_slow",
    )
    adaptive_100_labels = confidence_multiscale_labels(
        temporal,
        short_window=50,
        long_window=100,
        disagreement_margin_m=2.5,
        minimum_switch_run=50,
        maximum_internal_spike=50,
        dead_ball_mode="tagged_slow",
    )
    scenario_gated_control_labels = scenario_gated_labels(temporal)
    scenario_conservative_labels = scenario_gated_labels(
        temporal,
        allow_pressured_control=False,
    )
    scenario_receiver_switch_labels = scenario_gated_labels(
        temporal,
        allow_receiver_switch=True,
    )
    scenario_strict_receiver_labels = scenario_gated_labels(
        temporal,
        minimum_control_persistence=0.90,
        allow_pressured_control=False,
        allow_receiver_switch=True,
    )
    scenario_combined_labels = scenario_gated_labels(
        temporal,
        allow_receiver_switch=True,
        maximum_internal_spike=75,
    )
    scenario_bidirectional_labels = scenario_gated_labels(
        temporal,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        maximum_internal_spike=75,
    )
    scenario_bidirectional_long_reception_labels = scenario_gated_labels(
        temporal,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        receiver_horizon_frames=75,
        maximum_internal_spike=75,
    )
    scenario_bidirectional_receiver_agreement_labels = scenario_gated_labels(
        temporal,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        receiver_horizon_frames=75,
        require_receiver_path_agreement=True,
        maximum_internal_spike=75,
    )
    # New pipeline candidate: permit a cross-team switch only when the
    # anchor-only forecast also confirms a future touch and the proposed team.
    # This keeps the forecast from becoming a frame-level label by itself.
    scenario_bidirectional_anchor_forecast_labels = scenario_gated_labels(
        temporal,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        receiver_horizon_frames=75,
        require_anchor_forecast_confirmation=True,
        maximum_internal_spike=75,
    )
    scenario_d_speed2_labels = refine_dead_ball_labels(
        temporal, scenario_combined_labels, speed_threshold_mps=2.0, minimum_candidate_run=5
    )
    scenario_d_speed3_labels = refine_dead_ball_labels(
        temporal, scenario_combined_labels, speed_threshold_mps=3.0, minimum_candidate_run=5
    )
    scenario_d_speed3_long_labels = refine_dead_ball_labels(
        temporal, scenario_combined_labels, speed_threshold_mps=3.0, minimum_candidate_run=10
    )
    scenario_d_locked_labels = refine_dead_ball_labels(
        temporal,
        scenario_combined_labels,
        speed_threshold_mps=3.0,
        minimum_candidate_run=5,
        require_existing_d_adjacency=True,
    )
    scenario_d_locked_long_labels = refine_dead_ball_labels(
        temporal,
        scenario_combined_labels,
        speed_threshold_mps=3.0,
        minimum_candidate_run=10,
        require_existing_d_adjacency=True,
    )
    scenario_d_restart_boundary_labels = restart_boundary_dead_ball_labels(
        temporal, scenario_combined_labels, active_speed_mps=1.0, minimum_active_run=3
    )
    scenario_d_restart_boundary_strict_labels = restart_boundary_dead_ball_labels(
        temporal, scenario_combined_labels, active_speed_mps=2.0, minimum_active_run=3
    )
    scenario_bidirectional_d_labels = refine_dead_ball_labels(
        temporal,
        scenario_bidirectional_labels,
        speed_threshold_mps=3.0,
        minimum_candidate_run=10,
    )
    # Apply the same duration guard as the best ensemble, but to the
    # bidirectional bridge directly.  This tests whether the D improvement is
    # compatible with the better two-sided reception logic rather than being
    # an artefact of the older combined decoder.
    scenario_bidirectional_d_duration_labels = _remove_short_dead_runs(
        scenario_bidirectional_d_labels, minimum_run=25
    )
    scenario_d_activity_labels = refine_dead_ball_labels(
        temporal,
        scenario_combined_labels,
        speed_threshold_mps=3.0,
        minimum_candidate_run=10,
        maximum_player_activity_mps=1.5,
    )
    scenario_labels, decoder_report = decode_scenario_labels(temporal, rules)
    scenario_ensemble_labels = scenario_bidirectional_labels.copy()
    scenario_d_support = (
        temporal["graded"].to_numpy(bool)
        & temporal["smart_set_piece"].to_numpy(bool)
        & (scenario_labels == "D")
        & np.isin(temporal["scenario_context"].to_numpy(object), ["DEAD", "UNKNOWN", "OCCLUDED"])
    )
    scenario_ensemble_labels[scenario_d_support] = "D"
    scenario_ensemble_restart_guard_labels = scenario_bidirectional_labels.copy()
    scenario_d_restart_guard_support = scenario_d_support & ~temporal["restart_ball_active"].to_numpy(bool)
    scenario_ensemble_restart_guard_labels[scenario_d_restart_guard_support] = "D"
    scenario_ensemble_restart_guard_labels = _remove_short_dead_runs(
        scenario_ensemble_restart_guard_labels, minimum_run=25
    )
    scenario_d_restart_control_support = stop_dead_support_at_restart_control(
        temporal, scenario_d_support
    )
    scenario_ensemble_restart_control_labels = scenario_bidirectional_labels.copy()
    scenario_ensemble_restart_control_labels[scenario_d_restart_control_support] = "D"
    scenario_ensemble_restart_control_labels = _remove_short_dead_runs(
        scenario_ensemble_restart_control_labels, minimum_run=25
    )
    scenario_ensemble_duration_labels = _remove_short_dead_runs(scenario_ensemble_labels, minimum_run=25)
    scenario_long_ensemble_labels = scenario_bidirectional_long_reception_labels.copy()
    scenario_long_ensemble_labels[scenario_d_support] = "D"
    scenario_long_ensemble_duration_labels = _remove_short_dead_runs(
        scenario_long_ensemble_labels, minimum_run=25
    )
    scenario_receiver_agreement_ensemble_labels = scenario_bidirectional_receiver_agreement_labels.copy()
    scenario_receiver_agreement_ensemble_labels[scenario_d_support] = "D"
    scenario_receiver_agreement_ensemble_labels = _remove_short_dead_runs(
        scenario_receiver_agreement_ensemble_labels, minimum_run=25
    )
    scenario_interpolated_flight_labels = scenario_gated_labels(
        interpolated_flight_temporal,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        receiver_horizon_frames=75,
        maximum_internal_spike=75,
    )
    scenario_interpolated_flight_ensemble_labels = scenario_interpolated_flight_labels.copy()
    scenario_interpolated_flight_ensemble_labels[scenario_d_support] = "D"
    scenario_interpolated_flight_ensemble_labels = _remove_short_dead_runs(
        scenario_interpolated_flight_ensemble_labels, minimum_run=25
    )
    scenario_ensemble_anchor_protected_labels = protect_stable_control_anchors(
        temporal, scenario_ensemble_duration_labels, minimum_run=8
    )
    # A dead-ball interval shorter than 0.6 seconds is unlikely to be a useful
    # possession spell; this tests a moderate duration prior between the
    # 5-frame D detector and the current 1-second ensemble guard.
    scenario_ensemble_duration_short_labels = _remove_short_dead_runs(
        scenario_ensemble_labels, minimum_run=15
    )
    scenario_ensemble_duration_long_labels = _remove_short_dead_runs(scenario_ensemble_labels, minimum_run=50)
    scenario_activity_ensemble_labels = refine_dead_ball_labels(
        temporal,
        scenario_bidirectional_labels,
        speed_threshold_mps=3.0,
        minimum_candidate_run=10,
        maximum_player_activity_mps=1.5,
    )
    scenario_activity_ensemble_labels = _remove_short_dead_runs(
        scenario_activity_ensemble_labels, minimum_run=25
    )
    hmm_labels = AnchorInitializedHMM(rules).fit(temporal).predict(temporal)
    hsmm_labels = AnchorInitializedHSMM(rules).fit(temporal).predict(temporal)
    bayesian_hmm_labels = BayesianAnchorHMM(rules).fit(temporal).predict(temporal)
    constrained_viterbi_labels = RuleConstrainedViterbi(rules).fit(temporal).predict(temporal)
    selective_viterbi_labels = SelectiveRuleViterbi(rules).fit(temporal).predict(temporal)
    scenario_hsmm_labels = ScenarioHSMM(rules, minimum_flight_speed_mps=0.0).fit(temporal).predict(temporal)
    scenario_hsmm_strict_labels = ScenarioHSMM(
        rules,
        minimum_flight_speed_mps=float(rules["bridging"]["minimum_ball_speed_mps"]),
    ).fit(temporal).predict(temporal)
    restart_hsmm_labels = ScenarioHSMM(
        rules,
        minimum_flight_speed_mps=0.0,
        use_restart_boundaries=True,
        minimum_restart_persistence=0.60,
    ).fit(temporal).predict(temporal)
    restart_hsmm_strict_labels = ScenarioHSMM(
        rules,
        minimum_flight_speed_mps=0.0,
        use_restart_boundaries=True,
        minimum_restart_persistence=0.80,
    ).fit(temporal).predict(temporal)
    output_path = ROOT / "evaluation" / "results" / "baseline_temporal_anchor.csv"
    model.write_submission(output_path, labels)
    scenario_output_path = ROOT / "evaluation" / "results" / "baseline_scenario_decoder.csv"
    model.write_submission(scenario_output_path, scenario_labels)
    hmm_output_path = ROOT / "evaluation" / "results" / "baseline_hmm_anchor.csv"
    model.write_submission(hmm_output_path, hmm_labels)
    hsmm_output_path = ROOT / "evaluation" / "results" / "baseline_hsmm_anchor.csv"
    model.write_submission(hsmm_output_path, hsmm_labels)
    bayesian_hmm_output_path = ROOT / "evaluation" / "results" / "baseline_bayesian_hmm.csv"
    model.write_submission(bayesian_hmm_output_path, bayesian_hmm_labels)
    constrained_viterbi_output_path = ROOT / "evaluation" / "results" / "baseline_rule_constrained_viterbi.csv"
    model.write_submission(constrained_viterbi_output_path, constrained_viterbi_labels)
    selective_viterbi_output_path = ROOT / "evaluation" / "results" / "baseline_selective_rule_viterbi.csv"
    model.write_submission(selective_viterbi_output_path, selective_viterbi_labels)
    scenario_hsmm_output_path = ROOT / "evaluation" / "results" / "baseline_scenario_hsmm.csv"
    model.write_submission(scenario_hsmm_output_path, scenario_hsmm_labels)
    scenario_hsmm_strict_output_path = ROOT / "evaluation" / "results" / "baseline_scenario_hsmm_strict.csv"
    model.write_submission(scenario_hsmm_strict_output_path, scenario_hsmm_strict_labels)
    restart_hsmm_output_path = ROOT / "evaluation" / "results" / "baseline_restart_hsmm.csv"
    model.write_submission(restart_hsmm_output_path, restart_hsmm_labels)
    restart_hsmm_strict_output_path = ROOT / "evaluation" / "results" / "baseline_restart_hsmm_strict.csv"
    model.write_submission(restart_hsmm_strict_output_path, restart_hsmm_strict_labels)
    pressure_output_path = ROOT / "evaluation" / "results" / "baseline_pressure_anchor.csv"
    model.write_submission(pressure_output_path, pressure_labels)
    adaptive_output_path = ROOT / "evaluation" / "results" / "baseline_adaptive_multiscale.csv"
    model.write_submission(adaptive_output_path, adaptive_labels)
    adaptive_no_dead_output_path = ROOT / "evaluation" / "results" / "baseline_adaptive_no_dead.csv"
    model.write_submission(adaptive_no_dead_output_path, adaptive_no_dead_labels)
    adaptive_long_output_path = ROOT / "evaluation" / "results" / "baseline_adaptive_long_hysteresis.csv"
    model.write_submission(adaptive_long_output_path, adaptive_long_hysteresis_labels)
    scenario_gated_output_path = ROOT / "evaluation" / "results" / "baseline_scenario_gated_control.csv"
    model.write_submission(scenario_gated_output_path, scenario_gated_control_labels)
    for name, values in (
        ("baseline_scenario_conservative", scenario_conservative_labels),
        ("baseline_scenario_receiver_switch", scenario_receiver_switch_labels),
        ("baseline_scenario_strict_receiver", scenario_strict_receiver_labels),
        ("baseline_scenario_combined", scenario_combined_labels),
        ("baseline_scenario_bidirectional", scenario_bidirectional_labels),
        (
            "baseline_scenario_bidirectional_long_reception",
            scenario_bidirectional_long_reception_labels,
        ),
        (
            "baseline_scenario_bidirectional_receiver_agreement",
            scenario_bidirectional_receiver_agreement_labels,
        ),
        (
            "baseline_scenario_bidirectional_anchor_forecast",
            scenario_bidirectional_anchor_forecast_labels,
        ),
        ("baseline_scenario_bidirectional_d", scenario_bidirectional_d_labels),
        (
            "baseline_scenario_bidirectional_d_duration",
            scenario_bidirectional_d_duration_labels,
        ),
        ("baseline_scenario_bidirectional_ensemble", scenario_ensemble_labels),
        (
            "baseline_scenario_bidirectional_ensemble_restart_guard",
            scenario_ensemble_restart_guard_labels,
        ),
        (
            "baseline_scenario_bidirectional_ensemble_restart_control",
            scenario_ensemble_restart_control_labels,
        ),
        ("baseline_scenario_bidirectional_ensemble_duration", scenario_ensemble_duration_labels),
        (
            "baseline_scenario_bidirectional_long_ensemble_duration",
            scenario_long_ensemble_duration_labels,
        ),
        (
            "baseline_scenario_bidirectional_receiver_agreement_ensemble",
            scenario_receiver_agreement_ensemble_labels,
        ),
        (
            "baseline_scenario_bidirectional_interpolated_flight_ensemble",
            scenario_interpolated_flight_ensemble_labels,
        ),
        (
            "baseline_scenario_bidirectional_ensemble_anchor_protected",
            scenario_ensemble_anchor_protected_labels,
        ),
        (
            "baseline_scenario_bidirectional_ensemble_duration_short",
            scenario_ensemble_duration_short_labels,
        ),
        ("baseline_scenario_bidirectional_ensemble_duration_long", scenario_ensemble_duration_long_labels),
        ("baseline_scenario_activity_ensemble", scenario_activity_ensemble_labels),
        ("baseline_scenario_d_activity", scenario_d_activity_labels),
        ("baseline_scenario_d_speed2", scenario_d_speed2_labels),
        ("baseline_scenario_d_speed3", scenario_d_speed3_labels),
        ("baseline_scenario_d_speed3_long", scenario_d_speed3_long_labels),
        ("baseline_scenario_d_locked", scenario_d_locked_labels),
        ("baseline_scenario_d_locked_long", scenario_d_locked_long_labels),
        ("baseline_scenario_d_restart_boundary", scenario_d_restart_boundary_labels),
        ("baseline_scenario_d_restart_boundary_strict", scenario_d_restart_boundary_strict_labels),
    ):
        model.write_submission(ROOT / "evaluation" / "results" / f"{name}.csv", values)
    for name, values in (
        ("baseline_adaptive_transition_gate", adaptive_transition_gate_labels),
        ("baseline_adaptive_reachability_gate", adaptive_reachability_gate_labels),
        ("baseline_adaptive_dead_trim", adaptive_dead_trim_labels),
        ("baseline_adaptive_longer", adaptive_longer_labels),
        ("baseline_adaptive_100_context", adaptive_100_labels),
    ):
        model.write_submission(ROOT / "evaluation" / "results" / f"{name}.csv", values)

    feature_path = ROOT / "evaluation" / "reports" / "temporal_feature_sample.csv"
    anchor_audit_path = ROOT / "evaluation" / "reports" / "anchor_audit_public.csv"
    episode_path = ROOT / "evaluation" / "reports" / "scenario_episodes.csv"
    summary_path = ROOT / "evaluation" / "reports" / "scenario_summary.csv"
    sample_columns = [
        "frame", "closest_team", "closest_distance_m", "team_distance_margin_m",
        "scenario_context",
        "home_persistence", "away_persistence", "nearest_h_closing_rate_mps",
        "nearest_a_closing_rate_mps", "ball_speed_derived_mps", "candidate_pressure_m",
        "candidate_opponent_density_5m", "candidate_teammate_support_10m",
        "candidate_pass_options_15m", "candidate_isolated",
        "low_pressure_control", "pressurized_control", "pressure_anchor_team",
        "ball_acceleration_mps2", "ball_direction_change_rad", "ball_speed_drop_mps",
        "ball_speed_interpolated_mps", "ball_position_interpolated", "recent_flight_evidence",
        "ball_flight_candidate", "h_projected_path_distance_m", "a_projected_path_distance_m",
        "projected_receiver_team", "restart_start_candidate",
        "restart_end_candidate", "restart_ball_active", "home_future_anchor_persistence",
        "away_future_anchor_persistence",
        "opponent_closing_score", "arrival_switch_score", "projected_opponent_receiver",
        "pressure_switch_score", "switch_risk_score", "forecast_next_team",
        "h_fastest_arrival_s", "a_fastest_arrival_s", "arrival_time_margin_s",
        "likely_receiver_team", "long_ball_candidate",
        "anchor_team", "anchor_confidence_score",
        "anchor_source", "reception_anchor", "transition_anchor",
        "genuine_reception_anchor", "occlusion_anchor", "restart_confirmed_anchor",
    ]
    temporal[sample_columns].head(250).to_csv(feature_path, index=False)
    write_anchor_audit(temporal, anchor_audit_path)
    transition_path = ROOT / "evaluation" / "reports" / "transition_forecast_sample.csv"
    temporal.loc[temporal["switch_risk_score"] >= 0.50, [
        "frame", "closest_team", "switch_risk_score", "opponent_closing_score",
        "arrival_switch_score", "projected_opponent_receiver", "pressure_switch_score",
        "forecast_next_team",
    ]].head(500).to_csv(transition_path, index=False)
    episodes.to_csv(episode_path, index=False)
    episodes.groupby("scenario", dropna=False).agg(
        episodes=("episode_id", "count"),
        frames=("duration_frames", "sum"),
        mean_duration_s=("duration_s", "mean"),
    ).reset_index().to_csv(summary_path, index=False)
    temporal.groupby("scenario_context", dropna=False).agg(
        frames=("frame", "count"),
        graded_frames=("graded", "sum"),
    ).reset_index().to_csv(ROOT / "evaluation" / "reports" / "scenario_context_counts.csv", index=False)
    dead_audit = temporal.loc[temporal["smart_set_piece"] & temporal["graded"]].copy()
    dead_audit["speed_mps"] = pd.to_numeric(dead_audit["ball_speed_derived_mps"], errors="coerce")
    dead_audit["low_speed_1mps"] = dead_audit["speed_mps"].fillna(0.0) <= 1.0
    dead_audit["low_speed_2mps"] = dead_audit["speed_mps"].fillna(0.0) <= 2.0
    dead_audit["low_speed_3mps"] = dead_audit["speed_mps"].fillna(0.0) <= 3.0
    dead_audit["flight_blocked"] = dead_audit["ball_flight_candidate"].astype(bool)
    dead_audit["reception_blocked"] = dead_audit["genuine_reception_anchor"].astype(bool)
    dead_audit.groupby("restart_type", dropna=False).agg(
        smart_frames=("frame", "size"),
        low_speed_1mps=("low_speed_1mps", "sum"),
        low_speed_2mps=("low_speed_2mps", "sum"),
        low_speed_3mps=("low_speed_3mps", "sum"),
        flight_blocked=("flight_blocked", "sum"),
        reception_blocked=("reception_blocked", "sum"),
    ).reset_index().to_csv(ROOT / "evaluation" / "reports" / "dead_ball_event_audit.csv", index=False)
    print({**report, **bridge_report, **transition_report, **decoder_report})
    print("anchor frames:")
    print(temporal["anchor_source"].value_counts(dropna=False).to_string())
    print("restart types:")
    print(temporal.loc[temporal["restart_context"], "restart_type"].value_counts().to_string())
    print("scenario episodes:")
    print(episodes.groupby("scenario")["duration_frames"].agg(["count", "sum"]).to_string())
    print(pd.Series(labels).value_counts().sort_index())
    print(f"submission={output_path}")
    print(f"scenario_submission={scenario_output_path}")
    print(f"hmm_submission={hmm_output_path}")
    print(f"hsmm_submission={hsmm_output_path}")
    print(f"bayesian_hmm_submission={bayesian_hmm_output_path}")
    print(f"rule_constrained_viterbi_submission={constrained_viterbi_output_path}")
    print(f"selective_rule_viterbi_submission={selective_viterbi_output_path}")
    print(f"scenario_hsmm_submission={scenario_hsmm_output_path}")
    print(f"scenario_hsmm_strict_submission={scenario_hsmm_strict_output_path}")
    print(f"restart_hsmm_submission={restart_hsmm_output_path}")
    print(f"restart_hsmm_strict_submission={restart_hsmm_strict_output_path}")
    print(f"pressure_submission={pressure_output_path}")
    print(f"adaptive_submission={adaptive_output_path}")
    print(f"adaptive_no_dead_submission={adaptive_no_dead_output_path}")
    print(f"adaptive_long_submission={adaptive_long_output_path}")
    print(f"scenario_gated_control_submission={scenario_gated_output_path}")
    print("scenario_variant_submissions=4")
    print("additional_adaptive_submissions=5")
    print(f"feature_sample={feature_path}")
    print(f"anchor_audit={anchor_audit_path}")
    print(f"transition_forecast={transition_path}")
    print("forecast next-team counts:")
    print(temporal["forecast_next_team"].value_counts(dropna=False).to_string())
    print(f"episodes={episode_path}")
    print(f"scenario_summary={summary_path}")


if __name__ == "__main__":
    main()
