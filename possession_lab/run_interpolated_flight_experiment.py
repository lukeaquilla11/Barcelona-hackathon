"""Run the full observation-first possession experiment."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from possession_lab.baseline_pipeline import ClosestTeamRuleModel, graded_mask
from possession_lab.decoder_experiments import (
    _remove_short_dead_runs,
    classify_scenarios,
    dead_ball_boundary_evidence,
    low_motion_dead_ball_evidence,
    retain_no_switch_intervals,
    retain_weak_contact_arrival_handoffs,
    restart_provisional_labels,
    scenario_gated_labels,
    bridge_same_team_missing_ball_trajectory,
    scenario_confidence_scores,
    add_observation_probability_features,
    add_haiqal_transition_evidence,
    add_transition_channel_profile,
    add_transition_hypothesis_features,
    bootstrap_transition_stability,
    transition_channel_stability,
    duration_aware_scenario_decode,
    add_scenario_episode_features,
    build_transition_hypothesis_ledger,
    bounded_bidirectional_scenario_resolver,
    apply_switch_confirmation,
    apply_switch_confirmation_hysteresis,
    resolve_cross_team_bidirectional_switch,
    resolve_reception_switch_inside_interval,
    reception_transition_alignment_audit,
    resolve_aligned_reception_switch,
    apply_stable_reception_episode_switch,
    restart_control_boundary_labels,
    restart_phase_interval_resolver,
    restart_phase_boundary_interval_resolver,
    hierarchical_restart_fill_labels,
)
from possession_lab.scenario_decoder import decode_scenario_labels
from possession_lab.temporal_rules import (
    add_anchor_features,
    add_reachability_features,
    add_special_anchor_features,
    add_boundary_anchor_features,
    add_temporal_features,
    add_transition_forecast_features,
    add_switch_confirmation_features,
    add_reception_outcome_features,
    add_reception_geometry_features,
    add_reception_signal_features,
    add_first_time_pass_features,
    add_reception_episode_features,
    load_rules,
)
from starter.load_smart_tagging import load as load_smart_tagging
from possession_lab.calculations.features import build_engineered_features
from possession_lab.sequence_models import TransitionLedgerViterbi
from possession_lab.haiqal_baseline import add_evidence_features as add_haiqal_evidence_features


ROOT = Path(__file__).resolve().parents[1]


def write_public_transition_error_audit(
    ledger: pd.DataFrame,
    *,
    ground_truth_path: Path,
    output_path: Path,
    summary_path: Path,
) -> None:
    """Compare fixed transition hypotheses with public labels for evaluation only.

    This report is intentionally downstream of feature construction and label
    generation.  Public labels are not passed back into any detector, decoder,
    or confidence score; they only identify which interval hypotheses were
    near a labelled H/A change.
    """
    if ledger.empty or not ground_truth_path.exists():
        return
    ground_truth = pd.read_csv(ground_truth_path)
    gt_by_frame = pd.Series(
        ground_truth["label"].to_numpy(object),
        index=ground_truth["frame"].astype(int),
    )
    switches: list[int] = []
    previous = None
    for frame, label in gt_by_frame.items():
        label = str(label)
        if label in {"H", "A"} and previous in {"H", "A"} and label != previous:
            switches.append(int(frame))
        previous = label if label in {"H", "A"} else None

    rows: list[dict[str, object]] = []
    for row in ledger.to_dict("records"):
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        interval_frames = range(start, end + 1)
        interval_labels = gt_by_frame.reindex(interval_frames).dropna()
        labelled = interval_labels[interval_labels.isin(["H", "A", "D"])]
        nearest_distance = None
        nearest_switch = None
        if switches:
            nearest_switch = min(
                switches,
                key=lambda frame: 0 if start <= frame <= end else min(abs(frame - start), abs(frame - end)),
            )
            nearest_distance = 0 if start <= nearest_switch <= end else min(
                abs(nearest_switch - start), abs(nearest_switch - end)
            )
        post_team = gt_by_frame.get(nearest_switch) if nearest_switch is not None else None
        candidate_team = row.get("candidate_team")
        rows.append(
            {
                **row,
                "public_labelled_frames": int(len(labelled)),
                "public_gt_mode": str(labelled.mode().iloc[0]) if not labelled.empty else None,
                "public_gt_has_switch_inside": bool(any(start <= frame <= end for frame in switches)),
                "public_nearest_switch_frame": nearest_switch,
                "public_nearest_switch_distance_frames": nearest_distance,
                "public_switch_within_25_frames": bool(nearest_distance is not None and nearest_distance <= 25),
                "public_switch_team": post_team,
                "public_candidate_matches_switch_team": bool(
                    candidate_team in {"H", "A"}
                    and post_team in {"H", "A"}
                    and candidate_team == post_team
                    and nearest_distance is not None
                    and nearest_distance <= 25
                ),
            }
        )

    audit = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)

    # The public file grades only selected windows.  Keep the all-interval
    # audit for traceability, but do not let ungraded match time dominate the
    # aggregate error summary.
    summary_source = audit.loc[audit["public_labelled_frames"] > 0].copy()
    if summary_source.empty:
        summary_source = audit.copy()
    summary_source["near_switch"] = summary_source["public_switch_within_25_frames"]
    summary_source["candidate_match"] = summary_source["public_candidate_matches_switch_team"]
    summary = (
        summary_source.groupby(["state", "missing_ball_cause"], dropna=False)
        .agg(
            intervals=("interval_id", "size"),
            near_public_switch=("near_switch", "sum"),
            candidate_team_matches=("candidate_match", "sum"),
            median_switch_distance_frames=("public_nearest_switch_distance_frames", "median"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--george-ballfill",
        type=Path,
        default=None,
        help="optional George fused ball CSV; attaches source-aware observations before features/scenarios",
    )
    parser.add_argument(
        "--george-allow-model-filled",
        action="store_true",
        help="also accept George's carrier-HMM 'filled' source (diagnostic only)",
    )
    parser.add_argument(
        "--george-evidence-only",
        action="store_true",
        help="keep tactical coordinates unchanged; use George only as missing-ball/scenario evidence",
    )
    parser.add_argument(
        "--george-minimum-confidence",
        type=float,
        default=None,
        help=(
            "optional George confidence gate for coordinate replacement; "
            "does not suppress George auxiliary evidence"
        ),
    )
    args = parser.parse_args()
    rules = load_rules()
    model = ClosestTeamRuleModel()
    temporal = build_engineered_features(
        model=model,
        rules=rules,
        george_ballfill_path=args.george_ballfill,
        george_allow_model_filled=args.george_allow_model_filled,
        george_evidence_only=args.george_evidence_only,
        george_minimum_observation_confidence=args.george_minimum_confidence,
    )
    if args.george_ballfill is not None:
        print(f"George ball observations: {args.george_ballfill}")
    # Optional ablation: add Haiqal's raw distance/speed/dead-ball clues as
    # observations while keeping our scenario, anchor, interval, and decoder
    # layers in control. The independent Haiqal labels are never reused.
    haiqal_temporal = add_haiqal_evidence_features(temporal)
    # Clean combination experiment: retain Haiqal's raw observations, then
    # rebuild only our transition trust/interval layers. His standalone HMM
    # labels are deliberately not imported into this path.
    haiqal_transition_temporal = add_haiqal_transition_evidence(
        haiqal_temporal,
        maximum_contribution=0.04,
    )
    haiqal_transition_temporal = add_transition_hypothesis_features(
        haiqal_transition_temporal
    )

    interpolated = temporal.copy()
    interpolated["recent_flight_evidence"] = temporal["recent_flight_evidence_interpolated"]
    interpolated["ball_flight_candidate"] = temporal["ball_flight_candidate_interpolated"]
    labels = scenario_gated_labels(
        interpolated,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        receiver_horizon_frames=75,
        maximum_internal_spike=75,
    )
    scenario_labels, _ = decode_scenario_labels(temporal, rules)
    dead_support = (
        temporal["graded"].to_numpy(bool)
        & temporal["smart_set_piece"].to_numpy(bool)
        & (scenario_labels == "D")
        & temporal["scenario_context"].isin(["DEAD", "UNKNOWN", "OCCLUDED"]).to_numpy(bool)
    )
    labels[dead_support] = "D"
    labels = _remove_short_dead_runs(labels, minimum_run=25)
    output = ROOT / "evaluation" / "results" / "baseline_scenario_bidirectional_interpolated_flight_ensemble.csv"
    model.write_submission(output, labels)
    print(output)

    # Candidate: use only agreeing same-team anchors to fill a physically
    # coherent missing-ball bridge. Keep separate until public evaluation shows
    # that it improves both frame and spell quality.
    trajectory_bridge_labels = bridge_same_team_missing_ball_trajectory(
        temporal,
        labels,
        maximum_gap_seconds=float(rules["timeline"]["long_ball_max_seconds"]),
        require_coherent_trajectory=True,
        require_receiver_run=True,
    )
    trajectory_bridge_output = ROOT / "evaluation" / "results" / "baseline_same_team_missing_ball_trajectory.csv"
    model.write_submission(trajectory_bridge_output, trajectory_bridge_labels)
    print(trajectory_bridge_output)

    # Diagnostic Kalman variant: require the same conservative bridge to have
    # a bounded two-sided ball-state estimate. The estimate supplies geometric
    # support only; agreeing team anchors and receiver evidence remain required.
    kalman_bridge_labels = bridge_same_team_missing_ball_trajectory(
        temporal,
        labels,
        maximum_gap_seconds=float(rules["timeline"]["long_ball_max_seconds"]),
        require_coherent_trajectory=True,
        require_receiver_run=True,
        require_kalman_path=True,
    )
    kalman_bridge_output = ROOT / "evaluation" / "results" / "baseline_same_team_kalman_missing_ball_trajectory.csv"
    model.write_submission(kalman_bridge_output, kalman_bridge_labels)
    print(kalman_bridge_output)

    coordinate_bridge_labels = bridge_same_team_missing_ball_trajectory(
        temporal,
        labels,
        maximum_gap_seconds=float(rules["timeline"]["long_ball_max_seconds"]),
        require_coherent_trajectory=True,
        require_receiver_run=True,
        require_player_path=True,
    )
    coordinate_bridge_output = ROOT / "evaluation" / "results" / "baseline_same_team_coordinate_path_bridge.csv"
    model.write_submission(coordinate_bridge_output, coordinate_bridge_labels)
    print(coordinate_bridge_output)

    # Diagnostic candidate: only bridge a missing-ball gap when the same
    # temporary player path remains speed-compatible with the ball. A sharp
    # change is deliberately excluded from this candidate because it is more
    # consistent with a pass, deflection, or tackle than dribble continuity.
    dribble_bridge_labels = bridge_same_team_missing_ball_trajectory(
        temporal,
        labels,
        maximum_gap_seconds=float(rules["timeline"]["long_ball_max_seconds"]),
        require_coherent_trajectory=True,
        require_receiver_run=True,
        require_player_path=True,
        require_same_player_dribble=True,
    )
    dribble_bridge_output = ROOT / "evaluation" / "results" / "baseline_same_player_dribble_gap_bridge.csv"
    model.write_submission(dribble_bridge_output, dribble_bridge_labels)
    print(dribble_bridge_output)

    restart_labels = restart_provisional_labels(
        temporal,
        labels,
        minimum_active_run=int(rules["restart"]["minimum_active_run_frames"]),
        minimum_dead_run=int(rules["restart"]["minimum_dead_run_frames"]),
        control_run=8,
        post_restart_hold_frames=int(rules["restart"]["post_restart_hold_frames"]),
        post_restart_hold_frames_by_type={
            str(key): int(value)
            for key, value in rules["restart"].get(
                "post_restart_hold_frames_by_type", {}
            ).items()
        },
    )
    restart_output = (
        ROOT
        / "evaluation"
        / "results"
        / "baseline_restart_type_specific_interpolated_flight.csv"
    )
    model.write_submission(restart_output, restart_labels)
    print(restart_output)

    dead_ball_labels = dead_ball_boundary_evidence(
        temporal,
        labels,
        low_activity_median_mps=float(rules["dead_ball_evidence"]["low_activity_median_mps"]),
        low_activity_p90_mps=float(rules["dead_ball_evidence"]["low_activity_p90_mps"]),
        minimum_low_motion_frames=int(rules["dead_ball_evidence"]["minimum_low_motion_frames"]),
        maximum_out_of_bounds_bridge_frames=int(
            rules["dead_ball_evidence"]["maximum_out_of_bounds_bridge_frames"]
        ),
    )
    dead_ball_output = (
        ROOT
        / "evaluation"
        / "results"
        / "baseline_dead_ball_boundary_low_motion.csv"
    )
    model.write_submission(dead_ball_output, dead_ball_labels)
    print(dead_ball_output)

    # Diagnostic only: visible-ball whole-pitch settling can suggest an
    # untagged stoppage, but it must be audited against possession spells
    # before it is allowed into the canonical path.
    visible_rules = rules["dead_ball_evidence"]
    visible_settle_labels = low_motion_dead_ball_evidence(
        temporal,
        dead_ball_labels,
        activity_window_frames=int(visible_rules["visible_ball_activity_window_frames"]),
        minimum_support_fraction=float(visible_rules["visible_ball_minimum_support_fraction"]),
        minimum_candidate_run_frames=int(visible_rules["visible_ball_minimum_candidate_run_frames"]),
        maximum_ball_speed_mps=float(visible_rules["visible_ball_maximum_speed_mps"]),
        maximum_settling_speed_mps=float(visible_rules["visible_ball_maximum_settling_speed_mps"]),
        settling_acceleration_mps2=float(visible_rules["visible_ball_settling_acceleration_mps2"]),
        maximum_restart_bridge_frames=int(visible_rules["visible_ball_maximum_restart_bridge_frames"]),
        maximum_candidate_bridge_frames=int(visible_rules["visible_ball_maximum_candidate_bridge_frames"]),
        minimum_bridge_low_activity_fraction=float(visible_rules["visible_ball_minimum_bridge_low_activity_fraction"]),
    )
    visible_settle_output = (
        ROOT / "evaluation" / "results" / "diagnostic_visible_ball_low_motion_dead_ball.csv"
    )
    model.write_submission(visible_settle_output, visible_settle_labels)
    print(visible_settle_output)

    # Nonparametric transition-stability diagnostic. It resamples frames
    # within each inferred interval and estimates evidence aggregation
    # stability, not the probability that a hidden label is true.
    bootstrap_report = bootstrap_transition_stability(
        temporal,
        n_bootstrap=256,
        random_state=0,
    )
    bootstrap_report_output = (
        ROOT / "evaluation" / "reports" / "transition_bootstrap_stability.csv"
    )
    bootstrap_report_output.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_report.to_csv(bootstrap_report_output, index=False)
    if not bootstrap_report.empty:
        interval_lookup = bootstrap_report.set_index("interval_id")
        interval_ids = pd.to_numeric(
            temporal["transition_hypothesis_interval_id"], errors="coerce"
        ).fillna(-1).astype(int)
        temporal["transition_bootstrap_stable_state"] = interval_ids.map(
            interval_lookup["stable_state"]
        ).fillna("NO_INTERVAL")
        temporal["transition_bootstrap_stability"] = interval_ids.map(
            interval_lookup["stable_state_fraction"]
        ).fillna(0.0)
        temporal["transition_bootstrap_no_switch_lower"] = interval_ids.map(
            interval_lookup["no_switch_evidence_lower"]
        ).fillna(0.0)
    else:
        temporal["transition_bootstrap_stable_state"] = "NO_INTERVAL"
        temporal["transition_bootstrap_stability"] = 0.0
        temporal["transition_bootstrap_no_switch_lower"] = 0.0
    print(bootstrap_report_output)

    channel_report = transition_channel_stability(temporal)
    channel_report_output = (
        ROOT / "evaluation" / "reports" / "transition_channel_stability.csv"
    )
    channel_report.to_csv(channel_report_output, index=False)
    print(channel_report_output)

    # Structural transition ablation: the visible-ball candidate has strong
    # dead-ball performance, but its remaining errors are mostly H/A flips
    # inside flight, contested, reception, and unknown intervals. Retain the
    # prior team when the interval ledger says NO_SWITCH and no new control
    # anchor confirms the proposed team. This is an observation-only resolver,
    # not a public-label correction.
    no_switch_output = (
        ROOT / "evaluation" / "results" / "diagnostic_visible_ball_no_switch_interval.csv"
    )
    no_switch_labels = retain_no_switch_intervals(
        temporal,
        visible_settle_labels,
        minimum_no_switch_probability=0.65,
        minimum_interval_support=0.60,
        maximum_interval_frames=300,
        minimum_prior_team_run_frames=1,
    )
    model.write_submission(no_switch_output, no_switch_labels)
    print(no_switch_output)

    # Predeclared resolver-profile sweep.  These profiles vary only the
    # interval decision policy; they do not read public labels or fit to the
    # five-minute evaluation slice.  A lower no-switch gate and longer
    # interval are more conservative about accepting a handoff, while a
    # higher gate and shorter interval are more willing to release the prior
    # team.  Keep every profile as a separate artifact so 09 can rank both
    # frame and spell behaviour instead of silently selecting a threshold.
    resolver_profiles = {
        "conservative_long": {
            "minimum_no_switch_probability": 0.55,
            "minimum_interval_support": 0.50,
            "maximum_interval_frames": 600,
            "minimum_prior_team_run_frames": 1,
        },
        "conservative": {
            "minimum_no_switch_probability": 0.60,
            "minimum_interval_support": 0.55,
            "maximum_interval_frames": 450,
            "minimum_prior_team_run_frames": 1,
        },
        "balanced_current": {
            "minimum_no_switch_probability": 0.65,
            "minimum_interval_support": 0.60,
            "maximum_interval_frames": 300,
            "minimum_prior_team_run_frames": 1,
        },
        "aggressive": {
            "minimum_no_switch_probability": 0.75,
            "minimum_interval_support": 0.70,
            "maximum_interval_frames": 200,
            "minimum_prior_team_run_frames": 2,
        },
        "aggressive_short": {
            "minimum_no_switch_probability": 0.80,
            "minimum_interval_support": 0.80,
            "maximum_interval_frames": 150,
            "minimum_prior_team_run_frames": 3,
        },
    }
    for profile_name, profile_kwargs in resolver_profiles.items():
        profile_labels = retain_no_switch_intervals(
            temporal,
            visible_settle_labels,
            **profile_kwargs,
        )
        profile_output = (
            ROOT / "evaluation" / "results"
            / f"diagnostic_resolver_profile_{profile_name}.csv"
        )
        model.write_submission(profile_output, profile_labels)
        print(profile_output)

    # Local contact/arrival handoff diagnostic. It tests the measured failure
    # mode without suppressing an entire restart or possession interval: only
    # the transition boundary is guarded, and only when the new team lacks
    # combined contact + team-arrival support.
    contact_arrival_labels = retain_weak_contact_arrival_handoffs(
        temporal,
        no_switch_labels,
        boundary_frames=25,
        minimum_contact_score=0.45,
        minimum_consensus_score=0.42,
        minimum_handoff_score=0.58,
        minimum_no_switch_probability=0.70,
    )
    contact_arrival_output = (
        ROOT / "evaluation" / "results"
        / "diagnostic_contact_arrival_local_boundary.csv"
    )
    model.write_submission(contact_arrival_output, contact_arrival_labels)
    print(contact_arrival_output)

    # Conservative bootstrap guard: retain a no-switch interval only when
    # its empirical lower bound remains at least as strong as the existing
    # no-switch evidence gate. This is a diagnostic ablation, not a new truth
    # source or a public-label correction.
    bootstrap_no_switch_labels = retain_no_switch_intervals(
        temporal,
        visible_settle_labels,
        minimum_no_switch_probability=0.65,
        minimum_interval_support=0.60,
        maximum_interval_frames=300,
        minimum_prior_team_run_frames=1,
        minimum_bootstrap_no_switch_lower=0.65,
    )
    bootstrap_no_switch_output = (
        ROOT / "evaluation" / "results"
        / "diagnostic_visible_ball_no_switch_bootstrap_guard.csv"
    )
    model.write_submission(bootstrap_no_switch_output, bootstrap_no_switch_labels)
    print(bootstrap_no_switch_output)

    # Combined best-model ablation: Haiqal raw observations are allowed to
    # support a transition hypothesis, but the same visible-ball dead-ball
    # anchors and NO_SWITCH interval resolver still make the final decision.
    haiqal_no_switch_labels = retain_no_switch_intervals(
        haiqal_transition_temporal,
        visible_settle_labels,
        minimum_no_switch_probability=0.65,
        minimum_interval_support=0.60,
        maximum_interval_frames=300,
        minimum_prior_team_run_frames=1,
    )
    haiqal_no_switch_output = (
        ROOT / "evaluation" / "results"
        / "diagnostic_haiqal_visible_ball_no_switch_interval.csv"
    )
    model.write_submission(haiqal_no_switch_output, haiqal_no_switch_labels)
    print(haiqal_no_switch_output)

    # Channel-composition experiment: use George upstream and Haiqal only as
    # bounded observations, then rebuild the H/A/NO_SWITCH/CONTESTED interval
    # hypotheses from an explicit evidence vector.  These are predeclared
    # sensitivity profiles, not weights fitted to the public labels.  Keeping
    # them as separate submissions lets 09 rank them without silently changing
    # the promoted model.
    for channel_profile in ("anchor_dominant", "balanced", "independent_support"):
        profile_temporal = add_transition_channel_profile(
            haiqal_temporal,
            profile=channel_profile,
        )
        profile_temporal = add_transition_hypothesis_features(profile_temporal)
        profile_labels = retain_no_switch_intervals(
            profile_temporal,
            visible_settle_labels,
            minimum_no_switch_probability=0.65,
            minimum_interval_support=0.60,
            maximum_interval_frames=300,
            minimum_prior_team_run_frames=1,
        )
        profile_output = (
            ROOT / "evaluation" / "results"
            / f"diagnostic_channel_profile_{channel_profile}.csv"
        )
        model.write_submission(profile_output, profile_labels)
        print(profile_output)

    # Shared restart-phase constraint: the restart detector contributes phase
    # and switch permission to the interval resolver instead of writing a
    # provisional possession label first.  This full-length candidate is the
    # structural test of restart observations -> constraints -> resolver.
    restart_phase_labels = restart_phase_interval_resolver(
        temporal,
        no_switch_labels,
        minimum_control_run=8,
        maximum_live_search_frames=75,
        use_tracking_fallback=False,
        minimum_anchor_confidence=0.85,
    )
    restart_phase_output = (
        ROOT / "evaluation" / "results"
        / "diagnostic_no_switch_restart_phase_interval.csv"
    )
    model.write_submission(restart_phase_output, restart_phase_labels)
    print(restart_phase_output)

    # Local-boundary structural diagnostic: constrain only the first proposed
    # handoff inside a restart phase, rather than suppressing the full H/A run.
    # This is the candidate for testing whether the prior restart resolver was
    # too broad on full-match frames.
    for guard_frames in (5, 12, 25):
        restart_phase_boundary_labels = restart_phase_boundary_interval_resolver(
            temporal,
            no_switch_labels,
            minimum_control_run=8,
            maximum_live_search_frames=75,
            maximum_boundary_guard_frames=guard_frames,
            use_tracking_fallback=False,
            minimum_anchor_confidence=0.85,
        )
        restart_phase_boundary_output = (
            ROOT / "evaluation" / "results"
            / f"diagnostic_no_switch_restart_phase_boundary_guard{guard_frames:02d}.csv"
        )
        model.write_submission(restart_phase_boundary_output, restart_phase_boundary_labels)
        print(restart_phase_boundary_output)

    # Next structural ablations: test the two evidence signals identified by
    # the disagreement audit on top of the current best interval candidate.
    # These functions consume only engineered observations; public labels are
    # joined later by the evaluator and never enter either resolver.
    true_switch_labels = apply_switch_confirmation(
        temporal,
        no_switch_labels,
        minimum_score=0.65,
        minimum_run_frames=5,
    )
    true_switch_output = (
        ROOT / "evaluation" / "results"
        / "diagnostic_no_switch_true_switch_confirmation.csv"
    )
    model.write_submission(true_switch_output, true_switch_labels)
    print(true_switch_output)

    restart_activation_labels = restart_control_boundary_labels(
        temporal,
        no_switch_labels,
        minimum_control_run=8,
        maximum_live_search_frames=75,
        use_tracking_fallback=False,
        minimum_anchor_confidence=0.85,
    )
    restart_activation_output = (
        ROOT / "evaluation" / "results"
        / "diagnostic_no_switch_restart_activation.csv"
    )
    model.write_submission(restart_activation_output, restart_activation_labels)
    print(restart_activation_output)

    combined_transition_labels = apply_switch_confirmation(
        temporal,
        restart_activation_labels,
        minimum_score=0.65,
        minimum_run_frames=5,
    )
    combined_transition_output = (
        ROOT / "evaluation" / "results"
        / "diagnostic_no_switch_true_switch_restart_activation.csv"
    )
    model.write_submission(combined_transition_output, combined_transition_labels)
    print(combined_transition_output)

    # Guarded handoff diagnostics: unlike the direct confirmation ablation
    # above, these require a later sustained anchor before accepting the
    # candidate team.  The restart state-machine variant keeps D until a
    # bounded active/control episode is available.  These remain diagnostics
    # because they can abstain or delay a real switch on a short benchmark.
    hysteretic_handoff_labels = apply_switch_confirmation_hysteresis(
        temporal,
        no_switch_labels,
        minimum_score=0.65,
        candidate_run_frames=5,
        minimum_anchor_run_frames=8,
        anchor_lookahead_frames=75,
        merge_spikes=False,
    )
    hysteretic_handoff_output = (
        ROOT / "evaluation" / "results"
        / "diagnostic_no_switch_hysteretic_handoff.csv"
    )
    model.write_submission(hysteretic_handoff_output, hysteretic_handoff_labels)
    print(hysteretic_handoff_output)

    restart_state_labels = restart_provisional_labels(
        temporal,
        no_switch_labels,
        minimum_active_run=3,
        minimum_dead_run=5,
        control_run=8,
        post_restart_hold_frames=25,
    )
    restart_state_output = (
        ROOT / "evaluation" / "results"
        / "diagnostic_no_switch_restart_state_machine.csv"
    )
    model.write_submission(restart_state_output, restart_state_labels)
    print(restart_state_output)

    restart_handoff_labels = apply_switch_confirmation_hysteresis(
        temporal,
        restart_state_labels,
        minimum_score=0.65,
        candidate_run_frames=5,
        minimum_anchor_run_frames=8,
        anchor_lookahead_frames=75,
        merge_spikes=False,
    )
    restart_handoff_output = (
        ROOT / "evaluation" / "results"
        / "diagnostic_no_switch_restart_hysteretic_handoff.csv"
    )
    model.write_submission(restart_handoff_output, restart_handoff_labels)
    print(restart_handoff_output)

    switch_variants = {
        "baseline_switch_confirmation": {"minimum_score": 0.65, "minimum_run_frames": 5},
        "baseline_switch_confirmation_strict": {"minimum_score": 0.75, "minimum_run_frames": 5},
    }
    for variant_name, variant_kwargs in switch_variants.items():
        switch_labels = apply_switch_confirmation(
            temporal,
            dead_ball_labels,
            **variant_kwargs,
        )
        switch_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(switch_output, switch_labels)
        print(switch_output)

    hysteresis_variants = {
        "baseline_switch_confirmation_anchor_guard": {"merge_spikes": False},
        "baseline_switch_confirmation_anchor_guard_merge": {"merge_spikes": True},
    }
    for variant_name, variant_kwargs in hysteresis_variants.items():
        guarded_labels = apply_switch_confirmation_hysteresis(
            temporal,
            dead_ball_labels,
            minimum_score=0.65,
            candidate_run_frames=5,
            minimum_anchor_run_frames=8,
            anchor_lookahead_frames=75,
            **variant_kwargs,
        )
        guarded_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(guarded_output, guarded_labels)
        print(guarded_output)

    cross_team_labels = resolve_cross_team_bidirectional_switch(
        temporal,
        dead_ball_labels,
        max_anchor_gap_frames=75,
        minimum_anchor_confidence=0.85,
        minimum_switch_score=0.65,
        minimum_switch_run_frames=5,
    )
    cross_team_output = (
        ROOT / "evaluation" / "results" / "baseline_cross_team_bidirectional_switch.csv"
    )
    model.write_submission(cross_team_output, cross_team_labels)
    print(cross_team_output)

    reception_switch_variants = {
        "baseline_reception_switch_evidence": {
            "minimum_reception_score": 0.55,
            "minimum_post_control_frames": 3,
            "require_strict_reception": False,
        },
        "baseline_reception_switch_persistent": {
            "minimum_reception_score": 0.65,
            "minimum_post_control_frames": 8,
            "require_strict_reception": False,
        },
        "baseline_reception_switch_strict": {
            "minimum_reception_score": 0.65,
            "minimum_post_control_frames": 8,
            "require_strict_reception": True,
        },
    }
    for variant_name, variant_kwargs in reception_switch_variants.items():
        reception_switch_labels = resolve_reception_switch_inside_interval(
            temporal,
            dead_ball_labels,
            max_anchor_gap_frames=75,
            minimum_anchor_confidence=0.85,
            **variant_kwargs,
        )
        reception_switch_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(reception_switch_output, reception_switch_labels)
        print(reception_switch_output)

    enhanced_reception_switch_variants = {
        "baseline_enhanced_reception_switch": {
            "minimum_reception_score": 0.55,
            "minimum_post_control_frames": 3,
            "use_enhanced_reception_signal": True,
        },
        "baseline_enhanced_reception_switch_persistent": {
            "minimum_reception_score": 0.65,
            "minimum_post_control_frames": 8,
            "use_enhanced_reception_signal": True,
        },
        "baseline_enhanced_reception_switch_sustained": {
            "minimum_reception_score": 0.65,
            "minimum_post_control_frames": 8,
            "use_enhanced_reception_signal": True,
            "require_sustained_reception_signal": True,
        },
    }
    for variant_name, variant_kwargs in enhanced_reception_switch_variants.items():
        enhanced_labels = resolve_reception_switch_inside_interval(
            temporal,
            dead_ball_labels,
            max_anchor_gap_frames=75,
            minimum_anchor_confidence=0.85,
            **variant_kwargs,
        )
        enhanced_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(enhanced_output, enhanced_labels)
        print(enhanced_output)

    episode_variants = {
        "baseline_stable_reception_episode_switch_comfort": (3, 0.90, 0.0),
        "baseline_stable_reception_episode_switch_comfort_strict": (8, 0.95, 0.0),
        "baseline_stable_reception_episode_switch_post_control": (8, 0.95, 0.5),
    }
    for variant_name, (minimum_episode_frames, minimum_receiver_confidence, minimum_receiver_near_ball_fraction) in episode_variants.items():
        episode_labels = apply_stable_reception_episode_switch(
            temporal,
            dead_ball_labels,
            minimum_episode_frames=minimum_episode_frames,
            minimum_stability=0.75,
            minimum_receiver_confidence=minimum_receiver_confidence,
            minimum_receiver_near_ball_fraction=minimum_receiver_near_ball_fraction,
        )
        episode_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(episode_output, episode_labels)
        print(episode_output)

    restart_boundary_variants = {
        "baseline_restart_hinted_control_boundary": {
            "use_tracking_fallback": False,
            "minimum_control_run": 8,
        },
        "baseline_restart_tracking_control_boundary": {
            "use_tracking_fallback": True,
            "minimum_control_run": 8,
        },
        "baseline_restart_tracking_control_boundary_strict": {
            "use_tracking_fallback": True,
            "minimum_control_run": 12,
        },
    }
    for variant_name, variant_kwargs in restart_boundary_variants.items():
        boundary_labels = restart_control_boundary_labels(
            temporal,
            dead_ball_labels,
            maximum_live_search_frames=75,
            minimum_anchor_confidence=0.85,
            **variant_kwargs,
        )
        boundary_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(boundary_output, boundary_labels)
        print(boundary_output)

    hierarchy_variants = {
        "baseline_hierarchical_restart_fill_hinted": {"use_tracking_fallback": False},
        "baseline_hierarchical_restart_fill_tracking": {"use_tracking_fallback": True},
    }
    for variant_name, variant_kwargs in hierarchy_variants.items():
        hierarchy_labels = hierarchical_restart_fill_labels(
            temporal,
            dead_ball_labels,
            minimum_control_run=8,
            maximum_live_search_frames=75,
            minimum_anchor_confidence=0.85,
            **variant_kwargs,
        )
        hierarchy_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(hierarchy_output, hierarchy_labels)
        print(hierarchy_output)

    reception_variants = {
        "baseline_reception_path_agreement": {
            "require_receiver_path_agreement": True,
        },
        "baseline_reception_receiver_persistence": {
            "minimum_receiver_persistence_frames": 5,
        },
        "baseline_reception_path_persistence": {
            "require_receiver_path_agreement": True,
            "minimum_receiver_persistence_frames": 5,
        },
    }
    for variant_name, variant_kwargs in reception_variants.items():
        raw_variant = scenario_gated_labels(
            interpolated,
            allow_receiver_switch=True,
            allow_bidirectional_bridge=True,
            receiver_horizon_frames=75,
            maximum_internal_spike=75,
            **variant_kwargs,
        )
        raw_variant = _remove_short_dead_runs(raw_variant, minimum_run=25)
        variant_labels = dead_ball_boundary_evidence(
            temporal,
            raw_variant,
            low_activity_median_mps=float(rules["dead_ball_evidence"]["low_activity_median_mps"]),
            low_activity_p90_mps=float(rules["dead_ball_evidence"]["low_activity_p90_mps"]),
            minimum_low_motion_frames=int(rules["dead_ball_evidence"]["minimum_low_motion_frames"]),
            maximum_out_of_bounds_bridge_frames=int(
                rules["dead_ball_evidence"]["maximum_out_of_bounds_bridge_frames"]
            ),
        )
        variant_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(variant_output, variant_labels)
        print(variant_output)

    strict_reception_labels = scenario_gated_labels(
        interpolated,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        receiver_horizon_frames=75,
        maximum_internal_spike=75,
        require_strict_reception=True,
    )
    strict_reception_output = (
        ROOT / "evaluation" / "results" / "baseline_strict_reception_guard.csv"
    )
    model.write_submission(strict_reception_output, strict_reception_labels)
    print(strict_reception_output)

    # Diagnostic candidate: use the anchor-only future-touch forecast as an
    # additional veto/confirmation for a receiver switch. The forecast never
    # labels a frame by itself; it only authorises a switch already supported
    # by the bounded reception logic.
    forecast_confirmed_labels = scenario_gated_labels(
        interpolated,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        receiver_horizon_frames=75,
        require_anchor_forecast_confirmation=True,
        maximum_internal_spike=75,
    )
    forecast_confirmed_labels = _remove_short_dead_runs(
        forecast_confirmed_labels, minimum_run=25
    )
    forecast_confirmed_output = (
        ROOT / "evaluation" / "results" / "baseline_anchor_forecast_confirmation.csv"
    )
    model.write_submission(forecast_confirmed_output, forecast_confirmed_labels)
    print(forecast_confirmed_output)

    # Candidate: repair only suspicious scenario intervals when two bounded,
    # independent stable-control anchors agree on the same team.
    resolver_variants = {
        "baseline_bounded_bidirectional_short": {
            "max_anchor_gap_frames": 25,
            "minimum_interval_frames": 1,
        },
        "baseline_bounded_bidirectional_long_ball": {
            "max_anchor_gap_frames": 75,
            "minimum_interval_frames": 26,
            "long_ball_only": True,
        },
    }
    for variant_name, variant_kwargs in resolver_variants.items():
        candidate = bounded_bidirectional_scenario_resolver(
            temporal,
            dead_ball_labels,
            minimum_anchor_confidence=0.85,
            **variant_kwargs,
        )
        candidate_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(candidate_output, candidate)
        print(candidate_output)

    # Stage 1: expose scenario confidence without changing the official labels.
    scored_features = add_observation_probability_features(
        scenario_confidence_scores(interpolated)
    )
    confidence_columns = [
        "frame", "scenario_context", "scenario_context_causal", "scenario_episode_id",
        "scenario_episode_length_frames", "scenario_episode_position_frames",
        "scenario_episode_fraction", "scenario_episode_start", "scenario_episode_end",
        "scenario_state", "scenario_confidence", "scenario_margin",
        "state_score_H_CONTROL", "state_score_A_CONTROL", "state_score_DEAD_BALL",
        "state_score_BALL_IN_FLIGHT", "state_score_CONTESTED", "state_score_OCCLUDED",
        "state_score_RECEPTION", "reception_score_H", "reception_score_A",
        "post_control_score_H", "post_control_score_A",
        "h_contact_arrival_coherence", "a_contact_arrival_coherence",
        "h_contact_past_support", "a_contact_past_support",
        "h_arrival_future_support", "a_arrival_future_support",
        "contact_arrival_coherence_score", "contact_arrival_coherence_margin",
        "contact_arrival_coherence_team", "contact_arrival_coherence_available",
        "directional_reception_score_H", "directional_reception_score_A",
        "contest_score", "decoder_eligible",
        "transition_before_anchor_team", "transition_after_anchor_team",
        "transition_opposite_anchor_pair", "transition_candidate_team",
        "transition_trust_H", "transition_trust_A", "transition_trust_score",
        "transition_trust_margin", "transition_trust_p_H", "transition_trust_p_A",
        "transition_trust_p_no_switch", "transition_trust_calibrated",
        "transition_trust_candidate", "transition_trust_reason",
        "anchor_evidence_distance_score", "anchor_evidence_margin_score",
        "anchor_evidence_persistence_score", "anchor_evidence_distance_stability_score",
        "anchor_evidence_ball_control_score", "anchor_evidence_low_pressure_score",
        "anchor_evidence_density_score", "anchor_evidence_vector_score",
        "anchor_evidence_component_count",
        "observation_probability_H", "observation_probability_A",
        "observation_probability_D", "observation_probability_X",
        "observation_probability_entropy", "observation_probability_margin",
        "observation_probability_calibrated", "observation_probability_model",
        "observation_probability_uncertainty_score",
        "observation_probability_flight_support",
        "observation_probability_control_continuity_support",
        "observation_probability_contest_support",
        "scenario_significant_player_count", "scenario_home_significant_count",
        "scenario_away_significant_count", "scenario_player_density_near_ball",
        "candidate_pressure_m", "candidate_opponent_density_5m",
        "candidate_pass_options_15m", "opponent_closing_speed_mps",
        "short_option_blocked", "gegenpress_score", "pressure_switch_score",
        "landing_player_density_signal", "landing_pass_options_15m",
        "reception_pressure_score", "reception_pressure_context",
        "scenario_receiver_path_support_H", "scenario_receiver_path_support_A",
        "scenario_crowd_score", "scenario_balanced_crowd_candidate",
        "scenario_crowd_episode_candidate",
        "transition_activity_score", "transition_hypothesis_interval_id",
        "transition_hypothesis_start", "transition_hypothesis_end",
        "transition_hypothesis_activity", "transition_hypothesis_team",
        "transition_hypothesis_state", "transition_hypothesis_p_H",
        "transition_hypothesis_p_A", "transition_hypothesis_p_no_switch",
        "transition_hypothesis_p_contested", "transition_hypothesis_calibrated",
        "transition_kick_motion", "transition_no_switch_support",
        "transition_no_switch_cause_support", "transition_contested_cause_support",
        "transition_dead_ball_cause_support", "transition_uncertainty_cause_support",
        "transition_kalman_path_available", "transition_kalman_reliable",
        "transition_kalman_uncertainty_m", "transition_kalman_support",
        "transition_missing_ball_cause",
        "kick_episode_id", "kick_episode_start", "kick_episode_end",
        "kick_episode_end_reason", "kick_episode_distance_m",
        "kick_episode_contact_change", "kick_episode_observation_source",
        "kick_episode_kalman_supported",
    ]
    confidence_output = ROOT / "evaluation" / "reports" / "scenario_confidence_scores.csv"
    confidence_output.parent.mkdir(parents=True, exist_ok=True)
    scored_features.loc[:, [column for column in confidence_columns if column in scored_features]].to_csv(
        confidence_output, index=False
    )
    print(confidence_output)

    probability_columns = [
        "frame", "scenario_context", "scenario_state", "scenario_confidence",
        "observation_probability_H", "observation_probability_A",
        "observation_probability_D", "observation_probability_X",
        "observation_probability_entropy", "observation_probability_margin",
        "observation_probability_calibrated", "observation_probability_model",
        "observation_probability_uncertainty_score",
        "observation_probability_flight_support",
        "observation_probability_control_continuity_support",
        "observation_probability_contest_support",
        "missing_ball_cause", "missing_ball_cause_confidence",
        "scenario_crowd_score", "transition_hypothesis_state",
    ]
    probability_output = ROOT / "evaluation" / "reports" / "observation_probability_audit.csv"
    scored_features.loc[:, [column for column in probability_columns if column in scored_features]].to_csv(
        probability_output,
        index=False,
    )
    print(probability_output)

    transition_ledger_output = ROOT / "evaluation" / "reports" / "transition_hypothesis_ledger.csv"
    transition_ledger = build_transition_hypothesis_ledger(scored_features)
    transition_ledger.to_csv(
        transition_ledger_output,
        index=False,
    )
    print(transition_ledger_output)
    public_transition_audit_output = ROOT / "evaluation" / "reports" / "transition_public_error_audit.csv"
    public_transition_summary_output = ROOT / "evaluation" / "reports" / "transition_public_error_summary.csv"
    write_public_transition_error_audit(
        transition_ledger,
        ground_truth_path=ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv",
        output_path=public_transition_audit_output,
        summary_path=public_transition_summary_output,
    )
    print(public_transition_audit_output)
    print(public_transition_summary_output)

    # Evidence audit for missing-ball intervals. These columns are diagnostic
    # only: they make the dribble/pass/deflection/tackle interpretation
    # inspectable without converting an occlusion into a hand-written label.
    missing_ball_columns = [
        "frame", "ball_gap_id", "ball_gap_frames", "ball_gap_pre_speed_mps",
        "ball_gap_post_speed_mps", "ball_gap_speed_change_mps",
        "ball_gap_sharp_acceleration_risk", "gap_player_path_supported",
        "gap_player_path_team", "gap_player_speed_before_mps",
        "gap_player_speed_after_mps", "gap_player_ball_speed_match_mps",
        "gap_same_player_dribble_candidate", "gap_tackle_risk_candidate",
        "ball_direction_change_rad", "closest_distance_m",
        "candidate_opponent_density_5m", "opponent_closing_speed_mps",
        "scenario_significant_player_count", "scenario_home_significant_count",
        "scenario_away_significant_count", "scenario_player_density_near_ball",
        "scenario_crowd_score", "scenario_balanced_crowd_candidate",
        "scenario_crowd_episode_candidate",
        "ball_x_kalman", "ball_y_kalman", "ball_kalman_speed_mps",
        "ball_kalman_position_uncertainty_m", "ball_kalman_velocity_change_mps",
        "ball_kalman_path_available", "ball_kalman_reliable",
        "ball_kalman_prediction_confidence", "ball_kalman_gap_frames",
        "ball_kalman_displacement_m", "ball_kalman_direction_rad",
        "ball_kalman_acceleration_mps2", "ball_kalman_direction_change_rad",
        "missing_ball_cause", "missing_ball_cause_confidence",
        "missing_ball_cause_dead_or_out_score", "missing_ball_cause_flight_score",
        "missing_ball_cause_control_score", "missing_ball_cause_deflection_score",
        "missing_ball_cause_crowd_score", "missing_ball_cause_visibility_score",
        "missing_ball_cause_observation_count",
        "kick_episode_id", "kick_episode_distance_m",
        "kick_episode_contact_change", "kick_episode_observation_source",
        "kick_episode_kalman_supported",
        "scenario_context",
    ]
    missing_ball_audit_output = ROOT / "evaluation" / "reports" / "missing_ball_evidence_audit.csv"
    scored_features.loc[
        :, [column for column in missing_ball_columns if column in scored_features]
    ].loc[
        scored_features["ball_gap_id"].notna()
        if "ball_gap_id" in scored_features
        else scored_features["ball_x"].isna() | scored_features["ball_y"].isna()
    ].to_csv(missing_ball_audit_output, index=False)
    print(missing_ball_audit_output)

    kalman_audit_columns = [
        "frame", "ball_x", "ball_y", "ball_x_kalman", "ball_y_kalman",
        "ball_kalman_speed_mps", "ball_kalman_position_uncertainty_m",
        "ball_kalman_observation_residual_m", "ball_kalman_velocity_change_mps",
        "ball_kalman_path_available", "ball_kalman_reliable",
        "ball_kalman_prediction_confidence", "scenario_context",
    ]
    kalman_missing = scored_features["ball_x"].isna() | scored_features["ball_y"].isna()
    kalman_audit_output = ROOT / "evaluation" / "reports" / "kalman_ball_trajectory_audit.csv"
    scored_features.loc[
        kalman_missing,
        [column for column in kalman_audit_columns if column in scored_features],
    ].to_csv(kalman_audit_output, index=False)
    print(kalman_audit_output)

    kalman_summary = pd.DataFrame(
        {
            "metric": [
                "missing_ball_frames",
                "bidirectional_path_frames",
                "reliable_bidirectional_path_frames",
                "median_uncertainty_m_on_path",
                "p90_uncertainty_m_on_path",
            ],
            "value": [
                int(kalman_missing.sum()),
                int(scored_features.loc[kalman_missing, "ball_kalman_path_available"].sum()),
                int(scored_features.loc[kalman_missing, "ball_kalman_reliable"].sum()),
                float(scored_features.loc[kalman_missing & scored_features["ball_kalman_path_available"], "ball_kalman_position_uncertainty_m"].median()),
                float(scored_features.loc[kalman_missing & scored_features["ball_kalman_path_available"], "ball_kalman_position_uncertainty_m"].quantile(0.90)),
            ],
        }
    )
    kalman_summary_output = ROOT / "evaluation" / "reports" / "kalman_ball_trajectory_summary.csv"
    kalman_summary.to_csv(kalman_summary_output, index=False)
    print(kalman_summary_output)

    scenario_summary = (
        scored_features.groupby("scenario_context_causal", dropna=False)
        .agg(
            frames=("frame", "size"),
            episodes=("scenario_episode_id", "nunique"),
            median_episode_frames=("scenario_episode_length_frames", "median"),
            decoder_eligible_frames=("decoder_eligible", "sum"),
        )
        .reset_index()
        .sort_values(["frames", "scenario_context_causal"], ascending=[False, True])
    )
    scenario_summary_output = ROOT / "evaluation" / "reports" / "scenario_summary.csv"
    scenario_summary.to_csv(scenario_summary_output, index=False)
    print(scenario_summary_output)

    reception_audit = reception_transition_alignment_audit(temporal)
    reception_audit_output = ROOT / "evaluation" / "reports" / "reception_transition_alignment.csv"
    reception_audit.to_csv(reception_audit_output, index=False)
    reception_audit_summary = pd.DataFrame(
        {
            "metric": [
                "candidate_frames",
                "opposite_anchor_pairs",
                "candidate_matches_after_team",
                "opposite_and_matching_candidates",
            ],
            "value": [
                len(reception_audit),
                int(reception_audit["opposite_anchor_pair"].sum()) if not reception_audit.empty else 0,
                int(reception_audit["candidate_matches_after_team"].sum()) if not reception_audit.empty else 0,
                int((reception_audit["opposite_anchor_pair"] & reception_audit["candidate_matches_after_team"]).sum()) if not reception_audit.empty else 0,
            ],
        }
    )
    reception_audit_summary_output = ROOT / "evaluation" / "reports" / "reception_transition_alignment_summary.csv"
    reception_audit_summary.to_csv(reception_audit_summary_output, index=False)
    print(reception_audit_output)
    print(reception_audit_summary_output)

    aligned_switch_variants = {
        "baseline_aligned_reception_switch": {"use_sustained_signal": False},
        "baseline_aligned_sustained_reception_switch": {"use_sustained_signal": True},
        # Smart Tagging team metadata is a weak directional cue. It is only
        # allowed to support a switch already backed by opposite anchors and
        # reception geometry; it cannot create a label by itself.
        "baseline_aligned_reception_smart_team_context": {
            "use_sustained_signal": False,
            "use_team_context": True,
        },
        "baseline_aligned_physics_veto": {
            "use_sustained_signal": False,
            "use_physics_veto": True,
        },
        "baseline_trusted_transition": {
            "use_sustained_signal": False,
            "use_physics_veto": True,
            "minimum_transition_trust": 0.55,
        },
        "baseline_transition_interval_hypothesis": {
            "use_sustained_signal": False,
            "use_physics_veto": True,
            "require_transition_hypothesis": True,
            "minimum_transition_activity": 0.25,
        },
    }
    for variant_name, variant_kwargs in aligned_switch_variants.items():
        aligned_labels = resolve_aligned_reception_switch(
            temporal,
            dead_ball_labels,
            max_anchor_gap_frames=75,
            minimum_anchor_confidence=0.85,
            minimum_reception_score=0.55,
            **variant_kwargs,
        )
        aligned_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(aligned_output, aligned_labels)
        print(aligned_output)

    # Structural diagnostic: the interval ledger changes the cost of a
    # proposed switch, while hard anchors and graded masks remain protected.
    ledger_viterbi_labels = TransitionLedgerViterbi(rules).fit(
        temporal,
        base_labels=dead_ball_labels,
    ).predict(temporal)
    ledger_viterbi_output = ROOT / "evaluation" / "results" / "baseline_transition_ledger_viterbi.csv"
    model.write_submission(ledger_viterbi_output, ledger_viterbi_labels)
    print(ledger_viterbi_output)

    # Diagnostic probability-emission variant. The observation probabilities
    # are normalized but explicitly uncalibrated; this path is not promoted
    # unless it improves public metrics without breaking anchor protections.
    probability_ledger_labels = TransitionLedgerViterbi(
        rules,
        use_probability_emissions=True,
    ).fit(
        temporal,
        base_labels=dead_ball_labels,
    ).predict(temporal)
    probability_ledger_output = ROOT / "evaluation" / "results" / "baseline_probability_transition_ledger_viterbi.csv"
    model.write_submission(probability_ledger_output, probability_ledger_labels)
    print(probability_ledger_output)

    # Combined ablation: Haiqal contributes raw observations to our existing
    # probability-emission/transition-ledger path. His standalone labels and
    # Viterbi path remain excluded, so any change is attributable to the
    # additional evidence rather than decoder replacement.
    haiqal_probability_labels = TransitionLedgerViterbi(
        rules,
        use_probability_emissions=True,
    ).fit(
        haiqal_temporal,
        base_labels=dead_ball_labels,
    ).predict(haiqal_temporal)
    haiqal_probability_output = ROOT / "evaluation" / "results" / "baseline_haiqal_evidence_probability_ledger_viterbi.csv"
    model.write_submission(haiqal_probability_output, haiqal_probability_labels)
    print(haiqal_probability_output)

    # Stage 2: conservative duration-aware decoding on top of those scores.
    hsmm_labels = duration_aware_scenario_decode(
        scored_features,
        dead_ball_labels,
        minimum_team_run=8,
        switch_penalty=0.35,
        base_evidence_weight=0.75,
    )
    hsmm_output = ROOT / "evaluation" / "results" / "baseline_scenario_confidence_duration.csv"
    model.write_submission(hsmm_output, hsmm_labels)
    print(hsmm_output)

    # Future-looking reception decoding is intentionally not part of the
    # active runner. The implementation remains available as a documented
    # research branch for an explicitly offline smoother/live system with
    # permitted latency; it must not enter a causal submission unnoticed.


if __name__ == "__main__":
    main()

