"""Composable pipeline specifications for benchmark experiments.

Each benchmark selects components explicitly instead of duplicating the full
possession algorithm. The feature table is shared; only the decision stages
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from possession.baseline_pipeline import graded_mask
from possession.decoder_experiments import (
    _remove_short_dead_runs,
    scenario_gated_labels,
    dead_ball_boundary_evidence,
    duration_aware_scenario_decode,
    literal_out_of_bounds_dead_ball_evidence,
    release_restart_dead_ball_on_control,
    resolve_aligned_reception_switch,
    apply_control_event_anchor_experiment,
    strict_set_piece_dead_ball_smoothing,
)
from possession.scenario_decoder import decode_scenario_labels
from possession.sequence_models import (
    AnchorInitializedHMM,
    AnchorInitializedHSMM,
    BayesianAnchorHMM,
    RuleConstrainedViterbi,
    SelectiveRuleViterbi,
)
from possession.models.stage_hq_viterbi import stage_hq_labels


@dataclass(frozen=True)
class PipelineSpec:
    """Declare which evidence and decision layers a benchmark uses."""

    name: str
    anchor_source: str = "selected_anchor_team"
    use_smart_set_piece: bool = True
    use_dead_ball_boundary: bool = True
    use_literal_out_of_bounds: bool = False
    transition_rule: str = "aligned_reception"
    use_physics_veto: bool = False
    decoder: str = "none"
    protect_dead_ball: bool = True
    protect_outside_graded: bool = True
    minimum_control_anchor_confidence: float = 0.65


def available_pipeline_specs() -> dict[str, PipelineSpec]:
    """Return named, reproducible benchmark configurations."""
    return {
        "protected_aligned_reception": PipelineSpec(
            name="protected_aligned_reception",
        ),
        "aligned_physics_veto": PipelineSpec(
            name="aligned_physics_veto",
            use_physics_veto=True,
        ),
        "stable_anchor_only": PipelineSpec(
            name="stable_anchor_only",
            anchor_source="anchor_team",
            use_dead_ball_boundary=False,
            transition_rule="none",
        ),
        "duration_decoder_diagnostic": PipelineSpec(
            name="duration_decoder_diagnostic",
            transition_rule="none",
            decoder="duration_viterbi",
        ),
        "rule_viterbi_diagnostic": PipelineSpec(
            name="rule_viterbi_diagnostic",
            transition_rule="none",
            decoder="viterbi",
        ),
        "selective_viterbi_diagnostic": PipelineSpec(
            name="selective_viterbi_diagnostic",
            transition_rule="none",
            decoder="selective_viterbi",
        ),
        "anchor_hmm_diagnostic": PipelineSpec(
            name="anchor_hmm_diagnostic",
            transition_rule="none",
            decoder="hmm",
        ),
        "anchor_hsmm_diagnostic": PipelineSpec(
            name="anchor_hsmm_diagnostic",
            transition_rule="none",
            decoder="hsmm",
        ),
        "bayesian_anchor_hmm_diagnostic": PipelineSpec(
            name="bayesian_anchor_hmm_diagnostic",
            transition_rule="none",
            decoder="bayesian_hmm",
        ),
        "stage_hq_viterbi": PipelineSpec(
            name="stage_hq_viterbi",
            use_smart_set_piece=False,
            use_dead_ball_boundary=False,
            transition_rule="none",
            decoder="stage_hq_viterbi",
            protect_dead_ball=False,
        ),
        "evidence_bounded_deadball": PipelineSpec(
            name="evidence_bounded_deadball",
            use_dead_ball_boundary=True,
            transition_rule="aligned_reception",
            use_physics_veto=True,
            protect_dead_ball=False,
        ),
        "literal_out_of_bounds_diagnostic": PipelineSpec(
            name="literal_out_of_bounds_diagnostic",
            transition_rule="none",
            use_literal_out_of_bounds=True,
        ),
        "strict_set_piece_deadball": PipelineSpec(
            name="strict_set_piece_deadball",
            use_dead_ball_boundary=True,
            transition_rule="aligned_reception",
            use_physics_veto=True,
            protect_dead_ball=False,
        ),
        "control_event_anchor_experiment": PipelineSpec(
            name="control_event_anchor_experiment",
            use_dead_ball_boundary=True,
            transition_rule="aligned_reception",
            use_physics_veto=True,
            protect_dead_ball=False,
            minimum_control_anchor_confidence=0.65,
        ),
        "control_event_anchor_strict": PipelineSpec(
            name="control_event_anchor_strict",
            use_dead_ball_boundary=True,
            transition_rule="aligned_reception",
            use_physics_veto=True,
            protect_dead_ball=False,
            minimum_control_anchor_confidence=0.85,
        ),
        "strict_control_event_anchor_experiment": PipelineSpec(
            name="strict_control_event_anchor_experiment",
            use_dead_ball_boundary=True,
            transition_rule="aligned_reception",
            use_physics_veto=True,
            protect_dead_ball=False,
        ),
    }


def _base_anchor_labels(features: pd.DataFrame, spec: PipelineSpec) -> np.ndarray:
    """Create labels from the selected anchor source without inventing teams."""
    labels = np.full(len(features), "X", dtype="<U1")
    graded = graded_mask(len(features))
    if spec.use_smart_set_piece and "smart_set_piece" in features:
        labels[graded & features["smart_set_piece"].fillna(False).to_numpy(bool)] = "D"
    if spec.anchor_source not in features:
        return labels
    teams = features[spec.anchor_source].to_numpy(object)
    valid = graded & np.isin(teams, ["H", "A"])
    labels[valid] = teams[valid]
    return labels


def apply_pipeline_spec(
    features: pd.DataFrame,
    spec: PipelineSpec,
    rules: dict[str, Any],
) -> np.ndarray:
    """Apply one declared benchmark pipeline to a shared feature table."""
    labels = _base_anchor_labels(features, spec)
    if spec.use_dead_ball_boundary:
        labels = dead_ball_boundary_evidence(
            features,
            labels,
            low_activity_median_mps=float(rules["dead_ball_evidence"]["low_activity_median_mps"]),
            low_activity_p90_mps=float(rules["dead_ball_evidence"]["low_activity_p90_mps"]),
            minimum_low_motion_frames=int(rules["dead_ball_evidence"]["minimum_low_motion_frames"]),
            maximum_out_of_bounds_bridge_frames=int(rules["dead_ball_evidence"]["maximum_out_of_bounds_bridge_frames"]),
        )
    if spec.use_literal_out_of_bounds:
        evidence = rules["dead_ball_evidence"]
        labels = literal_out_of_bounds_dead_ball_evidence(
            features,
            labels,
            minimum_run_frames=int(evidence["minimum_out_of_bounds_run_frames"]),
            boundary_tolerance=float(evidence["out_of_bounds_boundary_tolerance"]),
        )
    if spec.transition_rule == "aligned_reception":
        # Reuse the established aligned-reception baseline handling. All model
        # variants receive the same engineered evidence and scenarios; the
        # spec changes only how those observations are converted into labels.
        interpolated = features.copy()
        if "recent_flight_evidence_interpolated" in interpolated:
            interpolated["recent_flight_evidence"] = interpolated["recent_flight_evidence_interpolated"]
        if "ball_flight_candidate_interpolated" in interpolated:
            interpolated["ball_flight_candidate"] = interpolated["ball_flight_candidate_interpolated"]
        labels = scenario_gated_labels(
            interpolated,
            allow_receiver_switch=True,
            allow_bidirectional_bridge=True,
            receiver_horizon_frames=75,
            maximum_internal_spike=75,
        )
        scenario_labels, _ = decode_scenario_labels(interpolated, rules)
        dead_support = (
            interpolated.get("graded", pd.Series(False, index=interpolated.index)).to_numpy(bool)
            & interpolated.get("smart_set_piece", pd.Series(False, index=interpolated.index)).to_numpy(bool)
            & (scenario_labels == "D")
            & interpolated.get("scenario_context", pd.Series("UNKNOWN", index=interpolated.index)).isin(
                ["DEAD", "UNKNOWN", "OCCLUDED"]
            ).to_numpy(bool)
        )
        labels[dead_support] = "D"
        labels = _remove_short_dead_runs(labels, minimum_run=25)
        labels = dead_ball_boundary_evidence(
            interpolated,
            labels,
            low_activity_median_mps=float(rules["dead_ball_evidence"]["low_activity_median_mps"]),
            low_activity_p90_mps=float(rules["dead_ball_evidence"]["low_activity_p90_mps"]),
            minimum_low_motion_frames=int(rules["dead_ball_evidence"]["minimum_low_motion_frames"]),
            maximum_out_of_bounds_bridge_frames=int(rules["dead_ball_evidence"]["maximum_out_of_bounds_bridge_frames"]),
        )
        labels = resolve_aligned_reception_switch(
            interpolated,
            labels,
            max_anchor_gap_frames=75,
            minimum_anchor_confidence=0.85,
            minimum_reception_score=0.55,
            use_physics_veto=spec.use_physics_veto,
        )
        if spec.name == "evidence_bounded_deadball":
            labels = release_restart_dead_ball_on_control(
                interpolated,
                labels,
                minimum_anchor_confidence=0.85,
                require_goal_kick_release=True,
            )
    elif spec.transition_rule not in {"none", "aligned_reception"}:
        raise ValueError(f"Unknown transition rule: {spec.transition_rule}")
    if spec.decoder == "duration_viterbi":
        labels = duration_aware_scenario_decode(features, labels)
    elif spec.decoder == "viterbi":
        labels = RuleConstrainedViterbi(rules).fit(features).predict(features)
    elif spec.decoder == "selective_viterbi":
        labels = SelectiveRuleViterbi(rules).fit(features).predict(features)
    elif spec.decoder == "hmm":
        labels = AnchorInitializedHMM(rules).fit(features).predict(features)
    elif spec.decoder == "hsmm":
        labels = AnchorInitializedHSMM(rules).fit(features).predict(features)
    elif spec.decoder == "bayesian_hmm":
        labels = BayesianAnchorHMM(rules).fit(features).predict(features)
    elif spec.decoder == "stage_hq_viterbi":
        labels = stage_hq_labels(features)
    elif spec.decoder != "none":
        raise ValueError(f"Unknown decoder: {spec.decoder}")
    # Every decoder is a consistency layer, not permission to override hard
    # evidence. Restore the same protections after the decoder has run.
    graded = graded_mask(len(features))
    if spec.name == "evidence_bounded_deadball":
        # A confirmed literal out-of-pitch run is D regardless of what the
        # nearest-player or decoder layers inferred. The diagnostic helper
        # requires the configured minimum run, protecting against one-frame
        # coordinate spikes.
        evidence = rules["dead_ball_evidence"]
        labels = literal_out_of_bounds_dead_ball_evidence(
            features,
            labels,
            minimum_run_frames=int(evidence["minimum_out_of_bounds_run_frames"]),
            boundary_tolerance=float(evidence["out_of_bounds_boundary_tolerance"]),
        )
    if spec.name == "strict_set_piece_deadball":
        labels = strict_set_piece_dead_ball_smoothing(
            features,
            labels,
            minimum_anchor_confidence=0.85,
        )
    if spec.name in {"control_event_anchor_experiment", "strict_control_event_anchor_experiment"}:
        labels = apply_control_event_anchor_experiment(
            features,
            labels,
            minimum_control_anchor_confidence=spec.minimum_control_anchor_confidence,
        )
    if spec.name == "strict_control_event_anchor_experiment":
        labels = strict_set_piece_dead_ball_smoothing(features, labels, minimum_anchor_confidence=0.85)
    if spec.protect_dead_ball and "smart_set_piece" in features:
        set_piece = features["smart_set_piece"].fillna(False).to_numpy(bool)
        labels[graded & set_piece] = "D"
    if spec.protect_outside_graded:
        labels[~graded] = "X"
    return labels.astype("<U1")
