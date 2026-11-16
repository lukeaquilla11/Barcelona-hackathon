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

from possession_lab.baseline_pipeline import graded_mask
from possession_lab.decoder_experiments import (
    dead_ball_boundary_evidence,
    duration_aware_scenario_decode,
    literal_out_of_bounds_dead_ball_evidence,
    resolve_aligned_reception_switch,
)
from possession_lab.sequence_models import (
    AnchorInitializedHMM,
    AnchorInitializedHSMM,
    BayesianAnchorHMM,
    RuleConstrainedViterbi,
    SelectiveRuleViterbi,
    TransitionLedgerViterbi,
)


@dataclass(frozen=True)
class PipelineSpec:
    """Declare which evidence and decision layers a benchmark uses."""

    name: str
    anchor_source: str = "selected_anchor_team"
    use_smart_set_piece: bool = True
    use_dead_ball_boundary: bool = True
    use_literal_out_of_bounds: bool = False
    transition_rule: str = "aligned_reception"
    decoder: str = "none"
    protect_dead_ball: bool = True
    protect_outside_graded: bool = True


def available_pipeline_specs() -> dict[str, PipelineSpec]:
    """Return named, reproducible benchmark configurations."""
    return {
        "protected_aligned_reception": PipelineSpec(
            name="protected_aligned_reception",
        ),
        "trusted_transition_diagnostic": PipelineSpec(
            name="trusted_transition_diagnostic",
            transition_rule="trusted_aligned_reception",
        ),
        "transition_interval_hypothesis_diagnostic": PipelineSpec(
            name="transition_interval_hypothesis_diagnostic",
            transition_rule="transition_interval_hypothesis",
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
        "transition_ledger_viterbi_diagnostic": PipelineSpec(
            name="transition_ledger_viterbi_diagnostic",
            transition_rule="none",
            decoder="transition_ledger_viterbi",
        ),
        "probability_transition_ledger_viterbi_diagnostic": PipelineSpec(
            name="probability_transition_ledger_viterbi_diagnostic",
            transition_rule="none",
            decoder="probability_transition_ledger_viterbi",
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
        "literal_out_of_bounds_diagnostic": PipelineSpec(
            name="literal_out_of_bounds_diagnostic",
            transition_rule="none",
            use_literal_out_of_bounds=True,
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
        labels = resolve_aligned_reception_switch(features, labels)
    elif spec.transition_rule == "trusted_aligned_reception":
        labels = resolve_aligned_reception_switch(
            features,
            labels,
            minimum_transition_trust=0.55,
            use_physics_veto=True,
        )
    elif spec.transition_rule == "transition_interval_hypothesis":
        labels = resolve_aligned_reception_switch(
            features,
            labels,
            use_physics_veto=True,
            require_transition_hypothesis=True,
            minimum_transition_activity=0.25,
        )
    elif spec.transition_rule not in {"none", "aligned_reception"}:
        raise ValueError(f"Unknown transition rule: {spec.transition_rule}")
    if spec.decoder == "duration_viterbi":
        labels = duration_aware_scenario_decode(features, labels)
    elif spec.decoder == "viterbi":
        labels = RuleConstrainedViterbi(rules).fit(features).predict(features)
    elif spec.decoder == "selective_viterbi":
        labels = SelectiveRuleViterbi(rules).fit(features).predict(features)
    elif spec.decoder == "transition_ledger_viterbi":
        labels = TransitionLedgerViterbi(rules).fit(
            features,
            base_labels=labels,
        ).predict(features)
    elif spec.decoder == "probability_transition_ledger_viterbi":
        labels = TransitionLedgerViterbi(
            rules,
            use_probability_emissions=True,
        ).fit(
            features,
            base_labels=labels,
        ).predict(features)
    elif spec.decoder == "hmm":
        labels = AnchorInitializedHMM(rules).fit(features).predict(features)
    elif spec.decoder == "hsmm":
        labels = AnchorInitializedHSMM(rules).fit(features).predict(features)
    elif spec.decoder == "bayesian_hmm":
        labels = BayesianAnchorHMM(rules).fit(features).predict(features)
    elif spec.decoder != "none":
        raise ValueError(f"Unknown decoder: {spec.decoder}")
    # Every decoder is a consistency layer, not permission to override hard
    # evidence. Restore the same protections after the decoder has run.
    graded = graded_mask(len(features))
    if spec.protect_dead_ball and "smart_set_piece" in features:
        set_piece = features["smart_set_piece"].fillna(False).to_numpy(bool)
        labels[graded & set_piece] = "D"
    if spec.anchor_source in features:
        teams = features[spec.anchor_source].to_numpy(object)
        for team in ("H", "A"):
            labels[graded & (teams == team)] = team
    if spec.protect_outside_graded:
        labels[~graded] = "X"
    return labels.astype("<U1")
