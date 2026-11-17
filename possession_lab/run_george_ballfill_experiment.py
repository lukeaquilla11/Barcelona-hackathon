"""Run the current best resolver with George's ball observations as an ablation.

This is deliberately a separate runner from the organiser-facing full model:
the no-George output remains the reference, while this command writes a
different candidate and a compact observation audit.  It never reads ground
truth while constructing labels.

``--evidence-only`` keeps the tactical ball path unchanged and passes only
George's source/uncertainty observations into missing-ball cause and
transition evidence.  Without it, the runner performs the older coordinate
fill ablation.

Usage:
    python -m possession_lab.run_george_ballfill_experiment \
        --ballfill /path/to/BAR-ATH_ball.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from possession_lab.baseline_pipeline import ClosestTeamRuleModel
from possession_lab.calculations.features import build_engineered_features
from possession_lab.decoder_experiments import (
    _remove_short_dead_runs,
    dead_ball_boundary_evidence,
    low_motion_dead_ball_evidence,
    retain_no_switch_intervals,
    scenario_gated_labels,
)
from possession_lab.scenario_decoder import decode_scenario_labels
from possession_lab.temporal_rules import load_rules


ROOT = Path(__file__).resolve().parents[1]


_NUMERIC_AUDIT_COLUMNS = (
    "ball_x",
    "ball_y",
    "ball_displacement_m",
    "ball_speed_derived_mps",
    "ball_acceleration_mps2",
    "ball_speed_evidence_mps",
    "ball_kalman_speed_mps",
    "ball_gap_implied_speed_mps",
    "closest_distance_m",
    "scenario_crowd_score",
)

_CATEGORICAL_AUDIT_COLUMNS = (
    "scenario_context",
    "scenario_context_causal",
    "missing_ball_cause",
    "anchor_type",
    "reception_signal",
    "long_ball_candidate",
    "ball_in_flight",
    "ball_missing_observed",
)


def build_current_best_labels(features: pd.DataFrame, rules: dict) -> np.ndarray:
    """Apply the current visible-ball + NO_SWITCH resolver to observations."""
    interpolated = features.copy()
    interpolated["recent_flight_evidence"] = features["recent_flight_evidence_interpolated"]
    interpolated["ball_flight_candidate"] = features["ball_flight_candidate_interpolated"]
    labels = scenario_gated_labels(
        interpolated,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        receiver_horizon_frames=75,
        maximum_internal_spike=75,
    )
    scenario_labels, _ = decode_scenario_labels(features, rules)
    dead_support = (
        features["graded"].to_numpy(bool)
        & features["smart_set_piece"].to_numpy(bool)
        & (scenario_labels == "D")
        & features["scenario_context"].isin(["DEAD", "UNKNOWN", "OCCLUDED"]).to_numpy(bool)
    )
    labels[dead_support] = "D"
    labels = _remove_short_dead_runs(labels, minimum_run=25)

    dead_rules = rules["dead_ball_evidence"]
    labels = dead_ball_boundary_evidence(
        features,
        labels,
        low_activity_median_mps=float(dead_rules["low_activity_median_mps"]),
        low_activity_p90_mps=float(dead_rules["low_activity_p90_mps"]),
        minimum_low_motion_frames=int(dead_rules["minimum_low_motion_frames"]),
        maximum_out_of_bounds_bridge_frames=int(dead_rules["maximum_out_of_bounds_bridge_frames"]),
    )

    labels = low_motion_dead_ball_evidence(
        features,
        labels,
        activity_window_frames=int(dead_rules["visible_ball_activity_window_frames"]),
        minimum_support_fraction=float(dead_rules["visible_ball_minimum_support_fraction"]),
        minimum_candidate_run_frames=int(dead_rules["visible_ball_minimum_candidate_run_frames"]),
        maximum_ball_speed_mps=float(dead_rules["visible_ball_maximum_speed_mps"]),
        maximum_settling_speed_mps=float(dead_rules["visible_ball_maximum_settling_speed_mps"]),
        settling_acceleration_mps2=float(dead_rules["visible_ball_settling_acceleration_mps2"]),
        maximum_restart_bridge_frames=int(dead_rules["visible_ball_maximum_restart_bridge_frames"]),
        maximum_candidate_bridge_frames=int(dead_rules["visible_ball_maximum_candidate_bridge_frames"]),
        minimum_bridge_low_activity_fraction=float(dead_rules["visible_ball_minimum_bridge_low_activity_fraction"]),
    )
    return retain_no_switch_intervals(
        features,
        labels,
        minimum_no_switch_probability=0.65,
        minimum_interval_support=0.60,
        maximum_interval_frames=300,
        minimum_prior_team_run_frames=1,
    )


def _audit_feature_changes(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    """Summarise upstream signal changes without reading any ground truth.

    This is intentionally an observation/scenario audit, not an accuracy report.  A filled ball
    may change speed or scenario evidence while the conservative resolver quite correctly keeps
    the final H/A/D label unchanged.  That distinction is the point of this experiment.
    """
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate feature tables have different lengths")
    rows: list[dict[str, object]] = []
    filled = candidate.get("ball_filled_from_george", pd.Series(False, index=candidate.index)).fillna(False).to_numpy(bool)
    graded = candidate.get("graded", pd.Series(True, index=candidate.index)).fillna(False).to_numpy(bool)

    for column in _NUMERIC_AUDIT_COLUMNS:
        if column not in reference or column not in candidate:
            continue
        before = pd.to_numeric(reference[column], errors="coerce").to_numpy(float)
        after = pd.to_numeric(candidate[column], errors="coerce").to_numpy(float)
        finite_both = np.isfinite(before) & np.isfinite(after)
        changed = finite_both & (np.abs(after - before) > 1e-9)
        gained = ~np.isfinite(before) & np.isfinite(after)
        rows.append(
            {
                "kind": "numeric",
                "feature": column,
                "changed_frames": int(changed.sum()),
                "changed_graded_frames": int((changed & graded).sum()),
                "filled_frames_changed": int((changed & filled).sum()),
                "finite_values_gained": int(gained.sum()),
                "mean_abs_delta_changed": float(np.mean(np.abs(after[changed] - before[changed]))) if changed.any() else 0.0,
                "max_abs_delta_changed": float(np.max(np.abs(after[changed] - before[changed]))) if changed.any() else 0.0,
            }
        )

    for column in _CATEGORICAL_AUDIT_COLUMNS:
        if column not in reference or column not in candidate:
            continue
        before = reference[column].astype("string").fillna("<NA>").to_numpy(object)
        after = candidate[column].astype("string").fillna("<NA>").to_numpy(object)
        changed = before != after
        rows.append(
            {
                "kind": "categorical",
                "feature": column,
                "changed_frames": int(changed.sum()),
                "changed_graded_frames": int((changed & graded).sum()),
                "filled_frames_changed": int((changed & filled).sum()),
                "finite_values_gained": 0,
                "mean_abs_delta_changed": np.nan,
                "max_abs_delta_changed": np.nan,
            }
        )

    audit = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "n_frames",
                "n_filled_from_george",
                "n_filled_from_george_graded",
                "n_scenario_context_changes",
                "n_scenario_context_changes_graded",
                "n_missing_ball_cause_changes",
                "n_missing_ball_cause_changes_graded",
            ],
            "value": [
                len(candidate),
                int(filled.sum()),
                int((filled & graded).sum()),
                int(((reference.get("scenario_context", pd.Series(index=reference.index, dtype=object)).astype("string").fillna("<NA>").to_numpy(object))
                     != candidate.get("scenario_context", pd.Series(index=candidate.index, dtype=object)).astype("string").fillna("<NA>").to_numpy(object)).sum()),
                int(((reference.get("scenario_context", pd.Series(index=reference.index, dtype=object)).astype("string").fillna("<NA>").to_numpy(object))
                     != candidate.get("scenario_context", pd.Series(index=candidate.index, dtype=object)).astype("string").fillna("<NA>").to_numpy(object))[graded].sum()),
                int(((reference.get("missing_ball_cause", pd.Series(index=reference.index, dtype=object)).astype("string").fillna("<NA>").to_numpy(object))
                     != candidate.get("missing_ball_cause", pd.Series(index=candidate.index, dtype=object)).astype("string").fillna("<NA>").to_numpy(object)).sum()),
                int(((reference.get("missing_ball_cause", pd.Series(index=reference.index, dtype=object)).astype("string").fillna("<NA>").to_numpy(object))
                     != candidate.get("missing_ball_cause", pd.Series(index=candidate.index, dtype=object)).astype("string").fillna("<NA>").to_numpy(object))[graded].sum()),
            ],
        }
    )
    summary.to_csv(output_path.with_name(output_path.stem + "_summary.csv"), index=False)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ballfill", required=True, type=Path)
    parser.add_argument(
        "--allow-model-filled",
        action="store_true",
        help="also allow George's carrier-HMM 'filled' source; off by default",
    )
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="keep tactical coordinates unchanged; use George only as auxiliary evidence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation/results/diagnostic_george_fused_observed_no_switch_interval.csv",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="also rebuild the no-George feature table and write an upstream signal/scenario audit",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT / "evaluation/reports/george_ballfill_feature_audit.csv",
    )
    args = parser.parse_args()

    rules = load_rules()
    model = ClosestTeamRuleModel()
    features = build_engineered_features(
        model=model,
        rules=rules,
        george_ballfill_path=args.ballfill,
        george_allow_model_filled=args.allow_model_filled,
        george_evidence_only=args.evidence_only,
    )
    labels = build_current_best_labels(features, rules)
    model.write_submission(args.output, labels)

    if args.compare_baseline:
        baseline_features = build_engineered_features(model=model, rules=rules)
        audit = _audit_feature_changes(
            baseline_features,
            features,
            output_path=args.audit_output,
        )
        print(args.audit_output)
        print(audit.to_string(index=False))

    filled = features["ball_filled_from_george"].fillna(False).to_numpy(bool)
    audit = pd.DataFrame(
        {
            "metric": [
                "n_frames",
                "n_george_available",
                "n_george_eligible",
                "n_tactical_gaps_filled",
                "n_graded_tactical_gaps_filled",
            ],
            "value": [
                len(features),
                int(features["ball_fill_available"].fillna(False).sum()),
                int(features["ball_fill_eligible"].fillna(False).sum()),
                int(filled.sum()),
                int((filled & features["graded"].to_numpy(bool)).sum()),
            ],
        }
    )
    audit_path = ROOT / "evaluation/reports/george_ballfill_observation_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    print(args.output)
    print(audit_path)
    print(audit.to_string(index=False))
    print("filled sources:")
    print(features.loc[filled, "ball_fill_source"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
