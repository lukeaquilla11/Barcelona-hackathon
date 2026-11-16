"""Run the current best resolver with George's ball observations as an ablation.

This is deliberately a separate runner from the organiser-facing full model:
the no-George output remains the reference, while this command writes a
different candidate and a compact observation audit.  It never reads ground
truth while constructing labels.

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ballfill", required=True, type=Path)
    parser.add_argument(
        "--allow-model-filled",
        action="store_true",
        help="also allow George's carrier-HMM 'filled' source; off by default",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation/results/diagnostic_george_fused_observed_no_switch_interval.csv",
    )
    args = parser.parse_args()

    rules = load_rules()
    model = ClosestTeamRuleModel()
    features = build_engineered_features(
        model=model,
        rules=rules,
        george_ballfill_path=args.ballfill,
        george_allow_model_filled=args.allow_model_filled,
    )
    labels = build_current_best_labels(features, rules)
    model.write_submission(args.output, labels)

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
