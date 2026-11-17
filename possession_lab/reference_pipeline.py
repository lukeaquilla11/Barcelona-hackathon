"""Canonical organiser-facing possession pipeline.

This module is intentionally small.  It builds the shared observation table,
applies the promoted rule-first evidence path, writes one final submission,
and optionally runs the public evaluator.  Research ablations remain in
``run_interpolated_flight_experiment.py`` and the appendix notebooks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from possession_lab.baseline_pipeline import ClosestTeamRuleModel
from possession_lab.calculations.features import build_engineered_features
from possession_lab.decoder_experiments import (
    dead_ball_boundary_evidence,
    _remove_short_dead_runs,
    forward_settling_dead_ball_evidence,
    low_motion_dead_ball_evidence,
    resolve_hybrid_three_family_transition_states,
    retain_no_switch_intervals,
    scenario_gated_labels,
)
from possession_lab.scenario_decoder import decode_scenario_labels
from possession_lab.temporal_rules import load_rules


ROOT = Path(__file__).resolve().parents[1]


def build_reference_labels(
    features: pd.DataFrame,
    rules: dict[str, Any],
) -> np.ndarray:
    """Apply the promoted observation-first policy to one feature table."""
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

    # Reproduce the promoted reference's two boundary stages before applying
    # the final hybrid resolver.  The first stage catches explicit low-motion
    # stoppages; the second catches visible-ball settling without imposing a
    # fixed dead-ball duration.
    dead_rules = rules["dead_ball_evidence"]
    labels = dead_ball_boundary_evidence(
        features,
        labels,
        low_activity_median_mps=float(dead_rules["low_activity_median_mps"]),
        low_activity_p90_mps=float(dead_rules["low_activity_p90_mps"]),
        minimum_low_motion_frames=int(dead_rules["minimum_low_motion_frames"]),
        maximum_out_of_bounds_bridge_frames=int(
            dead_rules["maximum_out_of_bounds_bridge_frames"]
        ),
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
        minimum_bridge_low_activity_fraction=float(
            dead_rules["visible_ball_minimum_bridge_low_activity_fraction"]
        ),
    )

    labels = retain_no_switch_intervals(
        features,
        labels,
        minimum_no_switch_probability=0.65,
        minimum_interval_support=0.60,
        maximum_interval_frames=300,
        minimum_prior_team_run_frames=1,
    )

    # The state layer proposes explanations; this conservative hybrid is the
    # only final-label resolver used by the reference pipeline.
    settling = forward_settling_dead_ball_evidence(
        features,
        labels,
        lookahead_frames=75,
        maximum_current_speed_mps=10.0,
        maximum_terminal_speed_mps=4.0,
        minimum_speed_drop_mps=4.0,
        minimum_activity_drop_mps=0.8,
        minimum_receiver_separation_m=3.0,
        minimum_candidate_run_frames=12,
    )
    return resolve_hybrid_three_family_transition_states(
        features,
        settling,
        settling,
        minimum_enter_d_support=0.75,
        minimum_control_run_frames=3,
        minimum_switch_confidence=0.60,
        maximum_local_change_frames=75,
        maximum_forward_settling_frames=75,
    )


def run_reference_pipeline(
    *,
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    score_public: bool = True,
) -> Path:
    """Build raw observations, write the final CSV, and optionally score it."""
    rules = load_rules()
    model = ClosestTeamRuleModel(data_dir=data_dir)
    features = build_engineered_features(model=model, rules=rules)
    labels = build_reference_labels(features, rules)
    output = Path(output_path or ROOT / "evaluation" / "results" / "final_submission.csv")
    model.write_submission(output, labels)

    if score_public:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "evaluation" / "run_evaluation.py"),
                "--gt",
                "public",
                "--results",
                str(output.parent),
                "--out",
                str(ROOT / "evaluation"),
                "--no-detail",
            ],
            cwd=ROOT,
            check=True,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation" / "results" / "final_submission.csv",
    )
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()
    output = run_reference_pipeline(
        data_dir=args.data_dir or os.environ.get("MTAH_DATA_DIR"),
        output_path=args.output,
        score_public=not args.no_score,
    )
    print(f"final submission: {output}")


if __name__ == "__main__":
    main()
