"""Evaluate a label-separated MLE calibration experiment.

This runner does not create a possession submission.  It calibrates the
existing rule-evidence vector on the first contiguous half of the reviewed
public slice and reports log loss/Brier score on the second half.  The split
is deliberately explicit so the public labels cannot silently improve the
same frames used for evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from possession_lab.calibration import (
    fit_temperature_mle,
    multiclass_brier_score,
    multiclass_log_loss,
)


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("H", "A", "D", "X")


def main() -> None:
    probability_path = ROOT / "evaluation" / "reports" / "observation_probability_audit.csv"
    ground_truth_path = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    report_path = ROOT / "evaluation" / "reports" / "probability_calibration_public_holdout.csv"
    if not probability_path.exists():
        raise FileNotFoundError(f"run the evidence pipeline first: {probability_path}")

    probability = pd.read_csv(probability_path)
    ground_truth = pd.read_csv(ground_truth_path)
    table = probability[["frame", "observation_probability_H", "observation_probability_A", "observation_probability_D", "observation_probability_X"]].merge(
        ground_truth[["frame", "label"]], on="frame", how="inner", validate="one_to_one"
    )
    table = table[table["label"].isin(CLASSES[:3])].copy()
    if len(table) < 4:
        raise ValueError("not enough reviewed public frames for a calibration split")
    table = table.sort_values("frame").reset_index(drop=True)
    midpoint = len(table) // 2
    calibration = np.arange(len(table)) < midpoint
    holdout = ~calibration
    probabilities = table[[
        "observation_probability_H",
        "observation_probability_A",
        "observation_probability_D",
        "observation_probability_X",
    ]].to_numpy(float)
    logits = np.log(np.maximum(probabilities, 1e-12))
    labels = table["label"].to_numpy(str)

    calibrator = fit_temperature_mle(logits, labels, classes=CLASSES, mask=calibration)
    uncalibrated = probabilities[holdout]
    calibrated = calibrator.predict_proba(logits[holdout])
    summary = pd.DataFrame([
        {"split": "holdout", "model": "rule_evidence_softmax", "temperature": 1.0,
         "log_loss": multiclass_log_loss(uncalibrated, labels[holdout], classes=CLASSES),
         "brier": multiclass_brier_score(uncalibrated, labels[holdout], classes=CLASSES)},
        {"split": "holdout", "model": "mle_temperature_scaled", "temperature": calibrator.temperature,
         "log_loss": multiclass_log_loss(calibrated, labels[holdout], classes=CLASSES),
         "brier": multiclass_brier_score(calibrated, labels[holdout], classes=CLASSES)},
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_path, index=False)
    metadata = {
        "calibration_frames": int(calibration.sum()),
        "holdout_frames": int(holdout.sum()),
        "calibration_frame_start": int(table.loc[calibration, "frame"].min()),
        "calibration_frame_end": int(table.loc[calibration, "frame"].max()),
        "holdout_frame_start": int(table.loc[holdout, "frame"].min()),
        "holdout_frame_end": int(table.loc[holdout, "frame"].max()),
        "temperature": float(calibrator.temperature),
        "labels_are_training_input": True,
        "purpose": "separate calibration experiment; not a submission model",
    }
    (report_path.with_suffix(".json")).write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"-> {report_path}")


if __name__ == "__main__":
    main()
