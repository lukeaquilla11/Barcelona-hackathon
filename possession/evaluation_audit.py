"""Deterministic diagnosis of frame and transition errors against available ground truth."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _events(labels: pd.Series) -> list[dict[str, object]]:
    """Return label-change events, without crossing missing frames or GT ``X`` rows."""
    values = labels.astype("string")
    frames = values.index.to_numpy(dtype=int)
    events: list[dict[str, object]] = []
    for index in range(1, len(values)):
        previous, current = values.iloc[index - 1], values.iloc[index]
        if frames[index] != frames[index - 1] + 1:
            continue
        if pd.isna(previous) or pd.isna(current) or previous == "X" or current == "X":
            continue
        if previous != current:
            events.append({"frame": int(frames[index]), "from": str(previous), "to": str(current)})
    return events


def classify_prediction_errors(
    features: pd.DataFrame,
    ground_truth: pd.Series,
    predictions: pd.Series,
    *,
    transition_tolerance_frames: int = 25,
    fps: float = 25.0,
) -> Mapping[str, pd.DataFrame]:
    """Explain exactly where a prediction differs from GT and what kind of error it is.

    Returns ``frame_errors`` (one row per wrong graded frame), ``transition_errors`` (missed,
    spurious, mistimed, or wrong-destination switches), and ``summary``.  The function is
    deterministic: it only aligns frame-indexed labels, matches transitions within the supplied
    tolerance, and copies available scenario/anchor/confidence columns from ``features``.
    """
    gt = ground_truth.copy()
    pred = predictions.copy()
    gt.index = pd.to_numeric(gt.index, errors="raise").astype(int)
    pred.index = pd.to_numeric(pred.index, errors="raise").astype(int)
    aligned = pd.DataFrame({"ground_truth": gt.astype("string"), "prediction": pred.astype("string")})
    aligned = aligned.dropna(subset=["ground_truth", "prediction"])
    aligned["frame"] = aligned.index.astype(int)
    graded = aligned["ground_truth"].ne("X")
    wrong = graded & aligned["ground_truth"].ne(aligned["prediction"])

    gt_events = _events(aligned["ground_truth"])
    pred_events = _events(aligned["prediction"])
    transition_rows: list[dict[str, object]] = []
    matched_predictions: set[int] = set()
    for gt_event in gt_events:
        candidates = [
            event for event in pred_events
            if event["to"] == gt_event["to"]
            and abs(int(event["frame"]) - int(gt_event["frame"])) <= transition_tolerance_frames
            and int(event["frame"]) not in matched_predictions
        ]
        if candidates:
            chosen = min(candidates, key=lambda event: abs(int(event["frame"]) - int(gt_event["frame"])))
            matched_predictions.add(int(chosen["frame"]))
            delta = int(chosen["frame"]) - int(gt_event["frame"])
            error_type = "correct_transition" if delta == 0 else ("late_transition" if delta > 0 else "early_transition")
            transition_rows.append({
                "ground_truth_frame": int(gt_event["frame"]),
                "predicted_frame": int(chosen["frame"]),
                "ground_truth_from": gt_event["from"],
                "ground_truth_to": gt_event["to"],
                "predicted_from": chosen["from"],
                "predicted_to": chosen["to"],
                "frame_delta": delta,
                "error_type": error_type,
                "match_clock_s": float((features.set_index("frame").loc[int(gt_event["frame"]), "match_clock_s"])
                                        if "match_clock_s" in features.columns and int(gt_event["frame"]) in set(features["frame"]) else np.nan),
            })
        else:
            transition_rows.append({
                "ground_truth_frame": int(gt_event["frame"]),
                "predicted_frame": np.nan,
                "ground_truth_from": gt_event["from"],
                "ground_truth_to": gt_event["to"],
                "predicted_from": np.nan,
                "predicted_to": np.nan,
                "frame_delta": np.nan,
                "error_type": "missed_transition",
                "match_clock_s": np.nan,
            })
    for event in pred_events:
        if int(event["frame"]) in matched_predictions:
            continue
        transition_rows.append({
            "ground_truth_frame": np.nan,
            "predicted_frame": int(event["frame"]),
            "ground_truth_from": np.nan,
            "ground_truth_to": np.nan,
            "predicted_from": event["from"],
            "predicted_to": event["to"],
            "frame_delta": np.nan,
            "error_type": "spurious_transition",
            "match_clock_s": np.nan,
        })

    frame_errors = aligned.loc[wrong].copy()
    frame_errors["frame"] = frame_errors.index.astype(int)
    frame_errors = frame_errors.reset_index(drop=True)
    frame_errors["error_type"] = np.select(
        [
            frame_errors["ground_truth"].eq("D") & frame_errors["prediction"].isin(["H", "A"]),
            frame_errors["prediction"].eq("D") & frame_errors["ground_truth"].isin(["H", "A"]),
            frame_errors["ground_truth"].isin(["H", "A"]) & frame_errors["prediction"].isin(["H", "A"]),
        ],
        ["missed_deadball", "overheld_deadball", "wrong_team"],
        default="state_or_transition_boundary_error",
    )
    frame_errors["match_clock_s"] = np.nan
    feature_columns = [
        "smart_context", "smart_restart_type", "smart_event_count", "scenario_context",
        "foul_stoppage_candidate", "foul_stoppage_confidence",
        "foul_stoppage_turnover_candidate", "foul_stoppage_convergence_candidate",
        "foul_stoppage_restart_candidate",
        "closest_team", "selected_anchor_team", "transition_anchor",
        "selected_anchor_confidence", "scenario_confidence", "scenario_margin",
        "controlled_reception_signal_score",
    ]
    feature_view = features.set_index("frame")
    available = [column for column in feature_columns if column in feature_view.columns]
    if available:
        frame_errors = frame_errors.join(feature_view[available], on="frame", how="left")
    if "match_clock_s" in feature_view.columns:
        frame_errors["match_clock_s"] = frame_errors["frame"].map(feature_view["match_clock_s"])
    frame_errors["match_time"] = pd.to_timedelta(frame_errors["match_clock_s"], unit="s").astype(str)
    transition_errors = pd.DataFrame(transition_rows).sort_values(
        ["ground_truth_frame", "predicted_frame"], na_position="last"
    )
    summary = pd.DataFrame(
        [{"error_type": key, "count": int(value)} for key, value in frame_errors["error_type"].value_counts().items()]
    )
    return {"frame_errors": frame_errors.reset_index(drop=True),
            "transition_errors": transition_errors.reset_index(drop=True),
            "summary": summary}
