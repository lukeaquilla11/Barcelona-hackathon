"""Evaluation-only frame and interval error audits for possession outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _mode_or_none(values: pd.Series) -> Any:
    values = values.dropna()
    if values.empty:
        return None
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else None


def _error_runs(errors: pd.DataFrame) -> pd.DataFrame:
    """Collapse contiguous identical error types into interval-level rows."""
    if errors.empty:
        return pd.DataFrame(
            columns=[
                "error_run_id", "start_frame", "end_frame", "duration_frames",
                "error_type", "gt_label", "pred_label",
            ]
        )

    rows: list[dict[str, Any]] = []
    start = 0
    previous_frame = int(errors.iloc[0]["frame"])
    previous_type = str(errors.iloc[0]["error_type"])
    for position in range(1, len(errors)):
        row = errors.iloc[position]
        frame = int(row["frame"])
        error_type = str(row["error_type"])
        contiguous = frame == previous_frame + 1
        if contiguous and error_type == previous_type:
            previous_frame = frame
            continue
        block = errors.iloc[start:position]
        rows.append(
            {
                "start_frame": int(block["frame"].iloc[0]),
                "end_frame": int(block["frame"].iloc[-1]),
                "duration_frames": int(len(block)),
                "error_type": previous_type,
                "gt_label": _mode_or_none(block["gt_label"]),
                "pred_label": _mode_or_none(block["pred_label"]),
            }
        )
        start = position
        previous_frame = frame
        previous_type = error_type

    block = errors.iloc[start:]
    rows.append(
        {
            "start_frame": int(block["frame"].iloc[0]),
            "end_frame": int(block["frame"].iloc[-1]),
            "duration_frames": int(len(block)),
            "error_type": previous_type,
            "gt_label": _mode_or_none(block["gt_label"]),
            "pred_label": _mode_or_none(block["pred_label"]),
        }
    )
    output = pd.DataFrame(rows)
    output.insert(0, "error_run_id", np.arange(len(output), dtype=int))
    return output


def _read_context(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["frame", *columns])
    table = pd.read_csv(path, low_memory=False)
    keep = [column for column in ["frame", *columns] if column in table.columns]
    return table[keep].drop_duplicates("frame")


def build_mislabel_audit(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    output_dir: str | Path,
    *,
    scenario_path: str | Path | None = None,
    probability_path: str | Path | None = None,
    missing_ball_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Write evaluation-only frame, run, and context mislabel reports.

    Ground truth is used only after the prediction already exists.  No output
    from this function is consumed by the detector or decoder.
    """
    ground_truth = pd.read_csv(ground_truth_path)
    prediction = pd.read_csv(prediction_path)
    required = {"frame", "label"}
    if not required.issubset(ground_truth.columns) or not required.issubset(prediction.columns):
        raise ValueError("ground truth and prediction must contain frame and label columns")
    if len(ground_truth) != len(prediction) or not ground_truth["frame"].equals(prediction["frame"]):
        raise ValueError("ground truth and prediction frame tables are not aligned")

    audit = pd.DataFrame(
        {
            "frame": ground_truth["frame"].to_numpy(),
            "gt_label": ground_truth["label"].astype(str).to_numpy(),
            "pred_label": prediction["label"].astype(str).to_numpy(),
        }
    )
    audit["graded"] = audit["gt_label"].ne("X")
    audit["correct"] = audit["gt_label"].eq(audit["pred_label"])
    audit["error_type"] = audit["gt_label"] + "_TO_" + audit["pred_label"]
    audit["is_mislabel"] = audit["graded"] & ~audit["correct"]

    context_specs = [
        (scenario_path, [
            "scenario_context", "scenario_context_causal", "scenario_state",
            "scenario_confidence", "scenario_margin", "state_score_DEAD_BALL",
            "state_score_H_CONTROL", "state_score_A_CONTROL", "state_score_CONTESTED",
            "transition_hypothesis_state", "transition_trust_score",
            "transition_trust_p_no_switch", "transition_trust_candidate",
        ]),
        (probability_path, [
            "scenario_context", "scenario_context_causal", "scenario_state",
            "transition_hypothesis_state",
            "observation_probability_H", "observation_probability_A",
            "observation_probability_D", "observation_probability_X",
            "observation_probability_entropy", "observation_probability_margin",
            "missing_ball_cause", "missing_ball_cause_confidence", "scenario_crowd_score",
        ]),
        (missing_ball_path, [
            "ball_kalman_path_available", "ball_kalman_reliable",
            "ball_kalman_position_uncertainty_m", "ball_gap_sharp_acceleration_risk",
            "gap_same_player_dribble_candidate", "gap_tackle_risk_candidate",
        ]),
    ]
    for path, columns in context_specs:
        if path is not None:
            context = _read_context(Path(path), columns)
            if len(context.columns) > 1:
                audit = audit.merge(context, on="frame", how="left")

    errors = audit[audit["is_mislabel"]].copy()
    runs = _error_runs(errors)

    if errors.empty:
        summary = pd.DataFrame(columns=["error_type", "frames", "runs", "mean_run_frames", "max_run_frames"])
        context_summary = pd.DataFrame()
    else:
        summary = (
            errors.groupby("error_type", as_index=False)
            .agg(frames=("frame", "size"), gt_label=("gt_label", "first"), pred_label=("pred_label", "first"))
            .merge(
                runs.groupby("error_type", as_index=False).agg(
                    runs=("error_run_id", "size"),
                    mean_run_frames=("duration_frames", "mean"),
                    max_run_frames=("duration_frames", "max"),
                ),
                on="error_type",
            )
            .sort_values(["frames", "error_type"], ascending=[False, True])
        )
        # Prefer the causal context when a newer pipeline emits it, but fall
        # back to the frame-level scenario context used by the current
        # probability audit.  Episode-level reports do not have a frame key
        # and are intentionally ignored by _read_context.
        context_column = next(
            (
                column
                for column in (
                    "scenario_context_causal",
                    "scenario_context",
                    "scenario_state",
                    "transition_hypothesis_state",
                )
                if column in errors
            ),
            None,
        )
        if context_column is not None:
            group_columns = ["error_type", context_column]
            aggregations = {"frames": ("frame", "size")}
            if "observation_probability_entropy" in errors:
                aggregations["mean_entropy"] = (
                    "observation_probability_entropy",
                    "mean",
                )
            context_summary = (
                errors.groupby(group_columns, dropna=False, as_index=False)
                .agg(**aggregations)
                .rename(columns={context_column: "context_value"})
                .assign(context_dimension=context_column)
                .sort_values("frames", ascending=False)
            )
        else:
            context_summary = pd.DataFrame()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "visible_ball_mislabel_frames.csv", index=False)
    errors.to_csv(output / "visible_ball_mislabel_errors.csv", index=False)
    runs.to_csv(output / "visible_ball_mislabel_runs.csv", index=False)
    summary.to_csv(output / "visible_ball_mislabel_summary.csv", index=False)
    context_summary.to_csv(output / "visible_ball_mislabel_context_summary.csv", index=False)
    return {
        "audit": audit,
        "errors": errors,
        "runs": runs,
        "summary": summary,
        "context_summary": context_summary,
    }
