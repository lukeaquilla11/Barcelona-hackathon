"""Adapter for the ``stage/hq`` three-state Viterbi baseline.

The friend's model is retained as a comparable decoder, not as a second data
pipeline.  It consumes the same canonical evidence table as every other model
and deliberately keeps the original clues and costs: nearest-team control,
ball visibility/speed, player activity, out-of-play evidence, and Smart
Tagging as a weak prior.  This makes model comparisons about *handling the
evidence*, rather than about different preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


H, A, D = 0, 1, 2

PARAMS = {
    "dist_scale": 2.0,
    "near": 4.0,
    "fast_ball": 6.0,
    "fast_weight": 0.3,
    "tag_weight": 0.03,
    "speed_dead": 1.3,
    "speed_scale": 0.35,
    "speed_weight": 0.6,
    "missing_dead": 0.15,
    "out_weight": 2.0,
    "switch_cost": 12.0,
    "dead_cost": 10.0,
}


def _values(features: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    return pd.to_numeric(
        features.get(name, pd.Series(default, index=features.index)), errors="coerce"
    ).to_numpy(float)


def canonical_stage_hq_inputs(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map canonical feature names to the friend's original clue names."""
    ball_x = _values(features, "ball_x_m")
    ball_y = _values(features, "ball_y_m")
    visible = np.isfinite(ball_x) & np.isfinite(ball_y)
    out = features.get(
        "ball_out_of_play_anchor",
        features.get("ball_out_of_play", pd.Series(False, index=features.index)),
    )
    return {
        "dH": _values(features, "nearest_h_distance_m"),
        "dA": _values(features, "nearest_a_distance_m"),
        "vis": visible,
        "bspeed": _values(features, "ball_speed_derived_mps", 0.0),
        "pspeed": _values(features, "player_activity_median_mps"),
        "out": out.fillna(False).to_numpy(bool),
    }


def _rolling_mean(values: np.ndarray, window: int = 25) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window, min_periods=1)
        .mean()
        .to_numpy(float)
    )


def closeness_clue(inputs: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    """Return positive evidence for H and negative evidence for A."""
    d_home, d_away = inputs["dH"], inputs["dA"]
    visible = inputs["vis"] & np.isfinite(d_home) & np.isfinite(d_away)
    difference = np.clip((d_away - d_home) / params["dist_scale"], -3, 3)
    nearest = np.fmin(d_home, d_away)
    near = np.exp(-np.nan_to_num(nearest, nan=100.0) / params["near"])
    speed = np.nan_to_num(inputs["bspeed"], nan=0.0)
    control_weight = np.where(speed > params["fast_ball"], params["fast_weight"], 1.0)
    return np.where(visible, difference * near * control_weight, 0.0)


def dead_ball_clue(inputs: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    """Reproduce stage/hq's calm/missing/out-of-play dead-ball evidence."""
    activity = _rolling_mean(inputs["pspeed"], 25)
    activity = np.where(np.isfinite(activity), activity, params["speed_dead"])
    clue = np.clip(
        (params["speed_dead"] - activity) / params["speed_scale"], -3, 3
    ) * params["speed_weight"]
    clue += np.where(inputs["vis"], 0.0, params["missing_dead"])
    clue += np.where(inputs["out"], params["out_weight"], 0.0)
    return clue


def frame_scores(features: pd.DataFrame, params: dict[str, float] | None = None) -> np.ndarray:
    """Build the friend's n-by-3 H/A/D evidence score matrix."""
    params = {**PARAMS, **(params or {})}
    inputs = canonical_stage_hq_inputs(features)
    control = closeness_clue(inputs, params)
    scores = np.zeros((len(features), 3), dtype=float)
    scores[:, H] = control / 2.0
    scores[:, A] = -control / 2.0
    scores[:, D] = dead_ball_clue(inputs, params)
    return scores


def viterbi(scores: np.ndarray, switch_cost: float, dead_cost: float) -> np.ndarray:
    """Decode the best H/A/D path with the original stage/hq costs."""
    if len(scores) == 0:
        return np.empty(0, dtype=np.int8)
    transitions = np.array(
        [[0, -switch_cost, -dead_cost],
         [-switch_cost, 0, -dead_cost],
         [-dead_cost, -dead_cost, 0]],
        dtype=float,
    )
    back = np.zeros((len(scores), 3), dtype=np.int8)
    path_score = scores[0] + np.array([-dead_cost, -dead_cost, 0.0])
    for frame in range(1, len(scores)):
        candidates = path_score[:, None] + transitions
        back[frame] = candidates.argmax(axis=0)
        path_score = candidates.max(axis=0) + scores[frame]
    path = np.empty(len(scores), dtype=np.int8)
    path[-1] = path_score.argmax()
    for frame in range(len(scores) - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


def stage_hq_labels(features: pd.DataFrame, params: dict[str, float] | None = None) -> np.ndarray:
    """Return labels for the canonical table using the stage/hq decoder."""
    params = {**PARAMS, **(params or {})}
    scores = frame_scores(features, params)
    path = viterbi(scores, params["switch_cost"], params["dead_cost"])
    return np.array(["H", "A", "D"], dtype="<U1")[path]


__all__ = [
    "PARAMS",
    "canonical_stage_hq_inputs",
    "frame_scores",
    "stage_hq_labels",
    "viterbi",
]
