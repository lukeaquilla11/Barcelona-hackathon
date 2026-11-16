"""Independent adapter for the raw-observation Haiqal possession model.

This module deliberately does *not* call ``build_engineered_features``.  It
ports the method on ``origin/haiqal`` into the current package boundary so it
can be compared fairly with the scenario-aware models:

    raw tracking -> distance/speed/dead-ball clues -> Viterbi -> H/A/D/X

The model has no access to our scenario labels, anchors, transition ledger,
probability emissions, Kalman features, or public ground truth.  Keeping that
boundary explicit is important: this is an independent baseline, not a hidden
ensemble with the current pipeline.
"""

from __future__ import annotations

import os
import warnings
from typing import Mapping

import numpy as np

from games import GAME, GAMES, asset_paths, half_video_blocks
from possession_lab.baseline_pipeline import load_raw_atd_compact, slot_team_map
from starter.load_smart_tagging import load as load_smart_tagging


H, A, D = 0, 1, 2

PARAMS: dict[str, float] = {
    # Raw control clues from the Haiqal branch.
    "dist_scale": 2.0,
    "near": 4.0,
    "fast_ball": 6.0,
    "fast_weight": 0.3,
    "tag_weight": 0.03,
    # Raw dead-ball clues.
    "speed_dead": 1.3,
    "speed_scale": 0.35,
    "speed_weight": 0.6,
    "missing_dead": 0.15,
    "out_weight": 2.0,
    # Viterbi transition penalties.
    "switch_cost": 12.0,
    "dead_cost": 10.0,
}


def rolling_mean(values: np.ndarray, half: int) -> np.ndarray:
    """Centred NaN-ignoring rolling mean, matching the Haiqal implementation."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    safe = np.where(valid, values, 0.0)
    cumulative = np.r_[0.0, np.cumsum(safe)]
    counts = np.r_[0, np.cumsum(valid.astype(np.int64))]
    index = np.arange(len(values))
    low = np.maximum(0, index - half)
    high = np.minimum(len(values), index + half + 1)
    count = counts[high] - counts[low]
    return np.where(
        count > 0,
        (cumulative[high] - cumulative[low]) / np.maximum(count, 1),
        np.nan,
    )


def tag_prior(n_frames: int, *, game: str = GAME, data_dir: str | os.PathLike[str] | None = None) -> np.ndarray:
    """Return Haiqal's weak Smart Tagging prior: +1 home, -1 away, 0 unknown."""
    attack_codes = {
        "build up",
        "progression",
        "attacking transition",
        "creating chances",
        "long ball",
        "ball in final third",
        "players in the box",
    }
    defend_codes = {
        "defending in defensive third",
        "defending in middle third",
        "defending in attacking third",
    }
    prior = np.zeros(n_frames, dtype=float)
    for event in load_smart_tagging(game=game, data_dir=data_dir):
        code = (event.get("code") or "").strip().lower()
        team_name = str(event.get("tags", {}).get("Team", "")).strip().lower()
        sign = 1 if team_name.startswith("barc") else -1 if team_name.startswith("atl") else 0
        if not sign:
            continue
        if code in defend_codes:
            sign = -sign
        elif code not in attack_codes:
            continue
        start = max(0, int(event["start_frame"]))
        end = min(n_frames, int(event["end_frame"]))
        if start < end:
            prior[start:end] += sign
    return np.sign(prior)


def features(
    *,
    game: str = GAME,
    data_dir: str | os.PathLike[str] | None = None,
) -> dict[str, np.ndarray]:
    """Build only the raw signals consumed by the Haiqal model.

    The ATD stores x/y in pitch-normalised coordinates.  Haiqal's original
    loader converted those coordinates to metres before calculating speed and
    distance; this adapter does the same and keeps the exact raw missingness.
    """
    raw = load_raw_atd_compact(game=game, data_dir=data_dir, source="tactical")
    pitch_length, pitch_width = GAMES[game]["pitch"]
    slots = sorted(
        int(column[:-2])
        for column in raw.columns
        if column.endswith("_x") and column[:-2].isdigit()
    )
    player_x = np.column_stack([raw[f"{slot}_x"].to_numpy(float) for slot in slots]) * pitch_length
    player_y = np.column_stack([raw[f"{slot}_y"].to_numpy(float) for slot in slots]) * pitch_width
    ball_x = raw["ball_x"].to_numpy(float) * pitch_length
    ball_y = raw["ball_y"].to_numpy(float) * pitch_width

    paths = slot_team_map(asset_paths(game=game, data_dir=data_dir, source="tactical")["atd_meta"])
    team_by_slot = np.array([paths[slot] for slot in slots], dtype=object)

    dx = player_x - ball_x[:, None]
    dy = player_y - ball_y[:, None]
    distance = np.hypot(dx, dy)
    safe_distance = np.where(np.isfinite(distance), distance, np.inf)
    d_home = np.min(np.where(team_by_slot[None, :] == "H", safe_distance, np.inf), axis=1)
    d_away = np.min(np.where(team_by_slot[None, :] == "A", safe_distance, np.inf), axis=1)
    d_home[~np.isfinite(d_home)] = np.nan
    d_away[~np.isfinite(d_away)] = np.nan

    player_speed = np.full(player_x.shape, np.nan, dtype=float)
    player_speed[1:] = np.hypot(np.diff(player_x, axis=0), np.diff(player_y, axis=0)) * GAMES[game]["fps"]
    player_speed[player_speed > 12.0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_player_speed = np.nanmedian(player_speed, axis=1)

    ball_speed = np.full(len(ball_x), np.nan, dtype=float)
    ball_speed[1:] = np.hypot(np.diff(ball_x), np.diff(ball_y)) * GAMES[game]["fps"]
    visible = np.isfinite(ball_x) & np.isfinite(ball_y)
    outside = (
        (ball_x < -0.5)
        | (ball_x > pitch_length + 0.5)
        | (ball_y < -0.5)
        | (ball_y > pitch_width + 0.5)
    )
    return {
        "vis": visible,
        "bx": ball_x,
        "by": ball_y,
        "dH": d_home,
        "dA": d_away,
        "pspeed": median_player_speed,
        "nplayers": np.isfinite(player_x).sum(axis=1),
        "bspeed": ball_speed,
        "out": outside,
    }


def closeness_clue(F: Mapping[str, np.ndarray], params: Mapping[str, float], *, prior: np.ndarray | None = None) -> np.ndarray:
    """Positive favours home; negative favours away when the ball is controllable."""
    d_home, d_away = F["dH"], F["dA"]
    visible = F["vis"] & np.isfinite(d_home) & np.isfinite(d_away)
    difference = np.clip((d_away - d_home) / params["dist_scale"], -3, 3)
    near = np.exp(-np.fmin(d_home, d_away) / params["near"])
    ball_speed = np.nan_to_num(F["bspeed"], nan=0.0)
    control = np.where(ball_speed > params["fast_ball"], params["fast_weight"], 1.0)
    score = np.where(visible, difference * near * control, 0.0)
    if params["tag_weight"] and prior is not None:
        score = score + params["tag_weight"] * prior
    return score


def dead_ball_clue(F: Mapping[str, np.ndarray], params: Mapping[str, float]) -> np.ndarray:
    """Score visible inactivity, missing ball, and literal out-of-pitch evidence."""
    player_speed = rolling_mean(F["pspeed"], 25)
    speed = np.where(np.isfinite(player_speed), player_speed, params["speed_dead"])
    score = np.clip(
        (params["speed_dead"] - speed) / params["speed_scale"], -3, 3
    ) * params["speed_weight"]
    score += np.where(F["vis"], 0.0, params["missing_dead"])
    score += np.where(F["out"], params["out_weight"], 0.0)
    return score


def add_evidence_features(
    frame_table,
    *,
    game: str = GAME,
    data_dir: str | os.PathLike[str] | None = None,
):
    """Attach Haiqal's raw clues to an existing canonical evidence table.

    This is the safe integration boundary: Haiqal contributes observations,
    while the current scenario, anchor, transition-ledger, and decoder layers
    remain responsible for ownership.  It never copies Haiqal's final labels.
    """
    out = frame_table.copy()
    raw = features(game=game, data_dir=data_dir)
    expected = len(out)
    if len(raw["vis"]) != expected:
        raise ValueError(f"Haiqal evidence has {len(raw['vis'])} frames; expected {expected}")
    prior = tag_prior(expected, game=game, data_dir=data_dir)
    params = PARAMS
    out["haiqal_distance_home_m"] = raw["dH"]
    out["haiqal_distance_away_m"] = raw["dA"]
    out["haiqal_distance_margin_m"] = raw["dA"] - raw["dH"]
    out["haiqal_ball_speed_mps"] = raw["bspeed"]
    out["haiqal_player_speed_mps"] = raw["pspeed"]
    out["haiqal_visible"] = raw["vis"]
    out["haiqal_out_of_pitch"] = raw["out"]
    out["haiqal_closeness_score"] = closeness_clue(raw, params, prior=prior)
    out["haiqal_dead_ball_score"] = dead_ball_clue(raw, params)
    out["haiqal_tag_prior"] = prior
    out["haiqal_evidence_source"] = "raw_distance_speed_deadball"
    return out


def frame_scores(
    F: Mapping[str, np.ndarray],
    params: Mapping[str, float] | None = None,
    *,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Return an ``n x 3`` raw evidence score for H, A, and D."""
    params = PARAMS if params is None else params
    closeness = closeness_clue(F, params, prior=prior)
    scores = np.zeros((len(closeness), 3), dtype=float)
    scores[:, H] += closeness / 2
    scores[:, A] -= closeness / 2
    scores[:, D] += dead_ball_clue(F, params)
    return scores


def viterbi(emissions: np.ndarray, switch_cost: float, dead_cost: float) -> np.ndarray:
    """Decode the best H/A/D path using Haiqal's three-state Viterbi costs."""
    if len(emissions) == 0:
        return np.empty(0, dtype=np.int8)
    transition = np.array(
        [[0.0, -switch_cost, -dead_cost],
         [-switch_cost, 0.0, -dead_cost],
         [-dead_cost, -dead_cost, 0.0]],
    )
    back = np.zeros((len(emissions), 3), dtype=np.int8)
    score = emissions[0] + np.array([-dead_cost, -dead_cost, 0.0])
    for frame in range(1, len(emissions)):
        candidates = score[:, None] + transition
        back[frame] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + emissions[frame]
    path = np.empty(len(emissions), dtype=np.int8)
    path[-1] = score.argmax()
    for frame in range(len(emissions) - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


def predict(
    F: Mapping[str, np.ndarray] | None = None,
    *,
    game: str = GAME,
    data_dir: str | os.PathLike[str] | None = None,
    **overrides: float,
) -> np.ndarray:
    """Return Haiqal labels for every tactical frame, without scenario inputs."""
    params = dict(PARAMS, **overrides)
    F = features(game=game, data_dir=data_dir) if F is None else F
    prior = tag_prior(len(F["vis"]), game=game, data_dir=data_dir) if params["tag_weight"] else None
    emissions = frame_scores(F, params, prior=prior)
    labels = np.full(len(emissions), "X", dtype="<U1")
    state_labels = np.array(["H", "A", "D"])
    for _, first, last in half_video_blocks(game):
        labels[first : last + 1] = state_labels[
            viterbi(emissions[first : last + 1], params["switch_cost"], params["dead_cost"])
        ]
    return labels


def write(path: str | os.PathLike[str], labels: np.ndarray) -> str:
    """Write the standard ``frame,label`` submission file."""
    import pandas as pd

    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)
    return path


def counts(labels: np.ndarray) -> dict[str, int]:
    """Return a small whole-match sanity report, matching Haiqal's helper."""
    graded = labels[labels != "X"]
    starts = np.r_[True, graded[1:] != graded[:-1]] if len(graded) else np.array([], dtype=bool)
    spells = graded[starts]
    return {
        "possessions": int(np.isin(spells, ["H", "A"]).sum()),
        "stoppages": int((spells == "D").sum()),
    }

