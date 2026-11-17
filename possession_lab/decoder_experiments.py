"""Evidence-preserving temporal decoders for the possession baseline.

These decoders do not use the public labels for fitting.  They combine the
nearest-player observation at multiple time scales and expose the decision
signals so each experiment can be audited.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from starter.baseline import AWAY, HOME, NOBODY, smooth
from possession_lab.baseline_pipeline import graded_mask


def nearest_codes(features: pd.DataFrame) -> np.ndarray:
    """Convert the supplied nearest-team observation to starter integer codes."""
    team = features["closest_team"].to_numpy(dtype=object)
    return np.where(team == "H", HOME, np.where(team == "A", AWAY, NOBODY)).astype(np.int8)


def _run_length_filter(labels: np.ndarray, minimum_run: int) -> np.ndarray:
    """Remove isolated team spikes while preserving the surrounding label."""
    out = labels.astype(object, copy=True)
    if minimum_run <= 1:
        return out
    start = 0
    while start < len(out):
        end = start + 1
        while end < len(out) and out[end] == out[start]:
            end += 1
        if out[start] in {"H", "A"} and end - start < minimum_run:
            before = out[start - 1] if start else None
            after = out[end] if end < len(out) else None
            replacement = before if before in {"H", "A"} else after
            if replacement in {"H", "A"}:
                out[start:end] = replacement
        start = end
    return out


def _merge_internal_spikes(labels: np.ndarray, maximum_gap: int) -> np.ndarray:
    """Replace a short run when the same label exists immediately on both sides."""
    out = labels.astype(object, copy=True)
    if maximum_gap <= 0:
        return out
    start = 0
    while start < len(out):
        end = start + 1
        while end < len(out) and out[end] == out[start]:
            end += 1
        before = out[start - 1] if start else None
        after = out[end] if end < len(out) else None
        if end - start <= maximum_gap and before in {"H", "A", "D"} and before == after:
            out[start:end] = before
        start = end
    return out


def _remove_short_dead_runs(labels: np.ndarray, minimum_run: int) -> np.ndarray:
    """Replace isolated short D runs with neighbouring H/A evidence."""
    out = labels.astype(object, copy=True)
    if minimum_run <= 1:
        return out
    start = 0
    while start < len(out):
        end = start + 1
        while end < len(out) and out[end] == out[start]:
            end += 1
        if out[start] == "D" and end - start < minimum_run:
            before = out[start - 1] if start else None
            after = out[end] if end < len(out) else None
            if before in {"H", "A"} and before == after:
                out[start:end] = before
            elif before in {"H", "A"}:
                out[start:end] = before
            elif after in {"H", "A"}:
                out[start:end] = after
        start = end
    return out


def dead_ball_boundary_evidence(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    low_activity_median_mps: float = 1.5,
    low_activity_p90_mps: float = 2.0,
    minimum_low_motion_frames: int = 25,
    maximum_out_of_bounds_bridge_frames: int = 75,
) -> np.ndarray:
    """Add two physically interpretable dead-ball boundary signals.

    The first signal uses an explicitly out-of-bounds ball coordinate only to
    bridge to a nearby dead-ball episode already supported by the decoder. It
    therefore cannot turn an arbitrary live interval into ``D``.

    The second signal detects a sustained whole-pitch freeze while the ball is
    missing. A short low-motion run is ignored; a qualifying run is expanded
    only across the same contiguous missing-ball episode. This is useful for
    stoppages that Smart Tagging does not label, while avoiding the assumption
    that every missing-ball frame is dead.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)

    ball_x = pd.to_numeric(features.get("ball_x", pd.Series(np.nan, index=features.index)), errors="coerce")
    ball_y = pd.to_numeric(features.get("ball_y", pd.Series(np.nan, index=features.index)), errors="coerce")
    out_of_bounds = ((ball_x < 0.0) | (ball_x > 1.0) | (ball_y < 0.0) | (ball_y > 1.0)).to_numpy(bool)
    out_of_bounds = out_of_bounds & in_play

    # Bridge only from an observed out-of-bounds boundary to a nearby D run.
    # Existing D evidence remains the anchor; this operation only repairs its
    # leading edge and short internal gaps.
    start = 0
    while start < n_frames:
        if not out_of_bounds[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and out_of_bounds[end]:
            end += 1
        search = end
        while search < min(n_frames, end + maximum_out_of_bounds_bridge_frames):
            if output[search] == "D":
                bridge_end = search
                cursor = search
                while cursor < n_frames:
                    if output[cursor] == "D":
                        bridge_end = cursor + 1
                        cursor += 1
                        continue
                    gap_end = cursor
                    while gap_end < n_frames and output[gap_end] != "D":
                        gap_end += 1
                    if gap_end >= n_frames or gap_end - cursor > maximum_out_of_bounds_bridge_frames:
                        break
                    bridge_end = gap_end + 1
                    cursor = gap_end
                output[start:bridge_end] = "D"
                break
            search += 1
        start = end

    missing_ball = ball_x.isna() | ball_y.isna()
    median_activity = pd.to_numeric(
        features.get("player_activity_median_mps", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    )
    p90_activity = pd.to_numeric(
        features.get("player_activity_p90_mps", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    )
    low_motion_missing = (
        in_play
        & missing_ball.to_numpy(bool)
        & (median_activity.to_numpy(float) <= low_activity_median_mps)
        & (p90_activity.to_numpy(float) <= low_activity_p90_mps)
    )

    start = 0
    while start < n_frames:
        if not low_motion_missing[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and low_motion_missing[end]:
            end += 1
        if end - start >= minimum_low_motion_frames:
            episode_start = start
            while episode_start > 0 and missing_ball.iloc[episode_start - 1] and in_play[episode_start - 1]:
                episode_start -= 1
            episode_end = end
            while episode_end < n_frames and missing_ball.iloc[episode_end] and in_play[episode_end]:
                episode_end += 1
            output[episode_start:episode_end] = "D"
        start = end

    output[~in_play] = "X"
    return output


def low_motion_dead_ball_evidence(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    activity_window_frames: int = 50,
    minimum_support_fraction: float = 0.80,
    minimum_candidate_run_frames: int = 20,
    maximum_ball_speed_mps: float = 6.0,
    maximum_settling_speed_mps: float = 4.0,
    settling_acceleration_mps2: float = -4.0,
    maximum_restart_bridge_frames: int = 150,
    maximum_candidate_bridge_frames: int = 100,
    minimum_bridge_low_activity_fraction: float = 0.80,
) -> np.ndarray:
    """Add a conservative visible-ball low-motion stoppage hypothesis.

    This is deliberately separate from ``dead_ball_boundary_evidence``.  The
    existing rule is anchored by Smart Tagging, missing-ball settling, or an
    observed out-of-bounds coordinate.  This rule handles the harder case in
    which the ball remains visible but the whole tracked scene settles without
    a restart tag.  It is evidence for a D interval, not proof of a restart:
    slow player motion alone is insufficient, so the ball must also be slow or
    decelerating for most of a trailing window.

    The output is an experiment layer.  It never writes labels outside the
    grading mask and never changes X.  A future promotion should require a
    held-out spell audit because live possession can also contain walking and
    a slow ball.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    if n_frames == 0:
        return output

    median_activity = pd.to_numeric(
        features.get("player_activity_median_mps", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    p90_activity = pd.to_numeric(
        features.get("player_activity_p90_mps", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    speed = pd.to_numeric(
        features.get("ball_speed_derived_mps", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    acceleration = pd.to_numeric(
        features.get("ball_acceleration_mps2", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    missing_ball = (
        pd.to_numeric(features.get("ball_x", pd.Series(np.nan, index=features.index)), errors="coerce").isna()
        | pd.to_numeric(features.get("ball_y", pd.Series(np.nan, index=features.index)), errors="coerce").isna()
    ).to_numpy(bool)
    restart_type = features.get("restart_type", pd.Series(None, index=features.index)).to_numpy(object)
    kickoff = np.isin(restart_type, ["Kick Off (Start)", "Kick Off (After Goal)"])

    low_activity = (
        np.isfinite(median_activity)
        & np.isfinite(p90_activity)
        & (median_activity <= 1.5)
        & (p90_activity <= 2.0)
    )
    visible_ball_support = (
        (np.isfinite(speed) & (speed <= float(maximum_ball_speed_mps)))
        | (np.isfinite(speed) & (speed <= float(maximum_settling_speed_mps)))
        | (np.isfinite(acceleration) & (acceleration <= float(settling_acceleration_mps2)))
    )
    support = (low_activity & visible_ball_support).astype(float)
    window = max(1, int(activity_window_frames))
    sustained = (
        pd.Series(support, index=features.index)
        .rolling(window, min_periods=window)
        .mean()
        .to_numpy(float)
        >= float(minimum_support_fraction)
    )
    candidate = sustained & in_play & ~missing_ball & ~kickoff

    # Avoid one-frame D insertions.  The run filter is causal and is an
    # evidence gate, not a claim that the exact stoppage boundary is known.
    minimum_run = max(1, int(minimum_candidate_run_frames))
    accepted_candidate = np.zeros(n_frames, dtype=bool)
    start = 0
    while start < n_frames:
        if not candidate[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and candidate[end]:
            end += 1
        if end - start >= minimum_run:
            output[start:end] = "D"
            accepted_candidate[start:end] = True
        start = end

    # If a settling candidate appears inside the same tagged restart episode
    # shortly after an existing D anchor, treat the intervening H/A island as
    # an unresolved boundary rather than a new live possession.  The bridge is
    # bounded, requires continuous restart context, and is cancelled by an
    # explicit restart-end signal.  This is the interval-level alternative to
    # overwriting each nearest-player observation independently.
    restart_context = features.get(
        "restart_context", features.get("smart_set_piece", pd.Series(False, index=features.index))
    ).to_numpy(bool)
    restart_end = features.get(
        "restart_end_candidate", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    bridge_limit = max(1, int(maximum_restart_bridge_frames))
    starts = np.flatnonzero(accepted_candidate & ~np.r_[False, accepted_candidate[:-1]])
    accepted_runs: list[tuple[int, int]] = []
    for candidate_start in starts:
        candidate_end = candidate_start + 1
        while candidate_end < n_frames and accepted_candidate[candidate_end]:
            candidate_end += 1
        accepted_runs.append((int(candidate_start), int(candidate_end)))
        previous_d = candidate_start - 1
        while previous_d >= 0 and output[previous_d] != "D":
            previous_d -= 1
        if previous_d < 0 or candidate_start - previous_d > bridge_limit:
            continue
        interval = slice(previous_d, candidate_end)
        if restart_context[interval].all() and not restart_end[interval].any():
            output[previous_d:candidate_end] = "D"

    # A pair of visible-ball settling candidates can also be separated by a
    # short, high-speed/occluded-looking burst.  Keep that burst as D only when
    # the surrounding scene remains settled and no confident team anchor can
    # explain it as live possession.  This is an explicit ``no-switch / no-
    # ownership`` alternative, not an inference that a ball path itself means
    # dead ball.
    anchor_team = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)
    anchor_confidence = pd.to_numeric(
        features.get("anchor_confidence_score", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    confident_anchor = np.isin(anchor_team, ["H", "A"]) & (anchor_confidence >= 0.85)
    bridge_limit = max(1, int(maximum_candidate_bridge_frames))
    for (_first_start, first_end), (next_start, _next_end) in zip(
        accepted_runs, accepted_runs[1:]
    ):
        gap_end = next_start
        if next_start - first_end <= 0 or next_start - first_end > bridge_limit:
            continue
        gap = slice(first_end, gap_end)
        if (
            low_activity[gap].mean() >= float(minimum_bridge_low_activity_fraction)
            and not confident_anchor[gap].any()
        ):
            output[first_end:gap_end] = "D"

    output[~in_play] = "X"
    return output.astype("<U1")


def literal_out_of_bounds_dead_ball_evidence(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    minimum_run_frames: int = 2,
    boundary_tolerance: float = 0.0,
) -> np.ndarray:
    """Mark sustained literal out-of-pitch ball coordinates as ``D``.

    ATD positions use normalized pitch coordinates.  A coordinate outside the
    pitch is strong dead-ball evidence, but a one-frame tracker spike should
    not create a one-frame ``D`` spell.  This function is therefore an
    auditable diagnostic layer: it marks only contiguous out-of-bounds runs of
    the configured minimum length and always protects the grading mask.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    ball_x = pd.to_numeric(
        features.get("ball_x", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    ball_y = pd.to_numeric(
        features.get("ball_y", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    tolerance = max(0.0, float(boundary_tolerance))
    out_of_bounds = (
        np.isfinite(ball_x)
        & np.isfinite(ball_y)
        & ((ball_x < -tolerance) | (ball_x > 1.0 + tolerance)
           | (ball_y < -tolerance) | (ball_y > 1.0 + tolerance))
        & in_play
    )
    minimum_run = max(1, int(minimum_run_frames))
    start = 0
    while start < n_frames:
        if not out_of_bounds[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and out_of_bounds[end]:
            end += 1
        if end - start >= minimum_run:
            output[start:end] = "D"
        start = end
    output[~in_play] = "X"
    return output


def add_missing_ball_cause_features(features: pd.DataFrame) -> pd.DataFrame:
    """Score likely causes of a missing-ball interval without assigning a team.

    Missing coordinates are an observation-quality problem, not a possession
    state.  This layer keeps the possible causes separate: dead/out of play,
    flight occlusion, control continuity, and deflection/contest.  Scores are
    evidence for later transition logic; the selected cause is not a label.
    """
    out = features.copy()
    index = out.index
    n = len(out)

    def numeric(name: str, default: float = np.nan) -> np.ndarray:
        return pd.to_numeric(
            out.get(name, pd.Series(default, index=index)), errors="coerce"
        ).to_numpy(float)

    def boolean(name: str) -> np.ndarray:
        return out.get(name, pd.Series(False, index=index)).fillna(False).to_numpy(bool)

    ball_x = numeric("ball_x")
    ball_y = numeric("ball_y")
    missing = ~np.isfinite(ball_x) | ~np.isfinite(ball_y)
    out_of_bounds = (
        np.isfinite(ball_x)
        & np.isfinite(ball_y)
        & ((ball_x < 0.0) | (ball_x > 1.0) | (ball_y < 0.0) | (ball_y > 1.0))
    )
    smart_set_piece = boolean("smart_set_piece")
    # A restart tag is context, not proof of dead ball. In particular, an
    # active goal kick/corner can have missing coordinates while play is live.
    restart_context = boolean("restart_context")
    low_motion = (
        (numeric("player_activity_median_mps", np.inf) <= 1.5)
        & (numeric("player_activity_p90_mps", np.inf) <= 2.0)
    )
    boundary_anchor = boolean("ball_out_of_play_anchor") | out_of_bounds

    # Signals that survive a short occlusion are treated as interval evidence
    # rather than frame-local truth.  Endpoint speeds are included because a
    # missing segment can hide the kick that launched the flight.
    speed = numeric("ball_speed_derived_mps", 0.0)
    flight_signal = (
        boolean("recent_flight_evidence")
        | boolean("ball_flight_candidate_interpolated")
        | boolean("long_ball_evidence")
        | (np.nan_to_num(speed, nan=0.0) >= 8.0)
    )
    deflection_signal = (
        boolean("gap_tackle_risk_candidate")
        | boolean("ball_gap_sharp_acceleration_risk")
        | boolean("deflection_direction_change_veto")
        | boolean("post_long_ball_direction_change_risk")
    )
    opponent_density = numeric("candidate_opponent_density_5m", 0.0)
    teammate_support = numeric("candidate_teammate_support_10m", 0.0)
    nearest_h = numeric("nearest_h_distance_m")
    nearest_a = numeric("nearest_a_distance_m")
    scenario_crowd_score = np.clip(
        np.nan_to_num(numeric("scenario_crowd_score", 0.0), nan=0.0),
        0.0,
        1.0,
    )
    opponent_density_score = np.clip(
        (np.nan_to_num(opponent_density, nan=0.0) - 1.0) / 3.0,
        0.0,
        1.0,
    )
    both_teams_near = (nearest_h <= 4.0) & (nearest_a <= 4.0)
    crowd_signal_score = np.maximum(
        scenario_crowd_score,
        np.maximum(opponent_density_score, np.where(both_teams_near, 1.0, 0.0)),
    )
    valid_tracking = boolean("valid")
    reprojection_error = numeric("reprojection_error")
    visibility_signal = (~valid_tracking) | (np.isfinite(reprojection_error) & (reprojection_error > 0.75))
    # George is observation-quality evidence only. An alternate-camera
    # observation can explain a tactical gap and refine its likely cause, but
    # it does not select H/A ownership by itself.
    george_gap_observed = boolean("ball_fill_observed_during_tactical_gap")
    george_flight = george_gap_observed & boolean("ball_fill_flight_observation")
    george_dead = np.where(
        george_gap_observed,
        np.clip(np.nan_to_num(numeric("ball_fill_dead_probability", 0.0), nan=0.0), 0.0, 1.0),
        0.0,
    )
    visibility_signal = visibility_signal | george_gap_observed
    control_signal = (
        boolean("gap_same_player_dribble_candidate")
        | (
            boolean("gap_player_path_supported")
            & ~deflection_signal
            & ~boolean("ball_gap_sharp_acceleration_risk")
        )
    )

    # Expand endpoint/anchor evidence across each internal missing run.  This
    # is deliberately bounded to the observed run; it does not invent a path
    # for leading/trailing missing data.
    interval_signals = {
        "dead_or_out": np.maximum(
            (boundary_anchor | (smart_set_piece & low_motion)).astype(float),
            george_dead,
        ),
        "flight": np.maximum(flight_signal.astype(float), george_flight.astype(float)),
        "deflection": deflection_signal.astype(float),
        "control": control_signal.astype(float),
        "crowd": crowd_signal_score,
        "visibility": visibility_signal.astype(float),
    }
    expanded = {name: values.copy() for name, values in interval_signals.items()}
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])

    # A missing run is more informative when the two visible endpoints form
    # a coherent path. This is an observation-quality signal only: it does
    # not infer possession from the public labels. A coherent path supports
    # flight/control continuity; an incoherent path remains unresolved.
    gap_trajectory_continuity = np.zeros(n, dtype=float)
    gap_same_team_reappearance = np.zeros(n, dtype=float)
    gap_opponent_reappearance = np.zeros(n, dtype=float)
    anchor_values = out.get(
        "anchor_team", pd.Series(None, index=index, dtype=object)
    ).to_numpy(object)
    for start, end in zip(starts, ends):
        before, after = start - 1, end + 1
        if before < 1 or after + 1 >= n:
            continue
        left_dx = ball_x[before] - ball_x[before - 1]
        left_dy = ball_y[before] - ball_y[before - 1]
        right_dx = ball_x[after + 1] - ball_x[after]
        right_dy = ball_y[after + 1] - ball_y[after]
        left_norm = float(np.hypot(left_dx, left_dy))
        right_norm = float(np.hypot(right_dx, right_dy))
        finite_vectors = np.all(np.isfinite([left_dx, left_dy, right_dx, right_dy]))
        if finite_vectors and left_norm > 1e-6 and right_norm > 1e-6:
            cosine = float((left_dx * right_dx + left_dy * right_dy) / (left_norm * right_norm))
            speed_ratio = min(left_norm, right_norm) / max(left_norm, right_norm)
            continuity = np.clip(0.5 * (cosine + 1.0), 0.0, 1.0) * np.clip(speed_ratio, 0.0, 1.0)
        else:
            continuity = 0.0
        if continuity <= 0.0:
            continue
        gap_trajectory_continuity[start : end + 1] = continuity
        prior = anchor_values[before] if anchor_values[before] in {"H", "A"} else None
        future = anchor_values[after] if anchor_values[after] in {"H", "A"} else None
        if prior is not None and future is not None:
            if prior == future:
                gap_same_team_reappearance[start : end + 1] = continuity
            else:
                gap_opponent_reappearance[start : end + 1] = continuity
    for start, end in zip(starts, ends):
        before, after = start - 1, end + 1
        if before < 0 and after >= n:
            continue
        left = max(0, before)
        right = min(n - 1, after)
        for name, values in interval_signals.items():
            evidence = float(np.nanmax(values[left : right + 1]))
            expanded[name][start : end + 1] = evidence

    dead_score = np.clip(expanded["dead_or_out"], 0.0, 1.0)
    dead_score = np.maximum(dead_score, np.where(smart_set_piece & low_motion, 0.8, 0.0))
    flight_score = np.clip(expanded["flight"], 0.0, 1.0)
    deflection_score = np.clip(expanded["deflection"], 0.0, 1.0)
    control_score = np.clip(expanded["control"], 0.0, 1.0)
    crowd_score = np.clip(expanded["crowd"], 0.0, 1.0)
    visibility_score = np.clip(expanded["visibility"], 0.0, 1.0)
    continuity_score = np.clip(gap_trajectory_continuity, 0.0, 1.0)
    same_team_score = np.clip(gap_same_team_reappearance, 0.0, 1.0)
    opponent_score = np.clip(gap_opponent_reappearance, 0.0, 1.0)
    scores = np.column_stack([
        dead_score,
        crowd_score,
        visibility_score,
        deflection_score,
        control_score,
        flight_score,
    ])
    names = np.array([
        "DEAD_OR_OUT",
        "CROWD_OCCLUSION",
        "VISIBILITY_UNCERTAINTY",
        "DEFLECTION_OR_CONTEST",
        "CONTROL_CONTINUITY",
        "FLIGHT_OCCLUSION",
    ], dtype=object)
    winner = names[np.argmax(scores, axis=1)]
    maximum = np.max(scores, axis=1)
    winner[(maximum <= 0.0) | ~missing] = "UNKNOWN"
    confidence = np.where(missing, maximum, 0.0)
    observation_count = (
        expanded["dead_or_out"].astype(int)
        + expanded["flight"].astype(int)
        + expanded["control"].astype(int)
        + expanded["deflection"].astype(int)
    )

    out["missing_ball_cause"] = winner
    out["missing_ball_cause_confidence"] = confidence
    out["missing_ball_cause_dead_or_out_score"] = dead_score
    out["missing_ball_cause_flight_score"] = flight_score
    out["missing_ball_cause_control_score"] = control_score
    out["missing_ball_cause_deflection_score"] = deflection_score
    out["missing_ball_cause_crowd_score"] = crowd_score
    out["missing_ball_cause_visibility_score"] = visibility_score
    out["missing_ball_cause_george_observed_score"] = george_gap_observed.astype(float)
    out["missing_ball_cause_george_flight_score"] = george_flight.astype(float)
    out["missing_ball_cause_george_dead_score"] = george_dead
    out["missing_ball_gap_trajectory_continuity_score"] = continuity_score
    out["missing_ball_gap_same_team_reappearance_score"] = same_team_score
    out["missing_ball_gap_opponent_reappearance_score"] = opponent_score
    out["missing_ball_cause_observation_count"] = observation_count
    return out


def confidence_multiscale_labels(
    features: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 125,
    disagreement_margin_m: float = 1.5,
    minimum_switch_run: int = 12,
    maximum_internal_spike: int = 25,
    dead_ball_mode: str = "tagged_slow",
    require_transition_support: bool = False,
    require_reachability_support: bool = False,
) -> np.ndarray:
    """Decode nearest-player evidence with adaptive temporal context.

    ``short_window`` responds faster to genuine changes.  ``long_window``
    suppresses flicker.  When the windows disagree, the geometric margin and
    local persistence decide whether to switch; this is not a public-label
    fit.  The result is mapped to the official H/A/D/X alphabet.
    """
    codes = nearest_codes(features)
    short = smooth(codes, short_window)
    long = smooth(codes, long_window)
    labels = np.where(short == "H", "H", np.where(short == "A", "A", "D")).astype(object)

    margin = pd.to_numeric(features["team_distance_margin_m"], errors="coerce").to_numpy(float)
    closest = pd.to_numeric(features["closest_distance_m"], errors="coerce").to_numpy(float)
    short_team = short
    long_team = long
    disagreement = (short_team != long_team) & np.isin(short_team, ["H", "A"]) & np.isin(long_team, ["H", "A"])

    # A short-window switch is trusted only when the closest-player evidence
    # has a meaningful margin and is not merely a one-frame crossing.
    short_is_home = short_team == "H"
    short_is_away = short_team == "A"
    margin_supports_short = np.where(short_is_home, margin >= disagreement_margin_m,
                                     np.where(short_is_away, margin <= -disagreement_margin_m, False))
    labels[disagreement & ~margin_supports_short] = long_team[disagreement & ~margin_supports_short]

    if require_transition_support and "forecast_next_team" in features:
        forecast = features["forecast_next_team"].to_numpy(dtype=object)
        labels[disagreement & (forecast != short_team)] = long_team[disagreement & (forecast != short_team)]
    if require_reachability_support and "likely_receiver_team" in features:
        receiver = features["likely_receiver_team"].to_numpy(dtype=object)
        labels[disagreement & (receiver != short_team)] = long_team[disagreement & (receiver != short_team)]

    labels = _run_length_filter(labels, minimum_switch_run)

    if dead_ball_mode in {"tagged_slow", "tagged_edges"} and "smart_set_piece" in features:
        speed = pd.to_numeric(features.get("ball_speed_derived_mps"), errors="coerce").to_numpy(float)
        dead = features["smart_set_piece"].to_numpy(bool) & (np.nan_to_num(speed, nan=0.0) <= 1.0)
        if dead_ball_mode == "tagged_edges":
            edges = features.get("restart_start_candidate", pd.Series(False, index=features.index)).to_numpy(bool).copy()
            edges |= features.get("restart_end_candidate", pd.Series(False, index=features.index)).to_numpy(bool)
            dead &= edges
        labels[dead] = "D"
    elif dead_ball_mode == "none":
        pass
    else:
        raise ValueError(f"unknown dead_ball_mode: {dead_ball_mode}")

    labels = _merge_internal_spikes(labels, maximum_internal_spike)

    in_play = graded_mask(len(features))
    output = np.full(len(features), "X", dtype="<U1")
    output[in_play] = np.where(np.isin(labels[in_play], ["H", "A", "D"]), labels[in_play], "X")
    # A missing-ball frame is not automatically dead ball.  Use the local
    # temporal decision where one exists; otherwise abstain.
    output[in_play & ~np.isfinite(closest)] = labels[in_play & ~np.isfinite(closest)]
    return output


def classify_scenarios(
    features: pd.DataFrame,
    minimum_control_persistence: float = 0.80,
    maximum_control_distance_m: float = 2.0,
    minimum_control_margin_m: float = 1.0,
    maximum_control_speed_mps: float = 3.0,
    causal: bool = False,
) -> np.ndarray:
    """Classify frame context without assigning a possession label.

    ``causal=True`` excludes the future-confirmed reception diagnostic. The
    default remains the legacy scoring path until causal candidates are proven
    not to regress the selected baseline.
    """
    closest_team = features["closest_team"].to_numpy(dtype=object)
    closest_distance = pd.to_numeric(features["closest_distance_m"], errors="coerce").to_numpy(float)
    margin = pd.to_numeric(features["team_distance_margin_m"], errors="coerce").to_numpy(float)
    speed = pd.to_numeric(
        features.get("ball_speed_derived_mps", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    home_persistence = pd.to_numeric(
        features.get("home_persistence", pd.Series(0.0, index=features.index)), errors="coerce"
    ).to_numpy(float)
    away_persistence = pd.to_numeric(
        features.get("away_persistence", pd.Series(0.0, index=features.index)), errors="coerce"
    ).to_numpy(float)
    persistence = np.maximum(home_persistence, away_persistence)
    pressure = pd.to_numeric(
        features.get("candidate_pressure_m", pd.Series(np.nan, index=features.index)), errors="coerce"
    ).to_numpy(float)
    density = pd.to_numeric(
        features.get("candidate_opponent_density_5m", pd.Series(np.nan, index=features.index)), errors="coerce"
    ).to_numpy(float)
    nearest_h = pd.to_numeric(
        features.get("nearest_h_distance_m", pd.Series(np.nan, index=features.index)), errors="coerce"
    ).to_numpy(float)
    nearest_a = pd.to_numeric(
        features.get("nearest_a_distance_m", pd.Series(np.nan, index=features.index)), errors="coerce"
    ).to_numpy(float)
    set_piece = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)
    missing_ball = (
        pd.to_numeric(features.get("ball_x", pd.Series(np.nan, index=features.index)), errors="coerce").isna()
        | pd.to_numeric(features.get("ball_y", pd.Series(np.nan, index=features.index)), errors="coerce").isna()
    ).to_numpy(bool)
    recent_flight = features.get("recent_flight_evidence", pd.Series(False, index=features.index)).to_numpy(bool)
    # Keep the scenario detector causal: the stronger genuine-reception
    # diagnostic may use later confirmation, so it is not used here.
    causal_reception = features.get(
        "reception_anchor", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    future_reception = features.get(
        "genuine_reception_anchor", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    reception_signal = causal_reception if causal else future_reception
    restart_active = features.get(
        "restart_ball_active", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    reprojection_error = pd.to_numeric(
        features.get("reprojection_error", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    valid_tracking = features.get(
        "valid", pd.Series(True, index=features.index)
    ).fillna(False).to_numpy(bool)
    direction_change = pd.to_numeric(
        features.get("ball_direction_change_rad", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    acceleration = pd.to_numeric(
        features.get("ball_acceleration_mps2", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)

    scenarios = np.full(len(features), "UNKNOWN", dtype=object)
    slow_dead = set_piece & (np.nan_to_num(speed, nan=0.0) <= 1.0)
    scenarios[slow_dead] = "DEAD"

    # Smart Tagging is an interval/context signal.  Preserve that distinction
    # explicitly: a tagged restart can already be live even when the ball has
    # not yet reached a controlled player.  This is useful for diagnosis, but
    # is deliberately not a possession label.
    restart_context = set_piece & ~slow_dead
    restart_control_block = restart_context if causal else np.zeros(len(features), dtype=bool)
    scenarios[restart_context] = "RESTART_CONTEXT"
    restart_live = restart_context & restart_active
    scenarios[restart_live] = "RESTART_LIVE_BOUNDARY"
    scenarios[missing_ball & ~slow_dead & ~restart_context] = "OCCLUDED"
    scenarios[reception_signal & ~slow_dead & ~restart_context & ~missing_ball] = "RECEPTION"

    # These are causal diagnostics, not possession labels. They identify
    # frames where nearest-player assignment is least trustworthy.
    tracking_artifact = (
        (~valid_tracking)
        | (np.isfinite(reprojection_error) & (reprojection_error > 0.75))
    ) & ~missing_ball & ~slow_dead & ~restart_context
    if causal:
        scenarios[tracking_artifact] = "TRACKING_ARTIFACT"

    deflection_candidate = (
        np.isfinite(direction_change)
        & (direction_change >= 0.8)
        & (np.nan_to_num(speed, nan=0.0) >= 4.0)
        & ~missing_ball
        & ~slow_dead
        & ~restart_control_block
        & ~tracking_artifact
        & ~reception_signal
    )
    if causal:
        scenarios[deflection_candidate] = "DEFLECTION_CANDIDATE"

    miscontrol_candidate = (
        np.isin(closest_team, ["H", "A"])
        & np.isfinite(closest_distance)
        & (closest_distance > maximum_control_distance_m * 1.25)
        & (closest_distance <= 6.0)
        & (persistence >= minimum_control_persistence)
        & (np.nan_to_num(speed, nan=0.0) < 8.0)
        & (
            (np.nan_to_num(acceleration, nan=0.0) > 8.0)
            | (np.nan_to_num(speed, nan=0.0) > maximum_control_speed_mps)
        )
        & ~missing_ball
        & ~slow_dead
        & ~restart_context
        & ~tracking_artifact
        & ~reception_signal
    )
    if causal:
        scenarios[miscontrol_candidate] = "MISCONTROL_CANDIDATE"
    else:
        tracking_artifact = np.zeros(len(features), dtype=bool)
        deflection_candidate = np.zeros(len(features), dtype=bool)
        miscontrol_candidate = np.zeros(len(features), dtype=bool)

    flight = (
        (np.nan_to_num(speed, nan=0.0) >= 8.0)
        | (recent_flight & (closest_distance > maximum_control_distance_m))
    ) & ~slow_dead & ~reception_signal & ~restart_context
    flight &= ~tracking_artifact & ~deflection_candidate & ~miscontrol_candidate
    scenarios[flight] = "BALL_IN_FLIGHT"

    # A flight interval with a plausible receiver is more specific than a
    # generic flight interval.  It remains a candidate until post-contact
    # persistence confirms that the receiving team actually controlled it.
    receiver_eta = pd.to_numeric(
        features.get("arrival_time_margin_s", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    receiver_candidate = (
        flight
        & np.isfinite(receiver_eta)
        & (np.abs(receiver_eta) <= 0.75)
        & ~restart_context
    )
    scenarios[receiver_candidate] = "RECEPTION_CANDIDATE"

    both_teams_close = (nearest_h <= 4.0) & (nearest_a <= 4.0)
    convergence = (np.nan_to_num(density, nan=0.0) >= 2.0) | both_teams_close
    contested = (
        convergence
        & ~flight
        & ~slow_dead
        & ~reception_signal
        & ~tracking_artifact
        & ~deflection_candidate
        & ~miscontrol_candidate
        & ~missing_ball
        & ~restart_context
    )

    controlled = (
        np.isin(closest_team, ["H", "A"])
        & np.isfinite(closest_distance)
        & (closest_distance <= maximum_control_distance_m)
        & (np.abs(margin) >= minimum_control_margin_m)
        & (persistence >= minimum_control_persistence)
        & (np.nan_to_num(speed, nan=0.0) <= maximum_control_speed_mps)
        & ~slow_dead
        & ~flight
        & ~restart_control_block
        & ~reception_signal
        & ~tracking_artifact
        & ~deflection_candidate
        & ~miscontrol_candidate
    )
    scenarios[controlled] = "CONTROLLED"

    # Stable control takes precedence over broad convergence evidence. Only
    # the remaining frames are true contest candidates.
    contested &= ~controlled
    scenarios[contested] = "CONTESTED"

    # A player can retain the ball under pressure when the geometry remains
    # stable. Pressure is therefore a separate scenario, not an automatic loss.
    pressured = (
        np.isin(closest_team, ["H", "A"])
        & np.isfinite(closest_distance)
        & (closest_distance <= maximum_control_distance_m * 1.25)
        & (np.abs(margin) >= minimum_control_margin_m)
        & (persistence >= minimum_control_persistence)
        & ~slow_dead
        & ~flight
        & ~contested
        & ~reception_signal
        & ~tracking_artifact
        & ~deflection_candidate
        & ~miscontrol_candidate
        & ~missing_ball
        & ~controlled
        & ~restart_context
    )
    scenarios[pressured] = "PRESSURED_CONTROL"

    # A sign change in the team distance margin is a transition candidate,
    # not proof of a possession switch.  It identifies where the temporal
    # decoder should inspect the surrounding episode more carefully.
    margin_series = pd.Series(margin, index=features.index)
    margin_sign_change = margin_series.mul(margin_series.shift(1)).lt(0).to_numpy()
    transition_candidate = (
        margin_sign_change
        & ~np.isin(
            scenarios,
            [
                "DEAD", "RESTART_CONTEXT", "RESTART_LIVE_BOUNDARY", "OCCLUDED",
                "TRACKING_ARTIFACT", "DEFLECTION_CANDIDATE", "MISCONTROL_CANDIDATE",
            ],
        )
        & ~missing_ball
    )
    scenarios[transition_candidate] = "TRANSITION_CANDIDATE"
    return scenarios


def add_scenario_episode_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add episode IDs, lengths, and boundaries for the scenario timeline.

    Episodes are contiguous runs of the diagnostic ``scenario_context``.  The
    function does not infer possession and does not use the public labels; it
    only makes scenario durations and boundaries available to later rules or
    sequence models.
    """
    out = features.copy()
    scenario = out.get(
        "scenario_context", pd.Series("UNKNOWN", index=out.index, dtype=object)
    ).astype(object)
    changed = scenario.ne(scenario.shift()).fillna(True)
    episode_id = changed.cumsum().astype(int)
    out["scenario_episode_id"] = episode_id.to_numpy()
    lengths = episode_id.value_counts().to_dict()
    out["scenario_episode_length_frames"] = episode_id.map(lengths).to_numpy()
    out["scenario_episode_start"] = changed.to_numpy(bool)
    out["scenario_episode_end"] = changed.shift(-1, fill_value=True).to_numpy(bool)
    position = out.groupby("scenario_episode_id", sort=False).cumcount()
    out["scenario_episode_position_frames"] = position.to_numpy()
    out["scenario_episode_fraction"] = (
        out["scenario_episode_position_frames"]
        / np.maximum(out["scenario_episode_length_frames"] - 1, 1)
    )
    return out


def operational_reception_labels(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    required_signals: frozenset[str] | set[str] | None = None,
    post_control_fraction: float = 0.60,
    minimum_directional_score: float = 0.15,
    minimum_reception_score: float = 0.45,
    maximum_future_frames: int = 125,
    control_run_frames: int = 8,
) -> np.ndarray:
    """Resolve only reception episodes with independent post-contact evidence.

    A candidate episode must have a stable H/A label before it, a different
    stable H/A team after it, and receiver evidence from three distinct ideas:
    arrival/reception score, directional approach, and persistence of the
    receiving team after the episode.  The handoff is placed near the end of
    the ambiguous episode, rather than at the first nearest-player crossing.

    This is an operational rule candidate, not a fitted classifier.  It uses
    future frames because the competition output is an offline timeline; a
    live version would emit the decision only after the post-contact window.
    """
    scored = scenario_confidence_scores(features)
    required = frozenset(required_signals or {
        "reception_score", "directional_approach", "post_control"
    })
    valid_signals = {"reception_score", "directional_approach", "post_control"}
    unknown_signals = required - valid_signals
    if unknown_signals:
        raise ValueError(f"unknown reception signals: {sorted(unknown_signals)}")
    if "scenario_context" not in scored:
        scored["scenario_context"] = classify_scenarios(scored)
    scored = add_scenario_episode_features(scored)
    labels = np.asarray(base_labels, dtype="<U1").copy()
    graded = scored.get("graded", pd.Series(True, index=scored.index)).to_numpy(bool)
    scenario = scored["scenario_context"].to_numpy(object)
    candidate_mask = graded & np.isin(
        scenario,
        ["BALL_IN_FLIGHT", "RECEPTION_CANDIDATE", "CONTESTED", "TRANSITION_CANDIDATE"],
    )
    closest = scored.get("closest_team", pd.Series(None, index=scored.index)).to_numpy(object)
    distance = pd.to_numeric(
        scored.get("closest_distance_m", pd.Series(np.nan, index=scored.index)), errors="coerce"
    ).to_numpy(float)
    set_piece = scored.get("smart_set_piece", pd.Series(False, index=scored.index)).to_numpy(bool)
    reception = {
        "H": scored["reception_score_H"].to_numpy(float),
        "A": scored["reception_score_A"].to_numpy(float),
    }
    directional = {
        "H": scored["directional_reception_score_H"].to_numpy(float),
        "A": scored["directional_reception_score_A"].to_numpy(float),
    }
    post_control = {
        "H": scored["post_control_score_H"].to_numpy(float),
        "A": scored["post_control_score_A"].to_numpy(float),
    }

    start = 0
    while start < len(labels):
        if not candidate_mask[start]:
            start += 1
            continue
        end = start + 1
        while end < len(labels) and candidate_mask[end]:
            end += 1

        before = labels[start - 1] if start else "X"
        if before not in {"H", "A"}:
            start = end
            continue

        # Find a future controlled run, bounded to avoid jumping across
        # unrelated possessions or a long restart interval.
        after = None
        future_limit = min(len(labels), end + maximum_future_frames)
        cursor = end
        while cursor < future_limit:
            team = closest[cursor]
            if team not in {"H", "A"} or set_piece[cursor] or not np.isfinite(distance[cursor]) or distance[cursor] > 2.5:
                cursor += 1
                continue
            run_end = cursor + 1
            while (
                run_end < future_limit
                and closest[run_end] == team
                and not set_piece[run_end]
                and np.isfinite(distance[run_end])
                and distance[run_end] <= 2.5
            ):
                run_end += 1
            if run_end - cursor >= control_run_frames:
                after = str(team)
                break
            cursor = run_end

        if after not in {"H", "A"} or after == before:
            start = end
            continue

        episode_reception = np.nanmax(reception[after][start:end])
        episode_direction = np.nanmax(directional[after][start:end])
        episode_post = np.nanmax(post_control[after][start:end])
        checks = {
            "reception_score": episode_reception >= minimum_reception_score,
            "directional_approach": episode_direction >= minimum_directional_score,
            "post_control": episode_post >= post_control_fraction,
        }
        if all(checks[signal] for signal in required):
            # Keep the prior team through most of the flight/contest and
            # switch only at the episode's evidence-supported end.
            handoff = max(start, end - control_run_frames)
            labels[start:handoff] = before
            labels[handoff:end] = after
        start = end

    labels[~graded] = "X"
    return labels


def bridge_same_team_missing_ball_trajectory(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    maximum_gap_seconds: float = 5.0,
    require_coherent_trajectory: bool = True,
    require_receiver_run: bool = True,
    require_player_path: bool = False,
    require_same_player_dribble: bool = False,
    require_kalman_path: bool = False,
) -> np.ndarray:
    """Fill only bounded missing-ball gaps between agreeing H/A anchors.

    This is an offline resolver for a specific occlusion case. It never
    creates a possession switch. A gap is filled only when the strongest
    available team anchor before and after the gap agrees, the gap is no longer
    than the configured horizon, and the observed endpoint trajectory supports
    a plausible same-team receiver. ATD slots are temporary tracks, so the
    receiver test is intentionally team-level rather than an identity claim.
    """
    labels = np.asarray(base_labels, dtype="<U1").copy()
    graded = features.get("graded", pd.Series(True, index=features.index)).to_numpy(bool)
    ball_x = pd.to_numeric(features.get("ball_x", pd.Series(np.nan, index=features.index)), errors="coerce")
    ball_y = pd.to_numeric(features.get("ball_y", pd.Series(np.nan, index=features.index)), errors="coerce")
    missing = (ball_x.isna() | ball_y.isna()).to_numpy(bool)
    gap_frames_limit = max(1, round(float(maximum_gap_seconds) * 25))
    anchor = features.get(
        "selected_anchor_team",
        features.get("anchor_team", pd.Series(None, index=features.index)),
    ).to_numpy(object)
    coherent = features.get(
        "ball_gap_trajectory_coherent", pd.Series(False, index=features.index)
    ).fillna(False).to_numpy(bool)
    run_supported = features.get(
        "ball_gap_receiver_run_supported", pd.Series(False, index=features.index)
    ).fillna(False).to_numpy(bool)
    player_path_supported = features.get(
        "gap_player_path_supported", pd.Series(False, index=features.index)
    ).fillna(False).to_numpy(bool)
    player_path_team = features.get(
        "gap_player_path_team", pd.Series(None, index=features.index)
    ).to_numpy(object)
    same_player_dribble = features.get(
        "gap_same_player_dribble_candidate", pd.Series(False, index=features.index)
    ).fillna(False).to_numpy(bool)
    receiver = features.get(
        "ball_gap_receiver_team", pd.Series(None, index=features.index)
    ).to_numpy(object)
    kalman_reliable = features.get(
        "ball_kalman_reliable", pd.Series(False, index=features.index)
    ).fillna(False).to_numpy(bool)

    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
    for start, end in zip(starts, ends):
        before_candidates = np.flatnonzero(np.isin(anchor[:start], ["H", "A"]))
        after_candidates = np.flatnonzero(np.isin(anchor[end + 1 :], ["H", "A"]))
        if not len(before_candidates) or not len(after_candidates):
            continue
        before = int(before_candidates[-1])
        after = int(end + 1 + after_candidates[0])
        gap_frames = end - start + 1
        if gap_frames > gap_frames_limit or not (graded[start : end + 1].all()):
            continue
        if anchor[before] != anchor[after] or anchor[before] not in {"H", "A"}:
            continue
        gap_coherent = coherent[start : end + 1].any()
        gap_receiver = receiver[start : end + 1]
        receiver_agrees = np.any(gap_receiver == anchor[before])
        gap_run = run_supported[start : end + 1].any()
        if require_coherent_trajectory and not gap_coherent:
            continue
        if require_receiver_run and not (gap_run and receiver_agrees):
            continue
        if require_player_path and not np.any(
            player_path_supported[start : end + 1]
            & (player_path_team[start : end + 1] == anchor[before])
        ):
            continue
        if require_same_player_dribble and not same_player_dribble[start : end + 1].any():
            continue
        if require_kalman_path and not kalman_reliable[start : end + 1].any():
            continue
        labels[start : end + 1] = anchor[before]
    labels[~graded] = "X"
    return labels


def scenario_confidence_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable, label-free confidence scores for each frame.

    The scores are evidence summaries, not public-label probabilities.  They
    deliberately keep ``closest_team`` as an observation rather than treating
    the nearest player as possession on its own.  A later temporal decoder can
    combine these scores with duration and transition constraints.
    """
    out = features.copy()
    n = len(out)
    closest = out.get("closest_team", pd.Series(None, index=out.index)).to_numpy(object)
    distance = pd.to_numeric(out.get("closest_distance_m", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    margin = pd.to_numeric(out.get("team_distance_margin_m", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    speed = pd.to_numeric(out.get("ball_speed_evidence_mps", out.get("ball_speed_derived_mps", pd.Series(np.nan, index=out.index))), errors="coerce").to_numpy(float)
    persistence_h = pd.to_numeric(out.get("home_persistence", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).to_numpy(float)
    persistence_a = pd.to_numeric(out.get("away_persistence", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).to_numpy(float)
    pressure = pd.to_numeric(out.get("candidate_pressure_m", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    set_piece = out.get("smart_set_piece", pd.Series(False, index=out.index)).to_numpy(bool)
    restart_type = out.get("restart_type", pd.Series(None, index=out.index)).to_numpy(object)
    flight = out.get("ball_flight_candidate_interpolated", out.get("ball_flight_candidate", pd.Series(False, index=out.index))).to_numpy(bool)
    missing = pd.to_numeric(out.get("ball_x", pd.Series(np.nan, index=out.index)), errors="coerce").isna().to_numpy(bool) | pd.to_numeric(out.get("ball_y", pd.Series(np.nan, index=out.index)), errors="coerce").isna().to_numpy(bool)
    reception = out.get("genuine_reception_anchor", pd.Series(False, index=out.index)).to_numpy(bool)
    context = out.get("scenario_context", pd.Series("UNKNOWN", index=out.index)).to_numpy(object)
    graded = out.get("graded", pd.Series(True, index=out.index)).to_numpy(bool)

    def clipped(value: np.ndarray, low: float, high: float) -> np.ndarray:
        return np.clip((np.nan_to_num(value, nan=low) - low) / (high - low), 0.0, 1.0)

    closeness = 1.0 - clipped(distance, 0.0, 4.0)
    margin_strength = clipped(np.abs(margin), 0.0, 4.0)
    low_speed = 1.0 - clipped(speed, 1.0, 8.0)
    low_pressure = clipped(pressure, 2.0, 10.0)
    base_control = 0.32 * closeness + 0.25 * margin_strength + 0.20 * low_speed + 0.15 * low_pressure
    base_control = np.clip(base_control, 0.0, 1.0)

    out["state_score_H_CONTROL"] = np.where(closest == "H", base_control * (0.5 + 0.5 * persistence_h), 0.0)
    out["state_score_A_CONTROL"] = np.where(closest == "A", base_control * (0.5 + 0.5 * persistence_a), 0.0)
    # Smart Tagging intervals are context windows, not exact dead-ball spans.
    # A tagged frame earns dead-ball confidence only when the ball is actually
    # slow, or when it is missing during a non-kickoff low-motion interval.
    # Kickoff context is excluded because the first visible live-ball frames
    # can have missing coordinates while the official label is already H/A.
    low_motion_missing = (
        missing
        & (pd.to_numeric(out.get("player_activity_median_mps", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float) <= 1.5)
        & (pd.to_numeric(out.get("player_activity_p90_mps", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float) <= 2.0)
    )
    observed_slow = np.isfinite(speed) & (speed <= 1.0)
    ball_x = pd.to_numeric(out.get("ball_x", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    ball_y = pd.to_numeric(out.get("ball_y", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    out_of_bounds = np.isfinite(ball_x) & np.isfinite(ball_y) & ((ball_x < 0.0) | (ball_x > 1.0) | (ball_y < 0.0) | (ball_y > 1.0))
    dead_evidence = set_piece & ~np.isin(restart_type, ["Kick Off (Start)", "Kick Off (After Goal)"]) & (observed_slow | low_motion_missing | out_of_bounds)
    out["state_score_DEAD_BALL"] = np.where(dead_evidence, 1.0, 0.0)
    out["state_score_BALL_IN_FLIGHT"] = np.where(flight | (np.nan_to_num(speed, nan=0.0) >= 8.0), 1.0, 0.0)
    arrival = pd.to_numeric(out.get("arrival_time_margin_s", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    likely = out.get("likely_receiver_team", pd.Series(None, index=out.index)).to_numpy(object)
    projected = out.get("projected_receiver_team", pd.Series(None, index=out.index)).to_numpy(object)
    future_h = pd.to_numeric(out.get("home_future_anchor_persistence", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).to_numpy(float)
    future_a = pd.to_numeric(out.get("away_future_anchor_persistence", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).to_numpy(float)
    # A reception is stronger when raw tracking continues to support the same
    # team after contact, independently of the anchor heuristic.
    closest_series = pd.Series(closest, index=out.index)
    post_h = closest_series.eq("H")[::-1].rolling(10, min_periods=1).mean()[::-1].to_numpy(float)
    post_a = closest_series.eq("A")[::-1].rolling(10, min_periods=1).mean()[::-1].to_numpy(float)
    closing_h = pd.to_numeric(out.get("nearest_h_closing_rate_mps", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    closing_a = pd.to_numeric(out.get("nearest_a_closing_rate_mps", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    nearest_h = pd.to_numeric(out.get("nearest_h_distance_m", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    nearest_a = pd.to_numeric(out.get("nearest_a_distance_m", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    speed_drop = pd.to_numeric(out.get("ball_speed_drop_mps", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    flight_support = np.clip((np.nan_to_num(speed, nan=0.0) - 4.0) / 8.0, 0.0, 1.0)
    flight_support = np.maximum(flight_support, flight.astype(float))
    deceleration = np.clip(np.nan_to_num(speed_drop, nan=0.0) / 4.0, 0.0, 1.0)
    receiver_close = 1.0 - clipped(distance, 0.0, 4.0)
    h_reception = (
        0.25 * flight_support
        + 0.25 * receiver_close * (closest == "H")
        + 0.20 * (((likely == "H") | (projected == "H")).astype(float))
        + 0.15 * clipped(arrival, 0.0, 2.0)
        + 0.15 * future_h
        + 0.15 * deceleration
    ) / 1.30
    a_reception = (
        0.25 * flight_support
        + 0.25 * receiver_close * (closest == "A")
        + 0.20 * (((likely == "A") | (projected == "A")).astype(float))
        + 0.15 * clipped(-arrival, 0.0, 2.0)
        + 0.10 * future_a
        + 0.10 * post_a
        + 0.10 * np.clip(np.nan_to_num(closing_a, nan=0.0) / 3.0, 0.0, 1.0)
        + 0.15 * deceleration
    ) / 1.30
    h_reception = (
        0.25 * flight_support
        + 0.25 * receiver_close * (closest == "H")
        + 0.20 * (((likely == "H") | (projected == "H")).astype(float))
        + 0.15 * clipped(arrival, 0.0, 2.0)
        + 0.10 * future_h
        + 0.10 * post_h
        + 0.10 * np.clip(np.nan_to_num(closing_h, nan=0.0) / 3.0, 0.0, 1.0)
        + 0.15 * deceleration
    ) / 1.15
    h_reception = np.clip(h_reception, 0.0, 1.0)
    a_reception = np.clip(a_reception, 0.0, 1.0)
    out["reception_score_H"] = h_reception
    out["reception_score_A"] = a_reception
    out["post_control_score_H"] = post_h
    out["post_control_score_A"] = post_a
    out["directional_reception_score_H"] = np.clip(np.nan_to_num(closing_h, nan=0.0) / 3.0, 0.0, 1.0)
    out["directional_reception_score_A"] = np.clip(np.nan_to_num(closing_a, nan=0.0) / 3.0, 0.0, 1.0)
    out["state_score_RECEPTION"] = np.maximum(h_reception, a_reception)

    both_near = np.isfinite(nearest_h) & np.isfinite(nearest_a) & (nearest_h <= 4.0) & (nearest_a <= 4.0)
    margin_uncertainty = 1.0 - clipped(np.abs(margin), 0.0, 4.0)
    density = pd.to_numeric(out.get("candidate_opponent_density_5m", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    contest_score = (
        0.35 * both_near.astype(float)
        + 0.25 * margin_uncertainty
        + 0.20 * clipped(density, 0.0, 4.0)
        + 0.20 * flight_support
    )
    out["contest_score"] = np.clip(contest_score, 0.0, 1.0)
    out["state_score_CONTESTED"] = np.where(context == "CONTESTED", np.maximum(0.8, out["contest_score"]), out["contest_score"])
    out["state_score_OCCLUDED"] = np.where(missing, 1.0, 0.0)
    score_columns = [column for column in out.columns if column.startswith("state_score_")]
    scores = out[score_columns].to_numpy(float)
    best = np.nanmax(scores, axis=1)
    second = np.partition(scores, -2, axis=1)[:, -2] if len(score_columns) > 1 else np.zeros(n)
    out["scenario_confidence"] = best
    out["scenario_margin"] = best - second
    out["scenario_state"] = np.asarray([score_columns[index].removeprefix("state_score_") for index in np.argmax(scores, axis=1)], dtype=object)
    out.loc[~graded, "scenario_state"] = "OUTSIDE_GRADED"
    out.loc[~graded, "scenario_confidence"] = 1.0
    # The temporal decoder is appropriate only for ambiguous live-ball
    # evidence. Smart Tagging is a context interval rather than an exact
    # boundary, so restart-context frames are handled by the explicit rule
    # layer instead. Hard dead-ball evidence is never handed to Viterbi.
    live_ambiguous = np.isin(out["scenario_state"], ["BALL_IN_FLIGHT", "CONTESTED", "OCCLUDED"])
    live_ambiguous |= out["scenario_confidence"].to_numpy(float) < 0.70
    out["decoder_eligible"] = graded & ~set_piece & live_ambiguous
    return out


def add_observation_probability_features(
    features: pd.DataFrame,
    *,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Convert heterogeneous evidence into an uncalibrated state likelihood.

    The output is a normalized observation vector over ``H``, ``A``, ``D``,
    and ``X`` (uncertain/abstain).  It is not a probability claim yet: the
    component scores are rule-derived and have not been calibrated against a
    human-reviewed corpus.  The vector is nevertheless useful as an
    emission-like input to a later path decoder because it keeps the evidence
    channels visible and mutually exclusive only at the final state layer.

    This function deliberately separates observation likelihood from
    transition ownership.  Missing-ball, crowd, flight, and deflection
    evidence increase ``X``/contested uncertainty; they do not select H or A.
    Deterministic anchors add strong evidence but remain enforceable by the
    downstream anchor layer.
    """
    out = features.copy()
    if "state_score_H_CONTROL" not in out:
        out = scenario_confidence_scores(out)

    def numeric(name: str, default: float = 0.0) -> np.ndarray:
        return pd.to_numeric(
            out.get(name, pd.Series(default, index=out.index)), errors="coerce"
        ).fillna(default).to_numpy(float)

    def boolean(name: str) -> np.ndarray:
        # Pandas may expose a read-only NumPy view here. Several evidence
        # channels are combined in-place below, so request an owned writable
        # array rather than mutating a view into the DataFrame.
        return out.get(name, pd.Series(False, index=out.index)).fillna(False).to_numpy(
            bool,
            copy=True,
        )

    def bounded(name: str, default: float = 0.0) -> np.ndarray:
        return np.clip(numeric(name, default), 0.0, 1.0)

    control_h = bounded("state_score_H_CONTROL")
    control_a = bounded("state_score_A_CONTROL")
    reception_h = bounded("reception_score_H")
    reception_a = bounded("reception_score_A")
    post_h = bounded("post_control_score_H")
    post_a = bounded("post_control_score_A")
    directional_h = bounded("directional_reception_score_H")
    directional_a = bounded("directional_reception_score_A")
    contest = bounded("contest_score")
    dead = np.maximum(
        bounded("state_score_DEAD_BALL"),
        bounded("missing_ball_cause_dead_or_out_score"),
    )
    flight = np.maximum(
        bounded("state_score_BALL_IN_FLIGHT"),
        bounded("missing_ball_cause_flight_score"),
    )
    deflection = bounded("missing_ball_cause_deflection_score")
    crowd = np.maximum(
        bounded("scenario_crowd_score"),
        bounded("missing_ball_cause_crowd_score"),
    )
    visibility = bounded("missing_ball_cause_visibility_score")
    control_continuity = bounded("missing_ball_cause_control_score")
    missing = boolean("ball_missing_observed")
    missing |= ~np.isfinite(pd.to_numeric(out.get("ball_x", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float))
    missing |= ~np.isfinite(pd.to_numeric(out.get("ball_y", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float))

    # Optional Haiqal integration: raw distance/speed/dead-ball clues are
    # additional observations only.  The independent Haiqal labels are never
    # copied into this evidence vector.  Keeping the contribution small makes
    # this an ablation of evidence weighting, not a second hidden decoder.
    haiqal_closeness = np.clip(numeric("haiqal_closeness_score") / 3.0, -1.0, 1.0)
    haiqal_dead = bounded("haiqal_dead_ball_score")
    if "haiqal_closeness_score" in out:
        control_h = np.clip(control_h + 0.20 * np.maximum(haiqal_closeness, 0.0), 0.0, 1.0)
        control_a = np.clip(control_a + 0.20 * np.maximum(-haiqal_closeness, 0.0), 0.0, 1.0)
        dead = np.maximum(dead, 0.15 * haiqal_dead)

    # The signed margin is an additional observation channel.  It is never
    # used as a hard nearest-player ownership rule.
    margin = np.clip(numeric("team_distance_margin_m") / 4.0, -1.0, 1.0)
    margin_h = np.maximum(margin, 0.0)
    margin_a = np.maximum(-margin, 0.0)

    anchor = out.get(
        "selected_anchor_team",
        out.get("anchor_team", pd.Series(None, index=out.index, dtype=object)),
    ).to_numpy(object)
    selected_confidence = numeric("selected_anchor_confidence", np.nan)
    fallback_confidence = numeric("anchor_confidence_score", 0.0)
    anchor_confidence = np.clip(
        np.where(np.isfinite(selected_confidence), selected_confidence, fallback_confidence),
        0.0,
        1.0,
    )
    anchor_h = (anchor == "H").astype(float) * anchor_confidence
    anchor_a = (anchor == "A").astype(float) * anchor_confidence

    uncertainty = np.clip(
        0.45 * missing.astype(float)
        + 0.25 * visibility
        + 0.20 * crowd
        + 0.20 * deflection
        + 0.15 * contest,
        0.0,
        1.0,
    )
    # Flight/control continuity support a live continuation, but do not say
    # which team owns it. They therefore reduce the strength of an abstention
    # only when no strong contest/visibility signal is present.
    uncertainty = np.clip(
        uncertainty - 0.20 * np.maximum(flight, control_continuity) * (1.0 - contest),
        0.0,
        1.0,
    )

    logits = np.column_stack([
        1.20 * control_h + 0.85 * reception_h + 0.35 * post_h + 0.25 * directional_h
        + 0.45 * margin_h + 3.50 * anchor_h,
        1.20 * control_a + 0.85 * reception_a + 0.35 * post_a + 0.25 * directional_a
        + 0.45 * margin_a + 3.50 * anchor_a,
        2.40 * dead,
        1.90 * uncertainty + 0.35 * contest,
    ])
    logits[:, 0] -= 0.75 * (dead + uncertainty + 0.5 * contest)
    logits[:, 1] -= 0.75 * (dead + uncertainty + 0.5 * contest)
    logits /= max(float(temperature), 1e-6)
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)

    out["observation_probability_H"] = probabilities[:, 0]
    out["observation_probability_A"] = probabilities[:, 1]
    out["observation_probability_D"] = probabilities[:, 2]
    out["observation_probability_X"] = probabilities[:, 3]
    out["observation_probability_entropy"] = (
        -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
        / np.log(4.0)
    )
    ordered = np.sort(probabilities, axis=1)
    out["observation_probability_margin"] = ordered[:, -1] - ordered[:, -2]
    out["observation_probability_calibrated"] = False
    out["observation_probability_model"] = "RULE_EVIDENCE_SOFTMAX_UNCALIBRATED"
    out["observation_probability_uncertainty_score"] = uncertainty
    out["observation_probability_flight_support"] = flight
    out["observation_probability_control_continuity_support"] = control_continuity
    out["observation_probability_contest_support"] = contest
    return out


def add_transition_trust_features(
    features: pd.DataFrame,
    *,
    max_anchor_gap_frames: int = 75,
    minimum_anchor_confidence: float = 0.85,
) -> pd.DataFrame:
    """Build an auditable, label-free trust vector for a team switch.

    This is deliberately not a possession classifier and the softmax output is
    not a calibrated probability.  It is a compact state-transition belief:
    ``H``, ``A``, or ``NO_SWITCH``.  The components make the reason for a
    proposed switch inspectable: opposite-anchor agreement, receiver control,
    post-contact persistence, approach/arrival geometry, and ball-contact
    physics.  Missing or contested evidence lowers trust rather than forcing a
    turnover.
    """
    out = features.copy()
    n = len(out)

    def numeric(name: str, default: float = np.nan) -> np.ndarray:
        return pd.to_numeric(
            out.get(name, pd.Series(default, index=out.index)), errors="coerce"
        ).to_numpy(float)

    def boolean(name: str) -> np.ndarray:
        return out.get(name, pd.Series(False, index=out.index)).fillna(False).to_numpy(bool)

    def team_column(team: str, primary: str, fallback: str | None = None) -> np.ndarray:
        first = numeric(primary)
        if fallback:
            second = numeric(fallback)
            first = np.where(np.isfinite(first), first, second)
        return first

    anchor = out.get(
        "selected_anchor_team",
        out.get("anchor_team", pd.Series(None, index=out.index, dtype=object)),
    ).to_numpy(object)
    anchor_confidence = numeric("selected_anchor_confidence")
    if not np.isfinite(anchor_confidence).any():
        anchor_confidence = numeric("anchor_confidence_score", 0.0)
    reliable = np.isin(anchor, ["H", "A"]) & (
        np.nan_to_num(anchor_confidence, nan=0.0) >= float(minimum_anchor_confidence)
    )

    before_team = np.full(n, None, dtype=object)
    after_team = np.full(n, None, dtype=object)
    opposite_anchor_pair = np.zeros(n, dtype=bool)
    for position in range(n):
        before = position - 1
        before_limit = max(-1, position - int(max_anchor_gap_frames) - 1)
        while before > before_limit and not reliable[before]:
            before -= 1
        after = position + 1
        after_limit = min(n, position + int(max_anchor_gap_frames) + 1)
        while after < after_limit and not reliable[after]:
            after += 1
        if before > before_limit:
            before_team[position] = anchor[before]
        if after < after_limit:
            after_team[position] = anchor[after]
        opposite_anchor_pair[position] = (
            before_team[position] in {"H", "A"}
            and after_team[position] in {"H", "A"}
            and before_team[position] != after_team[position]
        )

    candidate_team = out.get(
        "reception_control_team_geometry",
        out.get("likely_receiver_team", pd.Series(None, index=out.index, dtype=object)),
    ).to_numpy(object)
    projected = out.get("projected_receiver_team", pd.Series(None, index=out.index, dtype=object)).to_numpy(object)
    candidate_team = np.where(
        np.isin(candidate_team, ["H", "A"]), candidate_team,
        np.where(np.isin(projected, ["H", "A"]), projected, None),
    )

    distance_h = numeric("nearest_h_distance_m")
    distance_a = numeric("nearest_a_distance_m")
    nearest_distance = numeric("closest_distance_m")
    proximity_h = np.clip(1.0 - np.nan_to_num(distance_h, nan=8.0) / 4.0, 0.0, 1.0)
    proximity_a = np.clip(1.0 - np.nan_to_num(distance_a, nan=8.0) / 4.0, 0.0, 1.0)

    closing_h = team_column("h", "nearest_h_closing_rate_mps", "h_fastest_closing_speed_mps")
    closing_a = team_column("a", "nearest_a_closing_rate_mps", "a_fastest_closing_speed_mps")
    approach_h = np.clip(np.nan_to_num(closing_h, nan=0.0) / 3.0, 0.0, 1.0)
    approach_a = np.clip(np.nan_to_num(closing_a, nan=0.0) / 3.0, 0.0, 1.0)

    arrival = numeric("arrival_time_margin_s")
    arrival_h = np.clip(np.nan_to_num(-arrival, nan=0.0) / 2.0, 0.0, 1.0)
    arrival_a = np.clip(np.nan_to_num(arrival, nan=0.0) / 2.0, 0.0, 1.0)
    consensus_h = np.clip(numeric("h_arrival_consensus_score", 0.0), 0.0, 1.0)
    consensus_a = np.clip(numeric("a_arrival_consensus_score", 0.0), 0.0, 1.0)
    # Replace part of the single-player arrival evidence with team consensus;
    # this improves the observation rather than merely increasing its weight.
    arrival_h = 0.55 * arrival_h + 0.45 * consensus_h
    arrival_a = 0.55 * arrival_a + 0.45 * consensus_a

    coherence_h = np.clip(
        numeric("h_contact_arrival_coherence", 0.0), 0.0, 1.0
    )
    coherence_a = np.clip(
        numeric("a_contact_arrival_coherence", 0.0), 0.0, 1.0
    )

    reception_h = numeric("reception_score_H", 0.0)
    reception_a = numeric("reception_score_A", 0.0)
    reception_signal = numeric("controlled_reception_signal_score", 0.0)
    reception_h = np.where(np.isfinite(reception_h), reception_h, reception_signal)
    reception_a = np.where(np.isfinite(reception_a), reception_a, reception_signal)

    post_h = numeric("post_control_score_H", 0.0)
    post_a = numeric("post_control_score_A", 0.0)
    post_h = np.where(np.isfinite(post_h), post_h, numeric("h_receiver_future_near_fraction", 0.0))
    post_a = np.where(np.isfinite(post_a), post_a, numeric("a_receiver_future_near_fraction", 0.0))

    speed = numeric("ball_speed_evidence_mps")
    if not np.isfinite(speed).any():
        speed = numeric("ball_speed_derived_mps", 0.0)
    speed_drop = numeric("reception_ball_deceleration_mps")
    speed_drop = np.where(np.isfinite(speed_drop), speed_drop, numeric("ball_speed_drop_mps", 0.0))
    direction_change = numeric("ball_direction_change_rad", 0.0)
    flight = boolean("recent_flight_evidence") | (np.nan_to_num(speed, nan=0.0) >= 8.0)
    contact_change = np.clip(np.nan_to_num(speed_drop, nan=0.0) / 4.0, 0.0, 1.0)
    contact_change = np.maximum(contact_change, np.clip(np.nan_to_num(direction_change, nan=0.0) / 1.2, 0.0, 1.0) * flight)

    contest = numeric("contest_score", 0.0)
    if not np.isfinite(contest).any():
        contest = numeric("candidate_opponent_density_5m", 0.0) / 4.0
    ball_x = numeric("ball_x")
    ball_y = numeric("ball_y")
    occluded = boolean("ball_missing_observed") | ~np.isfinite(ball_x) | ~np.isfinite(ball_y)

    anchor_h = ((after_team == "H") & (before_team == "A")).astype(float)
    anchor_a = ((after_team == "A") & (before_team == "H")).astype(float)
    valid_h = candidate_team == "H"
    valid_a = candidate_team == "A"

    # This is a trust score, not a probability.  The anchor-pair term is
    # intentionally dominant: local geometry cannot create a switch without
    # a later, independent controlled state agreeing with it.
    trust_h = (
        0.28 * anchor_h
        + 0.20 * np.nan_to_num(reception_h, nan=0.0)
        + 0.16 * np.nan_to_num(post_h, nan=0.0)
        + 0.10 * proximity_h
        + 0.08 * approach_h
        + 0.07 * arrival_h
        + 0.06 * contact_change
        + 0.05 * flight.astype(float)
        + 0.05 * coherence_h
    )
    trust_a = (
        0.28 * anchor_a
        + 0.20 * np.nan_to_num(reception_a, nan=0.0)
        + 0.16 * np.nan_to_num(post_a, nan=0.0)
        + 0.10 * proximity_a
        + 0.08 * approach_a
        + 0.07 * arrival_a
        + 0.06 * contact_change
        + 0.05 * flight.astype(float)
        + 0.05 * coherence_a
    )
    trust_h = np.clip(trust_h - 0.10 * np.nan_to_num(contest, nan=0.0) - 0.10 * occluded, 0.0, 1.0)
    trust_a = np.clip(trust_a - 0.10 * np.nan_to_num(contest, nan=0.0) - 0.10 * occluded, 0.0, 1.0)
    trust_h = np.where(valid_h | opposite_anchor_pair, trust_h, 0.0)
    trust_a = np.where(valid_a | opposite_anchor_pair, trust_a, 0.0)

    # Softmax is useful for comparing alternatives, but it is explicitly
    # marked uncalibrated until a labelled calibration set exists.
    no_switch_logit = 1.25 + 1.5 * np.nan_to_num(contest, nan=0.0) + 0.8 * (~opposite_anchor_pair)
    logits = np.column_stack([4.0 * trust_h, 4.0 * trust_a, no_switch_logit])
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    trust_score = np.maximum(trust_h, trust_a)
    trust_margin = np.abs(trust_h - trust_a)

    out["transition_before_anchor_team"] = before_team
    out["transition_after_anchor_team"] = after_team
    out["transition_opposite_anchor_pair"] = opposite_anchor_pair
    out["transition_candidate_team"] = candidate_team
    out["transition_trust_H"] = trust_h
    out["transition_trust_A"] = trust_a
    out["transition_trust_score"] = trust_score
    out["transition_trust_margin"] = trust_margin
    out["transition_trust_p_H"] = probabilities[:, 0]
    out["transition_trust_p_A"] = probabilities[:, 1]
    out["transition_trust_p_no_switch"] = probabilities[:, 2]
    out["transition_trust_calibrated"] = False
    out["transition_trust_candidate"] = opposite_anchor_pair & np.isin(candidate_team, ["H", "A"])
    out["transition_trust_reason"] = np.select(
        [~opposite_anchor_pair, occluded, np.nan_to_num(contest, nan=0.0) >= 0.6],
        ["NO_OPPOSITE_ANCHORS", "OCCLUDED", "CONTESTED"],
        default="RECEPTION_AND_ANCHOR_EVIDENCE",
    )
    return out


def add_contact_arrival_coherence_features(
    features: pd.DataFrame,
    *,
    contact_lookback_frames: int = 12,
    arrival_lookahead_frames: int = 25,
) -> pd.DataFrame:
    """Build soft event-level contact-to-arrival evidence for handoffs.

    A single frame can show a direction change near a defender or a player
    entering the ball radius without a real change of control. This feature
    links two different observations over a bounded interval: a plausible
    contact/motion change followed by team-level arrival, reception, or
    post-control support. It is evidence for a transition hypothesis, not a
    transition decision.
    """
    out = features.copy()

    def numeric(name: str, default: float = 0.0) -> np.ndarray:
        return pd.to_numeric(
            out.get(name, pd.Series(default, index=out.index)), errors="coerce"
        ).fillna(default).to_numpy(float)

    def boolean(name: str) -> np.ndarray:
        return out.get(
            name, pd.Series(False, index=out.index)
        ).fillna(False).to_numpy(bool)

    candidate = out.get(
        "reception_control_team_geometry",
        out.get("likely_receiver_team", pd.Series(None, index=out.index)),
    ).to_numpy(object)
    deceleration = np.clip(numeric("reception_ball_deceleration_mps") / 4.0, 0.0, 1.0)
    direction = np.clip(numeric("ball_direction_change_rad") / 1.2, 0.0, 1.0)
    flight = boolean("recent_flight_evidence") | (numeric("ball_speed_evidence_mps") >= 8.0)
    contact = np.maximum(deceleration, direction * flight.astype(float))

    lookback = max(1, int(contact_lookback_frames))
    lookahead = max(1, int(arrival_lookahead_frames))
    for team in ("H", "A"):
        reception = np.clip(numeric(f"reception_score_{team}"), 0.0, 1.0)
        consensus = np.clip(numeric(f"{team.lower()}_arrival_consensus_score"), 0.0, 1.0)
        post_control = np.clip(numeric(f"post_control_score_{team}"), 0.0, 1.0)
        future_near = np.clip(
            numeric(f"{team.lower()}_receiver_future_near_fraction"), 0.0, 1.0
        )
        arrival = np.maximum.reduce([reception, consensus, post_control, future_near])

        contact_team = contact * (candidate == team)
        contact_past = pd.Series(contact_team).rolling(
            lookback, min_periods=1
        ).max().to_numpy(float)
        future_arrival = pd.Series(arrival[::-1]).rolling(
            lookahead, min_periods=1
        ).max().to_numpy(float)[::-1]
        coherence = np.sqrt(np.clip(contact_past * future_arrival, 0.0, 1.0))
        out[f"{team.lower()}_contact_arrival_coherence"] = coherence
        out[f"{team.lower()}_contact_past_support"] = contact_past
        out[f"{team.lower()}_arrival_future_support"] = future_arrival

    h = out["h_contact_arrival_coherence"].to_numpy(float)
    a = out["a_contact_arrival_coherence"].to_numpy(float)
    out["contact_arrival_coherence_score"] = np.maximum(h, a)
    out["contact_arrival_coherence_margin"] = h - a
    out["contact_arrival_coherence_team"] = np.select(
        [h > a + 0.08, a > h + 0.08], ["H", "A"], default=None
    )
    out["contact_arrival_coherence_available"] = (
        out["contact_arrival_coherence_score"] > 0.0
    )
    return out


def add_haiqal_transition_evidence(
    features: pd.DataFrame,
    *,
    maximum_contribution: float = 0.04,
) -> pd.DataFrame:
    """Add Haiqal's raw closeness as a weak transition observation.

    Haiqal's standalone model uses distance/speed observations in its own
    HMM.  This adapter does not import that HMM path or its labels.  It adds
    only a small signed closeness contribution to the already-computed
    transition trust vector, after the current pipeline has constructed
    anchors and reception evidence.  The contribution is intentionally
    bounded and is therefore a signal ablation, not a newly fitted model.

    The downstream interval builder still decides between ``SWITCH_CANDIDATE``,
    ``NO_SWITCH``, and ``CONTESTED``; closeness alone cannot create a switch.
    """
    out = features.copy()
    if "haiqal_closeness_score" not in out:
        return out

    closeness = pd.to_numeric(out["haiqal_closeness_score"], errors="coerce").fillna(0.0)
    closeness = np.clip(closeness.to_numpy(float) / 3.0, -1.0, 1.0)
    contribution = float(max(0.0, maximum_contribution))

    trust_h = pd.to_numeric(
        out.get("transition_trust_H", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    trust_a = pd.to_numeric(
        out.get("transition_trust_A", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    out["transition_trust_H"] = np.clip(
        trust_h + contribution * np.maximum(closeness, 0.0), 0.0, 1.0
    )
    out["transition_trust_A"] = np.clip(
        trust_a + contribution * np.maximum(-closeness, 0.0), 0.0, 1.0
    )
    out["transition_haiqal_closeness_support_H"] = np.maximum(closeness, 0.0)
    out["transition_haiqal_closeness_support_A"] = np.maximum(-closeness, 0.0)
    out["transition_haiqal_evidence_weight"] = contribution
    return out


def add_transition_channel_profile(
    features: pd.DataFrame,
    *,
    profile: str = "balanced",
) -> pd.DataFrame:
    """Compose independent transition channels without importing labels.

    The current trust score is intentionally broad because it must remain the
    production-compatible baseline.  This experiment makes the proposed
    decomposition explicit: anchor agreement, ball motion/contact, player
    arrival, restart/no-switch context, and bounded Haiqal support.  A profile
    replaces the ownership part of the trust vector with a weighted channel
    composition; it never writes H/A/D/X labels and never reads ground truth.

    ``balanced`` is a predeclared, interpretable profile rather than a fit to
    the public slice.  The other profiles are sensitivity checks showing how
    much the result depends on anchor dominance or on requiring independent
    motion/arrival support.  The interval resolver remains responsible for
    deciding whether the proposed ownership is accepted.
    """
    out = features.copy()

    def numeric(name: str, default: float = 0.0) -> np.ndarray:
        return pd.to_numeric(
            out.get(name, pd.Series(default, index=out.index)), errors="coerce"
        ).fillna(default).to_numpy(float)

    def boolean(name: str) -> np.ndarray:
        return out.get(name, pd.Series(False, index=out.index)).fillna(False).to_numpy(bool)

    before = out.get(
        "transition_before_anchor_team",
        pd.Series(None, index=out.index, dtype=object),
    ).to_numpy(object)
    after = out.get(
        "transition_after_anchor_team",
        pd.Series(None, index=out.index, dtype=object),
    ).to_numpy(object)
    anchor_h = ((before == "A") & (after == "H")).astype(float)
    anchor_a = ((before == "H") & (after == "A")).astype(float)

    contact = np.maximum(
        np.clip(numeric("reception_ball_deceleration_mps") / 4.0, 0.0, 1.0),
        np.clip(numeric("ball_direction_change_rad") / 1.2, 0.0, 1.0),
    )
    flight = boolean("recent_flight_evidence") | (numeric("ball_speed_evidence_mps") >= 8.0)
    motion = np.clip(0.60 * contact + 0.40 * flight.astype(float), 0.0, 1.0)
    candidate = out.get(
        "transition_candidate_team",
        out.get("reception_control_team_geometry", pd.Series(None, index=out.index)),
    ).to_numpy(object)
    motion_h = motion * (candidate == "H")
    motion_a = motion * (candidate == "A")

    nearest_h = np.clip(1.0 - np.nan_to_num(numeric("nearest_h_distance_m", np.nan), nan=8.0) / 4.0, 0.0, 1.0)
    nearest_a = np.clip(1.0 - np.nan_to_num(numeric("nearest_a_distance_m", np.nan), nan=8.0) / 4.0, 0.0, 1.0)
    approach_h = np.clip(numeric("nearest_h_closing_rate_mps") / 3.0, 0.0, 1.0)
    approach_a = np.clip(numeric("nearest_a_closing_rate_mps") / 3.0, 0.0, 1.0)
    arrival_margin = numeric("arrival_time_margin_s", np.nan)
    arrival_h = np.clip(np.nan_to_num(-arrival_margin, nan=0.0) / 2.0, 0.0, 1.0)
    arrival_a = np.clip(np.nan_to_num(arrival_margin, nan=0.0) / 2.0, 0.0, 1.0)
    reception_h = np.clip(numeric("reception_score_H"), 0.0, 1.0)
    reception_a = np.clip(numeric("reception_score_A"), 0.0, 1.0)
    arrival_h = np.clip(0.50 * reception_h + 0.25 * nearest_h + 0.15 * approach_h + 0.10 * arrival_h, 0.0, 1.0)
    arrival_a = np.clip(0.50 * reception_a + 0.25 * nearest_a + 0.15 * approach_a + 0.10 * arrival_a, 0.0, 1.0)

    closeness = np.clip(numeric("haiqal_closeness_score") / 3.0, -1.0, 1.0)
    haiqal_h = np.maximum(closeness, 0.0)
    haiqal_a = np.maximum(-closeness, 0.0)

    no_switch = np.maximum(
        numeric("transition_no_switch_cause_support"),
        np.maximum(numeric("missing_ball_cause_flight_score"), numeric("missing_ball_cause_control_score")),
    )
    restart = boolean("smart_set_piece") | boolean("restart_ball_active") | boolean("restart_live_boundary_anchor")
    no_switch = np.clip(np.maximum(no_switch, 0.80 * restart.astype(float)), 0.0, 1.0)
    contested = np.maximum(
        numeric("contest_score"),
        np.maximum(numeric("missing_ball_cause_deflection_score"), numeric("missing_ball_cause_crowd_score")),
    )

    profiles: dict[str, dict[str, float]] = {
        "anchor_dominant": {"anchor": 0.45, "motion": 0.20, "arrival": 0.30, "haiqal": 0.05},
        "balanced": {"anchor": 0.35, "motion": 0.25, "arrival": 0.35, "haiqal": 0.05},
        "independent_support": {"anchor": 0.30, "motion": 0.30, "arrival": 0.35, "haiqal": 0.05},
    }
    if profile not in profiles:
        raise ValueError(f"unknown transition channel profile: {profile}")
    weights = profiles[profile]
    support_h = (
        weights["anchor"] * anchor_h
        + weights["motion"] * motion_h
        + weights["arrival"] * arrival_h
        + weights["haiqal"] * haiqal_h
    )
    support_a = (
        weights["anchor"] * anchor_a
        + weights["motion"] * motion_a
        + weights["arrival"] * arrival_a
        + weights["haiqal"] * haiqal_a
    )

    out["transition_channel_anchor_H"] = anchor_h
    out["transition_channel_anchor_A"] = anchor_a
    out["transition_channel_motion_H"] = motion_h
    out["transition_channel_motion_A"] = motion_a
    out["transition_channel_arrival_H"] = arrival_h
    out["transition_channel_arrival_A"] = arrival_a
    out["transition_channel_haiqal_H"] = haiqal_h
    out["transition_channel_haiqal_A"] = haiqal_a
    out["transition_channel_restart_no_switch"] = restart.astype(float)
    out["transition_channel_no_switch_support"] = no_switch
    out["transition_channel_contested_support"] = np.clip(contested, 0.0, 1.0)
    out["transition_channel_support_H"] = np.clip(support_h, 0.0, 1.0)
    out["transition_channel_support_A"] = np.clip(support_a, 0.0, 1.0)
    out["transition_channel_profile"] = profile
    out["transition_channel_weight_anchor"] = weights["anchor"]
    out["transition_channel_weight_motion"] = weights["motion"]
    out["transition_channel_weight_arrival"] = weights["arrival"]
    out["transition_channel_weight_haiqal"] = weights["haiqal"]
    out["transition_trust_H"] = np.clip(support_h, 0.0, 1.0)
    out["transition_trust_A"] = np.clip(support_a, 0.0, 1.0)
    return out


def add_transition_hypothesis_features(
    features: pd.DataFrame,
    *,
    minimum_activity: float = 0.25,
    merge_gap_frames: int = 2,
) -> pd.DataFrame:
    """Represent transition evidence as interval hypotheses.

    This separates two questions that were previously mixed together:

    1. Is a transition-like event happening in this interval?
    2. If so, which team is the best ownership hypothesis?

    The interval also keeps explicit ``NO_SWITCH`` and ``CONTESTED`` states.
    The resulting softmax values are audit weights, not calibrated
    probabilities. The resolver can use the interval state as an additional
    guard, while deterministic anchors remain authoritative.
    """
    out = features.copy()
    n = len(out)

    def numeric(name: str, default: float = np.nan) -> np.ndarray:
        return pd.to_numeric(
            out.get(name, pd.Series(default, index=out.index)), errors="coerce"
        ).to_numpy(float)

    def boolean(name: str) -> np.ndarray:
        return out.get(name, pd.Series(False, index=out.index)).fillna(False).to_numpy(bool)

    trust_h = np.nan_to_num(numeric("transition_trust_H", 0.0), nan=0.0)
    trust_a = np.nan_to_num(numeric("transition_trust_A", 0.0), nan=0.0)
    reception_h = np.nan_to_num(numeric("reception_score_H", 0.0), nan=0.0)
    reception_a = np.nan_to_num(numeric("reception_score_A", 0.0), nan=0.0)
    reception_signal = np.maximum(reception_h, reception_a)
    contact_change = np.maximum(
        np.clip(np.nan_to_num(numeric("reception_ball_deceleration_mps", 0.0), nan=0.0) / 4.0, 0.0, 1.0),
        np.clip(np.nan_to_num(numeric("ball_direction_change_rad", 0.0), nan=0.0) / 1.2, 0.0, 1.0),
    )
    kick_motion = np.clip(
        np.nan_to_num(numeric("kick_episode_contact_change", 0.0), nan=0.0),
        0.0,
        1.0,
    )
    kick_start = boolean("kick_episode_start")
    # A kick is an important observation, but it is not a possession change
    # by itself. Most kicks are passes or clearances made by the current
    # holder. Keep it as continuity/no-switch evidence rather than allowing
    # it to open a transition interval.
    kick_no_switch_support = kick_start & (kick_motion >= 0.25)
    scenario_crowd = np.clip(
        np.nan_to_num(numeric("scenario_crowd_score", 0.0), nan=0.0),
        0.0,
        1.0,
    )
    contest = np.clip(np.nan_to_num(numeric("contest_score", 0.0), nan=0.0), 0.0, 1.0)
    # Crowd geometry raises the contested/uncertain alternative, but it is
    # deliberately not allowed to create an ownership switch on its own.
    contest = np.maximum(contest, 0.60 * scenario_crowd)
    cause_dead = np.clip(np.nan_to_num(numeric("missing_ball_cause_dead_or_out_score", 0.0), nan=0.0), 0.0, 1.0)
    cause_flight = np.clip(np.nan_to_num(numeric("missing_ball_cause_flight_score", 0.0), nan=0.0), 0.0, 1.0)
    cause_control = np.clip(np.nan_to_num(numeric("missing_ball_cause_control_score", 0.0), nan=0.0), 0.0, 1.0)
    cause_deflection = np.clip(np.nan_to_num(numeric("missing_ball_cause_deflection_score", 0.0), nan=0.0), 0.0, 1.0)
    cause_crowd = np.clip(np.nan_to_num(numeric("missing_ball_cause_crowd_score", 0.0), nan=0.0), 0.0, 1.0)
    cause_visibility = np.clip(np.nan_to_num(numeric("missing_ball_cause_visibility_score", 0.0), nan=0.0), 0.0, 1.0)
    george_gap_observed = boolean("ball_fill_observed_during_tactical_gap")
    george_confidence = np.clip(
        np.nan_to_num(numeric("ball_fill_observation_confidence", 0.0), nan=0.0),
        0.0,
        1.0,
    )
    george_flight_support = george_gap_observed * boolean("ball_fill_flight_observation") * george_confidence
    george_dead_support = george_gap_observed * np.clip(
        np.nan_to_num(numeric("ball_fill_dead_probability", 0.0), nan=0.0),
        0.0,
        1.0,
    ) * george_confidence
    george_uncertainty_support = george_gap_observed * (1.0 - george_confidence)
    trajectory_continuity = np.clip(
        np.nan_to_num(
            numeric("missing_ball_gap_trajectory_continuity_score", 0.0),
            nan=0.0,
        ),
        0.0,
        1.0,
    )
    # Cause-specific evidence changes the alternatives, not the owner. A
    # flight/control interpretation supports continuity; crowd/deflection
    # supports contested handling; dead/out supports the dead-ball alternative.
    no_switch_cause_support = np.maximum(
        kick_no_switch_support.astype(float),
        np.maximum(cause_flight, np.maximum(cause_control, 0.75 * trajectory_continuity)),
    )
    no_switch_cause_support = np.maximum(no_switch_cause_support, george_flight_support)
    contested_cause_support = np.maximum(cause_deflection, 0.85 * cause_crowd)
    dead_cause_support = np.maximum(cause_dead, george_dead_support)
    uncertainty_cause_support = np.maximum(
        cause_visibility,
        np.maximum(cause_crowd, george_uncertainty_support),
    )
    # A channel-profile experiment may provide a conservative restart/
    # continuity support term.  It is allowed to strengthen NO_SWITCH, but it
    # cannot manufacture H/A ownership or bypass the resolver's anchor and
    # reception protections.
    profile_no_switch = np.clip(
        np.nan_to_num(numeric("transition_channel_no_switch_support", 0.0), nan=0.0),
        0.0,
        1.0,
    )
    no_switch_cause_support = np.maximum(no_switch_cause_support, profile_no_switch)
    profile_contested = np.clip(
        np.nan_to_num(numeric("transition_channel_contested_support", 0.0), nan=0.0),
        0.0,
        1.0,
    )
    contested_cause_support = np.maximum(contested_cause_support, profile_contested)
    opposite_anchor_pair = boolean("transition_opposite_anchor_pair")
    kalman_available = boolean("ball_kalman_path_available")
    kalman_reliable = boolean("ball_kalman_reliable")
    kalman_uncertainty = numeric("ball_kalman_position_uncertainty_m", np.nan)
    missing_cause = out.get(
        "missing_ball_cause", pd.Series("UNKNOWN", index=out.index)
    ).fillna("UNKNOWN").to_numpy(object)
    candidate_team = out.get(
        "transition_candidate_team",
        out.get("reception_control_team_geometry", pd.Series(None, index=out.index)),
    ).to_numpy(object)
    candidate_team = np.where(np.isin(candidate_team, ["H", "A"]), candidate_team, None)
    ball_missing = boolean("ball_missing_observed")

    # Activity says that something worth investigating is happening. It does
    # not choose a team. Anchor disagreement is strong activity evidence, but
    # geometry/contact evidence can open a hypothesis before anchors resolve.
    activity = np.clip(
        0.30 * np.maximum(trust_h, trust_a)
        + 0.25 * reception_signal
        + 0.20 * contact_change
        + 0.20 * opposite_anchor_pair.astype(float)
        + 0.05 * np.maximum(trust_h, trust_a) * (1.0 - contest),
        0.0,
        1.0,
    )
    raw_activity = (
        opposite_anchor_pair
        | (np.isin(candidate_team, ["H", "A"]) & ((reception_signal >= 0.25) | (contact_change >= 0.25)))
        | (contact_change >= 0.45)
    )

    # Merge short gaps so one physical event is not split by a noisy frame.
    active = np.zeros(n, dtype=bool)
    active_positions = np.flatnonzero(raw_activity)
    if active_positions.size:
        start = previous = int(active_positions[0])
        for position in active_positions[1:]:
            position = int(position)
            if position - previous <= int(merge_gap_frames) + 1:
                previous = position
                continue
            active[start : previous + 1] = True
            start = previous = position
        active[start : previous + 1] = True

    interval_id = np.full(n, -1, dtype=int)
    interval_start = np.full(n, -1, dtype=int)
    interval_end = np.full(n, -1, dtype=int)
    interval_activity = np.zeros(n, dtype=float)
    interval_team = np.full(n, None, dtype=object)
    interval_state = np.full(n, "NO_SWITCH", dtype=object)
    p_h = np.zeros(n, dtype=float)
    p_a = np.zeros(n, dtype=float)
    p_no_switch = np.ones(n, dtype=float)
    p_contested = np.zeros(n, dtype=float)
    interval_counter = 0
    positions = np.flatnonzero(active)
    cursor = 0
    while cursor < len(positions):
        start = int(positions[cursor])
        end = start
        while cursor + 1 < len(positions) and positions[cursor + 1] == end + 1:
            cursor += 1
            end = int(positions[cursor])
        window = slice(start, end + 1)
        mean_h = float(np.mean(trust_h[window]))
        mean_a = float(np.mean(trust_a[window]))
        activity_score = float(np.max(activity[window]))
        contest_score = float(np.max(contest[window]))
        ownership_margin = abs(mean_h - mean_a)
        contested_strength = max(
            contest_score,
            activity_score * (1.0 - min(1.0, ownership_margin * 4.0)),
            float(np.max(contested_cause_support[window])),
        )
        no_switch_strength = float(np.max(no_switch_cause_support[window]))
        dead_strength = float(np.max(dead_cause_support[window]))
        uncertainty_strength = float(np.max(uncertainty_cause_support[window]))

        if ball_missing[window].mean() >= 0.5:
            state = "OCCLUDED"
            team = None
        elif activity_score < float(minimum_activity):
            state = "NO_SWITCH"
            team = None
        elif contest_score >= 0.6 or ownership_margin < 0.08:
            state = "CONTESTED"
            team = None
        else:
            state = "SWITCH_CANDIDATE"
            team = "H" if mean_h > mean_a else "A"

        logits = np.array([
            2.0 * mean_h,
            2.0 * mean_a,
            2.0 * (1.0 - activity_score) + 0.8 * no_switch_strength + 0.5 * dead_strength,
            2.0 * contested_strength + 0.4 * uncertainty_strength,
        ])
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= max(float(probabilities.sum()), 1e-12)
        interval_id[window] = interval_counter
        interval_start[window] = start
        interval_end[window] = end
        interval_activity[window] = activity_score
        interval_team[window] = team
        interval_state[window] = state
        p_h[window], p_a[window], p_no_switch[window], p_contested[window] = probabilities
        interval_counter += 1
        cursor += 1

    out["transition_activity_score"] = activity
    out["transition_hypothesis_interval_id"] = interval_id
    out["transition_hypothesis_start"] = interval_start
    out["transition_hypothesis_end"] = interval_end
    out["transition_hypothesis_activity"] = interval_activity
    out["transition_hypothesis_team"] = interval_team
    out["transition_hypothesis_state"] = interval_state
    out["transition_hypothesis_p_H"] = p_h
    out["transition_hypothesis_p_A"] = p_a
    out["transition_hypothesis_p_no_switch"] = p_no_switch
    out["transition_hypothesis_p_contested"] = p_contested
    out["transition_kick_motion"] = kick_motion
    out["transition_no_switch_support"] = no_switch_cause_support
    out["transition_no_switch_cause_support"] = no_switch_cause_support
    out["transition_contested_cause_support"] = contested_cause_support
    out["transition_dead_ball_cause_support"] = dead_cause_support
    out["transition_uncertainty_cause_support"] = uncertainty_cause_support
    out["transition_george_observed"] = george_gap_observed
    out["transition_george_confidence"] = george_confidence
    out["transition_george_flight_support"] = george_flight_support
    out["transition_george_dead_support"] = george_dead_support
    out["transition_george_uncertainty_support"] = george_uncertainty_support
    out["transition_kalman_path_available"] = kalman_available
    out["transition_kalman_reliable"] = kalman_reliable
    out["transition_kalman_uncertainty_m"] = kalman_uncertainty
    out["transition_kalman_support"] = kalman_available & kalman_reliable
    out["transition_missing_ball_cause"] = missing_cause
    out["transition_hypothesis_calibrated"] = False
    return out


def build_transition_hypothesis_ledger(features: pd.DataFrame) -> pd.DataFrame:
    """Summarise transition intervals without using ground truth.

    The ledger is the handoff between observations and a future decoder. Each
    row answers: what interval was active, what evidence supported it, what
    conflicted, which anchors bracketed it, and what a downstream decoder is
    allowed to do. Public labels must be joined outside this function for
    evaluation only.
    """
    required = "transition_hypothesis_interval_id"
    if required not in features:
        return pd.DataFrame()

    table = features.loc[features[required].fillna(-1).ge(0)].copy()
    if table.empty:
        return pd.DataFrame(
            columns=["interval_id", "start_frame", "end_frame", "state", "decoder_action"]
        )

    def numeric(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
        return pd.to_numeric(frame.get(name, pd.Series(default, index=frame.index)), errors="coerce")

    def first_team(frame: pd.DataFrame, name: str) -> str | None:
        values = frame.get(name, pd.Series(index=frame.index, dtype=object)).dropna()
        valid = values[values.isin(["H", "A"])]
        return str(valid.iloc[0]) if not valid.empty else None

    rows: list[dict[str, object]] = []
    for interval_id, interval in table.groupby(required, sort=True):
        interval_id = int(interval_id)
        state = str(interval["transition_hypothesis_state"].iloc[0])
        team = first_team(interval, "transition_hypothesis_team")
        before_team = first_team(interval, "transition_before_anchor_team")
        after_team = first_team(interval, "transition_after_anchor_team")
        candidate_team = first_team(interval, "transition_candidate_team")
        scenarios = interval.get("scenario_context", pd.Series(dtype=object)).dropna().astype(str)

        evidence: list[str] = []
        if interval.get("transition_opposite_anchor_pair", pd.Series(False, index=interval.index)).fillna(False).any():
            evidence.append("OPPOSITE_ANCHORS")
        if max(
            numeric(interval, "reception_score_H", 0.0).max(skipna=True),
            numeric(interval, "reception_score_A", 0.0).max(skipna=True),
        ) >= 0.25:
            evidence.append("RECEPTION_GEOMETRY")
        if numeric(interval, "ball_direction_change_rad", 0.0).max(skipna=True) >= 0.8:
            evidence.append("DIRECTION_CHANGE")
        if numeric(interval, "reception_ball_deceleration_mps", 0.0).max(skipna=True) >= 1.0:
            evidence.append("CONTACT_DECELERATION")
        if numeric(interval, "contact_arrival_coherence_score", 0.0).max(skipna=True) >= 0.45:
            evidence.append("CONTACT_ARRIVAL_COHERENCE")
        if interval.get("kick_episode_start", pd.Series(False, index=interval.index)).fillna(False).any():
            evidence.append("KICK_MOTION")
        if numeric(interval, "contest_score", 0.0).max(skipna=True) >= 0.6:
            evidence.append("CONTEST")
        if interval.get("ball_missing_observed", pd.Series(False, index=interval.index)).fillna(False).any():
            evidence.append("OCCLUSION")
        if interval.get("transition_george_observed", pd.Series(False, index=interval.index)).fillna(False).any():
            evidence.append("GEORGE_OBSERVATION")
        cause_values = interval.get(
            "transition_missing_ball_cause",
            pd.Series("UNKNOWN", index=interval.index),
        ).fillna("UNKNOWN").astype(str)
        cause_set = set(cause_values[cause_values != "UNKNOWN"])
        if "DEAD_OR_OUT" in cause_set:
            evidence.append("DEAD_OR_OUT_CONTEXT")
        if "FLIGHT_OCCLUSION" in cause_set:
            evidence.append("FLIGHT_OCCLUSION")
        if "CROWD_OCCLUSION" in cause_set:
            evidence.append("CROWD_OCCLUSION")
        if "VISIBILITY_UNCERTAINTY" in cause_set:
            evidence.append("VISIBILITY_UNCERTAINTY")
        if "CONTROL_CONTINUITY" in cause_set:
            evidence.append("CONTROL_CONTINUITY")
        if "DEFLECTION_OR_CONTEST" in cause_set:
            evidence.append("DEFLECTION_RISK")
        if interval.get("transition_kalman_reliable", pd.Series(False, index=interval.index)).fillna(False).any():
            evidence.append("KALMAN_BRIDGED_PATH")
        if not evidence:
            evidence.append("MOTION_ACTIVITY")

        conflicts: list[str] = []
        if "CONTEST" in evidence:
            conflicts.append("COMPETING_PLAYERS")
        if "OCCLUSION" in evidence:
            conflicts.append("MISSING_BALL")
        if before_team and after_team and before_team == after_team:
            conflicts.append("SAME_ANCHOR_CONTINUITY")
        if state == "CONTESTED":
            conflicts.append("OWNERSHIP_MARGIN_SMALL")

        if state == "SWITCH_CANDIDATE" and before_team and after_team and before_team != after_team and team == after_team:
            decoder_action = "ELIGIBLE_AFTER_ANCHOR_CHECK"
        elif state in {"CONTESTED", "OCCLUDED"}:
            decoder_action = "HOLD_OR_ABSTAIN"
        else:
            decoder_action = "RETAIN_PRIOR_STATE"

        rows.append(
            {
                "interval_id": interval_id,
                "start_frame": int(interval["frame"].min()),
                "end_frame": int(interval["frame"].max()),
                "duration_frames": int(len(interval)),
                "state": state,
                "hypothesis_team": team,
                "candidate_team": candidate_team,
                "activity_max": float(numeric(interval, "transition_hypothesis_activity", 0.0).max(skipna=True)),
                "activity_mean": float(numeric(interval, "transition_activity_score", 0.0).mean()),
                "p_H": float(numeric(interval, "transition_hypothesis_p_H", 0.0).max(skipna=True)),
                "p_A": float(numeric(interval, "transition_hypothesis_p_A", 0.0).max(skipna=True)),
                "p_no_switch": float(numeric(interval, "transition_hypothesis_p_no_switch", 0.0).max(skipna=True)),
                "p_contested": float(numeric(interval, "transition_hypothesis_p_contested", 0.0).max(skipna=True)),
                "contact_arrival_coherence_max": float(
                    numeric(interval, "contact_arrival_coherence_score", 0.0).max(skipna=True)
                ),
                "contact_arrival_coherence_margin": float(
                    numeric(interval, "contact_arrival_coherence_margin", 0.0).mean()
                ),
                "contact_arrival_coherence_team": first_team(
                    interval, "contact_arrival_coherence_team"
                ),
                "no_switch_support_max": float(
                    numeric(interval, "transition_no_switch_support", 0.0).max(skipna=True)
                ),
                "no_switch_cause_support_max": float(
                    numeric(interval, "transition_no_switch_cause_support", 0.0).max(skipna=True)
                ),
                "contested_cause_support_max": float(
                    numeric(interval, "transition_contested_cause_support", 0.0).max(skipna=True)
                ),
                "dead_ball_cause_support_max": float(
                    numeric(interval, "transition_dead_ball_cause_support", 0.0).max(skipna=True)
                ),
                "uncertainty_cause_support_max": float(
                    numeric(interval, "transition_uncertainty_cause_support", 0.0).max(skipna=True)
                ),
                "george_observation_confidence_max": float(
                    numeric(interval, "transition_george_confidence", 0.0).max(skipna=True)
                ),
                "george_flight_support_max": float(
                    numeric(interval, "transition_george_flight_support", 0.0).max(skipna=True)
                ),
                "george_dead_support_max": float(
                    numeric(interval, "transition_george_dead_support", 0.0).max(skipna=True)
                ),
                "george_uncertainty_support_max": float(
                    numeric(interval, "transition_george_uncertainty_support", 0.0).max(skipna=True)
                ),
                "before_anchor_team": before_team,
                "after_anchor_team": after_team,
                "opposite_anchor_pair": bool(interval.get("transition_opposite_anchor_pair", pd.Series(False, index=interval.index)).fillna(False).any()),
                "scenario_context": str(scenarios.mode().iloc[0]) if not scenarios.empty else "UNKNOWN",
                "closest_distance_min_m": float(numeric(interval, "closest_distance_m").min(skipna=True)),
                "closing_rate_max_mps": float(max(numeric(interval, "nearest_h_closing_rate_mps", 0.0).max(skipna=True), numeric(interval, "nearest_a_closing_rate_mps", 0.0).max(skipna=True))),
                "ball_speed_max_mps": float(numeric(interval, "ball_speed_derived_mps", 0.0).max(skipna=True)),
                "kalman_path_available": bool(interval.get("transition_kalman_path_available", pd.Series(False, index=interval.index)).fillna(False).any()),
                "kalman_reliable": bool(interval.get("transition_kalman_reliable", pd.Series(False, index=interval.index)).fillna(False).any()),
                "kalman_uncertainty_max_m": float(numeric(interval, "transition_kalman_uncertainty_m").max(skipna=True)),
                "missing_ball_cause": str(cause_values.mode().iloc[0]) if not cause_values.empty else "UNKNOWN",
                "evidence_reason": "|".join(evidence),
                "conflict_reason": "|".join(conflicts) if conflicts else "NONE",
                "decoder_action": decoder_action,
            }
        )
    return pd.DataFrame(rows).sort_values(["start_frame", "interval_id"]).reset_index(drop=True)


def bootstrap_transition_stability(
    features: pd.DataFrame,
    *,
    n_bootstrap: int = 256,
    random_state: int = 0,
) -> pd.DataFrame:
    """Estimate interval-state stability by resampling observed frames.

    This is a nonparametric uncertainty diagnostic. It resamples frames
    within each already-detected transition interval and recomputes the mean
    interval evidence for ``H``, ``A``, ``NO_SWITCH``, and ``CONTESTED``. It
    estimates how stable the current evidence aggregation is under empirical
    resampling; it is not a calibrated probability of the hidden possession
    state and never reads ground truth.
    """
    if int(n_bootstrap) < 2:
        raise ValueError("n_bootstrap must be at least 2")
    required = "transition_hypothesis_interval_id"
    if required not in features:
        return pd.DataFrame()

    interval_ids = pd.to_numeric(features[required], errors="coerce").fillna(-1).to_numpy(int)
    valid_ids = np.unique(interval_ids[interval_ids >= 0])
    if not len(valid_ids):
        return pd.DataFrame()

    def values(name: str, default: float = 0.0) -> np.ndarray:
        return pd.to_numeric(
            features.get(name, pd.Series(default, index=features.index)),
            errors="coerce",
        ).fillna(default).to_numpy(float)

    # Do not resample the already-aggregated interval probabilities: those
    # are constant by construction inside an interval and would produce a
    # misleading stability of 1.0. Resample the underlying frame evidence,
    # then reconstruct the interval logits from the same evidence channels.
    trust_h = values("transition_trust_H")
    trust_a = values("transition_trust_A")
    activity = values("transition_activity_score")
    no_switch_support = values("transition_no_switch_support")
    dead_support = values("transition_dead_ball_cause_support")
    contest = values("contest_score")
    contested_support = values("transition_contested_cause_support")
    uncertainty_support = values("transition_uncertainty_cause_support")
    names = np.array(["H", "A", "NO_SWITCH", "CONTESTED"], dtype=object)
    rng = np.random.default_rng(int(random_state))
    rows: list[dict[str, object]] = []
    for interval_id in valid_ids:
        positions = np.flatnonzero(interval_ids == interval_id)
        sample_positions = rng.choice(
            positions,
            size=(int(n_bootstrap), len(positions)),
            replace=True,
        )
        mean_h = trust_h[sample_positions].mean(axis=1)
        mean_a = trust_a[sample_positions].mean(axis=1)
        mean_activity = activity[sample_positions].mean(axis=1)
        mean_no_switch = no_switch_support[sample_positions].mean(axis=1)
        mean_dead = dead_support[sample_positions].mean(axis=1)
        mean_contest = contest[sample_positions].mean(axis=1)
        mean_contested = contested_support[sample_positions].mean(axis=1)
        mean_uncertainty = uncertainty_support[sample_positions].mean(axis=1)
        ownership_margin = np.abs(mean_h - mean_a)
        contested_strength = np.maximum.reduce([
            mean_contest,
            mean_activity * (1.0 - np.minimum(1.0, ownership_margin * 4.0)),
            mean_contested,
        ])
        logits = np.column_stack([
            2.0 * mean_h,
            2.0 * mean_a,
            2.0 * (1.0 - mean_activity) + 0.8 * mean_no_switch + 0.5 * mean_dead,
            2.0 * contested_strength + 0.4 * mean_uncertainty,
        ])
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        winners = np.argmax(probabilities, axis=1)
        winner_counts = np.bincount(winners, minlength=len(names))
        winner = int(np.argmax(winner_counts))
        no_switch_samples = probabilities[:, 2]
        rows.append(
            {
                "interval_id": int(interval_id),
                "start_frame": int(positions.min()),
                "end_frame": int(positions.max()),
                "frame_count": int(len(positions)),
                "stable_state": str(names[winner]),
                "stable_state_fraction": float(winner_counts[winner] / int(n_bootstrap)),
                "bootstrap_H_fraction": float((winners == 0).mean()),
                "bootstrap_A_fraction": float((winners == 1).mean()),
                "bootstrap_no_switch_fraction": float((winners == 2).mean()),
                "bootstrap_contested_fraction": float((winners == 3).mean()),
                "no_switch_evidence_mean": float(no_switch_samples.mean()),
                "no_switch_evidence_lower": float(np.percentile(no_switch_samples, 2.5)),
                "no_switch_evidence_upper": float(np.percentile(no_switch_samples, 97.5)),
                "bootstrap_samples": int(n_bootstrap),
                "bootstrap_random_state": int(random_state),
            }
        )
    return pd.DataFrame(rows)


def transition_channel_stability(features: pd.DataFrame) -> pd.DataFrame:
    """Audit transition hypotheses after leaving out one evidence family.

    Each ablation removes one observation family, rebuilds the transition
    trust vector and interval hypotheses, and compares the resulting state
    with the full evidence path. This is an empirical sensitivity measure,
    not a learned feature weight and not a ground-truth score. Restart context
    is included explicitly even when the current core transition hypothesis
    does not consume it; that no-op is itself useful architectural evidence.
    """
    required = "transition_hypothesis_interval_id"
    if required not in features:
        return pd.DataFrame()

    interval_ids = pd.to_numeric(features[required], errors="coerce").fillna(-1).to_numpy(int)
    valid_ids = np.unique(interval_ids[interval_ids >= 0])
    if not len(valid_ids):
        return pd.DataFrame()
    full_states = features.get(
        "transition_hypothesis_state",
        pd.Series("NO_SWITCH", index=features.index),
    ).fillna("NO_SWITCH").to_numpy(object)

    channel_columns = {
        "anchors": [
            "anchor_team", "selected_anchor_team", "anchor_confidence_score",
            "selected_anchor_confidence", "pressure_anchor_team",
        ],
        "ball_motion_contact": [
            "reception_score_H", "reception_score_A", "controlled_reception_signal_score",
            "post_control_score_H", "post_control_score_A",
            "directional_reception_score_H", "directional_reception_score_A",
            "reception_ball_deceleration_mps", "ball_direction_change_rad",
            "kick_episode_start", "kick_episode_contact_change",
            "reception_control_team_geometry", "projected_receiver_team",
            "missing_ball_gap_trajectory_continuity_score",
        ],
        "player_arrival": [
            "nearest_h_distance_m", "nearest_a_distance_m", "closest_distance_m",
            "nearest_h_closing_rate_mps", "nearest_a_closing_rate_mps",
            "h_fastest_closing_speed_mps", "a_fastest_closing_speed_mps",
            "arrival_time_margin_s", "h_receiver_future_near_fraction",
            "a_receiver_future_near_fraction", "likely_receiver_team",
            "h_arrival_consensus_score", "a_arrival_consensus_score",
            "h_arrival_consensus_fraction", "a_arrival_consensus_fraction",
            "arrival_consensus_margin", "arrival_consensus_team",
        ],
        "missing_ball_cause": [
            "missing_ball_cause", "missing_ball_cause_dead_or_out_score",
            "missing_ball_cause_flight_score", "missing_ball_cause_control_score",
            "missing_ball_cause_deflection_score", "missing_ball_cause_crowd_score",
            "missing_ball_cause_visibility_score", "ball_missing_observed",
            "missing_ball_gap_trajectory_continuity_score",
            "missing_ball_gap_same_team_reappearance_score",
            "missing_ball_gap_opponent_reappearance_score",
        ],
        "restart_context": [
            "restart_context", "smart_set_piece", "restart_ball_active",
            "restart_live_boundary_anchor", "restart_hinted_control_anchor",
            "restart_fallback_control_anchor", "restart_end_candidate",
        ],
        "haiqal": [
            "haiqal_distance_home_m", "haiqal_distance_away_m",
            "haiqal_distance_margin_m", "haiqal_ball_speed_mps",
            "haiqal_player_speed_mps", "haiqal_visible", "haiqal_out_of_pitch",
            "haiqal_closeness_score", "haiqal_dead_ball_score", "haiqal_tag_prior",
        ],
    }

    def mode(values: np.ndarray) -> str:
        series = pd.Series(values, dtype=object)
        return str(series.mode().iloc[0]) if not series.empty else "NO_SWITCH"

    rows: list[dict[str, object]] = []
    for channel, columns in channel_columns.items():
        ablated = features.copy()
        for column in columns:
            if column not in ablated:
                continue
            if pd.api.types.is_bool_dtype(ablated[column]):
                ablated[column] = False
            elif pd.api.types.is_numeric_dtype(ablated[column]):
                ablated[column] = 0.0
            else:
                ablated[column] = None
        rebuilt = add_transition_hypothesis_features(
            add_transition_trust_features(ablated)
        )
        rebuilt_states = rebuilt["transition_hypothesis_state"].fillna("NO_SWITCH").to_numpy(object)
        for interval_id in valid_ids:
            positions = np.flatnonzero(interval_ids == interval_id)
            if not len(positions):
                continue
            full_state = mode(full_states[positions])
            ablated_state = mode(rebuilt_states[positions])
            rows.append(
                {
                    "channel": channel,
                    "interval_id": int(interval_id),
                    "start_frame": int(positions.min()),
                    "end_frame": int(positions.max()),
                    "full_state": full_state,
                    "ablated_state": ablated_state,
                    "state_agrees": bool(full_state == ablated_state),
                    "ablated_frame_state_change_fraction": float(
                        np.mean(full_states[positions] != rebuilt_states[positions])
                    ),
                }
            )
    return pd.DataFrame(rows)


def duration_aware_scenario_decode(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    minimum_team_run: int = 8,
    switch_penalty: float = 0.35,
    base_evidence_weight: float = 0.75,
) -> np.ndarray:
    """Decode official labels from scenario evidence with a conservative HSMM-style pass.

    This is a finite-state Viterbi decoder with explicit minimum team-run
    duration.  It is intentionally not EM-fitted: the public labels are never
    used, and the current rule output is only treated as algorithmic evidence.
    """
    scored = scenario_confidence_scores(features)
    n = len(scored)
    graded = scored.get("graded", pd.Series(True, index=scored.index)).to_numpy(bool)
    states = ("H", "A", "D")
    evidence = np.column_stack([
        scored["state_score_H_CONTROL"].to_numpy(float),
        scored["state_score_A_CONTROL"].to_numpy(float),
        scored["state_score_DEAD_BALL"].to_numpy(float),
    ])
    base = np.asarray(base_labels, dtype=object)
    eligible = scored["decoder_eligible"].to_numpy(bool)
    evidence += np.column_stack([base == state for state in states]).astype(float) * base_evidence_weight
    labels = np.full(n, "X", dtype=object)
    # Decode each contiguous graded block independently.  The ungraded gap
    # between halves is an intentional reset, not an impossible observation
    # that should poison the dynamic-programming scores.
    starts = np.flatnonzero(graded & ~np.r_[False, graded[:-1]])
    ends = np.flatnonzero(graded & ~np.r_[graded[1:], False]) + 1
    for block_start, block_end in zip(starts, ends):
        length = block_end - block_start
        block_evidence = evidence[block_start:block_end]
        score = np.full((length, 3), -np.inf, dtype=float)
        back = np.zeros((length, 3), dtype=np.int8)
        score[0] = block_evidence[0]
        for offset in range(1, length):
            for current in range(3):
                candidate = score[offset - 1].copy()
                candidate[np.arange(3) != current] -= switch_penalty
                previous = int(np.argmax(candidate))
                score[offset, current] = candidate[previous] + block_evidence[offset, current]
                back[offset, current] = previous
        state = int(np.argmax(score[-1]))
        path = np.empty(length, dtype=np.int8)
        for offset in range(length - 1, -1, -1):
            path[offset] = state
            if offset:
                state = int(back[offset, state])
        labels[block_start:block_end] = np.asarray(states, dtype="<U1")[path]

    # Semi-Markov duration constraint: remove short H/A runs only when they
    # are surrounded by the same state. D is never removed by this pass.
    labels = _run_length_filter(labels, minimum_team_run)
    labels[~graded] = "X"
    labels[graded & ~eligible] = base[graded & ~eligible]
    return labels.astype("<U1")


def scenario_gated_labels(
    features: pd.DataFrame,
    minimum_control_persistence: float = 0.80,
    maximum_control_distance_m: float = 2.0,
    minimum_control_margin_m: float = 1.0,
    maximum_control_speed_mps: float = 3.0,
    maximum_internal_spike: int = 50,
    allow_pressured_control: bool = True,
    allow_receiver_switch: bool = False,
    allow_bidirectional_bridge: bool = False,
    receiver_horizon_frames: int = 25,
    require_receiver_path_agreement: bool = False,
    require_strict_reception: bool = False,
    require_anchor_forecast_confirmation: bool = False,
    minimum_receiver_persistence_frames: int = 0,
) -> np.ndarray:
    """Use nearest-player team only after identifying comfortable control.

    The nearest player is still computed for every frame, but it is treated as
    a candidate observation outside the control regime. Ambiguous flight,
    contest, and missing-ball intervals are filled from bounded neighbouring
    control evidence, preferring the preceding team when the sides disagree.
    This keeps scenario identification separate from possession assignment.
    """
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    closest_team = features["closest_team"].to_numpy(dtype=object)
    closest_distance = pd.to_numeric(features["closest_distance_m"], errors="coerce").to_numpy(float)
    margin = pd.to_numeric(features["team_distance_margin_m"], errors="coerce").to_numpy(float)
    speed = pd.to_numeric(
        features.get("ball_speed_derived_mps", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    home_persistence = pd.to_numeric(
        features.get("home_persistence", pd.Series(0.0, index=features.index)), errors="coerce"
    ).to_numpy(float)
    away_persistence = pd.to_numeric(
        features.get("away_persistence", pd.Series(0.0, index=features.index)), errors="coerce"
    ).to_numpy(float)
    persistence = np.maximum(home_persistence, away_persistence)
    pressure = pd.to_numeric(
        features.get("candidate_pressure_m", pd.Series(np.nan, index=features.index)), errors="coerce"
    ).to_numpy(float)
    density = pd.to_numeric(
        features.get("candidate_opponent_density_5m", pd.Series(np.nan, index=features.index)), errors="coerce"
    ).to_numpy(float)
    set_piece = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)
    flight = features.get("ball_flight_candidate", pd.Series(False, index=features.index)).to_numpy(bool).copy()
    flight |= np.isfinite(speed) & (speed >= 8.0) & (closest_distance > maximum_control_distance_m)
    scenarios = classify_scenarios(
        features,
        minimum_control_persistence=minimum_control_persistence,
        maximum_control_distance_m=maximum_control_distance_m,
        minimum_control_margin_m=minimum_control_margin_m,
        maximum_control_speed_mps=maximum_control_speed_mps,
    )
    accepted_control = ["CONTROLLED", "PRESSURED_CONTROL"] if allow_pressured_control else ["CONTROLLED"]
    comfortable = np.isin(scenarios, accepted_control)

    output = np.full(n_frames, "X", dtype=object)
    dead = in_play & set_piece & (np.nan_to_num(speed, nan=0.0) <= 1.0)
    output[dead] = "D"
    output[in_play & comfortable & ~dead] = closest_team[in_play & comfortable & ~dead]

    # Scenario labels are evidence used to select the filling policy. They are
    # intentionally not returned as possession labels.
    ambiguous = in_play & ~comfortable & ~dead

    # Fill each ambiguous episode using nearest confident control on both
    # sides. If they disagree, the preceding value wins as configured by the
    # project policy; a following value is used only at the beginning.
    start = 0
    while start < n_frames:
        if not ambiguous[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and ambiguous[end]:
            end += 1
        before = None
        for index in range(start - 1, -1, -1):
            if output[index] in {"H", "A"}:
                before = output[index]
                break
        after = None
        for index in range(end, n_frames):
            if output[index] in {"H", "A"}:
                after = output[index]
                break
        fill = before if before in {"H", "A"} else after
        if fill in {"H", "A"}:
            output[start:end] = fill
        start = end

    if allow_receiver_switch:
        likely = features.get("likely_receiver_team", pd.Series(None, index=features.index)).to_numpy(object)
        projected = features.get(
            "projected_receiver_team", pd.Series(None, index=features.index)
        ).to_numpy(object)
        arrival = pd.to_numeric(
            features.get("arrival_time_margin_s", pd.Series(np.nan, index=features.index)), errors="coerce"
        ).to_numpy(float)
        anchor = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)
        strict_reception = features.get(
            "genuine_reception_strict_anchor", pd.Series(False, index=features.index)
        ).to_numpy(bool)
        forecast_confirmed = features.get(
            "anchor_forecast_confirmed", pd.Series(False, index=features.index)
        ).to_numpy(bool)
        forecast_team = features.get(
            "forecast_touch_team", pd.Series(None, index=features.index, dtype=object)
        ).to_numpy(object)
        switch_scenarios = np.isin(scenarios, ["BALL_IN_FLIGHT", "CONTESTED"])
        if allow_bidirectional_bridge:
            switch_scenarios = ambiguous.copy()
        start = 0
        while start < n_frames:
            if not switch_scenarios[start] or not in_play[start]:
                start += 1
                continue
            end = start + 1
            while end < n_frames and switch_scenarios[end] and in_play[end]:
                end += 1
            before = output[start - 1] if start else None
            future = end
            while future < n_frames and output[future] not in {"H", "A"}:
                future += 1
            after = output[future] if future < n_frames else None
            if before in {"H", "A"} and after in {"H", "A"} and before != after:
                horizon = max(1, int(receiver_horizon_frames))
                tail_start = max(start, end - horizon)
                receiver_votes = int(
                    np.sum(
                        ((likely[tail_start:end] == after) & (projected[tail_start:end] == after))
                        if require_receiver_path_agreement
                        else (likely[tail_start:end] == after)
                    )
                )
                receiver_margin = arrival[tail_start:end]
                if after == "H":
                    margin_support = np.nanmedian(receiver_margin[np.isfinite(receiver_margin)]) if np.isfinite(receiver_margin).any() else np.nan
                    margin_support = np.isfinite(margin_support) and margin_support >= 0.25
                else:
                    margin_support = np.nanmedian(receiver_margin[np.isfinite(receiver_margin)]) if np.isfinite(receiver_margin).any() else np.nan
                    margin_support = np.isfinite(margin_support) and margin_support <= -0.25
                future_anchor = np.any(anchor[end:min(n_frames, end + horizon)] == after)
                strict_support = np.any(strict_reception[tail_start:end] & (likely[tail_start:end] == after))
                receiver_persistence = int(np.sum(likely[tail_start:end] == after))
                forecast_support = np.any(
                    forecast_confirmed[end:min(n_frames, end + horizon)]
                    & (forecast_team[end:min(n_frames, end + horizon)] == after)
                )
                if (
                    receiver_votes >= 3
                    and margin_support
                    and future_anchor
                    and (not require_strict_reception or strict_support)
                    and (not require_anchor_forecast_confirmation or forecast_support)
                    and receiver_persistence >= int(minimum_receiver_persistence_frames)
                ):
                    output[max(start, end - 10):end] = after
            start = end

    output[in_play & (output == "X")] = "D"
    output[~in_play] = "X"
    # Preserve genuine dead-ball evidence after temporal filling.
    output[dead] = "D"
    output = _merge_internal_spikes(output, maximum_internal_spike)
    output[~in_play] = "X"
    return output.astype("<U1")


def bounded_bidirectional_scenario_resolver(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    max_anchor_gap_frames: int = 75,
    minimum_anchor_confidence: float = 0.85,
    minimum_interval_frames: int = 1,
    long_ball_only: bool = False,
    reject_internal_opposing_anchor: bool = True,
) -> np.ndarray:
    """Resolve only suspicious intervals using two-sided, independent anchors.

    The forward/backward lookup is allowed for this offline competition output,
    but it is bounded and does not use ground truth. A gap is changed only when
    both sides identify the same team and the side anchors are stable-control
    anchors. ``pressure_anchor_team`` is used as an independent confirmation
    when available; nearest-player evidence alone cannot authorize the fill.
    Cross-team gaps are deliberately left unchanged because a future anchor
    alone does not establish the exact switch frame. ``long_ball_only`` is a
    separate hypothesis: long gaps may be bridged only when most of the gap
    has supplied/derived flight evidence.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    scenarios = features.get(
        "scenario_context_causal",
        features.get("scenario_context", pd.Series("UNKNOWN", index=features.index)),
    ).to_numpy(dtype=object)
    suspicious = np.isin(
        scenarios,
        [
            "OCCLUDED", "BALL_IN_FLIGHT", "CONTESTED", "RECEPTION_CANDIDATE",
            "TRANSITION_CANDIDATE", "MISCONTROL_CANDIDATE", "DEFLECTION_CANDIDATE",
            "TRACKING_ARTIFACT",
        ],
    ) & in_play
    anchor = features.get(
        "selected_anchor_team",
        features.get("anchor_team", pd.Series(None, index=features.index)),
    ).to_numpy(object)
    anchor_confidence = pd.to_numeric(
        features.get(
            "selected_anchor_confidence",
            features.get("anchor_confidence_score", pd.Series(0.0, index=features.index)),
        ),
        errors="coerce",
    ).to_numpy(float)
    pressure_anchor = features.get(
        "pressure_anchor_team", pd.Series(None, index=features.index)
    ).to_numpy(object)

    reliable = np.isin(anchor, ["H", "A"]) & (
        np.nan_to_num(anchor_confidence, nan=0.0) >= minimum_anchor_confidence
    )
    # Require the independent pressure-based signal to agree when it exists.
    pressure_known = np.isin(pressure_anchor, ["H", "A"])
    reliable &= (~pressure_known) | (pressure_anchor == anchor)

    start = 0
    while start < n_frames:
        if not suspicious[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and suspicious[end]:
            end += 1
        interval_length = end - start
        if interval_length < int(minimum_interval_frames):
            start = end
            continue
        if interval_length > int(max_anchor_gap_frames):
            start = end
            continue
        if long_ball_only:
            flight = features.get(
                "ball_flight_candidate", pd.Series(False, index=features.index)
            ).to_numpy(bool)
            speed = pd.to_numeric(
                features.get("ball_speed_derived_mps", pd.Series(np.nan, index=features.index)),
                errors="coerce",
            ).to_numpy(float)
            flight_evidence = flight[start:end] | (np.nan_to_num(speed[start:end], nan=0.0) >= 8.0)
            if interval_length <= 25 or float(flight_evidence.mean()) < 0.5:
                start = end
                continue

        before = start - 1
        before_limit = max(-1, start - int(max_anchor_gap_frames) - 1)
        while before > before_limit and not reliable[before]:
            before -= 1
        after = end
        after_limit = min(n_frames, end + int(max_anchor_gap_frames))
        while after < after_limit and not reliable[after]:
            after += 1

        if (
            before > before_limit
            and after < after_limit
            and anchor[before] in {"H", "A"}
            and anchor[before] == anchor[after]
        ):
            internal_opposite = reliable[start:end] & (anchor[start:end] != anchor[before])
            if not reject_internal_opposing_anchor or not internal_opposite.any():
                output[start:end] = anchor[before]
        start = end

    output[~in_play] = "X"
    return output.astype("<U1")


def resolve_cross_team_bidirectional_switch(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    max_anchor_gap_frames: int = 75,
    minimum_anchor_confidence: float = 0.85,
    minimum_switch_score: float = 0.65,
    minimum_switch_run_frames: int = 5,
) -> np.ndarray:
    """Localise a cross-team switch inside a bounded uncertain interval.

    Same-team gaps are handled by ``bounded_bidirectional_scenario_resolver``.
    Here, a different team on each side is not enough to rewrite the gap: a
    sustained switch-confirmation run for the right-hand team must occur inside
    the interval.  Only the suffix beginning at that run is changed, avoiding
    the unsafe assumption that the whole gap belongs to the future team.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    scenarios = features.get(
        "scenario_context_causal",
        features.get("scenario_context", pd.Series("UNKNOWN", index=features.index)),
    ).to_numpy(dtype=object)
    suspicious = np.isin(
        scenarios,
        [
            "OCCLUDED", "BALL_IN_FLIGHT", "CONTESTED", "RECEPTION_CANDIDATE",
            "TRANSITION_CANDIDATE", "MISCONTROL_CANDIDATE", "DEFLECTION_CANDIDATE",
            "TRACKING_ARTIFACT",
        ],
    ) & in_play
    anchor = features.get(
        "selected_anchor_team",
        features.get("anchor_team", pd.Series(None, index=features.index)),
    ).to_numpy(object)
    confidence = pd.to_numeric(
        features.get(
            "selected_anchor_confidence",
            features.get("anchor_confidence_score", pd.Series(0.0, index=features.index)),
        ),
        errors="coerce",
    ).to_numpy(float)
    reliable = np.isin(anchor, ["H", "A"]) & (
        np.nan_to_num(confidence, nan=0.0) >= minimum_anchor_confidence
    )
    scores = pd.to_numeric(
        features.get("switch_confirmation_score", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    switch_team = features.get(
        "switch_confirmation_team", pd.Series(None, index=features.index)
    ).to_numpy(object)
    eligible_switch = (scores >= minimum_switch_score) & np.isin(switch_team, ["H", "A"])

    start = 0
    while start < n_frames:
        if not suspicious[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and suspicious[end]:
            end += 1
        before = start - 1
        before_limit = max(-1, start - int(max_anchor_gap_frames) - 1)
        while before > before_limit and not reliable[before]:
            before -= 1
        after = end
        after_limit = min(n_frames, end + int(max_anchor_gap_frames))
        while after < after_limit and not reliable[after]:
            after += 1
        if (
            before > before_limit
            and after < after_limit
            and anchor[before] in {"H", "A"}
            and anchor[after] in {"H", "A"}
            and anchor[before] != anchor[after]
        ):
            candidate_start = None
            cursor = start
            while cursor < end:
                if eligible_switch[cursor] and switch_team[cursor] == anchor[after]:
                    candidate_end = cursor + 1
                    while (
                        candidate_end < end
                        and eligible_switch[candidate_end]
                        and switch_team[candidate_end] == anchor[after]
                    ):
                        candidate_end += 1
                    if candidate_end - cursor >= int(minimum_switch_run_frames):
                        candidate_start = cursor
                        break
                    cursor = candidate_end
                else:
                    cursor += 1
            if candidate_start is not None:
                output[candidate_start:end] = anchor[after]
        start = end

    output[~in_play] = "X"
    return output.astype("<U1")


def resolve_reception_switch_inside_interval(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    max_anchor_gap_frames: int = 75,
    minimum_anchor_confidence: float = 0.85,
    minimum_reception_score: float = 0.55,
    minimum_post_control_frames: int = 3,
    require_strict_reception: bool = False,
    use_enhanced_reception_signal: bool = False,
    require_sustained_reception_signal: bool = False,
) -> np.ndarray:
    """Localise H/A switches only at an evidence-backed reception event.

    A different team anchor before and after a bounded suspicious interval is
    only a candidate transition.  The interval is rewritten from its first
    reception event for the future-anchor team, and only non-D/non-X frames
    are touched.  The event must agree with the future team and have both
    controlled-reception geometry and post-arrival anchor persistence.

    Same-team receptions are deliberately not changed: they describe a pass
    or through-ball continuation, not a possession switch.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    scenarios = features.get(
        "scenario_context_causal",
        features.get("scenario_context", pd.Series("UNKNOWN", index=features.index)),
    ).to_numpy(object)
    suspicious = np.isin(
        scenarios,
        [
            "OCCLUDED", "BALL_IN_FLIGHT", "CONTESTED", "RECEPTION_CANDIDATE",
            "TRANSITION_CANDIDATE", "MISCONTROL_CANDIDATE", "DEFLECTION_CANDIDATE",
            "TRACKING_ARTIFACT",
        ],
    ) & in_play
    anchor = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)
    confidence = pd.to_numeric(
        features.get("anchor_confidence_score", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    reliable = np.isin(anchor, ["H", "A"]) & (confidence >= float(minimum_anchor_confidence))
    receiver = features.get(
        "reception_receiver_team",
        features.get("likely_receiver_team", pd.Series(None, index=features.index)),
    ).to_numpy(object)
    reception_score = pd.to_numeric(
        features.get("controlled_reception_evidence_score", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    post_run = pd.to_numeric(
        features.get("post_reception_anchor_run_frames", pd.Series(0, index=features.index)),
        errors="coerce",
    ).fillna(0).to_numpy(int)
    genuine = features.get(
        "genuine_reception_strict_anchor" if require_strict_reception else "genuine_reception_anchor",
        pd.Series(False, index=features.index),
    ).to_numpy(bool)
    opponent_reception = features.get(
        "opponent_reception_candidate", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    enhanced = features.get(
        "sustained_reception_signal" if require_sustained_reception_signal else "controlled_reception_signal",
        pd.Series(False, index=features.index),
    ).to_numpy(bool)
    enhanced_team = features.get(
        "reception_control_team_geometry", pd.Series(None, index=features.index)
    ).to_numpy(object)

    start = 0
    while start < n_frames:
        if not suspicious[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and suspicious[end]:
            end += 1
        before = start - 1
        before_limit = max(-1, start - int(max_anchor_gap_frames) - 1)
        while before > before_limit and not reliable[before]:
            before -= 1
        after = end
        after_limit = min(n_frames, end + int(max_anchor_gap_frames))
        while after < after_limit and not reliable[after]:
            after += 1
        if before <= before_limit or after >= after_limit or anchor[before] == anchor[after]:
            start = end
            continue
        future_team = anchor[after]
        if use_enhanced_reception_signal:
            event_mask = (
                enhanced
                & (enhanced_team == future_team)
                & (reception_score >= float(minimum_reception_score))
                & (post_run >= int(minimum_post_control_frames))
            )
        else:
            event_mask = (
                genuine
                & opponent_reception
                & (receiver == future_team)
                & (reception_score >= float(minimum_reception_score))
                & (post_run >= int(minimum_post_control_frames))
            )
        event_positions = np.flatnonzero(event_mask[start:end])
        if event_positions.size:
            switch = start + int(event_positions[0])
            release_end = min(end, n_frames)
            positions = np.flatnonzero(
                ~np.isin(output[switch:release_end], ["D", "X"])
            )
            if positions.size:
                output[switch + positions] = future_team
        start = end

    output[~in_play] = "X"
    return output.astype("<U1")


def reception_transition_alignment_audit(
    features: pd.DataFrame,
    *,
    max_anchor_gap_frames: int = 75,
    minimum_anchor_confidence: float = 0.85,
) -> pd.DataFrame:
    """Audit whether reception candidates sit between opposite anchors.

    The result is diagnostic only.  It answers whether a candidate reception
    has the structural evidence needed by a transition resolver, without
    changing any submission labels.
    """
    n_frames = len(features)
    anchor = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)
    confidence = pd.to_numeric(
        features.get("anchor_confidence_score", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    reliable = np.isin(anchor, ["H", "A"]) & (confidence >= float(minimum_anchor_confidence))
    candidate = (
        features.get("controlled_reception_signal", pd.Series(False, index=features.index)).to_numpy(bool)
        | features.get("sustained_reception_signal", pd.Series(False, index=features.index)).to_numpy(bool)
        | features.get("short_touch_signal", pd.Series(False, index=features.index)).to_numpy(bool)
        | features.get("genuine_reception_anchor", pd.Series(False, index=features.index)).to_numpy(bool)
    )
    frame = features.get("frame", pd.Series(np.arange(n_frames), index=features.index)).to_numpy()
    candidate_team = features.get(
        "reception_control_team_geometry", pd.Series(None, index=features.index)
    ).to_numpy(object)
    scenario = features.get(
        "scenario_context_causal",
        features.get("scenario_context", pd.Series("UNKNOWN", index=features.index)),
    ).to_numpy(object)
    rows: list[dict[str, object]] = []
    for position in np.flatnonzero(candidate):
        before = position - 1
        before_limit = max(-1, position - int(max_anchor_gap_frames) - 1)
        while before > before_limit and not reliable[before]:
            before -= 1
        after = position + 1
        after_limit = min(n_frames, position + int(max_anchor_gap_frames) + 1)
        while after < after_limit and not reliable[after]:
            after += 1
        before_team = anchor[before] if before > before_limit else None
        after_team = anchor[after] if after < after_limit else None
        rows.append(
            {
                "frame": frame[position],
                "position": position,
                "candidate_team": candidate_team[position],
                "scenario_context_causal": scenario[position],
                "before_anchor_frame": frame[before] if before > before_limit else None,
                "before_anchor_team": before_team,
                "after_anchor_frame": frame[after] if after < after_limit else None,
                "after_anchor_team": after_team,
                "opposite_anchor_pair": before_team in {"H", "A"}
                and after_team in {"H", "A"}
                and before_team != after_team,
                "candidate_matches_after_team": candidate_team[position] == after_team,
            }
        )
    return pd.DataFrame(rows)


def resolve_aligned_reception_switch(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    max_anchor_gap_frames: int = 75,
    minimum_anchor_confidence: float = 0.85,
    minimum_reception_score: float = 0.55,
    use_sustained_signal: bool = False,
    use_team_context: bool = False,
    use_physics_veto: bool = False,
    minimum_transition_trust: float | None = None,
    require_transition_hypothesis: bool = False,
    minimum_transition_activity: float | None = None,
) -> np.ndarray:
    """Resolve an opposite-anchor interval from aligned reception evidence.

    Unlike scenario-gated decoders, this searches directly around reception
    candidates.  The before/after anchor pair is the guard; the candidate's
    geometry identifies the transition frame.  It never changes ``D`` or ``X``.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    anchor = features.get(
        "selected_anchor_team",
        features.get("anchor_team", pd.Series(None, index=features.index)),
    ).to_numpy(object)
    confidence = pd.to_numeric(
        features.get(
            "selected_anchor_confidence",
            features.get("anchor_confidence_score", pd.Series(0.0, index=features.index)),
        ),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    reliable = np.isin(anchor, ["H", "A"]) & (confidence >= float(minimum_anchor_confidence))
    candidate = features.get(
        "sustained_reception_signal" if use_sustained_signal else "controlled_reception_signal",
        pd.Series(False, index=features.index),
    ).to_numpy(bool).copy()
    score = pd.to_numeric(
        features.get("controlled_reception_signal_score", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    candidate &= score >= float(minimum_reception_score)
    if use_physics_veto:
        # A controlled-reception candidate must not be created by the
        # departure impulse itself. Positive acceleration and direction-change
        # vetoes remain evidence for a pass/deflection/second kick instead.
        candidate &= ~features.get(
            "sharp_acceleration_pass_veto", pd.Series(False, index=features.index)
        ).fillna(False).to_numpy(bool)
        candidate &= ~features.get(
            "deflection_direction_change_veto", pd.Series(False, index=features.index)
        ).fillna(False).to_numpy(bool)
    candidate_team = features.get(
        "reception_control_team_geometry", pd.Series(None, index=features.index)
    ).to_numpy(object)
    if minimum_transition_trust is not None and "transition_trust_score" in features:
        trust = pd.to_numeric(
            features["transition_trust_score"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        candidate &= trust >= float(minimum_transition_trust)
    if require_transition_hypothesis:
        hypothesis_state = features.get(
            "transition_hypothesis_state", pd.Series("NO_SWITCH", index=features.index)
        ).to_numpy(object)
        hypothesis_team = features.get(
            "transition_hypothesis_team", pd.Series(None, index=features.index)
        ).to_numpy(object)
        candidate &= hypothesis_state == "SWITCH_CANDIDATE"
        candidate &= candidate_team == hypothesis_team
    if minimum_transition_activity is not None:
        activity = pd.to_numeric(
            features.get("transition_hypothesis_activity", pd.Series(0.0, index=features.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(float)
        candidate &= activity >= float(minimum_transition_activity)
    context_team = features.get(
        "smart_transition_attack_team",
        features.get(
            "smart_progression_or_build_up_team",
            pd.Series(None, index=features.index, dtype=object),
        ),
    ).to_numpy(object)

    for position in np.flatnonzero(candidate):
        before = position - 1
        before_limit = max(-1, position - int(max_anchor_gap_frames) - 1)
        while before > before_limit and not reliable[before]:
            before -= 1
        after = position + 1
        after_limit = min(n_frames, position + int(max_anchor_gap_frames) + 1)
        while after < after_limit and not reliable[after]:
            after += 1
        if before <= before_limit or after >= after_limit:
            continue
        if anchor[before] == anchor[after] or candidate_team[position] != anchor[after]:
            continue
        if use_team_context:
            # Smart Tagging is only a weak directional cue. Require the
            # future receiver team to agree somewhere in this episode; the
            # tag alone never creates an H/A switch.
            context_start = max(0, before + 1)
            context_end = min(n_frames, after)
            if not np.any(context_team[context_start:context_end] == anchor[after]):
                continue
        # Only the interval from the evidence-backed transition to the future
        # anchor is rewritten; dead-ball and outside-graded frames stay locked.
        end = min(after, n_frames)
        positions = np.flatnonzero(~np.isin(output[position:end], ["D", "X"]))
        if positions.size:
            output[position + positions] = anchor[after]

    output[~in_play] = "X"
    return output.astype("<U1")


def apply_switch_confirmation(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    minimum_score: float = 0.65,
    minimum_run_frames: int = 5,
) -> np.ndarray:
    """Apply only sustained, scenario-gated switch-confirmation evidence."""
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    scenario = features.get(
        "scenario_context_causal",
        features.get("scenario_context", pd.Series("UNKNOWN", index=features.index)),
    ).to_numpy(object)
    eligible = np.isin(
        scenario,
        ["BALL_IN_FLIGHT", "CONTESTED", "RECEPTION_CANDIDATE", "MISCONTROL_CANDIDATE", "TRANSITION_CANDIDATE"],
    ) & in_play
    scores = pd.to_numeric(
        features.get("switch_confirmation_score", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    candidate = features.get(
        "switch_confirmation_team", pd.Series(None, index=features.index)
    ).to_numpy(object)
    candidate = np.where(scores >= minimum_score, candidate, None)

    start = 0
    while start < n_frames:
        if not eligible[start] or candidate[start] not in {"H", "A"}:
            start += 1
            continue
        end = start + 1
        while (
            end < n_frames
            and eligible[end]
            and candidate[end] == candidate[start]
        ):
            end += 1
        before = output[start - 1] if start else None
        if end - start >= int(minimum_run_frames) and before in {"H", "A"}:
            if candidate[start] != before:
                output[start:end] = candidate[start]
        start = end

    output[~in_play] = "X"
    return output.astype("<U1")


def retain_no_switch_intervals(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    minimum_no_switch_probability: float = 0.65,
    minimum_interval_support: float = 0.60,
    maximum_interval_frames: int = 300,
    minimum_prior_team_run_frames: int = 1,
    minimum_bootstrap_no_switch_lower: float | None = None,
) -> np.ndarray:
    """Retain the prior team through an ambiguous live-ball interval.

    A nearest-player observation can change before possession has changed. The
    transition layer already exposes that possibility as ``NO_SWITCH`` versus
    ``SWITCH_CANDIDATE``, but the earlier decoders only used the candidate
    branch. This resolver makes the alternative operational: when a proposed
    H/A run is mostly supported by ``NO_SWITCH`` evidence, and there is no
    strong new control anchor inside the run, preserve the preceding team.

    This is an interval hypothesis resolver, not a label smoother and not a
    ground-truth lookup. It uses only derived observations, the existing
    scenario detector, and the rule-based transition evidence. Explicit D/X
    frames and strong new anchors remain protected. The result is deliberately
    exposed as an ablation before promotion to the organiser-facing model.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)

    scenario = features.get(
        "scenario_context_causal",
        features.get("scenario_context", pd.Series("UNKNOWN", index=features.index)),
    ).to_numpy(object)
    live_scenarios = np.isin(
        scenario,
        [
            "BALL_IN_FLIGHT",
            "CONTESTED",
            "RECEPTION_CANDIDATE",
            "MISCONTROL_CANDIDATE",
            "PRESSURED_CONTROL",
            "CONTROLLED",
            "UNKNOWN",
            "TRANSITION_CANDIDATE",
        ],
    )
    hypothesis = features.get(
        "transition_hypothesis_state",
        pd.Series("NO_SWITCH", index=features.index),
    ).to_numpy(object)
    p_no_switch = pd.to_numeric(
        features.get(
            "transition_hypothesis_p_no_switch",
            pd.Series(1.0, index=features.index),
        ),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    anchor_team = features.get(
        "selected_anchor_team",
        features.get("anchor_team", pd.Series(None, index=features.index)),
    ).to_numpy(object)
    anchor_confidence = pd.to_numeric(
        features.get(
            "selected_anchor_confidence",
            features.get("anchor_confidence_score", pd.Series(0.0, index=features.index)),
        ),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    reception_signal = features.get(
        "controlled_reception_signal", pd.Series(False, index=features.index)
    ).fillna(False).to_numpy(bool)
    reception_score = pd.to_numeric(
        features.get(
            "controlled_reception_signal_score",
            pd.Series(0.0, index=features.index),
        ),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    bootstrap_lower = pd.to_numeric(
        features.get(
            "transition_bootstrap_no_switch_lower",
            pd.Series(np.nan, index=features.index),
        ),
        errors="coerce",
    ).to_numpy(float)

    def prior_run_length(position: int, team: str) -> int:
        cursor = position - 1
        while cursor >= 0 and output[cursor] == team:
            cursor -= 1
        return position - cursor - 1

    position = 1
    while position < n_frames:
        previous = output[position - 1]
        current = output[position]
        if previous not in {"H", "A"} or current not in {"H", "A"} or previous == current:
            position += 1
            continue

        end = position + 1
        while end < n_frames and output[end] == current:
            end += 1
        if end - position > int(maximum_interval_frames):
            position = end
            continue

        interval = slice(position, end)
        support = (
            in_play[interval]
            & live_scenarios[interval]
            & (hypothesis[interval] == "NO_SWITCH")
            & (p_no_switch[interval] >= float(minimum_no_switch_probability))
        )
        if minimum_bootstrap_no_switch_lower is not None:
            support &= (
                np.isfinite(bootstrap_lower[interval])
                & (
                    bootstrap_lower[interval]
                    >= float(minimum_bootstrap_no_switch_lower)
                )
            )
        support_fraction = float(np.mean(support)) if end > position else 0.0
        strong_anchor = np.any(
            (anchor_team[interval] == current)
            & (anchor_confidence[interval] >= 0.85)
        )
        strong_reception = np.any(
            (reception_signal[interval] & (reception_score[interval] >= 0.55))
        )
        enough_prior_context = prior_run_length(position, previous) >= int(
            minimum_prior_team_run_frames
        )
        if (
            support_fraction >= float(minimum_interval_support)
            and enough_prior_context
            and not strong_anchor
            and not strong_reception
        ):
            output[position:end] = previous
        position = end

    output[~in_play] = "X"
    return output.astype("<U1")


def retain_weak_contact_arrival_handoffs(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    boundary_frames: int = 25,
    minimum_contact_score: float = 0.45,
    minimum_consensus_score: float = 0.42,
    minimum_handoff_score: float = 0.58,
    minimum_no_switch_probability: float = 0.70,
) -> np.ndarray:
    """Guard only a local handoff boundary when contact/arrival disagree.

    The existing interval resolver can preserve a whole proposed H/A run when
    the evidence is ambiguous. This diagnostic is narrower: it examines only
    the first ``boundary_frames`` of each proposed handoff and asks whether
    ball contact, team-level arrival consensus, and receiver control agree.
    Without that agreement, and without a new strong anchor, it retains the
    prior team locally. It never changes D/X and never creates a new team.

    The rule is a signal-based transition test, not a supervised correction.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    scenario = features.get(
        "scenario_context_causal",
        features.get("scenario_context", pd.Series("UNKNOWN", index=features.index)),
    ).to_numpy(object)
    live = np.isin(
        scenario,
        [
            "BALL_IN_FLIGHT", "CONTESTED", "RECEPTION_CANDIDATE",
            "MISCONTROL_CANDIDATE", "PRESSURED_CONTROL", "CONTROLLED",
            "UNKNOWN", "TRANSITION_CANDIDATE",
        ],
    )
    p_no_switch = pd.to_numeric(
        features.get("transition_hypothesis_p_no_switch", pd.Series(1.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    anchor_team = features.get(
        "selected_anchor_team", features.get("anchor_team", pd.Series(None, index=features.index))
    ).to_numpy(object)
    anchor_confidence = pd.to_numeric(
        features.get("selected_anchor_confidence", features.get("anchor_confidence_score", pd.Series(0.0, index=features.index))),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    candidate = features.get(
        "transition_candidate_team",
        features.get("reception_control_team_geometry", pd.Series(None, index=features.index)),
    ).to_numpy(object)
    direction = pd.to_numeric(
        features.get("ball_direction_change_rad", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    deceleration = pd.to_numeric(
        features.get("reception_ball_deceleration_mps", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    contact = np.maximum(
        np.clip(deceleration / 4.0, 0.0, 1.0),
        np.clip(direction / 1.2, 0.0, 1.0),
    )
    receiver_score = pd.to_numeric(
        features.get("controlled_reception_signal_score", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    consensus_h = pd.to_numeric(
        features.get("h_arrival_consensus_score", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    consensus_a = pd.to_numeric(
        features.get("a_arrival_consensus_score", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)

    position = 1
    while position < n_frames:
        previous = output[position - 1]
        current = output[position]
        if previous not in {"H", "A"} or current not in {"H", "A"} or previous == current:
            position += 1
            continue
        end = position + 1
        while end < n_frames and output[end] == current:
            end += 1
        boundary_end = min(end, position + int(boundary_frames))
        boundary = slice(position, boundary_end)
        candidate_consensus = np.where(current == "H", consensus_h, consensus_a)
        handoff_score = (
            0.45 * contact
            + 0.35 * candidate_consensus
            + 0.20 * receiver_score
        )
        strong_anchor = np.any(
            (anchor_team[boundary] == current) & (anchor_confidence[boundary] >= 0.85)
        )
        has_candidate = candidate[boundary] == current
        coherent = (
            in_play[boundary]
            & live[boundary]
            & (p_no_switch[boundary] >= float(minimum_no_switch_probability))
            & has_candidate
            & (contact[boundary] >= float(minimum_contact_score))
            & (candidate_consensus[boundary] >= float(minimum_consensus_score))
            & (handoff_score[boundary] >= float(minimum_handoff_score))
        )
        # If the new team lacks a coherent local contact+arrival handoff and
        # no strong anchor protects it, retain only the boundary portion.
        if not strong_anchor and not np.any(coherent):
            output[position:boundary_end] = previous
        position = end
    output[~in_play] = "X"
    return output.astype("<U1")


def apply_switch_confirmation_hysteresis(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    minimum_score: float = 0.65,
    candidate_run_frames: int = 5,
    minimum_anchor_run_frames: int = 8,
    anchor_lookahead_frames: int = 75,
    merge_spikes: bool = False,
    maximum_spike_gap: int = 25,
) -> np.ndarray:
    """Keep a proposed switch only when a later stable control anchor agrees.

    The switch score proposes a possible change, but does not itself establish
    possession.  This hysteresis layer accepts that proposal only if the new
    team later has a contiguous, high-confidence anchor run.  It is deliberately
    parameterised by football-time windows rather than fitted to submission
    labels, and it never changes ``D`` or ``X`` directly.
    """
    base = np.asarray(base_labels, dtype="<U1")
    proposed = apply_switch_confirmation(
        features,
        base,
        minimum_score=minimum_score,
        minimum_run_frames=candidate_run_frames,
    )
    output = base.copy()
    n_frames = len(features)
    anchor_team = features.get(
        "anchor_team", pd.Series(None, index=features.index)
    ).to_numpy(object)
    anchor_confidence = pd.to_numeric(
        features.get(
            "anchor_confidence_score", pd.Series(0.0, index=features.index)
        ),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    anchor_ok = np.isin(anchor_team, ["H", "A"]) & (anchor_confidence >= 0.85)

    start = 0
    while start < n_frames:
        changed = proposed[start] != base[start] and proposed[start] in {"H", "A"}
        if not changed:
            start += 1
            continue
        end = start + 1
        while (
            end < n_frames
            and proposed[end] == proposed[start]
            and proposed[end] != base[end]
        ):
            end += 1

        candidate_team = proposed[start]
        search_end = min(n_frames, end + int(anchor_lookahead_frames))
        confirmed = False
        cursor = end
        while cursor < search_end:
            if not (anchor_ok[cursor] and anchor_team[cursor] == candidate_team):
                cursor += 1
                continue
            anchor_end = cursor + 1
            while (
                anchor_end < search_end
                and anchor_ok[anchor_end]
                and anchor_team[anchor_end] == candidate_team
            ):
                anchor_end += 1
            if anchor_end - cursor >= int(minimum_anchor_run_frames):
                confirmed = True
                break
            cursor = anchor_end

        if confirmed:
            output[start:end] = proposed[start:end]
        start = end

    if merge_spikes:
        output = _merge_internal_spikes(output, maximum_gap=int(maximum_spike_gap))
    output[~graded_mask(n_frames)] = "X"
    return output.astype("<U1")


def apply_stable_reception_episode_switch(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    minimum_episode_frames: int = 3,
    minimum_stability: float = 0.75,
    minimum_receiver_confidence: float = 0.90,
    minimum_receiver_near_ball_fraction: float = 0.0,
) -> np.ndarray:
    """Apply only stable possible-turnover reception episodes.

    Same-team continuation episodes intentionally leave labels untouched.  A
    turnover episode can change only in-play H/A frames and must have a stable
    receiver team for the entire aggregated episode.  This is an experiment,
    not a claim that every soft episode is a true turnover.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    outcome = features.get(
        "reception_episode_outcome", pd.Series("UNRESOLVED", index=features.index)
    ).to_numpy(object)
    length = pd.to_numeric(
        features.get("reception_episode_length_frames", pd.Series(0, index=features.index)),
        errors="coerce",
    ).fillna(0).to_numpy(int)
    stability = pd.to_numeric(
        features.get("reception_episode_stability", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    receiver_confidence = pd.to_numeric(
        features.get(
            "reception_episode_receiver_confidence", pd.Series(0.0, index=features.index)
        ),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    receiver_near_fraction = pd.to_numeric(
        features.get(
            "reception_episode_receiver_near_ball_fraction",
            pd.Series(0.0, index=features.index),
        ),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    receiver = features.get(
        "reception_receiver_team", pd.Series(None, index=features.index)
    ).to_numpy(object)
    eligible = (
        in_play
        & (outcome == "POSSIBLE_TURNOVER")
        & (length >= int(minimum_episode_frames))
        & (stability >= float(minimum_stability))
        & (receiver_confidence >= float(minimum_receiver_confidence))
        & (receiver_near_fraction >= float(minimum_receiver_near_ball_fraction))
        & np.isin(receiver, ["H", "A"])
    )
    output[eligible] = receiver[eligible]
    output[~in_play] = "X"
    return output.astype("<U1")


def add_restart_phase_features(
    features: pd.DataFrame,
    *,
    minimum_control_run: int = 8,
    maximum_live_search_frames: int = 75,
    use_tracking_fallback: bool = False,
    minimum_anchor_confidence: float = 0.85,
) -> pd.DataFrame:
    """Expose restart timing as evidence rather than possession labels.

    The phase is one of ``OUTSIDE_RESTART``, ``PRE_RESTART``,
    ``ACTIVE_UNCONTROLLED``, or ``CONTROL_CONFIRMED``.  Smart Tagging and
    ball activity open the phase; a sustained team anchor closes the uncertain
    part.  The function never writes H/A/D/X labels and is therefore safe to
    pass to a later interval resolver or constrained path decoder.
    """
    out = features.copy()
    n = len(out)
    context = out.get(
        "restart_context",
        out.get("smart_set_piece", pd.Series(False, index=out.index)),
    ).fillna(False).to_numpy(bool)
    active = out.get(
        "restart_ball_active",
        out.get("restart_live_boundary_anchor", pd.Series(False, index=out.index)),
    ).fillna(False).to_numpy(bool)
    live_anchor = out.get(
        "restart_live_boundary_anchor", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    hinted = out.get(
        "restart_hinted_control_anchor", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    fallback = out.get(
        "restart_fallback_control_anchor", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    anchor_team = out.get(
        "anchor_team", pd.Series(None, index=out.index, dtype=object)
    ).to_numpy(object)
    confidence = pd.to_numeric(
        out.get("anchor_confidence_score", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    eligible_control = hinted | (fallback if use_tracking_fallback else False)
    eligible_control &= np.isin(anchor_team, ["H", "A"])
    eligible_control &= confidence >= float(minimum_anchor_confidence)

    phase = np.full(n, "OUTSIDE_RESTART", dtype=object)
    phase[context & ~active] = "PRE_RESTART"
    phase[context & active] = "ACTIVE_UNCONTROLLED"
    control_confirmed = np.zeros(n, dtype=bool)
    control_team = np.full(n, None, dtype=object)
    control_confidence = np.zeros(n, dtype=float)

    # Search only within a restart episode plus a bounded post-tag window.
    # This is timing evidence; it does not carry the team through the window.
    start = 0
    while start < n:
        if not context[start]:
            start += 1
            continue
        end = start + 1
        while end < n and context[end]:
            end += 1
        live_positions = np.flatnonzero(live_anchor[start:end] | active[start:end])
        if live_positions.size == 0:
            start = end
            continue
        live_start = start + int(live_positions[0])
        search_end = min(n, end + int(maximum_live_search_frames))
        cursor = live_start
        while cursor < search_end:
            if not eligible_control[cursor]:
                cursor += 1
                continue
            team = anchor_team[cursor]
            run_end = cursor + 1
            while (
                run_end < search_end
                and eligible_control[run_end]
                and anchor_team[run_end] == team
            ):
                run_end += 1
            if run_end - cursor >= int(minimum_control_run):
                control_confirmed[cursor:run_end] = True
                control_team[cursor:run_end] = team
                control_confidence[cursor:run_end] = confidence[cursor:run_end]
                phase[cursor:run_end] = "CONTROL_CONFIRMED"
                break
            cursor = run_end
        start = end

    out["restart_phase"] = phase
    out["restart_phase_active"] = phase != "OUTSIDE_RESTART"
    out["restart_phase_control_confirmed"] = control_confirmed
    out["restart_phase_team"] = control_team
    out["restart_phase_confidence"] = control_confidence
    out["restart_phase_switch_allowed"] = np.isin(
        phase, ["CONTROL_CONFIRMED", "OUTSIDE_RESTART"]
    )
    out["restart_phase_model"] = "observation_only_restart_phase_v1"
    return out


def restart_phase_interval_resolver(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    minimum_control_run: int = 8,
    maximum_live_search_frames: int = 75,
    use_tracking_fallback: bool = False,
    minimum_anchor_confidence: float = 0.85,
) -> np.ndarray:
    """Resolve labels using restart phase as a transition constraint.

    ``PRE_RESTART`` and ``ACTIVE_UNCONTROLLED`` block a new H/A handoff but do
    not invent D.  ``CONTROL_CONFIRMED`` permits the existing resolver's H/A
    evidence, and can release a D frame only when a sustained anchor names the
    team.  This is intentionally a shared constraint layer rather than a
    restart-specific possession classifier.
    """
    phased = add_restart_phase_features(
        features,
        minimum_control_run=minimum_control_run,
        maximum_live_search_frames=maximum_live_search_frames,
        use_tracking_fallback=use_tracking_fallback,
        minimum_anchor_confidence=minimum_anchor_confidence,
    )
    output = np.asarray(base_labels, dtype="<U1").copy()
    in_play = graded_mask(len(phased))
    phase = phased["restart_phase"].to_numpy(object)
    control_team = phased["restart_phase_team"].to_numpy(object)

    # A restart is not allowed to create a new live owner while the ball is
    # active but control is still unconfirmed.  Preserve the preceding live
    # team only when the supplied candidate actually proposes a handoff.
    for position in range(1, len(output)):
        if phase[position] not in {"PRE_RESTART", "ACTIVE_UNCONTROLLED"}:
            continue
        if output[position] not in {"H", "A"}:
            continue
        previous = output[position - 1]
        if previous in {"H", "A"} and previous != output[position]:
            output[position] = previous

    # A D label can be released at the first confirmed control frame, but only
    # with the independent sustained anchor team.  No restart hint alone can
    # create possession.
    release = (
        (phase == "CONTROL_CONFIRMED")
        & (output == "D")
        & np.isin(control_team, ["H", "A"])
        & in_play
    )
    output[release] = control_team[release]
    output[~in_play] = "X"
    return output.astype("<U1")


def restart_phase_boundary_interval_resolver(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    minimum_control_run: int = 8,
    maximum_live_search_frames: int = 75,
    maximum_boundary_guard_frames: int = 25,
    use_tracking_fallback: bool = False,
    minimum_anchor_confidence: float = 0.85,
) -> np.ndarray:
    """Apply restart constraints locally around a proposed label boundary.

    The broader restart-phase resolver is useful for auditing, but it can
    suppress an entire H/A run while a restart phase remains active.  This
    variant limits the constraint to the boundary where the base candidate
    first proposes a new owner.  A previous H/A owner is retained only until
    confirmed control or the bounded guard expires.  A D predecessor is kept
    until confirmed control when available, but is also bounded so an
    unconfirmed restart cannot create an unbounded dead-ball spell.

    This function is intentionally a resolver wrapper: it consumes restart
    evidence and a pre-existing candidate, and never creates an H/A owner from
    restart context alone.
    """
    phased = add_restart_phase_features(
        features,
        minimum_control_run=minimum_control_run,
        maximum_live_search_frames=maximum_live_search_frames,
        use_tracking_fallback=use_tracking_fallback,
        minimum_anchor_confidence=minimum_anchor_confidence,
    )
    base = np.asarray(base_labels, dtype="<U1")
    output = base.copy()
    in_play = graded_mask(len(phased))
    phase = phased["restart_phase"].to_numpy(object)
    control_team = phased["restart_phase_team"].to_numpy(object)
    live_phases = {"PRE_RESTART", "ACTIVE_UNCONTROLLED"}
    guard = max(0, int(maximum_boundary_guard_frames))
    search = max(0, int(maximum_live_search_frames))

    # Inspect only boundaries in the supplied candidate.  This is the key
    # difference from restart_phase_interval_resolver: later frames in the
    # proposed H/A run are not repeatedly treated as fresh transitions.
    for position in range(1, len(base)):
        previous = base[position - 1]
        candidate = base[position]
        if candidate not in {"H", "A"} or previous not in {"H", "A", "D"}:
            continue
        if candidate == previous or phase[position] not in live_phases:
            continue

        limit = min(len(base), position + search + 1)
        control_positions = np.flatnonzero(
            (phase[position:limit] == "CONTROL_CONFIRMED")
            & (control_team[position:limit] == candidate)
        )
        control_position = (
            position + int(control_positions[0]) if control_positions.size else None
        )
        if control_position is None:
            boundary_end = min(len(base), position + guard)
        else:
            boundary_end = min(len(base), control_position)
            if boundary_end > position + guard:
                boundary_end = min(len(base), position + guard)

        if previous in {"H", "A"}:
            output[position:boundary_end] = previous
        elif previous == "D":
            output[position:boundary_end] = "D"

    output[~in_play] = "X"
    return output.astype("<U1")


def restart_provisional_labels(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    minimum_active_run: int = 3,
    minimum_dead_run: int = 5,
    control_run: int = 8,
    post_restart_hold_frames: int = 25,
    post_restart_hold_frames_by_type: dict[str, int] | None = None,
) -> np.ndarray:
    """Apply a bounded restart prior without treating a tag as ground truth.

    Before a tagged restart becomes active, ``D`` is retained.  Once sustained
    ball activity is detected, the tagged team (when available) is carried as
    a provisional team.  Throw-ins and kickoffs have no reliable team hint in
    the supplied XML, so they use the first sustained post-restart anchor.
    A competing team can replace the provisional team only after a sustained
    anchor run.  The intervention is bounded to the tagged interval plus a
    short post-interval hold; normal decoder output is untouched elsewhere.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    context = features.get(
        "smart_set_piece", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    active = features.get(
        "restart_ball_active", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    anchor = features.get(
        "anchor_team", pd.Series(None, index=features.index)
    ).to_numpy(dtype=object)
    hint = features.get(
        "restart_team_hint", pd.Series(None, index=features.index)
    ).to_numpy(dtype=object)
    restart_type = features.get(
        "restart_type", pd.Series(None, index=features.index)
    ).to_numpy(dtype=object)
    hold_by_type = post_restart_hold_frames_by_type or {}

    start = 0
    while start < n_frames:
        if not context[start] or not in_play[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and context[end] and in_play[end]:
            end += 1

        # Find the first sustained live-ball run inside the approximate tag.
        live_start = None
        cursor = start
        while cursor < end:
            if not active[cursor]:
                cursor += 1
                continue
            live_end = cursor + 1
            while live_end < end and active[live_end]:
                live_end += 1
            if live_end - cursor >= minimum_active_run:
                live_start = cursor
                break
            cursor = live_end

        if live_start is None:
            start = end
            continue

        # The part before the inferred kick remains dead-ball.  This prevents
        # the broad Smart Tagging interval from switching to H/A too early.
        # Advantage is protected here: if play stayed active, the first
        # sustained active run begins at the event boundary and this slice is
        # empty, so no D label is invented.
        if live_start - start >= minimum_dead_run:
            output[start:live_start] = "D"

        event_hint = next(
            (value for value in hint[start:end] if value in {"H", "A"}),
            None,
        )
        event_type = next(
            (value for value in restart_type[start:end] if value in hold_by_type),
            None,
        )
        hold_frames = int(hold_by_type.get(event_type, post_restart_hold_frames))
        search_end = min(n_frames, end + max(0, hold_frames))
        sustained = np.zeros(n_frames, dtype=bool)
        for team in ("H", "A"):
            positions = np.flatnonzero(anchor[live_start:search_end] == team)
            if positions.size == 0:
                continue
            run_start = int(positions[0])
            previous = run_start
            for position in positions[1:]:
                position = int(position)
                if position != previous + 1:
                    if previous - run_start + 1 >= control_run:
                        sustained[live_start + run_start : live_start + previous + 1] = True
                    run_start = position
                previous = position
            if previous - run_start + 1 >= control_run:
                sustained[live_start + run_start : live_start + previous + 1] = True

        first_sustained = np.flatnonzero(sustained[live_start:search_end])
        inferred_team = event_hint
        if inferred_team is None and first_sustained.size:
            inferred_team = anchor[live_start + int(first_sustained[0])]
        if inferred_team not in {"H", "A"}:
            start = end
            continue

        # Carry the restarting side through the bounded live transition.
        output[live_start:search_end] = inferred_team

        # A sustained opposing anchor is the only permitted handoff.  This is
        # deliberately stricter than nearest-player proximity.
        opposite = "A" if inferred_team == "H" else "H"
        opposite_positions = np.flatnonzero(
            sustained[live_start:search_end]
            & (anchor[live_start:search_end] == opposite)
        )
        if opposite_positions.size:
            switch = live_start + int(opposite_positions[0])
            output[switch:search_end] = opposite
        start = end

    output[~in_play] = "X"
    return output


def restart_control_boundary_labels(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    minimum_control_run: int = 8,
    maximum_live_search_frames: int = 75,
    use_tracking_fallback: bool = False,
    minimum_anchor_confidence: float = 0.85,
) -> np.ndarray:
    """Release restart ``D`` labels only after stable post-restart control.

    Smart Tagging identifies an approximate restart context, not the exact
    frame at which possession becomes controlled.  This resolver therefore
    keeps the supplied labels unchanged except for ``D`` frames inside a
    bounded restart interval.  It releases those frames at the first run of
    ``minimum_control_run`` consecutive, same-team stable-control anchors
    after the ball becomes active.  Team-hinted anchors are preferred; the
    tracking fallback is optional because it is weaker evidence.

    The function is deliberately a boundary rule, not a possession classifier:
    it never invents H/A labels and never changes frames outside the restart
    context or the original ``D`` region.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    context = features.get(
        "restart_context", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    live = features.get(
        "restart_live_boundary_anchor", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    hinted = features.get(
        "restart_hinted_control_anchor", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    fallback = features.get(
        "restart_fallback_control_anchor", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    anchor = features.get(
        "anchor_team", pd.Series(None, index=features.index)
    ).to_numpy(dtype=object)
    confidence = pd.to_numeric(
        features.get(
            "anchor_confidence_score", pd.Series(0.0, index=features.index)
        ),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    eligible_control = hinted | (fallback if use_tracking_fallback else False)
    eligible_control &= np.isin(anchor, ["H", "A"])
    eligible_control &= confidence >= float(minimum_anchor_confidence)

    start = 0
    while start < n_frames:
        if not context[start] or not in_play[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and context[end] and in_play[end]:
            end += 1

        live_positions = np.flatnonzero(live[start:end])
        if live_positions.size == 0:
            start = end
            continue
        live_start = start + int(live_positions[0])
        search_end = min(n_frames, end + int(maximum_live_search_frames))

        control_start = None
        cursor = live_start
        while cursor < search_end:
            if not eligible_control[cursor]:
                cursor += 1
                continue
            run_team = anchor[cursor]
            run_end = cursor + 1
            while (
                run_end < search_end
                and eligible_control[run_end]
                and anchor[run_end] == run_team
            ):
                run_end += 1
            if run_end - cursor >= int(minimum_control_run):
                control_start = cursor
                break
            cursor = run_end

        if control_start is not None:
            # Only release existing D labels; H/A, X, and unrelated context
            # remain exactly as supplied by the selected baseline.
            release_end = min(control_start, end)
            d_positions = np.flatnonzero(output[live_start:release_end] == "D")
            if d_positions.size:
                output[live_start + d_positions] = anchor[control_start]
        start = end

    output[~in_play] = "X"
    return output.astype("<U1")


def hierarchical_restart_fill_labels(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    *,
    minimum_control_run: int = 8,
    maximum_live_search_frames: int = 75,
    minimum_anchor_confidence: float = 0.85,
    use_tracking_fallback: bool = False,
) -> np.ndarray:
    """Fill only uncertain live-ball frames while locking trusted ``D``/``X``.

    This is the hierarchy intended for offline decoding: dead-ball and
    outside-graded labels are protected; restart anchors may only fill frames
    in an explicitly uncertain live scenario before a stable controlled run.
    Existing confident H/A frames and all non-restart frames are left alone.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    context = features.get("restart_context", pd.Series(False, index=features.index)).to_numpy(bool)
    live = features.get("restart_live_boundary_anchor", pd.Series(False, index=features.index)).to_numpy(bool)
    hinted = features.get("restart_hinted_control_anchor", pd.Series(False, index=features.index)).to_numpy(bool)
    fallback = features.get("restart_fallback_control_anchor", pd.Series(False, index=features.index)).to_numpy(bool)
    anchor = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)
    confidence = pd.to_numeric(
        features.get("anchor_confidence_score", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    scenario = features.get(
        "scenario_context_causal",
        features.get("scenario_context", pd.Series("UNKNOWN", index=features.index)),
    ).to_numpy(object)
    uncertain = np.isin(
        scenario,
        ["UNKNOWN", "OCCLUDED", "BALL_IN_FLIGHT", "CONTESTED", "RECEPTION_CANDIDATE", "TRANSITION_CANDIDATE"],
    )
    eligible_control = hinted | (fallback if use_tracking_fallback else False)
    eligible_control &= np.isin(anchor, ["H", "A"])
    eligible_control &= confidence >= float(minimum_anchor_confidence)

    start = 0
    while start < n_frames:
        if not context[start] or not in_play[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and context[end] and in_play[end]:
            end += 1
        live_positions = np.flatnonzero(live[start:end])
        if live_positions.size == 0:
            start = end
            continue
        live_start = start + int(live_positions[0])
        search_end = min(n_frames, end + int(maximum_live_search_frames))
        control_start = None
        cursor = live_start
        while cursor < search_end:
            if not eligible_control[cursor]:
                cursor += 1
                continue
            team = anchor[cursor]
            run_end = cursor + 1
            while run_end < search_end and eligible_control[run_end] and anchor[run_end] == team:
                run_end += 1
            if run_end - cursor >= int(minimum_control_run):
                control_start = cursor
                control_team = team
                break
            cursor = run_end
        if control_start is not None:
            fill_end = min(control_start, end)
            positions = np.flatnonzero(
                uncertain[live_start:fill_end]
                & ~np.isin(output[live_start:fill_end], ["D", "X"])
            )
            if positions.size:
                output[live_start + positions] = control_team
        start = end
    output[~in_play] = "X"
    return output.astype("<U1")


def refine_dead_ball_labels(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    speed_threshold_mps: float = 2.0,
    minimum_candidate_run: int = 5,
    require_existing_d_adjacency: bool = False,
    maximum_player_activity_mps: float | None = None,
) -> np.ndarray:
    """Recover coherent dead-ball intervals without relabelling active play.

    Smart Tagging supplies the event context; low derived ball speed supplies
    local evidence. Flight, reception, and control evidence block a D label.
    Candidate runs must be long enough to be an interval, unless they already
    touch an existing D interval from the base decoder.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    smart = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)
    speed = pd.to_numeric(
        features.get("ball_speed_derived_mps", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    scenarios = features.get("scenario_context", pd.Series("UNKNOWN", index=features.index)).to_numpy(object)
    flight = features.get("ball_flight_candidate", pd.Series(False, index=features.index)).to_numpy(bool)
    reception = features.get("genuine_reception_anchor", pd.Series(False, index=features.index)).to_numpy(bool)
    player_activity = pd.to_numeric(
        features.get("player_activity_median_mps", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    blocked_context = np.isin(
        scenarios,
        ["CONTROLLED", "PRESSURED_CONTROL", "BALL_IN_FLIGHT", "RECEPTION"],
    )
    candidate = (
        in_play
        & smart
        & (np.nan_to_num(speed, nan=0.0) <= speed_threshold_mps)
        & ~flight
        & ~reception
        & ~blocked_context
    )
    if maximum_player_activity_mps is not None:
        candidate &= np.nan_to_num(player_activity, nan=np.inf) <= maximum_player_activity_mps

    start = 0
    while start < n_frames:
        if not candidate[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and candidate[end]:
            end += 1
        touches_existing_d = np.any(output[start:end] == "D")
        touches_existing_d |= start > 0 and output[start - 1] == "D"
        touches_existing_d |= end < n_frames and output[end] == "D"
        eligible = end - start >= minimum_candidate_run or np.any(output[start:end] == "D")
        if require_existing_d_adjacency:
            eligible &= touches_existing_d
        if eligible:
            output[start:end] = "D"
        start = end

    output[~in_play] = "X"
    return output


def stop_dead_support_at_restart_control(
    features: pd.DataFrame,
    dead_support: np.ndarray,
) -> np.ndarray:
    """Stop Smart Tagging dead-ball support after confirmed restart control.

    Smart Tagging intervals are deliberately broad.  A matching
    ``restart_confirmed_anchor`` is stronger local evidence that the restart
    has become active, so D support is retained only before the first such
    anchor in each tagged interval.  The function changes only the candidate
    D mask; it does not assign H/A or infer an unobserved touch.
    """
    support = np.asarray(dead_support, dtype=bool).copy()
    context = features.get(
        "smart_set_piece", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    confirmed = features.get(
        "restart_confirmed_anchor", pd.Series(False, index=features.index)
    ).to_numpy(bool)
    start = 0
    while start < len(support):
        if not context[start]:
            start += 1
            continue
        end = start + 1
        while end < len(support) and context[end]:
            end += 1
        local_confirmed = np.flatnonzero(confirmed[start:end])
        if local_confirmed.size:
            support[start + int(local_confirmed[0]) : end] = False
        start = end
    return support


def protect_stable_control_anchors(
    features: pd.DataFrame,
    labels: np.ndarray,
    minimum_run: int = 8,
) -> np.ndarray:
    """Protect sustained high-confidence control anchors from later filling.

    The stable-control anchor has independent distance, margin, persistence,
    speed, and distance-variance evidence.  Only contiguous runs of the same
    anchored team are protected, so a single noisy anchor frame cannot create
    a new possession spell.
    """
    output = np.asarray(labels, dtype="<U1").copy()
    anchor = features.get(
        "anchor_team", pd.Series(None, index=features.index)
    ).to_numpy(dtype=object)
    context = features.get(
        "scenario_context", pd.Series("UNKNOWN", index=features.index)
    ).to_numpy(dtype=object)
    eligible = np.isin(context, ["CONTROLLED", "PRESSURED_CONTROL"])
    eligible &= np.isin(anchor, ["H", "A"])
    start = 0
    while start < len(output):
        if not eligible[start]:
            start += 1
            continue
        end = start + 1
        while (
            end < len(output)
            and eligible[end]
            and anchor[end] == anchor[start]
        ):
            end += 1
        if end - start >= minimum_run:
            output[start:end] = anchor[start]
        start = end
    return output


def restart_boundary_dead_ball_labels(
    features: pd.DataFrame,
    base_labels: np.ndarray,
    active_speed_mps: float = 1.0,
    minimum_active_run: int = 3,
    minimum_dead_run: int = 5,
) -> np.ndarray:
    """Use sustained ball activity to close D inside a Smart Tagging window.

    Smart Tagging marks a tactical context interval, not the exact instant of
    the restart. Within each contiguous tagged run, D is allowed before the
    first sustained active-ball run and is stopped at that inferred boundary.
    The method is intentionally conservative: it never changes frames outside
    Smart Tagging or frames with control/flight/reception evidence.
    """
    output = np.asarray(base_labels, dtype="<U1").copy()
    n_frames = len(features)
    in_play = graded_mask(n_frames)
    context = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)
    speed = pd.to_numeric(
        features.get("ball_speed_derived_mps", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    acceleration = pd.to_numeric(
        features.get("ball_acceleration_mps2", pd.Series(np.nan, index=features.index)),
        errors="coerce",
    ).to_numpy(float)
    scenarios = features.get("scenario_context", pd.Series("UNKNOWN", index=features.index)).to_numpy(object)
    flight = features.get("ball_flight_candidate", pd.Series(False, index=features.index)).to_numpy(bool)
    reception = features.get("genuine_reception_anchor", pd.Series(False, index=features.index)).to_numpy(bool)
    blocked = np.isin(scenarios, ["CONTROLLED", "PRESSURED_CONTROL", "BALL_IN_FLIGHT", "RECEPTION"])
    active = (
        np.isfinite(speed)
        & (speed >= active_speed_mps)
        & (~np.isfinite(acceleration) | (acceleration >= 0.0))
    )

    start = 0
    while start < n_frames:
        if not context[start] or not in_play[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and context[end] and in_play[end]:
            end += 1
        boundary = end
        active_start = start
        while active_start < end:
            if not active[active_start]:
                active_start += 1
                continue
            active_end = active_start + 1
            while active_end < end and active[active_end]:
                active_end += 1
            if active_end - active_start >= minimum_active_run:
                boundary = active_start
                break
            active_start = active_end

        dead_end = boundary
        candidate = (
            np.ones(dead_end - start, dtype=bool)
            & (np.nan_to_num(speed[start:dead_end], nan=0.0) <= active_speed_mps * 3.0)
            & ~blocked[start:dead_end]
            & ~flight[start:dead_end]
            & ~reception[start:dead_end]
        )
        if candidate.size >= minimum_dead_run:
            candidate_positions = np.flatnonzero(candidate)
            if candidate_positions.size >= minimum_dead_run:
                output[start + candidate_positions] = "D"
        start = end

    output[~in_play] = "X"
    return output
