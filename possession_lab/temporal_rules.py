"""Transparent temporal features and anchor-based gap filling.

This module is an experiment layer on top of ``baseline_pipeline``. It does
not fit on the public labels. It computes causal-looking temporal features,
creates algorithmic confidence anchors, and bridges only locally justified
same-team gaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "possession_rules.yaml"


def load_rules(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the human-readable experiment configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _value(rules: dict[str, Any], *keys: str) -> Any:
    value: Any = rules
    for key in keys:
        value = value[key]
    return value


def _numeric(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table:
        return pd.Series(np.nan, index=table.index, dtype=float)
    return pd.to_numeric(table[column], errors="coerce")


def add_temporal_features(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Append lagged, rolling, velocity, persistence, and pressure features.

    Rolling windows are trailing windows: row ``t`` uses rows up to and
    including ``t``. This makes the features usable in a live/causal version.
    """
    rules = load_rules() if rules is None else rules
    fps = float(_value(rules, "timeline", "fps"))
    window = max(1, round(float(_value(rules, "timeline", "persistence_window_seconds")) * fps))
    out = features.copy()

    for team in ("h", "a"):
        distance = _numeric(out, f"nearest_{team}_distance_m")
        out[f"nearest_{team}_distance_lag_{window}f_m"] = distance.shift(window)
        out[f"nearest_{team}_distance_change_{window}f_m"] = distance - distance.shift(window)
        out[f"nearest_{team}_closing_rate_mps"] = (
            distance.shift(window) - distance
        ) / (window / fps)
        out[f"nearest_{team}_distance_roll_mean_m"] = distance.rolling(window, min_periods=1).mean()
        out[f"nearest_{team}_distance_roll_min_m"] = distance.rolling(window, min_periods=1).min()
        out[f"nearest_{team}_distance_roll_std_m"] = distance.rolling(window, min_periods=2).std().fillna(0.0)

    closest = out.get("closest_team", pd.Series(None, index=out.index, dtype=object))
    out["home_persistence"] = (closest == "H").astype(float).rolling(window, min_periods=1).mean()
    out["away_persistence"] = (closest == "A").astype(float).rolling(window, min_periods=1).mean()
    out["same_team_as_previous_frame"] = closest.eq(closest.shift()).fillna(False)
    out["team_distance_margin_abs_m"] = _numeric(out, "team_distance_margin_m").abs()

    ball_x = _numeric(out, "ball_x")
    ball_y = _numeric(out, "ball_y")
    interpolation_limit = int(_value(rules, "reception", "interpolation_max_frames"))
    ball_x_interp = ball_x.interpolate(limit=interpolation_limit, limit_area="inside")
    ball_y_interp = ball_y.interpolate(limit=interpolation_limit, limit_area="inside")
    out["ball_position_interpolated"] = (ball_x.isna() | ball_y.isna()) & ball_x_interp.notna() & ball_y_interp.notna()
    out["ball_x_interp"] = ball_x_interp
    out["ball_y_interp"] = ball_y_interp
    dx = ball_x.diff() * float(_value(rules, "pitch", "length_m"))
    dy = ball_y.diff() * float(_value(rules, "pitch", "width_m"))
    out["ball_displacement_m"] = np.hypot(dx, dy)
    out["ball_speed_derived_mps"] = out["ball_displacement_m"] * fps
    out["ball_direction_rad"] = np.arctan2(dy, dx)
    out["ball_speed_roll_mean_mps"] = out["ball_speed_derived_mps"].rolling(window, min_periods=1).mean()
    out["ball_acceleration_mps2"] = out["ball_speed_derived_mps"].diff() * fps
    direction_delta = out["ball_direction_rad"].diff()
    out["ball_direction_change_rad"] = np.arctan2(np.sin(direction_delta), np.cos(direction_delta)).abs()
    # A one-frame angle is useful for contact timing but is noisy under
    # tracking jitter. Keep short-window extrema and forward context as
    # separate observations so transition and dead-ball logic can distinguish
    # a genuine turn from ordinary straight-line deceleration.
    direction_window = max(3, min(9, window))
    direction_change = out["ball_direction_change_rad"]
    out["ball_direction_change_roll_max_rad"] = direction_change.rolling(
        direction_window, min_periods=1
    ).max()
    out["ball_direction_change_roll_mean_rad"] = direction_change.rolling(
        direction_window, min_periods=1
    ).mean()
    out["ball_direction_change_forward_max_rad"] = pd.Series(
        direction_change.to_numpy(float)[::-1], index=out.index
    ).rolling(direction_window, min_periods=1).max().to_numpy(float)[::-1]
    out["ball_speed_drop_mps"] = out["ball_speed_derived_mps"].shift(1) - out["ball_speed_derived_mps"]
    interp_dx = ball_x_interp.diff() * float(_value(rules, "pitch", "length_m"))
    interp_dy = ball_y_interp.diff() * float(_value(rules, "pitch", "width_m"))
    out["ball_speed_interpolated_mps"] = np.hypot(interp_dx, interp_dy) * fps
    out["ball_direction_interpolated_rad"] = np.arctan2(interp_dy, interp_dx)
    flight_window = max(1, int(_value(rules, "reception", "pre_flight_window_frames")))
    flight_speed = float(_value(rules, "bridging", "minimum_ball_speed_mps"))
    # A short internal ball gap should not erase evidence that the ball was in
    # flight. Use the observed speed where available and the interpolated speed
    # only where the configured interpolation limit makes it trustworthy.
    speed_evidence = out["ball_speed_derived_mps"].where(
        out["ball_speed_derived_mps"].notna(), out["ball_speed_interpolated_mps"]
    )
    # George's source-aware path is an optional third evidence channel for a
    # tactical gap.  Only eligible, non-zero-confidence rows may contribute;
    # model-imputed ``filled`` rows remain excluded unless the caller
    # explicitly enabled them in the adapter.
    george_eligible = out.get(
        "ball_fill_eligible", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    george_confidence = _numeric(out, "ball_fill_observation_confidence").fillna(0.0).to_numpy(float)
    george_supported = george_eligible & (george_confidence > 0.0)
    george_speed = _numeric(out, "ball_fill_speed_mps").to_numpy(float)
    george_displacement = _numeric(out, "ball_fill_displacement_m").to_numpy(float)
    george_direction = _numeric(out, "ball_fill_direction_rad").to_numpy(float)
    tactical_gap = out["ball_speed_derived_mps"].isna().to_numpy(bool)
    george_gap_speed = george_supported & tactical_gap & np.isfinite(george_speed)
    speed_evidence = speed_evidence.to_numpy(float).copy()
    speed_evidence[george_gap_speed] = george_speed[george_gap_speed]
    speed_evidence = pd.Series(speed_evidence, index=out.index, dtype=float)
    out["ball_speed_george_evidence_mps"] = np.where(
        george_gap_speed, george_speed, np.nan
    )
    out["ball_displacement_george_evidence_m"] = np.where(
        george_supported & np.isfinite(george_displacement), george_displacement, np.nan
    )
    out["ball_direction_george_evidence_rad"] = np.where(
        george_supported & np.isfinite(george_direction), george_direction, np.nan
    )
    out["ball_flight_george_evidence"] = (
        george_supported
        & out.get("ball_fill_long_ball_candidate", pd.Series(False, index=out.index)).fillna(False).to_numpy(bool)
    )
    out["ball_flight_george_distance_m"] = np.where(
        george_supported,
        _numeric(out, "ball_fill_flight_distance_m"),
        np.nan,
    )
    out["ball_speed_evidence_mps"] = speed_evidence
    george_flight = out["ball_flight_george_evidence"]
    out["recent_flight_evidence_interpolated"] = (
        speed_evidence.shift(1).rolling(flight_window, min_periods=1).max()
        >= flight_speed
    ) | george_flight.shift(1, fill_value=False).rolling(flight_window, min_periods=1).max().astype(bool)
    out["ball_flight_candidate_interpolated"] = (
        out["recent_flight_evidence_interpolated"]
        & (_numeric(out, "closest_distance_m") > float(_value(rules, "anchors", "stable_control_max_distance_m")))
    )
    out["recent_flight_evidence"] = (
        out["ball_speed_derived_mps"].shift(1).rolling(flight_window, min_periods=1).max() >= flight_speed
    ) | george_flight.shift(1, fill_value=False).rolling(flight_window, min_periods=1).max().astype(bool)
    out["ball_flight_candidate"] = (
        out["recent_flight_evidence"]
        & (_numeric(out, "closest_distance_m") > float(_value(rules, "anchors", "stable_control_max_distance_m")))
    )

    # Whole-pitch movement is an independent dead-ball signal. It is computed
    # from temporary ATD slot positions, so it does not require player IDs.
    slot_numbers = sorted(
        int(column[:-2])
        for column in out.columns
        if column.endswith("_x") and column[:-2].isdigit() and f"{column[:-2]}_y" in out
    )
    player_speeds = []
    pitch_length = float(_value(rules, "pitch", "length_m"))
    pitch_width = float(_value(rules, "pitch", "width_m"))
    for slot in slot_numbers:
        dx = _numeric(out, f"{slot}_x").diff() * pitch_length * fps
        dy = _numeric(out, f"{slot}_y").diff() * pitch_width * fps
        player_speeds.append(np.hypot(dx, dy).to_numpy(float))
    if player_speeds:
        speed_matrix = np.column_stack(player_speeds)
        valid_rows = np.isfinite(speed_matrix).any(axis=1)
        median_speed = np.full(len(out), np.nan, dtype=float)
        p90_speed = np.full(len(out), np.nan, dtype=float)
        median_speed[valid_rows] = np.nanmedian(speed_matrix[valid_rows], axis=1)
        p90_speed[valid_rows] = np.nanpercentile(speed_matrix[valid_rows], 90, axis=1)
        out["player_activity_median_mps"] = median_speed
        out["player_activity_p90_mps"] = p90_speed
    else:
        out["player_activity_median_mps"] = np.nan
        out["player_activity_p90_mps"] = np.nan

    # Pressure is the distance from the nearest opponent to the nearest player
    # of the candidate team. It is intentionally team-level because ATD slots
    # are temporary tracks, not stable player identities.
    ball_pressure = np.where(
        closest == "H",
        _numeric(out, "nearest_a_distance_m"),
        np.where(closest == "A", _numeric(out, "nearest_h_distance_m"), np.nan),
    )
    player_pressure = _numeric(out, "closest_player_opponent_distance_m")
    out["candidate_pressure_m"] = player_pressure.where(player_pressure.notna(), pd.Series(ball_pressure, index=out.index))
    out["candidate_opponent_density_5m"] = _numeric(out, "closest_player_opponent_density_5m")
    out["candidate_teammate_support_10m"] = _numeric(out, "closest_player_teammate_support_10m")
    out["candidate_pass_options_15m"] = _numeric(out, "closest_player_pass_options_15m")
    out["candidate_isolated"] = out["closest_player_isolated"].fillna(False)
    out["candidate_low_pressure"] = out["candidate_pressure_m"] >= float(
        _value(rules, "anchors", "low_pressure_opponent_distance_m")
    )
    approach_threshold = 0.5
    out["home_receiver_approaching"] = out["nearest_h_closing_rate_mps"] >= approach_threshold
    out["away_receiver_approaching"] = out["nearest_a_closing_rate_mps"] >= approach_threshold
    out["home_receiver_open"] = (
        out["home_receiver_approaching"]
        & (_numeric(out, "nearest_a_distance_m") >= float(_value(rules, "anchors", "low_pressure_opponent_distance_m")))
    )
    out["away_receiver_open"] = (
        out["away_receiver_approaching"]
        & (_numeric(out, "nearest_h_distance_m") >= float(_value(rules, "anchors", "low_pressure_opponent_distance_m")))
    )
    return out


def add_missing_ball_trajectory_features(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Describe the observed trajectory on either side of an internal ball gap.

    The ATD export can lose the ball for a run of frames.  This function does
    not invent a label for those frames.  Instead, it connects the last
    observed point before the gap to the first observed point after it and
    records whether that connection is physically and geometrically coherent.

    ``gap_*`` columns are broadcast across the missing run so scenario code can
    inspect one row per frame.  Player support is deliberately evidence-only:
    a player closing toward the reconstructed path can support a reception
    hypothesis, but it cannot force an H/A switch on its own.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    n_frames = len(out)
    fps = float(_value(rules, "timeline", "fps"))
    pitch_length = float(_value(rules, "pitch", "length_m"))
    pitch_width = float(_value(rules, "pitch", "width_m"))
    trajectory_rules = rules.get("trajectory", {})
    context_frames = max(1, int(trajectory_rules.get("context_frames", 5)))
    max_speed = float(trajectory_rules.get("maximum_implied_ball_speed_mps", 40.0))
    max_direction_change = float(trajectory_rules.get("maximum_direction_change_rad", np.pi / 2))
    interpolation_limit = int(_value(rules, "reception", "interpolation_max_frames"))

    numeric_defaults = {
        "ball_missing_gap_frames": np.nan,
        "ball_missing_gap_seconds": np.nan,
        "ball_gap_displacement_m": np.nan,
        "ball_gap_implied_speed_mps": np.nan,
        "ball_gap_direction_rad": np.nan,
        "ball_gap_direction_change_rad": np.nan,
        "ball_gap_line_residual_m": np.nan,
        "ball_gap_receiver_closing_speed_mps": np.nan,
        "ball_gap_receiver_speed_ratio": np.nan,
        "ball_gap_receiver_eta_s": np.nan,
        "ball_gap_pre_speed_mps": np.nan,
        "ball_gap_post_speed_mps": np.nan,
        "ball_gap_speed_change_mps": np.nan,
    }
    for column, default in numeric_defaults.items():
        out[column] = default
    out["ball_gap_trajectory_coherent"] = False
    out["ball_gap_receiver_team"] = None
    out["ball_gap_receiver_run_supported"] = False
    out["ball_gap_sharp_acceleration_risk"] = False

    ball_x = _numeric(out, "ball_x").to_numpy(float)
    ball_y = _numeric(out, "ball_y").to_numpy(float)
    missing = ~np.isfinite(ball_x) | ~np.isfinite(ball_y)
    if n_frames == 0:
        return out

    # Find only internal missing runs. Leading/trailing missing values have no
    # two-sided observed trajectory and therefore cannot pass this audit.
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
    for start, end in zip(starts, ends):
        before = start - 1
        after = end + 1
        if before < 0 or after >= n_frames:
            continue
        gap_frames = after - before - 1
        if gap_frames <= 0 or gap_frames > interpolation_limit:
            continue

        dx_norm = ball_x[after] - ball_x[before]
        dy_norm = ball_y[after] - ball_y[before]
        dx_m = dx_norm * pitch_length
        dy_m = dy_norm * pitch_width
        displacement = float(np.hypot(dx_m, dy_m))
        duration_s = gap_frames / fps
        implied_speed = displacement / duration_s if duration_s > 0 else np.nan
        gap_direction = float(np.arctan2(dy_m, dx_m))

        pre_start = max(0, before - context_frames + 1)
        post_end = min(n_frames, after + context_frames)
        pre_x = ball_x[pre_start : before + 1]
        pre_y = ball_y[pre_start : before + 1]
        post_x = ball_x[after:post_end]
        post_y = ball_y[after:post_end]
        pre_valid = np.isfinite(pre_x) & np.isfinite(pre_y)
        post_valid = np.isfinite(post_x) & np.isfinite(post_y)

        def _context_velocity(x_values: np.ndarray, y_values: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
            valid_indices = np.flatnonzero(valid)
            if len(valid_indices) < 2:
                return np.nan, np.nan
            i0, i1 = valid_indices[-2:]
            return (
                (x_values[i1] - x_values[i0]) * pitch_length * fps,
                (y_values[i1] - y_values[i0]) * pitch_width * fps,
            )

        pre_vx, pre_vy = _context_velocity(pre_x, pre_y, pre_valid)
        post_vx, post_vy = _context_velocity(post_x, post_y, post_valid)
        pre_speed = float(np.hypot(pre_vx, pre_vy)) if np.isfinite(pre_vx) and np.isfinite(pre_vy) else np.nan
        post_speed = float(np.hypot(post_vx, post_vy)) if np.isfinite(post_vx) and np.isfinite(post_vy) else np.nan
        speed_change = post_speed - pre_speed if np.isfinite(pre_speed) and np.isfinite(post_speed) else np.nan
        if np.isfinite(pre_vx) and np.isfinite(pre_vy) and np.isfinite(post_vx) and np.isfinite(post_vy):
            pre_angle = np.arctan2(pre_vy, pre_vx)
            post_angle = np.arctan2(post_vy, post_vx)
            direction_change = float(abs(np.arctan2(np.sin(post_angle - pre_angle), np.cos(post_angle - pre_angle))))
        else:
            direction_change = np.nan

        # Residual of the observed context points from the endpoint line. A
        # low residual means the straight-line bridge is consistent with the
        # nearby observed path, not that the hidden path is known exactly.
        line_norm = np.hypot(dx_norm, dy_norm)
        residuals: list[float] = []
        if line_norm > 0:
            for x_values, y_values, valid in ((pre_x, pre_y, pre_valid), (post_x, post_y, post_valid)):
                for x_value, y_value in zip(x_values[valid], y_values[valid]):
                    residuals.append(float(abs(
                        (y_value - ball_y[before]) * pitch_width * dx_m
                        - (x_value - ball_x[before]) * pitch_length * dy_m,
                    ) / displacement))
        line_residual = float(np.mean(residuals)) if residuals else np.nan

        coherent = (
            np.isfinite(implied_speed)
            and implied_speed <= max_speed
            and (not np.isfinite(direction_change) or direction_change <= max_direction_change)
            and (not np.isfinite(line_residual) or line_residual <= max(pitch_length, pitch_width))
        )
        row_slice = slice(start, end + 1)
        out.loc[row_slice, "ball_missing_gap_frames"] = gap_frames
        out.loc[row_slice, "ball_missing_gap_seconds"] = duration_s
        out.loc[row_slice, "ball_gap_displacement_m"] = displacement
        out.loc[row_slice, "ball_gap_implied_speed_mps"] = implied_speed
        out.loc[row_slice, "ball_gap_direction_rad"] = gap_direction
        out.loc[row_slice, "ball_gap_direction_change_rad"] = direction_change
        out.loc[row_slice, "ball_gap_line_residual_m"] = line_residual
        out.loc[row_slice, "ball_gap_trajectory_coherent"] = coherent
        out.loc[row_slice, "ball_gap_pre_speed_mps"] = pre_speed
        out.loc[row_slice, "ball_gap_post_speed_mps"] = post_speed
        out.loc[row_slice, "ball_gap_speed_change_mps"] = speed_change
        out.loc[row_slice, "ball_gap_sharp_acceleration_risk"] = bool(
            np.isfinite(speed_change)
            and speed_change >= float(trajectory_rules.get("sharp_gap_speed_change_mps", 8.0))
        )

        # Reachability is calculated later in the canonical composition. If it
        # is already present, use the endpoint evidence to describe a possible
        # receiver run. Otherwise these fields remain neutral and are filled
        # by the next composition pass only when available.
        endpoint = after
        receiver_team = None
        likely = out.get("likely_receiver_team")
        projected = out.get("projected_receiver_team")
        if likely is not None and pd.notna(likely.iloc[endpoint]):
            receiver_team = likely.iloc[endpoint]
        elif projected is not None and pd.notna(projected.iloc[endpoint]):
            receiver_team = projected.iloc[endpoint]
        if receiver_team in ("H", "A"):
            closing = _numeric(out, f"{receiver_team.lower()}_fastest_closing_speed_mps").iloc[endpoint]
            eta = _numeric(out, f"{receiver_team.lower()}_fastest_arrival_s").iloc[endpoint]
            ratio = closing / implied_speed if np.isfinite(closing) and implied_speed > 0 else np.nan
            run_supported = bool(np.isfinite(closing) and closing > 0 and np.isfinite(eta))
            out.loc[row_slice, "ball_gap_receiver_team"] = receiver_team
            out.loc[row_slice, "ball_gap_receiver_closing_speed_mps"] = closing
            out.loc[row_slice, "ball_gap_receiver_speed_ratio"] = ratio
            out.loc[row_slice, "ball_gap_receiver_eta_s"] = eta
            out.loc[row_slice, "ball_gap_receiver_run_supported"] = run_supported

    return out


def add_kalman_ball_trajectory_features(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Estimate an occluded ball path and expose its uncertainty.

    This is a constant-velocity Kalman smoother over pitch-space ball
    coordinates. It is deliberately an observation layer: it fills a
    plausible position/velocity estimate only for internal gaps bracketed by
    observed ball positions, and it never assigns a team or possession label.
    The covariance is retained so downstream scenario logic can reduce trust
    when the reconstructed path is uncertain.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    n_frames = len(out)
    fps = float(_value(rules, "timeline", "fps"))
    dt = 1.0 / fps
    pitch_length = float(_value(rules, "pitch", "length_m"))
    pitch_width = float(_value(rules, "pitch", "width_m"))
    kalman_rules = rules.get("trajectory", {}).get("kalman", {})
    measurement_std_m = float(kalman_rules.get("measurement_noise_m", 0.35))
    acceleration_std_mps2 = float(kalman_rules.get("acceleration_noise_mps2", 8.0))
    maximum_uncertainty_m = float(kalman_rules.get("maximum_uncertainty_m", 8.0))
    maximum_gap_frames = int(
        kalman_rules.get(
            "maximum_gap_frames",
            _value(rules, "reception", "interpolation_max_frames"),
        )
    )

    output_defaults: dict[str, object] = {
        "ball_x_kalman": np.nan,
        "ball_y_kalman": np.nan,
        "ball_kalman_velocity_x_mps": np.nan,
        "ball_kalman_velocity_y_mps": np.nan,
        "ball_kalman_speed_mps": np.nan,
        "ball_kalman_position_uncertainty_m": np.nan,
        "ball_kalman_observation_residual_m": np.nan,
        "ball_kalman_velocity_change_mps": np.nan,
        "ball_kalman_displacement_m": np.nan,
        "ball_kalman_direction_rad": np.nan,
        "ball_kalman_acceleration_mps2": np.nan,
        "ball_kalman_direction_change_rad": np.nan,
        "ball_kalman_path_available": False,
        "ball_kalman_reliable": False,
        "ball_kalman_prediction_confidence": 0.0,
        "ball_kalman_gap_frames": np.nan,
    }
    for column, default in output_defaults.items():
        out[column] = default
    if n_frames == 0:
        return out

    ball_x = _numeric(out, "ball_x").to_numpy(float) * pitch_length
    ball_y = _numeric(out, "ball_y").to_numpy(float) * pitch_width
    observed_pair = np.isfinite(ball_x) & np.isfinite(ball_y)
    if observed_pair.sum() < 2:
        return out

    transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    acceleration_variance = acceleration_std_mps2 ** 2
    process = acceleration_variance * np.array(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
        dtype=float,
    )
    measurement_variance = measurement_std_m**2
    identity = np.eye(2, dtype=float)

    def smooth_axis(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        observed = np.isfinite(observations)
        valid = np.flatnonzero(observed)
        first = int(valid[0])
        filtered = np.full((n_frames, 2), np.nan, dtype=float)
        predicted = np.full((n_frames, 2), np.nan, dtype=float)
        filtered_cov = np.full((n_frames, 2, 2), np.nan, dtype=float)
        predicted_cov = np.full((n_frames, 2, 2), np.nan, dtype=float)

        initial_velocity = 0.0
        next_valid = valid[valid > first]
        if len(next_valid):
            next_index = int(next_valid[0])
            initial_velocity = (observations[next_index] - observations[first]) / (
                (next_index - first) * dt
            )
        state = np.array([observations[first], initial_velocity], dtype=float)
        covariance = np.diag([
            max(measurement_variance * 4.0, 0.25),
            max(acceleration_variance, 1.0),
        ])

        for frame in range(first, n_frames):
            if frame == first:
                predicted_state = state.copy()
                predicted_covariance = covariance.copy()
            else:
                predicted_state = transition @ state
                predicted_covariance = (
                    transition @ covariance @ transition.T + process
                )
            predicted[frame] = predicted_state
            predicted_cov[frame] = predicted_covariance

            if observed[frame]:
                innovation = observations[frame] - predicted_state[0]
                innovation_variance = predicted_covariance[0, 0] + measurement_variance
                gain = predicted_covariance[:, 0] / max(innovation_variance, 1e-12)
                state = predicted_state + gain * innovation
                covariance = (identity - np.outer(gain, np.array([1.0, 0.0]))) @ predicted_covariance
                covariance = 0.5 * (covariance + covariance.T)
            else:
                state = predicted_state
                covariance = predicted_covariance
            filtered[frame] = state
            filtered_cov[frame] = covariance

        smoothed = filtered.copy()
        smoothed_cov = filtered_cov.copy()
        last = int(valid[-1])
        for frame in range(last - 1, first - 1, -1):
            if not np.isfinite(predicted_cov[frame + 1]).all():
                continue
            smoother_gain = filtered_cov[frame] @ transition.T @ np.linalg.pinv(predicted_cov[frame + 1])
            smoothed[frame] = filtered[frame] + smoother_gain @ (
                smoothed[frame + 1] - predicted[frame + 1]
            )
            smoothed_cov[frame] = filtered_cov[frame] + smoother_gain @ (
                smoothed_cov[frame + 1] - predicted_cov[frame + 1]
            ) @ smoother_gain.T
            smoothed_cov[frame] = 0.5 * (smoothed_cov[frame] + smoothed_cov[frame].T)
        return smoothed, smoothed_cov

    x_state, x_cov = smooth_axis(ball_x)
    y_state, y_cov = smooth_axis(ball_y)
    last_observed_before = np.maximum.accumulate(
        np.where(observed_pair, np.arange(n_frames), -1)
    )
    next_observed_after = np.minimum.accumulate(
        np.where(observed_pair, np.arange(n_frames), n_frames)[::-1]
    )[::-1]
    missing_pair = ~observed_pair
    gap_lengths = np.zeros(n_frames, dtype=int)
    gap_starts = np.flatnonzero(missing_pair & ~np.r_[False, missing_pair[:-1]])
    gap_ends = np.flatnonzero(missing_pair & ~np.r_[missing_pair[1:], False])
    for start, end in zip(gap_starts, gap_ends):
        gap_lengths[start : end + 1] = int(end - start + 1)
    path_available = (
        missing_pair
        & (last_observed_before >= 0)
        & (next_observed_after < n_frames)
        & (gap_lengths <= maximum_gap_frames)
    )

    x_position = x_state[:, 0]
    y_position = y_state[:, 0]
    x_variance = np.clip(x_cov[:, 0, 0], 0.0, None)
    y_variance = np.clip(y_cov[:, 0, 0], 0.0, None)
    uncertainty = np.sqrt(x_variance + y_variance)
    speed = np.hypot(x_state[:, 1], y_state[:, 1])
    velocity_change = np.r_[np.nan, np.diff(speed)]
    kalman_dx = np.r_[np.nan, np.diff(x_position)]
    kalman_dy = np.r_[np.nan, np.diff(y_position)]
    kalman_displacement = np.hypot(kalman_dx, kalman_dy)
    kalman_direction = np.arctan2(kalman_dy, kalman_dx)
    kalman_acceleration = velocity_change * fps
    kalman_direction_delta = np.r_[
        np.nan,
        np.arctan2(
            np.sin(np.diff(kalman_direction)),
            np.cos(np.diff(kalman_direction)),
        ),
    ]
    kalman_direction_change = np.abs(kalman_direction_delta)
    observed_residual = np.hypot(ball_x - x_position, ball_y - y_position)
    reliable = path_available & np.isfinite(uncertainty) & (uncertainty <= maximum_uncertainty_m)
    confidence = np.where(
        path_available & np.isfinite(uncertainty),
        np.exp(-uncertainty / max(maximum_uncertainty_m, 1e-6)),
        0.0,
    )

    out.loc[:, "ball_x_kalman"] = x_position / pitch_length
    out.loc[:, "ball_y_kalman"] = y_position / pitch_width
    out.loc[:, "ball_kalman_velocity_x_mps"] = x_state[:, 1]
    out.loc[:, "ball_kalman_velocity_y_mps"] = y_state[:, 1]
    out.loc[:, "ball_kalman_speed_mps"] = speed
    out.loc[:, "ball_kalman_position_uncertainty_m"] = uncertainty
    out.loc[:, "ball_kalman_observation_residual_m"] = observed_residual
    out.loc[:, "ball_kalman_velocity_change_mps"] = velocity_change
    out.loc[:, "ball_kalman_displacement_m"] = kalman_displacement
    out.loc[:, "ball_kalman_direction_rad"] = kalman_direction
    out.loc[:, "ball_kalman_acceleration_mps2"] = kalman_acceleration
    out.loc[:, "ball_kalman_direction_change_rad"] = kalman_direction_change
    out.loc[:, "ball_kalman_path_available"] = path_available
    out.loc[:, "ball_kalman_reliable"] = reliable
    out.loc[:, "ball_kalman_prediction_confidence"] = confidence
    out.loc[:, "ball_kalman_gap_frames"] = np.where(path_available, gap_lengths, np.nan)
    return out


def add_ball_path_fusion_features(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Select one source-aware ball path for internal tactical gaps.

    George and Kalman can describe the same hidden interval, so their scores
    must not be added as if they were independent observations.  The
    precedence is direct George observation, George geometric flight, then a
    short low-uncertainty Kalman path.  Agreement between available sources is
    exposed separately for player-arrival and transition evidence; disagreement
    increases ambiguity rather than awarding a team.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    n = len(out)
    pitch_length = float(_value(rules, "pitch", "length_m"))
    pitch_width = float(_value(rules, "pitch", "width_m"))
    kalman_rules = rules.get("trajectory", {}).get("kalman", {})
    maximum_uncertainty = float(kalman_rules.get("maximum_uncertainty_m", 8.0))
    fusion_uncertainty = float(
        kalman_rules.get(
            "fusion_max_uncertainty_m",
            min(4.0, maximum_uncertainty),
        )
    )

    def numeric(name: str, default: float = np.nan) -> np.ndarray:
        return pd.to_numeric(
            out.get(name, pd.Series(default, index=out.index)), errors="coerce"
        ).to_numpy(float)

    tactical_x = numeric("ball_x")
    tactical_y = numeric("ball_y")
    tactical_missing = ~np.isfinite(tactical_x) | ~np.isfinite(tactical_y)
    george_x = numeric("ball_fill_x")
    george_y = numeric("ball_fill_y")
    george_available = (
        out.get("ball_fill_eligible", pd.Series(False, index=out.index))
        .fillna(False)
        .to_numpy(bool)
        & np.isfinite(george_x)
        & np.isfinite(george_y)
    )
    source_class = out.get(
        "ball_fill_source_class",
        pd.Series("untrusted_or_metadata", index=out.index),
    ).fillna("untrusted_or_metadata").to_numpy(object)
    george_direct = (
        tactical_missing
        & george_available
        & (source_class == "direct_observation")
    )
    george_flight = (
        tactical_missing
        & george_available
        & (source_class == "geometric_reconstruction")
    )

    kalman_x = numeric("ball_x_kalman")
    kalman_y = numeric("ball_y_kalman")
    kalman_uncertainty = numeric("ball_kalman_position_uncertainty_m")
    kalman_available = (
        tactical_missing
        & out.get("ball_kalman_path_available", pd.Series(False, index=out.index))
        .fillna(False)
        .to_numpy(bool)
        & out.get("ball_kalman_reliable", pd.Series(False, index=out.index))
        .fillna(False)
        .to_numpy(bool)
        & np.isfinite(kalman_x)
        & np.isfinite(kalman_y)
        & np.isfinite(kalman_uncertainty)
        & (kalman_uncertainty <= fusion_uncertainty)
    )

    # Observed tactical coordinates remain the primary path whenever present.
    source = np.full(n, "UNRESOLVED", dtype=object)
    source[~tactical_missing & np.isfinite(tactical_x) & np.isfinite(tactical_y)] = "TACTICAL"
    source[george_direct] = "GEORGE_DIRECT"
    source[~george_direct & george_flight] = "GEORGE_FLIGHT"
    source[~george_direct & ~george_flight & kalman_available] = "KALMAN"

    fused_x = tactical_x.copy()
    fused_y = tactical_y.copy()
    fused_uncertainty = np.where(np.isfinite(tactical_x) & np.isfinite(tactical_y), 0.0, np.nan)
    direct_or_flight = george_direct | george_flight
    fused_x[direct_or_flight] = george_x[direct_or_flight]
    fused_y[direct_or_flight] = george_y[direct_or_flight]
    george_sigma = numeric("ball_fill_sigma_m")
    fused_uncertainty[direct_or_flight] = np.where(
        np.isfinite(george_sigma[direct_or_flight]),
        george_sigma[direct_or_flight],
        np.nan,
    )
    fused_x[~direct_or_flight & kalman_available] = kalman_x[~direct_or_flight & kalman_available]
    fused_y[~direct_or_flight & kalman_available] = kalman_y[~direct_or_flight & kalman_available]
    fused_uncertainty[~direct_or_flight & kalman_available] = kalman_uncertainty[
        ~direct_or_flight & kalman_available
    ]

    # Compare the alternative sources only where both describe the same gap.
    source_comparison = george_available & kalman_available
    george_direction = numeric("ball_fill_direction_rad")
    kalman_direction = numeric("ball_kalman_direction_rad")
    george_speed = numeric("ball_fill_speed_mps")
    kalman_speed = numeric("ball_kalman_speed_mps")
    distance = np.full(n, np.nan, dtype=float)
    distance[source_comparison] = np.hypot(
        (george_x[source_comparison] - kalman_x[source_comparison]) * pitch_length,
        (george_y[source_comparison] - kalman_y[source_comparison]) * pitch_width,
    )
    direction_delta = np.full(n, np.nan, dtype=float)
    direction_valid = source_comparison & np.isfinite(george_direction) & np.isfinite(kalman_direction)
    direction_delta[direction_valid] = np.abs(
        np.arctan2(
            np.sin(george_direction[direction_valid] - kalman_direction[direction_valid]),
            np.cos(george_direction[direction_valid] - kalman_direction[direction_valid]),
        )
    )
    speed_delta = np.full(n, np.nan, dtype=float)
    speed_valid = source_comparison & np.isfinite(george_speed) & np.isfinite(kalman_speed)
    speed_delta[speed_valid] = np.abs(george_speed[speed_valid] - kalman_speed[speed_valid])
    distance_score = np.exp(-np.nan_to_num(distance, nan=100.0) / 3.0)
    direction_score = np.exp(-np.nan_to_num(direction_delta, nan=np.pi) / 0.8)
    speed_score = np.exp(-np.nan_to_num(speed_delta, nan=100.0) / 8.0)
    agreement = np.full(n, np.nan, dtype=float)
    agreement[source_comparison] = np.mean(
        np.column_stack([distance_score, direction_score, speed_score]), axis=1
    )[source_comparison]

    # Derive motion from the selected path. These columns are used only as
    # missing-ball evidence; visible tactical motion remains unchanged.
    dx = np.r_[np.nan, np.diff(fused_x)] * pitch_length
    dy = np.r_[np.nan, np.diff(fused_y)] * pitch_width
    displacement = np.hypot(dx, dy)
    frame_delta = np.r_[np.nan, np.diff(pd.to_numeric(out["frame"], errors="coerce").to_numpy(float))]
    elapsed = frame_delta / float(_value(rules, "timeline", "fps"))
    valid_motion = np.isfinite(displacement) & np.isfinite(elapsed) & (elapsed > 0.0)
    speed = np.full(n, np.nan, dtype=float)
    speed[valid_motion] = displacement[valid_motion] / elapsed[valid_motion]
    direction = np.full(n, np.nan, dtype=float)
    direction[valid_motion & (displacement > 0.0)] = np.arctan2(
        dy[valid_motion & (displacement > 0.0)],
        dx[valid_motion & (displacement > 0.0)],
    )
    direction_change = np.full(n, np.nan, dtype=float)
    direction_change[1:] = np.abs(
        np.arctan2(
            np.sin(direction[1:] - direction[:-1]),
            np.cos(direction[1:] - direction[:-1]),
        )
    )
    acceleration = np.full(n, np.nan, dtype=float)
    acceleration[1:] = (speed[1:] - speed[:-1]) / np.maximum(elapsed[1:], 1e-6)

    out["ball_path_fused_x"] = fused_x
    out["ball_path_fused_y"] = fused_y
    out["ball_path_fused_source"] = source
    out["ball_path_fused_available"] = np.isfinite(fused_x) & np.isfinite(fused_y)
    out["ball_path_fused_uncertainty_m"] = fused_uncertainty
    out["ball_path_kalman_candidate"] = kalman_available
    out["ball_path_george_direct_candidate"] = george_direct
    out["ball_path_george_flight_candidate"] = george_flight
    out["ball_path_george_kalman_distance_m"] = distance
    out["ball_path_george_kalman_direction_delta_rad"] = direction_delta
    out["ball_path_george_kalman_speed_delta_mps"] = speed_delta
    out["ball_path_george_kalman_agreement_score"] = agreement
    out["ball_path_george_kalman_disagreement_score"] = np.where(
        np.isfinite(agreement), 1.0 - agreement, np.nan
    )
    out["ball_path_fused_displacement_m"] = displacement
    out["ball_path_fused_speed_mps"] = speed
    out["ball_path_fused_direction_rad"] = direction
    out["ball_path_fused_direction_change_rad"] = direction_change
    out["ball_path_fused_acceleration_mps2"] = acceleration
    out["ball_path_fusion_kalman_uncertainty_gate_m"] = fusion_uncertainty
    return out


def add_gap_player_path_features(
    features: pd.DataFrame,
    team_by_slot: dict[int, str],
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Measure same-team player/ball spatial continuity around ball gaps.

    Temporary slot numbers are used only to read the coordinates at the two
    observed endpoints. The evidence is geometric: a same-team candidate is
    useful when it is close to the ball at both endpoints and its displacement
    is aligned with the ball's endpoint displacement. This does not assume a
    globally stable player track.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    n_frames = len(out)
    pitch_length = float(_value(rules, "pitch", "length_m"))
    pitch_width = float(_value(rules, "pitch", "width_m"))
    zone_radius = float(_value(rules, "reception", "maximum_control_distance_m"))
    max_gap_frames = max(1, round(float(_value(rules, "timeline", "long_ball_max_seconds")) * float(_value(rules, "timeline", "fps"))))
    speed_match_limit = float(_value(rules, "trajectory", "same_player_speed_match_max_mps"))
    tackle_distance = float(_value(rules, "trajectory", "tackle_opponent_distance_m"))
    for column, default in {
        "gap_player_path_team": None,
        "gap_player_endpoint_distance_m": np.nan,
        "gap_player_path_alignment": np.nan,
        "gap_player_path_team_margin_m": np.nan,
        "gap_player_path_ambiguous": False,
        "gap_player_path_supported": False,
        "gap_player_speed_before_mps": np.nan,
        "gap_player_speed_after_mps": np.nan,
        "gap_player_ball_speed_match_mps": np.nan,
        "gap_same_player_dribble_candidate": False,
        "gap_tackle_risk_candidate": False,
    }.items():
        out[column] = default

    tactical_ball_x = _numeric(out, "ball_x").to_numpy(float)
    tactical_ball_y = _numeric(out, "ball_y").to_numpy(float)
    ball_x = _numeric(
        out,
        "ball_path_fused_x" if "ball_path_fused_x" in out else "ball_x",
    ).to_numpy(float)
    ball_y = _numeric(
        out,
        "ball_path_fused_y" if "ball_path_fused_y" in out else "ball_y",
    ).to_numpy(float)
    missing = ~np.isfinite(tactical_ball_x) | ~np.isfinite(tactical_ball_y)
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
    for start, end in zip(starts, ends):
        before, after = start - 1, end + 1
        if before < 0 or after >= n_frames:
            continue
        if end - start + 1 > max_gap_frames:
            continue
        ball_dx = (ball_x[after] - ball_x[before]) * pitch_length
        ball_dy = (ball_y[after] - ball_y[before]) * pitch_width
        ball_norm = float(np.hypot(ball_dx, ball_dy))
        candidates: dict[str, tuple[float, float, bool]] = {}
        for team in ("H", "A"):
            best: tuple[float, float, bool] | None = None
            slots = [slot for slot, slot_team in team_by_slot.items() if slot_team == team]
            for slot in slots:
                bx = _numeric(out, f"{slot}_x").iloc[before]
                by = _numeric(out, f"{slot}_y").iloc[before]
                ax = _numeric(out, f"{slot}_x").iloc[after]
                ay = _numeric(out, f"{slot}_y").iloc[after]
                if not all(np.isfinite(value) for value in (bx, by, ax, ay)):
                    continue
                before_distance = float(np.hypot((bx - ball_x[before]) * pitch_length, (by - ball_y[before]) * pitch_width))
                after_distance = float(np.hypot((ax - ball_x[after]) * pitch_length, (ay - ball_y[after]) * pitch_width))
                endpoint_distance = (before_distance + after_distance) / 2.0
                player_dx = (ax - bx) * pitch_length
                player_dy = (ay - by) * pitch_width
                alignment = (
                    (player_dx * ball_dx + player_dy * ball_dy)
                    / (float(np.hypot(player_dx, player_dy)) * ball_norm)
                    if float(np.hypot(player_dx, player_dy)) > 0 and ball_norm > 0
                    else np.nan
                )
                supported = endpoint_distance <= zone_radius
                candidate = (endpoint_distance, alignment, supported)
                if best is None or endpoint_distance < best[0]:
                    best = candidate
            if best is not None:
                candidates[team] = best
        supported = [(team, values) for team, values in candidates.items() if values[2]]
        if not supported:
            continue
        team, (distance, alignment, _) = min(supported, key=lambda item: item[1][0])
        supported_distances = sorted(values[0] for _, values in supported)
        second_distance = supported_distances[1] if len(supported_distances) > 1 else np.nan
        team_margin = second_distance - distance if np.isfinite(second_distance) else np.nan
        row_slice = slice(start, end + 1)
        out.loc[row_slice, "gap_player_path_team"] = team
        out.loc[row_slice, "gap_player_endpoint_distance_m"] = distance
        out.loc[row_slice, "gap_player_path_alignment"] = alignment
        out.loc[row_slice, "gap_player_path_team_margin_m"] = team_margin
        out.loc[row_slice, "gap_player_path_ambiguous"] = bool(
            np.isfinite(team_margin) and team_margin < 1.0
        )
        out.loc[row_slice, "gap_player_path_supported"] = True
        # A same-player dribble hypothesis requires the same temporary slot to
        # be closest on both visible sides and to move at approximately the
        # observed ball speed. This is evidence, not player identity proof.
        slot_before = pd.to_numeric(out.get("closest_slot", pd.Series(np.nan, index=out.index)), errors="coerce").iloc[before]
        slot_after = pd.to_numeric(out.get("closest_slot", pd.Series(np.nan, index=out.index)), errors="coerce").iloc[after]
        same_slot = np.isfinite(slot_before) and np.isfinite(slot_after) and int(slot_before) == int(slot_after)
        if same_slot:
            slot = int(slot_before)
            x = _numeric(out, f"{slot}_x")
            y = _numeric(out, f"{slot}_y")
            before_prev = max(0, before - 1)
            after_next = min(n_frames - 1, after + 1)
            fps = float(_value(rules, "timeline", "fps"))
            speed_before = float(np.hypot((x.iloc[before] - x.iloc[before_prev]) * pitch_length, (y.iloc[before] - y.iloc[before_prev]) * pitch_width) * fps)
            speed_after = float(np.hypot((x.iloc[after_next] - x.iloc[after]) * pitch_length, (y.iloc[after_next] - y.iloc[after]) * pitch_width) * fps)
            ball_before = float(_numeric(out, "ball_speed_derived_mps").iloc[before])
            ball_after = float(_numeric(out, "ball_speed_derived_mps").iloc[after])
            match_error = float(np.nanmean([abs(speed_before - ball_before), abs(speed_after - ball_after)]))
            opponent_before = float(_numeric(out, "closest_player_opponent_distance_m").iloc[before])
            opponent_after = float(_numeric(out, "closest_player_opponent_distance_m").iloc[after])
            direction_change = float(_numeric(out, "ball_gap_direction_change_rad").iloc[before])
            dribble = (
                np.isfinite(match_error)
                and match_error <= speed_match_limit
                and not bool(out.get("ball_gap_sharp_acceleration_risk", pd.Series(False, index=out.index)).iloc[before])
                and (not np.isfinite(direction_change) or direction_change < 0.35)
            )
            tackle = (
                (np.isfinite(opponent_before) and opponent_before <= tackle_distance)
                or (np.isfinite(opponent_after) and opponent_after <= tackle_distance)
            ) and (not np.isfinite(direction_change) or direction_change >= 0.35 or not dribble)
            out.loc[row_slice, "gap_player_speed_before_mps"] = speed_before
            out.loc[row_slice, "gap_player_speed_after_mps"] = speed_after
            out.loc[row_slice, "gap_player_ball_speed_match_mps"] = match_error
            out.loc[row_slice, "gap_same_player_dribble_candidate"] = dribble
            out.loc[row_slice, "gap_tackle_risk_candidate"] = tackle
    return out


def add_reachability_features(
    features: pd.DataFrame,
    team_by_slot: dict[int, str],
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Estimate which team can reach the ball first.

    For each visible slot, closing speed is the player's velocity projected onto
    the player-to-ball direction. Arrival time is distance divided by positive
    closing speed, with a small floor to keep the feature finite. This is a
    reachability proxy, not a claim that a player will continue unchanged.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    n_frames = len(out)
    fps = float(_value(rules, "timeline", "fps"))
    pitch_length = float(_value(rules, "pitch", "length_m"))
    pitch_width = float(_value(rules, "pitch", "width_m"))
    min_closing_speed = 0.5
    ball_x = _numeric(out, "ball_x_interp" if "ball_x_interp" in out else "ball_x").to_numpy(float)
    ball_y = _numeric(out, "ball_y_interp" if "ball_y_interp" in out else "ball_y").to_numpy(float)
    team_results: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {"H": [], "A": []}

    for slot, team in team_by_slot.items():
        x = _numeric(out, f"{slot}_x").to_numpy(float)
        y = _numeric(out, f"{slot}_y").to_numpy(float)
        dx = (ball_x - x) * pitch_length
        dy = (ball_y - y) * pitch_width
        distance = np.hypot(dx, dy)
        player_vx = np.diff(x, prepend=np.nan) * pitch_length * fps
        player_vy = np.diff(y, prepend=np.nan) * pitch_width * fps
        direction_x = np.divide(dx, distance, out=np.zeros(n_frames), where=np.isfinite(distance) & (distance > 0))
        direction_y = np.divide(dy, distance, out=np.zeros(n_frames), where=np.isfinite(distance) & (distance > 0))
        closing_speed = player_vx * direction_x + player_vy * direction_y
        closing_speed = np.where(np.isfinite(closing_speed), np.maximum(closing_speed, 0.0), np.nan)
        eta = distance / np.maximum(closing_speed, min_closing_speed)
        eta[~np.isfinite(distance)] = np.nan
        team_results[team].append((eta, closing_speed, distance))

    for team in ("H", "A"):
        if not team_results[team]:
            out[f"{team.lower()}_fastest_arrival_s"] = np.nan
            out[f"{team.lower()}_fastest_arrival_slot"] = np.nan
            out[f"{team.lower()}_fastest_closing_speed_mps"] = np.nan
            out[f"{team.lower()}_second_arrival_s"] = np.nan
            out[f"{team.lower()}_arrival_consensus_score"] = 0.0
            out[f"{team.lower()}_arrival_consensus_fraction"] = 0.0
            continue
        etas = np.column_stack([result[0] for result in team_results[team]])
        closings = np.column_stack([result[1] for result in team_results[team]])
        best = np.nanargmin(np.where(np.isfinite(etas), etas, np.inf), axis=1)
        best_eta = etas[np.arange(n_frames), best]
        best_closing = closings[np.arange(n_frames), best]
        no_candidate = ~np.isfinite(best_eta)
        best_eta[no_candidate] = np.nan
        best_closing[no_candidate] = np.nan
        slots = [slot for slot, slot_team in team_by_slot.items() if slot_team == team]
        best_slot = np.asarray([slots[index] for index in best], dtype=float)
        best_slot[no_candidate] = np.nan
        out[f"{team.lower()}_fastest_arrival_s"] = best_eta
        out[f"{team.lower()}_fastest_arrival_slot"] = best_slot
        out[f"{team.lower()}_fastest_closing_speed_mps"] = best_closing
        # A single fastest player is fragile around crossings and tracking
        # noise. Keep that ETA for compatibility, but also expose team-level
        # consensus from the best three reachable players. This is evidence,
        # not a possession decision.
        eta_scale = max(
            float(_value(rules, "reception", "projection_horizon_seconds")),
            1.0,
        )
        sorted_etas = np.sort(np.where(np.isfinite(etas), etas, np.inf), axis=1)
        second_eta = sorted_etas[:, 1] if sorted_etas.shape[1] > 1 else sorted_etas[:, 0]
        second_eta[~np.isfinite(second_eta)] = np.nan
        top_k = min(3, sorted_etas.shape[1])
        top_etas = sorted_etas[:, :top_k]
        top_support = np.exp(-np.minimum(top_etas, 4.0 * eta_scale) / eta_scale)
        top_support[~np.isfinite(top_etas)] = 0.0
        out[f"{team.lower()}_second_arrival_s"] = second_eta
        out[f"{team.lower()}_arrival_consensus_score"] = np.mean(top_support, axis=1)
        out[f"{team.lower()}_arrival_consensus_fraction"] = np.mean(
            top_etas <= (1.5 * eta_scale), axis=1
        )

    home_eta = _numeric(out, "h_fastest_arrival_s")
    away_eta = _numeric(out, "a_fastest_arrival_s")
    out["arrival_time_margin_s"] = away_eta - home_eta
    out["likely_receiver_team"] = np.select(
        [home_eta < away_eta, away_eta < home_eta], ["H", "A"], default=None
    )
    horizon = float(_value(rules, "reception", "projection_horizon_seconds"))
    ball_vx = _numeric(out, "ball_x_interp").diff() * pitch_length * fps
    ball_vy = _numeric(out, "ball_y_interp").diff() * pitch_width * fps
    projected_x = _numeric(out, "ball_x_interp") + ball_vx * horizon / pitch_length
    projected_y = _numeric(out, "ball_y_interp") + ball_vy * horizon / pitch_width
    projected_results: dict[str, list[np.ndarray]] = {"H": [], "A": []}
    for slot, team in team_by_slot.items():
        x = _numeric(out, f"{slot}_x")
        y = _numeric(out, f"{slot}_y")
        projected_results[team].append(np.hypot(
            (x - projected_x) * pitch_length,
            (y - projected_y) * pitch_width,
        ).to_numpy(float))
    for team in ("H", "A"):
        if projected_results[team]:
            distances = np.column_stack(projected_results[team])
            best_distance = np.full(n_frames, np.nan, dtype=float)
            valid_rows = np.isfinite(distances).any(axis=1)
            best_distance[valid_rows] = np.nanmin(distances[valid_rows], axis=1)
            out[f"{team.lower()}_projected_path_distance_m"] = best_distance
        else:
            out[f"{team.lower()}_projected_path_distance_m"] = np.nan
    out["projected_receiver_team"] = np.select(
        [
            _numeric(out, "h_projected_path_distance_m") < _numeric(out, "a_projected_path_distance_m"),
            _numeric(out, "a_projected_path_distance_m") < _numeric(out, "h_projected_path_distance_m"),
        ],
        ["H", "A"],
        default=None,
    )
    closest_team = out.get("closest_team", pd.Series(None, index=out.index, dtype=object))
    receiver_team_agreement = (
        out["likely_receiver_team"].notna()
        & out["projected_receiver_team"].notna()
        & out["likely_receiver_team"].eq(out["projected_receiver_team"])
    )
    nearest_disagrees = (
        np.isin(closest_team, ["H", "A"])
        & (
            (out["likely_receiver_team"].notna() & out["likely_receiver_team"].ne(closest_team))
            | (out["projected_receiver_team"].notna() & out["projected_receiver_team"].ne(closest_team))
        )
    )
    out["receiver_arrival_path_agreement"] = receiver_team_agreement
    consensus_h = _numeric(out, "h_arrival_consensus_score")
    consensus_a = _numeric(out, "a_arrival_consensus_score")
    consensus_margin = consensus_h - consensus_a
    out["arrival_consensus_margin"] = consensus_margin
    out["arrival_consensus_team"] = np.select(
        [consensus_margin >= 0.12, consensus_margin <= -0.12],
        ["H", "A"],
        default=None,
    )
    out["arrival_consensus_agreement"] = (
        out["arrival_consensus_team"].notna()
        & out["arrival_consensus_team"].eq(out["likely_receiver_team"])
    )
    out["nearest_player_receiver_disagreement"] = nearest_disagrees
    out["nearest_player_unreliable_candidate"] = (
        nearest_disagrees
        | (_numeric(out, "team_distance_margin_m").abs() < 1.0)
        | (_numeric(out, "ball_speed_derived_mps") >= 3.0)
        | (_numeric(out, "candidate_opponent_density_5m") >= 2.0)
    )
    out["long_ball_candidate"] = _numeric(out, "ball_speed_derived_mps") >= float(
        _value(rules, "bridging", "minimum_ball_speed_mps")
    )
    return out


def add_transition_forecast_features(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Estimate which team is more likely to control the next state.

    This is a feature layer, not a possession label. It combines opponent
    closing speed, arrival-time advantage, projected-path receiver, and local
    pressure. The result is intentionally inspectable and is not allowed to
    override hard anchors by itself.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    current = out.get("closest_team", pd.Series(None, index=out.index, dtype=object))
    pressure = _numeric(out, "candidate_pressure_m")
    density = _numeric(out, "candidate_opponent_density_5m")
    isolation = out.get("candidate_isolated", pd.Series(False, index=out.index)).fillna(False)
    arrival = _numeric(out, "arrival_time_margin_s")
    projected = out.get("projected_receiver_team", pd.Series(None, index=out.index, dtype=object))
    opponent_closing = np.where(
        current == "H",
        _numeric(out, "nearest_a_closing_rate_mps"),
        np.where(current == "A", _numeric(out, "nearest_h_closing_rate_mps"), np.nan),
    )
    opponent_closing_score = np.clip(np.nan_to_num(opponent_closing, nan=0.0) / 3.0, 0.0, 1.0)
    arrival_score = np.where(
        current == "H",
        np.clip(-arrival.to_numpy(float) / 2.0, 0.0, 1.0),
        np.where(current == "A", np.clip(arrival.to_numpy(float) / 2.0, 0.0, 1.0), 0.0),
    )
    projected_opponent = (
        ((current == "H") & (projected == "A"))
        | ((current == "A") & (projected == "H"))
    ).astype(float)
    pressure_score = np.clip(
        0.5 * (pressure <= 3.0).astype(float)
        + 0.3 * (density >= 2).astype(float)
        + 0.2 * isolation.astype(float),
        0.0,
        1.0,
    )
    risk = (
        0.35 * projected_opponent
        + 0.30 * arrival_score
        + 0.20 * opponent_closing_score
        + 0.15 * pressure_score
    )
    threshold = float(_value(rules, "transition_forecast", "switch_risk_threshold"))
    forecast = np.full(len(out), None, dtype=object)
    forecast[(current == "H") & (risk >= threshold)] = "A"
    forecast[(current == "A") & (risk >= threshold)] = "H"
    out["opponent_closing_score"] = opponent_closing_score
    out["arrival_switch_score"] = arrival_score
    out["projected_opponent_receiver"] = projected_opponent.astype(bool)
    out["pressure_switch_score"] = pressure_score
    out["switch_risk_score"] = risk
    out["forecast_next_team"] = forecast
    return out


def add_switch_confirmation_features(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build an inspectable switch score from receiver competition evidence.

    This is deliberately separate from nearest-player possession. It combines
    arrival advantage, projected-path agreement, ball deceleration, and the
    candidate receiver's closing speed. The output is evidence for a switch,
    not a label and not a fit to public ground truth.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    current = out.get("closest_team", pd.Series(None, index=out.index, dtype=object))
    arrival = _numeric(out, "arrival_time_margin_s")
    likely = out.get("likely_receiver_team", pd.Series(None, index=out.index, dtype=object))
    projected = out.get("projected_receiver_team", pd.Series(None, index=out.index, dtype=object))
    speed_drop = _numeric(out, "ball_speed_drop_mps")
    h_closing = _numeric(out, "h_fastest_closing_speed_mps")
    a_closing = _numeric(out, "a_fastest_closing_speed_mps")

    arrival_advantage = np.where(
        current == "H", np.clip(-arrival.to_numpy(float) / 2.0, 0.0, 1.0),
        np.where(current == "A", np.clip(arrival.to_numpy(float) / 2.0, 0.0, 1.0), 0.0),
    )
    projected_opponent = (
        ((current == "H") & (projected == "A"))
        | ((current == "A") & (projected == "H"))
    ).astype(float)
    likely_opponent = (
        ((current == "H") & (likely == "A"))
        | ((current == "A") & (likely == "H"))
    ).astype(float)
    receiver_agreement = (
        likely.notna() & projected.notna() & likely.eq(projected)
    ).astype(float).to_numpy()
    receiver_closing = np.where(
        likely == "H", h_closing.to_numpy(float),
        np.where(likely == "A", a_closing.to_numpy(float), np.nan),
    )
    closing_score = np.clip(np.nan_to_num(receiver_closing, nan=0.0) / 3.0, 0.0, 1.0)
    deceleration_score = np.clip(np.nan_to_num(speed_drop.to_numpy(float), nan=0.0) / 5.0, 0.0, 1.0)

    score = (
        0.30 * arrival_advantage
        + 0.25 * projected_opponent
        + 0.20 * likely_opponent
        + 0.15 * receiver_agreement
        + 0.05 * closing_score
        + 0.05 * deceleration_score
    )
    switch_team = np.full(len(out), None, dtype=object)
    switch_team[(current == "H") & (score >= 0.65)] = "A"
    switch_team[(current == "A") & (score >= 0.65)] = "H"
    out["switch_confirmation_score"] = score
    out["switch_confirmation_team"] = switch_team
    out["switch_receiver_agreement"] = receiver_agreement.astype(bool)
    out["switch_arrival_advantage"] = arrival_advantage
    out["switch_deceleration_score"] = deceleration_score
    # A switch score is only a candidate at frame t. Confirm it using the
    # following short run of team evidence; immediately after a turnover both
    # teams can be close, so keep the state sticky/contested until persistence.
    follow_window = int(_value(rules, "transition_forecast", "followthrough_window_frames"))
    confirm_min = int(_value(rules, "transition_forecast", "confirmation_min_frames"))
    observed_team = current.where(current.isin(["H", "A"]), likely)
    observed_team = observed_team.where(observed_team.isin(["H", "A"]), projected)
    follow_frames = np.zeros(len(out), dtype=int)
    for index, proposed in enumerate(switch_team):
        if proposed not in {"H", "A"}:
            continue
        future = observed_team.iloc[index + 1:index + 1 + follow_window]
        follow_frames[index] = int(future.eq(proposed).sum())
    follow_score = np.clip(follow_frames / max(follow_window, 1), 0.0, 1.0)
    transition_confidence = np.clip(0.65 * score + 0.35 * follow_score, 0.0, 1.0)
    transition_state = np.full(len(out), "NO_SWITCH", dtype=object)
    candidate_mask = switch_team != None
    transition_state[candidate_mask] = "SWITCH_CANDIDATE_STICKY"
    transition_state[candidate_mask & (follow_frames >= confirm_min)] = "SWITCH_CONFIRMED"
    out["transition_followthrough_frames"] = follow_frames
    out["transition_followthrough_score"] = follow_score
    out["transition_confidence"] = transition_confidence
    out["transition_state"] = transition_state
    return out


def add_anchor_forecast_features(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fast-forward candidate transitions until the next plausible touch.

    This is intentionally anchor-only. It forecasts the first future frame
    with reception/control evidence, then measures persistence after that
    frame. It does not rewrite the transition interval into H/A by itself.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    lookahead = int(_value(rules, "transition_forecast", "anchor_lookahead_frames"))
    minimum_support = int(_value(rules, "transition_forecast", "confirmation_min_frames"))
    current = out.get("closest_team", pd.Series(None, index=out.index, dtype=object))
    likely = out.get("likely_receiver_team", pd.Series(None, index=out.index, dtype=object))
    projected = out.get("projected_receiver_team", pd.Series(None, index=out.index, dtype=object))
    receiver = likely.where(likely.isin(["H", "A"]), projected)
    touch = (
        out.get("genuine_reception_strict_anchor", pd.Series(False, index=out.index)).fillna(False)
        | out.get("controlled_reception_signal", pd.Series(False, index=out.index)).fillna(False)
        | out.get("genuine_reception_anchor", pd.Series(False, index=out.index)).fillna(False)
    )
    anchor = out.get("anchor_team", pd.Series(None, index=out.index, dtype=object))
    previous_anchor = anchor.where(anchor.isin(["H", "A"])).ffill().shift()
    forecast_team = np.full(len(out), None, dtype=object)
    touch_frame = np.full(len(out), np.nan, dtype=float)
    touch_offset = np.full(len(out), np.nan, dtype=float)
    persistence = np.zeros(len(out), dtype=int)
    touch_low_pressure = np.zeros(len(out), dtype=bool)
    same_team_forecast = np.zeros(len(out), dtype=bool)
    base_confidence = pd.to_numeric(
        out.get("transition_confidence", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0).to_numpy(float)
    for index in range(len(out)):
        proposed = receiver.iloc[index]
        if proposed not in {"H", "A"}:
            proposed = out.get("switch_confirmation_team", pd.Series(None, index=out.index)).iloc[index]
        if proposed not in {"H", "A"}:
            continue
        end = min(len(out), index + 1 + lookahead)
        future_touch = np.flatnonzero(touch.iloc[index + 1:end].to_numpy(bool))
        if len(future_touch) == 0:
            continue
        first = index + 1 + int(future_touch[0])
        forecast_team[index] = proposed
        touch_frame[index] = float(out.iloc[first].frame)
        touch_offset[index] = float(first - index)
        same_team_forecast[index] = previous_anchor.iloc[index] == proposed
        touch_low_pressure[index] = bool(
            pd.to_numeric(out.get("candidate_pressure_m", pd.Series(np.nan, index=out.index)), errors="coerce").iloc[first] >= float(_value(rules, "anchors", "low_pressure_opponent_distance_m"))
            and pd.to_numeric(out.get("candidate_opponent_density_5m", pd.Series(np.nan, index=out.index)), errors="coerce").iloc[first] <= 1
        )
        future_anchor = anchor.iloc[first:min(len(out), first + lookahead)]
        persistence[index] = int(future_anchor.eq(proposed).sum())
    touch_score = np.where(
        np.isfinite(touch_offset),
        np.clip(1.0 - (touch_offset - 1.0) / max(lookahead, 1), 0.0, 1.0),
        0.0,
    )
    persistence_score = np.clip(persistence / max(minimum_support, 1), 0.0, 1.0)
    out["forecast_touch_team"] = forecast_team
    out["forecast_touch_frame"] = touch_frame
    out["forecast_touch_offset_frames"] = touch_offset
    out["forecast_post_touch_anchor_frames"] = persistence
    out["forecast_same_team_reception"] = same_team_forecast
    out["forecast_touch_low_pressure"] = touch_low_pressure
    out["anchor_forecast_confidence"] = np.clip(
        0.45 * base_confidence + 0.30 * touch_score + 0.25 * persistence_score,
        0.0,
        1.0,
    )
    # Same-team: a clean touch can immediately refresh the anchor. Opponent:
    # require the touch plus a persistent post-touch run to avoid spell flicker.
    out["same_team_touch_anchor_candidate"] = (
        pd.Series(forecast_team, index=out.index).isin(["H", "A"])
        & pd.Series(same_team_forecast, index=out.index)
        & pd.Series(touch_low_pressure, index=out.index)
    )
    out["opponent_touch_persistence_candidate"] = (
        pd.Series(forecast_team, index=out.index).isin(["H", "A"])
        & ~pd.Series(same_team_forecast, index=out.index)
        & (persistence >= minimum_support)
    )
    out["anchor_forecast_confirmed"] = (
        out["same_team_touch_anchor_candidate"]
        | out["opponent_touch_persistence_candidate"]
    )
    return out


def add_anchor_features(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Mark algorithmically high-confidence H/A anchors."""
    rules = load_rules() if rules is None else rules
    max_distance = float(_value(rules, "anchors", "stable_control_max_distance_m"))
    min_margin = float(_value(rules, "anchors", "minimum_distance_margin_m"))
    min_persistence = float(_value(rules, "anchors", "minimum_team_persistence"))
    max_distance_std = float(_value(rules, "anchors", "stable_distance_std_max_m"))
    max_ball_speed = float(_value(rules, "anchors", "stable_ball_speed_max_mps"))
    out = features.copy()
    anchor = pd.Series(None, index=out.index, dtype=object)
    home = (
        (out.get("closest_team") == "H")
        & (_numeric(out, "closest_distance_m") <= max_distance)
        & (_numeric(out, "team_distance_margin_m") >= min_margin)
        & (_numeric(out, "home_persistence") >= min_persistence)
        & (_numeric(out, "nearest_h_distance_roll_std_m") <= max_distance_std)
        & (_numeric(out, "ball_speed_derived_mps") <= max_ball_speed)
    )
    away = (
        (out.get("closest_team") == "A")
        & (_numeric(out, "closest_distance_m") <= max_distance)
        & (_numeric(out, "team_distance_margin_m") <= -min_margin)
        & (_numeric(out, "away_persistence") >= min_persistence)
        & (_numeric(out, "nearest_a_distance_roll_std_m") <= max_distance_std)
        & (_numeric(out, "ball_speed_derived_mps") <= max_ball_speed)
    )
    anchor.loc[home] = "H"
    anchor.loc[away] = "A"
    out["anchor_team"] = anchor
    out["anchor_confident"] = anchor.notna()
    out["anchor_confidence_score"] = np.select(
        [home, away],
        [
            np.minimum(_numeric(out, "home_persistence"), _numeric(out, "team_distance_margin_m") / (min_margin * 2)),
            np.minimum(_numeric(out, "away_persistence"), -_numeric(out, "team_distance_margin_m") / (min_margin * 2)),
        ],
        default=0.0,
    ).clip(0.0, 1.0)
    # Keep the deterministic anchor gate above unchanged, but expose the
    # underlying evidence vector so later calibration can learn reliability
    # without collapsing heterogeneous observations into one opaque number.
    distance = _numeric(out, "closest_distance_m")
    margin = _numeric(out, "team_distance_margin_m")
    persistence = np.where(
        out.get("closest_team", pd.Series(None, index=out.index)).eq("H"),
        _numeric(out, "home_persistence"),
        np.where(
            out.get("closest_team", pd.Series(None, index=out.index)).eq("A"),
            _numeric(out, "away_persistence"),
            np.nan,
        ),
    )
    distance_std = np.where(
        out.get("closest_team", pd.Series(None, index=out.index)).eq("H"),
        _numeric(out, "nearest_h_distance_roll_std_m"),
        np.where(
            out.get("closest_team", pd.Series(None, index=out.index)).eq("A"),
            _numeric(out, "nearest_a_distance_roll_std_m"),
            np.nan,
        ),
    )
    ball_speed = _numeric(out, "ball_speed_derived_mps")
    pressure = _numeric(out, "candidate_pressure_m")
    density = _numeric(out, "candidate_opponent_density_5m")
    out["anchor_evidence_distance_score"] = np.clip(
        1.0 - np.nan_to_num(distance, nan=max_distance * 2.0) / max(max_distance, 1e-6),
        0.0,
        1.0,
    )
    out["anchor_evidence_margin_score"] = np.clip(
        np.abs(np.nan_to_num(margin, nan=0.0)) / max(min_margin * 2.0, 1e-6),
        0.0,
        1.0,
    )
    out["anchor_evidence_persistence_score"] = np.clip(
        np.nan_to_num(persistence, nan=0.0), 0.0, 1.0
    )
    out["anchor_evidence_distance_stability_score"] = np.clip(
        1.0 - np.nan_to_num(distance_std, nan=max_distance_std * 2.0) / max(max_distance_std, 1e-6),
        0.0,
        1.0,
    )
    out["anchor_evidence_ball_control_score"] = np.clip(
        1.0 - np.nan_to_num(ball_speed, nan=max_ball_speed * 2.0) / max(max_ball_speed, 1e-6),
        0.0,
        1.0,
    )
    out["anchor_evidence_low_pressure_score"] = np.clip(
        np.nan_to_num(pressure, nan=0.0) / max(float(_value(rules, "anchors", "low_pressure_opponent_distance_m")), 1e-6),
        0.0,
        1.0,
    )
    out["anchor_evidence_density_score"] = np.clip(
        1.0 - np.nan_to_num(density, nan=3.0) / 3.0,
        0.0,
        1.0,
    )
    evidence_components = np.column_stack(
        [
            out["anchor_evidence_distance_score"].to_numpy(float),
            out["anchor_evidence_margin_score"].to_numpy(float),
            out["anchor_evidence_persistence_score"].to_numpy(float),
            out["anchor_evidence_distance_stability_score"].to_numpy(float),
            out["anchor_evidence_ball_control_score"].to_numpy(float),
            out["anchor_evidence_low_pressure_score"].to_numpy(float),
            out["anchor_evidence_density_score"].to_numpy(float),
        ]
    )
    out["anchor_evidence_vector_score"] = np.nanmean(evidence_components, axis=1)
    out["anchor_evidence_component_count"] = np.sum(evidence_components >= 0.5, axis=1)
    low_pressure = (
        (_numeric(out, "candidate_pressure_m") >= float(_value(rules, "anchors", "low_pressure_opponent_distance_m")))
        & (_numeric(out, "candidate_opponent_density_5m") <= 1)
        & (_numeric(out, "candidate_teammate_support_10m") >= 1)
    )
    pressurized_control = (
        (_numeric(out, "closest_distance_m") <= max_distance * 0.625)
        & (_numeric(out, "team_distance_margin_m").abs() >= min_margin * 1.5)
        & (np.maximum(_numeric(out, "home_persistence"), _numeric(out, "away_persistence")) >= 0.90)
        & (_numeric(out, "candidate_opponent_density_5m") <= 2)
    )
    pressure_anchor = pd.Series(None, index=out.index, dtype=object)
    pressure_anchor.loc[home & (low_pressure | pressurized_control)] = "H"
    pressure_anchor.loc[away & (low_pressure | pressurized_control)] = "A"
    out["pressure_anchor_team"] = pressure_anchor
    out["pressure_anchor"] = pressure_anchor.notna()
    out["low_pressure_control"] = low_pressure
    out["pressurized_control"] = pressurized_control
    return out


def add_pipeline_evidence_features(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
    team_by_slot: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Expose the staged evidence used by the possession pipeline.

    These columns are diagnostics and confidence modifiers, not direct H/A/D
    labels.  The order is deliberate: ball/restart evidence is checked first,
    then geometry/pressure, then local low-speed control.  Long-ball evidence
    reduces ordinary control confidence until a later reception anchor agrees.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    speed = _numeric(out, "ball_speed_derived_mps")
    acceleration = _numeric(out, "ball_acceleration_mps2")
    missing = _numeric(out, "ball_x").isna() | _numeric(out, "ball_y").isna()
    near_zero = float(_value(rules, "restart", "active_ball_speed_mps"))
    local_control_max = float(_value(rules, "anchors", "local_control_speed_max_mps"))
    long_speed = float(_value(rules, "bridging", "minimum_ball_speed_mps"))
    long_acceleration = float(_value(rules, "bridging", "long_ball_acceleration_mps2"))
    control_zone_radius = float(_value(rules, "anchors", "stable_control_max_distance_m"))
    direction_change = _numeric(out, "ball_direction_change_rad")

    out["ball_missing_observed"] = missing
    out["ball_near_zero_candidate"] = speed.le(near_zero) & ~missing
    out["restart_or_out_of_play_evidence"] = (
        out.get("smart_set_piece", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        | out.get("ball_out_of_play_anchor", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        | out.get("restart_context", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    )

    local_min = speed <= speed.rolling(9, center=True, min_periods=3).min().add(0.05)
    close = _numeric(out, "closest_distance_m") <= float(_value(rules, "reception_signal", "near_ball_distance_m"))
    margin = _numeric(out, "team_distance_margin_m").abs()
    out["local_low_speed_control_candidate"] = (
        speed.le(local_control_max) & local_min.fillna(False) & close & margin.ge(0.5) & ~missing
    )

    long_signal = (
        speed.ge(long_speed)
        & (acceleration.ge(long_acceleration) | speed.shift(1).ge(long_speed))
        & ~missing
    )
    long_score = (
        0.45 * np.clip((speed.fillna(0.0) - long_speed) / max(long_speed, 1.0), 0.0, 1.0)
        + 0.35 * np.clip(acceleration.fillna(0.0) / max(long_acceleration, 1.0), 0.0, 1.0)
        + 0.20 * long_signal.astype(float)
    ).clip(0.0, 1.0)
    out["long_ball_evidence"] = long_signal
    out["long_ball_evidence_score"] = long_score

    # Physics vetoes: these are scenario evidence, never direct possession
    # labels. A sharp deceleration is compatible with reception/control; a
    # sharp acceleration is compatible with a pass or deflection. Acceleration
    # plus a material direction change is specifically a miscontrol/deflection
    # candidate rather than ordinary straight-line ball flight.
    sharp_deceleration = _numeric(out, "ball_speed_drop_mps") >= max(
        2.0, float(_value(rules, "reception", "minimum_speed_drop_mps"))
    )
    sharp_acceleration = acceleration >= long_acceleration
    direction_change_signal = direction_change >= 0.35
    out["sharp_deceleration_reception_veto"] = sharp_deceleration & ~missing
    out["sharp_acceleration_pass_veto"] = sharp_acceleration & ~missing
    out["deflection_direction_change_veto"] = (
        sharp_acceleration & direction_change_signal & ~missing
    )
    out["direction_change_pass_or_deflection_candidate"] = (
        direction_change_signal & speed.ge(3.0) & ~missing
    )
    out["post_long_ball_direction_change_risk"] = (
        out["long_ball_evidence"] & direction_change_signal & ~missing
    )

    # Landing/contest evidence is deliberately exposed as raw observations.
    # It is not converted into a short/long class or a possession decision.
    # Reviewers can therefore see whether a high-speed ball has travelled,
    # slowed, changed direction, and entered a crowded or converging area.
    flight_distance_window = max(1, int(_value(rules, "reception", "pre_flight_window_frames")))
    out["ball_distance_travelled_recent_m"] = (
        out["ball_displacement_m"].where(~missing).rolling(flight_distance_window, min_periods=1).sum()
    )
    out["ball_peak_speed_recent_mps"] = speed.where(~missing).rolling(
        flight_distance_window, min_periods=1
    ).max()
    out["ball_speed_change_rate_mps2"] = acceleration
    out["landing_opponent_density_5m"] = _numeric(out, "candidate_opponent_density_5m")
    current_team_for_landing = out.get("closest_team", pd.Series(None, index=out.index, dtype=object))
    out["landing_opponent_closing_speed_mps"] = pd.Series(
        np.where(
            current_team_for_landing.eq("H"), _numeric(out, "nearest_a_closing_rate_mps"),
            np.where(current_team_for_landing.eq("A"), _numeric(out, "nearest_h_closing_rate_mps"), np.nan),
        ),
        index=out.index,
    )
    out["landing_opponent_approach_speed_mps"] = out["landing_opponent_closing_speed_mps"].clip(lower=0.0)
    out["landing_arrival_time_margin_s"] = _numeric(out, "arrival_time_margin_s")
    out["landing_pass_options_15m"] = _numeric(out, "candidate_pass_options_15m")
    out["landing_player_density_signal"] = (
        out["landing_opponent_density_5m"]
        .fillna(0.0)
        .add(_numeric(out, "candidate_teammate_support_10m").fillna(0.0))
    )
    # Project the observed direction onto visible player locations. This is
    # geometric evidence, not proof of a touch; a later reception anchor must
    # confirm whether the target was actually reached.
    target_team = np.full(len(out), None, dtype=object)
    target_slot = np.full(len(out), np.nan, dtype=float)
    target_angle = np.full(len(out), np.nan, dtype=float)
    target_distance = np.full(len(out), np.nan, dtype=float)
    if team_by_slot:
        ball_x = _numeric(out, "ball_x").to_numpy(float)
        ball_y = _numeric(out, "ball_y").to_numpy(float)
        direction = _numeric(out, "ball_direction_rad").to_numpy(float)
        length_m = float(_value(rules, "pitch", "length_m"))
        width_m = float(_value(rules, "pitch", "width_m"))
        slot_coordinates = {
            int(slot): (
                pd.to_numeric(out.get(f"{slot}_x", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float),
                pd.to_numeric(out.get(f"{slot}_y", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float),
            )
            for slot in team_by_slot
        }
        for index in range(len(out)):
            if not np.isfinite(ball_x[index]) or not np.isfinite(ball_y[index]) or not np.isfinite(direction[index]):
                continue
            candidates = []
            for slot, team in team_by_slot.items():
                px, py = slot_coordinates[int(slot)]
                if not np.isfinite(px[index]) or not np.isfinite(py[index]):
                    continue
                dx = (px[index] - ball_x[index]) * length_m
                dy = (py[index] - ball_y[index]) * width_m
                distance = float(np.hypot(dx, dy))
                target_direction = np.arctan2(dy, dx)
                angle = float(abs(np.arctan2(np.sin(direction[index] - target_direction), np.cos(direction[index] - target_direction))))
                if distance > control_zone_radius and angle <= 0.52:
                    candidates.append((angle, distance, int(slot), team))
            if candidates:
                angle, distance, slot, team = min(candidates, key=lambda item: (item[0], item[1]))
                target_angle[index], target_distance[index] = angle, distance
                target_slot[index], target_team[index] = slot, team
    out["direction_target_team"] = target_team
    out["direction_target_slot"] = target_slot
    out["direction_target_angle_error_rad"] = target_angle
    out["direction_target_distance_m"] = target_distance
    out["direction_target_found"] = pd.Series(target_team, index=out.index).notna()
    out["high_speed_no_control_signal"] = (
        speed.ge(long_speed)
        & (_numeric(out, "closest_distance_m") > control_zone_radius)
        & ~missing
    )
    out["high_speed_no_control_score"] = (
        0.6 * speed.ge(long_speed).astype(float)
        + 0.25 * np.clip(
            (_numeric(out, "closest_distance_m") - control_zone_radius) / 4.0,
            0.0,
            1.0,
        )
        + 0.15 * out["deflection_direction_change_veto"].astype(float)
    ).where(~missing, 0.0).clip(0.0, 1.0)

    closest_team = out.get("closest_team", pd.Series(None, index=out.index, dtype=object))
    opponent_closing = pd.Series(
        np.where(
            closest_team.eq("H"), _numeric(out, "nearest_a_closing_rate_mps"),
            np.where(closest_team.eq("A"), _numeric(out, "nearest_h_closing_rate_mps"), np.nan),
        ),
        index=out.index,
    )
    density = _numeric(out, "candidate_opponent_density_5m")
    pressure = _numeric(out, "candidate_pressure_m")
    pass_options = _numeric(out, "candidate_pass_options_15m")
    out["opponent_closing_speed_mps"] = opponent_closing
    out["opponent_directional_coverage_score"] = (
        0.45 * np.clip(density.fillna(0.0) / 3.0, 0.0, 1.0)
        + 0.35 * np.clip(opponent_closing.fillna(0.0) / 2.0, 0.0, 1.0)
        + 0.20 * np.clip((3.0 - pressure.fillna(3.0)) / 3.0, 0.0, 1.0)
    ).clip(0.0, 1.0)
    out["short_option_blocked"] = (
        density.ge(2) & pressure.le(3.0) & pass_options.fillna(0.0).le(0)
    )
    out["gegenpress_candidate"] = (
        out["short_option_blocked"]
        & opponent_closing.ge(0.5)
        & out["opponent_directional_coverage_score"].ge(0.55)
    )
    out["gegenpress_score"] = (
        out["opponent_directional_coverage_score"]
        * out["short_option_blocked"].astype(float)
    ).clip(0.0, 1.0)

    penalty = float(_value(rules, "anchors", "long_ball_confidence_penalty"))
    control_modifier = (1.0 - penalty * long_score).clip(0.0, 1.0)
    out["control_confidence_modifier"] = control_modifier
    if "anchor_confidence_score" in out:
        out["anchor_confidence_score_adjusted"] = (
            _numeric(out, "anchor_confidence_score") * control_modifier
        ).clip(0.0, 1.0)
    return out


def add_ball_kick_episode_features(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Group sharp ball departures into auditable kick-motion episodes.

    A kick episode starts at a meaningful acceleration or speed crossing. It
    ends at a controlled deceleration/reception, or immediately before the next
    sharp acceleration. Ordinary slowing after a speed peak is not enough by
    itself; the end also needs a control-speed/reception condition. These are
    motion episodes for plots and trust features, not possession labels.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    observed_speed = _numeric(out, "ball_speed_derived_mps").to_numpy(float)
    observed_acceleration = _numeric(out, "ball_acceleration_mps2").to_numpy(float)
    observed_direction = _numeric(out, "ball_direction_change_rad").to_numpy(float)
    missing = _numeric(out, "ball_x").isna().to_numpy(bool) | _numeric(out, "ball_y").isna().to_numpy(bool)
    kalman_reliable = out.get(
        "ball_kalman_reliable", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    kalman_supported = missing & kalman_reliable
    kalman_speed = _numeric(out, "ball_kalman_speed_mps").to_numpy(float)
    kalman_acceleration = _numeric(out, "ball_kalman_acceleration_mps2").to_numpy(float)
    kalman_direction = _numeric(out, "ball_kalman_direction_change_rad").to_numpy(float)
    kalman_displacement = _numeric(out, "ball_kalman_displacement_m").to_numpy(float)
    speed = np.where(kalman_supported & np.isfinite(kalman_speed), kalman_speed, observed_speed)
    acceleration = np.where(
        kalman_supported & np.isfinite(kalman_acceleration),
        kalman_acceleration,
        observed_acceleration,
    )
    direction = np.where(
        kalman_supported & np.isfinite(kalman_direction),
        kalman_direction,
        observed_direction,
    )
    displacement_observed = _numeric(out, "ball_displacement_m").to_numpy(float)
    displacement = np.where(
        kalman_supported & np.isfinite(kalman_displacement),
        kalman_displacement,
        displacement_observed,
    )
    george_supported = out.get(
        "ball_fill_eligible", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool) & out.get(
        "ball_fill_observation_confidence", pd.Series(0.0, index=out.index)
    ).fillna(0.0).to_numpy(float).clip(0.0, 1.0).astype(bool)
    george_speed = _numeric(out, "ball_speed_george_evidence_mps").to_numpy(float)
    george_displacement = _numeric(out, "ball_displacement_george_evidence_m").to_numpy(float)
    george_direction_change = _numeric(out, "ball_fill_direction_change_rad").to_numpy(float)
    george_long_ball = out.get(
        "ball_flight_george_evidence", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    george_gap_support = missing & george_supported
    speed = np.where(george_gap_support & np.isfinite(george_speed), george_speed, speed)
    displacement = np.where(
        george_gap_support & np.isfinite(george_displacement),
        george_displacement,
        displacement,
    )
    direction = np.where(
        george_gap_support & np.isfinite(george_direction_change),
        george_direction_change,
        direction,
    )
    speed = np.nan_to_num(speed, nan=0.0)
    acceleration = np.nan_to_num(acceleration, nan=0.0)
    direction = np.nan_to_num(direction, nan=0.0)
    displacement = np.nan_to_num(displacement, nan=0.0)
    speed_drop = np.nan_to_num(np.r_[0.0, speed[:-1]] - speed, nan=0.0)
    long_speed = float(_value(rules, "bridging", "minimum_ball_speed_mps"))
    long_acceleration = float(_value(rules, "bridging", "long_ball_acceleration_mps2"))
    control_speed = float(_value(rules, "anchors", "local_control_speed_max_mps"))
    minimum_drop = float(_value(rules, "reception", "minimum_speed_drop_mps"))
    direction_start = 0.8
    starts = (
        (~missing | kalman_supported)
        & (
            (acceleration >= long_acceleration)
            | ((speed >= long_speed) & (np.r_[0.0, speed[:-1]] < long_speed))
            | ((direction >= direction_start) & (speed >= 4.0))
            | george_long_ball
        )
    )
    controlled_reception = out.get(
        "controlled_reception_signal", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)

    episode_id = np.full(len(out), -1, dtype=int)
    episode_start = np.zeros(len(out), dtype=bool)
    episode_end = np.zeros(len(out), dtype=bool)
    end_reason = np.full(len(out), None, dtype=object)
    observation_source = np.full(len(out), None, dtype=object)
    observation_source[~missing] = "OBSERVED"
    observation_source[kalman_supported] = "KALMAN"
    observation_source[george_gap_support] = "GEORGE"
    current = -1
    active = False
    for index in range(len(out)):
        if starts[index]:
            if active and index > 0:
                episode_end[index - 1] = True
                end_reason[index - 1] = "NEXT_ACCELERATION"
            current += 1
            active = True
            episode_start[index] = True
        if not active:
            continue
        episode_id[index] = current
        controlled_end = (
            index > 0
            and speed[index] <= control_speed
            and speed_drop[index] >= minimum_drop
        ) or controlled_reception[index]
        if controlled_end:
            episode_end[index] = True
            end_reason[index] = "CONTROLLED_DECELERATION"
            active = False
    if active and len(out):
        episode_end[-1] = True
        end_reason[-1] = "END_OF_DATA"

    distance = pd.Series(0.0, index=out.index, dtype=float)
    valid_episode = episode_id >= 0
    if valid_episode.any():
        displacement_series = pd.Series(displacement, index=out.index, dtype=float)
        distance.loc[valid_episode] = displacement_series.loc[valid_episode].groupby(
            episode_id[valid_episode]
        ).cumsum()
    out["kick_episode_id"] = episode_id
    out["kick_episode_start"] = episode_start
    out["kick_episode_end"] = episode_end
    out["kick_episode_end_reason"] = end_reason
    out["kick_episode_distance_m"] = distance
    out["kick_episode_observation_source"] = observation_source
    out["kick_episode_kalman_supported"] = kalman_supported
    out["kick_episode_george_supported"] = george_gap_support
    out["kick_episode_long_ball_candidate"] = george_long_ball | (
        (distance.to_numpy(float) >= 3.0) & (episode_id >= 0)
    )
    out["kick_episode_contact_change"] = np.maximum(
        np.clip(np.abs(acceleration) / max(long_acceleration, 1.0), 0.0, 1.0),
        np.clip(direction / 1.2, 0.0, 1.0),
    )
    return out


def add_long_ball_origin_anchor_features(features: pd.DataFrame) -> pd.DataFrame:
    """Carry the last controlled team through an aerial flight interval.

    A long ball is still launched by the team that controlled it before the
    departure.  This feature keeps that origin team as a *continuity anchor*
    until the flight ends.  It does not claim that the origin team wins the
    landing contest: an opposing controlled reception and persistence can
    still open the H/A switch hypothesis after the landing.
    """
    out = features.copy()
    n = len(out)
    episode = pd.to_numeric(
        out.get("kick_episode_id", pd.Series(-1, index=out.index)), errors="coerce"
    ).fillna(-1).to_numpy(int)
    long_ball = out.get(
        "kick_episode_long_ball_candidate", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    tactical_gap = out.get(
        "ball_fill_during_tactical_gap", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    flight_observation = out.get(
        "ball_fill_flight_observation", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    observation_confidence = pd.to_numeric(
        out.get(
            "ball_fill_observation_confidence",
            pd.Series(0.0, index=out.index),
        ),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    # George's path is allowed to carry a continuity anchor only for the
    # problem it is meant to solve: a tactical ball gap.  On visible tactical
    # frames it remains useful geometry/provenance, but the tactical ball and
    # our own contact/arrival evidence stay authoritative.
    george_long_ball_gap = (
        long_ball
        & tactical_gap
        & flight_observation
        & (observation_confidence > 0.0)
    )
    anchor = out.get(
        "anchor_team", pd.Series(None, index=out.index, dtype=object)
    ).to_numpy(object)
    origin_team = np.full(n, None, dtype=object)
    active = np.zeros(n, dtype=bool)
    landing = np.zeros(n, dtype=bool)
    origin_confidence = np.zeros(n, dtype=float)

    for episode_id in np.unique(episode[episode >= 0]):
        positions = np.flatnonzero((episode == episode_id) & george_long_ball_gap)
        if not positions.size:
            continue
        start = int(positions[0])
        end = int(positions[-1])
        before = start
        while before >= 0 and anchor[before] not in {"H", "A"}:
            before -= 1
        if before < 0:
            continue
        team = anchor[before]
        origin_team[start : end + 1] = team
        active[start : end + 1] = True
        origin_confidence[start : end + 1] = np.clip(
            observation_confidence[start : end + 1],
            0.0,
            1.0,
        )
        # The final flight frame is a landing/control inspection boundary, not
        # an automatic team switch. The resolver needs post-flight evidence.
        landing[end] = True

    out["flight_origin_anchor_team"] = origin_team
    out["flight_origin_anchor_active"] = active
    out["flight_origin_anchor_confidence"] = origin_confidence
    out["flight_landing_boundary_candidate"] = landing
    return out


def apply_anchor_hierarchy(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Select the strongest available team anchor in configured priority order.

    This does not turn scenario evidence into possession. Flight, contest,
    occlusion, and reception candidates remain diagnostics unless a source
    supplies a team. Frames with no selected source are explicitly marked
    ``NO_ANCHOR`` for later trajectory resolution.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    hierarchy = rules.get("anchor_hierarchy", {})
    sources = hierarchy.get(
        "ordered_sources",
        ["restart_control_anchor_team", "pressure_anchor_team", "anchor_team"],
    )
    selected_team = pd.Series(None, index=out.index, dtype=object)
    selected_source = pd.Series(
        str(hierarchy.get("unresolved_source", "NO_ANCHOR")),
        index=out.index,
        dtype=object,
    )
    selected_confidence = pd.Series(0.0, index=out.index, dtype=float)
    adjusted_control = pd.to_numeric(
        out.get(
            "anchor_confidence_score_adjusted",
            out.get("anchor_confidence_score", pd.Series(0.0, index=out.index)),
        ),
        errors="coerce",
    ).fillna(0.0)
    control_modifier = pd.to_numeric(
        out.get("control_confidence_modifier", pd.Series(1.0, index=out.index)),
        errors="coerce",
    ).fillna(1.0).clip(0.0, 1.0)
    confidence_by_source = {
        "restart_control_anchor_team": control_modifier,
        "pressure_anchor_team": 0.95 * control_modifier,
        "anchor_team": adjusted_control,
    }
    for source in sources:
        if source not in out:
            continue
        values = out[source]
        if values.dtype == bool:
            candidate = out.get("anchor_team", pd.Series(None, index=out.index)).where(values)
        else:
            candidate = values.where(values.isin(["H", "A"]))
        available = selected_team.isna() & candidate.notna()
        selected_team.loc[available] = candidate.loc[available]
        selected_source.loc[available] = source
        confidence = confidence_by_source.get(source, pd.Series(1.0, index=out.index))
        if not isinstance(confidence, pd.Series):
            confidence = pd.Series(float(confidence), index=out.index)
        selected_confidence.loc[available] = confidence.loc[available].clip(0.0, 1.0)
    out["selected_anchor_team"] = selected_team
    out["selected_anchor_source"] = selected_source
    out["selected_anchor_confidence"] = selected_confidence
    out["has_selected_anchor"] = selected_team.notna()
    return out


def add_special_anchor_features(
    features: pd.DataFrame,
    events: list[dict[str, Any]],
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add restart, reception, transition, and occlusion anchor evidence.

    Smart Tagging intervals are treated as context windows, not exact restart
    instants. A special anchor is only emitted when tracking evidence confirms
    the team-level state around that context.
    """
    rules = load_rules() if rules is None else rules
    out = features.copy()
    n_frames = len(out)
    fps = float(_value(rules, "timeline", "fps"))
    window = max(1, round(float(_value(rules, "timeline", "persistence_window_seconds")) * fps))

    restart_context = np.zeros(n_frames, dtype=bool)
    restart_type = np.full(n_frames, None, dtype=object)
    restart_team_hint = np.full(n_frames, None, dtype=object)
    team_name_to_label = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}
    for event in events:
        event_type = (event.get("tags", {}).get("Type") or "").strip()
        if not event_type:
            continue
        start = max(0, int(event["start_frame"]))
        end = min(n_frames, int(event["end_frame"]) + 1)
        if start >= end:
            continue
        restart_context[start:end] = True
        restart_type[start:end] = event_type
        team_name = (event.get("tags", {}).get("Team") or "").strip().lower()
        if team_name in team_name_to_label:
            restart_team_hint[start:end] = team_name_to_label[team_name]

    out["restart_context"] = restart_context
    out["restart_type"] = restart_type
    out["restart_team_hint"] = restart_team_hint
    restart_team_hint_series = pd.Series(restart_team_hint, index=out.index, dtype=object)
    active_speed = float(_value(rules, "restart", "active_ball_speed_mps"))
    restart_series = pd.Series(restart_context, index=out.index)
    out["restart_start_candidate"] = restart_context & ~restart_series.shift(1, fill_value=False).to_numpy(bool)
    out["restart_end_candidate"] = restart_context & ~restart_series.shift(-1, fill_value=False).to_numpy(bool)
    out["restart_ball_active"] = restart_context & (_numeric(out, "ball_speed_derived_mps") >= active_speed)

    anchor = out["anchor_team"].copy()
    source = pd.Series(None, index=out.index, dtype=object)
    source.loc[anchor == "H"] = "stable_control"
    source.loc[anchor == "A"] = "stable_control"

    speed = _numeric(out, "ball_speed_derived_mps")
    post_window = max(1, int(_value(rules, "reception", "post_control_window_frames")))
    min_drop = float(_value(rules, "reception", "minimum_speed_drop_mps"))
    max_control_distance = float(_value(rules, "reception", "maximum_control_distance_m"))
    min_receiver_margin = float(_value(rules, "reception", "minimum_receiver_margin_s"))
    future_home = pd.Series((anchor == "H").to_numpy()[::-1], index=out.index).rolling(
        post_window, min_periods=1
    ).mean().to_numpy()[::-1]
    future_away = pd.Series((anchor == "A").to_numpy()[::-1], index=out.index).rolling(
        post_window, min_periods=1
    ).mean().to_numpy()[::-1]
    out["home_future_anchor_persistence"] = future_home
    out["away_future_anchor_persistence"] = future_away
    receiver_margin = _numeric(out, "arrival_time_margin_s")
    receiver_support = (
        ((anchor == "H") & (out["likely_receiver_team"] == "H") & (receiver_margin >= min_receiver_margin))
        | ((anchor == "A") & (out["likely_receiver_team"] == "A") & (receiver_margin <= -min_receiver_margin))
    )
    path_support = (
        ((anchor == "H") & (out["projected_receiver_team"] == "H"))
        | ((anchor == "A") & (out["projected_receiver_team"] == "A"))
    )
    future_control = (
        ((anchor == "H") & (future_home >= 0.6))
        | ((anchor == "A") & (future_away >= 0.6))
    )
    reception = (
        anchor.notna()
        & out["recent_flight_evidence"]
        & (
            (_numeric(out, "ball_speed_drop_mps") >= min_drop)
            | (_numeric(out, "ball_speed_derived_mps") <= float(_value(rules, "bridging", "minimum_ball_speed_mps")))
        )
        & (_numeric(out, "closest_distance_m") <= max_control_distance)
        & (receiver_support | path_support)
        & future_control
    )
    # A stronger reception event requires the receiver to appear after a
    # flight interval, not merely to be the nearest player in a fast frame.
    # The one-frame shift prevents the current control frame from creating its
    # own pre-flight evidence.
    pre_flight = out["recent_flight_evidence"].shift(1, fill_value=False).astype(bool)
    receiver_team = out["likely_receiver_team"].where(out["likely_receiver_team"].notna(), out["projected_receiver_team"])
    receiver_identity_support = (
        ((anchor == "H") & receiver_team.eq("H"))
        | ((anchor == "A") & receiver_team.eq("A"))
    )
    genuine_reception = (
        anchor.notna()
        & pre_flight
        & (
            (_numeric(out, "ball_speed_drop_mps") >= min_drop)
            | (_numeric(out, "ball_speed_derived_mps") <= float(_value(rules, "bridging", "minimum_ball_speed_mps")))
        )
        & (_numeric(out, "closest_distance_m") <= max_control_distance)
        & receiver_identity_support
        & future_control
    )
    # Offline confirmation signal: a genuine reception should coincide with a
    # local minimum in receiver distance and a short post-arrival low-speed
    # interval, rather than merely a single close frame.
    distance = _numeric(out, "closest_distance_m")
    local_distance_min = distance <= distance.rolling(5, center=True, min_periods=3).min().add(0.05)
    post_speed_mean = pd.Series(speed.to_numpy()[::-1], index=out.index).rolling(
        post_window, min_periods=1
    ).mean().to_numpy()[::-1]
    strict_reception = genuine_reception & local_distance_min.to_numpy(bool) & (
        post_speed_mean <= float(_value(rules, "bridging", "minimum_ball_speed_mps"))
    )
    out["genuine_reception_strict_anchor"] = strict_reception
    source.loc[strict_reception] = "genuine_reception_strict"
    source.loc[genuine_reception & ~strict_reception] = "genuine_reception"
    source.loc[reception & ~genuine_reception] = "confirmed_reception"

    # A transition anchor is an algorithmic switch confirmation: the newly
    # anchored team was approaching an open ball/receiver area in the preceding
    # window, and the previous anchor belonged to the other team.
    previous_anchor = anchor.where(anchor.notna()).ffill().shift()
    approaching_open = pd.Series(False, index=out.index)
    approaching_open |= (anchor == "H") & out["home_receiver_open"].rolling(window, min_periods=1).max().astype(bool).shift(1, fill_value=False)
    approaching_open |= (anchor == "A") & out["away_receiver_open"].rolling(window, min_periods=1).max().astype(bool).shift(1, fill_value=False)
    reachability_support = (
        ((anchor == "H") & (_numeric(out, "arrival_time_margin_s") > 0))
        | ((anchor == "A") & (_numeric(out, "arrival_time_margin_s") < 0))
    )
    transition = (
        anchor.notna()
        & previous_anchor.notna()
        & anchor.ne(previous_anchor)
        & approaching_open
        & reachability_support
    )
    source.loc[transition] = "confirmed_transition"

    # An occlusion anchor is valid only when the same team was anchored before
    # a short ball-coordinate gap and is anchored again afterward.
    ball_missing = _numeric(out, "ball_x").isna() | _numeric(out, "ball_y").isna()
    previous_team = anchor.where(anchor.notna()).ffill().shift()
    occlusion = anchor.notna() & anchor.eq(previous_team) & ball_missing.shift(1, fill_value=False)
    source.loc[occlusion] = "occlusion_continuity"

    # Restart hints are retained for inspection, but never override tracking.
    # The XML intervals are windows around an event, not exact touch frames.
    restart_confirmed = anchor.notna() & restart_context & restart_team_hint_series.eq(anchor)
    source.loc[restart_confirmed] = "restart_confirmed"

    out["anchor_source"] = source
    out["reception_anchor"] = reception
    out["genuine_reception_anchor"] = genuine_reception
    out["transition_anchor"] = transition
    out["occlusion_anchor"] = occlusion
    out["restart_confirmed_anchor"] = restart_confirmed
    out["n_anchor_evidence_types"] = source.notna().astype(int)
    return out


def add_boundary_anchor_features(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Expose restart-live, out-of-play, and dead-ball boundary anchors."""
    rules = load_rules() if rules is None else rules
    out = features.copy()
    ball_x = _numeric(out, "ball_x")
    ball_y = _numeric(out, "ball_y")
    speed = _numeric(out, "ball_speed_derived_mps")
    median_activity = _numeric(out, "player_activity_median_mps")
    p90_activity = _numeric(out, "player_activity_p90_mps")
    set_piece = out.get("smart_set_piece", pd.Series(False, index=out.index)).fillna(False).to_numpy(bool)
    restart_context = out.get("restart_context", pd.Series(False, index=out.index)).fillna(False).to_numpy(bool)
    restart_active = out.get("restart_ball_active", pd.Series(False, index=out.index)).fillna(False).to_numpy(bool)
    restart_type = out.get("restart_type", pd.Series(None, index=out.index)).to_numpy(object)
    anchor = out.get("anchor_team", pd.Series(None, index=out.index)).to_numpy(object)
    restart_hint = out.get("restart_team_hint", pd.Series(None, index=out.index)).to_numpy(object)

    out_of_bounds = (
        ball_x.notna() & ball_y.notna()
        & ((ball_x < 0.0) | (ball_x > 1.0) | (ball_y < 0.0) | (ball_y > 1.0))
    ).to_numpy(bool)
    low_motion = (
        (median_activity <= float(_value(rules, "dead_ball_evidence", "low_activity_median_mps")))
        & (p90_activity <= float(_value(rules, "dead_ball_evidence", "low_activity_p90_mps")))
        & (speed <= 1.0)
    ).to_numpy(bool)
    non_kickoff = ~np.isin(restart_type, ["Kick Off (Start)", "Kick Off (After Goal)"])
    dead_ball_boundary = set_piece & non_kickoff & low_motion
    restart_live_boundary = restart_context & restart_active
    hinted_control = (
        restart_live_boundary
        & np.isin(anchor, ["H", "A"])
        & np.isin(restart_hint, ["H", "A"])
        & (anchor == restart_hint)
    )
    fallback_control = restart_live_boundary & np.isin(anchor, ["H", "A"]) & ~np.isin(restart_hint, ["H", "A"])
    out["ball_out_of_play_anchor"] = out_of_bounds
    out["low_motion_dead_ball_anchor"] = dead_ball_boundary
    out["restart_live_boundary_anchor"] = restart_live_boundary
    out["restart_hinted_control_anchor"] = hinted_control
    out["restart_fallback_control_anchor"] = fallback_control
    out["restart_control_anchor_team"] = np.where(hinted_control | fallback_control, anchor, None)
    out["boundary_anchor_source"] = np.select(
        [hinted_control, fallback_control, dead_ball_boundary, out_of_bounds],
        ["RESTART_HINTED_CONTROL", "RESTART_TRACKING_CONTROL", "DEAD_BALL_LOW_MOTION", "BALL_OUT_OF_PLAY"],
        default="NONE",
    )
    return out


def add_reception_outcome_features(features: pd.DataFrame) -> pd.DataFrame:
    """Separate same-team receptions from possible turnovers.

    This is diagnostic evidence only.  It compares the last stable control
    anchor before a reception candidate with the team identified at the
    candidate receiver.  A genuine reception by the same team supports a long
    or through ball; an opponent reception is a turnover candidate.  Neither
    column directly overwrites the submission labels.
    """
    out = features.copy()
    anchor = out.get("anchor_team", pd.Series(None, index=out.index, dtype=object))
    previous_anchor = anchor.where(anchor.isin(["H", "A"])).ffill().shift()
    receiver = out.get("likely_receiver_team", pd.Series(None, index=out.index, dtype=object))
    projected = out.get("projected_receiver_team", pd.Series(None, index=out.index, dtype=object))
    receiver = receiver.where(receiver.isin(["H", "A"]), projected)
    genuine = out.get(
        "genuine_reception_anchor", pd.Series(False, index=out.index)
    ).fillna(False).astype(bool)
    receiver = receiver.where(receiver.isin(["H", "A"]))
    same_team = genuine & previous_anchor.isin(["H", "A"]) & receiver.eq(previous_anchor)
    opponent_team = genuine & previous_anchor.isin(["H", "A"]) & receiver.isin(["H", "A"]) & receiver.ne(previous_anchor)
    out["reception_previous_control_team"] = previous_anchor
    out["reception_receiver_team"] = receiver
    out["same_team_reception_candidate"] = same_team
    out["opponent_reception_candidate"] = opponent_team
    out["reception_outcome_candidate"] = np.select(
        [same_team, opponent_team],
        ["SAME_TEAM_CONTINUATION", "POSSIBLE_TURNOVER"],
        default="UNRESOLVED",
    )
    return out


def add_reception_geometry_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable geometry and persistence scores for reception analysis.

    These features combine independent observations but do not assign a
    possession label.  They are intended to distinguish a controlled arrival
    from a proximity-only event and to expose the evidence to later models.
    """
    out = features.copy()
    arrival = _numeric(out, "arrival_time_margin_s")
    projected_h = _numeric(out, "h_projected_path_distance_m")
    projected_a = _numeric(out, "a_projected_path_distance_m")
    projected_margin = projected_a - projected_h
    likely = out.get("likely_receiver_team", pd.Series(None, index=out.index, dtype=object))
    projected = out.get("projected_receiver_team", pd.Series(None, index=out.index, dtype=object))
    agreement = likely.isin(["H", "A"]) & projected.isin(["H", "A"]) & likely.eq(projected)
    out["receiver_arrival_margin_abs_s"] = arrival.abs()
    out["receiver_arrival_confidence"] = np.clip(arrival.abs().to_numpy(float) / 2.0, 0.0, 1.0)
    out["receiver_projected_path_margin_abs_m"] = projected_margin.abs()
    out["receiver_projected_path_confidence"] = np.clip(
        projected_margin.abs().to_numpy(float) / 10.0, 0.0, 1.0
    )
    out["receiver_evidence_agreement"] = agreement
    out["receiver_evidence_agreement_score"] = (
        0.5 * out["receiver_arrival_confidence"]
        + 0.3 * out["receiver_projected_path_confidence"]
        + 0.2 * agreement.astype(float)
    )
    pressure = _numeric(out, "candidate_pressure_m")
    density = _numeric(out, "candidate_opponent_density_5m")
    out["reception_pressure_score"] = (
        np.clip((pressure.to_numpy(float) - 1.5) / 4.0, 0.0, 1.0)
        * np.clip(1.0 - density.to_numpy(float) / 5.0, 0.0, 1.0)
    )
    anchor = out.get("anchor_team", pd.Series(None, index=out.index, dtype=object)).to_numpy(object)
    post_run = np.zeros(len(out), dtype=int)
    for index in range(len(out) - 1, -1, -1):
        if anchor[index] not in {"H", "A"}:
            post_run[index] = 0
        elif index + 1 < len(out) and anchor[index + 1] == anchor[index]:
            post_run[index] = post_run[index + 1] + 1
        else:
            post_run[index] = 1
    out["post_reception_anchor_run_frames"] = post_run
    out["controlled_reception_evidence_score"] = (
        0.35 * out["receiver_evidence_agreement_score"]
        + 0.25 * np.clip(post_run / 8.0, 0.0, 1.0)
        + 0.20 * out["reception_pressure_score"]
        + 0.20 * out.get("genuine_reception_anchor", pd.Series(False, index=out.index)).astype(float)
    )
    return out


def add_reception_signal_features(
    features: pd.DataFrame,
    *,
    rules: dict[str, Any] | None = None,
    near_ball_distance_m: float = 2.5,
    local_min_window_frames: int = 9,
    persistence_window_frames: int = 10,
) -> pd.DataFrame:
    """Add transparent reception signals from distance and motion geometry.

    These are evidence features, not labels.  A reception candidate requires
    an approach/local-minimum pattern and then measures whether the receiver
    remains near the ball.  A short touch is represented separately when the
    ball leaves the receiver's zone quickly.  Future persistence is intentional
    for the offline competition pipeline; a live version would use the
    trailing counterparts only.
    """
    if rules is not None:
        near_ball_distance_m = float(_value(rules, "reception_signal", "near_ball_distance_m"))
        local_min_window_frames = int(_value(rules, "reception_signal", "local_min_window_frames"))
        persistence_window_frames = int(_value(rules, "reception_signal", "persistence_window_frames"))
    out = features.copy()
    controlled_persistence_fraction = 0.50
    sustained_persistence_fraction = 0.70
    short_touch_persistence_fraction = 0.35
    minimum_signal_score = 0.55
    minimum_ball_deceleration_mps = 0.0
    if rules is not None:
        controlled_persistence_fraction = float(_value(rules, "reception_signal", "controlled_persistence_fraction"))
        sustained_persistence_fraction = float(_value(rules, "reception_signal", "sustained_persistence_fraction"))
        short_touch_persistence_fraction = float(_value(rules, "reception_signal", "short_touch_persistence_fraction"))
        minimum_signal_score = float(_value(rules, "reception_signal", "minimum_signal_score"))
        minimum_ball_deceleration_mps = float(_value(rules, "reception_signal", "minimum_ball_deceleration_mps"))
    window = max(3, int(local_min_window_frames))
    persistence = max(1, int(persistence_window_frames))
    distances: dict[str, pd.Series] = {
        team: _numeric(out, f"nearest_{team.lower()}_distance_m")
        for team in ("H", "A")
    }
    speeds = _numeric(out, "ball_speed_derived_mps")
    flight = out.get("recent_flight_evidence", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    likely = out.get("likely_receiver_team", pd.Series(None, index=out.index, dtype=object))
    projected = out.get("projected_receiver_team", pd.Series(None, index=out.index, dtype=object))

    team_scores: dict[str, pd.Series] = {}
    for team, distance in distances.items():
        closing = distance.shift(1) - distance
        local_min = distance <= distance.rolling(window, center=True, min_periods=3).min().add(0.10)
        near = distance <= float(near_ball_distance_m)
        future_near = pd.Series(
            near.to_numpy()[::-1], index=out.index
        ).rolling(persistence, min_periods=1).mean().to_numpy()[::-1]
        future_distance = pd.Series(
            distance.to_numpy()[::-1], index=out.index
        ).rolling(persistence, min_periods=1).mean().to_numpy()[::-1]
        rebound = future_distance - distance.to_numpy(float)
        out[f"{team.lower()}_receiver_closing_speed_mps"] = closing
        out[f"{team.lower()}_receiver_local_distance_min"] = local_min.fillna(False)
        out[f"{team.lower()}_receiver_future_near_fraction"] = future_near
        out[f"{team.lower()}_receiver_distance_rebound_m"] = rebound
        out[f"{team.lower()}_receiver_near_ball"] = near

        agreement = (likely == team) & (projected == team)
        approach_score = np.clip(closing.to_numpy(float) / 3.0, 0.0, 1.0)
        persistence_score = np.clip(future_near / 0.75, 0.0, 1.0)
        local_score = local_min.fillna(False).astype(float).to_numpy()
        agreement_score = agreement.astype(float).to_numpy()
        team_scores[team] = pd.Series(
            0.30 * approach_score
            + 0.20 * local_score
            + 0.30 * persistence_score
            + 0.20 * agreement_score,
            index=out.index,
        )

    home_distance = distances["H"]
    away_distance = distances["A"]
    margin = away_distance - home_distance
    control_team = pd.Series(
        np.select(
            [margin >= 0.5, margin <= -0.5],
            ["H", "A"],
            default=None,
        ),
        index=out.index,
        dtype=object,
    )
    out["reception_control_team_geometry"] = control_team
    out["reception_control_distance_margin_m"] = margin.abs()
    out["reception_ball_deceleration_mps"] = speeds.shift(1) - speeds
    out["reception_receiver_team_agreement"] = (
        control_team.notna() & likely.eq(control_team) & projected.eq(control_team)
    )
    best_score = pd.Series(
        np.where(control_team.eq("H"), team_scores["H"], team_scores["A"]),
        index=out.index,
    )
    near_control = np.where(
        control_team.eq("H"), home_distance <= near_ball_distance_m,
        away_distance <= near_ball_distance_m,
    )
    future_near_control = np.where(
        control_team.eq("H"),
        out["h_receiver_future_near_fraction"],
        out["a_receiver_future_near_fraction"],
    )
    out["controlled_reception_signal_score"] = best_score
    out["controlled_reception_signal"] = (
        flight
        & near_control
        & (future_near_control >= controlled_persistence_fraction)
        & (best_score >= minimum_signal_score)
    )
    out["short_touch_signal"] = (
        flight
        & near_control
        & (future_near_control < short_touch_persistence_fraction)
        & (out["reception_ball_deceleration_mps"] >= minimum_ball_deceleration_mps)
    )
    out["sustained_reception_signal"] = (
        out["controlled_reception_signal"]
        & (future_near_control >= sustained_persistence_fraction)
        & out["reception_receiver_team_agreement"]
    )
    return out


def add_first_time_pass_features(
    features: pd.DataFrame,
    *,
    lookback_frames: int = 25,
    lookahead_frames: int = 50,
) -> pd.DataFrame:
    """Add offline evidence for first-time passes versus turnovers.

    A first-time pass is allowed to leave the receiver immediately.  The
    receiver therefore needs approach evidence before the candidate and a
    same-team controlled anchor later; persistence at the original receiver
    is not required.  An opponent controlled anchor in the same horizon is
    recorded separately as turnover evidence.
    """
    out = features.copy()
    n_frames = len(out)
    anchor = out.get("anchor_team", pd.Series(None, index=out.index, dtype=object))
    previous_team = anchor.where(anchor.isin(["H", "A"])).ffill().shift()
    receiver = out.get("likely_receiver_team", pd.Series(None, index=out.index, dtype=object))
    projected = out.get("projected_receiver_team", pd.Series(None, index=out.index, dtype=object))
    receiver = receiver.where(receiver.isin(["H", "A"]), projected)
    receiver_distance = np.where(
        receiver.eq("H"), _numeric(out, "nearest_h_distance_m"),
        np.where(receiver.eq("A"), _numeric(out, "nearest_a_distance_m"), np.nan),
    )
    receiver_closing = np.where(
        receiver.eq("H"), _numeric(out, "nearest_h_closing_rate_mps"),
        np.where(receiver.eq("A"), _numeric(out, "nearest_a_closing_rate_mps"), np.nan),
    )
    receiver_distance_series = pd.Series(receiver_distance, index=out.index, dtype=float)
    approach_amount = receiver_distance_series.shift(int(lookback_frames)) - receiver_distance_series
    approach = (
        (approach_amount >= 1.0)
        & (pd.Series(receiver_closing, index=out.index).fillna(0.0) >= 0.5)
    )

    anchor_array = anchor.to_numpy(object)
    future_same = np.zeros(n_frames, dtype=bool)
    future_opponent = np.zeros(n_frames, dtype=bool)
    previous_array = previous_team.to_numpy(object)
    for index in range(n_frames):
        end = min(n_frames, index + 1 + int(lookahead_frames))
        window = anchor_array[index + 1:end]
        team = previous_array[index]
        if team not in {"H", "A"}:
            continue
        future_same[index] = np.any(window == team)
        future_opponent[index] = np.any(window == ("A" if team == "H" else "H"))

    future_receiver = np.zeros(n_frames, dtype=bool)
    anchor_confidence = pd.to_numeric(
        out.get("anchor_confidence_score", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    future_receiver_confidence = np.zeros(n_frames, dtype=float)
    future_receiver_near_ball_fraction = np.zeros(n_frames, dtype=float)
    receiver_array = receiver.to_numpy(object)
    for index in range(n_frames):
        end = min(n_frames, index + 1 + int(lookahead_frames))
        team = receiver_array[index]
        if team in {"H", "A"}:
            future_mask = anchor_array[index + 1:end] == team
            future_receiver[index] = np.any(future_mask)
            if future_receiver[index]:
                future_receiver_confidence[index] = float(
                    np.nanmax(anchor_confidence[index + 1:end][future_mask])
                )
            distance_column = "nearest_h_distance_m" if team == "H" else "nearest_a_distance_m"
            future_distance = _numeric(out, distance_column).to_numpy(float)[index + 1:end]
            if len(future_distance):
                valid = np.isfinite(future_distance)
                if valid.any():
                    future_receiver_near_ball_fraction[index] = float(
                        np.mean(future_distance[valid] <= 3.0)
                    )

    genuine = out.get(
        "genuine_reception_anchor", pd.Series(False, index=out.index)
    ).fillna(False).astype(bool)
    fast_departure = _numeric(out, "ball_speed_derived_mps").shift(-1).fillna(0.0) >= 3.0
    ball_speed = _numeric(out, "ball_speed_derived_mps")
    ball_speed_drop = _numeric(out, "ball_speed_drop_mps")
    receiver_close = receiver_distance_series <= 3.0
    contact_evidence = (
        out.get("recent_flight_evidence", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        & receiver.isin(["H", "A"])
        & receiver_close
        & (
            approach
            | (ball_speed_drop >= 1.0)
            | (ball_speed <= 3.0)
        )
    )
    soft_same_team = contact_evidence & previous_team.eq(receiver) & future_receiver
    soft_opponent = contact_evidence & previous_team.isin(["H", "A"]) & previous_team.ne(receiver) & future_receiver
    same_team_continuation = genuine & approach & future_same & ~future_opponent
    possible_first_time = genuine & approach & fast_departure & future_same
    possible_turnover = genuine & approach & future_opponent & ~future_same
    out["reception_lookback_approach_m"] = approach_amount
    out["reception_lookback_approach"] = approach
    out["future_same_team_control_anchor"] = future_same
    out["future_opponent_control_anchor"] = future_opponent
    out["future_receiver_control_anchor"] = future_receiver
    out["future_receiver_anchor_confidence"] = future_receiver_confidence
    out["future_receiver_near_ball_fraction"] = future_receiver_near_ball_fraction
    out["soft_reception_event_candidate"] = contact_evidence
    out["soft_same_team_continuation_candidate"] = soft_same_team
    out["soft_opponent_controlled_reception_candidate"] = soft_opponent
    out["first_time_pass_candidate"] = possible_first_time
    out["first_time_pass_after_long_ball_risk"] = (
        possible_first_time
        & out.get("post_long_ball_direction_change_risk", pd.Series(False, index=out.index))
    )
    out["same_team_continuation_candidate"] = same_team_continuation
    out["opponent_controlled_reception_candidate"] = possible_turnover
    return out


def add_reception_episode_features(
    features: pd.DataFrame,
    *,
    maximum_gap_frames: int = 3,
    minimum_support_frames: int = 3,
    minimum_stability: float = 0.75,
) -> pd.DataFrame:
    """Aggregate soft reception candidates into stable local episodes."""
    out = features.copy()
    n_frames = len(out)
    event = out.get(
        "soft_reception_event_candidate", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    same = out.get(
        "soft_same_team_continuation_candidate", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    opponent = out.get(
        "soft_opponent_controlled_reception_candidate", pd.Series(False, index=out.index)
    ).fillna(False).to_numpy(bool)
    episode_id = np.full(n_frames, -1, dtype=int)
    episode_length = np.zeros(n_frames, dtype=int)
    episode_stability = np.zeros(n_frames, dtype=float)
    episode_receiver_confidence = np.zeros(n_frames, dtype=float)
    episode_receiver_near_ball_fraction = np.zeros(n_frames, dtype=float)
    episode_outcome = np.full(n_frames, "UNRESOLVED", dtype=object)
    receiver_confidence = pd.to_numeric(
        out.get("future_receiver_anchor_confidence", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    receiver_near_fraction = pd.to_numeric(
        out.get("future_receiver_near_ball_fraction", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)

    positions = np.flatnonzero(event)
    if len(positions):
        groups: list[tuple[int, int]] = []
        start = int(positions[0])
        previous = start
        for position in positions[1:]:
            position = int(position)
            if position - previous > int(maximum_gap_frames):
                groups.append((start, previous + 1))
                start = position
            previous = position
        groups.append((start, previous + 1))

        for identifier, (start, end) in enumerate(groups):
            members = event[start:end]
            same_count = int(np.sum(same[start:end] & members))
            opponent_count = int(np.sum(opponent[start:end] & members))
            support = max(same_count, opponent_count)
            stability = support / max(1, int(np.sum(members)))
            confidence = float(np.nanmax(receiver_confidence[start:end][members])) if np.any(members) else 0.0
            near_fraction = float(np.nanmax(receiver_near_fraction[start:end][members])) if np.any(members) else 0.0
            if support >= int(minimum_support_frames) and stability >= float(minimum_stability):
                outcome = "SAME_TEAM_CONTINUATION" if same_count >= opponent_count else "POSSIBLE_TURNOVER"
            else:
                outcome = "UNRESOLVED"
            episode_id[start:end][members] = identifier
            episode_length[start:end][members] = int(np.sum(members))
            episode_stability[start:end][members] = stability
            episode_receiver_confidence[start:end][members] = confidence
            episode_receiver_near_ball_fraction[start:end][members] = near_fraction
            episode_outcome[start:end][members] = outcome

    out["reception_episode_id"] = episode_id
    out["reception_episode_length_frames"] = episode_length
    out["reception_episode_stability"] = episode_stability
    out["reception_episode_receiver_confidence"] = episode_receiver_confidence
    out["reception_episode_receiver_near_ball_fraction"] = episode_receiver_near_ball_fraction
    out["reception_episode_outcome"] = episode_outcome
    out["stable_reception_episode"] = episode_outcome != "UNRESOLVED"
    return out


def bridge_same_team_anchor_gaps(
    labels: np.ndarray,
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Bridge gaps between same-team anchors, with short/long gap limits.

    This function only fills ``None``/``X`` gaps. It never overwrites a known
    H/A/D label, and it never bridges across Smart Tagging set-piece frames.
    """
    rules = load_rules() if rules is None else rules
    out = labels.astype(object, copy=True)
    anchors = features["anchor_team"].to_numpy(dtype=object)
    set_piece = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)
    short_limit = int(_value(rules, "bridging", "short_pass_max_frames"))
    long_limit = int(_value(rules, "bridging", "long_ball_max_frames"))
    stats = {"short_bridges": 0, "long_bridges": 0, "rejected_bridges": 0, "frames_filled": 0}

    anchor_positions = np.flatnonzero(pd.notna(anchors))
    for left, right in zip(anchor_positions[:-1], anchor_positions[1:]):
        team = anchors[left]
        if team != anchors[right] or right - left <= 1:
            continue
        gap = right - left - 1
        if set_piece[left : right + 1].any():
            stats["rejected_bridges"] += 1
            continue
        if gap <= short_limit:
            kind = "short_bridges"
        elif gap <= long_limit:
            # A long bridge requires a coherent moving-ball episode. We use
            # derived speed as evidence, but leave uncertain episodes alone.
            speed = _numeric(features, "ball_speed_derived_mps").iloc[left + 1 : right]
            minimum_speed = float(_value(rules, "bridging", "minimum_ball_speed_mps"))
            if speed.dropna().empty or speed.max() < minimum_speed:
                stats["rejected_bridges"] += 1
                continue
            kind = "long_bridges"
        else:
            stats["rejected_bridges"] += 1
            continue
        gap_positions = np.arange(left + 1, right)
        missing = pd.isna(out[gap_positions]) | (out[gap_positions] == "X")
        out[gap_positions[missing]] = team
        stats[kind] += 1
        stats["frames_filled"] += int(missing.sum())
    return out, stats


def apply_confirmed_transition_switches(
    labels: np.ndarray,
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Fill a confirmed cross-team gap up to the receiving team's anchor.

    For a positive receiver-approach transition, the previous team owns the
    interval until the new team is actually anchored. This does not claim that
    every flight frame is a touch; it encodes the team-level possession
    convention used by the submission labels.
    """
    rules = load_rules() if rules is None else rules
    out = labels.astype(object, copy=True)
    stats = {"confirmed_switches": 0, "transition_frames_filled": 0}
    if not bool(_value(rules, "bridging", "use_confirmed_transition_switch")):
        return out, stats

    anchor_positions = np.flatnonzero(features["anchor_team"].notna().to_numpy())
    transition_flags = features["transition_anchor"].to_numpy(bool)
    for left, right in zip(anchor_positions[:-1], anchor_positions[1:]):
        left_team = features.iloc[left]["anchor_team"]
        right_team = features.iloc[right]["anchor_team"]
        if left_team == right_team or not transition_flags[right]:
            continue
        gap_positions = np.arange(left + 1, right)
        missing = pd.isna(out[gap_positions]) | (out[gap_positions] == "X")
        out[gap_positions[missing]] = left_team
        stats["confirmed_switches"] += 1
        stats["transition_frames_filled"] += int(missing.sum())
    return out, stats


def build_scenario_episodes(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Collapse frame evidence into contiguous, inspectable scenario episodes."""
    rules = load_rules() if rules is None else rules
    n_frames = len(features)
    scenario = np.full(n_frames, "UNKNOWN", dtype=object)
    anchor_team = features["anchor_team"].to_numpy(dtype=object)
    restart = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)
    transition = features.get("transition_anchor", pd.Series(False, index=features.index)).to_numpy(bool)
    long_ball = features.get("long_ball_candidate", pd.Series(False, index=features.index)).to_numpy(bool)
    close_distance = _numeric(features, "closest_distance_m").to_numpy(float)
    control_radius = float(_value(rules, "anchors", "stable_control_max_distance_m"))

    scenario[long_ball & (close_distance > control_radius)] = "BALL_IN_FLIGHT"
    scenario[anchor_team == "H"] = "H_CONTROLLED"
    scenario[anchor_team == "A"] = "A_CONTROLLED"
    scenario[transition] = "CONFIRMED_TRANSITION"
    scenario[restart] = "RESTART"

    candidate = pd.Series(anchor_team, index=features.index, dtype=object).ffill().to_numpy(dtype=object)
    episode_start = np.r_[True, scenario[1:] != scenario[:-1]]
    starts = np.flatnonzero(episode_start)
    ends = np.r_[starts[1:] - 1, n_frames - 1]
    rows = []
    for episode_id, (start, end) in enumerate(zip(starts, ends)):
        window = features.iloc[start : end + 1]
        rows.append({
            "episode_id": episode_id,
            "start_frame": int(start),
            "end_frame": int(end),
            "duration_frames": int(end - start + 1),
            "duration_s": float((end - start + 1) / _value(rules, "timeline", "fps")),
            "scenario": scenario[start],
            "candidate_team": candidate[start] if candidate[start] in {"H", "A"} else None,
            "max_ball_speed_mps": float(_numeric(window, "ball_speed_derived_mps").max()),
            "min_home_arrival_s": float(_numeric(window, "h_fastest_arrival_s").min()),
            "min_away_arrival_s": float(_numeric(window, "a_fastest_arrival_s").min()),
            "mean_pressure_m": float(_numeric(window, "candidate_pressure_m").mean()),
            "graded_frames": int(window.get("graded", pd.Series(False, index=window.index)).sum()),
        })
    return pd.DataFrame(rows)
