"""Generic, label-free audit of when broadcast evidence is trustworthy.

This module does not replace tactical labels. Tactical and broadcast recordings
have different frame counts and no shared frame key. It identifies intervals on
the broadcast timeline where the recording itself provides enough evidence to
be considered for a later alignment step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from games import GAME, GAMES, require
from starter.load_homography import load as load_homography
from possession_lab.temporal_rules import load_rules


@dataclass(frozen=True)
class BroadcastAuditConfig:
    """Conservative, interpretable thresholds for an audit block."""

    fps: float = 25.0
    block_frames: int = 25
    minimum_players: int = 12
    maximum_reprojection_error: float = 0.75
    minimum_geometry_fraction: float = 0.80


def _raw_frame_signals(path: str | Path) -> Iterator[tuple[int, int, bool, float, float]]:
    """Yield frame-level signals incrementally from one raw ATD export."""

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            frame_text, payload = line.rstrip("\n").split(":", 1)
            player_text, ball_text = payload.rsplit(":", 1)
            visible_players = 0
            for position in player_text.split(";"):
                x, y, *_ = position.split(",")
                if x != "NaN" and y != "NaN":
                    visible_players += 1
            ball = ball_text.split(",")
            ball_visible = len(ball) >= 2 and ball[0] != "NaN" and ball[1] != "NaN"
            ball_x = float(ball[0]) if ball_visible else np.nan
            ball_y = float(ball[1]) if ball_visible else np.nan
            yield int(frame_text) - 1, visible_players, ball_visible, ball_x, ball_y


def _raw_block_geometry(path: str | Path, block_frames: int) -> Iterator[dict[str, float]]:
    """Yield compact, team-agnostic pitch-geometry signatures per block."""

    block: list[tuple[int, list[tuple[float, float]], tuple[float, float] | None]] = []

    def flush(items: list[tuple[int, list[tuple[float, float]], tuple[float, float] | None]]) -> dict[str, float] | None:
        if not items:
            return None
        player_points = [point for _, points, _ in items for point in points]
        ball_points = [ball for _, _, ball in items if ball is not None]
        histogram = np.zeros(12, dtype=float)
        if player_points:
            points = np.asarray(player_points, dtype=float)
            histogram, _, _ = np.histogram2d(
                points[:, 0], points[:, 1], bins=(4, 3), range=((0.0, 1.0), (0.0, 1.0))
            )
            histogram = histogram.ravel() / max(float(histogram.sum()), 1.0)
            centroid_x, centroid_y = points.mean(axis=0)
        else:
            centroid_x = centroid_y = np.nan
        if ball_points:
            ball = np.asarray(ball_points, dtype=float)
            ball_x, ball_y = ball.mean(axis=0)
        else:
            ball_x = ball_y = np.nan
        row = {
            "frame_start": float(items[0][0]),
            "frame_end": float(items[-1][0]),
            "player_count_mean": float(np.mean([len(points) for _, points, _ in items])),
            "player_centroid_x": float(centroid_x),
            "player_centroid_y": float(centroid_y),
            "ball_mean_x": float(ball_x),
            "ball_mean_y": float(ball_y),
            "ball_visible_fraction": float(len(ball_points) / len(items)),
        }
        row.update({f"player_hist_{index}": float(value) for index, value in enumerate(histogram)})
        return row

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            frame_text, payload = line.rstrip("\n").split(":", 1)
            player_text, ball_text = payload.rsplit(":", 1)
            points = []
            for position in player_text.split(";"):
                x, y, *_ = position.split(",")
                if x != "NaN" and y != "NaN":
                    points.append((float(x), float(y)))
            ball = ball_text.split(",")
            ball_point = (float(ball[0]), float(ball[1])) if ball[0] != "NaN" and ball[1] != "NaN" else None
            block.append((int(frame_text) - 1, points, ball_point))
            if len(block) == block_frames:
                row = flush(block)
                if row is not None:
                    yield row
                block = []
    row = flush(block)
    if row is not None:
        yield row


def build_source_signatures(source: str, game: str = GAME, block_frames: int | None = None) -> pd.DataFrame:
    """Build compact signatures used only for local tactical/broadcast matching."""

    fps = float(GAMES[game]["fps"])
    block_frames = round(fps) if block_frames is None else block_frames
    raw_path = require(["atd_raw"], game, None, source)[0]
    signatures = pd.DataFrame(_raw_block_geometry(raw_path, block_frames))
    quality = build_broadcast_audit(
        source=source,
        game=game,
        config=BroadcastAuditConfig(fps=fps, block_frames=block_frames),
    )
    return signatures.merge(quality, on=["frame_start", "frame_end"], how="left", suffixes=("", "_quality"))


def find_local_alignment_candidates(
    tactical_signatures: pd.DataFrame,
    broadcast_signatures: pd.DataFrame,
    tactical_needs: pd.DataFrame,
    top_k: int = 3,
    expected_players: float = 22.0,
) -> pd.DataFrame:
    """Rank broadcast blocks against only the tactical blocks that need help.

    The score is based on player occupancy, player centroid, player count, and
    ball position when available. It is an alignment diagnostic, not a label
    transfer rule. The returned confidence is intentionally conservative.
    """

    histogram_columns = [f"player_hist_{index}" for index in range(12)]
    broadcast = broadcast_signatures[broadcast_signatures["live_evidence"]].reset_index(drop=True)
    rows: list[dict[str, float | int | bool]] = []
    if broadcast.empty:
        return pd.DataFrame()
    for _, need in tactical_needs.iterrows():
        target = tactical_signatures[
            (tactical_signatures["frame_end"] >= need["frame_start"])
            & (tactical_signatures["frame_start"] <= need["frame_end"])
        ]
        for _, source_row in target.iterrows():
            costs = np.zeros(len(broadcast), dtype=float)
            target_hist = source_row[histogram_columns].to_numpy(dtype=float)
            source_hist = broadcast[histogram_columns].to_numpy(dtype=float)
            costs += 0.55 * np.abs(source_hist - target_hist).sum(axis=1)
            costs += 0.15 * np.hypot(
                broadcast["player_centroid_x"] - source_row["player_centroid_x"],
                broadcast["player_centroid_y"] - source_row["player_centroid_y"],
            ).fillna(1.0).to_numpy()
            costs += 0.15 * (broadcast["player_count_mean"] - source_row["player_count_mean"]).abs().to_numpy() / expected_players
            if np.isfinite(source_row["ball_mean_x"]):
                costs += 0.15 * np.hypot(
                    broadcast["ball_mean_x"] - source_row["ball_mean_x"],
                    broadcast["ball_mean_y"] - source_row["ball_mean_y"],
                ).fillna(1.0).to_numpy()
            order = np.argsort(costs)[:top_k]
            best = float(costs[order[0]])
            second = float(costs[order[1]]) if len(order) > 1 else best + 1.0
            for rank, index in enumerate(order, start=1):
                cost = float(costs[index])
                rows.append(
                    {
                        "need_frame_start": int(need["frame_start"]),
                        "need_frame_end": int(need["frame_end"]),
                        "tactical_frame": int(source_row["frame_start"]),
                        "broadcast_frame": int(broadcast.iloc[index]["frame_start"]),
                        "rank": rank,
                        "geometry_cost": cost,
                        "cost_margin": second - best,
                        "alignment_confidence": float(np.exp(-4.0 * cost) * min(1.0, max(0.0, (second - cost) / 0.25))),
                        "candidate_accepted": False,
                        "reason": str(need["reason"]),
                    }
                )
    matches = pd.DataFrame(rows)
    if matches.empty:
        return matches
    matches["temporal_consistency"] = False

    # A local match must also be temporally coherent. Independent nearest
    # neighbours are allowed to jump backwards or repeatedly choose one frame.
    # Only rank-1 candidates inside a need window can become accepted.
    for _, group in matches[matches["rank"] == 1].groupby("need_frame_start"):
        group = group.sort_values("tactical_frame")
        if len(group) < 3:
            continue
        tactical = group["tactical_frame"].to_numpy(dtype=float)
        broadcast = group["broadcast_frame"].to_numpy(dtype=float)
        t_delta = np.diff(tactical)
        b_delta = np.diff(broadcast)
        valid_steps = t_delta > 0
        if not valid_steps.any():
            continue
        slopes = b_delta[valid_steps] / t_delta[valid_steps]
        monotonic = bool(np.mean(b_delta[valid_steps] >= 0) >= 0.90)
        slope_ok = bool(np.median(slopes) >= 0.50 and np.median(slopes) <= 1.50)
        coefficients = np.polyfit(tactical, broadcast, 1)
        residual = broadcast - np.polyval(coefficients, tactical)
        residual_ok = bool(np.median(np.abs(residual)) <= 100.0)
        if monotonic and slope_ok and residual_ok:
            accepted_index = group.index
            accepted_index = accepted_index[matches.loc[accepted_index, "geometry_cost"] <= 0.45]
            accepted_index = accepted_index[matches.loc[accepted_index, "cost_margin"] >= 0.05]
            matches.loc[accepted_index, "candidate_accepted"] = True
            matches.loc[accepted_index, "temporal_consistency"] = True
    return matches


def find_local_sequence_candidates(
    tactical_signatures: pd.DataFrame,
    broadcast_signatures: pd.DataFrame,
    tactical_needs: pd.DataFrame,
    sequence_blocks: int = 5,
    top_k: int = 3,
    expected_players: float = 22.0,
) -> pd.DataFrame:
    """Match short geometry sequences only around tactical need windows."""

    histogram_columns = [f"player_hist_{index}" for index in range(12)]
    feature_columns = histogram_columns + ["player_centroid_x", "player_centroid_y", "player_count_mean"]
    broadcast = broadcast_signatures.reset_index(drop=True)
    live = broadcast["live_evidence"].fillna(False).to_numpy(bool)
    candidate_starts = [
        index for index in range(0, len(broadcast) - sequence_blocks + 1)
        if live[index:index + sequence_blocks].all()
    ]
    broadcast_matrix = broadcast[feature_columns].to_numpy(float).copy()
    broadcast_matrix[:, -1] /= expected_players
    broadcast_matrix[:, -3:-1] = np.nan_to_num(broadcast_matrix[:, -3:-1], nan=0.5)
    candidate_matrix = np.stack(
        [broadcast_matrix[start:start + sequence_blocks] for start in candidate_starts],
        axis=0,
    )
    rows: list[dict[str, float | int | bool]] = []
    for _, need in tactical_needs.iterrows():
        target_index = int(np.argmin(np.abs(tactical_signatures["frame_start"].to_numpy() - need["frame_start"])))
        target = tactical_signatures.iloc[target_index:target_index + sequence_blocks]
        if len(target) < sequence_blocks:
            continue
        target_matrix = target[feature_columns].to_numpy(float).copy()
        target_matrix[:, -1] /= expected_players
        target_matrix[:, -3:-1] = np.nan_to_num(target_matrix[:, -3:-1], nan=0.5)
        costs = np.mean(np.abs(candidate_matrix - target_matrix[None, :, :]), axis=(1, 2))
        order = np.argsort(costs)[:top_k]
        for rank, index in enumerate(order, start=1):
            cost = float(costs[index])
            second = float(costs[np.argsort(costs)[1]]) if len(costs) > 1 else cost + 1.0
            rows.append(
                {
                    "need_frame_start": int(need["frame_start"]),
                    "need_frame_end": int(need["frame_end"]),
                    "tactical_sequence_start": int(target["frame_start"].iloc[0]),
                    "broadcast_sequence_start": int(broadcast["frame_start"].iloc[candidate_starts[index]]),
                    "sequence_duration_s": sequence_blocks,
                    "rank": rank,
                    "sequence_geometry_cost": cost,
                    "cost_margin": second - cost,
                    "sequence_confidence": float(np.exp(-8.0 * cost) * min(1.0, max(0.0, (second - cost) / 0.10))),
                    "candidate_accepted": bool(rank == 1 and cost <= 0.08 and second - cost >= 0.01),
                    "reason": str(need["reason"]),
                }
            )
    return pd.DataFrame(rows)


def build_broadcast_audit(
    source: str = "broadcast",
    game: str = GAME,
    config: BroadcastAuditConfig | None = None,
) -> pd.DataFrame:
    """Build one evidence row per time block on the selected source timeline."""

    if config is None:
        rules = load_rules()
        broadcast_rules = rules["broadcast"]
        fps = float(GAMES[game]["fps"])
        config = BroadcastAuditConfig(
            fps=fps,
            block_frames=int(broadcast_rules["audit_block_frames"]),
            minimum_players=int(broadcast_rules["minimum_visible_players"]),
            maximum_reprojection_error=float(broadcast_rules["maximum_reprojection_error"]),
            minimum_geometry_fraction=float(broadcast_rules["minimum_geometry_fraction"]),
        )
    raw_path = require(["atd_raw"], game, None, source)[0]
    _, reprojection_error, homography_valid = load_homography(game=game, source=source)
    rows: list[dict[str, float | int | bool]] = []
    block: list[tuple[int, int, bool, float, float]] = []

    def flush(items: list[tuple[int, int, bool, float, float]]) -> None:
        if not items:
            return
        frames = np.asarray([item[0] for item in items], dtype=int)
        visible_players = np.asarray([item[1] for item in items], dtype=float)
        ball_visible = np.asarray([item[2] for item in items], dtype=bool)
        geometry_ok = homography_valid[frames] & (
            reprojection_error[frames] <= config.maximum_reprojection_error
        )
        geometry_fraction = float(np.mean(geometry_ok))
        player_fraction = float(np.mean(visible_players >= config.minimum_players))
        ball_fraction = float(np.mean(ball_visible))
        live_evidence = geometry_fraction >= config.minimum_geometry_fraction and player_fraction >= 0.80
        rows.append(
            {
                "frame_start": int(frames[0]),
                "frame_end": int(frames[-1]),
                "time_start_s": float(frames[0] / config.fps),
                "time_end_s": float(frames[-1] / config.fps),
                "visible_players_median": float(np.median(visible_players)),
                "visible_players_p10": float(np.percentile(visible_players, 10)),
                "ball_visible_fraction": ball_fraction,
                "homography_valid_fraction": float(np.mean(homography_valid[frames])),
                "geometry_ok_fraction": geometry_fraction,
                "player_track_ok_fraction": player_fraction,
                "reprojection_error_median": float(np.median(reprojection_error[frames])),
                "live_evidence": bool(live_evidence),
                "ball_evidence": bool(live_evidence and ball_fraction >= 0.50),
            }
        )

    for signal in _raw_frame_signals(raw_path):
        block.append(signal)
        if len(block) == config.block_frames:
            flush(block)
            block = []
    flush(block)
    return pd.DataFrame(rows)


def summarise_candidate_windows(audit: pd.DataFrame, fps: float = 25.0) -> pd.DataFrame:
    """Collapse adjacent live-evidence blocks into candidate broadcast windows."""

    columns = ["frame_start", "frame_end", "duration_s", "blocks", "mean_geometry_ok", "mean_ball_visible"]
    if audit.empty:
        return pd.DataFrame(columns=columns)
    candidates = audit[audit["live_evidence"]].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    block_size = int(audit["frame_end"].iloc[0] - audit["frame_start"].iloc[0] + 1)
    candidates["window_id"] = candidates["frame_start"].diff().fillna(block_size).gt(block_size).cumsum()
    return (
        candidates.groupby("window_id", as_index=False)
        .agg(
            frame_start=("frame_start", "min"),
            frame_end=("frame_end", "max"),
            blocks=("frame_start", "size"),
            mean_geometry_ok=("geometry_ok_fraction", "mean"),
            mean_ball_visible=("ball_visible_fraction", "mean"),
        )
        .assign(duration_s=lambda table: (table["frame_end"] - table["frame_start"] + 1) / fps)
        [["frame_start", "frame_end", "duration_s", "blocks", "mean_geometry_ok", "mean_ball_visible"]]
    )


def identify_tactical_need_windows(
    tactical_audit: pd.DataFrame,
    ball_missing_fraction: float = 0.50,
    minimum_players: int = 12,
    graded_ranges: tuple[tuple[int, int], ...] | None = None,
    fps: float = 25.0,
) -> pd.DataFrame:
    """Find tactical blocks where broadcast evidence may be worth searching.

    This is deliberately a *need detector*, not a possession labeler. A block is
    interesting when tactical ball visibility is poor or the tactical player
    tracks are degraded. Blocks with normal tactical evidence are never sent to
    the broadcast alignment search.
    """

    if tactical_audit.empty:
        return pd.DataFrame(columns=["frame_start", "frame_end", "duration_s", "reason"])
    if graded_ranges is None:
        in_graded_play = np.ones(len(tactical_audit), dtype=bool)
    else:
        in_graded_play = np.zeros(len(tactical_audit), dtype=bool)
        for start, end in graded_ranges:
            in_graded_play |= (
                (tactical_audit["frame_end"].to_numpy() >= start)
                & (tactical_audit["frame_start"].to_numpy() <= end)
            )
    needs = tactical_audit[in_graded_play & (
        (1.0 - tactical_audit["ball_visible_fraction"] >= ball_missing_fraction)
        | (tactical_audit["visible_players_p10"] < minimum_players)
    )].copy()
    if needs.empty:
        return pd.DataFrame(columns=["frame_start", "frame_end", "duration_s", "reason"])
    reasons = np.where(
        1.0 - needs["ball_visible_fraction"] >= ball_missing_fraction,
        "tactical_ball_missing",
        "tactical_tracking_degraded",
    )
    needs["reason"] = reasons
    clipped_rows = []
    ranges_to_clip = graded_ranges if graded_ranges is not None else ((-np.inf, np.inf),)
    for _, row in needs.iterrows():
        for start, end in ranges_to_clip:
            clipped_start = max(int(row["frame_start"]), start)
            clipped_end = min(int(row["frame_end"]), end)
            if clipped_start <= clipped_end:
                item = row.copy()
                item["frame_start"] = clipped_start
                item["frame_end"] = clipped_end
                item["duration_s"] = (clipped_end - clipped_start + 1) / fps
                clipped_rows.append(item)
    needs = pd.DataFrame(clipped_rows)
    block_size = int(tactical_audit["frame_end"].iloc[0] - tactical_audit["frame_start"].iloc[0] + 1)
    needs["window_id"] = needs["frame_start"].diff().fillna(block_size).gt(block_size).cumsum()
    return (
        needs.groupby(["window_id", "reason"], as_index=False)
        .agg(frame_start=("frame_start", "min"), frame_end=("frame_end", "max"))
        .assign(duration_s=lambda table: (table["frame_end"] - table["frame_start"] + 1) / 25.0)
        [["frame_start", "frame_end", "duration_s", "reason"]]
    )

