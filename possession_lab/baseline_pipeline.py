"""Staged, inspectable baseline for the MTAH possession task.

Pipeline order:
    supplied signals -> remove always-null columns -> basic distances
    -> spatial features -> transparent rule-based labels -> submission CSV

The public five-minute labels are a sanity check only; they must not be used
to tune thresholds or train on the hidden match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys
import xml.etree.ElementTree as ET
from typing import Iterable

import numpy as np
import pandas as pd

# Allow both ``python possession_lab/baseline_pipeline.py`` and notebook imports
# from the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from games import GAME, GAMES, asset_paths, half_video_blocks, require, source_frames
from starter.load_smart_tagging import load as load_smart_tagging
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES


LABELS = ("H", "A", "D", "X")


@dataclass(frozen=True)
class BaselineConfig:
    """All baseline assumptions and thresholds in one place."""

    pitch_length_m: float = 105.3
    pitch_width_m: float = 68.0
    density_radius_m: float = 5.0
    zone_x_edges: tuple[float, ...] = (0.0, 1 / 3, 2 / 3, 1.0)
    zone_y_edges: tuple[float, ...] = (0.0, 1 / 3, 2 / 3, 1.0)
    set_piece_as_dead_ball: bool = True
    missing_ball_policy: str = "prefer_previous"

    @property
    def pitch(self) -> tuple[float, float]:
        return self.pitch_length_m, self.pitch_width_m


def slot_team_map(meta_path: str | os.PathLike[str]) -> dict[int, str]:
    """Map each temporary ATD slot to H/A using the supplied metadata XML."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"expected two teams in {meta_path}, found {len(teams)}")
    home_id = teams[0].get("id")
    return {
        slot: ("H" if player.get("teamId") == home_id else "A")
        for slot, player in enumerate(root.iter("Player"), start=1)
    }


def graded_mask(n_frames: int, game: str = GAME) -> np.ndarray:
    """Return the official graded-frame mask for the tactical timeline."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first : last + 1] = True
    return mask


def add_match_time_columns(frame_table: pd.DataFrame, game: str = GAME) -> pd.DataFrame:
    """Add exact video time and kickoff-anchored match-clock columns.

    ``video_time_s`` is the tactical timeline time at 25 fps. ``half_elapsed_s``
    is signed relative to the actual kickoff frame: negative values identify the
    graded pre-second-half kickoff interval. ``match_clock_s`` uses 0:00 at the
    first-half kickoff and 45:00 at the second-half kickoff. This is an analytical
    match clock, not a claim about the referee's stoppage-time display.
    """
    out = frame_table.copy()
    frame = pd.to_numeric(out["frame"], errors="coerce").to_numpy(float)
    fps = GAMES[game]["fps"]
    out["video_time_s"] = frame / fps
    # The second graded block begins before the recorded second-half kickoff.
    # Treat that interval as 2H context, then expose its negative elapsed time
    # rather than silently calling it first-half play.
    second_block_start = half_video_blocks(game)[1][1]
    out["match_half"] = np.where(frame < second_block_start, "1H", "2H")
    second_kickoff = KICKOFF_VIDEO_FRAMES["2nd Half"]
    first_kickoff = KICKOFF_VIDEO_FRAMES["1st Half"]
    half_elapsed = np.where(
        out["match_half"].to_numpy() == "1H",
        (frame - first_kickoff) / fps,
        (frame - second_kickoff) / fps,
    )
    out["half_elapsed_s"] = half_elapsed
    out["match_clock_s"] = np.where(out["match_half"] == "1H", half_elapsed, 45 * 60 + half_elapsed)
    out["match_clock_status"] = np.where(
        (out["match_half"] == "2H") & (half_elapsed < 0),
        "2H pre-kickoff",
        "in-play timeline",
    )
    return out


def drop_all_null_columns(
    table: pd.DataFrame,
    keep: Iterable[str] = (),
    row_mask: np.ndarray | pd.Series | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns empty in the selected rows and return their names."""
    keep = set(keep)
    inspected = table if row_mask is None else table.loc[row_mask]
    dropped = [column for column in table.columns if column not in keep and inspected[column].isna().all()]
    return table.drop(columns=dropped), dropped


def load_supplied_frame_table(game: str = GAME, data_dir: str | os.PathLike[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Load one raw row per tactical frame before feature engineering.

    ATD is aligned to zero-based video frames with ``frame = frame_id - 1``.
    Homography is merged on that key. Smart Tagging stays separate until
    ``add_context_features`` so the raw table remains auditable.
    """
    paths = asset_paths(game=game, data_dir=data_dir, source="tactical")
    atd = load_raw_atd_compact(game=game, data_dir=data_dir, source="tactical")
    atd, dropped_atd = drop_all_null_columns(atd, keep=("frame",))
    homography = pd.read_csv(paths["homography"])
    table = atd.merge(homography, on="frame", how="left", validate="one_to_one")
    graded = graded_mask(len(table), game)
    table, dropped_in_graded = drop_all_null_columns(table, keep=("frame",), row_mask=graded)
    report = {
        "raw_shape_after_empty_drop": table.shape,
        "dropped_all_missing_atd_columns": dropped_atd,
        "dropped_all_missing_on_graded_frames": dropped_in_graded,
        "n_frames": len(table),
        "frame_range": (int(table["frame"].min()), int(table["frame"].max())),
    }
    return table, report


def load_raw_atd_compact(
    game: str = GAME,
    data_dir: str | os.PathLike[str] | None = None,
    source: str = "tactical",
) -> pd.DataFrame:
    """Load the raw ATD into compact numeric columns without kloppy expansion.

    The raw export contains one 1-based frame id, 46 temporary slots with three
    numeric values each, and a three-value ball record. The possession pipeline
    needs the slot ``x``/``y`` coordinates and ball ``x``/``y``; retaining the
    third raw value keeps the parser inspectable without creating kloppy's large
    object-heavy dataframe.
    """

    raw_path = require(["atd_raw"], game, data_dir, source)[0]
    n_frames = source_frames(source, game)
    slot_count = None
    values = None
    rows_read = 0
    with open(raw_path, encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            rows_read += 1
            frame_text, payload = line.rstrip("\n").split(":", 1)
            player_text, ball_text = payload.rsplit(":", 1)
            player_values = player_text.split(";")
            if slot_count is None:
                slot_count = len(player_values)
                values = np.full((n_frames, slot_count * 3 + 3), np.nan, dtype=np.float32)
            frame_id = int(frame_text)
            target = frame_id - 1
            if target < 0 or target >= n_frames:
                raise ValueError(f"frame id {frame_id} is outside 1..{n_frames}")
            for slot, position in enumerate(player_values):
                parts = position.split(",")
                for component in range(min(3, len(parts))):
                    if parts[component] != "NaN":
                        values[target, slot * 3 + component] = float(parts[component])
            for component, part in enumerate(ball_text.split(",")[:3]):
                if part != "NaN":
                    values[target, slot_count * 3 + component] = float(part)
    if values is None or rows_read != n_frames:
        raise ValueError(f"{raw_path}: expected {n_frames} rows, got {rows_read}")
    columns = {}
    for slot in range(1, slot_count + 1):
        columns[f"{slot}_x"] = values[:, (slot - 1) * 3]
        columns[f"{slot}_y"] = values[:, (slot - 1) * 3 + 1]
        columns[f"{slot}_d"] = values[:, (slot - 1) * 3 + 2]
    columns["ball_x"] = values[:, slot_count * 3]
    columns["ball_y"] = values[:, slot_count * 3 + 1]
    columns["ball_z"] = values[:, slot_count * 3 + 2]
    columns["frame_id"] = np.arange(1, n_frames + 1, dtype=np.int32)
    columns["frame"] = np.arange(n_frames, dtype=np.int32)
    return pd.DataFrame(columns)


def _interval_count(events: Iterable[dict], n_frames: int) -> np.ndarray:
    """Count event intervals efficiently using a difference array."""
    changes = np.zeros(n_frames + 1, dtype=np.int32)
    for event in events:
        start = max(0, int(event["start_frame"]))
        end = min(n_frames, int(event["end_frame"]))
        if start < end:
            changes[start] += 1
            changes[end] -= 1
    return np.cumsum(changes[:-1])


def add_context_features(frame_table: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    """Convert Smart Tagging intervals into inspectable per-frame context flags.

    Smart Tagging is represented as context rather than a frame-level label.
    Code-specific flags let scenario logic ask the right follow-up question:
    for example, inspect reception evidence during ``LONG BALL`` and inspect
    low-motion/boundary evidence during ``SET PIECES``.
    """
    out = frame_table.copy()
    out["smart_event_count"] = _interval_count(events, len(out))
    normalised_codes = {
        str(event.get("code") or "").strip().lower()
        for event in events
        if str(event.get("code") or "").strip()
    }
    team_name_to_label = {
        "barcelona": "H",
        "atletic club": "A",
        "athletic club": "A",
    }
    for code in sorted(normalised_codes):
        slug = "_".join(code.split())
        code_events = [event for event in events if str(event.get("code") or "").strip().lower() == code]
        out[f"smart_{slug}"] = _interval_count(code_events, len(out)) > 0
        # Preserve the event's team dimension.  The boolean flag above is
        # useful for timing, but by itself loses whether the tag describes H
        # or A.  This remains context evidence; it is not an H/A label.
        team_context = np.full(len(out), None, dtype=object)
        for event in code_events:
            team_name = str((event.get("tags") or {}).get("Team") or "").strip().lower()
            team = team_name_to_label.get(team_name)
            if team is None:
                continue
            start = max(0, int(event["start_frame"]))
            end = min(len(out), int(event["end_frame"]) + 1)
            team_context[start:end] = team
        out[f"smart_{slug}_team"] = team_context
    set_piece_events = [event for event in events if (event.get("code") or "").strip().lower() == "set pieces"]
    out["smart_set_piece"] = _interval_count(set_piece_events, len(out)) > 0
    # Keep the detailed restart type as inspectable context.  A tagged type is
    # an interval hint, not the exact kick/reception frame.
    restart_type = np.full(len(out), None, dtype=object)
    for event in set_piece_events:
        event_type = (event.get("tags") or {}).get("Type")
        if not event_type:
            continue
        start = max(0, int(event["start_frame"]))
        end = min(len(out), int(event["end_frame"]) + 1)
        restart_type[start:end] = event_type
    out["smart_restart_type"] = restart_type
    # A transition tag is most useful when it agrees with the team that later
    # controls the ball.  Keep the components separate for inspection and add
    # one context-only pair flag; no label is assigned here.
    attacking_team = out.get(
        "smart_attacking_transition_team",
        pd.Series(None, index=out.index, dtype=object),
    )
    defensive_team = out.get(
        "smart_defensive_transition_team",
        pd.Series(None, index=out.index, dtype=object),
    )
    out["smart_transition_pair"] = (
        attacking_team.notna()
        & defensive_team.notna()
        & attacking_team.ne(defensive_team)
    )
    out["smart_transition_attack_team"] = attacking_team
    # Progression/build-up tags are weak positive possession context; they are
    # intentionally not allowed to override a geometric reception anchor.
    progression_team = out.get(
        "smart_progression_team",
        pd.Series(None, index=out.index, dtype=object),
    )
    build_up_team = out.get(
        "smart_build_up_team",
        pd.Series(None, index=out.index, dtype=object),
    )
    out["smart_progression_or_build_up_team"] = progression_team.where(
        progression_team.notna(), build_up_team
    )
    return out


def _distance_matrix(frame_table: pd.DataFrame, slots: list[int], config: BaselineConfig) -> np.ndarray:
    """Return frame-by-slot ball distances in metres; unavailable values are NaN."""
    ball_x = pd.to_numeric(frame_table["ball_x"], errors="coerce").to_numpy(float)
    ball_y = pd.to_numeric(frame_table["ball_y"], errors="coerce").to_numpy(float)
    distances = np.full((len(frame_table), len(slots)), np.nan, dtype=float)
    for column, slot in enumerate(slots):
        x_name, y_name = f"{slot}_x", f"{slot}_y"
        if x_name not in frame_table or y_name not in frame_table:
            continue
        x = pd.to_numeric(frame_table[x_name], errors="coerce").to_numpy(float)
        y = pd.to_numeric(frame_table[y_name], errors="coerce").to_numpy(float)
        distances[:, column] = np.hypot((x - ball_x) * config.pitch_length_m, (y - ball_y) * config.pitch_width_m)
    return distances


def add_basic_features(frame_table: pd.DataFrame, team_by_slot: dict[int, str], config: BaselineConfig = BaselineConfig()) -> pd.DataFrame:
    """Add nearest-player distances, slots, team, and distance margin."""
    out = frame_table.copy()
    for team in ("H", "A"):
        slots = [slot for slot, slot_team in team_by_slot.items() if slot_team == team]
        distances = _distance_matrix(out, slots, config)
        safe = np.where(np.isfinite(distances), distances, np.inf)
        nearest_index = np.argmin(safe, axis=1)
        nearest_distance = safe[np.arange(len(out)), nearest_index]
        available = np.isfinite(nearest_distance)
        nearest_distance[~available] = np.nan
        nearest_slot = np.array([slots[index] for index in nearest_index], dtype=float)
        nearest_slot[~available] = np.nan
        out[f"nearest_{team.lower()}_slot"] = nearest_slot
        out[f"nearest_{team.lower()}_distance_m"] = nearest_distance

    home = out["nearest_h_distance_m"].to_numpy(float)
    away = out["nearest_a_distance_m"].to_numpy(float)
    home_ok, away_ok = np.isfinite(home), np.isfinite(away)
    nearest_team = np.full(len(out), None, dtype=object)
    nearest_team[home_ok & (~away_ok | (home <= away))] = "H"
    nearest_team[away_ok & (~home_ok | (away < home))] = "A"
    out["closest_team"] = nearest_team
    out["closest_slot"] = np.where(nearest_team == "H", out["nearest_h_slot"], out["nearest_a_slot"])
    out["closest_distance_m"] = np.fmin(home, away)
    out["team_distance_margin_m"] = away - home
    return out


def _zone_index(values: pd.Series, edges: tuple[float, ...]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    result = np.full(len(numeric), np.nan, dtype=float)
    valid = np.isfinite(numeric)
    result[valid] = np.clip(np.digitize(numeric[valid], edges[1:-1]), 0, len(edges) - 2)
    return pd.Series(result, index=values.index, dtype="Int64")


def add_spatial_features(frame_table: pd.DataFrame, team_by_slot: dict[int, str], config: BaselineConfig = BaselineConfig()) -> pd.DataFrame:
    """Add configurable ball zones and counts of players near the ball."""
    out = frame_table.copy()
    out["ball_zone_x"] = _zone_index(out["ball_x"], config.zone_x_edges)
    out["ball_zone_y"] = _zone_index(out["ball_y"], config.zone_y_edges)
    for team in ("H", "A"):
        slots = [slot for slot, slot_team in team_by_slot.items() if slot_team == team]
        distances = _distance_matrix(out, slots, config)
        out[f"{team.lower()}_players_within_radius"] = np.sum(np.isfinite(distances) & (distances <= config.density_radius_m), axis=1)
        out[f"{team.lower()}_visible_players"] = np.sum(np.isfinite(distances), axis=1)

    # Context around the current closest player. These are geometric signals,
    # not possession labels: pressure is opponent proximity, support is nearby
    # teammates, and pass options are teammates within a configurable radius.
    n_frames = len(out)
    slot_coordinates: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for slot in team_by_slot:
        slot_coordinates[slot] = (
            pd.to_numeric(out.get(f"{slot}_x", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float),
            pd.to_numeric(out.get(f"{slot}_y", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float),
        )
    closest_slot = pd.to_numeric(out["closest_slot"], errors="coerce").to_numpy(float)
    closest_team = out["closest_team"].to_numpy(object)
    closest_x = np.full(n_frames, np.nan)
    closest_y = np.full(n_frames, np.nan)
    for slot, (x, y) in slot_coordinates.items():
        mask = closest_slot == slot
        closest_x[mask] = x[mask]
        closest_y[mask] = y[mask]

    for column in (
        "closest_player_opponent_distance_m",
        "closest_player_opponent_density_5m",
        "closest_player_teammate_support_10m",
        "closest_player_pass_options_15m",
    ):
        out[column] = np.nan

    for team in ("H", "A"):
        team_slots = [slot for slot, slot_team in team_by_slot.items() if slot_team == team]
        opponent_slots = [slot for slot, slot_team in team_by_slot.items() if slot_team != team]
        opponent_columns = []
        teammate_columns = []
        candidate_mask = closest_team == team
        for slot in opponent_slots:
            x, y = slot_coordinates[slot]
            opponent_columns.append(np.hypot((x - closest_x) * config.pitch_length_m, (y - closest_y) * config.pitch_width_m))
        for slot in team_slots:
            x, y = slot_coordinates[slot]
            distance = np.hypot((x - closest_x) * config.pitch_length_m, (y - closest_y) * config.pitch_width_m)
            distance[closest_slot == slot] = np.nan
            teammate_columns.append(distance)
        if not opponent_columns:
            continue
        opponents = np.column_stack(opponent_columns)
        teammates = np.column_stack(teammate_columns) if teammate_columns else np.full((n_frames, 0), np.nan)
        nearest_opponent = np.full(n_frames, np.nan, dtype=float)
        valid_opponent_rows = np.isfinite(opponents).any(axis=1)
        nearest_opponent[valid_opponent_rows] = np.nanmin(opponents[valid_opponent_rows], axis=1)
        nearest_opponent[~candidate_mask] = np.nan
        opponent_density = np.sum(np.isfinite(opponents) & (opponents <= config.density_radius_m), axis=1).astype(float)
        opponent_density[~candidate_mask] = np.nan
        support = np.sum(np.isfinite(teammates) & (teammates <= 10.0), axis=1).astype(float)
        support[~candidate_mask] = np.nan
        pass_options = np.sum(np.isfinite(teammates) & (teammates <= 15.0), axis=1).astype(float)
        pass_options[~candidate_mask] = np.nan
        for column, values in (
            ("closest_player_opponent_distance_m", nearest_opponent),
            ("closest_player_opponent_density_5m", opponent_density),
            ("closest_player_teammate_support_10m", support),
            ("closest_player_pass_options_15m", pass_options),
        ):
            out[column] = out[column].where(out[column].notna(), pd.Series(values, index=out.index))
    opponent_density = pd.to_numeric(out["closest_player_opponent_density_5m"], errors="coerce")
    support = pd.to_numeric(out["closest_player_teammate_support_10m"], errors="coerce")
    out["closest_player_isolated"] = (opponent_density >= 2) & (support == 0)
    return out


def fill_missing_with_bracketing(labels: np.ndarray) -> np.ndarray:
    """Fill unknown labels from both sides, preferring the previous H/A label."""
    out = labels.astype(object, copy=True)
    known = np.isin(out, ["H", "A"])
    previous = np.full(len(out), None, dtype=object)
    following = np.full(len(out), None, dtype=object)
    last = None
    for index in range(len(out)):
        previous[index] = last
        if known[index]:
            last = out[index]
    last = None
    for index in range(len(out) - 1, -1, -1):
        following[index] = last
        if known[index]:
            last = out[index]
    for index in np.flatnonzero(~known):
        out[index] = previous[index] if previous[index] is not None else following[index]
    return out


@dataclass
class ClosestTeamRuleModel:
    """Transparent rule model built on the closest-player feature."""

    game: str = GAME
    data_dir: str | os.PathLike[str] | None = None
    config: BaselineConfig = field(default_factory=BaselineConfig)

    def build_features(self, include_spatial: bool = True) -> tuple[pd.DataFrame, dict]:
        table, report = load_supplied_frame_table(self.game, self.data_dir)
        paths = asset_paths(self.game, self.data_dir, source="tactical")
        team_by_slot = slot_team_map(paths["atd_meta"])
        events = load_smart_tagging(game=self.game, data_dir=self.data_dir)
        table = add_context_features(table, events)
        table = add_basic_features(table, team_by_slot, self.config)
        if include_spatial:
            table = add_spatial_features(table, team_by_slot, self.config)
        report.update({"n_smart_events": len(events), "feature_shape": table.shape})
        return table, report

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Produce exactly one official label per frame; non-graded frames are X."""
        in_play = graded_mask(len(features), self.game)
        labels = np.full(len(features), None, dtype=object)
        labels[in_play] = features["closest_team"].to_numpy(dtype=object)[in_play]
        if self.config.set_piece_as_dead_ball:
            labels[in_play & features["smart_set_piece"].to_numpy(bool)] = "D"
        missing = in_play & pd.isna(labels)
        if missing.any() and self.config.missing_ball_policy == "prefer_previous":
            filled = fill_missing_with_bracketing(labels)
            labels[missing] = filled[missing]
        output = np.full(len(features), "X", dtype="<U1")
        output[in_play] = np.where(pd.isna(labels[in_play]), "X", labels[in_play])
        return output

    @staticmethod
    def write_submission(path: str | os.PathLike[str], labels: np.ndarray) -> str:
        """Write the required zero-based frame/label CSV and validate its alphabet."""
        invalid = set(labels) - set(LABELS)
        if invalid:
            raise ValueError(f"invalid labels: {sorted(invalid)}")
        os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
        pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)
        return os.fspath(path)


if __name__ == "__main__":
    model = ClosestTeamRuleModel()
    features, report = model.build_features()
    labels = model.predict(features)
    output = model.write_submission("evaluation/results/baseline_nearest_setpiece.csv", labels)
    print(report)
    print(pd.Series(labels).value_counts().sort_index())
    print(output)
