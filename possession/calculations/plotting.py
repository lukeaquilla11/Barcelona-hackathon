"""Compact pitch plots for auditing ball gaps and nearest-player evidence.

The ATD export stores positions normalised to the pitch.  This module converts
them to metres for display: x is 0--105.3 m and y is 0--68 m.  The plotting
functions are diagnostics only; they never create submission labels.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

def _apply_frame_time_axis(
    ax,
    *,
    fps: float = 25.0,
    frame_to_match_clock: Mapping[int, float] | None = None,
) -> None:
    """Show tactical frames with the kickoff-anchored match clock underneath."""
    from matplotlib.ticker import FuncFormatter

    def format_tick(value, _position):
        frame = int(round(value))
        if frame_to_match_clock:
            keys = np.asarray(list(frame_to_match_clock), dtype=float)
            vals = np.asarray([frame_to_match_clock[int(key)] for key in keys], dtype=float)
            seconds = float(np.interp(value, keys, vals))
        else:
            seconds = max(0, value / fps)
        if seconds < 0:
            clock = f"pre-kickoff ({seconds:.1f}s)"
        else:
            clock = f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
        return f"{frame}\n({clock})"

    ax.xaxis.set_major_formatter(FuncFormatter(format_tick))
    ax.set_xlabel("tactical frame\n(match clock from first kickoff)")



def _metres(values: pd.Series, scale: float) -> pd.Series:
    """Convert normalised ATD coordinates to metres, preserving missing values."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric * scale


def select_representative_ball_gap(
    features: pd.DataFrame,
    *,
    prefer_player_supported: bool = True,
) -> tuple[int, int] | None:
    """Return an internal missing-ball interval suitable for a diagnostic plot."""
    if "ball_missing_gap_frames" in features:
        gaps = features.loc[features["ball_missing_gap_frames"].notna()].copy()
    else:
        # Fallback for a basic feature table: identify internal missing runs
        # directly, so the diagnostic can be used before engineered features
        # have been added.
        missing = features["ball_x"].isna() | features["ball_y"].isna()
        starts = np.flatnonzero(missing.to_numpy() & ~np.r_[False, missing.to_numpy()[:-1]])
        ends = np.flatnonzero(missing.to_numpy() & ~np.r_[missing.to_numpy()[1:], False])
        candidates = [
            (int(features.iloc[start]["frame"]), int(features.iloc[end]["frame"]))
            for start, end in zip(starts, ends)
            if start > 0 and end + 1 < len(features)
        ]
        return min(candidates, key=lambda pair: pair[1] - pair[0]) if candidates else None
    if gaps.empty:
        return None
    if prefer_player_supported and "gap_player_path_supported" in gaps:
        supported = gaps.loc[gaps["gap_player_path_supported"].fillna(False)]
        if not supported.empty:
            gaps = supported
    # Prefer a short, readable internal gap.  Very long missing runs are still
    # useful for aggregate audits, but a single plot should show the local
    # before/after evidence rather than compressing an entire phase of play.
    row = gaps.sort_values(["ball_missing_gap_frames", "frame"]).iloc[0]
    start = int(row["frame"])
    length = max(1, int(row["ball_missing_gap_frames"]))
    return start, start + length - 1


def select_ball_gap_event_window(
    features: pd.DataFrame,
    gap: tuple[int, int],
    *,
    minimum_context_frames: int = 200,
    control_radius_m: float = 2.5,
) -> dict[str, int]:
    """Select a data-driven before/after window around an internal ball gap.

    The returned control frames are evidence anchors for a diagnostic plot,
    not hand-written possession labels.  The search prefers existing stable
    or selected anchors and falls back to the nearest observed player-ball
    control geometry on each side.
    """
    gap_start, gap_end = gap
    table = features.sort_values("frame")
    before = table.loc[table.frame.lt(gap_start) & table.ball_x.notna() & table.ball_y.notna()]
    after = table.loc[table.frame.gt(gap_end) & table.ball_x.notna() & table.ball_y.notna()]
    if before.empty or after.empty:
        return {
            "start_frame": max(0, gap_start - minimum_context_frames),
            "end_frame": gap_end + minimum_context_frames,
            "pre_control_frame": gap_start - 1,
            "post_control_frame": gap_end + 1,
        }
    pre_mask = before.get("selected_anchor_team", pd.Series(None, index=before.index)).isin(["H", "A"])
    post_mask = after.get("selected_anchor_team", pd.Series(None, index=after.index)).isin(["H", "A"])
    pre_control = before.loc[pre_mask]
    post_control = after.loc[post_mask]
    if pre_control.empty:
        pre_control = before.loc[before.get("closest_distance_m", pd.Series(np.inf, index=before.index)).le(control_radius_m)]
    if post_control.empty:
        post_control = after.loc[after.get("closest_distance_m", pd.Series(np.inf, index=after.index)).le(control_radius_m)]
    pre_frame = int(pre_control.iloc[-1].frame) if not pre_control.empty else int(before.iloc[-1].frame)
    post_frame = int(post_control.iloc[0].frame) if not post_control.empty else int(after.iloc[0].frame)
    return {
        "start_frame": max(int(table.frame.min()), pre_frame - minimum_context_frames // 2),
        "end_frame": min(int(table.frame.max()), post_frame + minimum_context_frames // 2),
        "pre_control_frame": pre_frame,
        "post_control_frame": post_frame,
    }


def _slot_columns(features: pd.DataFrame, team_by_slot: Mapping[int, str]) -> list[int]:
    return sorted(
        slot for slot in team_by_slot
        if f"{slot}_x" in features and f"{slot}_y" in features
    )


def select_persistent_nearby_slots(
    features: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    *,
    team_by_slot: Mapping[int, str],
    radius_m: float = 12.0,
    n_players: int | None = None,
) -> tuple[list[int], float, bool]:
    """Choose a fixed union of nearby coordinate streams for one plot window.

    A slot enters the fixed set if it is within ``radius_m`` on *any* frame in
    the window. The union is then plotted on every frame, so a slot cannot be
    silently replaced when it moves away. ``n_players`` is an optional display
    cap; ``None`` keeps the complete union. The boolean reports whether the
    requested radius produced at least one candidate.
    """
    if n_players is not None and n_players < 1:
        raise ValueError("n_players must be positive")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")

    table = features.sort_values("frame")
    window = table.loc[table["frame"].between(start_frame, end_frame)].copy()
    if window.empty:
        return [], float(radius_m), False

    bx = pd.to_numeric(window.get("ball_x_interp", window["ball_x"]), errors="coerce")
    by = pd.to_numeric(window.get("ball_y_interp", window["ball_y"]), errors="coerce")
    if bx.isna().any():
        bx = bx.interpolate(limit_area="inside")
    if by.isna().any():
        by = by.interpolate(limit_area="inside")

    records: list[tuple[float, float, int]] = []
    for slot in _slot_columns(window, team_by_slot):
        px = pd.to_numeric(window[f"{slot}_x"], errors="coerce")
        py = pd.to_numeric(window[f"{slot}_y"], errors="coerce")
        distances = np.hypot((px - bx) * PITCH_LENGTH_M, (py - by) * PITCH_WIDTH_M)
        observed_distances = distances.dropna()
        if observed_distances.empty:
            continue
        if bool((observed_distances <= radius_m).any()):
            records.append((float(observed_distances.min()), float(observed_distances.mean()), slot))

    records.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = [slot for _, _, slot in records]
    if n_players is not None:
        selected = selected[:n_players]
    return selected, float(radius_m), bool(selected)


def plot_trajectory_window(
    features: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    *,
    team_by_slot: Mapping[int, str],
    reference_frame: int | None = None,
    nearby_radius_m: float = 12.0,
    nearby_count: int = 6,
    include_slots: Sequence[int] | None = None,
    zoom_to_involved: bool = False,
    show_all_reference_players: bool = True,
    show_nearest_geometry: bool = False,
    touch_radius_m: float = 1.5,
    ax=None,
):
    """Plot observed ball paths, a dotted missing bridge, players, and nearest path.

    The nearest-player line is intentionally labelled as *nearest geometry*:
    ATD slots are not asserted to be persistent player identities.  The plot
    shows why Smart Tagging is a scenario clue rather than a frame-level label.
    """
    import matplotlib.pyplot as plt
    from starter.utils import draw_pitch

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 7))
    table = features.sort_values("frame").copy()
    window = table.loc[table["frame"].between(start_frame, end_frame)].copy()
    if window.empty:
        raise ValueError("plot window contains no frames")
    if reference_frame is None:
        reference_frame = int(round((start_frame + end_frame) / 2))
    ref = window.iloc[(window["frame"] - reference_frame).abs().argmin()]

    # Reuse the starter notebook's football-pitch drawing, but with the
    # corrected 105.3 x 68 m dimensions used by the feature calculations.
    draw_pitch(ax, length=PITCH_LENGTH_M, width=PITCH_WIDTH_M)
    ax.set_xlim(0, PITCH_LENGTH_M)
    ax.set_ylim(0, PITCH_WIDTH_M)
    ax.set_xlabel("pitch x (m), origin at pitch corner")
    ax.set_ylabel("pitch y (m), 0–68 m supplied coordinate range")
    ax.set_title(f"Ball-gap audit: frames {start_frame}–{end_frame}; reference frame {int(ref['frame'])}")

    bx = _metres(window["ball_x"], PITCH_LENGTH_M)
    by = _metres(window["ball_y"], PITCH_WIDTH_M)
    observed = bx.notna() & by.notna()
    # Fade older movement and emphasize the most recent coordinates.  This is
    # visual recency only; it does not alter any feature or label calculation.
    obs_idx = np.flatnonzero(observed.to_numpy())
    for pos, (left, right) in enumerate(zip(obs_idx[:-1], obs_idx[1:])):
        if right != left + 1:
            continue
        recency = (pos + 1) / max(1, len(obs_idx) - 1)
        ax.plot([bx.iloc[left], bx.iloc[right]], [by.iloc[left], by.iloc[right]],
                color="#ffffff", linewidth=0.8 + 1.7 * recency,
                alpha=0.25 + 0.75 * recency,
                label="observed ball path" if pos == 0 else None)
    if len(obs_idx):
        ax.scatter(bx.iloc[obs_idx], by.iloc[obs_idx], s=10,
                   color="#ffffff", edgecolor="#111111", linewidth=0.45,
                   alpha=np.linspace(0.3, 1.0, len(obs_idx)), zorder=5)

    # Draw each internal missing run as a distinct dotted bridge.
    missing = ~observed.to_numpy()
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
    for left, right in zip(starts, ends):
        before, after = left - 1, right + 1
        if before >= 0 and after < len(window) and observed.iloc[before] and observed.iloc[after]:
            ax.plot([bx.iloc[before], bx.iloc[after]], [by.iloc[before], by.iloc[after]],
                    color="#d95f02", linestyle=(0, (2, 2)), linewidth=2.2,
                    label="missing-ball bridge" if left == starts[0] else None)

    # Direction arrows make the path readable without requiring tracks.
    visible = np.flatnonzero(observed.to_numpy())
    for left, right in zip(visible[:-1], visible[1:]):
        if right == left + 1:
            ax.annotate("", xy=(bx.iloc[right], by.iloc[right]),
                        xytext=(bx.iloc[left], by.iloc[left]),
                        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "alpha": 0.65})

    # Match the frame-view convention: the ball itself is a white, black-edged
    # point.  The white line above is its observed trajectory; orange dotted
    # segments are the missing interval and are intentionally distinct.
    ref_ball_x = bx.iloc[(window["frame"] - int(ref["frame"])).abs().argmin()]
    ref_ball_y = by.iloc[(window["frame"] - int(ref["frame"])).abs().argmin()]
    if np.isfinite(ref_ball_x) and np.isfinite(ref_ball_y):
        ax.scatter([ref_ball_x], [ref_ball_y], s=70, color="white", edgecolor="black",
                   linewidth=1.1, zorder=6, label="ball at reference frame")

    slots = _slot_columns(features, team_by_slot)
    nearby_slots, effective_radius_m, requested_radius_satisfied = select_persistent_nearby_slots(
        window,
        start_frame,
        end_frame,
        team_by_slot=team_by_slot,
        radius_m=nearby_radius_m,
        n_players=nearby_count,
    )
    for slot in include_slots or ():
        if slot in slots and slot not in nearby_slots:
            nearby_slots.append(slot)
    for slot in slots:
        if not show_all_reference_players and nearby_slots and slot not in nearby_slots:
            continue
        x = _metres(pd.Series([ref[f"{slot}_x"]]), PITCH_LENGTH_M).iloc[0]
        y = _metres(pd.Series([ref[f"{slot}_y"]]), PITCH_WIDTH_M).iloc[0]
        if np.isfinite(x) and np.isfinite(y):
            team = team_by_slot[slot]
            color = "#d62728" if team == "H" else "#1f77b4"
            ax.scatter(x, y, s=26, color=color, edgecolor="black", linewidth=0.55,
                       alpha=0.42, zorder=4, label=f"{team} players at frame" if slot == slots[0] else None)

    if show_nearest_geometry:
        # Optional diagnostic only. The primary evidence remains the observed
        # ball path and the coordinate paths of the fixed nearby-slot union.
        nearest_x, nearest_y = [], []
        for _, row in window.iterrows():
            ball_x = row.get("ball_x_interp", row.get("ball_x"))
            ball_y = row.get("ball_y_interp", row.get("ball_y"))
            if not np.isfinite(ball_x) or not np.isfinite(ball_y):
                ball_x = pd.to_numeric(window["ball_x"], errors="coerce").interpolate(limit_area="inside").loc[row.name]
                ball_y = pd.to_numeric(window["ball_y"], errors="coerce").interpolate(limit_area="inside").loc[row.name]
            if not np.isfinite(ball_x) or not np.isfinite(ball_y):
                nearest_x.append(np.nan); nearest_y.append(np.nan); continue
            candidates = []
            for slot in slots:
                px, py = row.get(f"{slot}_x"), row.get(f"{slot}_y")
                if np.isfinite(px) and np.isfinite(py):
                    distance = np.hypot((px - ball_x) * PITCH_LENGTH_M,
                                        (py - ball_y) * PITCH_WIDTH_M)
                    candidates.append((distance, px, py))
            if candidates:
                _, px, py = min(candidates)
                nearest_x.append(px * PITCH_LENGTH_M); nearest_y.append(py * PITCH_WIDTH_M)
            else:
                nearest_x.append(np.nan); nearest_y.append(np.nan)
        ax.plot(nearest_x, nearest_y, color="#9467bd", linewidth=1.8, linestyle="--",
                marker=".", markersize=4, label="nearest-player geometry (optional)")

    # Select a constant candidate set once for the whole window.  This is a
    # coordinate-stream diagnostic, not persistent identity tracking.
    for slot in nearby_slots:
        x = _metres(pd.to_numeric(window[f"{slot}_x"], errors="coerce"), PITCH_LENGTH_M)
        y = _metres(pd.to_numeric(window[f"{slot}_y"], errors="coerce"), PITCH_WIDTH_M)
        team = team_by_slot[slot]
        color = "#d62728" if team == "H" else "#1f77b4"
        valid = x.notna() & y.notna()
        valid_idx = np.flatnonzero(valid.to_numpy())
        for pos, (left, right) in enumerate(zip(valid_idx[:-1], valid_idx[1:])):
            if right != left + 1:
                continue
            recency = (right - valid_idx[0]) / max(1, valid_idx[-1] - valid_idx[0])
            ax.plot([x.iloc[left], x.iloc[right]], [y.iloc[left], y.iloc[right]],
                color=color, linewidth=1.8 + 1.2 * recency,
                    alpha=0.35 + 0.60 * recency, zorder=7,
                    label=f"{team} player coordinate paths" if slot == nearby_slots[0] and pos == 0 else None)
        valid_points = np.flatnonzero(valid.to_numpy())
        if len(valid_points) >= 2:
            ax.scatter([x.iloc[valid_points[0]], x.iloc[valid_points[-1]]],
                       [y.iloc[valid_points[0]], y.iloc[valid_points[-1]]],
                       s=24, color=color, edgecolor="white", linewidth=0.7,
                       zorder=8, label="player path start / end" if slot == nearby_slots[0] else None)

    # Touch candidates: player coordinates that fall directly on/near an
    # observed ball coordinate. This is a proximity proxy, not touch identity.
    touch_label_used = False
    ball_x_m = _metres(pd.to_numeric(window["ball_x"], errors="coerce"), PITCH_LENGTH_M)
    ball_y_m = _metres(pd.to_numeric(window["ball_y"], errors="coerce"), PITCH_WIDTH_M)
    ball_valid = ball_x_m.notna() & ball_y_m.notna()
    for slot in nearby_slots:
        player_x_m = _metres(pd.to_numeric(window[f"{slot}_x"], errors="coerce"), PITCH_LENGTH_M)
        player_y_m = _metres(pd.to_numeric(window[f"{slot}_y"], errors="coerce"), PITCH_WIDTH_M)
        player_valid = player_x_m.notna() & player_y_m.notna()
        touch = ball_valid & player_valid & (
            np.hypot(player_x_m - ball_x_m, player_y_m - ball_y_m) <= touch_radius_m
        )
        if touch.any():
            team = team_by_slot[slot]
            color = "#d62728" if team == "H" else "#1f77b4"
            touch_idx = np.flatnonzero(touch.to_numpy())
            alpha = np.linspace(0.35, 1.0, len(touch_idx))
            ax.scatter(
                player_x_m.iloc[touch_idx], player_y_m.iloc[touch_idx],
                s=78, color=color, edgecolor="black", linewidth=1.3,
                alpha=alpha, zorder=7,
                label=(f"{team} player-ball proximity ≤{touch_radius_m:.1f} m"
                       if not touch_label_used else None),
            )
            touch_label_used = True

    radius_note = (
        f"fixed union: {len(nearby_slots)} slots entered ≤{effective_radius_m:.1f} m at least once"
        if requested_radius_satisfied
        else "fixed union: no slot entered the requested radius"
    )
    ax.text(0.01, 0.03, radius_note, transform=ax.transAxes, fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#999"})
    if zoom_to_involved and nearby_slots:
        x_values = list(bx.loc[observed].dropna())
        y_values = list(by.loc[observed].dropna())
        for slot in nearby_slots:
            x_values.extend(_metres(pd.to_numeric(window[f"{slot}_x"], errors="coerce"), PITCH_LENGTH_M).dropna())
            y_values.extend(_metres(pd.to_numeric(window[f"{slot}_y"], errors="coerce"), PITCH_WIDTH_M).dropna())
        if x_values and y_values:
            margin_m = 8.0
            ax.set_xlim(max(0.0, min(x_values) - margin_m), min(PITCH_LENGTH_M, max(x_values) + margin_m))
            ax.set_ylim(max(0.0, min(y_values) - margin_m), min(PITCH_WIDTH_M, max(y_values) + margin_m))

    smart = window.get("smart_set_piece", pd.Series(False, index=window.index)).fillna(False)
    if smart.any():
        ax.text(0.01, 0.99,
                "Smart Tagging set-piece context present\n"
                "→ scenario clue, not exact ball/reception boundary",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#999"})
    ax.legend(loc="upper right", fontsize=8)
    return ax.figure, ax


def select_evidence_window(
    features: pd.DataFrame,
    *,
    kind: str = "missing_ball",
    context_frames: int = 50,
) -> tuple[int, int] | None:
    """Select an algorithmic window around a missing-ball gap or transition."""
    if kind == "missing_ball":
        gap = select_representative_ball_gap(features)
        if gap is None:
            return None
        center = (gap[0] + gap[1]) // 2
    elif kind == "transition":
        candidates = features.loc[features.get("transition_anchor", pd.Series(False, index=features.index)).fillna(False)]
        if candidates.empty:
            candidates = features.loc[features.get("genuine_reception_anchor", pd.Series(False, index=features.index)).fillna(False)]
        if candidates.empty:
            return None
        center = int(candidates.iloc[0]["frame"])
    elif kind == "anchor":
        candidates = features.loc[features.get("selected_anchor_team", pd.Series(index=features.index)).notna()]
        if candidates.empty:
            return None
        center = int(candidates.iloc[0]["frame"])
    else:
        raise ValueError("kind must be 'missing_ball', 'transition', or 'anchor'")
    return max(0, center - context_frames), min(int(features["frame"].max()), center + context_frames)


def select_ball_gap_event_window(
    features: pd.DataFrame,
    gap: tuple[int, int],
    *,
    minimum_context_frames: int = 75,
    control_radius_m: float = 2.5,
) -> dict[str, int]:
    """Choose a gap plot from nearby control evidence, not a fixed width.

    The pre-boundary is the latest observed frame before the gap at which the
    ball is within ``control_radius_m`` of a player.  The post-boundary is the
    earliest analogous frame after the gap.  If either boundary is unavailable,
    the nearest observed frame is used.  A minimum context is then added on
    both sides for the diagnostic plot. This controls what is displayed, not
    the possession decision or the amount of look-ahead used by the live-safe
    pipeline.
    """
    gap_start, gap_end = gap
    table = features.sort_values("frame").copy()
    observed = table["ball_x"].notna() & table["ball_y"].notna()
    distance = pd.to_numeric(table.get("closest_distance_m"), errors="coerce")
    controlled = observed & distance.le(control_radius_m)

    before = table.loc[table["frame"] < gap_start]
    after = table.loc[table["frame"] > gap_end]
    before_control = table.loc[controlled & table["frame"].lt(gap_start)]
    after_control = table.loc[controlled & table["frame"].gt(gap_end)]

    pre_anchor = int(before_control.iloc[-1]["frame"]) if not before_control.empty else int(before.iloc[-1]["frame"])
    post_anchor = int(after_control.iloc[0]["frame"]) if not after_control.empty else int(after.iloc[0]["frame"])
    start = min(pre_anchor, gap_start - minimum_context_frames)
    end = max(post_anchor, gap_end + minimum_context_frames)
    return {
        "start_frame": max(0, int(start)),
        "end_frame": min(int(table["frame"].max()), int(end)),
        "pre_control_frame": pre_anchor,
        "post_control_frame": post_anchor,
    }


def plot_evidence_window(
    features: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    *,
    team_by_slot: Mapping[int, str],
    ground_truth: pd.Series | None = None,
    predicted_labels: pd.Series | None = None,
):
    """Show pitch geometry above a timeline of scenarios, anchors, and switches."""
    import matplotlib.pyplot as plt

    window = features.loc[features["frame"].between(start_frame, end_frame)].copy()
    if window.empty:
        raise ValueError("evidence window contains no frames")
    fig, (pitch_ax, timeline_ax) = plt.subplots(
        2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True
    )
    plot_trajectory_window(
        features,
        start_frame,
        end_frame,
        team_by_slot=team_by_slot,
        reference_frame=int(window.iloc[len(window) // 2]["frame"]),
        ax=pitch_ax,
    )
    # Keep the pitch-and-timeline composite on the same canonical timeline helper used by
    # standalone evidence plots. This prevents the two views from drifting in labels, colours,
    # match-clock formatting, missing-ball shading, or ground-truth rendering.
    plot_timeline_window(
        features,
        start_frame,
        end_frame,
        ground_truth=ground_truth,
        predicted_labels=predicted_labels,
        ax=timeline_ax,
    )
    return fig, (pitch_ax, timeline_ax)


def _plot_label_strip(ax, frames, labels: pd.Series, *, y: float, legend_prefix: str):
    """Draw contiguous H/A/D labels as one consistent colour strip."""
    from matplotlib.patches import Rectangle

    frame_values = np.asarray(frames, dtype=int)
    label_values = labels.reindex(frame_values).to_numpy()
    colors = {"H": "#d62728", "A": "#1f77b4", "D": "#999999"}
    if len(frame_values) == 0:
        return

    start = 0
    labelled = set()
    for i in range(1, len(frame_values) + 1):
        same_run = (
            i < len(frame_values)
            and frame_values[i] == frame_values[i - 1] + 1
            and label_values[i] == label_values[start]
        )
        if same_run:
            continue
        label = label_values[start]
        if label in colors:
            ax.add_patch(Rectangle(
                (frame_values[start] - 0.5, y),
                frame_values[i - 1] - frame_values[start] + 1,
                0.28,
                facecolor=colors[label],
                edgecolor="none",
                label=(
                    {"H": f"{legend_prefix} H / Home", "A": f"{legend_prefix} A / Away",
                     "D": f"{legend_prefix} D / dead / unavailable"}[label]
                    if label not in labelled else "_nolegend_"
                ),
                alpha=0.95,
                zorder=1,
            ))
            labelled.add(label)
        start = i


def _plot_ground_truth_strip(ax, frames, ground_truth: pd.Series, *, y: float = 1.64):
    """Draw contiguous public H/A/D labels as one consistent colour strip."""
    _plot_label_strip(ax, frames, ground_truth, y=y, legend_prefix="ground truth")


def _plot_mismatch_windows(
    ax,
    frames,
    ground_truth: pd.Series,
    predicted_labels: pd.Series,
    *,
    y_min: float = -0.5,
    y_max: float = 2.35,
):
    """Shade contiguous graded frames where prediction and public truth disagree."""
    frame_values = np.asarray(frames, dtype=int)
    gt = ground_truth.reindex(frame_values).astype("string").to_numpy()
    pred = predicted_labels.reindex(frame_values).astype("string").to_numpy()
    mismatch = pd.notna(gt) & pd.notna(pred) & (gt != "X") & (gt != pred)
    if not mismatch.any():
        return
    starts = np.flatnonzero(mismatch & ~np.r_[False, mismatch[:-1]])
    stops = np.flatnonzero(mismatch & ~np.r_[mismatch[1:], False])
    for index, (start, stop) in enumerate(zip(starts, stops)):
        ax.axvspan(
            frame_values[start] - 0.5,
            frame_values[stop] + 0.5,
            color="#d62728",
            alpha=0.10,
            label="label disagreement window" if index == 0 else "_nolegend_",
            zorder=0,
        )


def _plot_anchor_margin(ax, window: pd.DataFrame) -> None:
    """Overlay the continuous nearest-team distance difference.

    ``team_distance_margin_m`` is ``nearest_a_distance_m -
    nearest_h_distance_m``: positive values mean Home is closer to the ball,
    negative values mean Away is closer. The categorical dots remain useful
    for frame-level observations; this trace shows the shape and strength of
    the competition between the two teams.
    """
    if "team_distance_margin_m" not in window:
        return
    margin = pd.to_numeric(window["team_distance_margin_m"], errors="coerce")
    if not margin.notna().any():
        return
    margin_ax = ax.twinx()
    x = window["frame"].to_numpy()
    values = margin.to_numpy(float)
    margin_ax.plot(x, values, color="#6a3d9a", linewidth=1.4, alpha=0.82, label="distance margin (A − H)")
    margin_ax.axhline(0.0, color="#6a3d9a", linewidth=0.8, linestyle="--", alpha=0.45)
    margin_ax.fill_between(x, 0.0, values, where=np.nan_to_num(values, nan=0.0) >= 0.0,
                           color="#d62728", alpha=0.07, interpolate=True)
    margin_ax.fill_between(x, 0.0, values, where=np.nan_to_num(values, nan=0.0) < 0.0,
                           color="#1f77b4", alpha=0.07, interpolate=True)
    margin_ax.set_ylabel("distance margin (m)\npositive = Home closer", color="#6a3d9a", fontsize=8)
    margin_ax.tick_params(axis="y", labelcolor="#6a3d9a", labelsize=8)
    margin_ax.grid(False)
    margin_ax.set_zorder(ax.get_zorder() + 1)
    margin_ax.patch.set_visible(False)


def plot_timeline_window(
    features: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    *,
    ground_truth: pd.Series | None = None,
    predicted_labels: pd.Series | None = None,
    ax=None,
):
    """Plot only the scenario/anchor evidence timeline for a frame window."""
    import matplotlib.pyplot as plt

    window = features.loc[features["frame"].between(start_frame, end_frame)].copy()
    if window.empty:
        raise ValueError("timeline window contains no frames")
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_title("Evidence timeline — observations, not hand-written labels")
    _apply_frame_time_axis(
        ax,
        frame_to_match_clock=dict(zip(window["frame"].astype(int), window["match_clock_s"].astype(float)))
        if "match_clock_s" in window else None,
    )
    ax.set_yticks([0, 0.5, 1, 1.78, 2.08])
    ax.set_yticklabels(["Away", "Dead / unavailable", "Home", "Ground truth", "Prediction"])
    ax.set_ylim(-0.5, 2.4)
    ax.grid(axis="x", alpha=0.2)
    x = window["frame"].to_numpy()
    for column, y, color, label in (
        ("closest_team", 0.82, "#555555", "nearest-team observation"),
        ("selected_anchor_team", 1.18, "#111111", "selected anchor"),
    ):
        if column not in window:
            continue
        values = window[column].map({"A": 0, "H": 1}).to_numpy(dtype=float)
        ax.scatter(x, values * 0.28 + y - 0.14, s=10, color=color, label=label)
    scenario_colors = {
        "DEAD": "#e6ab02", "RESTART_LIVE_BOUNDARY": "#e6ab02",
        "BALL_IN_FLIGHT": "#66a61e", "CONTESTED": "#e7298a",
        "OCCLUDED": "#7570b3", "RECEPTION_CANDIDATE": "#d95f02",
        "CONTROLLED": "#1b9e77", "PRESSURED_CONTROL": "#1b9e77",
    }
    if "scenario_context" in window:
        for scenario, color in scenario_colors.items():
            mask = window["scenario_context"].eq(scenario).to_numpy()
            if mask.any():
                ax.scatter(x[mask], np.full(mask.sum(), 0.18), s=14,
                           color=color, label=f"scenario: {scenario}")
    if "transition_anchor" in window:
        mask = window["transition_anchor"].fillna(False).to_numpy(bool)
        ax.scatter(x[mask], np.full(mask.sum(), 1.42), marker="v", s=55,
                   color="#d62728", label="transition anchor")
    if "ball_x" in window:
        missing = window["ball_x"].isna() | window["ball_y"].isna()
        for row in window.loc[missing].itertuples():
            ax.axvline(row.frame, color="#d95f02", alpha=0.08, linewidth=2)
    _plot_anchor_margin(ax, window)
    if ground_truth is not None:
        gt = ground_truth.reindex(window["frame"]).to_numpy()
        _plot_ground_truth_strip(ax, window["frame"], ground_truth, y=1.64)
        if pd.Series(gt).notna().sum() == 0:
            ax.text(0.99, 0.04, "ground truth unavailable for this frame range",
                    transform=ax.transAxes, ha="right", color="#a33", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#999"})
    if predicted_labels is not None:
        _plot_label_strip(ax, window["frame"], predicted_labels, y=1.94, legend_prefix="prediction")
        if ground_truth is not None:
            _plot_mismatch_windows(ax, window["frame"], ground_truth, predicted_labels)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=3, fontsize=8)
    return ax.figure, ax
