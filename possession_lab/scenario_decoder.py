"""Frame-level decoder for the inspectable scenario episode layer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from possession_lab.temporal_rules import load_rules


def _value(rules: dict[str, Any], *keys: str) -> Any:
    value: Any = rules
    for key in keys:
        value = value[key]
    return value


def collapse_short_team_spikes(labels: np.ndarray, max_spike_frames: int) -> tuple[np.ndarray, int]:
    """Replace short H/A runs surrounded by the same opposing context."""
    out = labels.astype(object, copy=True)
    collapsed = 0
    index = 0
    while index < len(out):
        end = index + 1
        while end < len(out) and out[end] == out[index]:
            end += 1
        run = out[index]
        before = out[index - 1] if index else None
        after = out[end] if end < len(out) else None
        if run in {"H", "A"} and before == after and before in {"H", "A"} and end - index <= max_spike_frames:
            out[index:end] = before
            collapsed += end - index
        index = end
    return out, collapsed


def decode_scenario_labels(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Decode scenario evidence into a valid frame/label sequence.

    This is deliberately deterministic. It is the rule-based comparator for
    the later HMM/HSMM models, not a learned sequence model.
    """
    rules = load_rules() if rules is None else rules
    n_frames = len(features)
    graded = features.get("graded", pd.Series(True, index=features.index)).to_numpy(bool)
    labels = np.full(n_frames, "X", dtype=object)
    closest = features.get("closest_team", pd.Series(None, index=features.index)).to_numpy(object)
    closest = np.where(pd.isna(closest), "X", closest)
    labels[graded] = closest[graded]
    set_piece = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)
    labels[graded & set_piece] = "D"

    # A confirmed transition is the only special evidence allowed to overwrite
    # a non-dead-ball team decision.
    transition = features.get("transition_anchor", pd.Series(False, index=features.index)).to_numpy(bool)
    anchor = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)
    labels[graded & transition & pd.notna(anchor)] = anchor[graded & transition & pd.notna(anchor)]

    labels, collapsed = collapse_short_team_spikes(
        labels,
        int(_value(rules, "decoder", "max_team_spike_frames")),
    )
    return labels, {"short_team_spike_frames_collapsed": collapsed}
