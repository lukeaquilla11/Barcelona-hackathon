"""Optional adapter from the existing 46-slot match loader to the isolated supervised model.

This is an opt-in diagnostic/evidence interface. It does not change or replace the
observation-first resolver, nor does it read ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .supervised_training.schema import Match as SupervisedMatch
from .supervised_training import model as supervised_model


def to_supervised_match(match, name="consolidated_match", fps=25.0):
    """Adapt possession.load_data.Match to the training model's metres-based schema."""
    length, width = match.pitch
    return SupervisedMatch(
        name=name,
        P=np.asarray(match.players_xy, dtype=np.float32),
        B=np.asarray(match.ball_xy, dtype=np.float32),
        team=np.asarray(match.track_team, dtype=np.int8),
        period=np.asarray(match.half, dtype=np.int32),
        labels=None,
        fps=float(fps), L=float(length), W=float(width),
    )


def to_supervised_match_from_frame_table(
    frame_table: pd.DataFrame,
    *,
    team_by_slot: dict[int, str],
    period: np.ndarray,
    pitch: tuple[float, float] = (105.3, 68.0),
    fps: float = 25.0,
    name: str = "consolidated_features",
):
    """Adapt the canonical engineered table to Kinson's feature schema.

    This is deliberately a feature-only bridge.  It preserves the current
    table's observed/George ball coordinates and does not accept labels.  The
    period array is supplied by the tactical timeline so Kinson's past/future
    windows cannot cross halftime.
    """
    slots = sorted(
        int(column[:-2])
        for column in frame_table.columns
        if column.endswith("_x")
        and column[:-2].isdigit()
        and f"{column[:-2]}_y" in frame_table.columns
        and int(column[:-2]) in team_by_slot
    )
    if not slots:
        raise ValueError("canonical frame table has no player x/y slots for supervised evidence")
    players = np.stack(
        [
            frame_table[[f"{slot}_x", f"{slot}_y"]].to_numpy(dtype=np.float32)
            for slot in slots
        ],
        axis=1,
    )
    ball = frame_table[["ball_x", "ball_y"]].to_numpy(dtype=np.float32)
    team = np.asarray(
        [0 if team_by_slot[slot] == "H" else 1 for slot in slots],
        dtype=np.int8,
    )
    return SupervisedMatch(
        name=name,
        P=players,
        B=ball,
        team=team,
        period=np.asarray(period, dtype=np.int32),
        labels=None,
        fps=float(fps),
        L=float(pitch[0]),
        W=float(pitch[1]),
    )


def predict_supervised_evidence(model_path, match, name="consolidated_match", fps=25.0):
    """Return frame-aligned H/A/D probabilities and decoded labels as a DataFrame."""
    clf, decoder, _ = supervised_model.load(str(model_path))
    adapted = to_supervised_match(match, name=name, fps=fps)
    probabilities = supervised_model.predict_proba(clf, adapted)
    decoded = supervised_model.decode(probabilities, adapted.blocks(), decoder, adapted.fps)
    return pd.DataFrame({
        "p_supervised_home": probabilities[:, 0],
        "p_supervised_away": probabilities[:, 1],
        "p_supervised_dead": probabilities[:, 2],
        "supervised_label": supervised_model.labels_to_str(decoded),
    }, index=pd.RangeIndex(len(probabilities), name="frame"))


def add_supervised_probability_evidence(
    features: pd.DataFrame,
    model_path,
    *,
    team_by_slot: dict[int, str],
    period: np.ndarray,
    pitch: tuple[float, float] = (105.3, 68.0),
    fps: float = 25.0,
    maximum_contribution: float = 0.0,
) -> pd.DataFrame:
    """Attach Kinson-style probabilities as bounded auxiliary evidence.

    The classifier is allowed to provide ``p_supervised_*`` observations, but
    its decoded Viterbi path is never copied into the possession labels.  A
    non-zero ``maximum_contribution`` adds only a capped signed H/A support to
    the existing transition-trust vector; it cannot create an anchor,
    suppress D/X, or bypass the interval resolver.
    """
    clf, decoder, _info = supervised_model.load(str(model_path))
    adapted = to_supervised_match_from_frame_table(
        features,
        team_by_slot=team_by_slot,
        period=period,
        pitch=pitch,
        fps=fps,
    )
    probabilities = supervised_model.predict_proba(clf, adapted)
    decoded = supervised_model.decode(probabilities, adapted.blocks(), decoder, adapted.fps)
    out = features.copy()
    out["p_supervised_home"] = probabilities[:, 0]
    out["p_supervised_away"] = probabilities[:, 1]
    out["p_supervised_dead"] = probabilities[:, 2]
    out["supervised_evidence_margin"] = probabilities[:, 0] - probabilities[:, 1]
    out["supervised_evidence_live_support"] = 1.0 - probabilities[:, 2]
    out["supervised_evidence_model"] = "KINSON_HGB_SILVER_AUXILIARY"
    out["supervised_evidence_decoded_used"] = False
    out["supervised_decoded_label"] = supervised_model.labels_to_str(decoded)
    out["supervised_evidence_maximum_contribution"] = float(maximum_contribution)

    # Keep the contribution small and require the classifier to prefer a live
    # H/A alternative over D. This is evidence weighting, not label transfer.
    margin = probabilities[:, 0] - probabilities[:, 1]
    live_support = np.clip(1.0 - probabilities[:, 2], 0.0, 1.0)
    signed = np.clip(margin, -1.0, 1.0) * live_support
    contribution = float(max(0.0, maximum_contribution)) * signed
    trust_h = pd.to_numeric(
        out.get("transition_trust_H", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    trust_a = pd.to_numeric(
        out.get("transition_trust_A", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    out["transition_kinson_support_H"] = np.maximum(contribution, 0.0)
    out["transition_kinson_support_A"] = np.maximum(-contribution, 0.0)
    out["transition_trust_H"] = np.clip(trust_h + np.maximum(contribution, 0.0), 0.0, 1.0)
    out["transition_trust_A"] = np.clip(trust_a + np.maximum(-contribution, 0.0), 0.0, 1.0)
    return out
