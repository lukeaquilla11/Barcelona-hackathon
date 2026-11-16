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
