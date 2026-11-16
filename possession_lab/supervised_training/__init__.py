"""Isolated supervised H/A/D model training imported from Working File Mini.

This package intentionally does not replace the observation-first possession pipeline.
"""
from .schema import Match, H, A, D, CLASSES
from .model import fit, predict_proba, decode, evaluate, save, load

__all__ = ["Match", "H", "A", "D", "CLASSES", "fit", "predict_proba", "decode", "evaluate", "save", "load"]
