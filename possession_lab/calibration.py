"""Label-separated calibration helpers for evidence and transition models.

The main pipeline is deliberately hand-specified.  This module contains
optional experiments that require an explicit calibration set and must never
silently consume the public evaluation labels.

Implemented experiments:

* temperature scaling by maximum likelihood for a frame-level logit vector;
* Dirichlet-smoothed transition probabilities from a trusted state sequence.

The second experiment is the same mathematical idea used by Haiqal's
``BayesianAnchorHMM``: it estimates a transition matrix from algorithmic
anchor states, not from public ground truth.  It is a transition-prior
experiment, not calibration of the rule evidence probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DEFAULT_CLASSES = ("H", "A", "D", "X")


def _as_class_indices(labels: Sequence[str] | np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Convert string or integer labels to validated class indices."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if np.issubdtype(values.dtype, np.integer):
        indices = values.astype(int, copy=False)
    else:
        lookup = {name: index for index, name in enumerate(classes)}
        try:
            indices = np.asarray([lookup[str(value)] for value in values], dtype=int)
        except KeyError as exc:
            raise ValueError(f"unknown class label: {exc.args[0]!r}") from exc
    if np.any(indices < 0) or np.any(indices >= len(classes)):
        raise ValueError("label index is outside the class range")
    return indices


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert logits to probabilities with numerically stable temperature scaling."""
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2:
        raise ValueError("logits must have shape (n_samples, n_classes)")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = values / float(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)


def multiclass_log_loss(
    probabilities: np.ndarray,
    labels: Sequence[str] | np.ndarray,
    *,
    classes: Sequence[str] | None = None,
) -> float:
    """Return mean negative log likelihood for a probability matrix."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    label_classes = tuple(classes) if classes is not None else tuple(map(str, range(values.shape[1])))
    indices = _as_class_indices(labels, label_classes)
    if len(indices) != len(values):
        raise ValueError("probabilities and labels have different lengths")
    return float(-np.log(np.maximum(values[np.arange(len(values)), indices], 1e-12)).mean())


@dataclass(frozen=True)
class TemperatureCalibrator:
    """A fitted one-parameter MLE calibration layer."""

    temperature: float
    classes: tuple[str, ...] = DEFAULT_CLASSES

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature to new evidence logits."""
        values = np.asarray(logits)
        if values.shape[-1] != len(self.classes):
            raise ValueError("logit class count does not match calibrator classes")
        return softmax(values, self.temperature)


def fit_temperature_mle(
    logits: np.ndarray,
    labels: Sequence[str] | np.ndarray,
    *,
    classes: Sequence[str] = DEFAULT_CLASSES,
    mask: np.ndarray | None = None,
    minimum_temperature: float = 0.05,
    maximum_temperature: float = 20.0,
    iterations: int = 80,
) -> TemperatureCalibrator:
    """Fit temperature scaling by minimizing held-out negative log likelihood.

    ``mask`` must identify a declared calibration split.  This function does
    not know whether labels came from public ground truth, reviewed data, or
    another source; callers are responsible for keeping calibration and final
    evaluation frames disjoint.
    """
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(classes):
        raise ValueError("logits must have shape (n_samples, len(classes))")
    indices = _as_class_indices(labels, classes)
    if len(indices) != len(values):
        raise ValueError("logits and labels have different lengths")
    selected = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if len(selected) != len(values):
        raise ValueError("calibration mask has a different length")
    if not selected.any():
        raise ValueError("calibration mask selects no samples")
    if minimum_temperature <= 0 or maximum_temperature <= minimum_temperature:
        raise ValueError("temperature bounds are invalid")

    calibration_logits = values[selected]
    calibration_labels = indices[selected]

    def objective(log_temperature: float) -> float:
        probabilities = softmax(calibration_logits, float(np.exp(log_temperature)))
        return float(-np.log(np.maximum(probabilities[np.arange(len(probabilities)), calibration_labels], 1e-12)).mean())

    # Golden-section search in log-temperature keeps the search positive and
    # is sufficient for the one-dimensional convex temperature objective.
    left = float(np.log(minimum_temperature))
    right = float(np.log(maximum_temperature))
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(int(iterations)):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = objective(x2)
    temperature = float(np.exp((left + right) / 2.0))
    return TemperatureCalibrator(temperature=temperature, classes=tuple(classes))


def fit_dirichlet_transition_posterior(
    states: Iterable[str],
    *,
    classes: Sequence[str] = DEFAULT_CLASSES,
    alpha: float = 1.0,
) -> np.ndarray:
    """Estimate a transition matrix using a Dirichlet posterior mean.

    Consecutive valid states contribute counts.  Missing/unknown states break
    the transition, so an occlusion is not silently treated as a transition.
    With a symmetric Dirichlet prior ``alpha``, each row is:

    ``P(next=j | current=i) = (count[i,j] + alpha) / (row_count[i] + K*alpha)``.

    This is the Bayesian transition-prior idea used by Haiqal's diagnostic
    model, generalized so our trusted anchor/ledger sequence can be supplied.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    names = tuple(classes)
    lookup = {name: index for index, name in enumerate(names)}
    counts = np.full((len(names), len(names)), float(alpha), dtype=float)
    previous: int | None = None
    for state in states:
        current = lookup.get(str(state))
        if current is None:
            previous = None
            continue
        if previous is not None:
            counts[previous, current] += 1.0
        previous = current
    return counts / counts.sum(axis=1, keepdims=True)


def multiclass_brier_score(
    probabilities: np.ndarray,
    labels: Sequence[str] | np.ndarray,
    *,
    classes: Sequence[str] | None = None,
) -> float:
    """Return the multiclass Brier score for calibration evaluation."""
    values = np.asarray(probabilities, dtype=float)
    label_classes = tuple(classes) if classes is not None else tuple(map(str, range(values.shape[1])))
    indices = _as_class_indices(labels, label_classes)
    target = np.zeros_like(values)
    target[np.arange(len(values)), indices] = 1.0
    return float(np.mean(np.sum((values - target) ** 2, axis=1)))
