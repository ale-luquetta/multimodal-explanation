"""Threshold optimization utilities for binary classification.

Implements Youden's J statistic: J(t) = TPR(t) - FPR(t). The optimal threshold
maximizes this on the ROC curve. Standard when classes are balanced (our case
after downsampling) and when FP/FN have equivalent cost.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def youden_optimal_threshold(y_true, y_proba) -> float:
    """Return the threshold t that maximizes TPR(t) - FPR(t) on the ROC curve.

    Args:
        y_true: (N,) binary labels (0/1).
        y_proba: (N,) predicted probabilities in [0, 1].

    Returns:
        Optimal threshold in [0, 1].
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba).astype(float).ravel()
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    thr = float(thresholds[idx])
    return float(np.clip(thr, 0.0, 1.0))
