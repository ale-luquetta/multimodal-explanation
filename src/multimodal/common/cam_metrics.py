"""Faithfulness metrics for Grad-CAM heatmaps.

Adapted (Keras-friendly reimplementation) from `pytorch-grad-cam`:
  - cam_mult_confidence: drop in confidence after multiplying input by (1 - CAM)
  - road_score: average drop across percentiles when removing top-p% most relevant pixels

Both metrics quantify how *faithful* the Grad-CAM is to the model's decision —
higher scores mean the highlighted regions actually drive the prediction.

predict_fn must accept a batched image array (B, H, W, 3) and return shape
(B, 1) sigmoid probabilities. Closures handle multimodal cases (text fixed).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

PredictFn = Callable[[np.ndarray], np.ndarray]


def cam_mult_confidence(image: np.ndarray, heatmap: np.ndarray, predict_fn: PredictFn) -> float:
    """Confidence drop after masking the image by (1 - heatmap).

    Args:
        image: (H, W, 3) float in [0, 1].
        heatmap: (H, W) float in [0, 1] — same H, W as image.
        predict_fn: function (B, H, W, 3) -> (B, 1) probabilities.

    Returns:
        |orig_prob - masked_prob|. Larger = more faithful CAM.
    """
    if heatmap.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"heatmap shape {heatmap.shape[:2]} must match image {image.shape[:2]}"
        )
    masked = image * (1.0 - heatmap[..., None])
    orig = float(predict_fn(image[None])[0, 0])
    after = float(predict_fn(masked[None])[0, 0])
    return abs(orig - after)


def road_score(
    image: np.ndarray,
    heatmap: np.ndarray,
    predict_fn: PredictFn,
    percentiles: Sequence[int] = (20, 40, 60, 80),
) -> dict[str, float]:
    """ROAD-style faithfulness: remove top-p% most relevant pixels and measure prediction drop.

    Pixels above the percentile threshold are replaced by the image's mean (a
    simple debias choice). Returns the per-percentile drops plus their mean.

    Args:
        image: (H, W, 3) float in [0, 1].
        heatmap: (H, W) float in [0, 1].
        predict_fn: function (B, H, W, 3) -> (B, 1).
        percentiles: which top-p% to remove. Defaults to [20, 40, 60, 80].

    Returns:
        Dict with `mean` and `per_percentile` (list of floats aligned with `percentiles`).
    """
    if heatmap.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"heatmap shape {heatmap.shape[:2]} must match image {image.shape[:2]}"
        )

    flat = heatmap.flatten()
    orig = float(predict_fn(image[None])[0, 0])
    fill_value = float(image.mean())

    per_percentile: list[float] = []
    for p in percentiles:
        threshold = float(np.percentile(flat, 100 - p))
        mask = heatmap >= threshold
        masked = image.copy()
        masked[mask] = fill_value
        after = float(predict_fn(masked[None])[0, 0])
        per_percentile.append(abs(orig - after))

    return {
        "mean": float(np.mean(per_percentile)) if per_percentile else 0.0,
        "per_percentile": per_percentile,
        "percentiles": list(percentiles),
    }
