"""Object-level ablation primitives for XAI faithfulness.

Used by ``multimodal.explainability.multimodal_v2`` and
``multimodal.explainability.unimodal_v2`` to measure ΔP_img: the absolute drop
in predicted probability when the top-K salient UI components are zeroed out
(ObEy-style top-K object deletion).

Primitive style mirrors ``cam_metrics.py``: small, pure, side-effect-free.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def zero_image_regions(
    img: np.ndarray,
    bounds_list: Sequence[Sequence[int]],
    fill: float | None = None,
) -> np.ndarray:
    """Return a copy of ``img`` with all regions in ``bounds_list`` zeroed.

    Args:
        img: (H, W, 3) float array, typically in [0, 1].
        bounds_list: iterable of (x0, y0, x1, y1) tuples.
        fill: replacement value; defaults to ``img.mean()`` computed once from
            the original image (so that sequential zeroing doesn't drift as the
            running mean shifts). ROAD-style debias choice.

    Returns:
        Perturbed image with the same dtype as input.
    """
    if fill is None:
        fill = float(img.mean())
    out = img.copy()
    h, w = out.shape[:2]
    for bounds in bounds_list:
        x0, y0, x1, y1 = [int(v) for v in bounds]
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = fill
    return out


def top_k_token_indices(
    attention_weights: np.ndarray,
    k: int,
    exclude: Sequence[int] | None = None,
) -> list[int]:
    """Return indices of the top-k attention weights, excluding special tokens.

    Args:
        attention_weights: 1D array of per-token weights (e.g. Bahdanau, shape (128,)).
        k: number of indices to return. Clamped to available valid positions.
        exclude: token positions to skip (e.g. CLS/SEP/PAD indices).

    Returns:
        List of at most ``k`` integer indices, sorted by descending weight.
    """
    weights = np.asarray(attention_weights).ravel().astype(float)
    exclude_set = set(int(i) for i in (exclude or []))
    candidates = [i for i in range(weights.shape[0]) if i not in exclude_set]
    if not candidates or k <= 0:
        return []
    candidates.sort(key=lambda i: weights[i], reverse=True)
    return candidates[: min(k, len(candidates))]


def compute_delta_p(
    image: np.ndarray,
    text_embedding: np.ndarray,
    bounds_list: Sequence[Sequence[int]],
    model,
) -> dict:
    """Compute ΔP_img for the multimodal model via 2 forward passes.

    Both passes carry the same (unperturbed) text embedding, so the only
    difference between them is the visual ablation.

    Args:
        image: (H, W, 3) float [0, 1].
        text_embedding: (T, D) — e.g. (128, 768).
        bounds_list: iterable of (x0, y0, x1, y1) — top-K image regions
            zeroed together for ΔP_img (ObEy-style top-K object deletion).
        model: tf.keras.Model accepting [image_batch, text_batch].

    Returns:
        dict with:
          - p_orig, p_img: raw sigmoid probabilities
          - delta_p_img: absolute drop vs p_orig
    """
    img_ablated = zero_image_regions(image, bounds_list)

    batch_images = np.stack([image, img_ablated], axis=0)
    batch_texts = np.stack([text_embedding, text_embedding], axis=0)
    preds = model.predict([batch_images, batch_texts], verbose=0)
    preds = np.asarray(preds).reshape(-1)

    p_orig = float(preds[0])
    p_img = float(preds[1])
    delta_p_img = abs(p_orig - p_img)

    return {
        "p_orig": p_orig,
        "p_img": p_img,
        "delta_p_img": delta_p_img,
    }


def compute_obi_delta_p(
    image: np.ndarray,
    bounds_list: Sequence[Sequence[int]],
    model,
) -> dict:
    """Single-input counterpart to ``compute_delta_p`` for unimodal models.

    Performs the ObEy-style ablation of top-K UI components and returns the
    absolute drop in confidence. Used by
    ``multimodal.explainability.unimodal_v2`` to produce ΔP_img comparable
    to the multimodal pipeline (same bounds semantics, same fill policy).

    Unimodal models accept a single input (``image_batch``), not the
    ``[image_batch, text_batch]`` list expected by the multimodal fusion —
    so this function calls ``model.predict(batch)`` directly.

    Args:
        image: (H, W, 3) float [0, 1].
        bounds_list: iterable of (x0, y0, x1, y1) — top-K image regions
            zeroed together for ΔP_img.
        model: ``tf.keras.Model`` accepting a single image input.

    Returns:
        dict with:
          - p_orig, p_img: raw sigmoid probabilities
          - delta_p_img: absolute drop vs p_orig
    """
    img_ablated = zero_image_regions(image, bounds_list)
    batch = np.stack([image, img_ablated], axis=0)
    preds = model.predict(batch, verbose=0)
    preds = np.asarray(preds).reshape(-1)

    p_orig = float(preds[0])
    p_img = float(preds[1])
    delta_p_img = abs(p_orig - p_img)

    return {
        "p_orig": p_orig,
        "p_img": p_img,
        "delta_p_img": delta_p_img,
    }
