#!/usr/bin/env python3
"""Validate the custom multimodal Grad-CAM against external reference
implementations.

Sanity check: compares ``GradCAMExplainer.compute_real_gradcam`` (the custom
one, in ``multimodal.explainability.multimodal``) with the reference
implementation from the ``tf_explain`` package, a consolidated XAI library for
TF 2.x, and with the pyimagesearch formulation.

### Method

- The multimodal pipeline takes 2 inputs (``[image_batch, text_batch]``).
  Since ``tf_explain.GradCAM`` assumes a single-input model, a **wrapper**
  ``keras.Model`` embeds the text embedding as a constant in the graph and
  exposes only the image port.
- Both implementations run on the same (image, text) pair. Raw 7x7 heatmaps
  are pulled out of ``tf_explain`` through its internal API
  (``get_gradients_and_filters`` + ``generate_ponderated_output``) and resized
  to 224x224 to be compared with the custom output.
- Agreement metrics: Pearson, IoU over the top-20% pixels, RMSE.

### Output

``results/explanations/validation/validate_gradcam_<ts>/``:
- ``comparison.png``: panels (original | custom | reference | diff map)
- ``metrics.json``: Pearson, IoU, RMSE and the predictions

### Dependencies

- ``pip install tf-explain``
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.gridspec import GridSpec
from tensorflow import keras

from multimodal.common.paths import RESULTS_DIR
from multimodal.explainability.multimodal import (
    GradCAMExplainer,
    IMG_SIZE,
    MAX_TEXT_LENGTH,
    load_original_image,
    preprocess_image_path,
)


def build_image_only_wrapper(
    multimodal_model: keras.Model,
    fixed_text_embedding: np.ndarray,
) -> keras.Model:
    """Wrap the multimodal model with a frozen text input, exposing only the
    image port. Lets single-input tools such as tf_explain run on the 2-input
    pipeline.

    Args:
        multimodal_model: Keras Model taking [image, text] inputs.
        fixed_text_embedding: (T, D) — the single embedding frozen into the graph.

    Returns:
        Keras Model with a single image input (H, W, 3) and the same output.
    """
    T, D = fixed_text_embedding.shape
    img_input = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    batch_size = tf.shape(img_input)[0]
    text_const = tf.constant(
        fixed_text_embedding.astype(np.float32)[np.newaxis, :, :],
        dtype=tf.float32,
    )
    text_tensor = tf.broadcast_to(text_const, (batch_size, T, D))
    output = multimodal_model([img_input, text_tensor])
    return keras.Model(
        inputs=img_input, outputs=output, name="img_only_wrapper"
    )


def tf_explain_raw_heatmap(
    wrapper: keras.Model,
    image_batch: np.ndarray,
    class_index: int = 0,
    layer_name: str | None = None,
) -> np.ndarray:
    """Extract the raw heatmap (7x7, before resizing) from tf_explain.GradCAM.

    Uses the internal APIs ``get_gradients_and_filters`` and
    ``generate_ponderated_output``, which avoids the overlay post-processing that ``explain()`` applies.
    """
    from tf_explain.core.grad_cam import GradCAM

    explainer = GradCAM()
    outputs, grads = explainer.get_gradients_and_filters(
        wrapper,
        image_batch,
        class_index=class_index,
        layer_name=layer_name,
        use_guided_grads=True,
    )
    cams = explainer.generate_ponderated_output(outputs, grads)
    cam = cams[0].numpy() if hasattr(cams[0], "numpy") else np.asarray(cams[0])
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)
    cam_resized = cv2.resize(
        cam.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
    )
    return cam_resized


def heatmap_metrics(h1: np.ndarray, h2: np.ndarray) -> dict:
    """Agreement between two 2D heatmaps in [0, 1]: Pearson, IoU@20%, RMSE."""
    a = h1.flatten().astype(float)
    b = h2.flatten().astype(float)
    # Pearson, guarding against std=0.
    if a.std() < 1e-12 or b.std() < 1e-12:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(a, b)[0, 1])
    # IoU top-20%.
    q1 = np.quantile(a, 0.80)
    q2 = np.quantile(b, 0.80)
    mask1 = h1 >= q1
    mask2 = h2 >= q2
    intersect = float(np.logical_and(mask1, mask2).sum())
    union = float(np.logical_or(mask1, mask2).sum())
    iou = intersect / (union + 1e-12)
    # RMSE.
    rmse = float(np.sqrt(np.mean((h1 - h2) ** 2)))
    return {"pearson": pearson, "iou_top20": iou, "rmse": rmse}


def _overlay(original: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Standard overlay: resize + COLORMAP_JET + addWeighted 0.5/0.5."""
    h, w = original.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_uint8 = np.uint8(255 * np.clip(hm_resized, 0, 1))
    hm_colored = cv2.cvtColor(
        cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
    )
    return cv2.addWeighted(original, 0.5, hm_colored, 0.5, 0)


def _diff_overlay(
    original: np.ndarray, diff: np.ndarray
) -> np.ndarray:
    """Diff overlay with a divergent colormap. diff = |h_a - h_b|."""
    h, w = original.shape[:2]
    diff_resized = cv2.resize(
        diff.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )
    # Normalise to [0, 1] for display.
    if diff_resized.max() > 0:
        diff_resized = diff_resized / (diff_resized.max() + 1e-8)
    diff_uint8 = np.uint8(255 * np.clip(diff_resized, 0, 1))
    diff_colored = cv2.cvtColor(
        cv2.applyColorMap(diff_uint8, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB
    )
    return cv2.addWeighted(original, 0.4, diff_colored, 0.6, 0)


def render_figure(
    image_path: str,
    heatmaps: dict[str, np.ndarray],
    metrics_pairs: dict[str, dict],
    prediction: float,
    output_path: str,
    package_name: str = "",
    screen_id: str = "",
) -> None:
    """Build the figure, laying it out for the implementations available.

    Args:
        heatmaps: dict {impl_name: heatmap in [0,1]}. Always contains
                  'custom'; may contain 'tfexplain' and 'pyimagesearch'.
        metrics_pairs: dict {pair: {pearson, iou_top20, rmse}} — e.g.:
                  'custom_vs_tfexplain', 'custom_vs_pyimagesearch'.

    Layout:
        - Row 1: original screenshot + one panel per available implementation.
        - Row 2: one diff panel per metric pair.
        - Row 3: metadata bar.
    """
    original = load_original_image(image_path)
    if original is None:
        raise RuntimeError(f"Failed to load {image_path}")

    impl_names = [n for n in ("custom", "tfexplain", "pyimagesearch") if n in heatmaps]
    n_impls = len(impl_names)

    # Row 1: original + n_impls heatmaps, so n_impls+1 columns.
    # Row 2: diff pairs (custom vs each other implementation).
    diff_pairs = [
        ("custom", other)
        for other in impl_names
        if other != "custom"
    ]
    n_cols = max(n_impls + 1, len(diff_pairs))

    fig = plt.figure(figsize=(5 * n_cols, 11))
    gs = GridSpec(
        3, n_cols,
        height_ratios=[6, 6, 0.5],
        hspace=0.15, wspace=0.05,
    )

    impl_titles = {
        "custom": "Grad-CAM (custom)\nmultimodal.compute_real_gradcam",
        "tfexplain": "Grad-CAM (tf_explain)\nexternal reference",
        "pyimagesearch": "Grad-CAM (pyimagesearch)\nRosebrock 2020 — Guided Grad-CAM",
    }

    # Row 1: original + overlays
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(original)
    ax.set_title("Screenshot original", fontsize=11, fontweight="bold")
    ax.axis("off")
    for i, name in enumerate(impl_names):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(_overlay(original, heatmaps[name]))
        ax.set_title(impl_titles.get(name, name), fontsize=11, fontweight="bold")
        ax.axis("off")

    # Row 2: diff overlays (custom vs each other implementation)
    for i, (a_name, b_name) in enumerate(diff_pairs):
        ax = fig.add_subplot(gs[1, i])
        diff = np.abs(heatmaps[a_name] - heatmaps[b_name])
        ax.imshow(_diff_overlay(original, diff))
        pair_key = f"{a_name}_vs_{b_name}"
        m = metrics_pairs.get(pair_key, {})
        pearson = m.get("pearson", float("nan"))
        iou = m.get("iou_top20", float("nan"))
        rmse = m.get("rmse", float("nan"))
        ax.set_title(
            f"|Δ| ({a_name} − {b_name})\n"
            f"Pearson={pearson:.3f}  IoU@20={iou:.3f}  RMSE={rmse:.3f}",
            fontsize=10, fontweight="bold",
        )
        ax.axis("off")

    # Row 3: metadata bar
    ax_meta = fig.add_subplot(gs[2, :])
    ax_meta.axis("off")
    info_text = (
        f"App: {package_name}  |  Screen: {screen_id}  |  "
        f"Prediction: {prediction:.4f}  |  Impls: {', '.join(impl_names)}"
    )
    ax_meta.text(
        0.5, 0.5, info_text,
        ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5B8", edgecolor="gray"),
        transform=ax_meta.transAxes,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def validate(config: dict) -> None:
    """Entry point: config → one figure + metrics + JSON.

    Supports up to 3 Grad-CAM implementations: custom is always on, tf_explain
    and pyimagesearch are enabled by flags. The figure layout adapts to how
    many are available.
    """
    # --- Config ---
    model_path = config.get("multimodal_model_path")
    image_path = config.get("image_path")
    emb_cache = config.get("embedding_cache_path")
    emb_idx = int(config.get("embedding_idx", 0))
    layer_name = config.get("output_layer_name")  # None → auto
    pkg = config.get("package_name", "")
    sid = config.get("screen_id", "")
    use_tfexplain = bool(config.get("use_tf_explain", True))
    use_pyimagesearch = bool(config.get("use_pyimagesearch", True))

    for name, val in [("multimodal_model_path", model_path),
                      ("image_path", image_path),
                      ("embedding_cache_path", emb_cache)]:
        if not val:
            raise ValueError(f"Invalid config: '{name}' is required.")

    # --- Load model ---
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    print(f"  Inputs: {[t.shape for t in model.inputs]}")
    print(f"  Output: {model.outputs[0].shape}")

    # --- Load image ---
    print(f"Loading image: {image_path}")
    image = preprocess_image_path(image_path)
    if image is None:
        raise RuntimeError(f"Failed to preprocess {image_path}")
    image_batch = np.expand_dims(image, 0).astype(np.float32)

    # --- Load embedding ---
    print(f"Loading embedding: {emb_cache}[{emb_idx}]")
    embeddings = np.load(str(emb_cache), mmap_mode="r")
    text_emb = np.asarray(embeddings[emb_idx]).astype(np.float32)
    text_batch = np.expand_dims(text_emb, 0)
    print(f"  embedding shape: {text_emb.shape}")

    # --- Baseline prediction ---
    pred_raw = model.predict([image_batch, text_batch], verbose=0)
    prediction = float(np.asarray(pred_raw).ravel()[0])
    print(f"  prediction: {prediction:.4f}")

    # --- Image-only wrapper (used by tf_explain and pyimagesearch) ---
    wrapper = None
    if use_tfexplain or use_pyimagesearch:
        wrapper = build_image_only_wrapper(model, text_emb)

    # Heatmaps and stats per implementation.
    heatmaps: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}

    # --- Custom Grad-CAM (always on) ---
    print("\n[1] Grad-CAM custom (multimodal.compute_real_gradcam)...")
    gradcam_custom = GradCAMExplainer(model)
    heatmap_custom, _ = gradcam_custom.compute_real_gradcam(image_batch, text_batch)
    heatmaps["custom"] = heatmap_custom
    stats["custom"] = {
        "max": float(heatmap_custom.max()),
        "mean": float(heatmap_custom.mean()),
        "std": float(heatmap_custom.std()),
    }
    print(f"  shape={heatmap_custom.shape}  max={heatmap_custom.max():.3f}  mean={heatmap_custom.mean():.3f}")

    # --- tf_explain Grad-CAM (off by default) ---
    # Incompatible with this architecture: tf_explain 0.3.1 (Rosebrock-style)
    # assumes a flat single-input model. The wrapper exposes only the submodel
    # as an opaque "layer", so out_relu/Conv_1 and friends stay hidden from
    # outside. Pyimagesearch works around it via image_pooling.input in the
    # graph. The flag is kept for completeness, but defaults to off.
    if use_tfexplain:
        print(
            "\n[2] tf_explain is enabled but incompatible with nested backbones "
            "(MobileNetV2 inside the multimodal model). "
            "Skipping. Set 'use_tf_explain: false' to silence this message."
        )

    # --- pyimagesearch Grad-CAM (optional, no external deps) ---
    if use_pyimagesearch:
        print("\n[3] Grad-CAM pyimagesearch (Guided Grad-CAM, Rosebrock 2020)...")
        try:
            from multimodal.tools._gradcam_pyimagesearch import (
                GradCAMMultimodal,
            )
            # Use multimodal_model directly rather than the wrapper, so
            # find_target_layer() can walk the inner MobileNetV2 layers and
            # find the 4D convs.
            pyim = GradCAMMultimodal(
                model=model, classIdx=0, layerName=layer_name
            )
            heatmap_pyim = pyim.compute_heatmap_float(image_batch, text_batch)
            heatmaps["pyimagesearch"] = heatmap_pyim
            effective = pyim.effective_layer_name or pyim.layerName
            stats["pyimagesearch"] = {
                "max": float(heatmap_pyim.max()),
                "mean": float(heatmap_pyim.mean()),
                "std": float(heatmap_pyim.std()),
                "target_layer": pyim.layerName,
                "effective_tensor": effective,
            }
            print(f"  shape={heatmap_pyim.shape}  max={heatmap_pyim.max():.3f}  mean={heatmap_pyim.mean():.3f}  layer={effective}")
        except Exception as e:
            print(f"\n[3] pyimagesearch failed: {e}")

    # --- Paired metrics (custom is always the reference) ---
    print("\nAgreement metrics:")
    metrics_pairs: dict[str, dict] = {}
    for name, hm in heatmaps.items():
        if name == "custom":
            continue
        pair_key = f"custom_vs_{name}"
        m = heatmap_metrics(heatmap_custom, hm)
        metrics_pairs[pair_key] = m
        print(f"  {pair_key}:")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    # With 2 other implementations, add the pair between them as well.
    other_names = [n for n in heatmaps if n != "custom"]
    if len(other_names) == 2:
        a, b = other_names
        m = heatmap_metrics(heatmaps[a], heatmaps[b])
        pair_key = f"{a}_vs_{b}"
        metrics_pairs[pair_key] = m
        print(f"  {pair_key} (triangulation):")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    # --- Output dir ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / "explanations" / "validation" / f"validate_gradcam_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Save metrics JSON ---
    results = {
        "package_name": pkg,
        "screen_id": sid,
        "prediction": prediction,
        "embedding_idx": emb_idx,
        "image_path": image_path,
        "implementations_run": list(heatmaps.keys()),
        "metrics_pairs": metrics_pairs,
        "stats": stats,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n→ {metrics_path}")

    # --- Save figure ---
    figure_path = out_dir / "comparison.png"
    render_figure(
        image_path=image_path,
        heatmaps=heatmaps,
        metrics_pairs=metrics_pairs,
        prediction=prediction,
        output_path=str(figure_path),
        package_name=pkg,
        screen_id=sid,
    )
    print(f"→ {figure_path}")
