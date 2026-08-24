#!/usr/bin/env python3
"""Multimodal explainability — post-fusion Grad-CAM (Conv2D after the text broadcast).

Builds on ``multimodal.explainability.multimodal_v2`` without touching it,
moving the Grad-CAM target layer to the Conv2D that sees both modalities:

- ``image_pooling.input`` is the output of the last MobileNetV2 conv, before the
  GlobalAveragePooling. That 7x7x1280 map depends on the image only, since the
  text enters the dense head, after the GAP.
- ``fusion_conv2d.output`` is the output of the post-fusion Conv2D. That 7x7x256
  map already carries channels derived from the image and from the text, after
  the spatial broadcast, so the heatmap reflects *multimodal* saliency rather
  than visual saliency alone.

The faithfulness, OBI, per-review analysis and SHAP logic is reused by
monkey-patching. What this module adds is the injection of
``GradCAMExplainerV3``, the redirection of the outputs to
``results/explanations/multimodal_v3/`` and the split of the per-review figure
into two files (``_visual`` and ``_texto``) through ``MultimodalExplanationV3``,
so the modalities can be cited as independent figures.
"""

from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.transforms import Bbox

from multimodal.common.paths import (
    RESULTS_EXPLANATIONS_MULTIMODAL_V3,
    RESULTS_TRAINING_MULTIMODAL_V3,
)
from multimodal.explainability.multimodal import GradCAMExplainer, ModelSelector
from multimodal.explainability.multimodal_v2 import MultimodalExplanationV2


class GradCAMExplainerV3(GradCAMExplainer):
    """Grad-CAM targeting the post-fusion Conv2D (``fusion_conv2d``).

    Overrides ``_find_mobilenet_and_conv`` and ``_build_grad_model`` from the
    base class to point at the multimodal fusion output instead of the raw
    MobileNetV2 feature map. ``compute_real_gradcam``, ``compute_saliency`` and
    the smoothed variant keep working unchanged, since they operate on the
    rebuilt ``grad_model``.
    """

    TARGET_CONV_LAYER_NAME = "fusion_conv2d"

    def _find_mobilenet_and_conv(self):
        """The Grad-CAM target is ``fusion_conv2d``, not a MobileNetV2 layer.

        MobileNetV2 is still located, as a diagnostic reference, but
        ``target_conv_name`` points at the post-fusion Conv2D.
        """
        # Locate the nested MobileNetV2 for diagnostics only.
        self.mobilenet_layer = None
        for layer in self.model.layers:
            if "mobilenet" in layer.name.lower():
                self.mobilenet_layer = layer
                break

        # Confirm the post-fusion Conv2D exists in the model.
        try:
            self.model.get_layer(self.TARGET_CONV_LAYER_NAME)
        except ValueError as e:
            raise ValueError(
                f"Layer '{self.TARGET_CONV_LAYER_NAME}' not found in the model. "
                f"GradCAMExplainerV3 expects the spatial-fusion architecture. "
                f"Use the base GradCAMExplainer for models without it."
            ) from e

        self.target_conv_name = self.TARGET_CONV_LAYER_NAME

        print("Grad-CAM configured (spatial fusion):")
        if self.mobilenet_layer is not None:
            print(f"  - MobileNetV2 layer: {self.mobilenet_layer.name} (reference only)")
        print(f"  - Target conv layer:  {self.target_conv_name} (post-fusion)")

    def _build_grad_model(self):
        """Build the grad_model returning (fusion_conv2d.output, model.output)."""
        try:
            fusion_conv = self.model.get_layer(self.TARGET_CONV_LAYER_NAME)
            conv_output_tensor = fusion_conv.output

            self.grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[conv_output_tensor, self.model.output],
            )

            print(f"  - Grad model built from the output of {self.TARGET_CONV_LAYER_NAME}")
            print(f"  - Feature maps shape: {conv_output_tensor.shape}")

        except Exception as e:
            print(f"ERROR building the grad_model: {e}")
            print("Saliency map remains available as a fallback.")
            self.grad_model = None

    def compute_real_gradcam(self, image_array, text_embedding):
        """Class-aware Grad-CAM: flips the loss sign when the predicted class is bad.

        The base class uses ``loss = prediction[:, 0]`` (= P(good)) for every
        app. On an app with P(good) close to 0, so a predicted class of bad, the
        gradient points in the direction that would *raise* P(good), while the
        relevant activations are the ones supporting the bad prediction. The
        result is that ``sum(alpha*A)`` comes out mostly negative, the final
        ReLU collapses the heatmap and the method falls back to SmoothGrad.

        Following the original Grad-CAM prescription (Selvaraju et al. 2017),
        the score of the class of interest is used instead. On a binary sigmoid
        head that means:
          - predicted good (pred >= 0.5): loss =  prediction[:, 0]
          - predicted bad  (pred <  0.5): loss = -prediction[:, 0]

        The heatmap then reflects "evidence for the predicted class" rather than
        "evidence for good", and Grad-CAM behaves symmetrically on both sides of
        the threshold, which removes the systematic collapse on bad apps.

        Everything else (7x7 to 224x224 resize, blur, normalisation, SmoothGrad
        fallback on genuinely null gradients) is inherited unchanged.
        """
        if self.grad_model is None:
            print("WARNING: grad_model unavailable, using the SmoothGrad fallback")
            return self._compute_saliency_fallback(image_array, text_embedding)

        img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        txt_tensor = tf.convert_to_tensor(text_embedding, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, prediction = self.grad_model(
                [img_tensor, txt_tensor], training=False
            )
            # Pick the gradient direction from the predicted class.
            # float(prediction[0, 0]) reads the scalar eagerly, which does not
            # break the tape: it keeps tracking the `prediction` tensor itself.
            pred_value = float(prediction[0, 0])
            if pred_value < 0.5:
                loss = -prediction[:, 0]
            else:
                loss = prediction[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("WARNING: null gradients on the feature maps, using the SmoothGrad fallback")
            return self._compute_saliency_fallback(image_array, text_embedding)

        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        conv_out = conv_outputs[0]
        weights = pooled_grads[0]

        heatmap = tf.reduce_sum(conv_out * weights, axis=-1)
        heatmap = tf.nn.relu(heatmap)

        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / (max_val + 1e-8)
        else:
            # Genuinely null heatmap even after the flip; rare, falls back.
            print("WARNING: heatmap collapsed even with the class-aware loss, using the SmoothGrad fallback")
            return self._compute_saliency_fallback(image_array, text_embedding)

        heatmap_np = heatmap.numpy()
        heatmap_resized = cv2.resize(
            heatmap_np, (image_array.shape[2], image_array.shape[1]),
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (9, 9), 0)
        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / (heatmap_resized.max() + 1e-8)

        return heatmap_resized, float(prediction[0, 0])


class ModelSelectorV3(ModelSelector):
    """ModelSelector listing only runs under ``results/training/multimodal_v3/``.

    Keeps the picker from offering models without ``fusion_conv2d``, whose
    selection would break GradCAMExplainerV3.
    """

    def __init__(self, base_paths=None):
        if base_paths is None:
            base_paths = [str(RESULTS_TRAINING_MULTIMODAL_V3)]
        super().__init__(base_paths=base_paths)


class MultimodalExplanationV3(MultimodalExplanationV2):
    """Same figure as the base class, written to two files instead of one.

    The base class builds a single PNG per (screen, review) with both
    modalities stacked. Image and text are needed as independent figures, so
    the layout is reused in full and only the moment of saving is intercepted:
    two regions of the same figure are written out.

        {base}_visual.png  ->  Screenshot | Grad-CAM | SHAP Visual + metadata
        {base}_texto.png   ->  Review text | Bahdanau | SHAP Textual

    No line of ``multimodal_v2`` is changed and the layout is not duplicated.
    """

    #: Axis indices in ``_render_per_review_figure``, in creation order
    #: (gs[0,0], gs[0,1], gs[0,2], gs[1,0], gs[1,1], gs[1,2], gs[2,:]).
    #: The order is stable but not a contract: reordering the ``add_subplot``
    #: calls in the base class means updating these indices too.
    VISUAL_AXES = (0, 1, 2)
    TEXTUAL_AXES = (3, 4, 5)
    META_AXIS = 6

    #: Vertical gap, as a fraction of the figure height, between the visual row
    #: and the repositioned metadata bar.
    META_GAP = 0.015

    def _render_per_review_figure(self, *args, **kwargs) -> None:
        original_savefig = plt.savefig

        def _split_savefig(out_path, **kw):
            fig = plt.gcf()
            axs = fig.get_axes()
            if len(axs) <= self.META_AXIS:
                # Unexpected layout: keep the base class behaviour.
                original_savefig(out_path, **kw)
                return

            def _region(indices):
                fig.canvas.draw()  # a valid renderer for get_tightbbox
                renderer = fig.canvas.get_renderer()
                bb = Bbox.union(
                    [axs[i].get_tightbbox(renderer) for i in indices]
                )
                inv = fig.dpi_scale_trans.inverted()
                return bb.transformed(inv).expanded(1.03, 1.03)

            dpi = kw.get("dpi", 150)
            base, ext = os.path.splitext(out_path)

            # The metadata bar sits below the text row. Cropping "visual row +
            # bar" in one go would drag the text row along, since Bbox.union
            # returns the rectangle enclosing everything. So the bar is moved
            # up against the visual row before the crop and put back right
            # after.
            meta_ax = axs[self.META_AXIS]
            meta_pos = meta_ax.get_position()
            visual_y0 = min(axs[i].get_position().y0 for i in self.VISUAL_AXES)
            meta_ax.set_position([
                meta_pos.x0,
                visual_y0 - meta_pos.height - self.META_GAP,
                meta_pos.width,
                meta_pos.height,
            ])
            # The repositioned bar overlaps the band where the text-row titles
            # are drawn, so those axes are hidden while saving to keep a slice
            # of title from leaking into the footer.
            for i in self.TEXTUAL_AXES:
                axs[i].set_visible(False)
            try:
                fig.savefig(
                    f"{base}_visual{ext}", dpi=dpi,
                    bbox_inches=_region(
                        tuple(self.VISUAL_AXES) + (self.META_AXIS,)
                    ),
                )
            finally:
                meta_ax.set_position(meta_pos)
                for i in self.TEXTUAL_AXES:
                    axs[i].set_visible(True)

            fig.savefig(
                f"{base}_texto{ext}", dpi=dpi,
                bbox_inches=_region(self.TEXTUAL_AXES),
            )

        plt.savefig = _split_savefig
        try:
            super()._render_per_review_figure(*args, **kwargs)
        finally:
            plt.savefig = original_savefig


def explain(config: dict) -> None:
    """V3 explainability entry point.

    Strategy: delegate to ``v2.explain(config)`` with three extra patches applied
    before the call:

    1. ``v1.GradCAMExplainer`` -> ``GradCAMExplainerV3`` (post-fusion target).
    2. ``v1.ModelSelector`` -> ``ModelSelectorV3``, whose picker lists only V3
       runs, so models without ``fusion_conv2d`` cannot be picked by accident.
    3. ``v2.RESULTS_EXPLANATIONS_MULTIMODAL_V2`` -> ``RESULTS_EXPLANATIONS_MULTIMODAL_V3``,
       so that when ``v2.explain`` redirects ``v1.RESULTS_EXPLANATIONS_MULTIMODAL``
       it lands on the V3 path.

    Every patch is restored in the ``finally``, so no state leaks between
    sequential runs.
    """
    import multimodal.explainability.multimodal as v1
    import multimodal.explainability.multimodal_v2 as v2

    saved_v1_gradcam_cls = v1.GradCAMExplainer
    saved_v1_selector_cls = v1.ModelSelector
    saved_v2_results_path = v2.RESULTS_EXPLANATIONS_MULTIMODAL_V2
    saved_v2_explanation_cls = v2.MultimodalExplanationV2

    v1.GradCAMExplainer = GradCAMExplainerV3
    v1.ModelSelector = ModelSelectorV3
    v2.RESULTS_EXPLANATIONS_MULTIMODAL_V2 = RESULTS_EXPLANATIONS_MULTIMODAL_V3
    # v2.explain reads this name when injecting the class into
    # v1.MultimodalExplanation, so swapping it here yields the split figure.
    v2.MultimodalExplanationV2 = MultimodalExplanationV3

    print("=" * 70)
    print("MULTIMODAL EXPLAINABILITY MODULE")
    print("Post-fusion Grad-CAM (fusion_conv2d) over the spatial-fusion model")
    print(f"Outputs in: {RESULTS_EXPLANATIONS_MULTIMODAL_V3}/")
    print("=" * 70)

    try:
        v2.explain(config)
    finally:
        v1.GradCAMExplainer = saved_v1_gradcam_cls
        v1.ModelSelector = saved_v1_selector_cls
        v2.RESULTS_EXPLANATIONS_MULTIMODAL_V2 = saved_v2_results_path
        v2.MultimodalExplanationV2 = saved_v2_explanation_cls
