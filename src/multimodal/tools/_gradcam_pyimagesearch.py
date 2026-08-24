#!/usr/bin/env python3
"""Grad-CAM reference implementation (Rosebrock 2020, pyimagesearch).

Source: https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/

Kept as an **external reference** for validating the custom Grad-CAM in
``multimodal.explainability.multimodal``. Differences from it:

- **Guided Grad-CAM variant**: masks ``(convOutputs > 0) × (grads > 0)``
  before weighting the feature maps, instead of a ``ReLU`` after the sum.
- **Min-max normalization**: `(h - min) / (max - min + eps)` keeps the
  structure even without ReLU.
- **Returns uint8** [0, 255]; ``compute_heatmap_float`` is provided to match
  the float [0, 1] range the metrics use.
- **Single-input model**: needs a wrapper to run on multimodal pipelines.

Minimal changes over the original code:
- Added ``compute_heatmap_float()``, returning a float32 ``np.ndarray`` in
  [0, 1] for numeric comparison with other implementations.
- The ``overlay_heatmap`` signature is preserved.
"""

import cv2
import numpy as np
import tensorflow as tf


class GradCAM:
    """Grad-CAM (Rosebrock 2020) — reference implementation."""

    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        """Find the last layer with a 4D output, assumed to be convolutional."""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        """Heatmap as uint8 [0, 255] — the reference code's original interface."""
        gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layerName).output,
                self.model.output,
            ],
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)

        # Guided gradients: positive features AND positive grads only.
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # Weights = mean of the guided gradients per channel.
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # Resize to the image size.
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # Min-max normalization to [0, 1], then scaled to uint8 [0, 255].
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        return heatmap

    def compute_heatmap_float(self, image, eps=1e-8):
        """Heatmap as float32 [0, 1], the range used for comparison, matching
        what the custom implementation returns.
        """
        return self.compute_heatmap(image, eps=eps).astype(np.float32) / 255.0

    def overlay_heatmap(
        self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS
    ):
        """Overlay as in the original code (VIRIDIS colormap)."""
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return (heatmap, output)


class GradCAMMultimodal(GradCAM):
    """Guided Grad-CAM variant for models with 2 inputs ``[image, text]`` and a
    nested backbone (MobileNetV2 carrying its own Input).

    ### Problem 1 — ``find_target_layer`` fails through a wrapper

    The original pyimagesearch code assumes a single-input model and walks
    ``model.layers`` in ``find_target_layer()``. Wrapping the multimodal model
    in an image-only wrapper makes the inner layers opaque.

    Fix: pass the ``multimodal_model`` directly.

    ### Problem 2 — graph disconnected when reaching an inner conv

    Even with the model passed directly, ``model.get_layer('Conv1').output``
    belongs to the inner graph of MobileNetV2, which has its own nested Input,
    and is disconnected from the outer graph ``self.model.inputs``. Building a
    ``tf.keras.Model`` from those tensors raises
    "Graph disconnected: cannot obtain value for tensor ... at layer 'Conv1'".

    Fix, the same one the custom ``multimodal.compute_real_gradcam`` uses: take
    the ``input`` of a top-level layer of the main model that receives the
    backbone output. Here that is ``image_pooling``
    (``GlobalAveragePooling2D``): its ``.input`` is the backbone feature-map
    tensor **already connected to the outer graph**. Grad-CAM is ranked over
    that tensor, typically shaped ``(1, 7, 7, 1280)``.

    When ``layerName`` is given explicitly and exists at top level, that choice
    is honoured; otherwise the ``image_pooling`` fallback applies.
    """

    # Top-level "hook" layer whose ``.input`` is the backbone feature-map
    # tensor connected to the outer graph.
    DEFAULT_TOP_LEVEL_LAYER = "image_pooling"

    # Known inner MobileNetV2 layer names matching the last conv tensor; all
    # equivalent to ``image_pooling.input`` in the outer graph. When one of
    # these canonical names is passed, it is aliased transparently to avoid
    # "Graph disconnected".
    MOBILENET_FINAL_CONV_ALIASES = {"out_relu", "Conv_1_bn", "Conv_1"}

    def __init__(self, model, classIdx, layerName=None):
        # super().__init__ is skipped on purpose: find_target_layer would look
        # for a 4D conv in model.layers, which on a nested backbone returns
        # submodel layers whose graph is disconnected.
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName or self.DEFAULT_TOP_LEVEL_LAYER
        # Effective name used in the graph, after alias resolution, exposed for
        # logging/JSON. Computed lazily in ``_get_conv_output_tensor``.
        self.effective_layer_name: str | None = None

    def _get_conv_output_tensor(self):
        """Return the feature-map tensor connected to the outer graph.

        Rules:

        1. ``layerName == image_pooling`` (top-level default): returns
           ``get_layer('image_pooling').input``, the (1, 7, 7, 1280) tensor
           feeding the pooling layer in the outer graph.

        2. ``layerName`` in ``MOBILENET_FINAL_CONV_ALIASES`` (``out_relu`` and
           friends): the same tensor as case 1, semantically identical to that
           MobileNetV2 layer's output. ``image_pooling.input`` is used to avoid
           "Graph disconnected" without losing the semantics.

        3. ``layerName`` is top level in the main model: returns
           ``get_layer(layerName).output`` directly.
        """
        # Transparent alias: canonical MobileNetV2 names to image_pooling.input.
        if self.layerName in self.MOBILENET_FINAL_CONV_ALIASES:
            print(
                f"  [pyimagesearch] layer '{self.layerName}' is internal to "
                f"MobileNetV2; using 'image_pooling.input' as a proxy "
                f"(same tensor, outer graph)."
            )
            layer = self.model.get_layer(self.DEFAULT_TOP_LEVEL_LAYER)
            self.effective_layer_name = (
                f"{self.layerName} via {self.DEFAULT_TOP_LEVEL_LAYER}.input"
            )
            return layer.input

        try:
            layer = self.model.get_layer(self.layerName)
        except Exception as e:
            raise ValueError(
                f"Layer '{self.layerName}' not found at the model top level. "
                f"Available layers: "
                f"{[l.name for l in self.model.layers]}. "
                f"Canonical MobileNetV2 names supported via alias: "
                f"{sorted(self.MOBILENET_FINAL_CONV_ALIASES)}."
            ) from e
        if self.layerName == self.DEFAULT_TOP_LEVEL_LAYER:
            self.effective_layer_name = f"{self.layerName}.input"
            return layer.input
        self.effective_layer_name = f"{self.layerName}.output"
        return layer.output

    def compute_heatmap(self, image_batch, text_batch, eps=1e-8):
        """Heatmap as uint8 [0, 255] from an image + text batch."""
        conv_output_tensor = self._get_conv_output_tensor()
        gradModel = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[conv_output_tensor, self.model.output],
        )

        with tf.GradientTape() as tape:
            inputs_img = tf.cast(image_batch, tf.float32)
            inputs_txt = tf.cast(text_batch, tf.float32)
            (convOutputs, predictions) = gradModel([inputs_img, inputs_txt])
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)

        # Guided gradients, same logic as the original code.
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image_batch.shape[2], image_batch.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        return heatmap

    def compute_heatmap_float(self, image_batch, text_batch, eps=1e-8):
        """Heatmap as float32 [0, 1], the range used for comparison."""
        return self.compute_heatmap(image_batch, text_batch, eps=eps).astype(
            np.float32
        ) / 255.0
