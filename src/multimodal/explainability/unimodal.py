#!/usr/bin/env python3
"""
Unimodal (image) explainability module
==================================================
Produces visual explanations (Grad-CAM) for the decisions of the unimodal app
rating classifier, which sees UI screenshots only.

Not a CLI entry point. The menu runs ``unimodal_v2.py``, which builds on this
module: it imports ``ImageExplanation``, ``ExplanationOutputManager``,
``load_original_image``, ``preprocess_image_path`` and ``IMG_SIZE``, swaps the
first two by monkey-patching and delegates to ``explain()`` here. The
interactive menu, the data loading and the mode logic all live in this file,
which makes it a library of that module rather than dead code.

Requirements:
    - A trained model under results/training/unimodal/resultados_*/app_rating_model.h5
    - Screenshots under the Rico screenshots folder
"""

import os
import glob
import json
import datetime
import warnings

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import load_model

from tqdm import tqdm

from multimodal.common.cam_metrics import cam_mult_confidence, road_score
from multimodal.common.paths import (
    RESULTS_EXPLANATIONS_UNIMODAL,
    RESULTS_TRAINING_UNIMODAL,
    RICO_DIR,
    RICO_HIERARCHIES_DIR,
)
from multimodal.common.semantic_saliency import rank_elements_by_saliency

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================
IMG_SIZE = 224
IMAGES_FOLDER = str(RICO_DIR / "screenshots")

# Optional Grad-CAM enhancements (configured via YAML)
GRADCAM_SMOOTHING = False         # if True, use augmentation smoothing (~6x slower)
GRADCAM_METRICS: list[str] = []   # subset of {"cam_mult", "road"}
ROAD_PERCENTILES = [20, 40, 60, 80]

# Optional: path to a multimodal run whose app list is inherited as the
# canonical sample (mode 4). When set, the scope prompt is skipped and exactly
# the apps of that run are processed, which gives compare_xai a 100% app-level
# intersection.
APPS_FROM_MULTIMODAL_RUN: str | None = None


# =============================================================================
# UTILITIES
# =============================================================================

def preprocess_image_path(image_path, img_size=IMG_SIZE):
    """Load and preprocess an image for the model (resized to img_size)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    return img


def load_original_image(image_path):
    """Load an image as RGB at its original resolution (uint8) for display."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_heatmap_on_image(img_rgb, heatmap, out_path, alpha=0.5):
    """Save the image with the heatmap overlaid."""
    hmap = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (colored.astype('float32') / 255.0) * alpha + img_rgb * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)
    cv2.imwrite(out_path, cv2.cvtColor((overlay * 255).astype('uint8'), cv2.COLOR_RGB2BGR))


# =============================================================================
# CLASS: ModelSelector
# =============================================================================

class ModelSelector:
    """Select and load trained models from the resultados_* folders."""

    def __init__(self, base_path=None):
        self.base_path = base_path if base_path is not None else str(RESULTS_TRAINING_UNIMODAL)

    def list_available_models(self):
        """Find trained models and return a list of dicts.

        Covers:
          - ``resultados_*`` folders (single hold-out training runs).
          - ``cv_*/best_fold`` folders (copy of the best fold, marked 🏆).
          - ``cv_*/fold_N`` subfolders (individual folds).
        """
        models: list[dict] = []

        # Hold-out runs (resultados_*)
        for folder in sorted(glob.glob(os.path.join(self.base_path, "resultados_*")), reverse=True):
            info = self._build_model_info(folder, os.path.basename(folder))
            if info is not None:
                models.append(info)

        # Cross-validation runs (cv_*/best_fold and cv_*/fold_N)
        for cv_folder in sorted(glob.glob(os.path.join(self.base_path, "cv_*")), reverse=True):
            if not os.path.isdir(cv_folder):
                continue
            cv_name = os.path.basename(cv_folder)

            best_path = os.path.join(cv_folder, "best_fold")
            if os.path.isdir(best_path):
                info = self._build_model_info(best_path, f"{cv_name}/best_fold 🏆")
                if info is not None:
                    models.append(info)

            for sub in sorted(os.listdir(cv_folder)):
                if sub.startswith("fold_"):
                    fold_path = os.path.join(cv_folder, sub)
                    info = self._build_model_info(fold_path, f"{cv_name}/{sub}")
                    if info is not None:
                        models.append(info)

        return models

    def _build_model_info(self, folder, display_name):
        """Build the info dict for a model saved in `folder`, or None if absent."""
        model_path = os.path.join(folder, "app_rating_model.h5")
        if not os.path.exists(model_path):
            return None
        csv_test = os.path.join(folder, 'classification_all.csv')
        csv_full = os.path.join(folder, 'classification_full_dataset.csv')
        parts = os.path.basename(folder).split("_")
        timestamp = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else "unknown"
        return {
            'folder': folder,
            'folder_name': display_name,
            'model_path': model_path,
            'timestamp': timestamp,
            'classification_csv': csv_test,
            'classification_csv_test': csv_test,
            'classification_csv_full': csv_full,
            'has_classification': os.path.exists(csv_test),
            'has_full_dataset': os.path.exists(csv_full),
        }

    def interactive_select_model(self):
        """Interactive model selection menu."""
        models = self.list_available_models()

        if not models:
            print(f"ERROR: no trained model found in {self.base_path}/resultados_*/")
            return None

        print("\n" + "=" * 60)
        print("AVAILABLE MODELS")
        print("=" * 60)

        for i, m in enumerate(models):
            status = "[WITH CSV]" if m['has_classification'] else "[NO CSV]"
            print(f"  [{i + 1}] {m['folder_name']} {status}")
        print("  [0] Back")

        print("=" * 60)

        while True:
            try:
                choice = input(
                    f"\nSelect a model (1-{len(models)}, 0 to go back) [1]: "
                ).strip()
                if choice == "":
                    choice = 1
                else:
                    choice = int(choice)

                if choice == 0:
                    return None
                if 1 <= choice <= len(models):
                    return models[choice - 1]
                else:
                    print(f"Please type a number between 0 and {len(models)}")
            except ValueError:
                print("Invalid input. Type a number.")

    def interactive_select_csv(self, model_info):
        """Ask whether to use the test-split CSV or the full-dataset one.

        Returns None when the user goes back, so the caller can abort the run.
        """
        if not model_info.get('has_full_dataset'):
            return model_info

        print("\n" + "-" * 70)
        print("CLASSIFICATION CSV:")
        print("-" * 70)
        print("  1 - Test split only (classification_all.csv)")
        print("  2 - Full dataset: train + val + test (classification_full_dataset.csv)")
        print("  0 - Back")
        print("-" * 70)

        choice = input("Choose (1 or 2, 0 to go back) [2]: ").strip() or "2"
        if choice == "0":
            return None
        if choice == "2":
            model_info['classification_csv'] = model_info['classification_csv_full']
            print("Using the full dataset.")
        else:
            model_info['classification_csv'] = model_info['classification_csv_test']
            print("Using the test split only.")
        return model_info

    def load_selected_model(self, model_info):
        """Load the Keras model from the .h5 file."""
        print(f"\nLoading model: {model_info['model_path']}")
        model = load_model(model_info['model_path'])
        print("Model loaded.")
        return model


# =============================================================================
# CLASS: ImageTestDataLoader
# =============================================================================

class ImageTestDataLoader:
    """
    Load the test data from the classification CSV. Image-only counterpart of the
    multimodal loader.
    """

    def __init__(self, model_info):
        self.model_info = model_info
        self.classification_df = None
        self._load_classification_csv()

    def _load_classification_csv(self):
        """Load the classification CSV of the selected model."""
        csv_path = self.model_info['classification_csv']
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Classification CSV not found: {csv_path}")

        self.classification_df = pd.read_csv(csv_path)
        print(f"CSV loaded: {len(self.classification_df)} samples")

        # Drop duplicates (same package + screen)
        if 'numero_da_tela' in self.classification_df.columns:
            self.classification_df = self.classification_df.drop_duplicates(
                subset=['package_name', 'numero_da_tela']
            ).reset_index(drop=True)
            print(f"Unique samples (duplicates removed): {len(self.classification_df)}")

    def get_sample(self, idx):
        """
        Return one sample for the explanation pipeline (image only).

        Returns:
            dict with image_path, image, package_name, screen_id,
            true_label, true_class, predicted_class, is_correct, etc.
        """
        row = self.classification_df.iloc[idx]

        # Load the image
        screen_id = str(row['numero_da_tela'])
        image_path = os.path.join(IMAGES_FOLDER, f"{screen_id}.jpg")

        if not os.path.exists(image_path):
            return None

        image = preprocess_image_path(image_path)
        if image is None:
            return None

        # Map the label
        true_label = 1 if row['label_real'] == 'bom' else 0

        return {
            'image_path': image_path,
            'image': image,
            'package_name': row['package_name'],
            'screen_id': screen_id,
            'true_label': true_label,
            'true_class': row['label_real'],
            'predicted_class': row['predito'],
            'is_correct': row['acerto'],
            'app_name': row.get('app', row['package_name']),
            'category': row.get('categoria', 'Unknown'),
            'rating': row.get('rating', 0),
        }

    def get_samples_for_app(self, package_name):
        """Return every sample of a given app."""
        app_df = self.classification_df[
            self.classification_df['package_name'] == package_name
        ]

        samples = []
        for idx in app_df.index:
            sample = self.get_sample(idx)
            if sample:
                samples.append(sample)

        return samples

    def get_all_apps(self):
        """Return the list of unique apps."""
        return self.classification_df['package_name'].unique().tolist()

    def get_error_samples(self):
        """Return the indices of the misclassified samples."""
        error_df = self.classification_df[self.classification_df['acerto'] == False]
        return error_df.index.tolist()

    def __len__(self):
        return len(self.classification_df)

    def __iter__(self):
        for idx in range(len(self)):
            sample = self.get_sample(idx)
            if sample:
                yield sample


# =============================================================================
# CLASS: GradCAMExplainer
# =============================================================================

class GradCAMExplainer:
    """
    Produce visual explanations with Grad-CAM, for the image-only model whose
    MobileNetV2 backbone sits inside a Sequential.
    """

    def __init__(self, model):
        self.model = model
        self.mobilenet_layer = None
        self.target_conv_name = None
        self.grad_model = None
        self._find_mobilenet_and_conv()
        self._build_grad_model()

    def _find_mobilenet_and_conv(self):
        """
        Find MobileNetV2 and its last convolutional layer. Both Sequential and
        Functional models are supported.
        """
        # Step 1: find MobileNetV2
        for layer in self.model.layers:
            if 'mobilenet' in layer.name.lower():
                self.mobilenet_layer = layer
                break

        if self.mobilenet_layer is None:
            raise ValueError("MobileNetV2 not found in the model. Check whether the "
                             "model actually uses MobileNetV2.")

        # Step 2: find the last convolutional layer inside MobileNetV2
        preferred_layers = ['out_relu', 'Conv_1_bn', 'Conv_1', 'block_16_project']

        for preferred in preferred_layers:
            try:
                layer = self.mobilenet_layer.get_layer(preferred)
                self.target_conv_name = preferred
                break
            except Exception:
                continue

        # Fallback: take any 4D layer
        if self.target_conv_name is None:
            for sublayer in reversed(self.mobilenet_layer.layers):
                try:
                    output_shape = sublayer.output_shape
                    if isinstance(output_shape, tuple) and len(output_shape) == 4:
                        self.target_conv_name = sublayer.name
                        break
                except Exception:
                    continue

        if self.target_conv_name is None:
            raise ValueError("No convolutional layer found inside MobileNetV2")

        print(f"Grad-CAM configured:")
        print(f"  - MobileNetV2 layer: {self.mobilenet_layer.name}")
        print(f"  - Target conv layer: {self.target_conv_name}")

    def _build_grad_model(self):
        """
        Build the model that exposes the intermediate feature maps. The
        GlobalAveragePooling2D layer is looked up by type, since this model does not
        name its layers.
        """
        try:
            # Find GlobalAveragePooling2D by type
            pooling_layer = None

            # Try by name first, for parity with the multimodal module
            try:
                pooling_layer = self.model.get_layer('image_pooling')
            except ValueError:
                pass

            # Fallback: search by type
            if pooling_layer is None:
                for layer in self.model.layers:
                    if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                        pooling_layer = layer
                        break

            if pooling_layer is None:
                raise ValueError("GlobalAveragePooling2D not found in the model")

            # The pooling input is the feature-map tensor connected to the graph
            conv_output_tensor = pooling_layer.input

            # Build a model returning the feature maps and the prediction
            self.grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[conv_output_tensor, self.model.output]
            )

            print(f"  - Grad model built from the input of the pooling layer: {pooling_layer.name}")
            print(f"  - Feature maps shape: {conv_output_tensor.shape}")

        except Exception as e:
            print(f"WARNING building the grad_model: {e}")
            print("Falling back to gradients taken directly on the image.")
            self.grad_model = None

    def compute_gradcam(self, image_array):
        """
        Compute the heatmap, preferring the feature maps and falling back to the
        gradients on the input image.

        Args:
            image_array: (1, 224, 224, 3) preprocessed image

        Returns:
            heatmap: (224, 224) heatmap normalized to [0, 1]
            prediction: float, predicted probability
        """
        if self.grad_model is not None:
            return self._compute_gradcam_featuremap(image_array)
        else:
            return self._compute_saliency_fallback(image_array)

    def _compute_gradcam_featuremap(self, image_array):
        """Grad-CAM over the intermediate feature maps (preferred method).

        The loss targets the predicted class on a binary sigmoid head:
        - predicted "good" (p >= 0.5) -> ``loss = p``, steering gradients to the
          features supporting the "good" decision;
        - predicted "bad"  (p <  0.5) -> ``loss = 1 - p``, likewise for "bad".

        Following Selvaraju et al. 2017, Grad-CAM is always relative to the
        class of interest. Without this, confidently-bad apps had every gradient
        negative and the heatmap silently collapsed after the ReLU.
        """
        img_tensor = tf.cast(image_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = self.grad_model(img_tensor, training=False)
            # Steer the loss to the predicted class (see docstring).
            pred_val = float(predictions[0, 0].numpy())
            loss = (
                predictions[:, 0]
                if pred_val >= 0.5
                else (1.0 - predictions[:, 0])
            )

        # Gradients with respect to the feature maps
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("WARNING: null gradients on the feature map, using the fallback")
            return self._compute_saliency_fallback(image_array)

        # Weights: global mean of the gradients per channel
        weights = tf.reduce_mean(grads, axis=(1, 2))  # (1, num_channels)

        # Weighted combination of the feature maps
        cam = tf.reduce_sum(weights[:, tf.newaxis, tf.newaxis, :] * conv_outputs, axis=-1)

        # Apply ReLU (positive activations only)
        cam = tf.nn.relu(cam)

        # Normalize
        cam = cam[0]  # Drop the batch dimension
        max_val = tf.reduce_max(cam)
        if max_val > 0:
            cam = cam / (max_val + 1e-8)
        else:
            cam = tf.zeros_like(cam)

        # Resize to the image size
        heatmap_np = cam.numpy()
        heatmap_np = cv2.resize(heatmap_np, (IMG_SIZE, IMG_SIZE))

        # Smooth
        heatmap_np = cv2.GaussianBlur(heatmap_np, (15, 15), 0)

        # Renormalize after the blur
        if heatmap_np.max() > 0:
            heatmap_np = heatmap_np / (heatmap_np.max() + 1e-8)

        prediction = float(predictions[0, 0])
        return heatmap_np, prediction

    def _compute_saliency_fallback(self, image_array):
        """
        Fallback: compute the saliency from the gradients taken directly on the
        image. Useful when the backbone is frozen and Grad-CAM does not work.
        """
        img_tensor = tf.Variable(image_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            prediction = self.model(img_tensor, training=False)
            loss = prediction[:, 0]

        grads = tape.gradient(loss, img_tensor)

        if grads is None:
            print("WARNING: null gradients, using a uniform heatmap")
            return np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32) * 0.5, float(prediction[0, 0])

        # Reduce to a 2D heatmap (absolute mean over the RGB channels)
        heatmap = tf.reduce_mean(tf.abs(grads[0]), axis=-1)

        # Normalize
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / (max_val + 1e-8)
        else:
            heatmap = tf.zeros_like(heatmap)

        # Smooth with a blur
        heatmap_np = heatmap.numpy()
        heatmap_np = cv2.GaussianBlur(heatmap_np, (15, 15), 0)

        # Renormalize after the blur
        if heatmap_np.max() > 0:
            heatmap_np = heatmap_np / (heatmap_np.max() + 1e-8)

        return heatmap_np, float(prediction[0, 0])

    def compute_gradcam_smoothed(self, image_array, n_aug=6):
        """Augmentation smoothing (pytorch-grad-cam style): average heatmaps over TTA.

        Generates n_aug augmentations of the input (h-flip + brightness),
        computes Grad-CAM on each, un-flips when needed, and averages.
        Slower (~n_aug × base cost) but visually cleaner.
        """
        augmentations = [
            ("id", lambda x: x),
            ("flip", lambda x: x[:, :, ::-1, :]),
            ("b+", lambda x: np.clip(x * 1.2, 0.0, 1.0)),
            ("b-", lambda x: np.clip(x * 0.8, 0.0, 1.0)),
            ("flip_b+", lambda x: np.clip(x[:, :, ::-1, :] * 1.2, 0.0, 1.0)),
            ("flip_b-", lambda x: np.clip(x[:, :, ::-1, :] * 0.8, 0.0, 1.0)),
        ][:n_aug]

        heatmaps = []
        preds = []
        for tag, aug in augmentations:
            aug_img = aug(image_array).astype(np.float32)
            hm, pred = self.compute_gradcam(aug_img)
            if "flip" in tag:
                hm = hm[:, ::-1].copy()
            heatmaps.append(hm)
            preds.append(pred)

        mean_heatmap = np.mean(np.stack(heatmaps, axis=0), axis=0)
        if mean_heatmap.max() > 0:
            mean_heatmap = mean_heatmap / (mean_heatmap.max() + 1e-8)
        mean_pred = float(np.mean(preds))
        return mean_heatmap, mean_pred

    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        """
        Build the visualization with the heatmap overlaid on the image.

        Accepts an image of any size; the heatmap is resized to match. The
        convention is to overlay onto the high-resolution image.

        Args:
            image: (H, W, 3) image as uint8 [0,255] or float [0,1]
            heatmap: (h, w) heatmap in [0, 1], resized to (H, W)
            alpha: blending factor

        Returns:
            blended: (H, W, 3) merged image as uint8
        """
        image_uint8 = image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)
        h, w = image_uint8.shape[:2]

        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        return cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)

    def aggregate_heatmaps_for_app(self, heatmaps, confidences):
        """
        Aggregate several heatmaps into an app-level one, using a mean weighted by
        confidence.

        Args:
            heatmaps: list of (H, W) arrays
            confidences: list of floats

        Returns:
            aggregated: (H, W) normalized aggregated heatmap
        """
        if not heatmaps:
            return None

        heatmaps = np.array(heatmaps)
        confidences = np.array(confidences)

        # Normalize the weights
        if confidences.sum() < 1e-8:
            weights = np.ones(len(confidences)) / len(confidences)
        else:
            weights = confidences / (confidences.sum() + 1e-8)

        # Weighted mean
        aggregated = np.tensordot(weights, heatmaps, axes=(0, 0))

        # Renormalize
        if aggregated.max() > aggregated.min():
            aggregated = (aggregated - aggregated.min()) / (aggregated.max() - aggregated.min() + 1e-8)

        return aggregated


# =============================================================================
# CLASS: ImageExplanation
# =============================================================================

class ImageExplanation:
    """
    Produce the visual explanations of the image-only model, using Grad-CAM to
    highlight the UI regions that drive the prediction.
    """

    def __init__(self, model, gradcam_explainer):
        self.model = model
        self.gradcam = gradcam_explainer

    def explain_screen(
        self,
        image_path,
        true_label,
        package_name,
        screen_id,
        output_dir,
        app_name=None,
        category=None,
    ):
        """
        Produce the visual explanation of one screen.

        Args:
            image_path: path to the image
            true_label: true label (0 or 1)
            package_name: package name
            screen_id: screen identifier
            output_dir: output directory
            app_name: friendly app name (optional, for the figure metadata)
            category: app category (optional, for the figure metadata)

        Returns:
            explanation: dict with the explanation data
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load the image
        image = preprocess_image_path(image_path)
        if image is None:
            return None

        image_batch = np.expand_dims(image, 0)

        # 2. Compute Grad-CAM, with or without augmentation smoothing
        if GRADCAM_SMOOTHING:
            heatmap, prediction = self.gradcam.compute_gradcam_smoothed(image_batch)
        else:
            heatmap, prediction = self.gradcam.compute_gradcam(image_batch)

        # 3. Render the figure
        viz_filename = f"{package_name.replace('.', '_')}_{screen_id}_explanation.png"
        viz_path = os.path.join(output_dir, viz_filename)

        self.generate_screen_visualization(
            image_path=image_path,
            heatmap=heatmap,
            prediction=prediction,
            true_label=true_label,
            output_path=viz_path
        )

        # 4. Build the explanation dict
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Semantic saliency: cross the heatmap with the Rico view hierarchy
        # and rank the most salient UI components.
        top_components = []
        hierarchy_path = RICO_HIERARCHIES_DIR / f"{screen_id}.json"
        if hierarchy_path.exists():
            ranked = rank_elements_by_saliency(heatmap, hierarchy_path)
            for r in ranked[:3]:
                top_components.append({
                    'component': r['component'],
                    'score_total': r['score_total'],
                    'score_mean': r['score_mean'],
                    'bounds_image': list(r['bounds_image']),
                })

        # Optional CAM faithfulness metrics
        cam_metrics: dict[str, float] = {}
        if GRADCAM_METRICS:
            predict_fn = lambda x: self.model.predict(x, verbose=0)
            if "cam_mult" in GRADCAM_METRICS:
                cam_metrics["cam_mult"] = cam_mult_confidence(image, heatmap, predict_fn)
            if "road" in GRADCAM_METRICS:
                road_result = road_score(image, heatmap, predict_fn, ROAD_PERCENTILES)
                cam_metrics["road_mean"] = road_result["mean"]
                cam_metrics["road_per_percentile"] = road_result["per_percentile"]

        explanation = {
            'package_name': package_name,
            'app_name': app_name,
            'category': category,
            'screen_id': screen_id,
            'image_path': image_path,
            'prediction': float(prediction),
            'predicted_class': 'good' if prediction > 0.5 else 'bad',
            'true_label': int(true_label),
            'true_class': 'good' if true_label == 1 else 'bad',
            'is_correct': (prediction > 0.5) == (true_label == 1),
            'confidence': float(confidence),
            'heatmap_stats': {
                'mean': float(heatmap.mean()),
                'max': float(heatmap.max()),
                'std': float(heatmap.std()),
                'hot_area_pct': float((heatmap > 0.5).sum() / heatmap.size)
            },
            'top_components': top_components,
            'cam_metrics': cam_metrics,
            'visualization_path': viz_path
        }

        return explanation

    def explain_app(self, package_name, screen_explanations, output_dir):
        """
        Produce the app-level aggregated explanation.

        Args:
            package_name: package name
            screen_explanations: list of screen explanations
            output_dir: output directory

        Returns:
            app_explanation: dict with the aggregated explanation
        """
        os.makedirs(output_dir, exist_ok=True)

        if not screen_explanations:
            return None

        # 1. Aggregate the heatmaps, weighted by confidence
        heatmaps = []
        confidences = []
        image_paths = []
        screen_heatmaps = []  # For the thumbnail gallery

        for expl in screen_explanations:
            image = preprocess_image_path(expl['image_path'])
            if image is not None:
                image_paths.append(expl['image_path'])
                # Recompute the heatmap
                img_batch = np.expand_dims(image, 0)
                heatmap, _ = self.gradcam.compute_gradcam(img_batch)
                heatmaps.append(heatmap)
                screen_heatmaps.append(heatmap)
                confidences.append(expl['confidence'])

        aggregated_heatmap = None
        if heatmaps:
            aggregated_heatmap = self.gradcam.aggregate_heatmaps_for_app(heatmaps, confidences)

        # 2. Compute the aggregated prediction
        predictions = [expl['prediction'] for expl in screen_explanations]
        app_score = float(np.mean(predictions))
        app_prediction = 'good' if app_score > 0.5 else 'bad'
        app_confidence = app_score if app_score > 0.5 else 1 - app_score

        true_label = screen_explanations[0]['true_label']

        # 3. Render the aggregated figure
        app_viz_path = None
        if image_paths and aggregated_heatmap is not None:
            app_viz_path = os.path.join(output_dir, f"{package_name.replace('.', '_')}_app_explanation.png")
            self._generate_app_visualization(
                representative_image_path=image_paths[0],
                aggregated_heatmap=aggregated_heatmap,
                all_image_paths=image_paths,
                all_screen_heatmaps=screen_heatmaps,
                app_score=app_score,
                true_label=true_label,
                num_screens=len(screen_explanations),
                output_path=app_viz_path
            )

        # 4. Compute the aggregated statistics
        hot_area_pcts = [e['heatmap_stats']['hot_area_pct'] for e in screen_explanations
                         if 'heatmap_stats' in e]
        avg_hot_area = float(np.mean(hot_area_pcts)) if hot_area_pcts else 0.0

        # 5. Build the explanation
        app_explanation = {
            'package_name': package_name,
            'num_screens': len(screen_explanations),
            'app_score': app_score,
            'app_prediction': app_prediction,
            'app_confidence': float(app_confidence),
            'true_label': true_label,
            'true_class': 'good' if true_label == 1 else 'bad',
            'is_correct': (app_score > 0.5) == (true_label == 1),
            'screen_predictions': predictions,
            'prediction_variance': float(np.var(predictions)),
            'avg_hot_area_pct': avg_hot_area,
            'visualization_path': app_viz_path,
            'summary': self._generate_summary(package_name, app_prediction,
                                              app_confidence, len(screen_explanations))
        }

        return app_explanation

    def generate_screen_visualization(self, image_path, heatmap, prediction, true_label, output_path):
        """
        Render the screen figure: original image | heatmap | overlay.

        Three columns, with the image loaded at its native resolution and the
        heatmap resized to match.

        Panel titles and the info bar below are deliberately in Portuguese: they
        are rendered into the PNGs, so translating them would make regenerated
        figures disagree with the ones already in use. Comments stay in English;
        figure content does not.
        """
        original = load_original_image(image_path)
        h, w = original.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        fig = plt.figure(figsize=(15, 7))
        gs = GridSpec(2, 3, height_ratios=[5, 1], width_ratios=[1, 1, 1])

        # === Column 1: original image ===
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(original)
        ax_orig.set_title("Imagem Original", fontsize=12, fontweight='bold')
        ax_orig.axis('off')

        # === Column 2: raw Grad-CAM heatmap (high resolution) ===
        ax_heatmap = fig.add_subplot(gs[0, 1])
        ax_heatmap.imshow(heatmap_resized, cmap='jet', interpolation='bilinear')
        ax_heatmap.set_title("Mapa de Saliencia (Grad-CAM)", fontsize=12, fontweight='bold')
        ax_heatmap.axis('off')

        # === Column 3: overlay (original image + heatmap) ===
        ax_blend = fig.add_subplot(gs[0, 2])
        blended = self.gradcam.overlay_heatmap(original, heatmap_resized, alpha=0.5)
        ax_blend.imshow(blended)
        ax_blend.set_title("Atencao Visual (Overlay)", fontsize=12, fontweight='bold')
        ax_blend.axis('off')

        # === Row 2: prediction info ===
        ax_info = fig.add_subplot(gs[1, :])
        ax_info.axis('off')

        pred_class = "BOM" if prediction > 0.5 else "RUIM"
        true_class = "BOM" if true_label == 1 else "RUIM"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        is_correct = (prediction > 0.5) == (true_label == 1)

        result_text = "CORRETO" if is_correct else "INCORRETO"
        result_color = 'green' if is_correct else 'red'

        info_text = (
            f"Predicao: {pred_class} (Confianca: {confidence:.1%})    |    "
            f"Classe Real: {true_class}    |    "
            f"Resultado: {result_text}"
        )

        ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                     fontsize=12, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7),
                     color=result_color, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    def _generate_app_visualization(self, representative_image_path, aggregated_heatmap,
                                    all_image_paths, all_screen_heatmaps,
                                    app_score, true_label, num_screens, output_path):
        """
        Render the app-level figure with a thumbnail gallery.

        Layout: representative image + aggregated heatmap + screen gallery + info
        bar, with the images loaded at their native resolution and the heatmaps
        resized to match.

        Panel titles and the info bar below are deliberately in Portuguese: they
        are rendered into the PNGs, so translating them would make regenerated
        figures disagree with the ones already in use. Comments stay in English;
        figure content does not.
        """
        representative_original = load_original_image(representative_image_path)
        rep_h, rep_w = representative_original.shape[:2]
        aggregated_heatmap_resized = cv2.resize(aggregated_heatmap, (rep_w, rep_h), interpolation=cv2.INTER_LINEAR)

        # Cap the gallery at 8 thumbnails
        max_thumbs = min(len(all_image_paths), 8)
        has_thumbs = max_thumbs > 1

        if has_thumbs:
            # Layout with the gallery
            thumb_rows = 1 if max_thumbs <= 4 else 2
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(2 + thumb_rows, 2,
                          height_ratios=[4] + [2] * thumb_rows + [1],
                          width_ratios=[1, 1])
        else:
            # Layout without the gallery
            fig = plt.figure(figsize=(14, 7))
            gs = GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[1, 1])

        # === Row 1, left: image with the aggregated heatmap (high res) ===
        ax_img = fig.add_subplot(gs[0, 0])
        blended = self.gradcam.overlay_heatmap(representative_original, aggregated_heatmap_resized, alpha=0.5)
        ax_img.imshow(blended)
        ax_img.set_title(f"Atencao Visual Agregada ({num_screens} telas)",
                         fontsize=12, fontweight='bold')
        ax_img.axis('off')

        # === Row 1, right: raw aggregated heatmap (high resolution) ===
        ax_heatmap = fig.add_subplot(gs[0, 1])
        ax_heatmap.imshow(aggregated_heatmap_resized, cmap='jet', interpolation='bilinear')
        ax_heatmap.set_title("Mapa de Saliencia Agregado", fontsize=12, fontweight='bold')
        ax_heatmap.axis('off')

        # === Thumbnail gallery ===
        if has_thumbs:
            thumbs_per_row = 4
            for i in range(max_thumbs):
                row_idx = i // thumbs_per_row
                col_idx = i % thumbs_per_row

                # Manual subplot placement inside the gallery area
                ax_thumb = fig.add_axes([
                    0.05 + col_idx * 0.23,           # left
                    0.52 - row_idx * 0.22 if thumb_rows == 2 else 0.35,  # bottom
                    0.20,                             # width
                    0.18 if thumb_rows == 2 else 0.20  # height
                ])

                thumb_original = load_original_image(all_image_paths[i])
                thumb_blend = self.gradcam.overlay_heatmap(
                    thumb_original, all_screen_heatmaps[i], alpha=0.5
                )
                ax_thumb.imshow(thumb_blend)
                ax_thumb.set_title(f"Tela {i+1}", fontsize=8)
                ax_thumb.axis('off')

        # === Last row: app info ===
        last_row = 1 + (thumb_rows if has_thumbs else 0)
        ax_info = fig.add_subplot(gs[last_row, :])
        ax_info.axis('off')

        pred_class = "BOM" if app_score > 0.5 else "RUIM"
        true_class = "BOM" if true_label == 1 else "RUIM"
        confidence = app_score if app_score > 0.5 else 1 - app_score
        is_correct = (app_score > 0.5) == (true_label == 1)

        result_text = "CORRETO" if is_correct else "INCORRETO"
        result_color = 'green' if is_correct else 'red'

        info_text = (
            f"Score do App: {app_score:.3f}    |    "
            f"Predicao: {pred_class} ({confidence:.1%})    |    "
            f"Classe Real: {true_class}    |    "
            f"Resultado: {result_text}"
        )

        ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                     fontsize=12, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7),
                     color=result_color, fontweight='bold')

        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    def _generate_summary(self, package_name, prediction, confidence, num_screens):
        """Build the textual summary of the explanation."""
        return (
            f"App '{package_name}' was classified as {prediction} "
            f"with {confidence:.1%} confidence, "
            f"from the visual analysis of {num_screens} screen(s)."
        )


# =============================================================================
# CLASS: ExplanationOutputManager
# =============================================================================

class ExplanationOutputManager:
    """Handle the saving of the explanations as JSON and CSV."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.screen_dir = os.path.join(output_dir, "screen_explanations")
        self.app_dir = os.path.join(output_dir, "app_explanations")

        os.makedirs(self.screen_dir, exist_ok=True)
        os.makedirs(self.app_dir, exist_ok=True)

        self.screen_explanations = []
        self.app_explanations = []

    def add_screen_explanation(self, explanation):
        if explanation:
            self.screen_explanations.append(explanation)

    def _save_semantic_saliency_stats(self):
        """Aggregate the top-N salient components per class and write the CSV."""
        rows = []
        for expl in self.screen_explanations:
            true_class = expl.get('true_class', '-')
            for rank, comp in enumerate(expl.get('top_components', []) or [], start=1):
                rows.append({
                    'component': comp['component'],
                    'true_class': true_class,
                    'rank': rank,
                    'score_total': comp['score_total'],
                    'score_mean': comp['score_mean'],
                })
        if not rows:
            return

        import pandas as pd
        df = pd.DataFrame(rows)
        pivot = (
            df.groupby(['component', 'true_class'])
              .agg(count=('rank', 'size'),
                   mean_score_total=('score_total', 'mean'),
                   mean_score_mean=('score_mean', 'mean'))
              .reset_index()
        )
        wide = pivot.pivot(index='component', columns='true_class',
                           values=['count', 'mean_score_total', 'mean_score_mean'])
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide = wide.fillna(0).reset_index()
        wide.to_csv(os.path.join(self.output_dir, 'semantic_saliency_stats.csv'), index=False)
        print(f"  - Semantic saliency stats: semantic_saliency_stats.csv ({len(wide)} components)")

    def add_app_explanation(self, explanation):
        if explanation:
            self.app_explanations.append(explanation)

    def save_all(self):
        """Write every explanation to JSON and CSV."""
        # JSON - screens
        screen_json_path = os.path.join(self.output_dir, 'screen_explanations.json')
        with open(screen_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.screen_explanations, f, indent=2, ensure_ascii=False, default=str)

        # CSV - semantic saliency aggregation (good vs bad)
        self._save_semantic_saliency_stats()

        # JSON - apps
        app_json_path = os.path.join(self.output_dir, 'app_explanations.json')
        with open(app_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.app_explanations, f, indent=2, ensure_ascii=False, default=str)

        # CSV - screens
        if self.screen_explanations:
            screen_df = pd.DataFrame([{
                'package_name': e['package_name'],
                'screen_id': e['screen_id'],
                'prediction': e['prediction'],
                'predicted_class': e['predicted_class'],
                'true_label': e['true_label'],
                'is_correct': e['is_correct'],
                'confidence': e['confidence'],
                'heatmap_mean': e['heatmap_stats']['mean'],
                'heatmap_max': e['heatmap_stats']['max'],
                'heatmap_std': e['heatmap_stats']['std'],
                'hot_area_pct': e['heatmap_stats']['hot_area_pct']
            } for e in self.screen_explanations])
            screen_df.to_csv(os.path.join(self.output_dir, 'screen_explanations.csv'), index=False)

        # CSV - apps
        if self.app_explanations:
            app_df = pd.DataFrame([{
                'package_name': e['package_name'],
                'num_screens': e['num_screens'],
                'app_score': e['app_score'],
                'app_prediction': e['app_prediction'],
                'true_label': e['true_label'],
                'is_correct': e['is_correct'],
                'app_confidence': e['app_confidence'],
                'prediction_variance': e['prediction_variance'],
                'avg_hot_area_pct': e.get('avg_hot_area_pct', 0),
                'summary': e['summary']
            } for e in self.app_explanations])
            app_df.to_csv(os.path.join(self.output_dir, 'app_explanations.csv'), index=False)

        print(f"\nExplanations saved in: {self.output_dir}/")
        print(f"  - {len(self.screen_explanations)} screen explanations")
        print(f"  - {len(self.app_explanations)} app explanations")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def explain(config: dict) -> None:
    """Unimodal (vision-only) explainability entry point driven by a YAML config dict."""
    global IMG_SIZE, GRADCAM_SMOOTHING, GRADCAM_METRICS, ROAD_PERCENTILES
    global APPS_FROM_MULTIMODAL_RUN

    IMG_SIZE = config.get("img_size", IMG_SIZE)
    GRADCAM_SMOOTHING = bool(config.get("gradcam_smoothing", GRADCAM_SMOOTHING))
    GRADCAM_METRICS = list(config.get("gradcam_metrics", GRADCAM_METRICS) or [])
    ROAD_PERCENTILES = list(config.get("road_percentiles", ROAD_PERCENTILES))
    _apps_src = config.get("apps_from_multimodal_run")
    APPS_FROM_MULTIMODAL_RUN = str(_apps_src).strip() if _apps_src else None

    print("=" * 70)
    print("UNIMODAL (IMAGE) EXPLAINABILITY MODULE")
    print("=" * 70)

    # 1. Select the model
    selector = ModelSelector()
    print(f"\nLooking for models in: {selector.base_path}")
    model_info = selector.interactive_select_model()

    if model_info is None:
        return

    # 1b. Choose which classification CSV to use (test split or full dataset)
    model_info = selector.interactive_select_csv(model_info)

    if model_info is None:
        return

    # 2. Load the model
    model = selector.load_selected_model(model_info)

    # 3. Check the model is unimodal
    num_inputs = len(model.inputs)
    if num_inputs != 1:
        print(f"\nWARNING: the model has {num_inputs} inputs; this module expects a unimodal model (1 input).")
        print("Use the multimodal explainability module for multimodal models.")
        confirm = input("Continue anyway? (y/n) [n]: ").strip().lower()
        if confirm != 'y':
            return

    # 4. Print the model structure for debugging
    print("\nModel structure:")
    for i, layer in enumerate(model.layers):
        print(f"  [{i}] {layer.name}: {type(layer).__name__}")

    # 5. Initialize the explainers
    print("\nInitialising the explainability components...")

    try:
        gradcam = GradCAMExplainer(model)
    except Exception as e:
        print(f"ERROR initialising Grad-CAM: {e}")
        return

    explainer = ImageExplanation(model, gradcam)

    # 6. Mode menu
    print("\n" + "-" * 70)
    print("EXPLANATION MODES:")
    print("-" * 70)
    print("  1 - Quick test (1 sample)")
    print("  2 - Explain one screen")
    print("  3 - Explain one app")
    print("  4 - Explain every app (needs the CSV)")
    print("  5 - Explain misclassifications only (needs the CSV)")
    print("  0 - Back")
    print("-" * 70)

    mode = input("\nSelect a mode (1-5, 0 to go back) [1]: ").strip() or "1"
    if mode == "0":
        # Nothing has been written yet, so returning leaves no partial output.
        return

    # Create the output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = str(RESULTS_EXPLANATIONS_UNIMODAL / f"explanations_unimodal_{timestamp}")
    output_manager = ExplanationOutputManager(output_dir)

    if mode == "1":
        # Quick test on a sample image
        print("\nRunning the quick test...")

        test_images = glob.glob(os.path.join(IMAGES_FOLDER, "*.jpg"))[:1]
        if not test_images:
            print(f"No image found in {IMAGES_FOLDER}")
            return

        img_path = test_images[0]
        print(f"Using image: {img_path}")

        explanation = explainer.explain_screen(
            image_path=img_path,
            true_label=1,
            package_name="test_app",
            screen_id=os.path.basename(img_path).replace('.jpg', ''),
            output_dir=output_manager.screen_dir
        )

        if explanation:
            output_manager.add_screen_explanation(explanation)
            print(f"\nExplanation written: {explanation['visualization_path']}")
            print(f"Prediction: {explanation['predicted_class']} ({explanation['confidence']:.1%})")

    elif mode in ["2", "3", "4", "5"]:
        # Check that the classification CSV exists
        if not model_info['has_classification']:
            print(f"\nERROR: this model has no classification CSV.")
            print(f"classification_all.csv was not found in {model_info['folder']}")
            return

        # Load the test data
        print("\nLoading the test data...")
        try:
            data_loader = ImageTestDataLoader(model_info)
        except Exception as e:
            print(f"ERROR loading the data: {e}")
            return

        if mode == "2":
            # Mode 2: explain one screen
            print("\n" + "-" * 50)
            print("MODE 2: EXPLAIN ONE SCREEN")
            print("-" * 50)

            screen_id = input("Screen number (e.g. 32059): ").strip()

            matches = data_loader.classification_df[
                data_loader.classification_df['numero_da_tela'].astype(str) == screen_id
            ]

            if matches.empty:
                print(f"Screen {screen_id} not found in the test set.")
                return

            idx = matches.index[0]
            sample = data_loader.get_sample(idx)

            if sample is None:
                print(f"Could not load the data of screen {screen_id}")
                return

            print(f"\nExplaining screen {screen_id}...")
            print(f"  App: {sample['package_name']}")
            print(f"  True label: {sample['true_class']}")
            print(f"  Prediction: {sample['predicted_class']}")

            explanation = explainer.explain_screen(
                image_path=sample['image_path'],
                true_label=sample['true_label'],
                package_name=sample['package_name'],
                screen_id=sample['screen_id'],
                output_dir=output_manager.screen_dir,
                app_name=sample.get('app_name'),
                category=sample.get('category'),
            )

            if explanation:
                output_manager.add_screen_explanation(explanation)
                print(f"\nExplanation written: {explanation['visualization_path']}")

        elif mode == "3":
            # Mode 3: explain one app
            print("\n" + "-" * 50)
            print("MODE 3: EXPLAIN ONE APP")
            print("-" * 50)

            apps = data_loader.get_all_apps()
            print(f"\n{len(apps)} apps available. Examples:")
            for app in apps[:10]:
                print(f"  {app}")
            if len(apps) > 10:
                print(f"  ... and {len(apps) - 10} more apps")

            package_name = input("\nApp package_name: ").strip()

            if package_name not in apps:
                print(f"App {package_name} not found.")
                return

            samples = data_loader.get_samples_for_app(package_name)
            print(f"\nExplaining {len(samples)} screens of {package_name}...")

            for sample in tqdm(samples, desc="Processing screens"):
                explanation = explainer.explain_screen(
                    image_path=sample['image_path'],
                    true_label=sample['true_label'],
                    package_name=sample['package_name'],
                    screen_id=sample['screen_id'],
                    output_dir=output_manager.screen_dir,
                    app_name=sample.get('app_name'),
                    category=sample.get('category'),
                )
                if explanation:
                    output_manager.add_screen_explanation(explanation)

            # Build the aggregated app explanation
            if output_manager.screen_explanations:
                app_explanation = explainer.explain_app(
                    package_name=package_name,
                    screen_explanations=output_manager.screen_explanations,
                    output_dir=output_manager.app_dir
                )
                if app_explanation:
                    output_manager.add_app_explanation(app_explanation)

        elif mode == "4":
            # Mode 4: explain every app. For each app in the CSV, process every
            # screen and aggregate. Mirrors mode 4 of the multimodal module:
            # per-app loop plus per-app directories when available.
            print("\n" + "-" * 50)
            print("MODE 4: EXPLAIN EVERY APP")
            print("-" * 50)

            all_apps = data_loader.get_all_apps()
            total_apps = len(all_apps)
            print(f"\nApps in the CSV: {total_apps}")

            # Canonical app list inherited from a multimodal run. When
            # ``APPS_FROM_MULTIMODAL_RUN`` is set, the scope prompt is skipped and
            # exactly the same package_names of that run are processed, so
            # compare_xai gets a 100% app-level intersection.
            if APPS_FROM_MULTIMODAL_RUN:
                multi_run_path = os.path.abspath(APPS_FROM_MULTIMODAL_RUN)
                app_json = os.path.join(multi_run_path, "app_explanations.json")
                if not os.path.exists(app_json):
                    print(
                        f"ERROR: apps_from_multimodal_run='{multi_run_path}' "
                        f"does not contain app_explanations.json."
                    )
                    return
                try:
                    with open(app_json, encoding="utf-8") as f:
                        multi_apps_data = json.load(f)
                except Exception as e:
                    print(f"ERROR reading {app_json}: {e}")
                    return
                canonical = [a["package_name"] for a in multi_apps_data]
                # Keep only apps present in this unimodal model's CSV.
                apps = [p for p in canonical if p in set(all_apps)]
                missing = [p for p in canonical if p not in set(all_apps)]
                print(
                    f"\n[INFO] canonical sample inherited from {os.path.basename(multi_run_path)}: "
                    f"{len(apps)}/{len(canonical)} apps present in this CSV."
                )
                if missing:
                    print(
                        f"   [WARN] {len(missing)} multimodal apps are missing here: "
                        f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
                    )
                scope = "canonical"
            else:
                # Scope=2 (top-N good + top-N bad by confidence) needs the
                # 'probabilidade' column, only in classification_full_dataset.csv.
                has_proba = "probabilidade" in data_loader.classification_df.columns
                if has_proba:
                    print("\nScope:")
                    print(f"  1 - Process ALL {total_apps} apps")
                    print("  2 - Top-N good + top-N bad by highest confidence (default N=20)")
                    scope = input("Choose (1/2) [2]: ").strip() or "2"
                else:
                    print(
                        "\n[INFO] the selected CSV has no 'probabilidade' column. "
                        "The confidence scope is unavailable; using 'all'."
                    )
                    scope = "1"

                if scope == "2":
                    raw_n = input("N per class [20]: ").strip()
                    try:
                        n_per_class = int(raw_n) if raw_n else 20
                    except ValueError:
                        n_per_class = 20

                    df = data_loader.classification_df
                    # Keep only correct predictions before aggregating: the
                    # explainability analysis targets confident and correct ones.
                    correct_df = df[df["acerto"] == True]
                    agg = (
                        correct_df.groupby(
                            ["package_name", "predito"]
                        )["probabilidade"]
                        .mean()
                        .reset_index()
                    )
                    agg["confianca_agg"] = (agg["probabilidade"] - 0.5).abs() * 2
                    bom = (
                        agg[agg["predito"] == "bom"]
                        .nlargest(n_per_class, "confianca_agg")["package_name"]
                        .tolist()
                    )
                    ruim = (
                        agg[agg["predito"] == "ruim"]
                        .nlargest(n_per_class, "confianca_agg")["package_name"]
                        .tolist()
                    )
                    apps = bom + ruim
                    n_filtered = len(df) - len(correct_df)
                    print(
                        f"\nSelected sample: {len(bom)} good + {len(ruim)} bad "
                        f"= {len(apps)} apps (top-N by aggregated confidence among "
                        f"correct predictions; {n_filtered} incorrect rows dropped)."
                    )
                else:
                    apps = all_apps
                    confirm = input(
                        f"Process ALL {total_apps} apps? (y/n) [n]: "
                    ).strip().lower()
                    if confirm != "y":
                        print("Cancelled.")
                        return

            total_apps_selected = len(apps)
            for app_idx, package_name in enumerate(apps, start=1):
                samples = data_loader.get_samples_for_app(package_name)
                if not samples:
                    print(
                        f"\n[{app_idx}/{total_apps_selected}] {package_name}: "
                        f"no sample, skipping."
                    )
                    continue

                print(
                    f"\n[{app_idx}/{total_apps_selected}] {package_name}: "
                    f"{len(samples)} screens"
                )

                # Per-app directories when available, otherwise the shared ones.
                _screen_dir = (
                    output_manager.screen_dir_for(package_name)
                    if hasattr(output_manager, "screen_dir_for")
                    else output_manager.screen_dir
                )
                _app_dir = (
                    output_manager.app_dir_for(package_name)
                    if hasattr(output_manager, "app_dir_for")
                    else output_manager.app_dir
                )

                for sample in tqdm(
                    samples,
                    desc=f"  screens of {package_name[:30]}",
                    leave=False,
                ):
                    explanation = explainer.explain_screen(
                        image_path=sample["image_path"],
                        true_label=sample["true_label"],
                        package_name=sample["package_name"],
                        screen_id=sample["screen_id"],
                        output_dir=_screen_dir,
                        app_name=sample.get("app_name"),
                        category=sample.get("category"),
                    )
                    if explanation:
                        output_manager.add_screen_explanation(explanation)

                # Per-app aggregation over the screens of this iteration.
                app_screens = [
                    e
                    for e in output_manager.screen_explanations
                    if e["package_name"] == package_name
                ]
                if app_screens:
                    app_explanation = explainer.explain_app(
                        package_name=package_name,
                        screen_explanations=app_screens,
                        output_dir=_app_dir,
                    )
                    if app_explanation:
                        output_manager.add_app_explanation(app_explanation)

        elif mode == "5":
            # Mode 5: explain misclassifications only
            print("\n" + "-" * 50)
            print("MODE 5: EXPLAIN MISCLASSIFICATIONS ONLY")
            print("-" * 50)

            error_indices = data_loader.get_error_samples()
            print(f"\nMisclassifications: {len(error_indices)}")

            if len(error_indices) == 0:
                print("No misclassification found.")
                return

            confirm = input(f"Process the {len(error_indices)} errors? (y/n) [y]: ").strip().lower()
            if confirm == 'n':
                print("Cancelled.")
                return

            print("\nProcessing the errors...")
            for idx in tqdm(error_indices, desc="Generating explanations"):
                sample = data_loader.get_sample(idx)
                if sample is None:
                    continue

                explanation = explainer.explain_screen(
                    image_path=sample['image_path'],
                    true_label=sample['true_label'],
                    package_name=sample['package_name'],
                    screen_id=sample['screen_id'],
                    output_dir=output_manager.screen_dir,
                    app_name=sample.get('app_name'),
                    category=sample.get('category'),
                )
                if explanation:
                    output_manager.add_screen_explanation(explanation)

    # Save the results
    output_manager.save_all()

    print("\nDone.")
