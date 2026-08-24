#!/usr/bin/env python3
"""
Multimodal explainability module
================================
Produces visual explanations (Grad-CAM + saliency map), textual ones (DeBERTa
attention + Bahdanau attention) and multimodal SHAP for the decisions of the
multimodal app-rating classifier.

Not a CLI entry point. This file is the base of the whole chain:
``multimodal_v3.py`` patches ``GradCAMExplainer`` and ``ModelSelector`` here,
``multimodal_v2.py`` swaps ``MultimodalExplanation`` and
``ExplanationOutputManager``, and it is the ``explain()`` of this module that
ends up running the mode menu (1-11), the ``TestDataLoader`` and all the screen
selection logic. It is also imported by ``tools/validate_gradcam.py``.

Based on the IEEE paper:
"Multimodal Classification of Onion Services for Proactive Cyber Threat
Intelligence Using Explainable Deep Learning" (Moraliyage et al., 2022)

The screen figure has four columns: saliency | Grad-CAM | DeBERTa attention |
Bahdanau attention. The Bahdanau attention is read from the Keras model itself
and shows which tokens the model actually used to classify; models trained
without that layer are handled gracefully.

Requirements:
    - a trained model in results/training/multimodal*/
    - screenshots in the Rico dataset folder
    - reviews in data/reviews_processed/
    - shap >= 0.45.0
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

import torch
from transformers import AutoTokenizer, AutoModel

import shap
from skimage.segmentation import slic

from tqdm import tqdm

from multimodal.common.paths import (
    CONSOLIDATED_SENTIMENT_FILE,
    EMBEDDINGS_CACHE_DIR,
    RESULTS_EXPLANATIONS_MULTIMODAL,
    RESULTS_EXPLANATIONS_MULTIMODAL_V2,
    RESULTS_TRAINING_MULTIMODAL,
    RESULTS_TRAINING_MULTIMODAL_V2,
    REVIEWS_PROCESSED_DIR,
    RICO_DIR,
    RICO_HIERARCHIES_DIR,
)
from multimodal.common.cam_metrics import cam_mult_confidence, road_score
from multimodal.common.nlg import generate_app_text, generate_screen_text
from multimodal.common.semantic_saliency import rank_elements_by_saliency

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================
IMG_SIZE = 224
MAX_TEXT_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBERTA_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
REVIEWS_FOLDER = str(REVIEWS_PROCESSED_DIR)
IMAGES_FOLDER = str(RICO_DIR / "screenshots")
SENTIMENT_MAPPING_FILE = str(CONSOLIDATED_SENTIMENT_FILE)
ASPECT = "interface"
SHAP_MAX_EVALS = 500

# Optional Grad-CAM enhancements (configured via YAML)
GRADCAM_SMOOTHING = False         # if True, use augmentation smoothing (~6x slower)
GRADCAM_METRICS: list[str] = []   # subset of {"cam_mult", "road"}
ROAD_PERCENTILES = [20, 40, 60, 80]

# Canonical app list inherited from another multimodal run (mode 4, canonical
# scope). When set via YAML (``apps_from_multimodal_run``), mode 4 skips the scope
# prompt and processes exactly the ``package_name`` values listed in the
# ``app_explanations.json`` of the referenced run, filtered by the apps present in
# the CSV of this model. Mirrors the mechanism in explainability/unimodal.py (the
# APPS_FROM_MULTIMODAL_RUN variable), which allows an app-level intersection
# between runs of different pipelines over the same set of apps.
APPS_FROM_MULTIMODAL_RUN: str | None = None

# Negation words used to detect bigrams
NEGATION_WORDS = {
    "no", "not", "don't", "dont", "doesn't", "didn't", "isn't", "wasn't",
    "aren't", "won't", "can't", "without", "never", "hardly", "barely"
}

# Stopwords filtered out of the attention visualization
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "also", "would",
    "could", "may", "much", "get", "got", "go", "went", "well", "really", "even",
    "still", "every", "many", "way", "like", "one", "two", "make", "made", "use",
    "used", "lot", "thing", "things", "something", "anything", "nothing"
}

# Copula/pronoun contractions that must be filtered as stopwords (the STOPWORDS
# filter only strips apostrophes at the edges, so "it's" would slip through).
# Contracted negations (don't, can't, won't, isn't...) are not included: they are
# kept for the sentiment analysis and for the negation-bigram logic.
CONTRACTIONS = {
    "it's", "that's", "there's", "here's", "he's", "she's", "what's", "who's",
    "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "we're", "we've",
    "they're", "they've", "let's", "that'll",
}

# Non-discriminative domain terms: they show up in almost every app-store review
# without separating good from bad. This is a choice specific to this dataset, not
# a canonical list from the literature.
DOMAIN_STOPWORDS = {"app", "apps", "application"}


def _is_filtered_word(word: str) -> bool:
    """True when the word must be dropped from the attention/textual SHAP.

    Dropped: empty strings, pure numbers, stopwords and non-negation contractions.
    The inner apostrophe is preserved in the comparison (e.g. "it's"); only edge
    punctuation is stripped, consistently with the rest of the pipeline.
    """
    key = word.lower().strip('.,!?;:()[]"\'')
    return (
        (not key)
        or key.isdigit()
        or key in STOPWORDS
        or key in CONTRACTIONS
        or key in DOMAIN_STOPWORDS
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_original_image(image_path):
    """Load the image in RGB at its original resolution (uint8) for display."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image_path(image_path, img_size=IMG_SIZE):
    """Load and preprocess an image for the model."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    return img


def merge_subtokens_and_attention(tokens, att):
    """
    Convert the DeBERTa subword tokens into readable words and aggregate the
    attention of every sub-token.

    Returns a list of (word, importance).
    """
    words = []
    weights = []

    current_word = ""
    current_weight = 0.0

    for tok, w in zip(tokens, att):
        # Start of the sequence: skip it.
        if tok in ["[CLS]", "<s>"]:
            continue
        # The first separator marks the end of the review segment; stop here so the
        # aspect segment (text_pair="interface") and the padding are not included in
        # the word aggregation.
        if tok in ["[SEP]", "</s>", "[PAD]", "<pad>"]:
            break

        # Subword normalization (covers BERT / SentencePiece / GPT styles)
        t = tok.replace("##", "").replace("▁", "").replace("Ġ", "").strip()

        # Decide whether this is a continuation of the previous token
        is_subtoken = tok.startswith("##") or (
            not tok.startswith("▁") and current_word != "" and not tok.startswith("[")
        )

        if is_subtoken:
            current_word += t
            current_weight += w
        else:
            if current_word != "":
                words.append(current_word)
                weights.append(current_weight)
            current_word = t
            current_weight = w

    # Last pending token
    if current_word:
        words.append(current_word)
        weights.append(current_weight)

    # Filter out stopwords, contractions and numbers.
    filtered = [(w, wt) for w, wt in zip(words, weights)
                if not _is_filtered_word(w)]
    if filtered:
        words, weights = zip(*filtered)
        words, weights = list(words), list(weights)
    else:
        return []

    # Normalize the weights
    total = sum(weights) + 1e-12
    weights = [w / total for w in weights]

    return list(zip(words, weights))


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
    """Select and load trained models from the results folders."""

    def __init__(self, base_paths=None):
        if base_paths is None:
            self.base_paths = [
                str(RESULTS_TRAINING_MULTIMODAL),
                str(RESULTS_TRAINING_MULTIMODAL_V2),
            ]
        elif isinstance(base_paths, str):
            self.base_paths = [base_paths]
        else:
            self.base_paths = list(base_paths)

    def list_available_models(self):
        """Look for trained models and return a list of dicts.

        Includes:
          - ``resultados_*`` folders (single hold-out trainings).
          - ``cv_*/best_fold`` folders (copy of the best fold, marked with a trophy).
          - ``cv_*/fold_N`` subfolders (individual folds).

        Searches both the ``multimodal/`` and ``multimodal_v2/`` result dirs.
        """
        models = []

        for base_path in self.base_paths:
            if not os.path.isdir(base_path):
                continue
            variant = os.path.basename(base_path)  # "multimodal" or "multimodal_v2"

            # Hold-out mode (resultados_*)
            for folder in sorted(glob.glob(os.path.join(base_path, "resultados_*")), reverse=True):
                display = f"[{variant}] {os.path.basename(folder)}"
                info = self._build_model_info(folder, display)
                if info is not None:
                    models.append(info)

            # Cross-validation mode (cv_*/best_fold and cv_*/fold_N)
            for cv_folder in sorted(glob.glob(os.path.join(base_path, "cv_*")), reverse=True):
                if not os.path.isdir(cv_folder):
                    continue
                cv_name = os.path.basename(cv_folder)

                best_path = os.path.join(cv_folder, "best_fold")
                if os.path.isdir(best_path):
                    info = self._build_model_info(best_path, f"[{variant}] {cv_name}/best_fold 🏆")
                    if info is not None:
                        models.append(info)

                for sub in sorted(os.listdir(cv_folder)):
                    if sub.startswith("fold_"):
                        fold_path = os.path.join(cv_folder, sub)
                        info = self._build_model_info(fold_path, f"[{variant}] {cv_name}/{sub}")
                        if info is not None:
                            models.append(info)

        return models

    def _build_model_info(self, folder, display_name):
        """Build the info dict of a model saved in `folder` (None when absent)."""
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
        """Interactive menu for model selection."""
        models = self.list_available_models()

        if not models:
            print(f"ERROR: no trained model found in: {self.base_paths}")
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

        choice = input("Choice (1 or 2, 0 to go back) [2]: ").strip() or "2"
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
        print("Model loaded successfully.")
        return model


# =============================================================================
# CLASS: TestDataLoader
# =============================================================================

class TestDataLoader:
    """
    Load the test data from the classification CSV and the embedding cache, and
    iterate over samples so explanations can be generated.
    """

    EMBEDDINGS_CACHE = str(EMBEDDINGS_CACHE_DIR)

    def __init__(self, model_info):
        self.model_info = model_info
        self.classification_df = None
        self.embeddings_cache = {}
        self.reviews_cache = {}  # Reviews cached per package_name
        self.reviews_index = None  # Index of package -> reviews file
        self._load_classification_csv()
        self._build_reviews_index()  # Pre-build the reviews index

    def _load_classification_csv(self):
        """Load the classification CSV of the selected model."""
        csv_path = self.model_info['classification_csv']
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Classification CSV not found: {csv_path}")

        full_df = pd.read_csv(csv_path)
        print(f"CSV loaded: {len(full_df)} screen-review pairs")

        # Keep the full DataFrame to reach every embedding_idx of a screen
        self.full_classification_df = full_df

        # For iteration, keep one entry per screen (the first occurrence)
        self.classification_df = full_df.drop_duplicates(
            subset=['package_name', 'numero_da_tela']
        ).reset_index(drop=True)
        print(f"Unique screens: {len(self.classification_df)}")

        # Average number of reviews per screen
        reviews_per_screen = full_df.groupby(['package_name', 'numero_da_tela']).size()
        print(f"Reviews per screen: min={reviews_per_screen.min()}, max={reviews_per_screen.max()}, mean={reviews_per_screen.mean():.1f}")

    def _build_reviews_index(self):
        """
        Build the package_name -> app_id index from the mapping file, which is much
        faster than scanning every reviews file.
        """
        print("Loading the reviews mapping...")
        self.reviews_index = {}

        if not os.path.exists(SENTIMENT_MAPPING_FILE):
            print(f"WARNING: mapping file not found: {SENTIMENT_MAPPING_FILE}")
            return

        try:
            df_mapping = pd.read_csv(SENTIMENT_MAPPING_FILE)
            for _, row in df_mapping.iterrows():
                app_id = str(row['app_id']).zfill(4)  # Ensure 4 digits (0002, 0004, ...)
                package_name = row['package_name']
                self.reviews_index[package_name] = app_id

            print(f"Mapping loaded: {len(self.reviews_index)} apps")
        except Exception as e:
            print(f"ERROR loading the mapping: {e}")

    def _get_safe_package_name(self, package_name):
        """Convert the package name into a filesystem-safe name."""
        return package_name.replace(".", "_")

    def _load_embedding_for_package(self, package_name):
        """Load the embeddings of an app from the cache."""
        if package_name in self.embeddings_cache:
            return self.embeddings_cache[package_name]

        safe_name = self._get_safe_package_name(package_name)
        emb_path = os.path.join(self.EMBEDDINGS_CACHE, f"{safe_name}.npy")

        if not os.path.exists(emb_path):
            print(f"WARNING: embeddings not found for {package_name}")
            return None

        # Use memory-mapping for efficiency
        self.embeddings_cache[package_name] = np.load(emb_path, mmap_mode='r')
        return self.embeddings_cache[package_name]

    def get_sample(self, idx):
        """
        Return a complete sample for the explanation pipeline.

        Returns:
            dict with:
                - image_path: path to the image
                - image: preprocessed array (224, 224, 3)
                - embedding: array (128, 768)
                - package_name: str
                - screen_id: str
                - true_label: int (0 or 1)
                - true_class: str ('ruim' or 'bom')
                - predicted_class: str
                - is_correct: bool
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

        # Load the embedding and the reviews using embedding_idx from the CSV
        package_name = row['package_name']
        embeddings = self._load_embedding_for_package(package_name)

        # Look up the embedding_idx values of this screen in the full DF, top 5 by confidence
        screen_rows = self.full_classification_df[
            (self.full_classification_df['package_name'] == package_name) &
            (self.full_classification_df['numero_da_tela'].astype(str) == screen_id)
        ]
        if len(screen_rows) > 0:
            top5 = screen_rows.nlargest(5, 'confianca')
            all_emb_indices = top5['embedding_idx'].tolist()
        else:
            all_emb_indices = [0]

        # Use the highest-confidence embedding for the visual prediction (saliency/Grad-CAM)
        emb_idx = int(all_emb_indices[0])
        if embeddings is None:
            embedding = np.zeros((MAX_TEXT_LENGTH, 768), dtype=np.float32)
        elif emb_idx < len(embeddings):
            embedding = embeddings[emb_idx].astype(np.float32)
        else:
            embedding = np.zeros((MAX_TEXT_LENGTH, 768), dtype=np.float32)

        # Map the label
        true_label = 1 if row['label_real'] == 'bom' else 0

        # Load the actual review texts (enough of them to cover every index)
        max_idx = max(all_emb_indices) + 1 if all_emb_indices else 1
        all_review_texts = self._load_review_texts(package_name, max_texts=max(50, max_idx))

        # Select the specific reviews that were paired with this screen
        screen_reviews = []
        if all_review_texts:
            for idx in all_emb_indices:
                if idx < len(all_review_texts):
                    screen_reviews.append(all_review_texts[idx])

        if screen_reviews:
            review_text = " ".join(screen_reviews[:3])
        elif all_review_texts:
            review_text = " ".join(all_review_texts[:3])
            screen_reviews = all_review_texts[:3]
        else:
            review_text = f"App {package_name}"

        # Mean probability over the pairs of this screen
        avg_prob = float(screen_rows['probabilidade'].mean()) if len(screen_rows) > 0 else float(row['probabilidade'])

        return {
            'image_path': image_path,
            'image': image,
            'embedding': embedding,
            'package_name': package_name,
            'screen_id': screen_id,
            'true_label': true_label,
            'true_class': row['label_real'],
            'predicted_class': row['predito'],
            'is_correct': row['acerto'],
            'app_name': row.get('app', package_name),
            'category': row.get('categoria', 'Unknown'),
            'rating': row.get('rating', 0),
            'review_text': review_text,
            'review_texts': screen_reviews,
            'all_review_texts': all_review_texts or [],
            'embedding_indices': all_emb_indices,
            'avg_probability': avg_prob,
            'num_reviews': len(all_emb_indices)
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

    def _load_review_texts(self, package_name, max_texts=10):
        """
        Load the actual review texts of the app, using the pre-built index for
        direct access to the right file.
        """
        # Check the cache first
        if package_name in self.reviews_cache:
            return self.reviews_cache[package_name][:max_texts]

        # Look up the app_id in the index
        app_id = self.reviews_index.get(package_name)
        if app_id is None:
            return None

        # Build the file path
        reviews_file = os.path.join(REVIEWS_FOLDER, f"app_reviews_{app_id}_with_aspects.csv")

        if not os.path.exists(reviews_file):
            return None

        try:
            df = pd.read_csv(reviews_file)

            # Filter by package_name (the file may hold several apps)
            if 'package_name' in df.columns:
                df_app = df[df['package_name'] == package_name]
            else:
                df_app = df

            if len(df_app) == 0:
                return None

            # Extract and cache the texts (uncapped, so embedding_idx always resolves)
            texts = df_app['sentence'].dropna().astype(str).tolist()
            if texts:
                self.reviews_cache[package_name] = texts
                return texts[:max_texts]

        except Exception as e:
            print(e)
            pass

        return None

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
    Produce visual explanations with Grad-CAM and saliency maps, handling the
    nested MobileNetV2 backbone of the multimodal architecture.

    Two modes are supported:
    - Saliency map: gradients with respect to the input image
    - Real Grad-CAM: gradients with respect to the convolutional feature maps
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
        Find the nested MobileNetV2 and its last convolutional layer. MobileNetV2
        is a functional sublayer of the main model.
        """
        # Step 1: find the nested MobileNetV2
        for layer in self.model.layers:
            if 'mobilenet' in layer.name.lower():
                self.mobilenet_layer = layer
                break

        if self.mobilenet_layer is None:
            raise ValueError("MobileNetV2 not found in the model. Check whether the "
                             "model was trained with the expected architecture.")

        # Step 2: find the last convolutional layer inside MobileNetV2. 'out_relu'
        # and 'Conv_1' are preferred, as they are the last ones before the pooling
        preferred_layers = ['out_relu', 'Conv_1_bn', 'Conv_1', 'block_16_project']

        for preferred in preferred_layers:
            try:
                layer = self.mobilenet_layer.get_layer(preferred)
                self.target_conv_name = preferred
                break
            except:
                continue

        # Fallback: look for any 4D layer
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
        Build the model that exposes the intermediate feature maps.

        Since MobileNetV2 is nested and carries its own Input, the input tensor of
        the GlobalAveragePooling2D layer must be used, because that one is
        connected to the graph of the main model.
        """
        # Find the image_pooling layer (GlobalAveragePooling2D). Its input is the
        # MobileNetV2 output as connected to the main graph.
        try:
            pooling_layer = self.model.get_layer('image_pooling')
            # The pooling input is the (7,7,1280) feature-map tensor on the graph
            conv_output_tensor = pooling_layer.input

            # Build a model that returns the feature maps and the prediction
            self.grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[conv_output_tensor, self.model.output]
            )

            print(f"  - Grad model built from the image_pooling input")
            print(f"  - Feature maps shape: {conv_output_tensor.shape}")

        except Exception as e:
            print(f"ERROR building grad_model: {e}")
            print("Falling back to the alternative approach...")

            # Alternative approach: use the model directly with GradientTape
            self.grad_model = None

    def compute_saliency(self, image_array, text_embedding):
        """
        Compute a saliency map from the gradients with respect to the input image,
        which is the more robust approach for models with a frozen backbone.

        Args:
            image_array: (1, 224, 224, 3) preprocessed image
            text_embedding: (1, 128, 768) DeBERTa embeddings

        Returns:
            heatmap: (224, 224) heatmap normalized to [0, 1]
            prediction: float, predicted probability
        """
        # Convert to a variable tensor so the gradients can be tracked
        img_tensor = tf.Variable(image_array, dtype=tf.float32)
        txt_tensor = tf.convert_to_tensor(text_embedding, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            prediction = self.model([img_tensor, txt_tensor], training=False)
            loss = prediction[:, 0]

        # Gradients with respect to the image
        grads = tape.gradient(loss, img_tensor)

        if grads is None:
            print("WARNING: null gradients - falling back to a uniform heatmap")
            return np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32) * 0.5, float(prediction[0, 0])

        # Reduce to a 2D heatmap (mean absolute value over the RGB channels)
        heatmap = tf.reduce_mean(tf.abs(grads[0]), axis=-1)

        # Normalize
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / (max_val + 1e-8)
        else:
            heatmap = tf.zeros_like(heatmap)

        # Blur it for a cleaner visualization
        heatmap_np = heatmap.numpy()
        heatmap_np = cv2.GaussianBlur(heatmap_np, (15, 15), 0)

        # Renormalize after the blur
        if heatmap_np.max() > 0:
            heatmap_np = heatmap_np / (heatmap_np.max() + 1e-8)

        return heatmap_np, float(prediction[0, 0])

    def compute_real_gradcam(self, image_array, text_embedding):
        """
        Compute the real Grad-CAM from the feature maps of the last convolutional
        layer.

        Grad-CAM = ReLU(sum(alpha_c * A_c)), where alpha_c = GAP(dY/dA_c) are the
        per-channel importance weights.

        Args:
            image_array: (1, 224, 224, 3) preprocessed image
            text_embedding: (1, 128, 768) DeBERTa embeddings

        Returns:
            heatmap: (224, 224) heatmap normalized to [0, 1]
            prediction: float, predicted probability
        """
        if self.grad_model is None:
            # Fall back to SmoothGrad when grad_model could not be built
            print("WARNING: grad_model unavailable, using the SmoothGrad fallback")
            return self._compute_saliency_fallback(image_array, text_embedding)

        img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        txt_tensor = tf.convert_to_tensor(text_embedding, dtype=tf.float32)

        # The default GradientTape (watch_accessed_variables=True) auto-watches every
        # trainable weight of the model. That is required because the backprop from
        # ``loss`` to ``conv_outputs`` goes through the layers after out_relu
        # (image_pooling -> fusion -> dense head), whose weights must be tracked for
        # the gradient flow to work.
        with tf.GradientTape() as tape:
            conv_outputs, prediction = self.grad_model(
                [img_tensor, txt_tensor], training=False
            )
            loss = prediction[:, 0]  # Score of the positive class

        # Gradients with respect to the feature maps (1, 7, 7, 1280)
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("WARNING: null gradients on the feature maps, using the SmoothGrad fallback")
            return self._compute_saliency_fallback(image_array, text_embedding)

        # Per-channel alpha weights (global average pooling of the gradients)
        # Shape: (1, 1280)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        # Weight the feature maps by the alphas
        # conv_outputs: (1, 7, 7, 1280)
        # pooled_grads: (1, 1280)
        conv_out = conv_outputs[0]  # (7, 7, 1280)
        weights = pooled_grads[0]   # (1280,)

        # Multiply each channel by its weight and sum
        heatmap = tf.reduce_sum(conv_out * weights, axis=-1)  # (7, 7)

        # Apply ReLU: Grad-CAM only considers positive activations
        heatmap = tf.nn.relu(heatmap)

        # Normalize to [0, 1]
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / (max_val + 1e-8)
        else:
            # Everything is zero, so fall back
            print("WARNING: zeroed heatmap, using the SmoothGrad fallback")
            return self._compute_saliency_fallback(image_array, text_embedding)

        # Resize from 7x7 to 224x224
        heatmap_np = heatmap.numpy()
        heatmap_resized = cv2.resize(heatmap_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # Blur it slightly for a cleaner visualization
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (9, 9), 0)

        # Renormalize after the resize and the blur
        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / (heatmap_resized.max() + 1e-8)

        return heatmap_resized, float(prediction[0, 0])

    def _compute_saliency_fallback(self, image_array, text_embedding):
        """
        Fallback: SmoothGrad for the cases where Grad-CAM fails. Noise is added
        several times and the gradients are averaged.
        """
        n_samples = 20
        noise_level = 0.1

        img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        txt_tensor = tf.convert_to_tensor(text_embedding, dtype=tf.float32)

        accumulated_grads = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        for _ in range(n_samples):
            # Add Gaussian noise
            noise = tf.random.normal(img_tensor.shape, stddev=noise_level)
            noisy_input = tf.Variable(img_tensor + noise)

            with tf.GradientTape() as tape:
                tape.watch(noisy_input)
                prediction = self.model([noisy_input, txt_tensor], training=False)
                loss = prediction[:, 0]

            grads = tape.gradient(loss, noisy_input)
            if grads is not None:
                # Mean over the RGB channels
                grad_map = tf.reduce_mean(tf.abs(grads[0]), axis=-1).numpy()
                accumulated_grads += grad_map

        # Mean over the samples
        heatmap = accumulated_grads / n_samples

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / (heatmap.max() + 1e-8)

        # Smooth
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        if heatmap.max() > 0:
            heatmap = heatmap / (heatmap.max() + 1e-8)

        # Get the prediction
        pred = self.model([img_tensor, txt_tensor], training=False)

        return heatmap, float(pred[0, 0])

    def compute_real_gradcam_smoothed(self, image_array, text_embedding, n_aug=6):
        """Augmentation smoothing for multimodal Grad-CAM.

        Applies n_aug TTA on the *image* (text stays fixed), computes real Grad-CAM
        on each, un-flips when needed, and averages the heatmaps.
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
            hm, pred = self.compute_real_gradcam(aug_img, text_embedding)
            if "flip" in tag:
                hm = hm[:, ::-1].copy()
            heatmaps.append(hm)
            preds.append(pred)

        mean_heatmap = np.mean(np.stack(heatmaps, axis=0), axis=0)
        if mean_heatmap.max() > 0:
            mean_heatmap = mean_heatmap / (mean_heatmap.max() + 1e-8)
        return mean_heatmap, float(np.mean(preds))

    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        """
        Build the visualization with the heatmap overlaid on the image.

        Any image size is accepted; the heatmap is resized to match, so the overlay
        can be drawn over the high-resolution screenshot.

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
# CLASS: TextAttentionExplainer
# =============================================================================

class TextAttentionExplainer:
    """
    Extract the DeBERTa attention weights for the textual explanation.
    """

    def __init__(self, device=DEVICE):
        self.device = device
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load_model(self):
        """Load DeBERTa (lazy loading)."""
        if self._loaded:
            return

        print(f"\nLoading DeBERTa for attention extraction ({self.device})...")
        self.tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL)
        self.model = AutoModel.from_pretrained(
            DEBERTA_MODEL,
            output_attentions=True
        ).to(self.device)
        self.model.eval()
        self._loaded = True
        print("DeBERTa loaded.")

    def unload_model(self):
        """Unload DeBERTa to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        self._loaded = False

    @torch.no_grad()
    def extract_attention(self, text, max_length=MAX_TEXT_LENGTH):
        """
        Extract the DeBERTa attention weights.

        Args:
            text: input text
            max_length: maximum sequence length

        Returns:
            tokens: list of token strings
            attention: array of attention weights
        """
        self.load_model()

        # Tokenize
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Forward pass with attention
        outputs = self.model(**encoded, output_attentions=True)

        # Attention of the last layer: (batch, heads, seq, seq)
        attention_last_layer = outputs.attentions[-1]

        # Mean over the heads: (batch, seq, seq)
        attention_mean_heads = attention_last_layer.mean(dim=1)

        # CLS -> tokens pattern: (seq,)
        cls_attention = attention_mean_heads[0, 0, :].cpu().numpy()

        # Tokens
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

        return tokens, cls_attention

    def get_word_attention(self, text, max_length=MAX_TEXT_LENGTH):
        """
        Extract the attention and convert it to word level.

        Returns:
            word_attention: list of (word, importance) sorted by importance
        """
        tokens, attention = self.extract_attention(text, max_length)
        word_attention = merge_subtokens_and_attention(tokens, attention)

        # Sort by importance
        word_attention_sorted = sorted(word_attention, key=lambda x: x[1], reverse=True)

        return word_attention_sorted

    @torch.no_grad()
    def aggregate_attention_for_app(self, reviews_list, top_k=15, max_length=MAX_TEXT_LENGTH):
        """
        Aggregate the attention of several reviews into an app-level explanation.

        Args:
            reviews_list: list of review strings
            top_k: number of top words to return
            max_length: maximum sequence length

        Returns:
            top_words: list of the top_k (word, importance) pairs
            all_word_attention: dict with every word and its importance
        """
        self.load_model()

        word_importance = {}

        for review in reviews_list:
            try:
                word_attention = self.get_word_attention(review, max_length)

                for word, weight in word_attention:
                    word_lower = word.lower()
                    if len(word_lower) > 1:  # Ignore single-character tokens
                        if word_lower not in word_importance:
                            word_importance[word_lower] = []
                        word_importance[word_lower].append(weight)
            except Exception as e:
                continue

        # Aggregate: mean importance per word
        aggregated = {w: np.mean(scores) for w, scores in word_importance.items()}

        # Sort by importance
        sorted_words = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)

        return sorted_words[:top_k], aggregated


# =============================================================================
# CLASS: BahdanauAttentionExplainer
# =============================================================================

class BahdanauAttentionExplainer:
    """
    Extract the Bahdanau attention weights from the trained Keras model, which show
    which tokens the model actually used to classify, unlike DeBERTa, which reports
    a generic importance.

    Only works with models that carry an 'attention_softmax' layer; for models
    without it, the extraction returns None.
    """

    def __init__(self, model):
        self.model = model
        self.attention_model = None
        self.available = self._check_availability()

        if self.available:
            self._build_attention_model()

    def _check_availability(self):
        """Check whether the model has a Bahdanau attention layer."""
        try:
            self.model.get_layer('attention_softmax')
            return True
        except ValueError:
            print("The model has no 'attention_softmax' (Bahdanau) layer.")
            print("Bahdanau explainability disabled.")
            return False

    def _build_attention_model(self):
        """Build the submodel that returns the prediction plus the attention weights."""
        attention_layer = self.model.get_layer('attention_softmax')
        self.attention_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[self.model.output, attention_layer.output]
        )
        print("Bahdanau attention submodel built successfully.")

    def extract_attention(self, image_batch, text_embedding_batch):
        """
        Extract the Bahdanau attention weights for a batch.

        Args:
            image_batch: (batch, 224, 224, 3)
            text_embedding_batch: (batch, 128, 768)

        Returns:
            prediction: float, probability of the positive class
            attention_weights: (128,), weight of each token
        """
        if not self.available:
            return None, None

        prediction, attention_weights = self.attention_model(
            [image_batch, text_embedding_batch], training=False
        )

        pred = float(prediction.numpy().squeeze())
        weights = attention_weights.numpy().squeeze()  # (128, 1) → (128,)
        if weights.ndim > 1:
            weights = weights.squeeze(-1)  # drop the last dimension when it is (128, 1)

        return pred, weights

    def get_word_attention(self, image_batch, text_embedding_batch, tokenizer, text):
        """
        Extract the Bahdanau attention and map it onto words.

        Args:
            image_batch: (1, 224, 224, 3)
            text_embedding_batch: (1, 128, 768)
            tokenizer: the DeBERTa tokenizer (to map tokens onto words)
            text: original review text

        Returns:
            word_attention: list of (word, importance) sorted by importance
        """
        if not self.available:
            return []

        _, attention_weights = self.extract_attention(image_batch, text_embedding_batch)
        if attention_weights is None:
            return []

        # Tokenize the text so the weights can be mapped onto words
        encoded = tokenizer(
            text,
            text_pair=ASPECT,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_TEXT_LENGTH
        )
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

        # merge_subtokens_and_attention converts the subtokens back into words
        word_attention = merge_subtokens_and_attention(tokens, attention_weights)

        # Sort by importance
        word_attention_sorted = sorted(word_attention, key=lambda x: x[1], reverse=True)
        return word_attention_sorted

    def aggregate_attention_for_app(self, samples, tokenizer, top_k=15):
        """
        Aggregate the Bahdanau attention of several samples into an app-level
        explanation.

        Args:
            samples: list of dicts with 'image', 'embedding' and 'text'
            tokenizer: the DeBERTa tokenizer
            top_k: number of top words to return

        Returns:
            top_words: list of (word, importance)
            all_word_attention: dict with every word and its importance
        """
        if not self.available:
            return [], {}

        word_importance = {}

        for sample in samples:
            try:
                image_batch = np.expand_dims(sample['image'], 0)
                text_batch = np.expand_dims(sample['embedding'], 0)
                text = sample.get('text', '')

                if not text:
                    continue

                word_attention = self.get_word_attention(
                    image_batch, text_batch, tokenizer, text
                )

                for word, weight in word_attention:
                    word_lower = word.lower()
                    if len(word_lower) > 1:
                        if word_lower not in word_importance:
                            word_importance[word_lower] = []
                        word_importance[word_lower].append(weight)
            except Exception:
                continue

        aggregated = {w: np.mean(scores) for w, scores in word_importance.items()}
        sorted_words = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)

        return sorted_words[:top_k], aggregated


# =============================================================================
# CLASS: SHAPExplainer
# =============================================================================

class SHAPExplainer:
    """
    Produce multimodal SHAP explanations that measure how much each image region
    and each word of the text contributes to the model prediction.

    Masking is used (blur for the image, token removal for the text) and the impact
    of each part on the model output is measured.
    """

    def __init__(self, keras_model, text_explainer):
        """
        Args:
            keras_model: the trained multimodal Keras model
            text_explainer: a TextAttentionExplainer instance (provides the tokenizer
                and DeBERTa)
        """
        self.keras_model = keras_model
        self.text_explainer = text_explainer
        # Make sure DeBERTa is loaded
        self.text_explainer.load_model()

    @torch.no_grad()
    def _generate_embeddings_batch(self, texts, batch_size=8):
        """
        Generate the DeBERTa embeddings for a batch of texts, with the same
        text_pair=ASPECT logic used by the training pipeline.

        Args:
            texts: list of strings (they may be masked by SHAP)
            batch_size: mini-batch size for inference

        Returns:
            np.array of shape (N, MAX_TEXT_LENGTH, 768)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                text_str = str(text).strip()

                # Empty or whitespace-only text -> zero embedding
                if not text_str:
                    batch_embeddings.append(
                        np.zeros((MAX_TEXT_LENGTH, 768), dtype=np.float32)
                    )
                    continue

                # Tokenize in the ABSA format: [CLS] text [SEP] interface [SEP]
                encoded = self.text_explainer.tokenizer(
                    text_str,
                    text_pair=ASPECT,
                    max_length=MAX_TEXT_LENGTH,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(self.text_explainer.device)
                attention_mask = encoded['attention_mask'].to(self.text_explainer.device)

                outputs = self.text_explainer.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                emb = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                batch_embeddings.append(emb)

            all_embeddings.extend(batch_embeddings)

            # Free GPU memory periodically
            if torch.cuda.is_available() and i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()

        return np.array(all_embeddings, dtype=np.float32)

    @staticmethod
    def _group_subword_shap(tokens, shap_values):
        """
        Group the DeBERTa subwords back into whole words, summing their SHAP values.

        DeBERTa uses SentencePiece: subwords that continue a word do not start with a
        space/underscore (e.g. '<U+2581>annoy', 'ing'). Tokens that start a new word
        begin with U+2581, or are the first word.

        Returns:
            list of (word, shap_value) tuples
        """
        if len(tokens) == 0:
            return []

        grouped = []
        current_word = ""
        current_val = 0.0

        for tok, val in zip(tokens, shap_values):
            tok_str = str(tok).strip()
            if not tok_str:
                continue

            # Special tokens: ignore them
            if tok_str in ('[CLS]', '[SEP]', '[PAD]', '[MASK]', '<s>', '</s>', '<pad>'):
                continue

            # SentencePiece: U+2581 marks the start of a new word. Tokens starting
            # with a regular space are handled the same way.
            is_new_word = tok_str.startswith('\u2581') or tok_str.startswith(' ')

            # Strip the prefix
            clean_tok = tok_str.lstrip('\u2581 ')

            if not clean_tok:
                continue

            if is_new_word and current_word:
                # Store the previous word and start a new one
                grouped.append((current_word, current_val))
                current_word = clean_tok
                current_val = float(val)
            elif not current_word:
                # First token
                current_word = clean_tok
                current_val = float(val)
            else:
                # Continuation of the word (subword)
                current_word += clean_tok
                current_val += float(val)

        # Store the last word
        if current_word:
            grouped.append((current_word, current_val))

        return grouped

    @staticmethod
    def _extract_negation_bigrams(word_shap_pairs):
        """
        Identify negation bigrams (e.g. 'not good', "doesn't work") and combine
        their SHAP values. Returns the bigrams found.

        Args:
            word_shap_pairs: list of (word, shap_value) from _group_subword_shap

        Returns:
            dict: {bigram_str: combined_shap_value}
            set: unigram indices consumed by the bigrams
        """
        bigrams = {}
        consumed = set()

        for j in range(len(word_shap_pairs) - 1):
            w1, v1 = word_shap_pairs[j]
            w2, v2 = word_shap_pairs[j + 1]

            if w1.lower() in NEGATION_WORDS and w2.lower():
                bigram = f"{w1.lower()} {w2.lower()}"
                bigrams[bigram] = float(v1) + float(v2)
                consumed.add(j)
                consumed.add(j + 1)

        return bigrams, consumed

    @staticmethod
    def _build_feature_ranking(word_shap_pairs, top_k=20):
        """
        Build the feature ranking (unigrams + negation bigrams), giving priority to
        the bigrams when they exist.

        Args:
            word_shap_pairs: list of (word, shap_value)
            top_k: number of features in the ranking

        Returns:
            list of (feature, shap_value) sorted by |shap_value| desc
        """
        bigrams, consumed_indices = SHAPExplainer._extract_negation_bigrams(word_shap_pairs)

        # Unigrams not consumed by a bigram
        unigrams = {}
        for j, (word, val) in enumerate(word_shap_pairs):
            if j not in consumed_indices:
                key = word.lower()
                # Sum the values when the same word appears more than once
                if key in unigrams:
                    unigrams[key] += float(val)
                else:
                    unigrams[key] = float(val)

        # Combine unigrams and bigrams
        all_features = {}
        all_features.update(unigrams)
        all_features.update(bigrams)

        # Sort by absolute value
        sorted_features = sorted(all_features.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_features[:top_k]

    def _compute_image_shap_superpixel(self, image, text_embedding, n_segments=50,
                                        n_samples=200, seed=42):
        """
        Compute the SHAP values of the image using superpixel segmentation (SLIC).
        Each superpixel is a binary feature (visible/blurred), and KernelExplainer
        estimates the contribution of each one.

        ``seed`` (default 42) sets ``np.random.seed(seed)`` before
        ``shap.KernelExplainer``, which makes the SHAP heatmap reproducible across
        runs for the same (image, text_embedding). That is essential for the
        analyses that rank components from the SHAP heatmap, which would otherwise
        be unstable.

        Args:
            image: (224, 224, 3) preprocessed image, float32 [0,1]
            text_embedding: (1, 128, 768) text embedding
            n_segments: number of SLIC superpixels
            n_samples: number of samples for KernelExplainer

        Returns:
            shap_heatmap: (224, 224) heatmap normalized to [0,1]
            segments: (224, 224) map of SLIC segments
        """
        # 1. Segment the image with SLIC
        # Convert to uint8, which SLIC handles better
        img_uint8 = (image * 255).astype(np.uint8)
        segments = slic(img_uint8, n_segments=n_segments, compactness=20)
        unique_segments = np.unique(segments)
        n_features = len(unique_segments)

        # 2. Build the blurred image used as baseline
        blurred = cv2.GaussianBlur(image, (31, 31), 0)

        # 3. Prediction function driven by the superpixel mask
        keras_model = self.keras_model

        # Pre-compute one boolean mask per superpixel, to avoid recomputing
        # `segments == seg_id` for every mask.
        segment_masks = [segments == seg_id for seg_id in unique_segments]

        def predict_superpixel(masks):
            """masks: (N, n_features) binary - 1=visible, 0=blurred.

            Batched version: every masked image is built as a single tensor and
            ``keras_model.predict`` is called once with the full batch (TF-Keras
            batches internally, 32 by default). Much faster on GPU than a
            per-sample loop for the usual SHAP settings with nsamples > 100.
            """
            n = len(masks)
            if n == 0:
                return np.array([])

            # Assemble (N, H, W, 3) with the masked images
            batch_imgs = np.broadcast_to(blurred, (n,) + blurred.shape).copy().astype(np.float32)
            for idx, mask in enumerate(masks):
                for j, seg_mask in enumerate(segment_masks):
                    if mask[j] == 1:
                        batch_imgs[idx][seg_mask] = image[seg_mask]

            # Replicate text_embedding (1, 128, 768) into (N, 128, 768)
            text_batch = np.broadcast_to(
                text_embedding, (n,) + text_embedding.shape[1:]
            )

            preds = keras_model.predict(
                [batch_imgs, np.ascontiguousarray(text_batch)], verbose=0,
            )
            return np.asarray(preds).flatten()

        # 4. Baseline: every superpixel blurred (an image with no information)
        background = np.zeros((1, n_features))

        # 5. KernelExplainer (fixed seed for reproducibility)
        np.random.seed(seed)
        explainer = shap.KernelExplainer(predict_superpixel, background)
        instance = np.ones((1, n_features))
        shap_values = explainer.shap_values(instance, nsamples=n_samples, silent=True)

        # 6. Map the SHAP values back onto pixels
        if isinstance(shap_values, list):
            sv = shap_values[0].flatten()
        else:
            sv = shap_values.flatten()

        shap_heatmap = np.zeros(image.shape[:2], dtype=np.float32)
        for i, seg_id in enumerate(unique_segments):
            if i < len(sv):
                shap_heatmap[segments == seg_id] = abs(sv[i])

        # 7. Total contribution of the image (sum of |SHAP| over the superpixels)
        image_shap_total = float(np.sum(np.abs(sv)))

        # 8. Normalize
        if shap_heatmap.max() > 0:
            shap_heatmap = shap_heatmap / (shap_heatmap.max() + 1e-8)

        return shap_heatmap, segments, image_shap_total

    def _make_text_predict_fn(self, image_batch):
        """
        Build the prediction function for the textual SHAP, with a fixed image: the
        image is held constant while the text varies.
        """
        keras_model = self.keras_model
        generate_emb = self._generate_embeddings_batch

        def predict(texts):
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            texts = [str(t) for t in texts]
            n = len(texts)
            embeddings = generate_emb(texts)
            img_batch = np.tile(image_batch, (n, 1, 1, 1))
            preds = keras_model.predict([img_batch, embeddings], verbose=0)
            return preds
        return predict

    def explain_instance(self, image_path, review_texts, output_dir, filename_prefix,
                         max_evals=SHAP_MAX_EVALS):
        """
        Produce the multimodal SHAP explanation of one instance (screen + reviews).
        - Image: SHAP over superpixels (the embedding of the first review is used
          as reference)
        - Text: SHAP computed per review, then aggregated via mean(|SHAP|)

        Args:
            image_path: path to the image
            review_texts: list of individual reviews (up to 5)
            output_dir: output directory
            filename_prefix: prefix of the output file name
            max_evals: maximum number of SHAP evaluations (more = more precise and
                slower)

        Returns:
            dict with the results of the SHAP explanation
        """
        os.makedirs(output_dir, exist_ok=True)

        # Make sure the reviews are a list
        if isinstance(review_texts, str):
            review_texts = [review_texts]
        review_texts = [r for r in review_texts if r and str(r).strip()]
        if not review_texts:
            print(f"ERROR: no valid review text.")
            return None

        # 1. Load and prepare the image
        img = preprocess_image_path(image_path)
        if img is None:
            print(f"ERROR: could not load the image: {image_path}")
            return None

        img_batch = np.expand_dims(img, 0)

        # 2. Embedding of the first review, used as reference for the image SHAP and
        # for the prediction
        ref_text = review_texts[0]
        ref_embedding = self._generate_embeddings_batch([ref_text])  # (1, 128, 768)

        # 3. Base prediction (with the first review)
        prediction = float(self.keras_model.predict([img_batch, ref_embedding], verbose=0)[0, 0])

        # === Image SHAP (SLIC superpixels + KernelExplainer) ===
        print(f"  Computing the image SHAP values (superpixels, nsamples={max_evals})...")
        image_shap_heatmap, _segments, _img_total = self._compute_image_shap_superpixel(
            image=img,
            text_embedding=ref_embedding,
            n_segments=100,
            n_samples=max_evals
        )

        # === Textual SHAP: each review is processed separately, then aggregated ===
        # A regex tokenizer is used so SHAP works with whole words. The inner predict
        # function (_make_text_predict_fn -> _generate_embeddings_batch) already does
        # the DeBERTa tokenization with text_pair=ASPECT when generating embeddings.
        print(f"  Computing the SHAP values of {len(review_texts)} reviews (max_evals={max_evals})...")
        text_masker = shap.maskers.Text(tokenizer=r'\S+')
        text_predict_fn = self._make_text_predict_fn(img_batch)
        text_explainer_obj = shap.Explainer(text_predict_fn, text_masker)

        # Collect the word_shap_pairs of each review
        all_word_shap_pairs = []
        global_feature_shap = {}  # feature -> list of SHAP values

        for rev_idx, review in enumerate(review_texts):
            print(f"    [text SHAP] Review {rev_idx + 1}/{len(review_texts)}...")
            try:
                text_shap_values = text_explainer_obj(
                    [review],
                    max_evals=max_evals,
                    batch_size=5
                )

                tokens = text_shap_values.data[0]
                vals = text_shap_values.values[0]
                if vals.ndim > 1:
                    vals = vals[:, 0]
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.tolist()

                # With the regex tokenizer the tokens are already whole words
                word_pairs = [
                    (str(tok).strip(), float(val))
                    for tok, val in zip(tokens, vals)
                    if str(tok).strip()
                ]
                all_word_shap_pairs.extend(word_pairs)

                # Extract the negation bigrams
                bigrams, consumed = self._extract_negation_bigrams(word_pairs)

                # Accumulate the unigrams (those not consumed by a bigram)
                for j, (word, val) in enumerate(word_pairs):
                    if j not in consumed:
                        key = word.lower()
                        global_feature_shap.setdefault(key, []).append(float(val))

                # Accumulate the bigrams
                for bigram, val in bigrams.items():
                    global_feature_shap.setdefault(bigram, []).append(float(val))

            except Exception as e:
                print(f"    [text SHAP] Error on review {rev_idx + 1}: {e}")

        # 4. Aggregate: mean(|SHAP|) per feature, top-20 ranking
        feature_importance = {
            feat: float(np.mean(np.abs(vals)))
            for feat, vals in global_feature_shap.items()
        }
        feature_mean_signed = {
            feat: float(np.mean(vals))
            for feat, vals in global_feature_shap.items()
        }
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        feature_ranking = sorted_features[:20]

        # 5. Render the combined visualization
        viz_path = os.path.join(output_dir, f"{filename_prefix}_shap_multimodal.png")
        print(f"  Rendering the combined SHAP figure ({len(review_texts)} reviews aggregated)...")

        self._generate_shap_visualization(
            image_path=image_path,
            image_shap_heatmap=image_shap_heatmap,
            feature_ranking=feature_ranking,
            feature_mean_signed=feature_mean_signed,
            prediction=prediction,
            review_texts=review_texts,
            output_path=viz_path
        )

        print(f"  Figure saved: {viz_path}")

        return {
            'visualization_path': viz_path,
            'prediction': prediction,
            'predicted_class': 'good' if prediction > 0.5 else 'bad',
            'confidence': prediction if prediction > 0.5 else 1 - prediction,
            'max_evals': max_evals,
            'num_reviews': len(review_texts),
            'image_path': image_path,
            'feature_ranking': feature_ranking,
            'feature_mean_signed': feature_mean_signed,
            'word_shap_pairs': all_word_shap_pairs
        }

    def _generate_shap_visualization(self, image_path, image_shap_heatmap, feature_ranking,
                                     feature_mean_signed, prediction, review_texts,
                                     output_path):
        """
        Render the combined multimodal SHAP figure at high resolution:
        - Row 1: image SHAP (left) + word barplot (right)
        - Row 2: prediction info
        - Row 3: the original reviews used in the analysis

        Panel titles, axis labels, the legend and the info bar below are
        deliberately in Portuguese: they are rendered into the PNGs, so
        translating them would make regenerated figures disagree with the ones
        already in use. Comments stay in English; figure content does not.
        """
        n_reviews = len(review_texts)

        # Layout: 3 rows (charts | info | reviews)
        # The height of the reviews section scales with the number of reviews
        review_height = max(1.5, n_reviews * 0.6)
        fig = plt.figure(figsize=(16, 9 + review_height))
        gs = GridSpec(3, 2, height_ratios=[5, 0.6, review_height], width_ratios=[1, 1],
                      hspace=0.3)

        # === Row 1, column 1: image SHAP (high resolution) ===
        ax_img = fig.add_subplot(gs[0, 0])

        original = load_original_image(image_path)
        h, w = original.shape[:2]
        heatmap_resized = cv2.resize(image_shap_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(original, 0.5, heatmap_colored, 0.5, 0)

        ax_img.imshow(blended)
        ax_img.set_title("SHAP - Importancia Visual\n(regioes da imagem)", fontsize=12, fontweight='bold')
        ax_img.axis('off')

        # === Row 1, column 2: textual SHAP (mean|SHAP| per word) ===
        ax_text = fig.add_subplot(gs[0, 1])

        if feature_ranking:
            feat_names = [p[0] for p in feature_ranking]
            feat_importance = [p[1] for p in feature_ranking]

            # Color from the mean sign: green contributes to GOOD, red to BAD
            colors = []
            for feat, _ in feature_ranking:
                signed = feature_mean_signed.get(feat, 0)
                colors.append('green' if signed > 0 else 'red')

            ax_text.barh(range(len(feat_names)), feat_importance, color=colors, alpha=0.7)
            ax_text.set_yticks(range(len(feat_names)))
            ax_text.set_yticklabels(feat_names, fontsize=9)
            ax_text.invert_yaxis()
            ax_text.set_xlabel('Mean |SHAP value|', fontsize=10)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='Contribui para BOM'),
                Patch(facecolor='red', alpha=0.7, label='Contribui para RUIM'),
            ]
            ax_text.legend(handles=legend_elements, loc='lower right', fontsize=8)
        else:
            ax_text.text(0.5, 0.5, "Sem dados de texto", ha='center', va='center',
                         fontsize=12, transform=ax_text.transAxes)

        ax_text.set_title(f"SHAP - Importancia Textual\n(agregado de {n_reviews} reviews)",
                          fontsize=12, fontweight='bold')

        # === Row 2: prediction info ===
        ax_info = fig.add_subplot(gs[1, :])
        ax_info.axis('off')

        pred_class = "BOM" if prediction > 0.5 else "RUIM"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        info_text = (
            f"Predicao: {pred_class} (Confianca: {confidence:.1%})    |    "
            f"Probabilidade: {prediction:.4f}    |    "
            f"Metodo: SHAP (Superpixel + Partition)    |    "
            f"Reviews: {n_reviews}"
        )
        ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                     fontsize=12, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7),
                     fontweight='bold')

        # === Row 3: the original reviews ===
        ax_reviews = fig.add_subplot(gs[2, :])
        ax_reviews.axis('off')
        ax_reviews.set_title("Reviews utilizados na analise SHAP", fontsize=11,
                             fontweight='bold', loc='left', pad=4)

        review_display = []
        for i, rev in enumerate(review_texts):
            # Truncate very long reviews so they fit in the figure
            rev_str = str(rev).strip()
            if len(rev_str) > 150:
                rev_str = rev_str[:147] + "..."
            review_display.append(f"[{i+1}] {rev_str}")

        reviews_text = "\n".join(review_display)
        ax_reviews.text(0.02, 0.95, reviews_text, transform=ax_reviews.transAxes,
                        fontsize=8, family='monospace', verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                                  alpha=0.5, edgecolor='gray'))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    def explain_app(self, package_name, samples, output_dir, max_evals=SHAP_MAX_EVALS):
        """
        Produce the app-level aggregated SHAP explanation: each screen is processed
        individually and the results are aggregated, including a global aggregation
        of the textual SHAP (mean|SHAP| per feature).

        Args:
            package_name: package name of the app
            samples: list of dicts (the app samples, from TestDataLoader)
            output_dir: output directory
            max_evals: maximum number of SHAP evaluations per screen

        Returns:
            dict with the aggregated results
        """
        os.makedirs(output_dir, exist_ok=True)

        screen_results = []
        safe_name = package_name.replace('.', '_')

        for i, sample in enumerate(tqdm(samples, desc=f"SHAP for {package_name}")):
            screen_id = sample['screen_id']
            screen_reviews = sample.get('review_texts', [])
            if not screen_reviews:
                all_reviews = sample.get('all_review_texts', [])
                screen_reviews = all_reviews[:5] if all_reviews else [f"App {package_name}"]
            prefix = f"{safe_name}_{screen_id}"

            result = self.explain_instance(
                image_path=sample['image_path'],
                review_texts=screen_reviews[:5],
                output_dir=output_dir,
                filename_prefix=prefix,
                max_evals=max_evals
            )

            if result:
                screen_results.append(result)

        if not screen_results:
            return None

        # Aggregate the predictions
        predictions = [r['prediction'] for r in screen_results]
        app_score = float(np.mean(predictions))
        app_prediction = 'good' if app_score > 0.5 else 'bad'
        app_confidence = app_score if app_score > 0.5 else 1 - app_score

        # === Global aggregation of the textual SHAP ===
        # Collect the word_shap_pairs of every screen and aggregate via mean|SHAP|
        global_feature_shap = {}  # feature -> list of SHAP values
        for result in screen_results:
            word_pairs = result.get('word_shap_pairs', [])
            if not word_pairs:
                continue

            # Extract the negation bigrams of this screen
            bigrams, consumed = self._extract_negation_bigrams(word_pairs)

            # Unigrams that were not consumed
            for j, (word, val) in enumerate(word_pairs):
                if j not in consumed:
                    key = word.lower()
                    global_feature_shap.setdefault(key, []).append(float(val))

            # Bigrams
            for bigram, val in bigrams.items():
                global_feature_shap.setdefault(bigram, []).append(float(val))

        # Compute mean|SHAP| and the mean signed value of each feature
        global_importance = {}
        global_mean_signed = {}
        for feat, vals in global_feature_shap.items():
            global_importance[feat] = float(np.mean(np.abs(vals)))
            global_mean_signed[feat] = float(np.mean(vals))

        # Global top-20 ranking
        sorted_global = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
        global_ranking = sorted_global[:20]

        # Render the aggregated summary
        summary_path = os.path.join(output_dir, f"{safe_name}_app_shap_summary.png")
        self._generate_app_summary_visualization(
            package_name=package_name,
            screen_results=screen_results,
            app_score=app_score,
            app_prediction=app_prediction,
            global_ranking=global_ranking,
            global_mean_signed=global_mean_signed,
            output_path=summary_path
        )

        return {
            'package_name': package_name,
            'num_screens': len(screen_results),
            'app_score': app_score,
            'app_prediction': app_prediction,
            'app_confidence': float(app_confidence),
            'screen_results': screen_results,
            'summary_path': summary_path,
            'prediction_variance': float(np.var(predictions)),
            'global_text_ranking': global_ranking,
            'global_mean_signed': global_mean_signed
        }

    def _generate_app_summary_visualization(self, package_name, screen_results,
                                            app_score, app_prediction,
                                            global_ranking, global_mean_signed,
                                            output_path):
        """
        Render the app-level SHAP summary figure, with the per-screen predictions,
        the global textual SHAP and the confidence distribution.

        Panel titles, axis labels, the legend and the info bar below are
        deliberately in Portuguese: they are rendered into the PNGs, so
        translating them would make regenerated figures disagree with the ones
        already in use. Comments stay in English; figure content does not.
        """
        n_screens = len(screen_results)

        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(1, 2, width_ratios=[1, 1.3])

        # === Chart 1: per-screen predictions ===
        ax1 = fig.add_subplot(gs[0, 0])
        predictions = [r['prediction'] for r in screen_results]
        colors = ['green' if p > 0.5 else 'red' for p in predictions]
        ax1.bar(range(n_screens), predictions, color=colors, alpha=0.7)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=app_score, color='blue', linestyle='-', linewidth=2, alpha=0.7,
                     label=f'Media: {app_score:.3f}')
        ax1.set_xlabel('Tela', fontsize=11)
        ax1.set_ylabel('Probabilidade (bom)', fontsize=11)
        ax1.set_title(f'Predicoes por Tela\n{package_name}', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.legend()

        # === Chart 2: global textual SHAP (aggregated over every screen) ===
        ax2 = fig.add_subplot(gs[0, 1])
        if global_ranking:
            feat_names = [f[0] for f in global_ranking]
            feat_importance = [f[1] for f in global_ranking]
            # Color from the mean sign: green contributes to GOOD, red to BAD
            feat_colors = []
            for feat, _ in global_ranking:
                signed = global_mean_signed.get(feat, 0)
                feat_colors.append('green' if signed > 0 else 'red')

            ax2.barh(range(len(feat_names)), feat_importance, color=feat_colors, alpha=0.7)
            ax2.set_yticks(range(len(feat_names)))
            ax2.set_yticklabels(feat_names, fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlabel('Mean |SHAP value|', fontsize=10)
            ax2.set_title(f'SHAP Global Textual\n(agregado de {n_screens} telas)',
                          fontsize=12, fontweight='bold')

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='Contribui para BOM'),
                Patch(facecolor='red', alpha=0.7, label='Contribui para RUIM'),
            ]
            ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)
        else:
            ax2.text(0.5, 0.5, "Sem dados textuais", ha='center', va='center',
                     fontsize=12, transform=ax2.transAxes)

        # General info
        pred_class = "BOM" if app_score > 0.5 else "RUIM"
        app_conf = app_score if app_score > 0.5 else 1 - app_score

        fig.suptitle(
            f"SHAP Multimodal - Resumo do App | Predicao: {pred_class} ({app_conf:.1%}) | "
            f"{n_screens} telas analisadas",
            fontsize=13, fontweight='bold', y=1.02
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  App summary saved: {output_path}")


# =============================================================================
# CLASS: MultimodalExplanation
# =============================================================================

class MultimodalExplanation:
    """
    Combine the visual and textual explanations into integrated figures, showing
    the saliency map and the real Grad-CAM side by side.
    """

    def __init__(self, model, gradcam_explainer, text_explainer, bahdanau_explainer=None):
        self.model = model
        self.gradcam = gradcam_explainer
        self.text_attn = text_explainer
        self.bahdanau = bahdanau_explainer

    def _get_dummy_embedding(self):
        """Return an empty embedding for the cases with no text."""
        return np.zeros((1, MAX_TEXT_LENGTH, 768), dtype=np.float32)

    def explain_screen(self, image_path, text, text_embedding, true_label,
                       package_name, screen_id, output_dir, review_texts=None,
                       app_name=None, category=None):
        """
        Produce the complete explanation of one screen.

        Args:
            image_path: path to the image
            text: concatenated review text (fallback)
            text_embedding: (128, 768) precomputed embedding
            true_label: true label (0 or 1)
            package_name: package name
            screen_id: screen identifier
            output_dir: output directory
            review_texts: list of individual reviews (optional, preferred)

        Returns:
            explanation: dict with every piece of data of the explanation
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load the image
        image = preprocess_image_path(image_path)
        if image is None:
            return None

        image_batch = np.expand_dims(image, 0)
        text_batch = np.expand_dims(text_embedding, 0)

        # 2. Real Grad-CAM (gradients on the feature maps). The DeBERTa attention,
        #    the saliency map and the score_total component ranking are produced by
        #    the per-review pipeline of the subclass, not here.
        if GRADCAM_SMOOTHING:
            real_gradcam_heatmap, prediction = self.gradcam.compute_real_gradcam_smoothed(
                image_batch, text_batch
            )
        else:
            real_gradcam_heatmap, prediction = self.gradcam.compute_real_gradcam(
                image_batch, text_batch
            )

        # 3. Optional Grad-CAM faithfulness metrics (cam_mult/road).
        cam_metrics: dict = {}
        if GRADCAM_METRICS:
            txt_fixed = text_batch
            predict_fn = lambda x: self.model.predict(
                [x, np.tile(txt_fixed, (x.shape[0], 1, 1))], verbose=0
            )
            if "cam_mult" in GRADCAM_METRICS:
                cam_metrics["cam_mult"] = cam_mult_confidence(image, real_gradcam_heatmap, predict_fn)
            if "road" in GRADCAM_METRICS:
                road_result = road_score(image, real_gradcam_heatmap, predict_fn, ROAD_PERCENTILES)
                cam_metrics["road_mean"] = road_result["mean"]
                cam_metrics["road_per_percentile"] = road_result["per_percentile"]

        # 4. Build the explanation dict; the subclass enriches it further.
        confidence = prediction if prediction > 0.5 else 1 - prediction

        explanation = {
            'package_name': package_name,
            'screen_id': screen_id,
            'image_path': image_path,
            'text': text[:500] if text else "",
            'prediction': float(prediction),
            'predicted_class': 'good' if prediction > 0.5 else 'bad',
            'true_label': int(true_label),
            'true_class': 'good' if true_label == 1 else 'bad',
            'is_correct': (prediction > 0.5) == (true_label == 1),
            'confidence': float(confidence),
            'gradcam_stats': {
                'mean': float(real_gradcam_heatmap.mean()),
                'max': float(real_gradcam_heatmap.max()),
                'std': float(real_gradcam_heatmap.std()),
            },
            'cam_metrics': cam_metrics,
            'review_texts': review_texts or [],
            'app_name': app_name or package_name,
            'category': category or 'Unknown',
        }

        return explanation

    def explain_app(self, package_name, screen_explanations, output_dir,
                    reviews_list=None, all_app_reviews=None):
        """
        Produce the app-level aggregated explanation.

        Args:
            package_name: package name
            screen_explanations: list of screen explanations
            reviews_list: reviews used to compute the textual attention
            all_app_reviews: full list of reviews of the app (for display)
            output_dir: output directory

        Returns:
            app_explanation: dict with the aggregated explanation
        """
        os.makedirs(output_dir, exist_ok=True)

        if not screen_explanations:
            return None

        # 1. Aggregate the heatmaps (weighted by confidence)
        saliency_heatmaps = []
        gradcam_heatmaps = []
        confidences = []
        images = []
        image_paths = []

        for expl in screen_explanations:
            image = preprocess_image_path(expl['image_path'])
            if image is not None:
                images.append(image)
                image_paths.append(expl['image_path'])
                # Recompute the heatmaps
                img_batch = np.expand_dims(image, 0)
                text_batch = self._get_dummy_embedding()

                saliency_hm, _ = self.gradcam.compute_saliency(img_batch, text_batch)
                saliency_heatmaps.append(saliency_hm)

                gradcam_hm, _ = self.gradcam.compute_real_gradcam(img_batch, text_batch)
                gradcam_heatmaps.append(gradcam_hm)

                confidences.append(expl['confidence'])

        aggregated_saliency = None
        aggregated_gradcam = None
        if saliency_heatmaps:
            aggregated_saliency = self.gradcam.aggregate_heatmaps_for_app(saliency_heatmaps, confidences)
        if gradcam_heatmaps:
            aggregated_gradcam = self.gradcam.aggregate_heatmaps_for_app(gradcam_heatmaps, confidences)

        # 2. Aggregate the textual attention
        if reviews_list:
            top_words, _ = self.text_attn.aggregate_attention_for_app(reviews_list, top_k=20)
        else:
            # Aggregate the words of the screen explanations
            word_dict = {}
            for expl in screen_explanations:
                for word, weight in expl.get('top_words', []):
                    if word in word_dict:
                        word_dict[word].append(weight)
                    else:
                        word_dict[word] = [weight]
            # Average and sort
            top_words = sorted(
                [(word, np.mean(weights)) for word, weights in word_dict.items()],
                key=lambda x: x[1],
                reverse=True
            )[:20]

        # 3. Compute the aggregated prediction
        predictions = [expl['prediction'] for expl in screen_explanations]
        app_score = float(np.mean(predictions))
        app_prediction = 'good' if app_score > 0.5 else 'bad'
        app_confidence = app_score if app_score > 0.5 else 1 - app_score

        true_label = screen_explanations[0]['true_label']

        # 4. Render the aggregated figure
        app_viz_path = None
        if images and aggregated_saliency is not None and aggregated_gradcam is not None:
            app_viz_path = os.path.join(output_dir, f"{package_name.replace('.', '_')}_app_explanation.png")
            # Take app_name and category from the screen explanations
            first_expl = screen_explanations[0]
            self._generate_app_visualization(
                representative_image_path=image_paths[0],
                saliency_heatmap=aggregated_saliency,
                gradcam_heatmap=aggregated_gradcam,
                word_attention=top_words[:15],
                app_score=app_score,
                true_label=true_label,
                num_screens=len(screen_explanations),
                output_path=app_viz_path,
                review_texts=all_app_reviews or [],
                app_name=first_expl.get('app_name'),
                category=first_expl.get('category')
            )

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
            'top_influential_words': top_words[:15],
            'screen_predictions': predictions,
            'prediction_variance': float(np.var(predictions)),
            'visualization_path': app_viz_path,
            'summary': self._generate_text_summary(package_name, app_prediction, app_confidence, top_words[:10])
        }

        # NLG: caption / simple / detailed (deterministic, by package_name)
        app_explanation['text_explanation'] = generate_app_text(app_explanation, screen_explanations)

        return app_explanation

    def generate_combined_visualization(self, image_path, saliency_heatmap, gradcam_heatmap,
                                        word_attention, prediction, true_label, output_path,
                                        review_texts=None, app_name=None, category=None,
                                        bahdanau_attention=None):
        """
        Render the combined figure at high resolution.
        With Bahdanau available: 4 columns (saliency | Grad-CAM | DeBERTa | Bahdanau)
        Otherwise: 3 columns (saliency | Grad-CAM | DeBERTa)

        Panel titles, axis labels and the info bar below are deliberately in
        Portuguese: they are rendered into the PNGs, so translating them would
        make regenerated figures disagree with the ones already in use. Comments
        stay in English; figure content does not.
        """
        original = load_original_image(image_path)

        has_bahdanau = bahdanau_attention and len(bahdanau_attention) > 0
        n_cols = 4 if has_bahdanau else 3

        fig = plt.figure(figsize=(5 * n_cols + 5, 10))
        gs = GridSpec(3, n_cols, height_ratios=[4, 2, 1], width_ratios=[1] * n_cols)

        # === Column 0: saliency map (input gradients, high resolution) ===
        ax_saliency = fig.add_subplot(gs[0, 0])
        blended_saliency = self.gradcam.overlay_heatmap(original, saliency_heatmap, alpha=0.5)
        ax_saliency.imshow(blended_saliency)
        ax_saliency.set_title("Saliency Map\n(Input Gradients)", fontsize=12, fontweight='bold')
        ax_saliency.axis('off')

        # === Column 1: real Grad-CAM (feature maps, high resolution) ===
        ax_gradcam = fig.add_subplot(gs[0, 1])
        blended_gradcam = self.gradcam.overlay_heatmap(original, gradcam_heatmap, alpha=0.5)
        ax_gradcam.imshow(blended_gradcam)
        ax_gradcam.set_title("Grad-CAM\n(Feature Maps)", fontsize=12, fontweight='bold')
        ax_gradcam.axis('off')

        # === Column 2: DeBERTa attention ===
        ax_text = fig.add_subplot(gs[0, 2])
        self._draw_attention_barplot(ax_text, word_attention,
                                     "DeBERTa Attention\n(Importancia Generica)")

        # === Column 3: Bahdanau attention (when available) ===
        if has_bahdanau:
            ax_bahdanau = fig.add_subplot(gs[0, 3])
            self._draw_attention_barplot(ax_bahdanau, bahdanau_attention,
                                         "Bahdanau Attention\n(Aprendido pelo Modelo)",
                                         cmap='YlOrRd')

        # === Row 2: review texts (full width) ===
        ax_reviews = fig.add_subplot(gs[1, :])
        ax_reviews.axis('off')

        if review_texts and len(review_texts) > 0:
            ax_reviews.text(0.01, 0.98, "Reviews do App:", fontsize=11,
                           fontweight='bold', transform=ax_reviews.transAxes, va='top')

            review_display = []
            for i, r in enumerate(review_texts[:8]):
                truncated = r[:300] + "..." if len(r) > 300 else r
                review_display.append(f"{i+1}. {truncated}")

            reviews_text = "\n".join(review_display)
            ax_reviews.text(0.01, 0.80, reviews_text, fontsize=8,
                           transform=ax_reviews.transAxes, va='top',
                           ha='left', family='sans-serif',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3,
                                     mutation_aspect=0.5))
        else:
            ax_reviews.text(0.5, 0.5, "Sem reviews disponiveis", ha='center', va='center',
                           fontsize=10, transform=ax_reviews.transAxes, style='italic')

        # === Row 3: prediction info ===
        ax_info = fig.add_subplot(gs[2, :])
        ax_info.axis('off')

        pred_class = "BOM" if prediction > 0.5 else "RUIM"
        true_class = "BOM" if true_label == 1 else "RUIM"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        is_correct = (prediction > 0.5) == (true_label == 1)

        result_text = "CORRETO" if is_correct else "INCORRETO"
        result_color = 'green' if is_correct else 'red'

        app_display = app_name or ""
        cat_display = category or ""
        header = ""
        if app_display or cat_display:
            parts = [p for p in [app_display, cat_display] if p]
            header = " | ".join(parts) + "    |    "

        info_text = (
            f"{header}"
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

    def _draw_attention_barplot(self, ax, word_attention, title, cmap='RdYlGn_r'):
        """Draw the horizontal barplot of the textual attention on an axis.

        The axis label and the empty-data message are in Portuguese on purpose:
        they end up in the rendered figures.
        """
        if word_attention:
            words = [w for w, _ in word_attention]
            importances = [i for _, i in word_attention]

            colors = plt.cm.get_cmap(cmap)(np.linspace(0.2, 0.8, len(words)))
            ax.barh(range(len(words)), importances, color=colors)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Peso de Atencao', fontsize=10)
        else:
            ax.text(0.5, 0.5, "Sem dados de texto", ha='center', va='center',
                    fontsize=12, transform=ax.transAxes)

        ax.set_title(title, fontsize=12, fontweight='bold')

    def _generate_app_visualization(self, representative_image_path, saliency_heatmap, gradcam_heatmap,
                                    word_attention, app_score, true_label, num_screens, output_path,
                                    review_texts=None, app_name=None, category=None):
        """Render the app-level figure with 3 columns plus the reviews, at high
        resolution.

        Panel titles, axis labels and the info bar below are deliberately in
        Portuguese: they are rendered into the PNGs, so translating them would
        make regenerated figures disagree with the ones already in use. Comments
        stay in English; figure content does not.
        """
        representative_original = load_original_image(representative_image_path)

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, height_ratios=[4, 3, 1], width_ratios=[1, 1, 1])

        # Aggregated saliency map (high resolution)
        ax_saliency = fig.add_subplot(gs[0, 0])
        blended_saliency = self.gradcam.overlay_heatmap(representative_original, saliency_heatmap, alpha=0.5)
        ax_saliency.imshow(blended_saliency)
        ax_saliency.set_title(f"Saliency Map Agregado\n({num_screens} telas)", fontsize=12, fontweight='bold')
        ax_saliency.axis('off')

        # Aggregated real Grad-CAM (high resolution)
        ax_gradcam = fig.add_subplot(gs[0, 1])
        blended_gradcam = self.gradcam.overlay_heatmap(representative_original, gradcam_heatmap, alpha=0.5)
        ax_gradcam.imshow(blended_gradcam)
        ax_gradcam.set_title(f"Grad-CAM Agregado\n({num_screens} telas)", fontsize=12, fontweight='bold')
        ax_gradcam.axis('off')

        # Most influential words
        ax_text = fig.add_subplot(gs[0, 2])

        if word_attention:
            words = [w for w, _ in word_attention]
            importances = [i for _, i in word_attention]
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(words)))
            ax_text.barh(range(len(words)), importances, color=colors)
            ax_text.set_yticks(range(len(words)))
            ax_text.set_yticklabels(words, fontsize=9)
            ax_text.invert_yaxis()
            ax_text.set_xlabel('Peso de Atencao Agregado', fontsize=10)
        else:
            ax_text.text(0.5, 0.5, "Sem dados de texto", ha='center', va='center',
                         fontsize=12, transform=ax_text.transAxes)

        ax_text.set_title("Palavras Mais Influentes\n(Agregado)", fontsize=12, fontweight='bold')

        # App reviews (full width, up to 20 reviews)
        ax_reviews = fig.add_subplot(gs[1, :])
        ax_reviews.axis('off')

        if review_texts and len(review_texts) > 0:
            ax_reviews.text(0.01, 0.98, "Reviews do App:", fontsize=11,
                           fontweight='bold', transform=ax_reviews.transAxes, va='top')

            review_display = []
            for i, r in enumerate(review_texts[:20]):
                truncated = r[:300] + "..." if len(r) > 300 else r
                review_display.append(f"{i+1}. {truncated}")

            reviews_text = "\n".join(review_display)
            ax_reviews.text(0.01, 0.88, reviews_text, fontsize=7,
                           transform=ax_reviews.transAxes, va='top',
                           ha='left', family='sans-serif',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3,
                                     mutation_aspect=0.5))
        else:
            ax_reviews.text(0.5, 0.5, "Sem reviews disponiveis", ha='center', va='center',
                           fontsize=10, transform=ax_reviews.transAxes, style='italic')

        # App info
        ax_info = fig.add_subplot(gs[2, :])
        ax_info.axis('off')

        pred_class = "BOM" if app_score > 0.5 else "RUIM"
        true_class = "BOM" if true_label == 1 else "RUIM"
        confidence = app_score if app_score > 0.5 else 1 - app_score
        is_correct = (app_score > 0.5) == (true_label == 1)

        result_text = "CORRETO" if is_correct else "INCORRETO"
        result_color = 'green' if is_correct else 'red'

        app_display = app_name or ""
        cat_display = category or ""
        header = ""
        if app_display or cat_display:
            parts = [p for p in [app_display, cat_display] if p]
            header = " | ".join(parts) + "    |    "

        info_text = (
            f"{header}"
            f"Score do App: {app_score:.3f}    |    "
            f"Predicao: {pred_class} ({confidence:.1%})    |    "
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

    def _generate_text_summary(self, package_name, prediction, confidence, top_words):
        """Build the textual summary of the explanation."""
        words_str = ', '.join([w for w, _ in top_words[:5]]) if top_words else "N/A"
        return (
            f"App '{package_name}' was classified as {prediction} "
            f"with {confidence:.1%} confidence. "
            f"Most influential words: {words_str}."
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
        self.shap_dir = os.path.join(output_dir, "shap_explanations")

        os.makedirs(self.screen_dir, exist_ok=True)
        os.makedirs(self.app_dir, exist_ok=True)

        self.screen_explanations = []
        self.app_explanations = []

    def add_screen_explanation(self, explanation):
        if explanation:
            self.screen_explanations.append(explanation)

    def add_app_explanation(self, explanation):
        if explanation:
            self.app_explanations.append(explanation)

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

    def save_all(self):
        """Save every explanation as JSON and CSV."""
        # JSON - screens
        screen_json_path = os.path.join(self.output_dir, 'screen_explanations.json')
        with open(screen_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.screen_explanations, f, indent=2, ensure_ascii=False, default=str)

        # CSV - semantic saliency aggregated per class
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
                'top_5_words': str([w for w, _ in e['top_words'][:5]]),
                'caption': (e.get('text_explanation') or {}).get('caption', ''),
                'simple': (e.get('text_explanation') or {}).get('simple', ''),
                'detailed': (e.get('text_explanation') or {}).get('detailed', ''),
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
                'top_5_words': str([w for w, _ in e['top_influential_words'][:5]]),
                'summary': e['summary'],
                'caption': (e.get('text_explanation') or {}).get('caption', ''),
                'simple': (e.get('text_explanation') or {}).get('simple', ''),
                'detailed': (e.get('text_explanation') or {}).get('detailed', ''),
            } for e in self.app_explanations])
            app_df.to_csv(os.path.join(self.output_dir, 'app_explanations.csv'), index=False)

        print(f"\nExplanations saved in: {self.output_dir}/")
        print(f"  - {len(self.screen_explanations)} screen explanations")
        print(f"  - {len(self.app_explanations)} app explanations")


# =============================================================================
# FUNCTION: TOP-K BEST AND WORST SCREENS (GRAD-CAM)
# =============================================================================

def generate_top_bottom_gradcam(data_loader, gradcam_explainer, text_attn, output_dir, top_k=10):
    """
    Render two comparative figures: the top K screens of good apps and the top K
    screens of bad ones, each with the Grad-CAM overlay and its metadata.

    Panel titles and the info boxes are in Portuguese on purpose: they are rendered
    into the PNGs.

    Args:
        data_loader: TestDataLoader holding the classification CSV
        gradcam_explainer: an initialized GradCAMExplainer
        text_attn: TextAttentionExplainer (for the top words)
        output_dir: output directory
        top_k: number of screens per figure

    Returns:
        list of paths to the generated figures
    """
    os.makedirs(output_dir, exist_ok=True)
    df = data_loader.classification_df

    # Top K good (highest probability) and top K bad (lowest probability)
    df_bom = df[df['label_real'] == 'bom'].nlargest(top_k, 'probabilidade')
    df_ruim = df[df['label_real'] != 'bom'].nsmallest(top_k, 'probabilidade')

    print(f"\nTop {top_k} BOM: {len(df_bom)} screens selected")
    print(f"Top {top_k} RUIM: {len(df_ruim)} screens selected")

    # Load DeBERTa for the top words
    text_attn.load_model()

    output_paths = []

    for label, df_sel, filename in [
        ("BOM", df_bom, f"top{top_k}_gradcam_BOM.png"),
        ("RUIM", df_ruim, f"top{top_k}_gradcam_RUIM.png"),
    ]:
        print(f"\nProcessing the top {len(df_sel)} {label} screens...")
        screen_data = []

        for rank, (_, row) in enumerate(df_sel.iterrows()):
            # Find the index in classification_df so get_sample can be used
            match = data_loader.classification_df[
                (data_loader.classification_df['package_name'] == row['package_name']) &
                (data_loader.classification_df['numero_da_tela'] == row['numero_da_tela'])
            ]
            if len(match) == 0:
                continue
            idx = match.index[0]

            sample = data_loader.get_sample(idx)
            if sample is None:
                continue

            # Grad-CAM (overlay at high resolution)
            img_batch = np.expand_dims(sample['image'], 0)
            emb_batch = np.expand_dims(sample['embedding'], 0)
            heatmap, _ = gradcam_explainer.compute_real_gradcam(img_batch, emb_batch)
            original = load_original_image(sample['image_path'])
            blended = gradcam_explainer.overlay_heatmap(original, heatmap)

            # Top 3 words by attention
            top_words_str = ""
            try:
                review_text = sample.get('review_text', '')
                if review_text and review_text.strip():
                    word_attn = text_attn.get_word_attention(review_text)
                    top3 = [w for w, _ in word_attn[:3]]
                    top_words_str = ", ".join(top3)
            except Exception:
                pass

            prob = float(row['probabilidade'])
            conf = prob if prob > 0.5 else 1 - prob

            screen_data.append({
                'rank': rank + 1,
                'blended': blended,
                'app_name': sample.get('app_name', ''),
                'package_name': sample['package_name'],
                'category': sample.get('category', 'N/A'),
                'probability': prob,
                'confidence': conf,
                'screen_id': sample['screen_id'],
                'top_words': top_words_str,
            })

            print(f"  [{rank+1}] {sample['package_name']} screen={sample['screen_id']} prob={prob:.4f}")

        if not screen_data:
            print(f"  No screen available for {label}. Skipping.")
            continue

        # Render the grid figure
        n = len(screen_data)
        ncols = 5
        nrows = (n + ncols - 1) // ncols  # ceil division

        fig, axes = plt.subplots(nrows, ncols, figsize=(25, nrows * 7))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        fig.subplots_adjust(hspace=0.55)

        for i in range(nrows * ncols):
            row_idx = i // ncols
            col_idx = i % ncols
            ax = axes[row_idx, col_idx]

            if i < n:
                sd = screen_data[i]
                ax.imshow(sd['blended'])

                # Title: rank + app name
                app_display = sd['app_name'][:25] + ('...' if len(sd['app_name']) > 25 else '')
                ax.set_title(f"#{sd['rank']} - {app_display}", fontsize=10, fontweight='bold')

                # Info below the image
                pkg_display = sd['package_name'][:35] + ('...' if len(sd['package_name']) > 35 else '')
                info_lines = [
                    pkg_display,
                    f"Cat: {sd['category']} | Prob: {sd['probability']:.4f}",
                    f"Conf: {sd['confidence']:.1%} | Tela: {sd['screen_id']}",
                ]
                if sd['top_words']:
                    info_lines.append(f"Top: {sd['top_words']}")

                info_text = "\n".join(info_lines)
                ax.text(0.5, -0.02, info_text, transform=ax.transAxes,
                        fontsize=8, ha='center', va='top', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  alpha=0.8, edgecolor='gray'))
            else:
                ax.set_visible(False)

            ax.axis('off')

        color = 'green' if label == 'BOM' else 'red'
        fig.suptitle(
            f"Top {n} Telas - Apps {label} (Grad-CAM)",
            fontsize=16, fontweight='bold', color=color, y=1.01
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        output_paths.append(out_path)
        print(f"  Saved: {out_path}")

        # Write the CSV with the details of each screen
        csv_path = os.path.join(output_dir, filename.replace('.png', '_detalhes.csv'))
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['rank', 'app_name', 'package_name', 'category',
                             'screen_id', 'probabilidade', 'confianca', 'top3_atencao'])
            for sd in screen_data:
                writer.writerow([
                    sd['rank'], sd['app_name'], sd['package_name'], sd['category'],
                    sd['screen_id'], f"{sd['probability']:.4f}",
                    f"{sd['confidence']:.1%}", sd['top_words']
                ])
        output_paths.append(csv_path)
        print(f"  Details: {csv_path}")

    return output_paths


def _load_v2_top_words(v2_run_dir=None, top_k=3, key="top_bahdanau_aggregated"):
    """Load the aggregated word list of each app from the ``app_explanations.json``
    of a multimodal run, so the top words shown in the app-level figures can be
    reproduced exactly.

    Args:
        v2_run_dir: folder of the run (holding app_explanations.json). When None,
            the most recent run in RESULTS_EXPLANATIONS_MULTIMODAL_V2 is discovered
            automatically.
        top_k: number of words per app to return.
        key: JSON key holding the ``[[word, score], ...]`` list. Use
            ``"top_bahdanau_aggregated"`` (post-fusion attention) or
            ``"top_shap_textual_aggregated"`` (textual SHAP).

    Returns:
        dict {package_name: "w1, w2, w3"}, or {} when nothing is found.
    """
    try:
        if v2_run_dir is not None:
            candidates = [v2_run_dir]
        else:
            runs = sorted(glob.glob(
                os.path.join(str(RESULTS_EXPLANATIONS_MULTIMODAL_V2), "*")
            ))
            # Newest first; runs without a valid JSON are skipped (for instance the
            # run in progress, when mode 10 is triggered from within it).
            candidates = [r for r in reversed(runs) if os.path.isdir(r)]

        for run_dir in candidates:
            json_path = os.path.join(run_dir, "app_explanations.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            mapping = {}
            for app in data:
                pkg = app.get("package_name")
                agg = app.get(key) or []
                words = [item[0] for item in agg[:top_k]]
                if pkg and words:
                    mapping[pkg] = ", ".join(words)
            if mapping:
                print(f"[Mode 10] '{key}' loaded from: "
                      f"{os.path.basename(run_dir)} ({len(mapping)} apps)")
                return mapping
        return {}
    except Exception as e:
        print(f"[Mode 10] Failed to load '{key}': {e}")
        return {}


def _load_v2_screen_review_candidates(v2_run_dir=None):
    """Read the (screen x polarized review) candidates of each app from the
    ``screen_explanations.json`` of a multimodal run, so mode 10 can pick the screen
    with the highest confidence over the same set of screens and reviews that run
    analyzed.

    For each screen it uses ``confidence`` (the screen confidence) and, from each
    item of ``per_review_analyses``, ``embedding_idx`` and ``p_orig`` (the
    prediction of that review on that screen). ``embedding_idx = -1`` marks an
    on-the-fly review, outside the cache.

    Args:
        v2_run_dir: folder of the run (holding screen_explanations.json). When None,
            the most recent run is discovered automatically.

    Returns:
        dict {package_name: [{'screen_id': str, 'emb_idx': int,
        'p_orig': float, 'screen_confidence': float}, ...]}, or {}.
    """
    try:
        if v2_run_dir is not None:
            candidates_dirs = [v2_run_dir]
        else:
            runs = sorted(glob.glob(
                os.path.join(str(RESULTS_EXPLANATIONS_MULTIMODAL_V2), "*")
            ))
            candidates_dirs = [r for r in reversed(runs) if os.path.isdir(r)]

        for run_dir in candidates_dirs:
            json_path = os.path.join(run_dir, "screen_explanations.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            mapping = {}
            for scr in data:
                pkg = scr.get("package_name")
                sid = scr.get("screen_id")
                conf = scr.get("confidence")
                if not pkg or sid is None:
                    continue
                for pr in scr.get("per_review_analyses") or []:
                    p_orig = pr.get("p_orig")
                    if p_orig is None:
                        continue
                    mapping.setdefault(pkg, []).append({
                        "screen_id": str(sid),
                        "emb_idx": pr.get("embedding_idx"),
                        "p_orig": float(p_orig),
                        "screen_confidence": float(conf) if conf is not None else 0.0,
                    })
            if mapping:
                print(f"[Mode 10] screen x review candidates (by confidence) "
                      f"loaded from: {os.path.basename(run_dir)} "
                      f"({len(mapping)} apps)")
                return mapping
        return {}
    except Exception as e:
        print(f"[Mode 10] Failed to load the candidates: {e}")
        return {}


def _select_top_apps_unique_screens(data_loader, gradcam_explainer, top_k=10, v2_run_dir=None):
    """Select the top-K good/bad apps plus one representative screen per app.

    The criterion matches mode 4 ("explain every app", scope 2): the top-K apps of
    each predicted class are picked by aggregated confidence per app, that is, the
    mean ``probabilidade`` over the correct rows (``acerto == True``) of each
    (package_name, predito) pair, with ``confianca_agg = |prob - 0.5| * 2``. Then,
    for each app, the screen with the highest confidence is chosen from the
    (screen x polarized review) candidates read from ``screen_explanations.json``,
    so the same set analyzed by the per-review pipeline is used. The candidates are
    sorted by screen confidence and, within a screen, by the review most confident
    in the direction of the class; pairs whose Grad-CAM comes out zeroed (the
    SmoothGrad fallback) are skipped. The embedding is always loaded from the cache
    through ``embedding_idx``.

    Shared by the Grad-CAM and SHAP top-K modes, which guarantees the same
    screens/reviews in both figures (Grad-CAM, with the fallback skip, runs a single
    time here).

    Args:
        data_loader: TestDataLoader holding the classification CSV.
        gradcam_explainer: GradCAMExplainer (computes the heatmap and prediction).
        top_k: number of apps per class.
        v2_run_dir: folder of the run holding screen_explanations.json (the
            screen x review candidates). When None, the most recent run is used.

    Returns:
        dict {'BOM': [sel, ...], 'RUIM': [sel, ...]} where each ``sel`` is
        ``{'rank', 'pkg', 'crow', 'image_path', 'screen_id', 'emb_idx',
        'image', 'embedding', 'heatmap', 'prob'}``.
    """
    full_df = data_loader.full_classification_df
    correct_df = full_df[full_df['acerto'] == True]
    agg = (
        correct_df.groupby(['package_name', 'predito'])['probabilidade']
        .mean()
        .reset_index()
    )
    agg['confianca_agg'] = (agg['probabilidade'] - 0.5).abs() * 2

    bom_pkgs = (
        agg[agg['predito'] == 'bom']
        .nlargest(top_k, 'confianca_agg')['package_name']
        .tolist()
    )
    ruim_pkgs = (
        agg[agg['predito'] == 'ruim']
        .nlargest(top_k, 'confianca_agg')['package_name']
        .tolist()
    )

    # Screen x review candidates, used to pick the highest-confidence screen.
    cand_map = _load_v2_screen_review_candidates(v2_run_dir=v2_run_dir)

    print(f"\nTop {top_k} BOM apps (by aggregated confidence): {len(bom_pkgs)} selected")
    print(f"Top {top_k} RUIM apps (by aggregated confidence): {len(ruim_pkgs)} selected")
    if not cand_map:
        print("[Mode 10] WARNING: screen_explanations.json not found; "
              "check mode10_v2_run_dir. Aborting the selection.")
        return {"BOM": [], "RUIM": []}

    def _gradcam_capture_fallback(img_batch, emb_batch):
        """Compute Grad-CAM while capturing stdout; returns (heatmap, pred, used_fallback)."""
        import io as _io
        import contextlib as _ctx
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            hm, pr = gradcam_explainer.compute_real_gradcam(img_batch, emb_batch)
        out = buf.getvalue()
        if out.strip():
            print(out, end="")  # re-emit so no log is lost
        return hm, pr, ("fallback" in out.lower())

    selections = {}
    for label, pkgs, prefer_high in [
        ("BOM", bom_pkgs, True),
        ("RUIM", ruim_pkgs, False),
    ]:
        print(f"\nSelecting the highest-confidence screen (top {len(pkgs)} {label} apps)...")
        chosen_list = []

        for rank, pkg in enumerate(pkgs):
            # Cached candidates (emb_idx>=0): sorted by screen confidence and, within
            # a screen, by the review most confident in the direction of the class
            # (BOM: high p_orig; RUIM: low p_orig).
            cands = [
                c for c in cand_map.get(pkg, [])
                if c['emb_idx'] is not None and c['emb_idx'] >= 0
            ]
            if not cands:
                print(f"  [warn] {pkg}: no cached candidate in the JSON. Skipping.")
                continue
            cands.sort(
                key=lambda c: (
                    c['screen_confidence'],
                    c['p_orig'] if prefer_high else (1.0 - c['p_orig']),
                ),
                reverse=True,
            )

            embeddings = data_loader._load_embedding_for_package(pkg)

            chosen = None       # pair with no fallback
            first_valid = None  # first loadable pair (used as a last resort)

            for c in cands:
                screen_id = c['screen_id']
                emb_idx = c['emb_idx']
                image_path = os.path.join(IMAGES_FOLDER, f"{screen_id}.jpg")
                if not os.path.exists(image_path):
                    continue
                image = preprocess_image_path(image_path)
                if image is None:
                    continue
                if embeddings is None or emb_idx >= len(embeddings):
                    continue
                emb = embeddings[emb_idx].astype(np.float32)

                img_batch = np.expand_dims(image, 0)
                emb_batch = np.expand_dims(emb, 0)
                heatmap, pred, used_fb = _gradcam_capture_fallback(img_batch, emb_batch)

                app_rows = full_df[full_df['package_name'] == pkg]
                crow = app_rows.iloc[0] if len(app_rows) else None
                sel = {
                    'rank': rank + 1, 'pkg': pkg, 'crow': crow,
                    'image_path': image_path, 'screen_id': screen_id,
                    'emb_idx': emb_idx, 'image': image, 'embedding': emb,
                    'heatmap': heatmap, 'prob': float(pred),
                }
                if first_valid is None:
                    first_valid = sel
                if not used_fb:
                    chosen = sel
                    break
                print(f"  [skip] {pkg} screen={screen_id} (review_idx={emb_idx}): "
                      f"zeroed Grad-CAM, trying the next pair")

            if chosen is None:
                if first_valid is None:
                    print(f"  No loadable screen for {pkg}. Skipping.")
                    continue
                chosen = first_valid
                print(f"  [warn] {pkg}: every pair fell back; "
                      f"using the most confident one")

            chosen_list.append(chosen)

        selections[label] = chosen_list

    return selections


def _select_apps_from_folder_unique_screens(data_loader, gradcam_explainer, apps_dir):
    """Mode 11: select the apps listed in a previous run folder, one screen each.

    Differences from mode 10 (:func:`_select_top_apps_unique_screens`):

    - No top-K by confidence: exactly the ``package_name`` values listed in
      ``apps_dir/app_explanations.json`` are used.
    - The good/bad split follows the class predicted by the current model,
      aggregated per (package_name, predito) over the correct rows of its CSV.
    - ``apps_dir`` is reused as the source of the (screen x review) candidates.
    - Apps present in the folder but with no correct row in the CSV of this model
      are skipped, with a warning.

    The logic that picks the best screen/review mirrors the mode 10 routine, kept
    duplicated on purpose so mode 10 stays untouched.

    Args:
        data_loader: TestDataLoader.
        gradcam_explainer: GradCAMExplainer.
        apps_dir: folder of a previous run holding ``app_explanations.json`` and
            ``screen_explanations.json``.

    Returns:
        dict {'BOM': [sel, ...], 'RUIM': [sel, ...]}, the same structure as mode 10.
    """
    # --- 1. Canonical app list from the reference folder ---
    app_json = os.path.join(apps_dir, "app_explanations.json")
    if not os.path.exists(app_json):
        print(f"[Mode 11] ERROR: {app_json} does not exist.")
        return {"BOM": [], "RUIM": []}
    try:
        with open(app_json, encoding="utf-8") as f:
            ref_apps = json.load(f)
    except Exception as e:
        print(f"[Mode 11] ERROR reading {app_json}: {e}")
        return {"BOM": [], "RUIM": []}

    folder_pkgs = [a["package_name"] for a in ref_apps if "package_name" in a]
    folder_set = set(folder_pkgs)
    print(f"\n[Mode 11] Apps in the reference folder: {len(folder_pkgs)}")

    # --- 2. Good/bad split from the classification of the current model ---
    full_df = data_loader.full_classification_df
    correct_df = full_df[full_df['acerto'] == True]
    agg = (
        correct_df.groupby(['package_name', 'predito'])['probabilidade']
        .mean()
        .reset_index()
    )
    agg['confianca_agg'] = (agg['probabilidade'] - 0.5).abs() * 2
    restricted = agg[agg['package_name'].isin(folder_set)]

    bom_pkgs = (
        restricted[restricted['predito'] == 'bom']
        .sort_values('confianca_agg', ascending=False)['package_name']
        .tolist()
    )
    ruim_pkgs = (
        restricted[restricted['predito'] == 'ruim']
        .sort_values('confianca_agg', ascending=False)['package_name']
        .tolist()
    )

    classified = set(bom_pkgs) | set(ruim_pkgs)
    missing = [p for p in folder_pkgs if p not in classified]
    print(f"[Mode 11] Classified as BOM by this model: {len(bom_pkgs)}")
    print(f"[Mode 11] Classified as RUIM by this model: {len(ruim_pkgs)}")
    if missing:
        print(
            f"   [warn] {len(missing)} apps from the folder have no correct row in the "
            f"CSV of this model (skipped): "
            f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
        )

    # --- 3. Screen x review candidates from the same reference folder ---
    cand_map = _load_v2_screen_review_candidates(v2_run_dir=apps_dir)
    if not cand_map:
        print(f"[Mode 11] WARNING: screen_explanations.json not found in "
              f"'{apps_dir}'. Aborting the selection.")
        return {"BOM": [], "RUIM": []}

    def _gradcam_capture_fallback(img_batch, emb_batch):
        """Compute Grad-CAM while capturing stdout; returns (heatmap, pred, used_fallback)."""
        import io as _io
        import contextlib as _ctx
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            hm, pr = gradcam_explainer.compute_real_gradcam(img_batch, emb_batch)
        out = buf.getvalue()
        if out.strip():
            print(out, end="")
        return hm, pr, ("fallback" in out.lower())

    # --- 4. Loop over apps: pick the best (screen, review) and run Grad-CAM ---
    selections = {}
    for label, pkgs, prefer_high in [
        ("BOM", bom_pkgs, True),
        ("RUIM", ruim_pkgs, False),
    ]:
        print(f"\nSelecting the highest-confidence screen ({len(pkgs)} {label} apps)...")
        chosen_list = []

        for rank, pkg in enumerate(pkgs):
            cands = [
                c for c in cand_map.get(pkg, [])
                if c['emb_idx'] is not None and c['emb_idx'] >= 0
            ]
            if not cands:
                print(f"  [warn] {pkg}: no cached candidate in the JSON. Skipping.")
                continue
            cands.sort(
                key=lambda c: (
                    c['screen_confidence'],
                    c['p_orig'] if prefer_high else (1.0 - c['p_orig']),
                ),
                reverse=True,
            )

            embeddings = data_loader._load_embedding_for_package(pkg)

            chosen = None
            first_valid = None

            for c in cands:
                screen_id = c['screen_id']
                emb_idx = c['emb_idx']
                image_path = os.path.join(IMAGES_FOLDER, f"{screen_id}.jpg")
                if not os.path.exists(image_path):
                    continue
                image = preprocess_image_path(image_path)
                if image is None:
                    continue
                if embeddings is None or emb_idx >= len(embeddings):
                    continue
                emb = embeddings[emb_idx].astype(np.float32)

                img_batch = np.expand_dims(image, 0)
                emb_batch = np.expand_dims(emb, 0)
                heatmap, pred, used_fb = _gradcam_capture_fallback(img_batch, emb_batch)

                app_rows = full_df[full_df['package_name'] == pkg]
                crow = app_rows.iloc[0] if len(app_rows) else None
                sel = {
                    'rank': rank + 1, 'pkg': pkg, 'crow': crow,
                    'image_path': image_path, 'screen_id': screen_id,
                    'emb_idx': emb_idx, 'image': image, 'embedding': emb,
                    'heatmap': heatmap, 'prob': float(pred),
                }
                if first_valid is None:
                    first_valid = sel
                if not used_fb:
                    chosen = sel
                    break
                print(f"  [skip] {pkg} screen={screen_id} (review_idx={emb_idx}): "
                      f"zeroed Grad-CAM, trying the next pair")

            if chosen is None:
                if first_valid is None:
                    print(f"  No loadable screen for {pkg}. Skipping.")
                    continue
                chosen = first_valid
                print(f"  [warn] {pkg}: every pair fell back; "
                      f"using the most confident one")

            chosen_list.append(chosen)

        selections[label] = chosen_list

    return selections


def generate_top_bottom_gradcam_unique_apps(data_loader, gradcam_explainer, text_attn, output_dir, top_k=10, bahdanau_attn=None, v2_run_dir=None, selections=None):
    """
    Render two comparative figures: the top K good apps and the top K bad ones, with
    a single representative screen per app (the most confident one).

    The app/screen selection comes from :func:`_select_top_apps_unique_screens`
    (top-K apps by confidence + highest-confidence screen per app, candidates read
    from screen_explanations.json). When ``selections`` is provided, that selection
    is reused, which guarantees the same screen/review as the companion SHAP figure;
    otherwise it is computed here.

    For the words shown in each figure, the Bahdanau attention is preferred (it is
    post-fusion, so it reflects the image-text interaction) as read from
    ``top_bahdanau_aggregated``, which keeps the app-level figures consistent. The
    local Bahdanau aggregation is used only when the app is missing from the JSON.

    Panel titles and the info boxes are in Portuguese on purpose: they are rendered
    into the PNGs.

    Args:
        data_loader: TestDataLoader holding the classification CSV
        gradcam_explainer: an initialized GradCAMExplainer
        text_attn: TextAttentionExplainer (provides the tokenizer and the fallback)
        output_dir: output directory
        top_k: number of apps per figure
        bahdanau_attn: BahdanauAttentionExplainer (preferred source of the top words;
            when None or unavailable, the DeBERTa self-attention of ``text_attn`` is
            used instead)
        v2_run_dir: folder of the run holding app_explanations.json (top words)
        selections: dict precomputed by _select_top_apps_unique_screens; when None,
            it is computed internally.

    Returns:
        list of paths to the generated figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Top words per app: read top_bahdanau_aggregated from the most recent run so
    # the words of the app-level figures are reproduced exactly.
    v2_top_words = _load_v2_top_words(v2_run_dir=v2_run_dir, top_k=3)

    if selections is None:
        selections = _select_top_apps_unique_screens(
            data_loader, gradcam_explainer, top_k=top_k, v2_run_dir=v2_run_dir
        )

    text_attn.load_model()

    output_paths = []

    for label, filename in [
        ("BOM", f"top{top_k}_gradcam_unique_apps_BOM.png"),
        ("RUIM", f"top{top_k}_gradcam_unique_apps_RUIM.png"),
    ]:
        sel_list = selections.get(label, [])
        print(f"\nProcessing the top {len(sel_list)} {label} apps...")
        screen_data = []

        for sel in sel_list:
            pkg = sel['pkg']
            crow = sel['crow']
            screen_id = sel['screen_id']
            heatmap = sel['heatmap']

            original = load_original_image(sel['image_path'])
            blended = gradcam_explainer.overlay_heatmap(original, heatmap)

            # Top words: from the run JSON (app-level match), else the local
            # Bahdanau aggregation.
            top_words_str = ""
            try:
                if pkg in v2_top_words:
                    top_words_str = v2_top_words[pkg]
                elif bahdanau_attn is not None and bahdanau_attn.available:
                    text_attn.load_model()
                    app_samples = data_loader.get_samples_for_app(pkg)
                    for s in app_samples:
                        s['text'] = s.get('review_text', '')
                    agg_words, _ = bahdanau_attn.aggregate_attention_for_app(
                        app_samples, text_attn.tokenizer, top_k=3
                    )
                    top_words_str = ", ".join(w for w, _ in agg_words)
            except Exception:
                pass

            prob = sel['prob']
            conf = prob if prob > 0.5 else 1 - prob

            screen_data.append({
                'rank': sel['rank'],
                'blended': blended,
                'app_name': crow.get('app', pkg),
                'package_name': pkg,
                'category': crow.get('categoria', 'N/A'),
                'probability': prob,
                'confidence': conf,
                'screen_id': screen_id,
                'top_words': top_words_str,
            })

            print(f"  [{sel['rank']}] {pkg} screen={screen_id} prob={prob:.4f}")

        if not screen_data:
            print(f"  No screen available for {label}. Skipping.")
            continue

        n = len(screen_data)
        ncols = 5
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(25, nrows * 7))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        fig.subplots_adjust(hspace=0.55)

        for i in range(nrows * ncols):
            row_idx = i // ncols
            col_idx = i % ncols
            ax = axes[row_idx, col_idx]

            if i < n:
                sd = screen_data[i]
                ax.imshow(sd['blended'])

                app_display = sd['app_name'][:25] + ('...' if len(sd['app_name']) > 25 else '')
                ax.set_title(f"#{sd['rank']} - {app_display}", fontsize=10, fontweight='bold')

                pkg_display = sd['package_name'][:35] + ('...' if len(sd['package_name']) > 35 else '')
                info_lines = [
                    pkg_display,
                    f"Cat: {sd['category']} | Prob: {sd['probability']:.4f}",
                    f"Conf: {sd['confidence']:.1%} | Tela: {sd['screen_id']}",
                ]
                if sd['top_words']:
                    info_lines.append(f"Top: {sd['top_words']}")

                info_text = "\n".join(info_lines)
                ax.text(0.5, -0.02, info_text, transform=ax.transAxes,
                        fontsize=8, ha='center', va='top', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  alpha=0.8, edgecolor='gray'))
            else:
                ax.set_visible(False)

            ax.axis('off')

        color = 'green' if label == 'BOM' else 'red'
        fig.suptitle(
            f"Top {n} Apps {label} - 1 Tela por App (Grad-CAM)",
            fontsize=16, fontweight='bold', color=color, y=1.01
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        output_paths.append(out_path)
        print(f"  Saved: {out_path}")

        csv_path = os.path.join(output_dir, filename.replace('.png', '_detalhes.csv'))
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['rank', 'app_name', 'package_name', 'category',
                             'screen_id', 'probabilidade', 'confianca', 'top3_atencao'])
            for sd in screen_data:
                writer.writerow([
                    sd['rank'], sd['app_name'], sd['package_name'], sd['category'],
                    sd['screen_id'], f"{sd['probability']:.4f}",
                    f"{sd['confidence']:.1%}", sd['top_words']
                ])
        output_paths.append(csv_path)
        print(f"  Details: {csv_path}")

    return output_paths


def generate_top_bottom_shap_unique_apps(data_loader, shap_explainer, gradcam_explainer, output_dir, top_k=10, v2_run_dir=None, selections=None, n_segments=50, n_samples=None):
    """
    Render two comparative figures: the top K good apps and the top K bad ones, with
    a single representative screen per app, using the visual SHAP (SLIC superpixels)
    as heatmap and the textual SHAP as top words.

    Companion of :func:`generate_top_bottom_gradcam_unique_apps`: it consumes the
    same app/screen selection (through the ``selections`` precomputed by
    :func:`_select_top_apps_unique_screens`), so each app appears with the same
    representative screen in both the Grad-CAM and the SHAP figures.

    - Visual SHAP: ``_compute_image_shap_superpixel`` on the most confident screen,
      with ``n_segments=50`` and ``seed=42``, using the same embedding
      (``sel['embedding']``, from the cache) as Grad-CAM, so both figures agree.
    - Textual SHAP: reads ``top_shap_textual_aggregated`` from the run (the same
      app-level strategy used for the Bahdanau words), which guarantees an exact
      match with the app-level figures.

    Panel titles and the info boxes are in Portuguese on purpose: they are rendered
    into the PNGs.

    Args:
        data_loader: TestDataLoader holding the classification CSV
        shap_explainer: an initialized SHAPExplainer (visual SHAP)
        gradcam_explainer: GradCAMExplainer (provides overlay_heatmap and, when
            ``selections`` is None, the screen selection)
        output_dir: output directory
        top_k: number of apps per figure
        v2_run_dir: folder of the run holding app_explanations.json (SHAP words)
        selections: dict precomputed by _select_top_apps_unique_screens; when None,
            it is computed internally.
        n_segments: number of SLIC superpixels for the visual SHAP
        n_samples: KernelExplainer samples (default: SHAP_MAX_EVALS)

    Returns:
        list of paths to the generated figures
    """
    os.makedirs(output_dir, exist_ok=True)

    if n_samples is None:
        n_samples = SHAP_MAX_EVALS

    # Top words per app from the aggregated textual SHAP of the run (app-level match).
    v2_shap_words = _load_v2_top_words(
        v2_run_dir=v2_run_dir, top_k=3, key="top_shap_textual_aggregated"
    )

    if selections is None:
        selections = _select_top_apps_unique_screens(
            data_loader, gradcam_explainer, top_k=top_k, v2_run_dir=v2_run_dir
        )

    total_apps = sum(len(selections.get(l, [])) for l in ("BOM", "RUIM"))
    print(f"\n*** WARNING: the visual SHAP is significantly slower than Grad-CAM. ***")
    print(f"*** Estimate: ~3-5 min per screen ({total_apps} apps = ~{total_apps*4} min on GPU) ***\n")

    output_paths = []

    for label, filename in [
        ("BOM", f"top{top_k}_shap_unique_apps_BOM.png"),
        ("RUIM", f"top{top_k}_shap_unique_apps_RUIM.png"),
    ]:
        sel_list = selections.get(label, [])
        print(f"\nProcessing the top {len(sel_list)} {label} apps (visual + textual SHAP)...")
        screen_data = []

        for sel in sel_list:
            pkg = sel['pkg']
            crow = sel['crow']
            screen_id = sel['screen_id']
            image = sel['image']

            # Visual SHAP: uses the same embedding as Grad-CAM (the review of the
            # most confident screen, from the cache), with n_segments=50 and
            # seed=42, so both figures stay coherent.
            ref_embedding = np.expand_dims(sel['embedding'], 0)  # (1, 128, 768)

            print(f"  [{sel['rank']}] {pkg} screen={screen_id} - image SHAP...")
            try:
                heatmap, _segments, _img_total = shap_explainer._compute_image_shap_superpixel(
                    image=image,
                    text_embedding=ref_embedding,
                    n_segments=n_segments,
                    n_samples=n_samples,
                )
            except Exception as e:
                print(f"    Image SHAP error ({pkg} screen={screen_id}): {e}. Skipping the app.")
                continue

            original = load_original_image(sel['image_path'])
            blended = gradcam_explainer.overlay_heatmap(original, heatmap, alpha=0.5)

            # Textual SHAP: the app-level words read from the run JSON.
            top_words_str = v2_shap_words.get(pkg, "")

            prob = sel['prob']
            conf = prob if prob > 0.5 else 1 - prob

            screen_data.append({
                'rank': sel['rank'],
                'blended': blended,
                'app_name': crow.get('app', pkg),
                'package_name': pkg,
                'category': crow.get('categoria', 'N/A'),
                'probability': prob,
                'confidence': conf,
                'screen_id': screen_id,
                'top_words': top_words_str,
            })

            print(f"  [{sel['rank']}] {pkg} screen={screen_id} prob={prob:.4f} "
                  f"SHAP=[{top_words_str}]")

        if not screen_data:
            print(f"  No screen available for {label}. Skipping.")
            continue

        n = len(screen_data)
        ncols = 5
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(25, nrows * 7))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        fig.subplots_adjust(hspace=0.55)

        for i in range(nrows * ncols):
            row_idx = i // ncols
            col_idx = i % ncols
            ax = axes[row_idx, col_idx]

            if i < n:
                sd = screen_data[i]
                ax.imshow(sd['blended'])

                app_display = sd['app_name'][:25] + ('...' if len(sd['app_name']) > 25 else '')
                ax.set_title(f"#{sd['rank']} - {app_display}", fontsize=10, fontweight='bold')

                pkg_display = sd['package_name'][:35] + ('...' if len(sd['package_name']) > 35 else '')
                info_lines = [
                    pkg_display,
                    f"Cat: {sd['category']} | Prob: {sd['probability']:.4f}",
                    f"Conf: {sd['confidence']:.1%} | Tela: {sd['screen_id']}",
                ]
                if sd['top_words']:
                    info_lines.append(f"SHAP: {sd['top_words']}")

                info_text = "\n".join(info_lines)
                ax.text(0.5, -0.02, info_text, transform=ax.transAxes,
                        fontsize=8, ha='center', va='top', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  alpha=0.8, edgecolor='gray'))
            else:
                ax.set_visible(False)

            ax.axis('off')

        color = 'green' if label == 'BOM' else 'red'
        fig.suptitle(
            f"Top {n} Apps {label} - 1 Tela por App (SHAP)",
            fontsize=16, fontweight='bold', color=color, y=1.01
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        output_paths.append(out_path)
        print(f"  Saved: {out_path}")

        csv_path = os.path.join(output_dir, filename.replace('.png', '_detalhes.csv'))
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['rank', 'app_name', 'package_name', 'category',
                             'screen_id', 'probabilidade', 'confianca', 'top3_shap'])
            for sd in screen_data:
                writer.writerow([
                    sd['rank'], sd['app_name'], sd['package_name'], sd['category'],
                    sd['screen_id'], f"{sd['probability']:.4f}",
                    f"{sd['confidence']:.1%}", sd['top_words']
                ])
        output_paths.append(csv_path)
        print(f"  Details: {csv_path}")

    return output_paths


def generate_top_bottom_shap(data_loader, shap_explainer, output_dir, top_k=10):
    """
    Render two comparative figures: the top K screens of good apps and the top K
    screens of bad ones, each with the SHAP superpixel overlay and the top 3 SHAP
    words.

    Panel titles and the info boxes are in Portuguese on purpose: they are rendered
    into the PNGs.

    Args:
        data_loader: TestDataLoader holding the classification CSV
        shap_explainer: an initialized SHAPExplainer
        output_dir: output directory
        top_k: number of screens per figure

    Returns:
        list of paths to the generated figures
    """
    os.makedirs(output_dir, exist_ok=True)
    df = data_loader.classification_df

    # Top K good (highest probability) and top K bad (lowest probability)
    df_bom = df[df['label_real'] == 'bom'].nlargest(top_k, 'probabilidade')
    df_ruim = df[df['label_real'] != 'bom'].nsmallest(top_k, 'probabilidade')

    print(f"\nTop {top_k} BOM: {len(df_bom)} screens selected")
    print(f"Top {top_k} RUIM: {len(df_ruim)} screens selected")

    total_screens = len(df_bom) + len(df_ruim)
    print(f"\n*** WARNING: the SHAP mode is significantly slower than Grad-CAM. ***")
    print(f"*** Estimate: ~3-5 min per screen ({total_screens} screens = ~{total_screens*4} min on GPU) ***\n")

    output_paths = []

    for label, df_sel, filename in [
        ("BOM", df_bom, f"top{top_k}_shap_BOM.png"),
        ("RUIM", df_ruim, f"top{top_k}_shap_RUIM.png"),
    ]:
        print(f"\nProcessing the top {len(df_sel)} {label} screens (multimodal SHAP)...")
        screen_data = []

        for rank, (_, row) in enumerate(df_sel.iterrows()):
            match = data_loader.classification_df[
                (data_loader.classification_df['package_name'] == row['package_name']) &
                (data_loader.classification_df['numero_da_tela'] == row['numero_da_tela'])
            ]
            if len(match) == 0:
                continue
            idx = match.index[0]

            sample = data_loader.get_sample(idx)
            if sample is None:
                continue

            img = sample['image']
            img_batch = np.expand_dims(img, 0)

            # Collect the reviews (up to 5)
            review_texts = []
            if 'review_texts' in sample and sample['review_texts']:
                review_texts = [r for r in sample['review_texts'] if r and str(r).strip()][:5]
            if not review_texts:
                review_text = sample.get('review_text', '')
                if review_text and review_text.strip():
                    review_texts = [review_text]

            # Reference embedding (first review) for the image SHAP
            if review_texts:
                ref_embedding = shap_explainer._generate_embeddings_batch([review_texts[0]])
            else:
                ref_embedding = np.expand_dims(sample['embedding'], 0)

            # === Image SHAP (superpixels) ===
            print(f"  [{rank+1}/{len(df_sel)}] {sample['package_name']} screen={sample['screen_id']} - image SHAP...")
            heatmap, _segments, image_shap_total = shap_explainer._compute_image_shap_superpixel(
                image=img,
                text_embedding=ref_embedding,
                n_segments=100,
                n_samples=SHAP_MAX_EVALS
            )

            # Overlay the heatmap on the high-resolution image
            original = load_original_image(sample['image_path'])
            blended = gradcam_explainer.overlay_heatmap(original, heatmap, alpha=0.5)

            # === Textual SHAP (top 3 words) ===
            top_words_str = ""
            text_shap_total = 0.0
            if review_texts:
                print(f"    Text SHAP ({len(review_texts)} reviews)...")
                try:
                    text_masker = shap.maskers.Text(tokenizer=r'\S+')
                    text_predict_fn = shap_explainer._make_text_predict_fn(img_batch)
                    text_explainer_obj = shap.Explainer(text_predict_fn, text_masker)

                    global_feature_shap = {}
                    all_text_abs_shap = []

                    for rev_idx, review in enumerate(review_texts):
                        try:
                            text_shap_values = text_explainer_obj(
                                [review],
                                max_evals=SHAP_MAX_EVALS,
                                batch_size=5
                            )

                            tokens = text_shap_values.data[0]
                            vals = text_shap_values.values[0]
                            if vals.ndim > 1:
                                vals = vals[:, 0]
                            if isinstance(tokens, np.ndarray):
                                tokens = tokens.tolist()

                            word_pairs = [
                                (str(tok).strip(), float(val))
                                for tok, val in zip(tokens, vals)
                                if str(tok).strip()
                            ]

                            # Accumulate the total |SHAP| of the text
                            all_text_abs_shap.extend([abs(v) for _, v in word_pairs])

                            # Negation bigrams
                            bigrams, consumed = SHAPExplainer._extract_negation_bigrams(word_pairs)

                            for j, (word, val) in enumerate(word_pairs):
                                if j not in consumed:
                                    key = word.lower()
                                    global_feature_shap.setdefault(key, []).append(float(val))

                            for bigram, val in bigrams.items():
                                global_feature_shap.setdefault(bigram, []).append(float(val))

                        except Exception as e:
                            print(f"      Error on review {rev_idx+1}: {e}")

                    # Aggregate: mean(|SHAP|), top 3
                    if global_feature_shap:
                        feature_importance = {
                            feat: float(np.mean(np.abs(vals)))
                            for feat, vals in global_feature_shap.items()
                        }
                        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                        top3 = [f for f, _ in sorted_features[:3]]
                        top_words_str = ", ".join(top3)

                    text_shap_total = float(np.sum(all_text_abs_shap)) if all_text_abs_shap else 0.0

                except Exception as e:
                    print(f"    Text SHAP error: {e}")

            # Relative contribution of the image vs the text
            total_shap = image_shap_total + text_shap_total
            if total_shap > 0:
                img_contrib = image_shap_total / total_shap * 100
                txt_contrib = text_shap_total / total_shap * 100
            else:
                img_contrib = 50.0
                txt_contrib = 50.0

            prob = float(row['probabilidade'])
            conf = prob if prob > 0.5 else 1 - prob

            screen_data.append({
                'rank': rank + 1,
                'blended': blended,
                'app_name': sample.get('app_name', ''),
                'package_name': sample['package_name'],
                'category': sample.get('category', 'N/A'),
                'probability': prob,
                'confidence': conf,
                'screen_id': sample['screen_id'],
                'top_words': top_words_str,
                'img_contrib': img_contrib,
                'txt_contrib': txt_contrib,
            })

            print(f"    prob={prob:.4f} top_words=[{top_words_str}]")

        if not screen_data:
            print(f"  No screen available for {label}. Skipping.")
            continue

        # Render the grid figure
        n = len(screen_data)
        ncols = 5
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(25, nrows * 7))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        fig.subplots_adjust(hspace=0.55)

        for i in range(nrows * ncols):
            row_idx = i // ncols
            col_idx = i % ncols
            ax = axes[row_idx, col_idx]

            if i < n:
                sd = screen_data[i]
                ax.imshow(sd['blended'])

                app_display = sd['app_name'][:25] + ('...' if len(sd['app_name']) > 25 else '')
                ax.set_title(f"#{sd['rank']} - {app_display}", fontsize=10, fontweight='bold')

                pkg_display = sd['package_name'][:35] + ('...' if len(sd['package_name']) > 35 else '')
                info_lines = [
                    pkg_display,
                    f"Cat: {sd['category']} | Prob: {sd['probability']:.4f}",
                    f"Conf: {sd['confidence']:.1%} | Tela: {sd['screen_id']}",
                    f"Img: {sd['img_contrib']:.0f}% | Txt: {sd['txt_contrib']:.0f}%",
                ]
                if sd['top_words']:
                    info_lines.append(f"SHAP: {sd['top_words']}")

                info_text = "\n".join(info_lines)
                ax.text(0.5, -0.02, info_text, transform=ax.transAxes,
                        fontsize=8, ha='center', va='top', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  alpha=0.8, edgecolor='gray'))
            else:
                ax.set_visible(False)

            ax.axis('off')

        color = 'green' if label == 'BOM' else 'red'
        fig.suptitle(
            f"Top {n} Telas - Apps {label} (SHAP Multimodal)",
            fontsize=16, fontweight='bold', color=color, y=1.01
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        output_paths.append(out_path)
        print(f"  Saved: {out_path}")

        # Write the CSV with the details of each screen
        csv_path = os.path.join(output_dir, filename.replace('.png', '_detalhes.csv'))
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['rank', 'app_name', 'package_name', 'category',
                             'screen_id', 'probabilidade', 'confianca',
                             'img_contrib_%', 'txt_contrib_%', 'top3_shap_words'])
            for sd in screen_data:
                writer.writerow([
                    sd['rank'], sd['app_name'], sd['package_name'], sd['category'],
                    sd['screen_id'], f"{sd['probability']:.4f}",
                    f"{sd['confidence']:.1%}",
                    f"{sd['img_contrib']:.1f}", f"{sd['txt_contrib']:.1f}",
                    sd['top_words']
                ])
        output_paths.append(csv_path)
        print(f"  Details: {csv_path}")

    return output_paths


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def _pick_resume_run_dir():
    """Offer the recent runs so mode 4 can continue an interrupted one.

    Only folders carrying ``apps/<pkg>/app_explanation.json`` snapshots are
    offered: that file holds the dicts of an app that finished, which the CSVs
    alone cannot rebuild (they carry no ``top_words``, no aggregated attention
    and no app-level NLG). Runs produced before those snapshots existed are
    therefore not listed, since continuing one would silently drop those fields
    from the global indexes.

    Returns the chosen folder, or None to start a new run.
    """
    base = str(RESULTS_EXPLANATIONS_MULTIMODAL)
    runs = sorted(
        (d for d in glob.glob(os.path.join(base, "explicabilidade_v6_*"))
         if os.path.isdir(d)),
        key=os.path.getmtime,
        reverse=True,
    )[:10]

    resumable = []
    for d in runs:
        n_done = len(glob.glob(os.path.join(d, "apps", "*", "app_explanation.json")))
        if n_done:
            resumable.append((d, n_done))

    if not resumable:
        return None

    print("\nRuns that can be continued:")
    for i, (d, n_done) in enumerate(resumable, start=1):
        print(f"  {i} - {os.path.basename(d)} ({n_done} apps finished)")
    print("  0 - Start a new run")

    choice = input(f"Continue which run? (0-{len(resumable)}) [0]: ").strip() or "0"
    try:
        idx = int(choice)
    except ValueError:
        idx = 0
    if 1 <= idx <= len(resumable):
        return resumable[idx - 1][0]
    return None


def explain(config: dict) -> None:
    """Multimodal explainability entry point driven by a YAML config dict."""
    global IMG_SIZE, MAX_TEXT_LENGTH, DEBERTA_MODEL, ASPECT, SHAP_MAX_EVALS
    global GRADCAM_SMOOTHING, GRADCAM_METRICS, ROAD_PERCENTILES
    global APPS_FROM_MULTIMODAL_RUN

    IMG_SIZE = config.get("img_size", IMG_SIZE)
    MAX_TEXT_LENGTH = config.get("max_text_length", MAX_TEXT_LENGTH)
    DEBERTA_MODEL = config.get("deberta_model", DEBERTA_MODEL)
    ASPECT = config.get("aspect", ASPECT)
    SHAP_MAX_EVALS = config.get("shap_max_evals", SHAP_MAX_EVALS)
    GRADCAM_SMOOTHING = bool(config.get("gradcam_smoothing", GRADCAM_SMOOTHING))
    GRADCAM_METRICS = list(config.get("gradcam_metrics", GRADCAM_METRICS) or [])
    ROAD_PERCENTILES = list(config.get("road_percentiles", ROAD_PERCENTILES))
    _apps_src = config.get("apps_from_multimodal_run")
    APPS_FROM_MULTIMODAL_RUN = str(_apps_src).strip() if _apps_src else None

    # Run used as the source of top_bahdanau_aggregated in mode 10 (top-K apps).
    # When None or absent, mode 10 discovers the most recent run in
    # results/explanations/multimodal_v2/ automatically.
    v2_run_dir = config.get("mode10_v2_run_dir", None)

    # Folder of apps used by mode 11 ("apps from a folder"). It must point to a
    # previous XAI run holding ``app_explanations.json`` and
    # ``screen_explanations.json``. When None or absent, mode 11 falls back to
    # ``mode10_v2_run_dir``.
    mode11_apps_dir = config.get("mode11_apps_dir", None)

    # With ``skip_shap=true``, modes 10 and 11 only render the Grad-CAM + Bahdanau
    # figures, skipping the SHAP stage (visual and textual).
    skip_shap = bool(config.get("skip_shap", False))

    print("=" * 70)
    print("MULTIMODAL EXPLAINABILITY MODULE")
    print("Saliency map + Grad-CAM + DeBERTa attention + Bahdanau attention + SHAP")
    print("=" * 70)

    # 1. Select the model
    selector = ModelSelector()
    model_info = selector.interactive_select_model()

    if model_info is None:
        return

    # 1b. Choose which classification CSV to use (test split or full dataset)
    model_info = selector.interactive_select_csv(model_info)

    if model_info is None:
        return

    # 2. Load the model
    model = selector.load_selected_model(model_info)

    # 3. Show the model structure, for debugging
    print("\nModel structure:")
    for i, layer in enumerate(model.layers):
        print(f"  [{i}] {layer.name}: {type(layer).__name__}")

    # 4. Initialize the explainers
    print("\nInitializing the explainability components...")

    try:
        gradcam = GradCAMExplainer(model)
    except Exception as e:
        print(f"ERROR initializing Grad-CAM: {e}")
        return

    text_attn = TextAttentionExplainer()

    # Bahdanau attention is only available for models that carry the layer
    bahdanau = BahdanauAttentionExplainer(model)
    if bahdanau.available:
        print("Bahdanau attention: ACTIVE")
    else:
        print("Bahdanau attention: INACTIVE (model without an attention_softmax layer)")

    explainer = MultimodalExplanation(model, gradcam, text_attn, bahdanau)

    # 5. Mode menu
    print("\n" + "-" * 70)
    print("EXPLANATION MODES:")
    print("-" * 70)
    print("  1 - Quick test (1 sample)")
    print("  2 - Explain a specific screen")
    print("  3 - Explain a specific app")
    print("  4 - Explain every app (requires the CSV)")
    print("  5 - Explain the errors only (requires the CSV)")
    print("  6 - Multimodal SHAP explanation (specific screen)")
    print("  7 - Multimodal SHAP explanation (whole app)")
    print("  8 - Top best and worst screens (Grad-CAM)")
    print("  9 - Top best and worst screens (multimodal SHAP)")
    print("  10 - Top best and worst apps (1 screen per app, Grad-CAM)")
    print("  11 - Apps from a previous run folder (mode11_apps_dir, Grad-CAM + SHAP)")
    print("  0 - Back")

    mode = input(f"\nChoose the mode (1-11, 0 to go back) [1]: ").strip()
    if mode == "":
        mode = "1"
    if mode == "0":
        # Nothing has been written yet, so returning leaves no partial output.
        return

    # Check whether the CSV is available for the modes that need it
    if mode in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"] and not model_info['has_classification']:
        print(f"\nERROR: mode {mode} requires the classification CSV.")
        print(f"  Expected: {model_info['classification_csv']}")
        return

    # Create the output directory. Mode 4 may instead continue an interrupted
    # run: the folder is reused, the apps already finished are reloaded from
    # their per-app snapshots and skipped in the loop below.
    resumed_apps: set = set()
    resume_dir = _pick_resume_run_dir() if mode == "4" else None
    if resume_dir:
        output_dir = resume_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(RESULTS_EXPLANATIONS_MULTIMODAL / f"explicabilidade_v6_{timestamp}")
    output_manager = ExplanationOutputManager(output_dir)

    if resume_dir:
        # ``resume_from_disk`` only exists in the per-app output manager (v2+);
        # the base one keeps no per-app snapshot to reload.
        reload_snapshots = getattr(output_manager, "resume_from_disk", None)
        if reload_snapshots is None:
            print("[resume] This output manager keeps no per-app snapshot; "
                  "nothing to reload, every app will be processed again.")
        else:
            resumed_apps = reload_snapshots()
            print(
                f"\n[resume] Continuing {os.path.basename(output_dir)}: "
                f"{len(resumed_apps)} apps reloaded "
                f"({len(output_manager.screen_explanations)} screens); "
                f"they will be skipped."
            )

    # Load the data when needed
    data_loader = None
    if mode in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]:
        try:
            data_loader = TestDataLoader(model_info)
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            return

    if mode == "1":
        # Mode 1: quick test
        print("\n" + "-" * 50)
        print("MODE 1: QUICK TEST")
        print("-" * 50)

        # Take the first available image
        test_images = glob.glob(os.path.join(IMAGES_FOLDER, "*.jpg"))[:1]
        if not test_images:
            print("No image found in the screenshots folder")
            return

        image_path = test_images[0]
        screen_id = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Using image: {image_path}")

        # Use a dummy text and embedding
        dummy_text = "This app has a good interface design"
        dummy_embedding = np.zeros((MAX_TEXT_LENGTH, 768), dtype=np.float32)

        explanation = explainer.explain_screen(
            image_path=image_path,
            text=dummy_text,
            text_embedding=dummy_embedding,
            true_label=1,
            package_name="test",
            screen_id=screen_id,
            output_dir=output_manager.screen_dir
        )

        if explanation:
            output_manager.add_screen_explanation(explanation)
            print(f"\nExplanation generated: {explanation['visualization_path']}")

    elif mode == "2":
        # Mode 2: explain a specific screen
        print("\n" + "-" * 50)
        print("MODE 2: EXPLAIN A SPECIFIC SCREEN")
        print("-" * 50)

        total = len(data_loader)
        print(f"\nTotal samples: {total}")

        while True:
            try:
                idx = input(f"Sample index (0-{total - 1}): ").strip()
                idx = int(idx)
                if 0 <= idx < total:
                    break
                print(f"Index out of range (0-{total - 1})")
            except ValueError:
                print("Type a valid number.")

        sample = data_loader.get_sample(idx)
        if sample is None:
            print("Error loading the sample.")
            return

        print(f"\nApp: {sample['app_name']}")
        print(f"Package: {sample['package_name']}")
        print(f"Screen: {sample['screen_id']}")
        print(f"Label: {sample['true_class']}")

        explanation = explainer.explain_screen(
            image_path=sample['image_path'],
            text=sample.get('review_text', f"App {sample['app_name']}"),
            text_embedding=sample['embedding'],
            true_label=sample['true_label'],
            package_name=sample['package_name'],
            screen_id=sample['screen_id'],
            output_dir=output_manager.screen_dir,
            review_texts=sample.get('review_texts', []),
            app_name=sample.get('app_name'),
            category=sample.get('category')
        )

        if explanation:
            output_manager.add_screen_explanation(explanation)
            print(f"\nExplanation generated: {explanation['visualization_path']}")

    elif mode == "3":
        # Mode 3: explain a specific app
        print("\n" + "-" * 50)
        print("MODE 3: EXPLAIN A SPECIFIC APP")
        print("-" * 50)

        # List the available apps
        apps = data_loader.get_all_apps()
        print(f"\n{len(apps)} apps available. Examples:")
        for i, app in enumerate(apps[:10]):
            print(f"  {app}")
        if len(apps) > 10:
            print(f"  ... and {len(apps) - 10} more apps")

        package_name = input("\nType the package_name of the app: ").strip()

        if package_name not in apps:
            print(f"App {package_name} not found.")
            return

        samples = data_loader.get_samples_for_app(package_name)
        print(f"\nGenerating explanations for {len(samples)} screens of {package_name}...")

        # Per-app routing when the output manager supports it.
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

        for sample in tqdm(samples, desc="Processing screens"):
            explanation = explainer.explain_screen(
                image_path=sample['image_path'],
                text=sample.get('review_text', f"App {sample['app_name']}"),
                text_embedding=sample['embedding'],
                true_label=sample['true_label'],
                package_name=sample['package_name'],
                screen_id=sample['screen_id'],
                output_dir=_screen_dir,
                review_texts=sample.get('review_texts', []),
                app_name=sample.get('app_name'),
                category=sample.get('category')
            )
            if explanation:
                output_manager.add_screen_explanation(explanation)

        # Build the aggregated app explanation
        if output_manager.screen_explanations:
            # Load every review of the app, for the full display
            all_reviews = data_loader._load_review_texts(package_name, max_texts=200)
            app_explanation = explainer.explain_app(
                package_name=package_name,
                screen_explanations=output_manager.screen_explanations,
                output_dir=_app_dir,
                all_app_reviews=all_reviews or []
            )
            if app_explanation:
                output_manager.add_app_explanation(app_explanation)

    elif mode == "4":
        # Mode 4: explain every app. For each app of the CSV it repeats the mode 3
        # flow (explain the screens, then aggregate). The per-review artifact cache
        # is released after each app, to keep memory usage bounded.
        print("\n" + "-" * 50)
        print("MODE 4: EXPLAIN EVERY APP")
        print("-" * 50)

        all_apps = data_loader.get_all_apps()
        total_apps = len(all_apps)
        print(f"\nTotal apps in the CSV: {total_apps}")

        # Canonical list inherited from another multimodal run: when
        # ``APPS_FROM_MULTIMODAL_RUN`` is set, the scope prompt is skipped and
        # exactly the ``package_name`` values of that run are processed, filtered by
        # the intersection with the CSV of this model. This allows app-level XAI
        # comparisons between pipelines over the same canonical set of apps.
        if APPS_FROM_MULTIMODAL_RUN:
            ref_run_path = os.path.abspath(APPS_FROM_MULTIMODAL_RUN)
            app_json = os.path.join(ref_run_path, "app_explanations.json")
            if not os.path.exists(app_json):
                print(
                    f"ERROR: apps_from_multimodal_run='{ref_run_path}' "
                    f"has no app_explanations.json."
                )
                return
            try:
                with open(app_json, encoding="utf-8") as f:
                    ref_apps_data = json.load(f)
            except Exception as e:
                print(f"ERROR reading {app_json}: {e}")
                return
            canonical = [a["package_name"] for a in ref_apps_data]
            apps = [p for p in canonical if p in set(all_apps)]
            missing = [p for p in canonical if p not in set(all_apps)]
            print(
                f"\n[INFO] canonical sample inherited from {os.path.basename(ref_run_path)}: "
                f"{len(apps)}/{len(canonical)} apps present in this CSV."
            )
            if missing:
                print(
                    f"   [WARN] {len(missing)} apps of the reference run are missing "
                    f"in this model: "
                    f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
                )
            scope = "canonical"
        else:
            print("\nScope:")
            print(f"  1 - Process ALL {total_apps} apps")
            print(f"  2 - Top-N good + top-N bad by highest confidence (default N=20)")
            scope = input("Choice (1/2) [2]: ").strip() or "2"

        if scope == "canonical":
            # ``apps`` was already filled by the APPS_FROM_MULTIMODAL_RUN logic
            # above, so there is nothing else to do here.
            pass
        elif scope == "2":
            raw_n = input("N per class [20]: ").strip()
            try:
                n_per_class = int(raw_n) if raw_n else 20
            except ValueError:
                n_per_class = 20

            # Aggregate per (app, predicted class) over the correctly classified
            # pairs (``acerto == True``). The explainability analysis focuses on
            # predictions that are both confident and correct; the errors are out of
            # scope here.
            full_df = data_loader.full_classification_df
            correct_df = full_df[full_df['acerto'] == True]
            agg = (
                correct_df.groupby(['package_name', 'predito'])['probabilidade']
                .mean()
                .reset_index()
            )
            agg['confianca_agg'] = (agg['probabilidade'] - 0.5).abs() * 2
            bom = (
                agg[agg['predito'] == 'bom']
                .nlargest(n_per_class, 'confianca_agg')['package_name']
                .tolist()
            )
            ruim = (
                agg[agg['predito'] == 'ruim']
                .nlargest(n_per_class, 'confianca_agg')['package_name']
                .tolist()
            )
            apps = bom + ruim
            n_filtered_pairs = len(full_df) - len(correct_df)
            print(
                f"\nSelected sample: {len(bom)} good + {len(ruim)} bad "
                f"= {len(apps)} apps (top-N by aggregated confidence over the "
                f"correct pairs; {n_filtered_pairs} incorrect pairs dropped from "
                f"the aggregation)."
            )
        else:
            apps = all_apps
            confirm = input(
                f"Confirm processing ALL {total_apps} apps? (y/n) [n]: "
            ).strip().lower()
            if confirm != 'y':
                print("Operation cancelled.")
                return

        total_apps = len(apps)
        for app_idx, package_name in enumerate(apps, start=1):
            # Already finished in the run being continued: its figures, CSVs and
            # dicts are on disk and were reloaded into the output manager.
            if package_name in resumed_apps:
                print(f"\n[{app_idx}/{total_apps}] {package_name}: already done, skipping.")
                continue

            samples = data_loader.get_samples_for_app(package_name)
            if not samples:
                print(f"\n[{app_idx}/{total_apps}] {package_name}: no sample, skipping.")
                continue

            print(f"\n[{app_idx}/{total_apps}] {package_name}: {len(samples)} screens")

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
                samples, desc=f"  screens of {package_name[:30]}", leave=False
            ):
                explanation = explainer.explain_screen(
                    image_path=sample['image_path'],
                    text=sample.get('review_text', f"App {sample['app_name']}"),
                    text_embedding=sample['embedding'],
                    true_label=sample['true_label'],
                    package_name=sample['package_name'],
                    screen_id=sample['screen_id'],
                    output_dir=_screen_dir,
                    review_texts=sample.get('review_texts', []),
                    app_name=sample.get('app_name'),
                    category=sample.get('category')
                )
                if explanation:
                    output_manager.add_screen_explanation(explanation)

            # App-level aggregation over the screens processed in this iteration.
            app_screens = [e for e in output_manager.screen_explanations
                           if e['package_name'] == package_name]
            if app_screens:
                all_reviews = data_loader._load_review_texts(package_name, max_texts=200)
                app_explanation = explainer.explain_app(
                    package_name=package_name,
                    screen_explanations=app_screens,
                    output_dir=_app_dir,
                    all_app_reviews=all_reviews or []
                )
                if app_explanation:
                    output_manager.add_app_explanation(app_explanation)

            # Release the per-review artifact cache of this app, when it exists.
            artifacts = getattr(explainer, "_pr_artifacts_cache", None)
            if isinstance(artifacts, dict):
                artifacts.pop(package_name, None)

    elif mode == "5":
        # Mode 5: explain the errors only
        print("\n" + "-" * 50)
        print("MODE 5: EXPLAIN THE ERRORS ONLY")
        print("-" * 50)

        error_indices = data_loader.get_error_samples()
        print(f"\nTotal classification errors: {len(error_indices)}")

        if len(error_indices) == 0:
            print("No error found.")
            return

        confirm = input(f"Process the {len(error_indices)} errors? (y/n) [y]: ").strip().lower()
        if confirm == 'n':
            print("Operation cancelled.")
            return

        print("\nProcessing the errors...")
        for idx in tqdm(error_indices, desc="Generating explanations"):
            sample = data_loader.get_sample(idx)
            if sample is None:
                continue

            explanation = explainer.explain_screen(
                image_path=sample['image_path'],
                text=sample.get('review_text', f"App {sample['app_name']}"),
                text_embedding=sample['embedding'],
                true_label=sample['true_label'],
                package_name=sample['package_name'],
                screen_id=sample['screen_id'],
                output_dir=output_manager.screen_dir,
                review_texts=sample.get('review_texts', []),
                app_name=sample.get('app_name'),
                category=sample.get('category')
            )
            if explanation:
                output_manager.add_screen_explanation(explanation)

    elif mode == "6":
        # Mode 6: multimodal SHAP explanation (specific screen)
        print("\n" + "-" * 50)
        print("MODE 6: MULTIMODAL SHAP EXPLANATION (SPECIFIC SCREEN)")
        print("-" * 50)

        total = len(data_loader)
        print(f"\nTotal samples: {total}")

        while True:
            try:
                idx = input(f"Sample index (0-{total - 1}): ").strip()
                idx = int(idx)
                if 0 <= idx < total:
                    break
                print(f"Index out of range (0-{total - 1})")
            except ValueError:
                print("Type a valid number.")

        sample = data_loader.get_sample(idx)
        if sample is None:
            print("Error loading the sample.")
            return

        # Collect the individual reviews (up to 5)
        screen_reviews = sample.get('review_texts', [])
        if not screen_reviews:
            # Fallback: use all_review_texts
            all_reviews = sample.get('all_review_texts', [])
            screen_reviews = all_reviews[:5] if all_reviews else [f"App {sample['app_name']}"]

        print(f"\nApp: {sample['app_name']}")
        print(f"Package: {sample['package_name']}")
        print(f"Screen: {sample['screen_id']}")
        print(f"Label: {sample['true_class']}")
        print(f"Available reviews: {len(screen_reviews)}")
        for i, rev in enumerate(screen_reviews[:5]):
            print(f"  [{i+1}] {rev[:80]}{'...' if len(rev) > 80 else ''}")

        print(f"\nWARNING: multimodal SHAP with {len(screen_reviews[:5])} reviews x max_evals={SHAP_MAX_EVALS} may take several minutes...")
        confirm = input("Continue? (y/n) [y]: ").strip().lower()
        if confirm == 'n':
            print("Operation cancelled.")
            return

        # Create the SHAP directory
        os.makedirs(output_manager.shap_dir, exist_ok=True)

        # Instantiate the SHAPExplainer
        print("\nInitializing the SHAP explainer...")
        shap_explainer = SHAPExplainer(model, text_attn)

        prefix = f"{sample['package_name'].replace('.', '_')}_{sample['screen_id']}"

        result = shap_explainer.explain_instance(
            image_path=sample['image_path'],
            review_texts=screen_reviews[:5],
            output_dir=output_manager.shap_dir,
            filename_prefix=prefix
        )

        if result:
            print(f"\nSHAP explanation generated: {result['visualization_path']}")
            print(f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.1%})")
            print(f"Reviews processed: {result['num_reviews']}")

    elif mode == "7":
        # Mode 7: multimodal SHAP explanation (whole app)
        print("\n" + "-" * 50)
        print("MODE 7: MULTIMODAL SHAP EXPLANATION (WHOLE APP)")
        print("-" * 50)

        # List the available apps
        apps = data_loader.get_all_apps()
        print(f"\n{len(apps)} apps available. Examples:")
        for i, app in enumerate(apps[:10]):
            print(f"  {app}")
        if len(apps) > 10:
            print(f"  ... and {len(apps) - 10} more apps")

        package_name = input("\nType the package_name of the app: ").strip()

        if package_name not in apps:
            print(f"App {package_name} not found.")
            return

        samples = data_loader.get_samples_for_app(package_name)
        print(f"\nApp: {package_name}")
        print(f"Available screens: {len(samples)}")

        print(f"\nWARNING: multimodal SHAP over {len(samples)} screens may take a long time...")
        print(f"  (~{len(samples) * 3}-{len(samples) * 10} minutes depending on the hardware)")
        confirm = input("Continue? (y/n) [y]: ").strip().lower()
        if confirm == 'n':
            print("Operation cancelled.")
            return

        # Create the SHAP directory
        os.makedirs(output_manager.shap_dir, exist_ok=True)

        # Instantiate the SHAPExplainer
        print("\nInitializing the SHAP explainer...")
        shap_explainer = SHAPExplainer(model, text_attn)

        result = shap_explainer.explain_app(
            package_name=package_name,
            samples=samples,
            output_dir=output_manager.shap_dir
        )

        if result:
            print(f"\nApp SHAP explanation finished.")
            print(f"  Screens processed: {result['num_screens']}")
            print(f"  Prediction: {result['app_prediction']} (confidence: {result['app_confidence']:.1%})")
            print(f"  Summary: {result['summary_path']}")

    elif mode == "8":
        # Mode 8: top best and worst screens (Grad-CAM)
        print("\n" + "-" * 50)
        print("MODE 8: TOP BEST AND WORST SCREENS (GRAD-CAM)")
        print("-" * 50)

        paths = generate_top_bottom_gradcam(
            data_loader=data_loader,
            gradcam_explainer=gradcam,
            text_attn=text_attn,
            output_dir=output_manager.screen_dir
        )

        if paths:
            print(f"\nFigures generated:")
            for p in paths:
                print(f"  {p}")

    elif mode == "9":
        # Mode 9: top best and worst screens (multimodal SHAP)
        print("\n" + "-" * 50)
        print("MODE 9: TOP BEST AND WORST SCREENS (MULTIMODAL SHAP)")
        print("-" * 50)

        shap_exp = SHAPExplainer(model, text_attn)
        paths = generate_top_bottom_shap(
            data_loader=data_loader,
            shap_explainer=shap_exp,
            output_dir=output_manager.shap_dir
        )

        if paths:
            print(f"\nFigures generated:")
            for p in paths:
                print(f"  {p}")

    elif mode == "10":
        # Mode 10: top best and worst apps (1 screen per app). Renders two Grad-CAM
        # figures (+ Bahdanau) and two SHAP figures (visual + textual), all over the
        # same screens per app.
        print("\n" + "-" * 50)
        print("MODE 10: TOP APPS (1 SCREEN PER APP, GRAD-CAM + SHAP)")
        print("-" * 50)

        # Shared selection: top-K apps by confidence plus the screen with the
        # highest confidence per app (candidates from screen_explanations.json,
        # embedding from the cache). Computed a single time, so Grad-CAM and SHAP
        # show exactly the same screen and the same review.
        selections = _select_top_apps_unique_screens(
            data_loader, gradcam, top_k=10, v2_run_dir=v2_run_dir
        )

        paths = generate_top_bottom_gradcam_unique_apps(
            data_loader=data_loader,
            gradcam_explainer=gradcam,
            text_attn=text_attn,
            bahdanau_attn=bahdanau,
            output_dir=output_manager.screen_dir,
            v2_run_dir=v2_run_dir,
            selections=selections,
        )

        if skip_shap:
            print("\n[Mode 10] SHAP disabled by config (skip_shap=true). "
                  "Grad-CAM + Bahdanau only.")
        else:
            print("\n" + "-" * 50)
            print("MODE 10: RENDERING THE SHAP FIGURES (VISUAL + TEXTUAL)")
            print("-" * 50)
            shap_exp = SHAPExplainer(model, text_attn)
            shap_paths = generate_top_bottom_shap_unique_apps(
                data_loader=data_loader,
                shap_explainer=shap_exp,
                gradcam_explainer=gradcam,
                output_dir=output_manager.shap_dir,
                v2_run_dir=v2_run_dir,
                selections=selections,
            )
            paths = (paths or []) + (shap_paths or [])

        if paths:
            print(f"\nFigures generated:")
            for p in paths:
                print(f"  {p}")

    elif mode == "11":
        # Mode 11: apps from a previous run folder, using the same selection that run
        # made, with Grad-CAM and SHAP produced by the current model. Useful to
        # compare figures of two pipelines side by side over exactly the same apps.
        print("\n" + "-" * 50)
        print("MODE 11: APPS FROM A FOLDER (SAME SELECTION, 1 SCREEN PER APP)")
        print("-" * 50)

        apps_dir = mode11_apps_dir or v2_run_dir
        if not apps_dir:
            print("ERROR: mode 11 requires 'mode11_apps_dir' (or 'mode10_v2_run_dir' "
                  "as a fallback) in the config.")
            return
        apps_dir = os.path.abspath(apps_dir)
        if not os.path.isdir(apps_dir):
            print(f"ERROR: '{apps_dir}' does not exist or is not a directory.")
            return

        print(f"[Mode 11] Source of apps and candidates: {apps_dir}")

        selections = _select_apps_from_folder_unique_screens(
            data_loader, gradcam, apps_dir=apps_dir,
        )

        # Limit each class to the N most confident apps. The selection already comes
        # sorted by aggregated confidence desc, so the leading slice is enough.
        # ``mode11_max_apps`` defaults to 10; use 0 (or a negative value) for no limit.
        mode11_max_apps = int(config.get("mode11_max_apps", 10))
        if mode11_max_apps > 0:
            for _lbl in ("BOM", "RUIM"):
                _sel = selections.get(_lbl, [])
                if len(_sel) > mode11_max_apps:
                    print(f"[Mode 11] Limiting {_lbl}: {len(_sel)} -> "
                          f"{mode11_max_apps} most confident apps.")
                    selections[_lbl] = _sel[:mode11_max_apps]

        paths = generate_top_bottom_gradcam_unique_apps(
            data_loader=data_loader,
            gradcam_explainer=gradcam,
            text_attn=text_attn,
            bahdanau_attn=bahdanau,
            output_dir=output_manager.screen_dir,
            v2_run_dir=apps_dir,
            selections=selections,
            top_k=(mode11_max_apps or 10),
        )

        if skip_shap:
            print("\n[Mode 11] SHAP disabled by config (skip_shap=true). "
                  "Grad-CAM + Bahdanau only.")
        else:
            print("\n" + "-" * 50)
            print("MODE 11: RENDERING THE SHAP FIGURES (VISUAL + TEXTUAL)")
            print("-" * 50)
            shap_exp = SHAPExplainer(model, text_attn)
            shap_paths = generate_top_bottom_shap_unique_apps(
                data_loader=data_loader,
                shap_explainer=shap_exp,
                gradcam_explainer=gradcam,
                output_dir=output_manager.shap_dir,
                v2_run_dir=apps_dir,
                selections=selections,
                top_k=(mode11_max_apps or 10),
            )
            paths = (paths or []) + (shap_paths or [])

        if paths:
            print(f"\nFigures generated:")
            for p in paths:
                print(f"  {p}")

    # Save the results
    output_manager.save_all()

    # Free memory
    text_attn.unload_model()

    print("\nDone.")
