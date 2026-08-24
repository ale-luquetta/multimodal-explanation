#!/usr/bin/env python3
"""
Multimodal app rating analysis - spatial fusion
=================================================================================
The image and text branches are combined by **spatial fusion**: the text
vector is projected, broadcast (tiled) over the 7x7 grid of the MobileNetV2
feature map, concatenated along the channels and processed by a post-fusion
Conv2D. Only then is GlobalAveragePooling2D applied.

Rationale: pooling the image branch before the fusion destroys the spatial
structure, which makes the Grad-CAM look almost identical to the unimodal one
even when the classification differs. Fusing before the pooling lets the text
influence *where* the post-fusion Conv2D concentrates activation.
  Historical baseline in VQA (Antol et al. 2015, Zhou et al. 2015), generalised
  by FiLM (Perez et al. AAAI 2018).
- Grad-CAM target layer in this variant: ``fusion_conv2d``.
- The text projection width for the spatial fusion is set by the key
  ``text_spatial_proj_dim`` in ``configs/multimodal.yaml`` (default 128).
- Results are written to ``results/training/multimodal_v3/`` so they do not overwrite
  v1/v2.

The rest of the pipeline (DeBERTa ABSA + Bahdanau Attention in the text branch,
streaming, metrics, CV, transfer learning of the phase (i) backbone) is the same
as v2.
"""

import os
import gc
import datetime
import json

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
from tqdm import tqdm
import random

from multimodal.common.balancing import balance_split
from multimodal.common.cv import kfold_app_stratified
from multimodal.common.paths import (
    EMBEDDINGS_CACHE_DIR,
    RESULTS_TRAINING_MULTIMODAL_V3,
    REVIEWS_PROCESSED_DIR,
    RICO_DIR,
)
from multimodal.common.splits import split_apps_stratified
from multimodal.common.thresholds import youden_optimal_threshold

warnings.filterwarnings('ignore')

# Settings
RANDOM_SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 3e-5
L2_REG = 1e-4               # L2 regularisation on the Dense and LSTM layers
COLUMN = 'Average Rating Updated'
APP_PERCENTAGE = 0.1
MAX_TEXT_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REVIEWS_FOLDER = str(REVIEWS_PROCESSED_DIR)
EMBEDDINGS_CACHE = str(EMBEDDINGS_CACHE_DIR)
ASPECT = "interface"
RESULTS_DIR = str(RESULTS_TRAINING_MULTIMODAL_V3)

# Optional fine-tuning of the backbone (two-phase training).
# When False (default), only phase 1 (frozen backbone) runs.
FINE_TUNE_BACKBONE = False
FINE_TUNE_LR = 2e-6
FINE_TUNE_EPOCHS = 15
UNFREEZE_LAST_N_LAYERS = 10

# Path to the unimodal image model checkpoint, phase (i).
# When set and present, the MobileNetV2 backbone is initialised from these weights.
# When None or missing, falls back to ImageNet weights.
IMAGE_BACKBONE_CHECKPOINT = None

# Width of the text projection before broadcasting it over the feature-map
# grid (h x w x TEXT_SPATIAL_PROJ_DIM). Set through the config.
TEXT_SPATIAL_PROJ_DIM = 128

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def _resolve_embeddings_cache_dir(config: dict) -> str:
    """Resolve which ``EMBEDDINGS_CACHE_DIR`` subfolder the training should use.

    Resolution order:
    1. When ``config['embeddings_cache_subdir']`` is set and points to an
       existing subfolder holding .npy files, return that subfolder.
    2. Otherwise list the available subfolders plus the root (when it has .npy)
       and show an interactive picker.
    3. When no .npy exists anywhere, return the root; training can generate the
       embeddings through options 1/3 of the submenu.
    """
    from multimodal.common.menu import pick_option

    base = EMBEDDINGS_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)

    subdirs_with_npy = [
        p.name for p in sorted(base.iterdir())
        if p.is_dir() and any(p.glob("*.npy"))
    ]
    root_has_npy = any(base.glob("*.npy"))

    requested = config.get("embeddings_cache_subdir")
    if isinstance(requested, str) and requested:
        candidate = base / requested
        if candidate.exists() and any(candidate.glob("*.npy")):
            return str(candidate)
        print(f"⚠️  Cache '{requested}' not found in {base}; showing the picker.")

    options = list(subdirs_with_npy)
    if root_has_npy:
        options.append("(root — data/embeddings_cache/)")

    if not options:
        print(f"⚠️  No embeddings cache found in {base}. Using the empty root.")
        return str(base)

    if len(options) == 1:
        only = options[0]
        print(f"Only cache available: {only} — using it.")
        return str(base) if only.startswith("(root") else str(base / only)

    idx = pick_option("Select embeddings cache", options, back_label="Cancel")
    if idx is None:
        raise SystemExit("Cancelled by the user.")
    chosen = options[idx]
    return str(base) if chosen.startswith("(root") else str(base / chosen)


# =============================================================================
# STREAMING CLASS: MultimodalSequence
# =============================================================================
class MultimodalSequence(tf.keras.utils.Sequence):
    """
    Loads data on demand, keeping a single batch in memory.
    Allows an arbitrarily large texts_per_image without exhausting memory.

    Estimated memory: ~16 MB per batch (16 samples)
    """

    def __init__(self, index_df, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                 shuffle=True, use_float16_embeddings=True):
        """
        Args:
            index_df: DataFrame with columns [image_path, embedding_file, embedding_idx, label, package_name]
            img_size: image size (224 for MobileNetV2)
            batch_size: batch size
            shuffle: whether to shuffle the indices every epoch
            use_float16_embeddings: whether the embeddings are stored as float16
        """
        self.index_df = index_df.reset_index(drop=True)
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_float16 = use_float16_embeddings
        self.indices = np.arange(len(index_df))

        # Cache of opened embedding files (memory-mapped)
        self._embedding_cache = {}

        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.index_df) / self.batch_size))

    def __getitem__(self, idx):
        """Return one batch of data."""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        batch_rows = self.index_df.iloc[batch_indices]

        images = []
        embeddings = []
        labels = []

        for _, row in batch_rows.iterrows():
            # Load the image from disk
            img = self._load_image(row['image_path'])
            if img is not None:
                images.append(img)

                # Load the embedding from cache (memory-mapped for efficiency)
                emb = self._load_embedding(row['embedding_file'], row['embedding_idx'])
                embeddings.append(emb)

                labels.append(row['label'])

        if len(images) == 0:
            # Fallback: return an empty batch with the right shapes
            return (
                {
                    'image_input': np.zeros((1, self.img_size, self.img_size, 3), dtype=np.float32),
                    'text_embeddings': np.zeros((1, MAX_TEXT_LENGTH, 768), dtype=np.float32)
                },
                np.array([0])
            )

        return (
            {
                'image_input': np.array(images, dtype=np.float32),
                'text_embeddings': np.array(embeddings, dtype=np.float32)
            },
            np.array(labels, dtype=np.float32)
        )

    def _load_image(self, image_path):
        """Load and preprocess one image."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            return img
        except Exception:
            return None

    def _load_embedding(self, embedding_file, embedding_idx):
        """Load the embedding from file using memory-mapping."""
        if embedding_file not in self._embedding_cache:
            # mmap_mode='r' keeps the whole file out of memory
            self._embedding_cache[embedding_file] = np.load(embedding_file, mmap_mode='r')

        emb = self._embedding_cache[embedding_file][embedding_idx]

        # Convert to float32 when needed
        if self.use_float16 and emb.dtype == np.float16:
            emb = emb.astype(np.float32)

        return emb

    def on_epoch_end(self):
        """Called at the end of every epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_labels(self):
        """Return every label (useful for class_weight)."""
        return self.index_df['label'].values

    def get_package_names(self):
        """Return every package name."""
        return self.index_df['package_name'].values


# =============================================================================
# MAIN CLASS: MultimodalAppAnalyzer
# =============================================================================
class MultimodalAppAnalyzer:
    """
    Multimodal architecture with streaming: CNN (MobileNetV2) + BiLSTM + Bahdanau Attention

    Key points:
    - BiLSTM + Bahdanau Attention in the text branch (replaces GlobalAveragePooling1D)
    """

    def __init__(self, data_path=None,
                 images_path=None, img_size=IMG_SIZE, results_dir=RESULTS_DIR,
                 load_deberta=True):
        """
        Args:
            load_deberta: when False, DeBERTa is not loaded (useful with a warm cache)
        """
        self.data_path = data_path if data_path is not None else str(RICO_DIR / "rico_and_sentiment.csv")
        self.images_path = images_path if images_path is not None else str(RICO_DIR / "screenshots")
        self.img_size = img_size
        self.data = None
        self.model = None
        self.history = None
        self.results_dir = results_dir
        self.deberta_model = None
        self.tokenizer = None

        os.makedirs(EMBEDDINGS_CACHE, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        if load_deberta:
            self._load_deberta()

    def _load_deberta(self):
        """Load the DeBERTa model used to extract the embeddings."""
        print("Loading the DeBERTa model...")
        self.tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.deberta_model = AutoModel.from_pretrained("yangheng/deberta-v3-base-absa-v1.1").to(DEVICE)
        self.deberta_model.eval()
        print(f"DeBERTa loaded on {DEVICE}")

    def unload_deberta(self):
        """Unload DeBERTa to free GPU memory once the embeddings exist."""
        if self.deberta_model is not None:
            del self.deberta_model
            self.deberta_model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        torch.cuda.empty_cache()
        gc.collect()
        print("DeBERTa unloaded - GPU memory freed")

    # =========================================================================
    # BACKWARD-COMPATIBILITY HELPERS
    # =========================================================================
    def load_previous_model(self, model_path):
        """
        Load an .h5 model trained earlier.
        Useful to resume training or to run inference.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = load_model(model_path)
        print(f"Model loaded from: {model_path}")
        return self.model

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    def load_data(self, min_screens_per_app=None):
        """Load the data and select the top/bottom apps by rating."""
        print("Loading app data...")
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.dropna(subset=[COLUMN, 'App Package Name'])
        self.data[COLUMN] = pd.to_numeric(self.data[COLUMN], errors='coerce')
        self.data = self.data.dropna(subset=[COLUMN])

        if min_screens_per_app and 'UI Count' in self.data.columns:
            print(f"\nMinimum filter: {min_screens_per_app} screens per app")
            self.data['UI Count'] = pd.to_numeric(self.data['UI Count'], errors='coerce')
            self.data = self.data[self.data['UI Count'] >= min_screens_per_app].copy()

        # Select the top/bottom APP_PERCENTAGE
        self.data = self.data.sort_values(COLUMN, ascending=True)
        n = len(self.data)
        n_x = max(1, int(n * APP_PERCENTAGE))
        worst = self.data.iloc[:n_x].copy()
        best = self.data.iloc[-n_x:].copy()
        worst['label'] = 0
        worst['label_name'] = 'ruim'
        best['label'] = 1
        best['label_name'] = 'bom'
        self.data = pd.concat([worst, best]).reset_index(drop=True)

        print(f"\nApps selected: {len(self.data)}")
        print(self.data['label_name'].value_counts())
        return self.data

    def find_matching_images(self, texts_per_image=50):
        """Match screenshots and load the per-app text lists."""
        print("Looking for matching screenshots and texts...")
        ui_details = pd.read_csv(
            str(RICO_DIR / "ui_details.csv"),
            usecols=[0, 1],
            names=['UI Number', 'App Package Name'],
            header=0
        )

        app_details_df = pd.read_csv(str(RICO_DIR / "app_details.csv"), header=0)

        ui_details = ui_details[ui_details['App Package Name'].isin(self.data['App Package Name'])]

        image_mapping = {}
        text_mapping = {}
        app_index_mapping = {}  # Maps package_name -> formatted_idx

        for _, row in ui_details.iterrows():
            ui_num = row['UI Number']
            img_filename = f"{ui_num}.jpg"
            package_name = row['App Package Name']

            if package_name in self.data['App Package Name'].values:
                matches = app_details_df.index[app_details_df['App Package Name'] == package_name].tolist()
                if not matches:
                    continue
                idx = matches[0]
                formatted_idx = f"{idx:04d}"
                img_path = os.path.join(self.images_path, img_filename)
                if os.path.exists(img_path):
                    image_mapping.setdefault(package_name, []).append(img_path)
                    app_index_mapping[package_name] = formatted_idx
                    if package_name not in text_mapping:
                        texts_list = self.load_screen_text(formatted_idx, package_name, texts_per_image)
                        if texts_list is None or len(texts_list) == 0:
                            texts_list = ["Application screen"]
                        text_mapping[package_name] = texts_list

        self.data['has_images'] = self.data['App Package Name'].isin(image_mapping.keys())
        self.data_with_images = self.data[self.data['has_images']].copy()
        self.app_index_mapping = app_index_mapping

        print(f"Apps with screenshots and text: {len(self.data_with_images)}")
        return image_mapping, text_mapping

    def load_screen_text(self, index, package_name, texts_per_image=50):
        """Load the relevant texts from the app's reviews file."""
        reviews_file = os.path.join(REVIEWS_FOLDER, f"app_reviews_{index}_with_aspects.csv")
        if not os.path.exists(reviews_file):
            return None

        try:
            df_reviews = pd.read_csv(reviews_file)
            if df_reviews.empty:
                return None

            for col in ['interface_pos', 'interface_neg']:
                if col in df_reviews.columns:
                    df_reviews[col] = pd.to_numeric(df_reviews[col], errors='coerce').fillna(0.0)
                else:
                    df_reviews[col] = 0.0

            df_reviews['sentence'] = df_reviews.get('sentence', pd.Series([''] * len(df_reviews))).astype(str)
            df_reviews = df_reviews[df_reviews['sentence'].str.strip() != ""]

            df_reviews['interface_relevance'] = df_reviews['interface_pos']
            df_top = df_reviews.head(texts_per_image)
            texts = df_top['sentence'].dropna().astype(str).tolist()

            if len(texts) == 0:
                texts = df_reviews['sentence'].dropna().astype(str).head(texts_per_image).tolist()

            texts = [t[:2000] for t in texts]
            return texts

        except Exception as e:
            print(f"Error reading the reviews of {package_name}: {e}")
            return None

    # =========================================================================
    # EMBEDDING EXTRACTION
    # =========================================================================
    def extract_deberta_embeddings(self, text):
        """Extract the DeBERTa embeddings of one text."""
        if self.deberta_model is None:
            raise RuntimeError("DeBERTa is not loaded. Use load_deberta=True or _load_deberta()")

        encoded = self.tokenizer(text, text_pair=ASPECT, max_length=MAX_TEXT_LENGTH,
                                 padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoded['input_ids'].to(DEVICE)
        attention_mask = encoded['attention_mask'].to(DEVICE)

        with torch.no_grad():
            outputs = self.deberta_model(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state.squeeze(0)
        embeddings_np = embeddings.cpu().numpy()
        return embeddings_np

    def generate_missing_embeddings_cache(self, texts_per_app: int = 50) -> None:
        """
        Generate DeBERTa ABSA embeddings for apps with no .npy in EMBEDDINGS_CACHE yet.

        Walks every CSV in REVIEWS_FOLDER (reviews_processed) and, for each app
        whose cache is missing, takes the top-N texts and stores the embeddings
        as float16. The operation is resumable: later runs skip cached apps.
        """
        import glob
        from pathlib import Path

        if self.deberta_model is None:
            self._load_deberta()

        csv_pattern = os.path.join(REVIEWS_FOLDER, "app_reviews_*_with_aspects.csv")
        csv_files = sorted(glob.glob(csv_pattern))
        print(f"\nFound {len(csv_files)} processed CSVs in {REVIEWS_FOLDER}")
        print(f"Existing cache: {EMBEDDINGS_CACHE}")
        print(f"Texts per app: {texts_per_app}\n")

        generated = 0
        skipped = 0
        failed = 0

        for csv_file in tqdm(csv_files, desc="Generating embeddings"):
            try:
                df_head = pd.read_csv(csv_file, nrows=1)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                failed += 1
                continue

            if df_head.empty or 'package_name' not in df_head.columns:
                failed += 1
                continue

            package_name = str(df_head['package_name'].iloc[0])
            safe_pkg_name = package_name.replace('.', '_').replace('/', '_')
            emb_file = os.path.join(EMBEDDINGS_CACHE, f"{safe_pkg_name}.npy")

            if os.path.exists(emb_file):
                skipped += 1
                continue

            # Parse NNNN from filename: app_reviews_NNNN_with_aspects.csv
            stem = Path(csv_file).stem
            parts = stem.split('_')
            try:
                index = int(parts[2])
            except (ValueError, IndexError):
                failed += 1
                continue

            texts = self.load_screen_text(index, package_name, texts_per_image=texts_per_app)
            if not texts:
                failed += 1
                continue

            try:
                app_embeddings = [self.extract_deberta_embeddings(text) for text in texts]
                app_embeddings_np = np.array(app_embeddings, dtype=np.float16)
                np.save(emb_file, app_embeddings_np)
                generated += 1
            except Exception as e:
                print(f"Error generating embeddings for {package_name}: {e}")
                failed += 1

        print(f"\n=== Cache generation finished ===")
        print(f"  Generated: {generated}")
        print(f"  Skipped:   {skipped} (already cached)")
        print(f"  Failures:  {failed}")

    # =========================================================================
    # STREAMING INDEX CREATION
    # =========================================================================
    def create_multimodal_index(self, image_mapping, text_mapping,
                                 max_samples_per_class=6000,
                                 max_screens_per_app=8,
                                 texts_per_image=10,
                                 use_cached_index=True):
        """
        Build the index used for data streaming.
        Loads NO data into memory: only builds the mapping and pre-computes
        the embeddings.

        Args:
            texts_per_image: number of texts per image (5, 10, 50+)
            use_cached_index: when True, reuse an existing index if present

        Returns:
            DataFrame holding the index for MultimodalSequence
        """
        index_path = os.path.join(EMBEDDINGS_CACHE, 'index.csv')

        # Check whether the index already exists
        if use_cached_index and os.path.exists(index_path):
            print(f"Loading the existing index: {index_path}")
            index_df = pd.read_csv(index_path)
            print(f"Index loaded: {len(index_df)} samples")
            return index_df

        print(f"\nBuilding the streaming index (texts_per_image={texts_per_image})...")
        print("The first run takes a while, since the embeddings are generated...")

        index_rows = []
        class_counts = {0: 0, 1: 0}

        for _, row in tqdm(self.data_with_images.iterrows(),
                           total=len(self.data_with_images),
                           desc="Processing apps"):
            package_name = row['App Package Name']
            label = int(row['label'])

            if class_counts[label] >= max_samples_per_class:
                continue

            imgs = image_mapping.get(package_name, [])
            texts = text_mapping.get(package_name, ["Application screen"])

            if len(imgs) == 0:
                continue

            imgs = imgs[:max_screens_per_app]

            # Embeddings file for this app
            safe_pkg_name = package_name.replace('.', '_').replace('/', '_')
            emb_file = os.path.join(EMBEDDINGS_CACHE, f"{safe_pkg_name}.npy")

            # Check whether the embeddings already exist
            if os.path.exists(emb_file):
                app_embeddings = np.load(emb_file, mmap_mode='r')
                num_cached_embeddings = len(app_embeddings)
            else:
                # Pre-compute this app's embeddings once
                app_embeddings = []
                for text in texts[:texts_per_image]:  # At most texts_per_image texts per app
                    emb = self.extract_deberta_embeddings(text)
                    app_embeddings.append(emb)

                # Save the app embeddings to their own file (float16)
                app_embeddings = np.array(app_embeddings, dtype=np.float16)
                np.save(emb_file, app_embeddings)
                num_cached_embeddings = len(app_embeddings)

                # Clear the GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Add the index entries
            for img_path in imgs:
                if class_counts[label] >= max_samples_per_class:
                    break

                # Pick the embedding indices for this image
                available_indices = list(range(num_cached_embeddings))
                num_to_select = min(texts_per_image, len(available_indices))

                # Random selection, disabled
                #if len(available_indices) >= texts_per_image:
                #    selected_indices = random.sample(available_indices, num_to_select)
                #else:
                #    selected_indices = random.choices(available_indices, k=num_to_select)

                # Deterministic selection:
                if len(available_indices) >= texts_per_image:
                    step = len(available_indices) / texts_per_image
                    selected_indices = [int(i * step) for i in range(texts_per_image)]
                else:
                    selected_indices = available_indices  # use them all, no duplicates

                for emb_idx in selected_indices:
                    if class_counts[label] >= max_samples_per_class:
                        break

                    index_rows.append({
                        'image_path': img_path,
                        'embedding_file': emb_file,
                        'embedding_idx': emb_idx,
                        'package_name': package_name,
                        'label': label
                    })
                    class_counts[label] += 1

        index_df = pd.DataFrame(index_rows)

        # Save the index
        index_df.to_csv(index_path, index=False)
        print(f"\nIndex built: {len(index_df)} samples")
        print(f"Distribution: {dict(index_df['label'].value_counts())}")
        print(f"Index written to: {index_path}")

        return index_df

    # =========================================================================
    # DATA SPLITTING
    # =========================================================================
    def split_index_by_apps(self, index_df, test_size=0.3, val_size=0.5, balance=True):
        """Split the index per app via common.splits, with optional balancing."""
        apps_train, apps_val, apps_test = split_apps_stratified(
            index_df,
            package_col="package_name",
            label_col="label",
            test_size=test_size,
            val_size=val_size,
            random_seed=RANDOM_SEED,
        )
        return self._filter_and_balance_by_apps(index_df, apps_train, apps_val, apps_test, balance)

    def _filter_and_balance_by_apps(self, index_df, apps_train, apps_val, apps_test, balance=True):
        """Filter index_df by app lists and apply optional balancing; shared by hold-out and CV."""
        train_index = index_df[index_df["package_name"].isin(apps_train)].copy()
        val_index = index_df[index_df["package_name"].isin(apps_val)].copy()
        test_index = index_df[index_df["package_name"].isin(apps_test)].copy()

        print("\nSamples per split (before balancing):")
        print(f"  Train: {len(train_index)} ({dict(train_index['label'].value_counts())})")
        print(f"  Val:   {len(val_index)} ({dict(val_index['label'].value_counts())})")
        print(f"  Test:  {len(test_index)} ({dict(test_index['label'].value_counts())})")

        if balance:
            train_index = balance_split(train_index, label_col="label", random_seed=RANDOM_SEED)
            val_index = balance_split(val_index, label_col="label", random_seed=RANDOM_SEED)
            test_index = balance_split(test_index, label_col="label", random_seed=RANDOM_SEED)

            print("\nSamples per split (after downsampling):")
            print(f"  Train: {len(train_index)} ({dict(train_index['label'].value_counts())})")
            print(f"  Val:   {len(val_index)} ({dict(val_index['label'].value_counts())})")
            print(f"  Test:  {len(test_index)} ({dict(test_index['label'].value_counts())})")

        return train_index, val_index, test_index

    # =========================================================================
    # MODEL
    # =========================================================================
    def _load_unimodal_backbone_weights(self, base_cnn):
        """Load the MobileNetV2 weights from the phase (i) checkpoint.

        The checkpoint is the full Keras model saved by the unimodal training,
        holding MobileNetV2 as a submodel with the classification head attached.
        This isolates the MobileNetV2 submodel and transfers its weights to
        ``base_cnn``.

        When the path is empty or the file is missing, falls back to ImageNet
        with a warning.
        """
        ckpt = IMAGE_BACKBONE_CHECKPOINT
        if not ckpt or not os.path.exists(ckpt):
            print(f"Phase (i) checkpoint not found ({ckpt!r}); falling back to ImageNet.")
            imagenet = MobileNetV2(weights='imagenet', include_top=False,
                                   input_shape=(self.img_size, self.img_size, 3))
            base_cnn.set_weights(imagenet.get_weights())
            return

        print(f"[v2] Loading the backbone weights from phase (i): {ckpt}")
        unimodal = load_model(ckpt, compile=False)

        submodel = None
        for layer in unimodal.layers:
            if isinstance(layer, tf.keras.Model) and 'mobilenet' in layer.name.lower():
                submodel = layer
                break

        if submodel is not None:
            base_cnn.set_weights(submodel.get_weights())
            print(f"MobileNetV2 weights transferred (submodel '{submodel.name}').")
            return

        print("MobileNetV2 submodel not found; trying load_weights(by_name=True, skip_mismatch=True).")
        base_cnn.load_weights(ckpt, by_name=True, skip_mismatch=True)

    def build_multimodal_model(self):
        """Build the multimodal model with spatial fusion (broadcast + Conv2D).

        - The MobileNetV2 spatial feature map (7x7x1280) is preserved up to the
          fusion; GlobalAveragePooling2D happens *after* the post-fusion Conv2D.
          That lets the text influence the spatial structure of the activation
          map before pooling, which is what makes a post-fusion Grad-CAM
          sensitive to the text modality.
        - The text branch produces a (batch, TEXT_SPATIAL_PROJ_DIM) vector that
          is tiled to (batch, 7, 7, TEXT_SPATIAL_PROJ_DIM), concatenated
          channel-wise with the image feature map and passed through a 3x3
          Conv2D.
        """
        print("\nBuilding the multimodal model (spatial fusion: broadcast + post-fusion Conv2D)...")

        # === Image branch (keeps the spatial map) ===
        image_input = layers.Input(shape=(self.img_size, self.img_size, 3), name='image_input')

        base_cnn = MobileNetV2(weights=None, include_top=False,
                               input_shape=(self.img_size, self.img_size, 3))
        self._load_unimodal_backbone_weights(base_cnn)
        base_cnn.trainable = False

        # Stored so the optional fine-tuning phase can run (see unfreeze_backbone).
        self.base_cnn = base_cnn

        # Feature map (h, w, C_img); the GAP only comes after the fusion.
        image_feature_map = base_cnn(image_input)
        image_feature_map = layers.Dropout(0.3, name='image_fmap_dropout')(image_feature_map)

        # === Text branch: Conv1D + Bahdanau attention ===
        text_input = layers.Input(shape=(MAX_TEXT_LENGTH, 768), name='text_embeddings')

        x_text = layers.Dense(128, activation='relu', name='text_projection',
                              kernel_regularizer=regularizers.l2(L2_REG))(text_input)
        x_text = layers.Dropout(0.3)(x_text)

        x_text = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu',
                               name='text_conv1d',
                               kernel_regularizer=regularizers.l2(L2_REG))(x_text)
        x_text = layers.BatchNormalization(name='text_conv_bn')(x_text)
        x_text = layers.Dropout(0.3)(x_text)

        attention_hidden = layers.Dense(128, activation='tanh', name='attention_hidden',
                                        kernel_regularizer=regularizers.l2(L2_REG))(x_text)
        attention_scores = layers.Dense(1, name='attention_score')(attention_hidden)
        attention_weights = layers.Softmax(axis=1, name='attention_softmax')(attention_scores)
        x_text = layers.Multiply(name='attention_multiply')([x_text, attention_weights])
        x_text = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1),
                               name='attention_context')(x_text)  # (batch, 128)
        x_text = layers.Dropout(0.4)(x_text)

        # Text projection whose width controls the text share of the fused map.
        text_features_spatial = layers.Dense(TEXT_SPATIAL_PROJ_DIM, activation='relu',
                                              name='text_spatial_proj',
                                              kernel_regularizer=regularizers.l2(L2_REG))(x_text)

        # === Spatial fusion: broadcast -> concat -> Conv2D ===
        # Image feature-map shape: (batch, h, w, C_img); h=w=7 for MobileNetV2 @224.
        h = image_feature_map.shape[1]
        w = image_feature_map.shape[2]
        proj_dim = TEXT_SPATIAL_PROJ_DIM

        # (batch, P) → (batch, 1, 1, P) → tile → (batch, h, w, P)
        text_spatial = layers.Reshape((1, 1, proj_dim), name='text_reshape')(text_features_spatial)
        text_spatial = layers.Lambda(
            lambda t: tf.tile(t, [1, h, w, 1]),
            output_shape=(h, w, proj_dim),
            name='text_tile',
        )(text_spatial)

        # Concat on the channels: (h, w, C_img) + (h, w, P) → (h, w, C_img + P)
        fused_map = layers.Concatenate(axis=-1, name='spatial_fusion_concat')(
            [image_feature_map, text_spatial]
        )

        # Post-fusion Conv2D, the Grad-CAM target of this architecture.
        fused_map = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu',
                                  name='fusion_conv2d',
                                  kernel_regularizer=regularizers.l2(L2_REG))(fused_map)
        fused_map = layers.BatchNormalization(name='fusion_conv_bn')(fused_map)
        fused_map = layers.Dropout(0.4, name='fusion_conv_dropout')(fused_map)

        # GAP only now, over the fused map.
        fused = layers.GlobalAveragePooling2D(name='fusion_pooling')(fused_map)

        fused = layers.Dense(128, activation='relu', name='fusion_dense1',
                             kernel_regularizer=regularizers.l2(L2_REG))(fused)
        fused = layers.Dropout(0.3)(fused)
        fused = layers.Dense(64, activation='relu', name='fusion_dense2',
                             kernel_regularizer=regularizers.l2(L2_REG))(fused)

        output = layers.Dense(1, activation='sigmoid', name='output')(fused)

        model = Model(inputs=[image_input, text_input], outputs=output,
                      name='multimodal_app_rating_v3_spatial')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )

        print("\nModel architecture (spatial fusion):")
        model.summary()

        self.model = model
        return model

    # =========================================================================
    # TRAINING WITH STREAMING
    # =========================================================================
    def unfreeze_backbone(self):
        """Optional phase 2: unfreeze the last ``UNFREEZE_LAST_N_LAYERS`` layers of
        MobileNetV2 and recompile with ``FINE_TUNE_LR``.
        """
        if not hasattr(self, "base_cnn") or self.base_cnn is None:
            print("⚠️  base_cnn unavailable; skipping the unfreeze.")
            return self.model

        print("\n" + "=" * 60)
        print("PHASE 2: BACKBONE FINE-TUNING")
        print("=" * 60)

        total_layers = len(self.base_cnn.layers)
        freeze_until = max(0, total_layers - UNFREEZE_LAST_N_LAYERS)

        self.base_cnn.trainable = True
        for layer in self.base_cnn.layers[:freeze_until]:
            layer.trainable = False

        n_trainable = sum(1 for l in self.base_cnn.layers if l.trainable)
        n_frozen = sum(1 for l in self.base_cnn.layers if not l.trainable)
        print(f"  MobileNetV2: {n_trainable} layers unfrozen, {n_frozen} frozen")
        print(f"  Learning rate: {LEARNING_RATE} → {FINE_TUNE_LR}")

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )

        trainable_params = sum(tf.keras.backend.count_params(w) for w in self.model.trainable_weights)
        total_params = sum(tf.keras.backend.count_params(w) for w in self.model.weights)
        print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100*trainable_params/total_params:.1f}%)")
        return self.model

    def train_model(self, train_index, val_index):
        """Training in one or two phases.

        Phase 1 (always): frozen backbone, ``EPOCHS`` epochs at ``LEARNING_RATE``.
        Phase 2 (when ``FINE_TUNE_BACKBONE=True``): unfreezes the last
        ``UNFREEZE_LAST_N_LAYERS`` layers and trains for ``FINE_TUNE_EPOCHS``
        epochs at ``FINE_TUNE_LR``.
        """
        train_seq = MultimodalSequence(train_index, batch_size=BATCH_SIZE, shuffle=True)
        val_seq = MultimodalSequence(val_index, batch_size=BATCH_SIZE, shuffle=False)

        # Phase 1: frozen backbone
        print("\n" + "=" * 60)
        print(f"PHASE 1: FROZEN BACKBONE ({EPOCHS} epochs, lr={LEARNING_RATE})")
        print("=" * 60)
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7
        )
        self.history = self.model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=EPOCHS,
            callbacks=[early_stopping, reduce_lr],
            workers=1,
            use_multiprocessing=False,
            verbose=1,
        )

        # Phase 2: optional fine-tuning
        if FINE_TUNE_BACKBONE:
            self.unfreeze_backbone()
            early_stopping_p2 = callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
            reduce_lr_p2 = callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8
            )
            print(f"\nStarting phase 2 ({FINE_TUNE_EPOCHS} epochs, lr={FINE_TUNE_LR})...")
            history_p2 = self.model.fit(
                train_seq,
                validation_data=val_seq,
                epochs=FINE_TUNE_EPOCHS,
                callbacks=[early_stopping_p2, reduce_lr_p2],
                workers=1,
                use_multiprocessing=False,
                verbose=1,
            )
            # Concatenate the histories so plot_training_history shows both phases
            for key, vals in history_p2.history.items():
                self.history.history.setdefault(key, []).extend(vals)

        print("\nTraining finished.")
        return self.history

    # =========================================================================
    # EVALUATION
    # =========================================================================
    def evaluate_model(self, test_index):
        """Evaluate at pair level (screen-review): threshold 0.5 plus Youden's J."""
        print("Evaluating the model (screen-review pair level)...")

        test_seq = MultimodalSequence(test_index, batch_size=BATCH_SIZE, shuffle=False)

        # Predictions
        y_pred_proba = self.model.predict(test_seq)
        y_pred_proba = np.asarray(y_pred_proba).reshape(-1)
        y_pred = (y_pred_proba > 0.5).astype(int)

        y_test = test_index['label'].values

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Ruim', 'Bom'], zero_division=0)

        print(f"\n=== RESULTS (SCREEN-REVIEW PAIR LEVEL, threshold=0.5) ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(report)

        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, ['Ruim', 'Bom'], suffix='_pair_level')

        youden_block = self._build_youden_block(y_test, y_pred_proba, level_label='PAIR LEVEL')

        # Write the metrics to a text file
        self._save_metrics_to_file(
            filename='metrics_pair_level.txt',
            title='METRICS - PAIR LEVEL (SCREEN-REVIEW)',
            accuracy=accuracy,
            report=report,
            confusion_matrix=cm,
            n_samples=len(y_test),
            youden_block=youden_block,
        )

        return y_pred, y_pred_proba, y_test

    def _build_youden_block(self, y_true, y_proba, level_label=''):
        """Compute the Youden's J threshold and return the recomputed metrics.

        Returns None when the optimal threshold equals 0.5.
        """
        y_true = np.asarray(y_true).astype(int).ravel()
        y_proba = np.asarray(y_proba).astype(float).ravel()
        try:
            t_opt = youden_optimal_threshold(y_true, y_proba)
        except ValueError:
            return None
        if abs(t_opt - 0.5) < 1e-6:
            return None
        y_pred_opt = (y_proba > t_opt).astype(int)
        acc = accuracy_score(y_true, y_pred_opt)
        report = classification_report(y_true, y_pred_opt, target_names=['Ruim', 'Bom'], zero_division=0)
        cm = confusion_matrix(y_true, y_pred_opt)
        print(f"\n=== YOUDEN'S J ({level_label}) — optimal threshold: {t_opt:.4f} ===")
        print(f"Accuracy: {acc:.4f}")
        print(report)
        return {'threshold': t_opt, 'accuracy': acc, 'report': report, 'confusion_matrix': cm}

    # =========================================================================
    # PLOTTING AND SAVING
    # =========================================================================
    def _save_metrics_to_file(self, filename, title, accuracy, report, confusion_matrix,
                               n_samples, extra_info=None, youden_block=None):
        """Write the evaluation metrics to a text file.

        When ``youden_block`` is given (dict with 'threshold', 'accuracy', 'report',
        'confusion_matrix'), adds a secondary section with the metrics
        recomputed at the optimal Youden's J threshold.
        """
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Data/Hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de amostras: {n_samples}\n")
            if extra_info:
                f.write(f"{extra_info}\n")
            f.write("\n")

            f.write("-" * 70 + "\n")
            f.write("ACCURACY\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")

            f.write("-" * 70 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("-" * 70 + "\n")
            f.write(report)
            f.write("\n")

            f.write("-" * 70 + "\n")
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 70 + "\n")
            f.write("              Predito\n")
            f.write("            Ruim    Bom\n")
            f.write(f"Real Ruim   {confusion_matrix[0][0]:4d}   {confusion_matrix[0][1]:4d}\n")
            f.write(f"     Bom    {confusion_matrix[1][0]:4d}   {confusion_matrix[1][1]:4d}\n\n")

            # Metrics derived from the confusion matrix
            tn, fp, fn, tp = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]
            f.write("-" * 70 + "\n")
            f.write("DETAILED METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Verdadeiros Negativos (TN): {tn}\n")
            f.write(f"Falsos Positivos (FP): {fp}\n")
            f.write(f"Falsos Negativos (FN): {fn}\n")
            f.write(f"Verdadeiros Positivos (TP): {tp}\n\n")

            if (tp + fp) > 0:
                precision = tp / (tp + fp)
                f.write(f"Precision (good class): {precision:.4f}\n")
            if (tp + fn) > 0:
                recall = tp / (tp + fn)
                f.write(f"Recall (good class): {recall:.4f}\n")
            if (tn + fn) > 0:
                specificity = tn / (tn + fp)
                f.write(f"Specificity (bad class): {specificity:.4f}\n")

            if youden_block is not None:
                yb_cm = youden_block['confusion_matrix']
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"THRESHOLD OTIMIZADO (Youden's J) — t = {youden_block['threshold']:.4f}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Accuracy: {youden_block['accuracy']:.4f} "
                        f"({youden_block['accuracy']*100:.2f}%)\n\n")
                f.write("CLASSIFICATION REPORT (optimised threshold)\n")
                f.write("-" * 70 + "\n")
                f.write(youden_block['report'])
                f.write("\n")
                f.write("CONFUSION MATRIX (optimised threshold)\n")
                f.write("-" * 70 + "\n")
                f.write("              Predito\n")
                f.write("            Ruim    Bom\n")
                f.write(f"Real Ruim   {yb_cm[0][0]:4d}   {yb_cm[0][1]:4d}\n")
                f.write(f"     Bom    {yb_cm[1][0]:4d}   {yb_cm[1][1]:4d}\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"Metrics written to: {filepath}")

    def plot_confusion_matrix(self, cm, classes, suffix=''):
        """Plot the confusion matrix and save it under results_dir."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion matrix')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predito')
        plt.tight_layout()

        filename = f'confusion_matrix{suffix}.png'
        path = os.path.join(self.results_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix written to: {path}")

    def plot_training_history(self):
        """Plot the training history."""
        if self.history is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(self.history.history['accuracy'], label='Treino')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].legend()

        axes[0, 1].plot(self.history.history['loss'], label='Treino')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Loss')
        axes[0, 1].legend()

        axes[1, 0].plot(self.history.history['precision'], label='Treino')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].legend()

        axes[1, 1].plot(self.history.history['recall'], label='Treino')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_history.png'), dpi=300)
        plt.close()

    def analyze_errors(self, test_index, y_pred, y_pred_proba):
        """Analyse the misclassifications."""
        y_test = test_index['label'].values
        pk_test = test_index['package_name'].values
        img_paths_test = test_index['image_path'].values
        emb_idx_test = test_index['embedding_idx'].values

        print("\n=== ERROR ANALYSIS ===")
        errors = y_test != y_pred
        error_indices = np.where(errors)[0]
        print(f"Total errors: {len(error_indices)}")

        if len(error_indices) == 0:
            print("No misclassification found.")
        else:
            print("\nExamples of misclassifications:")
            for i, idx in enumerate(error_indices[:5]):
                true_label = y_test[idx]
                pred_label = y_pred[idx]
                package_name = pk_test[idx]
                img_path = img_paths_test[idx]

                print(f"\nError {i + 1}:")
                print(f"  True class: {'Bom' if true_label == 1 else 'Ruim'}")
                print(f"  Predicted: {'Bom' if pred_label == 1 else 'Ruim'}")

                app_rows = self.data_with_images[
                    self.data_with_images['App Package Name'] == package_name
                    ]

                if len(app_rows) > 0:
                    app_info = app_rows.iloc[0]
                    app_name = app_info['App'] if 'App' in app_info else '-'
                    rating = app_info[COLUMN] if COLUMN in app_info else '-'
                    label_name = app_info['label_name'] if 'label_name' in app_info else '-'
                    categoria = app_info['Category'] if 'Category' in app_info else '-'

                    print(f"  Package Name: {package_name}")
                    print(f"  App: {app_name}")
                    print(f"  Rating: {rating}")
                    print(f"  True label: {label_name}")
                    print(f"  Category: {categoria}")

                    ui_number = os.path.splitext(os.path.basename(img_path))[0]
                    print(f"  Screen number (UI Number): {ui_number}")

        todas_classificacoes = []
        erros_para_csv = []

        for idx in range(len(y_test)):
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            is_error = true_label != pred_label
            package_name = pk_test[idx]
            img_path = img_paths_test[idx]

            app_rows = self.data_with_images[
                self.data_with_images['App Package Name'] == package_name
                ]

            if len(app_rows) > 0:
                app_info = app_rows.iloc[0]
                app_name = app_info.get('App', '-')
                rating = app_info.get(COLUMN, '-')
                label_name = app_info.get('label_name', '-')
                categoria = app_info.get('Category', '-')
            else:
                app_name = '-'
                rating = '-'
                label_name = '-'
                categoria = '-'

            numero_tela = os.path.splitext(os.path.basename(img_path))[0]

            prob = float(y_pred_proba[idx])
            confianca = abs(prob - 0.5) * 2  # 0 = undecided, 1 = fully confident

            classificacao = {
                'valor_real': 'bom' if true_label == 1 else 'ruim',
                'predito': 'bom' if pred_label == 1 else 'ruim',
                'probabilidade': round(prob, 4),
                'confianca': round(confianca, 4),
                'acerto': not is_error,
                'package_name': package_name,
                'app': app_name,
                'rating': rating,
                'label_real': label_name,
                'categoria': categoria,
                'numero_da_tela': numero_tela,
                'embedding_idx': int(emb_idx_test[idx])
            }
            todas_classificacoes.append(classificacao)

            if is_error:
                erros_para_csv.append({
                    'valor_real': 'bom' if true_label == 1 else 'ruim',
                    'predito': 'bom' if pred_label == 1 else 'ruim',
                    'probabilidade': round(prob, 4),
                    'confianca': round(confianca, 4),
                    'package_name': package_name,
                    'app': app_name,
                    'rating': rating,
                    'label_real': label_name,
                    'categoria': categoria,
                    'numero_da_tela': numero_tela,
                    'embedding_idx': int(emb_idx_test[idx])
                })

        if todas_classificacoes:
            df_all = pd.DataFrame(todas_classificacoes)
            df_all.to_csv(os.path.join(self.results_dir, 'classification_all.csv'), index=False)
            print(
                f"\nWrote '{self.results_dir}/classification_all.csv' with every classification ({len(todas_classificacoes)} rows).")
            acertos = df_all['acerto'].sum()
            total = len(df_all)
            acuracia = acertos / total
            print(f"Accuracy: {acertos}/{total} = {acuracia:.4f}")

        if erros_para_csv:
            df_errors = pd.DataFrame(erros_para_csv)
            df_errors.to_csv(os.path.join(self.results_dir, 'classification_errors.csv'), index=False)
            print(
                f"Wrote '{self.results_dir}/classification_errors.csv' with the misclassifications ({len(erros_para_csv)} rows).")

            # Errors aggregated per screen
            df_errors_by_screen = df_errors.groupby(['package_name', 'numero_da_tela']).agg(
                probabilidade_media=('probabilidade', 'mean'),
                confianca_media=('confianca', 'mean'),
                num_reviews=('probabilidade', 'count'),
                valor_real=('valor_real', 'first'),
                predito=('predito', 'first'),
                app=('app', 'first'),
                rating=('rating', 'first'),
                label_real=('label_real', 'first'),
                categoria=('categoria', 'first'),
            ).reset_index()
            df_errors_by_screen.to_csv(os.path.join(self.results_dir, 'classification_errors_by_screen.csv'), index=False)
            print(f"Wrote '{self.results_dir}/classification_errors_by_screen.csv' ({len(df_errors_by_screen)} screens with errors)")

            # Errors aggregated per app
            df_errors_by_app = df_errors_by_screen.groupby('package_name').agg(
                probabilidade_media=('probabilidade_media', 'mean'),
                confianca_media=('confianca_media', 'mean'),
                num_telas=('numero_da_tela', 'count'),
                valor_real=('valor_real', 'first'),
                predito=('predito', 'first'),
                app=('app', 'first'),
                rating=('rating', 'first'),
                label_real=('label_real', 'first'),
                categoria=('categoria', 'first'),
            ).reset_index()
            df_errors_by_app.to_csv(os.path.join(self.results_dir, 'classification_errors_by_app.csv'), index=False)
            print(f"Wrote '{self.results_dir}/classification_errors_by_app.csv' ({len(df_errors_by_app)} apps with errors)")

        # Per-screen aggregation (mean probability over the screen-review pairs)
        if todas_classificacoes:
            df_by_screen = df_all.groupby(['package_name', 'numero_da_tela']).agg(
                probabilidade_media=('probabilidade', 'mean'),
                #confianca_media=('confianca', 'mean'),
                num_reviews=('probabilidade', 'count'),
                valor_real=('valor_real', 'first'),
                app=('app', 'first'),
                rating=('rating', 'first'),
                label_real=('label_real', 'first'),
                categoria=('categoria', 'first'),
            ).reset_index()
            df_by_screen['confianca_agregada'] = (df_by_screen['probabilidade_media'] - 0.5).abs() * 2
            df_by_screen['predito'] = df_by_screen['probabilidade_media'].apply(lambda p: 'bom' if p > 0.5 else 'ruim')
            df_by_screen['acerto'] = df_by_screen['valor_real'] == df_by_screen['predito']
            df_by_screen.to_csv(os.path.join(self.results_dir, 'classification_by_screen.csv'), index=False)
            acertos_tela = df_by_screen['acerto'].sum()
            total_tela = len(df_by_screen)
            print(f"Wrote '{self.results_dir}/classification_by_screen.csv' ({total_tela} screens, accuracy: {acertos_tela}/{total_tela} = {acertos_tela/total_tela:.4f})")

            # Errors aggregated per screen
            df_errors_screen = df_by_screen[~df_by_screen['acerto']]
            if len(df_errors_screen) > 0:
                df_errors_screen.to_csv(os.path.join(self.results_dir, 'classification_errors_by_screen.csv'), index=False)
                print(f"Wrote '{self.results_dir}/classification_errors_by_screen.csv' ({len(df_errors_screen)} screens with errors)")

            # Screen-level metrics
            y_screen_true = (df_by_screen['valor_real'] == 'bom').astype(int).values
            y_screen_pred = (df_by_screen['predito'] == 'bom').astype(int).values
            acc_screen = accuracy_score(y_screen_true, y_screen_pred)
            report_screen = classification_report(y_screen_true, y_screen_pred, target_names=['Ruim', 'Bom'], zero_division=0)
            cm_screen = confusion_matrix(y_screen_true, y_screen_pred)

            print(f"\n=== RESULTS (SCREEN LEVEL) ===")
            print(f"Accuracy: {acc_screen:.4f}")
            print(report_screen)

            self.plot_confusion_matrix(cm_screen, ['Ruim', 'Bom'], suffix='_screen_level')
            self._save_metrics_to_file(
                filename='metrics_screen_level.txt',
                title='METRICS - SCREEN LEVEL (AGGREGATED)',
                accuracy=acc_screen,
                report=report_screen,
                confusion_matrix=cm_screen,
                n_samples=total_tela,
                extra_info=f"Total de telas avaliadas: {total_tela}"
            )

            # Per-app aggregation (mean probability over the app's screens)
            df_by_app = df_by_screen.groupby('package_name').agg(
                probabilidade_media=('probabilidade_media', 'mean'),
                #confianca_media=('confianca_media', 'mean'),
                num_telas=('numero_da_tela', 'count'),
                valor_real=('valor_real', 'first'),
                app=('app', 'first'),
                rating=('rating', 'first'),
                label_real=('label_real', 'first'),
                categoria=('categoria', 'first'),
            ).reset_index()
            df_by_app['confianca_agregada'] = (df_by_app['probabilidade_media'] - 0.5).abs() * 2
            df_by_app['predito'] = df_by_app['probabilidade_media'].apply(lambda p: 'bom' if p > 0.5 else 'ruim')
            df_by_app['acerto'] = df_by_app['valor_real'] == df_by_app['predito']
            df_by_app.to_csv(os.path.join(self.results_dir, 'classification_by_app.csv'), index=False)
            acertos_app = df_by_app['acerto'].sum()
            total_app = len(df_by_app)
            print(f"Wrote '{self.results_dir}/classification_by_app.csv' ({total_app} apps, accuracy: {acertos_app}/{total_app} = {acertos_app/total_app:.4f})")

            # Errors aggregated per app
            df_errors_app = df_by_app[~df_by_app['acerto']]
            if len(df_errors_app) > 0:
                df_errors_app.to_csv(os.path.join(self.results_dir, 'classification_errors_by_app.csv'), index=False)
                print(f"Wrote '{self.results_dir}/classification_errors_by_app.csv' ({len(df_errors_app)} apps with errors)")

            # App-level metrics
            y_app_true = (df_by_app['valor_real'] == 'bom').astype(int).values
            y_app_pred = (df_by_app['predito'] == 'bom').astype(int).values
            acc_app = accuracy_score(y_app_true, y_app_pred)
            report_app = classification_report(y_app_true, y_app_pred, target_names=['Ruim', 'Bom'], zero_division=0)
            cm_app = confusion_matrix(y_app_true, y_app_pred)

            print(f"\n=== RESULTS (APP LEVEL) ===")
            print(f"Accuracy: {acc_app:.4f}")
            print(report_app)

            self.plot_confusion_matrix(cm_app, ['Ruim', 'Bom'], suffix='_app_level')
            self._save_metrics_to_file(
                filename='metrics_app_level.txt',
                title='METRICS - APP LEVEL (AGGREGATED)',
                accuracy=acc_app,
                report=report_app,
                confusion_matrix=cm_app,
                n_samples=total_app,
                extra_info=f"Total de apps avaliados: {total_app}"
            )

    def save_full_dataset_predictions(self, train_index, val_index, test_index):
        """Run inference over train+val+test and write classification_full_dataset.csv.

        Serves explainability: lets it iterate over every app of the filtered dataset, not only the test split.
        """
        print("\n" + "=" * 80)
        print("📊 PREDICTIONS OVER THE FULL DATASET (train + val + test)")
        print("=" * 80)

        full_index = pd.concat(
            [
                train_index.assign(split='train'),
                val_index.assign(split='val'),
                test_index.assign(split='test'),
            ],
            ignore_index=True,
        )
        print(f"Total samples: {len(full_index)}")

        full_seq = MultimodalSequence(full_index, batch_size=BATCH_SIZE, shuffle=False)
        y_proba = np.asarray(self.model.predict(full_seq)).reshape(-1)
        y_pred = (y_proba > 0.5).astype(int)
        y_true = full_index['label'].values

        rows = []
        for i in range(len(full_index)):
            true_label = int(y_true[i])
            pred_label = int(y_pred[i])
            proba = float(y_proba[i])
            row = full_index.iloc[i]
            package_name = row['package_name']
            img_path = row['image_path']
            split = row['split']

            app_rows = self.data_with_images[self.data_with_images['App Package Name'] == package_name]
            if len(app_rows) > 0:
                info = app_rows.iloc[0]
                app_name = info.get('App', '-')
                rating = info.get(COLUMN, '-')
                label_name = info.get('label_name', '-')
                categoria = info.get('Category', '-')
            else:
                app_name = rating = label_name = categoria = '-'

            numero_tela = os.path.splitext(os.path.basename(img_path))[0]

            rows.append({
                'split': split,
                'valor_real': 'bom' if true_label == 1 else 'ruim',
                'predito': 'bom' if pred_label == 1 else 'ruim',
                'acerto': true_label == pred_label,
                'probabilidade': proba,
                'confianca': round(abs(proba - 0.5) * 2, 4),
                'package_name': package_name,
                'app': app_name,
                'rating': rating,
                'label_real': label_name,
                'categoria': categoria,
                'numero_da_tela': numero_tela,
                'embedding_idx': int(row['embedding_idx']) if 'embedding_idx' in row else -1,
            })

        df_full = pd.DataFrame(rows)
        out_path = os.path.join(self.results_dir, 'classification_full_dataset.csv')
        df_full.to_csv(out_path, index=False)

        print(f"Wrote: {out_path} ({len(df_full)} rows)")
        for split_name in ['train', 'val', 'test']:
            n = int((df_full['split'] == split_name).sum())
            print(f"  {split_name}: {n}")

        # Per-screen aggregation (mean probability across reviews).
        df_by_screen = df_full.groupby(['package_name', 'numero_da_tela']).agg(
            probabilidade_media=('probabilidade', 'mean'),
            num_reviews=('probabilidade', 'count'),
            split=('split', 'first'),
            valor_real=('valor_real', 'first'),
            app=('app', 'first'),
            rating=('rating', 'first'),
            label_real=('label_real', 'first'),
            categoria=('categoria', 'first'),
        ).reset_index()
        df_by_screen['confianca_agregada'] = (df_by_screen['probabilidade_media'] - 0.5).abs() * 2
        df_by_screen['predito'] = df_by_screen['probabilidade_media'].apply(
            lambda p: 'bom' if p > 0.5 else 'ruim'
        )
        df_by_screen['acerto'] = df_by_screen['valor_real'] == df_by_screen['predito']
        screen_path = os.path.join(self.results_dir, 'classification_full_by_screen.csv')
        df_by_screen.to_csv(screen_path, index=False)
        print(f"Wrote: {screen_path} ({len(df_by_screen)} screens)")

        # Per-app aggregation (mean across screens).
        df_by_app = df_by_screen.groupby('package_name').agg(
            probabilidade_media=('probabilidade_media', 'mean'),
            num_telas=('numero_da_tela', 'count'),
            split=('split', 'first'),
            valor_real=('valor_real', 'first'),
            app=('app', 'first'),
            rating=('rating', 'first'),
            label_real=('label_real', 'first'),
            categoria=('categoria', 'first'),
        ).reset_index()
        df_by_app['confianca_agregada'] = (df_by_app['probabilidade_media'] - 0.5).abs() * 2
        df_by_app['predito'] = df_by_app['probabilidade_media'].apply(
            lambda p: 'bom' if p > 0.5 else 'ruim'
        )
        df_by_app['acerto'] = df_by_app['valor_real'] == df_by_app['predito']
        app_path = os.path.join(self.results_dir, 'classification_full_by_app.csv')
        df_by_app.to_csv(app_path, index=False)
        print(f"Wrote: {app_path} ({len(df_by_app)} apps)")

    def save_model(self, filename='app_rating_model.h5'):
        """Save the model."""
        if self.model is not None:
            path = os.path.join(self.results_dir, filename)
            self.model.save(path)
            print(f"Model saved: {path}")

    # =========================================================================
    # FULL PIPELINE
    # =========================================================================
    def run_complete_analysis(self, min_screens_per_app=3, max_screens_per_app=8,
                               texts_per_image=10, max_samples_per_class=6000,
                               use_cached_index=True, balance_splits=True,
                               use_cross_validation=False, cv_k=5):
        """Full streaming pipeline; supports single hold-out or cross-validation.

        Args:
            texts_per_image: texts per image during training.
            max_samples_per_class: cap on samples per class in the index.
            use_cached_index: reuse an existing index when available.
            balance_splits: balances each split by downsampling.
            use_cross_validation: when True, runs app-stratified k-fold.
            cv_k: number of folds.
        """
        print("=" * 80)
        print("MULTIMODAL APP RATING ANALYSIS - SPATIAL FUSION")
        print("=" * 80 + "\n")
        if use_cross_validation:
            print(f"🔁 Mode: cross-validation (k={cv_k})")
        else:
            print("🆕 Mode: single hold-out")

        # 1. Build the index once, shared across folds
        index_df = self._prepare_index(
            min_screens_per_app=min_screens_per_app,
            max_screens_per_app=max_screens_per_app,
            texts_per_image=texts_per_image,
            max_samples_per_class=max_samples_per_class,
            use_cached_index=use_cached_index,
        )

        base_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        if use_cross_validation:
            cv_root = RESULTS_TRAINING_MULTIMODAL_V3 / f"cv_{base_time}"
            cv_root.mkdir(parents=True, exist_ok=True)

            folds = list(kfold_app_stratified(
                index_df,
                k=cv_k,
                package_col="package_name",
                label_col="label",
                random_seed=RANDOM_SEED,
            ))

            for fold_idx, (apps_train, apps_val, apps_test) in enumerate(folds, start=1):
                fold_dir = cv_root / f"fold_{fold_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                self.results_dir = str(fold_dir)
                self.model = None
                self.history = None

                print("\n" + "=" * 80)
                print(f"🔁 FOLD {fold_idx} / {cv_k}")
                print("=" * 80)

                self._run_fold_from_apps(
                    index_df=index_df,
                    apps_train=apps_train, apps_val=apps_val, apps_test=apps_test,
                    balance_splits=balance_splits,
                )

            self._aggregate_cv_metrics(cv_root, cv_k)
            print(f"\n=== CROSS-VALIDATION FINISHED ===")
            print(f"Consolidated results in: {cv_root}/")
        else:
            self.start_time = base_time
            self.results_dir = str(RESULTS_TRAINING_MULTIMODAL_V3 / f"resultados_{self.start_time}")
            os.makedirs(self.results_dir, exist_ok=True)

            train_index, val_index, test_index = self.split_index_by_apps(
                index_df, balance=balance_splits
            )
            self._run_fold_internal(train_index, val_index, test_index)

            print("\nMultimodal analysis finished.")
            print(f"\nResults saved in: {self.results_dir}/")
            print(f"To reprocess the embeddings, delete the folder {EMBEDDINGS_CACHE}/")

    def _prepare_index(self, min_screens_per_app, max_screens_per_app,
                       texts_per_image, max_samples_per_class, use_cached_index):
        """Load data, match screenshots, build the streaming index, then unload DeBERTa."""
        self.load_data(min_screens_per_app=min_screens_per_app)

        print("\nLoading images and texts...")
        image_mapping, text_mapping = self.find_matching_images(texts_per_image=texts_per_image)

        index_df = self.create_multimodal_index(
            image_mapping, text_mapping,
            max_samples_per_class=max_samples_per_class,
            max_screens_per_app=max_screens_per_app,
            texts_per_image=texts_per_image,
            use_cached_index=use_cached_index,
        )
        self.unload_deberta()
        return index_df

    def _run_fold_from_apps(self, index_df, apps_train, apps_val, apps_test, balance_splits):
        """Filter the index by app, balance it and run the full train/evaluate pipeline."""
        train_index, val_index, test_index = self._filter_and_balance_by_apps(
            index_df, apps_train, apps_val, apps_test, balance=balance_splits,
        )
        self._run_fold_internal(train_index, val_index, test_index)

    def _run_fold_internal(self, train_index, val_index, test_index):
        """Build, train, evaluate and save one fold from ready-made splits."""
        self.build_multimodal_model()
        self.train_model(train_index, val_index)

        y_pred, y_pred_proba, _ = self.evaluate_model(test_index)
        self.plot_training_history()
        self.analyze_errors(test_index, y_pred, y_pred_proba)
        self.save_full_dataset_predictions(train_index, val_index, test_index)
        self.save_model()

    def _aggregate_cv_metrics(self, cv_root, n_folds):
        """Aggregate per-fold metrics (pair and app), summarise as mean ± sd, and copy the fold
        with the best ``accuracy_app`` to ``cv_root/best_fold/``."""
        import shutil as _shutil
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        rows = []
        for fold_idx in range(1, n_folds + 1):
            fold_dir = cv_root / f"fold_{fold_idx}"
            try:
                df_pair = pd.read_csv(fold_dir / "classification_all.csv")
                y_true_p = (df_pair['valor_real'] == 'bom').astype(int).values
                y_pred_p = (df_pair['predito'] == 'bom').astype(int).values
                df_app = pd.read_csv(fold_dir / "classification_by_app.csv")
                y_true_a = (df_app['valor_real'] == 'bom').astype(int).values
                y_pred_a = (df_app['predito'] == 'bom').astype(int).values
            except FileNotFoundError as e:
                print(f"⚠️  Fold {fold_idx}: missing CSV ({e}). Skipping.")
                continue

            rows.append({
                'fold': fold_idx,
                'accuracy_pair': accuracy_score(y_true_p, y_pred_p),
                'precision_pair': precision_score(y_true_p, y_pred_p, zero_division=0),
                'recall_pair': recall_score(y_true_p, y_pred_p, zero_division=0),
                'f1_pair': f1_score(y_true_p, y_pred_p, zero_division=0),
                'accuracy_app': accuracy_score(y_true_a, y_pred_a),
                'precision_app': precision_score(y_true_a, y_pred_a, zero_division=0),
                'recall_app': recall_score(y_true_a, y_pred_a, zero_division=0),
                'f1_app': f1_score(y_true_a, y_pred_a, zero_division=0),
            })

        if not rows:
            print("⚠️  No fold produced a CSV — aggregation skipped.")
            return

        df = pd.DataFrame(rows)
        df.to_csv(cv_root / 'cv_summary.csv', index=False)

        best_row = df.loc[df['accuracy_app'].idxmax()]
        best_fold_idx = int(best_row['fold'])
        best_acc_app = float(best_row['accuracy_app'])

        metrics_cols = [c for c in df.columns if c != 'fold']
        with open(cv_root / 'cv_metrics_aggregated.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"CROSS-VALIDATION RESULTS (k={n_folds})\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Data/Hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"{'Metric':<25} {'Mean':>10} {'Std':>10}\n")
            f.write("-" * 70 + "\n")
            for col in metrics_cols:
                f.write(f"{col:<25} {df[col].mean():>10.4f} {df[col].std():>10.4f}\n")
            f.write("\n")
            f.write(f"🏆 Melhor fold por accuracy_app: fold_{best_fold_idx} "
                    f"(accuracy_app = {best_acc_app:.4f})\n")
            f.write(f"    Copiado para: {cv_root.name}/best_fold/ (use na explicabilidade)\n")

        # Copy the best fold to best_fold/ so explainability can pick it directly
        best_src = cv_root / f"fold_{best_fold_idx}"
        best_dst = cv_root / "best_fold"
        if best_dst.exists():
            _shutil.rmtree(best_dst)
        _shutil.copytree(best_src, best_dst)

        print(f"\n📊 CV summary: {cv_root}/cv_summary.csv")
        print(f"📊 Aggregated metrics: {cv_root}/cv_metrics_aggregated.txt")
        print(f"🏆 Best fold by accuracy_app: fold_{best_fold_idx} "
              f"(accuracy_app = {best_acc_app:.4f})")
        print(f"    Copy available at: {cv_root}/best_fold/")


# =============================================================================
# ENTRY POINT (called from multimodal.cli)
# =============================================================================
def train(config: dict) -> None:
    """Multimodal training entry point driven by a YAML config dict."""
    global RANDOM_SEED, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, L2_REG
    global COLUMN, APP_PERCENTAGE, MAX_TEXT_LENGTH, ASPECT
    global FINE_TUNE_BACKBONE, FINE_TUNE_LR, FINE_TUNE_EPOCHS, UNFREEZE_LAST_N_LAYERS
    global IMAGE_BACKBONE_CHECKPOINT
    global TEXT_SPATIAL_PROJ_DIM
    global EMBEDDINGS_CACHE

    RANDOM_SEED = config.get("random_seed", RANDOM_SEED)
    IMG_SIZE = config.get("img_size", IMG_SIZE)
    BATCH_SIZE = config.get("batch_size", BATCH_SIZE)
    EPOCHS = config.get("epochs", EPOCHS)
    LEARNING_RATE = config.get("learning_rate", LEARNING_RATE)
    L2_REG = config.get("l2_reg", L2_REG)
    COLUMN = config.get("column", COLUMN)
    APP_PERCENTAGE = config.get("app_percentage", APP_PERCENTAGE)
    MAX_TEXT_LENGTH = config.get("max_text_length", MAX_TEXT_LENGTH)
    ASPECT = config.get("aspect", ASPECT)
    FINE_TUNE_BACKBONE = bool(config.get("fine_tune_backbone", FINE_TUNE_BACKBONE))
    FINE_TUNE_LR = float(config.get("fine_tune_lr", FINE_TUNE_LR))
    FINE_TUNE_EPOCHS = int(config.get("fine_tune_epochs", FINE_TUNE_EPOCHS))
    UNFREEZE_LAST_N_LAYERS = int(config.get("unfreeze_last_n_layers", UNFREEZE_LAST_N_LAYERS))
    IMAGE_BACKBONE_CHECKPOINT = config.get("image_backbone_checkpoint", IMAGE_BACKBONE_CHECKPOINT)
    TEXT_SPATIAL_PROJ_DIM = int(config.get("text_spatial_proj_dim", TEXT_SPATIAL_PROJ_DIM))

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("=" * 80)
    print("MULTIMODAL MODEL SETUP - SPATIAL FUSION")
    print("=" * 80)

    if not os.path.exists(REVIEWS_FOLDER):
        print(f"\nWARNING: folder '{REVIEWS_FOLDER}' not found.")
        input("\nPress ENTER to continue anyway...")

    EMBEDDINGS_CACHE = _resolve_embeddings_cache_dir(config)

    print(f"\nDevice: {DEVICE}")
    print(f"Embeddings cache: {EMBEDDINGS_CACHE}/")
    print(f"Results will be written to: {RESULTS_DIR}/")

    print("\n" + "-" * 80)
    print("OPTIONS:")
    print("  1 - Train a new model (build the index and embeddings)")
    print("  2 - Use the cached index (faster)")
    print("  3 - Force reprocessing (delete the cache)")
    print("  4 - Generate embeddings for apps with no cache (no training)")
    print("-" * 80)

    choice = input("\nChoice (1/2/3/4) [2]: ").strip() or "2"

    use_cache = True

    if choice == "1":
        use_cache = False
    elif choice == "3":
        import shutil
        if os.path.exists(EMBEDDINGS_CACHE):
            shutil.rmtree(EMBEDDINGS_CACHE)
            print(f"Cache removed: {EMBEDDINGS_CACHE}")
        use_cache = False
    elif choice == "4":
        cache_texts_per_app = config.get("cache_texts_per_app", 50)
        analyzer = MultimodalAppAnalyzer(img_size=IMG_SIZE, results_dir=RESULTS_DIR)
        analyzer.generate_missing_embeddings_cache(texts_per_app=cache_texts_per_app)
        return

    min_screens_per_app = config.get("min_screens_per_app", 3)
    max_screens_per_app = config.get("max_screens_per_app", 8)
    texts_per_image = config.get("texts_per_image", 10)
    max_samples_per_class = config.get("max_samples_per_class", 10000)
    balance_splits = config.get("balance_splits", True)
    use_cross_validation = bool(config.get("use_cross_validation", False))
    cv_k = int(config.get("cv_k", 5))

    print("\n" + "=" * 80 + "\n")

    analyzer = MultimodalAppAnalyzer(img_size=IMG_SIZE, results_dir=RESULTS_DIR)
    analyzer.run_complete_analysis(
        min_screens_per_app=min_screens_per_app,
        max_screens_per_app=max_screens_per_app,
        texts_per_image=texts_per_image,
        max_samples_per_class=max_samples_per_class,
        use_cached_index=use_cache,
        balance_splits=balance_splits,
        use_cross_validation=use_cross_validation,
        cv_k=cv_k,
    )
