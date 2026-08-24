#!/usr/bin/env python3
"""
App rating analysis with a CNN - full version with balancing
================================================================================
Loads a pre-trained model when one is available.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
import datetime

from multimodal.common.balancing import balance_split
from multimodal.common.cv import kfold_app_stratified
from multimodal.common.paths import RESULTS_TRAINING_UNIMODAL, RICO_DIR
from multimodal.common.splits import split_apps_stratified
from multimodal.common.thresholds import youden_optimal_threshold

warnings.filterwarnings('ignore')

# Settings
RANDOM_SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
L2_REG = 1e-4
COLUMN = 'Average Rating Updated'
APP_PERCENTAGE = 0.1

# Optional fine-tuning of the backbone (two-phase training).
# When False (default), only phase 1 (frozen backbone) runs.
FINE_TUNE_BACKBONE = False
FINE_TUNE_LR = 2e-6
FINE_TUNE_EPOCHS = 15
UNFREEZE_LAST_N_LAYERS = 10


class AppRatingAnalyzer:
    """Main class for the app rating analysis."""

    def __init__(self, data_path=None,
                 images_path=None, img_size=IMG_SIZE):
        self.data_path = data_path if data_path is not None else str(RICO_DIR / "rico_and_sentiment.csv")
        self.images_path = images_path if images_path is not None else str(RICO_DIR / "screenshots")
        self.img_size = img_size
        self.data = None
        self.model = None
        self.history = None
        self.label_encoder = LabelEncoder()
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)

    def load_data(self, min_screens_per_app=None):
        """Load and prepare the app data."""
        print("Loading app data...")
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.dropna(subset=[COLUMN, 'App Package Name'])
        self.data[COLUMN] = pd.to_numeric(self.data[COLUMN], errors='coerce')
        self.data = self.data.dropna(subset=[COLUMN])

        # Apply the minimum-screens filter before selecting the top/bottom 10%
        if min_screens_per_app is not None and 'UI Count' in self.data.columns:
            print(f"\n🔍 Pre-selection filter: at least {min_screens_per_app} screens per app")
            apps_before = len(self.data)

            # Convert UI Count to numeric and filter
            self.data['UI Count'] = pd.to_numeric(self.data['UI Count'], errors='coerce')
            self.data = self.data[self.data['UI Count'] >= min_screens_per_app].copy()

            apps_after = len(self.data)
            apps_removed = apps_before - apps_after

            print(f"   Apps before the filter: {apps_before}")
            print(f"   Apps after the filter: {apps_after}")
            print(f"   Apps removed: {apps_removed} ({apps_removed / apps_before * 100:.1f}%)")

        self.data = self.data.sort_values(COLUMN, ascending=True)
        n = len(self.data)
        n_x = int(n * APP_PERCENTAGE)
        worst = self.data.iloc[:n_x].copy()
        best = self.data.iloc[-n_x:].copy()
        worst['label'] = 0
        worst['label_name'] = 'ruim'
        best['label'] = 1
        best['label_name'] = 'bom'
        self.data = pd.concat([worst, best]).reset_index(drop=True)
        print(f"\nApps selected: {len(self.data)}")
        print(f"Label distribution:")
        print(self.data['label_name'].value_counts())

        # Show UI Count statistics when available
        if 'UI Count' in self.data.columns:
            print(f"\n📊 UI Count statistics on the selected apps:")
            print(f"   Mean: {self.data['UI Count'].mean():.2f}")
            print(f"   Median: {self.data['UI Count'].median():.0f}")
            print(f"   Min: {self.data['UI Count'].min():.0f}")
            print(f"   Max: {self.data['UI Count'].max():.0f}")

        return self.data

    def find_matching_images(self):
        """Find the screenshots matching the selected apps."""
        print("Looking for matching screenshots via ui_details.csv...")
        ui_details = pd.read_csv(str(RICO_DIR / "ui_details.csv"), usecols=[0, 1],
                                 names=['UI Number', 'App Package Name'], header=0)
        image_mapping = {}
        for _, row in ui_details.iterrows():
            img_filename = f"{row['UI Number']}.jpg"
            package_name = row['App Package Name']
            if package_name in self.data['App Package Name'].values:
                if package_name not in image_mapping:
                    image_mapping[package_name] = []
                img_path = os.path.join(self.images_path, img_filename)
                if os.path.exists(img_path):
                    image_mapping[package_name].append(img_path)
        self.data['has_images'] = self.data['App Package Name'].isin(image_mapping.keys())
        self.data_with_images = self.data[self.data['has_images']].copy()
        print(f"Apps with matching screenshots: {len(self.data_with_images)}")
        return image_mapping

    def preprocess_image(self, image_path):
        """Preprocess one image for the CNN input."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            return img
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def create_dataset(self, image_mapping, max_samples_per_class=4500,
                       max_screens_per_app=None):
        """Build the image and label dataset.

        Args:
            image_mapping: mapping from package name to image paths
            max_samples_per_class: cap on samples per class (total)
            max_screens_per_app: cap on screens per app (for balancing)

        Note: min_screens_per_app is applied in load_data() using UI Count
        """
        print("Building the image dataset...")

        # Derive max_screens_per_app automatically when not provided
        if max_screens_per_app is None:
            screens_per_app_by_class = {0: [], 1: []}
            for _, row in self.data_with_images.iterrows():
                package_name = row['App Package Name']
                label = row['label']
                if package_name in image_mapping:
                    num_screens = len(image_mapping[package_name])
                    screens_per_app_by_class[label].append(num_screens)

            if screens_per_app_by_class[0] and screens_per_app_by_class[1]:
                median_0 = int(np.median(screens_per_app_by_class[0]))
                median_1 = int(np.median(screens_per_app_by_class[1]))
                max_screens_per_app = min(median_0, median_1)
                print(f"\n🎯 Capping screens per app automatically:")
                print(f"   Median, bad apps: {median_0} screens")
                print(f"   Median, good apps: {median_1} screens")
                print(f"   Cap chosen: {max_screens_per_app} screens per app\n")

        X = []
        y = []
        package_names = []
        image_paths = []

        class_counts = {0: 0, 1: 0}
        app_screen_counts = {}

        for _, row in self.data_with_images.iterrows():
            package_name = row['App Package Name']
            label = row['label']

            if class_counts[label] >= max_samples_per_class:
                continue

            if package_name in image_mapping:
                if package_name not in app_screen_counts:
                    app_screen_counts[package_name] = 0

                for img_path in image_mapping[package_name]:
                    if class_counts[label] >= max_samples_per_class:
                        break

                    if max_screens_per_app and app_screen_counts[package_name] >= max_screens_per_app:
                        break

                    img = self.preprocess_image(img_path)
                    if img is not None:
                        X.append(img)
                        y.append(label)
                        package_names.append(package_name)
                        image_paths.append(img_path)
                        class_counts[label] += 1
                        app_screen_counts[package_name] += 1

        X = np.array(X)
        y = np.array(y)

        print(f"\nDataset built: {len(X)} images")
        print(f"Class distribution: {np.bincount(y)}")

        screens_used = list(app_screen_counts.values())
        print(f"\n📊 Screens used per app:")
        print(f"   Mean: {np.mean(screens_used):.2f}")
        print(f"   Median: {np.median(screens_used):.0f}")
        print(f"   Min: {min(screens_used)}")
        print(f"   Max: {max(screens_used)}")

        return X, y, package_names, image_paths

    def diagnose_data_split(self, package_names, y, splits_info):
        """Detailed balance diagnostics."""
        print("\n" + "=" * 80)
        print("🔍 DATA BALANCE DIAGNOSTICS")
        print("=" * 80)

        df_global = pd.DataFrame({'package': package_names, 'label': y})
        apps_stats = df_global.groupby('package').agg({'label': ['first', 'count']}).reset_index()
        apps_stats.columns = ['package', 'label', 'num_screens']

        print(f"\n📊 FULL DATASET OVERVIEW:")
        print(f"   Apps in total: {len(apps_stats)}")
        print(f"   Screens in total: {len(package_names)}")
        print(f"   Mean screens per app: {apps_stats['num_screens'].mean():.2f}")
        print(f"   Median screens per app: {apps_stats['num_screens'].median():.0f}")

        print(f"\n{'─' * 80}")
        print("📈 PER-SPLIT BREAKDOWN:")
        print(f"{'─' * 80}\n")

        results = []
        for split_name, pk_split in splits_info.items():
            y_split = [y[i] for i, pkg in enumerate(package_names) if pkg in set(pk_split)]
            unique_apps = len(set(pk_split))
            num_screens = len(pk_split)
            avg_screens = num_screens / unique_apps if unique_apps > 0 else 0

            screens_label_0 = sum([1 for label in y_split if label == 0])
            screens_label_1 = sum([1 for label in y_split if label == 1])

            results.append({
                'split': split_name.upper(),
                'apps': unique_apps,
                'telas': num_screens,
                'telas_ruim': screens_label_0,
                'telas_bom': screens_label_1,
                'telas_por_app': avg_screens
            })

            print(f"   {split_name.upper():5s}:")
            print(f"   ├─ Apps: {unique_apps:4d}")
            print(f"   ├─ Screens: {num_screens:4d} ({screens_label_0:4d} bad, {screens_label_1:4d} good)")
            print(f"   └─ Mean: {avg_screens:5.2f} screens/app\n")

        df_results = pd.DataFrame(results)
        print(f"{'─' * 80}")
        print(df_results.to_string(index=False))

        total_screens = df_results['telas'].sum()
        expected_train = total_screens * 0.7

        train_diff = abs(df_results[df_results['split'] == 'TRAIN']['telas'].values[0] - expected_train)
        max_diff_pct = (train_diff / expected_train) * 100

        print(f"\n{'─' * 80}")
        print("⚖️  CLASS BALANCE:")
        print(f"{'─' * 80}")

        for _, row in df_results.iterrows():
            ratio = row['telas_ruim'] / row['telas_bom'] if row['telas_bom'] > 0 else 0
            print(f"\n   {row['split']:5s}: ratio {ratio:.3f} ", end='')
            if abs(ratio - 1.0) < 0.1:
                print("✅ Perfectly balanced!")
            elif abs(ratio - 1.0) < 0.3:
                print("✅ Well balanced")
            else:
                print("⚠️  Imbalanced")

        print(f"\n{'=' * 80}\n")

        if hasattr(self, 'results_dir'):
            report_path = os.path.join(self.results_dir, 'balanceamento_diagnostico.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(df_results.to_string(index=False))
            print(f"📄 Diagnostics written to: {report_path}")

        return df_results

    def build_model(self):
        """Build the CNN, mirroring the multimodal vision branch and head without the text."""
        print("Building the CNN...")
        base_model = MobileNetV2(
            weights='imagenet', include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        base_model.trainable = False
        # Stored so the optional fine-tuning phase can run (see unfreeze_backbone).
        self.base_model = base_model
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(name='image_pooling'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', name='image_dense',
                         kernel_regularizer=regularizers.l2(L2_REG)),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu', name='head_dense1',
                         kernel_regularizer=regularizers.l2(L2_REG)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', name='head_dense2',
                         kernel_regularizer=regularizers.l2(L2_REG)),
            layers.Dense(1, activation='sigmoid', name='output'),
        ])
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )
        print("Model built.")
        return self.model

    def load_model(self, model_path):
        """Load a previously trained model."""
        try:
            print(f"\nLoading model from: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded.")
            return True
        except Exception as e:
            print(f"❌ Error loading the model: {e}")
            return False

    def unfreeze_backbone(self):
        """Optional phase 2: unfreeze the last ``UNFREEZE_LAST_N_LAYERS`` layers of the
        MobileNetV2 and recompiles with ``FINE_TUNE_LR``.
        """
        if not hasattr(self, "base_model") or self.base_model is None:
            print("⚠️  base_model unavailable; skipping the unfreeze.")
            return self.model

        print("\n" + "=" * 60)
        print("PHASE 2: BACKBONE FINE-TUNING")
        print("=" * 60)

        total_layers = len(self.base_model.layers)
        freeze_until = max(0, total_layers - UNFREEZE_LAST_N_LAYERS)

        self.base_model.trainable = True
        for layer in self.base_model.layers[:freeze_until]:
            layer.trainable = False

        n_trainable = sum(1 for l in self.base_model.layers if l.trainable)
        n_frozen = sum(1 for l in self.base_model.layers if not l.trainable)
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

    def train_model(self, X_train, y_train, X_val, y_val):
        """Training in one or two phases.

        Phase 1 (always): frozen backbone, ``EPOCHS`` epochs at ``LEARNING_RATE``.
        Phase 2 (when ``FINE_TUNE_BACKBONE=True``): unfreezes the last
        ``UNFREEZE_LAST_N_LAYERS`` layers and trains for ``FINE_TUNE_EPOCHS``
        epochs at ``FINE_TUNE_LR``.
        """
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
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[early_stopping, reduce_lr], verbose=1,
        )

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
                X_train, y_train, validation_data=(X_val, y_val),
                epochs=FINE_TUNE_EPOCHS, batch_size=BATCH_SIZE,
                callbacks=[early_stopping_p2, reduce_lr_p2], verbose=1,
            )
            for key, vals in history_p2.history.items():
                self.history.history.setdefault(key, []).extend(vals)

        print("Training finished.")
        return self.history

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model at screen level (threshold 0.5 plus Youden's J)."""
        print("Evaluating the model...")
        y_pred_proba = self.model.predict(X_test)
        y_proba_flat = np.asarray(y_pred_proba).flatten()
        y_pred = (y_proba_flat > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Ruim', 'Bom'], zero_division=0)
        print(f"\n=== RESULTS (SCREEN LEVEL, threshold=0.5) ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(report)
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, ['Ruim', 'Bom'], suffix='_screen_level')

        youden_block = self._build_youden_block(y_test, y_proba_flat, level_label='SCREEN LEVEL')

        self._save_metrics_to_file(
            filename='metrics_screen_level.txt',
            title='METRICS - SCREEN LEVEL',
            accuracy=accuracy,
            report=report,
            confusion_matrix=cm,
            n_samples=len(y_test),
            youden_block=youden_block,
        )
        return y_pred, y_pred_proba

    def _build_youden_block(self, y_true, y_proba, level_label=''):
        """Compute the Youden's J threshold and return the recomputed metrics.

        Returns None when the optimal threshold equals 0.5 (nothing to add).
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

    def evaluate_model_app_level(self, y_test, y_pred_proba, pk_test):
        """App-level evaluation by aggregating probabilities."""
        print("\n" + "=" * 80)
        print("🎯 APP-LEVEL EVALUATION")
        print("=" * 80 + "\n")

        app_data = {}
        for i, pkg in enumerate(pk_test):
            if pkg not in app_data:
                app_data[pkg] = {'true_label': int(y_test[i]), 'probabilities': []}
            app_data[pkg]['probabilities'].append(float(y_pred_proba[i]))

        print(f"📊 Apps in the test set: {len(app_data)}")

        rows = []
        for pkg, data in app_data.items():
            true_label = data['true_label']
            mean_proba = float(np.mean(data['probabilities']))
            pred_label = 1 if mean_proba > 0.5 else 0

            app_rows = self.data_with_images[self.data_with_images['App Package Name'] == pkg]
            if len(app_rows) > 0:
                info = app_rows.iloc[0]
                app_name = info.get('App', '-')
                rating = info.get(COLUMN, '-')
                label_real = info.get('label_name', '-')
                categoria = info.get('Category', '-')
            else:
                app_name = rating = label_real = categoria = '-'

            rows.append({
                'package_name': pkg,
                'app': app_name,
                'rating': rating,
                'categoria': categoria,
                'label_real': label_real,
                'valor_real': 'bom' if true_label == 1 else 'ruim',
                'predito': 'bom' if pred_label == 1 else 'ruim',
                'acerto': true_label == pred_label,
                'num_telas': len(data['probabilities']),
                'probabilidade_media': mean_proba,
                'confianca_agregada': abs(mean_proba - 0.5) * 2,
            })

        df_by_app = pd.DataFrame(rows)
        df_by_app.to_csv(os.path.join(self.results_dir, 'classification_by_app.csv'), index=False)
        print(f"Wrote '{self.results_dir}/classification_by_app.csv' ({len(df_by_app)} apps)")

        df_errors_app = df_by_app[~df_by_app['acerto']]
        if len(df_errors_app) > 0:
            df_errors_app.to_csv(os.path.join(self.results_dir, 'classification_errors_by_app.csv'), index=False)
            print(f"Wrote '{self.results_dir}/classification_errors_by_app.csv' ({len(df_errors_app)} apps with errors)")

        y_app_true = (df_by_app['valor_real'] == 'bom').astype(int).values
        y_app_pred = (df_by_app['predito'] == 'bom').astype(int).values

        acc = accuracy_score(y_app_true, y_app_pred)
        report = classification_report(y_app_true, y_app_pred, target_names=['Ruim', 'Bom'], zero_division=0)
        cm = confusion_matrix(y_app_true, y_app_pred)

        print(f"   ✅ Accuracy: {acc:.4f}")
        print(report)

        self.plot_confusion_matrix(cm, ['Ruim', 'Bom'], suffix='_app_level')

        youden_block = self._build_youden_block(
            y_app_true, df_by_app['probabilidade_media'].values, level_label='APP LEVEL'
        )

        self._save_metrics_to_file(
            filename='metrics_app_level.txt',
            title='METRICS - APP LEVEL (AGGREGATED)',
            accuracy=acc,
            report=report,
            confusion_matrix=cm,
            n_samples=len(df_by_app),
            extra_info=f"Total de apps avaliados: {len(df_by_app)}",
            youden_block=youden_block,
        )

        print(f"\n{'=' * 80}\n")
        return {'accuracy': acc, 'y_true': y_app_true.tolist(), 'y_pred': y_app_pred.tolist()}

    def plot_confusion_matrix(self, cm, classes, suffix=''):
        """Plot the confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion matrix')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predito')
        plt.tight_layout()
        filename = f'confusion_matrix{suffix}.png'
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

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
                f.write(f"Recall (classe Bom): {recall:.4f}\n")
            if (tn + fp) > 0:
                specificity = tn / (tn + fp)
                f.write(f"Especificidade (classe Ruim): {specificity:.4f}\n")

            if youden_block is not None:
                yb_cm = youden_block['confusion_matrix']
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"OPTIMISED THRESHOLD (Youden's J) — t = {youden_block['threshold']:.4f}\n")
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

    def analyze_errors(self, X_test, y_test, y_pred, pk_test, img_paths_test):
        """Analyse the misclassifications."""
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

            classificacao = {
                'valor_real': 'bom' if true_label == 1 else 'ruim',
                'predito': 'bom' if pred_label == 1 else 'ruim',
                'acerto': not is_error,
                'package_name': package_name,
                'app': app_name,
                'rating': rating,
                'label_real': label_name,
                'categoria': categoria,
                'numero_da_tela': numero_tela
            }
            todas_classificacoes.append(classificacao)

            if is_error:
                erros_para_csv.append({
                    'valor_real': 'bom' if true_label == 1 else 'ruim',
                    'predito': 'bom' if pred_label == 1 else 'ruim',
                    'package_name': package_name,
                    'app': app_name,
                    'rating': rating,
                    'label_real': label_name,
                    'categoria': categoria,
                    'numero_da_tela': numero_tela
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
            pd.DataFrame(erros_para_csv).to_csv(os.path.join(self.results_dir, 'classification_errors.csv'),
                                                index=False)
            print(
                f"Wrote '{self.results_dir}/classification_errors.csv' with the misclassifications ({len(erros_para_csv)} rows).")

    def save_full_dataset_predictions(self, X_full, y_full, pk_full, img_paths_full, split_labels):
        """Run inference over the full dataset (train+val+test) and write classification_full_dataset.csv.

        Serves explainability: lets it iterate over every app of the filtered dataset, not only the test split.
        """
        print("\n" + "=" * 80)
        print("📊 PREDICTIONS OVER THE FULL DATASET (train + val + test)")
        print("=" * 80)
        print(f"Total samples: {len(y_full)}")

        y_proba = self.model.predict(X_full).flatten()
        y_pred = (y_proba > 0.5).astype(int)

        rows = []
        for i in range(len(y_full)):
            true_label = int(y_full[i])
            pred_label = int(y_pred[i])
            proba = float(y_proba[i])
            package_name = pk_full[i]
            img_path = img_paths_full[i]
            split = split_labels[i]

            app_rows = self.data_with_images[self.data_with_images['App Package Name'] == package_name]
            if len(app_rows) > 0:
                info = app_rows.iloc[0]
                app_name = info.get('App', '-')
                rating = info.get(COLUMN, '-')
                label_real = info.get('label_name', '-')
                categoria = info.get('Category', '-')
            else:
                app_name = rating = label_real = categoria = '-'

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
                'label_real': label_real,
                'categoria': categoria,
                'numero_da_tela': numero_tela,
            })

        df_full = pd.DataFrame(rows)
        out_path = os.path.join(self.results_dir, 'classification_full_dataset.csv')
        df_full.to_csv(out_path, index=False)

        print(f"Wrote: {out_path} ({len(df_full)} rows)")
        for split_name in ['train', 'val', 'test']:
            n = int((df_full['split'] == split_name).sum())
            print(f"  {split_name}: {n}")

        # Per-screen aggregation, for parity with the multimodal pipeline. The
        # data is already screen-level, with no reviews to aggregate, so this
        # only adds confianca_agregada; num_reviews = 1.
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
        df_by_screen['confianca_agregada'] = (
            df_by_screen['probabilidade_media'] - 0.5
        ).abs() * 2
        df_by_screen['predito'] = df_by_screen['probabilidade_media'].apply(
            lambda p: 'bom' if p > 0.5 else 'ruim'
        )
        df_by_screen['acerto'] = df_by_screen['valor_real'] == df_by_screen['predito']
        screen_path = os.path.join(
            self.results_dir, 'classification_full_by_screen.csv'
        )
        df_by_screen.to_csv(screen_path, index=False)
        print(f"Wrote: {screen_path} ({len(df_by_screen)} screens)")

        # Per-app aggregation (mean of the per-screen probabilities).
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
        df_by_app['confianca_agregada'] = (
            df_by_app['probabilidade_media'] - 0.5
        ).abs() * 2
        df_by_app['predito'] = df_by_app['probabilidade_media'].apply(
            lambda p: 'bom' if p > 0.5 else 'ruim'
        )
        df_by_app['acerto'] = df_by_app['valor_real'] == df_by_app['predito']
        app_path = os.path.join(
            self.results_dir, 'classification_full_by_app.csv'
        )
        df_by_app.to_csv(app_path, index=False)
        print(f"Wrote: {app_path} ({len(df_by_app)} apps)")

    def save_model(self, filename='app_rating_model.h5'):
        """Save the model."""
        if self.model is not None:
            path = os.path.join(self.results_dir, filename)
            self.model.save(path)
            print(f"Model saved: {path}")

    def run_complete_analysis(self, balance_splits=False,
                              max_screens_per_app=None, min_screens_per_app=None,
                              load_model_path=None,
                              use_cross_validation=False, cv_k=5):
        """Run the full analysis in single hold-out or cross-validation mode.

        Args:
            balance_splits: when True, balances each split by downsampling.
            max_screens_per_app: cap on screens per app (None = automatic).
            min_screens_per_app: minimum-screens filter, applied before APP_PERCENTAGE.
            load_model_path: path to a pre-trained model (ignored in CV).
            use_cross_validation: when True, runs app-stratified k-fold (one subfolder per fold).
            cv_k: number of folds when use_cross_validation=True.
        """
        print("=== APP RATING ANALYSIS ===")
        print(f"Split balancing: {balance_splits}")
        if min_screens_per_app:
            print(f"Minimum screens filter (UI Count): {min_screens_per_app}")
        if use_cross_validation:
            print(f"🔁 Mode: cross-validation (k={cv_k})")
        elif load_model_path:
            print(f"🔄 Mode: load an existing model (hold-out)")
        else:
            print(f"🆕 Mode: train a new model (single hold-out)")
        print("Starting the full analysis...\n")

        X, y, package_names, image_paths = self._prepare_dataset(
            min_screens_per_app=min_screens_per_app,
            max_screens_per_app=max_screens_per_app,
        )
        if X is None:
            return

        base_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        if use_cross_validation:
            cv_root = RESULTS_TRAINING_UNIMODAL / f"cv_{base_time}"
            cv_root.mkdir(parents=True, exist_ok=True)

            apps_df = pd.DataFrame({"package_name": package_names, "label": y})
            folds = list(kfold_app_stratified(
                apps_df,
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

                self._run_fold(
                    X=X, y=y, package_names=package_names, image_paths=image_paths,
                    apps_train=apps_train, apps_val=apps_val, apps_test=apps_test,
                    balance_splits=balance_splits, load_model_path=None,
                )

            self._aggregate_cv_metrics(cv_root, cv_k)
            print(f"\n=== CROSS-VALIDATION FINISHED ===")
            print(f"Consolidated results in: {cv_root}/")
        else:
            self.start_time = base_time
            self.results_dir = str(RESULTS_TRAINING_UNIMODAL / f"resultados_{self.start_time}")
            os.makedirs(self.results_dir, exist_ok=True)

            apps_df = pd.DataFrame({"package_name": package_names, "label": y})
            apps_train, apps_val, apps_test = split_apps_stratified(
                apps_df,
                package_col="package_name",
                label_col="label",
                test_size=0.3,
                val_size=0.5,
                random_seed=RANDOM_SEED,
            )

            self._run_fold(
                X=X, y=y, package_names=package_names, image_paths=image_paths,
                apps_train=apps_train, apps_val=apps_val, apps_test=apps_test,
                balance_splits=balance_splits, load_model_path=load_model_path,
            )

            print("\n=== ANALYSIS FINISHED ===")
            print(f"Files in: {self.results_dir}/")

    def _prepare_dataset(self, min_screens_per_app, max_screens_per_app):
        """Load the data as (X, y, package_names, image_paths); (None,)*4 when empty."""
        self.load_data(min_screens_per_app=min_screens_per_app)
        image_mapping = self.find_matching_images()
        if len(self.data_with_images) == 0:
            print("No matching screenshot found.")
            return None, None, None, None
        X, y, package_names, image_paths = self.create_dataset(
            image_mapping,
            max_screens_per_app=max_screens_per_app,
        )
        if len(X) == 0:
            print("No valid screenshot found.")
            return None, None, None, None
        return X, y, package_names, image_paths

    def _run_fold(self, X, y, package_names, image_paths,
                  apps_train, apps_val, apps_test,
                  balance_splits, load_model_path):
        """Run one full iteration: filter, balance, train, evaluate, save."""

        def filter_by_apps(X, y, pk, img_paths, selected_apps):
            mask = [p in set(selected_apps) for p in pk]
            return (
                X[mask],
                y[mask],
                [p for i, p in enumerate(pk) if mask[i]],
                [ip for i, ip in enumerate(img_paths) if mask[i]],
            )

        X_train, y_train, pk_train, img_paths_train = filter_by_apps(X, y, package_names, image_paths, apps_train)
        X_val, y_val, pk_val, img_paths_val = filter_by_apps(X, y, package_names, image_paths, apps_val)
        X_test, y_test, pk_test, img_paths_test = filter_by_apps(X, y, package_names, image_paths, apps_test)

        if balance_splits:
            print("\n" + "=" * 80)
            print("🎯 BALANCING BY DOWNSAMPLING IN EACH SPLIT")
            print("=" * 80)

            def _balance_arrays(X, y, pk, img_paths):
                df = pd.DataFrame({"orig_idx": range(len(y)), "label": y})
                balanced_df = balance_split(df, label_col="label", random_seed=RANDOM_SEED)
                kept = balanced_df["orig_idx"].tolist()
                return X[kept], y[kept], [pk[i] for i in kept], [img_paths[i] for i in kept]

            X_train, y_train, pk_train, img_paths_train = _balance_arrays(X_train, y_train, pk_train, img_paths_train)
            X_val, y_val, pk_val, img_paths_val = _balance_arrays(X_val, y_val, pk_val, img_paths_val)
            X_test, y_test, pk_test, img_paths_test = _balance_arrays(X_test, y_test, pk_test, img_paths_test)

        splits_info = {'train': pk_train, 'val': pk_val, 'test': pk_test}
        self.diagnose_data_split(package_names, y, splits_info)

        if load_model_path:
            print("\n" + "=" * 80)
            print("🔄 LOADING THE PRE-TRAINED MODEL")
            print("=" * 80 + "\n")
            if not self.load_model(load_model_path):
                print("⚠️  Could not load the model. Training a new one...")
                self.build_model()
                self.train_model(X_train, y_train, X_val, y_val)
            else:
                print("✅ Model loaded. Skipping training.\n")
        else:
            print("\n" + "=" * 80)
            print("🆕 BUILDING AND TRAINING A NEW MODEL")
            print("=" * 80 + "\n")
            self.build_model()
            self.train_model(X_train, y_train, X_val, y_val)

        print("\n" + "=" * 80)
        print("📊 STARTING THE EVALUATIONS")
        print("=" * 80)

        y_pred, y_pred_proba = self.evaluate_model(X_test, y_test)
        self.evaluate_model_app_level(y_test, y_pred_proba, pk_test)

        if self.history:
            self.plot_training_history()

        self.analyze_errors(X_test, y_test, y_pred, pk_test, img_paths_test)

        X_full = np.concatenate([X_train, X_val, X_test])
        y_full = np.concatenate([y_train, y_val, y_test])
        pk_full = list(pk_train) + list(pk_val) + list(pk_test)
        img_paths_full = list(img_paths_train) + list(img_paths_val) + list(img_paths_test)
        split_labels = (['train'] * len(y_train)) + (['val'] * len(y_val)) + (['test'] * len(y_test))
        self.save_full_dataset_predictions(X_full, y_full, pk_full, img_paths_full, split_labels)

        if not load_model_path:
            self.save_model()

    def _aggregate_cv_metrics(self, cv_root, n_folds):
        """Read classification_all.csv and classification_by_app.csv of each fold, summarise as mean ± sd,
        and copy the fold with the best ``accuracy_app`` to ``cv_root/best_fold/``."""
        import shutil as _shutil
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        rows = []
        for fold_idx in range(1, n_folds + 1):
            fold_dir = cv_root / f"fold_{fold_idx}"
            try:
                df_screen = pd.read_csv(fold_dir / "classification_all.csv")
                y_true_s = (df_screen['valor_real'] == 'bom').astype(int).values
                y_pred_s = (df_screen['predito'] == 'bom').astype(int).values
                df_app = pd.read_csv(fold_dir / "classification_by_app.csv")
                y_true_a = (df_app['valor_real'] == 'bom').astype(int).values
                y_pred_a = (df_app['predito'] == 'bom').astype(int).values
            except FileNotFoundError as e:
                print(f"⚠️  Fold {fold_idx}: missing CSV ({e}). Skipping.")
                continue

            rows.append({
                'fold': fold_idx,
                'accuracy_screen': accuracy_score(y_true_s, y_pred_s),
                'precision_screen': precision_score(y_true_s, y_pred_s, zero_division=0),
                'recall_screen': recall_score(y_true_s, y_pred_s, zero_division=0),
                'f1_screen': f1_score(y_true_s, y_pred_s, zero_division=0),
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


def find_available_models():
    """Find available .h5 models under the results folders."""
    models = []
    root = RESULTS_TRAINING_UNIMODAL
    if not root.exists():
        return models
    for folder in os.listdir(root):
        if folder.startswith('resultados_'):
            model_path = os.path.join(root, folder, 'app_rating_model.h5')
            if os.path.exists(model_path):
                models.append({
                    'path': model_path,
                    'folder': folder,
                    'timestamp': folder.replace('resultados_', '')
                })
    return sorted(models, key=lambda x: x['timestamp'], reverse=True)


def train(config: dict) -> None:
    """Unimodal (vision-only) training entry point driven by a YAML config dict."""
    global RANDOM_SEED, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, L2_REG
    global COLUMN, APP_PERCENTAGE
    global FINE_TUNE_BACKBONE, FINE_TUNE_LR, FINE_TUNE_EPOCHS, UNFREEZE_LAST_N_LAYERS

    RANDOM_SEED = config.get("random_seed", RANDOM_SEED)
    IMG_SIZE = config.get("img_size", IMG_SIZE)
    BATCH_SIZE = config.get("batch_size", BATCH_SIZE)
    EPOCHS = config.get("epochs", EPOCHS)
    LEARNING_RATE = config.get("learning_rate", LEARNING_RATE)
    L2_REG = config.get("l2_reg", L2_REG)
    COLUMN = config.get("column", COLUMN)
    APP_PERCENTAGE = config.get("app_percentage", APP_PERCENTAGE)
    FINE_TUNE_BACKBONE = bool(config.get("fine_tune_backbone", FINE_TUNE_BACKBONE))
    FINE_TUNE_LR = float(config.get("fine_tune_lr", FINE_TUNE_LR))
    FINE_TUNE_EPOCHS = int(config.get("fine_tune_epochs", FINE_TUNE_EPOCHS))
    UNFREEZE_LAST_N_LAYERS = int(config.get("unfreeze_last_n_layers", UNFREEZE_LAST_N_LAYERS))

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    analyzer = AppRatingAnalyzer(img_size=IMG_SIZE)

    available_models = find_available_models()

    print("\n" + "=" * 80)
    print("🤖 MODEL SETUP")
    print("=" * 80)

    model_to_load = None

    if available_models:
        print(f"\n✅ {len(available_models)} trained model(s) found:\n")
        for i, model in enumerate(available_models, 1):
            print(f"   {i}. {model['folder']}")

        print("\n" + "-" * 80)
        print("\nOptions:")
        print("  0 - Train a NEW model")
        for i, model in enumerate(available_models, 1):
            print(f"  {i} - Load model {model['timestamp']}")
        print("-" * 80)

        try:
            choice = input("\nChoose an option (0 to train a new one): ").strip()
            choice = int(choice) if choice else 0

            if choice > 0 and choice <= len(available_models):
                model_to_load = available_models[choice - 1]['path']
                print(f"\n✅ Selected: {model_to_load}")
            else:
                print("\n🆕 Training a new model...")
        except (ValueError, KeyboardInterrupt):
            print("\n🆕 Training a new model...")
    else:
        print("\n📝 No trained model found. A new one will be trained.\n")

    print("=" * 80 + "\n")

    balance_splits = config.get("balance_splits", False)
    max_screens_per_app = config.get("max_screens_per_app", 8)
    min_screens_per_app = config.get("min_screens_per_app", 3)
    use_cross_validation = bool(config.get("use_cross_validation", False))
    cv_k = int(config.get("cv_k", 5))

    analyzer.run_complete_analysis(
        balance_splits=balance_splits,
        max_screens_per_app=max_screens_per_app,
        min_screens_per_app=min_screens_per_app,
        load_model_path=model_to_load,
        use_cross_validation=use_cross_validation,
        cv_k=cv_k,
    )