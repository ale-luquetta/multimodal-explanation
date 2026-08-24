"""App-level stratified k-fold cross-validation.

Yields k partitions in which every app appears in exactly one test fold.
Stratification follows the same logic as ``split_apps_stratified``: label ×
screen-count quartile, with a label-only fallback.

Within the remaining (non-test) apps of each fold, a further hold-out is applied
to produce train/val subsets, preserving stratification.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def kfold_app_stratified(
    df: pd.DataFrame,
    k: int = 5,
    package_col: str = "package_name",
    label_col: str = "label",
    val_size: float = 0.1765,
    random_seed: int = 42,
) -> Iterator[tuple[list, list, list]]:
    """Generate k folds of (apps_train, apps_val, apps_test) stratified at app level.

    Args:
        df: One row per screen/sample. Must contain ``package_col`` and ``label_col``.
        k: Number of folds.
        package_col: Column with the app identifier.
        label_col: Column with the binary/categorical label.
        val_size: Fraction of non-test apps used as validation. Default ≈15% of
            the total dataset (0.1765 of 85% ≈ 15%), matching the hold-out ratios
            used elsewhere in the project.
        random_seed: Seed for reproducibility.

    Yields:
        (apps_train, apps_val, apps_test) — lists of package identifiers, one
        triple per fold.
    """
    screen_counts = df.groupby(package_col).size().to_dict()
    df_apps = df.groupby(package_col).agg({label_col: "first"}).reset_index()
    df_apps["num_screens"] = df_apps[package_col].map(screen_counts)

    quartile_labels = ["Q1", "Q2", "Q3", "Q4"]
    try:
        df_apps["screen_quartile"] = pd.qcut(
            df_apps["num_screens"], q=4, labels=quartile_labels, duplicates="drop"
        )
    except ValueError:
        df_apps["screen_quartile"] = pd.cut(
            df_apps["num_screens"], bins=4, labels=quartile_labels, duplicates="drop"
        )
    df_apps["strata"] = df_apps[label_col].astype(str) + "_" + df_apps["screen_quartile"].astype(str)

    # Use label+quartile if possible; otherwise fall back to label.
    try:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
        fold_iter = list(skf.split(df_apps, df_apps["strata"]))
    except ValueError:
        print("⚠️  Stratification by (label + quartile) failed in the CV; "
              "falling back to label only.")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
        fold_iter = list(skf.split(df_apps, df_apps[label_col]))

    for fold_idx, (trainval_idx, test_idx) in enumerate(fold_iter):
        df_trainval = df_apps.iloc[trainval_idx]
        df_test = df_apps.iloc[test_idx]

        try:
            df_train, df_val = train_test_split(
                df_trainval,
                test_size=val_size,
                stratify=df_trainval["strata"],
                random_state=random_seed + fold_idx,
            )
        except ValueError:
            df_train, df_val = train_test_split(
                df_trainval,
                test_size=val_size,
                stratify=df_trainval[label_col],
                random_state=random_seed + fold_idx,
            )

        yield (
            df_train[package_col].tolist(),
            df_val[package_col].tolist(),
            df_test[package_col].tolist(),
        )
