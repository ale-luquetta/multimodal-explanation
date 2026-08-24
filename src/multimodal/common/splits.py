"""App-level stratified train/val/test split shared by unimodal and multimodal training.

Splitting is done at the app level (each package goes entirely to one split),
stratified by (label, screen-count quartile) with a label-only fallback.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_apps_stratified(
    df: pd.DataFrame,
    package_col: str = "package_name",
    label_col: str = "label",
    test_size: float = 0.3,
    val_size: float = 0.5,
    random_seed: int = 42,
) -> tuple[list, list, list]:
    """Split apps into train/val/test stratified by label + screen-count quartile.

    Args:
        df: One row per screen/sample. Must contain `package_col` and `label_col`.
        package_col: Column with the app/package identifier.
        label_col: Column with the binary/categorical label.
        test_size: Fraction of apps split off as (val + test) from the full set.
        val_size: Fraction of the (val + test) set assigned to test (so val gets 1 - val_size).
        random_seed: Seed for reproducibility.

    Returns:
        (apps_train, apps_val, apps_test) — lists of package identifiers.
    """
    screen_counts = df.groupby(package_col).size().to_dict()
    df_apps = df.groupby(package_col).agg({label_col: "first"}).reset_index()
    df_apps["num_screens"] = df_apps[package_col].map(screen_counts)

    print("\n" + "=" * 80)
    print("📊 SPLIT: stratify by label + screen quartile")
    print("=" * 80)
    print(f"Apps: {len(df_apps)} | Screens: {len(df)}")
    print(
        f"Screens per app — mean: {df_apps['num_screens'].mean():.2f}, "
        f"median: {df_apps['num_screens'].median():.0f}, "
        f"min: {df_apps['num_screens'].min()}, "
        f"max: {df_apps['num_screens'].max()}"
    )

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

    try:
        apps_train_df, apps_temp_df = train_test_split(
            df_apps, test_size=test_size, stratify=df_apps["strata"], random_state=random_seed
        )
    except ValueError:
        print("⚠️  Stratification by (label + quartile) failed; falling back to label only.")
        apps_train_df, apps_temp_df = train_test_split(
            df_apps, test_size=test_size, stratify=df_apps[label_col], random_state=random_seed
        )

    try:
        apps_val_df, apps_test_df = train_test_split(
            apps_temp_df, test_size=val_size, stratify=apps_temp_df["strata"], random_state=random_seed
        )
    except ValueError:
        apps_val_df, apps_test_df = train_test_split(
            apps_temp_df, test_size=val_size, stratify=apps_temp_df[label_col], random_state=random_seed
        )

    _print_split_stats(apps_train_df, apps_val_df, apps_test_df, label_col)

    return (
        apps_train_df[package_col].tolist(),
        apps_val_df[package_col].tolist(),
        apps_test_df[package_col].tolist(),
    )


def _print_split_stats(apps_train_df, apps_val_df, apps_test_df, label_col: str) -> None:
    print("\n📊 Split result (apps and screens):")
    for split_name, split_df in [
        ("TRAIN", apps_train_df),
        ("VAL", apps_val_df),
        ("TEST", apps_test_df),
    ]:
        num_apps = len(split_df)
        total_screens = split_df["num_screens"].sum()
        avg_screens = split_df["num_screens"].mean()
        apps_0 = int((split_df[label_col] == 0).sum())
        apps_1 = int((split_df[label_col] == 1).sum())
        screens_0 = int(split_df.loc[split_df[label_col] == 0, "num_screens"].sum())
        screens_1 = int(split_df.loc[split_df[label_col] == 1, "num_screens"].sum())
        print(f"\n{split_name}:")
        print(f"  Apps:    {num_apps:4d} (label=0: {apps_0}, label=1: {apps_1})")
        print(f"  Screens: {total_screens:4d} (label=0: {screens_0}, label=1: {screens_1})")
        print(f"  Mean screens/app: {avg_screens:.2f}")
    print()
