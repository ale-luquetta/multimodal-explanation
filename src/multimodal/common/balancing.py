"""Class balancing via downsampling of the majority class."""

from __future__ import annotations

import pandas as pd


def balance_split(
    split_df: pd.DataFrame,
    label_col: str = "label",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Downsample the majority class so all classes match the minority count."""
    class_counts = split_df[label_col].value_counts()
    min_count = class_counts.min()

    if class_counts.max() == min_count:
        return split_df

    parts = []
    for label in split_df[label_col].unique():
        class_df = split_df[split_df[label_col] == label]
        if len(class_df) > min_count:
            class_df = class_df.sample(n=min_count, random_state=random_seed)
        parts.append(class_df)

    return pd.concat(parts).reset_index(drop=True)
