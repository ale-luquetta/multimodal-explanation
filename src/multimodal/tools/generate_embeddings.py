#!/usr/bin/env python3
"""DeBERTa ABSA embeddings generator with configurable selection criteria.

Standalone tool triggered from the CLI ("Generate text embeddings cache").
Writes a self-contained cache under ``data/embeddings_cache/<cache_name>/``,
with one ``.npy`` per app plus a ``cache_config.yaml`` describing how it was
built.

Training consumes the cache through ``embeddings_cache_subdir`` (a string in
the YAML) or through an interactive picker when that field is absent.
"""

from __future__ import annotations

import gc
import glob
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from multimodal.common.paths import (
    EMBEDDINGS_CACHE_DIR,
    REVIEWS_PROCESSED_DIR,
    RICO_DIR,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEXT_LENGTH = 128
ASPECT = "interface"
VALID_CACHE_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


def _load_labels(rico_path: str, column: str, app_percentage: float) -> dict[str, int]:
    """Computes app labels via top/bottom ``app_percentage`` of ``column``.

    Same logic as ``load_data`` in training: sort by rating and take the
    top/bottom N*app_percentage. Returns {package_name: 0|1}; apps outside the
    top/bottom are absent from the dict.
    """
    df = pd.read_csv(rico_path)
    df = df.dropna(subset=[column, "App Package Name"])
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column]).sort_values(column, ascending=True)

    n = len(df)
    n_x = max(1, int(n * app_percentage))
    worst = df.iloc[:n_x]
    best = df.iloc[-n_x:]

    labels: dict[str, int] = {}
    for pkg in worst["App Package Name"]:
        labels[str(pkg)] = 0
    for pkg in best["App Package Name"]:
        labels[str(pkg)] = 1
    return labels


def _select_reviews(df: pd.DataFrame, criterion: str, label: int | None,
                    n: int, min_words: int) -> list[str]:
    """Filter reviews by min_words and pick the top-N by criterion."""
    df = df[df["sentence"].astype(str).apply(lambda s: len(s.split()) >= min_words)].copy()
    if df.empty:
        return []

    if criterion == "interface_polarized":
        if label not in (0, 1):
            return []
        # bom (label=1): most positive first; ruim (label=0): most negative first.
        ascending = label == 0
        df = df.sort_values("interface_pos", ascending=ascending, kind="mergesort")
    # criterion "default": keeps the original CSV order.

    texts = df.head(n)["sentence"].astype(str).tolist()
    return [t[:2000] for t in texts]


def _extract_embedding(tokenizer, model, text: str) -> np.ndarray:
    encoded = tokenizer(
        text,
        text_pair=ASPECT,
        max_length=MAX_TEXT_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()


def generate(config: dict) -> None:
    """Entry point invoked by CLI. Reads config, generates the cache."""
    cache_name = config.get("cache_name")
    if not isinstance(cache_name, str) or not VALID_CACHE_NAME.match(cache_name):
        raise ValueError(
            f"Invalid cache_name: {cache_name!r}. Use only [a-zA-Z0-9_-]."
        )

    texts_per_app = int(config.get("texts_per_app", 20))
    min_words = int(config.get("min_words", 3))
    criterion = config.get("selection_criterion", "default")
    if criterion not in ("default", "interface_polarized"):
        raise ValueError(
            f"Invalid selection_criterion: {criterion!r}. Use 'default' or 'interface_polarized'."
        )
    app_percentage = float(config.get("app_percentage", 0.1))
    column = config.get("column", "Average Rating Updated")
    rico_path = config.get("rico_data_path") or str(RICO_DIR / "rico_and_sentiment.csv")

    cache_dir = EMBEDDINGS_CACHE_DIR / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "cache_name": cache_name,
        "texts_per_app": texts_per_app,
        "min_words": min_words,
        "selection_criterion": criterion,
        "app_percentage": app_percentage if criterion == "interface_polarized" else None,
        "column": column if criterion == "interface_polarized" else None,
        "rico_data_path": rico_path if criterion == "interface_polarized" else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": DEVICE,
    }
    with open(cache_dir / "cache_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)

    print("=" * 70)
    print(f"Embeddings cache: {cache_dir}")
    print(f"Criterion:        {criterion}")
    print(f"texts_per_app:    {texts_per_app}")
    print(f"min_words:        {min_words}")
    if criterion == "interface_polarized":
        print(f"app_percentage:   {app_percentage}")
    print("=" * 70)

    labels: dict[str, int] | None = None
    if criterion == "interface_polarized":
        print(f"Computing labels from {rico_path} (top/bottom {app_percentage*100:.1f}%)...")
        labels = _load_labels(rico_path, column, app_percentage)
        n_bom = sum(1 for v in labels.values() if v == 1)
        n_ruim = sum(1 for v in labels.values() if v == 0)
        print(f"  Apps labelled: bom={n_bom}, ruim={n_ruim} (total={len(labels)})")

    print("\nLoading DeBERTa ABSA (yangheng/deberta-v3-base-absa-v1.1)...")
    tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    model = AutoModel.from_pretrained("yangheng/deberta-v3-base-absa-v1.1").to(DEVICE)
    model.eval()
    print(f"DeBERTa ready on {DEVICE}.")

    csv_pattern = os.path.join(str(REVIEWS_PROCESSED_DIR), "app_reviews_*_with_aspects.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    print(f"\nFound {len(csv_files)} review CSVs in {REVIEWS_PROCESSED_DIR}.")

    # For the polarized criterion, pre-filter the CSVs by the app NNNN ids
    # of the top/bottom apps (row index in app_details.csv = filename suffix),
    # so we do not open thousands of CSVs only to drop them by label.
    if criterion == "interface_polarized" and labels:
        app_details = pd.read_csv(RICO_DIR / "app_details.csv", header=0)
        pkg_to_idx = {
            str(pkg): i for i, pkg in enumerate(app_details["App Package Name"])
        }
        relevant_stems = set()
        for pkg in labels.keys():
            idx = pkg_to_idx.get(pkg)
            if idx is not None:
                relevant_stems.add(f"app_reviews_{idx:04d}_with_aspects")
        filtered = [f for f in csv_files if Path(f).stem in relevant_stems]
        print(
            f"Label pre-filter (top/bottom {app_percentage*100:.1f}%): "
            f"{len(filtered)} / {len(csv_files)} CSVs will be processed.\n"
        )
        csv_files = filtered
    else:
        print()

    generated = 0
    skipped_cached = 0
    skipped_label = 0
    skipped_empty = 0
    failed = 0

    for csv_file in tqdm(csv_files, desc="Generating"):
        try:
            df_head = pd.read_csv(csv_file, nrows=1)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            failed += 1
            continue

        if df_head.empty or "package_name" not in df_head.columns:
            failed += 1
            continue

        package_name = str(df_head["package_name"].iloc[0])
        safe_pkg = package_name.replace(".", "_").replace("/", "_")
        out_file = cache_dir / f"{safe_pkg}.npy"

        if out_file.exists():
            skipped_cached += 1
            continue

        label: int | None = None
        if criterion == "interface_polarized":
            label = labels.get(package_name) if labels else None
            if label is None:
                skipped_label += 1
                continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading the full CSV {csv_file}: {e}")
            failed += 1
            continue

        if df.empty:
            skipped_empty += 1
            continue

        for col in ("interface_pos", "interface_neg"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0

        df["sentence"] = df.get("sentence", pd.Series([""] * len(df))).astype(str)
        df = df[df["sentence"].str.strip() != ""]

        texts = _select_reviews(df, criterion, label, texts_per_app, min_words)
        if not texts:
            skipped_empty += 1
            continue

        try:
            embs = [_extract_embedding(tokenizer, model, t) for t in texts]
            np.save(out_file, np.array(embs, dtype=np.float16))
            generated += 1
        except Exception as e:
            print(f"\nError generating embeddings for {package_name}: {e}")
            failed += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Generation finished.")
    print(f"  Generated:                     {generated}")
    print(f"  Skipped (already cached):      {skipped_cached}")
    print(f"  Skipped (outside top/bottom):  {skipped_label}")
    print(f"  Skipped (no valid reviews):    {skipped_empty}")
    print(f"  Failures:                      {failed}")
    print("=" * 70)
    print(f"Cache available at: {cache_dir}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
