#!/usr/bin/env python3
"""Multimodal explainability — faithfulness (ΔP_img), object-level deletion, full OBI.

Not a CLI entry point. The menu runs ``multimodal_v3.py``, which builds on this
module: it swaps ``MultimodalExplanationV2`` and the output directory by
monkey-patching and delegates to the ``explain()`` here, which in turn delegates
to ``multimodal.py``. The per-review analysis, ΔP_img, OBI, SHAP and the per-app
figure all live in this file.

Builds on ``multimodal.explainability.multimodal`` without touching it, adding:

1. **Faithfulness (ΔP_img)** per screen: the top-K most salient UI components
   are zeroed out together and the change in the model confidence is measured.
   Object-level deletion in the ObEy sense, and more interpretable than a
   continuous Grad-CAM map: "erasing the DatePicker moves confidence by X%".

2. **Full OBI report** (`components_ranked.csv`): every leaf component of the
   screen, ranked by mean saliency per pixel, not only the top-3 used in the
   narrative.

3. **Per-review analysis**: the whole XAI pipeline is run once per polarized
   review of the app, so image and text evidence can be read pair by pair.

Architecture: every explainer of the base module is reused through import and
subclassing. The entry point ``explain(config)`` delegates to the base
``explain`` and injects the classes defined here (monkey-patch local to the
function), which preserves the interactive modes and the whole model/data
selection logic.

Output goes to
``results/explanations/multimodal_v2/explicabilidade_v6_<timestamp>/``.
"""

from __future__ import annotations

import csv
import json
import os
import string
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec

from multimodal.common.faithfulness import (
    compute_delta_p,
    top_k_token_indices,
)
from multimodal.common.paths import (
    EMBEDDINGS_CACHE_DIR,
    RESULTS_EXPLANATIONS_MULTIMODAL_V2,
    REVIEWS_PROCESSED_DIR,
    RICO_DIR,
    RICO_HIERARCHIES_DIR,
)
from multimodal.common.semantic_saliency import rank_elements_by_saliency
from multimodal.explainability.multimodal import (
    ASPECT,
    ExplanationOutputManager,
    IMG_SIZE,
    MAX_TEXT_LENGTH,
    MultimodalExplanation,
    SHAPExplainer,
    STOPWORDS,
    _is_filtered_word,
    generate_app_text,
    generate_screen_text,
    load_original_image,
    merge_subtokens_and_attention,
    preprocess_image_path,
)

# Module config (overwritten in explain(config))
ABLATION_ENABLED: bool = True
ABLATION_TOP_K_TOKENS: int = 3
# UI components are ranked by score_mean (saliency per pixel) and the top-K are
# zeroed out together for ΔP_img, following the "top-K object deletion" semantics
# of the ObEy paper, which stabilizes the measure when the signal is spread over
# sibling components.
OBI_TOP_K_COMPONENTS: int = 3
# Per-review analysis: runs the full XAI pipeline for each review selected among
# the 20 cached ones, with quality (min. words) and polarization (interface_pos)
# filters applied per label.
PER_REVIEW_ANALYSIS: bool = False
PER_REVIEW_TOP_K: int = 5
PER_REVIEW_MIN_WORDS: int = 5
# Source of the reviews used in the per-review analysis:
#   - "cache":     uses only the ~20 cached ones. The embedding comes from the .npy.
#   - "raw":       ignores the cache and looks for polarized reviews in the raw CSV
#                  ``app_reviews_<NNNN>_with_aspects.csv``. The embedding is computed
#                  on the fly via ``self.text_attn.model.last_hidden_state``.
#                  Allows picking reviews longer than the 20 in the cache (useful
#                  when the cached ones are very short).
#   - "raw+cache": merge of both sources (dedup by sentence); keeps the cached ones
#                  and enriches them with non-cached raw reviews.
PER_REVIEW_SOURCE: str = "cache"
# Minimum number of words required by the polarized criterion when the review comes
# from the raw CSV. Usually larger than PER_REVIEW_MIN_WORDS (which applies to the
# cache) so that the on-the-fly choice is richer.
PER_REVIEW_MIN_WORDS_RAW: int = 15
# If True, writes one PNG per (screen × review) in screen_explanations/per_review/
# and suppresses the combined per-screen figure. Implies PER_REVIEW_ANALYSIS=True.
PER_REVIEW_GENERATE_FIGURES: bool = False
# If True, computes SHAP (visual + textual) per review. Implies figures=true and
# adds a significant cost (~10-20 min per app).
PER_REVIEW_INCLUDE_SHAP: bool = False
PER_REVIEW_SHAP_MAX_EVALS: int = 500

# Criterion used to pick the (screen, review) pair that represents the app in the
# aggregated figure of the per-app modes:
#   - "delta_p_img":  largest ΔP under occlusion of the top Grad-CAM regions, that
#                     is, the screen whose visual explanation is the most faithful.
#   - "confidence":   highest screen confidence, the same criterion used by the
#                     screen-ranking modes (``screen_confidence``), which makes both
#                     figures show the same screen for a given app. Ties between
#                     reviews of the same screen are broken by ΔP_img.
APP_REP_SCREEN_CRITERION: str = "delta_p_img"

# Cache directory active for this run. Overwritten in explain() when the YAML sets
# embeddings_cache_subdir. Referenced by _load_cached_review_texts (criterion
# replay) and by _per_review_analysis (.npy load).
ACTIVE_CACHE_DIR: Path = EMBEDDINGS_CACHE_DIR

# Module-level cache for app_details.csv (loaded once per run).
_APP_DETAILS_CACHE: pd.DataFrame | None = None
# Cache: package_name -> list of dicts of the top-20 cached reviews.
_REVIEWS_BY_APP_CACHE: dict[str, list[dict]] = {}
# Cache of the cache_config.yaml read from the active subfolder (None = default criterion).
_CACHE_CONFIG: dict | None = None


def _aggregate_by_word(
    pairs: list[tuple[str, float]],
    signed: bool = False,
) -> list[tuple[str, float]]:
    """Sum weights per unique word (case-insensitive, trailing punctuation removed).

    With ``signed=True`` the result is sorted by |value| desc (the sign is kept).
    Otherwise it is sorted by value desc (Bahdanau convention: always positive).
    """
    agg: dict[str, float] = {}
    for w, val in pairs:
        key = w.strip('.,!?;:()[]"\'').lower()
        if not key:
            continue
        agg[key] = agg.get(key, 0.0) + float(val)
    if signed:
        return sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)


def _reciprocal_rank_fusion(
    bahdanau_ranking: list[tuple[str, float]],
    shap_ranking: list[tuple[str, float]],
    k: int = 60,
    top_n: int = 15,
) -> list[dict]:
    """Reciprocal Rank Fusion between the Bahdanau and textual SHAP rankings.

    For each word, score = sum of 1/(k + rank_in_method). Words present in both
    rankings accumulate both terms; words present in a single method only add
    their own. ``k=60`` is the conventional value (Cormack et al., 2009).

    Returns a list of dicts sorted by ``rrf_score`` desc, keeping track of the
    original ranks.
    """
    bah_map = {w.lower(): (i + 1, v) for i, (w, v) in enumerate(bahdanau_ranking)}
    shap_map = {w.lower(): (i + 1, v) for i, (w, v) in enumerate(shap_ranking)}

    all_words = set(bah_map) | set(shap_map)
    results: list[dict] = []
    for w in all_words:
        r_b = bah_map.get(w, (None, None))[0]
        r_s = shap_map.get(w, (None, None))[0]
        shap_signed = shap_map.get(w, (None, 0.0))[1]
        rrf = 0.0
        if r_b is not None:
            rrf += 1.0 / (k + r_b)
        if r_s is not None:
            rrf += 1.0 / (k + r_s)
        in_both = r_b is not None and r_s is not None
        results.append({
            "word": w,
            "rrf_score": round(rrf, 6),
            "rank_bahdanau": r_b,
            "rank_shap": r_s,
            "shap_signed": round(float(shap_signed), 6),
            "in_both": bool(in_both),
        })
    results.sort(key=lambda r: r["rrf_score"], reverse=True)
    return results[:top_n]



def _aggregate_word_attention(
    word_attention: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Sum weights of repeated words (case-insensitive, trailing punctuation removed).

    ``merge_subtokens_and_attention`` returns one tuple per occurrence of the word
    in the text; the barplot needs unique items. Returns a list sorted by
    aggregated weight, descending.
    """
    agg: dict[str, float] = {}
    for w, weight in word_attention:
        key = w.strip('.,!?;:()[]"\'').lower()
        if not key:
            continue
        agg[key] = agg.get(key, 0.0) + float(weight)
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)


def _apply_negation_bigrams(
    pairs: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Detect (negation, next-word) pairs and combine them into bigrams.

    Reuses ``SHAPExplainer._extract_negation_bigrams`` from the base module, which
    holds the logic calibrated with ``NEGATION_WORDS`` and returns (bigrams,
    consumed indices).

    E.g. ``[('not', -0.8), ('working', +0.4)]`` ->
    ``[('not working', -0.4)]``.

    Applied symmetrically to Bahdanau (attention sum) and textual SHAP (signed
    sum), so both lenses share the same treatment of negations. This avoids the
    artifact where "working" alone gets a positive SHAP value in a bad app
    despite the surrounding negative context.
    """
    bigrams, consumed = SHAPExplainer._extract_negation_bigrams(pairs)
    result: list[tuple[str, float]] = []
    for idx, (w, v) in enumerate(pairs):
        if idx not in consumed:
            result.append((w, v))
    for bigram, v in bigrams.items():
        result.append((bigram, float(v)))
    return result


def _is_punct_only_token(tok: str) -> bool:
    """Return True when ``tok`` (a decoded DeBERTa SentencePiece token) is made
    only of punctuation, after stripping the word-start marker (U+2581) and
    whitespace. Zeroing these tokens in the textual ablation adds noise with no
    semantic value.
    """
    clean = tok.replace("▁", "").strip()
    if not clean:
        return True
    return all(c in string.punctuation for c in clean)


def _get_app_nnnn(package_name: str) -> str | None:
    """Map package_name to NNNN (row index in app_details.csv, zero-padded).

    Mirrors the logic of the training pipeline
    (``multimodal_v3.find_matching_images``). Returns None when the app is not in
    app_details.csv.
    """
    global _APP_DETAILS_CACHE
    if _APP_DETAILS_CACHE is None:
        _APP_DETAILS_CACHE = pd.read_csv(RICO_DIR / "app_details.csv", header=0)
    matches = _APP_DETAILS_CACHE.index[
        _APP_DETAILS_CACHE["App Package Name"] == package_name
    ].tolist()
    if not matches:
        return None
    return f"{matches[0]:04d}"


def _load_cached_review_texts(
    package_name: str,
    true_label: int,
    n_cached: int = 20,
) -> list[dict]:
    """Load the metadata of the N cached reviews of an app, aligned with the .npy
    index (embedding_idx = position in the selected DataFrame).

    Replays the exact criterion used when the cache was generated. Reading
    ``cache_config.yaml`` (when present in ``ACTIVE_CACHE_DIR``) determines:

      - selection_criterion in {"default", "interface_polarized"}
      - texts_per_app (N)
      - min_words
      - app_percentage (documentation only; the app label arrives through the
        ``true_label`` parameter)

    When ``cache_config.yaml`` is absent, the default criterion is assumed
    (head(N) of the non-empty sentences, no min_words filter), which matches
    ``multimodal_v3.generate_missing_embeddings_cache``.

    Returns a list of dicts with: embedding_idx, sentence, interface_pos,
    interface_neg, interface_sentiment, word_count.
    """
    cache_key = (package_name, int(true_label))
    if cache_key in _REVIEWS_BY_APP_CACHE:
        return _REVIEWS_BY_APP_CACHE[cache_key]

    nnnn = _get_app_nnnn(package_name)
    if nnnn is None:
        _REVIEWS_BY_APP_CACHE[cache_key] = []
        return []

    reviews_file = REVIEWS_PROCESSED_DIR / f"app_reviews_{nnnn}_with_aspects.csv"
    if not reviews_file.exists():
        _REVIEWS_BY_APP_CACHE[cache_key] = []
        return []

    try:
        df = pd.read_csv(reviews_file)
    except Exception:
        _REVIEWS_BY_APP_CACHE[cache_key] = []
        return []

    if df.empty:
        _REVIEWS_BY_APP_CACHE[cache_key] = []
        return []

    for col in ("interface_pos", "interface_neg", "interface_neu"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    df["sentence"] = df.get("sentence", pd.Series([""] * len(df))).astype(str)
    df = df[df["sentence"].str.strip() != ""]

    # Replay the criterion recorded in cache_config.yaml, when it exists.
    if _CACHE_CONFIG is not None:
        criterion = str(_CACHE_CONFIG.get("selection_criterion", "default"))
        texts_per_app = int(_CACHE_CONFIG.get("texts_per_app", n_cached))
        min_words = int(_CACHE_CONFIG.get("min_words", 0))
        if min_words > 0:
            df = df[
                df["sentence"].astype(str).apply(lambda s: len(s.split()) >= min_words)
            ]
        if criterion == "interface_polarized":
            ascending = int(true_label) == 0
            df = df.sort_values("interface_pos", ascending=ascending, kind="mergesort")
        df = df.head(texts_per_app)
    else:
        # No cache config available: default criterion, no min_words filter.
        df = df.head(n_cached)

    df = df.reset_index(drop=True)

    rows: list[dict] = []
    for i, row in df.iterrows():
        rows.append(
            {
                "embedding_idx": int(i),
                "sentence": str(row["sentence"]),
                "interface_pos": float(row["interface_pos"]),
                "interface_neg": float(row["interface_neg"]),
                "interface_sentiment": str(row.get("interface_sentiment", "")),
                "word_count": len(str(row["sentence"]).split()),
            }
        )
    _REVIEWS_BY_APP_CACHE[cache_key] = rows
    return rows


def _ensure_full_aggregates(df_full: pd.DataFrame, out_dir: str) -> None:
    """Build classification_full_by_{screen,app}.csv from df_full (pair-level) when
    they are not already in the same folder. Mirrors the aggregations the training
    pipeline produces inside ``save_full_dataset_predictions``.
    """
    screen_path = os.path.join(out_dir, "classification_full_by_screen.csv")
    app_path = os.path.join(out_dir, "classification_full_by_app.csv")
    need_screen = not os.path.exists(screen_path)
    need_app = not os.path.exists(app_path)
    if not need_screen and not need_app:
        return

    screen_agg = {
        "probabilidade_media": ("probabilidade", "mean"),
        "num_reviews": ("probabilidade", "count"),
        "valor_real": ("valor_real", "first"),
    }
    for col in ("split", "app", "rating", "label_real", "categoria"):
        if col in df_full.columns:
            screen_agg[col] = (col, "first")
    df_by_screen = (
        df_full.groupby(["package_name", "numero_da_tela"])
        .agg(**screen_agg)
        .reset_index()
    )
    df_by_screen["confianca_agregada"] = (
        (df_by_screen["probabilidade_media"] - 0.5).abs() * 2
    )
    df_by_screen["predito"] = df_by_screen["probabilidade_media"].apply(
        lambda p: "bom" if p > 0.5 else "ruim"
    )
    df_by_screen["acerto"] = df_by_screen["valor_real"] == df_by_screen["predito"]

    if need_screen:
        df_by_screen.to_csv(screen_path, index=False)
        print(
            f"[v2] Wrote {os.path.basename(screen_path)} ({len(df_by_screen)} screens)"
        )

    if need_app:
        app_agg = {
            "probabilidade_media": ("probabilidade_media", "mean"),
            "num_telas": ("numero_da_tela", "count"),
            "valor_real": ("valor_real", "first"),
        }
        for col in ("split", "app", "rating", "label_real", "categoria"):
            if col in df_by_screen.columns:
                app_agg[col] = (col, "first")
        df_by_app = (
            df_by_screen.groupby("package_name").agg(**app_agg).reset_index()
        )
        df_by_app["confianca_agregada"] = (
            (df_by_app["probabilidade_media"] - 0.5).abs() * 2
        )
        df_by_app["predito"] = df_by_app["probabilidade_media"].apply(
            lambda p: "bom" if p > 0.5 else "ruim"
        )
        df_by_app["acerto"] = df_by_app["valor_real"] == df_by_app["predito"]
        df_by_app.to_csv(app_path, index=False)
        print(
            f"[v2] Wrote {os.path.basename(app_path)} ({len(df_by_app)} apps)"
        )


def _select_polarized_from_cached(
    package_name: str,
    true_label: int,
    top_k: int,
    min_words: int,
) -> list[dict]:
    """Select the top-K polarized reviews among the cached ones of the app.

    Criterion:
      - word_count >= min_words
      - interface_sentiment in {positive, negative}  (neutral ones are dropped)
      - sorted by interface_pos: desc for a good app (label=1), asc for a bad one
      - top-K

    Returns the sorted list of the K selected reviews (fewer when the filters
    prune too much).
    """
    cached = _load_cached_review_texts(package_name, true_label=int(true_label))
    if not cached:
        return []

    filtered = [
        r
        for r in cached
        if r["word_count"] >= min_words
        and r["interface_sentiment"] in ("positive", "negative")
    ]
    if not filtered:
        return []

    filtered.sort(
        key=lambda r: r["interface_pos"],
        reverse=(true_label == 1),
    )
    return filtered[:top_k]


def _select_polarized_from_raw(
    package_name: str,
    true_label: int,
    top_k: int,
    min_words: int,
    exclude_sentences: set[str] | None = None,
) -> list[dict]:
    """Select the top-K polarized reviews from the raw CSV
    ``app_reviews_<NNNN>_with_aspects.csv``, ignoring the embedding cache.

    Same criterion as ``_select_polarized_from_cached`` (min_words +
    interface_sentiment in {positive, negative} + sorting by interface_pos), but
    scanning all reviews of the app, not only the 20 cached ones. Every returned
    item has ``embedding_idx=None``, signalling that the embedding must be
    computed on the fly inside the XAI pipeline.

    ``exclude_sentences``: when non-empty, drops reviews whose text (sentence) is
    already in it, used by the "raw+cache" mode to avoid duplicating the cached
    reviews.
    """
    nnnn = _get_app_nnnn(package_name)
    if nnnn is None:
        return []

    reviews_file = REVIEWS_PROCESSED_DIR / f"app_reviews_{nnnn}_with_aspects.csv"
    if not reviews_file.exists():
        return []

    try:
        df = pd.read_csv(reviews_file)
    except Exception:
        return []

    if df.empty:
        return []

    for col in ("interface_pos", "interface_neg", "interface_neu"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    df["sentence"] = df.get("sentence", pd.Series([""] * len(df))).astype(str)
    df = df[df["sentence"].str.strip() != ""]
    if "interface_sentiment" not in df.columns:
        return []

    df["_word_count"] = df["sentence"].apply(lambda s: len(str(s).split()))
    df = df[df["_word_count"] >= int(min_words)]
    df = df[df["interface_sentiment"].isin(["positive", "negative"])]
    if exclude_sentences:
        df = df[~df["sentence"].isin(exclude_sentences)]
    if df.empty:
        return []

    ascending = int(true_label) == 0
    df = df.sort_values("interface_pos", ascending=ascending, kind="mergesort")
    df = df.head(int(top_k))

    rows: list[dict] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "embedding_idx": None,  # on-the-fly
                "sentence": str(row["sentence"]),
                "interface_pos": float(row["interface_pos"]),
                "interface_neg": float(row["interface_neg"]),
                "interface_sentiment": str(row["interface_sentiment"]),
                "word_count": int(row["_word_count"]),
            }
        )
    return rows


def _select_polarized_reviews(
    package_name: str,
    true_label: int,
    top_k: int,
    min_words_cache: int,
    min_words_raw: int,
    source: str,
) -> list[dict]:
    """Dispatcher by source (``source`` in {"cache", "raw", "raw+cache"}).

    - "cache":     top-K of the 20 cached reviews (``_select_polarized_from_cached``).
                   Every item has an integer ``embedding_idx`` (lookup in the .npy).
    - "raw":       top-K over ALL reviews of the app from the raw CSV. Every item
                   has ``embedding_idx=None`` (embedding computed on the fly).
    - "raw+cache": merge of both sources, with the cached reviews first (they are
                   already in the cache and therefore cheaper). Deduplicates by
                   ``sentence`` and truncates at top-K.
    """
    src = (source or "cache").lower().strip()
    if src == "cache":
        return _select_polarized_from_cached(
            package_name=package_name,
            true_label=int(true_label),
            top_k=int(top_k),
            min_words=int(min_words_cache),
        )
    if src == "raw":
        return _select_polarized_from_raw(
            package_name=package_name,
            true_label=int(true_label),
            top_k=int(top_k),
            min_words=int(min_words_raw),
        )
    if src in ("raw+cache", "cache+raw", "merge"):
        # In "raw+cache" the quality filter is uniform: both sources use
        # ``min_words_raw`` as the threshold. The cache is only preferred because
        # it is cheaper (it avoids an on-the-fly DeBERTa forward pass). So a short
        # cached review (<min_words_raw) is discarded and a raw one takes its slot
        # with a long enough review.
        cached = _select_polarized_from_cached(
            package_name=package_name,
            true_label=int(true_label),
            top_k=int(top_k),
            min_words=int(min_words_raw),
        )
        remaining = max(0, int(top_k) - len(cached))
        if remaining == 0:
            return cached[: int(top_k)]
        exclude = {c["sentence"] for c in cached}
        raw = _select_polarized_from_raw(
            package_name=package_name,
            true_label=int(true_label),
            top_k=remaining,
            min_words=int(min_words_raw),
            exclude_sentences=exclude,
        )
        return (cached + raw)[: int(top_k)]
    # Fallback: treat it as cache.
    print(
        f"[WARN] unknown per_review_source={source!r}; falling back to 'cache'."
    )
    return _select_polarized_from_cached(
        package_name=package_name,
        true_label=int(true_label),
        top_k=int(top_k),
        min_words=int(min_words_cache),
    )


# =============================================================================
# ORCHESTRATOR
# =============================================================================
class MultimodalExplanationV2(MultimodalExplanation):
    """Extend ``MultimodalExplanation`` with faithfulness, object-level deletion
    and the full OBI ranking.

    The base class handles the whole original sequence (Grad-CAM, saliency,
    DeBERTa attention, Bahdanau attention, visualization, top-3 components); a
    short second pass here adds the extra measurements.
    """

    @property
    def shap_explainer(self) -> SHAPExplainer:
        """Lazy-init SHAPExplainer, reusing the same model + text_attn."""
        if not hasattr(self, "_shap_explainer_cached"):
            self._shap_explainer_cached = SHAPExplainer(self.model, self.text_attn)
        return self._shap_explainer_cached

    @property
    def _per_review_artifacts(self) -> dict:
        """In-memory cache of transient artifacts (heatmaps + lists) keyed by
        (pkg, screen_id, rank), consumed by the aggregation in ``explain_app``.

        Structure::
            { pkg: { screen_id: { rank: {
                'gradcam': np.ndarray (224,224),
                'shap_visual': np.ndarray (224,224) | None,
                'word_attention': list[(word, weight)],
                'shap_textual': list[(word, signed_value)],
                'confidence': float,
                'image_path': str,
                'delta_p_img': float | None,
                'review_text': str,
                'embedding_idx': int,
            }}}}
        """
        if not hasattr(self, "_pr_artifacts_cache"):
            self._pr_artifacts_cache = {}
        return self._pr_artifacts_cache

    def _compute_shap_for_review(
        self,
        image: np.ndarray,
        image_batch: np.ndarray,
        text_embedding_batch: np.ndarray,
        review_text: str,
        max_evals: int,
    ) -> tuple[np.ndarray | None, list[tuple[str, float]]]:
        """Visual SHAP (superpixel) + textual SHAP (regex tokenizer) for one review.

        Returns:
            shap_visual_heatmap: (H, W) float [0,1], or None on failure.
            shap_textual_pairs: unsorted list of (word, signed shap_value).
        """
        import shap  # lazy, heavy

        shap_visual: np.ndarray | None = None
        try:
            shap_visual, _, _ = self.shap_explainer._compute_image_shap_superpixel(
                image=image,
                text_embedding=text_embedding_batch,
                n_segments=50,
                n_samples=max_evals,
            )
        except Exception as e:
            print(f"  [SHAP vis] failed: {e}")

        word_shap_pairs: list[tuple[str, float]] = []
        try:
            text_predict_fn = self.shap_explainer._make_text_predict_fn(image_batch)
            text_masker = shap.maskers.Text(tokenizer=r"\S+")
            text_explainer_obj = shap.Explainer(text_predict_fn, text_masker)
            result = text_explainer_obj(
                [review_text], max_evals=max_evals, batch_size=5
            )
            tokens = result.data[0]
            vals = result.values[0]
            if vals.ndim > 1:
                vals = vals[:, 0]
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            word_shap_pairs = [
                (str(tok).strip(), float(val))
                for tok, val in zip(tokens, vals)
                if str(tok).strip()
            ]
            # Filter symmetric to Bahdanau (via merge_subtokens_and_attention):
            # drops stopwords, contractions and numbers, keeps negations
            # (STOPWORDS/CONTRACTIONS exclude 'no'/'not' and don't/can't...).
            word_shap_pairs = [
                (w, v) for w, v in word_shap_pairs
                if not _is_filtered_word(w)
            ]
            # Combine adjacent negations into bigrams (e.g. "not working").
            word_shap_pairs = _apply_negation_bigrams(word_shap_pairs)
        except Exception as e:
            print(f"  [SHAP txt] failed: {e}")

        return shap_visual, word_shap_pairs

    def explain_screen(
        self,
        image_path,
        text,
        text_embedding,
        true_label,
        package_name,
        screen_id,
        output_dir,
        review_texts=None,
        app_name=None,
        category=None,
    ):
        explanation = super().explain_screen(
            image_path=image_path,
            text=text,
            text_embedding=text_embedding,
            true_label=true_label,
            package_name=package_name,
            screen_id=screen_id,
            output_dir=output_dir,
            review_texts=review_texts,
            app_name=app_name,
            category=category,
        )
        if explanation is None:
            return None

        # Default values (kept even when the block below does not run).
        explanation["faithfulness"] = {}
        explanation["top_components_ablated"] = []
        explanation["top_components"] = []
        explanation["components_full_ranked"] = []

        hierarchy_path = RICO_HIERARCHIES_DIR / f"{screen_id}.json"
        if not hierarchy_path.exists():
            return explanation

        # Recompute the heatmap (cheap) to get the 224x224 matrix, needed both by
        # the full ranking and by the image ablation.
        image = preprocess_image_path(image_path)
        if image is None:
            return explanation
        image_batch = np.expand_dims(image, 0)
        text_batch = np.expand_dims(text_embedding, 0)

        real_gradcam_heatmap, _ = self.gradcam.compute_real_gradcam(image_batch, text_batch)

        # rank_elements_by_saliency returns the list sorted by score_total, which
        # favors large components. OBI prefers score_mean (saliency per pixel), as
        # it highlights focused components over spread-out containers.
        ranked_full = rank_elements_by_saliency(
            real_gradcam_heatmap, hierarchy_path, image_shape=(IMG_SIZE, IMG_SIZE)
        )
        ranked_full = sorted(
            ranked_full, key=lambda c: c["score_mean"], reverse=True
        )
        explanation["components_full_ranked"] = [
            {
                "rank": i,
                "component": c["component"],
                "score_total": c["score_total"],
                "score_mean": c["score_mean"],
                "bounds_image": list(c["bounds_image"]),
                "bounds_rico": list(c["bounds_rico"]),
            }
            for i, c in enumerate(ranked_full)
        ]

        if not ABLATION_ENABLED or not ranked_full:
            return explanation

        # Bahdanau must be available in order to pick the top-K tokens.
        if self.bahdanau is None or not self.bahdanau.available:
            return explanation

        self.text_attn.load_model()
        _, attn = self.bahdanau.extract_attention(image_batch, text_batch)
        if attn is None:
            return explanation

        use_text = (review_texts[0] if review_texts else (text or "")).strip()
        if not use_text:
            return explanation

        encoded = self.text_attn.tokenizer(
            use_text,
            text_pair=ASPECT,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        )
        ids = encoded["input_ids"][0].tolist()
        tokens_decoded = self.text_attn.tokenizer.convert_ids_to_tokens(ids)
        special_ids = {
            self.text_attn.tokenizer.cls_token_id,
            self.text_attn.tokenizer.sep_token_id,
            self.text_attn.tokenizer.pad_token_id,
        }
        # Exclude special tokens (CLS/SEP/PAD) and punctuation-only tokens: zeroing
        # punctuation adds noise to the ablation with no semantic value.
        exclude = [
            i
            for i, (tid, tok) in enumerate(zip(ids, tokens_decoded))
            if tid in special_ids or _is_punct_only_token(tok)
        ]

        top_tok = top_k_token_indices(
            np.asarray(attn).ravel(), ABLATION_TOP_K_TOKENS, exclude=exclude
        )
        if not top_tok:
            return explanation

        # Top-K components (K = OBI_TOP_K_COMPONENTS) zeroed out together for ΔP_img.
        k = max(1, min(OBI_TOP_K_COMPONENTS, len(ranked_full)))
        top_k = ranked_full[:k]
        top_bounds_list = [c["bounds_image"] for c in top_k]

        delta_p_info = compute_delta_p(
            image=image,
            text_embedding=text_batch[0],
            bounds_list=top_bounds_list,
            model=self.model,
        )

        top_tokens_decoded = []
        for ti in top_tok:
            try:
                tok_str = self.text_attn.tokenizer.convert_ids_to_tokens([ids[ti]])[0]
            except Exception:
                tok_str = ""
            top_tokens_decoded.append({"index": int(ti), "token": tok_str})

        # ``faithfulness`` block: the record of the visual ablation.
        # ``delta_p_img`` feeds the screen-level CSV and compare_xai, while
        # ``p_orig``/``p_img`` are the two probabilities it comes from (kept for
        # auditing, since ``delta_p_img`` is an absolute value and on its own does
        # not reveal the direction of the change).
        explanation["faithfulness"] = {
            "p_orig": delta_p_info["p_orig"],
            "p_img": delta_p_info["p_img"],
            "delta_p_img": delta_p_info["delta_p_img"],
        }
        explanation["top_components_ablated"] = [
            {
                "rank": i,
                "component": c["component"],
                "bounds_image": list(c["bounds_image"]),
                "bounds_rico": list(c["bounds_rico"]),
                "score_total": c["score_total"],
                "score_mean": c["score_mean"],
            }
            for i, c in enumerate(top_k)
        ]
        # ``top_components`` is the key ``common.nlg`` consumes to build the
        # component sentences (``_aggregate_top_components``). Same format used by
        # ``unimodal_v2``, so both pipelines share a single contract: without it
        # the multimodal NLG loses the ``simple`` sentence and the component list
        # of the ``detailed`` level.
        explanation["top_components"] = [
            {
                "component": c["component"],
                "score_total": c["score_total"],
                "score_mean": c["score_mean"],
                "bounds_image": list(c["bounds_image"]),
            }
            for c in top_k
        ]
        explanation["top_k_tokens"] = top_tokens_decoded

        # OBI(SHAP) is computed per review in ``_per_review_analysis`` (where the
        # visual SHAP is already produced), not here. Pair-level is its natural
        # granularity, matching the Grad-CAM measurements of the per-review pass.

        # Per-review analysis: runs the full XAI pipeline for each polarized review
        # selected among the cached ones (optional, controlled by a flag).
        if PER_REVIEW_ANALYSIS:
            explanation["per_review_analyses"] = self._per_review_analysis(
                image=image,
                image_batch=image_batch,
                package_name=package_name,
                screen_id=screen_id,
                true_label=true_label,
                image_path=image_path,
                output_dir=output_dir,
                extra_metadata={
                    "app_name": app_name or package_name,
                    "category": category or "Unknown",
                    "prediction": float(explanation.get("prediction", 0.0)),
                    "predicted_class": explanation.get("predicted_class", "?"),
                    "true_class": explanation.get("true_class", "?"),
                    "is_correct": bool(explanation.get("is_correct", False)),
                    "confidence": float(explanation.get("confidence", 0.0)),
                },
            )
        else:
            explanation["per_review_analyses"] = []

        # With per-review figures on, drop the combined screen figure
        # (saliency + DeBERTa + Bahdanau) produced by super().explain_screen.
        if PER_REVIEW_GENERATE_FIGURES:
            old_viz = explanation.get("visualization_path")
            if old_viz and os.path.exists(old_viz):
                try:
                    os.remove(old_viz)
                except OSError:
                    pass
            explanation["visualization_path"] = None

        # review_texts carries the polarized texts selected for this screen.
        pr_list = explanation.get("per_review_analyses") or []
        if pr_list:
            explanation["review_texts"] = [
                pr.get("review_text", "") for pr in pr_list
            ]

        # text_explanation: deterministic English NLG (caption/simple/detailed)
        # produced by nlg.py.
        explanation.setdefault("app_name", app_name or package_name)
        explanation.setdefault("category", category or "Unknown")
        explanation["text_explanation"] = generate_screen_text(explanation)
        return explanation

    def _compute_embedding_on_the_fly(self, text: str) -> np.ndarray | None:
        """Compute the DeBERTa ABSA embedding (128, 768) for one review at runtime.

        Reuses ``self.text_attn.model`` (DeBERTa is already loaded for attention),
        so the checkpoint and the tokenization hyperparameters match the training
        pipeline (``training/multimodal_v3.extract_deberta_embeddings``):
        text_pair=ASPECT, max_length=MAX_TEXT_LENGTH, padding=max_length.

        Returns a float32 ``np.ndarray`` (128, 768), or ``None`` on failure.
        """
        import torch  # lazy: torch is only needed when per_review_source != "cache"

        try:
            self.text_attn.load_model()
            encoded = self.text_attn.tokenizer(
                text,
                text_pair=ASPECT,
                padding="max_length",
                truncation=True,
                max_length=MAX_TEXT_LENGTH,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.text_attn.device)
            attention_mask = encoded["attention_mask"].to(self.text_attn.device)
            with torch.no_grad():
                outputs = self.text_attn.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            emb = outputs.last_hidden_state.squeeze(0)  # (128, 768)
            return emb.detach().cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[WARN] failed to compute the on-the-fly embedding: {e}")
            return None

    def _per_review_analysis(
        self,
        image: np.ndarray,
        image_batch: np.ndarray,
        package_name: str,
        screen_id: str,
        true_label: int,
        image_path: str | None = None,
        output_dir: str | None = None,
        extra_metadata: dict | None = None,
    ) -> list[dict]:
        """Run the full XAI pipeline for each polarized review (as selected by
        ``PER_REVIEW_SOURCE``) and collect its metrics. One output row per review.

        Supported review sources:
          - "cache":     the ~20 cached reviews (embedding from the .npy).
          - "raw":       reviews from the raw CSV (embedding computed on the fly).
          - "raw+cache": merge, cached ones first, raw ones fill up to top-K.
        """
        reviews = _select_polarized_reviews(
            package_name=package_name,
            true_label=int(true_label),
            top_k=PER_REVIEW_TOP_K,
            min_words_cache=PER_REVIEW_MIN_WORDS,
            min_words_raw=PER_REVIEW_MIN_WORDS_RAW,
            source=PER_REVIEW_SOURCE,
        )
        if not reviews:
            return []

        safe_pkg = package_name.replace(".", "_").replace("/", "_")
        # Load the .npy only when at least one selected review comes from the cache
        # (integer embedding_idx). Otherwise (pure source="raw") the I/O is skipped.
        needs_cache = any(r.get("embedding_idx") is not None for r in reviews)
        app_embeddings = None
        if needs_cache:
            emb_file = ACTIVE_CACHE_DIR / f"{safe_pkg}.npy"
            if not emb_file.exists():
                # The cache was needed but is missing: drop the reviews that depend
                # on embedding_idx and continue with the on-the-fly ones, if any.
                reviews = [r for r in reviews if r.get("embedding_idx") is None]
                if not reviews:
                    return []
            else:
                try:
                    app_embeddings = np.load(str(emb_file), mmap_mode="r")
                except Exception:
                    reviews = [r for r in reviews if r.get("embedding_idx") is None]
                    if not reviews:
                        return []

        hierarchy_path = RICO_HIERARCHIES_DIR / f"{screen_id}.json"
        has_hierarchy = hierarchy_path.exists()

        results: list[dict] = []
        for rank, review in enumerate(reviews):
            idx = review.get("embedding_idx")
            if idx is None:
                # On-the-fly embedding (source "raw", or the raw part of "raw+cache").
                embedding = self._compute_embedding_on_the_fly(review["sentence"])
                if embedding is None:
                    continue
                # Marker -1 in the CSV identifies on-the-fly reviews.
                idx_for_row = -1
            else:
                if app_embeddings is None or idx >= len(app_embeddings):
                    continue
                embedding = np.asarray(app_embeddings[idx]).astype(np.float32)
                idx_for_row = int(idx)
            text_batch_j = np.expand_dims(embedding, 0)

            gradcam_j, _ = self.gradcam.compute_real_gradcam(
                image_batch, text_batch_j
            )

            # Rank components by score_mean (same as the main analysis).
            top_bounds_list: list = []
            top_classes: list[str] = []
            if has_hierarchy:
                raw_ranked = rank_elements_by_saliency(
                    gradcam_j, hierarchy_path, image_shape=(IMG_SIZE, IMG_SIZE)
                )
                ranked_j = sorted(
                    raw_ranked, key=lambda c: c["score_mean"], reverse=True
                )
                k = max(1, min(OBI_TOP_K_COMPONENTS, len(ranked_j)))
                top_bounds_list = [c["bounds_image"] for c in ranked_j[:k]]
                top_classes = [c["component"] for c in ranked_j[:k]]

            # Bahdanau attention + CLS/SEP/PAD/punctuation filter.
            top_tok: list[int] = []
            top_tokens_decoded: list[str] = []
            if (
                self.bahdanau is not None
                and self.bahdanau.available
                and top_bounds_list
            ):
                _, attn = self.bahdanau.extract_attention(
                    image_batch, text_batch_j
                )
                if attn is not None:
                    self.text_attn.load_model()
                    encoded = self.text_attn.tokenizer(
                        review["sentence"],
                        text_pair=ASPECT,
                        padding="max_length",
                        truncation=True,
                        max_length=MAX_TEXT_LENGTH,
                        return_tensors="pt",
                    )
                    ids = encoded["input_ids"][0].tolist()
                    tokens_decoded = self.text_attn.tokenizer.convert_ids_to_tokens(
                        ids
                    )
                    special_ids = {
                        self.text_attn.tokenizer.cls_token_id,
                        self.text_attn.tokenizer.sep_token_id,
                        self.text_attn.tokenizer.pad_token_id,
                    }
                    exclude = [
                        i
                        for i, (tid, tok) in enumerate(zip(ids, tokens_decoded))
                        if tid in special_ids or _is_punct_only_token(tok)
                    ]
                    top_tok = top_k_token_indices(
                        np.asarray(attn).ravel(),
                        ABLATION_TOP_K_TOKENS,
                        exclude=exclude,
                    )
                    top_tokens_decoded = [
                        tokens_decoded[ti]
                        for ti in top_tok
                        if ti < len(tokens_decoded)
                    ]

            if not top_tok or not top_bounds_list:
                continue

            delta_p_info = compute_delta_p(
                image=image,
                text_embedding=embedding,
                bounds_list=top_bounds_list,
                model=self.model,
            )

            # Bahdanau attention mapped onto words (for the figure and the CSV).
            word_attention = merge_subtokens_and_attention(
                tokens_decoded, np.asarray(attn).ravel()
            )
            # Combine adjacent negations into bigrams (symmetric to SHAP). Bahdanau
            # values are always positive (softmax attention), so the bigram sums both
            # attentions, reflecting the total weight the sequence "not X" received.
            word_attention = _apply_negation_bigrams(word_attention)
            word_attention = sorted(
                word_attention, key=lambda x: x[1], reverse=True
            )

            result_row = {
                "review_rank": rank,
                "embedding_idx": int(idx_for_row),
                "review_source": (
                    "raw" if idx_for_row == -1 else "cache"
                ),
                "review_text": review["sentence"],
                "word_count": review["word_count"],
                "interface_pos": review["interface_pos"],
                "interface_neg": review["interface_neg"],
                "interface_sentiment": review["interface_sentiment"],
                "top_components": ", ".join(top_classes),
                "top_k_tokens": ", ".join(top_tokens_decoded),
                "p_orig": delta_p_info["p_orig"],
                "p_img": delta_p_info["p_img"],
                "delta_p_img": delta_p_info["delta_p_img"],
                "gradcam_mean": float(gradcam_j.mean()),
                "gradcam_max": float(gradcam_j.max()),
            }
            results.append(result_row)

            # Per-review SHAP (optional, expensive).
            shap_visual = None
            shap_textual: list[tuple[str, float]] = []
            if PER_REVIEW_INCLUDE_SHAP and PER_REVIEW_GENERATE_FIGURES:
                print(
                    f"  [SHAP per-review] {package_name}/{screen_id} "
                    f"r{rank} (idx {idx_for_row})..."
                )
                shap_visual, shap_textual = self._compute_shap_for_review(
                    image=image,
                    image_batch=image_batch,
                    text_embedding_batch=text_batch_j,
                    review_text=review["sentence"],
                    max_evals=PER_REVIEW_SHAP_MAX_EVALS,
                )

            # OBI(SHAP): SHAP-based ranking and ablation, parallel to the Grad-CAM
            # one computed above. It reuses the per-review ``shap_visual`` (no extra
            # SHAP cost) and the same ``top_tok`` (the textual token selection does
            # not depend on the visual saliency method). Only ``top_bounds_shap``
            # changes, which is what affects ``delta_p_img``.
            top_bounds_shap_list: list = []
            top_classes_shap_list: list[str] = []
            delta_p_info_shap: dict = {}
            if shap_visual is not None and has_hierarchy and top_tok:
                try:
                    raw_ranked_shap = rank_elements_by_saliency(
                        shap_visual,
                        hierarchy_path,
                        image_shape=(IMG_SIZE, IMG_SIZE),
                    )
                    ranked_shap_j = sorted(
                        raw_ranked_shap,
                        key=lambda c: c["score_mean"],
                        reverse=True,
                    )
                    if ranked_shap_j:
                        k_shap_j = max(
                            1,
                            min(OBI_TOP_K_COMPONENTS, len(ranked_shap_j)),
                        )
                        top_bounds_shap_list = [
                            c["bounds_image"]
                            for c in ranked_shap_j[:k_shap_j]
                        ]
                        top_classes_shap_list = [
                            c["component"]
                            for c in ranked_shap_j[:k_shap_j]
                        ]
                        delta_p_info_shap = compute_delta_p(
                            image=image,
                            text_embedding=embedding,
                            bounds_list=top_bounds_shap_list,
                            model=self.model,
                        )
                except Exception as e:
                    print(
                        f"  [OBI(SHAP)] failed on r{rank} "
                        f"({package_name}/{screen_id}): {e}"
                    )

            if delta_p_info_shap:
                result_row["top_components_shap"] = ", ".join(
                    top_classes_shap_list
                )
                result_row["delta_p_img_shap"] = delta_p_info_shap[
                    "delta_p_img"
                ]
            else:
                result_row["top_components_shap"] = None
                result_row["delta_p_img_shap"] = None

            # Enrich result_row (which goes to the JSON) with the aggregated top-15.
            word_attention_top15 = [
                [w_, round(float(wt), 6)]
                for w_, wt in _aggregate_by_word(word_attention)[:15]
            ]
            shap_textual_top15 = [
                [w_, round(float(val), 6)]
                for w_, val in _aggregate_by_word(shap_textual, signed=True)[:15]
            ]
            shap_visual_stats = None
            if shap_visual is not None:
                shap_visual_stats = {
                    "mean": float(np.mean(shap_visual)),
                    "max": float(np.max(shap_visual)),
                    "std": float(np.std(shap_visual)),
                }
            result_row["word_attention_top15"] = word_attention_top15
            result_row["shap_textual_top15"] = shap_textual_top15
            result_row["shap_visual_stats"] = shap_visual_stats

            # Store transient artifacts for the aggregation done in explain_app.
            if PER_REVIEW_GENERATE_FIGURES:
                pr_conf = (
                    float(extra_metadata.get("confidence", 1.0))
                    if extra_metadata
                    else 1.0
                )
                self._per_review_artifacts.setdefault(package_name, {}).setdefault(
                    str(screen_id), {}
                )[int(rank)] = {
                    "gradcam": np.asarray(gradcam_j),
                    "shap_visual": (
                        np.asarray(shap_visual) if shap_visual is not None else None
                    ),
                    "word_attention": list(word_attention),
                    "shap_textual": list(shap_textual),
                    "confidence": pr_conf,
                    "image_path": image_path,
                    "delta_p_img": (
                        float(delta_p_info["delta_p_img"])
                        if delta_p_info.get("delta_p_img") is not None
                        else None
                    ),
                    "review_text": review["sentence"],
                    "embedding_idx": int(idx_for_row),
                }

            # Per-review figure rendering.
            if PER_REVIEW_GENERATE_FIGURES and image_path and output_dir:
                try:
                    self._render_per_review_figure(
                        image_path=image_path,
                        gradcam_heatmap=gradcam_j,
                        word_attention=word_attention,
                        result_row=result_row,
                        package_name=package_name,
                        screen_id=screen_id,
                        metadata=extra_metadata or {},
                        output_dir=output_dir,
                        shap_visual_heatmap=shap_visual,
                        shap_textual_pairs=shap_textual,
                    )
                except Exception as e:
                    print(
                        f"[v2] Failed to render the per-review figure "
                        f"({package_name}/{screen_id}/r{rank}): {e}"
                    )
        return results

    # =========================================================================
    # APP-LEVEL (summary)
    # =========================================================================
    def explain_app(
        self,
        package_name,
        screen_explanations,
        output_dir,
        reviews_list=None,
        all_app_reviews=None,
    ):
        """Render the app summary figure by aggregating the per-review data.

        When ``PER_REVIEW_GENERATE_FIGURES`` is on, the figure is a 3x3 summary
        (main screenshot + aggregated Grad-CAM + aggregated visual SHAP | reviews +
        aggregated Bahdanau + aggregated textual SHAP | metadata bar). Otherwise
        the base implementation is used.
        """
        # super is always called to fill the dict (app_score, is_correct, etc.).
        app_explanation = super().explain_app(
            package_name=package_name,
            screen_explanations=screen_explanations,
            output_dir=output_dir,
            reviews_list=reviews_list,
            all_app_reviews=all_app_reviews,
        )
        if not app_explanation or not PER_REVIEW_GENERATE_FIGURES:
            return app_explanation

        # Drop the figure produced by the base implementation.
        old_viz = app_explanation.get("visualization_path")
        if old_viz and os.path.exists(old_viz):
            try:
                os.remove(old_viz)
            except OSError:
                pass

        # Load the transient artifacts from the run cache.
        artifacts_by_screen = self._per_review_artifacts.get(package_name, {})
        if not artifacts_by_screen:
            # No per-review data available; the summary cannot be composed.
            app_explanation["visualization_path"] = None
            return app_explanation

        # Flatten into a plain list of (screen_id, rank, artifact_dict).
        flat: list[tuple[str, int, dict]] = []
        for sid, ranks in artifacts_by_screen.items():
            for r, art in ranks.items():
                flat.append((sid, int(r), art))

        # 1) Representative screen + review, following APP_REP_SCREEN_CRITERION:
        #    - "delta_p_img": largest ΔP_img, the pair whose visual explanation
        #      proved the most faithful.
        #    - "confidence": highest screen confidence (same criterion as the
        #      screen-ranking modes). Since ``confidence`` is constant within a
        #      screen, ΔP_img breaks ties among the reviews of the chosen screen.
        def _dp(art: dict) -> float:
            """ΔP_img used for sorting; missing/None sorts last."""
            v = art.get("delta_p_img")
            return float(v) if v is not None else -1e9

        if APP_REP_SCREEN_CRITERION == "confidence":
            best = max(flat, key=lambda t: (t[2].get("confidence", -1e9), _dp(t[2])))
        else:
            best = max(flat, key=lambda t: _dp(t[2]))
        rep_screen_id = best[0]
        rep_rank = best[1]
        rep_art = best[2]
        rep_image_path = rep_art["image_path"]

        # 2) Grad-CAM and visual SHAP come from the same representative (screen,
        # review). Aggregating visual heatmaps across different screens makes no
        # geometric sense, since each screen has its own layout and positions.
        agg_gradcam = rep_art["gradcam"]
        agg_shap_visual = rep_art.get("shap_visual")

        # 3) Textual Bahdanau aggregation (sum per word, case-insensitive).
        agg_bahdanau: dict[str, float] = {}
        for _, _, art in flat:
            for w_, wt in art.get("word_attention") or []:
                key = w_.strip('.,!?;:()[]"\'').lower()
                if not key:
                    continue
                agg_bahdanau[key] = agg_bahdanau.get(key, 0.0) + float(wt)
        agg_bahdanau_sorted = sorted(
            agg_bahdanau.items(), key=lambda x: x[1], reverse=True
        )[:15]

        # 4) Textual SHAP aggregation (signed sum, same treatment).
        agg_shap_text: dict[str, float] = {}
        for _, _, art in flat:
            for w_, val in art.get("shap_textual") or []:
                key = w_.strip('.,!?;:()[]"\'').lower()
                if not key:
                    continue
                agg_shap_text[key] = agg_shap_text.get(key, 0.0) + float(val)
        agg_shap_text_sorted = sorted(
            agg_shap_text.items(), key=lambda x: abs(x[1]), reverse=True
        )[:15]

        # 5) Reviews, one per rank, with the full text truncated to fit the box.
        reviews_by_rank: dict[int, dict] = {}
        for _, r, art in flat:
            if r not in reviews_by_rank:
                reviews_by_rank[r] = {
                    "rank": r,
                    "embedding_idx": art.get("embedding_idx"),
                    "text": art.get("review_text", ""),
                }
        top_reviews = [reviews_by_rank[r] for r in sorted(reviews_by_rank.keys())]

        # 6) Aggregated metadata.
        first = screen_explanations[0]
        app_confidence = float(app_explanation.get("app_confidence", 0.0))
        app_prediction = app_explanation.get("app_prediction", "?")
        true_class = app_explanation.get("true_class", "?")
        is_correct = bool(app_explanation.get("is_correct", False))
        n_screens = int(app_explanation.get("num_screens", len(screen_explanations)))
        app_name = first.get("app_name") or package_name
        category = first.get("category") or "-"

        # 7) Rendering.
        new_viz_path = os.path.join(
            output_dir,
            f"{package_name.replace('.', '_')}_app_explanation.png",
        )
        try:
            self._render_app_figure(
                output_path=new_viz_path,
                rep_image_path=rep_image_path,
                rep_screen_id=rep_screen_id,
                rep_review_rank=rep_rank,
                agg_gradcam=agg_gradcam,
                agg_shap_visual=agg_shap_visual,
                agg_bahdanau=agg_bahdanau_sorted,
                agg_shap_textual=agg_shap_text_sorted,
                top_reviews=top_reviews,
                metadata={
                    "package_name": package_name,
                    "app_name": app_name,
                    "category": category,
                    "n_screens": n_screens,
                    "app_prediction": app_prediction,
                    "app_confidence": app_confidence,
                    "true_class": true_class,
                    "is_correct": is_correct,
                },
            )
            app_explanation["visualization_path"] = new_viz_path
            print(
                f"  [v2] App summary figure saved: "
                f"{os.path.basename(new_viz_path)}"
            )
        except Exception as e:
            print(f"[v2] Failed to render the summary figure of {package_name}: {e}")
            app_explanation["visualization_path"] = None

        # Enrich the app dict with the per-review aggregations.
        rrf = _reciprocal_rank_fusion(
            agg_bahdanau_sorted, agg_shap_text_sorted, top_n=15,
        )
        common = [r for r in rrf if r["in_both"]]
        # Both keys are rebuilt below from the per-review data, or dropped.
        for key in ("top_influential_words", "summary"):
            app_explanation.pop(key, None)
        app_explanation["representative_screen"] = {
            "screen_id": rep_screen_id,
            "review_rank": int(rep_rank),
            "delta_p_img": (
                float(rep_art["delta_p_img"])
                if rep_art.get("delta_p_img") is not None
                else None
            ),
            "confidence": float(rep_art.get("confidence", 0.0)),
            "selection_criterion": APP_REP_SCREEN_CRITERION,
            "image_path": rep_image_path,
        }
        app_explanation["top_bahdanau_aggregated"] = [
            [w, round(float(v), 6)] for w, v in agg_bahdanau_sorted
        ]
        app_explanation["top_shap_textual_aggregated"] = [
            [w, round(float(v), 6)] for w, v in agg_shap_text_sorted
        ]
        app_explanation["top_common_words"] = rrf
        app_explanation["polarized_reviews"] = top_reviews
        # Deterministic English NLG (caption/simple/detailed) via nlg.py. The
        # influential words are the aggregated post-fusion Bahdanau ones.
        app_explanation["top_influential_words"] = [
            [w, v] for w, v in app_explanation.get("top_bahdanau_aggregated", [])
        ][:15]
        app_explanation["text_explanation"] = generate_app_text(
            app_explanation, screen_explanations,
        )

        return app_explanation

    def _render_app_figure(
        self,
        output_path: str,
        rep_image_path: str,
        rep_screen_id: str,
        rep_review_rank: int,
        agg_gradcam: np.ndarray,
        agg_shap_visual: np.ndarray | None,
        agg_bahdanau: list[tuple[str, float]],
        agg_shap_textual: list[tuple[str, float]],
        top_reviews: list[dict],
        metadata: dict,
    ) -> None:
        """Same 3x3 layout as the per-review figure, but for the whole app::

            [main screenshot   | aggregated Grad-CAM  | aggregated visual SHAP]
            [reviews text box  | aggregated Bahdanau  | aggregated textual SHAP]
            [metadata bar]

        Panel titles, axis labels and the metadata bar below are deliberately in
        Portuguese: they are rendered into the PNGs, so translating them would
        make regenerated figures disagree with the ones already in use. Comments
        stay in English; figure content does not.
        """
        original = load_original_image(rep_image_path)
        if original is None:
            raise RuntimeError(f"Could not load {rep_image_path}")
        h, w = original.shape[:2]

        # Grad-CAM overlay
        gc_resized = cv2.resize(agg_gradcam, (w, h), interpolation=cv2.INTER_LINEAR)
        gc_uint8 = np.uint8(255 * np.clip(gc_resized, 0, 1))
        gc_colored = cv2.applyColorMap(gc_uint8, cv2.COLORMAP_JET)
        gc_colored = cv2.cvtColor(gc_colored, cv2.COLOR_BGR2RGB)
        gradcam_overlay = cv2.addWeighted(original, 0.5, gc_colored, 0.5, 0)

        # SHAP visual overlay (when available)
        shap_overlay = None
        if agg_shap_visual is not None:
            sv_resized = cv2.resize(
                agg_shap_visual, (w, h), interpolation=cv2.INTER_LINEAR
            )
            sv_uint8 = np.uint8(255 * np.clip(sv_resized, 0, 1))
            sv_colored = cv2.applyColorMap(sv_uint8, cv2.COLORMAP_JET)
            sv_colored = cv2.cvtColor(sv_colored, cv2.COLOR_BGR2RGB)
            shap_overlay = cv2.addWeighted(original, 0.5, sv_colored, 0.5, 0)

        # 4 rows x 6 columns:
        #   Row 0 (visual):   3 equal panels (2 cols each)
        #   Row 1 (text):     2 panels (Bahdanau + SHAP, 3 cols each)
        #   Row 2 (reviews):  full-width text box with the selected reviews
        #   Row 3 (metadata): full-width info bar
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(
            4, 6,
            height_ratios=[3.5, 2.5, 2.8, 0.4],
            hspace=0.35, wspace=0.6,
        )

        # Row 0: aggregated visuals (3 panels)
        ax_orig = fig.add_subplot(gs[0, 0:2])
        ax_orig.imshow(original)
        ax_orig.set_title(
            f"Screenshot principal\n"
            f"(tela {rep_screen_id}, review #{rep_review_rank} — maior "
            f"$\\Delta P_{{img}}$)",
            fontsize=10, fontweight="bold",
        )
        ax_orig.axis("off")

        ax_cam = fig.add_subplot(gs[0, 2:4])
        ax_cam.imshow(gradcam_overlay)
        ax_cam.set_title(
            "Grad-CAM (tela principal)",
            fontsize=11, fontweight="bold",
        )
        ax_cam.axis("off")

        ax_shap_vis = fig.add_subplot(gs[0, 4:6])
        if shap_overlay is not None:
            ax_shap_vis.imshow(shap_overlay)
        else:
            ax_shap_vis.text(
                0.5, 0.5,
                "SHAP Visual\n(não computado)",
                ha="center", va="center", fontsize=11, color="gray",
                transform=ax_shap_vis.transAxes,
            )
        ax_shap_vis.set_title(
            "SHAP — Importância Visual (tela principal)",
            fontsize=11, fontweight="bold",
        )
        ax_shap_vis.axis("off")

        # Row 1: barplots (Bahdanau + textual SHAP, half the width each)
        ax_bah = fig.add_subplot(gs[1, 0:3])
        if agg_bahdanau:
            words = [w_ for w_, _ in agg_bahdanau]
            vals = [v for _, v in agg_bahdanau]
            y_pos = np.arange(len(words))
            ax_bah.barh(y_pos, vals, color=plt.cm.RdYlGn_r(
                np.linspace(0.2, 0.8, len(words))
            ))
            ax_bah.set_yticks(y_pos)
            ax_bah.set_yticklabels(words, fontsize=9)
            ax_bah.invert_yaxis()
            ax_bah.set_xlabel("Atenção somada (agregado)", fontsize=9)
        else:
            ax_bah.text(
                0.5, 0.5, "sem atenção agregada",
                ha="center", va="center", fontsize=10, color="gray",
                transform=ax_bah.transAxes,
            )
        ax_bah.set_title(
            "Bahdanau Attention (agregado)",
            fontsize=11, fontweight="bold",
        )

        ax_shap_txt = fig.add_subplot(gs[1, 3:6])
        if agg_shap_textual:
            words = [w_ for w_, _ in agg_shap_textual]
            vals = [v for _, v in agg_shap_textual]
            y_pos = np.arange(len(words))
            colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
            ax_shap_txt.barh(y_pos, vals, color=colors)
            ax_shap_txt.set_yticks(y_pos)
            ax_shap_txt.set_yticklabels(words, fontsize=9)
            ax_shap_txt.invert_yaxis()
            ax_shap_txt.axvline(0, color="gray", linewidth=0.5)
            ax_shap_txt.set_xlabel(
                "SHAP soma (+ bom / − ruim)", fontsize=9,
            )
        else:
            ax_shap_txt.text(
                0.5, 0.5,
                "SHAP Textual\n(não computado)",
                ha="center", va="center", fontsize=11, color="gray",
                transform=ax_shap_txt.transAxes,
            )
            ax_shap_txt.axis("off")
        ax_shap_txt.set_title(
            "SHAP — Importância Textual (agregado)",
            fontsize=11, fontweight="bold",
        )

        # Row 2: reviews (full-width single column, up to 2 lines each).
        ax_reviews = fig.add_subplot(gs[2, :])
        if top_reviews:
            # Each review: header + up to 2 text lines (wrapped at ~240 chars per
            # line to use the full 18 in width of the figure); the overflow becomes
            # an ellipsis at the end of the second line.
            blocks = []
            wrap_width = 240
            for r in top_reviews:
                txt = (r.get("text") or "").strip()
                wrapped = textwrap.wrap(txt, width=wrap_width)
                if len(wrapped) > 2:
                    line1 = wrapped[0]
                    line2 = wrapped[1]
                    if len(line2) > wrap_width - 3:
                        line2 = line2[: wrap_width - 3] + "..."
                    else:
                        line2 = line2 + "..."
                    body = f"{line1}\n{line2}"
                else:
                    body = "\n".join(wrapped) if wrapped else ""
                header = f"[{r.get('rank')}] "
                blocks.append(f"{header}{body}")
            full_text = "\n\n".join(blocks)
            ax_reviews.text(
                0.01, 0.98, full_text,
                ha="left", va="top", fontsize=8,
                transform=ax_reviews.transAxes, clip_on=True,
                family="sans-serif",
            )
        else:
            ax_reviews.text(
                0.5, 0.5, "sem reviews disponíveis",
                ha="center", va="center",
                fontsize=10, color="gray",
                transform=ax_reviews.transAxes,
            )
        ax_reviews.set_title(
            f"Top-{len(top_reviews)} reviews do app",
            fontsize=11, fontweight="bold",
        )
        ax_reviews.axis("off")

        # Row 3: metadata bar
        ax_meta = fig.add_subplot(gs[3, :])
        pred = str(metadata.get("app_prediction", "?")).upper()
        conf = float(metadata.get("app_confidence", 0.0)) * 100
        true_cls = str(metadata.get("true_class", "?")).upper()
        is_correct = bool(metadata.get("is_correct", False))
        result_symbol = "✓" if is_correct else "✗"
        meta_text = (
            f"App: {metadata.get('app_name', metadata.get('package_name', '-'))}   |   "
            f"Categoria: {metadata.get('category', '-')}   |   "
            f"Telas: {metadata.get('n_screens', '?')}   |   "
            f"Predição: {pred} (conf. {conf:.1f}%)   |   "
            f"Classe real: {true_cls}   |   "
            f"Resultado: {result_symbol}"
        )
        ax_meta.text(
            0.5, 0.5, meta_text,
            ha="center", va="center", fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#F5F5B8" if is_correct else "#F5B8B8",
                edgecolor="gray",
            ),
            transform=ax_meta.transAxes,
        )
        ax_meta.axis("off")

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _render_per_review_figure(
        self,
        image_path: str,
        gradcam_heatmap: np.ndarray,
        word_attention: list[tuple[str, float]],
        result_row: dict,
        package_name: str,
        screen_id: str,
        metadata: dict,
        output_dir: str,
        shap_visual_heatmap: np.ndarray | None = None,
        shap_textual_pairs: list[tuple[str, float]] | None = None,
    ) -> None:
        """Render one PNG per (screen x review) with a 3 x 2 panel grid plus a
        metadata bar::

            [original screenshot | Grad-CAM overlay | visual SHAP]
            [review text         | Bahdanau barplot | textual SHAP]
            [screen_id | app | category | prediction | confidence | true | result]

        The Grad-CAM overlay follows the same recipe as the SHAP one: resize to the
        original resolution, COLORMAP_JET, 0.5/0.5 blend with the native image.

        Panel titles, axis labels and the metadata bar below are deliberately in
        Portuguese: they are rendered into the PNGs, so translating them would
        make regenerated figures disagree with the ones already in use. Comments
        stay in English; figure content does not.
        """
        per_review_dir = os.path.join(output_dir, "per_review")
        os.makedirs(per_review_dir, exist_ok=True)

        safe_pkg = package_name.replace(".", "_").replace("/", "_")
        rank = int(result_row.get("review_rank", 0))
        out_path = os.path.join(
            per_review_dir, f"{safe_pkg}_{screen_id}_r{rank}.png"
        )

        # --- Original image (native resolution) + Grad-CAM overlay ---
        original = load_original_image(image_path)
        if original is None:
            print(f"[v2] Failed to load the original {image_path}, skipping figure.")
            return

        h, w = original.shape[:2]
        hm_resized = cv2.resize(gradcam_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        hm_uint8 = np.uint8(255 * np.clip(hm_resized, 0, 1))
        hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
        gradcam_overlay = cv2.addWeighted(original, 0.5, hm_colored, 0.5, 0)

        # --- matplotlib layout ---
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(
            3, 3,
            height_ratios=[3.5, 2.0, 0.4],
            width_ratios=[1, 1, 1],
            hspace=0.35, wspace=0.45,
        )

        # Row 1: visual
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(original)
        ax_orig.set_title("Screenshot original", fontsize=11, fontweight="bold")
        ax_orig.axis("off")

        ax_cam = fig.add_subplot(gs[0, 1])
        ax_cam.imshow(gradcam_overlay)
        ax_cam.set_title(
            "Grad-CAM (overlay na resolução original)",
            fontsize=11, fontweight="bold",
        )
        ax_cam.axis("off")

        ax_shap_vis = fig.add_subplot(gs[0, 2])
        if shap_visual_heatmap is not None:
            shap_resized = cv2.resize(
                shap_visual_heatmap, (w, h), interpolation=cv2.INTER_LINEAR
            )
            shap_uint8 = np.uint8(255 * np.clip(shap_resized, 0, 1))
            shap_colored = cv2.applyColorMap(shap_uint8, cv2.COLORMAP_JET)
            shap_colored = cv2.cvtColor(shap_colored, cv2.COLOR_BGR2RGB)
            shap_overlay = cv2.addWeighted(original, 0.5, shap_colored, 0.5, 0)
            ax_shap_vis.imshow(shap_overlay)
        else:
            ax_shap_vis.text(
                0.5, 0.5,
                "SHAP Visual\n(desabilitado nesta run)",
                ha="center", va="center",
                fontsize=11, color="gray",
                transform=ax_shap_vis.transAxes,
            )
        ax_shap_vis.set_title(
            "SHAP — Importância Visual", fontsize=11, fontweight="bold"
        )
        ax_shap_vis.axis("off")

        # Row 2: textual
        ax_text = fig.add_subplot(gs[1, 0])
        review_raw = (result_row.get("review_text") or "").strip()
        # Manual wrapping (matplotlib.ax.text ignores wrap=True inside axes).
        review_wrapped = textwrap.fill(review_raw, width=42)
        header = f"Review #{rank} (idx {result_row.get('embedding_idx')})"
        ax_text.text(
            0.04, 0.97,
            f"{header}\n\n{review_wrapped}",
            ha="left", va="top",
            fontsize=9,
            transform=ax_text.transAxes,
            clip_on=True,
        )
        ax_text.set_title("Review text", fontsize=11, fontweight="bold")
        ax_text.axis("off")

        ax_bah = fig.add_subplot(gs[1, 1])
        # Aggregate multiple occurrences of the same word (weights are summed).
        agg_attention = _aggregate_word_attention(word_attention)[:15]
        if agg_attention:
            words = [w for w, _ in agg_attention]
            weights = [val for _, val in agg_attention]
            y_pos = np.arange(len(words))
            ax_bah.barh(y_pos, weights, color=plt.cm.RdYlGn_r(
                np.linspace(0.2, 0.8, len(words))
            ))
            ax_bah.set_yticks(y_pos)
            ax_bah.set_yticklabels(words, fontsize=9)
            ax_bah.invert_yaxis()
            ax_bah.set_xlabel("Peso de atenção (agregado)", fontsize=9)
        else:
            ax_bah.text(
                0.5, 0.5, "sem atenção disponível",
                ha="center", va="center", fontsize=10, color="gray",
                transform=ax_bah.transAxes,
            )
        ax_bah.set_title("Bahdanau Attention", fontsize=11, fontweight="bold")

        ax_shap_txt = fig.add_subplot(gs[1, 2])
        if shap_textual_pairs:
            # Aggregate per word (lowercase, trailing punctuation removed), sum the
            # signed values, sort by |value| desc and keep the top-15.
            agg: dict[str, float] = {}
            for w_, val in shap_textual_pairs:
                key = w_.strip('.,!?;:()[]"\'').lower()
                if not key:
                    continue
                agg[key] = agg.get(key, 0.0) + float(val)
            sorted_pairs = sorted(
                agg.items(), key=lambda x: abs(x[1]), reverse=True
            )[:15]
            if sorted_pairs:
                words = [w_ for w_, _ in sorted_pairs]
                vals = [v for _, v in sorted_pairs]
                y_pos = np.arange(len(words))
                colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
                ax_shap_txt.barh(y_pos, vals, color=colors)
                ax_shap_txt.set_yticks(y_pos)
                ax_shap_txt.set_yticklabels(words, fontsize=9)
                ax_shap_txt.invert_yaxis()
                ax_shap_txt.axvline(0, color="gray", linewidth=0.5)
                ax_shap_txt.set_xlabel("SHAP value (+ bom / − ruim)", fontsize=9)
            else:
                ax_shap_txt.text(
                    0.5, 0.5, "SHAP textual vazio",
                    ha="center", va="center", fontsize=10, color="gray",
                    transform=ax_shap_txt.transAxes,
                )
        else:
            ax_shap_txt.text(
                0.5, 0.5,
                "SHAP Textual\n(desabilitado nesta run)",
                ha="center", va="center",
                fontsize=11, color="gray",
                transform=ax_shap_txt.transAxes,
            )
            ax_shap_txt.axis("off")
        ax_shap_txt.set_title(
            "SHAP — Importância Textual", fontsize=11, fontweight="bold"
        )

        # Row 3: metadata bar
        ax_meta = fig.add_subplot(gs[2, :])
        result_symbol = "✓" if metadata.get("is_correct") else "✗"
        meta_text = (
            f"Screen: {screen_id}   |   "
            f"App: {metadata.get('app_name', package_name)}   |   "
            f"Categoria: {metadata.get('category', '-')}   |   "
            f"Predição: {metadata.get('predicted_class', '?').upper()} "
            f"(conf. {metadata.get('confidence', 0)*100:.1f}%)   |   "
            f"Classe real: {metadata.get('true_class', '?').upper()}   |   "
            f"Resultado: {result_symbol}"
        )
        ax_meta.text(
            0.5, 0.5, meta_text,
            ha="center", va="center", fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#F5F5B8" if metadata.get("is_correct") else "#F5B8B8",
                edgecolor="gray",
            ),
            transform=ax_meta.transAxes,
        )
        ax_meta.axis("off")

        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# OUTPUT MANAGER
# =============================================================================
class ExplanationOutputManagerV2(ExplanationOutputManager):
    """Organize the outputs per app under ``apps/<safe_pkg>/`` plus global JSONs.

    Structure::

        output_dir/
        ├── apps/
        │   └── <safe_pkg>/
        │       ├── app_explanation.png
        │       ├── screen_explanations/
        │       │   └── per_review/
        │       │       └── <pkg>_<screen>_r<rank>.png
        │       ├── screen_explanations.csv
        │       ├── per_review_analyses.csv
        │       ├── components_ranked.csv
        │       ├── app_explanation.json      (snapshot of the app dict)
        │       └── screen_explanations.json  (snapshot of its screen dicts)
        ├── app_nlg.csv                  (appended as each app finishes)
        ├── screen_explanations.json     (global index)
        └── app_explanations.json        (global index)

    The CSVs, the JSON snapshots and the NLG rows of each app are written as soon
    as it finishes, in ``add_app_explanation``, not only in the final
    ``save_all``. That way an interrupted run keeps everything the apps already
    processed produced, including the fields that exist in no CSV.
    """

    def __init__(self, output_dir: str):
        # super().__init__ is not called, to avoid creating the root folders
        # ``screen_explanations/`` and ``app_explanations/`` automatically: with the
        # per-app layout they would always stay empty. The essential attributes are
        # replicated here; those root paths only exist as a fallback for the
        # screen-level modes, whose ``explain_screen`` lazily creates the folder via
        # ``os.makedirs(output_dir, exist_ok=True)`` before writing, so no mode
        # breaks.
        self.output_dir = output_dir
        self.screen_dir = os.path.join(output_dir, "screen_explanations")
        self.app_dir = os.path.join(output_dir, "app_explanations")
        self.shap_dir = os.path.join(output_dir, "shap_explanations")
        self.apps_dir = os.path.join(output_dir, "apps")
        os.makedirs(self.apps_dir, exist_ok=True)
        self.screen_explanations = []
        self.app_explanations = []
        # Apps whose CSVs were already written by ``add_app_explanation``, so that
        # ``save_all`` does not rewrite what is already on disk.
        self._apps_csvs_written: set = set()

    @staticmethod
    def _safe_pkg(package_name: str) -> str:
        return package_name.replace(".", "_").replace("/", "_")

    def app_dir_for(self, package_name: str) -> str:
        """Root directory of the app (holds the CSVs and the figure subfolders)."""
        path = os.path.join(self.apps_dir, self._safe_pkg(package_name))
        os.makedirs(path, exist_ok=True)
        return path

    def screen_dir_for(self, package_name: str) -> str:
        """Subfolder holding the screen figures of the app."""
        path = os.path.join(self.app_dir_for(package_name), "screen_explanations")
        os.makedirs(path, exist_ok=True)
        return path

    def add_app_explanation(self, explanation):
        """Write the CSVs of the app as soon as it finishes.

        ``save_all`` runs a single time, at the end of the whole run. If the
        execution is interrupted, every figure exists but no CSV was written.
        Writing here, each app is complete as soon as it leaves the loop.
        ``save_all`` still rewrites everything at the end, so the final state is
        the same.
        """
        super().add_app_explanation(explanation)
        pkg = (explanation or {}).get("package_name")
        if not pkg:
            return
        try:
            self._write_app_csvs(pkg)
            self._apps_csvs_written.add(pkg)
        except Exception as e:  # noqa: BLE001 - an I/O error must not abort the run
            print(f"[v2] Failed to write the partial CSVs of {pkg}: {e}")
        try:
            self._write_app_json(pkg, explanation)
        except Exception as e:  # noqa: BLE001 - an I/O error must not abort the run
            print(f"[v2] Failed to write the partial JSONs of {pkg}: {e}")
        try:
            self._append_app_nlg(explanation)
        except Exception as e:  # noqa: BLE001 - an I/O error must not abort the run
            print(f"[v2] Failed to append the NLG of {pkg}: {e}")

    def _write_app_json(self, pkg: str, explanation: dict):
        """Snapshot the full dicts of one app under ``apps/<safe_pkg>/``.

        The global ``screen_explanations.json`` / ``app_explanations.json`` are
        only written by ``save_all``, at the very end of the run. Everything the
        analysis produces beyond the CSV columns (``top_words``,
        ``top_bahdanau_aggregated``, ``polarized_reviews``, the app-level
        ``text_explanation``, ...) therefore lives in memory alone until then, so
        an interrupted run loses it for every app already processed.

        Writing the same dicts per app, as each one finishes, keeps them on disk::

            apps/<safe_pkg>/app_explanation.json      (the app dict)
            apps/<safe_pkg>/screen_explanations.json  (its screen dicts)

        ``save_all`` still writes the global indexes at the end, so the final
        state of a complete run is unchanged; these files are the redundancy that
        survives an interruption.
        """
        app_dir = self.app_dir_for(pkg)
        with open(
            os.path.join(app_dir, "app_explanation.json"), "w", encoding="utf-8",
        ) as f:
            json.dump(explanation, f, indent=2, ensure_ascii=False, default=str)

        app_screens = [
            e for e in self.screen_explanations if e["package_name"] == pkg
        ]
        with open(
            os.path.join(app_dir, "screen_explanations.json"), "w", encoding="utf-8",
        ) as f:
            json.dump(app_screens, f, indent=2, ensure_ascii=False, default=str)

    def _append_app_nlg(self, explanation: dict):
        """Append the app-level NLG to ``app_nlg.csv`` as each app finishes.

        The app-level caption/simple/detailed are generated in ``explain_app``
        but reach disk only through ``save_all``; they appear in no figure and in
        no per-app CSV (which carry the *screen*-level NLG). Appending here means
        an interrupted run keeps the NLG of the apps it did finish.

        This is the only writer of the file: ``save_all`` does not rewrite it,
        since the appended rows already cover every app, in the same order.
        """
        tx = (explanation or {}).get("text_explanation") or {}
        pkg = (explanation or {}).get("package_name", "")
        nlg_path = os.path.join(self.output_dir, "app_nlg.csv")
        write_header = not os.path.exists(nlg_path)
        with open(nlg_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["package_name", "level", "text"])
            for lvl in ("caption", "simple", "detailed"):
                w.writerow([pkg, lvl, tx.get(lvl, "")])

    def resume_from_disk(self) -> set:
        """Reload the per-app snapshots of an interrupted run into memory.

        Reads every ``apps/<safe_pkg>/app_explanation.json`` plus its
        ``screen_explanations.json`` back into ``app_explanations`` /
        ``screen_explanations``, so a continued run ends with global indexes
        covering the whole set, not only the apps processed after the restart.

        An app counts as finished only when both snapshots exist: they are
        written by ``add_app_explanation``, which runs after the app-level
        aggregation, so their presence means the app left the loop complete. A
        folder holding only figures (interrupted mid-app, or produced before the
        snapshots existed) is ignored and the app is processed again.

        Returns the set of ``package_name`` already covered, which the caller
        uses to skip them. Their CSVs are marked as written, since they are
        already on disk.
        """
        done: set = set()
        if not os.path.isdir(self.apps_dir):
            return done

        for entry in sorted(os.listdir(self.apps_dir)):
            app_dir = os.path.join(self.apps_dir, entry)
            app_json = os.path.join(app_dir, "app_explanation.json")
            screens_json = os.path.join(app_dir, "screen_explanations.json")
            if not (os.path.exists(app_json) and os.path.exists(screens_json)):
                continue
            try:
                with open(app_json, encoding="utf-8") as f:
                    app_expl = json.load(f)
                with open(screens_json, encoding="utf-8") as f:
                    screens = json.load(f)
            except Exception as e:  # noqa: BLE001 - a corrupt snapshot is skipped
                print(f"[resume] Failed to read the snapshot of {entry}: {e}")
                continue

            pkg = (app_expl or {}).get("package_name")
            if not pkg:
                continue
            self.screen_explanations.extend(screens or [])
            self.app_explanations.append(app_expl)
            self._apps_csvs_written.add(pkg)
            done.add(pkg)

        return done

    def _write_app_csvs(self, pkg: str):
        """Write the CSVs of one app under ``apps/<safe_pkg>/``.

        Kept separate from ``save_all`` so it can be called as soon as each app
        finishes, and not only at the end of the whole run.
        """
        app_dir = self.app_dir_for(pkg)
        app_screens = [
            e for e in self.screen_explanations if e["package_name"] == pkg
        ]

        # screen_explanations.csv (per-app)
        screen_rows = []
        for e in app_screens:
            # ``synergy`` is how the block was named in earlier runs; kept in the
            # lookup so previously generated JSONs can still be read.
            syn = e.get("faithfulness") or e.get("synergy") or {}
            top_list = e.get("top_components_ablated") or []
            top_classes = (
                ", ".join(c["component"] for c in top_list) if top_list else None
            )
            tx = e.get("text_explanation") or {}

            # SHAP aggregation from the per-review analyses of the screen. Since
            # OBI(SHAP) is computed per review, the screen level reports the mean
            # over the pairs.
            pr_list = e.get("per_review_analyses") or []
            shap_pairs = [
                pr for pr in pr_list
                if pr.get("delta_p_img_shap") is not None
            ]
            if shap_pairs:
                delta_p_img_shap = float(
                    np.mean([pr["delta_p_img_shap"] for pr in shap_pairs])
                )
                # ``top_components_shap`` at screen level comes from rank=0, which
                # is more informative than an average.
                rank0 = next(
                    (pr for pr in shap_pairs if pr.get("review_rank") == 0),
                    shap_pairs[0],
                )
                top_components_shap = rank0.get("top_components_shap")
            else:
                delta_p_img_shap = None
                top_components_shap = None

            screen_rows.append({
                "screen_id": e["screen_id"],
                "prediction": e["prediction"],
                "predicted_class": e["predicted_class"],
                "true_label": e["true_label"],
                "is_correct": e["is_correct"],
                "confidence": e["confidence"],
                "caption": tx.get("caption", ""),
                "simple": tx.get("simple", ""),
                "detailed": tx.get("detailed", ""),
                # OBI(Grad-CAM) for the primary review (one value per screen).
                # ``delta_p_img`` carries the visual faithfulness; compare_xai
                # prefers the per-review mean but falls back to this value when
                # per_review_analyses.csv does not exist.
                "delta_p_img": syn.get("delta_p_img"),
                "p_orig": syn.get("p_orig"),
                "top_components": top_classes,
                "top_components_delta_p": syn.get("delta_p_img"),
                # OBI(SHAP) aggregated over the polarized reviews, which allows a
                # Grad-CAM x SHAP triangulation.
                "delta_p_img_shap": delta_p_img_shap,
                "top_components_shap": top_components_shap,
            })
        screen_df = pd.DataFrame(screen_rows)
        screen_df.to_csv(
            os.path.join(app_dir, "screen_explanations.csv"), index=False,
        )

        # components_ranked.csv (per-app)
        # The ``score_pct`` column is the fraction of the total screen saliency each
        # component concentrates, which supports statements such as "component X
        # concentrates Y% of the saliency".
        comp_rows = []
        for e in app_screens:
            comps = e.get("components_full_ranked", []) or []
            if not comps:
                continue
            total_screen_saliency = sum(
                float(c.get("score_total", 0.0)) for c in comps
            ) + 1e-12
            for c in comps:
                comp_rows.append({
                    "screen_id": e["screen_id"],
                    "true_class": e.get("true_class", "-"),
                    "rank": c["rank"],
                    "component": c["component"],
                    "score_total": c["score_total"],
                    "score_mean": c["score_mean"],
                    "score_pct": round(
                        100.0 * float(c["score_total"]) / total_screen_saliency,
                        2,
                    ),
                    "bounds_image": str(c["bounds_image"]),
                })
        if comp_rows:
            pd.DataFrame(comp_rows).to_csv(
                os.path.join(app_dir, "components_ranked.csv"), index=False,
            )

        # The OBI(SHAP) ranking has no screen-level CSV: SHAP is computed per review
        # (in _per_review_analysis), not per screen. The top SHAP components per
        # (screen x review) live in ``per_review_analyses.csv``, in the
        # ``top_components_shap`` column.

        # per_review_analyses.csv (per-app)
        pr_rows = []
        for e in app_screens:
            for pr in e.get("per_review_analyses", []) or []:
                # Drop the bulky fields (the top-15 lists) from the CSV; they are
                # kept in the JSON only.
                pr_csv = {
                    k: v for k, v in pr.items()
                    if k not in {"word_attention_top15",
                                  "shap_textual_top15",
                                  "shap_visual_stats"}
                }
                pr_rows.append({
                    "screen_id": e["screen_id"],
                    "true_class": e.get("true_class", "-"),
                    **pr_csv,
                })
        if pr_rows:
            pd.DataFrame(pr_rows).to_csv(
                os.path.join(app_dir, "per_review_analyses.csv"),
                index=False,
            )


    def save_all(self):
        """Write the per-app CSVs under apps/<pkg>/."""
        if not self.screen_explanations:
            print(f"\nNo explanation to save in: {self.output_dir}")
            return

        # Global indexes (JSON) at the root: a catalog of every dict.
        with open(
            os.path.join(self.output_dir, "screen_explanations.json"),
            "w", encoding="utf-8",
        ) as f:
            json.dump(self.screen_explanations, f, indent=2,
                      ensure_ascii=False, default=str)
        with open(
            os.path.join(self.output_dir, "app_explanations.json"),
            "w", encoding="utf-8",
        ) as f:
            json.dump(self.app_explanations, f, indent=2,
                      ensure_ascii=False, default=str)

        # app_nlg.csv (app | level | generated text) is not written here: each app
        # appends its three rows in ``add_app_explanation`` as it finishes, in the
        # same order this loop would use, so the file is already complete.

        # Per-app CSVs are already written by ``add_app_explanation`` as each app
        # finishes. Here only the apps whose screens exist but that never went
        # through that hook are covered (the screen-level modes, which emit no app
        # explanation), so nothing already on disk is rewritten.
        apps = sorted({e["package_name"] for e in self.screen_explanations})
        for pkg in apps:
            if pkg not in self._apps_csvs_written:
                self._write_app_csvs(pkg)

        print(f"\nExplanations saved in: {self.output_dir}/")
        print(f"  - {len(apps)} apps in apps/")
        print(f"  - global JSONs: screen_explanations.json, app_explanations.json")


# =============================================================================
# ENTRY POINT (called from multimodal.cli)
# =============================================================================
def explain(config: dict) -> None:
    """Explainability entry point: delegates to the base ``explain`` with the
    classes defined here injected.

    Every interactive mode of the base module is preserved; what changes is:
    - ``MultimodalExplanation``   -> ``MultimodalExplanationV2`` (faithfulness/OBI)
    - ``ExplanationOutputManager``-> ``ExplanationOutputManagerV2`` (per-app CSVs)
    - ``RESULTS_EXPLANATIONS_MULTIMODAL`` -> ``RESULTS_EXPLANATIONS_MULTIMODAL_V2``
    """
    import multimodal.explainability.multimodal as v1

    global ABLATION_ENABLED, ABLATION_TOP_K_TOKENS, OBI_TOP_K_COMPONENTS
    global PER_REVIEW_ANALYSIS, PER_REVIEW_TOP_K, PER_REVIEW_MIN_WORDS
    global PER_REVIEW_SOURCE, PER_REVIEW_MIN_WORDS_RAW
    global PER_REVIEW_GENERATE_FIGURES
    global PER_REVIEW_INCLUDE_SHAP, PER_REVIEW_SHAP_MAX_EVALS
    global APP_REP_SCREEN_CRITERION
    global ACTIVE_CACHE_DIR, _CACHE_CONFIG
    ABLATION_ENABLED = bool(config.get("ablation_enabled", ABLATION_ENABLED))
    ABLATION_TOP_K_TOKENS = int(
        config.get("ablation_top_k_tokens", ABLATION_TOP_K_TOKENS)
    )
    OBI_TOP_K_COMPONENTS = int(
        config.get("obi_top_k_components", OBI_TOP_K_COMPONENTS)
    )
    PER_REVIEW_ANALYSIS = bool(config.get("per_review_analysis", PER_REVIEW_ANALYSIS))
    PER_REVIEW_TOP_K = int(config.get("per_review_top_k", PER_REVIEW_TOP_K))
    PER_REVIEW_MIN_WORDS = int(config.get("per_review_min_words", PER_REVIEW_MIN_WORDS))
    _src = str(config.get("per_review_source", PER_REVIEW_SOURCE)).lower().strip()
    if _src not in ("cache", "raw", "raw+cache", "cache+raw", "merge"):
        print(
            f"[WARN] unknown per_review_source={_src!r}; falling back to 'cache'. "
            f"Valid options: 'cache' | 'raw' | 'raw+cache'."
        )
        _src = "cache"
    PER_REVIEW_SOURCE = _src
    PER_REVIEW_MIN_WORDS_RAW = int(
        config.get("per_review_min_words_raw", PER_REVIEW_MIN_WORDS_RAW)
    )
    PER_REVIEW_GENERATE_FIGURES = bool(
        config.get("per_review_generate_figures", PER_REVIEW_GENERATE_FIGURES)
    )
    PER_REVIEW_INCLUDE_SHAP = bool(
        config.get("per_review_include_shap", PER_REVIEW_INCLUDE_SHAP)
    )
    PER_REVIEW_SHAP_MAX_EVALS = int(
        config.get("per_review_shap_max_evals", PER_REVIEW_SHAP_MAX_EVALS)
    )
    _crit = str(
        config.get("app_rep_screen_criterion", APP_REP_SCREEN_CRITERION)
    ).lower().strip()
    if _crit not in ("delta_p_img", "confidence"):
        print(
            f"[WARN] unknown app_rep_screen_criterion={_crit!r}; falling back to "
            f"'delta_p_img'. Valid options: 'delta_p_img' | 'confidence'."
        )
        _crit = "delta_p_img"
    APP_REP_SCREEN_CRITERION = _crit
    # Figures require the analysis, so the dependency is enforced here.
    if PER_REVIEW_GENERATE_FIGURES and not PER_REVIEW_ANALYSIS:
        PER_REVIEW_ANALYSIS = True
        print(
            "[INFO] per_review_generate_figures=true requires the analysis; "
            "enabling per_review_analysis automatically."
        )
    # Per-review SHAP requires the figures (it is rendered inside them).
    if PER_REVIEW_INCLUDE_SHAP and not PER_REVIEW_GENERATE_FIGURES:
        PER_REVIEW_GENERATE_FIGURES = True
        print(
            "[INFO] per_review_include_shap=true requires the figures; "
            "enabling per_review_generate_figures automatically."
        )

    # Resolve the active cache (root, or the subfolder declared in the YAML).
    subdir = config.get("embeddings_cache_subdir")
    if isinstance(subdir, str) and subdir:
        candidate = EMBEDDINGS_CACHE_DIR / subdir
        if candidate.exists():
            ACTIVE_CACHE_DIR = candidate
        else:
            print(
                f"[WARN] embeddings_cache_subdir='{subdir}' not found in "
                f"{EMBEDDINGS_CACHE_DIR}; using the root."
            )
            ACTIVE_CACHE_DIR = EMBEDDINGS_CACHE_DIR
    else:
        ACTIVE_CACHE_DIR = EMBEDDINGS_CACHE_DIR

    # Read cache_config.yaml when it exists in the active subfolder; it is used by
    # _load_cached_review_texts to replay the selection criterion.
    cfg_path = ACTIVE_CACHE_DIR / "cache_config.yaml"
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                _CACHE_CONFIG = yaml.safe_load(f) or {}
            print(
                f"Cache config detected: criterion="
                f"{_CACHE_CONFIG.get('selection_criterion', '?')} | "
                f"N={_CACHE_CONFIG.get('texts_per_app', '?')} | "
                f"min_words={_CACHE_CONFIG.get('min_words', '?')}"
            )
        except Exception as e:
            print(f"[WARN] failed to read {cfg_path}: {e}; using the default criterion.")
            _CACHE_CONFIG = None
    else:
        _CACHE_CONFIG = None

    saved_explanation_cls = v1.MultimodalExplanation
    saved_output_cls = v1.ExplanationOutputManager
    saved_results_path = v1.RESULTS_EXPLANATIONS_MULTIMODAL
    saved_loader_cache = v1.TestDataLoader.EMBEDDINGS_CACHE
    saved_load_csv = v1.TestDataLoader._load_classification_csv

    v1.MultimodalExplanation = MultimodalExplanationV2
    v1.ExplanationOutputManager = ExplanationOutputManagerV2
    v1.RESULTS_EXPLANATIONS_MULTIMODAL = RESULTS_EXPLANATIONS_MULTIMODAL_V2
    # The base TestDataLoader is pointed at the same cache subfolder, so the main
    # analysis and the per-review one read from the same directory.
    v1.TestDataLoader.EMBEDDINGS_CACHE = str(ACTIVE_CACHE_DIR)

    # ``TestDataLoader.get_sample`` uses nlargest(5, 'confianca'), but a
    # classification CSV may only carry 'probabilidade'. The column is derived when
    # missing and persisted to disk, so later runs do not need the patch. The
    # aggregates classification_full_by_{screen,app}.csv are generated as well when
    # they are absent.
    def _patched_load_classification_csv(loader_self):
        csv_path = loader_self.model_info.get("classification_csv")
        if csv_path and os.path.exists(csv_path):
            try:
                df_disk = pd.read_csv(csv_path)
                if (
                    "confianca" not in df_disk.columns
                    and "probabilidade" in df_disk.columns
                ):
                    df_disk["confianca"] = (
                        (df_disk["probabilidade"].astype(float) - 0.5).abs() * 2
                    ).round(4)
                    df_disk.to_csv(csv_path, index=False)
                    print(f"[v2] Backfilled 'confianca' in {csv_path}")

                # For classification_full_dataset.csv, build the per-screen and
                # per-app aggregates when they do not exist, mirroring what
                # save_full_dataset_predictions does during training.
                basename = os.path.basename(csv_path)
                if (
                    basename == "classification_full_dataset.csv"
                    and "numero_da_tela" in df_disk.columns
                    and "package_name" in df_disk.columns
                ):
                    base_dir = os.path.dirname(csv_path)
                    _ensure_full_aggregates(df_disk, base_dir)
            except Exception as e:
                print(f"[v2] Warning: failed to prepare {csv_path}: {e}")

        saved_load_csv(loader_self)

    v1.TestDataLoader._load_classification_csv = _patched_load_classification_csv

    print("=" * 70)
    print("MULTIMODAL EXPLAINABILITY MODULE")
    print("Faithfulness (delta P_img) + object-level deletion (ObEy) + full OBI")
    print(
        f"Ablation: {ABLATION_ENABLED} | top-K tokens: {ABLATION_TOP_K_TOKENS} | "
        f"top-K components: {OBI_TOP_K_COMPONENTS} (ranked by score_mean)"
    )
    print(
        f"Per-review analysis: {PER_REVIEW_ANALYSIS} "
        f"(top-{PER_REVIEW_TOP_K} reviews, min_words={PER_REVIEW_MIN_WORDS}) | "
        f"figures: {PER_REVIEW_GENERATE_FIGURES} | "
        f"SHAP: {PER_REVIEW_INCLUDE_SHAP} (max_evals={PER_REVIEW_SHAP_MAX_EVALS})"
    )
    print(f"Active cache dir: {ACTIVE_CACHE_DIR}")
    print("=" * 70)

    try:
        v1.explain(config)
    finally:
        v1.MultimodalExplanation = saved_explanation_cls
        v1.ExplanationOutputManager = saved_output_cls
        v1.RESULTS_EXPLANATIONS_MULTIMODAL = saved_results_path
        v1.TestDataLoader.EMBEDDINGS_CACHE = saved_loader_cache
        v1.TestDataLoader._load_classification_csv = saved_load_csv
