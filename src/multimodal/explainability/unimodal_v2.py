#!/usr/bin/env python3
"""Unimodal explainability — OBI ΔP_img + SHAP Visual + per-app output.

Mirrors the architecture of ``multimodal_v2.py``, injecting its classes into
the ``unimodal.explain`` flow by monkey-patching, and adds:

1. **Object-level deletion (ObEy-style, ICCV 2023)**: zeroes the top-K (K=3)
   most salient UI components of the Grad-CAM, ranked by ``score_mean`` over
   the Rico hierarchy, and measures the confidence drop, ``ΔP_img``. It is
   comparable 1:1 with the ``ΔP_img`` the multimodal pipeline produces, which
   is what makes the unimodal vs multimodal faithfulness comparison possible.

2. **SHAP Visual**: ``shap.KernelExplainer`` over SLIC superpixels of the
   image. A causal per-pixel attribution complementing Grad-CAM attention,
   matching the "SHAP Visual" column of the multimodal figure.

3. **Raw Grad-CAM heatmap as ``.npy``** per screen (224x224 float32), which
   allows heatmap similarity comparisons between the two pipelines.

4. **Per-app output layout** (``apps/<pkg>/``) plus a per-app
   ``components_ranked.csv``, identical to the multimodal side.

**Figure parity with the multimodal pipeline**, dropping the text rows that do
not apply here:

    Multimodal per-screen:         Unimodal per-screen:
    ─────────────────────────────  ─────────────────────────────
    screenshot | Grad-CAM | SHAP   screenshot | Grad-CAM | SHAP
    ─────────────────────────────  ─────────────────────────────
    review text | Bahdanau | SHAP  (dropped: no text branch)
    ─────────────────────────────
        metadata bar                   metadata bar

Outputs in ``results/explanations/unimodal_v2/explicabilidade_v2_<timestamp>/``.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

from multimodal.common.faithfulness import compute_obi_delta_p
from multimodal.common.paths import (
    RESULTS_EXPLANATIONS_UNIMODAL_V2,
    RICO_HIERARCHIES_DIR,
)
from multimodal.common.semantic_saliency import rank_elements_by_saliency
from multimodal.explainability.unimodal import (
    ExplanationOutputManager,
    IMG_SIZE,
    ImageExplanation,
    load_original_image,
    preprocess_image_path,
)

# Module config (overwritten in explain(config))
# Components are ranked by ``score_mean`` (per-pixel saliency) and the top-K
# are zeroed together, following the top-K object deletion semantics of ObEy.
OBI_TOP_K_COMPONENTS: int = 3
# Per-screen visual SHAP (KernelExplainer over SLIC superpixels).
SHAP_ENABLED: bool = True
SHAP_MAX_EVALS: int = 500
# Number of SLIC superpixels, matching the multimodal SHAPExplainer.
SHAP_N_SEGMENTS: int = 50

# Module-level cache {package_name: screen_id}: the representative screen of
# the multimodal run when ``APPS_FROM_MULTIMODAL_RUN`` is set. Read once per
# run via ``_multimodal_rep_screens()``.
_MULTIMODAL_REP_CACHE: dict[str, str] | None = None


def _multimodal_rep_screens() -> dict[str, str]:
    """Load {package_name -> representative_screen.screen_id} from the
    ``app_explanations.json`` of the multimodal run referenced by
    ``unimodal.APPS_FROM_MULTIMODAL_RUN``. Module-level cache; returns an empty
    dict when the config is unset or the file is missing.
    """
    global _MULTIMODAL_REP_CACHE
    if _MULTIMODAL_REP_CACHE is not None:
        return _MULTIMODAL_REP_CACHE
    import multimodal.explainability.unimodal as v1
    src = getattr(v1, "APPS_FROM_MULTIMODAL_RUN", None)
    if not src:
        _MULTIMODAL_REP_CACHE = {}
        return _MULTIMODAL_REP_CACHE
    app_json = Path(src) / "app_explanations.json"
    if not app_json.exists():
        _MULTIMODAL_REP_CACHE = {}
        return _MULTIMODAL_REP_CACHE
    try:
        with open(app_json, encoding="utf-8") as f:
            apps = json.load(f)
    except Exception as e:
        print(f"[unimodal_v2] Failed to read {app_json}: {e}")
        _MULTIMODAL_REP_CACHE = {}
        return _MULTIMODAL_REP_CACHE
    mapping: dict[str, str] = {}
    for a in apps:
        pkg = a.get("package_name")
        rep = a.get("representative_screen") or {}
        sid = rep.get("screen_id")
        if pkg and sid is not None:
            mapping[str(pkg)] = str(sid)
    _MULTIMODAL_REP_CACHE = mapping
    return mapping


# =============================================================================
# SHAP IMAGE EXPLAINER (single-input, adapted from multimodal_v2.SHAPExplainer)
# =============================================================================
class SHAPImageExplainer:
    """SHAP over SLIC superpixels for the single-input image model.

    Adapted from the multimodal ``SHAPExplainer``: calls ``model.predict(batch)``
    directly instead of ``[image_batch, text_batch]``. The rest of the logic
    (SLIC + KernelExplainer + mapping back to pixels) is identical, with the
    same hyperparameters (n_segments, compactness, blur kernel, baseline).
    """

    def __init__(self, keras_model):
        self.keras_model = keras_model
        self._deps_ok = self._check_deps()

    @staticmethod
    def _check_deps() -> bool:
        try:
            import shap  # noqa: F401
            from skimage.segmentation import slic  # noqa: F401
            return True
        except ImportError as e:
            print(
                f"⚠️  SHAP/skimage unavailable ({e}); SHAP Visual will be skipped."
            )
            return False

    def compute(
        self,
        image: np.ndarray,
        n_segments: int = SHAP_N_SEGMENTS,
        n_samples: int = SHAP_MAX_EVALS,
        seed: int = 42,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """Return (shap_heatmap [224x224], segments [224x224], total_abs_shap).

        ``seed`` (default 42) sets ``np.random.seed`` before the KernelExplainer,
        which keeps the SHAP heatmap reproducible across runs on the same
        ``image``. That matters for the analyses that rank components from the
        SHAP heatmap.

        Returns (None, None, 0.0) when SHAP/skimage are not installed.
        """
        if not self._deps_ok:
            return None, None, 0.0

        import shap
        from skimage.segmentation import slic

        # 1. Segment the image with SLIC (same hyperparameters as the multimodal one)
        img_uint8 = (image * 255).astype(np.uint8)
        segments = slic(img_uint8, n_segments=n_segments, compactness=20)
        unique_segments = np.unique(segments)
        n_features = len(unique_segments)

        # 2. Blurred baseline (the "off" state of a feature for the KernelExplainer)
        blurred = cv2.GaussianBlur(image, (31, 31), 0)

        # Pre-compute per-segment masks so they are not rebuilt on every eval.
        segment_masks = [segments == seg_id for seg_id in unique_segments]

        model = self.keras_model

        def predict_superpixel(masks: np.ndarray) -> np.ndarray:
            """masks: (N, n_features) binary, 1=visible, 0=blurred.

            Batched: builds every masked image and calls ``model.predict`` once
            with the full batch, a 10-30x speedup on GPU over a per-sample loop.
            """
            n = len(masks)
            if n == 0:
                return np.array([])
            batch_imgs = (
                np.broadcast_to(blurred, (n,) + blurred.shape)
                .copy()
                .astype(np.float32)
            )
            for idx, mask in enumerate(masks):
                for j, seg_mask in enumerate(segment_masks):
                    if mask[j] == 1:
                        batch_imgs[idx][seg_mask] = image[seg_mask]
            preds = model.predict(batch_imgs, verbose=0)
            return np.asarray(preds).flatten()

        # 3. KernelExplainer: the baseline is every superpixel blurred. Fixing the
        # seed before shap_values makes the heatmap reproducible across runs on the
        # same input.
        np.random.seed(seed)
        background = np.zeros((1, n_features))
        explainer = shap.KernelExplainer(predict_superpixel, background)
        instance = np.ones((1, n_features))
        shap_values = explainer.shap_values(
            instance, nsamples=n_samples, silent=True
        )

        # 4. Map the SHAP values → pixels.
        if isinstance(shap_values, list):
            sv = shap_values[0].flatten()
        else:
            sv = np.asarray(shap_values).flatten()

        shap_heatmap = np.zeros(image.shape[:2], dtype=np.float32)
        for i, seg_id in enumerate(unique_segments):
            if i < len(sv):
                shap_heatmap[segments == seg_id] = abs(sv[i])

        image_shap_total = float(np.sum(np.abs(sv)))

        # 5. Normalise to [0, 1] for display.
        if shap_heatmap.max() > 0:
            shap_heatmap = shap_heatmap / (shap_heatmap.max() + 1e-8)

        return shap_heatmap, segments, image_shap_total


# =============================================================================
# ORCHESTRATOR V2
# =============================================================================
class UnimodalExplanationV2(ImageExplanation):
    """Extends ``ImageExplanation`` with OBI ΔP_img, SHAP Visual and a new figure.

    Grad-CAM, the initial ranking and the basic statistics are delegated to the
    base class. The overrides re-rank by ``score_mean``, compute ΔP_img through
    top-K deletion, run SHAP Visual and replace the figure with a layout of 3
    visual panels plus a metadata bar, matching the multimodal figure without
    its text rows.
    """

    @property
    def shap_explainer(self) -> SHAPImageExplainer:
        if not hasattr(self, "_shap_explainer_cached"):
            self._shap_explainer_cached = SHAPImageExplainer(self.model)
        return self._shap_explainer_cached

    @property
    def _shap_visual_cache(self) -> dict:
        """In-memory cache of SHAP Visual heatmaps keyed by (pkg, screen_id).

        Filled in ``explain_screen`` after SHAP runs and consumed in
        ``explain_app`` to avoid recomputing it. SHAP is stochastic: two
        KernelExplainer calls over the same image give slightly different
        values. Reusing the cached heatmap keeps the SHAP panel of the app
        figure pixel-identical to the per-screen figure of the same screen.
        """
        if not hasattr(self, "_shap_visual_cache_dict"):
            self._shap_visual_cache_dict = {}
        return self._shap_visual_cache_dict

    # =========================================================================
    # SCREEN-LEVEL
    # =========================================================================
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
        """The base class produces Grad-CAM and top_components by score_total;
        this override re-ranks by score_mean, computes ΔP_img through top-K
        deletion, runs SHAP Visual and replaces the figure.
        """
        # Point output_dir at the per-app folder.
        active_mgr = getattr(UnimodalExplanationV2, "_active_output_manager", None)
        if active_mgr is not None:
            output_dir = active_mgr.screen_dir_for(package_name)

        explanation = super().explain_screen(
            image_path=image_path,
            true_label=int(true_label),
            package_name=package_name,
            screen_id=screen_id,
            output_dir=output_dir,
            app_name=app_name,
            category=category,
        )
        if explanation is None:
            return None

        # Drop the base-class figure, replaced further down.
        old_viz = explanation.get("visualization_path")
        if old_viz and os.path.exists(old_viz):
            try:
                os.remove(old_viz)
            except OSError:
                pass

        # Recompute heatmap and image for the block below.
        image = preprocess_image_path(image_path)
        if image is None:
            return explanation
        image_batch = np.expand_dims(image, 0)
        heatmap, prediction = self.gradcam.compute_gradcam(image_batch)
        safe_pkg = package_name.replace(".", "_").replace("/", "_")

        # OBI: re-rank by score_mean and compute ΔP_img by top-K deletion.
        components_full_ranked: list[dict] = []
        top_components_ablated: list[dict] = []
        obi_info: dict = {}
        hierarchy_path = RICO_HIERARCHIES_DIR / f"{screen_id}.json"
        if hierarchy_path.exists():
            raw_ranked = rank_elements_by_saliency(
                heatmap, hierarchy_path, image_shape=(IMG_SIZE, IMG_SIZE)
            )
            ranked_by_mean = sorted(
                raw_ranked, key=lambda c: c["score_mean"], reverse=True
            )
            components_full_ranked = [
                {
                    "rank": i,
                    "component": c["component"],
                    "score_total": c["score_total"],
                    "score_mean": c["score_mean"],
                    "bounds_image": list(c["bounds_image"]),
                    "bounds_rico": list(c["bounds_rico"]),
                }
                for i, c in enumerate(ranked_by_mean)
            ]
            if ranked_by_mean:
                k = max(1, min(OBI_TOP_K_COMPONENTS, len(ranked_by_mean)))
                top_k = ranked_by_mean[:k]
                top_bounds_list = [c["bounds_image"] for c in top_k]
                obi_info = compute_obi_delta_p(
                    image=image, bounds_list=top_bounds_list, model=self.model
                )
                top_components_ablated = [
                    {
                        "rank": i,
                        "component": c["component"],
                        "bounds_image": list(c["bounds_image"]),
                        "score_total": c["score_total"],
                        "score_mean": c["score_mean"],
                    }
                    for i, c in enumerate(top_k)
                ]

        # Visual SHAP (optional, expensive).
        shap_visual = None
        shap_total = 0.0
        if SHAP_ENABLED:
            try:
                shap_visual, _segments, shap_total = self.shap_explainer.compute(
                    image=image, n_samples=SHAP_MAX_EVALS
                )
                if shap_visual is not None:
                    # Cached for reuse in ``explain_app``, which keeps the
                    # per-screen figure and the app figure identical pixel by
                    # pixel (KernelExplainer is stochastic, but with a fixed seed
                    # it is also deterministic across runs).
                    self._shap_visual_cache[(package_name, str(screen_id))] = (
                        np.asarray(shap_visual)
                    )
            except Exception as e:
                print(
                    f"[unimodal_v2] SHAP failed on {package_name}/{screen_id}: {e}"
                )

        # OBI(SHAP): a parallel ranking from the visual SHAP heatmap.
        # ``rank_elements_by_saliency`` is heatmap-agnostic, so any 2D array
        # works. This enables a Grad-CAM vs SHAP triangulation at UI-component
        # level. Extra cost: two predicts, inside compute_obi_delta_p.
        components_full_ranked_shap: list[dict] = []
        top_components_ablated_shap: list[dict] = []
        obi_info_shap: dict = {}
        if shap_visual is not None and hierarchy_path.exists():
            try:
                raw_ranked_shap = rank_elements_by_saliency(
                    shap_visual,
                    hierarchy_path,
                    image_shape=(IMG_SIZE, IMG_SIZE),
                )
                ranked_shap = sorted(
                    raw_ranked_shap,
                    key=lambda c: c["score_mean"],
                    reverse=True,
                )
                components_full_ranked_shap = [
                    {
                        "rank": i,
                        "component": c["component"],
                        "score_total": c["score_total"],
                        "score_mean": c["score_mean"],
                        "bounds_image": list(c["bounds_image"]),
                        "bounds_rico": list(c["bounds_rico"]),
                    }
                    for i, c in enumerate(ranked_shap)
                ]
                if ranked_shap:
                    k_shap = max(
                        1, min(OBI_TOP_K_COMPONENTS, len(ranked_shap))
                    )
                    top_k_shap = ranked_shap[:k_shap]
                    top_bounds_shap = [
                        c["bounds_image"] for c in top_k_shap
                    ]
                    obi_info_shap = compute_obi_delta_p(
                        image=image,
                        bounds_list=top_bounds_shap,
                        model=self.model,
                    )
                    top_components_ablated_shap = [
                        {
                            "rank": i,
                            "component": c["component"],
                            "bounds_image": list(c["bounds_image"]),
                            "score_total": c["score_total"],
                            "score_mean": c["score_mean"],
                        }
                        for i, c in enumerate(top_k_shap)
                    ]
            except Exception as e:
                print(
                    f"[unimodal_v2] OBI(SHAP) failed on "
                    f"{package_name}/{screen_id}: {e}"
                )

        # Add the new fields to the explanation dict.
        explanation["components_full_ranked"] = components_full_ranked
        explanation["top_components_ablated"] = top_components_ablated
        explanation["obi"] = obi_info
        explanation["components_full_ranked_shap"] = components_full_ranked_shap
        explanation["top_components_ablated_shap"] = top_components_ablated_shap
        explanation["obi_shap"] = obi_info_shap
        explanation["shap_visual_stats"] = (
            {
                "mean": float(np.mean(shap_visual)),
                "max": float(np.max(shap_visual)),
                "std": float(np.std(shap_visual)),
                "total_abs": float(shap_total),
            }
            if shap_visual is not None
            else None
        )

        # Overwrite ``top_components`` with the score_mean ordering, to stay
        # consistent with the OBI ranking.
        explanation["top_components"] = [
            {
                "component": c["component"],
                "score_total": c["score_total"],
                "score_mean": c["score_mean"],
                "bounds_image": c["bounds_image"],
            }
            for c in components_full_ranked[:3]
        ]

        # Render the screen figure (visual row + metadata bar).
        viz_path = os.path.join(
            output_dir, f"{safe_pkg}_{screen_id}_explanation.png"
        )
        try:
            self._render_screen_figure(
                image_path=image_path,
                heatmap=heatmap,
                shap_visual=shap_visual,
                explanation=explanation,
                output_path=viz_path,
            )
            explanation["visualization_path"] = viz_path
        except Exception as e:
            print(f"[unimodal_v2] Failed to render the screen figure: {e}")

        return explanation

    # =========================================================================
    # APP-LEVEL
    # =========================================================================
    def explain_app(self, package_name, screen_explanations, output_dir):
        """Render the app figure from the representative screen, with 3 visual
        panels (screenshot | Grad-CAM | SHAP) of the main screen.

        As on the multimodal side, the heatmap shown is the one of the screen
        picked as representative, not an aggregate, so the app figure is
        identical to the per-screen figure of that same screen.
        """
        active_mgr = getattr(UnimodalExplanationV2, "_active_output_manager", None)
        if active_mgr is not None:
            output_dir = active_mgr.app_dir_for(package_name)

        app_explanation = super().explain_app(
            package_name=package_name,
            screen_explanations=screen_explanations,
            output_dir=output_dir,
        )
        if app_explanation is None:
            return None

        # Drop the figure produced by the base implementation.
        old_viz = app_explanation.get("visualization_path")
        if old_viz and os.path.exists(old_viz):
            try:
                os.remove(old_viz)
            except OSError:
                pass

        # Representative screen: when APPS_FROM_MULTIMODAL_RUN is set, inherit
        # the screen_id from ``representative_screen`` in the multimodal
        # app_explanations.json, whatever criterion picked it there
        # (``app_rep_screen_criterion``: max ΔP_img or max screen confidence).
        # That keeps the two pipelines paired screen by screen for the same
        # package. Fallback: max(confidence) on the unimodal side, which is the
        # standalone behaviour.
        canonical_rep = _multimodal_rep_screens().get(package_name)
        rep = None
        if canonical_rep:
            matches = [
                e for e in screen_explanations
                if str(e.get("screen_id")) == canonical_rep
            ]
            if matches:
                rep = matches[0]
            else:
                print(
                    f"[unimodal_v2] [WARN] canonical screen {canonical_rep} from "
                    f"the multimodal run not found in {package_name}; "
                    f"falling back to max(confidence)."
                )
        if rep is None:
            rep = max(
                screen_explanations,
                key=lambda e: float(e.get("confidence", 0.0)),
            )
        rep_image_path = rep["image_path"]
        rep_screen_id = rep.get("screen_id")

        # Propagate app_name / category (every screen shares them).
        app_explanation.setdefault("app_name", rep.get("app_name"))
        app_explanation.setdefault("category", rep.get("category"))

        # Grad-CAM and SHAP of the representative screen; no aggregation over the
        # screens of the app (the multimodal side does the same). This guarantees
        # that the main-screen Grad-CAM panel of the app figure is identical to the
        # Grad-CAM panel of that screen's own figure.
        rep_heatmap = None
        rep_shap = None
        rep_img = preprocess_image_path(rep_image_path)
        if rep_img is not None:
            rep_heatmap, _ = self.gradcam.compute_gradcam(
                np.expand_dims(rep_img, 0)
            )
            # Visual SHAP: the cache filled in ``explain_screen`` comes first, as it
            # holds the exact value of the per-screen figure. It is only recomputed
            # when that cache is missing, for instance when SHAP failed the first
            # time around.
            if SHAP_ENABLED and rep.get("shap_visual_stats") is not None:
                cache_key = (package_name, str(rep_screen_id))
                cached = self._shap_visual_cache.get(cache_key)
                if cached is not None:
                    rep_shap = np.asarray(cached)
                else:
                    try:
                        sh, _, _ = self.shap_explainer.compute(
                            image=rep_img, n_samples=SHAP_MAX_EVALS
                        )
                        rep_shap = sh
                    except Exception as e:
                        print(
                            f"[unimodal_v2] SHAP of the main screen failed: {e}"
                        )

        safe_pkg = package_name.replace(".", "_").replace("/", "_")
        viz_path = os.path.join(output_dir, f"{safe_pkg}_app_explanation.png")
        try:
            self._render_app_figure(
                representative_image_path=rep_image_path,
                aggregated_heatmap=rep_heatmap,
                aggregated_shap=rep_shap,
                app_explanation=app_explanation,
                output_path=viz_path,
                representative_screen_id=rep_screen_id,
            )
            app_explanation["visualization_path"] = viz_path
        except Exception as e:
            print(f"[unimodal_v2] Failed to render the app figure: {e}")

        return app_explanation

    # =========================================================================
    # FIGURES (multimodal layout minus the text rows)
    #
    # Panel titles and metadata-bar labels below are deliberately in Portuguese:
    # they are rendered into the PNGs, so translating them would make
    # regenerated figures disagree with the ones already in use. Comments stay
    # in English; figure content does not.
    # =========================================================================
    def _render_screen_figure(
        self,
        image_path: str,
        heatmap: np.ndarray,
        shap_visual: np.ndarray | None,
        explanation: dict,
        output_path: str,
    ) -> None:
        """Per-screen figure, mirroring ``_render_per_review_figure`` of the
        multimodal module **minus** the textual analysis row (review, Bahdanau,
        textual SHAP). Titles and metadata bar are kept identical.

        Layout::

            [original screenshot | Grad-CAM (overlay) | visual SHAP]
            [metadata bar: screen | app | category | prediction | true | result]
        """
        original = load_original_image(image_path)
        if original is None:
            return

        # Reuse the base-class ``overlay_heatmap``, the same one
        # ``generate_screen_visualization`` uses. It handles dtype and resize
        # internally, which avoids the divergences that produced an overlay
        # with no original image behind it.
        gradcam_overlay = self.gradcam.overlay_heatmap(original, heatmap, alpha=0.5)
        shap_overlay = None
        if shap_visual is not None:
            shap_overlay = self.gradcam.overlay_heatmap(
                original, shap_visual, alpha=0.5
            )

        # Layout: 2 rows (visual + metadata), 3 columns.
        fig = plt.figure(figsize=(16, 7))
        gs = GridSpec(
            2, 3,
            height_ratios=[3.5, 0.4],
            width_ratios=[1, 1, 1],
            hspace=0.15, wspace=0.05,
        )

        # Row 0 col 0: original screenshot
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(original)
        ax_orig.set_title("Screenshot original", fontsize=11, fontweight="bold")
        ax_orig.axis("off")

        # Row 0 col 1: Grad-CAM overlay
        ax_cam = fig.add_subplot(gs[0, 1])
        ax_cam.imshow(gradcam_overlay)
        ax_cam.set_title(
            "Grad-CAM (overlay na resolução original)",
            fontsize=11, fontweight="bold",
        )
        ax_cam.axis("off")

        # Row 0 col 2: SHAP Visual
        ax_shap = fig.add_subplot(gs[0, 2])
        if shap_overlay is not None:
            ax_shap.imshow(shap_overlay)
        else:
            ax_shap.text(
                0.5, 0.5,
                "SHAP Visual\n(desabilitado nesta run)",
                ha="center", va="center",
                fontsize=11, color="gray",
                transform=ax_shap.transAxes,
            )
        ax_shap.set_title(
            "SHAP — Importância Visual", fontsize=11, fontweight="bold"
        )
        ax_shap.axis("off")

        # Row 1: metadata bar (same format as the multimodal per-review one).
        ax_meta = fig.add_subplot(gs[1, :])
        is_correct = bool(explanation.get("is_correct", False))
        pred_class = str(explanation.get("predicted_class", "?")).upper()
        true_class = str(explanation.get("true_class", "?")).upper()
        conf = float(explanation.get("confidence", 0.0)) * 100
        result_symbol = "✓" if is_correct else "✗"
        meta_text = (
            f"Screen: {explanation.get('screen_id', '-')}   |   "
            f"App: {explanation.get('app_name') or explanation.get('package_name', '-')}   |   "
            f"Categoria: {explanation.get('category', '-')}   |   "
            f"Predição: {pred_class} (conf. {conf:.1f}%)   |   "
            f"Classe real: {true_class}   |   "
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

    def _render_app_figure(
        self,
        representative_image_path: str,
        aggregated_heatmap: np.ndarray | None,
        aggregated_shap: np.ndarray | None,
        app_explanation: dict,
        output_path: str,
        representative_screen_id: str | None = None,
    ) -> None:
        """App-level figure, mirroring ``_render_app_figure`` of the multimodal
        module **minus** the textual analysis row (Bahdanau, textual SHAP) and
        the reviews row. Titles and metadata bar identical to the multimodal.

        Layout::

            [main screenshot | Grad-CAM of the main screen | visual SHAP]
            [metadata bar: app | category | screens | prediction | true | result]
        """
        original = load_original_image(representative_image_path)
        if original is None:
            return

        # Same ``overlay_heatmap`` used by ``_render_screen_figure``, for
        # consistency and dtype robustness.
        gradcam_overlay = None
        if aggregated_heatmap is not None:
            gradcam_overlay = self.gradcam.overlay_heatmap(
                original, aggregated_heatmap, alpha=0.5
            )
        shap_overlay = None
        if aggregated_shap is not None:
            shap_overlay = self.gradcam.overlay_heatmap(
                original, aggregated_shap, alpha=0.5
            )

        # Layout: 2 rows (visual + metadata), 3 columns.
        fig = plt.figure(figsize=(16, 7))
        gs = GridSpec(
            2, 3,
            height_ratios=[3.5, 0.4],
            width_ratios=[1, 1, 1],
            hspace=0.15, wspace=0.05,
        )

        # Row 0 col 0: main screenshot
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(original)
        screen_tag = (
            f" (tela {representative_screen_id})"
            if representative_screen_id
            else ""
        )
        ax_orig.set_title(
            f"Screenshot principal{screen_tag}",
            fontsize=10, fontweight="bold",
        )
        ax_orig.axis("off")

        # Row 0 col 1: Grad-CAM of the main screen
        ax_cam = fig.add_subplot(gs[0, 1])
        if gradcam_overlay is not None:
            ax_cam.imshow(gradcam_overlay)
        else:
            ax_cam.imshow(original)
        ax_cam.set_title(
            "Grad-CAM (tela principal)",
            fontsize=11, fontweight="bold",
        )
        ax_cam.axis("off")

        # Row 0 col 2: visual SHAP of the main screen
        ax_shap = fig.add_subplot(gs[0, 2])
        if shap_overlay is not None:
            ax_shap.imshow(shap_overlay)
        else:
            ax_shap.text(
                0.5, 0.5,
                "SHAP Visual\n(não computado)",
                ha="center", va="center",
                fontsize=11, color="gray",
                transform=ax_shap.transAxes,
            )
        ax_shap.set_title(
            "SHAP — Importância Visual (tela principal)",
            fontsize=11, fontweight="bold",
        )
        ax_shap.axis("off")

        # Row 1: metadata bar (same format as the multimodal app one).
        ax_meta = fig.add_subplot(gs[1, :])
        is_correct = bool(app_explanation.get("is_correct", False))
        pred = str(app_explanation.get("app_prediction", "?")).upper()
        conf = float(app_explanation.get("app_confidence", 0.0)) * 100
        true_cls = str(app_explanation.get("true_class", "?")).upper()
        result_symbol = "✓" if is_correct else "✗"
        # app_name and category come from the first screen; all screens share them.
        app_name = app_explanation.get("app_name") or app_explanation.get(
            "package_name", "-"
        )
        category = app_explanation.get("category", "-")
        meta_text = (
            f"App: {app_name}   |   "
            f"Categoria: {category}   |   "
            f"Telas: {app_explanation.get('num_screens', '?')}   |   "
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


# =============================================================================
# OUTPUT MANAGER V2 (per-app folders + components_ranked.csv)
# =============================================================================
class ExplanationOutputManagerV2(ExplanationOutputManager):
    """Lay the output out as ``apps/<pkg>/`` and add ``components_ranked.csv``.

    ``self.screen_dir`` / ``self.app_dir`` are kept as a fallback and are not
    created on disk by default; the writers use ``screen_dir_for(pkg)`` and
    ``app_dir_for(pkg)`` to keep each app isolated.
    """

    def __init__(self, output_dir: str):
        # super().__init__ is skipped so the flat folders screen_explanations/ and
        # app_explanations/ are not created at the root, where the per-app layout
        # would leave them empty. The essential attributes are replicated here.
        self.output_dir = output_dir
        self.screen_dir = os.path.join(output_dir, "screen_explanations")
        self.app_dir = os.path.join(output_dir, "app_explanations")
        self.apps_dir = os.path.join(output_dir, "apps")
        os.makedirs(self.apps_dir, exist_ok=True)
        self.screen_explanations = []
        self.app_explanations = []
        # Registered so UnimodalExplanationV2 can redirect its output_dir.
        UnimodalExplanationV2._active_output_manager = self

    # -------- per-app path resolvers (same as the multimodal module) --------
    def _safe_pkg(self, package_name: str) -> str:
        return str(package_name).replace(".", "_").replace("/", "_")

    def screen_dir_for(self, package_name: str) -> str:
        d = os.path.join(
            self.apps_dir, self._safe_pkg(package_name), "screen_explanations"
        )
        os.makedirs(d, exist_ok=True)
        return d

    def app_dir_for(self, package_name: str) -> str:
        d = os.path.join(self.apps_dir, self._safe_pkg(package_name))
        os.makedirs(d, exist_ok=True)
        return d

    def save_all(self) -> None:
        # JSON global (screen + app)
        with open(
            os.path.join(self.output_dir, "screen_explanations.json"),
            "w", encoding="utf-8",
        ) as f:
            json.dump(
                self.screen_explanations, f, indent=2, ensure_ascii=False,
                default=str,
            )
        with open(
            os.path.join(self.output_dir, "app_explanations.json"),
            "w", encoding="utf-8",
        ) as f:
            json.dump(
                self.app_explanations, f, indent=2, ensure_ascii=False,
                default=str,
            )

        # Group the screens per app and write the per-app CSVs.
        screens_by_app: dict[str, list[dict]] = {}
        for e in self.screen_explanations:
            screens_by_app.setdefault(e["package_name"], []).append(e)

        for pkg, screens in screens_by_app.items():
            app_dir = self.app_dir_for(pkg)

            # screen_explanations.csv, per app
            rows = []
            for e in screens:
                obi = e.get("obi") or {}
                obi_shap = e.get("obi_shap") or {}
                top_list = e.get("top_components_ablated") or []
                top_list_shap = e.get("top_components_ablated_shap") or []
                top_classes = (
                    ", ".join(c["component"] for c in top_list)
                    if top_list
                    else None
                )
                top_classes_shap = (
                    ", ".join(c["component"] for c in top_list_shap)
                    if top_list_shap
                    else None
                )
                rows.append({
                    "package_name": e["package_name"],
                    "screen_id": e["screen_id"],
                    "prediction": e["prediction"],
                    "predicted_class": e["predicted_class"],
                    "true_label": e["true_label"],
                    "is_correct": e["is_correct"],
                    "confidence": e["confidence"],
                    "heatmap_mean": e["heatmap_stats"]["mean"],
                    "heatmap_max": e["heatmap_stats"]["max"],
                    "heatmap_std": e["heatmap_stats"]["std"],
                    "hot_area_pct": e["heatmap_stats"]["hot_area_pct"],
                    # OBI(Grad-CAM)
                    "p_orig": obi.get("p_orig"),
                    "p_img": obi.get("p_img"),
                    "delta_p_img": obi.get("delta_p_img"),
                    "top_components": top_classes,
                    # OBI(SHAP): parallel ranking from the visual SHAP heatmap.
                    "p_img_shap": obi_shap.get("p_img"),
                    "delta_p_img_shap": obi_shap.get("delta_p_img"),
                    "top_components_shap": top_classes_shap,
                })
            pd.DataFrame(rows).to_csv(
                os.path.join(app_dir, "screen_explanations.csv"),
                index=False,
            )

            # components_ranked.csv per app, one row per (screen, component).
            # The ``score_pct`` column is the fraction of the screen's total
            # saliency each component concentrates.
            comp_rows = []
            for e in screens:
                comps = e.get("components_full_ranked", []) or []
                if not comps:
                    continue
                total_screen_saliency = sum(
                    float(c.get("score_total", 0.0)) for c in comps
                ) + 1e-12
                for c in comps:
                    comp_rows.append({
                        "package_name": e["package_name"],
                        "screen_id": e["screen_id"],
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
                    os.path.join(app_dir, "components_ranked.csv"),
                    index=False,
                )

            # components_ranked_shap.csv (per app): the parallel ranking from
            # visual SHAP. Same schema, enabling Grad-CAM vs SHAP triangulation.
            comp_shap_rows = []
            for e in screens:
                comps_shap = e.get("components_full_ranked_shap", []) or []
                if not comps_shap:
                    continue
                total_screen_saliency_shap = sum(
                    float(c.get("score_total", 0.0)) for c in comps_shap
                ) + 1e-12
                for c in comps_shap:
                    comp_shap_rows.append({
                        "package_name": e["package_name"],
                        "screen_id": e["screen_id"],
                        "rank": c["rank"],
                        "component": c["component"],
                        "score_total": c["score_total"],
                        "score_mean": c["score_mean"],
                        "score_pct": round(
                            100.0 * float(c["score_total"])
                            / total_screen_saliency_shap,
                            2,
                        ),
                        "bounds_image": str(c["bounds_image"]),
                    })
            if comp_shap_rows:
                pd.DataFrame(comp_shap_rows).to_csv(
                    os.path.join(app_dir, "components_ranked_shap.csv"),
                    index=False,
                )

        # Global CSV with one summary row per app
        if self.app_explanations:
            app_df = pd.DataFrame([
                {
                    "package_name": e["package_name"],
                    "num_screens": e["num_screens"],
                    "app_score": e["app_score"],
                    "app_prediction": e["app_prediction"],
                    "true_label": e["true_label"],
                    "is_correct": e["is_correct"],
                    "app_confidence": e["app_confidence"],
                    "prediction_variance": e["prediction_variance"],
                    "avg_hot_area_pct": e.get("avg_hot_area_pct", 0),
                }
                for e in self.app_explanations
            ])
            app_df.to_csv(
                os.path.join(self.output_dir, "app_explanations.csv"),
                index=False,
            )

        print(f"\nExplanations saved in: {self.output_dir}/")
        print(f"  - {len(self.screen_explanations)} screen explanations")
        print(f"  - {len(self.app_explanations)} app explanations")
        print(f"  - global JSONs: screen_explanations.json, app_explanations.json")
        print(f"  - per-app CSVs in apps/<pkg>/")


# =============================================================================
# ENTRY POINT (called from multimodal.cli)
# =============================================================================
def explain(config: dict) -> None:
    """Unimodal explainability entry point: injects this module's classes into
    the base flow by monkey-patching.

    The interactive modes of the base module are preserved (quick test, screen,
    app, all samples, errors only); what changes is:
    - ``ImageExplanation`` -> ``UnimodalExplanationV2`` (OBI + SHAP + figures)
    - ``ExplanationOutputManager`` -> ``ExplanationOutputManagerV2`` (per-app)
    - ``RESULTS_EXPLANATIONS_UNIMODAL`` -> ``RESULTS_EXPLANATIONS_UNIMODAL_V2``
    """
    import multimodal.explainability.unimodal as v1

    global OBI_TOP_K_COMPONENTS, SHAP_ENABLED, SHAP_MAX_EVALS, SHAP_N_SEGMENTS
    global _MULTIMODAL_REP_CACHE
    OBI_TOP_K_COMPONENTS = int(
        config.get("obi_top_k_components", OBI_TOP_K_COMPONENTS)
    )
    SHAP_ENABLED = bool(config.get("shap_enabled", SHAP_ENABLED))
    SHAP_MAX_EVALS = int(config.get("shap_max_evals", SHAP_MAX_EVALS))
    SHAP_N_SEGMENTS = int(config.get("shap_n_segments", SHAP_N_SEGMENTS))
    # Reset the representative-screen cache; it is read once on the first
    # explain_app, from whatever APPS_FROM_MULTIMODAL_RUN the base explain() set.
    _MULTIMODAL_REP_CACHE = None

    saved_explanation_cls = v1.ImageExplanation
    saved_output_cls = v1.ExplanationOutputManager
    saved_results_path = v1.RESULTS_EXPLANATIONS_UNIMODAL

    v1.ImageExplanation = UnimodalExplanationV2
    v1.ExplanationOutputManager = ExplanationOutputManagerV2
    v1.RESULTS_EXPLANATIONS_UNIMODAL = RESULTS_EXPLANATIONS_UNIMODAL_V2

    print("=" * 70)
    print("UNIMODAL EXPLAINABILITY MODULE (OBI + visual SHAP)")
    print(
        f"  OBI top-K: {OBI_TOP_K_COMPONENTS}  |  SHAP enabled: {SHAP_ENABLED}  "
        f"|  SHAP max_evals: {SHAP_MAX_EVALS}"
    )
    print("=" * 70)

    try:
        v1.explain(config)
    finally:
        v1.ImageExplanation = saved_explanation_cls
        v1.ExplanationOutputManager = saved_output_cls
        v1.RESULTS_EXPLANATIONS_UNIMODAL = saved_results_path
        UnimodalExplanationV2._active_output_manager = None
