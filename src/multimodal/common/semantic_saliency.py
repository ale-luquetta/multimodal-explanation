"""Cross a saliency heatmap with a Rico view hierarchy to rank UI components.

Based on `xui-software/xui/quantizer.py::locate_fixated_elements` (Leiva et al.).
The heatmap is projected onto each leaf element's bounding box, yielding:
  - score_total: sum of heatmap values inside the bbox (favors large components)
  - score_mean:  score_total / bbox area  (favors concentrated attention)
Results are sorted by score_total descending.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Rico's canonical screen size used in `bounds` normalization.
RICO_SCREEN_WIDTH = 1440
RICO_SCREEN_HEIGHT = 2560


def _iter_leaf_elements(view_hierarchy_path: Path) -> list[dict[str, Any]]:
    """Load Rico JSON and return leaf elements (with componentLabel + bounds)."""
    with open(view_hierarchy_path, "r", encoding="utf-8") as f:
        root = json.load(f)
    # Rico v1 wraps in 'activity'; semantic_annotations (v2) does not.
    if "activity" in root:
        root = root["activity"]["root"]

    elements: list[dict[str, Any]] = []

    def _traverse(node: dict[str, Any]) -> None:
        if "children" not in node and "componentLabel" in node and "bounds" in node:
            elements.append(
                {
                    "component": node["componentLabel"],
                    "bounds": tuple(node["bounds"]),
                    "node": node,
                }
            )
        for child in node.get("children", []) or []:
            _traverse(child)

    _traverse(root)
    return elements


def rank_elements_by_saliency(
    heatmap: np.ndarray,
    view_hierarchy_path: str | Path,
    image_shape: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Rank UI components by cumulative saliency inside their bbox.

    Args:
        heatmap: 2D saliency array (values in [0, 1] preferred).
        view_hierarchy_path: path to the Rico JSON.
        image_shape: (H, W) of the target image; heatmap and bboxes are mapped to it.
                     If None, heatmap's own shape is used.

    Returns:
        List of dicts sorted by score_total desc, each with:
          - component (str)
          - score_total (float)
          - score_mean (float)
          - bounds_image (x0, y0, x1, y1) — in image coords
          - bounds_rico  (x0, y0, x1, y1) — original Rico coords
    """
    elements = _iter_leaf_elements(Path(view_hierarchy_path))
    if not elements:
        return []

    if image_shape is None:
        h, w = heatmap.shape[:2]
    else:
        h, w = image_shape
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    x_scale = w / RICO_SCREEN_WIDTH
    y_scale = h / RICO_SCREEN_HEIGHT

    results: list[dict[str, Any]] = []
    for el in elements:
        x0, y0, x1, y1 = el["bounds"]
        x0s = max(0, int(x0 * x_scale))
        y0s = max(0, int(y0 * y_scale))
        x1s = min(w, int(x1 * x_scale))
        y1s = min(h, int(y1 * y_scale))
        if x1s <= x0s or y1s <= y0s:
            continue
        region = heatmap[y0s:y1s, x0s:x1s]
        area = int((x1s - x0s) * (y1s - y0s))
        score_total = float(region.sum())
        results.append(
            {
                "component": el["component"],
                "score_total": score_total,
                "score_mean": score_total / area if area > 0 else 0.0,
                "bounds_image": (x0s, y0s, x1s, y1s),
                "bounds_rico": el["bounds"],
            }
        )

    results.sort(key=lambda r: r["score_total"], reverse=True)
    return results
