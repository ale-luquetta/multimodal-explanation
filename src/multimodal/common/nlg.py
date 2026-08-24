"""Template-based NLG for multimodal classification explanations.

Inspired by xui-software/xui/nlg.py (Leiva et al.). Two key adaptations:
  - Templates describe the model's *decision* (good/bad + evidence), not the UI itself.
  - Variation is deterministic: the same explanation always produces the same text.
    A hash of a stable seed key (e.g., screen_id) selects each option, so the wording
    varies across screens but is reproducible across runs.

Three levels per target:
  - caption  — one short sentence
  - simple   — 2-3 sentences, mentions the dominant UI component
  - detailed — paragraph with components, review words, app metadata
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

# Rico canonical screen size — matches semantic_saliency.py for position descriptions.
RICO_SCREEN_WIDTH = 1440
RICO_SCREEN_HEIGHT = 2560


def _pick(options: list[str], seed_key: str) -> str:
    """Deterministic choice: the same seed_key always picks the same option."""
    if not options:
        return ""
    # Stable hash across processes: hash() on str is randomised by
    # PYTHONHASHSEED, which would break the reproducibility this module promises.
    digest = hashlib.md5(seed_key.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(options)
    return options[idx]


def _confidence_phrase(confidence: float, seed_key: str) -> str:
    if confidence >= 0.85:
        return _pick(["with high confidence", "confidently", "with strong evidence"], seed_key)
    if confidence >= 0.65:
        return _pick(["with moderate confidence", "fairly confidently"], seed_key)
    return _pick(["with low confidence", "tentatively", "with weak evidence"], seed_key)


def _verb_classified(seed_key: str) -> str:
    return _pick(["was classified as", "was labeled as", "was predicted as"], seed_key)


def _position_desc(bounds: list[int] | tuple[int, int, int, int],
                   image_shape: tuple[int, int] | None = None) -> str:
    """Return a coarse position label for a bounding box (image coords)."""
    if not bounds or len(bounds) != 4:
        return ""
    x0, y0, x1, y1 = bounds
    h, w = image_shape if image_shape else (RICO_SCREEN_HEIGHT, RICO_SCREEN_WIDTH)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    v = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "center")
    horiz = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")

    if v == "center" and horiz == "center":
        return "in the center"
    if v == "center":
        return f"on the {horiz}"
    if horiz == "center":
        return f"at the {v}"
    return f"at the {v}-{horiz}"


def _area_desc(bounds: list[int] | tuple[int, int, int, int],
               image_shape: tuple[int, int] | None = None) -> str:
    if not bounds or len(bounds) != 4:
        return ""
    x0, y0, x1, y1 = bounds
    w_b, h_b = x1 - x0, y1 - y0
    h, w = image_shape if image_shape else (RICO_SCREEN_HEIGHT, RICO_SCREEN_WIDTH)
    area_ratio = (w_b * h_b) / max(1, h * w)
    if area_ratio > 0.30:
        return "large"
    if area_ratio < 0.05:
        return "small"
    return ""


def _format_words(top_words: list[Any], k: int = 3) -> str:
    """Format the top-k words as 'a', 'b', 'c'. Accepts list of strings or (word, score) tuples."""
    if not top_words:
        return ""
    items = []
    for w in top_words[:k]:
        if isinstance(w, (list, tuple)):
            items.append(str(w[0]))
        else:
            items.append(str(w))
    if len(items) == 1:
        return f"'{items[0]}'"
    if len(items) == 2:
        return f"'{items[0]}' and '{items[1]}'"
    return ", ".join(f"'{x}'" for x in items[:-1]) + f", and '{items[-1]}'"


# =============================================================================
# Screen-level generators
# =============================================================================

def gen_screen_caption(expl: dict) -> str:
    seed = str(expl.get("screen_id", ""))
    pred = expl.get("predicted_class", "unknown")
    conf = float(expl.get("confidence", 0.0))
    pct = int(round(conf * 100))
    verb = _verb_classified(seed)
    return f"This screen {verb} {pred} with {pct}% confidence."


def gen_screen_simple(expl: dict) -> str:
    seed = str(expl.get("screen_id", ""))
    caption = gen_screen_caption(expl)

    components = expl.get("top_components") or []
    if not components:
        return caption + " " + _pick(
            ["No salient UI components were identified for this decision.",
             "The model's attention was diffuse across the screen."],
            seed,
        )

    main = components[0]
    pos = _position_desc(main.get("bounds_image", []))
    area = _area_desc(main.get("bounds_image", []))
    pos_phrase = f" {pos}" if pos else ""
    area_phrase = f" {area}" if area else ""
    focus_verb = _pick(
        ["focused primarily on", "directed most attention to", "concentrated on"], seed
    )
    return f"{caption} The model {focus_verb} a{area_phrase} {main['component']}{pos_phrase}."


def gen_screen_detailed(expl: dict) -> str:
    seed = str(expl.get("screen_id", ""))
    pred = expl.get("predicted_class", "unknown")
    conf = float(expl.get("confidence", 0.0))
    pct = int(round(conf * 100))
    app_name = expl.get("app_name") or expl.get("package_name", "an app")
    category = expl.get("category", "Unknown")
    components = expl.get("top_components") or []
    top_words = expl.get("top_words") or []
    conf_phrase = _confidence_phrase(conf, seed)

    parts = [
        f"This screen of {app_name} (category: {category}) {_verb_classified(seed)} "
        f"{pred} with {pct}% confidence ({conf_phrase})."
    ]

    if components:
        comp_descriptions = []
        for c in components[:3]:
            pos = _position_desc(c.get("bounds_image", []))
            area = _area_desc(c.get("bounds_image", []))
            qualifiers = " ".join(p for p in (area, c["component"]) if p)
            pos_phrase = f" {pos}" if pos else ""
            comp_descriptions.append(f"a {qualifiers}{pos_phrase}")
        if len(comp_descriptions) == 1:
            comp_text = comp_descriptions[0]
        else:
            comp_text = ", ".join(comp_descriptions[:-1]) + f", and {comp_descriptions[-1]}"
        verb = _pick(["focused on", "attended to", "gave priority to"], seed)
        parts.append(f"The model {verb} {comp_text}.")

    if top_words:
        words_text = _format_words(top_words, k=3)
        if words_text:
            parts.append(f"From the user reviews, the most influential words were {words_text}.")

    closing = _pick(
        [
            "These visual and textual cues together support the classification.",
            "Both modalities contributed to the final decision.",
            "The convergence of image and text evidence justifies the prediction.",
        ],
        seed,
    )
    if components and top_words:
        parts.append(closing)

    return " ".join(parts)


# =============================================================================
# App-level generators
# =============================================================================

def _app_category(
    app_expl: dict, screen_explanations: list[dict] | None
) -> str:
    """App category, looked up first on the screens and then on the app."""
    for s in screen_explanations or []:
        cat = s.get("category")
        if cat:
            return str(cat)
    return str(app_expl.get("category") or "")


def _aggregate_top_components(screen_explanations: list[dict]) -> list[tuple[str, int]]:
    """Count how often each component appears in the top-3 across screens."""
    counter: Counter[str] = Counter()
    for s in screen_explanations:
        for c in s.get("top_components") or []:
            counter[c["component"]] += 1
    return counter.most_common()


def gen_app_caption(app_expl: dict) -> str:
    seed = str(app_expl.get("package_name", ""))
    pred = app_expl.get("app_prediction", "unknown")
    conf = float(app_expl.get("app_confidence", 0.0))
    pct = int(round(conf * 100))
    n = app_expl.get("num_screens", 0)
    name = app_expl.get("package_name", "the app")
    verb = _verb_classified(seed)
    return f"{name} {verb} {pred} with average confidence {pct}% across {n} screens."


def gen_app_simple(app_expl: dict, screen_explanations: list[dict] | None = None) -> str:
    seed = str(app_expl.get("package_name", ""))
    caption = gen_app_caption(app_expl)

    dominant = ""
    if screen_explanations:
        ranked = _aggregate_top_components(screen_explanations)
        if ranked:
            dominant = ranked[0][0]

    if not dominant:
        return caption

    verb = _pick(
        ["recurringly attended to", "consistently focused on", "frequently emphasized"], seed
    )
    return f"{caption} Across these screens, the model {verb} {dominant} components."


def gen_app_detailed(app_expl: dict, screen_explanations: list[dict] | None = None) -> str:
    seed = str(app_expl.get("package_name", ""))
    variance = float(app_expl.get("prediction_variance", 0.0))

    # The ``detailed`` level opens by repeating ``simple`` in full (caption +
    # dominant component), so the three levels form an additive progression:
    # each one contains the previous and adds information.
    parts = [gen_app_simple(app_expl, screen_explanations)]

    if variance > 0.05:
        parts.append(
            _pick(
                [
                    f"Per-screen predictions varied substantially (variance {variance:.2f}), "
                    "suggesting the app shows mixed visual or textual signals.",
                    f"The screens disagreed considerably (variance {variance:.2f}), "
                    "indicating heterogeneous evidence across the UI.",
                ],
                seed,
            )
        )
    else:
        parts.append(
            _pick(
                [
                    "Per-screen predictions were stable, indicating consistent evidence across the UI.",
                    "The screens agreed on the prediction, pointing to a coherent signal.",
                ],
                seed,
            )
        )

    if screen_explanations:
        ranked = _aggregate_top_components(screen_explanations)
        if ranked:
            top3 = ranked[:3]
            comp_list = ", ".join(f"{c} ({n}x)" for c, n in top3)
            parts.append(f"The most recurring salient components were: {comp_list}.")

    top_words = app_expl.get("top_influential_words") or []
    if top_words:
        words_text = _format_words(top_words, k=5)
        if words_text:
            parts.append(f"The most influential review words across screens were {words_text}.")

    # Category last: it is context metadata, not evidence for the decision.
    # The category lives on the screens, not on ``app_explanation``, with a
    # fallback to the app in case some pipeline promotes it.
    category = _app_category(app_expl, screen_explanations)
    if category:
        n = app_expl.get("num_screens", 0)
        parts.append(
            f"The app belongs to the {category} category and was analyzed "
            f"across {n} screens."
        )

    return " ".join(parts)


# =============================================================================
# Public API
# =============================================================================

def generate_screen_text(explanation: dict) -> dict[str, str]:
    """Returns {caption, simple, detailed} for a screen explanation."""
    return {
        "caption": _realize(gen_screen_caption(explanation)),
        "simple": _realize(gen_screen_simple(explanation)),
        "detailed": _realize(gen_screen_detailed(explanation)),
    }


def generate_app_text(app_explanation: dict,
                       screen_explanations: list[dict] | None = None) -> dict[str, str]:
    """Returns {caption, simple, detailed} for an app explanation."""
    return {
        "caption": _realize(gen_app_caption(app_explanation)),
        "simple": _realize(gen_app_simple(app_explanation, screen_explanations)),
        "detailed": _realize(gen_app_detailed(app_explanation, screen_explanations)),
    }


# =============================================================================
# Realization (post-processing)
# =============================================================================

_VOWELS = set("aeiouAEIOU")


def _realize(text: str) -> str:
    text = " ".join(text.split())
    text = text.replace(" .", ".").replace(" ,", ",").replace("a (", "a (")
    text = _fix_indefinite_articles(text)
    text = _capitalize_sentences(text)
    if not text.endswith("."):
        text += "."
    return text


def _fix_indefinite_articles(text: str) -> str:
    """Convert 'a X' → 'an X' when X starts with a vowel sound."""
    words = text.split()
    for i in range(len(words) - 1):
        cur = words[i]
        nxt = words[i + 1].lstrip("'\"(")
        if cur in ("a", "A") and nxt and nxt[0] in _VOWELS:
            words[i] = "an" if cur == "a" else "An"
    return " ".join(words)


def _capitalize_sentences(text: str) -> str:
    sentences = text.split(". ")
    out = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if s[0].islower() and not _starts_with_identifier(s):
            s = s[0].upper() + s[1:]
        out.append(s)
    return ". ".join(out).strip()


def _starts_with_identifier(sentence: str) -> bool:
    """True when the sentence starts with an identifier that must not be capitalised.

    Package names (``com.hp.pregnancy.lite``) open several NLG sentences and are
    case-sensitive: capitalising them would produce ``Com.hp.pregnancy.lite``,
    which is not the app's real identifier.
    """
    first_token = sentence.split(" ", 1)[0]
    return "." in first_token.rstrip(".")
