"""Interactive numbered-menu for training and explainability."""

import importlib

from multimodal.common.config import load_yaml
from multimodal.common.menu import pick_option, pick_yaml_from
from multimodal.common.paths import CONFIGS_DIR


def _run(module_path: str, attr: str, cfg: dict):
    """Lazy-import ``module_path`` and invoke ``attr(cfg)``.

    Defers TensorFlow / PyTorch / transformers imports until the user actually
    picks an option, keeping CLI boot near ~2 s regardless of how many heavy
    modules are registered in MAIN_OPTIONS.
    """
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)(cfg)


# Each entry: (label, action, default_config_filename).
# When the default YAML exists under CONFIGS_DIR, the picker is skipped.
# An entry whose action is a list of entries opens a submenu instead of running
# anything; its last item is "Back" and returns to the parent menu.
# Labels carry no version marker: one variant of each pipeline is exposed.
# The module names do carry one because each explainability module builds on
# the previous by monkey-patching — see the header of each of them.
EXTRAS_OPTIONS = [
    ("Validate Grad-CAM",
        lambda cfg: _run("multimodal.tools.validate_gradcam", "validate", cfg),
        "validate_gradcam.yaml"),
    ("Generate text embeddings cache",
        lambda cfg: _run("multimodal.tools.generate_embeddings", "generate", cfg),
        "embeddings_generation.yaml"),
]

MAIN_OPTIONS = [
    ("Train unimodal",
        lambda cfg: _run("multimodal.training.unimodal", "train", cfg),
        "unimodal.yaml"),
    ("Train multimodal",
        lambda cfg: _run("multimodal.training.multimodal_v3", "train", cfg),
        "multimodal.yaml"),
    ("Run unimodal explainability",
        lambda cfg: _run("multimodal.explainability.unimodal_v2", "explain", cfg),
        "explainability_unimodal.yaml"),
    ("Run multimodal explainability",
        lambda cfg: _run("multimodal.explainability.multimodal_v3", "explain", cfg),
        "explainability_multimodal.yaml"),
    ("Compare unimodal vs multimodal XAI (ΔP_img)",
        lambda cfg: _run("multimodal.tools.compare_xai", "compare", cfg),
        "compare_xai.yaml"),
    ("UI component patterns per app (OBI rankings)",
        lambda cfg: _run("multimodal.tools.component_patterns", "generate", cfg),
        "component_patterns.yaml"),
    ("Extras", EXTRAS_OPTIONS, None),
]


def _run_menu(title: str, options: list, back_label: str = "Back") -> None:
    """Render `options` in a loop until the user picks the back entry.

    A submenu entry re-enters this function with its own list, so leaving it
    returns to the caller's loop, that is, to the parent menu.
    """
    while True:
        idx = pick_option(title, [label for label, _, _ in options], back_label=back_label)
        if idx is None:
            return

        label, action, default_yaml = options[idx]

        if isinstance(action, list):
            _run_menu(label, action)
            continue

        config_path = CONFIGS_DIR / default_yaml if default_yaml else None
        if config_path is None or not config_path.exists():
            if config_path is not None:
                print(f"Default config '{default_yaml}' not found — falling back to picker.")
            config_path = pick_yaml_from(CONFIGS_DIR, f"Select config for: {label}")
            if config_path is None:
                continue

        config = load_yaml(config_path)
        print(f"\n>>> {label}  (config: {config_path.name})")
        try:
            action(config)
        except NotImplementedError as e:
            print(f"[stub] {e}")
        except KeyboardInterrupt:
            print("\nInterrupted.")


def run() -> None:
    _run_menu("Multimodal Research", MAIN_OPTIONS, back_label="Exit")
    print("Bye.")
