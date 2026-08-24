"""Canonical project paths. Import from here instead of hardcoding strings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIGS_DIR = PROJECT_ROOT / "configs"

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RICO_DIR = DATASETS_DIR / "rico"
RICO_SCREENSHOTS_DIR = RICO_DIR / "screenshots"
RICO_HIERARCHIES_DIR = RICO_DIR / "hierarchies"
DATASETS_REVIEWS_DIR = DATASETS_DIR / "reviews"
CONSOLIDATED_SENTIMENT_FILE = DATASETS_REVIEWS_DIR / "consolidated_sentiment_analysis.csv"
REVIEWS_DIR = DATA_DIR / "reviews"
REVIEWS_PROCESSED_DIR = DATA_DIR / "reviews_processed"
EMBEDDINGS_CACHE_DIR = DATA_DIR / "embeddings_cache"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_UNIMODAL_DIR = MODELS_DIR / "unimodal"
MODELS_MULTIMODAL_DIR = MODELS_DIR / "multimodal"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TRAINING_UNIMODAL = RESULTS_DIR / "training" / "unimodal"
RESULTS_TRAINING_MULTIMODAL = RESULTS_DIR / "training" / "multimodal"
RESULTS_TRAINING_MULTIMODAL_V2 = RESULTS_DIR / "training" / "multimodal_v2"
RESULTS_TRAINING_MULTIMODAL_V3 = RESULTS_DIR / "training" / "multimodal_v3"
RESULTS_EXPLANATIONS_UNIMODAL = RESULTS_DIR / "explanations" / "unimodal"
RESULTS_EXPLANATIONS_UNIMODAL_V2 = RESULTS_DIR / "explanations" / "unimodal_v2"
RESULTS_EXPLANATIONS_MULTIMODAL = RESULTS_DIR / "explanations" / "multimodal"
RESULTS_EXPLANATIONS_MULTIMODAL_V2 = RESULTS_DIR / "explanations" / "multimodal_v2"
RESULTS_EXPLANATIONS_MULTIMODAL_V3 = RESULTS_DIR / "explanations" / "multimodal_v3"
RESULTS_EXPLANATIONS_COMPARISON = RESULTS_DIR / "explanations" / "comparison"
RESULTS_EXPLANATIONS_COMPONENTS = RESULTS_DIR / "explanations" / "components"
