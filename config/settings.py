"""
WildGuard - config/settings.py

Single source of truth for every path, seed, ratio, and threshold used
across dataset preparation, training, and the dashboard. Nothing in the
rest of the codebase hardcodes a filesystem path - everything reads from
here (or from an env var override), so the project is reproducible on any
machine.

The Windows path given in the spec is wired in as the DEFAULT source
location for THIS machine only, via an environment variable fallback -
change WILDGUARD_ANIMALS_ROOT (or edit the default below) to point at the
actual dataset location on your machine.
"""

import os

# --------------------------------------------------------------------------
# Project root (this file lives in <root>/config/settings.py)
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Source dataset location (configurable; NOT hardcoded through the codebase)
# --------------------------------------------------------------------------
# On the machine this spec was written for, the real path is:
#   C:\Users\Bhavana\Downloads\wildguard_project\wildguard_project\archive\animals\animals
# That is wired in here as the default, but every script accepts
# --animals-root to override it, and it is read once, in one place.
DEFAULT_ANIMALS_ROOT_WINDOWS = (
    r"C:\Users\Bhavana\Downloads\wildguard_project\wildguard_project\archive\animals\animals"
)
# Cross-platform candidates tried (in order) when no explicit path is given
# and the Windows path above doesn't exist on this OS (e.g. this sandbox,
# or a Linux/Mac dev machine using a locally-copied dataset).
ANIMALS_ROOT_CANDIDATES = [
    os.environ.get("WILDGUARD_ANIMALS_ROOT", ""),
    DEFAULT_ANIMALS_ROOT_WINDOWS,
    os.path.join(PROJECT_ROOT, "archive", "animals", "animals"),
]
MERGED_DIRNAME = "merged"  # excluded from train/test/validate, used only for `mixed`

# --------------------------------------------------------------------------
# Output dataset dump
# --------------------------------------------------------------------------
DATASET_DUMP_ROOT = os.path.join(PROJECT_ROOT, "data", "all")
SPLIT_DIRS = {
    "train": os.path.join(DATASET_DUMP_ROOT, "train"),
    "validate": os.path.join(DATASET_DUMP_ROOT, "validate"),
    "test": os.path.join(DATASET_DUMP_ROOT, "test"),
    "mixed": os.path.join(DATASET_DUMP_ROOT, "mixed"),
}
MANIFEST_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
MANIFEST_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "manifest.json")

# --------------------------------------------------------------------------
# Config / model artifacts
# --------------------------------------------------------------------------
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
SPECIES_SENSITIVITY_PATH = os.path.join(CONFIG_DIR, "species_sensitivity.json")
POLICY_CONFIG_PATH = os.path.join(CONFIG_DIR, "policy.json")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "wildguard_xgb.joblib")
FEATURE_SCHEMA_PATH = os.path.join(MODELS_DIR, "feature_schema.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Split ratios (applied per-species, at the ORIGINAL-image level, before any
# synthetic injection - guarantees no near-duplicate variant of the same
# source image can cross a train/validate/test boundary)
# --------------------------------------------------------------------------
SPLIT_RATIOS = {"train": 0.70, "validate": 0.15, "test": 0.15}

# --------------------------------------------------------------------------
# Synthetic injection distribution (per split, applied independently).
# Must sum to 1.0. "hard_negative_d" = clean image of a high-sensitivity
# species, deliberately unlabelled as a leak - teaches the model that
# species sensitivity alone does not imply metadata/context leakage.
# --------------------------------------------------------------------------
INJECTION_DISTRIBUTION = {
    "clean": 0.50,
    "category_a": 0.12,   # explicit metadata (EXIF/IPTC/XMP/PNG chunks)
    "category_b": 0.08,   # visible OCR text
    "category_c": 0.08,   # filename / caption leak
    "hard_negative_d": 0.07,  # sensitive species, but clean (no leak features)
    "category_e": 0.08,   # steganography / covert channel
    "multi": 0.07,        # 2-3 stacked categories from {a,b,c,e}
}
assert abs(sum(INJECTION_DISTRIBUTION.values()) - 1.0) < 1e-6

HIGH_SENSITIVITY_THRESHOLD = 0.70  # species_sensitivity >= this qualifies for hard_negative_d

# --------------------------------------------------------------------------
# Policy engine thresholds (0-100 scale). Configurable, NOT hardcoded deep
# in the scoring code - see policy/policy_engine.py.
# --------------------------------------------------------------------------
DEFAULT_POLICY_THRESHOLDS = {
    "safe_max": 30,
    "review_max": 70,
    "quarantine_min": 90,
}

# --------------------------------------------------------------------------
# Category-level explanation thresholds (deterministic rule layer - used to
# label a category HIGH/MEDIUM/LOW/NONE independently of the ML model)
# --------------------------------------------------------------------------
CATEGORY_THRESHOLDS = {
    "metadata": {"high": 0.5, "medium": 0.2},     # fraction of category-A fields present
    "ocr": {"high": 1, "medium": 0},               # number of OCR hits (>=1 -> HIGH)
    "context": {"high": 0.5, "medium": 0.25},     # max(filename_leak, caption_leak)
    "species": {"high": 0.7, "medium": 0.4},      # species_sensitivity
    "stego": {"high": 0.65, "medium": 0.35},      # composite stego score
}

for _d in (
    DATASET_DUMP_ROOT, *SPLIT_DIRS.values(), CONFIG_DIR, MODELS_DIR, OUTPUT_DIR, UPLOADS_DIR,
    os.path.join(PROJECT_ROOT, "data"),
):
    os.makedirs(_d, exist_ok=True)


def resolve_animals_root(explicit: str = None) -> str:
    """Resolve the source animals-dataset root, trying candidates in order."""
    candidates = ([explicit] if explicit else []) + ANIMALS_ROOT_CANDIDATES
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "Could not locate the animals dataset. Pass --animals-root explicitly, "
        "or set the WILDGUARD_ANIMALS_ROOT environment variable.\n"
        f"Tried: {[c for c in candidates if c]}"
    )
