"""
WildGuard - policy/policy_engine.py

Composes the final 0-100 risk score from THREE explicitly separate inputs
(per spec section 19 - never conflate the raw model probability with the
final score):

    raw_model_probability        (XGBoost, statistical pattern layer)
          +
    deterministic_signal_score   (metadata/OCR/context/stego evidence,
                                   independent of the ML model - see
                                   section 43: the dashboard must never
                                   depend exclusively on XGBoost)
          +
    species_sensitivity          (risk multiplier, kept separate from
                                   "proof of leakage" per section 10)
          ↓
    final_risk_score (0-100)
          ↓
    policy decision (SAFE / REVIEW / SANITIZE / QUARANTINE)

Weights and thresholds are loaded from config/policy.json - NOT hardcoded
here - so they can be tuned without touching code.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any

from features.feature_pipeline import ExtractionBundle


@dataclass
class CategoryVerdict:
    category: str          # "metadata" | "ocr" | "context" | "species" | "stego"
    level: str              # "NONE" | "LOW" | "MEDIUM" | "HIGH"
    score: float             # 0-1 deterministic sub-score for this category
    evidence: List[str] = field(default_factory=list)   # human-readable flagged factors


@dataclass
class PolicyResult:
    model_probability: float
    deterministic_signal_score: float
    species_sensitivity: float
    final_risk_score: float
    decision: str
    category_verdicts: List[CategoryVerdict]
    flagged_factors: List[str]


def load_policy_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _level_from_thresholds(score: float, thresholds: Dict[str, float]) -> str:
    if score >= thresholds["high"]:
        return "HIGH"
    if score >= thresholds["medium"]:
        return "MEDIUM"
    if score > 0:
        return "LOW"
    return "NONE"


def _human_readable_factors(bundle: ExtractionBundle) -> List[str]:
    """
    Section 22/43: full, plain-English sentences - never raw feature-index
    dumps. Each sentence is only emitted when its underlying deterministic
    evidence is actually present (never fabricated).
    """
    factors = []
    m, o, c, s, sp = bundle.metadata, bundle.ocr, bundle.context, bundle.stego, bundle.species

    if m.has_gps:
        factors.append("GPS coordinates were detected in the image's EXIF metadata.")
    if m.has_gps_altitude or m.has_gps_timestamp:
        factors.append("GPS altitude/timestamp fields were found alongside the coordinates, "
                        "adding further precision to the leaked location.")
    if m.has_camera_make or m.has_camera_model:
        factors.append("A camera/device make or model identifier was found in the metadata.")
    if m.has_serial_number:
        factors.append("A camera/lens serial number was found in the metadata - this can "
                        "uniquely fingerprint the specific device used.")
    if m.has_owner_name:
        factors.append("An owner/artist name was found embedded in the metadata.")
    if m.description_contains_location or m.keywords_contain_location:
        factors.append("The image description or keyword metadata contains location-specific wording.")
    if m.has_thumbnail:
        factors.append("An embedded thumbnail was found, which may retain data stripped from the main image.")

    for hit in o.hits:
        if hit.label == "gps_coordinate_stamp":
            factors.append(f'OCR detected a GPS coordinate stamped directly onto the image pixels: "{hit.text}".')
        elif hit.label == "camera_trap_id":
            factors.append(f'OCR detected a camera-trap/site identifier: "{hit.text}".')
        elif hit.label == "zone_marker":
            factors.append(f'OCR detected a location marker printed on the image: "{hit.text}".')

    if c.filename_has_location_keyword:
        factors.append("The filename contains a sensitive location or reserve-zone keyword.")
    if c.filename_has_distance_pattern:
        factors.append('The filename contains a distance pattern (e.g. "5km_from_checkpoint").')
    if c.caption_has_location_keyword or c.caption_has_distance_pattern:
        factors.append("The caption/description text references a specific location or distance from a landmark.")

    if s.has_eof_trailing_bytes:
        factors.append(f"The file contains {s.trailing_byte_count} bytes appended after the "
                        f"legitimate end-of-image marker - a possible covert data channel.")
    if s.lsb_variance_score > 0.6:
        factors.append("The image's least-significant-bit distribution deviates from a natural "
                        "photo's statistical baseline, suggesting possible steganographic content.")
    if s.suspicious_png_chunk_count > 0:
        factors.append("Non-standard PNG chunk types were found in the file structure.")

    if sp.sensitivity >= 0.7:
        factors.append(f"The detected species ({sp.species}) has a high location-sensitivity score "
                        f"({sp.sensitivity:.2f}) - poaching/trafficking risk is elevated if location "
                        f"is also exposed.")

    return factors


def evaluate_policy(
    bundle: ExtractionBundle, model_probability: float, policy_config: Dict[str, Any]
) -> PolicyResult:
    thresholds = policy_config["thresholds"]
    weights = policy_config["weights"]
    cat_thresholds_default = {
        "metadata": {"high": 0.5, "medium": 0.2},
        "ocr": {"high": 1, "medium": 0},
        "context": {"high": 0.5, "medium": 0.25},
        "species": {"high": 0.7, "medium": 0.4},
        "stego": {"high": 0.65, "medium": 0.35},
    }
    cat_thresholds = policy_config.get("category_thresholds", cat_thresholds_default)

    m, o, c, s, sp = bundle.metadata, bundle.ocr, bundle.context, bundle.stego, bundle.species

    # --- Category-level deterministic sub-scores (independent of XGBoost) ---
    metadata_flags = [m.has_gps, m.has_gps_altitude, m.has_gps_timestamp, m.has_camera_make,
                       m.has_camera_model, m.has_serial_number, m.has_owner_name,
                       m.has_iptc_location, m.has_iptc_description, m.has_iptc_keywords]
    metadata_score = sum(1 for f in metadata_flags if f) / len(metadata_flags)

    ocr_score = 1.0 if o.hits else 0.0
    context_score = max(c.filename_leak_score, c.caption_leak_score)
    species_score = sp.sensitivity
    stego_score = max(s.lsb_variance_score, min(s.trailing_byte_count_norm * 1.2, 1.0))

    category_verdicts = [
        CategoryVerdict("metadata", _level_from_thresholds(metadata_score, cat_thresholds["metadata"]), metadata_score),
        CategoryVerdict("ocr", "HIGH" if o.hits else "NONE", ocr_score),
        CategoryVerdict("context", _level_from_thresholds(context_score, cat_thresholds["context"]), context_score),
        CategoryVerdict("species", _level_from_thresholds(species_score, cat_thresholds["species"]), species_score),
        CategoryVerdict("stego", _level_from_thresholds(stego_score, cat_thresholds["stego"]), stego_score),
    ]

    # Deterministic signal score = max of the (non-species) category scores,
    # so ANY single strong deterministic finding (e.g. raw GPS) drives risk
    # up even if the ML model alone would score it low (spec section 43).
    deterministic_signal_score = max(metadata_score, ocr_score, context_score, stego_score)

    final_fraction = (
        weights["model_probability"] * model_probability
        + weights["deterministic_signals"] * deterministic_signal_score
        + weights["species_sensitivity"] * species_score
    )
    final_risk_score = max(0.0, min(100.0, final_fraction * 100.0))

    if final_risk_score >= thresholds["quarantine_min"]:
        decision = "QUARANTINE"
    elif final_risk_score > thresholds["review_max"]:
        decision = "SANITIZE"
    elif final_risk_score >= thresholds["safe_max"]:
        decision = "REVIEW"
    else:
        decision = "SAFE"

    # Deterministic override: if strong direct evidence exists (raw GPS
    # found, or an OCR-matched GPS/camera-trap stamp), never allow the
    # policy to settle below REVIEW even if the blended score is low -
    # spec section 43's explicit requirement.
    strong_direct_evidence = m.has_gps or o.gps_regex_match or o.camtrap_id_match
    if strong_direct_evidence and decision == "SAFE":
        decision = "REVIEW"
        final_risk_score = max(final_risk_score, thresholds["safe_max"])

    flagged_factors = _human_readable_factors(bundle)

    return PolicyResult(
        model_probability=model_probability,
        deterministic_signal_score=deterministic_signal_score,
        species_sensitivity=species_score,
        final_risk_score=round(final_risk_score, 2),
        decision=decision,
        category_verdicts=category_verdicts,
        flagged_factors=flagged_factors,
    )
