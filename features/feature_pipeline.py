"""
WildGuard - features/feature_pipeline.py

Combines Category A/B/C/D/E extractors into a single deterministic,
numeric-only feature vector for XGBoost. Raw OCR text, raw filenames, and
raw captions are NEVER included directly - only the derived scores/flags
from each category module.

This is the SAME code path used at:
  - dataset preparation time (labeling training rows)
  - training time (building the feature matrix)
  - inference time (the dashboard's Analyse page)
so there is zero drift between what the model was trained on and what it
sees at inference.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional

from features.ingest import ingest_image
from features.metadata_features import extract_metadata_features, MetadataResult
from features.ocr_features import extract_ocr_features, OCRResult
from features.context_features import extract_context_features, ContextResult
from features.species_features import extract_species_features, SpeciesResult, load_species_table
from features.stego_features import extract_stego_features, StegoResult

# Canonical, ordered feature schema. Every training row and every inference
# call produces EXACTLY this vector (missing keys are a bug, not a
# silent zero-fill), so there is no schema drift between train and serve.
FEATURE_COLUMNS = [
    # Category A - metadata
    "has_gps", "has_gps_altitude", "has_gps_timestamp", "has_datetime_original",
    "has_camera_make", "has_camera_model", "has_serial_number", "has_owner_name",
    "has_iptc_location", "has_iptc_creator", "has_iptc_description", "has_iptc_keywords",
    "has_thumbnail", "has_png_text_chunks", "has_app_segments",
    "description_contains_location", "keywords_contain_location",
    "gps_field_count_norm", "metadata_block_count_norm", "app_segment_count_norm",
    "png_chunk_count_norm",
    # Category B - OCR (derived numeric only; raw text excluded by design)
    "ocr_text_detected", "ocr_mean_confidence", "ocr_num_text_boxes_norm",
    "ocr_gps_regex_match", "ocr_camtrap_id_match", "ocr_location_keyword_match",
    "ocr_suspicious_token_count_norm",
    # Category C - filename/caption context
    "filename_leak_score", "caption_leak_score",
    "filename_has_location_keyword", "filename_has_distance_pattern",
    "filename_has_coordinate_pattern", "caption_has_location_keyword",
    "caption_has_distance_pattern", "caption_has_coordinate_pattern",
    "filename_entropy_norm",
    # Category D - species sensitivity
    "species_sensitivity",
    # Category E - steganography
    "lsb_mean_deviation", "lsb_variance", "lsb_entropy_norm", "lsb_variance_score",
    "has_eof_trailing_bytes", "trailing_byte_count_norm", "trailing_entropy",
    "alpha_channel_variance", "suspicious_png_chunk_count_norm",
]

TARGET_COLUMN = "risk_label"


@dataclass
class ExtractionBundle:
    metadata: MetadataResult
    ocr: OCRResult
    context: ContextResult
    species: SpeciesResult
    stego: StegoResult
    feature_vector: Dict[str, float]


def extract_all_features(
    image_path: str,
    species_name: Optional[str],
    caption: Optional[str],
    species_table: Dict[str, float],
) -> ExtractionBundle:
    img, raw_bytes, fmt = ingest_image(image_path)
    filename = os.path.basename(image_path)

    metadata = extract_metadata_features(img, raw_bytes, fmt)
    ocr = extract_ocr_features(img)
    context = extract_context_features(filename, caption, species_names=list(species_table.keys()))
    species = extract_species_features(species_name, species_table)
    stego = extract_stego_features(img, raw_bytes, fmt)

    vector = {}
    vector.update(metadata.to_feature_dict())
    vector.update(ocr.to_feature_dict())
    vector.update(context.to_feature_dict())
    vector.update(species.to_feature_dict())
    vector.update(stego.to_feature_dict())

    # Guarantee schema completeness (fail loudly rather than silently drift)
    missing = [c for c in FEATURE_COLUMNS if c not in vector]
    if missing:
        raise RuntimeError(f"Feature pipeline is missing expected columns: {missing}")

    ordered_vector = {c: float(vector[c]) for c in FEATURE_COLUMNS}
    return ExtractionBundle(
        metadata=metadata, ocr=ocr, context=context, species=species, stego=stego,
        feature_vector=ordered_vector,
    )
