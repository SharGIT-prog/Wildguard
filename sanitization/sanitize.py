"""
WildGuard - sanitization/sanitize.py

Orchestrates Levels 4-6 for an unsafe/reviewed image:
  Level 4: sanitize (strip ALL metadata, blur ALL OCR boxes, safe filename,
           sanitize caption) - always every applicable operation,
           regardless of what was individually detected (defense-in-depth).
  Level 5: SHA-256 of the exact sanitized bytes.
  Level 6: build + embed the provenance record into the sanitized artifact.
"""

import os
from dataclasses import dataclass
from typing import Optional

from features.ingest import ingest_image
from features.ocr_features import OCRResult
from sanitization.metadata_strip import strip_all_metadata, encode_clean
from sanitization.ocr_redaction import blur_ocr_regions
from sanitization.filename_sanitization import generate_safe_filename
from sanitization.caption_sanitization import sanitize_caption
from provenance.hash import sha256_bytes
from provenance.embed import build_provenance_record, embed_provenance


@dataclass
class SanitizationOutput:
    output_path: str
    original_hash: str
    artifact_hash: str
    container_hash: str
    sanitized_caption: str
    provenance_artifact_id: str


def sanitize_and_provenance(
    image_path: str, output_dir: str, decision: str, caption: Optional[str] = None,
    ocr_result: Optional[OCRResult] = None,
) -> SanitizationOutput:
    os.makedirs(output_dir, exist_ok=True)
    img, raw_bytes, fmt = ingest_image(image_path)
    original_hash = sha256_bytes(raw_bytes)  # H0

    # Level 4 - always every applicable operation, defense-in-depth:
    stripped = strip_all_metadata(img)
    hits = ocr_result.hits if ocr_result else []
    redacted = blur_ocr_regions(stripped, hits)
    sanitized_bytes = encode_clean(redacted, fmt)  # S
    artifact_hash = sha256_bytes(sanitized_bytes)  # H1 - computed BEFORE embedding

    sanitized_caption = sanitize_caption(caption)

    # Level 6 - build + embed provenance (record contains H0 and H1, never
    # its own eventual container hash - see provenance/embed.py docstring)
    record = build_provenance_record(
        parent_hash=original_hash,
        artifact_hash=artifact_hash,
        operation="sanitized",
        previous_status=decision,
        current_status="sanitized",
    )
    container_bytes = embed_provenance(sanitized_bytes, fmt, record)  # P
    container_hash = sha256_bytes(container_bytes)  # H2

    ext = "jpg" if fmt == "JPEG" else "png"
    safe_name = generate_safe_filename(ext)
    output_path = os.path.join(output_dir, safe_name)
    with open(output_path, "wb") as f:
        f.write(container_bytes)

    return SanitizationOutput(
        output_path=output_path,
        original_hash=original_hash,
        artifact_hash=artifact_hash,
        container_hash=container_hash,
        sanitized_caption=sanitized_caption,
        provenance_artifact_id=record.artifact_id,
    )
