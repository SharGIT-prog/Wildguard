"""
WildGuard - provenance/verify.py

Level 7: verification. Given an uploaded provenance-bearing image, checks:
  1. Is an embedded provenance record present and parseable at all?
  2. artifact_hash check: does the recovered pre-embedding content
     (byte-exact strip of the marker) hash to the record's own
     `artifact_hash` field? If not, the image content changed AFTER
     sanitization but the provenance marker is still attached (tampering).
  3. container_hash check: informational - the hash of the file exactly as
     uploaded, useful for comparing two copies of the "same" distributed
     file.
  4. parent_hash presence: sanity-checks the chain link exists.
  5. Drift check: re-runs the metadata scanner on the CURRENT artifact and
     flags any metadata that shouldn't exist on something already
     sanitized (defense-in-depth: catches reintroduced EXIF/GPS even if
     the hash checks somehow passed).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from provenance.hash import sha256_bytes
from provenance.extract import extract_provenance_from_bytes
from features.ingest import ingest_image
from features.metadata_features import extract_metadata_features


@dataclass
class VerificationResult:
    provenance_found: bool
    valid: bool
    reasons: List[str] = field(default_factory=list)
    record: Optional[Dict[str, Any]] = None
    computed_container_hash: Optional[str] = None
    computed_artifact_hash: Optional[str] = None
    recorded_artifact_hash: Optional[str] = None
    recorded_parent_hash: Optional[str] = None
    hash_match: Optional[bool] = None
    metadata_drift_flags: List[str] = field(default_factory=list)


REQUIRED_FIELDS = {"version", "artifact_id", "parent_hash", "artifact_hash",
                    "timestamp", "operation", "algorithm", "previous_status", "current_status"}


def verify_image(image_path: str) -> VerificationResult:
    img, raw_bytes, fmt = ingest_image(image_path)
    container_hash = sha256_bytes(raw_bytes)

    record, artifact_bytes = extract_provenance_from_bytes(raw_bytes, fmt)

    if record is None:
        return VerificationResult(
            provenance_found=False,
            valid=False,
            reasons=["No embedded WildGuard provenance record was found in this image. "
                     "It may never have been processed by WildGuard, or the marker was stripped."],
            computed_container_hash=container_hash,
        )

    result = VerificationResult(
        provenance_found=True, valid=True,
        record=record, computed_container_hash=container_hash,
        recorded_artifact_hash=record.get("artifact_hash"),
        recorded_parent_hash=record.get("parent_hash"),
    )

    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        result.valid = False
        result.reasons.append(f"Provenance record is missing required field(s): {sorted(missing)}.")
        return result

    if record.get("algorithm") != "SHA-256":
        result.valid = False
        result.reasons.append(f"Unexpected hash algorithm recorded: {record.get('algorithm')!r}.")

    if artifact_bytes is not None:
        computed_artifact_hash = sha256_bytes(artifact_bytes)
        result.computed_artifact_hash = computed_artifact_hash
        result.hash_match = (computed_artifact_hash == record.get("artifact_hash"))
        if not result.hash_match:
            result.valid = False
            result.reasons.append(
                "The current artifact content does not match the hash recorded at sanitization "
                "time. The image has been modified since WildGuard processed it."
            )
    else:
        result.valid = False
        result.hash_match = False
        result.reasons.append("Could not recover the pre-embedding artifact bytes for hash comparison.")

    if not record.get("parent_hash"):
        result.valid = False
        result.reasons.append("Provenance record has no parent_hash - the chain link to the "
                               "original (pre-sanitization) artifact is missing.")

    # Drift check: re-scan current metadata; a genuinely sanitized artifact
    # should show none of these regardless of hash outcome.
    try:
        metadata = extract_metadata_features(img, raw_bytes, fmt)
        for field_name, human in [
            ("has_gps", "GPS metadata"), ("has_camera_make", "camera make metadata"),
            ("has_camera_model", "camera model metadata"), ("has_serial_number", "device serial number"),
            ("has_owner_name", "owner/artist metadata"), ("has_iptc_location", "IPTC/XMP location metadata"),
        ]:
            if getattr(metadata, field_name):
                result.metadata_drift_flags.append(human)
    except Exception:
        pass

    if result.metadata_drift_flags:
        result.valid = False
        result.reasons.append(
            "Metadata drift detected: the current file contains metadata fields "
            f"({', '.join(result.metadata_drift_flags)}) that a sanitized artifact should not have. "
            "This suggests metadata was re-added after sanitization."
        )

    if result.valid:
        result.reasons.append("Provenance is intact: the artifact hash matches the recorded "
                               "value and no disallowed metadata was reintroduced.")

    return result
