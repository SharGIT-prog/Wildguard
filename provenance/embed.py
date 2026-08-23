"""
WildGuard - provenance/embed.py

Level 6: mock blockchain-style provenance, embedded INSIDE the artifact
itself (never a separate .json file).

--- Hash protocol (spec section 26, avoids circularity) ---

    H0 = SHA256(original artifact)                # pre-sanitization
    sanitize -> S (bytes)
    H1 = SHA256(S)                                  # artifact_hash: computed
                                                      #   BEFORE this record
                                                      #   is embedded
    record = {parent_hash: H0, artifact_hash: H1, ...}
    embed record into S (byte-exact insertion, see provenance/container.py)
       -> P (bytes)
    H2 = SHA256(P)                                  # container_hash: computed
                                                      #   AFTER embedding

Authoritative hashes:
    artifact_hash  (H1) - the sanitized content's own hash, recorded INSIDE
                    the provenance record. Recoverable from P at verify time
                    by byte-exact stripping of the embedded marker (never a
                    re-encode), so no circularity: the record never needs to
                    contain a hash of itself.
    container_hash (H2) - hash of the exact bytes as distributed/downloaded;
                    used for straightforward bit-for-bit tamper detection.
"""

import json
import uuid
import datetime as dt
from dataclasses import dataclass, asdict

from provenance.hash import sha256_bytes
from provenance.container import insert_jpeg_comment, insert_png_text

PROVENANCE_VERSION = 1
JPEG_COMMENT_PREFIX = b"WILDGUARD_PROV_V1:"
PNG_TEXT_KEY = b"wildguard_provenance"


@dataclass
class ProvenanceRecord:
    version: int
    artifact_id: str
    parent_hash: str      # H0 - hash of the pre-sanitization original artifact
    artifact_hash: str    # H1 - hash of the sanitized artifact BEFORE embedding
    timestamp: str
    operation: str         # "sanitized" | "clean_passthrough"
    algorithm: str
    previous_status: str   # e.g. "unsafe" / "review" / "safe"
    current_status: str    # e.g. "sanitized"


def build_provenance_record(
    parent_hash: str, artifact_hash: str, operation: str,
    previous_status: str, current_status: str,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        version=PROVENANCE_VERSION,
        artifact_id=uuid.uuid4().hex,
        parent_hash=parent_hash,
        artifact_hash=artifact_hash,
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        operation=operation,
        algorithm="SHA-256",
        previous_status=previous_status,
        current_status=current_status,
    )


def embed_provenance(sanitized_bytes: bytes, fmt: str, record: ProvenanceRecord) -> bytes:
    """
    Embeds `record` into `sanitized_bytes` via byte-exact insertion (the
    original S bytes are untouched elsewhere in the file - only a new
    segment/chunk is added). Returns the container bytes P; H2 = sha256(P).
    """
    payload_json = json.dumps(asdict(record), sort_keys=True).encode("utf-8")
    if fmt == "JPEG":
        return insert_jpeg_comment(sanitized_bytes, JPEG_COMMENT_PREFIX + payload_json)
    elif fmt == "PNG":
        return insert_png_text(sanitized_bytes, PNG_TEXT_KEY, payload_json)
    else:
        raise ValueError(f"Unsupported format for provenance embedding: {fmt}")
