"""
WildGuard - provenance/extract.py

Extracts the embedded provenance JSON record (and recovers the byte-exact
pre-embedding artifact bytes, for H1 recomputation) from a provenance-
bearing image - no separate ledger file needed.
"""

import json
from typing import Optional, Tuple, Dict, Any

from features.ingest import ingest_image
from provenance.container import (
    extract_jpeg_comment, strip_jpeg_comment,
    extract_png_text, strip_png_text,
)
from provenance.embed import JPEG_COMMENT_PREFIX, PNG_TEXT_KEY


def extract_provenance_from_bytes(container_bytes: bytes, fmt: str) -> Tuple[Optional[Dict[str, Any]], Optional[bytes]]:
    """
    Returns (record_dict_or_None, artifact_bytes_before_embedding_or_None).
    The second element is the exact byte-for-byte content that existed
    BEFORE the provenance marker was inserted - i.e. it should hash to the
    record's own `artifact_hash` field if the file has not been tampered
    with since sanitization.
    """
    if fmt == "JPEG":
        raw = extract_jpeg_comment(container_bytes)
        if raw is None or not raw.startswith(JPEG_COMMENT_PREFIX):
            return None, None
        payload = raw[len(JPEG_COMMENT_PREFIX):]
        try:
            record = json.loads(payload.decode("utf-8"))
        except Exception:
            return None, None
        artifact_bytes = strip_jpeg_comment(container_bytes)
        return record, artifact_bytes

    elif fmt == "PNG":
        raw = extract_png_text(container_bytes, PNG_TEXT_KEY)
        if raw is None:
            return None, None
        try:
            record = json.loads(raw.decode("utf-8"))
        except Exception:
            return None, None
        artifact_bytes = strip_png_text(container_bytes, PNG_TEXT_KEY)
        return record, artifact_bytes

    return None, None


def extract_provenance_from_file(image_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[bytes], Optional[str]]:
    img, raw_bytes, fmt = ingest_image(image_path)
    record, artifact_bytes = extract_provenance_from_bytes(raw_bytes, fmt)
    return record, artifact_bytes, fmt
