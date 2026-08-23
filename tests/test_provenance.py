"""
WildGuard - tests/test_provenance.py
Valid artifact, modified artifact, corrupted provenance, mismatched parent
hash, mismatched artifact hash.
"""
import json
import os

from features.ingest import ingest_image
from provenance.hash import sha256_bytes
from provenance.embed import build_provenance_record, embed_provenance
from provenance.container import insert_jpeg_comment, strip_jpeg_comment, extract_jpeg_comment
from provenance.verify import verify_image
from sanitization.sanitize import sanitize_and_provenance


def test_valid_artifact_verifies(gps_jpeg, tmp_output_dir):
    result = sanitize_and_provenance(gps_jpeg, tmp_output_dir, "SANITIZE", caption=None, ocr_result=None)
    v = verify_image(result.output_path)
    assert v.provenance_found is True
    assert v.valid is True
    assert v.hash_match is True


def test_modified_artifact_fails_verification(gps_jpeg, tmp_output_dir):
    result = sanitize_and_provenance(gps_jpeg, tmp_output_dir, "SANITIZE", caption=None, ocr_result=None)
    tampered_path = result.output_path + ".tampered.jpg"
    with open(result.output_path, "rb") as f:
        data = f.read()
    with open(tampered_path, "wb") as f:
        f.write(data + b"INJECTED")
    v = verify_image(tampered_path)
    assert v.provenance_found is True
    assert v.valid is False
    assert v.hash_match is False


def test_corrupted_provenance_json_not_found(gps_jpeg, tmp_output_dir):
    """A COM segment present but not valid WildGuard JSON should behave
    like 'no provenance found', not crash."""
    with open(gps_jpeg, "rb") as f:
        raw = f.read()
    corrupted = insert_jpeg_comment(raw, b"NOT_WILDGUARD_DATA")
    path = os.path.join(tmp_output_dir, "corrupted.jpg")
    with open(path, "wb") as f:
        f.write(corrupted)
    v = verify_image(path)
    assert v.provenance_found is False


def test_image_with_no_provenance_reports_not_found(clean_jpeg):
    v = verify_image(clean_jpeg)
    assert v.provenance_found is False
    assert v.valid is False


def test_mismatched_artifact_hash_detected(gps_jpeg, tmp_output_dir):
    """Manually build a record with a WRONG artifact_hash and confirm
    verification correctly flags the mismatch."""
    img, raw_bytes, fmt = ingest_image(gps_jpeg)
    wrong_hash = sha256_bytes(b"not the real content")
    record = build_provenance_record(
        parent_hash=sha256_bytes(raw_bytes),
        artifact_hash=wrong_hash,
        operation="sanitized", previous_status="unsafe", current_status="sanitized",
    )
    container = embed_provenance(raw_bytes, fmt, record)
    path = os.path.join(tmp_output_dir, "mismatched.jpg")
    with open(path, "wb") as f:
        f.write(container)

    v = verify_image(path)
    assert v.provenance_found is True
    assert v.hash_match is False
    assert v.valid is False


def test_mismatched_parent_hash_flagged(gps_jpeg, tmp_output_dir):
    img, raw_bytes, fmt = ingest_image(gps_jpeg)
    record = build_provenance_record(
        parent_hash="",  # missing/empty parent hash
        artifact_hash=sha256_bytes(raw_bytes),
        operation="sanitized", previous_status="unsafe", current_status="sanitized",
    )
    container = embed_provenance(raw_bytes, fmt, record)
    path = os.path.join(tmp_output_dir, "no_parent.jpg")
    with open(path, "wb") as f:
        f.write(container)

    v = verify_image(path)
    assert v.valid is False
    assert any("parent_hash" in r for r in v.reasons)


def test_container_byte_exact_roundtrip(gps_jpeg):
    """Stripping the embedded comment must recover the EXACT pre-embedding
    bytes - this is what makes H1 verification non-circular and correct."""
    with open(gps_jpeg, "rb") as f:
        raw = f.read()
    payload = b"WILDGUARD_PROV_V1:" + json.dumps({"a": 1}).encode()
    embedded = insert_jpeg_comment(raw, payload)
    recovered = strip_jpeg_comment(embedded)
    assert recovered == raw
    assert extract_jpeg_comment(embedded) == payload
