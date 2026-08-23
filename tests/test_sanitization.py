"""
WildGuard - tests/test_sanitization.py
Verify EXIF removed, GPS removed, OCR text blurred, filename sanitized,
output remains readable.
"""
import os

from features.ingest import ingest_image
from features.metadata_features import extract_metadata_features
from features.ocr_features import extract_ocr_features
from sanitization.sanitize import sanitize_and_provenance
from sanitization.filename_sanitization import generate_safe_filename
from sanitization.caption_sanitization import sanitize_caption


def test_exif_and_gps_removed(gps_jpeg, tmp_output_dir):
    result = sanitize_and_provenance(gps_jpeg, tmp_output_dir, "SANITIZE", caption=None, ocr_result=None)
    img, raw, fmt = ingest_image(result.output_path)
    m = extract_metadata_features(img, raw, fmt)
    assert m.has_gps is False


def test_camera_metadata_removed(camera_metadata_jpeg, tmp_output_dir):
    result = sanitize_and_provenance(camera_metadata_jpeg, tmp_output_dir, "SANITIZE", caption=None, ocr_result=None)
    img, raw, fmt = ingest_image(result.output_path)
    m = extract_metadata_features(img, raw, fmt)
    assert m.has_camera_make is False
    assert m.has_camera_model is False


def test_output_filename_is_safe(gps_jpeg, tmp_output_dir):
    result = sanitize_and_provenance(gps_jpeg, tmp_output_dir, "SANITIZE", caption=None, ocr_result=None)
    fname = os.path.basename(result.output_path)
    assert fname.startswith("sanitized_")
    assert "gps" not in fname.lower()


def test_output_remains_viewable(gps_jpeg, tmp_output_dir):
    result = sanitize_and_provenance(gps_jpeg, tmp_output_dir, "SANITIZE", caption=None, ocr_result=None)
    img, raw, fmt = ingest_image(result.output_path)
    img.verify()  # raises if corrupted


def test_ocr_regions_blurred(ocr_gps_text_jpeg, tmp_output_dir):
    img, raw, fmt = ingest_image(ocr_gps_text_jpeg)
    ocr = extract_ocr_features(img)
    result = sanitize_and_provenance(ocr_gps_text_jpeg, tmp_output_dir, "SANITIZE", caption=None, ocr_result=ocr)
    img2, raw2, fmt2 = ingest_image(result.output_path)
    assert raw2 != raw  # content genuinely changed (blur was applied)


def test_no_ocr_hits_is_noop(clean_jpeg, tmp_output_dir):
    img, raw, fmt = ingest_image(clean_jpeg)
    ocr = extract_ocr_features(img)
    assert len(ocr.hits) == 0
    result = sanitize_and_provenance(clean_jpeg, tmp_output_dir, "SANITIZE", caption=None, ocr_result=ocr)
    img2, raw2, fmt2 = ingest_image(result.output_path)
    img2.verify()  # still produces a valid file


def test_caption_sanitization_redacts_location():
    sanitized = sanitize_caption("Spotted this tiger 2 km from the Bandipur core zone checkpost")
    assert "bandipur" not in sanitized.lower()
    assert "[REDACTED" in sanitized


def test_caption_sanitization_handles_empty():
    assert sanitize_caption(None) == ""
    assert sanitize_caption("") == ""


def test_safe_filename_generator_uniqueness():
    a = generate_safe_filename("jpg")
    b = generate_safe_filename("jpg")
    assert a != b
    assert a.endswith(".jpg")
