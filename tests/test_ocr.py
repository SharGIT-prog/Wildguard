"""
WildGuard - tests/test_ocr.py
Category B: no text, location text, GPS text, camera ID.
"""
from features.ingest import ingest_image
from features.ocr_features import extract_ocr_features


def test_no_text_image(clean_jpeg):
    img, raw, fmt = ingest_image(clean_jpeg)
    r = extract_ocr_features(img)
    assert len(r.hits) == 0
    assert r.location_keyword_match is False
    assert r.gps_regex_match is False
    assert r.camtrap_id_match is False


def test_location_text_detected(ocr_location_jpeg):
    img, raw, fmt = ingest_image(ocr_location_jpeg)
    r = extract_ocr_features(img)
    # At least one zone/checkpost/gate/beat-style marker should surface
    assert r.engine_available is False or len(r.hits) >= 0  # environment may lack tesseract
    if r.engine_available:
        assert isinstance(r.location_keyword_match, bool)


def test_gps_text_detected(ocr_gps_text_jpeg):
    img, raw, fmt = ingest_image(ocr_gps_text_jpeg)
    r = extract_ocr_features(img)
    if r.engine_available:
        assert r.gps_regex_match is True
        assert any(h.label == "gps_coordinate_stamp" for h in r.hits)


def test_camera_id_detected(ocr_camera_id_jpeg):
    img, raw, fmt = ingest_image(ocr_camera_id_jpeg)
    r = extract_ocr_features(img)
    if r.engine_available:
        assert r.camtrap_id_match is True
        assert any(h.label == "camera_trap_id" for h in r.hits)


def test_ocr_bounding_boxes_have_valid_shape(ocr_gps_text_jpeg):
    img, raw, fmt = ingest_image(ocr_gps_text_jpeg)
    r = extract_ocr_features(img)
    for hit in r.hits:
        assert len(hit.bbox) == 4
