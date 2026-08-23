"""
WildGuard - tests/test_metadata.py
Category A: clean, GPS, camera metadata, timestamp images.
"""
from features.ingest import ingest_image
from features.metadata_features import extract_metadata_features


def test_clean_image_has_no_metadata(clean_jpeg):
    img, raw, fmt = ingest_image(clean_jpeg)
    m = extract_metadata_features(img, raw, fmt)
    assert m.has_gps is False
    assert m.has_camera_make is False
    assert m.has_datetime_original is False


def test_gps_image_detected(gps_jpeg):
    img, raw, fmt = ingest_image(gps_jpeg)
    m = extract_metadata_features(img, raw, fmt)
    assert m.has_gps is True
    assert m.gps_coordinates is not None
    lat, lon = m.gps_coordinates
    assert 12.9 < lat < 13.0
    assert 77.5 < lon < 77.7


def test_camera_metadata_detected(camera_metadata_jpeg):
    img, raw, fmt = ingest_image(camera_metadata_jpeg)
    m = extract_metadata_features(img, raw, fmt)
    assert m.has_camera_make is True
    assert m.has_camera_model is True
    assert m.has_gps is False


def test_timestamp_detected(timestamp_jpeg):
    img, raw, fmt = ingest_image(timestamp_jpeg)
    m = extract_metadata_features(img, raw, fmt)
    assert m.has_datetime_original is True


def test_feature_dict_is_numeric_only(gps_jpeg):
    img, raw, fmt = ingest_image(gps_jpeg)
    m = extract_metadata_features(img, raw, fmt)
    d = m.to_feature_dict()
    assert all(isinstance(v, float) for v in d.values())
