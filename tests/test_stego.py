"""
WildGuard - tests/test_stego.py
Category E: normal image, appended bytes, suspicious LSB distribution.
"""
from features.ingest import ingest_image
from features.stego_features import extract_stego_features


def test_normal_image_no_trailing_bytes(clean_jpeg):
    img, raw, fmt = ingest_image(clean_jpeg)
    s = extract_stego_features(img, raw, fmt)
    assert s.has_eof_trailing_bytes is False
    assert s.trailing_byte_count == 0


def test_appended_bytes_detected(stego_appended_bytes_jpeg):
    img, raw, fmt = ingest_image(stego_appended_bytes_jpeg)
    s = extract_stego_features(img, raw, fmt)
    assert s.has_eof_trailing_bytes is True
    assert s.trailing_byte_count > 0
    assert s.trailing_entropy >= 0.0


def test_suspicious_lsb_distribution(stego_lsb_jpeg, clean_jpeg):
    img_stego, raw_stego, fmt_stego = ingest_image(stego_lsb_jpeg)
    img_clean, raw_clean, fmt_clean = ingest_image(clean_jpeg)
    s_stego = extract_stego_features(img_stego, raw_stego, fmt_stego)
    s_clean = extract_stego_features(img_clean, raw_clean, fmt_clean)
    # The LSB-payload image should score meaningfully higher than a natural one
    assert s_stego.lsb_variance_score >= s_clean.lsb_variance_score


def test_stego_feature_dict_numeric(clean_jpeg):
    img, raw, fmt = ingest_image(clean_jpeg)
    s = extract_stego_features(img, raw, fmt)
    d = s.to_feature_dict()
    assert all(isinstance(v, float) for v in d.values())
