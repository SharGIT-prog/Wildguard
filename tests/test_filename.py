"""
WildGuard - tests/test_filename.py
Category C: safe filename, location filename, distance pattern.
"""
from features.context_features import extract_context_features


def test_safe_filename_low_score():
    r = extract_context_features("IMG_4092.jpg")
    assert r.filename_leak_score < 0.15
    assert r.filename_has_location_keyword is False


def test_location_filename_high_score():
    r = extract_context_features("tiger_bandipur_zone3.jpg")
    assert r.filename_leak_score > 0.5
    assert r.filename_has_location_keyword is True


def test_distance_pattern_filename():
    r = extract_context_features("antelope_5km_from_checkpoint.jpg")
    assert r.filename_has_distance_pattern is True


def test_caption_location_leak():
    r = extract_context_features("photo.jpg", caption="Spotted 2 km from the Bandipur checkpost")
    assert r.caption_leak_score > 0.3
    assert r.caption_has_location_keyword is True or r.caption_has_distance_pattern is True


def test_caption_benign():
    r = extract_context_features("photo.jpg", caption="Beautiful morning in the forest")
    assert r.caption_leak_score < 0.2
