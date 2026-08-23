"""
WildGuard - tests/test_hash.py
Verify deterministic SHA-256 calculation.
"""
from provenance.hash import sha256_bytes, sha256_file


def test_sha256_bytes_deterministic():
    data = b"wildguard test payload"
    assert sha256_bytes(data) == sha256_bytes(data)


def test_sha256_bytes_changes_with_content():
    assert sha256_bytes(b"a") != sha256_bytes(b"b")


def test_sha256_file_matches_bytes(clean_jpeg):
    with open(clean_jpeg, "rb") as f:
        data = f.read()
    assert sha256_file(clean_jpeg) == sha256_bytes(data)


def test_sha256_known_vector():
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
