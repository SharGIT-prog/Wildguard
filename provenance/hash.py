"""
WildGuard - provenance/hash.py

Level 5: SHA-256 hashing utilities. Deterministic - identical bytes always
produce an identical hash, verified by tests/test_hash.py.
"""

import hashlib


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
