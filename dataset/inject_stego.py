"""
WildGuard - dataset/inject_stego.py

Category E injector. Two lightweight, real (not simulated-by-label) covert
channels:
  1. LSB payload embedding - overwrites the least-significant bit of each
     RGB channel with a pseudo-random payload bitstream over a random
     region of the image, genuinely shifting the LSB-plane statistics away
     from a natural photo's baseline (this is real LSB steganography, just
     with a synthetic payload rather than a specific hidden message).
  2. Trailing bytes appended after the legitimate EOF marker (JPEG FFD9 /
     PNG IEND) - genuine appended bytes, not a flag.
"""

import io
import random
from typing import Tuple

import numpy as np
from PIL import Image

JPEG_EOF_MARKER = b"\xff\xd9"
PNG_IEND_CHUNK = b"IEND"


def inject_lsb_payload(img: Image.Image, rng: random.Random, coverage: float = None) -> Image.Image:
    """
    Overwrites the LSB of each RGB channel, over a randomly-sized/positioned
    rectangular region (coverage = fraction of image area, randomized if not
    given), with a pseudo-random bitstream. This is real bit manipulation on
    real pixels, not a synthetic label.
    """
    out = img.convert("RGB").copy()
    arr = np.asarray(out, dtype=np.uint8).copy()
    h, w, _ = arr.shape

    if coverage is None:
        coverage = rng.uniform(0.3, 0.9)
    region_h = max(1, int(h * (coverage ** 0.5)))
    region_w = max(1, int(w * (coverage ** 0.5)))
    y0 = rng.randint(0, max(0, h - region_h))
    x0 = rng.randint(0, max(0, w - region_w))

    region = arr[y0:y0 + region_h, x0:x0 + region_w, :]
    payload_bits = np.random.RandomState(rng.randint(0, 2**31 - 1)).randint(
        0, 2, size=region.shape, dtype=np.uint8
    )
    region_cleared = region & 0xFE  # zero out LSB
    region_new = region_cleared | payload_bits
    arr[y0:y0 + region_h, x0:x0 + region_w, :] = region_new

    return Image.fromarray(arr, mode="RGB")


def append_trailing_bytes(image_bytes: bytes, fmt: str, rng: random.Random,
                           byte_count: int = None) -> bytes:
    """Appends genuine extra bytes after the format's legitimate EOF marker."""
    if byte_count is None:
        byte_count = rng.randint(200, 3000)
    payload = bytes(rng.getrandbits(8) for _ in range(byte_count))

    if fmt == "JPEG":
        idx = image_bytes.rfind(JPEG_EOF_MARKER)
        if idx == -1:
            return image_bytes + payload  # malformed but still appends
        insert_at = idx + 2
        return image_bytes[:insert_at] + payload + image_bytes[insert_at:]
    elif fmt == "PNG":
        idx = image_bytes.rfind(PNG_IEND_CHUNK)
        if idx == -1:
            return image_bytes + payload
        insert_at = idx + len(PNG_IEND_CHUNK) + 4  # past IEND's CRC
        return image_bytes[:insert_at] + payload + image_bytes[insert_at:]
    return image_bytes + payload
