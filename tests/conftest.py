"""
WildGuard - tests/conftest.py

Shared fixtures: builds small synthetic images on the fly (clean, GPS,
camera metadata, OCR text, stego) so the test suite is self-contained and
doesn't depend on the (large, machine-specific) real animal dataset.
"""

import io
import os
import random
import sys
import tempfile

import piexif
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TMP_DIR = tempfile.mkdtemp(prefix="wildguard_tests_")


def _deg_to_dms(deg):
    d = int(deg)
    m_f = (deg - d) * 60
    m = int(m_f)
    s = (m_f - m) * 60
    return [(d, 1), (m, 1), (int(s * 100), 100)]


@pytest.fixture
def clean_jpeg():
    """
    A flat solid-color image is a pathological (not representative) LSB
    baseline - every pixel identical means the LSB plane is degenerate
    (constant 0 or 1), NOT the near-random ~0.5 mean a real photo's sensor
    noise naturally produces. Use mild synthetic noise so this fixture is a
    meaningful "natural" baseline for stego comparison tests.
    """
    import numpy as np
    rng = np.random.default_rng(123)
    base = np.zeros((240, 300, 3), dtype=np.uint8)
    base[:, :] = (60, 110, 70)
    noise = rng.integers(-8, 9, size=base.shape, endpoint=True)
    arr = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    path = os.path.join(TMP_DIR, "clean.jpg")
    Image.fromarray(arr, mode="RGB").save(path, quality=90)
    return path


@pytest.fixture
def gps_jpeg():
    path = os.path.join(TMP_DIR, "gps.jpg")
    Image.new("RGB", (300, 240), (60, 110, 70)).save(path, quality=90)
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b"N",
        piexif.GPSIFD.GPSLatitude: _deg_to_dms(12.9716),
        piexif.GPSIFD.GPSLongitudeRef: b"E",
        piexif.GPSIFD.GPSLongitude: _deg_to_dms(77.5946),
    }
    exif_bytes = piexif.dump({"0th": {}, "Exif": {}, "GPS": gps_ifd, "1st": {}, "thumbnail": None})
    piexif.insert(exif_bytes, path)
    return path


@pytest.fixture
def camera_metadata_jpeg():
    path = os.path.join(TMP_DIR, "camera.jpg")
    Image.new("RGB", (300, 240), (60, 110, 70)).save(path, quality=90)
    zeroth = {piexif.ImageIFD.Make: b"Canon", piexif.ImageIFD.Model: b"EOS 90D"}
    exif_bytes = piexif.dump({"0th": zeroth, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None})
    piexif.insert(exif_bytes, path)
    return path


@pytest.fixture
def timestamp_jpeg():
    path = os.path.join(TMP_DIR, "timestamp.jpg")
    Image.new("RGB", (300, 240), (60, 110, 70)).save(path, quality=90)
    exif_ifd = {piexif.ExifIFD.DateTimeOriginal: b"2024:05:12 07:15:00"}
    exif_bytes = piexif.dump({"0th": {}, "Exif": exif_ifd, "GPS": {}, "1st": {}, "thumbnail": None})
    piexif.insert(exif_bytes, path)
    return path


@pytest.fixture
def ocr_location_jpeg():
    from dataset.inject_ocr import inject_ocr_text
    rng = random.Random(1)
    img = Image.new("RGB", (400, 300), (80, 90, 70))
    out = inject_ocr_text(img, rng, num_tokens=1)
    # force a specific deterministic token by monkeypatching the pool briefly
    path = os.path.join(TMP_DIR, "ocr_zone.jpg")
    out.save(path, quality=90)
    return path


@pytest.fixture
def ocr_gps_text_jpeg():
    from PIL import ImageDraw
    img = Image.new("RGB", (400, 300), (10, 10, 10))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 320, 70], fill=(0, 0, 0))
    draw.text((30, 30), '12\u00b034\'56.7"N', fill=(255, 255, 255))
    path = os.path.join(TMP_DIR, "ocr_gps.jpg")
    img.save(path, quality=90)
    return path


@pytest.fixture
def ocr_camera_id_jpeg():
    from PIL import ImageDraw
    img = Image.new("RGB", (400, 300), (10, 10, 10))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 260, 70], fill=(0, 0, 0))
    draw.text((30, 30), "CAM-123", fill=(255, 255, 255))
    path = os.path.join(TMP_DIR, "ocr_camid.jpg")
    img.save(path, quality=90)
    return path


@pytest.fixture
def stego_appended_bytes_jpeg():
    path = os.path.join(TMP_DIR, "stego_trailing.jpg")
    Image.new("RGB", (300, 240), (60, 110, 70)).save(path, quality=90)
    with open(path, "ab") as f:
        f.write(b"HIDDEN_PAYLOAD" * 50)
    return path


@pytest.fixture
def stego_lsb_jpeg():
    from dataset.inject_stego import inject_lsb_payload
    import numpy as np
    rng = random.Random(2)
    np_rng = np.random.default_rng(2)
    base = np.zeros((240, 300, 3), dtype=np.uint8)
    base[:, :] = (60, 110, 70)
    noise = np_rng.integers(-8, 9, size=base.shape, endpoint=True)
    arr = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    out = inject_lsb_payload(img, rng, coverage=0.9)
    path = os.path.join(TMP_DIR, "stego_lsb.jpg")
    out.save(path, quality=95)
    return path


@pytest.fixture
def tmp_output_dir():
    d = os.path.join(TMP_DIR, "output")
    os.makedirs(d, exist_ok=True)
    return d
