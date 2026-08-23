"""
WildGuard - features/ingest.py

Shared image ingestion helper used by every feature module, dataset
injectors, and the dashboard. JPEG and PNG only (matches the spec's
lightweight-only constraint - no heavy format-conversion pipeline needed).
"""

import io
import os
from typing import Tuple

from PIL import Image


def ingest_image(image_path: str) -> Tuple[Image.Image, bytes, str]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as f:
        raw_bytes = f.read()
    if len(raw_bytes) == 0:
        raise ValueError("Empty file provided.")
    img = Image.open(io.BytesIO(raw_bytes))
    img.load()
    fmt = (img.format or "UNKNOWN").upper()
    if fmt not in ("JPEG", "PNG"):
        raise ValueError(f"Unsupported format '{fmt}'. WildGuard supports JPEG and PNG.")
    return img, raw_bytes, fmt
