"""
WildGuard - sanitization/metadata_strip.py

Defense-in-depth: strips EVERY metadata channel (EXIF, GPS, IPTC, XMP,
camera/software metadata, embedded thumbnails, PNG ancillary chunks)
unconditionally, regardless of what the detector actually found. Per spec
section 23: "Even if a particular feature was not detected, still perform
every relevant sanitization operation."
"""

import io
from PIL import Image


def strip_all_metadata(img: Image.Image) -> Image.Image:
    """
    Returns a pixel-identical copy of img with zero EXIF/XMP/IPTC/ancillary
    metadata. Rebuilding a fresh Image from the pixel buffer (rather than
    just deleting known tags) guarantees nothing survives, including
    metadata formats this codebase doesn't explicitly parse.
    """
    clean = Image.new(img.mode, img.size)
    clean.paste(img)
    return clean


def encode_clean(img: Image.Image, fmt: str, quality: int = 92) -> bytes:
    """Encodes a metadata-free image to bytes. No EXIF, no PNG text chunks,
    no thumbnail - PIL's default encoder writes none of these unless a
    caller explicitly passes exif=/pnginfo= kwargs, which this never does."""
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
    else:
        img.convert("RGB" if img.mode not in ("RGBA",) else "RGBA").save(buf, format="PNG")
    return buf.getvalue()
