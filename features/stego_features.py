"""
WildGuard - features/stego_features.py

Category E: steganographic / covert-channel features. Lightweight
statistical analysis only (no deep steganalysis):
  - spatial LSB statistics (mean, variance, entropy, per-channel, block
    irregularity)
  - trailing bytes past the legitimate EOF marker (JPEG FFD9 / PNG IEND),
    with trailing-byte entropy
  - PNG alpha-channel variance and suspicious/custom chunk counting
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from PIL import Image

JPEG_EOF_MARKER = b"\xff\xd9"
PNG_IEND_CHUNK = b"IEND"


@dataclass
class StegoResult:
    lsb_mean: float = 0.5
    lsb_variance: float = 0.0
    lsb_entropy: float = 0.0
    lsb_variance_score: float = 0.0          # composite 0-1 "how unnatural" score
    channel_lsb_means: List[float] = field(default_factory=list)
    has_eof_trailing_bytes: bool = False
    trailing_byte_count: int = 0
    trailing_byte_count_norm: float = 0.0
    trailing_entropy: float = 0.0
    alpha_channel_variance: float = 0.0
    suspicious_png_chunk_count: int = 0
    notes: List[str] = field(default_factory=list)

    def to_feature_dict(self) -> Dict[str, float]:
        return {
            "lsb_mean_deviation": abs(self.lsb_mean - 0.5) * 2,
            "lsb_variance": float(self.lsb_variance),
            "lsb_entropy_norm": float(self.lsb_entropy) / 1.0,  # already normalized to [0,1] below
            "lsb_variance_score": float(self.lsb_variance_score),
            "has_eof_trailing_bytes": float(self.has_eof_trailing_bytes),
            "trailing_byte_count_norm": float(self.trailing_byte_count_norm),
            "trailing_entropy": float(self.trailing_entropy),
            "alpha_channel_variance": float(self.alpha_channel_variance),
            "suspicious_png_chunk_count_norm": min(self.suspicious_png_chunk_count / 5.0, 1.0),
        }


def _byte_entropy(data: bytes) -> float:
    """Shannon entropy of a byte string, normalized to [0,1] (max = 8 bits -> 1.0)."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    entropy_bits = -np.sum(probs * np.log2(probs))
    return float(entropy_bits / 8.0)


def extract_stego_features(img: Image.Image, raw_bytes: bytes, fmt: str) -> StegoResult:
    result = StegoResult()

    # ---- Spatial LSB statistics ----
    try:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        lsb_planes = arr & 1
        result.lsb_mean = float(lsb_planes.mean())
        result.lsb_variance = float(lsb_planes.var())
        result.channel_lsb_means = [float(lsb_planes[:, :, c].mean()) for c in range(3)]

        # Bit-level entropy of the LSB plane (close to 1.0 for pure random
        # noise, which is itself a mild anomaly signal vs natural photo LSBs)
        flat_bits = lsb_planes.flatten()
        p1 = flat_bits.mean()
        p0 = 1 - p1
        bit_entropy = 0.0
        for p in (p0, p1):
            if p > 0:
                bit_entropy -= p * math.log2(p)
        result.lsb_entropy = float(bit_entropy)  # in [0,1] already (binary entropy)

        h, w = lsb_planes.shape[0], lsb_planes.shape[1]
        block = 8
        block_means = []
        for y in range(0, h - h % block, block):
            for x in range(0, w - w % block, block):
                block_means.append(lsb_planes[y:y + block, x:x + block].mean())
        block_means = np.array(block_means) if block_means else np.array([0.5])
        block_variance = float(block_means.var())
        natural_band_center = 0.02
        band_distance = abs(block_variance - natural_band_center)

        deviation_from_uniform = abs(result.lsb_mean - 0.5) * 2
        result.lsb_variance_score = float(
            max(0.0, min(1.0, 0.55 * deviation_from_uniform + 0.30 * min(band_distance * 10, 1.0)
                          + 0.15 * result.lsb_entropy))
        )
        if result.lsb_variance_score > 0.6:
            result.notes.append("LSB bit distribution deviates from natural-image baseline.")
    except Exception as e:
        result.notes.append(f"LSB analysis failed: {e}")

    # ---- Trailing bytes past EOF ----
    try:
        trailing_bytes = b""
        if fmt == "JPEG":
            idx = raw_bytes.rfind(JPEG_EOF_MARKER)
            if idx != -1 and len(raw_bytes) - (idx + 2) > 0:
                trailing_bytes = raw_bytes[idx + 2:]
        elif fmt == "PNG":
            idx = raw_bytes.rfind(PNG_IEND_CHUNK)
            if idx != -1:
                iend_end = idx + len(PNG_IEND_CHUNK) + 4
                if len(raw_bytes) - iend_end > 0:
                    trailing_bytes = raw_bytes[iend_end:]
        if trailing_bytes:
            result.has_eof_trailing_bytes = True
            result.trailing_byte_count = len(trailing_bytes)
            result.trailing_entropy = _byte_entropy(trailing_bytes)
            result.notes.append(f"{len(trailing_bytes)} bytes found appended past the {fmt} EOF marker.")
    except Exception as e:
        result.notes.append(f"EOF trailing-byte scan failed: {e}")
    result.trailing_byte_count_norm = min(result.trailing_byte_count / 5000.0, 1.0)

    # ---- PNG alpha channel + suspicious chunk count ----
    try:
        if "A" in img.getbands():
            alpha = np.asarray(img.getchannel("A"), dtype=np.float32)
            result.alpha_channel_variance = min(float(alpha.var()) / (128.0 ** 2), 1.0)
        if fmt == "PNG":
            result.suspicious_png_chunk_count = _count_suspicious_png_chunks(raw_bytes)
    except Exception as e:
        result.notes.append(f"Alpha/chunk analysis failed: {e}")

    return result


_STANDARD_PNG_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"gAMA", b"cHRM", b"sRGB", b"bKGD", b"pHYs"}


def _count_suspicious_png_chunks(raw_bytes: bytes) -> int:
    pos = 8
    suspicious = 0
    while pos + 8 <= len(raw_bytes):
        length = int.from_bytes(raw_bytes[pos:pos + 4], "big")
        ctype = raw_bytes[pos + 4:pos + 8]
        data_end = pos + 8 + length
        if data_end > len(raw_bytes):
            break
        if ctype not in _STANDARD_PNG_CHUNKS and ctype not in (b"tEXt", b"zTXt", b"iTXt", b"eXIf"):
            suspicious += 1  # non-standard / custom chunk type
        pos = data_end + 4
        if ctype == b"IEND":
            break
    return suspicious
