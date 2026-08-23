"""
WildGuard - provenance/container.py

Low-level, BYTE-EXACT primitives for embedding/extracting/stripping a
provenance payload directly in the container format (JPEG COM comment
segment / PNG tEXt chunk) via raw byte manipulation - deliberately NOT
through a Pillow re-encode/re-save, because re-encoding is not guaranteed
byte-identical even at "the same" quality setting (encoder version,
rounding, chroma subsampling defaults can all differ). Byte-exact
stripping is what makes the hash protocol's H1 (artifact_hash) exactly
recoverable from the final container - see embed.py / verify.py.
"""

import zlib
from typing import Optional

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"
JPEG_COM_MARKER = 0xFE
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# JPEG: COM (comment) marker segment, inserted immediately after SOI
# --------------------------------------------------------------------------
def insert_jpeg_comment(jpeg_bytes: bytes, payload: bytes) -> bytes:
    if jpeg_bytes[:2] != JPEG_SOI:
        raise ValueError("Not a valid JPEG (missing SOI marker)")
    seg_len = len(payload) + 2  # length field includes itself, per JPEG spec
    if seg_len > 65535:
        raise ValueError("Provenance payload too large for a single JPEG COM segment")
    segment = bytes([0xFF, JPEG_COM_MARKER]) + seg_len.to_bytes(2, "big") + payload
    return jpeg_bytes[:2] + segment + jpeg_bytes[2:]


def extract_jpeg_comment(jpeg_bytes: bytes) -> Optional[bytes]:
    pos = 2
    n = len(jpeg_bytes)
    while pos + 4 <= n:
        if jpeg_bytes[pos] != 0xFF:
            break
        marker = jpeg_bytes[pos + 1]
        if marker in (0xD8,) or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            pos += 2
            continue
        if marker == 0xD9:  # EOI
            break
        seg_len = int.from_bytes(jpeg_bytes[pos + 2:pos + 4], "big")
        if marker == JPEG_COM_MARKER:
            return jpeg_bytes[pos + 4:pos + 2 + seg_len]
        pos += 2 + seg_len
        if marker == 0xDA:  # start of scan - no more markers before compressed data
            break
    return None


def strip_jpeg_comment(jpeg_bytes: bytes) -> bytes:
    pos = 2
    n = len(jpeg_bytes)
    while pos + 4 <= n:
        if jpeg_bytes[pos] != 0xFF:
            break
        marker = jpeg_bytes[pos + 1]
        if marker in (0xD8,) or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            pos += 2
            continue
        if marker == 0xD9:
            break
        seg_len = int.from_bytes(jpeg_bytes[pos + 2:pos + 4], "big")
        if marker == JPEG_COM_MARKER:
            return jpeg_bytes[:pos] + jpeg_bytes[pos + 2 + seg_len:]
        pos += 2 + seg_len
        if marker == 0xDA:
            break
    return jpeg_bytes  # no COM segment present - already "stripped"


# --------------------------------------------------------------------------
# PNG: tEXt chunk with a dedicated keyword, inserted immediately before IEND
# --------------------------------------------------------------------------
def insert_png_text(png_bytes: bytes, keyword: bytes, text: bytes) -> bytes:
    if png_bytes[:8] != PNG_SIGNATURE:
        raise ValueError("Not a valid PNG (bad signature)")
    idx = png_bytes.rfind(b"IEND")
    if idx == -1:
        raise ValueError("Not a valid PNG (missing IEND chunk)")
    iend_chunk_start = idx - 4  # 4-byte length field precedes the 4-byte type
    data = keyword + b"\x00" + text
    chunk = (
        len(data).to_bytes(4, "big") + b"tEXt" + data
        + zlib.crc32(b"tEXt" + data).to_bytes(4, "big")
    )
    return png_bytes[:iend_chunk_start] + chunk + png_bytes[iend_chunk_start:]


def extract_png_text(png_bytes: bytes, keyword: bytes) -> Optional[bytes]:
    pos = 8
    n = len(png_bytes)
    while pos + 8 <= n:
        length = int.from_bytes(png_bytes[pos:pos + 4], "big")
        ctype = png_bytes[pos + 4:pos + 8]
        data_start = pos + 8
        data_end = data_start + length
        if data_end > n:
            break
        if ctype == b"tEXt":
            data = png_bytes[data_start:data_end]
            k, _, v = data.partition(b"\x00")
            if k == keyword:
                return v
        pos = data_end + 4
        if ctype == b"IEND":
            break
    return None


def strip_png_text(png_bytes: bytes, keyword: bytes) -> bytes:
    pos = 8
    out = bytearray(png_bytes[:8])
    n = len(png_bytes)
    while pos + 8 <= n:
        length = int.from_bytes(png_bytes[pos:pos + 4], "big")
        ctype = png_bytes[pos + 4:pos + 8]
        data_start = pos + 8
        data_end = data_start + length
        chunk_end = data_end + 4
        if data_end > n:
            out += png_bytes[pos:]
            break
        skip = False
        if ctype == b"tEXt":
            data = png_bytes[data_start:data_end]
            k, _, v = data.partition(b"\x00")
            if k == keyword:
                skip = True
        if not skip:
            out += png_bytes[pos:chunk_end]
        pos = chunk_end
        if ctype == b"IEND":
            break
    return bytes(out)
