#Script to deliberately tamper the image
"""
tamper_provenance_demo.py

STANDALONE tamper-proof-of-concept script for WildGuard's embedded
blockchain-style provenance. This file is fully self-contained (only needs
Pillow, which you likely already have) and does NOT import anything from
the wildguard_project codebase - run it from any folder you like, on any
WildGuard-sanitized image.

WHAT IT DOES
------------
Takes an image that WildGuard has already sanitized (i.e. it has an
embedded provenance record - a JPEG COM comment segment or a PNG tEXt
chunk), and modifies the file's bytes in a way that:

  1. Leaves the embedded provenance record itself completely intact and
     readable - WildGuard's Version page will still find it and display
     "PROVENANCE FOUND", never "NO PROVENANCE FOUND".
  2. Changes the underlying artifact content, so the hash WildGuard
     recomputes from the (marker-stripped) current file no longer matches
     the artifact_hash recorded INSIDE that provenance record.

This is exactly what a bad-faith actor re-editing/re-uploading an already
"verified safe" photo would produce, and is exactly the class of tampering
the embedded-provenance/hash-chain design exists to catch. Feeding the
output of this script into WildGuard's Version -> Verify Provenance page
should show "PROVENANCE COMPROMISED" with the reason "The current artifact
content does not match the hash recorded at sanitization time."

HOW (so you can see it isn't smoke and mirrors)
------------------------------------------------
- JPEG: WildGuard embeds its provenance as a COM (comment) marker segment
  placed immediately after the SOI marker (the very start of the file).
  WildGuard's verifier recovers the "pre-embedding" content by finding
  that FIRST comment segment and splicing it out, keeping the marker plus
  everything before and after it. This script locates that same first COM
  segment (so it never disturbs it) and appends extra bytes at the very
  end of the file, past the JPEG end-of-image marker (FFD9). Those
  trailing bytes survive WildGuard's splice-out-the-comment step, so they
  end up INSIDE the "recovered artifact" WildGuard rehashes - changing the
  hash while the provenance record itself stays perfectly parseable.

- PNG: WildGuard's verifier reconstructs the pre-embedding content by
  removing ONLY the tEXt chunk with the specific keyword "wildguard_
  provenance" and keeping every other chunk untouched. This script inserts
  a brand-new, differently-keyworded tEXt chunk ("wildguard_tamper_demo")
  right before the IEND chunk. WildGuard's verifier has no reason to
  remove that chunk (wrong keyword), so it survives into the "recovered
  artifact" and changes its hash - again without touching the real
  provenance chunk at all.

Neither technique requires understanding or reproducing WildGuard's
internal Python objects - it only relies on the two container formats'
own public byte-level structure (JPEG marker segments / PNG chunks), which
is exactly why it works against the real, unmodified verifier.

USAGE
-----
    python tamper_provenance_demo.py path/to/sanitized_image.jpg
    python tamper_provenance_demo.py path/to/sanitized_image.jpg --out path/to/tampered.jpg

If --out is omitted, writes "<original_name>_TAMPERED.<ext>" next to the input.
"""

import argparse
import os
import random
import struct
import sys
import zlib

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"
JPEG_COM_MARKER = 0xFE
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TAMPER_PNG_KEYWORD = b"wildguard_tamper_demo"


def detect_format(raw_bytes: bytes) -> str:
    if raw_bytes[:2] == JPEG_SOI:
        return "JPEG"
    if raw_bytes[:8] == PNG_SIGNATURE:
        return "PNG"
    raise ValueError(
        "Not a JPEG or PNG file (or missing/unrecognized header). "
        "WildGuard only embeds provenance in these two formats."
    )


def find_first_jpeg_com_segment_end(raw_bytes: bytes) -> int:
    """
    Walks JPEG marker segments from just after SOI and returns the byte
    offset immediately AFTER the first COM (comment) segment found - i.e.
    exactly where WildGuard's own embed step placed its provenance record,
    and exactly the boundary its verifier splices around. Used here only
    to confirm a provenance segment actually exists before we bother
    tampering - the tamper itself happens at the very end of the file, far
    away from this position, so it can never accidentally corrupt it.
    """
    pos = 2
    n = len(raw_bytes)
    while pos + 4 <= n:
        if raw_bytes[pos] != 0xFF:
            break
        marker = raw_bytes[pos + 1]
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            pos += 2
            continue
        if marker == 0xD9:
            break
        seg_len = int.from_bytes(raw_bytes[pos + 2:pos + 4], "big")
        if marker == JPEG_COM_MARKER:
            return pos + 2 + seg_len
        pos += 2 + seg_len
        if marker == 0xDA:
            break
    return -1


def tamper_jpeg(raw_bytes: bytes) -> bytes:
    com_end = find_first_jpeg_com_segment_end(raw_bytes)
    if com_end == -1:
        raise ValueError(
            "No embedded provenance (JPEG comment segment) found in this file. "
            "This script only tampers images that WildGuard has already sanitized "
            "and stamped with provenance - run WildGuard's Analyse page on an "
            "unsafe image first, then point this script at the sanitized output."
        )

    print(f"[tamper] Found the embedded provenance COM segment "
          f"(ends at byte offset {com_end}) - leaving it completely untouched.")

    # Genuine random payload appended PAST the legitimate JPEG EOI marker.
    # WildGuard's own strip logic (splice out only the first COM segment,
    # keep literally everything else) means these bytes survive straight
    # into the "recovered pre-embedding artifact" it rehashes.
    rng = random.Random()
    payload = bytes(rng.getrandbits(8) for _ in range(256))
    tampered = raw_bytes + payload
    print(f"[tamper] Appended {len(payload)} random bytes after the JPEG EOI marker "
          f"(this is the classic 'appended payload past EOF' covert-channel pattern "
          f"WildGuard's own Category-E steganography detector is built to look for).")
    return tampered


def insert_png_tamper_chunk(raw_bytes: bytes) -> bytes:
    idx = raw_bytes.rfind(b"IEND")
    if idx == -1:
        raise ValueError("Not a valid PNG (missing IEND chunk).")
    iend_chunk_start = idx - 4  # 4-byte length field precedes the 4-byte "IEND" type

    data = TAMPER_PNG_KEYWORD + b"\x00" + b"this-chunk-was-not-here-at-sanitization-time"
    chunk = (
        len(data).to_bytes(4, "big") + b"tEXt" + data
        + zlib.crc32(b"tEXt" + data).to_bytes(4, "big")
    )
    return raw_bytes[:iend_chunk_start] + chunk + raw_bytes[iend_chunk_start:]


def has_png_provenance_chunk(raw_bytes: bytes) -> bool:
    pos = 8
    n = len(raw_bytes)
    while pos + 8 <= n:
        length = int.from_bytes(raw_bytes[pos:pos + 4], "big")
        ctype = raw_bytes[pos + 4:pos + 8]
        data_start = pos + 8
        data_end = data_start + length
        if data_end > n:
            break
        if ctype == b"tEXt":
            data = raw_bytes[data_start:data_end]
            key, _, _ = data.partition(b"\x00")
            if key == b"wildguard_provenance":
                return True
        pos = data_end + 4
        if ctype == b"IEND":
            break
    return False


def tamper_png(raw_bytes: bytes) -> bytes:
    if not has_png_provenance_chunk(raw_bytes):
        raise ValueError(
            "No embedded provenance (wildguard_provenance tEXt chunk) found in this "
            "file. This script only tampers images that WildGuard has already "
            "sanitized and stamped with provenance - run WildGuard's Analyse page on "
            "an unsafe image first, then point this script at the sanitized output."
        )
    print("[tamper] Found the embedded wildguard_provenance tEXt chunk - leaving it "
          "completely untouched.")
    tampered = insert_png_tamper_chunk(raw_bytes)
    print(f"[tamper] Inserted a new, differently-keyworded tEXt chunk "
          f"('{TAMPER_PNG_KEYWORD.decode()}') just before IEND - a stand-in for "
          f"undetected content changes slipped in after sanitization.")
    return tampered


def main():
    parser = argparse.ArgumentParser(
        description="Tamper with a WildGuard-sanitized image's content while leaving "
                    "its embedded provenance record intact, to demonstrate tamper "
                    "detection on WildGuard's Version page."
    )
    parser.add_argument("image", help="Path to a WildGuard-sanitized image (has embedded provenance).")
    parser.add_argument("--out", default=None, help="Output path (default: <name>_TAMPERED.<ext> next to input).")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f"File not found: {args.image}")

    with open(args.image, "rb") as f:
        raw_bytes = f.read()

    fmt = detect_format(raw_bytes)
    print(f"[tamper] Detected format: {fmt}")

    if fmt == "JPEG":
        tampered_bytes = tamper_jpeg(raw_bytes)
        default_ext = "jpg"
    else:
        tampered_bytes = tamper_png(raw_bytes)
        default_ext = "png"

    if args.out:
        out_path = args.out
    else:
        base, _ = os.path.splitext(args.image)
        out_path = f"{base}_TAMPERED.{default_ext}"

    with open(out_path, "wb") as f:
        f.write(tampered_bytes)

    print(f"\n[tamper] Done. Wrote tampered file -> {out_path}")
    print("[tamper] Upload this file to WildGuard's Version page and click "
          "'Verify Provenance'.")
    print("[tamper] Expected result: PROVENANCE COMPROMISED (not 'no provenance found') - ")
    print("[tamper]   the embedded record is still there and readable, but the")
    print("[tamper]   recomputed content hash no longer matches the hash recorded")
    print("[tamper]   inside it, proving the file was modified after sanitization.")


if __name__ == "__main__":
    main()
