"""
WildGuard - sanitization/ocr_redaction.py

Blurs every OCR-detected text bounding box, regardless of whether that
specific text was individually classified as sensitive (defense-in-depth,
per spec section 23). If OCR found no boxes, this is a no-op.
"""

from typing import List
from PIL import Image, ImageFilter

from features.ocr_features import OCRHit


def blur_ocr_regions(img: Image.Image, hits: List[OCRHit], pad: int = 6) -> Image.Image:
    if not hits:
        return img
    out = img.convert("RGB")
    for hit in hits:
        x, y, w, h = hit.bbox
        if w <= 0 or h <= 0:
            continue
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(out.width, x + w + pad)
        y1 = min(out.height, y + h + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        region = out.crop((x0, y0, x1, y1))
        blurred = region.filter(ImageFilter.GaussianBlur(radius=max(10, min(w, h) // 2)))
        out.paste(blurred, (x0, y0))
    return out
