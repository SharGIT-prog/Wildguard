"""
WildGuard - dataset/inject_ocr.py

Category B injector. Renders real, readable text into image PIXELS (not a
metadata label) so the OCR detector at Level 3 has genuine text to find.
Position, token choice, and font size are randomized to prevent the model
(or a human skimming the dataset) from learning a trivial "text always
appears top-left in the same font" shortcut.
"""

import random
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

TEXT_TOKENS = [
    "ZONE 4", "CHECKPOST", "GATE 2", "BEAT 12", "CAM-123", "SITE-AB12",
    "12°34'56.7\"N", "77°34'12.3\"E", "ZONE 7", "GATE", "BEAT 5",
    "CAM-047", "SITE-XY9", "RESTRICTED AREA", "FOREST RANGE 3",
]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def inject_ocr_text(img: Image.Image, rng: random.Random, num_tokens: int = None) -> Image.Image:
    """Returns a NEW image with 1-3 random tokens rendered onto the pixels."""
    out = img.convert("RGB").copy()
    w, h = out.size
    draw = ImageDraw.Draw(out)

    if num_tokens is None:
        num_tokens = rng.randint(1, 3)
    tokens = rng.sample(TEXT_TOKENS, min(num_tokens, len(TEXT_TOKENS)))

    for token in tokens:
        font_size = rng.randint(max(20, h // 12), max(28, h // 7))
        font = _get_font(font_size)
        try:
            bbox = draw.textbbox((0, 0), token, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = font_size * len(token) // 2, font_size

        max_x = max(1, w - tw - 8)
        max_y = max(1, h - th - 8)
        x, y = rng.randint(0, max_x), rng.randint(0, max_y)

        # Solid high-contrast background box (always white-on-black,
        # regardless of underlying photo) for maximum OCR reliability.
        pad = 6
        draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0))
        draw.text((x, y), token, fill=(255, 255, 255), font=font)

    return out
