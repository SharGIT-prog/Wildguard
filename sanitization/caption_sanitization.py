"""
WildGuard - sanitization/caption_sanitization.py

Always sanitizes any provided caption/comment/tag text - removes or
replaces location-sensitive content, regardless of whether the caption was
individually flagged (defense-in-depth, per spec section 23).
"""

import re
from typing import Optional

from features.context_features import DISTANCE_FROM_REGEX, LOCATION_KEYWORDS, KNOWN_RESERVE_FRAGMENTS


def sanitize_caption(caption: Optional[str]) -> str:
    if not caption:
        return ""
    text = caption
    text = DISTANCE_FROM_REGEX.sub("[REDACTED DISTANCE] ", text)
    for frag in KNOWN_RESERVE_FRAGMENTS:
        text = re.sub(re.escape(frag), "[REDACTED LOCATION]", text, flags=re.IGNORECASE)
    for kw in LOCATION_KEYWORDS:
        text = re.sub(re.escape(kw), "[REDACTED]", text, flags=re.IGNORECASE)
    return text.strip()
