"""
WildGuard - features/context_features.py

Category C: context / filename / caption leaks. The filename is analyzed
SEPARATELY from any caption/comment/tag text. Deterministic regex + keyword
dictionary only - no ML, no raw text fed into the model (only derived
numeric scores/flags).
"""

import os
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List

LOCATION_KEYWORDS = [
    "zone", "range", "reserve", "sanctuary", "national park", "core area",
    "buffer zone", "beat", "compartment", "grid", "camp", "outpost",
    "waterhole", "salt lick", "den", "nest site", "burrow", "checkpost",
    "checkpoint", "gate",
]

DISTANCE_FROM_REGEX = re.compile(
    r"\d+(\.\d+)?\s*(km|kms|kilometers?|m|meters?|miles?|mi)\s+(from|near|away\s+from|of)\s+",
    re.IGNORECASE,
)
GENERIC_FILENAME_REGEX = re.compile(r"^(IMG|DSC|DCIM|PXL|PHOTO|MVIMG|animal)[\-_ ]?\d+", re.IGNORECASE)
COORD_DIGIT_REGEX = re.compile(r"\d{1,3}[._]\d{3,}")
DECIMAL_LATLON_REGEX = re.compile(r"\b\d{1,2}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}\b")

KNOWN_RESERVE_FRAGMENTS = [
    "bandipur", "nagarhole", "kanha", "corbett", "sundarbans", "kaziranga",
    "periyar", "gir", "ranthambore", "tadoba", "mudumalai", "wayanad",
    "bandhavgarh", "pench", "similipal",
]


@dataclass
class ContextResult:
    filename_leak_score: float = 0.0
    caption_leak_score: float = 0.0
    filename_has_location_keyword: bool = False
    filename_has_distance_pattern: bool = False
    filename_has_coordinate_pattern: bool = False
    filename_has_species_keyword: bool = False
    caption_has_location_keyword: bool = False
    caption_has_distance_pattern: bool = False
    caption_has_coordinate_pattern: bool = False
    filename_length: int = 0
    filename_entropy: float = 0.0

    def to_feature_dict(self) -> Dict[str, float]:
        return {
            "filename_leak_score": float(self.filename_leak_score),
            "caption_leak_score": float(self.caption_leak_score),
            "filename_has_location_keyword": float(self.filename_has_location_keyword),
            "filename_has_distance_pattern": float(self.filename_has_distance_pattern),
            "filename_has_coordinate_pattern": float(self.filename_has_coordinate_pattern),
            "caption_has_location_keyword": float(self.caption_has_location_keyword),
            "caption_has_distance_pattern": float(self.caption_has_distance_pattern),
            "caption_has_coordinate_pattern": float(self.caption_has_coordinate_pattern),
            "filename_entropy_norm": min(self.filename_entropy / 4.5, 1.0),
        }


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def extract_context_features(filename: str, caption: str = None, species_names: List[str] = None) -> ContextResult:
    result = ContextResult()
    caption = caption or ""
    species_names = species_names or []

    name_no_ext = os.path.splitext(filename or "")[0]
    lower_name = name_no_ext.lower()
    result.filename_length = len(name_no_ext)
    result.filename_entropy = _shannon_entropy(name_no_ext)

    fscore = 0.0
    if GENERIC_FILENAME_REGEX.match(name_no_ext):
        fscore = 0.02
    else:
        fscore += 0.12

    for frag in KNOWN_RESERVE_FRAGMENTS:
        if frag in lower_name:
            fscore += 0.45
            result.filename_has_location_keyword = True
            break
    if re.search(r"\bzone\s*\d+\b", lower_name) or re.search(r"\bbeat\s*\d+\b", lower_name):
        fscore += 0.30
        result.filename_has_location_keyword = True
    for kw in LOCATION_KEYWORDS:
        if kw.replace(" ", "") in lower_name.replace(" ", "").replace("_", "").replace("-", ""):
            fscore += 0.10
            result.filename_has_location_keyword = True
            break
    if DISTANCE_FROM_REGEX.search(name_no_ext.replace("_", " ")):
        fscore += 0.25
        result.filename_has_distance_pattern = True
    if COORD_DIGIT_REGEX.search(lower_name):
        fscore += 0.20
        result.filename_has_coordinate_pattern = True
    for sp in species_names:
        if sp.lower() in lower_name:
            result.filename_has_species_keyword = True
            break
    result.filename_leak_score = min(fscore, 1.0)

    cscore = 0.0
    lower_caption = caption.lower()
    for frag in KNOWN_RESERVE_FRAGMENTS:
        if frag in lower_caption:
            cscore += 0.40
            result.caption_has_location_keyword = True
            break
    if DISTANCE_FROM_REGEX.search(caption):
        cscore += 0.35
        result.caption_has_distance_pattern = True
    hits = sum(1 for kw in LOCATION_KEYWORDS if kw in lower_caption)
    if hits:
        result.caption_has_location_keyword = True
    cscore += min(hits * 0.12, 0.36)
    if DECIMAL_LATLON_REGEX.search(caption):
        cscore += 0.45
        result.caption_has_coordinate_pattern = True
    result.caption_leak_score = min(cscore, 1.0)

    return result
