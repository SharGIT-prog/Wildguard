"""
WildGuard - features/ocr_features.py

Category B: visible textual leaks. Deterministic OCR only (Tesseract via
pytesseract). Produces detected text, confidence, bounding boxes, and
regex-based matches for GPS-coordinate stamps, camera-trap IDs, and
zone/checkpost keywords. Raw OCR text is NEVER fed directly into the model -
only the derived numeric/boolean features below are.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import os
import sys

from PIL import Image

try:
    import pytesseract

    def _auto_configure_tesseract_windows():
        """
        The single most common Windows gotcha with pytesseract: the
        tesseract.exe binary isn't on PATH, so pytesseract can't find it
        even though it's installed. Honor an explicit override via the
        WILDGUARD_TESSERACT_CMD env var first, then probe the standard
        Windows install locations before giving up.
        """
        override = os.environ.get("WILDGUARD_TESSERACT_CMD")
        if override and os.path.isfile(override):
            pytesseract.pytesseract.tesseract_cmd = override
            return
        if sys.platform.startswith("win"):
            for candidate in (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ):
                if os.path.isfile(candidate):
                    pytesseract.pytesseract.tesseract_cmd = candidate
                    return

    _auto_configure_tesseract_windows()
    _TESSERACT_AVAILABLE = True
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        _TESSERACT_AVAILABLE = False
except ImportError:
    pytesseract = None
    _TESSERACT_AVAILABLE = False

# Coordinate regex: supports "12°34'56.7\"N" and practical variants
# (space tolerance, optional seconds, optional minute/second quote-mark
# style - OCR frequently misreads a double-quote as a single apostrophe,
# or drops the minute-mark tick entirely, so both are treated as optional).
GPS_COORD_REGEX = re.compile(
    r"\d{1,3}\s*°\s*\d{1,2}\s*['\u2019]?\s*[\d.]*\s*[\"'\u2019]?\s*[NSEW]", re.IGNORECASE
)
CAMTRAP_ID_REGEX = re.compile(r"\b(CAM|SITE)[-_ ]?[A-Z0-9]{2,10}\b", re.IGNORECASE)
ZONE_KEYWORD_REGEX = re.compile(
    r"\b(zone\s*\d+|checkpost|check\s*post|gate\s*\d*|beat\s*\d+|milestone|forest\s*range|watch\s*tower)\b",
    re.IGNORECASE,
)


@dataclass
class OCRHit:
    label: str
    text: str
    bbox: List[int]  # [x, y, w, h]
    confidence: float


@dataclass
class OCRResult:
    engine_available: bool
    raw_text: str = ""
    hits: List[OCRHit] = field(default_factory=list)
    mean_confidence: float = 0.0
    num_text_boxes: int = 0
    gps_regex_match: bool = False
    camtrap_id_match: bool = False
    location_keyword_match: bool = False
    suspicious_token_count: int = 0

    def to_feature_dict(self) -> Dict[str, float]:
        return {
            "ocr_text_detected": float(bool(self.raw_text.strip())),
            "ocr_mean_confidence": float(self.mean_confidence) / 100.0 if self.mean_confidence else 0.0,
            "ocr_num_text_boxes_norm": min(self.num_text_boxes / 10.0, 1.0),
            "ocr_gps_regex_match": float(self.gps_regex_match),
            "ocr_camtrap_id_match": float(self.camtrap_id_match),
            "ocr_location_keyword_match": float(self.location_keyword_match),
            "ocr_suspicious_token_count_norm": min(self.suspicious_token_count / 5.0, 1.0),
        }


def _locate_token_bbox(words, boxes, target_tokens) -> Optional[Tuple[int, int, int, int]]:
    if not target_tokens:
        return None
    tlen = len(target_tokens)
    norm_target = "".join(t.strip(".,\"'").lower() for t in target_tokens)
    for i in range(len(words) - tlen + 1):
        norm_window = "".join(w.strip(".,\"'").lower() for w in words[i:i + tlen])
        if norm_target and (norm_target in norm_window or norm_window in norm_target):
            xs = [boxes[j][0] for j in range(i, i + tlen)]
            ys = [boxes[j][1] for j in range(i, i + tlen)]
            xe = [boxes[j][0] + boxes[j][2] for j in range(i, i + tlen)]
            ye = [boxes[j][1] + boxes[j][3] for j in range(i, i + tlen)]
            return (min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys))
    return None


def extract_ocr_features(img) -> OCRResult:
    if not _TESSERACT_AVAILABLE:
        return OCRResult(engine_available=False)

    words, boxes, confs = [], [], []
    try:
        work = img.convert("L")
        if min(work.size) < 500:
            work = work.resize((work.width * 2, work.height * 2), Image.LANCZOS)
        scale = work.width / img.width

        # Multiple page-segmentation modes catch different layouts of
        # sparse, scattered stamped text; results are merged/deduped since
        # any single mode alone misses some real, visible text.
        seen = set()
        for psm in (11, 6, 3):
            try:
                data = pytesseract.image_to_data(work, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
            except Exception:
                continue
            n = len(data.get("text", []))
            for i in range(n):
                text = (data["text"][i] or "").strip()
                if not text:
                    continue
                box = (int(data["left"][i] / scale), int(data["top"][i] / scale),
                       int(data["width"][i] / scale), int(data["height"][i] / scale))
                key = (text.lower(), box[0] // 10, box[1] // 10)  # dedupe near-identical detections
                if key in seen:
                    continue
                seen.add(key)
                words.append(text)
                boxes.append(box)
                try:
                    c = float(data["conf"][i])
                except (ValueError, TypeError):
                    c = -1.0
                if c >= 0:
                    confs.append(c)
    except Exception:
        return OCRResult(engine_available=False)

    full_text = " ".join(words)
    result = OCRResult(engine_available=True, raw_text=full_text)
    result.num_text_boxes = len(words)
    result.mean_confidence = sum(confs) / len(confs) if confs else 0.0

    suspicious_count = 0
    for m in GPS_COORD_REGEX.finditer(full_text):
        result.gps_regex_match = True
        suspicious_count += 1
        bbox = _locate_token_bbox(words, boxes, m.group(0).split())
        result.hits.append(OCRHit("gps_coordinate_stamp", m.group(0), list(bbox) if bbox else [0, 0, 0, 0],
                                   result.mean_confidence))
    for m in CAMTRAP_ID_REGEX.finditer(full_text):
        result.camtrap_id_match = True
        suspicious_count += 1
        bbox = _locate_token_bbox(words, boxes, m.group(0).split())
        result.hits.append(OCRHit("camera_trap_id", m.group(0), list(bbox) if bbox else [0, 0, 0, 0],
                                   result.mean_confidence))
    for m in ZONE_KEYWORD_REGEX.finditer(full_text):
        result.location_keyword_match = True
        suspicious_count += 1
        bbox = _locate_token_bbox(words, boxes, m.group(0).split())
        result.hits.append(OCRHit("zone_marker", m.group(0), list(bbox) if bbox else [0, 0, 0, 0],
                                   result.mean_confidence))

    result.suspicious_token_count = suspicious_count
    return result
