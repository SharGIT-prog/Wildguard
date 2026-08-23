"""
WildGuard - features/metadata_features.py

Category A: explicit metadata leaks. Parses EXIF (JPEG) and ancillary chunks
(PNG) and returns a structured result distinguishing:
  1. metadata field exists (presence)
  2. metadata field's presence is itself sensitive (e.g. GPS, serial number)
  3. content-level sensitivity (e.g. a Description field that CONTAINS a
     location keyword vs one that's just present)

Deterministic, no ML. piexif for JPEG EXIF; a small hand-rolled PNG chunk
walker for PNG ancillary chunks (tEXt/zTXt/iTXt/eXIf).
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any

import piexif


@dataclass
class MetadataResult:
    format: str
    # presence flags (Category A "metadata field exists")
    has_gps: bool = False
    has_gps_altitude: bool = False
    has_gps_timestamp: bool = False
    has_datetime_original: bool = False
    has_camera_make: bool = False
    has_camera_model: bool = False
    has_serial_number: bool = False
    has_owner_name: bool = False
    has_iptc_location: bool = False
    has_iptc_creator: bool = False
    has_iptc_description: bool = False
    has_iptc_keywords: bool = False
    has_thumbnail: bool = False
    has_png_text_chunks: bool = False
    has_app_segments: bool = False
    # content-sensitivity flags (field present AND content is itself sensitive,
    # distinct from mere presence - e.g. a Description that mentions a zone)
    description_contains_location: bool = False
    keywords_contain_location: bool = False
    # counts (useful numeric features beyond booleans)
    gps_field_count: int = 0
    metadata_block_count: int = 0
    app_segment_count: int = 0
    png_chunk_count: int = 0
    # raw values kept for explanation/debugging (not fed to the model directly)
    gps_coordinates: Optional[Tuple[float, float]] = None
    raw_tags_found: List[str] = field(default_factory=list)

    def to_feature_dict(self) -> Dict[str, float]:
        """Numeric-only view suitable for the XGBoost feature vector."""
        return {
            "has_gps": float(self.has_gps),
            "has_gps_altitude": float(self.has_gps_altitude),
            "has_gps_timestamp": float(self.has_gps_timestamp),
            "has_datetime_original": float(self.has_datetime_original),
            "has_camera_make": float(self.has_camera_make),
            "has_camera_model": float(self.has_camera_model),
            "has_serial_number": float(self.has_serial_number),
            "has_owner_name": float(self.has_owner_name),
            "has_iptc_location": float(self.has_iptc_location),
            "has_iptc_creator": float(self.has_iptc_creator),
            "has_iptc_description": float(self.has_iptc_description),
            "has_iptc_keywords": float(self.has_iptc_keywords),
            "has_thumbnail": float(self.has_thumbnail),
            "has_png_text_chunks": float(self.has_png_text_chunks),
            "has_app_segments": float(self.has_app_segments),
            "description_contains_location": float(self.description_contains_location),
            "keywords_contain_location": float(self.keywords_contain_location),
            "gps_field_count_norm": min(self.gps_field_count / 4.0, 1.0),
            "metadata_block_count_norm": min(self.metadata_block_count / 10.0, 1.0),
            "app_segment_count_norm": min(self.app_segment_count / 5.0, 1.0),
            "png_chunk_count_norm": min(self.png_chunk_count / 10.0, 1.0),
        }


_LOCATION_HINT_WORDS = ("zone", "location", "gps", "km", "checkpost", "beat", "range", "reserve")


def _dms_to_decimal(dms, ref) -> Optional[float]:
    try:
        d = dms[0][0] / dms[0][1]
        m = dms[1][0] / dms[1][1]
        s = dms[2][0] / dms[2][1]
        value = d + m / 60.0 + s / 3600.0
        if ref in (b"S", b"W", "S", "W"):
            value = -value
        return value
    except Exception:
        return None


def extract_metadata_features(img, raw_bytes: bytes, fmt: str) -> MetadataResult:
    result = MetadataResult(format=fmt)

    if fmt == "JPEG":
        try:
            exif_dict = piexif.load(raw_bytes)
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        gps_ifd = exif_dict.get("GPS", {}) or {}
        zeroth_ifd = exif_dict.get("0th", {}) or {}
        exif_ifd = exif_dict.get("Exif", {}) or {}

        gps_field_count = 0
        if piexif.GPSIFD.GPSLatitude in gps_ifd and piexif.GPSIFD.GPSLongitude in gps_ifd:
            result.has_gps = True
            gps_field_count += 2
            lat = _dms_to_decimal(gps_ifd[piexif.GPSIFD.GPSLatitude], gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef))
            lon = _dms_to_decimal(gps_ifd[piexif.GPSIFD.GPSLongitude], gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef))
            if lat is not None and lon is not None:
                result.gps_coordinates = (lat, lon)
            result.raw_tags_found.append("GPSLatitude/GPSLongitude")
        if piexif.GPSIFD.GPSAltitude in gps_ifd:
            result.has_gps_altitude = True
            gps_field_count += 1
            result.raw_tags_found.append("GPSAltitude")
        if piexif.GPSIFD.GPSTimeStamp in gps_ifd or piexif.GPSIFD.GPSDateStamp in gps_ifd:
            result.has_gps_timestamp = True
            gps_field_count += 1
            result.raw_tags_found.append("GPSTimeStamp/GPSDateStamp")
        result.gps_field_count = gps_field_count

        if piexif.ExifIFD.DateTimeOriginal in exif_ifd or piexif.ExifIFD.DateTimeDigitized in exif_ifd:
            result.has_datetime_original = True
            result.raw_tags_found.append("DateTimeOriginal/DateTimeDigitized")

        if piexif.ImageIFD.Make in zeroth_ifd:
            result.has_camera_make = True
            result.raw_tags_found.append("Make")
        if piexif.ImageIFD.Model in zeroth_ifd:
            result.has_camera_model = True
            result.raw_tags_found.append("Model")
        if piexif.ExifIFD.LensSerialNumber in exif_ifd or piexif.ExifIFD.BodySerialNumber in exif_ifd:
            result.has_serial_number = True
            result.raw_tags_found.append("SerialNumber/LensSerialNumber")
        if piexif.ImageIFD.Artist in zeroth_ifd or piexif.ImageIFD.Copyright in zeroth_ifd:
            result.has_owner_name = True
            result.raw_tags_found.append("Artist/Copyright(OwnerName)")

        if piexif.ImageIFD.ImageDescription in zeroth_ifd:
            desc_val = zeroth_ifd[piexif.ImageIFD.ImageDescription]
            desc_str = desc_val.decode("utf-8", errors="ignore") if isinstance(desc_val, bytes) else str(desc_val)
            result.has_iptc_description = True
            result.raw_tags_found.append("ImageDescription")
            if any(k in desc_str.lower() for k in _LOCATION_HINT_WORDS):
                result.description_contains_location = True
                result.has_iptc_location = True

        thumb = exif_dict.get("thumbnail")
        if thumb:
            result.has_thumbnail = True
            result.raw_tags_found.append("EmbeddedThumbnail")

        # Lightweight XMP/IPTC scan within the raw APP1 bytes + count APPn segments
        app_segment_count = 0
        pos = 2
        while pos + 4 <= len(raw_bytes):
            if raw_bytes[pos] != 0xFF:
                break
            marker = raw_bytes[pos + 1]
            if marker == 0xD8 or marker == 0xD9:
                pos += 2
                continue
            if 0xE0 <= marker <= 0xEF:
                app_segment_count += 1
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                pos += 2
                continue
            if pos + 4 > len(raw_bytes):
                break
            seg_len = int.from_bytes(raw_bytes[pos + 2:pos + 4], "big")
            if seg_len < 2:
                break
            pos += 2 + seg_len
            if marker == 0xDA:  # start of scan - stop parsing segments
                break
        result.app_segment_count = app_segment_count
        result.has_app_segments = app_segment_count > 0

        lowered = raw_bytes[:200000].lower()
        if b"<iptc:location" in lowered or b"iptc:city" in lowered or b"photoshop:city" in lowered:
            result.has_iptc_location = True
            result.raw_tags_found.append("XMP:Location")
        if b"dc:creator" in lowered or b"xmp:creator" in lowered:
            result.has_iptc_creator = True
            result.raw_tags_found.append("XMP:Creator")
        if b"dc:subject" in lowered or b"iptc:keywords" in lowered:
            result.has_iptc_keywords = True
            if b"zone" in lowered or b"checkpost" in lowered or b"location" in lowered:
                result.keywords_contain_location = True
            result.raw_tags_found.append("XMP:Keywords")

        result.metadata_block_count = len(result.raw_tags_found)

    elif fmt == "PNG":
        pos = 8
        text_keys = []
        chunk_count = 0
        while pos + 8 <= len(raw_bytes):
            length = int.from_bytes(raw_bytes[pos:pos + 4], "big")
            ctype = raw_bytes[pos + 4:pos + 8]
            data_start = pos + 8
            data_end = data_start + length
            if data_end > len(raw_bytes):
                break
            data = raw_bytes[data_start:data_end]
            chunk_count += 1

            if ctype in (b"tEXt", b"zTXt", b"iTXt"):
                try:
                    key = data.split(b"\x00", 1)[0].decode("latin-1", errors="ignore")
                except Exception:
                    key = "?"
                text_keys.append(key.strip() or ctype.decode())
                blob = data.lower()
                if b"gps" in blob or b"location" in blob:
                    result.has_iptc_location = True
                    result.description_contains_location = True
                if b"author" in blob or b"creator" in blob or b"artist" in blob:
                    result.has_iptc_creator = True
                if b"description" in blob or b"comment" in blob:
                    result.has_iptc_description = True
                if b"keyword" in blob:
                    result.has_iptc_keywords = True
                    if b"zone" in blob or b"checkpost" in blob:
                        result.keywords_contain_location = True
                if b"owner" in blob:
                    result.has_owner_name = True
                if b"serial" in blob:
                    result.has_serial_number = True
                if b"make" in blob or b"model" in blob:
                    result.has_camera_make = True
                    result.has_camera_model = True

            if ctype == b"eXIf":
                result.raw_tags_found.append("eXIf chunk")
                try:
                    exif_dict = piexif.load(data)
                    gps_ifd = exif_dict.get("GPS", {}) or {}
                    if piexif.GPSIFD.GPSLatitude in gps_ifd:
                        result.has_gps = True
                        result.gps_field_count += 2
                        lat = _dms_to_decimal(gps_ifd[piexif.GPSIFD.GPSLatitude], gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef))
                        lon_dms = gps_ifd.get(piexif.GPSIFD.GPSLongitude, ((0, 1), (0, 1), (0, 1)))
                        lon = _dms_to_decimal(lon_dms, gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef))
                        if lat is not None and lon is not None:
                            result.gps_coordinates = (lat, lon)
                except Exception:
                    pass

            pos = data_end + 4
            if ctype == b"IEND":
                break

        result.has_png_text_chunks = len(text_keys) > 0
        result.png_chunk_count = chunk_count
        result.metadata_block_count = len(text_keys) + (1 if result.has_gps else 0)
        result.raw_tags_found.extend(text_keys)

    return result
