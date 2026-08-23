"""
WildGuard - dataset/inject_metadata.py

Category A injector. Writes REAL EXIF (JPEG) or ancillary chunk (PNG)
metadata into a copy of the source image - not a textual label claiming
metadata exists, but the actual artifact, so the detector genuinely learns
to recognize it during Level 3.

To avoid the model learning shortcuts (e.g. "GPS always co-occurs with this
exact camera model"), field SUBSETS, camera vendors, and GPS coordinates are
all randomized per call (see safety_against_leakage notes in prepare_dataset.py).
"""

import io
import random
import struct
from typing import Dict, Any, Tuple

import piexif
from PIL import Image

CAMERA_MAKES_MODELS = [
    ("Canon", "EOS 90D"), ("Canon", "EOS R5"), ("Nikon", "D850"),
    ("Nikon", "Z6 II"), ("Sony", "A7 IV"), ("Sony", "A9 II"),
    ("Olympus", "OM-D E-M1"), ("Fujifilm", "X-T4"), ("Panasonic", "Lumix GH6"),
]
OWNER_NAMES = ["J. Ranger", "Field Team 3", "K. Sharma", "Forest Watch Unit",
               "R. Verma", "Camp Station Alpha", "M. Patel"]

# Roughly bounds a real forest/reserve region so coordinates look plausible
LAT_RANGE = (8.0, 30.0)
LON_RANGE = (70.0, 90.0)


def _deg_to_dms_rational(deg: float):
    d = int(abs(deg))
    m_float = (abs(deg) - d) * 60
    m = int(m_float)
    s = (m_float - m) * 60
    return [(d, 1), (m, 1), (int(round(s * 100)), 100)]


def _random_gps() -> Tuple[float, float]:
    return (random.uniform(*LAT_RANGE), random.uniform(*LON_RANGE))


def inject_metadata_jpeg(img: Image.Image, raw_bytes: bytes, fields: Dict[str, bool],
                          rng: random.Random) -> bytes:
    """
    fields: dict of which sub-features to include, e.g.
        {"gps": True, "gps_altitude": True, "timestamp": False,
         "camera": True, "serial": False, "owner": False, "description": True}
    Returns new JPEG bytes with EXIF injected.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=rng.choice([80, 85, 88, 90, 92, 95]))
    base_bytes = buf.getvalue()

    zeroth_ifd, exif_ifd, gps_ifd = {}, {}, {}

    if fields.get("gps"):
        lat, lon = _random_gps()
        gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = b"N" if lat >= 0 else b"S"
        gps_ifd[piexif.GPSIFD.GPSLatitude] = _deg_to_dms_rational(lat)
        gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = b"E" if lon >= 0 else b"W"
        gps_ifd[piexif.GPSIFD.GPSLongitude] = _deg_to_dms_rational(lon)
        if fields.get("gps_altitude"):
            gps_ifd[piexif.GPSIFD.GPSAltitude] = (rng.randint(50, 2500), 1)
            gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 0
        if fields.get("gps_timestamp"):
            gps_ifd[piexif.GPSIFD.GPSTimeStamp] = [(rng.randint(0, 23), 1), (rng.randint(0, 59), 1), (0, 1)]
            gps_ifd[piexif.GPSIFD.GPSDateStamp] = f"2024:{rng.randint(1,12):02d}:{rng.randint(1,28):02d}"

    if fields.get("camera"):
        make, model = rng.choice(CAMERA_MAKES_MODELS)
        zeroth_ifd[piexif.ImageIFD.Make] = make.encode()
        zeroth_ifd[piexif.ImageIFD.Model] = model.encode()
    if fields.get("datetime_original"):
        exif_ifd[piexif.ExifIFD.DateTimeOriginal] = (
            f"2024:{rng.randint(1,12):02d}:{rng.randint(1,28):02d} "
            f"{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00"
        ).encode()
        exif_ifd[piexif.ExifIFD.DateTimeDigitized] = exif_ifd[piexif.ExifIFD.DateTimeOriginal]
    if fields.get("serial"):
        exif_ifd[piexif.ExifIFD.BodySerialNumber] = f"SN{rng.randint(10000000,99999999)}".encode()
    if fields.get("owner"):
        zeroth_ifd[piexif.ImageIFD.Artist] = rng.choice(OWNER_NAMES).encode()
    if fields.get("description"):
        desc = rng.choice([
            "Wildlife sighting near forest checkpost zone",
            "Camera trap capture, restricted area",
            "Field survey photograph",
        ])
        zeroth_ifd[piexif.ImageIFD.ImageDescription] = desc.encode()
    if fields.get("keywords"):
        # stashed inside a UserComment-like tag via 0th Software field as a
        # lightweight stand-in (keeps the EXIF writer dependency-free)
        zeroth_ifd[piexif.ImageIFD.Software] = b"keywords:zone,checkpost,restricted"

    exif_bytes = piexif.dump({"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": {}, "thumbnail": None})
    out = io.BytesIO()
    piexif.insert(exif_bytes, base_bytes, out)
    result = out.getvalue()

    if fields.get("thumbnail"):
        result = _reinsert_with_thumbnail(result, img)

    return result


def _reinsert_with_thumbnail(jpeg_bytes: bytes, img: Image.Image) -> bytes:
    try:
        exif_dict = piexif.load(jpeg_bytes)
        thumb_img = img.convert("RGB").copy()
        thumb_img.thumbnail((160, 160))
        thumb_buf = io.BytesIO()
        thumb_img.save(thumb_buf, format="JPEG")
        exif_dict["thumbnail"] = thumb_buf.getvalue()
        exif_bytes = piexif.dump(exif_dict)
        out = io.BytesIO()
        piexif.insert(exif_bytes, jpeg_bytes, out)
        return out.getvalue()
    except Exception:
        return jpeg_bytes


def inject_metadata_png(img: Image.Image, fields: Dict[str, bool], rng: random.Random) -> bytes:
    """Writes PNG tEXt chunks carrying equivalent metadata leak content."""
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()

    if fields.get("gps"):
        lat, lon = _random_gps()
        info.add_text("GPS", f"{lat:.6f},{lon:.6f}")
    if fields.get("camera"):
        make, model = rng.choice(CAMERA_MAKES_MODELS)
        info.add_text("Make", make)
        info.add_text("Model", model)
    if fields.get("serial"):
        info.add_text("SerialNumber", f"SN{rng.randint(10000000,99999999)}")
    if fields.get("owner"):
        info.add_text("Owner", rng.choice(OWNER_NAMES))
    if fields.get("description"):
        info.add_text("Description", "Wildlife sighting near forest checkpost zone")
    if fields.get("keywords"):
        info.add_text("Keywords", "zone,checkpost,restricted")

    buf = io.BytesIO()
    img.convert("RGBA" if img.mode == "RGBA" else "RGB").save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def random_field_subset(rng: random.Random, intensity: str = "full") -> Dict[str, bool]:
    """
    Randomized subset of Category-A sub-features. `intensity` controls how
    many fields tend to be set - "full" for the mixed/adversarial dataset,
    "partial" for normal single-category training variants (keeps positive
    examples from being trivially identical to each other).
    """
    all_fields = ["gps", "gps_altitude", "gps_timestamp", "camera",
                  "datetime_original", "serial", "owner", "description",
                  "keywords", "thumbnail"]
    if intensity == "full":
        return {f: True for f in all_fields}
    chosen = {}
    n_active = rng.randint(1, 4)
    active = rng.sample(all_fields, min(n_active, len(all_fields)))
    for f in all_fields:
        chosen[f] = f in active
    # gps_altitude/timestamp only meaningful if gps itself is on
    if not chosen["gps"]:
        chosen["gps_altitude"] = False
        chosen["gps_timestamp"] = False
    return chosen
