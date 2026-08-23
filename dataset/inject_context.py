"""
WildGuard - dataset/inject_context.py

Category C injector. Filename and caption/context leaks are properties of
how a photo is NAMED and DESCRIBED when shared - not artifacts embedded in
the image bytes. This mirrors the real dashboard flow (a user types an
optional caption when uploading; the filename is whatever they saved the
file as), so this injector simply generates the leaky filename/caption text
itself. Both travel through the manifest and are passed directly into
features/context_features.py at both training and inference time -
identical code path, no drift.
"""

import random
from typing import Tuple

RESERVES = ["bandipur", "nagarhole", "kanha", "corbett", "kaziranga",
            "periyar", "gir", "ranthambore", "tadoba", "wayanad"]
ZONE_WORDS = ["zone", "beat", "gate", "range"]
DISTANCE_UNITS = ["km", "m", "miles"]

BENIGN_PREFIXES = ["IMG", "DSC", "DCIM", "PXL", "animal"]


def generate_leaky_filename(rng: random.Random, species: str, serial: int, ext: str) -> str:
    reserve = rng.choice(RESERVES)
    pattern = rng.choice(["zone", "gate", "distance"])
    if pattern == "zone":
        tag = f"{rng.choice(ZONE_WORDS)}{rng.randint(1,9)}"
    elif pattern == "gate":
        tag = f"gate{rng.randint(1,6)}"
    else:
        tag = f"{rng.randint(1,9)}{rng.choice(DISTANCE_UNITS)}_from_checkpoint"
    return f"{species}_{reserve}_{tag}_{serial}.{ext}"


def generate_benign_filename(rng: random.Random, species: str, serial: int, ext: str) -> str:
    prefix = rng.choice(BENIGN_PREFIXES)
    return f"{prefix}_{rng.randint(1000,9999)}_{serial}.{ext}"


def generate_leaky_tag(rng: random.Random) -> str:
    """
    Just the descriptive leaky portion (no species/serial prefix - the
    caller composes the full {animal}_{serial}_{feature_type} filename per
    the mandatory naming convention). e.g. "bandipur_zone4", "gate3",
    "5km_from_checkpoint".
    """
    reserve = rng.choice(RESERVES)
    pattern = rng.choice(["zone", "gate", "distance"])
    if pattern == "zone":
        return f"{reserve}_{rng.choice(ZONE_WORDS)}{rng.randint(1,9)}"
    elif pattern == "gate":
        return f"{reserve}_gate{rng.randint(1,6)}"
    else:
        return f"{rng.randint(1,9)}{rng.choice(DISTANCE_UNITS)}_from_checkpoint"


def generate_leaky_caption(rng: random.Random, species: str) -> str:
    reserve = rng.choice(RESERVES)
    distance = rng.randint(1, 9)
    unit = rng.choice(DISTANCE_UNITS)
    templates = [
        f"Spotted this {species} {distance} {unit} from the {reserve.title()} core zone checkpost",
        f"{species.title()} sighting near {reserve.title()} beat {rng.randint(1,20)} waterhole",
        f"{species.title()} seen {distance} {unit} from forest range outpost, {reserve.title()} sanctuary",
    ]
    return rng.choice(templates)


def generate_benign_caption(rng: random.Random, species: str) -> str:
    templates = [
        "Beautiful morning in the forest",
        "Amazing wildlife encounter today",
        "Captured this shot on a recent trip",
        "Nature is incredible",
        f"Lovely {species} photo",
    ]
    return rng.choice(templates)
