"""
WildGuard - dataset/generate_manifest.py

Defines the manifest row schema and writes it to both CSV and JSON. The
manifest is the machine-readable record of every image in the generated
dataset dump (data/all/...): where it came from, what split it's in, what
was injected, and its ground-truth risk label. This manifest is what
model/train.py reads - it never re-derives labels from folder names or
filenames at training time, only from this manifest, so label logic lives
in exactly one place (the injector, at generation time).
"""

import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class ManifestRow:
    image_id: str
    original_path: str
    new_path: str
    animal: str
    split: str                    # train | validate | test | mixed
    feature_category: str         # clean | category_a | category_b | category_c |
                                   # hard_negative_d | category_e | multi | mixed_all
    feature_type: str             # short filename tag, e.g. "locationcoordinates"
    injected: bool
    injection_parameters: str     # JSON string of exact params used (reproducibility)
    risk_label: int               # 0 = safe, 1 = leakage risk (ground truth for XGBoost)
    species_sensitivity: float
    caption: str = ""


def write_manifest(rows: List[ManifestRow], csv_path: str, json_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    fieldnames = list(ManifestRow.__dataclass_fields__.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(row) for row in rows], f, indent=2)


def append_manifest_csv(rows: List[ManifestRow], csv_path: str, write_header: bool) -> None:
    """Used for incremental writes during large dataset generation runs."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(ManifestRow.__dataclass_fields__.keys())
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
