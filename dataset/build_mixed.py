"""
WildGuard - dataset/build_mixed.py

The merged/ directory (unlabelled/mislabelled images) is reserved
exclusively for the `mixed` dataset. Unlike train/validate/test, there is
no random distribution draw here - every single image gets ALL FIVE
categories (A, B, C, D, E) stacked, producing a maximally-contaminated
adversarial example. This set is for separate evaluation/stress-testing
only and is never used for training or the normal train/test split.
"""

import os
import json
import random
from typing import List, Dict

from dataset.inject_metadata import inject_metadata_jpeg, inject_metadata_png, random_field_subset
from dataset.inject_ocr import inject_ocr_text
from dataset.inject_context import generate_leaky_tag, generate_leaky_caption
from dataset.inject_stego import inject_lsb_payload, append_trailing_bytes
from dataset.generate_manifest import ManifestRow
from features.ingest import ingest_image
from features.species_features import extract_species_features

MIXED_SPECIES_LABEL = "unknown"  # merged/ is unlabelled per spec


def build_mixed_dataset(
    merged_paths: List[str], seed: int, species_table: Dict[str, float], out_root: str
) -> List[ManifestRow]:
    os.makedirs(out_root, exist_ok=True)
    rows: List[ManifestRow] = []
    species_sensitivity = extract_species_features(MIXED_SPECIES_LABEL, species_table).sensitivity

    for i, src_path in enumerate(merged_paths, start=1):
        rng = random.Random(seed * 7_919 + i)  # deterministic per-image, distinct stream from main split
        try:
            img, raw_bytes, src_fmt = ingest_image(src_path)
        except Exception as e:
            print(f"  [skip] {src_path}: {e}")
            continue

        target_fmt = src_fmt
        ext = "jpg" if target_fmt == "JPEG" else "png"
        work_img = img.convert("RGB")

        # Category B: OCR text
        work_img = inject_ocr_text(work_img, rng, num_tokens=rng.randint(2, 3))
        # Category E (part 1): LSB payload
        work_img = inject_lsb_payload(work_img, rng, coverage=rng.uniform(0.5, 0.95))

        # Category A: metadata (full intensity - every sub-field on)
        fields = random_field_subset(rng, intensity="full")
        if target_fmt == "JPEG":
            final_bytes = inject_metadata_jpeg(work_img, raw_bytes, fields, rng)
        else:
            final_bytes = inject_metadata_png(work_img, fields, rng)

        # Category E (part 2): trailing bytes, always present in mixed
        trailing_count = rng.randint(500, 4000)
        final_bytes = append_trailing_bytes(final_bytes, target_fmt, rng, trailing_count)

        # Category C: leaky filename + caption
        leaky_tag = generate_leaky_tag(rng)
        caption = generate_leaky_caption(rng, MIXED_SPECIES_LABEL)

        filename = f"mixed_{i}_all_{leaky_tag}.{ext}"
        dest_path = os.path.join(out_root, filename)
        with open(dest_path, "wb") as f:
            f.write(final_bytes)

        injection_params = {
            "category_a_fields": fields,
            "trailing_bytes": trailing_count,
            "caption": caption,
            "ocr_tokens": 2,
        }
        rows.append(ManifestRow(
            image_id=f"mixed_{i}",
            original_path=src_path,
            new_path=dest_path,
            animal=MIXED_SPECIES_LABEL,
            split="mixed",
            feature_category="mixed_all",
            feature_type=f"all_{leaky_tag}",
            injected=True,
            injection_parameters=json.dumps(injection_params),
            risk_label=1,
            species_sensitivity=species_sensitivity,
            caption=caption,
        ))

    return rows
