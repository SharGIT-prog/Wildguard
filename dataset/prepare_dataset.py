"""
WildGuard - dataset/prepare_dataset.py

Phases 1-2 of the implementation order: discovers the source dataset,
performs a deterministic per-species stratified split at the ORIGINAL-image
level (before any synthetic variant is generated - this is what prevents
near-duplicate variants of the same source photo from crossing train/
validate/test boundaries), then applies exactly ONE synthetic-injection
profile per image (clean / category A / B / C / hard-negative-D / E /
multi), writes the result into data/all/{split}/{species}/ following the
mandatory {animal}_{serial}_{feature_type}.{ext} naming convention, and
records everything in the manifest.

merged/ is discovered separately and handed to dataset/build_mixed.py -
it never participates in this split.

Usage:
    python -m dataset.prepare_dataset --animals-root "archive/animals/animals"
    python -m dataset.prepare_dataset --animals-root "..." --move   # see NOTE below

NOTE on "move, not copy" (spec section 4): the spec asks that original
source images be moved rather than copied into the generated structure, to
avoid leaving duplicates. Since every generated variant is enriched with
metadata/pixel changes (even "clean" images are re-encoded with randomized
JPEG quality - see section 38 leakage-safety notes), the generated file's
bytes are never byte-identical to the source, so this is implemented as:
by default, the source file is left untouched (safe); passing --move
relocates (not deletes) the original into data/_consumed_originals/ once
its variant has been generated, satisfying "no duplicate left in the
working source tree" without irreversibly destroying the user's only copy
of their source dataset. This is a deliberate, documented deviation from a
literal delete-the-original reading, per the spec's own instruction to
prioritize security/data-safety when a requirement is ambiguous.
"""

import os
import io
import sys
import json
import random
import shutil
import argparse
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    resolve_animals_root, SPLIT_DIRS, MANIFEST_CSV_PATH, MANIFEST_JSON_PATH,
    RANDOM_SEED, SPLIT_RATIOS, INJECTION_DISTRIBUTION, HIGH_SENSITIVITY_THRESHOLD,
    SPECIES_SENSITIVITY_PATH, PROJECT_ROOT,
)
from dataset.discover import discover_species_folders, discover_merged_folder, resolve_merged_folder, print_stats, MERGED_DIRNAME
from dataset.generate_manifest import ManifestRow, write_manifest
from dataset.inject_metadata import inject_metadata_jpeg, inject_metadata_png, random_field_subset
from dataset.inject_ocr import inject_ocr_text
from dataset.inject_context import (
    generate_leaky_tag, generate_leaky_caption, generate_benign_caption,
)
from dataset.inject_stego import inject_lsb_payload, append_trailing_bytes
from features.ingest import ingest_image
from features.species_features import load_species_table, extract_species_features

CONSUMED_ORIGINALS_DIR = os.path.join(PROJECT_ROOT, "data", "_consumed_originals")


# ==========================================================================
# Splitting (per-species, stratified, seeded, at the ORIGINAL-image level)
# ==========================================================================
def stratified_split(species_map: Dict[str, List[str]], ratios: Dict[str, float], seed: int):
    """
    For every species, shuffle deterministically and slice into
    train/validate/test according to `ratios`. Returns
    {split_name: [(species, path), ...]}.
    """
    rng = random.Random(seed)
    splits = {"train": [], "validate": [], "test": []}
    for species, paths in species_map.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["validate"]))
        train_paths = shuffled[:n_train]
        val_paths = shuffled[n_train:n_train + n_val]
        test_paths = shuffled[n_train + n_val:]
        for p in train_paths:
            splits["train"].append((species, p))
        for p in val_paths:
            splits["validate"].append((species, p))
        for p in test_paths:
            splits["test"].append((species, p))
    return splits


# ==========================================================================
# Injection category selection
# ==========================================================================
def draw_category(rng: random.Random, species_sensitivity: float) -> str:
    categories = list(INJECTION_DISTRIBUTION.keys())
    weights = list(INJECTION_DISTRIBUTION.values())
    choice = rng.choices(categories, weights=weights, k=1)[0]
    if choice == "hard_negative_d" and species_sensitivity < HIGH_SENSITIVITY_THRESHOLD:
        # Not eligible (species isn't sensitive enough to be a meaningful
        # hard negative) - fall back to clean.
        return "clean"
    return choice


# ==========================================================================
# Leakage-safety helpers (section 38): decouple irrelevant properties
# (format, JPEG quality) from the injected label so the model can't learn
# a trivial shortcut like "PNG => always leaked".
# ==========================================================================
def maybe_flip_format(fmt: str, rng: random.Random, prob: float = 0.12) -> str:
    if rng.random() < prob:
        return "PNG" if fmt == "JPEG" else "JPEG"
    return fmt


def plain_encode(img, fmt: str, rng: random.Random) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=rng.choice([78, 82, 85, 88, 90, 92, 95]))
    else:
        img.convert("RGB").save(buf, format="PNG", compress_level=rng.choice([3, 6, 9]))
    return buf.getvalue()


# ==========================================================================
# Per-image processing
# ==========================================================================
def process_single_image(
    src_path: str, species: str, split: str, serial: int, global_seed: int,
    species_table: Dict[str, float], out_root: str,
) -> Tuple[str, ManifestRow]:
    # Deterministic per-image RNG so the whole run is reproducible from
    # (global_seed, serial) regardless of processing order.
    rng = random.Random(global_seed * 1_000_003 + serial)

    img, raw_bytes, src_fmt = ingest_image(src_path)
    species_sensitivity = extract_species_features(species, species_table).sensitivity

    category = draw_category(rng, species_sensitivity)
    target_fmt = maybe_flip_format(src_fmt, rng)
    ext = "jpg" if target_fmt == "JPEG" else "png"

    work_img = img.convert("RGB")
    injection_params: Dict = {"category": category, "target_format": target_fmt}
    sub_categories_used = []  # for `multi`
    caption = generate_benign_caption(rng, species)
    feature_type = "clean"
    include_a = include_b = include_c = include_e = False

    if category == "category_a":
        include_a = True
    elif category == "category_b":
        include_b = True
    elif category == "category_c":
        include_c = True
    elif category == "category_e":
        include_e = True
    elif category == "hard_negative_d":
        pass  # no leak injection; species is already attached via manifest
    elif category == "multi":
        pool = ["a", "b", "c", "e"]
        n_pick = rng.randint(2, 3)
        picked = rng.sample(pool, n_pick)
        sub_categories_used = sorted(picked)
        include_a = "a" in picked
        include_b = "b" in picked
        include_c = "c" in picked
        include_e = "e" in picked
    # category == "clean": nothing set

    # --- Pixel-level modifications first (order matters: OCR text and LSB
    # payload both touch pixels; metadata injection re-encodes afterward) ---
    tag_parts = []
    if include_b:
        work_img = inject_ocr_text(work_img, rng)
        tag_parts.append("ocr")
    if include_e:
        work_img = inject_lsb_payload(work_img, rng)
        tag_parts.append("stego")

    # --- Encode (with or without Category-A metadata) ---
    if include_a:
        fields = random_field_subset(rng, intensity="partial")
        injection_params["category_a_fields"] = fields
        if target_fmt == "JPEG":
            final_bytes = inject_metadata_jpeg(work_img, raw_bytes, fields, rng)
        else:
            final_bytes = inject_metadata_png(work_img, fields, rng)
        if fields.get("gps"):
            tag_parts.insert(0, "gps")
        elif fields.get("camera"):
            tag_parts.insert(0, "camera")
        elif fields.get("serial") or fields.get("owner"):
            tag_parts.insert(0, "device")
        elif fields.get("description") or fields.get("keywords"):
            tag_parts.insert(0, "iptc")
        else:
            tag_parts.insert(0, "metadata")
    else:
        final_bytes = plain_encode(work_img, target_fmt, rng)

    # --- Trailing bytes (Category E secondary signal), byte-level, after encode ---
    if include_e and rng.random() < 0.7:
        trailing_count = rng.randint(200, 3000)
        final_bytes = append_trailing_bytes(final_bytes, target_fmt, rng, trailing_count)
        injection_params["trailing_bytes"] = trailing_count

    # --- Filename / caption (Category C) ---
    if include_c:
        leaky_tag = generate_leaky_tag(rng)
        caption = generate_leaky_caption(rng, species)
        injection_params["caption"] = caption
        feature_type = leaky_tag
    elif category == "hard_negative_d":
        feature_type = "species_sensitive_clean"
    elif category == "clean":
        feature_type = "clean"
    elif category == "multi":
        feature_type = "multi_" + "_".join(sub_categories_used)
        if include_c:
            pass  # already set above via leaky_tag path (include_c true covers this)
    else:
        feature_type = tag_parts[0] if tag_parts else category

    # Multi with include_c needs the leaky tag folded into feature_type too
    if category == "multi" and include_c:
        feature_type = "multi_" + "_".join(sub_categories_used)

    # Human-friendlier canonical tags matching the spec's own examples
    canonical_map = {
        "gps": "locationcoordinates", "camera": "camera_metadata",
        "device": "device_metadata", "iptc": "iptc_metadata",
        "ocr": "ocrtext", "stego": "steganography",
    }
    if category in ("category_a", "category_b", "category_e") and tag_parts:
        feature_type = canonical_map.get(tag_parts[0], tag_parts[0])

    risk_label = 1 if category in ("category_a", "category_b", "category_c", "category_e", "multi") else 0

    filename = f"{species}_{serial}_{feature_type}.{ext}"
    dest_dir = os.path.join(out_root, species)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    with open(dest_path, "wb") as f:
        f.write(final_bytes)

    row = ManifestRow(
        image_id=f"{split}_{serial}",
        original_path=src_path,
        new_path=dest_path,
        animal=species,
        split=split,
        feature_category=category,
        feature_type=feature_type,
        injected=category != "clean" and category != "hard_negative_d",
        injection_parameters=json.dumps(injection_params),
        risk_label=risk_label,
        species_sensitivity=species_sensitivity,
        caption=caption if include_c else "",
    )
    return dest_path, row


def relocate_original(src_path: str, species: str, move: bool):
    if not move:
        return
    dest_dir = os.path.join(CONSUMED_ORIGINALS_DIR, species)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src_path))
    try:
        shutil.move(src_path, dest)
    except Exception as e:
        print(f"  [warn] could not relocate original {src_path}: {e}")


# ==========================================================================
# Orchestration
# ==========================================================================
def prepare(animals_root: str, seed: int, move: bool, merged_root: str = None, progress_every: int = 500):
    species_table = load_species_table(SPECIES_SENSITIVITY_PATH)

    print(f"[WildGuard] Phase 1: discovering species folders (excludes merged/) ...")
    print(f"[WildGuard] animals-root (absolute): {os.path.abspath(animals_root)}")
    species_map = discover_species_folders(animals_root)
    merged_dir = resolve_merged_folder(animals_root, merged_root)
    merged_files = discover_merged_folder(animals_root, merged_root)
    if merged_dir:
        print(f"[WildGuard] merged/ resolved to (absolute): {os.path.abspath(merged_dir)}")
    else:
        print("[WildGuard] merged/ NOT found. Checked:")
        print(f"    {os.path.abspath(os.path.join(animals_root, MERGED_DIRNAME))}  (nested)")
        print(f"    {os.path.abspath(os.path.join(os.path.dirname(animals_root), MERGED_DIRNAME))}  (sibling)")
        if merged_root:
            print(f"    {os.path.abspath(merged_root)}  (--merged-root)")
        print("    If your merged/ folder is somewhere else, pass --merged-root explicitly.")
    print_stats(species_map, merged_files)

    print("\n[WildGuard] Phase 2: stratified split (train/validate/test), seed =", seed)
    splits = stratified_split(species_map, SPLIT_RATIOS, seed)
    for split_name, items in splits.items():
        print(f"  {split_name:10s} {len(items)} original images")

    print("\n[WildGuard] Phase 2: injecting synthetic features + writing dataset dump ...")
    all_rows: List[ManifestRow] = []
    serial = 0
    for split_name, items in splits.items():
        out_root = SPLIT_DIRS[split_name]
        for i, (species, src_path) in enumerate(items, start=1):
            serial += 1
            try:
                dest_path, row = process_single_image(
                    src_path, species, split_name, serial, seed, species_table, out_root
                )
            except Exception as e:
                print(f"  [skip] {src_path}: {e}")
                continue
            all_rows.append(row)
            relocate_original(src_path, species, move)
            if i % progress_every == 0:
                print(f"  {split_name}: {i}/{len(items)} processed")

    print(f"\n[WildGuard] Phase 2 complete: {len(all_rows)} images generated across train/validate/test.")

    # Mixed dataset (merged/, ALL categories A-E stacked) is handled by a
    # dedicated module since its rules are entirely different (no split,
    # no distribution draw - every image gets everything).
    if merged_files:
        from dataset.build_mixed import build_mixed_dataset
        print(f"\n[WildGuard] Building mixed/ dataset from {len(merged_files)} merged/ images "
              f"(ALL categories A-E applied to every image) ...")
        mixed_rows = build_mixed_dataset(merged_files, seed, species_table, SPLIT_DIRS["mixed"])
        all_rows.extend(mixed_rows)
        print(f"[WildGuard] mixed/ complete: {len(mixed_rows)} images.")
    else:
        print("\n[WildGuard] No merged/ directory found - skipping mixed/ dataset.")

    write_manifest(all_rows, MANIFEST_CSV_PATH, MANIFEST_JSON_PATH)
    print(f"\n[WildGuard] Manifest written -> {MANIFEST_CSV_PATH}  ({len(all_rows)} rows)")

    # Summary
    from collections import Counter
    cat_counts = Counter(r.feature_category for r in all_rows if r.split != "mixed")
    print("\nInjection category distribution (train+validate+test):")
    for cat, count in cat_counts.most_common():
        print(f"  {cat:20s} {count}")


def main():
    parser = argparse.ArgumentParser(description="WildGuard dataset preparation (Phases 1-2)")
    parser.add_argument("--animals-root", default=None)
    parser.add_argument("--merged-root", default=None,
                         help="Explicit path to merged/ if it's neither nested inside "
                              "--animals-root nor its sibling directory.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--move", action="store_true",
                         help="Relocate consumed originals to data/_consumed_originals/ "
                              "instead of leaving them in place (see module docstring).")
    args = parser.parse_args()

    root = resolve_animals_root(args.animals_root)
    print(f"[WildGuard] animals-root: {root}")
    prepare(root, args.seed, args.move, merged_root=args.merged_root)


if __name__ == "__main__":
    main()
