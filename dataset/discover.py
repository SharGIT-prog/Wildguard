"""
WildGuard - dataset/discover.py

Phase 1: dataset discovery and statistics. Walks the animals-root, excludes
the `merged` directory entirely (per spec - merged is reserved exclusively
for the `mixed` stress-test set), and reports per-species counts.

Run standalone for a quick report:
    python -m dataset.discover --animals-root "path/to/animals/animals"
"""

import os
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import resolve_animals_root, MERGED_DIRNAME

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def discover_species_folders(animals_root: str) -> dict:
    """Returns {species: [image_paths]}, EXCLUDING the merged/ directory."""
    if not os.path.isdir(animals_root):
        raise FileNotFoundError(f"animals-root not found: {animals_root}")

    species_map = {}
    for entry in sorted(os.listdir(animals_root)):
        if entry.lower() == MERGED_DIRNAME.lower():
            continue  # explicitly excluded from train/test/validate discovery
        class_dir = os.path.join(animals_root, entry)
        if not os.path.isdir(class_dir):
            continue
        files = [
            os.path.join(class_dir, f) for f in os.listdir(class_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if files:
            species_map[entry] = files
    return species_map


def resolve_merged_folder(animals_root: str, explicit: str = None) -> str:
    """
    Locates the merged/ directory. Tried in order:
      1. an explicitly-given path (--merged-root)
      2. animals_root/merged           (nested - what the original spec path showed)
      3. parent(animals_root)/merged   (sibling of the "animals" folder - common when
                                         the animals-root itself is named "animals",
                                         e.g. archive/animals/animals + archive/animals/merged)
    Returns the resolved path, or None if not found anywhere.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(os.path.join(animals_root, MERGED_DIRNAME))
    candidates.append(os.path.join(os.path.dirname(animals_root), MERGED_DIRNAME))
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None


def discover_merged_folder(animals_root: str, explicit: str = None) -> list:
    """Returns the list of image paths in merged/, or [] if absent anywhere."""
    merged_dir = resolve_merged_folder(animals_root, explicit)
    if merged_dir is None:
        return []
    return [
        os.path.join(merged_dir, f) for f in os.listdir(merged_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]


def print_stats(species_map: dict, merged_files: list) -> None:
    counts = {sp: len(files) for sp, files in species_map.items()}
    total = sum(counts.values())
    print(f"Species folders discovered: {len(species_map)}")
    print(f"Total labelled images (excludes merged/): {total}")
    if counts:
        avg = total / len(counts)
        print(f"  min per species: {min(counts.values())}   "
              f"max per species: {max(counts.values())}   avg: {avg:.1f}")
    print(f"merged/ (excluded from train/test/validate, used only for mixed): {len(merged_files)} images")
    print("\nPer-species counts:")
    for sp in sorted(counts):
        print(f"  {sp:20s} {counts[sp]}")


def main():
    parser = argparse.ArgumentParser(description="WildGuard Phase 1: dataset discovery")
    parser.add_argument("--animals-root", default=None)
    parser.add_argument("--merged-root", default=None,
                         help="Explicit path to merged/ if it isn't nested inside --animals-root "
                              "and isn't its sibling either.")
    args = parser.parse_args()

    root = resolve_animals_root(args.animals_root)
    print(f"animals-root (resolved, absolute): {os.path.abspath(root)}\n")
    species_map = discover_species_folders(root)
    merged_dir = resolve_merged_folder(root, args.merged_root)
    merged_files = discover_merged_folder(root, args.merged_root)
    if merged_dir:
        print(f"merged/ resolved to: {os.path.abspath(merged_dir)}")
    else:
        print("merged/ not found - checked:")
        print(f"  {os.path.abspath(os.path.join(root, MERGED_DIRNAME))}")
        print(f"  {os.path.abspath(os.path.join(os.path.dirname(root), MERGED_DIRNAME))}")
        if args.merged_root:
            print(f"  {os.path.abspath(args.merged_root)}")
    print()
    print_stats(species_map, merged_files)


if __name__ == "__main__":
    main()
