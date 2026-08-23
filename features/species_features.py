"""
WildGuard - features/species_features.py

Category D: wildlife species sensitivity. A configurable lookup table
(config/species_sensitivity.json), NOT hardcoded here. This is a RISK
feature, not evidence of leakage - it is combined with, but kept
numerically and explanatorily separate from, the metadata/OCR/context/stego
leakage signals (see policy/policy_engine.py).
"""

import json
from dataclasses import dataclass
from typing import Dict


@dataclass
class SpeciesResult:
    species: str
    sensitivity: float

    def to_feature_dict(self) -> Dict[str, float]:
        return {"species_sensitivity": float(self.sensitivity)}


def load_species_table(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_species_features(species_name: str, table: Dict[str, float]) -> SpeciesResult:
    if not species_name:
        return SpeciesResult(species="unknown", sensitivity=table.get("unknown", 0.4))
    key = species_name.strip().lower().replace("_", " ")
    sensitivity = float(table.get(key, table.get("unknown", 0.4)))
    return SpeciesResult(species=species_name, sensitivity=sensitivity)
