"""
WildGuard - model/predict.py

Phase 5: local inference pipeline. Given a single image path (+ optional
species/caption), runs the exact same feature pipeline used at training
time, gets the XGBoost probability, and hands everything to the policy
engine to produce category-level verdicts, human-readable flagged factors,
and the final SAFE/REVIEW/SANITIZE/QUARANTINE decision.

This is the module the Analyse dashboard page calls directly - nothing
about it is web-specific, so it also works from a plain Python REPL or a
CLI script.
"""

import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_PATH, SPECIES_SENSITIVITY_PATH, POLICY_CONFIG_PATH
from features.feature_pipeline import extract_all_features, FEATURE_COLUMNS, ExtractionBundle
from features.species_features import load_species_table
from policy.policy_engine import evaluate_policy, load_policy_config, PolicyResult


@dataclass
class AnalysisResult:
    image_path: str
    bundle: ExtractionBundle
    policy: PolicyResult


class WildGuardPredictor:
    """Loads the model/config once; reuse across multiple predict() calls
    (the dashboard keeps one instance alive for the life of the process)."""

    def __init__(self, model_path: str = MODEL_PATH,
                 species_path: str = SPECIES_SENSITIVITY_PATH,
                 policy_path: str = POLICY_CONFIG_PATH):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found at {os.path.abspath(model_path)}. "
                f"Run `python -m model.train` first, from this same project directory."
            )
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Found a file at {os.path.abspath(model_path)} but could not load it as a "
                f"model ({e}). It may be from an interrupted training run - delete it and "
                f"re-run `python -m model.train`."
            ) from e
        self.species_table = load_species_table(species_path)
        self.policy_config = load_policy_config(policy_path)

    def predict_probability(self, feature_vector: dict) -> float:
        x = np.array([[feature_vector[c] for c in FEATURE_COLUMNS]], dtype=float)
        proba = float(self.model.predict_proba(x)[0][1])
        return proba

    def analyse(self, image_path: str, species: Optional[str] = None,
                caption: Optional[str] = None) -> AnalysisResult:
        bundle = extract_all_features(image_path, species, caption, self.species_table)
        model_probability = self.predict_probability(bundle.feature_vector)
        policy_result = evaluate_policy(bundle, model_probability, self.policy_config)
        return AnalysisResult(image_path=image_path, bundle=bundle, policy=policy_result)
