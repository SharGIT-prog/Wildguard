"""
WildGuard - model/train.py

Phase 3: extract the deterministic tabular feature vector (features/feature_pipeline.py)
for every image in the generated dataset dump, reading labels EXCLUSIVELY
from the manifest (never re-derived from folder/filenames at this stage).

Phase 4: train an XGBoost binary classifier (risk_label: 0=safe, 1=leak),
evaluate on the held-out validate/test splits with accuracy/precision/
recall/F1/ROC-AUC/confusion-matrix, and separately evaluate on mixed/ as a
stress test (never used for training).

--- No train/test leakage ---
The manifest's `split` column was assigned once, at prepare_dataset.py time,
BEFORE any synthetic variant existed (split at the original-image level).
This script only ever reads that pre-assigned split - it never re-splits or
shuffles across train/validate/test, so no synthetic variant of a source
image used for training can appear in validation/test.

--- Time budget ---
OCR (Tesseract, subprocess per image) is the dominant per-image cost and is
highly hardware-dependent. This script calibrates on a small real sample
from the generated dataset first, then estimates whether extracting
features for the full manifest fits inside --time-budget-minutes; if not,
it takes a balanced, class-stratified subsample per split rather than
silently truncating (which would bias the dataset toward whatever species
happen to be listed first).

Usage:
    python -m model.train
    python -m model.train --time-budget-minutes 90
"""

import os
import sys
import time
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import xgboost as xgb
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MANIFEST_CSV_PATH, MODEL_PATH, FEATURE_SCHEMA_PATH, METRICS_PATH,
    SPECIES_SENSITIVITY_PATH, RANDOM_SEED,
)
from features.feature_pipeline import extract_all_features, FEATURE_COLUMNS, TARGET_COLUMN
from features.species_features import load_species_table


# ==========================================================================
# Calibration (mirrors the approach validated in the previous iteration -
# times a small real sample on THIS machine before committing to a size)
# ==========================================================================
def calibrate(manifest_df: pd.DataFrame, species_table, sample_size: int, seed: int) -> float:
    sample = manifest_df.sample(n=min(sample_size, len(manifest_df)), random_state=seed)
    t0 = time.time()
    ok = 0
    for _, row in sample.iterrows():
        try:
            extract_all_features(row["new_path"], row["animal"], row.get("caption", ""), species_table)
            ok += 1
        except Exception:
            continue
    elapsed = time.time() - t0
    if ok == 0:
        raise RuntimeError("Calibration failed: no sampled images could be processed.")
    return elapsed / ok


def stratified_cap(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """
    Caps `df` to at most max_rows, preserving (as closely as possible) the
    relative proportions of split x feature_category, so a time-budget cap
    never silently skews the dataset toward one split or category.
    """
    if len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    parts = []
    for (_, _), group in df.groupby(["split", "feature_category"]):
        n = max(1, int(round(len(group) * frac)))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    capped = pd.concat(parts, ignore_index=True)
    return capped.sample(n=min(max_rows, len(capped)), random_state=seed)


# ==========================================================================
# Phase 3: feature extraction over the manifest
# ==========================================================================
def extract_feature_matrix(df: pd.DataFrame, species_table, progress_every: int = 500) -> pd.DataFrame:
    rows = []
    n = len(df)
    t0 = time.time()
    skipped = 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            bundle = extract_all_features(row["new_path"], row["animal"], row.get("caption", ""), species_table)
        except Exception:
            skipped += 1
            continue
        record = dict(bundle.feature_vector)
        record[TARGET_COLUMN] = int(row["risk_label"])
        record["split"] = row["split"]
        record["feature_category"] = row["feature_category"]
        record["animal"] = row["animal"]
        record["image_id"] = row["image_id"]
        rows.append(record)
        if i % progress_every == 0 or i == n:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (n - i) / rate if rate > 0 else 0
            print(f"  extracted {i}/{n}  ({elapsed/60:.1f} min elapsed, ~{remaining/60:.1f} min remaining)")
    if skipped:
        print(f"  [warn] skipped {skipped} unreadable image(s)")
    return pd.DataFrame(rows)


# ==========================================================================
# Phase 4: train + evaluate
# ==========================================================================
def train_and_evaluate(feature_df: pd.DataFrame, seed: int):
    train_df = feature_df[feature_df["split"] == "train"]
    val_df = feature_df[feature_df["split"] == "validate"]
    test_df = feature_df[feature_df["split"] == "test"]
    mixed_df = feature_df[feature_df["split"] == "mixed"]

    X_train, y_train = train_df[FEATURE_COLUMNS].astype(float), train_df[TARGET_COLUMN].astype(int)
    X_val, y_val = val_df[FEATURE_COLUMNS].astype(float), val_df[TARGET_COLUMN].astype(int)
    X_test, y_test = test_df[FEATURE_COLUMNS].astype(float), test_df[TARGET_COLUMN].astype(int)

    # scale_pos_weight favors recall on the positive (leak) class, per spec
    # section 17: false negatives (missed leaks) are more dangerous than
    # false positives here.
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    scale_pos_weight = (n_neg / n_pos) * 1.15  # slight extra recall bias beyond pure class balance

    model = xgb.XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, random_state=seed, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    def eval_split(X, y, name):
        if len(X) == 0:
            return {}
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "n": int(len(y)),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
        }
        if len(set(y)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y, proba))
        cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
        metrics["confusion_matrix"] = {"labels": ["safe(0)", "leak(1)"], "matrix": cm}
        print(f"\n--- {name} ---")
        print(classification_report(y, pred, labels=[0, 1], target_names=["safe", "leak"], zero_division=0))
        return metrics

    results = {
        "validate": eval_split(X_val, y_val, "VALIDATE"),
        "test": eval_split(X_test, y_test, "TEST"),
    }
    if len(mixed_df) > 0:
        X_mixed, y_mixed = mixed_df[FEATURE_COLUMNS].astype(float), mixed_df[TARGET_COLUMN].astype(int)
        results["mixed_stress_test"] = eval_split(X_mixed, y_mixed, "MIXED (stress test, all-category adversarial)")

    importances = model.feature_importances_
    importance_map = {
        col: float(imp) for col, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda kv: kv[1], reverse=True)
    }
    results["feature_importance"] = importance_map
    results["train_rows"] = int(len(train_df))
    results["scale_pos_weight"] = float(scale_pos_weight)

    return model, results


def main():
    parser = argparse.ArgumentParser(description="WildGuard Phase 3-4: feature extraction + XGBoost training")
    parser.add_argument("--manifest", default=MANIFEST_CSV_PATH)
    parser.add_argument("--time-budget-minutes", type=float, default=90)
    parser.add_argument("--max-rows", type=int, default=None,
                         help="Force an exact row cap and skip auto-calibration.")
    parser.add_argument("--calibration-sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if not os.path.isfile(args.manifest):
        raise SystemExit(f"Manifest not found: {args.manifest}\nRun dataset.prepare_dataset first.")

    manifest_df = pd.read_csv(args.manifest)
    manifest_df["caption"] = manifest_df["caption"].fillna("")
    print(f"[WildGuard] Manifest loaded: {len(manifest_df)} rows "
          f"({manifest_df['split'].value_counts().to_dict()})")

    species_table = load_species_table(SPECIES_SENSITIVITY_PATH)

    if args.max_rows is not None:
        working_df = stratified_cap(manifest_df, args.max_rows, args.seed)
        print(f"[WildGuard] --max-rows given: using {len(working_df)} rows (calibration skipped).")
    else:
        print(f"[WildGuard] Calibrating feature-extraction speed on this machine "
              f"({args.calibration_sample_size} sample images) ...")
        sec_per_image = calibrate(manifest_df, species_table, args.calibration_sample_size, args.seed)
        budget_seconds = args.time_budget_minutes * 60.0 * 0.85  # safety margin
        max_rows = max(200, int(budget_seconds / sec_per_image))
        print(f"[WildGuard] Measured {sec_per_image*1000:.1f} ms/image on this machine.")
        working_df = stratified_cap(manifest_df, max_rows, args.seed)
        est_minutes = len(working_df) * sec_per_image / 60.0
        print(f"[WildGuard] Auto-selected {len(working_df)}/{len(manifest_df)} rows "
              f"(~{est_minutes:.1f} min, within the {args.time_budget_minutes:.0f}-min budget).")

    print("\n[WildGuard] Phase 3: extracting feature vectors ...")
    feature_df = extract_feature_matrix(working_df, species_table)
    print(f"[WildGuard] Feature extraction complete: {len(feature_df)} usable rows.")

    print("\n[WildGuard] Phase 4: training XGBoost + evaluating ...")
    model, results = train_and_evaluate(feature_df, args.seed)

    joblib.dump(model, MODEL_PATH)
    schema = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "random_seed": args.seed,
        "manifest_source": args.manifest,
        "rows_used": len(feature_df),
    }
    with open(FEATURE_SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[WildGuard] Model saved -> {os.path.abspath(MODEL_PATH)}")
    print(f"[WildGuard] Schema saved -> {os.path.abspath(FEATURE_SCHEMA_PATH)}")
    print(f"[WildGuard] Metrics saved -> {os.path.abspath(METRICS_PATH)}")
    if not os.path.isfile(MODEL_PATH):
        print("[WildGuard] WARNING: joblib.dump() reported success but the file is not "
              "at the expected path - check for permission errors above.")

    print("\n=== Summary ===")
    for split_name in ("validate", "test", "mixed_stress_test"):
        m = results.get(split_name)
        if not m:
            continue
        print(f"{split_name:20s} n={m['n']:5d}  acc={m['accuracy']:.3f}  "
              f"prec={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}"
              + (f"  auc={m['roc_auc']:.3f}" if "roc_auc" in m else ""))

    print("\nTop 10 feature importances:")
    for i, (k, v) in enumerate(results["feature_importance"].items()):
        if i >= 10:
            break
        print(f"  {k:32s} {v:.4f}")

    print(f"\n[WildGuard] DONE. Trained model is at:\n    {os.path.abspath(MODEL_PATH)}")
    print("[WildGuard] The dashboard checks this EXACT path. If it still reports "
          "'no trained model found', the dashboard process is looking at a different "
          "copy of the project than this training run wrote to - confirm both are "
          "launched from the same project directory.")


if __name__ == "__main__":
    main()
