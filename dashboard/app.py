"""
WildGuard - dashboard/app.py

Local dashboard (Flask). Not deployed - runs on localhost only, all
inference/OCR/hashing happens on this machine; no uploaded image ever
leaves it. Two primary areas per spec section 20-21/27-28:

    Analyse  - "Is this image safe, and what privacy risks does it contain?"
    Version  - "Is this image's embedded provenance valid?"

Run:
    python -m dashboard.app
    (then open http://127.0.0.1:5000)
"""

import os
import sys
import uuid
import traceback
from dataclasses import asdict

from flask import Flask, render_template, request, send_file, url_for, flash, redirect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import UPLOADS_DIR, OUTPUT_DIR, MODEL_PATH
from model.predict import WildGuardPredictor
from sanitization.sanitize import sanitize_and_provenance
from provenance.verify import verify_image

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.secret_key = "wildguard-local-dashboard"  # local-only tool, no auth needed
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB upload cap

_predictor = None


def get_predictor() -> WildGuardPredictor:
    global _predictor
    if _predictor is None:
        _predictor = WildGuardPredictor()
    return _predictor


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file_storage) -> str:
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    unique_name = f"upload_{uuid.uuid4().hex[:12]}.{ext}"
    path = os.path.join(UPLOADS_DIR, unique_name)
    file_storage.save(path)
    return path


@app.route("/")
def index():
    return redirect(url_for("analyse"))


# ==========================================================================
# ANALYSE
# ==========================================================================
@app.route("/analyse", methods=["GET"])
def analyse():
    return render_template("analyse.html", result=None,
                            model_missing=not os.path.isfile(MODEL_PATH),
                            model_path=os.path.abspath(MODEL_PATH), active="analyse")


@app.route("/analyse/scan", methods=["POST"])
def analyse_scan():
    if not os.path.isfile(MODEL_PATH):
        flash(f"No trained model found at {os.path.abspath(MODEL_PATH)}. "
              f"Run `python -m model.train` first, from this same project directory.", "error")
        return redirect(url_for("analyse"))

    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please choose an image file.", "error")
        return redirect(url_for("analyse"))
    if not allowed_file(file.filename):
        flash("Only JPEG and PNG images are supported.", "error")
        return redirect(url_for("analyse"))

    species = (request.form.get("species") or "").strip() or None
    caption = (request.form.get("caption") or "").strip() or None

    upload_path = save_upload(file)

    try:
        predictor = get_predictor()
        analysis = predictor.analyse(upload_path, species, caption)
    except Exception as e:
        flash(f"Analysis failed: {e}", "error")
        traceback.print_exc()
        return redirect(url_for("analyse"))

    sanitized_info = None
    decision = analysis.policy.decision
    if decision in ("SANITIZE", "QUARANTINE"):
        try:
            sanitized_info = sanitize_and_provenance(
                upload_path, OUTPUT_DIR, decision, caption, analysis.bundle.ocr
            )
        except Exception as e:
            flash(f"Sanitization failed: {e}", "error")
            traceback.print_exc()

    result = _build_analyse_view(analysis, sanitized_info, upload_path, species, caption)
    return render_template("analyse.html", result=result, model_missing=False, active="analyse")


@app.route("/analyse/sanitize", methods=["POST"])
def analyse_sanitize_manual():
    """Manual 'Sanitize Anyway' action for REVIEW-tier images."""
    upload_path = request.form.get("upload_path")
    decision = request.form.get("decision", "REVIEW")
    species = (request.form.get("species") or "").strip() or None
    caption = (request.form.get("caption") or "").strip() or None

    if not upload_path or not os.path.isfile(upload_path):
        flash("Original upload no longer available - please re-upload.", "error")
        return redirect(url_for("analyse"))

    try:
        predictor = get_predictor()
        analysis = predictor.analyse(upload_path, species, caption)
        sanitized_info = sanitize_and_provenance(
            upload_path, OUTPUT_DIR, decision, caption, analysis.bundle.ocr
        )
    except Exception as e:
        flash(f"Sanitization failed: {e}", "error")
        traceback.print_exc()
        return redirect(url_for("analyse"))

    result = _build_analyse_view(analysis, sanitized_info, upload_path, species, caption)
    return render_template("analyse.html", result=result, model_missing=False, active="analyse")


def _build_analyse_view(analysis, sanitized_info, upload_path, species, caption):
    b = analysis.bundle
    p = analysis.policy
    return {
        "upload_path": upload_path,
        "upload_filename": os.path.basename(upload_path),
        "species": species or "unspecified",
        "caption": caption or "",
        "risk_score": p.final_risk_score,
        "decision": p.decision,
        "model_probability": round(p.model_probability * 100, 1),
        "deterministic_signal_score": round(p.deterministic_signal_score * 100, 1),
        "species_sensitivity": round(p.species_sensitivity * 100, 1),
        "category_verdicts": [asdict(cv) for cv in p.category_verdicts],
        "flagged_factors": p.flagged_factors,
        "ocr_hits": [{"label": h.label, "text": h.text, "bbox": h.bbox} for h in b.ocr.hits],
        "ocr_engine_available": b.ocr.engine_available,
        "metadata_tags_found": b.metadata.raw_tags_found,
        "stego_notes": b.stego.notes,
        "feature_vector": b.feature_vector,
        "sanitized": sanitized_info is not None,
        "sanitized_filename": os.path.basename(sanitized_info.output_path) if sanitized_info else None,
        "original_hash": sanitized_info.original_hash if sanitized_info else None,
        "artifact_hash": sanitized_info.artifact_hash if sanitized_info else None,
        "container_hash": sanitized_info.container_hash if sanitized_info else None,
    }


@app.route("/download/<filename>")
def download_sanitized(filename):
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.isfile(path):
        flash("File not found.", "error")
        return redirect(url_for("analyse"))
    return send_file(path, as_attachment=True)


# ==========================================================================
# VERSION (provenance verification)
# ==========================================================================
@app.route("/version", methods=["GET"])
def version():
    return render_template("version.html", result=None, active="version")


@app.route("/version/verify", methods=["POST"])
def version_verify():
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please choose an image file.", "error")
        return redirect(url_for("version"))
    if not allowed_file(file.filename):
        flash("Only JPEG and PNG images are supported.", "error")
        return redirect(url_for("version"))

    upload_path = save_upload(file)
    try:
        v = verify_image(upload_path)
    except Exception as e:
        flash(f"Verification failed: {e}", "error")
        traceback.print_exc()
        return redirect(url_for("version"))

    result = {
        "upload_filename": os.path.basename(upload_path),
        "provenance_found": v.provenance_found,
        "valid": v.valid,
        "reasons": v.reasons,
        "record": v.record,
        "computed_container_hash": v.computed_container_hash,
        "computed_artifact_hash": v.computed_artifact_hash,
        "recorded_artifact_hash": v.recorded_artifact_hash,
        "recorded_parent_hash": v.recorded_parent_hash,
        "hash_match": v.hash_match,
        "metadata_drift_flags": v.metadata_drift_flags,
    }
    return render_template("version.html", result=result, active="version")


if __name__ == "__main__":
    from config.settings import PROJECT_ROOT
    print("WildGuard dashboard starting at http://127.0.0.1:5000")
    print(f"[WildGuard] Project root:  {os.path.abspath(PROJECT_ROOT)}")
    print(f"[WildGuard] Model path:    {os.path.abspath(MODEL_PATH)}  "
          f"({'FOUND' if os.path.isfile(MODEL_PATH) else 'NOT FOUND'})")
    app.run(host="127.0.0.1", port=5000, debug=False)
