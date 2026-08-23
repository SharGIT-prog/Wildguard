# WildGuard — Wildlife Image Location Privacy Protection System

A local application that scans wildlife photographs for data that could reveal an endangered animal's location — EXIF/GPS metadata, visible text stamped on pixels, leaky filenames/captions, and steganographic channels — scores the risk using an XGBoost model, sanitizes unsafe images, and embeds tamper-evident provenance directly inside the sanitized file.

All operations execute locally on the host machine.

---

## Core Architecture

* **Dataset Preparation (`dataset/`)**: Discovers species directories, executes a deterministic per-species stratified 70/15/15 train/validate/test split at the original-image level, and injects synthetic-leak profiles (Category A: Metadata, B: OCR text, C: Filename/Caption, D: Species hard-negatives, E: Steganography, and Multi-category) using real artifacts. Adversarial samples with all profiles stacked are generated to `data/all/mixed/`.
* **Feature Extraction & Model (`features/`, `model/`)**: Extracts a 47-column deterministic numeric feature vector across all input channels. Trains an `XGBClassifier` optimized for recall on unsafe classes using class-and-split-stratified sampling.
* **Policy Engine & Analysis (`policy/`, `dashboard/`)**: Evaluates model probability, deterministic signals, and species sensitivity tables to categorize risk into `SAFE`, `REVIEW`, `SANITIZE`, or `QUARANTINE`. Automatic deterministic overrides escalate images containing raw GPS or location stamps.
* **Sanitization (`sanitization/`)**: Strips all metadata channels, redacts OCR-detected pixel regions, standardizes filenames, and scrubs captions.
* **Embedded Provenance (`provenance/`)**: Computes SHA-256 digests and embeds provenance metadata records directly into JPEG COM segments or PNG `tEXt` chunks via byte-exact raw insertion, allowing non-destructive tamper and metadata-drift verification.

---

## Project Structure

```
wildguard_project/
├── config/
│   ├── settings.py                 # System paths, seeds, thresholds
│   ├── species_sensitivity.json    # Species risk lookup table
│   └── policy.json                 # Policy weights and decision thresholds
├── dataset/
│   ├── discover.py                 # Dataset discovery and inventory
│   ├── inject_metadata.py          # Category A injection
│   ├── inject_ocr.py               # Category B injection
│   ├── inject_context.py           # Category C injection
│   ├── inject_stego.py             # Category E injection
│   ├── prepare_dataset.py          # Pipeline orchestrator for splits and injection
│   ├── build_mixed.py              # Adversarial stress-test dataset generator
│   └── generate_manifest.py        # Manifest serialization (CSV/JSON)
├── features/
│   ├── ingest.py                   # Image loading and format normalization
│   ├── metadata_features.py        # EXIF/GPS feature extraction
│   ├── ocr_features.py             # Pixel text extraction
│   ├── context_features.py         # Filename and caption text extraction
│   ├── species_features.py         # Sensitivity scoring
│   ├── stego_features.py           # LSB and trailing byte analysis
│   └── feature_pipeline.py         # 47-column feature assembly
├── model/
│   ├── train.py                    # Training and validation pipeline
│   └── predict.py                  # Single-image inference
├── policy/
│   └── policy_engine.py            # Risk score calculation and action dispatch
├── sanitization/
│   ├── metadata_strip.py           # Full EXIF/XMP stripping
│   ├── ocr_redaction.py            # Coordinate-based pixel blurring
│   ├── filename_sanitization.py    # Deterministic safe renaming
│   ├── caption_sanitization.py     # Sensitive token removal
│   └── sanitize.py                 # Sanitization orchestration
├── provenance/
│   ├── hash.py                     # SHA-256 hashing utilities
│   ├── container.py                # Byte-exact JPEG COM / PNG tEXt insertion/extraction
│   ├── embed.py                    # Record constructor and injector
│   ├── extract.py                  # Record extractor and pre-embedding byte recovery
│   └── verify.py                   # Integrity and drift verification
├── dashboard/
│   ├── app.py                      # Flask web interface
│   ├── templates/                  # analyse.html, version.html, base.html
│   └── static/                     # CSS, JavaScript, static assets
├── tests/                          # Test suite
├── data/                           # Generated splits and manifests
├── models/                         # Trained model artifacts and schemas
├── output/                         # Sanitized image output
├── uploads/                        # Temporary processing directory
└── requirements.txt

```

---

## Installation & Setup

### 1. Dependencies

```bash
cd wildguard_project
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Tesseract OCR Engine

Tesseract is required for Category B optical character recognition.

* **Ubuntu / Debian:** `sudo apt-get install -y tesseract-ocr`
* **macOS:** `brew install tesseract`
* **Windows:** Install via the official installer. Set the path variable if installed outside standard directories:
```cmd
set WILDGUARD_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

```



### 3. Environment Configuration

Define the base path to your raw source images:

```bash
# Linux/macOS
export WILDGUARD_ANIMALS_ROOT="/path/to/archive/animals/animals"

# Windows (Command Prompt)
set WILDGUARD_ANIMALS_ROOT="C:\Users\Bhavana\Downloads\wildguard_project\wildguard_project\archive\animals\animals"

```

---

## Execution Pipeline

Execute the pipeline stages sequentially from the project root:

**Step 1: Discover Source Data**

```bash
python -m dataset.discover

```

**Step 2: Generate Split Dataset & Injections**

```bash
python -m dataset.prepare_dataset

```

**Step 3: Train XGBoost Model**

```bash
python -m model.train

```

**Step 4: Run Automated Tests**

```bash
python -m pytest tests/ -v

```

**Step 5: Launch Local Dashboard**

```bash
python -m dashboard.app

```

Access the application interface at `[http://127.0.0.1:5000](http://127.0.0.1:5000)`.

---

## Provenance Hash Lifecycle

Provenance tracking guarantees tamper detection without circular hash dependencies:

```
Original Image File
       │
       ▼
 [ SHA-256 ] ──────────────────────────────────────────► H0 (parent_hash)
       │
       ▼
 [ Sanitization Pipeline ]
       │
       ▼
 Sanitized Image (S)
       │
       ▼
 [ SHA-256 ] ──────────────────────────────────────────► H1 (artifact_hash)
       │                                                      │
       ▼                                                      ▼
 [ Container Embedding ] ◄── Metadata Record Construction ────┘
       │                     { parent_hash: H0, artifact_hash: H1, ... }
       ▼
 Embedded File (P)
       │
       ▼
 [ SHA-256 ] ──────────────────────────────────────────► H2 (container_hash)

```

During verification, the application extracts the JSON payload and strips the container segment to recover `S` directly from raw bytes. It then confirms `SHA-256(S) == H1` to ensure pixel and structural integrity.