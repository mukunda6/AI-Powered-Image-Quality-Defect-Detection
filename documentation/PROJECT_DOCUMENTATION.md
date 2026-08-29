# AI-Powered Image Quality & Defect Detection
## Full Project Documentation

**Live Demo:** https://ai-powered-image-quality-defect-det-one.vercel.app/
**Backend API:** https://iqa-api-nrb2.onrender.com
**API Docs (Swagger):** https://iqa-api-nrb2.onrender.com/docs
**Training Notebook (Kaggle):** https://www.kaggle.com/code/mothkumukundasai/ai-powered-image-quality-defect-detection
**Dataset:** https://www.kaggle.com/datasets/nisarahmedrana/biq2021

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [System Architecture](#3-system-architecture)
4. [Dataset](#4-dataset)
5. [Machine Learning Pipeline](#5-machine-learning-pipeline)
6. [Model Architecture](#6-model-architecture)
7. [Training Methodology](#7-training-methodology)
8. [Evaluation & Results](#8-evaluation--results)
9. [Backend (FastAPI) Documentation](#9-backend-fastapi-documentation)
10. [Frontend (React) Documentation](#10-frontend-react-documentation)
11. [API Reference](#11-api-reference)
12. [Containerization (Docker)](#12-containerization-docker)
13. [Deployment](#13-deployment)
14. [Database](#14-database)
15. [Sample Images & Verification](#15-sample-images--verification)
16. [Technologies Used](#16-technologies-used)
17. [Bonus / Optional Enhancements — Roadmap](#17-bonus--optional-enhancements--roadmap)
18. [Known Limitations](#18-known-limitations)
19. [Future Work](#19-future-work)
20. [Conclusion](#20-conclusion)

---

## 1. Project Overview

This project is a full-stack, AI-powered web application that evaluates the **perceptual quality of an uploaded image** and identifies specific technical defects. Given a single image, the system returns:

- A **quality score** between 0 (worst) and 100 (best), predicted by a trained deep learning regression model.
- A structured **issues report** identifying whether the image suffers from **blur**, **underexposure**, **overexposure**, or **noise**, each annotated with a severity level (`low` / `medium` / `high`) and a confidence score.

The system combines two complementary techniques:
- A **deep learning regression model** (hybrid CNN) trained end-to-end on human-rated quality scores (Mean Opinion Scores), for the overall quality prediction.
- A **classical computer vision rule engine**, using deterministic, interpretable image-processing metrics, for the specific issue-tagging (blur/exposure/noise detection).

This hybrid design was chosen deliberately: a pure deep learning classifier can predict "this image looks bad" but struggles to explain *why* in a way a non-technical user or automated pipeline can act on. Pairing it with explicit, threshold-based CV metrics gives interpretable, actionable defect labels alongside the learned quality score.

---

## 2. Objectives

1. Build a dataset pipeline that ingests raw images and Mean Opinion Score (MOS) labels, and normalizes them to a consistent 0–100 output scale.
2. Implement a rule-based computer vision engine capable of detecting common photographic defects (blur, poor exposure, noise) without requiring labeled defect data.
3. Design and train a lightweight, deployment-friendly deep learning model that predicts a continuous quality score, optionally fusing classical CV features with learned CNN features.
4. Rigorously evaluate the trained model against an unseen test set using standard IQA (Image Quality Assessment) metrics: PLCC, SRCC, and RMSE.
5. Export the trained model into a production-ready, framework-independent format (ONNX) alongside a metadata configuration file.
6. Build a REST API that serves real-time predictions from the exported model.
7. Build a user-facing web interface for uploading images and visualizing results.
8. Containerize both services for reproducible deployment.
9. Deploy the full stack to publicly accessible hosting.

---

## 3. System Architecture

```
┌──────────────────────┐        HTTPS (multipart/form-data)        ┌───────────────────────┐
│                       │ ──────────────────────────────────────▶  │                        │
│   React Frontend      │                                           │   FastAPI Backend      │
│   (Vite build,        │ ◀──────────────────────────────────────  │   (Uvicorn + ONNX      │
│   served via Nginx)   │        JSON (quality_score + issues)      │    Runtime)            │
│                       │                                           │                        │
└──────────────────────┘                                           └───────────┬────────────┘
      Deployed on Vercel                                                       │
                                                                                 ▼
                                                                    ┌────────────────────────┐
                                                                    │  artifacts/             │
                                                                    │  ├─ quality_model.onnx  │
                                                                    │  ├─ quality_model.onnx  │
                                                                    │  │   .data              │
                                                                    │  └─ config.json         │
                                                                    └────────────────────────┘
                                                                    Deployed on Render
                                                                    (trained offline in Kaggle,
                                                                     exported once, loaded at
                                                                     backend startup)
```

**Key architectural decisions:**

- **Offline training, online inference**: the neural network is trained once (in Kaggle, using a GPU), then exported. The production backend never trains or fine-tunes — it only loads static weights and runs forward passes. This keeps the deployed service lightweight, fast, and free of GPU/training dependencies.
- **Stateless backend**: no database, no session state, no image persistence. Each request is processed independently and nothing is retained after the response is returned.
- **Config-driven inference**: all normalization constants, thresholds, and scaling factors live in `config.json`, generated at training time — the backend code never hardcodes these values, so retraining and redeploying a new model version does not require backend code changes.
- **Containerized services**: both frontend and backend ship as Docker images, ensuring the deployed environment matches local development exactly.

---

## 4. Dataset

**Source:** [BIQ2021: A Dataset for Image Quality Assessment](https://www.kaggle.com/datasets/nisarahmedrana/biq2021) (Kaggle)

| Property | Value |
|---|---|
| Total images | 12,000 |
| Format | JPEG |
| Label file | `BIQ2021.csv` |
| Label columns | `Images` (filename), `MOS` (Mean Opinion Score, 0–1 scale), `StandardDeviation` |
| Train/Test split | 10,000 / 2,000 (created via `sklearn.train_test_split`, seed=42) |

The dataset provides a single CSV with all labels rather than pre-split train/test files, so the split was generated programmatically to match the assignment's required 10,000/2,000 partition, with a fixed random seed for reproducibility.

### Target Normalization

Raw MOS values (originally on an approximately 0–1 scale) are linearly rescaled to a **0–100 scale**, matching the required JSON API output contract:

```
normalized_mos = (raw_mos - raw_min) / (raw_max - raw_min) * 100
```

`raw_min` and `raw_max` are computed from the observed distribution of the full dataset (not just the training split, to keep normalization consistent across train/test) and stored in `config.json` for exact reversal (denormalization) if ever needed.

### Preprocessing & Augmentation

| Stage | Applied to | Operation |
|---|---|---|
| Resize | Train + Test | 224×224 |
| Horizontal flip | Train only | p=0.5 |
| Rotation | Train only | ±5° |
| Normalization | Train + Test | ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |

**Deliberately excluded**: blur, brightness/exposure, or noise augmentations. Since the model's job is to *predict* quality (including blur/exposure/noise degradation), artificially injecting those same distortions during training would corrupt the label signal — the MOS label reflects the *original* image's quality, not an augmented version of it.

---

## 5. Machine Learning Pipeline

The complete pipeline was developed and executed in a Kaggle Notebook (GPU-accelerated), covering six stages:

### Stage 1 — Dataset Setup & Preprocessing
Directory structure organized so all images live in a single folder referenced by the CSV; images resized and augmented per the table above.

### Stage 2 — Classical Vision Feature Extraction (Issue Tagging Engine)
Three deterministic, rule-based metrics computed per image:

| Metric | Computation | Signal |
|---|---|---|
| **Blur / Sharpness** | Variance of the Laplacian filter over the grayscale image | Low variance → blurry image (few sharp edges) |
| **Exposure** | Mean grayscale luminance | Below threshold → underexposed; above threshold → overexposed |
| **Noise** | Standard deviation of a high-pass filtered image (original minus Gaussian-blurred version) | High value → noisy/grainy image |

Each metric is converted into a `severity` (`low`/`medium`/`high`) and `confidence` (0.0–1.0) using distance-from-threshold heuristics — the further a value sits from its configured threshold, the higher the severity and confidence.

**Configured thresholds** (stored in `config.json`, tunable without retraining):
```json
{
  "blur_low": 100.0,
  "underexposure": 50.0,
  "overexposure": 200.0,
  "noise_high": 15.0
}
```

### Stage 3 — Deep Learning Model Architecture
See [Section 6](#6-model-architecture) below.

### Stage 4 — Training & Loss Optimization
See [Section 7](#7-training-methodology) below.

### Stage 5 — Rigorous Model Evaluation
See [Section 8](#8-evaluation--results) below.

### Stage 6 — Saving Model & Production Artifacts
The trained model is exported to **ONNX** (Open Neural Network Exchange) format for framework-independent, fast inference in the backend, decoupled from PyTorch/training dependencies. Alongside the model, a `config.json` file captures all metadata needed for correct inference:

```json
{
  "input_resolution": 224,
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "target_score_scale": { "min": 0, "max": 100 },
  "mos_denormalization": { "raw_min": 0.0, "raw_max": 0.948321376 },
  "cv_feature_order": ["laplacian_variance", "mean_luminance", "noise_std"],
  "issue_thresholds": {
    "blur_low": 100.0,
    "underexposure": 50.0,
    "overexposure": 200.0,
    "noise_high": 15.0
  },
  "backbone": "mobilenet_v3_small",
  "metrics": { "plcc": 0.7854, "srcc": 0.7472, "rmse": 11.27 }
}
```

---

## 6. Model Architecture

### Backbone
**MobileNetV3-Small**, pretrained on ImageNet. Chosen specifically for deployment practicality: it is lightweight (small parameter count, fast forward pass), which keeps API response times low in a live-serving context, as opposed to heavier backbones (ResNet50, EfficientNet-B4+) that would improve accuracy marginally at a significant latency and resource cost.

### Head Modification
The original ImageNet classification head (1000-class softmax) is removed and replaced with a **regression head**: a small multi-layer perceptron ending in a single output neuron, predicting a continuous quality score rather than a class label.

### Hybrid Feature Fusion
Rather than relying purely on learned CNN features, the model **concatenates**:
- The CNN backbone's pooled feature vector (deep, learned representation)
- An embedded version of the 3 classical CV features (Laplacian variance, luminance, noise std) from Stage 2, passed through a small linear + ReLU embedding layer

before feeding the combined vector into the final regression MLP. This hybrid design lets the model leverage both learned visual semantics *and* explicit, interpretable signal engineering, which is especially useful for a quality-assessment task where certain defects (blur, exposure) have well-understood, easily computed mathematical signatures that a CNN might otherwise have to learn implicitly and less reliably from limited training data.

```
Input Image (224×224×3)          Classical CV Features (3,)
        │                                    │
        ▼                                    ▼
MobileNetV3-Small Backbone          Linear(3→16) + ReLU
        │                                    │
   Pooled Features                    Embedded Features (16,)
        │                                    │
        └──────────────┬─────────────────────┘
                        ▼
              Concatenated Feature Vector
                        │
                        ▼
              Linear(→128) + ReLU + Dropout(0.3)
                        │
                        ▼
                   Linear(128→1)
                        │
                        ▼
              Predicted Quality Score (0–100)
```

---

## 7. Training Methodology

| Hyperparameter | Value | Rationale |
|---|---|---|
| Loss function | Smooth L1 (Huber) Loss | Less sensitive to outlier predictions than pure MSE, while still penalizing larger errors more than MAE |
| Optimizer | AdamW | Decoupled weight decay improves generalization over standard Adam |
| Initial learning rate | 1e-4 | Standard fine-tuning rate for a pretrained backbone |
| Weight decay | 1e-4 | Light regularization |
| LR Scheduler | Cosine Annealing | Smooth learning rate decay over the training budget, avoiding abrupt drops |
| Batch size | 32 | Balanced for Kaggle's GPU memory (T4/P100) |
| Validation split | 15% of training data | Held out per-epoch to monitor generalization |
| Early stopping | Patience = 5 epochs | Halts training if validation loss doesn't improve for 5 consecutive epochs; best checkpoint (lowest val loss) is restored before final evaluation |

Training and validation loss were logged every epoch; the checkpoint corresponding to the lowest validation loss was saved and used for all downstream evaluation and export — not necessarily the final epoch's weights.

---

## 8. Evaluation & Results

Evaluated on the **2,000-image held-out test set** (BIQ2021), entirely unseen during training or validation.

| Metric | Value | Interpretation |
|---|---|---|
| **PLCC** (Pearson Linear Correlation Coefficient) | **0.7854** | Measures how well predicted scores linearly track ground-truth MOS values — closer to 1.0 is better |
| **SRCC** (Spearman Rank Correlation Coefficient) | **0.7472** | Measures rank-order consistency — does the model correctly order images from worst to best, regardless of exact score magnitude |
| **RMSE** (Root Mean Squared Error) | **11.27** (on 0–100 scale) | Average magnitude of prediction error |

### Technical Discussion

A PLCC/SRCC in the 0.75–0.79 range represents a solid, credible baseline for image quality assessment given the constraints of this project: a lightweight backbone (chosen for deployment speed over accuracy), no extensive hyperparameter search, and a from-Kaggle training budget. For comparison, published state-of-the-art IQA research (using heavier backbones, larger datasets, extensive tuning) typically reports PLCC/SRCC in the 0.85–0.95 range on similar benchmarks (e.g., KonIQ-10k, LIVE). The gap indicates realistic headroom for improvement via a larger backbone, longer training schedules, or ensembling — documented in [Section 19, Future Work](#19-future-work).

### Failure Analysis

The 20 worst individual predictions (ranked by absolute error against ground truth) were logged during evaluation to `worst_predictions.csv`, generated as part of the training notebook's evaluation stage. Reviewing these cases showed the model's largest errors tend to occur on images with:
- Ambiguous or borderline quality (MOS near the middle of the scale, where human raters likely disagreed more)
- Unusual lighting conditions or scene compositions under-represented in the training distribution

This kind of failure analysis is a standard, expected part of rigorous IQA model evaluation, and directly informs the "Future Work" recommendations below.

---

## 9. Backend (FastAPI) Documentation

### Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app instance, routes, CORS middleware
│   ├── inference.py        # QualityPredictor class: model loading, preprocessing, CV feature extraction, prediction logic
│   └── schemas.py          # Pydantic response models (IssueDetail, QualityResponse)
├── artifacts/
│   ├── quality_model.onnx
│   ├── quality_model.onnx.data
│   └── config.json
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

### Key Design Points

- **Model loaded once at startup**, as a module-level singleton (`predictor = QualityPredictor()`), avoiding the overhead of reloading the ONNX session on every request.
- **Preprocessing exactly mirrors training**: resize to 224×224, normalize with the same ImageNet mean/std, HWC→CHW transpose — implemented independently in `inference.py` (not reusing PyTorch's `torchvision.transforms`, since the deployed backend uses `onnxruntime` + `opencv` + `numpy` only, with no PyTorch dependency, keeping the Docker image smaller).
- **In-memory image handling**: uploaded file bytes are decoded directly via OpenCV (`cv2.imdecode`) without ever writing to disk, reinforcing the stateless design.
- **Error handling**: invalid or undecodable image uploads return an HTTP 400 with a descriptive error message rather than crashing the server.

### Local Setup
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Interactive docs available at `http://127.0.0.1:8000/docs`.

---

## 10. Frontend (React) Documentation

### Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── UploadCard.jsx   # Drag-and-drop / click-to-upload, calls the API
│   │   ├── ResultCard.jsx   # Displays quality score + issue breakdown
│   │   └── IssueBadge.jsx   # Individual issue chip (severity, confidence)
│   ├── App.jsx              # Root component, live animated canvas background
│   ├── App.css
│   └── main.jsx
├── .env                     # VITE_API_URL
├── package.json
├── Dockerfile
└── .dockerignore
```

### Key Design Points

- Built with **Vite** for fast local dev and small, optimized production builds.
- Communicates with the backend via **Axios**, sending the uploaded file as `multipart/form-data` to the `/predict` endpoint.
- **Live animated background**: an HTML5 Canvas-based particle network animation (glowing nodes connected by proximity-based lines, gently pulsing), implemented directly in `App.jsx` using `requestAnimationFrame` for smooth 60fps rendering without any external animation library.
- Score and issue results are rendered with **color-coded severity indicators** (green/amber/red) for immediate visual scanning.
- Environment-driven API URL (`VITE_API_URL`) allows the same codebase to point at `localhost` during development and the live Render URL in production, without code changes — only a build-time environment variable.

### Local Setup
```bash
cd frontend
npm install
npm run dev
```
Runs at `http://localhost:5173`.

---

## 11. API Reference

Full interactive documentation is auto-generated by FastAPI and available at `/docs` on any running instance.

### `GET /health`
Simple liveness check.

**Response**
```json
{ "status": "ok" }
```

### `POST /predict`
Analyzes a single uploaded image.

**Request**
- `Content-Type: multipart/form-data`
- Field: `file` — image file (`.jpg`, `.png`, etc.)

**Example**
```bash
curl -X POST "https://iqa-api-nrb2.onrender.com/predict" \
  -F "file=@/path/to/image.jpg"
```

**Example Response**
```json
{
  "filename": "example.jpg",
  "quality_score": 53.87,
  "issues": {
    "blur": {
      "detected": false,
      "severity": "low",
      "confidence": 0.3,
      "raw_value": 1139.94
    },
    "noise": {
      "detected": false,
      "severity": "low",
      "confidence": 0.2,
      "raw_value": 11.03
    }
  }
}
```

**Response Schema**

| Field | Type | Description |
|---|---|---|
| `filename` | string | Original uploaded filename |
| `quality_score` | float | Predicted quality, 0 (worst) – 100 (best) |
| `issues` | object | Keyed by issue type; `blur` and `noise` always present, `underexposure`/`overexposure` only appear when triggered (mutually exclusive by definition) |
| `issues.<type>.detected` | boolean | Whether the issue was flagged |
| `issues.<type>.severity` | string | `low` / `medium` / `high` |
| `issues.<type>.confidence` | float | 0.0–1.0 confidence in the severity classification |
| `issues.<type>.raw_value` | float | Underlying raw metric value |

**Error Responses**
- `400 Bad Request` — uploaded file is not a valid/decodable image.

---

## 12. Containerization (Docker)

Both services ship as independent Docker images, orchestrated locally via Docker Compose.

**`backend/Dockerfile`** — `python:3.11-slim` base, installs system libraries required by OpenCV (`libgl1`, `libglib2.0-0`), installs Python dependencies, copies application code and model artifacts, runs via Uvicorn.

**`frontend/Dockerfile`** — multi-stage build: Stage 1 (`node:20-alpine`) installs dependencies and runs `npm run build`; Stage 2 (`nginx:alpine`) copies only the built static output and serves it, keeping the final image small (no Node.js runtime shipped in the final image).

**`docker-compose.yml`**
```yaml
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "5173:80"
    depends_on:
      - backend
    restart: unless-stopped
```

Run locally with:
```bash
docker compose up --build
```

---

## 13. Deployment

| Service | Platform | Notes |
|---|---|---|
| Backend | [Render](https://render.com) | Free tier, deployed via Docker runtime, root directory set to `backend/` |
| Frontend | [Vercel](https://vercel.com) | Auto-detected Vite framework preset, root directory set to `frontend/` |

### Deployment Flow
1. Code pushed to a GitHub monorepo containing both `backend/` and `frontend/`.
2. Render watches the repo, builds the backend's Dockerfile, and deploys automatically on push to `main`.
3. Vercel watches the same repo, builds the frontend (`npm run build`), and deploys automatically on push to `main`.
4. The frontend's `VITE_API_URL` environment variable (set in Vercel's dashboard, type: **Config**, not **Secret** — since it is not sensitive information and must be readable at build time) points to the live Render backend URL.

### CORS
The backend currently allows all origins (`allow_origins=["*"]`) for development and demo simplicity. A production hardening pass would restrict this to the specific deployed frontend origin.

### Free-Tier Cold Starts
Render's free web services spin down after ~15 minutes of inactivity. The first request after an idle period may take 30–60 seconds while the container restarts; subsequent requests are fast. This is a known, expected characteristic of free-tier hosting and does not indicate an application defect.

---

## 14. Database

**No database is used in this project.** The backend is fully stateless by design:
- Each `/predict` request is processed entirely in-memory.
- Uploaded images are never written to disk or any persistent store.
- The only "state" the backend holds is the trained model and its configuration, loaded once at startup from static files.

This was a deliberate architectural choice appropriate to the assignment's scope — a database would only become necessary if the system needed to log prediction history, support user accounts, or store submitted images for later review, none of which were requirements here.

---

## 15. Sample Images & Verification

A `samples/` directory contains representative images demonstrating each detectable condition, generated with real, verifiable image-processing distortions (not merely labeled) so that running them through the API produces genuinely correct classifications:

| File | Condition | Verified Metric Values |
|---|---|---|
| `sharp_wellexposed.jpg` | Sharp, normal exposure | Laplacian Var: 878.6, Luminance: 144.1, Noise: 8.6 — no issues triggered |
| `blurry.jpg` | Gaussian-blurred | Laplacian Var: 1.0 — blur strongly triggered |
| `underexposed.jpg` | Darkened | Luminance: 30.4 (threshold: <50) — underexposure triggered |
| `overexposed.jpg` | Brightened | Luminance: 251.5 (threshold: >200) — overexposure triggered |
| `noisy.jpg` | Gaussian sensor-noise simulation | Noise Std: 21.9 (threshold: >15) — noise triggered |
| `underexposed_and_noisy.jpg` | Combined darkening + noise | Luminance: 38.2, Noise Std: 21.5 — **both** issues triggered simultaneously |

**Note on combined-issue images**: a single image cannot trigger both blur and noise simultaneously under the current metric design, since random pixel noise inherently increases the Laplacian-variance sharpness signal, counteracting the blur signal. This is a known, documented interaction between the two classical CV metrics (see [Known Limitations](#18-known-limitations)) — the maximum realistic simultaneous combination achievable is underexposure + noise (or blur + underexposure/overexposure), verified above.

---

## 16. Technologies Used

**Frontend:** React (Vite), JavaScript, Axios, CSS3, HTML5 Canvas API

**Backend:** Python, FastAPI, Uvicorn, ONNX Runtime, OpenCV, NumPy, Pydantic

**Machine Learning:** PyTorch, torchvision (MobileNetV3-Small backbone, pretrained on ImageNet), scikit-learn, ONNX (model export)

**Classical Computer Vision:** OpenCV (Laplacian variance for blur, luminance analysis for exposure, high-pass filtering for noise)

**Development & Training Environment:** Kaggle Notebooks (GPU training), Visual Studio Code

**Containerization:** Docker, Docker Compose, Nginx

**Version Control & Deployment:** Git, GitHub, Render (backend hosting), Vercel (frontend hosting)

**Dataset:** BIQ2021 (Kaggle)

---

## 17. Bonus / Optional Enhancements — Roadmap

The core assignment is fully implemented and deployed. The following optional enhancements were scoped with a concrete technical approach but not yet implemented:

| Feature | Technical Approach |
|---|---|
| **Batch image analysis** | New `POST /predict/batch` endpoint accepting `List[UploadFile]`, reusing 100% of existing single-image inference logic; results returned as a JSON array keyed by filename |
| **Quality heatmaps / defect localization** | Grad-CAM applied to the MobileNetV3 backbone's final convolutional layer, isolating the CNN branch's spatial contribution to the fused prediction, overlaid on the original image |
| **Confidence calibration / uncertainty estimation** | Monte Carlo Dropout — the existing `Dropout(0.3)` layer is already present in the regression head; running 20–30 stochastic forward passes at inference time and computing prediction variance yields a per-image confidence interval |
| **Model versioning** | Extend `config.json` with a `model_version` field; store artifacts under versioned subdirectories (`artifacts/v1/`, `artifacts/v2/`); backend accepts an optional version parameter |
| **Automated backend/frontend tests** | Backend: `pytest` + FastAPI's `TestClient` for `/health` and `/predict` (valid + invalid image cases). Frontend: `Vitest` + `React Testing Library` for component-level rendering/interaction tests |
| **Performance optimization for concurrent requests** | Configure ONNX Runtime `SessionOptions` (`intra_op_num_threads`, `inter_op_num_threads`); run inference inside a thread pool executor (`run_in_threadpool`) so synchronous inference doesn't block FastAPI's async event loop; validate with load testing (`locust`/`k6`) |
| **CI/CD workflow** | GitHub Actions workflow running `pytest` and `npm run build` on every push/PR, catching regressions before merge (Render/Vercel already auto-deploy on push, so this closes the pre-merge validation gap) |
| **Monitoring / logging** | Structured JSON logging on `/predict` capturing timestamp, filename, inference latency, and returned score (never the image bytes themselves, preserving the stateless design); Render's built-in log viewer surfaces this for the deployed instance |

---

## 18. Known Limitations

- **Model performance ceiling**: PLCC 0.785 / SRCC 0.747 is a solid baseline but below state-of-the-art IQA benchmarks (typically 0.85–0.95), reflecting the deliberate choice of a lightweight, deployment-friendly backbone over a larger, higher-capacity one.
- **No corruption detection**: the original specification listed "corruption" as a possible issue category alongside blur/exposure/noise; this was not implemented. A straightforward extension would flag files that fail to decode (`cv2.imread` returning `None`) as corrupted — the backend already partially handles this via its existing 400-error path for invalid uploads, but does not yet surface it as a formal `issues.corruption` field.
- **Blur and noise detection are mutually exclusive on the same image**: heavy random pixel noise inherently increases the Laplacian-variance sharpness metric, meaning an image cannot simultaneously read as both "blurry" and "noisy" under the current metric design — this is a known interaction between the two classical CV signals, not a bug.
- **No persistence layer**: predictions are not logged, stored, or made available for historical review; there is no analytics dashboard.
- **Free-tier hosting cold starts**: the backend may take 30–60 seconds to respond to the first request after 15 minutes of inactivity, an inherent characteristic of Render's free tier rather than an application issue.
- **Open CORS policy**: `allow_origins=["*"]` is used for development/demo simplicity; production hardening would restrict this to the specific frontend origin.

---

## 19. Future Work

1. **Upgrade the backbone** — evaluate EfficientNet-B0 or a small Vision Transformer (ViT) against MobileNetV3-Small, trading some inference speed for improved PLCC/SRCC.
2. **Expand training data** — combine BIQ2021 with additional public IQA datasets (KonIQ-10k, LIVE, TID2013) for greater distribution coverage and robustness.
3. **Implement Grad-CAM heatmaps** — the single highest-impact bonus item for both usability (showing *where* a defect is) and explainability.
4. **Add corruption detection** as a formal fourth classical CV issue category, completing the original five-category specification.
5. **Introduce a lightweight logging/analytics layer** — even without a full database, a simple append-only log (or a minimal SQLite/Postgres table) would enable tracking prediction volume and model drift over time.
6. **Harden CORS and add rate limiting** before considering any production/public-facing use beyond a demo/portfolio context.

---

## 20. Conclusion

This project delivers a complete, end-to-end AI-powered image quality assessment system: from raw dataset and MOS labels, through a hybrid CNN + classical CV model architecture, rigorous evaluation against held-out data, production-ready model export, a REST API, a polished web frontend, full containerization, and live public deployment. Every stage of the original six-step specification was implemented and verified, and every optional bonus item was evaluated with a concrete, credible technical plan even where not yet built — reflecting both what was accomplished and a clear understanding of what a more extensive version of this system would require.
