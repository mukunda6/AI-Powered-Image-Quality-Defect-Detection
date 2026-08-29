# AI-Powered Image Quality & Defect Detection

A full-stack application that analyzes an uploaded image and returns:
- A **quality score** (0–100), predicted by a hybrid CNN + classical computer-vision regression model.
- An **issues report** flagging blur, underexposure, overexposure, and noise, each with a severity level and confidence score.

The system consists of three parts:
1. **ML Pipeline** — trained and evaluated in Kaggle (dataset preprocessing, classical CV feature engine, hybrid CNN regression model, training, evaluation).
2. **Backend API** — FastAPI service that loads the exported ONNX model and serves predictions.
3. **Frontend** — React (Vite) single-page app for uploading an image and viewing results.
   
- **Live deployed URL**: https://ai-powered-image-quality-defect-det-one.vercel.app/
---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [ML Pipeline: Training & Methodology](#ml-pipeline-training--methodology)
- [Evaluation Results](#evaluation-results)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Docker / Docker Compose](#docker--docker-compose)
- [Database](#database)
- [API Documentation](#api-documentation)
- [Sample Images](#sample-images)
- [Deployment](#deployment)
- [Deployed URLs](#deployed-urls)
- [Known Limitations](#known-limitations)

---

## Architecture Overview

```
┌─────────────────┐      HTTP (multipart/form-data)      ┌──────────────────┐
│  React Frontend  │ ───────────────────────────────────▶ │  FastAPI Backend  │
│  (Vite, Nginx)   │ ◀─────────────────────────────────── │  (ONNX Runtime)   │
└─────────────────┘         JSON (score + issues)          └──────────────────┘
                                                                     │
                                                                     ▼
                                                         quality_model.onnx
                                                         config.json
                                                    (trained in Kaggle, exported)
```

- The model is **trained once, offline, in Kaggle**, then exported as an ONNX file and a `config.json` metadata file.
- The backend performs **stateless, offline inference** — no external model-serving calls, no internet dependency at runtime.
- The frontend is a **static single-page app** that calls the backend's `/predict` endpoint.

---

## Repository Structure

```
image-quality-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app, routes, CORS
│   │   ├── inference.py       # Model loading, preprocessing, CV feature extraction, prediction
│   │   └── schemas.py         # Pydantic response models
│   ├── artifacts/
│   │   ├── quality_model.onnx
│   │   ├── quality_model.onnx.data
│   │   └── config.json
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadCard.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   └── IssueBadge.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── .env                   # VITE_API_URL
│   ├── package.json
│   ├── Dockerfile
│   └── .dockerignore
├── notebooks/
│   └── training_pipeline.ipynb   # Full Kaggle training notebook (see below)
├── samples/                      # Example images used for evaluation / demo
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Dataset

- **Source**: [BIQ2021: A Dataset for Image Quality Assessment](https://www.kaggle.com/datasets/nisarahmedrana/biq2021) (Kaggle) — `BIQ2021.csv` + accompanying images, 12,000 images total.
- **Labels**: Mean Opinion Score (MOS) per image, originally on a `0–1` scale, plus a `StandardDeviation` column (not used for training).
- **Split**: 10,000 train / 2,000 test, created via `sklearn.model_selection.train_test_split` (random seed fixed at 42) since the source data ships as a single CSV rather than pre-split files.
- **Target normalization**: raw MOS values are linearly rescaled from their observed `[min, max]` range to a `[0, 100]` scale to match the API's output contract.

Dataset is **not included in this repository** due to size — download it directly from the Kaggle link above, or via the Kaggle API:
```bash
kaggle datasets download -d nisarahmedrana/biq2021
```

**Training notebook (Kaggle, fully executed)**: [AI-Powered Image Quality Defect Detection](https://www.kaggle.com/code/mothkumukundasai/ai-powered-image-quality-defect-detection) — contains the complete, runnable pipeline: dataset loading, MOS normalization, classical CV feature extraction, hybrid CNN model definition, training loop, evaluation (PLCC/SRCC/RMSE), and ONNX export. A copy is also included in this repo at `notebooks/training_pipeline.ipynb`.

---

## ML Pipeline: Training & Methodology

### 1. Preprocessing
- Images resized to **224×224**.
- Training-only augmentations: horizontal flip (p=0.5), slight rotation (±5°). No blur/brightness augmentations, since those would corrupt the quality-label signal.
- Normalized with standard ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`.

### 2. Classical CV Issue-Tagging Engine
Rule-based metrics computed per image, independent of the neural network:

| Metric | Method | Purpose |
|---|---|---|
| Blur / Sharpness | Variance of Laplacian filter | Low variance → blur |
| Exposure | Mean grayscale luminance | Thresholds flag under/overexposure |
| Noise | Std. deviation of high-pass filtered image | High value → noisy image |

Each metric is converted into a `severity` (`low` / `medium` / `high`) and a `confidence` (0–1) score using distance-from-threshold heuristics. Thresholds are configurable and stored in `config.json`.

### 3. Model Architecture — Hybrid CNN + CV Feature Fusion
- **Backbone**: MobileNetV3-Small, pretrained on ImageNet (lightweight, fast inference — suitable for a live API).
- **Head**: Original classification head removed; deep features are concatenated with an embedded version of the 3 classical CV features (Laplacian variance, luminance, noise std), then passed through a small MLP regression head (`128 → 1`) with dropout for regularization.
- **Output**: A single continuous quality score (0–100).

### 4. Training Configuration
- **Loss**: Smooth L1 (Huber) loss.
- **Optimizer**: AdamW, initial LR `1e-4`, weight decay `1e-4`.
- **Scheduler**: Cosine Annealing over the training epoch budget.
- **Early Stopping**: training halts if validation loss does not improve for 5 consecutive epochs; best checkpoint (by val loss) is restored before final evaluation and export.
- **Validation**: 15% held out from the training split, evaluated after every epoch.

### 5. Export
- Model exported to **ONNX** (opset 18) for fast, framework-independent inference in the backend.
- `config.json` stores: input resolution, normalization mean/std, MOS denormalization bounds, CV feature order, and issue-detection thresholds — so the backend never hardcodes these values.

The complete, runnable notebook (as executed in Kaggle) is available at [kaggle.com/code/mothkumukundasai/ai-powered-image-quality-defect-detection](https://www.kaggle.com/code/mothkumukundasai/ai-powered-image-quality-defect-detection), and a copy is included in this repo at `notebooks/training_pipeline.ipynb`.

---

## Evaluation Results

Evaluated on the held-out **2,000-image test set** (BIQ2021), unseen during training:

| Metric | Value | Meaning |
|---|---|---|
| **PLCC** (Pearson Linear Correlation) | **0.785** | Absolute prediction accuracy vs. ground-truth MOS |
| **SRCC** (Spearman Rank Correlation) | **0.747** | Rank-ordering consistency (does the model rank images correctly by quality) |
| **RMSE** | **11.27** (on 0–100 scale) | Average magnitude of prediction error |

### Technical interpretation
A PLCC/SRCC in the 0.75–0.79 range is a solid baseline for image quality assessment using a lightweight backbone and hybrid feature fusion, without extensive hyperparameter search or a larger backbone (e.g., EfficientNet-B0 or ResNet50). Published IQA benchmarks with heavily tuned, larger models typically report PLCC/SRCC in the 0.85–0.95 range — there is room for improvement via a larger backbone, longer training, or an ensembled classical + deep scoring approach.

### Failure Analysis
The 20 worst predictions (by absolute error) from the test set are logged in `worst_predictions.csv` (generated during training) to document where the model struggles — typically images with ambiguous or borderline quality, or unusual lighting conditions not well represented in the training distribution.

---

## Backend Setup

### Requirements
- Python 3.11+
- pip

### Local setup (without Docker)

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Verify it's running:
```
http://127.0.0.1:8000/docs
```

### Environment / configuration
The backend has no required environment variables — all model paths and thresholds are resolved from `artifacts/config.json` relative to the app directory.

---

## Frontend Setup

### Requirements
- Node.js 18+
- npm

### Local setup

```bash
cd frontend
npm install
```

Create/edit `.env`:
```
VITE_API_URL=http://127.0.0.1:8000
```

Run the dev server:
```bash
npm run dev
```

Open:
```
http://localhost:5173
```

### Production build
```bash
npm run build
```
Outputs static files to `dist/`, servable by any static file host or Nginx (see Dockerfile).

---

## Docker / Docker Compose

Both services are containerized. From the repository root:

```bash
docker compose up --build
```

This will:
1. Build the backend image (Python 3.11-slim + system libs for OpenCV + FastAPI/ONNX Runtime dependencies).
2. Build the frontend image (multi-stage: Node 20 build → static files served via Nginx).
3. Start both containers, networked together.

Access:
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000/docs`

**`docker-compose.yml`:**
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

---

## Database

**This project does not use a database.** The backend is fully stateless:
- Each `/predict` request is processed independently — the uploaded image is analyzed in-memory and never persisted to disk or any datastore.
- The trained model and its configuration are the only "state," loaded once at server startup from static files (`artifacts/quality_model.onnx`, `artifacts/config.json`).

If a database were added in the future (e.g., to log prediction history or store user-submitted images), the natural extension point would be inside `app/main.py`'s `/predict` route, writing the request/response pair to a table before returning the response.

---

## API Documentation

Interactive Swagger UI is auto-generated by FastAPI at `/docs` on any running instance (local or deployed).

### `GET /health`
Health check.

**Response**
```json
{ "status": "ok" }
```

### `POST /predict`
Analyzes an uploaded image and returns a quality score plus detected issues.

**Request**
- `Content-Type: multipart/form-data`
- Field: `file` — an image file (`.jpg`, `.png`, etc.)

**Example (cURL)**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
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
    },
    "underexposure": {
      "detected": true,
      "severity": "medium",
      "confidence": 0.55,
      "raw_value": 42.1
    }
  }
}
```

**Error responses**
- `400 Bad Request` — file is not a valid image, or could not be decoded.

| Field | Type | Description |
|---|---|---|
| `filename` | string | Original uploaded filename |
| `quality_score` | float | Predicted quality, 0 (worst) – 100 (best) |
| `issues` | object | Keyed by issue type (`blur`, `noise`, `underexposure`, `overexposure`); only exposure issues that are actually triggered appear, blur/noise always appear |
| `issues.<type>.detected` | boolean | Whether this issue was flagged |
| `issues.<type>.severity` | string | `low` / `medium` / `high` |
| `issues.<type>.confidence` | float | 0–1 confidence in the severity classification |
| `issues.<type>.raw_value` | float | Underlying raw metric value (e.g., Laplacian variance) |

---

## Sample Images

The `samples/` directory contains example images representing different quality conditions used to validate the API during development:

| File | Condition | Expected behavior |
|---|---|---|
| `samples/sharp_wellexposed.jpg` | Sharp, normal exposure | High score, no issues detected |
| `samples/blurry.jpg` | Motion/focus blur | Blur flagged, medium/high severity |
| `samples/underexposed.jpg` | Dark/low-light | Underexposure flagged |
| `samples/overexposed.jpg` | Bright/washed out | Overexposure flagged |
| `samples/noisy.jpg` | High ISO / grainy | Noise flagged |

can download from sample inputs and check the sample outputs from code repo.

---

## Deployment

### Backend — Render
1. Push repository to GitHub.
2. On Render: **New → Web Service** → connect repo → set **Root Directory: `backend`** → **Runtime: Docker** → **Instance Type: Free**.
3. Render builds from `backend/Dockerfile` and exposes the service at a generated URL.

### Frontend — Vercel
1. On Vercel: **Add New → Project** → import the same repo → set **Root Directory: `frontend`**.
2. Framework preset: Vite (auto-detected).
3. Add environment variable:
   - `VITE_API_URL` = `<your deployed backend URL>`
   - Type: **Config** (not Secret — this value is not sensitive and must be readable at build time)
4. Deploy.

> **Note:** Vite bakes `VITE_*` environment variables into the static build at build time. Any change to `VITE_API_URL` requires a redeploy to take effect.

### CORS
The backend currently allows all origins (`allow_origins=["*"]`) for development convenience. For a production hardening pass, replace this with the specific deployed frontend origin in `backend/app/main.py`.

---

## Deployed URLs

| Service | URL |
|---|---|
| Frontend (Vercel) | [https://ai-powered-image-quality-defect-det-one.vercel.app](https://ai-powered-image-quality-defect-det-one.vercel.app/) |
| Backend API (Render) | [https://iqa-api-nrb2.onrender.com](https://iqa-api-nrb2.onrender.com) |
| API Docs (Swagger) | [https://iqa-api-nrb2.onrender.com/docs](https://iqa-api-nrb2.onrender.com/docs) |

- **Live deployed URL**: https://ai-powered-image-quality-defect-det-one.vercel.app/
> **Note on free-tier hosting:** the backend is hosted on Render's free tier, which spins down after 15 minutes of inactivity. The first request after idle time may take 30–60 seconds while the service wakes up; subsequent requests are fast.

---

## Known Limitations

- **Model performance**: PLCC 0.785 / SRCC 0.747 is a solid baseline but below state-of-the-art IQA benchmarks; a larger backbone or longer training schedule would likely improve this.
- **No corruption detection**: the original spec listed `corruption` as a possible issue category; the current classical CV engine covers blur, exposure, and noise, but does not yet detect file corruption or artifacts like compression blocking.
- **No persistence layer**: predictions are not logged or stored; there is no history or analytics dashboard.
- **Free-tier hosting cold starts**: as noted above, the backend may be slow to respond after inactivity on Render's free plan.
- **CORS is currently open (`*`)**: acceptable for a demo/evaluation deployment, should be restricted to the specific frontend origin for production use.
