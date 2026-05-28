# GravSense

Construction debris detection and volume estimation from a single photograph.

[![CI](https://github.com/amgharhind/gravsense/actions/workflows/ci.yml/badge.svg)](https://github.com/amgharhind/gravsense/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![GHCR](https://img.shields.io/badge/ghcr.io-gravsense-0d1117?logo=github)](https://github.com/amgharhind/gravsense/pkgs/container/gravsense)

---

## What it does

Upload a construction site photo. GravSense returns:

- **Pixel-precise debris mask** — GroundingDINO locates debris by text prompt, SAM generates the mask
- **Automatic reference-width calibration** — GroundingDINO detects a vehicle/truck and computes the real-world image scale
- **Automatic pile height** — Depth Anything V2 (metric outdoor) estimates pile depth from a single RGB photo
- **Surface area + volume** — derived from mask pixels × calibrated scale × depth

Everything runs server-side. The browser UI shows the result in four views: original image, debris overlay, full depth map, and depth-on-debris only.

---

## System architecture

```mermaid
flowchart TD
    Client(["Browser\nUpload image"])
    API["FastAPI — async / uvicorn\nCORS · 10 MB file guard · content-type + PIL validation"]

    Client -->|"POST /analyze"| API

    API -->|"step 1"| Det
    API -->|"step 2 — parallel"| Depth
    API -->|"step 2 — parallel"| Cal

    subgraph Det ["Debris Detection — grounded_sam.py"]
        GD1["GroundingDINO\ntext prompt → bounding boxes"]
        VF["Vegetation filter\ndrop tree / plant boxes"]
        SAM["SAM vit-base\nboxes → pixel mask"]
        GD1 --> VF --> SAM
    end

    subgraph Depth ["Pile Height — depth_estimator.py"]
        DA["Depth Anything V2\nMetric Outdoor Small\nper-pixel depth in metres"]
        DH["ground ring 40 px − pile median\n= pile height cm"]
        CM["Plasma colormap\ndepth map + depth-on-debris images"]
        DA --> DH
        DA --> CM
    end

    subgraph Cal ["Reference Width — auto_calibrate.py"]
        GD2["GroundingDINO\nquery: truck · lorry · car · van"]
        SC["bbox width px → real-world scale\nref_width_cm = known_cm ÷ fraction"]
        GD2 --> SC
    end

    SAM -->|"mask"| Vol
    DH  -->|"height cm"| Vol
    SC  -->|"ref_width_cm"| Vol

    Vol["volume.py\npixel_scale = ref_width ÷ img_width\narea = mask_pixels × scale²\nvolume = area × height"]

    Vol --> Resp["JSON response\nn_detections · surface_area_cm2 · volume_cm3\nauto_pile_height_cm · auto_reference_width_cm\n+ 4 base64 images"]
```

**CI/CD pipeline:**

```mermaid
flowchart LR
    Push(["git push\nmain"])
    Lint["ruff lint\ngravsense/ + tests/"]
    Test["pytest\nPython 3.10 + 3.11\nall inference mocked"]
    Build["Docker build\n2-stage Python 3.10-slim"]
    GHCR["GHCR push\nghcr.io/amgharhind/gravsense\n:latest · :sha"]

    Push --> Lint --> Test --> Build --> GHCR
```

**Baseline** — a SegFormer-b0 (ADE20K) detector is kept for benchmark comparison
(`?method=segformer`). It uses the dominant predicted class as debris — the original
research approach, intentionally preserved so both methods can be compared via the same API.

---

## Features

| | Grounded SAM (default) | SegFormer (baseline) |
|---|---|---|
| Detection | GroundingDINO text → boxes → SAM mask | ADE20K semantic seg, dominant class |
| Needs retraining | No — open-vocabulary | No |
| Debris in truck/skip | ✅ expanded query covers truck-bed scenes | ❌ class-based only |
| Vegetation filter | ✅ drops tree/plant false positives | ❌ none |
| Reference-width auto | ✅ GroundingDINO vehicle detection | ✅ same |
| Depth auto (height) | ✅ Depth Anything V2 Metric | ✅ same |
| Speed (CPU) | ~4–6 s | ~2–3 s |

---

## Quickstart

### Docker — pull from GHCR (no build needed)

```bash
docker pull ghcr.io/amgharhind/gravsense:latest
docker run -p 8000:8000 ghcr.io/amgharhind/gravsense:latest
```

### Docker — build from source

```bash
git clone https://github.com/amgharhind/gravsense
cd gravsense
docker compose up --build
```

Open **http://localhost:8000** for the UI, **http://localhost:8000/docs** for the API.

First start downloads ~800 MB of models (GroundingDINO + SAM + Depth Anything V2).
All subsequent starts are instant — models are cached in a named Docker volume.

### Local (Python 3.10 / 3.11)

```bash
# Create a virtual environment (Python 3.13 is not yet supported by PyTorch on Windows)
py -3.10 -m venv .venv && .venv\Scripts\activate   # Windows
# python3.10 -m venv .venv && source .venv/bin/activate  # Linux/Mac

pip install -e ".[dev]"
uvicorn gravsense.api.main:app --reload
```

---

## UI walkthrough

The browser UI at `/` provides:

1. **Upload** — drag-and-drop or click to browse
2. **Method** — Grounded SAM or SegFormer
3. **Calibration cards** — Auto/Manual toggle for each value
   - *Reference width*: auto-detects a vehicle in the image (GroundingDINO)
   - *Pile height*: auto-estimated via Depth Anything V2 after the mask is computed
4. **Analyze** — single request, results appear inline
5. **Image tabs**:
   - `Original` — uploaded photo
   - `Debris Mask` — green overlay on detected debris
   - `Depth Map` — full-image plasma colormap (yellow = close, purple = far)
   - `Depth on Debris` — depth colors inside the mask only, greyed outside
6. **Stats** — detections, surface area, pile height (with source badge), volume
7. **Pipeline log** — every step with status, detected labels + confidence, calibration source, depth value, and a dedicated **Gravat (debris) volume** result card at the end

---

## API

### `POST /analyze`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | image | — | JPEG or PNG, max 10 MB |
| `method` | `grounded_sam` \| `segformer` | `grounded_sam` | Detection method |
| `reference_width_cm` | float | 100 | Manual fallback — real-world image width (cm) |
| `assumed_height_cm` | float | 30 | Manual fallback — pile height (cm) |
| `auto_calibrate` | bool | true | Run Depth Anything V2 + GroundingDINO auto-calibration in parallel |
| `include_overlay` | bool | true | Include base64 images in response |

**Error codes:**

| Code | Reason |
|------|--------|
| `400` | Uploaded file is not an image |
| `413` | File exceeds 10 MB limit |
| `422` | Image bytes could not be decoded |

**Response (JSON):**

```json
{
  "method": "grounded_sam",
  "n_detections": 2,
  "mask_pixel_count": 84320,
  "surface_area_cm2": 3372.8,
  "volume_cm3": 142870.4,
  "reference_width_cm": 298.5,
  "assumed_height_cm": 42.3,
  "detected_labels": ["pile of construction debris", "rubble"],
  "detection_scores": [0.871, 0.743],
  "processing_time_s": 5.12,
  "auto_reference_width_cm": 298.5,
  "auto_reference_object": "truck",
  "auto_reference_confidence": 0.912,
  "auto_pile_height_cm": 42.3,
  "overlay_b64": "...",
  "depth_map_b64": "...",
  "depth_on_debris_b64": "..."
}
```

### `GET /health`

```json
{ "status": "ok", "version": "1.0.0" }
```

---

## Calibration — how it works automatically

**Reference width** (`auto_reference_object`)

GroundingDINO scans for vehicles with the query `"truck. lorry. van. car. vehicle."`.
When one is found, its bounding-box pixel width is compared against a known real-world
width (truck ≈ 240 cm, car ≈ 180 cm) to compute the full-image real-world scale:

```
reference_width_cm = known_object_cm × (image_width_px / bbox_width_px)
```

**Pile height** (`auto_pile_height_cm`)

Depth Anything V2 Metric Outdoor Small produces a per-pixel depth map in metres.
The debris mask from detection is dilated by 40 px to sample surrounding ground pixels:

```
height = |median(ground_ring_depths) − median(pile_depths)| × 100
```

When no vehicle is found or auto-calibrate is disabled, the API falls back to the
manual `reference_width_cm` / `assumed_height_cm` query parameters.

---

## Detection robustness

**Expanded text query**

The Grounded SAM detector uses a broad open-vocabulary query that covers the range of
debris scenes found on construction sites:

```
construction debris · rubble · gravel · building waste · gravat ·
demolition waste · rubble pile · sand pile · broken concrete ·
concrete rubble · crushed stone · aggregate · debris in truck ·
rubble in truck bed · construction waste · excavation debris ·
mixed debris · rock debris
```

Thresholds are set to `box_threshold=0.20 / text_threshold=0.15` to catch debris inside
truck beds and skips, where the visual context lowers model confidence.

**Vegetation false-positive filter**

After GroundingDINO returns detections, any box whose matched label contains a
vegetation term (`tree`, `plant`, `bush`, `grass`, `shrub`, `foliage`, …) is discarded
before SAM generates its mask. This prevents trees and hedges visible in the background
from being counted as debris.

---

## Project structure

```
gravsense/
├── core/
│   ├── grounded_sam.py        ← GroundingDINO + SAM detector
│   ├── segformer_detector.py  ← ADE20K SegFormer baseline (for comparison)
│   ├── depth_estimator.py     ← Depth Anything V2 + plasma colormap
│   ├── auto_calibrate.py      ← GroundingDINO reference-width detection
│   └── volume.py              ← mask → surface area → volume
└── api/
    ├── main.py                ← FastAPI routes, CORS, file limit, async inference
    ├── schemas.py             ← Pydantic response models
    └── static/
        └── index.html         ← Single-file browser UI (no build step)

tests/
└── test_api.py                ← pytest with fully mocked inference (no GPU needed)

notebooks/
└── (original Kaggle research — SegFormer exploration, K-Means, DeepLabV3)

Dockerfile                     ← Python 3.10-slim, 2-stage build
docker-compose.yml             ← API + HuggingFace model cache volume
ruff.toml                      ← linter config
.github/workflows/ci.yml       ← lint → test (py 3.10 + 3.11) → docker build → GHCR push
pyproject.toml
requirements.txt
```

---

## Run tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

All model inference is mocked. CI runs in ~60 seconds without a GPU.

---

## Models used

| Model | Source | Purpose |
|-------|--------|---------|
| `IDEA-Research/grounding-dino-tiny` | HuggingFace | Text → bounding boxes |
| `facebook/sam-vit-base` | HuggingFace | Boxes → pixel masks |
| `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` | HuggingFace | Monocular depth (metres) |
| `nvidia/segformer-b0-finetuned-ade-512-512` | HuggingFace | Semantic segmentation baseline |

All models download automatically on first request and are cached in the Docker volume.
No API keys or accounts required.

---

## Next upgrade — YOLO11-seg fine-tuned on construction waste

Fine-tune `yolo11n-seg.pt` on the Roboflow construction-waste dataset for
real-time (30+ FPS) inference — useful for video feeds or drone footage.
This adds a fourth `method=yolo` option to the existing API without changing
any other component.

---

<!--
## CV line

Built **GravSense**: a production debris detection and volume estimation system.
Replaced a fragile ADE20K dominant-class heuristic with an open-vocabulary
**Grounded SAM** pipeline (GroundingDINO text → bounding boxes → SAM masks).
Tuned the detector for real-world scenes: expanded the text query to cover
truck-bed and skip-container debris, lowered detection thresholds for low-contrast
scenes, and added a post-detection **vegetation filter** that drops tree/plant
false positives before SAM mask generation.
Added **Depth Anything V2** (metric outdoor) for automatic pile-height estimation
from a single RGB photo, and **GroundingDINO auto-calibration** for reference-width
detection — both running in parallel as part of the same API request.
Deployed as an async **FastAPI** service (CORS, 10 MB file guard, thread-pool executor,
Pydantic schemas), containerised with **Docker Compose** (model cache volume), browser
UI with depth visualisation and a dedicated **Gravat (debris) volume** result card,
and a **GitHub Actions** CI/CD pipeline (ruff lint, mocked-inference pytest on
Python 3.10 + 3.11, Docker build + push to GHCR on every green main commit).
-->
