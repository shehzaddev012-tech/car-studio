# Car Studio — AI Background Generator

> Upload a car photo taken anywhere (driveway, street, garage). Get back the **same car**, pixel-faithful, on a professional dealership studio backdrop.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (React SPA)                                            │
│  UploadZone → createJobs() POST /api/jobs                       │
│  JobQueue ← WebSocket /ws/jobs  ←  Redis pub/sub               │
│  BeforeAfterSlider  ← GET /api/jobs/:id/download               │
└───────────────────────┬─────────────────────────────────────────┘
                        │  HTTP + WebSocket
┌───────────────────────▼─────────────────────────────────────────┐
│  FastAPI API (stateless, horizontally scalable)                  │
│  POST /api/jobs → save image → DB job record → Celery queue     │
│  GET  /api/jobs/:id → read from SQLite / Postgres               │
│  WS   /ws/jobs  → aioredis subscribe → forward to client        │
└───────────┬──────────────────────────────┬──────────────────────┘
            │  SQLite / Postgres           │  Redis pub/sub
┌───────────▼──────────┐     ┌────────────▼──────────────────────┐
│  Database (job state) │     │  Redis (Celery broker + pub/sub)  │
└──────────────────────┘     └──────────┬────────────────────────┘
                                         │  task queue
┌────────────────────────────────────────▼────────────────────────┐
│  Celery Worker Pool (2–N processes)                             │
│                                                                  │
│  process_job(job_id)                                            │
│    1. segment_car()         rembg U²-Net  → car RGBA + mask     │
│    2. (car pixels locked)   never touched again                  │
│    3. generate_background() deterministic PIL studio backdrop    │
│    4. generate_shadow()     3-layer Gaussian soft shadow         │
│    5. composite_layers()    background → shadow → car           │
│    6. glass_cleanup_stub()  no-op (see upgrade path below)      │
│    7. quality_check()       SSIM ≥ 0.82 on masked car region   │
│    ✓ save result → storage                                       │
│    ✓ update DB → publish Redis → WebSocket → client             │
└─────────────────────────────────────────────────────────────────┘
            │
┌───────────▼───────────────────────────────────────────────────┐
│  Storage (local disk / swappable to S3)                        │
│    uploads/{job_id}.{ext}   ← original                        │
│    results/{job_id}.jpg     ← composite output                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
# 1. Clone / unzip the project
cd car-studio

# 2. Start all services (Redis + API + Worker + Frontend dev server)
docker compose up --build

# 3. Open the app
open http://localhost:5173
```

> **First run note:** `rembg` downloads the U²-Net model (~170 MB) the first time a worker processes a job. Subsequent runs are fast. The model is baked into the Docker image at build time.

### Option B — Local development (no Docker)

**Prerequisites:** Python 3.11+, Node.js 20+, Redis running on `localhost:6379`

```bash
# ── Backend ────────────────────────────────────────────────────────
cd backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start API server
uvicorn app.main:app --reload --port 8000

# In a second terminal — start the Celery worker
celery -A celery_worker.celery_app worker --loglevel=info --concurrency=2

# ── Frontend ───────────────────────────────────────────────────────
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Mock AI mode (no model download)

Set `MOCK_AI=true` to use OpenCV GrabCut instead of rembg. The full pipeline (background, shadow, compositing, QC) still runs — only the segmentation step is swapped.

```bash
MOCK_AI=true uvicorn app.main:app --reload
# Worker:
MOCK_AI=true celery -A celery_worker.celery_app worker --loglevel=info
```

Or with Docker Compose:
```bash
MOCK_AI=true docker compose up
```

---

## AI Pipeline Details

| Step | Implementation | Notes |
|------|---------------|-------|
| 1. Segmentation | `rembg` U²-Net (170 MB, local) | Alpha-matting enabled for clean edges on mirrors/antennae |
| 2. Car preservation | PIL alpha channel | Car pixels are never passed to any generative model |
| 3. Background | Pure PIL/NumPy (deterministic) | Radial gradient + infinity curve + floor zone |
| 4. Shadow | OpenCV 3-layer Gaussian blend | Perspective-squashed silhouette, area-light diffusion |
| 5. Compositing | `PIL.Image.alpha_composite` | background → shadow → car; pixel-exact |
| 6. Glass cleanup | **Stub** (no-op) | See upgrade path below |
| 7. Quality gate | scikit-image SSIM on masked region | Threshold 0.82; fails job if car region drifts |

---

## Swapping AI Providers

All AI logic is behind the `AIBackgroundService` interface in [backend/app/services/ai_background.py](backend/app/services/ai_background.py).

### Swap segmentation model (e.g. SAM 2)

Edit `pipeline.segment_car()` in [backend/app/services/pipeline.py](backend/app/services/pipeline.py):

```python
def segment_car(image: Image.Image, mock: bool = False) -> Tuple[Image.Image, Image.Image]:
    # Replace this body with your SAM 2 / custom model call.
    # Must return (car_rgba: RGBA Image, mask: L-mode Image)
    ...
```

### Enable glass cleanup

Edit `glass_cleanup_stub()` in `pipeline.py` — the function signature and position in the pipeline are already correct. Replace the body with:
1. Vehicle-part segmentation to detect window regions (SAM 2 + "windshield" prompt)
2. A masked inpainting call (SDXL-Inpaint / OpenCV `cv2.inpaint`) restricted to the glass mask only
3. Feathered blend back onto the composite

### Swap storage to S3

```env
# .env
STORAGE_BACKEND=s3
S3_BUCKET=my-dealership-bucket
S3_REGION=us-east-1
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
```

No code changes required — `get_storage()` in [backend/app/services/storage.py](backend/app/services/storage.py) returns the right adapter automatically.

### Swap database to Postgres

```env
DATABASE_URL=postgresql://user:password@host:5432/car_studio
```

No code changes required — SQLAlchemy handles it.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/jobs` | POST | Upload images, create jobs (returns immediately) |
| `/api/jobs/:id` | GET | Single job status + URLs |
| `/api/jobs?ids=id1,id2` | GET | Batch status |
| `/api/jobs/:id` | DELETE | Cancel pending job |
| `/api/jobs/:id/original` | GET | Stream original image |
| `/api/jobs/:id/download` | GET | Stream result image |
| `/ws/jobs` | WS | Live status push (subscribe by job ID) |
| `/health` | GET | Health check |
| `/docs` | GET | Auto-generated Swagger UI |

---

## Running Tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

Tests cover:
- All `JobRepository` state transitions (unit)
- Storage adapter: put/get/delete/exists/path-traversal guard (unit)
- Upload flow: single image, batch, bad MIME type, oversized file, cancel (integration)

---

## Known Limitations

1. **No authentication** — all jobs are world-readable by job ID. Add OAuth2/JWT for production.
2. **SQLite concurrency** — SQLite handles low concurrency well but use Postgres for > 5 simultaneous workers.
3. **rembg edge quality** — U²-Net occasionally struggles with:
   - Cars with the same colour as the background
   - Very dark cars on dark backgrounds
   - Motion blur / low-resolution inputs
   Production fix: upgrade to SAM 2 with a vehicle-category prompt.
4. **SSIM threshold tuning** — The default 0.82 may produce false-positive QC failures on images with heavy JPEG compression artifacts. Reduce to 0.75 if needed (edit `qc_ssim_threshold` in config).
5. **No result expiry** — result images accumulate on disk. Add a cleanup Celery beat task for production.
6. **Glass cleanup is a stub** — window reflections/dirt are not removed. See the production upgrade path above.
