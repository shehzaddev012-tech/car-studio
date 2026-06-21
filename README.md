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
│    1. Vertex AI image-segmentation-001  (sole provider)         │
│    2. SAM2 mask refinement + merge + edge refine                │
│    3. (vehicle pixels locked)   never regenerated               │
│    4. generate_background()   premium cyclorama studio          │
│    5. generate_shadow()         footprint + contact shadow      │
│    6. composite_layers()        background → shadow → vehicle   │
│    7. validate_output()         SSIM + LPIPS + coverage gates   │
│    ✗ NO fallbacks — fail job on any segmentation/QC error        │
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

# 2. Configure .env (required — app refuses to start without these)
#    GOOGLE_CLOUD_PROJECT=your-project-id
#    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# 3. Start all services
docker compose up --build

# 4. Open the app
open http://localhost:5173
```

> **Startup validation:** API and Celery workers verify `GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`, live Vertex AI connectivity, and SAM2 model load **before accepting traffic**. No silent fallbacks.

### Option B — Local development (no Docker)

**Prerequisites:** Python 3.11+, Node.js 20+, Redis, GCP service account with Vertex AI User role

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-ai.txt

# Set GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS in .env

uvicorn app.main:app --reload --port 8000
celery -A celery_worker.celery_app worker --loglevel=info --concurrency=1

# ── Frontend ───────────────────────────────────────────────────────
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## Strict Pipeline Policy

**Vehicle preservation is non-negotiable.** The system never regenerates, repairs, or reconstructs vehicle pixels.

| Policy | Behaviour |
|--------|-----------|
| Part validation | Mirrors, wheels, lights, plate, roof rails, antenna, edges checked across all mask stages |
| Pre-composite gate | Rejects before studio output if any part may be missing |
| Post-composite gate | SSIM + LPIPS verify pixel fidelity |
| User error | `"Vehicle could not be isolated with sufficient accuracy. Please upload a clearer image."` |
| On failure | **No output image** — job marked failed |

A perfect failure is better than an incorrect dealership image.

**No fallbacks.** Missing GCP credentials, Vertex failures, SAM2 failures, or quality-gate failures all reject the job — never silently switch providers.

---

## AI Pipeline Details

| Step | Implementation | Notes |
|------|---------------|-------|
| 1. Primary segmentation | Vertex AI `image-segmentation-001` (prompt: car) | GCP required for production |
| 2. Refinement | SAM 2.1 Hiera Base+ | Box + point prompts from Vertex mask |
| 3. Mask merge | Confidence scoring + union fallback | Prevents trimmed mirrors/spoilers |
| 4. Edge refine | Morphological cleanup + boundary anti-aliasing | Sub-pixel alpha feather |
| 5. Vehicle pixels | PIL alpha composite | **Never regenerated** |
| 6. Background | Deterministic cyclorama gradient | Carvana/OEM-style light grey studio |
| 7. Shadow | Footprint + tire contact + progressive blur | Realistic ground contact |
| 8. Quality gate | SSIM + LPIPS + coverage + missing-part detection | Rejects degraded outputs |

### Required `.env`

```env
AI_PROVIDER=compositing
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
VERTEX_MIN_CONFIDENCE=0.55
QC_SSIM_THRESHOLD=0.85
QC_LPIPS_THRESHOLD=0.08
LPIPS_ENABLED=true
```

Enable the **Vertex AI API** and grant the service account **Vertex AI User**.

---

## Swapping Storage / Database

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
3. **Vertex AI dependency** — segmentation requires GCP credentials and network access. The application will not start without a successful Vertex connectivity probe.
4. **SSIM / LPIPS tuning** — Defaults (0.85 / 0.08) are strict for dealership quality. Adjust via `QC_SSIM_THRESHOLD` and `QC_LPIPS_THRESHOLD` if needed.
5. **No result expiry** — result images accumulate on disk. Add a cleanup Celery beat task for production.
6. **Glass cleanup is a stub** — window reflections are preserved from the original photo (vehicle pixels are never regenerated).
