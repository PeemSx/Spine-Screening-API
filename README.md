# Spine Screening API

Backend API for the Spine Opportunistic Screening research project.

This service is intended for research and screening support. It must not present
measurements or model outputs as diagnoses. Results require clinical review.

## Current status

The API loads the compact HRNet-W18 CenterNet artifact once during application
startup. It implements deterministic image preprocessing, landmark decoding,
spine-chain candidate selection, Cobb geometry, and measurement-only vertebral
morphology. Readiness becomes successful only after strict checkpoint loading
and a full-size warm-up pass.

The demo API will accept one radiograph, process it synchronously, return the
prediction, and retain neither the uploaded image nor the result.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

## Local setup

```powershell
Copy-Item .env.example .env
uv sync --frozen
uv run --frozen uvicorn app.main:app --reload
```

The API documentation will be available at:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI document: `http://localhost:8000/openapi.json`
- Liveness: `http://localhost:8000/api/v1/health/live`
- Readiness: `http://localhost:8000/api/v1/health/ready`
- Model information: `http://localhost:8000/api/v1/model`
- Prediction: `POST http://localhost:8000/api/v1/predictions`

## Tests and linting

```powershell
uv run --frozen pytest
uv run --frozen ruff check .

# Optional real-checkpoint HTTP and pipeline smoke tests
$env:SPINE_RUN_MODEL_TESTS='1'
uv run --frozen pytest tests/test_integration/test_real_runtime.py -v
```

## Project layout

```text
app/
  api/
    dependencies.py       FastAPI-to-service dependency adapter
    v1/
      router.py           Versioned route composition
      endpoints/
        health.py         Liveness and model readiness
        model.py          Safe deployed-model metadata
        predictions.py    Single-radiograph HTTP workflow
  schemas/
    health.py             Health response contracts
    prediction.py         Landmark and measurement response contracts
    error.py              Stable problem-detail response
    model.py              Public model-release contract
  services/
    screening.py          Framework-independent use-case orchestration
  inference/
    runtime.py            Safe model lifecycle, warm-up, and inference
    manifest.py           Reproducible model artifact metadata
    model.py              HRNet/ResNet CenterNet architectures
    preprocessing.py      Decode, resize/pad, normalize, coordinate mapping
    decoder.py            Heatmap NMS and landmark reconstruction
    types.py              Internal inference value objects
  postprocessing/
    spine_chain.py        Duplicate suppression and chain optimization
    cobb.py               Measurement-only Cobb geometry
    morphology.py         Vertebral geometry and neighbor-relative features
  core/
    config.py             Validated environment settings
    lifespan.py           Process-wide resource startup and cleanup
    errors.py             Safe application errors and HTTP mapping
    logging.py            Payload-free application logging setup
  utils/
    image_validation.py   Bounded image-container validation
tests/          Automated tests mirroring the application
```

The dependency direction is `api -> services -> inference`. Pydantic schemas are
HTTP contracts; inference modules remain independent of FastAPI. `core` owns
cross-cutting process behavior, while `utils` contains narrow technical helpers.
Golden and focused tests protect the API pipeline against behavioral drift from
the canonical research implementation.

## Container deployment

The container listens on port `8000`, runs as non-root user ID `1000`, uses
CPU-only PyTorch packages, and defaults to production CPU inference with one
concurrent prediction.

The defaults work without runtime configuration. These environment variables
can still be supplied when an explicit override is preferred:

```text
SPINE_API_ENVIRONMENT=production
SPINE_API_MODEL_DEVICE=cpu
SPINE_API_INFERENCE_CONCURRENCY=1
SPINE_API_LOG_LEVEL=INFO
```

The compact checkpoint at `app/weight/best_center_f1.pt` is copied into the
container. A successful deployment must return `200` from both:

- `GET /api/v1/health/live`
- `GET /api/v1/health/ready`

Update the decoder's minimum peak-confidence cutoff without changing learned
model tensors:

```powershell
.venv\Scripts\python.exe scripts\update_peak_threshold.py `
  app\weight\best_center_f1.pt 0.1
```

Rebuild the API image after changing the artifact so the deployed container
receives the updated checkpoint.

The hosted API remains a research demonstration. Upload only anonymized test
images and do not use the service for identifiable patient data or diagnosis.

## Cloudflare Quick Tunnel

The development tunnel runs the API and Cloudflare's official `cloudflared`
container on one private Docker network. It publishes a temporary HTTPS URL
without requiring a Cloudflare account or exposing port `8000` beyond the local
machine.

Start the API and tunnel:

```powershell
docker compose -f compose.tunnel.yaml up --build
```

Wait for a log message containing a URL like:

```text
https://random-words.trycloudflare.com
```

Then test these public URLs:

```text
https://random-words.trycloudflare.com/api/v1/health/live
https://random-words.trycloudflare.com/api/v1/health/ready
https://random-words.trycloudflare.com/docs
```

To run in the background and follow only the tunnel logs:

```powershell
docker compose -f compose.tunnel.yaml up --build --detach
docker compose -f compose.tunnel.yaml logs --follow tunnel
```

Stop and remove the development containers and network:

```powershell
docker compose -f compose.tunnel.yaml down
```

Quick Tunnel URLs change whenever `cloudflared` restarts and have no uptime
guarantee. They are suitable for Postman and temporary integration testing, not
production. The API currently has no authentication, so never share its URL
publicly or upload patient-identifiable images. A stable frontend integration
will require a named tunnel, application authentication, and explicit CORS
configuration.
