# Spine Screening API

Backend API for the Spine Opportunistic Screening research project.

This service is intended for research and screening support. It must not present
measurements or model outputs as diagnoses. Results require clinical review.

## Current status

The API loads independent AP/PA four-point and LA six-point HRNet-W18 CenterNet
models during application startup. Both pipelines implement deterministic image
preprocessing, landmark decoding, spine-chain candidate selection, geometry,
and measurement-only vertebral morphology. Each pipeline advertises readiness
only after strict checkpoint loading and a full-size warm-up pass. A failure in
one model does not make the other pipeline unavailable.

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
- LA readiness: `http://localhost:8000/api/v1/la/health/ready`
- LA model information: `http://localhost:8000/api/v1/la/model`
- LA six-point prediction: `POST http://localhost:8000/api/v1/la/predictions`

The LA response returns `top_left`, `top_middle`, `top_right`, `bottom_left`,
`bottom_middle`, and `bottom_right` for each candidate in original-image pixel
coordinates. LA Cobb output is geometry agreement calculated from the four
outer landmarks. It is not a clinical AP/PA Cobb measurement.

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
        predictions.py    AP/PA single-radiograph HTTP workflow
        la.py             LA readiness, metadata, and prediction workflow
  schemas/
    health.py             Health response contracts
    prediction.py         AP/PA landmark and measurement response contracts
    la_prediction.py      LA six-point response contracts
    error.py              Stable problem-detail response
    model.py              Public model-release contract
  services/
    screening.py          AP/PA use-case orchestration
    la_screening.py       LA use-case and outer-landmark adapter
  inference/
    runtime.py            Safe AP/PA and LA lifecycle, warm-up, and inference
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
SPINE_API_MODEL_PATH=app/weight/AP_PA_best_center_f1.pt
SPINE_API_LA_MODEL_PATH=app/weight/LA_best_usable_recall.pt
SPINE_API_INFERENCE_CONCURRENCY=1
SPINE_API_LOG_LEVEL=INFO
```

The AP/PA artifact at `app/weight/AP_PA_best_center_f1.pt` and LA checkpoint at
`app/weight/LA_best_usable_recall.pt` are copied into the container. The
concurrency limit is shared across both models. A fully successful deployment
returns `200` from all readiness checks:

- `GET /api/v1/health/live`
- `GET /api/v1/health/ready`
- `GET /api/v1/la/health/ready`

Update the decoder's minimum peak-confidence cutoff without changing learned
model tensors:

```powershell
.venv\Scripts\python.exe scripts\update_peak_threshold.py `
  app\weight\AP_PA_best_center_f1.pt 0.1
```

Rebuild the API image after changing the artifact so the deployed container
receives the updated checkpoint.

The hosted API remains a research demonstration. Upload only anonymized test
images and do not use the service for identifiable patient data or diagnosis.

## Cloudflare named tunnel

The Compose stack runs the API and Cloudflare's official `cloudflared`
container on one private Docker network. A remotely managed named tunnel gives
the API a stable hostname without exposing port `8000` beyond the local machine.

In the Cloudflare dashboard, create or select a named tunnel, add a published
application route, and set its service URL to `http://api:8000`. From **Add a
replica**, copy the tunnel token into the ignored local `.env` file:

```text
TUNNEL_TOKEN=<tunnel-token>
```

Start the API and named tunnel in the background:

```powershell
docker compose -f compose.tunnel.yaml up --build --detach
docker compose -f compose.tunnel.yaml logs --follow tunnel
```

The tunnel should become healthy in Cloudflare and keep the configured public
hostname across container restarts. Stop and remove the local containers and
network with:

```powershell
docker compose -f compose.tunnel.yaml down
```

The API currently has no authentication. Do not upload patient-identifiable
images or expose it as a clinical diagnostic service.
