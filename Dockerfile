FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    SPINE_API_ENVIRONMENT=production \
    SPINE_API_MODEL_DEVICE=cpu \
    SPINE_API_INFERENCE_CONCURRENCY=1

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /usr/local/bin/

RUN groupadd --gid 1000 user && \
    useradd --uid 1000 --gid 1000 --create-home user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

USER user
WORKDIR /home/user/app

COPY --chown=user:user pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user:user app ./app

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
