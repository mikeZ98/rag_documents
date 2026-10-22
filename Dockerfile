# syntax=docker/dockerfile:1.7
# --------------------------------------------------------------------------- #
# rag_documents API image — multi-stage, non-root, CPU-only torch.
#
#   docker compose build api        # or: docker build -t rag_documents-api .
#
# builder  resolves the locked graph with uv into /app/.venv (no dev group, no
#          eval extra) and installs the project non-editable, so the runtime
#          stage needs neither uv nor the source tree.
# runtime  same base image (the venv's interpreter symlink must resolve), no
#          compilers, no uv, runs as an unprivileged user with a read-only venv.
#
# Model weights are NOT baked in: they are downloaded on first start into
# HF_HOME, which docker-compose bind-mounts from ./.cache/huggingface so the
# weights already fetched for local development are reused.
# --------------------------------------------------------------------------- #
ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7

FROM ${UV_IMAGE} AS uv

# ------------------------------- builder ----------------------------------- #
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first: this layer is reused until uv.lock / pyproject.toml change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable \
 && /app/.venv/bin/python -c "import src.main, torch; assert not torch.cuda.is_available()"

# ------------------------------- runtime ----------------------------------- #
FROM ${PYTHON_IMAGE} AS runtime

ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --system --gid "${APP_GID}" app \
 && useradd --system --uid "${APP_UID}" --gid app --home-dir /home/app --create-home \
        --shell /usr/sbin/nologin app \
 && install -d -o app -g app /home/app/.cache/huggingface /app/documents

ENV PATH=/app/.venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/app/.cache/huggingface/sentence-transformers \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_HOME=/tmp/torch \
    TOKENIZERS_PARALLELISM=false \
    DOCUMENTS_DIR=/app/documents \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app
# Owned by root and not writable by `app`: the code cannot be modified at runtime.
COPY --from=builder /app/.venv /app/.venv

USER app:app
EXPOSE 8000

# Liveness only: /health answers 200 with status "degraded" while the index is
# empty, so an unindexed service is still reported as running. The long start
# period covers the first-boot model download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "15"]
