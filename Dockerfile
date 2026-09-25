FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Pin the uv version for reproducible builds, e.g.:
# FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependency manifests only — cached layer; deps come from uv.lock, not pip.
# The BuildKit cache mount keeps uv's download cache OUTSIDE the image
# layers, so a changed lockfile only downloads NEW packages instead of
# re-downloading the whole set (layers are immutable and would otherwise
# discard the cache together with the venv).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Application code (src layout: law_api + data packages) + non-secret config.
COPY src ./src
COPY config ./config

# Install the project itself into the synced venv.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000

# --no-sync: the venv is fully synced at build time — container start must
# NOT re-run the sync (no re-download, no reinstall, no resolution).
CMD ["uv", "run", "--no-dev", "uvicorn", "law_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
