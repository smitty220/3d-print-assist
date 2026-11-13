# Multi-stage build for 3D Print Assist (self-hosted / Unraid).
# Local development with ./dev.sh is unchanged.

FROM node:22-slim AS frontend-build
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# OpenCASCADE (build123d / OCP) and headless rendering runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglu1-mesa \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY mcp/ /app/mcp/
COPY backend/pyproject.toml backend/uv.lock /app/backend/
COPY backend/src/ /app/backend/src/

WORKDIR /app/backend
RUN uv sync --frozen --no-dev

COPY --from=frontend-build /src/frontend/dist /app/static

ENV PRINT_ASSIST_DATA=/data \
    PRINT_ASSIST_STATIC_DIR=/app/static \
    PYTHONUNBUFFERED=1

VOLUME ["/data"]
EXPOSE 8010

CMD ["uv", "run", "--no-dev", "uvicorn", "print_assist.app:app", "--host", "0.0.0.0", "--port", "8010"]
