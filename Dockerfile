# SkinTokens API (production) — RunPod Serverless
# Thin layer on Dockerfile.base: FastAPI + uvicorn + entrypoint.
#
# Build context must be SkinTokens/ (monorepo subdir).
#
#   DOCKER_BUILDKIT=1 docker build -f Dockerfile.base -t sybiote/skintokens-base:latest .
#   DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE=sybiote/skintokens-base:latest -f Dockerfile -t sybiote/skintokens-api:latest .

ARG BASE_IMAGE=docker.io/sybiote/skintokens-base:latest
FROM ${BASE_IMAGE}

COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache -r requirements-api.txt

# Re-copy everything Dockerfile.base bakes in. RunPod builds this file from git
# on top of a base image pulled from Docker Hub, so anything this branch changes
# under src/, configs/ or bpy_server.py is invisible at runtime unless it is
# copied again here -- the base carries whatever was current when it was last
# pushed. Skipping this silently ships stale code: a fix can be committed,
# built and rolled out while every worker still runs the old file.
COPY src/ src/
COPY configs/ configs/
COPY bpy_server.py .

COPY api.py .
COPY bpy_supervisor.py .
COPY runtime.py .
COPY scripts/ scripts/
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV SKINTOKENS_APP_DIR=/app
ENV PORT=8080
ENV PORT_HEALTH=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:${PORT_HEALTH:-${PORT:-8080}}/ping || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["sh", "-c", "exec python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --no-access-log"]
