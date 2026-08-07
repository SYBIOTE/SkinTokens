"""
SkinTokens (TokenRig) microservice: auto-rig 3D meshes via file upload.
Exposes POST /rig (JSON or GLB), GET /health, and GET /ping (RunPod liveness).

Models load in a background thread: HTTP listens immediately (/ping 204),
then TokenRig loads from the volume.

Usage:
    python -m uvicorn api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.requests import Request

from runtime import RigOptions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(levelname)-5s  %(message)s",
)
logger = logging.getLogger("skintokens.api")

ALLOWED_EXTENSIONS = {"obj", "fbx", "glb"}
DEFAULT_TOP_K = 5
DEFAULT_TOP_P = 0.95
DEFAULT_TEMPERATURE = 1.0
DEFAULT_REPETITION_PENALTY = 2.0
DEFAULT_NUM_BEAMS = 10

_runtime: object | None = None
_load_error: str | None = None


def _get_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _validate_extension(filename: str) -> str:
    ext = _get_ext(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format '.{ext}'. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return ext


def _save_upload(upload: UploadFile, dest: str) -> None:
    with open(dest, "wb") as handle:
        shutil.copyfileobj(upload.file, handle)


def _cleanup_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _get_runtime():
    if _load_error is not None:
        raise HTTPException(503, f"Runtime failed to load: {_load_error}")
    if _runtime is None:
        raise HTTPException(503, "Runtime not loaded yet — server is still starting")
    return _runtime


def _probe_response() -> Response | dict:
    if _load_error is not None:
        raise HTTPException(503, f"Runtime failed to load: {_load_error}")
    if _runtime is None:
        return Response(status_code=204)
    return {"status": "ok"}


def _load_runtime() -> None:
    global _runtime, _load_error
    from runtime import SkinTokensRuntime

    app_dir = os.environ.get("SKINTOKENS_APP_DIR", "/app")
    logger.info("Background model load started (app_dir=%s)", app_dir)
    try:
        _runtime = SkinTokensRuntime(app_dir=app_dir)
        logger.info("Background model load finished — /ping will return 200")
    except Exception as exc:
        logger.exception("Failed to load runtime: %s", exc)
        _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runtime, _load_error
    _runtime = None
    _load_error = None
    thread = threading.Thread(target=_load_runtime, daemon=True)
    thread.start()
    yield
    _runtime = None
    _load_error = None


app = FastAPI(title="SkinTokens API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/ping":
        return await call_next(request)

    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    client = request.client.host if request.client else "unknown"
    logger.info(
        "%s %s %s %.1fms client=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        client,
    )
    return response


@app.get("/ping")
def ping():
    """RunPod load-balancer probe (204 while initializing, 200 when ready)."""
    return _probe_response()


@app.get("/health")
def health():
    return _probe_response()


@app.post("/rig")
async def rig_mesh(
    file: UploadFile = File(...),
    output_format: str = Form("json"),
    top_k: int = Form(DEFAULT_TOP_K),
    top_p: float = Form(DEFAULT_TOP_P),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    repetition_penalty: float = Form(DEFAULT_REPETITION_PENALTY),
    num_beams: int = Form(DEFAULT_NUM_BEAMS),
    use_skeleton: bool = Form(False),
    use_transfer: bool = Form(False),
    use_postprocess: bool = Form(False),
):
    """Unified TokenRig pipeline. Returns UniRig-compatible JSON or rigged GLB."""
    output_format = output_format.strip().lower()
    if output_format not in {"json", "glb"}:
        raise HTTPException(400, "output_format must be 'json' or 'glb'")

    runtime = _get_runtime()
    filename = file.filename or "mesh.glb"
    ext = _validate_extension(filename)

    options = RigOptions(
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        use_skeleton=use_skeleton,
        use_transfer=use_transfer,
        use_postprocess=use_postprocess,
    )

    tmpdir = tempfile.mkdtemp(prefix="skintokens_rig_")
    try:
        input_path = os.path.join(tmpdir, f"input.{ext}")
        _save_upload(file, input_path)

        if output_format == "json":
            data = runtime.generate_rig_json(input_path, options=options)
            return JSONResponse(content=data)

        glb_bytes = runtime.export_rig_glb(input_path, options=options)
        return Response(
            content=glb_bytes,
            media_type="model/gltf-binary",
            headers={"Content-Disposition": 'attachment; filename="rigged.glb"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Rigging failed: {exc}") from exc
    finally:
        _cleanup_dir(tmpdir)
