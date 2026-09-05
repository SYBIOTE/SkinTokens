"""
RunPod queue-endpoint handler for SkinTokens.

Swaps the transport only: the FastAPI app and the load-balancer deployment stay
as they are, and this module drives the same SkinTokensRuntime.

Why a queue at all: the load balancer drops any request that outruns its ~30s
gateway timeout *after* the GPU has finished. leon.glb logs `POST /rig 200
32191ms` server-side and the client still sees a 502, so the generation is paid
for and thrown away. A queue job has no such ceiling.

Payloads travel by reference, not by value. RunPod caps a /status body near 2MB;
base64 puts leon at 6.7MB in and ~9MB out, and even a small rigged model lands
on the line. So the job carries presigned S3 URLs and returns only a receipt:

    Job input:  {"op": "rig",
                 "input_url":  "<presigned GET>",
                 "output_url": "<presigned PUT>",
                 "output_format": "glb" | "json",
                 ...RigOptions fields}
    Job output: {"bytes": 1473368, "content_type": "model/gltf-binary"}
                or, for output_format=json, the rig JSON inline when it fits.
    On failure: {"error": "...", "code": "..."}

The worker holds no AWS credentials: each URL is a capability scoped to one
object, and expires on its own.
"""

import json
import os
import tempfile
import traceback
from typing import Any

import requests
import runpod

from bpy_supervisor import get_bpy_supervisor
from runtime import RigOptions, SkinTokensRuntime

OP_RIG = "rig"

# A presigned URL is not a secret we should echo into logs or job output, but a
# transfer failure has to say *something* useful. Cap what we report.
_HTTP_TIMEOUT = 300
_MAX_ERR = 400

_runtime: SkinTokensRuntime | None = None


def _error(code: str, message: str) -> dict:
    return {"error": message[:2000], "code": code}


def _get_runtime() -> SkinTokensRuntime:
    global _runtime
    if _runtime is None:
        _runtime = SkinTokensRuntime(app_dir=os.environ.get("SKINTOKENS_APP_DIR", "/app"))
    return _runtime


def _rig_options(payload: dict) -> RigOptions:
    """Build RigOptions from the job input, ignoring transport-only keys."""
    fields = RigOptions.__dataclass_fields__  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in payload.items() if k in fields}
    return RigOptions(**kwargs)


def _download(url: str, dest: str) -> None:
    with requests.get(url, stream=True, timeout=_HTTP_TIMEOUT) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)


def _upload(url: str, src: str, content_type: str) -> None:
    with open(src, "rb") as fh:
        r = requests.put(
            url,
            data=fh,
            headers={"Content-Type": content_type},
            timeout=_HTTP_TIMEOUT,
        )
    r.raise_for_status()


def _handle_rig(payload: dict) -> dict:
    input_url = payload.get("input_url")
    if not input_url:
        return _error("bad_input", "input_url is required")

    output_format = str(payload.get("output_format", "glb")).strip().lower()
    if output_format not in {"glb", "json"}:
        return _error("bad_input", "output_format must be 'glb' or 'json'")

    output_url = payload.get("output_url")
    if output_format == "glb" and not output_url:
        return _error("bad_input", "output_url is required for output_format=glb")

    try:
        options = _rig_options(payload)
    except TypeError as e:
        return _error("bad_input", f"invalid options: {e}")

    if not get_bpy_supervisor().is_healthy():
        return _error("unavailable", "bpy_server is not healthy")

    tmpdir = tempfile.mkdtemp(prefix="skintokens_rig_")
    try:
        src = os.path.join(tmpdir, "input.glb")
        try:
            _download(input_url, src)
        except Exception as e:  # noqa: BLE001 - the URL must not leak into the message
            return _error("input_fetch_failed", f"{type(e).__name__}: {str(e)[:_MAX_ERR]}")

        runtime = _get_runtime()

        if output_format == "json":
            data = runtime.generate_rig_json(src, options=options)
            # The 54k-sample rig JSON can exceed the /status cap on its own, so
            # send it through S3 too when the caller gave us somewhere to put it.
            if output_url:
                blob = os.path.join(tmpdir, "rig.json")
                with open(blob, "w") as fh:
                    json.dump(data, fh)
                _upload(output_url, blob, "application/json")
                return {"bytes": os.path.getsize(blob), "content_type": "application/json"}
            return data

        glb = runtime.export_rig_glb(src, options=options)
        out = os.path.join(tmpdir, "rigged.glb")
        with open(out, "wb") as fh:
            fh.write(glb)

        try:
            _upload(output_url, out, "model/gltf-binary")
        except Exception as e:  # noqa: BLE001
            return _error("output_put_failed", f"{type(e).__name__}: {str(e)[:_MAX_ERR]}")

        return {"bytes": os.path.getsize(out), "content_type": "model/gltf-binary"}
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def handler(job: dict) -> dict:
    payload = job.get("input") or {}

    op = payload.get("op", OP_RIG)
    if op != OP_RIG:
        return _error("bad_op", f"Unsupported op {op!r}; expected {OP_RIG!r}")

    try:
        return _handle_rig(payload)
    except Exception as e:  # noqa: BLE001 - a dead worker tells the caller nothing
        print(traceback.format_exc(), flush=True)
        return _error("rig_failed", f"{type(e).__name__}: {e}")


# bpy_server is a sidecar the runtime talks to over localhost. FastAPI started it
# from its lifespan; with no ASGI app here the handler owns it.
print(">>> [queue] starting bpy_server...", flush=True)
try:
    get_bpy_supervisor().start()
except Exception as e:  # noqa: BLE001 - jobs report the real error
    print(f">>> [queue] [WARNING] bpy_server failed to start: {e}", flush=True)

# Warm the model at worker start rather than on the first job, so the job that
# happens to arrive first does not also pay for the checkpoint load.
try:
    _get_runtime()
    print(">>> [queue] runtime ready", flush=True)
except Exception as e:  # noqa: BLE001
    print(f">>> [queue] [WARNING] runtime not loaded at startup: {e}", flush=True)

runpod.serverless.start({"handler": handler})
