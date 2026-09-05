# RunPod (Serverless) — SkinTokens queue worker

SkinTokens runs on a RunPod Serverless **queue** endpoint with a **network
volume** for checkpoints, shared with HY-Motion, Kimodo and UniRig.

## Why a queue and not a load balancer

The load-balancer endpoint could not serve this workload. Its gateway gives up
around 30s and returns 502 while the worker keeps going — `leon.glb` (86,855
verts) logged `POST /rig 200 32191ms` server-side for a request the client had
already lost. The GPU work was paid for and discarded, and anything much past
16k vertices sat in that range.

| model | verts | work | load balancer | queue |
|-------|-------|------|---------------|-------|
| Guardian_male_fixed.glb | 16,674 | ~22s | 200 | COMPLETED 59.2s |
| leon.glb | 86,855 | ~32s | **502, always** | COMPLETED 31.2s |

A queue job has no such ceiling: a cold worker makes a job slower, never failed.

The HTTP/FastAPI deployment is preserved on the **`runpod-load-balancer`**
branch if it is ever needed again.

## Layout

| File | Role |
|------|------|
| `Dockerfile.base` | CUDA 12.8 + bpy + PyTorch 2.7 cu128 + flash-attn + ML deps; checkpoints **not** baked in |
| `Dockerfile` | Queue worker on top of the base (built by RunPod from this branch) |
| `handler.py` | `runpod.serverless.start`; owns `bpy_server`, warms the model at import |
| `docker-entrypoint.sh` | Links checkpoints in from the volume, verifies them, execs the handler |

`Dockerfile` **re-copies `src/`, `configs/` and `bpy_server.py`**. RunPod builds
it on top of a base image pulled from Docker Hub, so anything changed in those
paths is invisible at runtime unless copied again — skipping it ships stale code
while every signal (build completed, new image tag, workers rolled out) says
otherwise.

## Job contract

Payloads travel by reference. RunPod caps a `/status` body near 2MB, and base64
puts leon at 6.7MB in and ~9MB out, so the job carries presigned S3 URLs and the
handler returns a receipt. The worker holds no AWS credentials: each URL is a
capability scoped to one object.

```jsonc
// POST /v2/{endpointId}/run
{"input": {
  "op": "rig",
  "input_url":  "<presigned GET for the staged upload>",
  "output_url": "<presigned PUT for the result>",
  "output_format": "glb",          // or "json"
  "use_postprocess": true,          // default true
  // optional generation knobs: top_k, top_p, temperature,
  // repetition_penalty, num_beams, do_sample, voxel_resolution
}}
```

```jsonc
// GET /v2/{endpointId}/status/{id}
{"status": "COMPLETED",
 "output": {"bytes": 1473432, "content_type": "model/gltf-binary"}}
```

Handler failures come back as `{"error": "...", "code": "..."}` rather than a
dead worker. Presigned URLs are never echoed into an error message.

## Endpoint settings

| Setting | Value |
|---------|-------|
| Type | **Queue** (fixed at creation — cannot be changed later) |
| Branch | `runpod-queue` |
| Data center | **EU-RO-1** (where the volume lives) |
| Network volume | `nirvana` (`vpwhvs7cia`) |
| Execution timeout | 600000 ms |
| Workers | min 0, max 2–3 |

No `PORT` / `PORT_HEALTH` and no exposed ports: the worker pulls jobs rather
than serving requests, so readiness is the SDK connecting, not a socket
accepting. Leave the **Model** field blank — the checkpoints come from the
network volume, which is faster and not limited to one model per endpoint.

The entrypoint links `/runpod-volume/skintokens/experiments` → `/app/experiments`
and `models` → `/app/models`, verifies them, then execs `handler.py`, which
starts `bpy_server` on :59876 and loads TokenRig (~15s) before taking jobs.

## Client

`SKINTOKENS_ENDPOINT_ID` selects the endpoint; `SKINTOKENS_RUNPOD_API_KEY` is an
endpoint-scoped key, falling back to the account-wide `RUNPOD_API_KEY` with a
one-time warning. See `nirvana-animate-saas/src/server/skintokens/queue-client.ts`.

## Checkpoints on the volume

```
/runpod-volume/skintokens/
├── experiments/
│   ├── articulation_xl_quantization_256_token_4/grpo_1400.ckpt
│   └── skin_vae_2_10_32768/last.ckpt
└── models/Qwen3-0.6B/
```

Seed once with `python download.py --model` and copy the result up, or pull
directly on a pod attached to the volume.
