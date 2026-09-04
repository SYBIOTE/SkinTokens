# RunPod (Serverless) — SkinTokens API

Deploy SkinTokens (TokenRig) on **[RunPod](https://www.runpod.io/)** Serverless with a **network volume** for checkpoints. This guide uses a **shared volume** with HY-Motion and UniRig.

## Layout

| File | Role |
|------|------|
| `Dockerfile.base` | CUDA 12.8 + bpy + PyTorch 2.7 cu128 + flash-attn + ML deps; checkpoints **not** baked in |
| `Dockerfile` | FastAPI API on top of `BASE_IMAGE` |
| `docker-entrypoint.sh` | Symlink volume → `/app/experiments` + `/app/models`, verify checkpoints |
| `bpy_supervisor.py` | Supervised `bpy_server` subprocess (started from FastAPI lifespan) |
| `scripts/ensure_checkpoints.py` | Fail fast if weights are missing |
| `runtime.py` | TokenRig inference + UniRig-compatible JSON serialization |
| `api.py` | HTTP service (`POST /rig`, `GET /ping`) |

## Checkpoints (~6–8 GB)

Three items required at runtime:

```
experiments/skin_vae_2_10_32768/last.ckpt
experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt
models/Qwen3-0.6B/          # config only
```

Download locally (one-time): `make ckpts` — see [ckpts/README.md](ckpts/README.md).

## Shared network volume

Recommended layout on one volume (same datacenter as HY-Motion / UniRig):

```
/runpod-volume/
  ckpts/          ← HY-Motion
  unirig/         ← UniRig
  skintokens/
    experiments/
      skin_vae_2_10_32768/last.ckpt
      articulation_xl_quantization_256_token_4/grpo_1400.ckpt
    models/
      Qwen3-0.6B/
```

**Volume size:** add ~8 GB for SkinTokens on top of existing HY-Motion + UniRig usage.

### Seed weights via S3 API

From `SkinTokens/` after `make ckpts`:

```bash
aws s3 sync ckpts/experiments/ s3://YOUR_VOLUME_ID/skintokens/experiments/ \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/

aws s3 sync ckpts/models/ s3://YOUR_VOLUME_ID/skintokens/models/ \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/
```

Verify all three checkpoint paths exist under `skintokens/`.

## Build & push

Context must be **`SkinTokens/`** (monorepo):

```bash
cd SkinTokens
DOCKER_BUILDKIT=1 docker build -f Dockerfile.base -t YOUR_USER/skintokens-base:latest .
docker push YOUR_USER/skintokens-base:latest

docker build --build-arg BASE_IMAGE=docker.io/YOUR_USER/skintokens-base:latest \
  -f Dockerfile -t YOUR_USER/skintokens-api:latest .
docker push YOUR_USER/skintokens-api:latest
```

**Note:** `api.py`, `runtime.py`, and `docker-entrypoint.sh` live in the thin `Dockerfile` layer. Rebuild **only** `Dockerfile` after those change. Rebuild `Dockerfile.base` when `src/`, `configs/`, or ML deps change.

In RunPod Git/Docker build, set build arg **`BASE_IMAGE=docker.io/YOUR_USER/skintokens-base:latest`**.

### Troubleshooting Git build: `skintokens-base:latest: pull access denied`

RunPod is pulling `docker.io/library/skintokens-base:latest` (no Docker Hub user). Fix:

1. **Build arg** in RunPod endpoint → Edit → Build → add:
   - Name: `BASE_IMAGE`
   - Value: `docker.io/YOUR_USER/skintokens-base:latest`
2. **Push base first:** `docker push YOUR_USER/skintokens-base:latest` (repo must be **public**, or add registry credentials in RunPod).

## RunPod Serverless endpoint

1. Attach the **same network volume** as HY-Motion / UniRig (same datacenter).
2. Worker image: **`skintokens-api`**.
3. **Load balancer** endpoint (direct HTTP to FastAPI paths).

### Environment

| Variable | Example | Notes |
|----------|---------|--------|
| `SKINTOKENS_APP_DIR` | `/app` | App root |
| `SKINTOKENS_CKPTS_ROOT` | `/runpod-volume/skintokens` | Optional; auto-detected if dir exists |
| `SKINTOKENS_MODEL_CKPT` | `experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt` | TokenRig checkpoint (default) |
| `NVIDIA_DRIVER_CAPABILITIES` | `graphics,compute,utility` | Required for Blender headless |
| `PORT` | `8080` | HTTP server port |
| `PORT_HEALTH` | `8080` | Health probe port |

Entrypoint links `/runpod-volume/skintokens/experiments` → `/app/experiments` and `models` → `/app/models`, then execs uvicorn as the foreground process. `bpy_server` is started and supervised from FastAPI lifespan (`bpy_supervisor.py`).

### Suggested endpoint settings

| Setting | Value | Why |
|---------|--------|-----|
| GPU | L4 24GB or RTX 4090 | TokenRig + VAE + Qwen backbone needs ≥14 GB VRAM |
| Datacenter | Same as network volume | Avoid cross-region volume latency |
| Request timeout | 600s | Unified autoregressive rigging can take several minutes |
| Active workers | `1` | Avoid cold start on every request |
| Max workers | `1` | One GPU process per worker |
| Expose HTTP Ports | `8080` | Must match `PORT` / `PORT_HEALTH` |

### Cold start

Startup sequence:

1. Symlink volume → `/app/experiments` and `/app/models`
2. uvicorn starts; FastAPI lifespan starts `bpy_server` and waits for `:59876/ping`
3. HTTP server listens; `/ping` returns **204** until bpy + TokenRig are ready
4. Background thread loads TokenRig + VAE from the volume
5. `/ping` returns **200** when bpy and models are ready (~2–5 min typical)

`/rig` uses a **sync** handler so long GPU jobs do not block the event loop — `/ping` stays responsive during inference (required for RunPod health probes).

Check worker logs for:

```
>>> [runtime] Loading TokenRig from ...
>>> [runtime] TokenRig ready (...s elapsed)
>>> [api] Background model load finished — /ping will return 200
```

### HTTP API

Auth (load balancer): `Authorization: Bearer YOUR_RUNPOD_API_KEY`

```bash
# Health
curl -sS https://YOUR_ID.api.runpod.ai/ping \
  -H "Authorization: Bearer $RUNPOD_API_KEY"

# Rig (JSON — UniRig-compatible schema)
curl -sS -X POST https://YOUR_ID.api.runpod.ai/rig \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -F "file=@model.glb" \
  -F "output_format=json"

# Rig (GLB export)
curl -sS -X POST https://YOUR_ID.api.runpod.ai/rig \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -F "file=@model.glb" \
  -F "output_format=glb" \
  -o rigged.glb
```

Optional form fields: `top_k`, `top_p`, `temperature`, `repetition_penalty`, `num_beams`, `use_skeleton`, `use_transfer`, `use_postprocess`.

### Probes

- `GET /ping` — **204** while TokenRig loads, **200** when ready
- `GET /health` — same semantics as `/ping`

## References

- Local checkpoints: [ckpts/README.md](ckpts/README.md)
- UniRig RunPod (shared volume): [../UniRig/RUNPOD.md](../UniRig/RUNPOD.md)
- HY-Motion RunPod: [../HY-Motion-1.0/RUNPOD.md](../HY-Motion-1.0/RUNPOD.md)
