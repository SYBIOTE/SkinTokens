# SkinTokens checkpoints (local mirror)

Weights are **not** baked into Docker images. Mount them at runtime or seed a RunPod network volume.

## Required files

```
experiments/skin_vae_2_10_32768/last.ckpt
experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt
models/Qwen3-0.6B/          # Qwen3 config only (no weight files)
```

## Download locally

From `SkinTokens/`:

```bash
make ckpts
# or: python download.py --model
```

This populates `experiments/` and `models/` in the repo root and copies them into `ckpts/` for volume seeding.

## Local Docker run

```bash
docker run --gpus all -p 8080:8080 \
  -v "$(pwd)/ckpts/experiments:/app/experiments:ro" \
  -v "$(pwd)/ckpts/models:/app/models:ro" \
  sybiote/skintokens-api:latest
```

## RunPod

See [RUNPOD.md](../RUNPOD.md) — upload under `s3://VOLUME_ID/skintokens/`.
