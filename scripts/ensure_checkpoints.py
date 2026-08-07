#!/usr/bin/env python3
"""Verify SkinTokens checkpoints under {SKINTOKENS_APP_DIR}.

Invoked at container start (via docker-entrypoint.sh) — verify only; exit 1 if missing.
See ckpts/README.md and RUNPOD.md for seeding instructions.
"""

from __future__ import annotations

import os
import sys

REQUIRED_FILES = [
    "experiments/skin_vae_2_10_32768/last.ckpt",
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
]

REQUIRED_DIRS = [
    "models/Qwen3-0.6B",
]


def main() -> int:
    app_dir = os.environ.get("SKINTOKENS_APP_DIR", "/app")
    missing: list[str] = []

    for rel in REQUIRED_FILES:
        path = os.path.join(app_dir, rel)
        if not os.path.isfile(path):
            missing.append(rel)

    for rel in REQUIRED_DIRS:
        path = os.path.join(app_dir, rel)
        if not os.path.isdir(path):
            missing.append(rel + "/")

    if missing:
        print("ERROR: SkinTokens checkpoint(s) missing:", file=sys.stderr)
        for rel in missing:
            print(f"  - {os.path.join(app_dir, rel)}", file=sys.stderr)
        print(
            "Seed ckpts on a network volume (see RUNPOD.md) or mount: "
            "-v ./ckpts/experiments:/app/experiments:ro "
            "-v ./ckpts/models:/app/models:ro",
            file=sys.stderr,
        )
        return 1

    print(">>> SkinTokens checkpoints OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
