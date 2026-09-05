# SkinTokens — build for RunPod / local Docker
.PHONY: ckpts build-base build-queue build push-base push-queue push

DOCKER_BUILD := DOCKER_BUILDKIT=1 docker build
IMAGE_USER ?= sybiote

ckpts:
	@echo ">>> Downloading checkpoints to ckpts/ (~6-8GB)..."
	python download.py --model
	@echo ">>> Organizing ckpts/ for volume seeding..."
	@mkdir -p ckpts/experiments ckpts/models
	@cp -a experiments/. ckpts/experiments/ 2>/dev/null || true
	@cp -a models/. ckpts/models/ 2>/dev/null || true
	@echo ">>> Done. Verify:"
	@ls -lh ckpts/experiments/skin_vae_2_10_32768/last.ckpt 2>/dev/null || true
	@ls -lh ckpts/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt 2>/dev/null || true
	@ls ckpts/models/Qwen3-0.6B/ 2>/dev/null || true

build-base:
	$(DOCKER_BUILD) -f Dockerfile.base -t $(IMAGE_USER)/skintokens-base:latest .

build-queue: build-base
	$(DOCKER_BUILD) --build-arg BASE_IMAGE=$(IMAGE_USER)/skintokens-base:latest \
		-f Dockerfile -t $(IMAGE_USER)/skintokens-queue:latest .

build: build-queue

push-base:
	docker push $(IMAGE_USER)/skintokens-base:latest

push-queue:
	docker push $(IMAGE_USER)/skintokens-queue:latest

push: push-base push-queue
