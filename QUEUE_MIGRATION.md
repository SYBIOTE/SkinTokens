# SkinTokens → RunPod queue endpoint

Branch: `runpod-queue`. Mirrors `HY-Motion-1.0`'s migration, but the payload
shape is different enough that the transport cannot be copied wholesale.

## Why

The load-balancer endpoint drops any request that outruns its ~30s gateway
timeout, *after* the GPU has done the work. Measured on `l216ag73mpvqw4`:

| model | verts | server time | gateway result |
|-------|-------|-------------|----------------|
| Guardian_male_fixed.glb | 16,674 | 22.0s / 23.2s | 200 |
| leon.glb | 86,855 | 32.2s / 37.0s | **502** |

The worker logged `POST /rig 200 32191.0ms` for the same request the client saw
as a 502. Anything past ~16k vertices is at risk today, and the user pays for a
generation they never receive. A queue endpoint has no such ceiling: submit to
/run, poll /status until terminal.

## The payload problem (differs from HY-Motion)

HY-Motion returns JSON that fits inline. SkinTokens takes a GLB in and returns a
rigged GLB out, and RunPod caps a /status body near 2MB. Base64 measured:

| | raw | base64 |
|--|-----|--------|
| leon.glb in | 5.05MB | **6.73MB** |
| leon rigged out | ~6.8MB | **~9MB** |
| guardian in | 1.39MB | 1.85MB |
| guardian rigged out | 1.47MB | **1.96MB** |

Inline base64 is not viable: leon exceeds the cap several times over, and even
guardian's output lands on the line. The job must carry references, not bytes.

## Design: presigned S3 both directions

Nothing new has to be built to hold the bytes. The studio **already** uploads the
input GLB to S3 staging before calling /rig — `uploadGlbToStaging` on the client,
`stagingObjectKey(userId, chatId)` on the server — and today the SaaS downloads
it again just to forward it as multipart. The queue version simply hands the
worker a presigned URL for the object that is already there, and the download
in `rig-from-staging.ts` goes away.

The result rides the same staging area: a sibling key under the same chat
prefix, written with the existing `lifecycle=staging` tag so the bucket's expiry
rule reaps it, and deleted explicitly on success exactly as
`deleteStagingInputGlb` already does for the input. Ephemeral by construction —
no new bucket, no new lifecycle policy, no long-lived artifacts.

So:

    job input  = {"op": "rig",
                  "input_url":  "<presigned GET for the staged upload>",
                  "output_url": "<presigned PUT for the result>",
                  "output_format": "glb",
                  ...generation options}

    job output = {"key": "...", "bytes": 1473368, "joints": 52}   # small

The worker downloads from `input_url`, runs the existing `export_rig_glb`, and
uploads the result to `output_url`. Nothing large crosses the queue, and the
worker needs no AWS credentials — the presigned URLs are the capability, scoped
to one object and expiring on their own.

## Server work

1. `handler.py` — `runpod.serverless.start`, dispatch on `op`, warm the runtime
   at import, return handler errors as `{"error", "code"}`.
2. `Dockerfile.queue` — **re-copy `src/`, `configs/`, `bpy_server.py`**. The base
   image bakes them; a4f0034 fixed exactly this trap for the API image and the
   queue image must not reintroduce it.
3. bpy_server must start from the handler, not FastAPI's lifespan.

## Client work

`src/server/motion/runpod-queue.ts` already exists and is provider-agnostic
(`runQueueJob`, `resolveQueueApiKey`). Needs a SkinTokens client that presigns
both URLs, submits, polls, then reads the result key from S3.

## Also worth fixing here

The mesh is loaded through bpy **twice** per request: `_load_original_vertices`
(runtime.py:302) and again by the dataloader. ~1.5s of the wall clock. The
second copy exists because the dataloader's transform normalises vertices into
[-1,1] in place, so the asset it yields is no longer in original space — but the
original bounds could be captured once rather than re-importing the file.
