"""
SkinTokens (TokenRig) in-process runtime: loads the unified rigging model once at
startup and reuses it across requests. Mesh I/O goes through the bpy_server sidecar.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from src.data.dataset import DatasetConfig, RigDatasetModule
from src.data.transform import Transform
from src.data.vertex_group import voxel_skin
from src.model.tokenrig import TokenRigResult
from src.rig_package.info.asset import Asset
from src.server.spec import (
    BPY_SERVER,
    bytes_to_object,
    get_model,
    object_to_bytes,
)
from src.tokenizer.parse import get_tokenizer

os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")

DEFAULT_MODEL_CKPT = (
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
)
MODEL_NAME = "articulation_xl_quantization_256_token_4"
SUPPORTED_EXT = {".obj", ".fbx", ".glb"}


def _preflight() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. SkinTokens requires a GPU. "
            "Run with: docker run --gpus all ..."
        )
    try:
        from flash_attn.modules.mha import MHA  # noqa: F401
    except Exception as exc:
        raise ImportError(
            f"Required native extensions failed to load: {exc}. "
            "Ensure LD_LIBRARY_PATH includes PyTorch lib dir."
        ) from exc


def _denormalize_joints(
    points: np.ndarray,
    original_vertices: np.ndarray,
    normalize_into: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Reverse AugmentAffine normalization into original model space."""
    bound_min = original_vertices.min(axis=0)
    bound_max = original_vertices.max(axis=0)
    denom = normalize_into[1] - normalize_into[0]
    scale = np.max((bound_max - bound_min) / denom)
    bias = (normalize_into[0] + normalize_into[1]) / 2.0
    return (points - bias) * scale + (bound_max + bound_min) / 2.0


def _zup_to_yup(points: np.ndarray) -> np.ndarray:
    """Blender Z-up -> glTF/Three.js Y-up: (x, y, z) -> (x, z, -y)."""
    result = np.empty_like(points)
    result[:, 0] = points[:, 0]
    result[:, 1] = points[:, 2]
    result[:, 2] = -points[:, 1]
    return result


def _load_original_vertices(input_path: str) -> np.ndarray:
    response = requests.get(
        f"{BPY_SERVER}/load",
        data=object_to_bytes(input_path),
        timeout=120,
    )
    response.raise_for_status()
    asset = bytes_to_object(response.content)
    if isinstance(asset, str):
        raise RuntimeError(f"bpy_server load failed: {asset}")
    if not isinstance(asset, Asset) or asset.vertices is None:
        raise RuntimeError("bpy_server returned invalid asset (no vertices)")
    return np.asarray(asset.vertices, dtype=np.float32)


def _post_bpy_payload(endpoint: str, payload: dict) -> Any:
    payload_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"skintokens_{endpoint}_",
            suffix=".pt",
            delete=False,
        ) as handle:
            handle.write(object_to_bytes(payload))
            payload_path = handle.name
        request_payload = {"payload_path": payload_path}
        response = requests.post(
            f"{BPY_SERVER}/{endpoint}",
            data=object_to_bytes(request_payload),
            timeout=300,
        )
        response.raise_for_status()
        result = bytes_to_object(response.content)
        if isinstance(result, dict) and result.get("error") is not None:
            raise RuntimeError(result.get("traceback") or result["error"])
        return result
    finally:
        if payload_path is not None:
            try:
                os.remove(payload_path)
            except OSError:
                pass


def _result_to_rig_json(
    result: TokenRigResult,
    normalized_sampled_vertices: np.ndarray,
    original_vertices: np.ndarray,
    skin: np.ndarray | None = None,
    group_per_vertex: int = 4,
) -> dict:
    detokenize = result.detokenize_output
    if detokenize is None:
        raise RuntimeError("Inference did not produce skeleton output")
    if skin is None:
        skin_pred = result.skin_pred
        if skin_pred is None:
            raise RuntimeError("Inference did not produce skin output")
        skin = skin_pred.detach().float().cpu().numpy()

    joints = np.asarray(detokenize.joints, dtype=np.float32)
    parents_raw = list(detokenize.parents)
    names = list(
        detokenize.joint_names or [f"bone_{i}" for i in range(joints.shape[0])]
    )
    skin = np.asarray(skin, dtype=np.float32)
    sampled_vertices = np.asarray(normalized_sampled_vertices, dtype=np.float32)

    joints = _zup_to_yup(_denormalize_joints(joints, original_vertices))
    sampled_vertices = _zup_to_yup(
        _denormalize_joints(sampled_vertices, original_vertices)
    )

    parents: list[int | None] = [
        None if int(p) < 0 else int(p) for p in parents_raw
    ]

    bones = []
    for i in range(joints.shape[0]):
        parent_idx = parents[i]
        if parent_idx is None:
            position = joints[i].tolist()
        else:
            position = (joints[i] - joints[parent_idx]).tolist()
        bones.append(
            {
                "name": names[i],
                "parent": names[parent_idx] if parent_idx is not None else None,
                "position": position,
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            }
        )

    num_bones = skin.shape[1]
    k = num_bones if group_per_vertex <= 0 else min(group_per_vertex, num_bones)
    topk_idx = np.argsort(-skin, axis=1)[:, :k]
    topk_w = np.take_along_axis(skin, topk_idx, axis=1)
    sums = topk_w.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    topk_w = topk_w / sums

    vertex_count = int(sampled_vertices.shape[0])
    return {
        "name": "SkinTokens Rig",
        "skeleton": {
            "name": "SkinTokens Skeleton",
            "bones": bones,
            "meta": {"model": MODEL_NAME},
        },
        "skin": {
            "vertexCount": vertex_count,
            "boneNames": names,
            "influencesPerVertex": int(k),
            "boneIndices": topk_idx.astype(np.int32).tolist(),
            "weights": topk_w.astype(np.float32).tolist(),
            "vertices": sampled_vertices.astype(np.float32).tolist(),
        },
        "meta": {
            "model": MODEL_NAME,
            "coordinateSystem": "y-up",
            "space": "original",
        },
    }


@dataclass
class RigOptions:
    """
    Generation and post-processing knobs for one /rig call.

    The sampling defaults below match this checkpoint's own `generate_kwargs`
    (recorded in grpo_1400.ckpt), i.e. the settings the model was validated
    under during RL fine-tuning. Deviating from them is a legitimate experiment
    -- do_sample=False in particular makes a run deterministic, which matters
    for a one-shot UX -- but it is a departure from how the policy was tuned,
    so change the defaults only on measured evidence.
    """

    top_k: int = 10
    top_p: float = 0.95
    temperature: float = 1.5
    repetition_penalty: float = 2.0
    num_beams: int = 10
    # False switches beam-sample -> deterministic beam search: same mesh always
    # yields the same skeleton, and the beams do max-likelihood search rather
    # than stochastic exploration. top_k/top_p/temperature are inert when False.
    do_sample: bool = True
    use_skeleton: bool = False
    # Geodesic voxel refinement of the predicted weights. On by default: it runs
    # over the real mesh at full resolution and is the only thing that stops
    # weights bleeding across gaps the 512-point encoder cannot resolve
    # (fingers, inner thighs), so a caller has to opt out rather than opt in.
    use_postprocess: bool = True
    # Voxel grid resolution for the use_postprocess geodesic pass. Higher
    # separates nearby surfaces (fingers, inner thighs) better, at a superlinear
    # cost in the shortest-path solve.
    voxel_resolution: int = 196
    group_per_vertex: int = 4


class SkinTokensRuntime:
    """Persistent TokenRig runtime."""

    def __init__(self, app_dir: str = "/app"):
        self._app_dir = app_dir
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._model_ckpt = os.environ.get("SKINTOKENS_MODEL_CKPT", DEFAULT_MODEL_CKPT)
        self._hf_path = os.environ.get("SKINTOKENS_HF_PATH") or None

        _preflight()
        t0 = time.perf_counter()
        ckpt_path = os.path.join(app_dir, self._model_ckpt)
        print(f">>> [runtime] Loading TokenRig from {ckpt_path}")
        self._model = get_model(ckpt_path, hf_path=self._hf_path)
        assert self._model.tokenizer_config is not None
        self._tokenizer = get_tokenizer(**self._model.tokenizer_config)
        self._transform = Transform.parse(
            **self._model.transform_config["predict_transform"]
        )
        print(
            f">>> [runtime] TokenRig ready ({time.perf_counter() - t0:.1f}s elapsed)"
        )

    def generate_rig_json(self, input_path: str, options: RigOptions | None = None) -> dict:
        options = options or RigOptions()
        with self._lock:
            return self._run_inference(input_path, options, output_format="json")

    def export_rig_glb(self, input_path: str, options: RigOptions | None = None) -> bytes:
        options = options or RigOptions()
        with self._lock:
            out_path = self._run_inference(
                input_path, options, output_format="glb"
            )
            try:
                with open(out_path, "rb") as handle:
                    return handle.read()
            finally:
                try:
                    os.remove(out_path)
                except OSError:
                    pass

    def _run_inference(
        self,
        input_path: str,
        options: RigOptions,
        output_format: str,
    ) -> dict | str:
        input_path = str(Path(input_path).resolve())
        ext = Path(input_path).suffix.lower()
        if ext not in SUPPORTED_EXT:
            raise ValueError(
                f"Unsupported format '{ext}'. Use: {', '.join(sorted(SUPPORTED_EXT))}"
            )

        original_vertices = _load_original_vertices(input_path)

        datapath = {
            "data_name": None,
            "loader": "bpy_server",
            "filepaths": {"articulation": [input_path]},
        }
        dataset_config = DatasetConfig.parse(
            shuffle=False,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            datapath=datapath,
        ).split_by_cls()

        module = RigDatasetModule(
            predict_dataset_config=dataset_config,
            predict_transform=self._transform,
            tokenizer=self._tokenizer,
            process_fn=self._model._process_fn,
        )
        dataloader = module.predict_dataloader()["articulation"]

        result: TokenRigResult | None = None
        normalized_sampled_vertices: np.ndarray | None = None

        for batch in dataloader:
            batch = {
                key: value.to("cuda") if isinstance(value, Tensor) else value
                for key, value in batch.items()
            }

            if not options.use_skeleton:
                batch.pop("skeleton_tokens", None)
                batch.pop("skeleton_mask", None)

            batch["generate_kwargs"] = {
                "max_length": 2048,
                "top_k": int(options.top_k),
                "top_p": float(options.top_p),
                "temperature": float(options.temperature),
                "repetition_penalty": float(options.repetition_penalty),
                "num_return_sequences": 1,
                "num_beams": int(options.num_beams),
                "do_sample": bool(options.do_sample),
            }

            skeleton_tokens = None
            if "skeleton_tokens" in batch and "skeleton_mask" in batch:
                mask = batch["skeleton_mask"][0] == 1
                skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()

            normalized_sampled_vertices = (
                batch["vertices"][0].detach().float().cpu().numpy()
            )

            preds: list[TokenRigResult] = self._model.predict_step(
                batch,
                skeleton_tokens=[skeleton_tokens]
                if skeleton_tokens is not None
                else None,
                make_asset=True,
            )["results"]
            result = preds[0]
            break

        if result is None or result.asset is None:
            raise RuntimeError("TokenRig prediction returned no asset")
        if normalized_sampled_vertices is None:
            raise RuntimeError("TokenRig prediction returned no sampled vertices")

        asset = result.asset
        skin_for_json: np.ndarray | None = None
        if options.use_postprocess:
            voxel = asset.voxel(resolution=int(options.voxel_resolution))
            asset.skin *= voxel_skin(
                grid=0,
                grid_coords=voxel.coords,
                joints=asset.joints,
                vertices=asset.vertices,
                faces=asset.faces,
                mode="square",
                voxel_size=voxel.voxel_size,
            )
            asset.normalize_skin()
            if asset.vertices is not None and asset.skin is not None:
                sampled_denorm = _denormalize_joints(
                    normalized_sampled_vertices, original_vertices
                )
                _, indices = cKDTree(asset.vertices).query(sampled_denorm, k=1)
                skin_for_json = np.asarray(asset.skin[indices], dtype=np.float32)

        if output_format == "json":
            return _result_to_rig_json(
                result,
                normalized_sampled_vertices,
                original_vertices,
                skin=skin_for_json,
                group_per_vertex=options.group_per_vertex,
            )

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as handle:
            out_path = handle.name

        # Always export through the transfer path.
        #
        # Asset carries geometry and skeleton only -- it has no fields for UVs,
        # materials or textures -- so the plain "export" path rebuilds bare
        # meshes from raw arrays and silently drops every appearance attribute:
        # a textured input comes back with 0 materials, 0 textures and no
        # TEXCOORD/COLOR channels, which also makes the texture unrecoverable
        # downstream since the UVs are gone too.
        #
        # "transfer" re-exports with use_origin=True, which keeps the original
        # file's Blender scene loaded so make_asset binds vertex groups onto the
        # real mesh objects. Materials, textures, UVs, vertex colours, original
        # scale and origin all survive; a rigged asset nobody can texture is not
        # a useful result, so this is not something a caller should be able to
        # turn off.
        # target_path is this request's upload, not asset.path. The asset comes
        # from the dataloader, and on a warm worker its .path can still name a
        # previous request's tmpdir -- already removed by the /rig finally block,
        # so bpy would fail loading a file that no longer exists.
        payload = {
            "source_asset": asset,
            "target_path": input_path,
            "export_path": out_path,
            "group_per_vertex": options.group_per_vertex,
        }
        res = _post_bpy_payload("transfer", payload)

        if res != "ok":
            raise RuntimeError(f"bpy_server export failed: {res}")
        return out_path
