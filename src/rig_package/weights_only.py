"""Checked correspondence for weights-only inference. No Blender or GPU dependency."""
import numpy as np


def restore_conditioned_skeleton(predicted, conditioned, decoded):
    """Skin columns follow the supplied token order, never guessed nearest joints."""
    names = list(conditioned.joint_names) if conditioned.joint_names is not None else []
    count = len(names)
    if not count or len(set(names)) != count:
        raise ValueError("AI weights requires unique authored bone names")
    if predicted.skin is None or predicted.skin.shape != (conditioned.N, count):
        raise ValueError("AI weights returned a different vertex or bone count")
    if decoded is None or decoded.J != count:
        raise ValueError("Skeleton tokenization changed the authored bone count")
    if not np.array_equal(decoded.parents, conditioned.parents):
        raise ValueError("Skeleton tokenization changed bone parents; nearby joints may overlap on the token grid")
    if not np.isfinite(predicted.skin).all() or (predicted.skin < 0).any():
        raise ValueError("AI weights returned invalid weights")
    for field in ("joint_names", "parents", "matrix_local", "lengths", "matrix_world",
                  "mesh_names", "vertex_bias", "face_bias"):
        value = getattr(conditioned, field)
        setattr(predicted, field, value.copy() if value is not None else None)


def weights_in_target_order(source, target):
    """The preserve-mode pipeline changes neither dense vertices nor partitions."""
    source_names = list(source.joint_names) if source.joint_names is not None else []
    target_names = list(target.joint_names) if target.joint_names is not None else []
    if (len(set(source_names)) != len(source_names) or
            len(set(target_names)) != len(target_names) or set(source_names) != set(target_names)):
        raise ValueError("AI weights bone names do not match the uploaded skeleton")
    if source.N != target.N or not np.array_equal(source.faces, target.faces):
        raise ValueError("AI weights changed mesh topology")
    if (not np.array_equal(source.mesh_names, target.mesh_names) or
            not np.array_equal(source.vertex_bias, target.vertex_bias)):
        raise ValueError("AI weights changed mesh vertex partitions")
    source_lookup = {name: i for i, name in enumerate(source_names)}
    def parent_names(asset, names):
        return {name: None if asset.parents[i] == -1 else names[asset.parents[i]] for i, name in enumerate(names)}
    if parent_names(source, source_names) != parent_names(target, target_names):
        raise ValueError("AI weights changed the uploaded bone hierarchy")
    skin = source.skin
    if skin is None or skin.shape != (target.N, len(source_names)) or not np.isfinite(skin).all() or (skin < 0).any():
        raise ValueError("AI weights returned invalid skin data")
    return skin[:, [source_lookup[name] for name in target_names]]


def sparse_weights(skin, limit):
    count = skin.shape[1] if limit <= 0 else min(limit, skin.shape[1])
    indices = np.argsort(-skin, axis=1)[:, :count]
    weights = np.take_along_axis(skin, indices, axis=1)
    sums = weights.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        raise ValueError("AI weights returned vertices with no influence")
    return indices, weights / sums
