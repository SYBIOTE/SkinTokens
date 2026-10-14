import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("weights_only", Path(__file__).parents[1] / "src/rig_package/weights_only.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def asset():
    return NS(N=2, joint_names=["Root", "Arm"], parents=np.array([-1, 0]),
              matrix_local=np.repeat(np.eye(4)[None], 2, axis=0), lengths=np.ones(2),
              matrix_world=np.eye(4), mesh_names=["catapult"], vertex_bias=np.array([2]),
              face_bias=np.array([0]), faces=np.empty((0, 3), dtype=int),
              skin=np.array([[1., 0.], [0., 1.]]))

class WeightOnlyTests(unittest.TestCase):
    def test_preserves_authored_matrices_names_and_zero_influence_bone(self):
        source, predicted = asset(), asset()
        source.skin[:, 1] = 0
        source.matrix_local[1, 0, 3] = 1.123456789
        predicted.joint_names = ["bone_0", "bone_1"]
        module.restore_conditioned_skeleton(predicted, source, NS(J=2, parents=[-1, 0]))
        self.assertEqual(predicted.joint_names, ["Root", "Arm"])
        np.testing.assert_array_equal(predicted.matrix_local, source.matrix_local)
        np.testing.assert_array_equal(predicted.skin, [[1, 0], [0, 1]])

    def test_remaps_columns_by_verified_bone_correspondence(self):
        source, target = asset(), asset()
        source.joint_names = ["Arm", "Root"]
        source.parents = np.array([1, -1])
        np.testing.assert_array_equal(module.weights_in_target_order(source, target), [[0, 1], [1, 0]])

    def test_fails_on_missing_bone_changed_topology_or_ambiguous_token_parent(self):
        source, target = asset(), asset()
        source.joint_names[1] = "Wrong"
        with self.assertRaisesRegex(ValueError, "bone names"):
            module.weights_in_target_order(source, target)
        source = asset(); source.vertex_bias = np.array([1, 2])
        with self.assertRaisesRegex(ValueError, "partitions"):
            module.weights_in_target_order(source, target)
        with self.assertRaisesRegex(ValueError, "parents"):
            module.restore_conditioned_skeleton(asset(), asset(), NS(J=2, parents=[-1, -1]))

    def test_sparse_weights_validate_and_normalize(self):
        indices, weights = module.sparse_weights(np.array([[.2, .8]]), 1)
        np.testing.assert_array_equal(indices, [[1]])
        np.testing.assert_array_equal(weights, [[1.]])
        with self.assertRaisesRegex(ValueError, "no influence"):
            module.sparse_weights(np.zeros((1, 2)), 4)
