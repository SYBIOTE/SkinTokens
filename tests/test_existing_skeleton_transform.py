"""CPU-only regression for request-specific preprocessing; no model download."""
import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from src.data.augment import AugmentAffine, AugmentNormalize, AugmentTrim
from src.data.transform import Transform
from src.rig_package.info.asset import Asset


tree = ast.parse((Path(__file__).parents[1] / "runtime.py").read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_transform_for_rig_mode")
# Load only this runtime function to avoid CUDA/model imports, but use the real
# production dataclasses and apply the resulting pipeline to a real Asset.
scope = dict(Transform=Transform, AugmentAffine=AugmentAffine,
             AugmentNormalize=AugmentNormalize, replace=replace)
exec(compile(ast.Module(body=[function], type_ignores=[]), "runtime.py", "exec"), scope)
prepare = scope["_transform_for_rig_mode"]


class ExistingSkeletonTests(unittest.TestCase):
    def test_weights_mode_excludes_trim_without_mutating_shared_transform(self):
        trim, normalize, sampler = AugmentTrim(), AugmentNormalize(), object()
        affine = AugmentAffine.parse(random_scale_p=1., random_shift_p=1.,
                                     random_scale=[0.5, 0.9])
        original = Transform(augments=[trim, affine, normalize], sampler=sampler, order=object())
        prepared = prepare(original, True)
        self.assertEqual(prepared.augments, [replace(affine, random_scale_p=0., random_shift_p=0.), normalize])
        self.assertEqual(affine.random_scale_p, 1.)
        self.assertEqual(affine.random_shift_p, 1.)
        self.assertIs(prepared.sampler, sampler)
        self.assertIsNone(prepared.order)
        self.assertIsNotNone(original.order)
        self.assertEqual(original.augments, [trim, affine, normalize])

    def test_normalizes_mesh_and_unweighted_joint_in_the_same_coordinate_frame(self):
        original = Transform.parse(augments=[
            {'__target__': 'trim'},
            {'__target__': 'affine', 'random_scale_p': 1., 'random_shift_p': 1.,
             'random_scale': [0.2, 0.8]},
            {'__target__': 'normalize'},
        ])
        matrices = np.repeat(np.eye(4)[None], 2, axis=0)
        matrices[:, :3, 3] = [[10., 0., 0.], [14., 1., 0.]]
        source = Asset(
            vertices=np.array([[10., 0., 0.], [12., 0., 0.], [10., 2., 0.]]),
            faces=np.array([[0, 1, 2]]),
            joint_names=['Root', 'CustomArm'], parents=np.array([-1, 0]),
            matrix_local=matrices, skin=np.array([[1., 0.]] * 3),
        )
        prepared = prepare(original, True)
        first, second = deepcopy(source), deepcopy(source)
        prepared.apply(first)
        prepared.apply(second)
        # Joint bounds participate too: center (12, 1, 0), uniform scale 1/2.
        np.testing.assert_allclose(first.vertices, [[-1., -.5, 0.], [0., -.5, 0.], [-1., .5, 0.]])
        np.testing.assert_allclose(first.joints, [[-1., -.5, 0.], [1., 0., 0.]])
        np.testing.assert_array_equal(first.vertices, second.vertices)
        np.testing.assert_array_equal(first.matrix_local, second.matrix_local)
        self.assertEqual(first.joint_names, source.joint_names)
        np.testing.assert_array_equal(first.parents, source.parents)
        np.testing.assert_array_equal(first.skin, source.skin)
        np.testing.assert_allclose(np.linalg.norm(first.vertex_normals, axis=1), 1.)

    def test_normal_only_or_empty_pipeline_has_no_affine_parameters(self):
        normalize = AugmentNormalize()
        self.assertEqual(prepare(Transform(augments=[normalize]), True).augments, [normalize])
        self.assertEqual(prepare(Transform(), True).augments, [])

    def test_new_rig_mode_keeps_original_pipeline(self):
        original = Transform(augments=[AugmentTrim(), AugmentAffine.parse(), AugmentNormalize()])
        self.assertIs(prepare(original, False), original)


if __name__ == "__main__":
    unittest.main()
