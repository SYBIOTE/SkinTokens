"""CPU-only regression for request-specific preprocessing; no model download."""
import ast
from dataclasses import dataclass, replace
from pathlib import Path
import unittest


@dataclass
class AugmentNormalize:
    random_scale_p: float = 0.5
    random_shift_p: float = 0.5


class AugmentTrim:
    pass


@dataclass
class Transform:
    augments: list
    sampler: object = None
    order: object = None


tree = ast.parse((Path(__file__).parents[1] / "runtime.py").read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_transform_for_rig_mode")
scope = dict(Transform=Transform, AugmentNormalize=AugmentNormalize, replace=replace)
exec(compile(ast.Module(body=[function], type_ignores=[]), "runtime.py", "exec"), scope)
prepare = scope["_transform_for_rig_mode"]


class ExistingSkeletonTests(unittest.TestCase):
    def test_weights_mode_excludes_trim_without_mutating_shared_transform(self):
        trim, normalize, sampler = AugmentTrim(), AugmentNormalize(), object()
        original = Transform([trim, normalize], sampler)
        prepared = prepare(original, True)
        self.assertEqual(prepared.augments, [AugmentNormalize(0., 0.)])
        self.assertEqual(normalize.random_scale_p, 0.5)
        self.assertIs(prepared.sampler, sampler)
        self.assertEqual(original.augments, [trim, normalize])

    def test_new_rig_mode_keeps_original_pipeline(self):
        original = Transform([AugmentTrim()])
        self.assertIs(prepare(original, False), original)


if __name__ == "__main__":
    unittest.main()
