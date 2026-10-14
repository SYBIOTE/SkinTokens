"""Run with: python -m unittest discover -s tests -p 'test_extract_mesh_empty_parts.py'.

Exercise the production extractor with Blender-shaped fixtures, without importing
the GPU/Blender runtime. This does not test Blender's GLB importer.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
import trimesh


def mesh(name, coords, triangle=False):
    vertices = [NS(co=co, index=i, groups=[]) for i, co in enumerate(coords)]
    polygons = [NS(edge_keys=[(0, 1), (1, 2), (2, 0)], normal=np.array([0, 0, 1]))] if triangle else []
    return NS(name=name, type="MESH", parent=None, matrix_world=np.eye(4),
              vertex_groups=[], data=NS(vertices=vertices, polygons=polygons))


def extract(objects):
    path = Path(__file__).parents[1] / "src/rig_package/parser/bpy.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "extract_mesh")
    scope = {"np": np, "trimesh": trimesh, "bpy": NS(
        data=NS(objects=objects),
        context=NS(view_layer=NS(objects={o.name: o for o in objects})))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope["extract_mesh"]([NS(name="Root"), NS(name="Custom")])


class ExtractMeshTests(unittest.TestCase):
    def test_empty_and_point_parts_keep_vertex_partitions(self):
        result = extract([
            mesh("empty", []),
            mesh("points", [[2, 2, 2]]),
            mesh("surface", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], True),
        ])
        np.testing.assert_array_equal(result["mesh_names"], ["empty", "points", "surface"])
        np.testing.assert_array_equal(result["vertex_bias"], [0, 1, 4])
        np.testing.assert_array_equal(result["face_bias"], [0, 0, 1])
        np.testing.assert_array_equal(result["faces"], [[1, 2, 3]])
        self.assertEqual(result["skin"].shape, (4, 2))
        self.assertEqual(result["vertices"].shape, (4, 3))

    def test_point_only_input_reports_missing_surface(self):
        with self.assertRaisesRegex(ValueError, "no triangle surfaces"):
            extract([mesh("points", [[0, 0, 0]])])

    def test_empty_input_reports_missing_vertices(self):
        with self.assertRaisesRegex(ValueError, "no mesh vertices"):
            extract([mesh("empty", [])])


if __name__ == "__main__":
    unittest.main()
