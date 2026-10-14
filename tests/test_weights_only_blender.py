"""Blender integration: run from SkinTokens with bpy, numpy, scipy and trimesh."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("bpy"), "requires Blender bpy")
class BlenderWeightsOnlyTests(unittest.TestCase):
    def test_added_unweighted_bone_and_rest_matrices_survive_glb_transfer(self):
        import bpy
        from src.rig_package.parser.bpy import BpyParser, clean_bpy, transfer_rigging

        clean_bpy()
        data = bpy.data.armatures.new("CatapultRig")
        rig = bpy.data.objects.new("CatapultRig", data)
        bpy.context.scene.collection.objects.link(rig)
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        root = data.edit_bones.new("Root")
        root.head = (0, 0, 0); root.tail = (0, 1, 0)
        arm = data.edit_bones.new("CustomArm")
        arm.head = (.123456, .5, .2); arm.tail = (.123456, 1.5, .4)
        arm.roll = .37; arm.parent = root
        unused = data.edit_bones.new("StillUnweighted")
        unused.head = (1, 0, 0); unused.tail = (1, 1, 0); unused.parent = root
        bpy.ops.object.mode_set(mode="OBJECT")
        rig.location = (3, -2, 1)

        mesh_data = bpy.data.meshes.new("Catapult")
        mesh_data.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        obj = bpy.data.objects.new("Catapult", mesh_data)
        bpy.context.scene.collection.objects.link(obj)
        obj.parent = rig
        obj.modifiers.new("Armature", "ARMATURE").object = rig
        obj.vertex_groups.new(name="Root").add([0, 1, 2], 1., "REPLACE")
        obj.vertex_groups.new(name="CustomArm")  # deliberately no initial weights

        with tempfile.TemporaryDirectory() as directory:
            source_path = str(Path(directory) / "input.glb")
            result_path = str(Path(directory) / "output.glb")
            bpy.ops.export_scene.gltf(filepath=source_path)
            original = BpyParser.load(source_path)
            self.assertIn("CustomArm", original.joint_names)
            arm_index = original.joint_names.index("CustomArm")
            self.assertTrue(np.all(original.skin[:, arm_index] == 0))
            prediction = original.copy()
            prediction.skin = np.zeros_like(original.skin)
            prediction.skin[:, arm_index] = 1.
            transfer_rigging(prediction, source_path, result_path, preserve_skeleton=True, group_per_vertex=4)
            returned = BpyParser.load(result_path)
            self.assertEqual(returned.joint_names, original.joint_names)
            np.testing.assert_array_equal(returned.parents, original.parents)
            np.testing.assert_allclose(returned.matrix_local, original.matrix_local, atol=1e-6)
            np.testing.assert_allclose(returned.matrix_world, original.matrix_world, atol=1e-6)
            np.testing.assert_allclose(returned.vertices, original.vertices, atol=1e-6)
            np.testing.assert_allclose(returned.skin[:, arm_index], 1., atol=1e-6)
        clean_bpy()
