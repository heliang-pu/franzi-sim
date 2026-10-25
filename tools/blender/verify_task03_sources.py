"""Verify packed task-3 sources; optionally render the actual OBJ/albedo exports.

Blender -b --factory-startup -P tools/blender/verify_task03_sources.py -- --render-exports
The atlas renders are Blender QA images, not a claim of MuJoCo runtime testing.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
import numpy as np
from mathutils import Matrix

import prop_lib as props


def render_exports(folder, data):
    props.reset_scene()
    objects = []
    for part in data["parts"]:
        previous = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(folder / part["visual"]), forward_axis="Y", up_axis="Z")
        for obj in set(bpy.data.objects) - previous:
            obj.data.materials.clear()
            material = props.pbr_material(part["name"] + " single baked atlas", "#FFFFFF", .5, 0)
            tree = material.node_tree
            shader = tree.nodes["Principled BSDF"]
            image = tree.nodes.new("ShaderNodeTexImage")
            image.image = bpy.data.images.load(str(folder / part["texture"]), check_existing=True)
            tree.links.new(image.outputs["Color"], shader.inputs["Base Color"])
            shader.inputs["Specular IOR Level"].default_value = part["mujoco_material"]["specular"]
            shader.inputs["Roughness"].default_value = 1 - part["mujoco_material"]["shininess"]
            obj.data.materials.append(material)
            objects.append(obj)
    bpy.context.view_layer.update()
    props.render_preview(objects, folder / "preview_export.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-exports", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    results = {}
    for prop_id in ("esd_tray", "tray_rail_set"):
        folder = props.ROOT / "assets" / "props" / prop_id
        data = json.loads((folder / "prop.json").read_text())
        bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
        scene = bpy.context.scene
        checks = {
            "cpu_cycles_64_samples": scene.render.engine == "CYCLES" and scene.cycles.device == "CPU"
            and scene.cycles.samples == 64,
            "metre_scene_units": scene.unit_settings.system == "METRIC" and scene.unit_settings.scale_length == 1,
            "source_prop_id": scene["prop_id"] == prop_id,
        }
        objects = []
        for part in data["parts"]:
            name = part["name"]
            obj = bpy.data.objects.get(part["source_visual_object"])
            checks[name + "_exists"] = obj is not None
            if obj is None:
                continue
            objects.append(obj)
            checks[name + "_identity_transform"] = max(
                abs(obj.matrix_world[i][j] - Matrix.Identity(4)[i][j])
                for i in range(4) for j in range(4)) < 1e-8
            checks[name + "_triangles_match_export_metadata"] = all(len(p.vertices) == 3 for p in obj.data.polygons)
            checks[name + "_principled_pbr"] = bool(obj.data.materials) and all(
                material and any(n.type == "BSDF_PRINCIPLED" for n in material.node_tree.nodes)
                for material in obj.data.materials)
            checks[name + "_packed_source_images"] = all(
                n.image is not None and n.image.packed_file is not None
                for material in obj.data.materials for n in material.node_tree.nodes if n.type == "TEX_IMAGE")
            checks[name + "_active_atlas_uv"] = obj.data.uv_layers.active.name == "UVMap"
            collision_objects = [bpy.data.objects.get(n) for n in part["source_collision_objects"]]
            checks[name + "_source_proxies_hidden"] = all(
                proxy is not None and proxy.hide_render and proxy.hide_get() for proxy in collision_objects)
        checks["source_bbox_matches_metadata"] = np.allclose(props.bbox_mm(objects), data["bbox_mm"], atol=.01)
        checks["source_triangles_match_metadata"] = props.triangle_count(objects) == data["triangles"]
        if prop_id == "esd_tray":
            mat = objects[0].data.materials[0]
            shader = mat.node_tree.nodes["Principled BSDF"]
            checks["conductive_pp_roughness_065"] = abs(shader.inputs["Roughness"].default_value - .65) < 1e-6
            checks["original_print_mask_packed"] = any(
                node.type == "TEX_IMAGE" and node.image and "esd_print_mask" in node.image.name
                and node.image.packed_file for node in mat.node_tree.nodes)
        else:
            rubber = bpy.data.materials.get("Black rubber bumper, 5 mm")
            checks["rubber_pbr_retained"] = rubber is not None and abs(
                rubber.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value - .9) < 1e-6
        # bpy values must become native bools before serialising the report.
        results[prop_id] = {key: bool(value) for key, value in checks.items()}
        if args.render_exports:
            render_exports(folder, data)
    path = Path(__file__).with_name("task03_source_checks.json")
    path.write_text(json.dumps(results, indent=2) + "\n")
    for prop_id, checks in results.items():
        print(prop_id, sum(checks.values()), "/", len(checks), "source checks passed", flush=True)
        for key, passed in checks.items():
            if not passed:
                print("FAIL", prop_id, key, flush=True)
    return int(any(not passed for checks in results.values() for passed in checks.values()))


if __name__ == "__main__":
    raise SystemExit(main())
