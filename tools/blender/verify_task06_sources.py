"""Inspect Task 06 source blends and render the retained steel gear variant.

Blender -b --factory-startup --python-exit-code 1 -P tools/blender/verify_task06_sources.py
Only writes Task 06 QA reports and assets/props/spur_gear/preview_steel.png.
"""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpy
import numpy as np

import prop_lib as props

IDS = ("hex_blank", "flanged_bushing", "l_bracket", "spur_gear", "foam_kit_tray")


def main():
    report = {}
    for prop_id in IDS:
        folder = props.ROOT / "assets" / "props" / prop_id
        data = json.loads((folder / "prop.json").read_text())
        bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
        scene, checks = bpy.context.scene, {}

        def check(name, condition, evidence):
            checks[name] = {"pass": bool(condition), "evidence": evidence}

        body = bpy.data.objects[data["parts"][0]["source_visual_object"]]
        check("metric_metres_Z_up", scene.unit_settings.system == "METRIC" and
              scene.unit_settings.scale_length == 1, [scene.unit_settings.system, scene.unit_settings.scale_length])
        check("CPU_Cycles_64", scene.render.engine == "CYCLES" and scene.cycles.samples == 64 and
              scene.cycles.device == "CPU", [scene.render.engine, scene.cycles.device, scene.cycles.samples])
        check("visual_transform_applied", np.allclose(np.asarray(body.matrix_world), np.eye(4), atol=1e-7),
              np.asarray(body.matrix_world).tolist())
        names = data["parts"][0]["source_collision_objects"]
        collisions = [bpy.data.objects[name] for name in names]
        check("collision_transforms_applied", all(np.allclose(np.asarray(obj.matrix_world), np.eye(4), atol=1e-7)
                                                   for obj in collisions), len(collisions))
        check("proxies_hidden_from_render", all(obj.hide_render for obj in collisions), len(collisions))
        check("all_visual_faces_triangulated", all(len(face.vertices) == 3 for face in body.data.polygons),
              len(body.data.polygons))
        check("one_export_UV_layer", len(body.data.uv_layers) == 1 and body.data.uv_layers[0].name == "UVMap",
              [layer.name for layer in body.data.uv_layers])
        materials = [slot.material for slot in body.material_slots]
        check("PBR_materials_restored_after_bake",
              all(mat and mat.node_tree.nodes.get("Principled BSDF") and
                  not any(node.name.startswith("Temporary bake target") for node in mat.node_tree.nodes)
                  for mat in materials), [mat.name for mat in materials])
        if prop_id == "flanged_bushing":
            check("bronze_metallic_1_and_turned_marks",
                  materials[0].node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value == 1 and
                  any(node.type == "GROUP" and node.node_tree.name.startswith("Machining: turned")
                      for node in materials[0].node_tree.nodes), materials[0].name)
        elif prop_id == "l_bracket":
            check("galvanised_spangle_procedural", any(node.type == "TEX_VORONOI" for node in materials[0].node_tree.nodes),
                  [node.name for node in materials[0].node_tree.nodes if node.type == "TEX_VORONOI"])
            extents = sorted([np.ptp(np.array([tuple(v.co) for v in obj.data.vertices]), axis=0).tolist()
                              for obj in collisions])
            check("two_leg_boxes_include_submillimetre_fillet_allowance",
                  np.allclose(extents, [[.00425, .06, .04], [.04, .00425, .04]], atol=1e-7), extents)
        elif prop_id == "foam_kit_tray":
            roughness = sorted(mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value
                               for mat in materials)
            check("PP_and_EVA_PBR_retained", len(roughness) == 2 and np.allclose(roughness, [.55, .95], atol=1e-6),
                  roughness)
        elif prop_id == "spur_gear":
            variant = next(v for v in data["variants"] if v["name"] == "steel")
            steel = bpy.data.materials.get(variant["source_material"])
            check("POM_and_steel_PBR_retained", steel is not None and steel.use_fake_user and
                  materials[0].node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value == 0 and
                  steel.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value == 1,
                  [materials[0].name, steel.name if steel else None])
            if steel:
                body.data.materials.clear()
                body.data.materials.append(steel)
                for face in body.data.polygons:
                    face.material_index = 0
                props.render_preview([body], folder / "preview_steel.png")
        report[prop_id] = checks
        print("TASK06_SOURCE", prop_id, sum(item["pass"] for item in checks.values()), "/", len(checks), flush=True)
    (HERE / "task06_source_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    failed = [(prop_id, name, item["evidence"]) for prop_id, checks in report.items()
              for name, item in checks.items() if not item["pass"]]
    for failure in failed:
        print("FAIL", *failure)
    if failed:
        raise RuntimeError(f"{len(failed)} Task 06 source checks failed")


if __name__ == "__main__":
    main()
