"""Pipeline proof only: a 100 mm cube, never a benchmark prop."""

import argparse
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
import prop_lib as props


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=props.ROOT / "assets/props/_test_cube")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    folder = args.output.resolve()
    allowed = (props.ROOT / "assets/props").resolve()
    if not folder.is_relative_to(allowed) or folder.name != "_test_cube":
        raise ValueError("Task 0 output must be assets/props/.../_test_cube")
    folder.mkdir(parents=True, exist_ok=True)
    props.reset_scene()
    mat = props.pbr_material("Test cube PBR", "#F5F3E9", 0.55, 0, wear=0.12, grime=0.12)
    cube = props.box("body", (100, 100, 100), location_mm=(0, 0, 50), bevel_mm=0.5, segments=3, material=mat)
    props.uv_unwrap(cube, "cube")
    label_path = props.make_label_png(
        folder / "checker_source.png", 1024, 1024, {"text": "TEST", "checker": True, "checker_cells": 8}
    )
    label_image = bpy.data.images.load(str(label_path))
    label_image.colorspace_settings.name = "sRGB"
    # Source coordinates are independent of the packed bake/export UV atlas.
    source_uv = cube.data.uv_layers.new(name="LabelUV")
    for face in cube.data.polygons:
        axis = max(range(3), key=lambda i: abs(face.normal[i]))
        for loop_index in face.loop_indices:
            v = cube.data.vertices[cube.data.loops[loop_index].vertex_index].co
            if axis == 0:
                u, w = face.normal.x / abs(face.normal.x) * v.y / 0.1 + 0.5, v.z / 0.1
            elif axis == 1:
                u, w = -face.normal.y / abs(face.normal.y) * v.x / 0.1 + 0.5, v.z / 0.1
            else:
                u, w = -v.y / 0.1 + 0.5, face.normal.z / abs(face.normal.z) * v.x / 0.1 + 0.5
            source_uv.data[loop_index].uv = (u, w)
    cube.data.uv_layers.active = cube.data.uv_layers["UVMap"]
    cube.data.uv_layers["UVMap"].active_render = True
    tree = mat.node_tree
    uv_node = tree.nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = "LabelUV"
    image_node = tree.nodes.new("ShaderNodeTexImage")
    image_node.image = label_image
    tree.links.new(uv_node.outputs["UV"], image_node.inputs["Vector"])
    # Keep the procedural wear/grime groups in the PBR graph while replacing
    # their base colour with the checker image.
    base_rgb = next(node for node in tree.nodes if node.type == "RGB")
    destinations = [link.to_socket for link in base_rgb.outputs[0].links]
    for link in list(base_rgb.outputs[0].links):
        tree.links.remove(link)
    for target in destinations:
        tree.links.new(image_node.outputs["Color"], target)
    props.bake_albedo([cube], 1024, folder / "body_albedo.png")
    props.export_obj(cube, folder / "body.obj")
    collision = props.box("body collision source", (100, 100, 100), location_mm=(0, 0, 50), bevel_mm=0)
    collision_files = props.export_collision("body", [collision], folder)
    collision.hide_render = True
    collision.hide_set(True)
    props.render_preview([cube], folder / "preview.png")
    actual_bbox = props.bbox_mm([cube])
    metadata = {
        "id": "_test_cube",
        "task": "pipeline_test",
        "units": "m",
        "origin": "bottom_center",
        "bbox_mm": [100.0, 100.0, 100.0],
        "budget_class": "small",
        "triangles": props.triangle_count([cube]),
        "source": "source.blend",
        "preview": "preview.png",
        "parts": [
            {
                "name": "body",
                "visual": "body.obj",
                "texture": "body_albedo.png",
                "texture_px": 1024,
                "collision": collision_files,
                "mass_kg": 0.1,
                "friction": 0.4,
                "mujoco_material": {"specular": 0.25, "shininess": 0.3, "reflectance": 0.0},
                "parent": None,
                "joint": None,
            }
        ],
        "variants": [],
        "notes": (
            "Pipeline proof only; not a benchmark prop. Cube spans X/Y +/-50 mm and Z 0..100 mm. "
            "0.5 mm three-segment edge bevel; one nominal box collider (corner deviation <1 mm). "
            "Checker and TEST source is generated with Pillow and packed in source.blend. "
            "Albedo is 64-sample CPU Cycles diffuse colour only, multiplied in linear light by "
            "mix(1, AO, 0.6), saved as 8-bit sRGB; source retains Principled PBR wear/grime. "
            f"Generated with Blender {bpy.app.version_string}; identity OBJ axes Y forward, Z up. "
            "The 100 mm cube is an export test and exceeds the recommended one-hand grasp width."
        ),
    }
    bpy.context.scene["prop_id"] = "_test_cube"
    bpy.context.scene["bake_recipe"] = "DIFFUSE COLOR * (0.4 + 0.6 * AO), linear -> 8-bit sRGB"
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    props.write_prop_json(metadata, folder / "prop.json")
    print(
        "PIPELINE_PROOF",
        {
            "blender": str(props.locate_blender()),
            "bbox_mm": actual_bbox,
            "triangles": metadata["triangles"],
            "collision": collision_files,
            "folder": str(folder),
        },
    )


if __name__ == "__main__":
    try:
        main()
        props.finalize_assets()
    except Exception:  # noqa: BLE001 - Blender otherwise exits zero after a script exception.
        traceback.print_exc()
        sys.exit(1)
