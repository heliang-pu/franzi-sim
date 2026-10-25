"""Render Task 04's actual OBJ + single-albedo files for texture/UV QA.

Headless Blender, optional -- --props <ids> --variant A|B|C. These supplementary
previews use the same 512 px / CPU Cycles 64 / three-light recipe as preview.png.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
import prop_lib as props

IDS = ("belt_conveyor", "sort_bin", "parcel_box_s", "parcel_box_m", "parcel_pouch")


def render_prop(prop_id, variant_name):
    folder = props.ROOT / "assets" / "props" / prop_id
    data = json.loads((folder / "prop.json").read_text())
    variant = next((variant for variant in data["variants"] if variant["name"] == variant_name), {})
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    objects = []
    for part in data["parts"]:
        previous = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(folder / part["visual"]), forward_axis="Y", up_axis="Z")
        for obj in set(bpy.data.objects) - previous:
            obj.data.materials.clear()
            material = props.pbr_material(part["name"] + " actual baked atlas", "#FFFFFF", .6, 0)
            tree = material.node_tree
            texture = tree.nodes.new("ShaderNodeTexImage")
            filename = variant.get("textures", {}).get(part["name"], part["texture"])
            texture.image = bpy.data.images.load(str(folder / filename), check_existing=True)
            texture.image.colorspace_settings.name = "sRGB"
            uv = tree.nodes.new("ShaderNodeUVMap")
            uv.uv_map = obj.data.uv_layers.active.name
            tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
            bsdf = tree.nodes["Principled BSDF"]
            tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
            bsdf.inputs["Specular IOR Level"].default_value = part["mujoco_material"]["specular"]
            bsdf.inputs["Roughness"].default_value = 1 - part["mujoco_material"]["shininess"]
            if prop_id == "belt_conveyor" and part["name"] == "frame":
                bsdf.inputs["Metallic"].default_value = .6
            obj.data.materials.append(material)
            objects.append(obj)
    bpy.context.view_layer.update()
    props.render_preview(objects, folder / f"preview_export_{variant_name}.png")
    print("TASK04_EXPORTED_ATLASES_RENDERED", prop_id, variant_name, len(objects), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    parser.add_argument("--variant", choices=list("ABC"), default="A")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    for prop_id in args.props:
        render_prop(prop_id, args.variant)


if __name__ == "__main__":
    main()
