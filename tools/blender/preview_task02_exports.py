"""Render actual OBJ + single-albedo exports, independent of source PBR.

Blender -b --factory-startup -P tools/blender/preview_task02_exports.py
Optional -- --layout B (default A). Writes preview_export_<layout>.png.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy

import prop_lib as props


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=list("ABC"), default="A")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    folder = props.ROOT / "assets" / "props" / "control_console"
    metadata = json.loads((folder / "prop.json").read_text())
    variant = next(v for v in metadata["variants"] if v["name"] == args.layout)
    props.reset_scene()
    objects = []
    for part in variant["parts"]:
        previous = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(folder / part["visual"]), forward_axis="Y", up_axis="Z")
        imported = list(set(bpy.data.objects) - previous)
        for obj in imported:
            obj.data.materials.clear()
            material = props.pbr_material(part["name"] + " baked atlas", "#FFFFFF", 0.5, 0)
            tree = material.node_tree
            bsdf = tree.nodes["Principled BSDF"]
            texture = tree.nodes.new("ShaderNodeTexImage")
            texture.image = bpy.data.images.load(str(folder / part["texture"]), check_existing=True)
            tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
            bsdf.inputs["Specular IOR Level"].default_value = part["mujoco_material"]["specular"]
            bsdf.inputs["Roughness"].default_value = 1 - part["mujoco_material"]["shininess"]
            if part["name"].endswith("bezel"):
                bsdf.inputs["Metallic"].default_value = 1
            if part.get("emission"):
                tree.links.new(texture.outputs["Color"], bsdf.inputs["Emission Color"])
                bsdf.inputs["Emission Strength"].default_value = part["emission"]
            obj.data.materials.append(material)
            objects.append(obj)
    bpy.context.view_layer.update()
    props.render_preview(objects, folder / ("preview_export_" + args.layout + ".png"))
    print("TASK02_EXPORTED_ATLASES_RENDERED", args.layout, len(objects), flush=True)


if __name__ == "__main__":
    main()
