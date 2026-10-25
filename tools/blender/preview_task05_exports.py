"""Render actual Task 05 OBJ + single-albedo exports for UV/colour review."""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpy
import prop_lib as props

IDS = ("carton_250x200x150", "carton_300x200x200", "carton_220x180x120", "roller_conveyor")


def render(prop_id):
    folder = props.ROOT / "assets" / "props" / prop_id
    data = json.loads((folder / "prop.json").read_text())
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
            texture.image = bpy.data.images.load(str(folder / part["texture"]), check_existing=True)
            texture.image.colorspace_settings.name = "sRGB"
            uv = tree.nodes.new("ShaderNodeUVMap")
            uv.uv_map = obj.data.uv_layers.active.name
            tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
            tree.links.new(texture.outputs["Color"], tree.nodes["Principled BSDF"].inputs["Base Color"])
            bsdf = tree.nodes["Principled BSDF"]
            bsdf.inputs["Specular IOR Level"].default_value = part["mujoco_material"]["specular"]
            bsdf.inputs["Roughness"].default_value = 1 - part["mujoco_material"]["shininess"]
            obj.data.materials.append(material)
            objects.append(obj)
    bpy.context.view_layer.update()
    props.render_preview(objects, folder / "preview_export.png")
    print("TASK05_EXPORTED_ATLASES_RENDERED", prop_id, len(objects), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    for prop_id in args.props:
        render(prop_id)


if __name__ == "__main__":
    main()
