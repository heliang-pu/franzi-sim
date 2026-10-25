"""Review the actual Task 07 OBJ/UV/albedo exports with the shared preview rig."""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpy

import prop_lib as props


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=("lidded_tote", "station_plate", "drawer_cabinet"),
                        default=["lidded_tote", "station_plate", "drawer_cabinet"])
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    for prop_id in args.props:
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
                mat = props.pbr_material(part["name"] + " single baked atlas", "#FFFFFF",
                                         1 - part["mujoco_material"]["shininess"], 0)
                tree = mat.node_tree
                texture = tree.nodes.new("ShaderNodeTexImage")
                texture.image = bpy.data.images.load(str(folder / part["texture"]), check_existing=True)
                texture.image.colorspace_settings.name = "sRGB"
                uv = tree.nodes.new("ShaderNodeUVMap")
                uv.uv_map = obj.data.uv_layers.active.name
                tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
                bsdf = tree.nodes["Principled BSDF"]
                tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
                bsdf.inputs["Specular IOR Level"].default_value = part["mujoco_material"]["specular"]
                obj.data.materials.append(mat)
                objects.append(obj)
        bpy.context.view_layer.update()
        props.render_preview(objects, folder / "preview_export.png")
        print("TASK07_EXPORTED_ATLASES_RENDERED", prop_id, len(objects), flush=True)


if __name__ == "__main__":
    main()
