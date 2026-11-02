"""Inspect every Task 17 exported atlas on its delivered OBJ, in six-thread Blender."""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpy
from mathutils import Vector
import prop_lib as props
from build_task17_parcels import render_1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--props', nargs='+', default=['parcel_cube_80','parcel_cube_75','parcel_cube_85','infeed_table','pallet_carton_160x80x80'])
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    qa = HERE / '_qa/task17'
    for prop_id in args.props:
        folder = props.ROOT / 'assets/props' / prop_id
        data = json.loads((folder/'prop.json').read_text())
        variants = data['variants'] or [{'name': 'default','textures': {}}]
        if prop_id.startswith('pallet_carton'):
            variants = [v for v in variants if v['name'] == 'imc']
        for variant in variants:
            props.reset_scene()
            bpy.ops.outliner.orphans_purge(do_recursive=True)
            objects = []
            for part in data['parts']:
                before = set(bpy.data.objects)
                bpy.ops.wm.obj_import(filepath=str(folder/part['visual']), forward_axis='Y', up_axis='Z')
                for obj in set(bpy.data.objects)-before:
                    mat = props.pbr_material('Delivered single albedo', '#FFFFFF', 1-part['mujoco_material']['shininess'], 0)
                    node = mat.node_tree.nodes.new('ShaderNodeTexImage')
                    node.image = bpy.data.images.load(str(folder/variant['textures'].get(part['name'],part['texture'])))
                    uv = mat.node_tree.nodes.new('ShaderNodeUVMap')
                    uv.uv_map = obj.data.uv_layers.active.name
                    mat.node_tree.links.new(uv.outputs['UV'],node.inputs['Vector'])
                    mat.node_tree.links.new(node.outputs['Color'],mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
                    obj.data.materials.clear()
                    obj.data.materials.append(mat)
                    objects.append(obj)
            if prop_id.startswith('pallet_carton'):
                render_1024(objects, qa/f'{prop_id}_{variant["name"]}_export.png')
            else:
                preview = folder/variant.get('preview','preview.png')
                render_1024(objects,preview)
                import shutil
                if variant['name'] in ('px','ny'):
                    render_1024(objects, qa/f'{prop_id}_{variant["name"]}_export.png', back=True)
                else:
                    shutil.copyfile(preview,qa/f'{prop_id}_{variant["name"]}_export.png')
            print('TASK17_EXPORT_PREVIEW',prop_id,variant['name'],flush=True)
    props.finalize_assets()


if __name__ == '__main__':
    main()
