"""Render delivered OBJ/atlas assets, including the light-grey foam variant.

blender -b --factory-startup -t 6 -P tools/blender/preview_task16_exports.py -- --props ID ...
Sources remain PBR; preview.png is the validator's 512 thumbnail, inspection.png
is the native 1024 / 64-sample three-light render.
"""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy
import prop_lib as props
from support_task16 import OUTPUT, image_material, render
from build_task16_assembly import IDS


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--props',nargs='+',default=[*IDS,'foam_kit_tray'])
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    for name in args.props:
        props.reset_scene(); bpy.ops.outliner.orphans_purge(do_recursive=True)
        folder=OUTPUT/name; data=json.loads((folder/'prop.json').read_text())
        variant=next((v for v in data['variants'] if name=='foam_kit_tray' and v['name']=='light_grey'),{})
        objects=[]
        for p in data['parts']:
            before=set(bpy.data.objects)
            bpy.ops.wm.obj_import(filepath=str(folder/p['visual']),forward_axis='Y',up_axis='Z')
            imported=[o for o in bpy.data.objects if o not in before and o.type=='MESH']
            file=variant.get('textures',{}).get(p['name'],p['texture'])
            if file:
                mat=image_material(p['name']+' delivered albedo',folder/file,'UVMap')
                bsdf=mat.node_tree.nodes['Principled BSDF']
                bsdf.inputs['Roughness'].default_value=.95 if name=='foam_kit_tray' else .5
                if name=='cable_gland': bsdf.inputs['Roughness'].default_value=.48
                if p['mujoco_material']['reflectance']>=.05:
                    bsdf.inputs['Metallic'].default_value=.8
                    bsdf.inputs['Roughness'].default_value=.3
            else:
                mat=props.pbr_material(p['name']+' rgba','#FFFFFF',.3,0)
                bsdf=mat.node_tree.nodes['Principled BSDF']
                for link in list(bsdf.inputs['Base Color'].links): mat.node_tree.links.remove(link)
                bsdf.inputs['Base Color'].default_value=p['rgba']
                bsdf.inputs['Alpha'].default_value=p['rgba'][3]
                if p['rgba'][3]<1: bsdf.inputs['Transmission Weight'].default_value=.75
            for obj in imported:
                obj.data.materials.clear(); obj.data.materials.append(mat)
                for face in obj.data.polygons: face.material_index=0
                objects.append(obj)
        render(objects,folder)
        data['preview_variant']='light_grey' if variant else 'default'
        data['preview_source']='Delivered OBJ meshes and exported albedos; 1024 native inspection, 512 thumbnail.'
        props.write_prop_json(data,folder/'prop.json')
        print('TASK16_EXPORTED_PREVIEW',name,flush=True)


if __name__=='__main__': main()
