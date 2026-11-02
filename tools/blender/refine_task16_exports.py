"""Targeted finish for front-label offsets and the full-resolution card atlas.

blender -b --factory-startup -t 6 -P tools/blender/refine_task16_exports.py --
The main builders already include these corrections for clean rebuilds.
"""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy
import prop_lib as props
from support_task16 import OUTPUT, bake, render, finalize_selected
from build_task16_assembly import IDS


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--props',nargs='+',default=['order_tray','rotor_shaft_fixture','order_card_stand'])
    parser.add_argument('--finalize-all',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    for name in args.props:
        folder=OUTPUT/name; data=json.loads((folder/'prop.json').read_text())
        if name=='rotor_shaft_fixture':
            data['collision_exclusions']=[['start_left','body'],['start_right','body']]
            data['notes']+=' Simulator must honor cap/support collision_exclusions for the full 6 mm button stroke.'
        bpy.ops.wm.open_mainfile(filepath=str(folder/'source.blend'))
        visuals=[bpy.data.objects[p['source_visual_object']] for p in data['parts']]
        p=next(p for p in data['parts'] if p['name']==('card' if name=='order_card_stand' else 'body'))
        obj=bpy.data.objects[p['source_visual_object']]
        small_heights={'relay':40,'terminal_block':35,'fuse_holder':40}
        if name in small_heights:
            faces=[f for f in obj.data.polygons if obj.data.materials[f.material_index].name.startswith(('Printed ','TB-4P ink'))]
            assert faces
            for i in {v for f in faces for v in f.vertices}: obj.data.vertices[i].co.z=props.mm(small_heights[name]+.02)
            obj.data.update(); props.export_obj(obj,folder/p['visual'])
        elif name!='order_card_stand':
            ink_names={'ORDER print','Fixture nameplate','start_left ink','start_right ink'}
            faces=[f for f in obj.data.polygons if any(obj.data.materials[f.material_index].name.startswith(n) for n in ink_names)]
            assert faces,(name,'no front artwork faces')
            for i in {v for f in faces for v in f.vertices}:
                obj.data.vertices[i].co.x=props.mm(-70.02 if name=='order_tray' else -130.02)
            obj.data.update()
            props.export_obj(obj,folder/p['visual'])
        if name not in small_heights:
            p['texture_px']=2048
            bake(obj,2048,folder,p['texture'])
        data['bbox_mm']=[round(v,6) for v in props.bbox_mm(visuals)]
        data['triangles']=props.triangle_count(visuals)
        if name not in small_heights: render(visuals,folder)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(folder/'source.blend'),check_existing=False)
        props.write_prop_json(data,folder/'prop.json')
        print('TASK16_REFINED',name,flush=True)
    if args.finalize_all: props._DELIVERED_PROPS.update([*IDS,'foam_kit_tray'])
    finalize_selected()


if __name__=='__main__': main()
