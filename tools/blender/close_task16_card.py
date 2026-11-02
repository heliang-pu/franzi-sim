"""Retain the full-front atlas while closing the card mesh for simulator import."""
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy
from mathutils import Vector
import prop_lib as props
from support_task16 import OUTPUT,mesh,finish,finalize_selected
folder=OUTPUT/'order_card_stand'
data=json.loads((folder/'prop.json').read_text())
bpy.ops.wm.open_mainfile(filepath=str(folder/'source.blend'))
old=bpy.data.objects['card']; material=old.data.materials[0]
normal=Vector(data['card']['front_normal']); u=Vector((0,-1,0)); v=Vector(data['card']['v_direction'])
center=sum((vertex.co for vertex in old.data.vertices),Vector())/len(old.data.vertices)*1000
if data['dimensions_mm'].get('card_thickness'): center+=normal*data['dimensions_mm']['card_thickness']/2
front=[center+u*a*148+v*b*105 for a,b in [(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)]]
back=[point-normal*.3 for point in front]
old.name='Old front sheet'
obj=mesh('card',[tuple(point) for point in front+back],[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],material)
uv=obj.data.uv_layers.new(name='UVMap')
for poly in obj.data.polygons:
    for li in poly.loop_indices:
        index=obj.data.loops[li].vertex_index
        uv.data[li].uv=[(0,0),(1,0),(1,1),(0,1)][index] if poly.index==0 else (0,0)
finish(obj); bpy.data.objects.remove(old,do_unlink=True)
props.export_obj(obj,folder/'card.obj')
visuals=[bpy.data.objects[p['source_visual_object']] for p in data['parts']]
data['triangles']=props.triangle_count(visuals)
data['dimensions_mm']['card_thickness']=.3
data['notes']=data['notes'].replace('Card is a front sheet backed by real acrylic;', 'Card is a closed 0.3 mm paper slab backed by real acrylic;')
data['notes']+=' Hidden card back/edge UVs sample the white lower-left pixel; the front retains its full-image mapping. Closed volume supports simulator mesh import.'
bpy.ops.file.pack_all(); bpy.ops.wm.save_as_mainfile(filepath=str(folder/'source.blend'),check_existing=False)
props.write_prop_json(data,folder/'prop.json')
finalize_selected()
