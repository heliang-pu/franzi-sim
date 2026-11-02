"""Build section 6.3 picking/assembly props, in millimetres.

Usage: blender -b --factory-startup -t 6 -P tools/blender/build_task16_assembly.py -- --props ID ...
Rebuild foam_kit_tray with build_task06_shaped_kitting.py -- --props foam_kit_tray.
"""
import argparse
import json
import math
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))


def artwork_worker() -> None:
    from PIL import Image, ImageDraw, ImageFont
    kind,path,title=sys.argv[-3:]
    w,h=(1480,1050) if kind=='card' else (1400,600) if kind.startswith('bin_') else (1024,1024) if kind=='bearing' else (1600,400)
    if kind=='label':
        ratio={'IMC RLY|24VDC':24/21,'FH-10A':46/18,'TB-4P':36/5,
               'START':48/11,'IMC BEARING PRE-ASSEMBLY':190/14,'ORDER':64/11}.get(title,4)
        h=round(w/ratio)
    im=Image.new('RGB',(w,h),'#F5F5EF'); d=ImageDraw.Draw(im)
    font=lambda size: ImageFont.truetype(str(HERE/'fonts'/'DejaVuSans-Bold.ttf'),size)
    ink='#182536'
    if kind=='card':
        d.rectangle((0,0,w,190),fill='#174C87')
        d.text((70,40),'WORK ORDER',font=font(105),fill='white')
        for y in (290,530,770):
            d.rounded_rectangle((65,y,1415,y+180),radius=15,outline='#7E8D9A',width=5)
            d.line((290,y+20,290,y+160),fill='#A8B1B8',width=3)
            d.line((1190,y+20,1190,y+160),fill='#A8B1B8',width=3)
    elif kind.startswith('bin_'):
        part=kind[4:]; x,y=80,85
        if part=='relay':
            d.rounded_rectangle((x,y,x+225,y+285),20,outline=ink,width=12)
            for xx in range(x+25,x+226,55): d.line((xx,y+285,xx,y+350),fill=ink,width=12)
            d.rectangle((x+35,y+35,x+190,y+100),outline=ink,width=9)
        elif part=='terminal':
            d.rectangle((x,y+30,x+260,y+300),outline=ink,width=12)
            for xx in (x+65,x+195):
                for yy in (y+95,y+230):
                    d.ellipse((xx-30,yy-30,xx+30,yy+30),outline=ink,width=9)
                    d.line((xx-20,yy,xx+20,yy),fill=ink,width=8)
        elif part=='gland':
            d.regular_polygon((x+130,y+170,130),6,rotation=30,outline=ink,width=12)
            d.ellipse((x+60,y+100,x+200,y+240),outline=ink,width=12)
            d.rectangle((x+55,y+290,x+205,y+350),outline=ink,width=10)
        else:
            d.rounded_rectangle((x+40,y,x+220,y+340),15,outline=ink,width=12)
            d.rectangle((x+70,y+50,x+190,y+210),outline=ink,width=9)
            d.line((x+85,y+280,x+175,y+280),fill=ink,width=12)
        d.text((425,85),title,font=font(125),fill=ink)
        d.text((430,250),'IMC Controls',font=font(65),fill=ink)
        xx=435
        for i in range(100):
            bar=3+(i*7%4)*2
            if i%3!=0: d.rectangle((xx,390,xx+bar,510),fill=ink)
            xx+=9
    elif kind=='bearing':
        # Image maps to a 100 mm square: the printed circle is exactly diameter 48.
        r=48/100*1024/2
        d.ellipse((512-r,512-r,512+r,512+r),outline='#465563',width=7)
        d.text((512,850),'6204-2RS',font=font(84),anchor='mm',fill=ink)
        # Background is POM grey, not a white paper label.
        import numpy as np
        a=np.asarray(im).copy(); a[(a[:,:,0]>230)&(a[:,:,1]>230)&(a[:,:,2]>225)]=[154,160,166]
        im=Image.fromarray(a)
    else:
        words=title.split('|')
        for i,line in enumerate(words):
            size=round(h/len(words)*.62)
            while d.textbbox((0,0),line,font=font(size))[2]>w*.9: size-=1
            d.text((w//2,int((i+.5)*h/len(words))),line,font=font(size),anchor='mm',fill=ink)
    im.save(path)


if __name__=='__main__' and '--artwork-worker' in sys.argv:
    artwork_worker(); raise SystemExit
if __name__=='__main__' and '--thumbnail' in sys.argv:
    from PIL import Image
    Image.open(sys.argv[-2]).convert('RGB').resize((512,512),Image.Resampling.LANCZOS).save(sys.argv[-1])
    raise SystemExit

import bpy
from mathutils import Matrix, Vector
import prop_lib as props
from support_task16 import (OUTPUT, PLASTIC, METAL, material, box, cylinder, mesh, loft,
    finish, boolean, art, image_material, label, start, part, bake, complete, finalize_selected)

IDS=('parts_bin','relay','terminal_block','cable_gland','fuse_holder','order_tray',
     'order_card_stand','rotor_shaft_fixture','bearing_tray')


def tray_geometry(size, wall, floor, mat):
    x,y,z=size
    body=box('Moulded tray',size,(0,0,z/2),mat,.6)
    boolean(body,box('Interior',(x-2*wall,y-2*wall,z+2),(0,0,floor+(z+2)/2),edge=.5))
    cols=[box('floor',(x,y,floor),(0,0,floor/2),edge=0)]
    for s in (-1,1):
        cols.append(box('X wall',(wall,y,z-floor),(s*(x-wall)/2,0,(z+floor)/2),edge=0))
        cols.append(box('Y wall',(x-2*wall,wall,z-floor),(0,s*(y-wall)/2,(z+floor)/2),edge=0))
    return body,cols


def build_parts_bin() -> None:
    folder=start('parts_bin'); blue=material('Blue PP #1F5FA8','#1F5FA8')
    body,cols=tray_geometry((180,150,38),2.5,3,blue)
    # Recessed card holder within the prescribed external envelope.
    boolean(body,box('70 x 30 recessed label slot',(1.2,70,30),(-90,0,19),edge=.3))
    arts=[art(folder,'bin_'+name,title) for name,title in
          [('relay','RLY-24V'),('terminal','TB-4P'),('gland','CG-M20'),('fuse','FH-10A')]]
    mat=image_material('Replaceable stock card',arts[0])
    face=label('White stock label',(-89.41,0,19),69,29,mat)
    p=part('body',[body,face],cols,.2,friction=.4)
    def variants():
        node=next(n for n in mat.node_tree.nodes if n.type=='TEX_IMAGE')
        result=[]
        for i,name in enumerate(('relay','terminal','gland','fuse')):
            node.image=bpy.data.images.load(str(arts[i]),check_existing=True)
            file='body_albedo.png' if i==0 else f'body_albedo_v{i}.png'
            if i: bake(p['obj'],2048,folder,file)
            result.append(dict(name=name,textures={'body':file},label_code=['RLY-24V','TB-4P','CG-M20','FH-10A'][i]))
        node.image=bpy.data.images.load(str(arts[0]),check_existing=True)
        return result
    complete(folder,[p],dict(outer_xyz=[180,150,38],wall=2.5,floor=3,label_slot_yz=[70,30]),
             dict(static=True,recommended_placement_m={'origins':[[.38,y,.9] for y in (.19,.36,-.19,-.36)]},
                  contents={'count':3,'axis':[1,0,0],'spacing_m':.055,'centers_m':[[-.055,0,.003],[0,0,.003],[.055,0,.003]]}),
             variant_callback=variants)


def top_label(folder,visuals,title,size,z,kind='label',center=(0,0)):
    path=art(folder,kind,title)
    mat=image_material('Printed '+title,path)
    visuals.append(label('Identification',(center[0],center[1],z),size[1],size[0],mat,False))


def build_relay() -> None:
    folder=start('relay'); blue=material('Opaque relay blue','#1E5AA8'); steel=material('Tin pins','#BFC3C7',True)
    # Pins occupy the bottom 3 mm of the specified total 40 mm envelope.
    visuals=[box('Relay housing',(28,28,37),(0,0,21.5),blue,.4)]
    for x in (-9,9):
        for y in (-9,-3,3,9): visuals.append(box('Tin contact pin',(1,1,3.2),(x,y,1.6),steel,.15))
    top_label(folder,visuals,'IMC RLY|24VDC',(21,24),40.02)
    p=part('body',visuals,[box('Specified box proxy',(28,28,40),(0,0,20),edge=0)],.05,size=1024)
    complete(folder,[p],dict(outer_xyz=[28,28,40],pins=8,pin_length=3),
             dict(budget_class='small',static=False),notes='Pins are visual only, as specified; total height includes pins.')


def build_terminal() -> None:
    folder=start('terminal_block'); grey=material('Terminal grey','#9AA0A6'); orange=material('Orange terminal wells','#EF6C00')
    steel=material('Clamp screws','#BFC3C7',True); dark=material('Recess shadow','#34383C')
    body=box('Terminal housing',(45,22,35),(0,0,17.5),grey,.4); visuals=[body]
    for x in (-16.5,-5.5,5.5,16.5):
        boolean(body,cylinder('Wiring well',3.8,5,(x,0,34),edge=0,count=32))
        ring=props.tube('Orange screw collar',3.8,2.6,2,(x,0,33.3),bevel_mm=.3,vertices=32,material=orange)
        screw=cylinder('Terminal screw',2.45,1,(x,0,32.2),steel,.2,32)
        boolean(screw,box('Screw slot',(3.6,.65,.6),(x,0,32.75),edge=.1))
        visuals.extend([ring,screw])
    marking=label('Top terminal ID',(0,0,35.02),36,5,
                  image_material('TB-4P ink',art(folder,'label','TB-4P')),False)
    for vertex in marking.data.vertices:
        x,y,z=vertex.co; vertex.co=(-y,x+.008,z)
    visuals.append(marking)
    p=part('body',visuals,[box('Box proxy',(45,22,35),(0,0,17.5),edge=0)],.05,size=1024)
    complete(folder,[p],dict(outer_xyz=[45,22,35],orange_wells=4,well_diameter=7.6),dict(budget_class='small',static=False),
             notes='Four shallow screw wells and screw slots are visual; prescribed single box collision bridges them.')


def build_gland() -> None:
    folder=start('cable_gland'); nylon=material('Black nylon','#202124',rough=.48)
    lower=cylinder('M20 threaded shank',9.55,10,(0,0,5),nylon,.3)
    visuals=[lower]
    for z in (1,2.5,4,5.5,7,8.5):
        visuals.append(loft('M20 thread crest',[(9.5,z-.45),(10,z),(9.5,z+.45)],32,nylon))
    visuals.append(props.hexagonal_prism('30 AF wrench flats',30,10,(0,0,15),bevel_mm=.4,material=nylon))
    visuals.append(loft('Domed compression cap',[(13.5,20),(14,20.5),(14,28),(13.8,30),(12.6,32.5),(10.5,34.5),(9,35)],64,nylon))
    # Shallow cable entry socket, kept within a convex 3-piece contact envelope.
    cap=visuals[-1]; boolean(cap,cylinder('Cable entry',4,1.3,(0,0,35),edge=.3,count=32))
    cols=[cylinder('Shank proxy',10,10,(0,0,5),edge=0,count=32),
          props.hexagonal_prism('Hex proxy',30,10,(0,0,15),bevel_mm=0),
          loft('Domed cap proxy',[(14,20),(14,28),(12.6,32.5),(9,35)],16)]
    p=part('body',visuals,cols,.04,size=1024)
    complete(folder,[p],dict(thread_diameter=20,thread_length=10,hex_across_flats=30,hex_height=10,cap_diameter=28,cap_height=15,total_height=35),
             dict(budget_class='small',static=False),notes='Thread crests are ring approximations (not a screw joint). Entry recess is 0.65 mm deep and omitted by its convex cap proxy.')


def build_fuse() -> None:
    folder=start('fuse_holder'); grey=material('Fuse grey','#8C9399'); dark=material('Cartridge seam','#3C4248')
    body=box('Fuse base',(25,60,40),(0,0,20),grey,.4)
    boolean(body,box('Front cartridge pocket',(2,42,27),(-12.3,0,21),edge=.5))
    visuals=[body,box('Fixed fuse cartridge',(.8,40,25),(-11.75,0,21),dark,.3),
             box('Cartridge pull tab',(.8,16,3),(-12.05,0,29),grey,.3)]
    top_label(folder,visuals,'FH-10A',(18,46),40.02)
    p=part('body',visuals,[box('Box proxy',(25,60,40),(0,0,20),edge=0)],.05,size=1024)
    complete(folder,[p],dict(outer_xyz=[25,60,40]),dict(budget_class='small',static=False),notes='Front cartridge is fixed geometry without a joint; single box collision per specification.')


def build_order_tray() -> None:
    folder=start('order_tray'); grey=material('Grey ESD PP','#838C94')
    body,cols=tray_geometry((140,200,30),2.5,3,grey)
    rib=box('Left-right divider',(135,2.5,12),(0,0,9),grey,.5)
    cols.append(box('Divider proxy',(135,2.5,12),(0,0,9),edge=0))
    visuals=[body,rib,label('ORDER front',(-70.02,0,16),64,11,image_material('ORDER print',art(folder,'label','ORDER')))]
    p=part('body',visuals,cols,.2,friction=.4)
    complete(folder,[p],dict(outer_xyz=[140,200,30],divider_height=12,wall=2.5,floor=3),
             dict(static=True,recommended_placement_m={'origin':[.38,0,.9]},
                  compartments=[{'hand':'left','y_sign':1},{'hand':'right','y_sign':-1}]))


def build_card() -> None:
    folder=start('order_card_stand'); acrylic=material('Clear acrylic','#D9E5F2',rough=.18)
    bsdf=acrylic.node_tree.nodes['Principled BSDF']; bsdf.inputs['Transmission Weight'].default_value=.75
    bsdf.inputs['Alpha'].default_value=.25
    angle=math.radians(15); v=Vector((math.sin(angle),0,math.cos(angle))); normal=Vector((-math.cos(angle),0,math.sin(angle)))
    bottom=Vector((-13,0,8)); center=bottom+v*52.5
    back=props.box('Acrylic inclined support',(1.8,152,109),tuple(center-normal*1.2),bevel_mm=.4,material=acrylic,rotation_deg=(0,15,0))
    base=box('Acrylic L base',(60,160,8),(0,0,4),acrylic,.5)
    body=part('body',[base,back],[box('Base proxy',(60,160,8),(0,0,4),edge=0)],.295,rgba=[.85,.90,.95,.25])
    path=art(folder,'card',filename='card_front.png')
    mat=image_material('Full A6 work order',path,'UVMap')
    u=Vector((0,-1,0)); c=center+normal*.01
    corners=[(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)]
    front=[tuple(c+u*a*148+v*b*105) for a,b in corners]
    back=[tuple(Vector(point)-normal*.3) for point in front]
    card=mesh('card',front+back,[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],mat)
    uv=card.data.uv_layers.new(name='UVMap')
    for poly in card.data.polygons:
        for li in poly.loop_indices:
            if poly.index==0:
                a,b=corners[card.data.loops[li].vertex_index]; uv.data[li].uv=(a+.5,b+.5)
            else: uv.data[li].uv=(0,0)
    col=props.box('Card and support proxy',(2,152,109),tuple(center-normal*.8),bevel_mm=0,rotation_deg=(0,15,0))
    p=part('card',[card],[col],.005,parent='body',size=2048,full_uv=True)
    bake(card,2048,folder,'card_albedo.png')
    complete(folder,[body,p],dict(base_xyz=[60,160,8],card_width=148,card_height=105,card_thickness=.3,rear_tilt_deg=15),
             dict(static=True,recommended_placement_m={'origin':[.56,0,.9]},
                  card={'part':'card','front_image':'card_front.png','image_size_px':[1480,1050],
                        'front_uv_bounds':[[0,0],[1,1]],'back_edge_uv_sample':[0,0],'thickness_m':.0003,'front_normal':list(normal),
                        'u_direction':[0,-1,0],'v_direction':list(v),'replace_texture':True}),
             notes='The card front uses the complete [0,1] UV rectangle, with no packing or margins. '
             'Native replacement artwork is 1480 x 1050. The unchanged validator only accepts square 1024/2048 part textures; '
             'the default card_albedo is a square bake of that artwork, geometrically restored to A6 by the full-face UVs. '
             'Acrylic is an untextured rgba part. Card is a closed 0.3 mm slab for simulator mesh loading; hidden back/edge UVs sample the white lower-left corner while the front uses the whole image. Support and card share one convex collision.')


def build_bearing_tray() -> None:
    folder=start('bearing_tray'); grey=material('Grey POM','#9AA0A6',rough=.45)
    body=box('Flat bearing tray',(100,100,10),(0,0,5),grey,.5)
    visuals=[body]
    top_label(folder,visuals,'6204-2RS',(100,100),10.005,'bearing')
    # Label sheet is trimmed to the inset flat top, with a shader projection instead.
    face=visuals.pop(); mat=face.data.materials[0]; bpy.data.objects.remove(face,do_unlink=True)
    body.data.materials.append(mat)
    uv=body.data.uv_layers.new(name='LabelUV')
    for poly in body.data.polygons:
        if poly.normal.z>.999 and poly.center.z>=0:
            poly.material_index=1
        for li in poly.loop_indices:
            co=body.data.vertices[body.data.loops[li].vertex_index].co*1000
            uv.data[li].uv=(.5-co.y/100,.5+co.x/100)
    p=part('body',[body],[box('Flat tray proxy',(100,100,10),(0,0,5),edge=0)],.2)
    complete(folder,[p],dict(outer_xyz=[100,100,10],marking_circle_diameter=48),
             dict(task='bearing_assembly',static=True,recommended_placement_m={'origins':[[.38,.30,.9],[.38,-.30,.9]]}),
             notes='Flat upper support surface; no recess or raised rim to obstruct fingertips.')


def build_fixture() -> None:
    folder=start('rotor_shaft_fixture')
    grey=material('Powder coated steel #6B7075','#6B7075'); steel=material('Ground shaft steel','#BFC3C7',True,.3)
    dark=material('Laminated silicon steel','#34383E',True,.45); green=material('Green start cap','#2E9E44',rough=.5)
    visuals=[box('Steel base',(260,440,30),(0,0,15),grey,1)]
    cols=[box('Base proxy',(260,440,30),(0,0,15),edge=0)]
    for y in (-120,120):
        # A genuine V groove is split into a bottom and two convex jaws.
        profiles=[ [(-40,30),(40,30),(40,40),(-40,40)],
                   [(-40,40),(0,40),(-20,60),(-40,60)],
                   [(0,40),(40,40),(40,60),(20,60)] ]
        for profile in profiles:
            verts=[(x,y+dy,z) for dy in (-30,30) for x,z in profile]; n=len(profile)
            faces=[tuple(reversed(range(n))),tuple(range(n,2*n))]+[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
            obj=mesh('V seat jaw',verts,faces,steel)
            props._select([obj]); bevel=obj.modifiers.new('Jaw edge break','BEVEL')
            bevel.width=.0004; bevel.segments=3
            bpy.ops.object.modifier_apply(modifier=bevel.name); visuals.append(obj)
            cols.append(mesh('V seat proxy',verts,faces))
        visuals.append(cylinder('Lower spindle',9,20,(0,y,50),steel,.3))
        cols.append(cylinder('Lower spindle proxy',9,20,(0,y,50),edge=0,count=32))
        core=cylinder('Laminated rotor core',18,50,(0,y,85),dark,.4)
        # Procedural closely spaced lamination lines retained in PBR and albedo.
        tree=dark.node_tree; bsdf=tree.nodes['Principled BSDF']
        if not tree.nodes.get('Lamination bands'):
            wave=tree.nodes.new('ShaderNodeTexWave'); wave.name='Lamination bands'; wave.bands_direction='Z'; wave.inputs['Scale'].default_value=1800
            coords=tree.nodes.new('ShaderNodeTexCoord'); tree.links.new(coords.outputs['Object'],wave.inputs['Vector'])
            ramp=tree.nodes.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color=(.018,.021,.026,1); ramp.color_ramp.elements[1].color=(.09,.10,.12,1)
            tree.links.new(wave.outputs['Fac'],ramp.inputs[0]); tree.links.new(ramp.outputs['Color'],bsdf.inputs['Base Color'])
        visuals.append(core); cols.append(cylinder('Core proxy',18,50,(0,y,85),edge=0,count=16))
        visuals.append(cylinder('Shoulder diameter 26',13,6,(0,y,113),steel,.15))
        cols.append(cylinder('Shoulder proxy',13,6,(0,y,113),edge=0,count=32))
        tip=9-4*math.tan(math.radians(30))
        visuals.append(loft('Precision bearing seat',[(9,116),(9,152),(tip,156)],96,steel,(0,y)))
        cols.append(cylinder('Precision 32-facet shaft',9,36,(0,y,134),edge=0,count=32))
        cols.append(loft('Precision lead-in',[(9,152),(tip,156)],32,None,(0,y)))
    parts=[]; buttons=[]
    for name,y in [('start_left',190),('start_right',-190)]:
        visuals.append(box('Button enclosure',(60,60,50),(-100,y,55),grey,1))
        visuals.append(cylinder('Start bezel',22,3,(-100,y,81.5),steel,.4))
        cols.append(cylinder('Bezel proxy',22,3,(-100,y,81.5),edge=0,count=32))
        cols.append(box('Button enclosure proxy',(60,60,50),(-100,y,55),edge=0))
        # Cap and short stem form one independently moving convex mushroom.
        cap=loft(name,[(15,80),(15,84),(19.5,84),(20,85),(20,94),(19,96)],64,green,(-100,y))
        capcol=cylinder('Cap proxy',20,12,(-100,y,90),edge=0,count=32)
        joint=dict(type='slide',axis=[0,0,-1],anchor_m=[-.1,y/1000,.08],range=[0,.006],damping=2.,frictionloss=.1,
                   spring={'stiffness':700.,'reference':0.})
        parts.append(part(name,[cap],[capcol],.02,parent='body',joint=joint,size=1024,friction=.8))
        buttons.append(dict(part=name,press_direction=[0,0,-1],travel_m=.006,pressed_threshold_m=.004,
                            center_m=[-.1,y/1000,.096]))
        visuals.append(label(name+' legend',(-130.02,y,56),48,11,image_material(name+' ink',art(folder,'label','START',name+'_label.png'))))
    visuals.append(cylinder('Station lamp post',10,250,(105,0,155),steel,.4))
    cols.append(cylinder('Post proxy',10,250,(105,0,155),edge=0,count=32))
    visuals.append(cylinder('Lamp foot',24,4,(105,0,278),grey,.4))
    cols.append(cylinder('Lamp foot proxy',24,4,(105,0,278),edge=0,count=32))
    lampmat=material('Unlit station lamp','#55705A',rough=.3)
    lens=loft('Station lens',[(25,280),(25,295),(24,300),(20,305)],64,lampmat,(105,0))
    lenscol=loft('Lens proxy',[(25,280),(25,295),(24,300),(20,305)],16,None,(105,0))
    lamp=part('station_lamp',[lens],[lenscol],.04,parent='body',rgba=[.12,.22,.14,1])
    lampmat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(.12,.22,.14,1)
    visuals.append(label('Fixture ID',(-130.02,0,17),190,14,image_material('Fixture nameplate',art(folder,'label','IMC BEARING PRE-ASSEMBLY'))))
    body=part('body',visuals,cols,11.92,mujoco=METAL)
    complete(folder,[body,*parts,lamp],dict(base_xyz=[260,440,30],station_centers_xy=[[0,120],[0,-120]],
        shaft_diameter=18,shaft_height=40,shaft_top_z=156,shoulder_diameter=26,shoulder_top_z=116,
        lead_in_axial_length=4,lead_in_angle_deg=30,lead_in_tip_diameter=2*tip,
        shaft_collision_facets=32,shaft_chord_error=9*(1-math.cos(math.pi/32))),
        dict(task='bearing_assembly',static=True,precision_mm=.05,recommended_placement_m={'origin':[.4,0,.9]},
             collision_exclusions=[['start_left','body'],['start_right','body']],
             buttons=buttons,lamps=[dict(part='station_lamp',on_rgba=[.08,.85,.18,1],emission=.8)],
             two_hand_start={'parts':['start_left','start_right'],'max_interval_s':.5,'requires_both_bearings_seated':True},
             bearing_seats=[dict(center_m=[0,y/1000,.116],axis=[0,0,1],shaft_diameter_m=.018,
                                 bearing_bore_m=.020,diametral_clearance_m=.002) for y in (120,-120)]),
        notes='32 shaft facets replace the listed 24: R9 at 24 facets has 0.077 mm chord error, exceeding the binding 0.05 mm tolerance; '
        '32 gives 0.04334 mm and 64 vertices. V clamps use three convex pieces each to preserve the groove. '
        'Lead-in is 4 mm axial length at 30 degrees to the shaft axis. No overhead crossbeam. '
        'Buttons and station lamp are separate functional parts; scene implements the 0.5 s interlock and the supplied cap/support contact exclusions.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--props',nargs='+',choices=IDS,default=list(IDS))
    parser.add_argument('--skip-existing',action='store_true',help='Resume completed manifests; unfinished props are rebuilt')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    builders=dict(zip(IDS,[build_parts_bin,build_relay,build_terminal,build_gland,build_fuse,build_order_tray,build_card,build_fixture,build_bearing_tray]))
    for name in args.props:
        if args.skip_existing and (OUTPUT/name/'prop.json').exists():
            props._DELIVERED_PROPS.add(name)
            continue
        builders[name]()
    finalize_selected()


if __name__=='__main__': main()
