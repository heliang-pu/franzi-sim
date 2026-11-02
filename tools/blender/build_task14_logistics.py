"""Build section 6.1 logistics props (geometry arguments in millimetres).

blender -b --factory-startup -t 6 -P tools/blender/build_task14_logistics.py -- --props ID ...
Artwork uses PROP_PYTHON and the bundled DejaVu Sans fonts.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from collections.abc import Callable, Sequence

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def artwork_worker() -> None:
    from PIL import Image, ImageDraw, ImageFont
    request = json.load(sys.stdin)
    kind = request['kind']
    colors = {'ok': ('#2E7D32', 'white'), 'recheck': ('#F9A825', '#111111'),
              'ng': ('#C62828', 'white')}
    bg, ink = colors.get(kind, ('#E5E7E8', '#16191D'))
    image = Image.new('RGB', (1200, 880), bg)
    d = ImageDraw.Draw(image)
    def text(y, body, size):
        font = ImageFont.truetype(str(HERE / 'fonts/DejaVuSans-Bold.ttf'), size)
        d.text((600, y), body, font=font, fill=ink, anchor='mm')
    if kind in colors:
        text(220, {'ok': 'OK', 'recheck': 'RECHECK', 'ng': 'NG'}[kind], 145 if kind == 'recheck' else 220)
        if kind == 'ok':
            text(435, 'STORAGE', 86)
            d.line([(445, 655), (555, 760), (785, 535)], fill=ink, width=45)
        elif kind == 'ng':
            text(435, 'QUARANTINE', 72)
            d.line((490, 560, 710, 780), fill=ink, width=45)
            d.line((710, 560, 490, 780), fill=ink, width=45)
        else:
            d.ellipse((440, 460, 690, 710), outline=ink, width=35)
            d.line((655, 675, 790, 810), fill=ink, width=42)
    else:
        text(330, kind, 145 if len(kind) < 10 else 95)
        text(610, request.get('brand', 'IMC Controls'), 70)
    image.save(request['path'])


if __name__ == '__main__' and '--artwork-worker' in sys.argv:
    artwork_worker()
    raise SystemExit

import bpy
from mathutils import Matrix, Vector
import prop_lib as p
from build_task01_tote_transfer import finish_mesh

IDS = ('tote_cart', 'line_feed_conveyor', 'takeaway_conveyor', 'induction_conveyor',
       'flow_lane', 'recheck_shelf', 'station_sign')
PAINT = {'specular': .3, 'shininess': .3, 'reflectance': 0.0}
METAL = {'specular': .7, 'shininess': .7, 'reflectance': .05}


def material(name: str, color: str, metal: bool = False) -> bpy.types.Material:
    return p.pbr_material(name, color, .3 if metal else .6, 1 if metal else 0,
                          wear=.035, grime=.12, machining='brushed' if metal else None)


def recolor(mat: bpy.types.Material, color: str) -> None:
    next(n for n in mat.node_tree.nodes if n.type == 'RGB').outputs[0].default_value = p._linear_color(color)
    mat.diffuse_color = p._linear_color(color)


def handling_smudges(mat: bpy.types.Material, center_m: tuple, radius_m: float) -> None:
    """Bake local hand grime into frequently touched grips and button caps."""
    tree = mat.node_tree
    bsdf = tree.nodes['Principled BSDF']
    color = bsdf.inputs['Base Color'].links[0].from_socket
    coords = tree.nodes.new('ShaderNodeTexCoord')
    distance = tree.nodes.new('ShaderNodeVectorMath')
    distance.operation = 'DISTANCE'
    tree.links.new(coords.outputs['Object'], distance.inputs[0])
    distance.inputs[1].default_value = center_m
    mask = p._math(tree, 'MAXIMUM', 0, p._math(tree, 'SUBTRACT', 1,
                   p._math(tree, 'DIVIDE', distance.outputs['Value'], radius_m)))
    noise = tree.nodes.new('ShaderNodeTexNoise')
    noise.name = 'Soft fingertip smears and handling residue'
    noise.inputs['Scale'].default_value = 350
    noise.inputs['Detail'].default_value = 3
    tree.links.new(coords.outputs['Object'], noise.inputs['Vector'])
    factor = p._math(tree, 'SUBTRACT', 1, p._math(tree, 'MULTIPLY', .18,
                     p._math(tree, 'MULTIPLY', mask, noise.outputs['Fac'])))
    mix = tree.nodes.new('ShaderNodeMixRGB')
    mix.name = 'Localized hand-contact smudges'
    mix.blend_type = 'MULTIPLY'
    mix.inputs[0].default_value = 1
    tree.links.new(color, mix.inputs[1])
    tree.links.new(factor, mix.inputs[2])
    tree.links.new(mix.outputs[0], bsdf.inputs['Base Color'])


class Build:
    def __init__(self, name: str, task: str, mass: float, origin: str = 'bottom_center') -> None:
        p.reset_scene()
        bpy.ops.outliner.orphans_purge(do_recursive=True)
        self.folder = p.ROOT / 'assets/props' / name
        self.folder.mkdir(parents=True, exist_ok=True)
        self.parts, self.visuals = [], []
        self.data = dict(id=name, task=task, units='m', origin=origin, static=True,
                         budget_class='large' if 'conveyor' in name or name == 'flow_lane' else 'medium',
                         variants=[], dimensions_mm={}, recommended_placement_m={},
                         notes='Geometry in the common prop frame, Z up, front -X. CPU Cycles 64-sample diffuse colour times (0.4 + 0.6 AO); Smart UV 66 degrees. Full PBR in source.blend. ',
                         mass_kg=mass)
        self.steel = material('Brushed zinc steel', '#BFC3C7', True)
        self.paint = material('NOVA blue powder coat', '#1E4C9A')
        self.dark = material('Dark rubber and drive guards', '#24282D')
        self.grey = material('Grey anti-slip rubber', '#6E7378')
        self.groups = {}

    def box(self, group: str, name: str, size: Sequence[float], loc: Sequence[float],
            mat: bpy.types.Material | None = None, collision: bool = True, bevel: float = 1,
            rotation: Sequence[float] = (0, 0, 0)) -> bpy.types.Object:
        v, c = self.groups.setdefault(group, ([], []))
        obj = p.box(name, size, loc, bevel_mm=bevel, segments=2, material=mat or self.paint, rotation_deg=rotation)
        v.append(obj)
        if collision:
            c.append(p.box('COL ' + name, size, loc, bevel_mm=0, rotation_deg=rotation))
            c[-1].hide_render = True
        return obj

    def cyl(self, group: str, name: str, radius: float, length: float, loc: Sequence[float],
            mat: bpy.types.Material | None = None, rotation: Sequence[float] = (0, 0, 0),
            collision: bool = True, vertices: int = 32, bevel: float = .5) -> bpy.types.Object:
        v, c = self.groups.setdefault(group, ([], []))
        obj = p.cylinder(name, radius, length, loc, material=mat or self.steel, rotation_deg=rotation,
                         vertices=vertices, bevel_mm=bevel, segments=2)
        v.append(obj)
        if collision:
            c.append(p.cylinder('COL ' + name, radius, length, loc, rotation_deg=rotation, vertices=16, bevel_mm=0))
            c[-1].hide_render = True
        return obj

    def bar(self, group: str, name: str, a: Sequence[float], b: Sequence[float], radius: float,
            mat: bpy.types.Material | None = None, collision: bool = True) -> bpy.types.Object:
        center = (Vector(a) + Vector(b)) / 2
        direction = Vector(b) - Vector(a)
        rotation = tuple(math.degrees(x) for x in direction.to_track_quat('Z', 'Y').to_euler())
        return self.cyl(group, name, radius, direction.length, center, mat, rotation, collision)

    def label(self, group: str, text: str, center: Sequence[float], size: Sequence[float],
              facing: str = 'front', brand: str = 'IMC Controls') -> tuple:
        path = self.folder / (text.lower().replace(' ', '_') + '_art.png')
        subprocess.run([str(p.VALIDATOR_PYTHON), '-B', str(Path(__file__)), '--artwork-worker'],
                       input=json.dumps(dict(kind=text, path=str(path), brand=brand)), text=True, check=True)
        mat = p.pbr_material(text + ' printed artwork', '#FFFFFF', .65, 0)
        tree = mat.node_tree
        uv = tree.nodes.new('ShaderNodeUVMap'); uv.uv_map = 'LabelUV'
        image = tree.nodes.new('ShaderNodeTexImage'); image.image = bpy.data.images.load(str(path)); image.extension = 'EXTEND'
        tree.links.new(uv.outputs['UV'], image.inputs['Vector'])
        tree.links.new(image.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
        w, h = size
        x, y, z = center
        if facing == 'front':
            vertices = [(x, y+w/2, z-h/2), (x, y-w/2, z-h/2), (x, y-w/2, z+h/2), (x, y+w/2, z+h/2)]
        else:
            vertices = [(x-w/2,y-h/2,z),(x+w/2,y-h/2,z),(x+w/2,y+h/2,z),(x-w/2,y+h/2,z)]
        obj = p._mesh_object(text + ' print', vertices, [(0,1,2,3)], (0,0,0), 0, 1, mat)
        uv = obj.data.uv_layers.new(name='LabelUV')
        for loop, coord in zip(uv.data, ((0,0),(1,0),(1,1),(0,1))): loop.uv = coord
        self.groups.setdefault(group, ([], []))[0].append(obj)
        return mat, image

    def part(self, name: str, mass: float, friction: float = .4, parent: str | None = None,
             joint: dict | None = None, rgba: Sequence[float] | None = None,
             shared: str | None = None) -> bpy.types.Object:
        visuals, collisions = self.groups.pop(name)
        obj = p.join_objects(visuals, name)
        for col in collisions:
            col.hide_render = True
        if rgba is None and not shared:
            p.uv_unwrap(obj)
        finish_mesh(obj)
        if rgba is None:
            if not shared:
                p.bake_albedo([obj], 1024, self.folder / (name + '_albedo.png'))
            texture = (shared or name) + '_albedo.png'
        else:
            texture = None
        files = p.export_collision(name, collisions, self.folder)
        for col in collisions:
            col.hide_render = True; col.hide_set(True)
        p.export_obj(obj, self.folder / (name + '.obj'))
        part = dict(name=name, visual=name+'.obj', texture=texture, texture_px=1024 if texture else None,
                    collision=files, mass_kg=mass, friction=friction, mujoco_material=METAL if name.startswith('roller') else PAINT,
                    parent=parent, joint=joint)
        if rgba is not None: part['rgba'] = rgba
        self.parts.append(part); self.visuals.append(obj)
        return obj

    def feet(self, xs: Sequence[float], ys: Sequence[float],
             top: float | Callable[[float], float], group: str = 'frame') -> None:
        for x in xs:
            for y in ys:
                z = top(x) if callable(top) else top
                self.box(group, '40 mm aluminium upright', (40,40,z-25), (x,y,(z+25)/2), self.steel, bevel=1)
                self.cyl(group, 'Inboard levelling foot', 24, 12, (x,y,6), self.dark)
                self.cyl(group, 'Levelling screw', 7, 24, (x,y,24), self.steel)
            self.box(group, 'Lower transverse brace', (30,abs(ys[-1]-ys[0]),30), (x,0,260), self.steel)

    def rollers(self, count: int, radius: float, length: float, pitch: float, first: float,
                top: float, axis: str = 'Y', mass: float = .3) -> None:
        loc = (first,0,top-radius) if axis == 'Y' else (85,first,top-radius)
        rotation = (90,0,0) if axis == 'Y' else (0,90,0)
        self.cyl('roller_00', 'Steel roller', radius, length, loc, self.steel, rotation, vertices=48, bevel=.4)
        joint = dict(type='hinge', axis=[0,1,0] if axis=='Y' else [1,0,0], anchor_m=[v/1000 for v in loc],
                     range=[-math.pi,math.pi], limited=False, continuous=True, damping=.001, frictionloss=0)
        template = self.part('roller_00', mass, .6, 'frame', joint)
        names = ['roller_00']
        for i in range(1,count):
            name = f'roller_{i:02d}'; names.append(name)
            obj = bpy.data.objects.new(name, template.data.copy()); bpy.context.collection.objects.link(obj)
            delta = Vector((pitch*i/1000,0,0) if axis=='Y' else (0,pitch*i/1000,0))
            obj.data.transform(Matrix.Translation(delta))
            center = [loc[j]+delta[j]*1000 for j in range(3)]
            col = p.cylinder('COL '+name, radius,length,center,rotation_deg=rotation,vertices=16,bevel_mm=0)
            col.hide_render=True
            self.groups[name] = ([obj],[col])
            self.part(name,mass,.6,'frame',dict(joint,anchor_m=[v/1000 for v in center]),shared='roller_00')
        self.data['conveyor'] = dict(roller_parts=names, direction=[1,0,0] if axis=='Y' else [0,-1,0],
                                     roller_top_m=top/1000, roller_radius_m=radius/1000,
                                     drive='velocity_actuators', positive_angular_velocity=True)

    def finish(self) -> None:
        assert not self.groups, self.groups.keys()
        if self.data.get('conveyor', {}).get('roller_parts'):
            names = self.data['conveyor']['roller_parts'] + [
                item['part'] for key in ('buttons', 'lamps') for item in self.data.get(key, [])]
            self.data['collision_exclusions'] = [['frame', name] for name in names]
        self.data.update(parts=self.parts,bbox_mm=[round(v,5) for v in p.bbox_mm(self.visuals)],
                         triangles=p.triangle_count(self.visuals),source='source.blend',preview='preview.png')
        p.render_preview(self.visuals,self.folder/'preview.png')
        p._select(self.visuals)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(self.folder/'source.blend'),check_existing=False)
        p.write_prop_json(self.data,self.folder/'prop.json')
        subprocess.run([str(p.VALIDATOR_PYTHON), '-B', str(HERE / 'check_task14_logistics.py'),
                        '--repair-collisions', '--props', self.data['id']], check=True)
        print('TASK14_COMPLETE',self.data['id'],flush=True)


def build_cart() -> None:
    b=Build('tote_cart','line_replenish tote_sorting',18)
    handling_smudges(b.paint, (.1025,0,.9375), .16)
    for rubber in (b.grey, b.dark):
        rubber.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = .9
    for z in (300,800):
        b.box('frame','Steel shelf',(260,520,4),(0,0,z-4),bevel=1)
        b.box('frame','Anti-slip mat',(254,514,2),(0,0,z-1),b.grey,collision=False,bevel=.5)
    # Deck and rubber together have exact support planes at 300 / 800 mm.
    for c, top in zip(b.groups['frame'][1], (.300, .800)):
        for v in c.data.vertices:
            if v.co.z > top - .004: v.co.z = top
    for s in (-1,1):
        b.box('frame','Deck lip X',(1.5,520,10),(s*129.25,0,805),bevel=.5)
        b.box('frame','Deck lip Y',(257,1.5,10),(0,s*259.25,805),bevel=.5)
    for x in (-102.5,102.5):
        for y in (-222.5,222.5):
            b.box('frame','25 mm square upright',(25,25,696),(x,y,448),bevel=1)
            b.cyl('frame','75 mm rubber caster',37.5,25,(x,y,37.5),b.dark,(0,90,0),collision=False)
            b.cyl('frame','Wheel hub',13,27,(x,y,37.5),b.steel,(0,90,0),collision=False)
            b.box('frame','Caster crown',(27,55,8),(x,y,88),b.steel,collision=False)
            for side in (-1,1): b.box('frame','Wheel fork',(3,28,50),(x+side*12,y,62),b.steel,collision=False,bevel=.5)
            b.cyl('frame','Swivel bearing',13,12,(x,y,94),b.steel,collision=False)
            b.groups['frame'][1].append(p.box('COL caster envelope',(28,75,100),(x,y,50),bevel_mm=0))
    # Bent handle: 25 mm tube with rounded shoulder bends, all within deck footprint.
    for y in (-222.5,222.5): b.bar('frame','Handle upright',(102.5,y,796),(102.5,y,912.5),12.5,b.paint)
    points=[(102.5,-222.5,912.5)]
    for i in range(1,9):
        t=math.pi*i/16; points.append((102.5,-197.5-25*math.cos(t),912.5+25*math.sin(t)))
    points.append((102.5,197.5,937.5))
    for i in range(1,9):
        t=math.pi*i/16; points.append((102.5,197.5+25*math.sin(t),912.5+25*math.cos(t)))
    # Fine bend sections are visual; the two uprights plus crossbar are three prescribed proxies.
    for a,c in zip(points,points[1:]): b.bar('frame','Bent push handle',a,c,12.5,b.paint,collision=False)
    b.groups['frame'][1].append(p.box('COL handle crossbar',(25,420,25),(102.5,0,937.5),bevel_mm=0))
    obj=b.part('frame',18,.5)
    for i,(name,color) in enumerate((('blue','#1E4C9A'),('red','#C62828'),('green','#2E7D32'))):
        tex='frame_albedo.png' if i==0 else f'frame_albedo_v{i}.png'
        if i:
            recolor(b.paint,color); p.bake_albedo([obj],1024,b.folder/tex); p.render_preview([obj],b.folder/f'preview_{name}.png')
        b.data['variants'].append(dict(name=name,textures={'frame':tex}))
    recolor(b.paint,'#1E4C9A')
    b.data['dimensions_mm']=dict(deck_xy=[260,520],deck_top=800,lower_shelf_top=300,upright_xy=[25,25],caster_diameter=75,caster_height=100,handle_top=950,footprint_xy=[260,520],upright_centers_xy=[[x,y] for x in (-102.5,102.5) for y in (-222.5,222.5)])
    b.data['recommended_placement_m']=dict(line_replenish=[.37,0,0],tote_sorting=[dict(position=[0,.50,0],yaw_degrees=90),dict(position=[0,-.50,0],yaw_degrees=-90)])
    b.data['notes']+='Y uprights are inset an extra 10 mm to keep complete 75 mm caster envelopes inside the deck. Caster boxes and three handle proxies are the task-prescribed simplifications. '
    b.finish()


def build_line() -> None:
    b=Build('line_feed_conveyor','line_replenish',45,'ground_entry_center')
    # Real C sections, decomposed into webs and flanges.
    for s in (-1,1):
        b.box('frame','C channel web',(1200,3,80),(600,s*188.5,1000),b.steel)
        for z in (961.5,1038.5): b.box('frame','C channel flange',(1200,20,3),(600,s*180,z),b.steel,bevel=.7)
    b.feet((100,1100),(-185,185),960)
    b.box('frame','Longitudinal brace',(1000,30,30),(600,185,400),b.steel)
    for s in (-1,1):
        b.bar('frame','12 mm guide',(360,s*171,1069),(1194,s*171,1069),6,b.steel)
        b.bar('frame','15 degree guide flare',(302,s*(171+58*math.tan(math.radians(15))),1069),(360,s*171,1069),6,b.steel)
        for x in (380,900,1170): b.bar('frame','Guide stanchion',(x,s*184,1038),(x,s*171,1069),4,b.steel)
    b.bar('frame','Terminal stop',(1194,-171,1084),(1194,171,1084),6,b.steel)
    b.box('frame','Drive chain cover',(520,35,95),(770,-211,945),b.dark,collision=False,bevel=2)
    b.cyl('frame','Gear motor',40,130,(700,-218,850),b.steel,(90,0,0),collision=False)
    b.bar('frame','Button support arm',(85,-185,975),(-50,-280,940),10,b.steel)
    b.box('frame','Button enclosure',(80,80,100),(-50,-280,950),material('Light grey enclosure','#D7D7D7'),bevel=2)
    b.cyl('frame','Button bezel',16,3,(-50,-280,1001.5),b.steel)
    b.label('frame','RELEASE',(-50,-304,1000.1),(66,15),facing='top')
    b.box('frame','Photoelectric sensor',(28,18,24),(70,184,1052),b.dark,collision=False)
    b.box('frame','Sensor blue lens',(14,1,12),(70,174.5,1052),material('Sensor glass','#1565C0'),collision=False,bevel=.3)
    green=material('Green mushroom cap','#2E9E44')
    handling_smudges(green,(-.05,-.28,1.018),.035)
    b.cyl('release_button','Mushroom stem',12,8,(-50,-280,1004),green,collision=False,bevel=.5)
    b.cyl('release_button','Mushroom button',20,12,(-50,-280,1012),green,vertices=48,bevel=2)
    b.part('frame',34.98)
    b.part('release_button',.02,.8,'frame',dict(type='slide',axis=[0,0,-1],anchor_m=[-.05,-.28,1.0],range=[0,.006],damping=1,frictionloss=.1,spring=dict(stiffness=900,reference=0)))
    b.rollers(20,24,330,60,30,1050,mass=.5)
    b.data['buttons']=[dict(part='release_button',press_direction=[0,0,-1],travel_m=.006,label='RELEASE')]
    b.data['dimensions_mm']=dict(length=1200,roller_diameter=48,roller_length=330,roller_count=20,roller_pitch=60,first_roller_x=30,roller_top=1050,frame_inner_width=340,guide_inner_width=330,guide_start_x=300,guide_height=25,button_box_center_xy=[-50,-280],button_box_top=1000,footprint_xy_bounds=[[0,1200],[-240,240]])
    b.data['recommended_placement_m']=[.5,0,0]
    b.data['notes']+='Guides and 15 degree flares begin at X300; loading segment is unobstructed. Guide upper surface is 25 mm above rollers; the flare starts at X302 so its radius stays beyond X300. Positive Y-axis roller speed transports +X. Motor, chain cover and photoelectric sensor are visual only. '
    b.finish()


def build_takeaway() -> None:
    b=Build('takeaway_conveyor','packing',12)
    b.box('packing_plate','Aluminium packing plate',(170,400,28),(-150,0,14),b.steel,bevel=1)
    b.box('packing_plate','Grey ESD mat',(170,400,2),(-150,0,29),b.grey,collision=False,bevel=.4)
    # Single task-prescribed plate collider includes its rubber surface.
    col=b.groups['packing_plate'][1][0]
    for v in col.data.vertices:
        if v.co.z>.014: v.co.z=.030
    b.box('frame','Roller bed',(300,800,3),(85,0,1.5),b.steel,bevel=.7)
    b.box('frame','Low near side rail',(8,800,25),(-61,0,15.5),b.steel,bevel=.7)
    b.box('frame','Far guide',(8,800,67),(231,0,36.5),b.paint,bevel=1)
    b.box('frame','Terminal stop',(284,4,67),(85,-398,36.5),b.paint,bevel=1)
    b.part('frame',5.8)
    b.part('packing_plate',3.6,.5,'frame')
    b.rollers(26,10,280,30,-375,30,axis='X',mass=.1)
    b.data['dimensions_mm']=dict(packing_plate_x=[-235,-65],packing_plate_y=[-200,200],packing_plate_top=30,roller_bed_x=[-65,235],roller_bed_y=[-400,400],roller_count=26,roller_pitch=30,roller_diameter=20,roller_length=280,roller_axis_z=20,near_rail_top=28,far_guide_top=70,end_stop_top=70)
    b.data['recommended_placement_m']=[.535,0,.90]
    b.data['notes']+='Positive X-axis angular velocity transports -Y. Roller centres range Y -375 to +375. No floor legs; support plane Z0 rests on the workbench. '
    b.finish()


def build_induction() -> None:
    b=Build('induction_conveyor','parcel_induction',50,'ground_entry_center')
    for s in (-1,1):
        b.box('frame','Side frame',(1100,25,60),(550,s*200.5,890),b.steel,bevel=2)
        b.box('frame','Stainless side guide',(1100,3,40),(550,s*186.5,940),b.steel,bevel=.7)
    b.box('frame','Entry plate',(3,370,50),(1.5,0,945),b.steel,bevel=.7)
    b.box('frame','End stop',(3,370,60),(1103.5,0,950),b.steel,bevel=.7)
    # The final roller reaches X1100: stop inner face X1102 leaves 2 mm.
    b.feet((50,500,1050),(-200,200),860)
    for y in (-280,280): b.box('frame','Scanner gantry upright',(40,40,1230),(650,y,615),b.steel,bevel=2)
    b.box('frame','Scanner crossbeam',(40,600,40),(650,0,1210),b.steel,bevel=2)
    b.box('frame','Scanner housing',(80,120,60),(650,0,1160),b.dark,bevel=2)
    b.box('frame','Downward red scanner window',(60,95,1),(650,0,1129.5),material('Red optical window','#B61E2E'),collision=False,bevel=.4)
    b.label('frame','SCAN',(629.9,0,1210),(150,31))
    b.cyl('frame','Lamp mounting boss',12,10,(650,280,1235),b.dark)
    # Keep the boss proxy 0.1 mm below the lamp hull at rest.
    for vertex in b.groups['frame'][1][-1].data.vertices:
        if vertex.co.z > 1.235:
            vertex.co.z = 1.2394
    b.cyl('frame','Drive motor',30,95,(850,-225,800),b.steel,(90,0,0),collision=False)
    b.part('frame',41.18)
    lampmat=material('Unlit scanner status lens','#145026')
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,radius=.02,location=(.650,.280,1.260))
    lamp=bpy.context.object; lamp.name='Scanner status sphere'; lamp.data.materials.append(lampmat)
    p._select([lamp]); bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
    # A sphere collider is an explicitly convex low-poly hull under 64 vertices.
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=.0205,location=(.650,.280,1.260))
    col=bpy.context.object; col.hide_render=True
    b.groups['scan_lamp']=([lamp],[col])
    b.part('scan_lamp',.02,.4,'frame',rgba=[.06,.30,.10,1])
    b.rollers(44,10,360,25,15,940,mass=.2)
    b.data['lamps']=[dict(part='scan_lamp',on_rgba=[.18,.80,.25,1],emission=.8,states={'read':[.18,.80,.25,1],'unread':[.9,.08,.06,1]})]
    b.data['conveyor']['speed_m_s']=.1
    b.data['dimensions_mm']=dict(length=1105,end_stop_inner_x=1102,roller_end_clearance=2,roller_count=44,roller_diameter=20,roller_length=360,roller_pitch=25,first_roller_x=15,roller_top=940,guide_inner_width=370,guide_top=960,entry_plate_top=970,end_plate_top=980,gantry_x=650,gantry_post_y=[-280,280],beam_bottom=1190,scanner_xyz=[80,120,60],lamp_diameter=40,footprint_xy_bounds=[[0,1100],[-300,300]])
    b.data['recommended_placement_m']=[.30,0,0]
    b.data['notes']+='Scanner 120 mm width is along Y; 80 mm depth along X. Gantry feet remain inside its 600 mm overall footprint. Lamp is a separate untextured dark lens; simulation changes rgba and emission. '
    b.finish()


def build_flow() -> None:
    b=Build('flow_lane','tote_sorting',30,'ground_pickup_end_center')
    slope=math.tan(math.radians(3)); z=lambda x:1050+x*slope
    b.feet((60,740),(-200,200),lambda x:z(x)-35)
    for y in (-200,200):
        b.bar('frame','Diagonal brace',(60,y,300),(740,y,940),12,b.steel)
        b.box('frame','Inclined rail',(800/math.cos(math.radians(3)),30,30),(400,y,z(400)-30),b.steel,rotation=(0,-3,0))
    b.box('frame','Front 30 mm stop',(30,340,30),(-15,0,1065),b.paint,bevel=1)
    # Vertical sheet edges end exactly at X220 / X800. Rotating a 40 mm
    # tall box would project its end corners into the clear pickup segment.
    def guide(x0: float, x1: float, y0: float, y1: float, side: int) -> None:
        vertices = []
        for x, yc in ((x0, side*y0), (x1, side*y1)):
            vertices.extend([(x,yc-1.5,z(x)), (x,yc+1.5,z(x)),
                             (x,yc+1.5,z(x)+40), (x,yc-1.5,z(x)+40)])
        faces=[(3,2,1,0),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
        visual=p._mesh_object('Slope-following guide plate',vertices,faces,(0,0,0),.5,2,b.steel)
        collider=p._mesh_object('COL guide plate',vertices,faces,(0,0,0),0,1,None)
        collider.hide_render=True
        b.groups['frame'][0].append(visual)
        b.groups['frame'][1].append(collider)
    for side in (-1,1):
        guide(260,800,161.5,161.5,side)
        guide(220,260,175,161.5,side)
    for x in (60,740):
        b.box('frame','Upper lane support crossmember',(30,400,30),(x,0,z(x)-47),b.steel)
    for y in (-130,130): b.bar('frame','Rear label post',(790,y,z(790)),(790,y,1280),6,b.steel,collision=False)
    b.box('frame','Rear label holder',(3,290,85),(790,0,1240),b.paint,collision=False)
    b.label('frame','FIFO',(788.4,0,1240),(265,65),brand='NOVA Logistics')
    b.part('frame',24)
    length=800/math.cos(math.radians(3))
    slab = b.box('lane','Low-friction lane proxy template',(length,340,5),(400,0,z(400)-2.5/math.cos(math.radians(3))),b.grey,rotation=(0,-3,0),bevel=.5)
    b.groups['lane'][0].remove(slab)
    bpy.data.objects.remove(slab, do_unlink=True)
    # Slotted visual deck exposes the three wheel strips; one continuous slab
    # collider intentionally bridges these visual wheel recesses (section 6.1).
    for lo, hi in ((-170,-131),(-89,-21),(21,89),(131,170)):
        b.box('lane','Flush deck between roller strips',(length,hi-lo,5),(400,(lo+hi)/2,z(400)-2.5/math.cos(math.radians(3))),b.grey,collision=False,bevel=.5,rotation=(0,-3,0))
    b.box('lane','Recessed strip backing',(length,340,3),(400,0,z(400)-32),b.grey,collision=False,bevel=.5,rotation=(0,-3,0))
    yellow=material('Yellow flow strips','#F2B705')
    for y in (-110,0,110):
        for side in (-1,1):
            b.box('lane','Yellow strip channel',(length,3,18),(400,y+side*19.5,z(400)-16),yellow,collision=False,bevel=.5,rotation=(0,-3,0))
        for x in range(20,800,34):
            b.cyl('lane','Visual gravity wheel',15,34,(x,y,z(x)-15/math.cos(math.radians(3))),yellow,(90,0,0),collision=False,vertices=24,bevel=.5)
    b.part('lane',6,.03,'frame')
    b.parts[-1]['contact_priority'] = 1
    b.data['notes'] += ('Lane friction must win over the goods: MuJoCo geom priority 1; '
                        'PhysX friction combine mode min. Friction 0.03 is below '
                        'tan(3 degrees) = 0.0524 so gravity moves the tote. ')
    b.data['conveyor']=dict(type='gravity_plane',surface_part='lane',roller_parts=[],direction=[-math.cos(math.radians(3)),0,-math.sin(math.radians(3))],roller_top_m=1.05,roller_radius_m=.015,slope_degrees=3,top_height_endpoints_m=[1.05,z(800)/1000],drive='gravity')
    b.data['dimensions_mm']=dict(lane_width=340,lane_horizontal_length=800,slope_degrees=3,low_top=1050,high_top=z(800),stop_inner_x=0,stop_top=1080,guide_start_x=220,guide_straight_start_x=260,guide_inner_width=320,guide_height=40,upright_x=[60,740],upright_y=[-200,200],footprint_xy_bounds=[[-30,800],[-224,224]])
    b.data['recommended_placement_m']=[.5,0,0]
    b.data['notes']+='Lane is one fixed low-friction part and one inclined slab collider. Its three rows of yellow D30 wheels are visual only and tangent to the support plane, as section 6.1 prescribes; these are not powered hinged rollers. '
    b.finish()


def build_shelf() -> None:
    b=Build('recheck_shelf','tote_sorting',12)
    recolor(b.paint,'#F9A825')
    for z in (300,800):
        b.box('frame','Shelf panel',(240,400,4),(0,0,z-2),b.grey)
    for s in (-1,1):
        b.box('frame','Shelf lip X',(2,400,10),(s*119,0,805),bevel=.5)
        b.box('frame','Shelf lip Y',(236,2,10),(0,s*199,805),bevel=.5)
    for x in (-87.5,87.5):
        for y in (-167.5,167.5): b.box('frame','25 mm upright',(25,25,796),(x,y,398),bevel=1)
    b.part('frame',12,.5)
    b.data['dimensions_mm']=dict(shelf_xy=[240,400],top=800,lower_shelf_top=300,lip_height=10,upright_outer_inset=20,footprint_xy=[240,400])
    b.data['recommended_placement_m']=[.36,0,0]
    b.finish()


def build_sign() -> None:
    b=Build('station_sign','tote_sorting',5)
    b.cyl('body','Cast iron base',125,15,(0,0,7.5),b.dark,vertices=64,bevel=1)
    # Base uses 32 sides: radial chord error is 0.602 mm (under 1 mm).
    bpy.data.objects.remove(b.groups['body'][1].pop(),do_unlink=True)
    b.groups['body'][1].append(p.cylinder('COL base',125,15,(0,0,7.5),vertices=32,bevel_mm=0))
    b.cyl('body','30 mm sign pole',15,1285,(0,0,657.5),b.steel)
    b.box('body','Aluminium sign',(2,300,220),(-17,0,1300),b.steel,bevel=.5)
    # Moulded border remains inside the specified 300 x 220 sign envelope.
    for y in (-147,147): b.box('body','Plastic border',(5,6,220),(-17,y,1300),b.dark,collision=False)
    for z in (1193,1407): b.box('body','Plastic border',(5,288,6),(-17,0,z),b.dark,collision=False)
    # Expand only the thin sign proxy to match the trim depth.
    for v in b.groups['body'][1][-1].data.vertices: v.co.x = -.017 + (v.co.x + .017) * 2.5
    mat,node=b.label('body','ok',(-19.6,0,1300),(286,206),brand='NOVA Logistics')
    obj=b.part('body',5,.5)
    b.data['variants']=[dict(name='ok',textures={'body':'body_albedo.png'})]
    for i,kind in enumerate(('recheck','ng'),1):
        path=b.folder/(kind+'_art.png')
        subprocess.run([str(p.VALIDATOR_PYTHON),'-B',str(Path(__file__)),'--artwork-worker'],input=json.dumps(dict(kind=kind,path=str(path))),text=True,check=True)
        node.image=bpy.data.images.load(str(path))
        tex=f'body_albedo_v{i}.png'; p.bake_albedo([obj],1024,b.folder/tex)
        p.render_preview([obj],b.folder/f'preview_{kind}.png')
        b.data['variants'].append(dict(name=kind,textures={'body':tex}))
    node.image=bpy.data.images.load(str(b.folder/'ok_art.png'))
    b.data['dimensions_mm']=dict(base_diameter=250,base_height=15,pole_diameter=30,sign_width=300,sign_height=220,aluminium_thickness=2,sign_center_z=1300)
    b.data['recommended_placement_m']={'note':'No placement specified in section 6.1; place outside robot base and arm paths, facing -X.'}
    b.data['notes']+='Opaque sign artwork has original check, magnifier and cross symbols, English DejaVu Sans Bold. Plastic border depth 5 mm is included in the sign collider. '
    b.finish()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--props',nargs='+',choices=IDS,default=list(IDS))
    parser.add_argument('--finalize-only',action='store_true',help='Repair and compact completed task-14 outputs without rebaking')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    builders=dict(zip(IDS,(build_cart,build_line,build_takeaway,build_induction,build_flow,build_shelf,build_sign)))
    if not args.finalize_only:
        for name in dict.fromkeys(args.props): builders[name]()
    completed = [name for name in args.props if (p.ROOT / 'assets' / 'props' / name / 'prop.json').is_file()]
    subprocess.run([str(p.VALIDATOR_PYTHON), '-B', str(HERE / 'check_task14_logistics.py'),
                    '--repair-collisions', '--props', *completed], check=True)
    p._DELIVERED_PROPS.update(completed)
    p.finalize_assets()


if __name__=='__main__':
    main()
