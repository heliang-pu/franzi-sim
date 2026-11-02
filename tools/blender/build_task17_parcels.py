"""Task 17 parcels and infeed table; geometry arguments use millimetres.

blender -b --factory-startup -t 6 -P tools/blender/build_task17_parcels.py --
Artwork runs under PROP_PYTHON. Five texture variants share one parcel mesh.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
OUTPUT = ROOT / 'assets/props'
QA = HERE / '_qa/task17'
PARCELS = {'parcel_cube_80': (80, 70, 60), 'parcel_cube_75': (75, 75, 75),
           'parcel_cube_85': (85, 65, 55)}
FACES = ('top', 'px', 'nx', 'py', 'ny')
NORMALS = {'top': [0, 0, 1], 'px': [1, 0, 0], 'nx': [-1, 0, 0],
           'py': [0, 1, 0], 'ny': [0, -1, 0], 'bottom': [0, 0, -1]}


def artwork_worker() -> None:
    """Draw physical-size face art, using only bundled DejaVu Sans fonts."""
    import random
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    request = json.load(sys.stdin)
    folder = Path(request['folder'])
    x, y, z = request['dimensions']
    scale = 16
    def font(size, bold=False):
        return ImageFont.truetype(str(HERE / 'fonts' / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), size)
    label = Image.new('RGBA', (56 * scale, 46 * scale), '#FAFAF5')
    d = ImageDraw.Draw(label)
    ink = '#17232A'
    d.rectangle((2, 2, 894, 734), outline='#D0D0C8', width=3)
    d.polygon([(0, 0), (15*scale, 0), (0, 15*scale)], fill='#EF6C00')
    d.text((275, 25), 'QF Express', font=font(62, True), fill=ink)
    d.text((272, 103), 'TRACK  QF 1700 8042', font=font(30, True), fill=ink)
    d.text((38, 196), 'FROM: NOVA / 12 FOUNDRY WAY', font=font(29), fill=ink)
    d.text((38, 241), 'TO: A. REED / 8 CIRCUIT ROAD', font=font(29), fill=ink)
    d.text((38, 288), 'NORTH DEPOT     ROUTE 17', font=font(29, True), fill=ink)
    # The barcode region fills the lower half of the physical 56 x 46 mm label.
    rng = random.Random('QF17008042')
    xx = 35
    while xx < 858:
        width = rng.choice((3, 5, 7))
        d.rectangle((xx, 367, min(xx+width, 860), 700), fill=ink)
        xx += width + rng.choice((3, 4, 6))
    label.save(folder / 'qf_label_source.png')
    label = label.rotate(2, Image.Resampling.BICUBIC, expand=True)
    for face in (*FACES, 'bottom'):
        width, height = (y, x) if face in ('top', 'bottom') else (y, z) if face in ('px', 'nx') else (x, z)
        size = (round(width*scale), round(height*scale))
        rng_np = np.random.default_rng(1700 + int(width) + int(height))
        colors = np.array([184, 139, 92], dtype=float)
        noise = rng_np.normal(0, 1.2, (size[1], size[0], 1))
        image = Image.fromarray(np.uint8(np.clip(colors + noise, 0, 255)), 'RGB')
        draw = ImageDraw.Draw(image)
        rough = Image.new('L', size, 217)
        rd = ImageDraw.Draw(rough)
        if face in ('top', 'px', 'nx'):
            # Top image horizontal is -Y; vertical is X. End images horizontal is Y.
            left, right = (width/2-24)*scale, (width/2+24)*scale
            bottom = size[1] if face == 'top' else 30*scale
            draw.rectangle((left, 0, right, bottom), fill='#8B5A2B')
            rd.rectangle((left, 0, right, bottom), fill=77)
            draw.line((width*scale/2, 0, width*scale/2, bottom), fill='#794D28', width=3)
            for i in range(9):
                yy = (i+1)*bottom/10
                start = left+19+(i*37)%160
                draw.line((start, yy, start+140, yy+12), fill='#A06C3A', width=2)
            bubble = (left+80, bottom*.86, left+160, bottom*.86+28)
            draw.ellipse(bubble, fill='#966532', outline='#AF7C47', width=2)
            rd.ellipse(bubble, fill=90)
        if face in ('py', 'ny'):
            for xx in (7*scale, 14*scale):
                draw.line((xx, (height-8)*scale, xx, (height-20)*scale), fill='#5E432B', width=8)
                draw.polygon([(xx, (height-22)*scale), (xx-3*scale, (height-17)*scale),
                              (xx+3*scale, (height-17)*scale)], fill='#5E432B')
            draw.text((22*scale, (height-17)*scale), 'QF', font=font(42, True), fill='#63472F')
        for marked in (False, True):
            result, surface = image.copy(), rough.copy()
            if marked:
                offset = ((size[0]-label.width)//2, (size[1]-label.height)//2)
                result.paste(label, offset, label)
                surface.paste(Image.new('L', label.size, 199), offset, label.getchannel('A'))
            suffix = 'label' if marked else 'plain'
            result.save(folder / f'{face}_{suffix}_source.png')
            surface.save(folder / f'{face}_{suffix}_roughness.png')


def compact_artwork_worker() -> None:
    """Keep editable artwork small; the build regenerates full-resolution inputs."""
    from PIL import Image
    folder = Path(json.load(sys.stdin)['folder'])
    for path in sorted(folder.glob('*.png')):
        if not path.name.endswith(('_source.png', '_roughness.png')):
            continue
        with Image.open(path) as source:
            image = source.copy()
        limit = 1024 if path.name == 'qf_label_source.png' else 256
        image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        image.save(path, optimize=True)


if '--compact-artwork-worker' in sys.argv:
    compact_artwork_worker()
    raise SystemExit(0)


if '--artwork-worker' in sys.argv:
    artwork_worker()
    raise SystemExit(0)

import bpy
from mathutils import Matrix, Vector
import prop_lib as props
from build_task05_carton_carry import boolean, bevel, combine, grain, image_material, wave_patch

PAPER = {'specular': .05, 'shininess': .10, 'reflectance': 0.0}
PAINT = {'specular': .30, 'shininess': .30, 'reflectance': 0.0}


def render_1024(objects: list, path: Path, *, back: bool = False) -> None:
    """Render at 1024 before compaction; shared helper fixes resolution at 512."""
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    low = Vector([min(p[i] for p in points) for i in range(3)])
    high = Vector([max(p[i] for p in points) for i in range(3)])
    center, span = (low+high)/2, max(high-low)
    scene = bpy.data.scenes.new('Task 17 1024 review')
    created = []
    try:
        scene.render.engine = 'CYCLES'
        scene.cycles.device, scene.cycles.samples, scene.cycles.seed = 'CPU', 64, 0
        scene.cycles.use_adaptive_sampling = False
        scene.render.threads_mode, scene.render.threads = 'FIXED', 6
        scene.render.resolution_x = scene.render.resolution_y = 1024
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode, scene.render.image_settings.color_depth = 'RGB', '8'
        scene.view_settings.view_transform, scene.view_settings.look = 'AgX', 'None'
        scene.world = bpy.data.worlds.new('Task 17 neutral grey')
        scene.world.use_nodes = True
        scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.18,.18,.18,1)
        scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .45
        for obj in objects:
            scene.collection.objects.link(obj)
        camera_data = bpy.data.cameras.new('Task 17 review camera')
        camera = bpy.data.objects.new('Task 17 review camera',camera_data)
        scene.collection.objects.link(camera)
        created.append(camera)
        camera.location = center + Vector((1.8,-1.35,1.25) if back else (-1.8,1.35,1.25))*span
        camera.rotation_euler = (center-camera.location).to_track_quat('-Z','Y').to_euler()
        camera_data.type, camera_data.ortho_scale = 'ORTHO', span*1.95
        camera_data.clip_start, camera_data.clip_end = max(span*.001,.0001),span*30
        scene.camera = camera
        for name, offset, power, size in [('Key',(-1.5,1.2,2.1),120,1.4),
                                         ('Fill',(-.8,-1.8,.9),55,1.6),('Rim',(1.2,.7,1.7),85,1.1)]:
            light_data = bpy.data.lights.new('Task 17 '+name,'AREA')
            light_data.energy,light_data.shape,light_data.size = power*span*span,'DISK',size*span
            light = bpy.data.objects.new('Task 17 '+name,light_data)
            scene.collection.objects.link(light)
            created.append(light)
            light.location = center+Vector(offset)*span
            light.rotation_euler = (center-light.location).to_track_quat('-Z','Y').to_euler()
        mesh = bpy.data.meshes.new('Task 17 review floor')
        mesh.from_pydata([(x*span,y*span,low.z-span*.003) for x,y in [(-5,-5),(5,-5),(5,5),(-5,5)]],[],[(0,1,2,3)])
        floor = bpy.data.objects.new('Task 17 review floor',mesh)
        floor.data.materials.append(props.pbr_material('Task 17 grey floor','#777777',.8,0))
        scene.collection.objects.link(floor)
        created.append(floor)
        path.parent.mkdir(parents=True,exist_ok=True)
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True,scene=scene.name)
        props._optimize_png(path)
        if path.parent.parent == OUTPUT:
            props._DELIVERED_PROPS.add(path.parent.name)
            QA.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,QA/f'{path.parent.name}_{path.stem}_1024.png')
    finally:
        world = scene.world
        for obj in created:
            data = obj.data
            bpy.data.objects.remove(obj,do_unlink=True)
            if data.users == 0:
                if isinstance(data,bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                elif isinstance(data,bpy.types.Camera):
                    bpy.data.cameras.remove(data)
                elif isinstance(data,bpy.types.Light):
                    bpy.data.lights.remove(data)
        bpy.data.scenes.remove(scene)
        bpy.data.worlds.remove(world)


def start(prop_id: str) -> Path:
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    bpy.context.scene.render.threads_mode = 'FIXED'
    bpy.context.scene.render.threads = 6
    folder = OUTPUT / prop_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def face_materials(folder: Path) -> dict:
    result = {}
    for face in NORMALS:
        material = image_material('QF ' + face, folder / f'{face}_plain_source.png', 'FaceUV')
        tree = material.node_tree
        color = next(n for n in tree.nodes if n.type == 'TEX_IMAGE')
        color.name = 'Face artwork'
        rough = tree.nodes.new('ShaderNodeTexImage')
        rough.name = 'Face roughness'
        rough.image = bpy.data.images.load(str(folder / f'{face}_plain_roughness.png'), check_existing=True)
        rough.image.colorspace_settings.name = 'Non-Color'
        uv = next(n for n in tree.nodes if n.type == 'UVMAP')
        tree.links.new(uv.outputs['UV'], rough.inputs['Vector'])
        tree.links.new(rough.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Roughness'])
        grain(material, 3900, .035, .025)
        result[face] = material
    return result


def map_faces(obj: bpy.types.Object, dimensions: tuple, materials: dict) -> None:
    x, y, z = dimensions
    indices = {}
    for face, material in materials.items():
        indices[face] = len(obj.data.materials)
        obj.data.materials.append(material)
    uv = obj.data.uv_layers.get('FaceUV') or obj.data.uv_layers.new(name='FaceUV')
    for polygon in obj.data.polygons:
        axis = max(range(3), key=lambda i: abs(polygon.normal[i]))
        sign = 1 if polygon.normal[axis] > 0 else -1
        face = ('px' if sign > 0 else 'nx') if axis == 0 else ('py' if sign > 0 else 'ny') if axis == 1 else ('top' if sign > 0 else 'bottom')
        # Internal faces retain the unprinted kraft liner.
        outer = abs(polygon.center[axis]*1000 - (sign*dimensions[axis]/2 if axis < 2 else z if sign > 0 else 0)) < 1.2
        if outer:
            polygon.material_index = indices[face]
        for index in polygon.loop_indices:
            p = obj.data.vertices[obj.data.loops[index].vertex_index].co * 1000
            if axis == 0:
                coord = (.5 + sign*p.y/y, p.z/z)
            elif axis == 1:
                coord = (.5 - sign*p.x/x, p.z/z)
            else:
                coord = (.5-p.y/y, .5+sign*p.x/x)
            uv.data[index].uv = coord


def set_label_face(materials: dict, folder: Path, label_face: str) -> None:
    for face, material in materials.items():
        suffix = 'label' if face == label_face else 'plain'
        for name, filename in [('Face artwork', f'{face}_{suffix}_source.png'),
                               ('Face roughness', f'{face}_{suffix}_roughness.png')]:
            image = bpy.data.images.load(str(folder / filename), check_existing=True)
            image.use_fake_user = True
            if name == 'Face roughness':
                image.colorspace_settings.name = 'Non-Color'
            material.node_tree.nodes[name].image = image


def save_compact_source(folder: Path) -> None:
    """Save PBR geometry with relative external art and no packed bake images."""
    subprocess.run([str(props.VALIDATOR_PYTHON), '-B', str(Path(__file__)),
                    '--compact-artwork-worker'],
                   input=json.dumps({'folder': str(folder)}), text=True, check=True)
    # Variant switching leaves fake users on artwork from previous builds.
    for image in bpy.data.images:
        image.use_fake_user = False
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    for image in list(bpy.data.images):
        filename = Path(image.filepath).name
        if not filename.endswith(('_source.png', '_roughness.png')):
            bpy.data.images.remove(image)
            continue
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(f'Missing PBR source artwork: {path}')
        if image.packed_file:
            image.unpack(method='REMOVE')
        image.filepath = str(path)
        image.reload()
        image.filepath = '//' + filename
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / 'source.blend'),
                               check_existing=False, compress=True, relative_remap=False)


def save_prop(folder: Path, obj: bpy.types.Object, collisions: list, *, mass: float, friction: float,
              dimensions: dict, variants: list, static: bool, notes: str) -> None:
    collision_files = props.export_collision('body', collisions, folder)
    for collider in collisions:
        collider.hide_render = True
        collider.hide_set(True)
    props.export_obj(obj, folder / 'body.obj')
    data = {'id': folder.name, 'task': 'parcel_induction', 'units': 'm', 'origin': 'bottom_center',
            'bbox_mm': [round(v, 5) for v in props.bbox_mm([obj])], 'triangles': props.triangle_count([obj]),
            'budget_class': 'medium' if static else 'small', 'static': static,
            'source': 'source.blend', 'preview': 'preview.png', 'dimensions_mm': dimensions,
            'recommended_placement_m': [{'origin_x': .40, 'origin_y': sy*.36, 'support_z': 0 if static else .90} for sy in (1, -1)],
            'parts': [{'name': 'body', 'visual': 'body.obj', 'texture': 'body_albedo.png',
                       'texture_px': 1024, 'collision': collision_files, 'mass_kg': mass, 'friction': friction,
                       'mujoco_material': PAINT if static else PAPER, 'parent': None, 'joint': None}],
            'variants': variants, 'buttons': [], 'lamps': [],
            'notes': notes + ' Metres, Z up, front -X; applied transforms and shared bottom-centre origin. '
            '64-sample CPU Cycles colour multiplied by mix(1, AO, 0.6), RGB8 sRGB. '
            'PBR source retains separate roughness, fine grain and edge wear. Instance color multiplier 0.97-1.03. '
            '1024 px review renders retained in tools/blender/_qa/task17; mandatory compactor delivers 512 px previews.'}
    props.write_prop_json(data, folder / 'prop.json')
    bpy.context.scene['task17_manifest'] = json.dumps(data)
    bpy.context.scene['bake_recipe'] = 'DIFFUSE COLOR * (0.4 + 0.6 * AO)'
    props._select([obj])
    save_compact_source(folder)
    print('TASK17_COMPLETE', folder.name, data['bbox_mm'], data['triangles'], flush=True)


def build_parcel(prop_id: str) -> None:
    folder = start(prop_id)
    x, y, z = PARCELS[prop_id]
    subprocess.run([str(props.VALIDATOR_PYTHON), '-B', str(__file__), '--artwork-worker'],
                   input=json.dumps({'folder': str(folder), 'dimensions': [x, y, z]}), text=True, check=True)
    kraft = grain(props.pbr_material('3 mm single-wall kraft liner', '#B88B5C', .85, 0, grime=.14), 3900, .06, .025)
    medium = props.pbr_material('Exposed single corrugated medium', '#C19A68', .92, 0)
    core_x, top = x-.30, z-.15
    shell = props.box('Closed 3 mm corrugated shell', (core_x, y, top), (0, 0, top/2), bevel_mm=0, material=kraft)
    boolean(shell, props.box('Internal air', (core_x-6, y-6, top-6), (0, 0, top/2), bevel_mm=0))
    boolean(shell, props.box('Top major flap seam along X', (core_x-6, .7, 3.5), (0, 0, top-1.5), bevel_mm=0))
    for sx in (-1, 1):
        for sy in (-1, 1):
            boolean(shell, props.cylinder('Shallow crushed corner', 1.5, top*.45,
                (sx*(core_x/2+.95), sy*(y/2-.45), top*.63), vertices=16, bevel_mm=0))
    patches = []
    for sy in (-1, 1):
        lo, hi = -core_x/2+5, -core_x/2+18
        boolean(shell, props.box('Exposed cut-edge channel', (hi-lo, .8, 2.2),
                                 ((lo+hi)/2, sy*y/2, top-1.5), bevel_mm=0))
        wave = wave_patch('Visible corrugated flap edge', lo, hi, sy*(y/2-.27), top, medium)
        for v in wave.data.vertices:
            v.co.z = props.mm(top) + .75*(v.co.z-props.mm(top))
        patches.append(wave)
    bevel(shell, .35)
    half, thickness = core_x/2, .15
    profile = [(-half-thickness, top-30), (-half-thickness, top+thickness),
               (half+thickness, top+thickness), (half+thickness, top-30),
               (half, top-30), (half, top), (-half, top), (-half, top-30)]
    tape = props.extruded_profile('48 mm sealing tape, two 30 mm folds', profile, 48,
                                  bevel_mm=.04, material=kraft)
    tape.data.transform(Matrix(((1,0,0,0),(0,0,1,0),(0,1,0,0),(0,0,0,1))))
    # Swapping Y/Z reverses handedness; combine() recalculates normals.
    body = combine([shell, tape, *patches], 'body')
    materials = face_materials(folder)
    map_faces(body, (x,y,z), materials)
    props.uv_unwrap(body)
    for uv in body.data.uv_layers['UVMap'].data:
        uv.uv = (.0005+.999*uv.uv.x, .0005+.999*uv.uv.y)
    variants = []
    for i, face in enumerate(FACES):
        set_label_face(materials, folder, face)
        texture = 'body_albedo.png' if i == 0 else f'body_albedo_v{i}.png'
        preview = 'preview.png' if i == 0 else f'preview_{face}.png'
        props.bake_albedo([body], 1024, folder / texture)
        render_1024([body], folder / preview)
        variants.append({'name': face, 'textures': {'body': texture}, 'label_face': face,
                         'label_normal': NORMALS[face], 'preview': preview})
    set_label_face(materials, folder, 'top')
    collider = props.box('COL nominal parcel', (x,y,z), (0,0,z/2), bevel_mm=0)
    save_prop(folder, body, [collider], mass=.12, friction=.6,
              dimensions={'nominal_xyz': [x,y,z], 'wall': 3, 'label_wh': [56,46],
                          'label_skew_deg': 2, 'orange_corner_leg': 15, 'tape_width': 48,
                          'tape_end_fold': 30, 'tape_axis': 'X', 'flap_seam': .7},
              variants=variants, static=False,
              notes='Five texture-only face variants with exactly one centred 56x46 mm QF Express label; '
              'orange 15 mm corner and lower-half barcode. Real hollow 3 mm shell, compressed corners, '
              'flap gap, exposed flutes and 48 mm tape folded 30 mm on both X ends. '
              'Fine tape wrinkles and shallow air bubbles are colour details; label and tape gloss use roughness maps. '
              'The single nominal solid-box collider intentionally ignores the sealed interior.')


def build_table() -> None:
    folder = start('infeed_table')
    steel = grain(props.pbr_material('NOVA grey powder-coated steel', '#59636C', .6, 0, wear=.12, grime=.15), 1600, .05, .02)
    hpl = grain(props.pbr_material('Grey HPL laminated top', '#969A9C', .65, 0, wear=.06, grime=.12), 2800, .045, .015)
    objects = [props.box('30 mm grey HPL tabletop', (300,280,30), (0,0,885), bevel_mm=1, material=hpl)]
    collisions = [props.box('COL tabletop', (300,280,30), (0,0,885), bevel_mm=0)]
    for xx in (-120,120):
        for yy in (-110,110):
            objects.append(props.box('40 mm square steel leg', (40,40,870), (xx,yy,435), bevel_mm=1, material=steel))
            collisions.append(props.box('COL square leg', (40,40,870), (xx,yy,435), bevel_mm=0))
    for yy in (-110,110):
        objects.append(props.box('Lower X crossbrace', (200,25,30), (0,yy,240), bevel_mm=1, material=steel))
        objects.append(props.box('Upper X apron', (200,25,40), (0,yy,842), bevel_mm=1, material=steel))
    for xx in (-120,120):
        objects.append(props.box('Lower Y crossbrace', (25,180,30), (xx,0,240), bevel_mm=1, material=steel))
        objects.append(props.box('Upper Y apron', (25,180,40), (xx,0,842), bevel_mm=1, material=steel))
    body = combine(objects, 'body')
    props.uv_unwrap(body)
    for collider in collisions:
        collider.hide_render = True
        collider.hide_set(True)
    props.bake_albedo([body], 2048, folder / 'body_albedo.png')
    render_1024([body], folder / 'preview.png')
    save_prop(folder, body, collisions, mass=15, friction=.5,
              dimensions={'tabletop_xyz': [300,280,30], 'top_surface_z': 900,
                          'leg_xy': [40,40], 'leg_centers_xy': [[x,y] for x in (-120,120) for y in (-110,110)],
                          'leg_footprint_xy': [280,260], 'crossbrace_center_z': 240},
              variants=[], static=True,
              notes='Static 15 kg infeed table; four inset 40x40 steel legs and crossbraces. '
              'Exactly five specified box colliders; braces and aprons are visual only. '
              'At the recommended placements the closest leg face is world X=0.26 m, outside the robot X<=0.24 m footprint. '
              'The 2048 px HPL/steel bake is downsampled to 1024 by the mandatory shared compactor. '
              'Friction 0.5 is an estimate; the specification gives no table friction.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--props', nargs='+', choices=[*PARCELS,'infeed_table'], default=[*PARCELS,'infeed_table'])
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    for prop_id in args.props:
        if prop_id == 'infeed_table':
            build_table()
        else:
            build_parcel(prop_id)
    props.finalize_assets()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
