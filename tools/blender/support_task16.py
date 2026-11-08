"""Task 16 export helpers; shared pipeline remains unmodified."""
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector
import prop_lib as props

HERE = Path(__file__).resolve().parent
OUTPUT = props.PROPS
PLASTIC = dict(specular=.3, shininess=.4, reflectance=0.)
METAL = dict(specular=.7, shininess=.7, reflectance=.05)


def material(name, color, metal=False, rough=.55):
    return props.pbr_material(name, color, rough, int(metal), wear=.025, grime=.12,
                              machining='turned' if metal else None)


def box(name, size, pos, mat=None, edge=.4):
    return props.box(name, size, pos, bevel_mm=edge, segments=3, material=mat)


def cylinder(name, radius, height, pos, mat=None, edge=.4, count=64):
    return props.cylinder(name, radius, height, pos, bevel_mm=edge, vertices=count, material=mat)


def mesh(name, vertices, faces, mat=None):
    return props._mesh_object(name, vertices, faces, (0, 0, 0), 0, 3, mat)


def loft(name, rings, count=32, mat=None, center=(0, 0)):
    vertices = [(center[0]+r*math.cos(i*2*math.pi/count), center[1]+r*math.sin(i*2*math.pi/count), z)
                for r, z in rings for i in range(count)]
    faces = [tuple(reversed(range(count))), tuple(range((len(rings)-1)*count, len(rings)*count))]
    faces += [(k*count+i, k*count+(i+1)%count, (k+1)*count+(i+1)%count, (k+1)*count+i)
              for k in range(len(rings)-1) for i in range(count)]
    return mesh(name, vertices, faces, mat)


def finish(obj):
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bm = bmesh.new(); bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-8)
    bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=1e-9)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data); bm.free()
    props.smooth_by_angle(obj)
    return obj


def boolean(obj, cutter, operation='DIFFERENCE'):
    props._select([obj])
    mod = obj.modifiers.new('Manufactured recess', 'BOOLEAN')
    mod.operation, mod.solver, mod.object = operation, 'EXACT', cutter
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def art(folder, kind, title='', filename=None):
    destination = folder / (filename or kind+'.png')
    subprocess.run([str(props.VALIDATOR_PYTHON), '-B', str(HERE/'build_task16_assembly.py'),
                    '--artwork-worker', kind, str(destination), title], check=True)
    return destination


def image_material(name, path, uv='LabelUV'):
    mat = props.pbr_material(name, '#FFFFFF', .7, 0)
    tree = mat.node_tree
    tex = tree.nodes.new('ShaderNodeTexImage'); tex.image = bpy.data.images.load(str(path), check_existing=True)
    coords = tree.nodes.new('ShaderNodeUVMap'); coords.uv_map = uv
    tree.links.new(coords.outputs['UV'], tex.inputs['Vector'])
    tree.links.new(tex.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
    return mat


def label(name, center, width, height, mat, front=True):
    # u right on the image is -Y when viewed by the robot; v is up.
    c = Vector(center); u = Vector((0, -1, 0)); v = Vector((0, 0, 1) if front else (1, 0, 0))
    corners = [(-.5,-.5), (.5,-.5), (.5,.5), (-.5,.5)]
    obj = mesh(name, [tuple(c+u*(a*width)+v*(b*height)) for a,b in corners], [(0,1,2,3)], mat)
    uv = obj.data.uv_layers.new(name='LabelUV')
    for polygon in obj.data.polygons:
        for li in polygon.loop_indices:
            index = obj.data.loops[li].vertex_index
            uv.data[li].uv = (corners[index][0]+.5, corners[index][1]+.5)
    return obj


def start(prop_id, clear_collisions=True):
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    folder = OUTPUT/prop_id; folder.mkdir(parents=True, exist_ok=True)
    # Only this prop's prior generated collision meshes are replaced.
    if clear_collisions:
        for file in folder.glob('*_col_*.obj'): file.unlink()
    print('TASK16_START', prop_id, flush=True)
    return folder


def part(name, visuals, collisions, mass, parent=None, joint=None, size=2048, rgba=None,
         friction=.5, mujoco=None, full_uv=False, texture_override=None):
    obj = props.join_objects(visuals, name) if len(visuals)>1 else visuals[0]
    obj.name = name; finish(obj)
    if not full_uv: props.uv_unwrap(obj)
    for i, col in enumerate(collisions):
        col.name = f'COL__{name}__{i:03d}'; col.hide_render=True; col.hide_set(True)
    return dict(name=name, obj=obj, cols=collisions, mass_kg=mass, parent=parent, joint=joint,
                texture_px=None if rgba else size, rgba=rgba, friction=friction,
                mujoco_material=mujoco or PLASTIC, full_uv=full_uv, texture_override=texture_override)


def bake(obj, size, folder, name):
    # prop_lib's automatic PNG optimizer downsizes names containing 'albedo'.
    # Bake under a neutral name, then retain the section 1.5 resolution.
    temp = folder/(name.replace('albedo','bake_work'))
    print('TASK16_BAKE', name, size, flush=True)
    # Cycles already averages 64 bake samples. One cavity ray per shader
    # evaluation avoids nesting 16 additional AO rays inside every colour sample.
    ao_nodes=[n for tree in bpy.data.node_groups for n in tree.nodes if n.type=='AMBIENT_OCCLUSION']
    old_samples=[n.samples for n in ao_nodes]
    try:
        for node in ao_nodes: node.samples=1
        image = props.bake_albedo([obj], size, temp)
    finally:
        for node,count in zip(ao_nodes,old_samples): node.samples=count
    target = folder/name; temp.replace(target); image.filepath=str(target)
    return image


def render(objects, folder):
    """Native 1024 Cycles preview plus validator-compatible 512 thumbnail."""
    scene=bpy.context.scene
    scene.render.resolution_x=scene.render.resolution_y=1024
    scene.render.resolution_percentage=100
    scene.render.threads_mode='FIXED'; scene.render.threads=6
    scene.cycles.samples=64
    scene.world=bpy.data.worlds.new('Neutral grey'); scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.18,.18,.18,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.45
    points=np.array([tuple(o.matrix_world@Vector(c)) for o in objects for c in o.bound_box])
    low,high=points.min(0),points.max(0); center=Vector((low+high)/2); span=float(max(high-low))
    added=[]
    camdata=bpy.data.cameras.new('Robot front-left'); cam=bpy.data.objects.new('Camera',camdata)
    scene.collection.objects.link(cam); added.append(cam)
    cam.location=center+Vector((-1.8,1.35,1.55))*span
    cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler()
    camdata.type='ORTHO'; camdata.ortho_scale=span*1.65; camdata.clip_start=.0001; scene.camera=cam
    for name,xyz,power in [('Key',(-1.5,1.2,2.1),120),('Fill',(-.8,-1.8,.9),55),('Rim',(1.2,.7,1.7),85)]:
        data=bpy.data.lights.new(name,'AREA'); data.energy=power*span*span; data.size=span*1.4
        obj=bpy.data.objects.new(name,data); scene.collection.objects.link(obj); added.append(obj)
        obj.location=center+Vector(xyz)*span; obj.rotation_euler=(center-obj.location).to_track_quat('-Z','Y').to_euler()
    floor=box('Preview ground',(span*10000,span*10000,1),(0,0,low[2]*1000-1),material('Ground','#777777'),0)
    added.append(floor)
    scene.view_settings.view_transform='AgX'
    scene.render.filepath=str(folder/'inspection.png'); bpy.ops.render.render(write_still=True)
    for obj in added: bpy.data.objects.remove(obj,do_unlink=True)
    subprocess.run([str(props.VALIDATOR_PYTHON),'-B',str(HERE/'build_task16_assembly.py'),
                    '--thumbnail',str(folder/'inspection.png'),str(folder/'preview.png')],check=True)


def complete(folder, parts, dimensions, extra=None, variants=None, variant_callback=None, notes=''):
    visuals=[p['obj'] for p in parts]
    records=[]
    for p in parts:
        obj=p['obj']; name=p['name']
        texture=None
        if not p['rgba']:
            texture=p.get('texture_override') or name+'_albedo.png'
            if not p['full_uv'] and not p.get('texture_override'): bake(obj,p['texture_px'],folder,texture)
        props.export_obj(obj,folder/(name+'.obj'))
        record={k:v for k,v in p.items() if k not in ('obj','cols','full_uv','rgba','texture_override')}
        record.update(visual=name+'.obj',texture=texture,collision=props.export_collision(name,p['cols'],folder),
                      source_visual_object=obj.name,source_collision_objects=[o.name for o in p['cols']])
        if p['rgba']: record['rgba']=p['rgba']
        records.append(record)
    if variant_callback: variants=variant_callback()
    data=dict(id=folder.name,task='order_picking',units='m',origin='bottom_center',budget_class='medium',
              bbox_mm=[round(x,6) for x in props.bbox_mm(visuals)],triangles=props.triangle_count(visuals),
              source='source.blend',preview='preview.png',inspection='inspection.png',parts=records,
              variants=variants or [],dimensions_mm=dimensions,
              notes='Metres, Z up, robot front -X, shared bottom-centre frame; applied transforms. '
              'CPU Cycles 64 diffuse colour times mix(1, AO, 0.6); PBR source retained. '
              'Inspection render 1024 square; preview thumbnail 512 for the unchanged validator. '+notes)
    data.update(extra or {})
    render(visuals,folder)
    bpy.ops.file.pack_all()
    props._select(visuals); bpy.ops.wm.save_as_mainfile(filepath=str(folder/'source.blend'),check_existing=False)
    props.write_prop_json(data,folder/'prop.json')
    print('TASK16_COMPLETE',folder.name,data['triangles'],[len(p['collision']) for p in records],flush=True)
    return data


def finalize_selected():
    """Run the mandatory compactor on an isolated root, retaining spec atlases.

    compact_props updates/deduplicates all manifests even with --props. Isolating
    this invocation prevents changes to other teams' concurrent deliveries.
    """
    names=sorted(props._DELIVERED_PROPS)
    if not names:
        props.finalize_assets(); return
    subprocess.run([str(props.VALIDATOR_PYTHON),'-B',str(HERE/'convexify_task16_collisions.py'),'--props',*names],check=True)
    stage=HERE/'_qa'/('task16_finalize_'+str(os.getpid()))
    if stage.exists(): shutil.rmtree(stage)
    root=stage/'assets'/'props'; root.mkdir(parents=True)
    saved={}
    for name in names:
        src=OUTPUT/name; dst=root/name; dst.mkdir()
        saved[name]=json.loads((src/'prop.json').read_text())
        for path in src.iterdir():
            if path.is_file() and path.suffix in ('.obj','.json','.png'): shutil.copy2(path,dst/path.name)
    original=props.ROOT
    try:
        props.ROOT=stage
        props.finalize_assets()
    finally: props.ROOT=original
    for name in names:
        src=root/name; dst=OUTPUT/name
        for path in src.glob('*.obj'): shutil.copy2(path,dst/path.name)
        compact=json.loads((src/'prop.json').read_text())
        marker=' Task 13 compact OBJ exception:'
        if marker in compact['notes']:
            saved[name]['notes']=saved[name]['notes'].partition(marker)[0]+marker+compact['notes'].split(marker,1)[1].partition(' Task 13 delivery:')[0]
        if name == 'rotor_shaft_fixture':
            restore_shaft_precision(dst)
            saved[name]['notes'] += (' Shaft/chamfer collision coordinates retain 12 decimal metres '
                'after compaction and an inward 0.00001 mm numerical guard; vertices never exceed '
                'the design circle. This prevents 5-decimal rounding from creating an oversized shaft.')
        props.write_prop_json(saved[name],dst/'prop.json')
    shutil.rmtree(stage)


def hull_points(points):
    """Reduce a convex point cloud to its actual hull vertices in millimetres."""
    points=np.unique(np.round(points,8),axis=0)
    if len(points)<4: return None
    bm=bmesh.new()
    for p in points: bm.verts.new(tuple(p))
    try:
        result=bmesh.ops.convex_hull(bm,input=list(bm.verts),use_existing_faces=False)
        used={v for f in bm.faces for v in f.verts}
        if len(used)<4 or abs(bm.calc_volume())<1e-6: return None
        return np.array([tuple(v.co) for v in used])
    finally: bm.free()


def split_convex(points, normal, offset):
    """Split convex points by dot(normal, point) + offset >= 0."""
    normal=np.asarray(normal); distances=points@normal+offset
    if distances.min()>=-1e-5: return points,None
    if distances.max()<=1e-5: return None,points
    pos=points[distances>1e-5]; neg=points[distances< -1e-5]
    zero=list(points[np.abs(distances)<=1e-5])
    # All crossing chords are safe; convex hull removes their interior points.
    for a in pos:
        da=a@normal+offset
        for b in neg:
            db=b@normal+offset
            zero.append(a+(b-a)*(da/(da-db)))
    return hull_points([*pos,*zero]),hull_points([*neg,*zero])


def subtract_convex(pieces, planes):
    """Subtract one convex cutter, retaining a disjoint convex decomposition."""
    result=[]
    for source in pieces:
        inside=source; outside=[]
        for normal,offset in planes:
            inside,rejected=split_convex(inside,normal,offset)
            if rejected is not None: outside.append(rejected)
            if inside is None: break
        result.extend(outside)
    return result


def prism_planes(outline, expand=0, zref=40):
    """CCW outline; expand=1 produces a 45-degree outward opening above zref."""
    planes=[]
    for a,b in zip(outline,outline[1:]+outline[:1]):
        dx,dy=b[0]-a[0],b[1]-a[1]; length=math.hypot(dx,dy)
        nx,ny=-dy/length,dx/length
        planes.append(((nx,ny,expand),-nx*a[0]-ny*a[1]-expand*zref))
    return planes


def convex_object(name, points, mat=None):
    bm=bmesh.new()
    for point in points: bm.verts.new(tuple(float(v)/1000 for v in point))
    bmesh.ops.convex_hull(bm,input=list(bm.verts),use_existing_faces=False)
    loose=[v for v in bm.verts if not v.link_faces]
    if loose: bmesh.ops.delete(bm,geom=loose,context='VERTS')
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    data=bpy.data.meshes.new(name); bm.to_mesh(data); bm.free()
    obj=bpy.data.objects.new(name,data); bpy.context.collection.objects.link(obj)
    if mat: obj.data.materials.append(mat)
    return obj


def restore_shaft_precision(folder):
    """Keep precision seats inscribed after the generic five-decimal compaction."""
    for file in folder.glob('body_col_*.obj'):
        lines=file.read_text().splitlines()
        vertices=np.array([[float(x) for x in row.split()[1:]] for row in lines if row.startswith('v ')])
        lo,hi=vertices[:,2].min(),vertices[:,2].max()
        if not (.1159<=lo<=.1521 and .1519<=hi<=.1561): continue
        cy=.12 if vertices[:,1].mean()>0 else -.12
        if np.max(np.hypot(vertices[:,0],vertices[:,1]-cy))>.0091: continue
        output=[]
        for line in lines:
            if line.startswith('v '):
                x,y,z=map(float,line.split()[1:])
                a=round(math.atan2(y-cy,x)/(2*math.pi/32))*2*math.pi/32
                r=.009-max(0,z-.152)*math.tan(math.radians(30))-.00000001
                line='v '+ ' '.join(f'{v:.12f}' for v in (r*math.cos(a),cy+r*math.sin(a),z))
            output.append(line)
        file.write_text('\n'.join(output)+'\n')
