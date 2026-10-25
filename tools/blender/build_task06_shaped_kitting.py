"""Build benchmark_props.md sections 1, 2 and 3.6, and only Task 06 assets.

Blender -b --factory-startup -P tools/blender/build_task06_shaped_kitting.py
        -- [--props hex_blank flanged_bushing l_bracket spur_gear foam_kit_tray]
Geometry is specified in millimetres; prop_lib exports metres in the prop frame.
--geometry-only writes inspection blends under tools/blender/_qa/task06/.
--tray-materials-only updates the existing tray's variants/previews, preserving
its exported geometry, UVs, collisions and original black albedo byte for byte.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bmesh
import bpy
from mathutils import Matrix, Vector

import prop_lib as props

IDS = ("hex_blank", "flanged_bushing", "l_bracket", "spur_gear", "foam_kit_tray")
OUTPUT = props.ROOT / "assets" / "props"
QA = HERE / "_qa" / "task06"
PREFLIGHT = []
GEOMETRY_ONLY = False
STEEL = {"specular": .50, "shininess": .50, "reflectance": .02}
BRONZE = {"specular": .60, "shininess": .60, "reflectance": .03}
ZINC = {"specular": .70, "shininess": .70, "reflectance": .05}
POM = {"specular": .30, "shininess": .40, "reflectance": 0.0}
EVA = {"specular": .02, "shininess": .05, "reflectance": 0.0}
TRAY_LIGHT_NOTES = (
    "Light-grey EVA variant: #A9AFB6 with the original fine foam-cell noise/bump, "
    "CPU Cycles 64-sample colour-only albedo * (0.2 + 0.8 * AO); pocket floors "
    "receive a subtle 0.80 linear colour multiplier. No direct light or highlights "
    "are baked. Black remains the default part texture and source material; both "
    "variants share exactly the same visual mesh, UVs, collisions and physics. "
    "Preview uses the exported light-grey atlas. Robot-camera QA is 640 x 480, "
    "0.6 m above the foam top, 30 degrees from vertical (0.693 m slant range), "
    "with the four existing parts beside the empty tray."
)
COMMON_NOTES = (
    "Metres, Z up, front -X, bottom-centre origin; all transforms applied. "
    "Triangulated visual OBJ with normals and a nonoverlapping Smart UV 66-degree atlas. "
    "Albedo is CPU Cycles 64-sample DIFFUSE COLOR * (0.4 + 0.6 * AO), saved as 8-bit sRGB; "
    "no direct illumination or highlights. Procedural Principled PBR retained in source.blend. "
    "Preview: CPU Cycles 64, 512 square, neutral grey and three lights, front-left above. "
)


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def area(polygon):
    return sum(cross(a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1])) / 2


def ccw(polygon):
    return polygon if area(polygon) > 0 else polygon[::-1]


def regular_polygon(radius, count, phase=0, center=(0, 0)):
    return [(center[0] + radius * math.cos(phase + 2 * math.pi * i / count),
             center[1] + radius * math.sin(phase + 2 * math.pi * i / count)) for i in range(count)]


def offset_polygon(polygon, distance):
    """Mitered parallel offset, preserving the vertex correspondence for lofts."""
    polygon = ccw(polygon)
    result = []
    for a, b, c in zip(polygon[-1:] + polygon[:-1], polygon, polygon[1:] + polygon[:1]):
        u, v = sub(b, a), sub(c, b)
        lu, lv = math.hypot(*u), math.hypot(*v)
        nu, nv = (u[1] / lu, -u[0] / lu), (v[1] / lv, -v[0] / lv)
        p = (b[0] + distance * nu[0], b[1] + distance * nu[1])
        q = (b[0] + distance * nv[0], b[1] + distance * nv[1])
        denominator = cross(u, v)
        t = cross(sub(q, p), v) / denominator if abs(denominator) > 1e-10 else 0
        result.append((p[0] + t * u[0], p[1] + t * u[1]))
    return result


def hull2(points):
    points = sorted(set((round(x, 9), round(y, 9)) for x, y in points))
    lower, upper = [], []
    for result, source in ((lower, points), (upper, points[::-1])):
        for point in source:
            while len(result) > 1 and cross(sub(result[-1], result[-2]), sub(point, result[-1])) <= 1e-8:
                result.pop()
            result.append(point)
    return lower[:-1] + upper[:-1]


def clip(polygon, evaluate):
    if not polygon:
        return []
    result = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        fa, fb = evaluate(a), evaluate(b)
        if fa >= -1e-8:
            result.append(a)
        if (fa < -1e-8 and fb > 1e-8) or (fb < -1e-8 and fa > 1e-8):
            t = fa / (fa - fb)
            result.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return result


def inside_convex(point, polygon):
    return all(cross(sub(b, a), sub(point, a)) >= -1e-7
               for a, b in zip(polygon, polygon[1:] + polygon[:1]))


def union_convex(a, b):
    """Boundary of two overlapping convex polygons, with their real concave joins."""
    segments = []
    for polygon, other in ((a, b), (b, a)):
        for p, q in zip(polygon, polygon[1:] + polygon[:1]):
            d = sub(q, p)
            cuts = [0.0, 1.0]
            for r, s in zip(other, other[1:] + other[:1]):
                e = sub(s, r)
                denominator = cross(d, e)
                if abs(denominator) < 1e-10:
                    continue
                t, u = cross(sub(r, p), e) / denominator, cross(sub(r, p), d) / denominator
                if -1e-8 <= t <= 1 + 1e-8 and -1e-8 <= u <= 1 + 1e-8:
                    cuts.append(min(1, max(0, t)))
            cuts = sorted(set(round(t, 10) for t in cuts))
            for lo, hi in zip(cuts, cuts[1:]):
                if hi - lo < 1e-8:
                    continue
                middle = (p[0] + (lo + hi) / 2 * d[0], p[1] + (lo + hi) / 2 * d[1])
                if not inside_convex(middle, other):
                    segments.append(((p[0] + lo * d[0], p[1] + lo * d[1]),
                                     (p[0] + hi * d[0], p[1] + hi * d[1])))
    first, current = segments.pop(0)
    boundary = [first]
    while segments:
        boundary.append(current)
        index = min(range(len(segments)), key=lambda i: math.dist(segments[i][0], current))
        start, end = segments.pop(index)
        if math.dist(start, current) > 1e-5:
            raise ValueError("Polygon union did not produce one closed boundary")
        current = end
    assert math.dist(current, first) < 1e-5
    return ccw(boundary)


def wall_polygons(outline, bounds, pole, merge=True):
    """Convex sectors tile a rectangular cell outside a star-shaped cavity."""
    xmin, xmax, ymin, ymax = bounds
    rectangle = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    polygons = []
    for a, b in zip(outline, outline[1:] + outline[:1]):
        assert cross(sub(a, pole), sub(b, pole)) >= -1e-6, (a, b, pole)
        p = clip(rectangle, lambda v: cross(sub(a, pole), sub(v, pole)))
        p = clip(p, lambda v: -cross(sub(b, pole), sub(v, pole)))
        p = clip(p, lambda v: -cross(sub(b, a), sub(v, a)))
        if len(p) >= 3 and abs(area(p)) > 1e-5:
            polygons.append(hull2(p))
    if merge:
        changed = True
        while changed:
            changed = False
            for i in range(len(polygons)):
                for j in range(i + 1, len(polygons)):
                    h = hull2(polygons[i] + polygons[j])
                    if abs(area(h) - area(polygons[i]) - area(polygons[j])) < 1e-5:
                        polygons[i] = h
                        polygons.pop(j)
                        changed = True
                        break
                if changed:
                    break
    return polygons


def circular_collision_outline(radius, notch_distance):
    """16 facets around circle + finger notch; 2 concave joins permit 2 merges.

    The round seat is circumscribed (no clearance loss). Notch facets are also
    circumscribed; their maximum excess clearance is 0.989 mm.
    """
    minimum = math.ceil(math.pi / math.acos(radius / (radius + .99)))
    for main_count in range(minimum, minimum + 3):
        for phase in range(24):
            main = regular_polygon(radius / math.cos(math.pi / main_count), main_count,
                                   phase * 2 * math.pi / main_count / 24)
            for notch_phase in range(16):
                notch = regular_polygon(12 / math.cos(math.pi / 8), 8,
                                        notch_phase * math.pi / 4 / 16, (-notch_distance, 0))
                combined = union_convex(main, notch)
                if len(combined) == 16:
                    return combined, {"round_sides": main_count, "boundary_segments": 16,
                                      "notch_sides": 8, "round_max_error_mm":
                                      radius * (1 / math.cos(math.pi / main_count) - 1)}
    raise ValueError(f"Could not allocate 16 facets to the R{radius} circular pocket and notch")


def mesh(name, vertices, faces, material=None):
    return props._mesh_object(name, vertices, faces, (0, 0, 0), 0, 3, material)


def loft(name, profiles, zs, material=None):
    n = len(profiles[0])
    vertices = [(x, y, z) for profile, z in zip(profiles, zs, strict=True) for x, y in profile]
    faces = [tuple(reversed(range(n))), tuple(range((len(profiles) - 1) * n, len(profiles) * n))]
    faces.extend((r * n + i, r * n + (i + 1) % n, (r + 1) * n + (i + 1) % n, (r + 1) * n + i)
                 for r in range(len(profiles) - 1) for i in range(n))
    return mesh(name, vertices, faces, material)


def revolve(name, profile, count, material):
    vertices = [(r * math.cos(2 * math.pi * i / count), r * math.sin(2 * math.pi * i / count), z)
                for r, z in profile for i in range(count)]
    faces = [(row * count + i, row * count + (i + 1) % count,
              ((row + 1) % len(profile)) * count + (i + 1) % count,
              ((row + 1) % len(profile)) * count + i)
             for row in range(len(profile)) for i in range(count)]
    return mesh(name, vertices, faces, material)


def boolean(target, cutter, operation="DIFFERENCE"):
    if not cutter.data.materials:
        cutter.data.materials.append(target.data.materials[0])
    props._select([target])
    modifier = target.modifiers.new(operation + " " + cutter.name, "BOOLEAN")
    modifier.operation, modifier.solver, modifier.object = operation, "EXACT", cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def bevel(obj, width, segments=3):
    props._select([obj])
    modifier = obj.modifiers.new("Manufactured edge radii", "BEVEL")
    modifier.width, modifier.segments, modifier.limit_method = props.mm(width), segments, "ANGLE"
    modifier.angle_limit, modifier.harden_normals = math.radians(30), True
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def finish(obj):
    """Clean at the OBJ writer's micrometre precision before UVs and normals."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for vertex in bm.verts:
        vertex.co = tuple(round(v, 6) for v in vertex.co)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=2e-7)
    bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=2e-7)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    seen, bad = set(), []
    for face in bm.faces:
        key = frozenset(face.verts)
        if key in seen or face.calc_area() <= 2e-18:
            bad.append(face)
        seen.add(key)
    if bad:
        bmesh.ops.delete(bm, geom=bad, context="FACES_ONLY")
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    props.smooth_by_angle(obj, 30)
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # Large coplanar caps remain flat next to their narrow chamfers.
    normal = obj.modifiers.new("Area-weighted manufacturing normals", "WEIGHTED_NORMAL")
    normal.keep_sharp, normal.mode, normal.weight = True, "FACE_AREA", 50
    bpy.ops.object.modifier_apply(modifier=normal.name)
    return obj


def move(obj, xyz=(0, 0, 0), rotation=(0, 0, 0)):
    obj.location = [props.mm(v) for v in xyz]
    obj.rotation_euler = [math.radians(v) for v in rotation]
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def hide_collisions(collisions):
    for obj in collisions:
        obj.hide_render = True
        obj.hide_set(True)


def grain(material, scale=3000, amplitude=.06, bump_mm=.015):
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    original = bsdf.inputs["Base Color"].links[0].from_socket
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 2
    tree.links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
    factor = props._math(tree, "ADD", 1 - amplitude / 2,
                         props._math(tree, "MULTIPLY", noise.outputs["Fac"], amplitude))
    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.blend_type, mix.inputs[0].default_value = "MULTIPLY", 1
    tree.links.new(original, mix.inputs[1])
    tree.links.new(factor, mix.inputs[2])
    tree.links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .18
    bump.inputs["Distance"].default_value = props.mm(bump_mm)
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material


def zinc_material():
    material = props.pbr_material("Galvanised steel #BFC3C7 - crystalline zinc spangle", "#BFC3C7",
                                  .30, 1, grime=.12, machining="brushed")
    tree, bsdf = material.node_tree, material.node_tree.nodes["Principled BSDF"]
    original = bsdf.inputs["Base Color"].links[0].from_socket
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    voronoi = tree.nodes.new("ShaderNodeTexVoronoi")
    voronoi.name = "1.7 mm zinc crystal spangle"
    voronoi.inputs["Scale"].default_value = 590
    voronoi.inputs["Randomness"].default_value = .85
    tree.links.new(coordinates.outputs["Object"], voronoi.inputs["Vector"])
    grayscale = tree.nodes.new("ShaderNodeRGBToBW")
    tree.links.new(voronoi.outputs["Color"], grayscale.inputs[0])
    variation = props._math(tree, "ADD", .78, props._math(tree, "MULTIPLY", grayscale.outputs[0], .40))
    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.blend_type, mix.inputs[0].default_value = "MULTIPLY", 1
    tree.links.new(original, mix.inputs[1])
    tree.links.new(variation, mix.inputs[2])
    tree.links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    tree.links.new(props._math(tree, "ADD", .22,
                             props._math(tree, "MULTIPLY", grayscale.outputs[0], .16)), bsdf.inputs["Roughness"])
    return material


def start(prop_id):
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    folder = OUTPUT / prop_id
    folder.mkdir(parents=True, exist_ok=True)
    print("TASK06_START", prop_id, flush=True)
    return folder


def bake_tray_light_variant(body, folder):
    """Replace only the EVA slot while baking; leave the black PBR untouched."""
    foam_index = next(i for i, mat in enumerate(body.data.materials)
                      if mat.name == "Black EVA foam #161616")
    black = body.data.materials[foam_index]
    light_name = "Light-grey EVA foam #A9AFB6"
    previous = bpy.data.materials.get(light_name)
    if previous and previous.users <= int(previous.use_fake_user):
        bpy.data.materials.remove(previous)
    light = black.copy()
    light.name = light_name
    light.diffuse_color = props._linear_color("#A9AFB6")
    tree = light.node_tree
    for node in tree.nodes:
        if node.type == "RGB":
            node.outputs[0].default_value = props._linear_color("#A9AFB6")
    # Tint existing upward-facing pocket floors, including the bushing shoulder
    # and every notch floor. This is shading only: no face/UV/material-index edits.
    bsdf = tree.nodes["Principled BSDF"]
    original = bsdf.inputs["Base Color"].links[0].from_socket
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    position = tree.nodes.new("ShaderNodeSeparateXYZ")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    normal = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(coordinates.outputs["Object"], position.inputs[0])
    tree.links.new(geometry.outputs["Normal"], normal.inputs[0])
    below_top = props._math(tree, "LESS_THAN", position.outputs["Z"], props.mm(42.25))
    above_bottom = props._math(tree, "GREATER_THAN", position.outputs["Z"], props.mm(3.5))
    upward = props._math(tree, "GREATER_THAN", normal.outputs["Z"], .92)
    floor = props._math(tree, "MULTIPLY", upward,
                        props._math(tree, "MULTIPLY", below_top, above_bottom))
    tint = tree.nodes.new("ShaderNodeMixRGB")
    tint.name = "Pocket floor tint: 0.80 linear, geometry unchanged"
    tint.blend_type = "MULTIPLY"
    tree.links.new(floor, tint.inputs[0])
    tree.links.new(original, tint.inputs[1])
    tint.inputs[2].default_value = (.80, .80, .80, 1)
    tree.links.new(tint.outputs[0], bsdf.inputs["Base Color"])
    light["base_color_srgb"] = "#A9AFB6"
    light["baked_ao_strength"] = .8
    light["pocket_floor_tint_linear"] = .8
    light.use_fake_user = True
    # An in-place rerun must not render a previously packed copy of this atlas.
    texture_path = (folder / "body_albedo_v1.png").resolve()
    for old_image in list(bpy.data.images):
        if (old_image.filepath and Path(bpy.path.abspath(old_image.filepath)).resolve() == texture_path
                and old_image.users <= int(old_image.use_fake_user)):
            bpy.data.images.remove(old_image)
    body.data.materials[foam_index] = light
    try:
        image = props.bake_albedo([body], 2048, texture_path, ao_strength=.8)
        image.use_fake_user = True
    finally:
        body.data.materials[foam_index] = black
    variants = [
        {"name": "black", "textures": {"body": "body_albedo.png"},
         "source_material": black.name},
        {"name": "light_grey", "textures": {"body": "body_albedo_v1.png"},
         "source_material": light.name, "foam_base_color_srgb": "#A9AFB6",
         "pocket_ao_strength": .8, "pocket_floor_tint_linear": .8},
    ]
    bpy.context.scene["material_variants"] = json.dumps(variants)
    return variants


def baked_render_material(name, texture, roughness=.95):
    material = props.pbr_material(name, "#FFFFFF", roughness, 0)
    image = bpy.data.images.load(str(texture.resolve()), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    material.node_tree.links.new(node.outputs["Color"],
                                 material.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    return material


def render_tray_light_variant(body, folder):
    """Preview and robot QA use the exported atlas, not an unbaked PBR proxy."""
    tray = body.copy()
    tray.data = body.data.copy()
    tray.name = "QA light-grey tray (exported albedo)"
    bpy.context.collection.objects.link(tray)
    tray.data.materials.clear()
    tray.data.materials.append(baked_render_material("QA light-grey EVA atlas", folder / "body_albedo_v1.png"))
    for polygon in tray.data.polygons:
        polygon.material_index = 0
    scene = None
    created = [tray]
    materials = [tray.data.materials[0]]
    try:
        props.render_preview([tray], folder / "preview.png")
        scene = bpy.data.scenes.new("Temporary foam tray robot-camera QA")
        scene.render.engine = "CYCLES"
        scene.cycles.device, scene.cycles.samples, scene.cycles.seed = "CPU", 64, 0
        scene.cycles.use_adaptive_sampling = False
        scene.render.resolution_x, scene.render.resolution_y = 640, 480
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode, scene.render.image_settings.color_depth = "RGB", "8"
        scene.view_settings.view_transform, scene.view_settings.look = "AgX", "None"
        scene.world = bpy.data.worlds.new("Neutral grey robot-camera world")
        scene.world.use_nodes = True
        background = scene.world.node_tree.nodes["Background"]
        background.inputs["Color"].default_value = (.18, .18, .18, 1)
        background.inputs["Strength"].default_value = .45
        scene.collection.objects.link(tray)
        # Reuse existing exported parts. Never build or write their assets here.
        placements = (("hex_blank", .065, -.235), ("l_bracket", -.065, -.245),
                      ("flanged_bushing", .065, .235), ("spur_gear", -.065, .245))
        for prop_id, x, y in placements:
            part_folder = props.ROOT / "assets" / "props" / prop_id
            part = json.loads((part_folder / "prop.json").read_text())["parts"][0]
            before = set(bpy.data.objects)
            bpy.ops.wm.obj_import(filepath=str(part_folder / part["visual"]), forward_axis="Y", up_axis="Z")
            imported = list(set(bpy.data.objects) - before)
            created.extend(imported)
            if len(imported) != 1 or imported[0].type != "MESH":
                raise ValueError(f"Expected one existing visual mesh for {prop_id}")
            obj = imported[0]
            obj.name = "QA existing " + prop_id
            obj.location = (x, y, 0)
            obj.data.materials.clear()
            mat = baked_render_material("QA existing " + prop_id + " atlas", part_folder / part["texture"], .45)
            materials.append(mat)
            obj.data.materials.append(mat)
            scene.collection.objects.link(obj)
        camera_data = bpy.data.cameras.new("Robot camera: 600 mm above foam, tilt 30 degrees")
        camera = bpy.data.objects.new(camera_data.name, camera_data)
        created.append(camera)
        scene.collection.objects.link(camera)
        target = Vector((0, 0, .043))
        camera.location = target + Vector((-.6 * math.tan(math.radians(30)), 0, .6))
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera_data.type, camera_data.lens, camera_data.sensor_width = "PERSP", 40, 36
        camera_data.clip_start, camera_data.clip_end = .01, 10
        scene.camera = camera
        for name, location, power, size in (
            ("QA Key", (-.55, .45, .9), 42, .65),
            ("QA Fill", (-.3, -.6, .5), 18, .75),
            ("QA Rim", (.5, .3, .75), 28, .55),
        ):
            data = bpy.data.lights.new(name, "AREA")
            data.energy, data.shape, data.size = power, "DISK", size
            obj = bpy.data.objects.new(name, data)
            created.append(obj)
            scene.collection.objects.link(obj)
            obj.location = location
            obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()
        floor = props.box("QA neutral grey ground", (2000, 2000, 2), (0, 0, -2), bevel_mm=0,
                          material=props.pbr_material("QA ground", "#777777", .8, 0))
        materials.append(floor.data.materials[0])
        created.append(floor)
        scene.collection.objects.link(floor)
        out_png = HERE / "_qa" / "foam_kit_tray_light_top.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(out_png)
        bpy.ops.render.render(write_still=True, scene=scene.name)
        camera_info = {"resolution_px": [640, 480], "height_above_foam_m": .6,
                       "tilt_from_vertical_deg": 30, "slant_range_m": (target - camera.location).length,
                       "position_m": list(camera.location), "target_m": list(target),
                       "focal_length_mm": 40, "sensor_width_mm": 36, "cycles_samples": 64,
                       "variant": "light_grey", "texture": "body_albedo_v1.png",
                       "parts_beside_tray": [p[0] for p in placements]}
        qa = HERE / "_qa" / "task12"
        qa.mkdir(parents=True, exist_ok=True)
        (qa / "camera.json").write_text(json.dumps(camera_info, indent=2) + "\n")
    finally:
        for obj in created:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data.users == 0:
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                elif isinstance(data, bpy.types.Camera):
                    bpy.data.cameras.remove(data)
                elif isinstance(data, bpy.types.Light):
                    bpy.data.lights.remove(data)
        if scene:
            world = scene.world
            bpy.data.scenes.remove(scene)
            if world and world.users == 0:
                bpy.data.worlds.remove(world)
        for material in materials:
            if material.users == 0:
                bpy.data.materials.remove(material)


def update_tray_materials_only():
    folder = OUTPUT / "foam_kit_tray"
    bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
    data = json.loads((folder / "prop.json").read_text())
    body = bpy.data.objects[data["parts"][0]["source_visual_object"]]
    data["variants"] = bake_tray_light_variant(body, folder)
    render_tray_light_variant(body, folder)
    if TRAY_LIGHT_NOTES not in data["notes"]:
        data["notes"] += " " + TRAY_LIGHT_NOTES
    bpy.context.scene["light_variant_bake_recipe"] = "CPU Cycles 64; DIFFUSE COLOR * mix(1, AO, .8); floor tint .80"
    bpy.context.preferences.filepaths.save_version = 0
    props._select([body])
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    props.write_prop_json(data, folder / "prop.json")
    print("TASK12_COMPLETE", folder.name, "black and light_grey; geometry/collision/default texture preserved", flush=True)


def complete(folder, body, collisions, size, mass, friction, material, dimensions, notes, extra=None,
             variant_material=None):
    hide_collisions(collisions)
    body.name = "body"
    finish(body)
    props.uv_unwrap(body)
    for loop in body.data.uv_layers["UVMap"].data:
        loop.uv = (.0005 + .999 * loop.uv.x, .0005 + .999 * loop.uv.y)
    preflight = {"id": folder.name, "bbox_mm": props.bbox_mm([body]),
                 "triangles": props.triangle_count([body]), "collision_count": len(collisions)}
    PREFLIGHT.append(preflight)
    print("TASK06_GEOMETRY", json.dumps(preflight), flush=True)
    assert preflight["triangles"] <= (50_000 if folder.name == "foam_kit_tray" else 10_000), preflight
    assert len(collisions) <= 64, preflight
    if GEOMETRY_ONLY:
        QA.mkdir(parents=True, exist_ok=True)
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(QA / f"{folder.name}_geometry.blend"), check_existing=False)
        return
    props.bake_albedo([body], size, folder / "body_albedo.png")
    variants = []
    if variant_material:
        original = list(body.data.materials)
        body.data.materials.clear()
        body.data.materials.append(variant_material)
        props.bake_albedo([body], size, folder / "body_albedo_v1.png")
        body.data.materials.clear()
        for mat in original:
            body.data.materials.append(mat)
        variant_material.use_fake_user = True
        variants = [
            {"name": "white_pom", "textures": {"body": "body_albedo.png"},
             "mass_kg": mass, "masses_kg": {"body": mass}, "mujoco_materials": {"body": POM}},
            {"name": "steel", "textures": {"body": "body_albedo_v1.png"},
             "mass_kg": round(mass * 7850 / 1410, 6),
             "masses_kg": {"body": round(mass * 7850 / 1410, 6)}, "mujoco_materials": {"body": ZINC},
             "source_material": variant_material.name},
        ]
    collision_files = props.export_collision("body", collisions, folder)
    assert len(collision_files) <= 64
    hide_collisions(collisions)
    props.export_obj(body, folder / "body.obj")
    if folder.name == "foam_kit_tray":
        variants = bake_tray_light_variant(body, folder)
        render_tray_light_variant(body, folder)
        notes += " " + TRAY_LIGHT_NOTES
        bpy.context.scene["light_variant_bake_recipe"] = "CPU Cycles 64; DIFFUSE COLOR * mix(1, AO, .8); floor tint .80"
    else:
        props.render_preview([body], folder / "preview.png")
    data = {"id": folder.name, "task": "shaped_kitting", "units": "m", "origin": "bottom_center",
            "bbox_mm": [round(v, 6) for v in props.bbox_mm([body])],
            "budget_class": "medium" if folder.name == "foam_kit_tray" else "small",
            "triangles": props.triangle_count([body]), "source": "source.blend", "preview": "preview.png",
            "parts": [{"name": "body", "visual": "body.obj", "texture": "body_albedo.png",
                       "texture_px": size, "collision": collision_files, "mass_kg": mass,
                       "friction": friction, "mujoco_material": material, "parent": None, "joint": None,
                       "source_visual_object": body.name,
                       "source_collision_objects": [obj.name for obj in collisions]}],
            "variants": variants, "dimensions_mm": dimensions,
            "notes": notes + " " + COMMON_NOTES + f"Generated with Blender {bpy.app.version_string}.",
            **(extra or {})}
    scene = bpy.context.scene
    scene["prop_id"], scene["dimensions_mm"] = folder.name, json.dumps(dimensions)
    scene["bake_recipe"] = "CPU Cycles 64; DIFFUSE COLOR * mix(1, AO, .6); 8-bit sRGB"
    scene["preview_recipe"] = "CPU Cycles 64; 512 square; three lights; front-left above; grey"
    scene["collision_piece_count"] = len(collision_files)
    bpy.context.preferences.filepaths.save_version = 0
    props._select([body])
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    props.write_prop_json(data, folder / "prop.json")
    print("TASK06_COMPLETE", folder.name, data["bbox_mm"], data["triangles"], len(collision_files), flush=True)


def hex_outline(across_flats=27):
    return regular_polygon(across_flats / math.sqrt(3), 6, math.pi / 6)


def l_outline(clearance=0, samples=12):
    """Side-lying bracket, 40 X x 60 Y, R4 inner fillet, mitered outer corners."""
    a, b, thickness, radius = 20 + clearance, 30 + clearance, 4 + 2 * clearance, 4 - clearance
    center = (-12, -22)
    outline = [(-a, -b), (a, -b), (a, -b + thickness), (center[0], center[1] - radius)]
    outline += [(center[0] + radius * math.cos(-math.pi / 2 - math.pi / 2 * i / samples),
                 center[1] + radius * math.sin(-math.pi / 2 - math.pi / 2 * i / samples))
                for i in range(1, samples + 1)]
    outline += [(-a + thickness, b), (-a, b)]
    return ccw(outline)


def involute_outline(module=1.5, teeth=30, pressure_angle=20, flank_samples=5):
    """Exact sampled involute flanks; radial root transitions below the base circle.

    r(t)=rb*sqrt(1+t^2); inv(t)=t-atan(t). The pitch-circle tooth thickness is
    pi*m/2, the standard addendum m and dedendum 1.25*m. No sinusoidal teeth.
    """
    rp = module * teeth / 2
    rb, ra, rf = rp * math.cos(math.radians(pressure_angle)), rp + module, rp - 1.25 * module
    inv_pitch = math.tan(math.radians(pressure_angle)) - math.radians(pressure_angle)
    base_half = math.pi / (2 * teeth) + inv_pitch
    tmax = math.sqrt((ra / rb) ** 2 - 1)
    result = []
    for tooth in range(teeth):
        center = tooth * 2 * math.pi / teeth
        polar = [(rf, center - base_half)]
        for i in range(flank_samples + 1):
            t = tmax * i / flank_samples
            polar.append((rb * math.sqrt(1 + t * t), center - base_half + t - math.atan(t)))
        top_half = base_half - tmax + math.atan(tmax)
        polar += [(ra, center - top_half + 2 * top_half * i / 3) for i in range(1, 3)]
        for i in range(flank_samples, -1, -1):
            t = tmax * i / flank_samples
            polar.append((rb * math.sqrt(1 + t * t), center + base_half - t + math.atan(t)))
        polar += [(rf, center + base_half)]
        gap = 2 * math.pi / teeth - 2 * base_half
        polar += [(rf, center + base_half + gap * i / 3) for i in range(1, 3)]
        result += [(r * math.cos(theta), r * math.sin(theta)) for r, theta in polar]
    return ccw(result), {"module_mm": module, "teeth": teeth, "pressure_angle_deg": pressure_angle,
                         "pitch_diameter": 2 * rp, "base_diameter": 2 * rb, "tip_diameter": 2 * ra,
                         "root_diameter": 2 * rf, "involute_samples_per_flank": flank_samples + 1,
                         "pitch_tooth_thickness": math.pi * module / 2}


def build_hex():
    folder = start("hex_blank")
    material = props.pbr_material("Blackened steel #2A2B2F", "#2A2B2F", .45, .8,
                                  grime=.12, machining="milled")
    full = hex_outline()
    inset = offset_polygon(full, -math.tan(math.radians(30)))
    body = loft("Hex blank, two 1 mm x 30 degree end chamfers", [inset, full, full, inset],
                [0, 1, 29, 30], material)
    bevel(body, .3, 3)
    collider = props.extruded_profile("Hexagonal prism collision", full, 30, (0, 0, 15), bevel_mm=0)
    complete(folder, body, [collider], 1024, .149, .30, STEEL,
             {"across_flats": 27, "height": 30, "end_chamfer_axial": 1,
              "end_chamfer_angle_deg": 30},
             "Blackened steel, 0.149 kg. End chamfer: 1 mm axial run at 30 degrees to the axis, "
             "0.577 mm normal setback; vertical corners softened 0.3 mm. One six-sided collision prism.")


def build_bushing():
    folder = start("flanged_bushing")
    material = props.pbr_material("Turned bronze #B08D57", "#B08D57", .35, 1,
                                  grime=.12, machining="turned")
    profile = [(10.5, 0), (14.5, 0), (15, .5), (15, 24.5), (15.5, 25), (19.5, 25),
               (20, 25.5), (20, 29.5), (19.5, 30), (10.5, 30), (10, 29.5), (10, .5)]
    body = revolve("Turned stepped through-bored bronze bushing", profile, 64, material)
    collisions = [props.cylinder("Body collision - bore intentionally omitted", 15, 25, (0, 0, 12.5),
                                 bevel_mm=0, vertices=32),
                  props.cylinder("Flange collision - bore intentionally omitted", 20, 5, (0, 0, 27.5),
                                 bevel_mm=0, vertices=32)]
    complete(folder, body, collisions, 1024, .128, .30, BRONZE,
             {"outer_diameter": 30, "bore_diameter": 20, "body_length": 25,
              "flange_diameter": 40, "flange_thickness": 5, "height": 30, "chamfer": .5},
             "Bronze, 0.128 kg. Real 20 mm through bore and 0.5 mm machined chamfers. "
             "The two prescribed convex cylinders omit the bore. Flange is at +Z for insertion.")


def capsule(length, width, samples=12):
    result = []
    for cx, begin in ((length / 2 - width / 2, -math.pi / 2),
                      (-length / 2 + width / 2, math.pi / 2)):
        result += [(cx + width / 2 * math.cos(begin + math.pi * i / samples),
                    width / 2 * math.sin(begin + math.pi * i / samples)) for i in range(samples + 1)]
    return ccw(result)


def build_bracket():
    folder = start("l_bracket")
    body = props.extruded_profile("Side-lying R4 galvanised angle", l_outline(), 40, (0, 0, 20),
                                  bevel_mm=0, material=zinc_material())
    for y in (-12, 15):
        cutter = props.extruded_profile("6.5 x 12 through slot", capsule(12, 6.5), 8, bevel_mm=0)
        # Capsule long direction -> Y; width -> Z; extrusion -> X.
        cutter.data.transform(Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))))
        move(cutter, (-18, y, 20))
        boolean(body, cutter)
    cutter = props.cylinder("6.5 mm short-leg bore", 3.25, 8, (7, -28, 20), bevel_mm=0,
                            vertices=32, rotation_deg=(90, 0, 0))
    boolean(body, cutter)
    bevel(body, .35, 3)
    # Grow only the inside face by .25 mm: two boxes then follow the R4 fillet
    # within .922 mm, while straight-leg excess is just .25 mm.
    collisions = [props.box("60 mm leg collision", (4.25, 60, 40), (-17.875, 0, 20), bevel_mm=0),
                  props.box("40 mm leg collision", (40, 4.25, 40), (0, -27.875, 20), bevel_mm=0)]
    complete(folder, body, collisions, 1024, .120, .30, ZINC,
             {"long_leg": 60, "short_leg": 40, "width": 40, "thickness": 4,
              "inner_bend_radius": 4, "slots": [6.5, 12], "slot_centers_y": [-12, 15],
              "short_leg_hole_diameter": 6.5, "side_lying_bbox": [40, 60, 40]},
             "Galvanised steel with procedural zinc spangle and fine straight brushing, 0.120 kg. "
             "Side-lying insertion orientation: 60 mm along Y, 40 mm along X, width along Z. "
             "R4 inner fillet and softened outer angle. Two 6.5 x 12 slots in long leg, one real "
             "6.5 mm bore in short leg. The prescribed two box proxies omit the holes. Each inside "
             "collision face is expanded .25 mm, keeping the R4 fillet deviation <=.922 mm and "
             "the straight-leg surface excess at .25 mm.")


def build_gear():
    folder = start("spur_gear")
    outline, dimensions = involute_outline()
    material = grain(props.pbr_material("White POM #ECEAE3", "#ECEAE3", .45, 0,
                                        wear=.012, grime=.12), scale=5500, amplitude=.035, bump_mm=.006)
    inset = offset_polygon(outline, -.3)
    body = loft("30-tooth true involute gear - 0.3 mm face chamfers", [inset, outline, outline, inset],
                [0, .3, 11.7, 12], material)
    hub = props.cylinder("One-sided 24 x 6 hub", 12, 6.3, (0, 0, 14.85), bevel_mm=.3,
                         segments=3, vertices=48, material=material)
    boolean(body, hub, "UNION")
    bore = props.cylinder("8 mm through shaft bore", 4, 22, (0, 0, 9), bevel_mm=0, vertices=48)
    boolean(body, bore)
    keyway = props.box("3 mm keyway - 1.4 mm radial depth", (2.2, 3, 22), (4.3, 0, 9), bevel_mm=0)
    boolean(body, keyway)
    # Bore entry relief is geometry, not an alpha decal.
    for z, rotation in ((0, 0), (18, 180)):
        bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=props.mm(4.3), radius2=props.mm(4),
                                       depth=props.mm(.3))
        cutter = bpy.context.object
        move(cutter, (0, 0, z + (.15 if z == 0 else -.15)), (rotation, 0, 0))
        boolean(body, cutter)
    collisions = [props.cylinder("Tip-circle collision - teeth and bore simplified", 24, 12,
                                  (0, 0, 6), bevel_mm=0, vertices=32),
                  props.cylinder("Hub collision", 12, 6, (0, 0, 15), bevel_mm=0, vertices=32)]
    dimensions.update({"face_width": 12, "hub_diameter": 24, "hub_height": 6,
                       "bore_diameter": 8, "keyway_width": 3, "keyway_radial_depth": 1.4})
    steel = props.pbr_material("Steel variant #9CA3AA", "#9CA3AA", .30, 1,
                               grime=.12, machining="turned")
    complete(folder, body, collisions, 1024, .029, .30, POM, dimensions,
             "Python-generated true involute flanks; standard 20 degree pressure angle selected "
             "because the specification does not state one. Standard full-depth dedendum 1.25m, "
             "radial root transitions below base circle and 0.3 mm face chamfers. Keyway width 3 mm; "
             "unspecified radial depth chosen as 1.4 mm. Hub up (+Z). White POM default 0.029 kg; "
             "steel variant mass 0.161454 kg estimated by density ratio 7850/1410, with its own "
             "texture, PBR and MuJoCo material override. Specified tip-circle collision bridges "
             "tooth gaps and ignores bore/keyway; tooth-root discrepancy 3.375 mm is the explicit "
             "section 3.6 cylinder simplification.", variant_material=steel)


def box_bounds(name, x0, x1, y0, y1, z0, z1, material=None, bevel_mm=0):
    return props.box(name, (x1 - x0, y1 - y0, z1 - z0),
                     ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                     bevel_mm=bevel_mm, material=material)


def extrude_walls(name, polygons, center, z0, z1):
    return [props.extruded_profile(f"{name}_{i:02d}", polygon, z1 - z0,
                                   (*center, (z0 + z1) / 2), bevel_mm=0)
            for i, polygon in enumerate(polygons)]


def build_tray():
    folder = start("foam_kit_tray")
    pp = grain(props.pbr_material("Grey PP shell #8A9097", "#8A9097", .55, 0, wear=.035, grime=.15),
               scale=3600, amplitude=.06)
    foam_material = grain(props.pbr_material("Black EVA foam #161616", "#161616", .95, 0,
                                             wear=.004, grime=.15),
                          scale=5500, amplitude=.20, bump_mm=.055)
    shell = props.box("260 x 380 x 50 grey PP shell", (260, 380, 50), (0, 0, 25),
                       bevel_mm=0, material=pp)
    boolean(shell, props.box("Open PP shell, 3 mm wall and floor", (254, 374, 55), (0, 0, 30.5),
                              bevel_mm=0))
    bevel(shell, 1, 3)
    foam = props.box("252 x 372 x 40 EVA insert", (252, 372, 40), (0, 0, 23),
                      bevel_mm=0, material=foam_material)
    # Outlines are derived from the parts, not guessed from a photograph.
    hex_slot = offset_polygon(hex_outline(), 1.5)
    l_slot = l_outline(clearance=1.5)
    gear_outline, gear_dims = involute_outline()
    gear_radius = max(math.hypot(x, y) for x, y in gear_outline) + 1.5
    pockets = [
        ("hex_blank", (55, -90), hex_slot, 18, 43, -15),
        ("flanged_bushing_counterbore", (55, 90), regular_polygon(21.5, 96), 38, 43, -21.5),
        ("flanged_bushing_body", (55, 90), regular_polygon(16.5, 96), 13, 38, -21.5),
        ("l_bracket", (-55, -90), l_slot, 18, 43, -21.5),
        ("spur_gear", (-55, 90), regular_polygon(gear_radius, 96), 33, 43, -gear_radius),
    ]
    for name, center, outline, z0, z1, _ in pockets:
        boolean(foam, props.extruded_profile(name + " pocket cutter", outline, z1 - z0 + .1,
                                             (*center, (z0 + z1) / 2 + .05), bevel_mm=0))
    # Same-depth -X finger notches, including the complete 30 mm bushing depth.
    for name, center, _, z0, z1, notch_x in (pockets[0], pockets[2], pockets[3], pockets[4]):
        if name == "l_bracket":
            # The straight L wall admits a literal half-circle, without enlarging its 7 mm leg.
            half = [(0, 12)] + [(12 * math.cos(math.pi / 2 + math.pi * i / 32),
                                  12 * math.sin(math.pi / 2 + math.pi * i / 32)) for i in range(1, 33)]
            cutter = props.extruded_profile("L half-circle finger notch", half, z1 - z0 + .1,
                                            (center[0] + notch_x, center[1], (z0 + z1) / 2 + .05), bevel_mm=0)
        else:
            cutter = props.cylinder(name + " 24 mm finger notch", 12, 43 - z0 + .1,
                                    (center[0] + notch_x, center[1], (z0 + 43) / 2 + .05),
                                    bevel_mm=0, vertices=64)
        boolean(foam, cutter)
    bevel(foam, .5, 3)
    body = props.join_objects([shell, foam], "body")

    collisions = [box_bounds("Common floor, shell plus foam below deepest pocket", -130, 130,
                             -190, 190, 0, 13),
                  box_bounds("Hex and L floors, plus lower inter-pocket fill", -126, 126,
                             -186, 0, 13, 18),
                  box_bounds("Gear floor plus lower front-right inter-pocket fill", -126, 0,
                             0, 186, 13, 33)]
    labels = ["base", "hex_l_floor", "gear_floor"]
    for name, bounds in (
        ("PP -X wall", (-130, -127, -190, 190)), ("PP +X wall", (127, 130, -190, 190)),
        ("PP -Y wall", (-127, 127, -190, -187)), ("PP +Y wall", (-127, 127, 187, 190)),
    ):
        collisions.append(box_bounds(name, *bounds, 0, 50))
        labels.append("shell_wall")
    # Six primary hex faces are clipped by the real notch. A six-sided notch
    # approximation stays within 0.915 mm of the 12 mm radius circle.
    hex_notch = regular_polygon(12.8, 6, 0, (-15, 0))
    hex_combined = union_convex(hex_slot, hex_notch)
    hex_walls = wall_polygons(hex_combined, (-55, 71, -96, 90), (-10, 0))
    collisions += extrude_walls("Hex face or notch wall", hex_walls, (55, -90), 18, 43)
    labels += ["hex_wall"] * len(hex_walls)
    layout = {"hex": {"primary_hex_faces": 6, "combined_boundary_segments": len(hex_combined),
                       "convex_wall_pieces": len(hex_walls), "notch_max_error_mm": .915}}
    for name, radius, distance, center, bounds, z0, z1 in (
        ("bushing_body", 16.5, 21.5, (55, 90), (-55, 71, -90, 96), 13, 38),
        ("bushing_counterbore", 21.5, 21.5, (55, 90), (-55, 71, -90, 96), 38, 43),
        ("gear", 25.5, 25.5, (-55, 90), (-71, 55, -90, 96), 33, 43),
    ):
        outline, info = circular_collision_outline(radius, distance)
        walls = wall_polygons(outline, bounds, (-distance + 6, 0))
        info["convex_wall_pieces"] = len(walls)
        layout[name] = info
        collisions += extrude_walls(name + " wall", walls, center, z0, z1)
        labels += [name + "_wall"] * len(walls)
    # Rectangular complement of the 63 x 43, 7 mm leg L, with a semicircular notch.
    l_boxes = [(-21.5, 21.5, -96, -31.5), (-21.5, 21.5, 31.5, 90),
               (21.5, 55, -96, 90), (-14, 21.5, -24, 31.5)]
    for index, (x0, x1, y0, y1) in enumerate(l_boxes):
        collisions.append(box_bounds(f"L rectangular fill {index}", x0 - 55, x1 - 55,
                                     y0 - 90, y1 - 90, 18, 43))
        labels.append("l_box")
    notch_r = 12 / math.cos(math.pi / 8)
    l_half = [(-21.5 + notch_r * math.cos(math.pi / 2 + i * math.pi / 4),
               notch_r * math.sin(math.pi / 2 + i * math.pi / 4)) for i in range(5)]
    l_notch_walls = wall_polygons(l_half, (-71, -21.5, -96, 90), (-21.5, 0), merge=False)
    collisions += extrude_walls("L notch wall", l_notch_walls, (-55, -90), 18, 43)
    labels += ["l_notch_wall"] * len(l_notch_walls)
    layout["l_bracket"] = {"rectangular_fills": 4, "notch_walls": len(l_notch_walls),
                            "inner_corner_collision_clearance": .5}
    print("TASK06_COLLISION_LAYOUT", json.dumps(layout), "TOTAL", len(collisions), flush=True)
    assert len(collisions) <= 64, layout
    pocket_metadata = [
        {"part": "hex_blank", "center_yx_mm": [-90, 55], "shape": "hexagon", "across_flats_mm": 30,
         "depth_mm": 25, "floor_z_mm": 18, "notch_center_xy_mm": [40, -90]},
        {"part": "flanged_bushing", "center_yx_mm": [90, 55], "shape": "stepped_circle",
         "steps": [{"diameter_mm": 43, "depth_mm": 5, "floor_z_mm": 38},
                   {"diameter_mm": 33, "depth_mm": 25, "floor_z_mm": 13}],
         "depth_mm": 30, "floor_z_mm": 13, "notch_center_xy_mm": [33.5, 90],
         "required_orientation": "flange_up"},
        {"part": "l_bracket", "center_yx_mm": [-90, -55], "shape": "side_lying_L",
         "outer_yx_mm": [63, 43], "leg_mm": 7, "depth_mm": 25, "floor_z_mm": 18,
         "notch_center_xy_mm": [-76.5, -90]},
        {"part": "spur_gear", "center_yx_mm": [90, -55], "shape": "circle", "diameter_mm": 51,
         "depth_mm": 10, "floor_z_mm": 33, "notch_center_xy_mm": [-80.5, 90],
         "required_orientation": "hub_up"},
    ]
    complete(folder, body, collisions, 2048, .6, .80, EVA,
             {"outer_xyz": [260, 380, 50], "shell_wall": 3, "shell_floor": 3,
              "foam_xyz": [252, 372, 40], "foam_bottom_z": 3, "foam_top_z": 43,
              "outline_clearance_per_side": 1.5, "notch_diameter": 24},
             "Static 0.6 kg PP tray with EVA insert; foam seated on the 3 mm shell floor, top Z=43 mm. "
             "Pockets derived from part outlines +1.5 mm per side; the gear uses its specified "
             "circular tip envelope. Bushing has 43 x 5 over 33 x 25, full 30 mm notch; flange up "
             "alone seats to its proper floor. Gear seat 51 x 10 leaves hub up, L side-lying. "
             "Each finger notch is 24 mm on -X, at pocket depth. Where a curved/hex wall meets "
             "the notch, its overlapping circle is used; the straight L wall has a literal semicircle. "
             "Collision uses 16 boundary wall segments per round step including the notch, with "
             "pairs at concave joins merged when their union is convex; seven hex/notch walls "
             "replace six plain hex walls so the finger opening is preserved. This interprets "
             "the nominal wall counts after adding finger notches while keeping the entire tray "
             "at most 64 convex pieces. Wall sectors extend to rectangular cell edges to fill "
             "between pockets; L and floor fill are boxes. Hole polygon apothems preserve seat "
             "clearance; maximum round/notch error <1 mm. Hex notch approximation has <=0.915 mm "
             "radial error. The 1 mm insert/shell gap is bridged only by the base proxy. Uniform "
             "MuJoCo material/friction uses EVA; complete PP/EVA PBR is retained in source.",
             {"static": True, "pockets": pocket_metadata, "collision_layout": layout,
              "collision_roles": dict(zip([f"body_col_{i:02d}.obj" for i in range(len(labels))], labels)),
              "recommended_placement_m": {"origin": [.4, 0, .8],
                                           "pocket_centers_world_xy": [[.455, -.09], [.455, .09],
                                                                       [.345, -.09], [.345, .09]]}})


def main():
    global GEOMETRY_ONLY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--tray-materials-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    GEOMETRY_ONLY = args.geometry_only
    if args.tray_materials_only:
        if GEOMETRY_ONLY:
            parser.error("--tray-materials-only cannot be combined with --geometry-only")
        update_tray_materials_only()
        return
    builders = {"hex_blank": build_hex, "flanged_bushing": build_bushing,
                "l_bracket": build_bracket, "spur_gear": build_gear, "foam_kit_tray": build_tray}
    for prop_id in IDS:  # Parts always precede their outline-derived foam pockets.
        if prop_id in args.props:
            builders[prop_id]()
    if GEOMETRY_ONLY:
        (HERE / "task06_geometry_preflight.json").write_text(json.dumps(PREFLIGHT, indent=2) + "\n")


if __name__ == "__main__":
    main()
    props.finalize_assets()
