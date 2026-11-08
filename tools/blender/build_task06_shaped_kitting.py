"""Build benchmark_props.md sections 1, 2, 3.6 and the section 6.3 foam revision.

Blender -b --factory-startup -t 6 -P tools/blender/build_task06_shaped_kitting.py
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
# Pocket outline clearance per side, mm. 4.0 since 2026-10-25 (was 1.5): at 1.5
# the L bracket's pocket allowed about 2 degrees of yaw, too tight for a
# camera-guided robot hand.
CLEAR = 4.0
OUTPUT = props.PROPS
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
            part_folder = props.PROPS / prop_id
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
    if len(data['parts']) > 1:
        from support_task16 import bake, render, material
        visuals=[bpy.data.objects[p['source_visual_object']] for p in data['parts']]
        light=grain(props.pbr_material('Light-grey EVA foam #A9AFB6','#A9AFB6',.95,0,
                                       wear=.004,grime=.12),scale=5500,amplitude=.20,bump_mm=.055)
        light.use_fake_user=True
        copies=[]
        for obj in visuals:
            copy=obj.copy(); copy.data=obj.data.copy(); bpy.context.collection.objects.link(copy)
            for i,mat in enumerate(copy.data.materials):
                if mat.name.startswith('Black EVA'): copy.data.materials[i]=light
            copies.append(copy); obj.hide_render=True
        atlas=props.join_objects(copies,'Existing shared atlas - light variant')
        bake(atlas,2048,folder,'body_albedo_v1.png')
        bpy.data.objects.remove(atlas,do_unlink=True)
        for obj in visuals: obj.hide_render=False
        originals=[list(obj.data.materials) for obj in visuals]
        for obj in visuals:
            for i,mat in enumerate(obj.data.materials):
                if mat.name.startswith('Black EVA'): obj.data.materials[i]=light
        render(visuals,folder)
        for obj,mats in zip(visuals,originals):
            for i,mat in enumerate(mats): obj.data.materials[i]=mat
        data['preview_variant']='light_grey'
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(folder/'source.blend'),check_existing=False)
        props.write_prop_json(data,folder/'prop.json')
        print('TASK16_MATERIALS_ONLY complete; black atlas, exported geometry and UVs preserved',flush=True)
        return
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
    outline = [(-a, -b), (a, -b), (a, -b + thickness)]
    if radius > 0:
        outline += [(center[0], center[1] - radius)]
        outline += [(center[0] + radius * math.cos(-math.pi / 2 - math.pi / 2 * i / samples),
                     center[1] + radius * math.sin(-math.pi / 2 - math.pi / 2 * i / samples))
                    for i in range(1, samples + 1)]
    else:  # an offset of 4 mm or more leaves the R4 fillet a sharp inner corner
        outline += [(-a + thickness, -b + thickness)]
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
    """Section 6.3: 20 mm seats, two deep X finger slots and 3 mm lead-ins."""
    from support_task16 import (start as start16, part as part16, complete as complete16,
        box as box16, material as material16, boolean as cut16, bake as bake16,
        convex_object, subtract_convex, prism_planes)
    import numpy as np
    folder = start16("foam_kit_tray", clear_collisions=not GEOMETRY_ONLY)
    pp = material16("Grey PP shell #8A9097", "#8A9097")
    black = grain(props.pbr_material("Black EVA foam #161616", "#161616", .95, 0,
                                     wear=.004, grime=.12), scale=5500, amplitude=.20, bump_mm=.055)
    light = grain(props.pbr_material("Light-grey EVA foam #A9AFB6", "#A9AFB6", .95, 0,
                                     wear=.004, grime=.12), scale=5500, amplitude=.20, bump_mm=.055)
    shell = box16("PP shell", (260,380,50), (0,0,25), pp, 1)
    cut16(shell, box16("Shell interior", (254,374,60), (0,0,33), edge=.6))
    shellcols = [box16("PP floor", (260,380,3), (0,0,1.5), edge=0)]
    for sign in (-1,1):
        shellcols += [box16("X wall", (3,380,47), (sign*128.5,0,26.5), edge=0),
                      box16("Y wall", (254,3,47), (0,sign*188.5,26.5), edge=0)]
    parts = [part16("body", [shell], shellcols, .18)]
    pockets = []
    def rectangle(x0,x1,y0,y1): return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
    def shifted(poly, cx,cy): return [(x+cx,y+cy) for x,y in poly]
    definitions = [
        ("hex_blank",55,-90, "hexagon", 17.5),
        ("flanged_bushing",55,90, "stepped_circle",24),
        ("l_bracket",-55,-90, "side_lying_L",24),
        ("spur_gear",-55,90, "circle",28),
    ]
    for name,cx,cy,shape,radius in definitions:
        bounds = (0,126,-186,0) if cx>0 and cy<0 else (0,126,0,186) if cx>0 else (-126,0,-186,0) if cy<0 else (-126,0,0,186)
        x0,x1,y0,y1 = bounds
        # Slots overlap the cavity so their full 26 mm width is open. The L's
        # opposite X face is the inner side of its vertical leg, not its bbox.
        intervals = [(-radius-22,-radius+8),(radius-8,radius+22)]
        if name == "l_bracket": intervals = [(-46,-22),(-14,10)]
        notches = [shifted(rectangle(a,b,-13,13),cx,cy) for a,b in intervals]
        metadata = dict(part=name, center_yx_mm=[cy,cx], center_xy_mm=[cx,cy], shape=shape,
                        depth_mm=20, floor_z_mm=23, clearance_per_side_mm=4,
                        entry_chamfer_mm=3, entry_chamfer_angle_deg=45,
                        finger_notches=[dict(side="-X" if i==0 else "+X",width_y_mm=26,
                          extension_x_mm=22,depth_mm=25,floor_z_mm=18,
                          center_xy_mm=[cx+(a+b)/2,cy],bounds_xy_mm=[cx+a,cx+b,cy-13,cy+13],
                          chamfer_mm=3) for i,(a,b) in enumerate(intervals)])
        if name == "flanged_bushing":
            metadata.update(steps=[dict(diameter_mm=48,depth_mm=5,floor_z_mm=38),
                                   dict(diameter_mm=38,depth_mm=20,floor_z_mm=23)],required_orientation="flange_up")
        elif name == "hex_blank": metadata['across_flats_mm']=35
        elif name == "spur_gear": metadata.update(diameter_mm=56,required_orientation="hub_up")
        else: metadata.update(outer_yx_mm=[68,48],leg_mm=12)
        pockets.append(metadata)
        layer_z = [3,18,23,38,40,43] if name == "flanged_bushing" else [3,18,23,40,43]
        allcols = []
        for z0,z1 in zip(layer_z,layer_z[1:]):
            points=np.array([(x,y,z) for x in (x0,x1) for y in (y0,y1) for z in (z0,z1)],dtype=float)
            chunks=[points]
            cutters=[]
            if z0>=23:
                if name=="hex_blank": cutters=[shifted(offset_polygon(hex_outline(),4),cx,cy)]
                elif name=="l_bracket":
                    cutters=[shifted(rectangle(-24,-12,-34,34),cx,cy),shifted(rectangle(-24,24,-34,-22),cx,cy)]
                else:
                    r=(19 if z0<38 else 24) if name=="flanged_bushing" else 28
                    cutters=[regular_polygon(r/math.cos(math.pi/16),16,center=(cx,cy))]
            if z0>=18: cutters += notches
            for outline in cutters:
                chunks=subtract_convex(chunks,prism_planes(outline,expand=1 if z0==40 else 0))
            allcols += [convex_object(name+" collision",chunk) for chunk in chunks]
        # Continuous visual pocket; chamfers are actual tapered cutters.
        foam=box16(name+" foam cell",(x1-x0,y1-y0,40),((x0+x1)/2,(y0+y1)/2,23),black,0)
        visual_cutters=[]
        if name=="hex_blank": visual_cutters=[(shifted(offset_polygon(hex_outline(),4),cx,cy),23)]
        elif name=="l_bracket": visual_cutters=[(shifted(l_outline(4),cx,cy),23)]
        else:
            r=19 if name=="flanged_bushing" else 28
            visual_cutters=[(regular_polygon(r,64,center=(cx,cy)),23)]
            if name=="flanged_bushing": visual_cutters.append((regular_polygon(24,64,center=(cx,cy)),38))
        for outline,z0 in visual_cutters+[(n,18) for n in notches]:
            cut16(foam,props.extruded_profile("Pocket / notch",outline,60-z0,(0,0,(60+z0)/2),bevel_mm=0))
        # Counterbore covers the lower bushing outline at the top, so only the
        # outer opening gets a chamfer. All notch upper edges have the same ramp.
        chamfer_outlines=[visual_cutters[-1][0]]+notches
        for outline in chamfer_outlines:
            outer=offset_polygon(outline,3)
            cutter=loft("3 x 45-degree lead-in",[outline,outer,outer],[40,43,50],black)
            cut16(foam,cutter)
        # Subdivide only when the collision budget requires it. Each split is
        # a fixed child; visual halves meet without overlapping triangles.
        if len(allcols)<=64:
            parts.append(part16("foam_"+name,[foam],allcols,.105,parent="body",friction=.8,
                                mujoco=EVA))
        else:
            split_z = 38 if name == "flanged_bushing" else 40
            lower,upper=[],[]
            for col in allcols:
                (lower if max(v.co.z for v in col.data.vertices)<(split_z+.01)/1000 else upper).append(col)
            low=foam.copy(); low.data=foam.data.copy(); bpy.context.collection.objects.link(low)
            cut16(low,box16("Remove upper foam",(400,500,40),(0,0,split_z+20),edge=0))
            cut16(foam,box16("Remove lower foam",(400,500,50),(0,0,split_z-25),edge=0))
            assert max(len(lower),len(upper))<=64, (name,len(lower),len(upper))
            parts += [part16("foam_"+name+"_lower",[low],lower,.097,parent="body",friction=.8,mujoco=EVA),
                      part16("foam_"+name+"_rim",[foam],upper,.008,parent="body",friction=.8,mujoco=EVA)]
        print("TASK16_FOAM",name,"collision pieces",len(allcols),flush=True)
    if GEOMETRY_ONLY:
        QA.mkdir(parents=True,exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(QA/'foam_kit_tray_geometry.blend'),check_existing=False)
        return
    # All fixed sections share one 2048 atlas. A temporary joined bake mesh
    # preserves a face-to-source mapping, so UVs are identical for both colours.
    copies=[]
    for index,p in enumerate(parts):
        obj=p['obj']; copy=obj.copy(); copy.data=obj.data.copy()
        bpy.context.collection.objects.link(copy)
        tag=copy.data.attributes.new('source_part','INT','FACE')
        face=copy.data.attributes.new('source_face','INT','FACE')
        for j in range(len(copy.data.polygons)):
            tag.data[j].value=index; face.data[j].value=j
        copies.append(copy)
    for p in parts: p['obj'].hide_render=True
    atlas=props.join_objects(copies,'Shared foam tray bake atlas')
    props.uv_unwrap(atlas)
    bake16(atlas,2048,folder,'body_albedo.png')
    original=list(atlas.data.materials)
    for i,mat in enumerate(atlas.data.materials):
        if mat==black: atlas.data.materials[i]=light
    bake16(atlas,2048,folder,'body_albedo_v1.png')
    for poly in atlas.data.polygons:
        index=atlas.data.attributes['source_part'].data[poly.index].value
        face=atlas.data.attributes['source_face'].data[poly.index].value
        obj=parts[index]['obj']; target=obj.data.polygons[face]
        # The join preserves loop order and the common prop frame.
        for source_loop,target_loop in zip(poly.loop_indices,target.loop_indices):
            obj.data.uv_layers['UVMap'].data[target_loop].uv=atlas.data.uv_layers['UVMap'].data[source_loop].uv
    bpy.data.objects.remove(atlas,do_unlink=True)
    for p in parts: p['obj'].hide_render=False
    for p in parts: p['texture_override']='body_albedo.png'
    variants=[dict(name='black',textures={p['name']:'body_albedo.png' for p in parts}),
              dict(name='light_grey',textures={p['name']:'body_albedo_v1.png' for p in parts})]
    complete16(folder,parts,dict(outer_xyz=[260,380,50],shell_wall=3,shell_floor=3,
                 foam_xyz=[252,372,40],foam_bottom_z=3,foam_top_z=43,outline_clearance_per_side=4,
                 pocket_depth=20,notch_width_y=26,notch_extension_x=22,notch_depth=25,chamfer=3),
               dict(task='shaped_kitting',static=True,pockets=pockets,
                    recommended_placement_m={'origin':[.4,0,.9]}),variants=variants,
               notes='Section 6.3 rebuild. All pocket centres unchanged. Convex 16-facet circular cavity '
               'proxies are circumscribed: minimum nominal clearance preserved, maximum radius error '
               '0.55 mm. Exact 3 x 45-degree collision lead-ins. Finger slots overlap the cavity '
               'to expose their full width, extend 22 mm beyond its X rim and end at Z18. '
               'L slots follow both X faces of the vertical leg at its centre. Foam partitions '
               'are fixed children of the PP shell, each at most 64 collision pieces. '
               'Bushing depth 20 mm is measured from foam top, including its 5 mm counterbore.')

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
    from support_task16 import finalize_selected
    finalize_selected()
