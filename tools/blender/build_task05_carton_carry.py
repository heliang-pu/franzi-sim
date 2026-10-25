"""Task 05: FEFCO 0201 cartons and a passive tabletop roller conveyor.

Blender -b --factory-startup -P this_file -- [--props ID ...] [--geometry-only].
All dimensions passed to prop_lib are millimetres. Only this task's folders
are built. --artwork-worker is an isolated Pillow entry point, without bpy.
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
OUTPUT = HERE.parents[1] / "assets" / "props"
CARTONS = {
    "carton_250x200x150": (200, 250, 150),
    "carton_300x200x200": (200, 300, 200),
    "carton_220x180x120": (180, 220, 120),
}
IDS = (*CARTONS, "roller_conveyor")


def artwork_worker():
    """Draw original generic handling marks, with an opaque white mask border."""
    from PIL import Image, ImageDraw
    from PIL.PngImagePlugin import PngInfo

    request = json.load(sys.stdin)
    image = Image.new("RGB", (1500, 460), "white")
    draw = ImageDraw.Draw(image)
    ink = "#111111"
    # Two upright arrows on a base line.
    for x in (155, 305):
        draw.polygon([(x, 60), (x - 70, 160), (x - 24, 160),
                      (x - 24, 338), (x + 24, 338), (x + 24, 160), (x + 70, 160)], fill=ink)
    draw.line((94, 369, 369, 369), fill=ink, width=22)
    # Umbrella canopy and hooked handle, using our own simple construction.
    points = [(750 + 160 * math.cos(math.pi + i * math.pi / 80),
               230 + 160 * math.sin(math.pi + i * math.pi / 80)) for i in range(81)]
    points += [(910, 230), (870, 210), (830, 230), (790, 210),
               (750, 230), (710, 210), (670, 230), (630, 210), (590, 230)]
    draw.polygon(points, fill=ink)
    draw.line((750, 220, 750, 331), fill=ink, width=20)
    draw.arc((694, 299, 750, 363), 0, 180, fill=ink, width=20)
    for x, y in ((590, 61), (900, 50), (665, 23)):
        draw.line((x + 15, y, x, y + 29), fill=ink, width=14)
    # Fragile stemmed glass, with a white zigzag crack in its bowl.
    draw.polygon([(1138, 67), (1352, 67), (1324, 207), (1280, 251),
                  (1210, 251), (1166, 207)], fill=ink)
    draw.line((1245, 238, 1245, 351), fill=ink, width=24)
    draw.line((1169, 366, 1321, 366), fill=ink, width=23)
    draw.line([(1258, 70), (1222, 126), (1266, 153), (1232, 212)], fill="white", width=15)
    info = PngInfo()
    info.add(b"sRGB", b"\x00")
    image.save(request["path"], pnginfo=info)


if __name__ == "__main__" and "--artwork-worker" in sys.argv:
    artwork_worker()
    raise SystemExit(0)

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

import prop_lib as props

PAPER = {"specular": .05, "shininess": .10, "reflectance": 0.0}
METAL = {"specular": .70, "shininess": .70, "reflectance": .05}
PAINT = {"specular": .30, "shininess": .30, "reflectance": 0.0}
PREFLIGHT = []
GEOMETRY_ONLY = False
COMMON_NOTES = (
    "Metres, Z up, front -X; identity object transforms and one common prop frame for all parts. "
    "30-degree split smoothing, weighted normals, triangulated OBJ with normals. "
    "Nonoverlapping Smart UV 66-degree atlases, 0.01 island margins. "
    "Albedo = CPU Cycles 64-sample diffuse COLOR * (0.4 + 0.6 * AO), 8-bit sRGB, "
    "without direct lighting or highlights. Full procedural Principled PBR and original label UVs "
    "are packed in source.blend. Preview: 1024 square, CPU Cycles 64, neutral grey, "
    "three area lights, front-left above. "
)


def multiply_color(material, factor, name):
    tree = material.node_tree
    socket = tree.nodes["Principled BSDF"].inputs["Base Color"]
    original = socket.links[0].from_socket
    node = tree.nodes.new("ShaderNodeMixRGB")
    node.name, node.blend_type = name, "MULTIPLY"
    node.inputs[0].default_value = 1
    tree.links.new(original, node.inputs[1])
    tree.links.new(factor, node.inputs[2])
    tree.links.new(node.outputs[0], socket)


def grain(material, scale, variation=.06, bump_mm=.025, stretch=(1, 1, 1)):
    tree = material.node_tree
    position = tree.nodes.new("ShaderNodeNewGeometry")
    vector = tree.nodes.new("ShaderNodeVectorMath")
    vector.operation = "MULTIPLY"
    vector.inputs[1].default_value = stretch
    tree.links.new(position.outputs["Position"], vector.inputs[0])
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 3
    tree.links.new(vector.outputs[0], noise.inputs["Vector"])
    factor = props._math(tree, "ADD", 1 - variation / 2,
                         props._math(tree, "MULTIPLY", noise.outputs["Fac"], variation))
    multiply_color(material, factor, "Centred fine colour grain")
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .22
    bump.inputs["Distance"].default_value = props.mm(bump_mm)
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs[0], tree.nodes["Principled BSDF"].inputs["Normal"])
    return material


def kraft_material():
    return grain(props.pbr_material("4 mm single-wall kraft liner", "#B88B5C", .85, 0,
                                     wear=.025, grime=.14), 3900, .06, .035, (1, .4, 1))


def tape_material():
    material = grain(props.pbr_material("Glossy brown 48 mm packing tape", "#8B5A2B", .30, 0),
                     1900, .055, .05, (1, .08, 1))
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["IOR"].default_value = 1.46
    bsdf.inputs["Coat Weight"].default_value = .32
    bsdf.inputs["Coat Roughness"].default_value = .2
    return material


def zinc_material():
    material = props.pbr_material("Galvanised steel #C3C8CC, zinc spangle and drawing marks",
                                  "#C3C8CC", .30, 1, grime=.12, machining="brushed")
    tree = material.node_tree
    coords = tree.nodes.new("ShaderNodeNewGeometry")
    voronoi = tree.nodes.new("ShaderNodeTexVoronoi")
    voronoi.inputs["Scale"].default_value = 1100
    tree.links.new(coords.outputs["Position"], voronoi.inputs["Vector"])
    grey = tree.nodes.new("ShaderNodeRGBToBW")
    tree.links.new(voronoi.outputs["Color"], grey.inputs[0])
    factor = props._math(tree, "ADD", .94, props._math(tree, "MULTIPLY", grey.outputs[0], .12))
    multiply_color(material, factor, "Zinc crystallites in diffuse colour")
    return material


def move(obj, location=(0, 0, 0), rotation=(0, 0, 0)):
    transform = Matrix.Translation(Vector(tuple(props.mm(v) for v in location)))
    for axis, degrees in zip("XYZ", rotation):
        transform = transform @ Matrix.Rotation(math.radians(degrees), 4, axis)
    obj.data.transform(transform)
    obj.data.update()
    return obj


def boolean(target, cutter, operation="DIFFERENCE"):
    props._select([target])
    modifier = target.modifiers.new("Actual manufactured opening", "BOOLEAN")
    modifier.operation, modifier.solver, modifier.object = operation, "EXACT", cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def bevel(obj, width):
    props._select([obj])
    modifier = obj.modifiers.new("Softened paper or sheet edge", "BEVEL")
    modifier.width, modifier.segments = props.mm(width), 3
    modifier.limit_method = "ANGLE"
    modifier.angle_limit = math.radians(30)
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def finish(obj):
    """Retain material/source-UV data while removing OBJ-precision degeneracy."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for vertex in bm.verts:
        vertex.co = tuple(round(value, 6) for value in vertex.co)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=2e-7)
    bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=2e-7)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    if not all(edge.is_manifold for edge in bm.edges):
        bm.free()
        raise ValueError(f"{obj.name}: nonmanifold visual mesh")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    props.smooth_by_angle(obj, 30)
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    modifier = obj.modifiers.new("Area-weighted manufacturing normals", "WEIGHTED_NORMAL")
    modifier.keep_sharp = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def combine(objects, name):
    for obj in objects:
        group = obj.vertex_groups.new(name=obj.name)
        group.add(list(range(len(obj.data.vertices))), 1, "REPLACE")
    obj = props.join_objects(objects, name) if len(objects) > 1 else objects[0]
    obj.name = name
    finish(obj)
    return obj


def start(prop_id):
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    folder = OUTPUT / prop_id
    folder.mkdir(parents=True, exist_ok=True)
    print("TASK05_START", prop_id, bpy.app.version_string, flush=True)
    return folder


def hide_collisions(objects):
    collection = bpy.data.collections.get("Collision proxies (hidden)")
    if collection is None:
        collection = bpy.data.collections.new("Collision proxies (hidden)")
        bpy.context.scene.collection.children.link(collection)
    for obj in objects:
        for previous in list(obj.users_collection):
            previous.objects.unlink(obj)
        collection.objects.link(obj)
        obj.hide_render = True
        obj.hide_set(True)


def image_material(name, path, uv_name):
    material = props.pbr_material(name, "#FFFFFF", .78, 0)
    tree = material.node_tree
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_name
    image = tree.nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(path), check_existing=True)
    image.image.colorspace_settings.name = "sRGB"
    image.image.use_fake_user = True
    image.extension = "EXTEND"
    tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
    tree.links.new(image.outputs["Color"], tree.nodes["Principled BSDF"].inputs["Base Color"])
    return material


def handling_material(kraft, path):
    material = kraft.copy()
    material.name = "Black generic arrows, umbrella, fragile glass printed on kraft"
    tree = material.node_tree
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "HandlingUV"
    image = tree.nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(path), check_existing=True)
    image.image.colorspace_settings.name = "sRGB"
    image.image.use_fake_user = True
    image.extension = "EXTEND"
    tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
    multiply_color(material, image.outputs["Color"], "Opaque black handling ink mask")
    return material


def wave_patch(name, xlo, xhi, y, top, material):
    """Real wavy paper medium inside the exposed 4 mm cut end of a flap."""
    steps = math.ceil((xhi - xlo) / .45)
    vertices, faces = [], []
    for index in range(steps + 1):
        x = xlo + (xhi - xlo) * index / steps
        z = top - 2 + 1.37 * math.sin(2 * math.pi * x / 4)
        vertices.extend([(x, y - .26, z - .08), (x, y + .26, z - .08),
                         (x, y + .26, z + .08), (x, y - .26, z + .08)])
        if index:
            a, b = 4 * (index - 1), 4 * index
            for side in range(4):
                after = (side + 1) % 4
                faces.append((a + side, a + after, b + after, b + side))
    faces += [(3, 2, 1, 0), tuple(4 * steps + i for i in range(4))]
    return props._mesh_object(name, vertices, faces, (0, 0, 0), 0, 3, material)


def tape_ribbon(length, top, material):
    half, thickness, fold = length / 2, .15, 50
    outer = [(-half - thickness, top - fold), (-half - thickness, top + thickness),
             (half + thickness, top + thickness), (half + thickness, top - fold)]
    inner = [(half, top - fold), (half, top), (-half, top), (-half, top - fold)]
    obj = props.extruded_profile("48 mm brown tape with two 50 mm end folds", outer + inner,
                                 48, bevel_mm=.04, segments=3, material=material)
    obj.data.transform(Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))))
    return obj


def build_carton(prop_id):
    x, y, height = CARTONS[prop_id]
    folder = start(prop_id)
    core_y, top = y - .30, height - .30
    kraft, tape = kraft_material(), tape_material()
    medium = props.pbr_material("Exposed corrugated medium and cut liners", "#C19A68", .92, 0,
                                grime=.13)
    body = props.box("4 mm FEFCO 0201 outer walls and major flaps", (x, core_y, top),
                     (0, 0, top / 2), bevel_mm=0, material=kraft)
    boolean(body, props.box("Internal air, 4 mm wall offset", (x - 8, core_y - 8, top - 8),
                            (0, 0, top / 2), bevel_mm=0))
    for z in (2, top - 2):
        boolean(body, props.box("1.5 mm gap between major flaps", (1.5, core_y - 8, 6),
                                (0, 0, z), bevel_mm=0))
    # Submillimetre local compression, rather than a perfectly rectangular silhouette.
    for sx in (-1, 1):
        for sy in (-1, 1):
            boolean(body, props.cylinder("Shallow dented folded corner", 1.8, top * .53,
                                         (sx * (x / 2 + 1.12), sy * (core_y / 2 - .65), top * .63),
                                         bevel_mm=.15, vertices=24))
    # Recess cut edges by 0.7 mm, leaving two liners and actual wavy medium.
    patches = []
    for sy in (-1, 1):
        for sx in (-1, 1):
            lo, hi = (25, x / 2 - 2) if sx == 1 else (-x / 2 + 2, -25)
            boolean(body, props.box("Open cut flute cross-section", (hi - lo, 1.4, 3.3),
                                    ((lo + hi) / 2, sy * (core_y / 2), top - 2), bevel_mm=0))
            patches.append(wave_patch("Visible wavy single-wall flap medium", lo, hi,
                                       sy * (core_y / 2 - .35), top, medium))
        boolean(body, props.box("Top flap fold score", (x - 2, .32, .24),
                                (0, sy * (core_y / 2 - 4.6), top), bevel_mm=.04))
    # An actual thin paper label, pressed into a shallow rectangular recess.
    label_center = (-x / 2 + .06, 0, height * .69)
    label_dims = (.12, 112, 43)
    boolean(body, props.box("Slightly skewed label seating", (.5, 112.3, 43.3),
                            (-x / 2, 0, height * .69), bevel_mm=.04, rotation_deg=(2, 0, 0)))
    bevel(body, .45)
    artwork = folder / "handling_icons.png"
    result = subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(__file__), "--artwork-worker"],
                            input=json.dumps({"path": str(artwork)}), text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr)
    printed = handling_material(kraft, artwork)
    print_index = len(body.data.materials)
    body.data.materials.append(printed)
    layer = body.data.uv_layers.new(name="HandlingUV")
    for face in body.data.polygons:
        center, normal = face.center, face.normal
        front = normal.x < -.65 and center.x < props.mm(-x / 2 + 1)
        side = abs(normal.y) > .65 and abs(center.y) > props.mm(core_y / 2 - 1)
        if (front or side) and props.mm(5) < center.z < props.mm(top - 5):
            face.material_index = print_index
        for loop_index in face.loop_indices:
            point = body.data.vertices[body.data.loops[loop_index].vertex_index].co
            horizontal = -point.y if front else point.x * (1 if normal.y > 0 else -1)
            layer.data[loop_index].uv = (.5 + horizontal / props.mm(132),
                                         .5 + (point.z - props.mm(height * .30)) / props.mm(40.48))
    label_path = folder / "nova_sku_label.png"
    props.make_label_png(label_path, 1568, 602, {
        "brand": "NOVA Logistics", "title": "NOVA Logistics · SKU-B0042",
        "lines": ["CARTON CARRY / SINGLE WALL / FEFCO 0201", "BATCH N26-0042  |  FICTIONAL INVENTORY"],
        "barcode": True, "seed": "nova-fictional-sku-b0042", "background": "#F4F1E7",
        "foreground": "#17191B",
    })
    label_mat = image_material("NOVA Logistics · SKU-B0042, fictional paper label", label_path, "LabelUV")
    label = props.box("NOVA SKU label skewed 2 degrees", label_dims, label_center,
                      bevel_mm=.025, segments=3, material=label_mat, rotation_deg=(2, 0, 0))
    label_uv = label.data.uv_layers.new(name="LabelUV")
    rotation = Matrix.Rotation(math.radians(-2), 3, "X")
    for loop in label.data.loops:
        local = rotation @ (label.data.vertices[loop.vertex_index].co
                            - Vector(tuple(props.mm(v) for v in label_center)))
        label_uv.data[loop.index].uv = (.5 - local.y / props.mm(112), .5 + local.z / props.mm(43))
    objects = [body, *patches, label, tape_ribbon(core_y, top, tape)]
    # The four internal minor flaps distinguish a regular slotted carton from
    # a solid block. Each has the same nominal half-short-side flap depth.
    minor_depth = x / 2 - 4
    for end in (-1, 1):
        for z in (6.02, top - 6.02):
            objects.append(props.box("Internal 4 mm minor flap", (x - 8.1, minor_depth, 4),
                                     (0, end * (core_y / 2 - 4.05 - minor_depth / 2), z),
                                     bevel_mm=.3, segments=3, material=kraft))
    for index, yy in enumerate((-.38, -.27, .29, .40)):
        objects.append(props.box("Fine tape wrinkle", (18, .5, .08),
                                 (-8 + index * 5, yy * core_y, top + .19),
                                 bevel_mm=.025, segments=3, material=tape,
                                 rotation_deg=(0, 0, (-1) ** index * 11)))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1)
    bubble = bpy.context.object
    bubble.scale = (props.mm(4), props.mm(1.6), props.mm(.12))
    props._finish(bubble, "Trapped air bubble in tape", (5, .13 * core_y, top + .18), 0, 3, tape)
    objects.append(bubble)
    visual = combine(objects, "body")
    bm = bmesh.new()
    bm.from_mesh(visual.data)
    volume = bm.calc_volume(signed=True)
    bm.free()
    shell_volume = (x * core_y * top - (x - 8) * (core_y - 8) * (top - 8)
                    + 4 * (x - 8.1) * minor_depth * 4) * 1e-9
    if not .85 < volume / shell_volume < 1.12:
        raise ValueError(f"{prop_id}: incorrect shell winding or 4 mm walls: {volume / shell_volume}")
    collider = props.box("COL nominal one-box carton", (x, y, height), (0, 0, height / 2), bevel_mm=0)
    variants = [{"name": f"mass_{mass:.1f}kg", "mass_kg": mass,
                 "part_masses_kg": {"body": mass}} for mass in (.5, 1.0, 1.5)]
    finish_asset(folder, [{"name": "body", "object": visual, "collisions": [collider],
                          "mass_kg": 1.0, "friction": .55, "material": PAPER}], variants,
                 {"nominal_xyz": [x, y, height], "size_convention": "Y x X x height",
                  "wall": 4, "major_flap_gap": 1.5, "minor_flap_nominal_depth": x / 2,
                  "tape_width": 48, "tape_end_fold": 50, "label_yz": [112, 43],
                  "label_skew_deg": 2, "maximum_corner_compression": .68},
                 "FEFCO 0201 regular slotted single-wall carton: 4 mm shell and four 4 mm inner "
                 "minor flaps; 1.5 mm gaps between the outer major flaps. A 48 mm #8B5A2B ribbon "
                 "follows the Y seam and folds 50 mm down both Y ends, with fine wrinkles and an "
                 "actual shallow trapped-air bubble. The 0.30 mm allowance for tape and bubbles "
                 "is inside the nominal external dimensions. Flap ends have real 0.7 mm-deep "
                 "cut sections with two liners and a 4 mm-pitch wavy paper medium; the 4 mm "
                 "effective board thickness is modelled as an opaque shell elsewhere. Corners "
                 "have shallow geometric compression. Black arrows, umbrella and fragile glass "
                 "are our original generic ink drawings on the front and both squeeze sides. "
                 "The fictional NOVA Logistics · SKU-B0042 paper label is skewed 2 degrees and "
                 "pressed into a 0.25 mm seat to preserve the nominal box envelope. "
                 "One full nominal box collider intentionally ignores air, flaps and the small "
                 "dents. The three masses represent invisible contents; shape and friction stay "
                 "the same. MuJoCo's single material per rigid body uses paper highlight settings; "
                 "tape gloss remains in the source PBR while its brown colour is in the atlas.",
                 {"mass_variants_kg": [.5, 1.0, 1.5], "squeeze_axis": [0, 1, 0],
                  "source_artwork": [artwork.name, label_path.name],
                  "shell_volume_m3": volume,
                  "recommended_placement_m": {"origin_x": .40, "origin_y": 0, "support_z": .85},
                  "functional_frame_offset_m": [0, 0, 0]})


def profile_along_x(name, yz_profile, length, x, material, radius=.45):
    obj = props.extruded_profile(name, yz_profile, length, bevel_mm=radius, segments=3, material=material)
    obj.data.transform(Matrix(((0, 0, 1, props.mm(x)), (1, 0, 0, 0),
                               (0, 1, 0, 0), (0, 0, 0, 1))))
    return obj


def build_conveyor():
    folder = start("roller_conveyor")
    blue = grain(props.pbr_material("Blue powder-coated 2 mm C-channel steel", "#1E4C9A", .60, 0,
                                    wear=.07, grime=.15), 1700, .06, .025)
    zinc = zinc_material()
    dark = props.pbr_material("Dark bearing shields and axle spring ends", "#373D42", .4, .8,
                              grime=.13)
    frame_objects, frame_collisions = [], []
    for sign in (-1, 1):
        yz = [(sign * yy, zz) for yy, zz in
              ((160, 0), (180, 0), (180, 40), (160, 40),
               (160, 38), (178, 38), (178, 2), (160, 2))]
        channel = profile_along_x("40x20x2 real C-channel side frame", yz, 480, 0, blue)
        for xx in range(-225, 226, 50):
            boolean(channel, props.cylinder("Axle bearing mounting bore", 3.2, 4,
                                             (xx, sign * 179, 35), bevel_mm=0, vertices=24,
                                             rotation_deg=(90, 0, 0)))
        frame_objects.append(channel)
        for yy, zz, dy, dz in ((179, 20, 2, 40), (169, 1, 18, 2), (169, 39, 18, 2)):
            frame_collisions.append(props.box("COL C-channel web or flange", (480, dy, dz),
                                               (0, sign * yy, zz), bevel_mm=0))
        for xx in (-200, 200):
            yz = [(sign * yy, zz) for yy, zz in
                  ((180, 0), (220, 0), (220, 2), (182, 2), (182, 40), (180, 40))]
            foot = profile_along_x("40x40 L-foot on tabletop", yz, 40, xx, blue, .4)
            frame_objects.append(foot)
            frame_collisions.extend([
                props.box("COL L-foot base", (40, 40, 2), (xx, sign * 200, 1), bevel_mm=0),
                props.box("COL L-foot upright", (40, 2, 38), (xx, sign * 181, 21), bevel_mm=0),
            ])
        # End-stop supports sit beyond the rollers in Y, outside their swept volume.
        frame_objects.append(props.box("Removable stop support ear", (35, 12, 2),
                                       (-242.5, sign * 164, 39), bevel_mm=.4, material=zinc))
        frame_collisions.append(props.box("COL stop support ear", (35, 12, 2),
                                           (-242.5, sign * 164, 39), bevel_mm=0))
        for xx in range(-225, 226, 50):
            frame_objects.append(props.cylinder("Stationary D6 axle and bearing spindle", 3, 23.8,
                                                (xx, sign * 170, 35), bevel_mm=.25, vertices=32,
                                                material=zinc, rotation_deg=(90, 0, 0)))
            frame_collisions.append(props.cylinder("COL fixed spindle", 3, 23.8,
                                                   (xx, sign * 170, 35), bevel_mm=0, vertices=16,
                                                   rotation_deg=(90, 0, 0)))
    frame = combine(frame_objects, "frame")
    # Make one roller with a unique UV atlas, then translate exact copies. Each
    # part keeps its own OBJ and PNG; congruent barrels can reuse one actual bake.
    barrel = props.tube("D30 x 316 galvanised hollow roller barrel", 15, 13.6, 316,
                        bevel_mm=.38, segments=3, vertices=64, material=zinc)
    move(barrel, rotation=(90, 0, 0))
    roller_objects = [barrel]
    for sign in (-1, 1):
        cap = props.tube("Pressed roller end bearing shield", 13.45, 3.2, 1.2,
                         bevel_mm=.25, segments=3, vertices=48, material=dark)
        move(cap, (0, sign * 157.3, 0), (90, 0, 0))
        roller_objects.append(cap)
    template = combine(roller_objects, "roller_00")
    props.uv_unwrap(template, "smart")
    specs = [{"name": "frame", "object": frame, "collisions": frame_collisions,
              "mass_kg": 4.0, "friction": .50, "material": PAINT}]
    for index in range(10):
        xx = -225 + 50 * index
        if index == 0:
            visual = template
        else:
            visual = bpy.data.objects.new(f"roller_{index:02d}", template.data.copy())
            bpy.context.collection.objects.link(visual)
        # Copy from an untranslated template until all clones have been made.
        specs.append({"name": f"roller_{index:02d}", "object": visual,
                      "atlas_prepared": True,
                      "collisions": [props.cylinder(f"COL roller {index:02d} 16-sided", 15, 316,
                                                     (xx, 0, 35), bevel_mm=0, vertices=16,
                                                     rotation_deg=(90, 0, 0))],
                      "mass_kg": .45, "friction": .30, "material": METAL, "parent": "frame",
                      "joint": {"type": "hinge", "parent": "frame", "axis": [0, 1, 0],
                                "anchor_m": [props.mm(xx), 0, .035],
                                "range": [-math.pi, math.pi], "limited": False, "continuous": True,
                                "damping": .001, "frictionloss": 0.0},
                      **({"atlas_copy_from": "roller_00"} if index else {})})
    for index, spec in enumerate(specs[1:]):
        move(spec["object"], (-225 + 50 * index, 0, 35))
    stop = combine([props.box("320x20x30 removable downstream stop bar", (20, 320, 30),
                              (-250, 0, 55), bevel_mm=2, segments=3, material=blue)], "end_stop")
    specs.append({"name": "end_stop", "object": stop,
                  "collisions": [props.box("COL removable end stop", (20, 320, 30),
                                            (-250, 0, 55), bevel_mm=2, segments=1)],
                  "mass_kg": .38, "friction": .50, "material": PAINT, "parent": "frame",
                  "removable": True})
    # The 480 mm frame and 20 mm outboard stop have a 500 mm fitted envelope.
    # Centre that entire initial assembly at the common bottom-centre origin.
    shift = (10, 0, 0)
    for spec in specs:
        move(spec["object"], shift)
        for collider in spec["collisions"]:
            move(collider, shift)
        if spec.get("joint"):
            spec["joint"]["anchor_m"][0] += .01
    finish_asset(folder, specs, [
        {"name": "stop_fitted", "disabled_parts": []},
        {"name": "stop_removed", "disabled_parts": ["end_stop"]},
    ], {"overall_length": 500, "frame_length": 480, "inside_width": 320,
        "roller_diameter": 30, "roller_length": 316,
        "roller_count": 10, "roller_pitch": 50, "roller_top_above_table": 50,
        "side_frame_hwt": [40, 20, 2], "side_frame_top_above_table": 40,
        "foot_base_xy": [40, 40], "end_stop_xyz": [20, 320, 30], "end_stop_top_above_rollers": 20,
        "fitted_overall_xyz": [500, 440, 70]},
        "Passive conveyor, flow -X, four actual 40x40 L-feet on the tabletop. Ten galvanised "
        "D30 x 316 hollow steel rollers, 50 mm pitch, top 50 mm; blue #1E4C9A real 40x20x2 "
        "C-channels with top at 40 mm preserve the 10 mm hand clearance. Thin sheet edges "
        "use 0.4-0.45 mm 3-segment radii so the 2 mm walls survive; the stop uses 2 mm radii. "
        "SPEC CONFLICT: a single convex collider cannot represent the 18x36 mm open C-channel "
        "within the general 1 mm rule. Each side therefore has three convex web/flange boxes, "
        "rather than one hull filling its groove. Feet, fixed axles and support ears also have "
        "individual convex proxies. Each rotating roller has exactly one 16-sided cylinder "
        "collider; its polygon vertices lie on the nominal D30 circle. "
        "The removable 320Y x 20X x 30Z stop has its top 20 mm above the rollers and is fitted "
        "immediately outside the -X end of the 480 mm side channels, so the complete fitted "
        "conveyor is exactly 500 mm long. The ten rollers occupy a 480 mm envelope including "
        "their radii. The stop is outboard of that envelope to avoid intersecting the first "
        "roller. Feet give 440 mm overall width; the internal clear width remains 320. "
        "All meshes, proxies and hinge anchors have been shifted +10 mm X together to centre "
        "the fitted assembly. Remove the end_stop part for the stop_removed variant. "
        "Continuous Y hinges use limited=false, continuous=true, damping=0.001 and frictionloss=0. "
        "The finite +/-pi range is present only for the existing schema; simulation must respect "
        "limited=false and allow unlimited rotation. Identical translated roller meshes retain "
        "identical UVs and reuse the actual roller_00 2048px Cycles bake in ten separate PNG files. "
        "Listed masses are reasonable modelling estimates because no equipment masses were specified.",
        {"flow_axis": [-1, 0, 0], "functional_frame_offset_m": [.01, 0, 0],
         "recommended_placement_m": {"frame_center_x": .40, "origin_y": 0, "support_z": .8},
         "static_parts": ["frame", "end_stop"]})


def finish_asset(folder, specs, variants, dimensions, notes, extra):
    visuals = [spec["object"] for spec in specs]
    collisions = [collider for spec in specs for collider in spec["collisions"]]
    hide_collisions(collisions)
    for spec in specs:
        visual = spec["object"]
        # Primitives already have a UVMap, but joining them does not create an
        # atlas. Re-unwrap the complete rigid body. Only explicitly prepared
        # congruent roller atlases may bypass this step.
        if not spec.get("atlas_prepared"):
            props.uv_unwrap(visual, "smart")
        for loop in visual.data.uv_layers["UVMap"].data:
            loop.uv = (.0005 + .999 * loop.uv.x, .0005 + .999 * loop.uv.y)
    preflight = {"id": folder.name, "bbox_mm": props.bbox_mm(visuals),
                 "triangles": props.triangle_count(visuals), "collision_count": len(collisions),
                 "parts": {spec["name"]: props.bbox_mm([spec["object"]]) for spec in specs}}
    PREFLIGHT.append(preflight)
    print("TASK05_GEOMETRY", json.dumps(preflight), flush=True)
    if GEOMETRY_ONLY:
        return
    parts = []
    for spec in specs:
        name, visual = spec["name"], spec["object"]
        visual.name = name
        texture = f"{name}_albedo.png"
        if spec.get("atlas_copy_from"):
            shutil.copyfile(folder / f"{spec['atlas_copy_from']}_albedo.png", folder / texture)
        else:
            visibility = [(other, other.hide_render) for other in bpy.context.scene.objects
                          if other.type == "MESH" and other != visual]
            for other, _ in visibility:
                other.hide_render = True
            visual.hide_render = False
            try:
                props.bake_albedo([visual], 2048, folder / texture)
            finally:
                for other, hidden in visibility:
                    other.hide_render = hidden
        collision_files = props.export_collision(name, spec["collisions"], folder)
        hide_collisions(spec["collisions"])
        part = {"name": name, "visual": props.export_obj(visual, folder / f"{name}.obj"),
                "texture": texture, "texture_px": 2048, "collision": collision_files,
                "mass_kg": spec["mass_kg"], "friction": spec["friction"],
                "mujoco_material": spec["material"], "parent": spec.get("parent"),
                "joint": spec.get("joint"), "source_visual_object": visual.name,
                "source_collision_objects": [obj.name for obj in spec["collisions"]]}
        for key in ("removable", "atlas_copy_from"):
            if key in spec:
                part[key] = spec[key]
        parts.append(part)
        print("TASK05_PART_EXPORTED", folder.name, name, flush=True)
    data = {"id": folder.name, "task": "carton_carry", "units": "m", "origin": "bottom_center",
            "bbox_mm": [round(value, 5) for value in props.bbox_mm(visuals)],
            "budget_class": "large" if folder.name == "roller_conveyor" else "medium",
            "triangles": props.triangle_count(visuals), "parts": parts, "variants": variants,
            "source": "source.blend", "preview": "preview.png", "dimensions_mm": dimensions,
            "notes": notes + " " + COMMON_NOTES + f"Generated with Blender {bpy.app.version_string}.",
            **extra}
    props.render_preview(visuals, folder / "preview.png")
    scene = bpy.context.scene
    scene["prop_id"] = folder.name
    scene["bake_recipe"] = "CPU Cycles 64; diffuse COLOR * mix(1, AO, 0.6); 8-bit sRGB"
    scene["dimensions_mm"] = json.dumps(dimensions)
    scene["functional_frame_offset_m"] = extra["functional_frame_offset_m"]
    scene["preview_recipe"] = "CPU Cycles 64; 512 square; grey; three lights; front-left above"
    bpy.context.preferences.filepaths.save_version = 0
    props._select(visuals)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    props.write_prop_json(data, folder / "prop.json")
    print("TASK05_COMPLETE", folder.name, data["bbox_mm"], data["triangles"], len(collisions), flush=True)


def main():
    global GEOMETRY_ONLY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    GEOMETRY_ONLY = args.geometry_only
    for prop_id in args.props:
        if prop_id in CARTONS:
            build_carton(prop_id)
        else:
            build_conveyor()
    if GEOMETRY_ONLY:
        (HERE / "task05_geometry_preflight.json").write_text(json.dumps(PREFLIGHT, indent=2) + "\n")


if __name__ == "__main__":
    main()
    props.finalize_assets()
