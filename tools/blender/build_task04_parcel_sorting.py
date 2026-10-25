"""Task 04: deterministic, metre-scale parcel-sorting props for Blender 5.x.

Run headless with -- --props <ids>, or -- --geometry-only for a cheap preflight.
Only this task's five asset folders are written. Artwork is generated through
prop_lib.make_label_png; an isolated Pillow worker enlarges the bin sign glyph.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUTPUT = HERE.parents[1] / "assets" / "props"
IDS = ("belt_conveyor", "sort_bin", "parcel_box_s", "parcel_box_m", "parcel_pouch")
ZONES = (("A", "#D32F2F"), ("B", "#1976D2"), ("C", "#388E3C"))


def artwork_worker():
    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo

    request = json.load(sys.stdin)
    path, letter, color = request["path"], request["letter"], request["color"]
    font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    if request.get("kind") == "shipping":
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1499, 329), fill="#F8F7F2")
        draw.rounded_rectangle((31, 31, 302, 302), 9, fill=color)
        draw.text((167, 167), letter, fill="white", font=ImageFont.truetype(font_path, 202), anchor="mm")
        draw.text((350, 155), "QF Express", fill="#202326",
                  font=ImageFont.truetype(font_path, 151), anchor="lm")
        draw.text((358, 272), "FICTIONAL PARCEL NETWORK", fill="#575A5C",
                  font=ImageFont.truetype(font_path, 31), anchor="lm")
        # An original forward-arrow emblem completes the typographic logo.
        draw.polygon(((1295, 95), (1380, 155), (1295, 215), (1318, 155)), fill=color)
    else:
        image = Image.new("RGB", (1200, 800), color)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, 620)
        draw.rounded_rectangle((20, 20, 1180, 780), 28, outline="#F5F5F1", width=12)
        draw.text((600, 400), letter, fill="white", font=font, anchor="mm", stroke_width=1)
    info = PngInfo()
    info.add(b"sRGB", b"\x00")
    image.save(path, pnginfo=info)


if __name__ == "__main__" and "--artwork-worker" in sys.argv:
    artwork_worker()
    raise SystemExit(0)

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

import prop_lib as props

METAL = {"specular": .7, "shininess": .7, "reflectance": .05}
PAPER = {"specular": .05, "shininess": .1, "reflectance": 0.0}
PVC = {"specular": .15, "shininess": .2, "reflectance": 0.0}
PE = {"specular": .3, "shininess": .35, "reflectance": 0.0}
COMMON_NOTES = (
    "Metres, Z up, front -X; all meshes and proxies share the prop origin and identity transforms. "
    "Smart UV 66 degrees with 0.01 island margins; 30-degree split smoothing and exported normals. "
    "Albedo is CPU Cycles 64-sample diffuse COLOR * (0.4 + 0.6 * AO), 8-bit sRGB; "
    "no direct light or highlights. Full procedural Principled PBR and source label UVs are packed "
    "in source.blend. Colour grain is centred on nominal colour with about +/-3 percent variation. "
)


def math_node(tree, operation, *inputs):
    return props._math(tree, operation, *inputs)


def multiply_color(material, factor, name):
    tree = material.node_tree
    socket = tree.nodes["Principled BSDF"].inputs["Base Color"]
    original = socket.links[0].from_socket
    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.name, mix.blend_type = name, "MULTIPLY"
    mix.inputs[0].default_value = 1
    tree.links.new(original, mix.inputs[1])
    tree.links.new(factor, mix.inputs[2])
    tree.links.new(mix.outputs[0], socket)


def position_channels(tree):
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(geometry.outputs["Position"], separate.inputs[0])
    return separate.outputs


def noise_finish(material, scale, strength, bump_mm, stretch=(1, 1, 1)):
    tree = material.node_tree
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    vector = tree.nodes.new("ShaderNodeVectorMath")
    vector.operation = "MULTIPLY"
    vector.inputs[1].default_value = stretch
    tree.links.new(geometry.outputs["Position"], vector.inputs[0])
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 3
    noise.inputs["Roughness"].default_value = .7
    tree.links.new(vector.outputs[0], noise.inputs["Vector"])
    shade = math_node(tree, "ADD", 1 - strength / 2,
                      math_node(tree, "MULTIPLY", noise.outputs["Fac"], strength))
    multiply_color(material, shade, "Fine material grain, centred variation")
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .22
    bump.inputs["Distance"].default_value = props.mm(bump_mm)
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs[0], tree.nodes["Principled BSDF"].inputs["Normal"])
    return material


def pvc_material():
    material = props.pbr_material("Green PVC, 6 mm embossed diamond weave", "#2F7D32", .75, 0,
                                  wear=.04, grime=.10)
    tree = material.node_tree
    xyz = position_channels(tree)
    diagonal = []
    for operation in ("ADD", "SUBTRACT"):
        phase = math_node(tree, operation, xyz["X"], xyz["Y"])
        wave = math_node(tree, "ABSOLUTE",
                         math_node(tree, "SINE", math_node(tree, "MULTIPLY", phase,
                                                           2 * math.pi / props.mm(6))))
        diagonal.append(math_node(tree, "LESS_THAN", wave, .15))
    mask = math_node(tree, "MAXIMUM", *diagonal)
    multiply_color(material, math_node(tree, "ADD", .97, math_node(tree, "MULTIPLY", mask, .07)),
                   "Diamond relief colour, retained in MuJoCo")
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .3
    bump.inputs["Distance"].default_value = props.mm(.08)
    tree.links.new(mask, bump.inputs["Height"])
    tree.links.new(bump.outputs[0], tree.nodes["Principled BSDF"].inputs["Normal"])
    return material


def galvanised_material():
    material = props.pbr_material("Galvanised sheet, crystalline zinc spangle", "#AEB4BA", .30, 1,
                                  grime=.13, machining="brushed")
    tree = material.node_tree
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    voronoi = tree.nodes.new("ShaderNodeTexVoronoi")
    voronoi.inputs["Scale"].default_value = 1050
    voronoi.feature = "F1"
    tree.links.new(geometry.outputs["Position"], voronoi.inputs["Vector"])
    grey = tree.nodes.new("ShaderNodeRGBToBW")
    tree.links.new(voronoi.outputs["Color"], grey.inputs[0])
    factor = math_node(tree, "ADD", .90, math_node(tree, "MULTIPLY", grey.outputs[0], .20))
    multiply_color(material, factor, "Zinc crystallites, diffuse spangle")
    return material


def cardboard_material():
    return noise_finish(props.pbr_material("Single-wall kraft liner, worn folded corners", "#B88B5C",
                                           .85, 0, wear=.07, grime=.12), 3400, .06, .035,
                        (1, .35, 1))


def flute_material(top_height):
    material = props.pbr_material("Exposed single-wall corrugation cross-section", "#BE996A", .9, 0)
    tree = material.node_tree
    xyz = position_channels(tree)
    phase = math_node(tree, "ADD", xyz["X"], xyz["Y"])
    wave = math_node(tree, "SINE", math_node(tree, "MULTIPLY", phase, 2 * math.pi / props.mm(3.5)))
    relative_z = math_node(tree, "SUBTRACT", xyz["Z"], props.mm(top_height - 3))
    medium_z = math_node(tree, "ADD", props.mm(1.5), math_node(tree, "MULTIPLY", wave, props.mm(.95)))
    medium = math_node(tree, "LESS_THAN",
                       math_node(tree, "ABSOLUTE", math_node(tree, "SUBTRACT", relative_z, medium_z)),
                       props.mm(.16))
    liners = math_node(tree, "MAXIMUM", math_node(tree, "LESS_THAN", relative_z, props.mm(.22)),
                       math_node(tree, "GREATER_THAN", relative_z, props.mm(2.78)))
    paper = math_node(tree, "MAXIMUM", medium, liners)
    multiply_color(material, math_node(tree, "ADD", .62, math_node(tree, "MULTIPLY", paper, .52)),
                   "Two liners and a wavy medium, 3 mm cut edge / 3.5 mm flute pitch")
    return material


def tape_material():
    material = noise_finish(props.pbr_material("48 mm clear amber packing tape over kraft", "#B18A59",
                                               .30, 0), 2100, .055, .045, (1, .08, 1))
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["IOR"].default_value = 1.46
    bsdf.inputs["Coat Weight"].default_value = .30
    bsdf.inputs["Coat Roughness"].default_value = .2
    return material


def pe_material():
    material = noise_finish(props.pbr_material("Puffy white PE with stretched film wrinkles", "#ECEDEA",
                                               .42, 0, grime=.10), 1800, .06, .14, (1, .09, 1))
    material.node_tree.nodes["Principled BSDF"].inputs["IOR"].default_value = 1.48
    material.node_tree.nodes["Principled BSDF"].inputs["Coat Weight"].default_value = .12
    return material


def move(obj, location=(0, 0, 0), rotation=(0, 0, 0)):
    transform = Matrix.Translation(Vector(tuple(props.mm(v) for v in location)))
    for axis, degrees in zip("XYZ", rotation):
        transform = transform @ Matrix.Rotation(math.radians(degrees), 4, axis)
    obj.data.transform(transform)
    obj.data.update()
    return obj


def bevel(obj, width, segments=3):
    props._select([obj])
    modifier = obj.modifiers.new("Manufactured edge radii", "BEVEL")
    modifier.width, modifier.segments = props.mm(width), segments
    modifier.limit_method, modifier.angle_limit = "ANGLE", math.radians(30)
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def difference(target, cutter):
    props._select([target])
    modifier = target.modifiers.new("Real opening", "BOOLEAN")
    modifier.operation, modifier.solver, modifier.object = "DIFFERENCE", "EXACT", cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def finish(obj):
    """Round to the OBJ export precision, remove degeneracies, retain source UVs."""
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
        raise ValueError(f"{obj.name}: visual is not a closed mesh")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    for slot in {polygon.material_index for polygon in obj.data.polygons}:
        if slot >= len(obj.data.materials) or obj.data.materials[slot] is None:
            raise ValueError(f"{obj.name}: a used face has no PBR material")
    props.smooth_by_angle(obj, 30)
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    modifier = obj.modifiers.new("Area-weighted manufacturing normals", "WEIGHTED_NORMAL")
    modifier.keep_sharp, modifier.mode = True, "FACE_AREA"
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
    print("TASK04_START", prop_id, bpy.app.version_string, flush=True)
    return folder


def recenter(visuals, collisions):
    points = np.asarray([tuple(vertex.co) for obj in visuals for vertex in obj.data.vertices])
    lower, upper = points.min(axis=0), points.max(axis=0)
    offset = np.array([-(lower[0] + upper[0]) / 2, -(lower[1] + upper[1]) / 2, -lower[2]])
    for obj in [*visuals, *collisions]:
        obj.data.transform(Matrix.Translation(Vector(offset)))
        obj.data.update()
    return [round(float(value), 7) for value in offset]


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


def image_material(name, path, roughness=.8):
    material = props.pbr_material(name, "#FFFFFF", roughness, 0)
    tree = material.node_tree
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "LabelUV"
    node = tree.nodes.new("ShaderNodeTexImage")
    node.name = "Variant artwork"
    node.image = bpy.data.images.load(str(path), check_existing=True)
    node.image.colorspace_settings.name = "sRGB"
    node.image.use_fake_user = True
    node.extension = "EXTEND"
    tree.links.new(uv.outputs["UV"], node.inputs["Vector"])
    tree.links.new(node.outputs["Color"], tree.nodes["Principled BSDF"].inputs["Base Color"])
    return material, node


def apply_label_uv(obj, width, height, center, angle=0, sign=False):
    layer = obj.data.uv_layers.new(name="LabelUV")
    rotation = Matrix.Rotation(-math.radians(angle), 3, "Z")
    for loop in obj.data.loops:
        local = obj.data.vertices[loop.vertex_index].co - Vector(tuple(props.mm(v) for v in center))
        local = rotation @ local
        if sign:
            coord = (.5 - local.y / props.mm(width), .5 + local.z / props.mm(height))
        else:
            coord = (.5 - local.y / props.mm(width), .5 + local.x / props.mm(height))
        layer.data[loop.index].uv = coord


def make_shipping_labels(folder, index):
    files = {}
    for zone_index, (letter, color) in enumerate(ZONES):
        path = folder / f"label_{letter}.png"
        props.make_label_png(path, 1500, 900, {
            "brand": "QF Express", "title": "QF Express", "zone": letter, "zone_color": color,
            "tracking": f"QF20261024{index:02d}{zone_index:03d}17",
            "barcode": True, "seed": f"task04-fictional-{index}-{letter}",
            "lines": ["TO: Mira Vale - 17 Nimbus Way, Cloudford",
                      "FROM: Oren Quill - 4 Aurora Rd, Northbay",
                      "FICTIONAL ROUTE / STANDARD PARCEL"],
            "background": "#F8F7F2", "foreground": "#202326",
        })
        result = subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(__file__), "--artwork-worker"],
                                input=json.dumps({"kind": "shipping", "path": str(path),
                                                  "letter": letter, "color": color}),
                                text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr)
        files[letter] = path.name
    return files


def make_signs(folder):
    files = {}
    for letter, color in ZONES:
        path = folder / f"zone_{letter}.png"
        props.make_label_png(path, 1200, 800, {"text": letter, "background": color,
                                              "foreground": "#FFFFFF"})
        result = subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(__file__), "--artwork-worker"],
                                input=json.dumps({"path": str(path), "letter": letter, "color": color}),
                                text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr)
        files[letter] = path.name
    return files


def rounded_rectangle(width, length, radius, segments=8):
    points = []
    for x, y, start_angle in ((width / 2 - radius, length / 2 - radius, 0),
                              (-width / 2 + radius, length / 2 - radius, 90),
                              (-width / 2 + radius, -length / 2 + radius, 180),
                              (width / 2 - radius, -length / 2 + radius, 270)):
        for index in range(segments + 1):
            angle = math.radians(start_angle + index * 90 / segments)
            points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    return points


def loft(name, rings, material):
    count = len(rings[0])
    vertices = [point for ring in rings for point in ring]
    faces = [tuple(reversed(range(count))), tuple(range((len(rings) - 1) * count, len(rings) * count))]
    for row in range(len(rings) - 1):
        for col in range(count):
            next_col = (col + 1) % count
            faces.append((row * count + col, row * count + next_col,
                          (row + 1) * count + next_col, (row + 1) * count + col))
    return props._mesh_object(name, vertices, faces, (0, 0, 0), 0, 3, material)


def extrusion40(name, length, position, rotation, material):
    # Shallow 0.65 mm actual longitudinal grooves keep the two mandated box
    # colliders within 1 mm of the outer 40x40 extrusion surface.
    edge = [(20, 20), (4, 20), (4, 19.35), (-4, 19.35), (-4, 20), (-20, 20)]
    profile = []
    for quarter in range(4):
        angle = quarter * math.pi / 2
        c, s = round(math.cos(angle)), round(math.sin(angle))
        profile.extend((c * x - s * y, s * x + c * y) for x, y in edge[:-1])
    obj = props.extruded_profile(name, profile, length, bevel_mm=.5, segments=3, material=material)
    return move(obj, position, rotation)


def belt_wrap(name, y, material):
    # Half-annulus in YZ, 25 mm roller + 3 mm PVC. No duplicate end faces.
    sign = 1 if y > 0 else -1
    profile = []
    for radius, indices in ((28, range(33)), (25, reversed(range(33)))):
        for index in indices:
            angle = -math.pi / 2 + index * math.pi / 32
            profile.append((sign * radius * math.cos(angle), radius * math.sin(angle)))
    # Extruded_profile's XY -> YZ, extrusion Z -> X.
    obj = props.extruded_profile(name, profile, 150, bevel_mm=.10, segments=2, material=material)
    transform = Matrix(((0, 0, 1, 0), (1, 0, 0, props.mm(y)),
                        (0, 1, 0, props.mm(772)), (0, 0, 0, 1)))
    obj.data.transform(transform)
    return obj


def bolt(name, center, material, axis="Z", radius=3, depth=2):
    rotation = (0, 90, 0) if axis == "X" else ((90, 0, 0) if axis == "Y" else (0, 0, 0))
    return props.cylinder(name, radius, depth, center, bevel_mm=.25, vertices=24,
                          material=material, rotation_deg=rotation)


def build_conveyor():
    folder = start("belt_conveyor")
    aluminium = props.pbr_material("Brushed aluminium 4040 profiles", "#C9CDD1", .30, 1,
                                    grime=.12, machining="brushed")
    stainless = props.pbr_material("Brushed stainless guides and end plate", "#BFC3C7", .30, 1,
                                    grime=.10, machining="brushed")
    black = props.pbr_material("Blackened fixings and sensor body", "#2A2B2F", .45, .8, grime=.12)
    rubber = props.pbr_material("Levelling foot rubber", "#151515", .9, 0, grime=.15)
    motor = props.pbr_material("Gearmotor grey powder coat", "#6B7075", .6, 0, wear=.13, grime=.14)
    green = pvc_material()
    red = noise_finish(props.pbr_material("Red microprismatic reflector", "#BA2429", .23, .12),
                       3600, .06, .03)
    lens = props.pbr_material("Photo-eye optical red lens", "#781F24", .18, .1)
    frame, collisions = [], []
    for side in (-1, 1):
        frame.append(extrusion40(f"Side beam {side:+d} 40x40x800", 800, (side * 108, 0, 770),
                                  (90, 0, 0), aluminium))
        collisions.append(props.box(f"COL side beam {side:+d}", (40, 800, 40),
                                    (side * 108, 0, 770), bevel_mm=0))
        frame.append(props.box(f"Stainless guide {side:+d} inner gap 160", (2, 760, 30),
                               (side * 81, 0, 815), bevel_mm=.35, material=stainless))
        collisions.append(props.box(f"COL guide {side:+d}", (2, 760, 30),
                                    (side * 81, 0, 815), bevel_mm=0))
        for y in (-300, -100, 100, 300):
            frame.append(props.box("Guide upright bracket at 200 mm pitch", (4, 16, 42),
                                   (side * 96, y, 806), bevel_mm=.6, material=stainless))
            frame.append(props.box("Guide bracket horizontal saddle", (22, 16, 2),
                                   (side * 91, y, 790.5), bevel_mm=.4, material=stainless))
            frame.append(bolt("Guide adjustment screw", (side * 98.5, y, 811), black, "X"))
        for y in (-300, 300):
            frame.append(extrusion40("40x40 vertical leg", 700, (side * 108, y, 400),
                                      (0, 0, 0), aluminium))
            frame.append(props.cylinder("D50 x 20 levelling foot", 25, 20, (side * 108, y, 10),
                                         bevel_mm=2, material=rubber, vertices=48))
            frame.append(props.cylinder("Threaded levelling stem", 6, 35, (side * 108, y, 35),
                                         bevel_mm=.4, material=stainless, vertices=24))
            frame.append(props.hexagonal_prism("Levelling locknut", 18, 7,
                                                (side * 108, y, 38), bevel_mm=.5, material=stainless))
            frame.append(props.box("Leg mounting angle", (4, 45, 42), (side * 129, y, 735),
                                   bevel_mm=1, material=stainless))
        frame.append(extrusion40("Longitudinal brace at z200", 600, (side * 108, 0, 200),
                                  (90, 0, 0), aluminium))
    for y in (-300, 300):
        frame.append(extrusion40("Transverse brace at z200", 176, (0, y, 200),
                                  (0, 90, 0), aluminium))
    for y in (-380, 380):
        frame.append(props.cylinder("D50 end drum, 760 mm centres", 25, 170, (0, y, 772),
                                     bevel_mm=1, material=stainless, rotation_deg=(0, 90, 0)))
        frame.append(props.cylinder("Drum axle", 8, 238, (0, y, 772), bevel_mm=.5,
                                     material=black, rotation_deg=(0, 90, 0), vertices=32))
        collisions.append(props.cylinder("COL D50 drum", 25, 170, (0, y, 772), bevel_mm=0,
                                          vertices=24, rotation_deg=(0, 90, 0)))
        frame.append(belt_wrap("3 mm PVC drum wrap", y, green))
        for side in (-1, 1):
            frame.append(props.box("Flanged drum bearing block", (9, 48, 42),
                                   (side * 99, y, 769), bevel_mm=2, material=black))
            frame.append(bolt("Bearing housing screw", (side * 105, y, 781), stainless, "X", 3.5))
    frame.append(props.box("PVC return run", (150, 760, 3), (0, 0, 745.5),
                           bevel_mm=.3, material=green))
    frame.append(props.box("Stainless 160x40x3 end plate, top840", (160, 3, 40),
                           (0, -401.5, 820), bevel_mm=.6, material=stainless))
    collisions.append(props.box("COL end plate", (160, 3, 40), (0, -401.5, 820), bevel_mm=0))
    # Gearbox output is coaxial with the downstream drum. Motor runs outward +X.
    frame.append(props.box("100x80x80 reduction gearbox", (100, 80, 80), (178, -380, 752),
                           bevel_mm=2, material=motor))
    frame.append(props.cylinder("D80 x150 gearmotor finned core", 36, 150, (303, -380, 752),
                                 bevel_mm=2, material=motor, rotation_deg=(0, 90, 0)))
    for x in (238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358):
        frame.append(props.cylinder("Motor cooling rib", 40, 2, (x, -380, 752), bevel_mm=.5,
                                     material=motor, vertices=48, rotation_deg=(0, 90, 0)))
    frame.append(props.box("Motor terminal box", (35, 36, 18), (288, -380, 797),
                           bevel_mm=2, material=motor))
    frame.append(props.box("Gearmotor support bracket", (108, 5, 12), (176, -346, 707),
                           bevel_mm=1, material=stainless))
    frame.append(props.box("Photo-eye 30x20x12 at z830", (12, 30, 20), (-108, -333, 830),
                           bevel_mm=1, material=black))
    frame.append(props.box("Photo-eye red optical window", (.7, 18, 10), (-101.65, -333, 830),
                           bevel_mm=.25, material=lens))
    frame.append(props.box("Photo-eye mounting bracket", (32, 3, 43), (-109, -349, 810),
                           bevel_mm=.7, material=stainless))
    frame.append(props.box("Reflector 30x30x3", (3, 30, 30), (108, -333, 830),
                           bevel_mm=.6, material=red))
    frame.append(props.box("Reflector stainless backing", (2, 34, 34), (110.6, -333, 830),
                           bevel_mm=.8, material=stainless))
    frame_visual = combine(frame, "frame")
    belt = combine([props.box("760x150x3 green carrying surface", (150, 760, 3), (0, 0, 798.5),
                               bevel_mm=.2, material=green)], "belt")
    belt_collision = props.box("COL moving belt", (150, 760, 3), (0, 0, 798.5), bevel_mm=0)
    offset = recenter([frame_visual, belt], [*collisions, belt_collision])
    return finish_asset(folder, [
        ("frame", frame_visual, collisions, 2048, 27.0, .35, METAL, None),
        ("belt", belt, [belt_collision], 2048, .55, .60, PVC, "frame"),
    ], [], {
        "transport_nominal_length": 800, "drum_diameter": 50, "drum_centres": 760,
        "drum_width": 170, "belt_xyz": [150, 760, 3], "belt_top_z": 800,
        "side_beam_section": [40, 40], "side_beam_top_z": 790,
        "guide_inner_gap": 160, "guide_thickness": 2, "guide_top_z": 830,
        "guide_bracket_pitch": 200, "levelling_feet": [50, 20],
        "end_plate_xyz": [160, 3, 40], "end_plate_top_z": 840,
        "photo_eye_xyz": [12, 30, 20], "photo_eye_optical_z": 830,
        "gearbox_xyz": [100, 80, 80], "motor_diameter_length": [80, 150],
    }, (
        "Origin is on the floor at the centre of the complete projected equipment footprint, including "
        "the asymmetric +X motor. The conveyor mechanical centreline is therefore offset from the origin; "
        "see functional_frame_offset_m. The mutually inconsistent nominal 800 mm transport length, "
        "760 mm drum centres and D50 drums are resolved in favour of the exact component dimensions: "
        "bare drums span 810 mm and 3 mm outer PVC wraps span 816 mm. The gearbox extends farther at -Y. "
        "belt is the mandated independent 760x150x3 top run, not a translating rigid conveyor loop; "
        "the simulator should apply surface velocity toward -Y, leaving its pose fixed. Wraps and return "
        "run are decorative frame geometry. Eight collision pieces: belt, two guides, end plate, two "
        "beams and two drums. Legs, motor, brackets and optics intentionally have no collision, per the "
        "task's small proxy budget. Shallow profile grooves are 0.65 mm deep so beam box proxies stay "
        "within the 1 mm outer-surface allowance. Photo-eye dimensions are permuted from 30x20x12 to "
        "12X x30Y x20Z so its optical face points across the belt."
    ), {"functional_frame_offset_m": offset, "flow_axis": [0, -1, 0],
        "simulation": {"fixed_base": True, "belt_part": "belt", "surface_velocity_axis": [0, -1, 0]},
        "recommended_placement_m": {"origin_x": round(.455 - offset[0], 5), "origin_y": 0,
                                     "origin_z": 0, "robot_side_belt_edge_x": .380}})


def build_sort_bin():
    folder = start("sort_bin")
    zinc = galvanised_material()
    steel = props.pbr_material("Sign post brushed steel", "#BFC3C7", .3, 1, grime=.12, machining="brushed")
    # A single closed shell has neither overlapping wall faces nor a filled cavity.
    shell = props.box("200x160x150 open 1 mm bin shell", (200, 160, 150), (0, 0, 75),
                      bevel_mm=0, material=zinc)
    difference(shell, props.box("Bin interior cutter", (198, 158, 152), (0, 0, 77), bevel_mm=0))
    bevel(shell, .28)
    lip_center = (-100 - 15 * math.cos(math.pi / 6), 0, 157.5)
    lip = props.box("30 mm entry ramp, rising outward 30 degrees", (30, 160, 1), lip_center,
                    bevel_mm=.18, material=zinc, rotation_deg=(0, 30, 0))
    objects = [shell, lip,
               props.cylinder("D20 sign post, height300", 10, 299, (88, 55, 150.5),
                              bevel_mm=1, material=steel),
               props.box("Sign attachment lower", (32, 3, 12), (98, 55, 274),
                         bevel_mm=.7, material=steel),
               props.box("Sign attachment upper", (32, 3, 12), (98, 55, 295),
                         bevel_mm=.7, material=steel)]
    for z in (274, 295):
        objects.append(bolt("Sign post clamp screw", (109, 55, z), steel, "X"))
    sign_files = make_signs(folder)
    ink, image_node = image_material("Zone sign painted front, variant A", folder / sign_files["A"], .60)
    sign = props.box("120x80 zone sign panel", (3, 120, 80), (112.5, 0, 300),
                     bevel_mm=.6, material=steel)
    sign.data.materials.append(ink)
    for face in sign.data.polygons:
        if face.normal.x < -.9:
            face.material_index = 1
    apply_label_uv(sign, 120, 80, (112.5, 0, 300), sign=True)
    body, sign = combine(objects, "body"), combine([sign], "sign")
    collisions = [props.box("COL bin bottom", (200, 160, 1), (0, 0, .5), bevel_mm=0),
                  props.box("COL bin -X wall", (1, 160, 149), (-99.5, 0, 75.5), bevel_mm=0),
                  props.box("COL bin +X wall", (1, 160, 149), (99.5, 0, 75.5), bevel_mm=0),
                  props.box("COL bin -Y wall", (198, 1, 149), (0, -79.5, 75.5), bevel_mm=0),
                  props.box("COL bin +Y wall", (198, 1, 149), (0, 79.5, 75.5), bevel_mm=0),
                  props.box("COL 30-degree entry lip", (30, 160, 1), lip_center,
                            bevel_mm=0, rotation_deg=(0, 30, 0))]
    offset = recenter([body, sign], collisions)
    return finish_asset(folder, [
        ("body", body, collisions, 2048, .77, .30, METAL, None),
        ("sign", sign, [], 2048, .03, .30, METAL, "body"),
    ], [("sign", image_node, sign_files)], {
        "bin_body_xyz": [200, 160, 150], "wall": 1, "entry_lip_width": 30,
        "entry_lip_angle_from_horizontal_deg": 30, "post_diameter": 20, "post_top_z": 300,
        "sign_yz": [120, 80], "sign_center_z": 300,
    }, (
        "A/B/C differ only in the sign texture (A red, B blue, C green); the body, post, geometry and "
        "mass stay identical. The 200x160x150 dimensions refer to the open bin body; the entry lip, "
        "post and 120x80 sign enlarge its total bbox. Lip width is measured along the 30 mm sloping "
        "sheet, rising 30 degrees above horizontal toward -X to guide a parcel into the bin. All "
        "parts are recentered together to the complete bottom-bbox centre, not independently. "
        "The bin remains open; exactly six proxies represent floor, four walls and entry lip. "
        "The post/sign are decorative and intentionally noncolliding under the prescribed six-piece "
        "budget. The 0.8 kg total is the specified static effective mass."
    ), {"functional_frame_offset_m": offset, "source_artwork": sign_files,
        "simulation": {"fixed_base": True},
        "recommended_placement_m": {"origin_x": .40, "origin_y_options": [-.30, 0, .30],
                                     "support_z": .8}})


def tape_ribbon(width, length, height, material):
    y, thickness, fold = length / 2, .12, 25
    # Closed constant-thickness ribbon around the two short ends and top.
    outer = [(-y - thickness, height - fold), (-y - thickness, height + thickness),
             (y + thickness, height + thickness), (y + thickness, height - fold)]
    inner = [(y, height - fold), (y, height), (-y, height), (-y, height - fold)]
    obj = props.extruded_profile("48 mm tape, top and end folds", outer + inner, width,
                                 bevel_mm=.035, segments=2, material=material)
    transform = Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    obj.data.transform(transform)
    return obj


def add_tape_wrinkles(objects, length, height, material):
    # Thin sealed bubbles, below the shipping-label surface, all inside nominal bbox.
    for index, y in enumerate((-length * .38, -length * .31, length * .30, length * .39)):
        x = -9 + index * 5.5
        objects.append(props.box("Tape fine wrinkle and trapped bubble", (18, .35, .06),
                                 (x, y, height + .14), bevel_mm=.025, segments=2,
                                 material=material, rotation_deg=(0, 0, (-1) ** index * 9)))


def parcel_label(folder, files, length, top, angle):
    material, node = image_material("QF Express paper shipping label A", folder / files["A"])
    center = (0, 0, top - .08)
    obj = props.box("Shipping label, 2-degree skew", (60, length, .16), center,
                    bevel_mm=.025, segments=2, material=material, rotation_deg=(0, 0, angle))
    apply_label_uv(obj, length, 60, center, angle=angle)
    return obj, node


def build_carton(prop_id):
    small = prop_id == "parcel_box_s"
    width, length, height = (75, 140, 60) if small else (80, 180, 80)
    folder = start(prop_id)
    core_y, core_z = length - .24, height - .28
    kraft, cut, tape = cardboard_material(), flute_material(core_z), tape_material()
    body = props.box("3 mm single-wall corrugated carton", (width, core_y, core_z),
                      (0, 0, core_z / 2), bevel_mm=0, material=kraft)
    # The real inner shell is retained even though the parcel collision is one box.
    difference(body, props.box("Carton internal air", (width - 6, core_y - 6, core_z - 6),
                                (0, 0, core_z / 2), bevel_mm=0))
    # Deliberately overlap the inner air by 0.5 mm. A merely tangent cutter
    # can leave a disconnected inner shell whose normals are later inverted.
    difference(body, props.box("1 mm top flap seam", (1, core_y - 6, 5),
                                (0, 0, core_z - 1), bevel_mm=0))
    # A very shallow end crease and corner dents are geometry, below 1 mm.
    for side in (-1, 1):
        difference(body, props.box("Top flap end score", (width - 2, .35, .28),
                                    (0, side * (core_y / 2 - 3.2), core_z - .02), bevel_mm=.06))
        dent = props.cylinder("Corner compression tool", 1.4, core_z * .52,
                               (side * (width / 2 + 1.0), -core_y / 2 + .3, core_z * .68),
                               bevel_mm=.2, vertices=24)
        difference(body, dent)
    # Split the outer side faces at the flap layer: the 3 mm cut cross-section
    # remains visible at both short ends outside the 48 mm tape strip.
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
                          plane_co=(0, 0, props.mm(core_z - 3)), plane_no=(0, 0, 1),
                          dist=1e-8, clear_inner=False, clear_outer=False)
    bm.to_mesh(body.data)
    bm.free()
    body.data.update()
    bevel(body, .40)
    cut_index = len(body.data.materials)
    body.data.materials.append(cut)
    for polygon in body.data.polygons:
        center = polygon.center
        if abs(center.x) < props.mm(1.1) and center.z > props.mm(core_z - 3.3):
            polygon.material_index = cut_index
        if (abs(center.y) > props.mm(core_y / 2 - .8) and center.z > props.mm(core_z - 3.1)
                and abs(polygon.normal.y) > .5):
            polygon.material_index = cut_index
    objects = [body, tape_ribbon(48, core_y, core_z, tape)]
    # Actual wavy paper at the exposed ends of the flap seams: liner + medium.
    for side in (-1, 1):
        rings = []
        for x in (side * .45, side * .65):
            ring = []
            for index in range(81):
                y = -(core_y - 6) / 2 + (core_y - 6) * index / 80
                z = core_z - 1.5 + .95 * math.sin(2 * math.pi * y / 3.5)
                ring.append((x, y, z))
            ring += [(x, y, z - .15) for _, y, z in reversed(ring)]
            rings.append(ring)
        objects.append(loft("Exposed wavy corrugated medium in cut flap", rings, cut))
    add_tape_wrinkles(objects, core_y, core_z, tape)
    files = make_shipping_labels(folder, 1 if small else 2)
    label, node = parcel_label(folder, files, 100, height, 2)
    objects.append(label)
    visual = combine(objects, "body")
    bm = bmesh.new()
    bm.from_mesh(visual.data)
    wall_volume = bm.calc_volume(signed=True)
    bm.free()
    nominal_wall_volume = (width * length * height - (width - 6) * (length - 6) * (height - 6)) * 1e-9
    if not .8 < wall_volume / nominal_wall_volume < 1.2:
        raise ValueError(f"{prop_id}: incorrect internal-cavity winding, volume={wall_volume}")
    print("TASK04_CARTON_WALL_VOLUME", prop_id, wall_volume, flush=True)
    collision = props.box("COL parcel one box", (width, length, height), (0, 0, height / 2), bevel_mm=0)
    offset = recenter([visual], [collision])
    return finish_asset(folder, [
        ("body", visual, [collision], 2048, .25 if small else .35, .60, PAPER, None),
    ], [("body", node, files)], {
        "nominal_xyz": [width, length, height], "corrugated_wall": 3, "flap_gap": 1,
        "tape_width": 48, "tape_end_fold": 25, "label_yx": [100, 60], "label_skew_deg": 2,
    }, (
        "Closed single-wall corrugated carton with 3 mm liner/medium wall, a real 1 mm flap gap, "
        "visible wavy cut flute medium, shallow end scores and submillimetre corner compression. "
        "48 mm tape wraps over the top and 25 mm down both ends with procedural wrinkles, thin "
        "physical bubbles and a coated PBR finish. Clear tape over kraft is flattened into its "
        "amber/kraft composite colour for the opaque MuJoCo atlas; alpha would be ignored. "
        "The top face label is 100x60 mm and skewed 2 degrees. Nominal outer dimensions include "
        "the tape and paper thickness. A/B/C change only the QF Express shipping-label artwork. "
        "Logo, tracking identifiers, decorative random bars, people and addresses are fictional. "
        "One full nominal box collision intentionally omits internal air and the flap seam."
    ), {"functional_frame_offset_m": offset, "source_artwork": files,
        "mass_range_kg": [.15, .35] if small else [.20, .50],
        "recommended_placement_m": {"origin_x": .40, "origin_y": .16, "support_z": .8},
        "grasp_width_mm": width})


def pouch_seal(material):
    outer = rounded_rectangle(80, 200, 9)
    inner = rounded_rectangle(76, 196, 8)
    count = len(outer)
    vertices = [(x, y, z) for z in (17.1, 17.9) for ring in (outer, inner) for x, y in ring]
    faces = []
    for index in range(count):
        j = (index + 1) % count
        faces.extend([(index, j, j + 2 * count, index + 2 * count),
                      (index + count, index + 3 * count, j + 3 * count, j + count),
                      (index, index + count, j + count, j),
                      (index + 2 * count, j + 2 * count, j + 3 * count, index + 3 * count)])
    return props._mesh_object("Continuous 2 mm heat-sealed edge flange", vertices, faces,
                              (0, 0, 0), .08, 2, material)


def build_pouch():
    folder = start("parcel_pouch")
    film = pe_material()
    seal = noise_finish(props.pbr_material("Compressed PE thermal welds", "#DEE1DE", .5, 0),
                         4400, .06, .045, (1, .12, 1))
    rings = []
    for z, width, length, radius in ((0, 66, 186, 10), (1, 70, 190, 11), (4, 75, 195, 12),
                                     (10, 78, 198, 12), (17.5, 78.5, 198.5, 11),
                                     (25, 78, 198, 12), (31, 76, 196, 12),
                                     (34.5, 72, 192, 11), (34.84, 70, 190, 10)):
        outline = rounded_rectangle(width, length, radius, 10)
        ring = []
        for index, (x, y) in enumerate(outline):
            # Geometric folds fade at the flat label support and bottom.
            envelope = math.sin(math.pi * z / 34.84) ** 2
            wrinkle = .30 * envelope * math.sin(index * 2.1 + z * .55)
            ring.append((x * (1 + wrinkle / width), y * (1 + wrinkle / length), z))
        rings.append(ring)
    pillow = loft("Puffy PE bag, rounded and lightly wrinkled", rings, film)
    # The broad top is flat enough to support the physical paper label.
    bevel(pillow, .18, 2)
    files = make_shipping_labels(folder, 3)
    label, node = parcel_label(folder, files, 90, 35, -2)
    visual = combine([pillow, pouch_seal(seal), label], "body")
    collision = props.box("COL prescribed 196x76x33 pouch box", (76, 196, 33),
                           (0, 0, 17.5), bevel_mm=0)
    offset = recenter([visual], [collision])
    return finish_asset(folder, [
        ("body", visual, [collision], 2048, .20, .50, PE, None),
    ], [("body", node, files)], {
        "nominal_xyz": [80, 200, 35], "collision_xyz": [76, 196, 33],
        "heat_seal_width": 2, "heat_seal_thickness": .8,
        "label_yx": [90, 60], "label_skew_deg": -2,
    }, (
        "Puffy white PE mailer with a rounded, asymmetrically wrinkled pillow shell and a continuous "
        "2 mm compressed heat-sealed perimeter flange at mid-height. A/B/C change only the 90x60 mm "
        "QF Express label artwork, skewed -2 degrees. Logo, tracking numbers, decorative random bars "
        "and addresses are fictional. The explicitly required single 196Y x76X x33Z box collider "
        "is centred at z17.5 mm. It necessarily protrudes more than the generic 1 mm tolerance "
        "near the curved pillow corners and omits the thin thermal flange; this task-specific "
        "box-proxy requirement takes precedence over a close-fitting convex decomposition."
    ), {"functional_frame_offset_m": offset, "source_artwork": files, "mass_range_kg": [.10, .30],
        "recommended_placement_m": {"origin_x": .40, "origin_y": -.16, "support_z": .8},
        "grasp_width_mm": 80})


GEOMETRY_ONLY = False
PREFLIGHT = []


def finish_asset(folder, specs, variant_specs, dimensions, notes, extra):
    visuals = [item[1] for item in specs]
    all_collisions = [collision for item in specs for collision in item[2]]
    hide_collisions(all_collisions)
    preflight = {"id": folder.name, "bbox_mm": props.bbox_mm(visuals),
                 "triangles": props.triangle_count(visuals), "collision_count": len(all_collisions),
                 "parts": {name: props.bbox_mm([visual]) for name, visual, *_ in specs}}
    PREFLIGHT.append(preflight)
    print("TASK04_GEOMETRY", json.dumps(preflight), flush=True)
    if GEOMETRY_ONLY:
        return
    parts = []
    for name, visual, collisions, size, mass, friction, material, parent in specs:
        visual.name = name
        props.uv_unwrap(visual, "smart")
        visibility = [(other, other.hide_render) for other in bpy.context.scene.objects
                      if other.type == "MESH" and other != visual]
        for other, _ in visibility:
            other.hide_render = True
        visual.hide_render = False
        try:
            props.bake_albedo([visual], size, folder / f"{name}_albedo.png")
            for variant_part, image_node, files in variant_specs:
                if variant_part != name:
                    continue
                for variant_index, (letter, _) in enumerate(ZONES[1:], 1):
                    image_node.image = bpy.data.images.load(str(folder / files[letter]), check_existing=True)
                    image_node.image.colorspace_settings.name = "sRGB"
                    image_node.image.use_fake_user = True
                    props.bake_albedo([visual], size, folder / f"{name}_albedo_v{variant_index}.png")
                image_node.image = bpy.data.images.load(str(folder / files["A"]), check_existing=True)
        finally:
            for other, hidden in visibility:
                other.hide_render = hidden
        collision_files = props.export_collision(name, collisions, folder)
        hide_collisions(collisions)
        parts.append({"name": name, "visual": props.export_obj(visual, folder / f"{name}.obj"),
                      "texture": f"{name}_albedo.png", "texture_px": size,
                      "collision": collision_files, "mass_kg": mass, "friction": friction,
                      "mujoco_material": material, "parent": parent, "joint": None,
                      "source_visual_object": visual.name,
                      "source_collision_objects": [obj.name for obj in collisions]})
    variants = []
    if variant_specs:
        for index, (letter, color) in enumerate(ZONES):
            textures = {name: f"{name}_albedo{'_v' + str(index) if index else ''}.png"
                        for name, _, _ in variant_specs}
            variants.append({"name": letter, "textures": textures, "zone_color": color})
    data = {"id": folder.name, "task": "parcel_sorting", "units": "m", "origin": "bottom_center",
            "bbox_mm": [round(value, 5) for value in props.bbox_mm(visuals)],
            "budget_class": "large" if folder.name == "belt_conveyor" else "medium",
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
    scene["preview_recipe"] = "CPU Cycles 64; 512x512; neutral grey; three area lights; front-left above"
    bpy.context.preferences.filepaths.save_version = 0
    props._select(visuals)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    props.write_prop_json(data, folder / "prop.json")
    print("TASK04_COMPLETE", folder.name, data["bbox_mm"], data["triangles"],
          len(all_collisions), flush=True)


def main():
    global GEOMETRY_ONLY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    GEOMETRY_ONLY = args.geometry_only
    for prop_id in args.props:
        if prop_id == "belt_conveyor":
            build_conveyor()
        elif prop_id == "sort_bin":
            build_sort_bin()
        elif prop_id == "parcel_pouch":
            build_pouch()
        else:
            build_carton(prop_id)
    if GEOMETRY_ONLY:
        (HERE / "task04_geometry_preflight.json").write_text(json.dumps(PREFLIGHT, indent=2) + "\n")


if __name__ == "__main__":
    main()
    props.finalize_assets()
