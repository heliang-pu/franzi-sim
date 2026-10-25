"""Build task 2 only, using the shared Blender 5.x prop pipeline.

Blender -b --factory-startup -P tools/blender/build_task02_panel_operation.py
    -- --props panel_devices control_console

The device library is intentionally NOT an assembly: every mesh is exported
in its own mounting-face frame. Console variants contain complete part lists
in the console frame, including transformed joints and collision geometry.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bmesh
import bpy
from mathutils import Matrix, Vector

import prop_lib as props


OUTPUT = props.ROOT / "assets" / "props"
C = 2 / math.sqrt(5)
S = 1 / math.sqrt(5)
SLOPE_LENGTH = math.sqrt(200**2 + 100**2)
BASE_X = -6.0  # Rear gland extends 12 mm: total visual bbox is centred at X=0.
ROTATION = Matrix(((0, C, -S), (-1, 0, 0), (0, S, C)))
NORMAL = Vector((-S, 0, C))
LEGENDS = ("START", "STOP", "AUTO", "MAN", "CYCLE START", "FEED HOLD")
ABS_MJ = {"specular": 0.30, "shininess": 0.40, "reflectance": 0.0}
METAL_MJ = {"specular": 0.70, "shininess": 0.70, "reflectance": 0.05}
SHEET_MJ = {"specular": 0.30, "shininess": 0.30, "reflectance": 0.0}
RECIPE = "64-sample CPU Cycles DIFFUSE COLOR * (0.4 + 0.6 * AO); 8-bit sRGB; no direct/highlight bake."


def log(*args):
    print("TASK02", *args, flush=True)


def rounded_rectangle(width, height, radius, segments=6):
    points = []
    for cx, cy, start in ((width / 2 - radius, height / 2 - radius, 0),
                          (-width / 2 + radius, height / 2 - radius, 90),
                          (-width / 2 + radius, -height / 2 + radius, 180),
                          (width / 2 - radius, -height / 2 + radius, 270)):
        for step in range(segments + 1):
            a = math.radians(start + 90 * step / segments)
            points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return points


def mesh(name, vertices, faces, material=None):
    return props._mesh_object(name, vertices, faces, (0, 0, 0), 0, 3, material)


def loft(name, bottom, top, material=None):
    n = len(bottom)
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    faces.extend((i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n))
    return mesh(name, [*bottom, *top], faces, material)


def revolve(name, profile, material=None, sectors=64):
    verts, rings, faces = [], [], []
    for radius, z in profile:
        if radius == 0:
            rings.append([len(verts)])
            verts.append((0, 0, z))
        else:
            rings.append(list(range(len(verts), len(verts) + sectors)))
            verts.extend((radius * math.cos(i * math.tau / sectors),
                          radius * math.sin(i * math.tau / sectors), z) for i in range(sectors))
    for a, b in zip(rings, rings[1:] + rings[:1]):
        if len(a) == len(b) == 1:
            continue
        for i in range(sectors):
            j = (i + 1) % sectors
            if len(a) == 1:
                faces.append((a[0], b[j], b[i]))
            elif len(b) == 1:
                faces.append((a[i], a[j], b[0]))
            else:
                faces.append((a[i], a[j], b[j], b[i]))
    return mesh(name, verts, faces, material)


def boolean(obj, cutter, operation="DIFFERENCE"):
    if not cutter.data.materials and obj.data.materials:
        cutter.data.materials.append(obj.data.materials[0])
    props._select([obj])
    mod = obj.modifiers.new(operation + " " + cutter.name, "BOOLEAN")
    mod.operation, mod.solver, mod.object = operation, "EXACT", cutter
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def bevel(obj, width, segments=3, angle=30):
    props._select([obj])
    mod = obj.modifiers.new("Manufactured edge rounds", "BEVEL")
    mod.width, mod.segments = props.mm(width), segments
    mod.limit_method, mod.angle_limit = "ANGLE", math.radians(angle)
    mod.harden_normals = True
    bpy.ops.object.modifier_apply(modifier=mod.name)


def clean(obj):
    """Remove slivers before Blender's six-decimal-place OBJ writer does."""
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for v in bm.verts:
        v.co = tuple(round(a, 6) for a in v.co)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=2e-7)
    bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=2e-7)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    seen, bad = set(), []
    for f in bm.faces:
        key = frozenset(f.verts)
        if key in seen or f.calc_area() <= 2e-18:
            bad.append(f)
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
    # Keep planar faces exactly planar, particularly around Boolean holes.
    normals = [tuple(n.vector) for n in obj.data.corner_normals]
    planar = (Vector((0, 0, 1)), Vector((1, 0, 0)), Vector((0, 1, 0)), NORMAL)
    for p in obj.data.polygons:
        if any(abs(p.normal.dot(axis)) > 0.999999 for axis in planar):
            for i in p.loop_indices:
                normals[i] = tuple(p.normal)
    obj.data.normals_split_custom_set(normals)
    return obj


def rr_box(name, width, height, depth, radius, material=None, location=(0, 0, 0), edge=0.3):
    return props.extruded_profile(name, rounded_rectangle(width, height, radius), depth,
                                  location_mm=location, bevel_mm=edge, segments=3, material=material)


def pose(u, v, height=1.5):
    return Vector((BASE_X - 100 + C * v, u, 60 + S * v)) + NORMAL * height


def transform(obj, position, rotation=ROTATION):
    obj.matrix_world = rotation.to_4x4()
    obj.matrix_world.translation = Vector(position) * 0.001
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def duplicate(obj, name):
    copied = obj.copy()
    copied.data = obj.data.copy()
    copied.name = name
    bpy.context.collection.objects.link(copied)
    copied.hide_set(False)
    copied.hide_render = False
    return copied


def hide(objects):
    for obj in objects:
        obj.hide_render = True
        obj.hide_set(True)


def isolated_bake(obj, size, path):
    log("BAKE_START", obj.name, size, Path(path).name)
    for other in bpy.context.scene.objects:
        if other.type == "MESH":
            other.hide_render = other != obj
    obj.hide_set(False)
    props.bake_albedo([obj], size, path)
    log("BAKE_DONE", Path(path).name)


def label_material(folder, stem, text, background="#FFFFFF", foreground="#101010", aspect=3):
    path = folder / (stem + "_print.png")
    # A single-cell checker requests the shared painter's centred text mode.
    props.make_label_png(path, 1536, round(1536 / aspect), {
        "text": text, "background": background, "foreground": foreground,
        "checker": True, "checker_cells": 1,
    })
    mat = props.pbr_material(stem + " printed enamel", background, 0.55, 0, wear=0.025, grime=0.12)
    tree = mat.node_tree
    image = tree.nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(path), check_existing=True)
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "PrintUV"
    tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
    rgb = next(n for n in tree.nodes if n.type == "RGB")
    for link in list(rgb.outputs[0].links):
        tree.links.new(image.outputs["Color"], link.to_socket)
    if stem.startswith("legend_"):
        legend_ink_mapping(mat)
    return mat


def legend_ink_mapping(material):
    """Leave a blank upper strip under the bezel at the mandated 15 mm offset."""
    tree = material.node_tree
    if tree.nodes.get("Legend ink vertical scale"):
        return
    image = next(n for n in tree.nodes if n.type == "TEX_IMAGE")
    vector = image.inputs["Vector"].links[0].from_socket
    scale = tree.nodes.new("ShaderNodeVectorMath")
    scale.name, scale.operation = "Legend ink vertical scale", "MULTIPLY"
    tree.links.new(vector, scale.inputs[0])
    scale.inputs[1].default_value = (1, 0.625, 1)
    offset = tree.nodes.new("ShaderNodeVectorMath")
    offset.name, offset.operation = "Legend lower ink band", "ADD"
    tree.links.new(scale.outputs[0], offset.inputs[0])
    offset.inputs[1].default_value = (0, 0.3325, 0)
    tree.links.new(offset.outputs[0], image.inputs["Vector"])


def print_uv(obj, width, height):
    uv = obj.data.uv_layers.new(name="PrintUV")
    for loop in obj.data.loops:
        v = obj.data.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (v.x / props.mm(width) + 0.5, v.y / props.mm(height) + 0.5)


def hand_material(name, color, roughness=0.45):
    mat = props.pbr_material(name, color, roughness, 0, wear=0.06, grime=0.15)
    tree = mat.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    original = bsdf.inputs["Base Color"].links[0].from_socket
    coord = tree.nodes.new("ShaderNodeTexCoord")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 420
    noise.inputs["Detail"].default_value = 3
    tree.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    shade = props._math(tree, "ADD", 0.94, props._math(tree, "MULTIPLY", noise.outputs["Fac"], 0.06))
    mul = tree.nodes.new("ShaderNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs[0].default_value = 1
    tree.links.new(original, mul.inputs[1])
    tree.links.new(shade, mul.inputs[2])
    tree.links.new(mul.outputs[0], bsdf.inputs["Base Color"])
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = props.mm(0.012)
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def part_record(name, visual, collision, mass, friction=0.4, material=None, parent=None, joint=None,
                texture=None, size=1024, **extra):
    return {"name": name, "visual": visual, "texture": texture or name + "_albedo.png",
            "texture_px": size, "collision": collision, "mass_kg": mass, "friction": friction,
            "mujoco_material": dict(material or ABS_MJ), "parent": parent, "joint": joint, **extra}


def save_source(folder, metadata, visible):
    # Validate the final quantized proxies, not the pre-export float32 hulls.
    note = " Convex proxy topology is rebuilt from unchanged one-micrometre OBJ points after export rounding."
    if note not in metadata["notes"]:
        metadata["notes"] += note
    props.write_prop_json(metadata, folder / "prop.json")
    result = subprocess.run([str(props.VALIDATOR_PYTHON), "-B",
                             str(Path(__file__).with_name("convexify_task02_collisions.py")),
                             "--props", metadata["id"]], capture_output=True, text=True)
    log(result.stdout.strip())
    if result.returncode:
        raise RuntimeError("Quantized collision export failed: " + result.stderr)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in visible
            obj.hide_set(obj not in visible)
    bpy.context.scene["prop_id"] = metadata["id"]
    bpy.context.scene["bake_recipe"] = RECIPE
    bpy.context.scene["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
    bpy.context.preferences.filepaths.save_version = 0
    props._select(visible)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    props.write_prop_json(metadata, folder / "prop.json")


def render_library_preview(objects, folder):
    """Stage temporary copies without changing any exported component frame."""
    display = []
    for index, (name, original) in enumerate(objects.items()):
        row, col = divmod(index, 3)
        obj = duplicate(original, "Display " + name)
        bottom = min(v.co.z for v in obj.data.vertices)
        obj.location = (props.mm((row - 1.5) * 72), props.mm((col - 1) * 76), -bottom)
        display.append(obj)
    hide(objects.values())
    # Matrix/bbox caches otherwise retain the prototypes' overlaid zero frames.
    bpy.context.view_layer.update()
    props.render_preview(display, folder / "preview.png")
    for obj in display:
        bpy.data.objects.remove(obj, do_unlink=True)


def build_panel_devices():
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    folder = OUTPUT / "panel_devices"
    folder.mkdir(parents=True, exist_ok=True)
    log("START", folder.name)
    parts, objects, proxies = [], {}, {}

    def add(name, obj, collision, mass, friction=0.4, material=None, parent=None, joint=None, **extra):
        obj.name = name
        clean(obj)
        props.uv_unwrap(obj)
        isolated_bake(obj, 1024, folder / (name + "_albedo.png"))
        visual = props.export_obj(obj, folder / (name + ".obj"))
        for i, p in enumerate(collision):
            p.name = f"COL__{name}__{i:02d}"
        files = props.export_collision(name, collision, folder)
        parts.append(part_record(name, visual, files, mass, friction, material, parent, joint,
                                 source_collision_objects=[p.name for p in collision],
                                 frame={"origin": "mounting_face_center", "press_direction": [0, 0, -1]},
                                 **extra))
        objects[name], proxies[name] = obj, collision
        hide([obj, *collision])

    chrome = props.pbr_material("Bright chrome bezel", "#BFC3C7", 0.30, 1, grime=0.12,
                                machining="turned")
    ring = props.tube("bezel", 14.75, 12.15, 3, (0, 0, 1.5), bevel_mm=0.3,
                     segments=2, vertices=48, material=chrome)
    ring_col = props.ring_segments("Bezel collision", 12.15, 14.75, 3, count=16,
                                  location_mm=(0, 0, 1.5))
    add("bezel", ring, ring_col, 0.008, material=METAL_MJ,
        dimensions_mm={"diameter": 29.5, "height": 3, "clear_bore": 24.3})

    for color, hex_color in (("green", "#2E9E44"), ("red", "#D32F2F"), ("yellow", "#F2B705")):
        name = "btn_" + color
        mat = hand_material(name + " handled rubber-faced ABS", hex_color)
        # Eight mm total cap envelope, six above bezel. The lower skirt steps
        # down to fit through the 22.3 mm mounting hole throughout its stroke.
        obj = revolve(name, [(0, 1), (10.4, 1), (10.8, 1.4), (10.8, 4.5),
                             (11.7, 5), (12, 5.3), (12, 8.5), (11.85, 8.85),
                             (11.5, 9), (0, 9)], mat, sectors=48)
        joint = {"type": "slide", "parent": "bezel", "axis": [0, 0, -1],
                 "anchor_m": [0, 0, 0], "range": [0.0, 0.004], "damping": 0.6,
                 "frictionloss": 0.08, "spring": {"stiffness": 400.0, "reference": 0.0}}
        collision = [props.cylinder(name + " cap collider", 11.5, 8, (0, 0, 5),
                                    bevel_mm=0, vertices=32)]
        add(name, obj, collision, 0.01, 0.8, parent="bezel", joint=joint,
            dimensions_mm={"cap_diameter": 24, "cap_envelope_height": 8,
                           "cap_top_above_mount": 9, "cap_top_above_bezel": 6,
                           "lower_skirt_diameter": 21.6, "travel": 4},
            base_color=hex_color)

    black = hand_material("Selector black handled ABS", "#1A1A1A")
    selector = props.cylinder("Selector base", 13.5, 6, (0, 0, 3), bevel_mm=0.35,
                              segments=2, vertices=48, material=black)
    lever = props.box("32 x 9 x 14 selector lever", (9, 32, 14.3), (0, 0, 12.85),
                      bevel_mm=0.4, segments=2, material=black)
    boolean(selector, lever, "UNION")
    tree = black.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    color = bsdf.inputs["Base Color"].links[0].from_socket
    coord, xyz = tree.nodes.new("ShaderNodeTexCoord"), tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(coord.outputs["Object"], xyz.inputs[0])
    mask = props._math(tree, "MULTIPLY",
                       props._math(tree, "LESS_THAN", props._math(tree, "ABSOLUTE", xyz.outputs["X"]), 0.00065),
                       props._math(tree, "GREATER_THAN", xyz.outputs["Z"], 0.0197))
    mask = props._math(tree, "MULTIPLY", mask, props._math(tree, "GREATER_THAN", xyz.outputs["Y"], 0.001))
    mask = props._math(tree, "MULTIPLY", mask, props._math(tree, "LESS_THAN", xyz.outputs["Y"], 0.013))
    mix = tree.nodes.new("ShaderNodeMixRGB")
    tree.links.new(mask, mix.inputs[0])
    tree.links.new(color, mix.inputs[1])
    mix.inputs[2].default_value = props._linear_color("#F5F5F5")
    tree.links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    joint = {"type": "hinge", "parent": "bezel", "axis": [0, 0, 1], "anchor_m": [0, 0, 0],
             "range": [-math.pi / 4, math.pi / 4], "damping": 0.02, "frictionloss": 0.025,
             "detents": {"positions_rad": [-math.pi / 4, math.pi / 4], "labels": ["MAN", "AUTO"],
                         "holding_torque_nm": 0.08, "snap_stiffness_nm_per_rad": 0.4,
                         "implementation": "bistable endpoint torque in simulator"}}
    add("selector_knob", selector,
        [props.box("Selector lever collision", (9, 32, 14), (0, 0, 13), bevel_mm=0)],
        0.025, 0.8, parent="bezel", joint=joint,
        dimensions_mm={"base_diameter": 27, "base_height": 6, "lever_xyz": [9, 32, 14]},
        recommended_initial_qpos=-math.pi / 4)

    # The yellow plate is a fixed rigid body, never part of the sliding head.
    yellow = label_material(folder, "emergency", "EMERGENCY STOP", "#F4C20D", aspect=1)
    # Paint two rim words, leaving the centre unobstructed by lettering.
    image = next(n for n in yellow.node_tree.nodes if n.type == "TEX_IMAGE")
    image.image = None
    yellow.node_tree.nodes.remove(image)
    bsdf = yellow.node_tree.nodes["Principled BSDF"]
    for link in list(bsdf.inputs["Base Color"].links):
        yellow.node_tree.links.remove(link)
    bsdf.inputs["Base Color"].default_value = props._linear_color("#F4C20D")
    disk = props.cylinder("estop_plate", 30, 1.5, (0, 0, 0.75), bevel_mm=0.3,
                          segments=2, vertices=64, material=yellow)
    # Text is converted and unioned as shallow recessed black-filled engraving.
    # Separate source text meshes share this fixed part and its baked atlas.
    text_material = props.pbr_material("Emergency plate black print", "#111111", 0.65, 0)
    lettering = []
    for word, y in (("EMERGENCY", 23), ("STOP", -24)):
        curve = bpy.data.curves.new(word + " printed type", "FONT")
        curve.body, curve.align_x, curve.align_y = word, "CENTER", "CENTER"
        curve.size, curve.resolution_u, curve.extrude = props.mm(3.5), 2, props.mm(0.015)
        obj = bpy.data.objects.new(word + " rim print", curve)
        bpy.context.collection.objects.link(obj)
        obj.location = (0, props.mm(y), props.mm(1.5))
        obj.data.materials.append(text_material)
        props._select([obj])
        bpy.ops.object.convert(target="MESH")
        lettering.append(bpy.context.object)
    disk = props.join_objects([disk, *lettering], "estop_plate")
    add("estop_plate", disk,
        [props.cylinder("Yellow disc collision", 30, 1.5, (0, 0, 0.75), bevel_mm=0, vertices=32)],
        0.012, dimensions_mm={"diameter": 60, "height": 1.5}, text="EMERGENCY STOP")

    red = hand_material("Emergency stop red mushroom", "#D32F2F")
    head = revolve("estop", [(0, -3.5), (6.6, -3.5), (7, -3.1), (7, 6.5),
                              (18.5, 6.5), (19.5, 7), (20, 8), (20, 22.5),
                              (19.7, 23.4), (18.8, 24.2), (18, 24.5), (0, 24.5)], red, sectors=48)
    joint = {"type": "slide", "parent": "estop_plate", "axis": [0, 0, -1],
             "anchor_m": [0, 0, 0], "range": [0.0, 0.005], "damping": 0.8, "frictionloss": 0.12,
             "latch": {"engage_at_m": 0.0048, "hold_at_m": 0.005,
                       "reset": "twist_to_release", "release_axis": [0, 0, 1],
                       "implementation": "simulator state machine; hold qpos after threshold"}}
    add("estop", head,
        [props.cylinder("Emergency head collision", 19.8, 18, (0, 0, 15.5), bevel_mm=0, vertices=32)],
        0.01, 0.8, parent="estop_plate", joint=joint,
        dimensions_mm={"head_diameter": 40, "head_height": 18, "head_bottom_z": 6.5,
                       "stem_diameter": 14, "travel": 5}, base_color="#D32F2F")

    for color, hex_color in (("green", "#2ECC40"), ("amber", "#FFB000"), ("red", "#E53935")):
        name = "pilot_" + color
        mat = props.pbr_material(name + " flat lens", hex_color, 0.25, 0, grime=0.10)
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Emission Color"].default_value = props._linear_color(hex_color)
        bsdf.inputs["Emission Strength"].default_value = 0.8
        obj = props.cylinder(name, 12, 6, (0, 0, 6), bevel_mm=0.35,
                             segments=2, vertices=48, material=mat)
        add(name, obj, [props.cylinder(name + " collision", 12, 6, (0, 0, 6),
                                       bevel_mm=0, vertices=32)], 0.008, parent="bezel",
            emission=0.8, dimensions_mm={"lens_diameter": 24, "lens_height": 6}, base_color=hex_color)

    legend = rr_box("legend_plate", 30, 10, 1, 0.5, edge=0.3, location=(0, 0, 0.5))
    print_uv(legend, 30, 10)
    legend_mat = label_material(folder, "legend_start", "START")
    legend.data.materials.append(legend_mat)
    add("legend_plate", legend,
        [props.box("Legend collision", (30, 10, 1), (0, 0, 0.5), bevel_mm=0)],
        0.002, dimensions_mm={"xyz": [30, 10, 1]})
    variants = []
    image_node = next(n for n in legend_mat.node_tree.nodes if n.type == "TEX_IMAGE")
    original = image_node.image
    for index, text in enumerate(LEGENDS):
        texture = "legend_plate_albedo.png" if index == 0 else f"legend_plate_albedo_v{index}.png"
        if index:
            mat = label_material(folder, "legend_" + text.lower().replace(" ", "_"), text)
            image_node.image = next(n.image for n in mat.node_tree.nodes if n.type == "TEX_IMAGE")
            isolated_bake(legend, 1024, folder / texture)
        variants.append({"name": text, "text": text, "textures": {"legend_plate": texture}})
    image_node.image = original

    notes = (
        "This is an independent-component library, NOT an overlaid physical assembly. Each visual/collider has "
        "its own mounting-face-centre origin and +Z outward; pressing is -Z. bbox_mm is only the union of "
        "these independent local frames, not the staged preview. Yellow estop_plate is an additional fixed "
        "part so the specified yellow disc never moves with estop. Red head is exactly diameter 40 x 18 mm; "
        "its lower face starts at Z6.5 to allow the 5 mm stroke above the 1.5 mm yellow plate. The integral "
        "14 mm stem is additional geometry. Button envelopes are diameter 24 x 8 with tops at Z9, six mm "
        "above the 3 mm bezel; their underside skirts step to diameter 21.6 for mounting-hole clearance. "
        "Single diameter-23 button cylinders approximate the stepped skirt within 1 mm, and the "
        "single e-stop head cylinder is diameter 39.6 to follow its rounded lip within 1 mm; cap/support contact pairs "
        "must be excluded in the simulator. The selector lever is 32 long along Y x 9 X x 14 Z on a 27 x 6 "
        "base; its symmetric base rotates with the lever. Per the task, its only collider is the lever box, "
        "omitting the fixed base. Zero-angle geometry is the neutral joint reference; recommended initial "
        "qpos is MAN=-45 degrees, because a symmetric -45/+45 joint range cannot put its zero at an endpoint. "
        "Endpoint detents and the e-stop latch are explicit simulator behaviour specifications, not Blender "
        "constraints. Spring stiffness, damping, frictionloss and detent forces were unspecified and are "
        "documented benchmark choices. Pilot emission 0.8 is retained in both PBR and MuJoCo metadata; "
        "emission is not baked into the albedo. Six legend atlases share the same mesh/UVs. " + RECIPE
    )
    metadata = {"id": "panel_devices", "task": "panel_operation", "units": "m",
                "origin": "each_component_mounting_face_center", "bbox_mm": props.bbox_mm(objects.values()),
                "budget_class": "small", "triangles": props.triangle_count(objects.values()),
                "source": "source.blend", "preview": "preview.png", "parts": parts,
                "variants": variants, "notes": notes,
                "nominal_mounting_diameter_mm": 22, "required_panel_hole_mm": 22.3}
    render_library_preview(objects, folder)
    save_source(folder, metadata, [objects["bezel"]])
    log("COMPLETE", "panel_devices", metadata["triangles"], "triangles")


def device_layout(layout):
    devices = [
        ("lamp_green", "pilot_green", -60, 150, "AUTO"),
        ("lamp_amber", "pilot_amber", 0, 150, "MAN"),
        ("lamp_red", "pilot_red", 60, 150, "STOP"),
        ("mode", "selector_knob", -90, 70, "MAN"),
        ("start", "btn_green", -20, 70, "START"),
        ("stop", "btn_red", 40, 70, "STOP"),
        ("emergency", "estop", 110, 110, "STOP"),
    ]
    if layout == "B":
        devices = [(name, part, -u, v, text) for name, part, u, v, text in devices]
    elif layout == "C":
        positions = {"mode": -110, "start": -50, "stop": 10, "emergency": 100}
        devices = [(name, part, positions.get(name, u), 170 if name.startswith("lamp") else 110,
                    "CYCLE START" if name == "start" else "FEED HOLD" if name == "stop" else text)
                   for name, part, u, v, text in devices]
    return devices


def mounting_cutter(u, v):
    obj = props.cylinder("22.3 mm apothem mounting bore", 11.15 / math.cos(math.pi / 64), 15,
                         bevel_mm=0, vertices=64)
    return transform(obj, pose(u, v, 0))


def shell_geometry(name, material, devices):
    plan = rounded_rectangle(200, 320, 3)
    outer = loft(name, [(BASE_X + x, y, 0) for x, y in plan],
                 [(BASE_X + x, y, 110 + 0.5 * x) for x, y in plan], material)
    plan = rounded_rectangle(197, 317, 1.5)
    cavity = loft("1.5 mm sheet interior", [(BASE_X + x, y, 1.5) for x, y in plan],
                  [(BASE_X + x, y, 110 + 0.5 * x - 1.5 / C) for x, y in plan])
    boolean(outer, cavity)
    for _, _, u, v, _ in devices:
        boolean(outer, mounting_cutter(u, v))
    bevel(outer, 0.3, segments=2)
    # Recessed right (-Y) nameplate stays inside the 320 mm base envelope.
    pocket = props.box("Right-side nameplate pocket", (72, 1.2, 20), (BASE_X - 5, -159.9, 30),
                        bevel_mm=0.3)
    boolean(outer, pocket)
    return outer


def shell_collisions():
    # OBJ positions have a 1 micrometre grid. Since roof Z = 0.5*X + b,
    # an even-micrometre X grid keeps every exported cap exactly planar.
    plan = [(round(x / 0.002) * 0.002, round(y, 3))
            for x, y in rounded_rectangle(200, 320, 3)]
    result = [loft("Base sheet collision", [(BASE_X + x, y, 0) for x, y in plan],
                   [(BASE_X + x, y, 1.5) for x, y in plan])]
    # Six prescribed shell pieces cannot preserve the seven bores. The roof's
    # upper normal face is +0.75 mm. True normal extrusion also follows the
    # roof edge rounds within 1 mm, unlike raising a sharp rectangle vertically.
    result.append(loft("Sloped roof collision",
                       [(BASE_X + x + 0.670, y, 110 + 0.5 * (x + 0.670) - 1.677) for x, y in plan],
                       [(BASE_X + x - 0.336, y, 110 + 0.5 * (x - 0.336) + 0.839) for x, y in plan]))

    def clip(points, axis, limit, sign):
        output = []
        for a, b in zip(points, points[1:] + points[:1]):
            da, db = sign * (a[axis] - limit), sign * (b[axis] - limit)
            if da >= -1e-8:
                output.append(a)
            if (da > 1e-8 and db < -1e-8) or (da < -1e-8 and db > 1e-8):
                t = da / (da - db)
                output.append(tuple(a[i] + t * (b[i] - a[i]) for i in range(2)))
        # Tangency endpoints can coincide with a clipping-plane vertex.
        unique = []
        for p in output:
            if not unique or math.dist(p, unique[-1]) > 1e-7:
                unique.append(p)
        if len(unique) > 1 and math.dist(unique[0], unique[-1]) < 1e-7:
            unique.pop()
        return unique

    for axis, width, label in ((1, 160, "Side sheet collision"), (0, 100, "Front/rear sheet collision")):
        for side in (-1, 1):
            strip = clip(plan, axis, side * (width - 1.5), side)
            result.append(loft(label, [(BASE_X + x, y, 0) for x, y in strip],
                               [(BASE_X + x, y, 110 + 0.5 * x) for x, y in strip]))
    return result


def add_nameplate_and_gland(body, folder, metal, black):
    mat = label_material(folder, "imc_op03", "IMC Controls · OP-03", "#E5E7E9", aspect=4)
    plaque = rr_box("IMC Controls OP-03 recessed nameplate", 70, 18, 0.6, 1,
                    material=mat, edge=0.15)
    print_uv(plaque, 70, 18)
    side_rotation = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    transform(plaque, (BASE_X - 5, -159.7, 30), side_rotation)
    # Keep gland separate in the scene/JSON, but fixed to the same shell body.
    gland = props.cylinder("Rear cable gland core", 6, 13.5, (BASE_X + 105.25, 0, 35),
                            bevel_mm=0.4, segments=2, vertices=32, material=black,
                            rotation_deg=(0, 90, 0))
    nut = props.hex_prism("Rear gland hex retaining nut", 18, 5, bevel_mm=0.4,
                          segments=2, material=metal)
    transform(nut, (BASE_X + 102.5, 0, 35), Matrix.Rotation(math.pi / 2, 3, "Y"))
    boolean(gland, nut, "UNION")
    return props.join_objects([body, plaque], body.name), gland


def load_library():
    folder = OUTPUT / "panel_devices"
    data = json.loads((folder / "prop.json").read_text())
    names = [p["name"] for p in data["parts"]]
    names += [n for p in data["parts"] for n in p["source_collision_objects"]]
    with bpy.data.libraries.load(str(folder / "source.blend"), link=False) as (source, target):
        if not set(names) <= set(source.objects):
            raise RuntimeError("Library source is missing visual/collision prototypes")
        target.objects = names
    loaded = {}
    for obj in target.objects:
        bpy.context.collection.objects.link(obj)
        loaded[obj.name] = obj
    hide(loaded.values())
    for path in folder.glob("*_albedo*.png"):
        shutil.copyfile(path, OUTPUT / "control_console" / path.name)
    return data, loaded


def build_control_console():
    # reset_scene clears objects, but the previous prop's saved filepath still
    # makes Blender reject append-from-self. A fresh datablock clears it too.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    folder = OUTPUT / "control_console"
    folder.mkdir(parents=True, exist_ok=True)
    log("START", folder.name)
    library, prototypes = load_library()
    source_parts = {p["name"]: p for p in library["parts"]}
    legend_textures = {v["text"]: v["textures"]["legend_plate"] for v in library["variants"]}
    grey = props.pbr_material("RAL 7035 light-grey powder-coated steel", "#D7D7D7", 0.60, 0,
                              wear=0.05, grime=0.15, machining="brushed")
    dark = props.pbr_material("Dark grey operating panel", "#3A3D40", 0.60, 0, wear=0.04, grime=0.15)
    metal = props.pbr_material("Gland brushed plated steel", "#BFC3C7", 0.30, 1, machining="brushed")
    black = hand_material("Gland black polymer", "#222326", 0.65)
    variants, all_objects, default_objects = [], [], []

    def export_part(obj, colliders, record, prefix):
        obj.name = prefix
        clean(obj)
        record["visual"] = props.export_obj(obj, folder / (prefix + ".obj"))
        record["collision"] = props.export_collision(prefix, colliders, folder)
        hide(colliders)
        all_objects.append(obj)
        return record

    for layout in "ABC":
        log("LAYOUT", layout)
        devices = device_layout(layout)
        suffix = "" if layout == "A" else "_" + layout
        body_name, panel_name = "body" + suffix, "panel" + suffix
        body = shell_geometry(body_name, grey, devices)
        body, gland = add_nameplate_and_gland(body, folder, metal, black)
        clean(body)
        props.uv_unwrap(body)
        isolated_bake(body, 2048, folder / (body_name + "_albedo.png"))
        rgb = next(n for n in grey.node_tree.nodes if n.type == "RGB")
        rgb.outputs[0].default_value = props._linear_color("#3B6EA5")
        grey.diffuse_color = props._linear_color("#3B6EA5")
        isolated_bake(body, 2048, folder / (body_name + "_albedo_v1.png"))
        rgb.outputs[0].default_value = props._linear_color("#D7D7D7")
        grey.diffuse_color = props._linear_color("#D7D7D7")
        panel = rr_box(panel_name, 280, 190, 1.5, 3, material=dark, edge=0)
        transform(panel, pose(0, SLOPE_LENGTH / 2, 0.75))
        for _, _, u, v, _ in devices:
            boolean(panel, mounting_cutter(u, v))
        bevel(panel, 0.3, segments=2)
        clean(panel)
        props.uv_unwrap(panel)
        isolated_bake(panel, 2048, folder / (panel_name + "_albedo.png"))
        # Shell + panel + small fittings mass totals exactly 4 kg below.
        body_record = part_record("body", "", [], 3.5, 0.5, SHEET_MJ, size=2048,
                                  texture=body_name + "_albedo.png")
        parts = [export_part(body, shell_collisions(), body_record, body_name)]
        parts.append(export_part(panel, [], part_record("panel", "", [], 0.3, 0.5, SHEET_MJ,
                                 parent="body", texture=panel_name + "_albedo.png", size=2048), panel_name))
        # Gland collision has its own fixed part so the shell still has six pieces.
        props.uv_unwrap(gland)
        isolated_bake(gland, 1024, folder / ("gland" + suffix + "_albedo.png"))
        gland_col = [props.cylinder("Rear gland core collision", 6, 12,
                                    (BASE_X + 106, 0, 35), bevel_mm=0, vertices=32,
                                    rotation_deg=(0, 90, 0))]
        nut_col = props.hex_prism("Rear gland nut collision", 18, 5, bevel_mm=0)
        transform(nut_col, (BASE_X + 102.5, 0, 35), Matrix.Rotation(math.pi / 2, 3, "Y"))
        gland_col.append(nut_col)
        parts.append(export_part(gland, gland_col,
                                 part_record("gland", "", [], 0.02, 0.4, parent="body",
                                             texture="gland" + suffix + "_albedo.png"), "gland" + suffix))
        visible = [body, panel, gland]
        placements, exclusions = [], []

        def install(template, name, u, v, parent, text=None):
            source = source_parts[template]
            obj = duplicate(prototypes[template], name + suffix)
            transform(obj, pose(u, v))
            if template == "selector_knob":
                # Applied mesh transforms must not move the procedural white
                # indicator away from its original device-local coordinates.
                frame = bpy.data.objects.new(name + suffix + " PBR local frame", None)
                bpy.context.collection.objects.link(frame)
                frame.matrix_world = ROTATION.to_4x4()
                frame.matrix_world.translation = pose(u, v) * 0.001
                for index, original in enumerate(obj.data.materials):
                    material = original.copy()
                    obj.data.materials[index] = material
                    for node in material.node_tree.nodes:
                        if node.type == "TEX_COORD":
                            node.object = frame
            colliders = [transform(duplicate(prototypes[n], "Placed " + n), pose(u, v))
                         for n in source["source_collision_objects"]]
            record = json.loads(json.dumps(source))
            for key in ("source_collision_objects", "frame", "recommended_initial_qpos"):
                record.pop(key, None)
            record["name"], record["parent"] = name, parent
            if text:
                record["texture"] = legend_textures[text]
                record["text"] = text
                # Source PBR must show the chosen legend too, not just the OBJ atlas.
                original = obj.data.materials[0]
                own_mat = original.copy()
                obj.data.materials[0] = own_mat
                node = next(n for n in own_mat.node_tree.nodes if n.type == "TEX_IMAGE")
                source_stem = "legend_" + text.lower().replace(" ", "_") + "_print.png"
                source_path = OUTPUT / "panel_devices" / source_stem
                node.image = bpy.data.images.load(str(source_path), check_existing=True)
            joint = record["joint"]
            if joint:
                joint["parent"] = parent
                joint["axis"] = [float(a) for a in ROTATION @ Vector(joint["axis"])]
                joint["anchor_m"] = [float(a) / 1000 for a in pose(u, v)]
                if "latch" in joint:
                    joint["latch"]["release_axis"] = list(NORMAL)
                exclusions.extend([[name, p] for p in ("body", "panel", parent, name + "_legend")])
            parts.append(export_part(obj, colliders, record, name + suffix))
            visible.append(obj)
            return name

        for name, template, u, v, text in devices:
            if template in ("btn_green", "btn_red", "pilot_green", "pilot_amber", "pilot_red"):
                parent = install("bezel", name + "_bezel", u, v, "panel")
            elif template == "estop":
                parent = install("estop_plate", name + "_plate", u, v, "panel")
            else:
                parent = "panel"
            install(template, name, u, v, parent)
            # Exact prescribed -15 mm legend offset, beneath overlapping trim.
            install("legend_plate", name + "_legend", u, v - 15, "panel", text)
            point = pose(u, v)
            label_point = pose(u, v - 15)
            placements.append({
                "name": name, "library_part": template, "u_mm": u, "v_mm": v,
                "position_m": [float(a) / 1000 for a in point],
                "quaternion_wxyz": [float(a) for a in ROTATION.to_quaternion()],
                "rotation_matrix": [list(row) for row in ROTATION],
                "press_direction": list(-NORMAL), "legend": text,
                "legend_u_v_mm": [u, v - 15],
                "legend_position_m": [float(a) / 1000 for a in label_point],
            })
        # Allocate the remainder to the enclosure instead of changing specified cap masses.
        parts[0]["mass_kg"] += 4.0 - sum(p["mass_kg"] for p in parts)
        for obj in visible:
            obj.hide_render = False
            obj.hide_set(False)
        for obj in all_objects:
            if obj not in visible:
                obj.hide_render = True
        preview = "preview.png" if layout == "A" else "preview_" + layout + ".png"
        props.render_preview(visible, folder / preview)
        placements_y = {"A": 0.055, "B": -0.055, "C": -0.045}[layout]
        variant = {
            "name": layout, "parts": parts, "devices": placements,
            "bbox_mm": props.bbox_mm(visible), "triangles": props.triangle_count(visible),
            "preview": preview, "collision_exclusions": exclusions,
            "textures": {"body": body_name + "_albedo.png"},
            "colour_variants": [
                {"name": "RAL7035", "textures": {"body": body_name + "_albedo.png"}},
                {"name": "blue", "textures": {"body": body_name + "_albedo_v1.png"}},
            ],
            "recommended_world_translation_m": [0.436, placements_y, 0.8],
        }
        variants.append(variant)
        if layout == "A":
            default_objects = visible
        hide(visible)

    # A directly usable blue variant accompanies the three complete layouts.
    blue = {"name": "blue", "layout": "A", "textures": {"body": "body_albedo_v1.png"}}
    metadata = {
        "id": "control_console", "task": "panel_operation", "units": "m", "origin": "bottom_center",
        "bbox_mm": variants[0]["bbox_mm"], "budget_class": "medium", "triangles": variants[0]["triangles"],
        "source": "source.blend", "preview": "preview.png", "parts": variants[0]["parts"],
        "variants": [*variants, blue], "mass_kg": 4.0, "friction": 0.5,
        "dimensions_mm": {"base_xyz": [200, 320, 0], "front_height": 60, "back_height": 160,
                          "sheet_thickness": 1.5, "corner_radius": 3, "panel_uv": [280, 190],
                          "panel_thickness": 1.5, "mounting_hole_diameter": 22.3,
                          "slope_degrees": math.degrees(math.atan(0.5)), "rear_gland_projection": 12},
        "panel_frame": {
            "u_direction": [0, 1, 0], "v_direction": [C, 0, S], "outward_normal": list(NORMAL),
            "slope_front_edge_m": [(BASE_X - 100) / 1000, 0, 0.060],
            "mounting_face_offset_m": 0.0015,
            "position_formula_mm": "[-106 + v*2/sqrt(5), u, 60 + v/sqrt(5)] + 1.5*[-1/sqrt(5),0,2/sqrt(5)]",
            "device_local_x": [0, -1, 0], "device_local_y": [C, 0, S],
            "quaternion_convention": "wxyz, active local-to-console rotation, right handed",
        },
        "notes": (
            "Default is layout A. A/B/C each provide a complete parts replacement (meshes, textures, "
            "colliders, transformed joints) and actual seven diameter-22.3 bores through BOTH sheet roof "
            "and dark panel. B mirrors device centres, not lettering. C pilot u values were unspecified "
            "and retain A's -60/0/+60 at v170. Panel 280 along u x 190 along slope is centred on the roof. "
            "The nominal 200 X x 320 Y base is centred at X=-6 mm; a rear gland protrudes 12 mm, giving "
            "the combined visual bottom-centre origin at zero and bbox 212 x 320 x 160 mm. Nameplate is "
            "recessed flush on robot-right (-Y). Thin sheet edge rounding is 0.3 mm because a general "
            "1 mm edge radius cannot fit 1.5 mm stock; plan corners retain R3. The six prescribed shell "
            "colliders intentionally fill mounting holes, and their roof is expanded outward 0.75 mm to "
            "approximate the additional 1.5 mm dark panel within 1 mm. The gland has a separate fixed "
            "collider. Actuator/support contact exclusions are explicitly supplied for these simplified "
            "colliders. At the prescribed 15 mm legend offset, 30 x 10 plates overlap 29.5 mm bezels and "
            "the 60 mm emergency disc: exact centre positions are retained; upper label edges sit under "
            "the trim, and the emergency legend is mostly obscured by its correctly printed yellow disc. "
            "This unavoidable placement conflict is not resolved by moving the mandated centres. "
            "Static trim/legend intersections are visual-only and adjacent fixed bodies should be welded "
            "or have contacts excluded by the scene importer. The selector's neutral geometry, endpoint "
            "detents, spring and latch conventions are inherited from panel_devices; simulator behaviour "
            "is metadata, not an executed dynamics demonstration. Recommended per-layout placements put "
            "all four actuation centres in x300-480 mm, |y|30-450 mm, z800-1000 mm; pilots need not be "
            "pressed. Enclosure mass absorbs the remainder so each complete layout totals exactly 4 kg. "
            "All geometry/UVs/normal exports share the console frame. Full Principled PBR and source "
            "printing images are packed in source.blend; preview_B/C show the other real layouts. " + RECIPE
        ),
    }
    save_source(folder, metadata, default_objects)
    log("COMPLETE", "control_console", metadata["triangles"], "triangles")


def refresh_previews(prop_id):
    """Re-render existing task-2 sources without rebuilding meshes or atlases."""
    folder = OUTPUT / prop_id
    metadata = json.loads((folder / "prop.json").read_text())
    bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
    if prop_id == "panel_devices":
        objects = {p["name"]: bpy.data.objects[p["name"]] for p in metadata["parts"]}
        render_library_preview(objects, folder)
        visible = [objects["bezel"]]
    else:
        visible = []
        for variant in metadata["variants"]:
            if "parts" not in variant:
                continue
            objects = [bpy.data.objects[Path(p["visual"]).stem] for p in variant["parts"]]
            for obj in bpy.context.scene.objects:
                if obj.type == "MESH":
                    obj.hide_render = obj not in objects
                    obj.hide_set(obj not in objects)
            bpy.context.view_layer.update()
            props.render_preview(objects, folder / variant["preview"])
            if variant["name"] == "A":
                visible = objects
        # Additional render of the real blue powder-coat variant.
        grey = next(m for m in bpy.data.materials if m.name.startswith("RAL 7035"))
        rgb = next(n for n in grey.node_tree.nodes if n.type == "RGB")
        rgb.outputs[0].default_value = props._linear_color("#3B6EA5")
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                obj.hide_render = obj not in visible
        props.render_preview(visible, folder / "preview_blue.png")
        rgb.outputs[0].default_value = props._linear_color("#D7D7D7")
    save_source(folder, metadata, visible)
    log("PREVIEWS_REFRESHED", prop_id)


def refresh_collisions(prop_id):
    """Update the refined <1 mm proxies without repeating colour/AO bakes."""
    folder = OUTPUT / prop_id
    metadata = json.loads((folder / "prop.json").read_text())
    bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
    if prop_id == "panel_devices":
        for part in metadata["parts"]:
            name = part["name"]
            if name not in ("btn_green", "btn_red", "btn_yellow", "estop"):
                continue
            for object_name in part["source_collision_objects"]:
                old = bpy.data.objects.get(object_name)
                if old:
                    bpy.data.objects.remove(old, do_unlink=True)
            radius, depth, z = (19.8, 18, 15.5) if name == "estop" else (11.5, 8, 5)
            obj = props.cylinder(f"COL__{name}__00", radius, depth, (0, 0, z), bevel_mm=0, vertices=32)
            part["collision"] = props.export_collision(name, [obj], folder)
            part["source_collision_objects"] = [obj.name]
            part["collision_dimensions_mm"] = {"diameter": 2 * radius, "height": depth}
            hide([obj])
        metadata["notes"] += (
            " Collision refinement: button cylinders are diameter 23 x 8 mm and emergency head cylinder "
            "diameter 39.6 x 18 mm, keeping rounded/stepped cap boundary errors below 1 mm."
        )
        visible = [bpy.data.objects["bezel"]]
    else:
        old_prefixes = ("Base sheet collision", "Sloped roof collision", "Side sheet collision",
                        "Front/rear sheet collision", "Placed COL__btn_green__", "Placed COL__btn_red__",
                        "Placed COL__estop__", "COL__btn_green__", "COL__btn_red__", "COL__btn_yellow__",
                        "COL__estop__")
        for obj in list(bpy.data.objects):
            if obj.type == "MESH" and obj.name.startswith(old_prefixes):
                bpy.data.objects.remove(obj, do_unlink=True)
        for variant in metadata["variants"]:
            if "parts" not in variant:
                continue
            mapping = {part["name"]: part for part in variant["parts"]}
            exclusions = variant.get("collision_exclusions", [])
            exclusions.extend([[p["name"], p["name"] + "_legend"]
                               for p in variant["parts"] if p["joint"]])
            variant["collision_exclusions"] = [list(pair) for pair in dict.fromkeys(tuple(p) for p in exclusions)]
            colliders = shell_collisions()
            for index, obj in enumerate(colliders):
                obj.name = f"COLL__body_{variant['name']}__{index:02d}"
            prefix = Path(mapping["body"]["visual"]).stem
            mapping["body"]["collision"] = props.export_collision(prefix, colliders, folder)
            hide(colliders)
            for device in variant["devices"]:
                name, template = device["name"], device["library_part"]
                if template not in ("btn_green", "btn_red", "estop"):
                    continue
                radius, depth, z = (19.8, 18, 15.5) if template == "estop" else (11.5, 8, 5)
                obj = props.cylinder(f"COLL__{name}_{variant['name']}__00", radius, depth,
                                     (0, 0, z), bevel_mm=0, vertices=32)
                transform(obj, pose(device["u_mm"], device["v_mm"]))
                prefix = Path(mapping[name]["visual"]).stem
                mapping[name]["collision"] = props.export_collision(prefix, [obj], folder)
                mapping[name]["collision_dimensions_mm"] = {"diameter": 2 * radius, "height": depth}
                hide([obj])
        default = next(v for v in metadata["variants"] if v["name"] == "A")
        metadata["parts"] = default["parts"]
        metadata["notes"] = metadata["notes"].replace("expanded outward 1 mm", "expanded outward 0.75 mm")
        metadata["notes"] += (
            " Collision refinement: all six shell proxies have clipped R3 plan footprints, with at most "
            "56 vertices each; rounded wall-strip corner underfill is less than 0.63 mm. Roof is extruded "
            "along the true normal, from -1.5 to +0.75 mm, so rim overhang also remains under 1 mm. "
            "Button/head proxy diameters are 23/39.6 mm to follow their stepped/rounded boundaries. "
            "v is measured from the full roof's front edge; the centred dark insert starts at v16.8034 mm."
        )
        visible = [bpy.data.objects[Path(p["visual"]).stem] for p in metadata["parts"]]
    save_source(folder, metadata, visible)
    log("COLLISIONS_REFRESHED", prop_id)


def refresh_legends(prop_id):
    """Improve caption clearance while keeping exact plate/device centres."""
    folder = OUTPUT / prop_id
    metadata = json.loads((folder / "prop.json").read_text())
    bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
    if prop_id == "panel_devices":
        obj = bpy.data.objects["legend_plate"]
        material = obj.data.materials[0]
        legend_ink_mapping(material)
        node = next(n for n in material.node_tree.nodes if n.type == "TEX_IMAGE")
        original = node.image
        for variant in metadata["variants"]:
            stem = "legend_" + variant["text"].lower().replace(" ", "_") + "_print.png"
            node.image = bpy.data.images.load(str(folder / stem), check_existing=True)
            isolated_bake(obj, 1024, folder / variant["textures"]["legend_plate"])
        node.image = original
        visible = [bpy.data.objects["bezel"]]
    else:
        library = OUTPUT / "panel_devices"
        for path in library.glob("legend_plate_albedo*.png"):
            shutil.copyfile(path, folder / path.name)
        for variant in metadata["variants"]:
            if "parts" not in variant:
                continue
            for part in variant["parts"]:
                if part["name"].endswith("_legend"):
                    obj = bpy.data.objects[Path(part["visual"]).stem]
                    for material in obj.data.materials:
                        legend_ink_mapping(material)
        prototype = bpy.data.objects.get("legend_plate")
        if prototype:
            legend_ink_mapping(prototype.data.materials[0])
        visible = [bpy.data.objects[Path(p["visual"]).stem] for p in metadata["parts"]]
    metadata["notes"] += (
        " Legend ink is enlarged vertically and placed in the lower band of each unchanged 30 x 10 mm "
        "plate, keeping text clear of the bezel at the prescribed 15 mm plate-centre offset; the "
        "emergency plate remains occluded by the mandatory 60 mm yellow disc."
    )
    save_source(folder, metadata, visible)
    log("LEGENDS_REFRESHED", prop_id)


def main():
    builders = {"panel_devices": build_panel_devices, "control_console": build_control_console}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=list(builders), default=list(builders))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--previews-only", action="store_true")
    mode.add_argument("--refresh-collisions", action="store_true")
    mode.add_argument("--refresh-legends", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    for prop_id in dict.fromkeys(args.props):
        if args.previews_only:
            refresh_previews(prop_id)
        elif args.refresh_collisions:
            refresh_collisions(prop_id)
        elif args.refresh_legends:
            refresh_legends(prop_id)
        else:
            builders[prop_id]()


if __name__ == "__main__":
    try:
        main()
        props.finalize_assets()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
