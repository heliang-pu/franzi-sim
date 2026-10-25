"""Build only the two section 3.3 props, using the shared Blender 5.x pipeline.

Blender -b --factory-startup -P tools/blender/build_task03_tray_push.py --
Optional --props restricts regeneration; --preview-only reuses the saved source.
All modelling dimensions are millimetres; exported coordinates are metres.
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


def draw_esd_mask(path):
    """Original generic ESD pictogram and lettering, never a copied logo."""
    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo

    image = Image.new("RGB", (1680, 500), "black")
    draw = ImageDraw.Draw(image)
    # A bordered warning triangle enclosing a stylised finger and spark.
    draw.polygon([(245, 28), (20, 450), (470, 450)], fill="white")
    draw.polygon([(245, 90), (73, 417), (417, 417)], fill="black")
    draw.line([(122, 296), (184, 296), (221, 254), (247, 245), (267, 259),
               (260, 283), (239, 306), (269, 331), (284, 367), (264, 388),
               (225, 360), (185, 352), (122, 352)], fill="white", width=23)
    draw.polygon([(328, 175), (285, 255), (321, 246), (300, 306),
                  (370, 221), (332, 232), (353, 175)], fill="white")
    font = None
    for name in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(name).is_file():
            font = ImageFont.truetype(name, 199)
            break
    if font is None:
        font = ImageFont.load_default(size=190)
    draw.text((530, 250), "ESD SAFE", fill="white", font=font, anchor="lm")
    info = PngInfo()
    info.add(b"sRGB", b"\x00")
    image.save(path, pnginfo=info)


if __name__ == "__main__" and "--draw-esd-mask" in sys.argv:
    draw_esd_mask(Path(sys.argv[-1]))
    raise SystemExit(0)

import bmesh
import bpy

import prop_lib as props

PP_MJ = {"specular": 0.25, "shininess": 0.30, "reflectance": 0.0}
AL_MJ = {"specular": 0.70, "shininess": 0.70, "reflectance": 0.05}
COMMON_NOTES = (
    "Metres, Z up, front -X, bottom-centre origin shared by every part and proxy. "
    "Triangulated OBJ with normals, 30-degree smoothing and nonoverlapping Smart UV 66-degree atlases. "
    "64-sample CPU Cycles base colour * (0.4 + 0.6 * AO), saved as 8-bit sRGB, without lighting or highlights. "
    "Full Principled PBR, source markings and procedural wear remain packed in source.blend. "
)


def node_math(tree, operation, *inputs):
    return props._math(tree, operation, *inputs)


def mix_color(tree, original, factor, color, name):
    node = tree.nodes.new("ShaderNodeMixRGB")
    node.name = name
    tree.links.new(factor, node.inputs[0])
    tree.links.new(original, node.inputs[1])
    node.inputs[2].default_value = props._linear_color(color)
    return node.outputs[0]


def coordinates(tree):
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    position = tree.nodes.new("ShaderNodeSeparateXYZ")
    normal = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(geometry.outputs["Position"], position.inputs[0])
    tree.links.new(geometry.outputs["Normal"], normal.inputs[0])
    return position.outputs, normal.outputs


def current_color(material):
    bsdf = material.node_tree.nodes["Principled BSDF"]
    return bsdf, bsdf.inputs["Base Color"].links[0].from_socket


def multiply_all(tree, values):
    result = values[0]
    for other in values[1:]:
        result = node_math(tree, "MULTIPLY", result, other)
    return result


def inside(tree, socket, lower, upper):
    return node_math(tree, "MULTIPLY", node_math(tree, "GREATER_THAN", socket, lower),
                     node_math(tree, "LESS_THAN", socket, upper))


def plastic_material(mask_path):
    mat = props.pbr_material("Conductive black PP #1B1B1B", "#1B1B1B", 0.65, 0, grime=0.15)
    tree = mat.node_tree
    bsdf, color = current_color(mat)
    pos, normal = coordinates(tree)
    # A 1.5 mm rim band, with deliberately modest lightening on black PP.
    rim = node_math(tree, "GREATER_THAN", pos["Z"], props.mm(48.5))
    rim = node_math(tree, "MULTIPLY", rim, 0.21)
    color = mix_color(tree, color, rim, "#66686A", "1.5 mm moulded rim wear")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    stretch = tree.nodes.new("ShaderNodeVectorMath")
    stretch.operation = "MULTIPLY"
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    tree.links.new(geometry.outputs["Position"], stretch.inputs[0])
    stretch.inputs[1].default_value = (800, 35, 1100)
    tree.links.new(stretch.outputs[0], noise.inputs["Vector"])
    noise.inputs["Scale"].default_value = 1
    noise.inputs["Detail"].default_value = 2
    streaks = node_math(tree, "MULTIPLY", node_math(tree, "GREATER_THAN", noise.outputs["Fac"], 0.70), 0.17)
    color = mix_color(tree, color, streaks, "#696C70", "Sparse long handling scuffs")
    # A broad faint handling patch surrounding each real finger opening.
    near_end = node_math(tree, "GREATER_THAN", node_math(tree, "ABSOLUTE", pos["Y"]), props.mm(146.3))
    handling = multiply_all(tree, [near_end, inside(tree, pos["X"], -.044, .044),
                                  inside(tree, pos["Z"], .016, .046)])
    handling = node_math(tree, "MULTIPLY", handling, 0.11)
    color = mix_color(tree, color, handling, "#55534E", "Faint finger oils around handles")
    image = bpy.data.images.load(str(mask_path), check_existing=True)
    image.colorspace_settings.name = "Non-Color"
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.name = "Original ESD triangle and ESD SAFE print mask"
    tex.image, tex.extension = image, "CLIP"
    uv = tree.nodes.new("ShaderNodeCombineXYZ")
    u = node_math(tree, "ADD", node_math(tree, "DIVIDE", pos["Y"], -.084), .5)
    v = node_math(tree, "DIVIDE", node_math(tree, "SUBTRACT", pos["Z"], .0125), .025)
    tree.links.new(u, uv.inputs["X"])
    tree.links.new(v, uv.inputs["Y"])
    tree.links.new(uv.outputs[0], tex.inputs["Vector"])
    exterior = node_math(tree, "LESS_THAN", normal["X"], -.8)
    exterior = node_math(tree, "MULTIPLY", exterior, node_math(tree, "LESS_THAN", pos["X"], -.099))
    print_mask = node_math(tree, "MULTIPLY", exterior, tex.outputs["Color"])
    color = mix_color(tree, color, print_mask, "#F2C230", "Yellow screen-printed generic ESD marking")
    tree.links.new(color, bsdf.inputs["Base Color"])
    grain = tree.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 1500
    grain.inputs["Detail"].default_value = 2
    tree.links.new(geometry.outputs["Position"], grain.inputs["Vector"])
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .13
    bump.inputs["Distance"].default_value = props.mm(.025)
    tree.links.new(grain.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def aluminium_material(name, axis):
    mat = props.pbr_material(name, "#B9BEC3", .30, 1, wear=.04, grime=.12, machining="brushed")
    # Keep extrusion marks parallel to each profile, rather than across it.
    group = next(n.node_tree for n in mat.node_tree.nodes if n.type == "GROUP" and "Machining" in n.node_tree.name)
    separate = next(n for n in group.nodes if n.type == "SEPXYZ")
    sine = next(n for n in group.nodes if n.type == "MATH" and n.operation == "SINE")
    phase_scale = sine.inputs[0].links[0].from_node
    group.links.new(separate.outputs[axis], phase_scale.inputs[0])
    return mat


def stop_material():
    mat = props.pbr_material("Milled anodised aluminium stop with hazard tape", "#B9BEC3", .30, 1,
                             wear=.04, grime=.15, machining="milled")
    tree = mat.node_tree
    bsdf, color = current_color(mat)
    pos, normal = coordinates(tree)
    tape = multiply_all(tree, [inside(tree, pos["X"], -.0185, .0185),
                              inside(tree, pos["Y"], -.3185, -.2815),
                              node_math(tree, "GREATER_THAN", normal["Z"], .9),
                              node_math(tree, "GREATER_THAN", pos["Z"], .0496)])
    phase = node_math(tree, "DIVIDE", node_math(tree, "ADD", pos["X"], pos["Y"]), .014)
    stripe = node_math(tree, "LESS_THAN", node_math(tree, "PINGPONG", phase, 1), .5)
    stripe_node = tree.nodes.new("ShaderNodeMixRGB")
    tree.links.new(stripe, stripe_node.inputs[0])
    stripe_node.inputs[1].default_value = props._linear_color("#F2C230")
    stripe_node.inputs[2].default_value = props._linear_color("#151515")
    printed = tree.nodes.new("ShaderNodeMixRGB")
    tree.links.new(tape, printed.inputs[0])
    tree.links.new(color, printed.inputs[1])
    tree.links.new(stripe_node.outputs[0], printed.inputs[2])
    tree.links.new(printed.outputs[0], bsdf.inputs["Base Color"])
    tree.links.new(node_math(tree, "SUBTRACT", 1, tape), bsdf.inputs["Metallic"])
    tree.links.new(node_math(tree, "ADD", .30, node_math(tree, "MULTIPLY", tape, .32)),
                   bsdf.inputs["Roughness"])
    return mat


def rounded_rectangle(width, height, radius, segments=12):
    result = []
    for cx, cy, start in ((width / 2 - radius, height / 2 - radius, 0),
                          (-width / 2 + radius, height / 2 - radius, 90),
                          (-width / 2 + radius, -height / 2 + radius, 180),
                          (width / 2 - radius, -height / 2 + radius, 270)):
        for step in range(segments + 1):
            angle = math.radians(start + 90 * step / segments)
            result.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return result


def hollow_tray(material):
    rings = [[(x, y, z) for x, y in rounded_rectangle(w, h, r)]
             for w, h, r, z in ((200, 300, 8, 0), (200, 300, 8, 50),
                               (194, 294, 5, 50), (194, 294, 5, 3))]
    n = len(rings[0])
    faces = [(i * n + j, i * n + (j + 1) % n,
              (i + 1) * n + (j + 1) % n, (i + 1) * n + j)
             for i in range(3) for j in range(n)]
    faces += [tuple(reversed(range(n))), tuple(range(3 * n, 4 * n))]
    return props._mesh_object("Continuous R8 tray shell", sum(rings, []), faces,
                              (0, 0, 0), 0, 3, material)


def move(obj, location=(0, 0, 0), rotation=(0, 0, 0)):
    obj.location = [props.mm(v) for v in location]
    obj.rotation_euler = [math.radians(v) for v in rotation]
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def difference(target, cutter):
    if not cutter.data.materials and target.data.materials:
        cutter.data.materials.append(target.data.materials[0])
    props._select([target])
    modifier = target.modifiers.new("Real cut: " + cutter.name, "BOOLEAN")
    modifier.operation, modifier.solver, modifier.object = "DIFFERENCE", "EXACT", cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def bevel(obj, amount):
    props._select([obj])
    modifier = obj.modifiers.new("Manufactured edge radii", "BEVEL")
    modifier.width, modifier.segments = props.mm(amount), 3
    modifier.limit_method, modifier.angle_limit = "ANGLE", math.radians(30)
    modifier.harden_normals = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def finish(obj):
    """Clean before UV/bake, including the OBJ writer's micrometre rounding."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for vertex in bm.verts:
        vertex.co = tuple(round(v, 6) for v in vertex.co)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=2e-7)
    bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=2e-7)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    if not all(edge.is_manifold for edge in bm.edges):
        bm.free()
        raise ValueError(f"{obj.name}: visual is not closed; refuse to bake broken geometry")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    props.smooth_by_angle(obj, 30)
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    modifier = obj.modifiers.new("Area-weighted manufacturing normals", "WEIGHTED_NORMAL")
    modifier.keep_sharp, modifier.mode, modifier.weight = True, "FACE_AREA", 50
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    print("TASK03_MESH", obj.name, props.triangle_count([obj]), flush=True)


def start(prop_id):
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    folder = OUTPUT / prop_id
    folder.mkdir(parents=True, exist_ok=True)
    print("TASK03_START", prop_id, bpy.app.version_string, flush=True)
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


def export_part(folder, name, visual, collisions, size, mass, friction, material, parent=None):
    visual.name = name
    finish(visual)
    props.uv_unwrap(visual)
    hide_collisions(collisions)
    # Only this part should contribute AO to its own atlas.
    others = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj != visual]
    visibility = [(obj, obj.hide_render) for obj in others]
    for obj in others:
        obj.hide_render = True
    try:
        props.bake_albedo([visual], size, folder / (name + "_albedo.png"))
    finally:
        for obj, hidden in visibility:
            obj.hide_render = hidden
    collision_files = props.export_collision(name, collisions, folder)
    hide_collisions(collisions)  # export selection unhides the input proxies
    return {
        "name": name, "visual": props.export_obj(visual, folder / (name + ".obj")),
        "texture": name + "_albedo.png", "texture_px": size, "collision": collision_files,
        "mass_kg": mass, "friction": friction, "mujoco_material": material,
        "parent": parent, "joint": None, "source_visual_object": visual.name,
        "source_collision_objects": [obj.name for obj in collisions],
    }


def complete(folder, visuals, parts, dimensions, notes, extra):
    prop_id = folder.name
    props.render_preview(visuals, folder / "preview.png")
    data = {
        "id": prop_id, "task": "tray_push", "units": "m", "origin": "bottom_center",
        "bbox_mm": [round(v, 5) for v in props.bbox_mm(visuals)], "budget_class": "medium",
        "triangles": props.triangle_count(visuals), "parts": parts, "variants": [],
        "source": "source.blend", "preview": "preview.png", "dimensions_mm": dimensions,
        "notes": notes + " " + COMMON_NOTES + f"Generated with Blender {bpy.app.version_string}.",
        **extra,
    }
    scene = bpy.context.scene
    scene["prop_id"] = prop_id
    scene["bake_recipe"] = "CPU Cycles 64; diffuse COLOR * mix(1, AO, 0.6); 8-bit sRGB"
    scene["dimensions_mm"] = json.dumps(dimensions)
    bpy.context.preferences.filepaths.save_version = 0
    props._select(visuals)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    props.write_prop_json(data, folder / "prop.json")
    print("TASK03_COMPLETE", prop_id, data["bbox_mm"], data["triangles"],
          sum(len(p["collision"]) for p in parts), flush=True)


def build_esd_tray():
    folder = start("esd_tray")
    mask = folder / "esd_print_mask.png"
    subprocess.run([str(props.VALIDATOR_PYTHON), str(Path(__file__).resolve()),
                    "--draw-esd-mask", str(mask)], check=True)
    body = hollow_tray(plastic_material(mask))
    for side in (-1, 1):
        cutter = props.extruded_profile("70 x 22 R6 finger hole", rounded_rectangle(70, 22, 6),
                                        12, bevel_mm=0)
        move(cutter, (0, side * 148.5, 32), (90, 0, 0))
        difference(body, cutter)
    # 1 mm around the rim/base; the small handle edges receive the same radius.
    bevel(body, 1.0)
    collisions = tray_collisions()
    part = export_part(folder, "body", body, collisions, 2048, .35, .25, PP_MJ)
    complete(folder, [body], [part], {
        "outside_xyz": [200, 300, 50], "wall": 3, "floor": 3,
        "outer_corner_radius": 8, "inner_corner_radius": 5, "hard_edge_radius": 1,
        "finger_hole_width_height": [70, 22], "finger_hole_corner_radius": 6,
        "finger_hole_centres": [[0, -148.5, 32], [0, 148.5, 32]],
        "esd_print_width_height": [84, 25],
    }, (
        "Continuous open-top conductive PP #1B1B1B, roughness 0.65; two real 70 x 22 mm "
        "short-wall finger openings centred at Z32 (chosen R6), nominal dimensions before the 1 mm edge roll. "
        "Original yellow ESD triangle/finger/spark and ESD SAFE screen print on -X. "
        "Five-piece collision is the mandated floor plus four walls, never a cavity-filling hull. "
        "This fixed convex-piece count cannot reproduce concave finger holes within the general 1 mm rule: "
        "the two holes are deliberately filled in the proxies. Each wall proxy is a convex strip clipped to the "
        "R8 outer outline; its inner plane is inset 0.5 mm into the cavity to keep the rounded corner seam error "
        "below 1 mm while retaining only four convex walls. The floor preserves its R8 outline with 56 vertices. "
        "Task 1 payloads are reused by scene assembly and are not rebuilt or embedded here. "
        "Friction 0.25 is the intended tray/rail sliding coefficient; use an explicit simulator contact pair if combining "
        "it with another material's coefficient. See the rail set's support-plane integration note."
    ), {
        "contact_pairs": [{"other_prop": "tray_rail_set", "other_part": "frame", "sliding_friction": .25}],
        "recommended_placement_m": {"x": .4, "initial_y": .25, "slide_axis": [0, -1, 0],
                                      "support_z": .8},
        "payload_reuse": ["flange_part", "abs_housing"],
    })


def tray_collisions():
    floor = props.extruded_profile("COL tray floor R8", rounded_rectangle(200, 300, 8, 6),
                                   3, (0, 0, 1.5), bevel_mm=0)
    result = [floor]
    outline = rounded_rectangle(200, 300, 8)
    for side in (-1, 1):
        for axis, threshold in ((0, 96.5), (1, 146.5)):
            clipped = []
            for first, second in zip(outline, outline[1:] + outline[:1]):
                a, b = side * first[axis] - threshold, side * second[axis] - threshold
                if a >= 0:
                    clipped.append(first)
                if (a < 0) != (b < 0):
                    t = a / (a - b)
                    clipped.append(tuple(x + t * (y - x) for x, y in zip(first, second)))
            result.append(props.extruded_profile("COL rounded wall strip", clipped, 47,
                                                  (0, 0, 26.5), bevel_mm=0))
    return result


def t_profile():
    # Four undercut slots: 6 mm mouth, 8.5 mm cavity, 5 mm depth. The
    # 0.75 mm diagonal webs keep adjacent cavities from intersecting.
    top = [(10, 10), (3, 10), (3, 8.5), (4.25, 8.5), (4.25, 5),
           (-4.25, 5), (-4.25, 8.5), (-3, 8.5), (-3, 10), (-10, 10)]
    profile = []
    for quarter in range(4):
        angle = quarter * math.pi / 2
        c, s = round(math.cos(angle)), round(math.sin(angle))
        profile.extend((c * x - s * y, s * x + c * y) for x, y in top[:-1])
    return profile


def extrusion(name, length, position, rotation, material):
    obj = props.extruded_profile(name, t_profile(), length, bevel_mm=0, material=material)
    bore = props.cylinder("4.2 mm extrusion core bore", 2.1, length + 4, bevel_mm=0, vertices=24)
    difference(obj, bore)
    bevel(obj, .3)
    return move(obj, position, rotation)


def socket_head(name, position, material):
    obj = props.cylinder(name, 4.2, 1.6, bevel_mm=.2, vertices=32, material=material)
    cutter = props.hexagonal_prism("4 mm hex socket", 4, 1.3, (0, 0, .65), bevel_mm=.08)
    difference(obj, cutter)
    return move(obj, position)


def build_tray_rail_set():
    folder = start("tray_rail_set")
    long_al = aluminium_material("Anodised 2020 rails, longitudinal brushing", "X")
    cross_al = aluminium_material("Anodised 2020 cross bars, transverse brushing", "Y")
    steel = props.pbr_material("Blackened socket fasteners", "#36383B", .4, .8, grime=.12)
    frame_objects = []
    for side in (-1, 1):
        frame_objects.append(extrusion("700 mm 2020 guide rail", 700, (side * 113, 0, 30),
                                        (90, 0, 0), long_al))
        frame_objects.append(extrusion("246 mm 2020 cross bar", 246, (0, side * 300, 10),
                                        (90, 0, 90), cross_al))
        for y in (-300, 300):
            frame_objects.append(socket_head("Recessed rail fixing", (side * 113, y, 39.8), steel))
    frame = props.join_objects(frame_objects, "frame")
    frame_collisions = []
    for side in (-1, 1):
        frame_collisions.append(props.box("COL 700 rail", (20, 700, 20), (side * 113, 0, 30), bevel_mm=0))
        frame_collisions.append(props.box("COL 246 cross bar", (246, 20, 20), (0, side * 300, 10), bevel_mm=0))
    block = props.box("40 x 40 x 30 aluminium stop", (40, 40, 30), (0, -300, 35),
                      bevel_mm=1, material=stop_material())
    rubber = props.pbr_material("Black rubber bumper, 5 mm", "#151515", .90, 0, grime=.15)
    bumper = props.box("5 mm bumper facing positive Y", (40, 5, 30), (0, -277.5, 35),
                       bevel_mm=.5, material=rubber)
    stop = props.join_objects([block, bumper], "stop")
    # One mandated stop hull includes the full 5 mm bumper thickness.
    stop_collision = [props.box("COL stop and bumper", (40, 45, 30), (0, -297.5, 35), bevel_mm=0)]
    parts = [export_part(folder, "frame", frame, frame_collisions, 2048, 1.0, .25, AL_MJ),
             export_part(folder, "stop", stop, stop_collision, 1024, .13, .9, AL_MJ, parent="frame")]
    complete(folder, [frame, stop], parts, {
        "rail_section_xz": [20, 20], "rail_length_y": 700, "rail_centres_x": [-113, 113],
        "rail_inside_spacing": 206, "t_slot_mouth": 6, "t_slot_cavity_width": 8.5,
        "t_slot_depth": 5, "extrusion_core_bore_diameter": 4.2,
        "cross_bar_xyz": [246, 20, 20], "cross_bar_centres_y": [-300, 300],
        "cross_bar_inset_from_rail_end": 50, "cross_bar_z_range": [0, 20],
        "rail_z_range": [20, 40], "stop_block_xyz": [40, 40, 30],
        "stop_block_centre": [0, -300, 35], "bumper_thickness_y": 5,
        "bumper_contact_plane_y": -275, "stop_z_range": [20, 50],
    }, (
        "Two 700 mm 20 x 20 anodised aluminium profiles at X +/-113, inside gap 206; "
        "two 246 mm cross bars at Y +/-300, exactly 50 mm inboard. Four real undercut 6 mm T-slot mouths "
        "per profile, chosen 8.5 mm cavities/5 mm depth and through core bores; "
        "0.3 mm three-segment profile edge breaks. "
        "Cross bars are below the rails so they do not block the tray travel. The 40 x 40 x 30 stop is at "
        "Y-300, atop the negative-end cross bar, with a 5 mm black rubber bumper facing +Y and original "
        "yellow/black 45-degree tape printed on its top. Mass is estimated (1.0 kg frame, 0.13 kg stop); "
        "section 3.3 gives no rail-set mass. Frame friction is 0.25 for the tray; stop friction is 0.9. "
        "Five prescribed convex proxies total: four full rectangular profile envelopes and one stop/bumper envelope. "
        "The general 1 mm collision fidelity rule is incompatible with one convex proxy per undercut profile: "
        "T slots and core bores are deliberately filled in these proxies. "
        "Integration constraint in the literal dimensions: a 200 mm tray is narrower than the 206 mm rail gap; "
        "these profiles are lateral guides and do not support its floor along the travel. The scene must provide "
        "a continuous sliding support plane flush with the cross-bar tops (rail-set local Z20), or an independently "
        "designed support. No unrequested support slab or change to the specified 206 mm gap is baked into this prop. "
        "The Y+250 suggested tray start extends 50 mm beyond the +Y rail end; the support surface must cover that overhang. "
        "Rigid bolted profiles and the fixed stop share the prop frame; no joint or free body is required for the rail set."
    ), {
        "contact_pairs": [{"other_prop": "esd_tray", "other_part": "body", "sliding_friction": .25}],
        "recommended_placement_m": {"x": .4, "y": 0, "bottom_z": .78,
                                      "external_sliding_support_z": .8,
                                      "tray_bottom_local_z": .020, "initial_tray_y": .25,
                                      "stopped_tray_y": -.125},
        "installation": {"fixed_to_world": True, "requires_external_sliding_support": True,
                           "sliding_support_local_z_m": .020,
                           "stop_bumper_normal": [0, 1, 0]},
    })


def preview_only(prop_id):
    folder = OUTPUT / prop_id
    data = json.loads((folder / "prop.json").read_text())
    bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
    objects = [bpy.data.objects[part["source_visual_object"]] for part in data["parts"]]
    props.render_preview(objects, folder / "preview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=("esd_tray", "tray_rail_set"),
                        default=["esd_tray", "tray_rail_set"])
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    for prop_id in args.props:
        if args.preview_only:
            preview_only(prop_id)
        else:
            {"esd_tray": build_esd_tray, "tray_rail_set": build_tray_rail_set}[prop_id]()


if __name__ == "__main__":
    main()
    props.finalize_assets()
