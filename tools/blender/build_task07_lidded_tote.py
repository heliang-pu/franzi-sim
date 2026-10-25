"""Build only benchmark_props.md sections 1, 2 and 3.7 assets.

Blender -b --factory-startup --python-exit-code 1 -P this_file -- [--props ID ...]
--geometry-only saves resumable inspection blends under tools/blender/_qa/task07.
--export-only resumes those blends without rebuilding other tasks or geometry.
--finish-only refreshes metadata and PBR finishing on already delivered props.
Dimensions are millimetres; prop_lib writes metres in a common closed prop frame.
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

import bpy
from mathutils import Matrix, Vector

import prop_lib as props
from build_task01_tote_transfer import (
    bevel_edges,
    boolean,
    finish_mesh,
    loft,
    move_mesh,
    plastic_material,
    revolve,
    rr_ring,
    rounded_rectangle,
)

IDS = ("lidded_tote", "station_plate", "drawer_cabinet")
OUTPUT = props.ROOT / "assets" / "props"
QA = HERE / "_qa" / "task07"
PP = {"specular": .25, "shininess": .30, "reflectance": 0.0}
ALUMINIUM = {"specular": .70, "shininess": .70, "reflectance": .05}
PAINT = {"specular": .30, "shininess": .30, "reflectance": 0.0}
COMMON_NOTES = (
    "Metres, Z up, front -X; closed parts share the same prop frame, all transforms applied. "
    "Triangulated OBJ with normals, 30-degree smoothing and a nonoverlapping Smart UV 66-degree atlas. "
    "Each rigid part has a 2048-square, 8-bit sRGB albedo: CPU Cycles 64-sample DIFFUSE COLOR * "
    "(0.4 + 0.6 * AO), without direct light or highlights. Full Principled PBR is packed in source.blend. "
    "The 512-square preview uses CPU Cycles 64, three lights, neutral grey and a front-left camera. "
    "Joint dynamics are integration parameters, not a claim of an executed robot episode. "
)


def box(name, size, center=(0, 0, 0), material=None, bevel=0, rotation=(0, 0, 0)):
    return props.box(name, size, center, bevel_mm=bevel, segments=3,
                     material=material, rotation_deg=rotation)


def hide_collisions(objects):
    for obj in objects:
        obj.hide_render = True
        obj.hide_set(True)
        obj.display_type = "WIRE"
        obj["role"] = "convex_collision_proxy"


def translate(objects, delta):
    transform = Matrix.Translation(Vector([props.mm(v) for v in delta]))
    for obj in objects:
        obj.data.transform(transform)
        obj.data.update()


def grip_smudges(material, center, radius=45):
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    original = bsdf.inputs["Base Color"].links[0].from_socket
    coord = tree.nodes.new("ShaderNodeTexCoord")
    distance = tree.nodes.new("ShaderNodeVectorMath")
    distance.operation = "DISTANCE"
    tree.links.new(coord.outputs["Object"], distance.inputs[0])
    distance.inputs[1].default_value = tuple(props.mm(v) for v in center)
    mask = props._math(tree, "MAXIMUM", 0, props._math(tree, "SUBTRACT", 1,
                       props._math(tree, "DIVIDE", distance.outputs["Value"], props.mm(radius))))
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.name = "Subtle handling fingerprints and dirt"
    noise.inputs["Scale"].default_value = 460
    noise.inputs["Detail"].default_value = 3
    tree.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    shade = props._math(tree, "SUBTRACT", 1, props._math(tree, "MULTIPLY", .17,
                        props._math(tree, "MULTIPLY", mask, noise.outputs["Fac"])))
    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.blend_type, mix.inputs[0].default_value = "MULTIPLY", 1
    tree.links.new(original, mix.inputs[1])
    tree.links.new(shade, mix.inputs[2])
    tree.links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    return material


def image_on_surface(obj, material, path, mapping, ink=False):
    layer = obj.data.uv_layers.get("PrintUV") or obj.data.uv_layers.new(name="PrintUV")
    for loop in obj.data.loops:
        xyz = tuple(float(v) * 1000 for v in obj.data.vertices[loop.vertex_index].co)
        layer.data[loop.index].uv = mapping(xyz)
    tree = material.node_tree
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "PrintUV"
    image = tree.nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(path), check_existing=False)
    image.image.colorspace_settings.name = "sRGB"
    image.extension = "EXTEND"
    tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
    if ink:
        bsdf = tree.nodes["Principled BSDF"]
        old = bsdf.inputs["Base Color"].links[0].from_socket
        mix = tree.nodes.new("ShaderNodeMixRGB")
        tree.links.new(image.outputs["Alpha"], mix.inputs[0])
        tree.links.new(old, mix.inputs[1])
        tree.links.new(image.outputs["Color"], mix.inputs[2])
        tree.links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    else:
        rgb = next(n for n in tree.nodes if n.type == "RGB")
        for link in list(rgb.outputs[0].links):
            tree.links.new(image.outputs["Color"], link.to_socket)


def label(folder, name, text, size, center, front=False, lines=()):
    path = props.make_label_png(folder / f"{name}_print_source.png", 1200, 500,
                              {"text": text, "lines": list(lines),
                               "background": "#F0EEE4", "foreground": "#252A30"})
    mat = props.pbr_material(name + " paper and printed ink", "#F0EEE4", .82, 0, grime=.07)
    obj = box(name, size, center, mat, bevel=.07)
    if front:
        image_on_surface(obj, mat, path,
                         lambda p: (.5 - (p[1] - center[1]) / size[1],
                                    .5 + (p[2] - center[2]) / size[2]))
    else:
        image_on_surface(obj, mat, path,
                         lambda p: (.5 - (p[1] - center[1]) / size[1],
                                    .5 + (p[0] - center[0]) / size[0]))
    return obj


def part(name, visual, collision, mass, friction, material, parent=None, joint=None):
    return {"name": name, "visual_object": visual.name,
            "collision_objects": [obj.name for obj in collision], "mass_kg": mass,
            "friction": friction, "mujoco_material": material, "parent": parent, "joint": joint}


def finish_metadata(data):
    """Integration details independent of UVs or bake pixels; safe on a resumed build."""
    if data["id"] == "lidded_tote":
        lid = next(p for p in data["parts"] if p["name"] == "lid")
        lid["joint"]["solimpfriction"] = [.9999, .9999, .001]
        lid["joint"]["solreffriction"] = [.002, 1]
        data["recommended_placement_m"] = {"body_origin": [.445, .15, .8],
                                             "nominal_pull_tab": [.334637, .15, .961937]}
        if "MuJoCo soft-friction" not in data["notes"]:
            data["notes"] += (
                " MuJoCo soft-friction defaults allow numerical creep even below the 0.4 N m "
                "threshold. Apply joint solimpfriction=[0.9999,0.9999,0.001] and "
                "solreffriction=[0.002,1] for static holding, with damping/frictionloss unchanged. "
                "Suggested body placement X445 mm puts the initial tab around X335 mm, visible "
                "to the head camera and inside the left hand workspace. The full 110-degree "
                "mechanical arc exceeds the fixed robot's nominal X window; task integration "
                "must use a partial opening or regrasp rather than assume one grasp follows "
                "the complete arc. The preferred 200 mm height band is also exceeded while opening."
            )
    elif data["id"] == "drawer_cabinet":
        shift = data["dimensions_mm"]["shell_center_x"]
        data["recommended_placement_m"] = {"shell_center": [.50, -.15, .8],
                                             "origin": [.50 - props.mm(shift), -.15, .8]}
    return data


def silkscreen_pbr(material):
    tree = material.node_tree
    if tree.nodes.get("Silkscreen nonmetallic mask"):
        return
    bsdf = tree.nodes["Principled BSDF"]
    mix = next(n for n in tree.nodes if n.type == "MIX_RGB" and n.inputs[0].is_linked)
    mask = mix.inputs[0].links[0].from_socket
    metal = props._math(tree, "SUBTRACT", 1, mask)
    metal.node.name = "Silkscreen nonmetallic mask"
    tree.links.new(metal, bsdf.inputs["Metallic"])
    tree.links.new(props._math(tree, "ADD", .30, props._math(tree, "MULTIPLY", .35, mask)),
                   bsdf.inputs["Roughness"])


def finish_delivered(prop_id):
    """Update metadata/PBR finishing without rebaking unchanged diffuse/AO pixels."""
    folder = OUTPUT / prop_id
    data = finish_metadata(json.loads((folder / "prop.json").read_text()))
    bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
    manifest = finish_metadata(json.loads(bpy.context.scene["task07_manifest"]))
    bpy.context.scene["task07_manifest"] = json.dumps(manifest)
    for p in manifest["parts"]:
        bpy.data.objects[p["visual_object"]]["joint_json"] = json.dumps(p["joint"])
    if prop_id == "station_plate":
        mat = next(m for m in bpy.data.materials if m.name.startswith("Milled aluminium #C5CACF"))
        silkscreen_pbr(mat)
        props.render_preview([bpy.data.objects["body"]], folder / "preview.png")
    elif prop_id == "drawer_cabinet":
        visuals = [bpy.data.objects[p["visual_object"]] for p in manifest["parts"]]
        upper = bpy.data.objects["drawer_upper"]
        upper.location.x = -.18
        bpy.context.view_layer.update()
        props.render_preview(visuals, folder / "preview_open.png")
        upper.location.x = 0
        bpy.context.view_layer.update()
    props.write_prop_json(data, folder / "prop.json")
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    print("TASK07_FINISHED", prop_id, flush=True)


def start(prop_id):
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    folder = OUTPUT / prop_id
    folder.mkdir(parents=True, exist_ok=True)
    print("TASK07_START", prop_id, flush=True)
    return folder


def complete(folder, parts, dimensions, notes, origin="bottom_center", extra=None):
    visuals = [bpy.data.objects[p["visual_object"]] for p in parts]
    for p, obj in zip(parts, visuals, strict=True):
        obj.name = p["name"]
        p["visual_object"] = obj.name
        obj["rigid_part"] = p["name"]
        obj["joint_json"] = json.dumps(p["joint"])
        finish_mesh(obj, weighted_normals=True)
        props.uv_unwrap(obj)
        # Guard the OBJ exporter's six-decimal UV rounding at atlas borders.
        for loop in obj.data.uv_layers["UVMap"].data:
            loop.uv = (.0005 + .999 * loop.uv.x, .0005 + .999 * loop.uv.y)
        hide_collisions([bpy.data.objects[name] for name in p["collision_objects"]])
    data = {"id": folder.name, "task": "lidded_tote", "units": "m", "origin": origin,
            "bbox_mm": [round(v, 6) for v in props.bbox_mm(visuals)],
            "triangles": props.triangle_count(visuals), "budget_class": "medium",
            "source": "source.blend", "preview": "preview.png", "parts": parts, "variants": [],
            "dimensions_mm": dimensions,
            "notes": notes + " " + COMMON_NOTES + f"Generated with Blender {bpy.app.version_string}.",
            **(extra or {})}
    assert data["triangles"] <= 50_000, data["triangles"]
    for p in parts:
        assert len(p["collision_objects"]) <= 64
    scene = bpy.context.scene
    scene["task07_manifest"] = json.dumps(data)
    scene["prop_id"] = folder.name
    scene["dimensions_mm"] = json.dumps(dimensions)
    scene["bake_recipe"] = "CPU Cycles 64; DIFFUSE COLOR * (0.4 + 0.6 * AO); RGB8 sRGB"
    scene["preview_recipe"] = "Cycles CPU 64; 512 square; grey; three lights; front-left"
    bpy.context.preferences.filepaths.save_version = 0
    QA.mkdir(parents=True, exist_ok=True)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(QA / f"{folder.name}_geometry.blend"), check_existing=False)
    (QA / f"{folder.name}_preflight.json").write_text(json.dumps(data, indent=2) + "\n")
    print("TASK07_GEOMETRY", folder.name, data["bbox_mm"], data["triangles"],
          [len(p["collision_objects"]) for p in parts], flush=True)
    if not ARGS.geometry_only:
        export_current()


def export_current():
    data = finish_metadata(json.loads(bpy.context.scene["task07_manifest"]))
    bpy.context.scene["task07_manifest"] = json.dumps(data)
    folder = OUTPUT / data["id"]
    visuals, exported = [], []
    for p in data["parts"]:
        obj = bpy.data.objects[p["visual_object"]]
        obj["joint_json"] = json.dumps(p["joint"])
        if data["id"] == "station_plate":
            silkscreen_pbr(next(m for m in obj.data.materials if m.name.startswith("Milled aluminium #C5CACF")))
        collisions = [bpy.data.objects[name] for name in p["collision_objects"]]
        hide_collisions(collisions)
        for other in bpy.data.objects:
            if other.type == "MESH":
                other.hide_render = other != obj
        obj.hide_set(False)
        obj.hide_render = False
        # Separate rigid parts must not bake an occlusion shadow from a closed neighbour.
        print("TASK07_BAKE", data["id"], p["name"], flush=True)
        props.bake_albedo([obj], 2048, folder / f"{p['name']}_albedo.png")
        collision_files = props.export_collision(p["name"], collisions, folder)
        hide_collisions(collisions)
        props.export_obj(obj, folder / f"{p['name']}.obj")
        exported.append({"name": p["name"], "visual": f"{p['name']}.obj",
                         "texture": f"{p['name']}_albedo.png", "texture_px": 2048,
                         "collision": collision_files, "mass_kg": p["mass_kg"],
                         "friction": p["friction"], "mujoco_material": p["mujoco_material"],
                         "parent": p["parent"], "joint": p["joint"],
                         "source_visual_object": obj.name,
                         "source_collision_objects": p["collision_objects"]})
        visuals.append(obj)
    for obj in visuals:
        obj.hide_render = False
        obj.hide_set(False)
    props.render_preview(visuals, folder / "preview.png")
    # Additional operating views use temporary transforms, never altered export coordinates.
    if data["id"] == "lidded_tote":
        lid = next(obj for obj in visuals if obj.name == "lid")
        anchor = Vector((.102, 0, .152))
        lid.matrix_world = (Matrix.Translation(anchor) @ Matrix.Rotation(math.radians(100), 4, "Y") @
                            Matrix.Translation(-anchor))
        bpy.context.view_layer.update()
        props.render_preview(visuals, folder / "preview_open.png")
        lid.matrix_world = Matrix.Identity(4)
        bpy.context.view_layer.update()
    elif data["id"] == "drawer_cabinet":
        upper = next(obj for obj in visuals if obj.name == "drawer_upper")
        upper.location.x = -.18
        bpy.context.view_layer.update()
        props.render_preview(visuals, folder / "preview_open.png")
        upper.location.x = 0
        bpy.context.view_layer.update()
    data["parts"] = exported
    props.write_prop_json(data, folder / "prop.json")
    # Curved handle facets can lose exact convexity at the OBJ writer's 1 um
    # precision. Re-hull each quantized piece without moving any point.
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(HERE / "convexify_task07_collisions.py"),
                    "--props", data["id"]], check=True)
    for image in bpy.data.images:
        if image.source == "FILE" and image.filepath:
            image.pack()
    props._select(visuals)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    print("TASK07_COMPLETE", data["id"], data["bbox_mm"], data["triangles"],
          sum(len(p["collision"]) for p in exported), flush=True)


def drafted_wall(name, axis, side, outer, inner, z0=4, z1=146):
    rings = []
    for z in (z0, z1):
        ox, oy = outer(z)
        ix, iy = inner(z)
        polygon = rounded_strip(ox, oy, ix if axis == "X" else iy, 5, axis, side)
        rings.append([(x, y, z) for x, y in polygon])
    return loft(name, rings)


def rounded_strip(ox, oy, inner, radius, axis, side):
    """Convex side strip clipped to the actual rounded outside contour."""
    if axis == "Y":
        ox, oy = oy, ox
    theta = math.acos(max(-1, min(1, (inner - (ox - radius)) / radius)))
    polygon = []
    for sign in (-1, 1):
        angles = [-theta + theta * i / 4 for i in range(5)] if sign < 0 else [theta * i / 4 for i in range(5)]
        for angle in angles:
            polygon.append((ox - radius + radius * math.cos(angle),
                            sign * (oy - radius) + radius * math.sin(angle)))
    if axis == "Y":
        polygon = [(-y, x) for x, y in polygon]
    if side < 0:
        polygon = [(-x, -y) for x, y in polygon]
    return polygon


def rim_proxy(name, ox, oy, inner, radius, z0, z1, axis, side):
    return props.extruded_profile(name, rounded_strip(ox, oy, inner, radius, axis, side),
                                  z1 - z0, (0, 0, (z0 + z1) / 2), bevel_mm=0)


def rounded_proxy(name, width, length, height, radius, center):
    return props.extruded_profile(name, rounded_rectangle(width, length, radius, segments=4),
                                  height, center, bevel_mm=0)


def make_hinge(y, body_items, lid_items, steel, pp):
    # Two fixed outer knuckles and one moving centre knuckle, with axial gaps.
    for offset in (-15.15, 15.15):
        obj = props.tube("Fixed hinge knuckle", 4, 2.62, 9.7, bevel_mm=.25,
                         vertices=32, material=steel)
        move_mesh(obj, (102, y + offset, 152), (90, 0, 0))
        body_items.append(obj)
    obj = props.tube("Moving hinge knuckle", 4, 2.62, 20.1, bevel_mm=.25,
                     vertices=32, material=steel)
    move_mesh(obj, (102, y, 152), (90, 0, 0))
    lid_items.append(obj)
    body_items.append(props.cylinder("Diameter 5 hinge pin", 2.5, 40, (102, y, 152),
                                     bevel_mm=.2, vertices=32, material=steel, rotation_deg=(90, 0, 0)))
    body_items.append(box("Fixed hinge leaf", (3, 39, 13), (99.8, y, 143.5), steel, .6))
    lid_items.append(box("Moving hinge leaf", (2.4, 20, 12.5), (100.8, y, 158.1), steel, .5))
    # Standoffs bridge the fixed leaf to the rim without floating hardware.
    body_items.append(box("Hinge mounting shoulder", (6.2, 39, 4), (97.0, y, 148), pp, .7))
    for dy in (-12, 12):
        body_items.append(props.cylinder("Fixed hinge rivet", 2.1, 1.0, (101.55, y + dy, 142),
                                         bevel_mm=.2, vertices=24, material=steel,
                                         rotation_deg=(0, 90, 0)))


def build_tote():
    folder = start("lidded_tote")
    pp = plastic_material("Grey PP #7A7F85 - mould grain and scuffs", "#7A7F85", .55, tote=True)
    grip_smudges(pp, (-111, 0, 162), 44)
    steel = props.pbr_material("Hinge brushed steel", "#BFC3C7", .30, 1,
                               grime=.12, machining="brushed")
    slope = math.tan(math.radians(2))
    thickness = 3 / math.cos(math.radians(2))

    def outer(z):
        return 94 - (150 - z) * slope, 144 - (150 - z) * slope

    def inner(z):
        x, y = outer(z)
        return x - thickness, y - thickness

    ox, oy = outer(0)
    ix, iy = inner(4)
    shoulder = outer(146)
    body = loft("Continuous drafted open PP shell", [
        rr_ring(2 * ox, 2 * oy, 5, 0, 4),
        rr_ring(2 * shoulder[0], 2 * shoulder[1], 5, 146, 4),
        rr_ring(200, 300, 5, 146, 4), rr_ring(200, 300, 5, 150, 4),
        rr_ring(2 * inner(150)[0], 2 * inner(150)[1], 2, 150, 4),
        rr_ring(2 * ix, 2 * iy, 2, 4, 4),
    ], pp, bevel=.7)
    boolean(body, box("Real catch receiving groove", (3, 19, 4.5), (-99.8, 0, 146)))
    bevel_edges(body, .25, angle=45)
    body_items = [body]
    # Closed 15 mm shell seats on the rim: 3 mm crown plus 12 mm full skirt.
    lid = loft("Continuous 304 x 204 x 15 lid shell", [
        rr_ring(204, 304, 4, 165, 4), rr_ring(204, 304, 4, 150, 4),
        rr_ring(201, 301, 2.5, 150, 4), rr_ring(201, 301, 2.5, 162, 4),
    ], pp, bevel=.45)
    root = (-101.5, 0, 163.5)
    tab_center = (root[0] - 9 * math.cos(math.radians(10)), 0,
                  root[2] - 9 * math.sin(math.radians(10)))
    tab = box("60 x 18 x 3 pull tab tilted down ten degrees", (18, 60, 3),
              tab_center, pp, .7, (0, -10, 0))
    boolean(lid, tab, "UNION")
    catch = box("Snap catch flexible stem", (1.6, 16, 9), (-101.0, 0, 149), pp, .35)
    boolean(lid, catch, "UNION")
    hook = props.extruded_profile("Snap catch inward tooth", [(-101.6, 144.8), (-98.6, 146),
                                  (-98.6, 147.6), (-101.6, 147.6)], 15,
                                  bevel_mm=.25, material=pp)
    # Extruded X/Z profile: the original profile's Y becomes Z, extrusion becomes Y.
    move_mesh(hook, (0, 0, 0), (90, 0, 0))
    boolean(lid, hook, "UNION")
    lid_items = [lid, label(folder, "parts_label", "PARTS · LINE 3", (40, 112, .16),
                            (-8, 0, 165.08), lines=("COMPONENT STORAGE", "OPEN / PICK / CLOSE"))]
    for y in (-94, 94):
        make_hinge(y, body_items, lid_items, steel, pp)
    body = props.join_objects(body_items, "body")
    lid = props.join_objects(lid_items, "lid")
    collisions = [rounded_proxy("Tote floor proxy", 2 * ox, 2 * oy, 4, 5, (0, 0, 2))]
    for axis in ("X", "Y"):
        for side in (-1, 1):
            collisions.append(drafted_wall("Drafted wall proxy", axis, side, outer, inner))
    for side in (-1, 1):
        collisions.append(rim_proxy("Rim X proxy", 100, 150, 93.8, 5, 146, 150, "X", side))
        collisions.append(rim_proxy("Rim Y proxy", 100, 150, 143.8, 5, 146, 150, "Y", side))
    lid_collisions = [rounded_proxy("Lid roof proxy", 204, 304, 3, 4, (0, 0, 163.5))]
    for side in (-1, 1):
        lid_collisions.append(rim_proxy("Lid X skirt proxy", 102, 152, 100.5, 4, 150, 162, "X", side))
        lid_collisions.append(rim_proxy("Lid Y skirt proxy", 102, 152, 150.5, 4, 150, 162, "Y", side))
    lid_collisions.append(box("Pull tab proxy", (18, 60, 3), tab_center, rotation=(0, -10, 0)))
    hinge = {"type": "hinge", "parent": "body", "axis": [0, 1, 0], "anchor_m": [.102, 0, .152],
             "range": [0.0, math.radians(110)], "damping": .05, "frictionloss": .4}
    complete(folder, [part("body", body, collisions, 1.0, .45, PP),
                      part("lid", lid, lid_collisions, .25, .45, PP, "body", hinge)],
             {"body_outer_rim_xyz": [200, 300, 150], "draft_degrees": 2,
              "wall_normal_thickness": 3, "floor_thickness": 4, "rim_width": 6, "rim_thickness": 4,
              "wall_top_xy": [188, 288], "wall_base_xy": [2 * ox, 2 * oy],
              "lid_crown_and_skirt_xyz": [204, 304, 15], "lid_shell_z": [150, 165],
              "lid_skirt_drop": 12, "lid_skirt_thickness": 1.5,
              "hinge_centers_y": [-94, 94], "hinge_length": 40, "hinge_pin_diameter": 5,
              "hinge_axis_xz": [102, 152], "pull_tab_yxz": [60, 18, 3], "pull_tab_down_degrees": 10},
             "The 200 x 300 x 150 mm envelope includes the 6 mm outward rim; 2-degree draft "
             "therefore applies to the 188 x 288 mm wall top. Origin exception: body_bottom_center "
             "preserves the exact hinge x=102, z=152 mm datum; the forward tab makes the complete "
             "assembly bounding box asymmetric. The specification does not locate the 15 mm lid "
             "vertically; it is seated at Z150..165, crown Z162..165 and full 12 mm skirt Z150..162. "
             "This raised crown keeps all four skirts and avoids rear-skirt/body interference around "
             "the mandated high hinge. Label thickness and hinge hardware are additional details. "
             "Positive rotation about +Y opens upward toward +X; the illustrative -Y axis in section "
             "1.8 would open downward. Frictionloss 0.4 N m exceeds the 0.25 kg lid's gravity torque "
             "and is intended to hold any opening angle; damping is 0.05. Snap catch return torque "
             "is 0.3 N m toward q=0 only within 0..3 degrees, to be implemented by the simulator, "
             "not as a full-range hinge spring. Body proxies expand the prescribed floor+four-wall "
             "decomposition with four rim bars to preserve the 6 mm overhang without filling the "
             "open cavity. Lid has exactly roof+four skirts+pull tab proxies. Pins, knuckles and "
             "compliant catch are non-contact hardware; the receiving groove is visual and must "
             "not lock the simulated hinge. Rounded proxy corners differ by up to 1 mm. "
             "Contents reuse flange_part, abs_housing and hex_blank; they are not copied or rebuilt.",
             origin="body_bottom_center",
             extra={"recommended_placement_m": {"body_origin": [.42, .15, .8],
                                                      "nominal_pull_tab": [.31, .15, .962]},
                    "interaction": {"lid_grip_span_mm": 60, "catch_return_torque_Nm": .3,
                                    "catch_active_range_rad": [0, math.radians(3)],
                                    "reuse_prop_ids": ["flange_part", "abs_housing", "hex_blank"]}})


def build_station():
    folder = start("station_plate")
    aluminium = props.pbr_material("Milled aluminium #C5CACF", "#C5CACF", .30, 1,
                                   grime=.12, machining="milled")
    steel = props.pbr_material("Blackened countersunk screws", "#34383E", .4, .8, grime=.10)
    plate = box("150 X x 200 Y x 10 aluminium plate", (150, 200, 10), (0, 0, 5), aluminium, 1)
    items = [plate]
    for x in (-62, 62):
        for y in (-87, 87):
            cutter = revolve("Real through bore and 90-degree countersink", [
                (0, -1), (2.6, -1), (2.6, 7.5), (5.6, 10.5), (0, 10.5)], 32)
            move_mesh(cutter, (x, y, 0))
            boolean(plate, cutter)
            screw = revolve("Flush M4 countersunk screw", [(0, .7), (2, .7), (2, 7.4),
                             (4.4, 9.8), (4.4, 9.88), (0, 9.88)], 32, steel)
            for size in ((5.6, .9, .9), (.9, 5.6, .9)):
                boolean(screw, box("Recessed screw cross drive", size, (0, 0, 9.75)))
            bevel_edges(screw, .12, angle=35)
            move_mesh(screw, (x, y, 0))
            items.append(screw)
    bevel_edges(plate, .18, angle=45)
    art = folder / "station_print_source.png"
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(HERE / "task07_artwork.py")],
                   input=json.dumps({"path": str(art)}), text=True, check=True)
    image_on_surface(plate, aluminium, art, lambda p: (.5 - p[1] / 200, .5 + p[0] / 150), ink=True)
    # Mask printing to the planar top; projected ink must not wrap onto edges or bores.
    tree = aluminium.node_tree
    mix = next(n for n in tree.nodes if n.type == "MIX_RGB" and n.inputs[0].is_linked)
    alpha = mix.inputs[0].links[0].from_socket
    geom = tree.nodes.new("ShaderNodeNewGeometry")
    normal = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(geom.outputs["Normal"], normal.inputs[0])
    tree.links.new(props._math(tree, "MULTIPLY", alpha,
                              props._math(tree, "GREATER_THAN", normal.outputs["Z"], .95)), mix.inputs[0])
    silkscreen_pbr(aluminium)
    body = props.join_objects(items, "body")
    collision = [box("Plate box proxy", (150, 200, 10), (0, 0, 5))]
    complete(folder, [part("body", body, collision, .80, .35, ALUMINIUM)],
             {"outside_xyz": [150, 200, 10], "screw_centers_xy": [[x, y] for x in (-62, 62) for y in (-87, 87)],
              "through_hole_diameter": 5.2, "countersink_diameter_at_top": 10.2,
              "part_outline_diameter": 60, "center_cross_span": 14},
             "Static 200 Y x 150 X x 10 mm milled aluminium station. Four real through holes with "
             "90-degree countersinks contain flush cross-drive screws. Yellow ST-02, the 60 mm "
             "flange outline, bore/bolt references and centre cross are surface print baked into "
             "the plate atlas, without raised decal geometry. The task-prescribed single box proxy "
             "fills the occupied screw holes. Station mass 0.80 kg and friction 0.35 are reasonable "
             "defaults because section 3.7 gives neither; fix the plate to the worktop in simulation.",
             extra={"fixed": True, "recommended_placement_m": {"origin": [.40, -.20, .80]}})


def arch_handle(z, steel):
    # 100 mm overall span, 15 mm overall projection from the drawer face.
    path = [(-126, -46, z), (-133, -43, z), (-136, -35, z),
            (-136, 35, z), (-133, 43, z), (-126, 46, z)]
    curve = bpy.data.curves.new("Arched handle solid sweep", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth, curve.bevel_resolution = props.mm(4), 3
    curve.resolution_u = 3
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(path) - 1)
    for point, xyz in zip(spline.bezier_points, path, strict=True):
        point.co = [props.mm(v) for v in xyz]
        point.handle_left_type = point.handle_right_type = "AUTO"
    curve.use_fill_caps = True
    obj = bpy.data.objects.new("Silver arched pull handle", curve)
    bpy.context.collection.objects.link(obj)
    props._select([obj])
    bpy.ops.object.convert(target="MESH")
    obj = bpy.context.object
    # Auto Bezier overshoot is normalised to exact outside 100 Y x 15 X mm.
    coords = [v.co.copy() for v in obj.data.vertices]
    lo = Vector(tuple(min(p[i] for p in coords) for i in range(3)))
    hi = Vector(tuple(max(p[i] for p in coords) for i in range(3)))
    for vertex in obj.data.vertices:
        vertex.co.x = props.mm(-140) + (vertex.co.x - lo.x) / (hi.x - lo.x) * props.mm(18)
        vertex.co.y = -props.mm(50) + (vertex.co.y - lo.y) / (hi.y - lo.y) * props.mm(100)
    obj.data.materials.append(steel)
    # Use the actual swept mesh split by Y for faithful convex handle proxies.
    proxies = []
    import bmesh
    for index, (y0, y1) in enumerate(zip((-51, -42, -30, 30, 42), (-42, -30, 30, 42, 51), strict=True)):
        mesh = obj.data.copy()
        bm = bmesh.new()
        bm.from_mesh(mesh)
        for boundary, normal in ((y0, (0, -1, 0)), (y1, (0, 1, 0))):
            geometry = list(bm.verts) + list(bm.edges) + list(bm.faces)
            bmesh.ops.bisect_plane(bm, geom=geometry, dist=1e-8,
                                  plane_co=(0, props.mm(boundary), 0), plane_no=normal,
                                  clear_outer=True, clear_inner=False)
        bm.to_mesh(mesh)
        bm.free()
        proxy = bpy.data.objects.new(f"Arched handle proxy {index}", mesh)
        bpy.context.collection.objects.link(proxy)
        proxies.append(proxy)
    return obj, proxies


def build_cabinet():
    folder = start("drawer_cabinet")
    paint = plastic_material("Blue-grey powder-coated steel #3E6C9C", "#3E6C9C", .60)
    steel = props.pbr_material("Brushed silver drawer pulls and rails", "#BFC3C7", .30, 1,
                               grime=.13, machining="brushed")
    shell_specs = [((250, 300, 1), (0, 0, .5)), ((250, 300, 1), (0, 0, 199.5)),
                   ((248, 1, 198), (0, -149.5, 100)), ((248, 1, 198), (0, 149.5, 100)),
                   ((1, 298, 198), (124.5, 0, 100))]
    shell_items = [box("1 mm steel cabinet shell", size, center, paint, .35) for size, center in shell_specs]
    body_col = [box("Cabinet shell proxy", size, center) for size, center in shell_specs]
    # Fixed slides occupy the space outside the exact 270 mm drawer cavity.
    for y in (-142, 142):
        for z in (18, 118):
            shell_items.append(box("Fixed telescopic guide rail", (226, 11, 12), (-6, y, z), steel, .4))
    body = props.join_objects(shell_items, "body")
    definitions = [("drawer_lower", 0, "D-01"), ("drawer_upper", 100, "D-02")]
    drawers = []
    for name, offset, code in definitions:
        # A continuous shell provides the exact 230 X x 270 Y x 80 Z clear cavity.
        drawer = loft(name + " continuous open drawer", [
            rr_ring(232, 272, 1.2, 5 + offset, 2), rr_ring(232, 272, 1.2, 86 + offset, 2),
            rr_ring(230, 270, .2, 86 + offset, 2), rr_ring(230, 270, .2, 6 + offset, 2),
        ], paint, bevel=.25)
        translate([drawer], (-8, 0, 0))
        panel = box("290 x 95 drawer front", (1, 290, 95), (-124.5, 0, 50 + offset), paint, .35)
        boolean(drawer, panel, "UNION")
        grip_mat = steel.copy()
        grip_mat.name = name + " brushed handle fingerprints"
        grip_smudges(grip_mat, (-136, 0, 68 + offset), 55)
        handle, handle_col = arch_handle(68 + offset, grip_mat)
        items = [drawer, handle]
        for y in (-46, 46):
            items.append(props.cylinder("Handle mounting boss", 5, 2.0, (-125.5, y, 68 + offset),
                                         bevel_mm=.3, vertices=24, material=grip_mat,
                                         rotation_deg=(0, 90, 0)))
        # Fifty by twenty millimetre label pocket, with physical retaining rails.
        card = label(folder, name + "_label", code, (.15, 48, 18), (-125.08, -92, 30 + offset),
                     front=True, lines=("PARTS / LINE 3",))
        items.append(card)
        for y in (-117.5, -66.5):
            items.append(box("Label slot vertical rail", (.9, 1, 20), (-125.1, y, 30 + offset), steel, .2))
        items.append(box("Label slot lower retaining rail", (.9, 52, 1),
                         (-125.1, -92, 20 + offset), steel, .2))
        visual = props.join_objects(items, name)
        drawer_specs = [((232, 272, 1), (-8, 0, 5.5 + offset)),
                        ((232, 1, 80), (-8, -135.5, 46 + offset)),
                        ((232, 1, 80), (-8, 135.5, 46 + offset)),
                        ((1, 270, 80), (-123.5, 0, 46 + offset)),
                        ((1, 270, 80), (107.5, 0, 46 + offset)),
                        ((1, 290, 95), (-124.5, 0, 50 + offset))]
        collisions = [box("Drawer floor/wall/front proxy", size, center) for size, center in drawer_specs]
        collisions.extend(handle_col)
        joint = {"type": "slide", "parent": "body", "axis": [-1, 0, 0], "anchor_m": [0, 0, 0],
                 "range": [0.0, .18], "damping": 1.0, "frictionloss": .15}
        drawers.append(part(name, visual, collisions, .6, .30, PAINT, "body", joint))
    # Unlike the tote, the cabinet has no specified absolute hinge datum. Centre
    # the complete closed bounding box, including the asymmetric handles.
    all_parts = [part("body", body, body_col, 4.0, .5, PAINT), *drawers]
    visuals = [bpy.data.objects[p["visual_object"]] for p in all_parts]
    xs = [v.co.x * 1000 for obj in visuals for v in obj.data.vertices]
    shift = -(min(xs) + max(xs)) / 2
    every_object = visuals + [bpy.data.objects[n] for p in all_parts for n in p["collision_objects"]]
    translate(every_object, (shift, 0, 0))
    for p in drawers:
        p["joint"]["anchor_m"] = [props.mm(shift), 0, 0]
    complete(folder, all_parts,
             {"cabinet_shell_xyz": [250, 300, 200], "shell_thickness": 1,
              "shell_center_x": shift, "drawer_front_yz": [290, 95],
              "drawer_clear_cavity_xyz": [230, 270, 80], "drawer_floor_z": [5, 105],
              "handle_span_projection": [100, 15], "handle_grip_diameter": 8,
              "label_slot_yz": [50, 20], "drawer_stroke": 180},
             "Optional two-drawer variant, built in full. Closed steel shell is 250 X x 300 Y x 200 Z mm "
             "with 1 mm sheets; both 290 x 95 mm fronts and both 230 X x 270 Y x 80 mm cavities fit "
             "without reducing the specified dimensions. The complete closed assembly, including "
             "15 mm projecting handles, is translated to a bottom bounding-box-centre datum; "
             f"the nominal shell centre is X={shift:.6f} mm in that frame. Each drawer has its own "
             "-X slide, stroke 0..0.180 m, damping 1.0, and default frictionloss 0.15 N. Cabinet "
             "4 kg is fixed; each drawer is 0.6 kg. The 100 mm arched handles have real open finger "
             "clearance, brushed silver surfaces and handling smudges. A 50 x 20 mm label pocket "
             "is retained on each front. Five shell proxies and six pan/front proxies per drawer "
             "are supplemented by five convex handle pieces per drawer so robot fingers do not "
             "pass through the pulls or a filled-in arch. Non-contact telescopic guide rails are "
             "visual only. Sheet edges use 0.25..0.35 mm bevels to fit the 1 mm steel; handle edges "
             "are rounded. Friction defaults are cabinet 0.50 and drawers 0.30. At the example "
             "placement, the first 60 mm of pull travel lies in the robot's nominal X300..480 mm "
             "workspace; the full 180 mm joint stroke is available for scene-specific placement.",
             extra={"fixed": True,
                    "recommended_placement_m": {"shell_center": [.4925, -.15, .8],
                                                 "origin": [.4925 - props.mm(shift), -.15, .8]},
                    "robot_nominal_pull_travel_m": .06})


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--geometry-only", action="store_true")
    modes.add_argument("--export-only", action="store_true")
    modes.add_argument("--finish-only", action="store_true")
    ARGS = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    for prop_id in ARGS.props:
        if ARGS.finish_only:
            finish_delivered(prop_id)
        elif ARGS.export_only:
            bpy.ops.wm.open_mainfile(filepath=str(QA / f"{prop_id}_geometry.blend"))
            export_current()
        else:
            {"lidded_tote": build_tote, "station_plate": build_station,
             "drawer_cabinet": build_cabinet}[prop_id]()


if __name__ == "__main__":
    main()
    props.finalize_assets()
