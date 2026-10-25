"""Build benchmark_props.md sections 1, 2 and 3.11, and no other props.

Blender -b --factory-startup --python-exit-code 1 -P this_file -- [--props ID ...]
--geometry-only saves resumable sources; --export-only resumes their export.
--render-exports renders the delivered OBJ/albedo files for independent visual QA.
All geometry arguments are millimetres; exported coordinates are metres, Z up.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
OUTPUT = ROOT / "assets" / "props"
QA = HERE / "_qa" / "task11"
IDS = ("workbench", "factory_floor", "safety_fence", "background_rack")


def floor_artwork():
    """Deterministic, seamless 1 m epoxy substrate, including dragged scuffs."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    from PIL.PngImagePlugin import PngInfo

    folder = OUTPUT / "factory_floor"
    folder.mkdir(parents=True, exist_ok=True)
    n = 1024
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 1, (n, n)).astype(np.float32)
    smooth = noise.copy()
    for shift in (2, 5, 13, 31, 63):
        smooth = (smooth + np.roll(smooth, shift, 0) + np.roll(smooth, -shift, 0)
                  + np.roll(smooth, shift, 1) + np.roll(smooth, -shift, 1)) / 5
    grain = noise * 0.6 + smooth * 20
    color = np.array([140, 145, 150], dtype=np.float32)
    array = np.clip(color + grain[:, :, None], 0, 255).astype(np.uint8)
    image = Image.fromarray(array, "RGB")
    scuffs = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    draw = ImageDraw.Draw(scuffs)
    randomizer = random.Random(11)
    for _ in range(28):
        x, y = randomizer.randrange(n), randomizer.randrange(n)
        length = randomizer.randint(35, 220)
        angle = randomizer.uniform(-0.5, 0.5)
        for dx in (-n, 0, n):
            for dy in (-n, 0, n):
                xy = [(x + dx, y + dy),
                      (x + dx + length, y + dy + length * angle)]
                draw.line(xy, fill=(43, 48, 51, randomizer.randint(15, 32)), width=2)
    scuffs = scuffs.filter(ImageFilter.GaussianBlur(0.7))
    image = Image.alpha_composite(image.convert("RGBA"), scuffs).convert("RGB")
    info = PngInfo()
    info.add(b"sRGB", b"\x00")
    image.save(folder / "epoxy_1m_source.png", pnginfo=info)


if __name__ == "__main__" and "--floor-artwork-worker" in sys.argv:
    floor_artwork()
    raise SystemExit(0)

import bmesh
import bpy
from mathutils import Vector

import prop_lib as props

PAINT = {"specular": 0.30, "shininess": 0.30, "reflectance": 0.0}
PLASTIC = {"specular": 0.25, "shininess": 0.30, "reflectance": 0.0}
PAPER = {"specular": 0.05, "shininess": 0.10, "reflectance": 0.0}
COMMON = (
    "Static environment geometry; mass values are descriptive, not free-body dynamics. "
    "Metres, Z up, robot-facing front -X; transforms applied in the shared prop frame. "
    "30-degree split smoothing and triangulated OBJ with exported normals. "
    "New parts have nonoverlapping Smart UV 66-degree atlases with 0.01 island margins. "
    "Albedo = CPU Cycles 64-sample diffuse COLOR * (0.4 + 0.6 * AO), RGB8 sRGB, "
    "without direct illumination or highlights. Full procedural Principled PBR and source "
    "images remain packed in source.blend. Source roughness variation cannot be represented "
    "by MuJoCo's single highlight setting per part. Preview: CPU Cycles 64, 512 square, "
    "neutral grey, three area lights, front-left above. "
)


def log(*args):
    print("TASK11", *args, flush=True)


def start(prop_id):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    folder = OUTPUT / prop_id
    folder.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    log("START", prop_id, bpy.app.version_string)
    return folder


def box(name, size, center, material, edge=2, segments=3):
    obj = props.box(name, size, center, edge, segments, material)
    if material and material.get("task11_edge_material") and edge:
        # Pointiness interpolates across an entire sparsely tessellated box
        # face. Give only actual bevel faces the worn paint, keeping broad
        # surfaces at the specified colour instead of washing them grey.
        obj.data.materials.append(bpy.data.materials[material["task11_edge_material"]])
        for face in obj.data.polygons:
            if max(abs(v) for v in face.normal) < .99999:
                face.material_index = 1
    return obj


def paint(name, color, roughness=.60):
    material = props.pbr_material(name, color, roughness, 0, grime=.15, machining="brushed")
    rgb = props._linear_color(color)
    worn = tuple(.82 * value + .18 * .55 for value in rgb[:3])
    encoded = [round(255 * (12.92 * v if v <= .0031308 else 1.055 * v ** (1 / 2.4) - .055))
               for v in worn]
    edge_color = "#" + "".join(f"{v:02X}" for v in encoded)
    edge = props.pbr_material(name + " / 2 mm worn bevels", edge_color, .48, .2, grime=.15,
                              machining="brushed")
    material["task11_edge_material"] = edge.name
    return material


def rod(name, a, b, radius, material, vertices=12, edge=0.15):
    delta = Vector(b) - Vector(a)
    center = (Vector(a) + Vector(b)) / 2
    obj = props.cylinder(name, radius, delta.length, bevel_mm=edge, segments=2,
                         vertices=vertices, material=material)
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    obj.location = center / 1000
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def beam(name, a, b, width, depth, material):
    delta = Vector(b) - Vector(a)
    obj = box(name, (width, depth, delta.length), (0, 0, 0), material, edge=1)
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    obj.location = (Vector(a) + Vector(b)) / 2000
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def merge(objects, name):
    obj = props.join_objects(objects, name) if len(objects) > 1 else objects[0]
    obj.name = name
    return obj


def finish(obj):
    """Prevent tiny bevel slivers collapsing in the six-decimal OBJ writer."""
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for vertex in bm.verts:
        vertex.co = tuple(round(v, 6) for v in vertex.co)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=2e-7)
    bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=2e-7)
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    seen, bad = set(), []
    for face in bm.faces:
        key = frozenset(face.verts)
        if key in seen or face.calc_area() <= 2e-18:
            bad.append(face)
        else:
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
    compact_materials(obj)


def compact_materials(obj):
    """Remove unused empty slots without touching existing mesh/UV topology."""
    old = list(obj.data.materials)
    materials = list(dict.fromkeys(m for m in old if m is not None))
    assignments = [materials.index(old[p.material_index]) if old[p.material_index] is not None else 0
                   for p in obj.data.polygons]
    obj.data.materials.clear()
    for material in materials:
        obj.data.materials.append(material)
    for polygon, index in zip(obj.data.polygons, assignments, strict=True):
        polygon.material_index = index


def record(name, obj, mass, material=PAINT, friction=0.5, collisions=(), parent=None, **extra):
    for i, collision in enumerate(collisions):
        collision.name = f"COL__{name}__{i:02d}"
        collision.hide_render = True
        collision.hide_set(True)
    obj["rigid_part"] = name
    return {"name": name, "visual": f"{name}.obj", "texture": f"{name}_albedo.png",
            "texture_px": 2048, "collision": [], "mass_kg": mass, "friction": friction,
            "mujoco_material": dict(material), "parent": parent, "joint": None,
            "source_visual_object": obj.name,
            "source_collision_objects": [c.name for c in collisions], **extra}


def fit_uv(obj):
    """Uniformly fit an overflowing packed atlas, without stacking any islands.

    Hundreds of actual peg holes can exhaust Blender's fixed fractional packing
    margins. Preserve the complete layout, then normalize it into the PNG.
    """
    layer = obj.data.uv_layers["UVMap"]
    lower = [min(v.uv[i] for v in layer.data) for i in range(2)]
    upper = [max(v.uv[i] for v in layer.data) for i in range(2)]
    if min(lower) >= 0 and max(upper) <= 1:
        return 1.0
    scale = .9998 / max(upper[i] - lower[i] for i in range(2))
    for corner in layer.data:
        corner.uv = tuple(.0001 + (corner.uv[i] - lower[i]) * scale for i in range(2))
    log("NORMALIZE_PACKED_UV", obj.name, scale)
    return scale


def uv_area(obj):
    import numpy as np

    layer = obj.data.uv_layers["UVMap"]
    data = np.empty(len(obj.data.loops) * 2, dtype=np.float32)
    layer.data.foreach_get("uv", data)
    corners = data.reshape(-1, 3, 2)  # finish() triangulates every polygon.
    a, b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    return float(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2)


def unwrap_dense(obj):
    """Preserve positive-area islands when a perforated sheet exhausts gutters."""
    props._select([obj])
    obj.data.uv_layers.active = obj.data.uv_layers["UVMap"]
    obj.data.uv_layers["UVMap"].active_render = True
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=.002, margin_method="FRACTION")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, scale=True, merge_overlap=False,
                           margin_method="FRACTION", margin=.002, shape_method="AABB")
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.data.update()
    if uv_area(obj) < .01:
        raise ValueError(f"UV packing collapsed: {obj.name}")


def complete(folder, parts, dimensions, notes, **extra):
    visuals = []
    for part in parts:
        obj = bpy.data.objects[part["source_visual_object"]]
        if part.get("library_instance"):
            # The library is already cleaned and triangulated. Re-welding
            # transformed seams can replace a loop UV on a tiny triangle,
            # so preserve its topology and UV corners verbatim.
            compact_materials(obj)
        else:
            finish(obj)
        if not part.get("library_instance"):
            props.uv_unwrap(obj)
            if uv_area(obj) < .01:
                unwrap_dense(obj)
                part["uv_island_margin"] = .002
            for corner in obj.data.uv_layers["UVMap"].data:
                corner.uv = (0.0001 + 0.9998 * corner.uv.x, 0.0001 + 0.9998 * corner.uv.y)
            scale = fit_uv(obj)
            if scale < 1:
                part["uv_layout_fit_scale"] = scale
        visuals.append(obj)
    data = {"id": folder.name, "task": "environment", "units": "m", "origin": "bottom_center",
            "bbox_mm": [round(v, 4) for v in props.bbox_mm(visuals)],
            "triangles": props.triangle_count(visuals), "budget_class": "large", "static": True,
            "source": "source.blend", "preview": "preview.png", "parts": parts, "variants": [],
            "dimensions_mm": dimensions, "notes": notes + " " + COMMON,
            "build_blender_version": bpy.app.version_string, **extra}
    if any("uv_island_margin" in p for p in parts):
        data["notes"] += (" Blender's fixed 0.01 fractional margin collapses the numerous peg-hole "
                          "UV islands to zero area. This dense part uses Smart UV at the same "
                          "66 degrees with a 0.002 fractional island margin (about 4 pixels at "
                          "2048), preserving positive-area, nonoverlapping islands. The actual "
                          "uv_island_margin is recorded per affected part.")
    if any("uv_layout_fit_scale" in p for p in parts):
        data["notes"] += (" The perforated sheet's numerous islands exceed the unit tile at a fixed "
                          "0.01 fractional packing margin; its entire packed layout is uniformly "
                          "normalized into [0,1], preserving nonoverlap and proportional gutters. "
                          "The normalization factor is recorded per affected part.")
    if data["triangles"] > 150000:
        raise ValueError(f"Triangle budget exceeded: {data['triangles']}")
    bpy.context.scene["task11_manifest"] = json.dumps(data)
    bpy.context.scene["bake_recipe"] = "CPU Cycles 64 DIFFUSE COLOR * (0.4 + 0.6 * AO), RGB8 sRGB"
    props._select(visuals)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(QA / f"{folder.name}_geometry.blend"), check_existing=False)
    (QA / f"{folder.name}_preflight.json").write_text(json.dumps(data, indent=2) + "\n")
    log("GEOMETRY", folder.name, data["bbox_mm"], data["triangles"])
    if not ARGS.geometry_only:
        export_current()


def export_current():
    data = json.loads(bpy.context.scene["task11_manifest"])
    folder = OUTPUT / data["id"]
    previous = json.loads((folder / "prop.json").read_text()) if (folder / "prop.json").is_file() else {}
    old_parts = {part["name"]: part for part in previous.get("parts", [])}
    visuals = [bpy.data.objects[p["source_visual_object"]] for p in data["parts"]]
    for part, obj in zip(data["parts"], visuals, strict=True):
        changed_uv = any(part.get(key) != old_parts.get(part["name"], {}).get(key)
                         for key in ("uv_island_margin", "uv_layout_fit_scale"))
        if not part.get("library_instance"):
            if uv_area(obj) < .01:
                unwrap_dense(obj)
                part["uv_island_margin"] = .002
                changed_uv = True
                data["notes"] += (" The peg-hole atlas required a 0.002 fractional margin because "
                                  "0.01 collapsed its numerous islands to zero area. Smart UV "
                                  "remains 66 degrees with nonoverlapping positive-area islands.")
            scale = fit_uv(obj)
            if scale < 1:
                changed_uv = True
                part["uv_layout_fit_scale"] = scale
                data["notes"] += (" The fixed 0.01 fractional packing margin overflows on this "
                                  "part's many small islands. Its whole packed UV layout is "
                                  "uniformly normalized into the unit PNG, preserving nonoverlap "
                                  "and proportional gutters; uv_layout_fit_scale records this "
                                  "necessary margin reduction for the perforated/rod geometry.")
        for other in bpy.context.scene.objects:
            if other.type == "MESH":
                other.hide_render = other != obj
        obj.hide_set(False)
        if not part.get("library_instance") and (changed_uv or not
                (ARGS.reuse_bakes and (folder / part["texture"]).is_file())):
            log("BAKE", data["id"], part["name"])
            props.bake_albedo([obj], part["texture_px"], folder / part["texture"])
        part["visual"] = props.export_obj(obj, folder / part["visual"])
        part["collision"] = props.export_collision(
            part["name"], [bpy.data.objects[n] for n in part["source_collision_objects"]], folder)
        for n in part["source_collision_objects"]:
            bpy.data.objects[n].hide_render = True
            bpy.data.objects[n].hide_set(True)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in visuals
            obj.hide_set(obj not in visuals)
    props.render_preview(visuals, folder / "preview.png")
    props.write_prop_json(data, folder / "prop.json")
    bpy.context.scene["task11_manifest"] = json.dumps(data)
    for image in bpy.data.images:
        if image.source == "FILE":
            image.use_fake_user = True
    props._select(visuals)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    log("COMPLETE", data["id"], data["triangles"])


def fine_surface(material, period=0.0015, strength=0.025):
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1 / period
    noise.inputs["Detail"].default_value = 2
    tree.links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
    base = bsdf.inputs["Base Color"].links[0].from_socket
    tint = tree.nodes.new("ShaderNodeMapRange")
    tint.inputs["To Min"].default_value = 1 - strength
    tint.inputs["To Max"].default_value = 1 + strength
    tree.links.new(noise.outputs["Fac"], tint.inputs["Value"])
    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1
    tree.links.new(base, mix.inputs[1])
    tree.links.new(tint.outputs["Result"], mix.inputs[2])
    tree.links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.00002
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def pegboard(material):
    """One welded, watertight sheet with 288 actual octagonal 6 mm holes."""
    vertices, faces, lookup = [], [], {}

    def vertex(point):
        key = tuple(round(v, 6) for v in point)
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append(key)
        return lookup[key]

    for row in range(12):
        z = 845 + row * 50
        for col in range(24):
            y = -575 + col * 50
            outer, inner = [], []
            for i in range(8):
                a = i * math.tau / 8
                c, s = math.cos(a), math.sin(a)
                scale = 25 / max(abs(c), abs(s))
                outer.append((y + scale * c, z + scale * s))
                inner.append((y + 3 * c, z + 3 * s))
            for i in range(8):
                j = (i + 1) % 8
                for x in (333, 335):
                    loop = [(x, *outer[i]), (x, *outer[j]), (x, *inner[j]), (x, *inner[i])]
                    faces.append(tuple(vertex(p) for p in loop))
                faces.append(tuple(vertex(p) for p in ((333, *inner[i]), (335, *inner[i]),
                                                       (335, *inner[j]), (333, *inner[j]))))
                a, b = outer[i], outer[j]
                boundary = (abs(abs(a[0]) - 600) < 1e-5 and abs(abs(b[0]) - 600) < 1e-5
                            or abs(a[1] - b[1]) < 1e-5 and (abs(a[1] - 820) < 1e-5
                                                                         or abs(a[1] - 1420) < 1e-5))
                if boundary:
                    faces.append(tuple(vertex(p) for p in ((333, *a), (335, *a), (335, *b), (333, *b))))
    return props._mesh_object("Perforated 1200 x 600 sheet / real holes", vertices, faces,
                              (0, 0, 0), 0.3, 2, material)


def flat_tool(name, outline_yz, center, thickness, material, angle=0):
    obj = props.extruded_profile(name, outline_yz, thickness, bevel_mm=0.4,
                                 segments=2, material=material)
    # The local 2-D profile becomes YZ; its extrusion becomes X.
    for v in obj.data.vertices:
        y, z, x = v.co
        a = math.radians(angle)
        v.co = (x + center[0] / 1000,
                y * math.cos(a) - z * math.sin(a) + center[1] / 1000,
                y * math.sin(a) + z * math.cos(a) + center[2] / 1000)
    obj.data.update()
    return obj


def build_workbench():
    folder = start("workbench")
    steel = paint("Blue-black powdercoat #3B3F44", "#3B3F44")
    zinc = props.pbr_material("Levelling screws / brushed zinc", "#BFC3C7", .30, 1, .04, .12, "brushed")
    rubber = props.pbr_material("Rubber foot pads", "#151515", .9, 0, .03, .15)
    topmat = props.pbr_material("40 mm laminate worktop", "#777C79", .60, 0, .08, .12)
    esd = paint("2 mm grey-green ESD #6B7B6E", "#6B7B6E", .68)
    fine_surface(esd, .0015, .03)
    pvc = paint("Black PVC front edge", "#181B1C", .55)
    boardmat = props.pbr_material("Warm grey powdercoat pegboard", "#A6A8A2", .60, 0, .12, .15)
    toolsteel = props.pbr_material("Tools / brushed steel", "#BFC3C7", .30, 1, .10, .15, "brushed")
    red = props.pbr_material("Worn red tool grips", "#A7392D", .58, 0, .14, .16)
    blue = props.pbr_material("Worn blue tool grips", "#264D71", .60, 0, .14, .16)
    frame = []
    for x in (-280, 280):
        for y in (-530, 530):
            frame.append(box("45 mm steel leg", (45, 45, 700), (x, y, 398), steel))
            frame.append(props.cylinder("Levelling rubber foot", 28, 10, (x, y, 5),
                                         bevel_mm=1.5, vertices=24, material=rubber))
            frame.append(props.cylinder("M16 levelling spindle", 8, 40, (x, y, 30),
                                         bevel_mm=.5, vertices=16, material=zinc))
            frame.append(props.hexagonal_prism("Levelling locknut", 24, 8, (x, y, 44), material=zinc))
    for z in (175, 724):
        for x in (-280, 280):
            frame.append(box("Long frame rail", (45, 1105, 40), (x, 0, z), steel))
        for y in (-530, 530):
            frame.append(box("Side frame rail", (560, 45, 40), (0, y, z), steel))
    frame.append(box("Lower folded steel shelf", (550, 1030, 3), (0, 0, 197.5), steel, edge=1))
    for x in (-274, 274):
        frame.append(box("Shelf return lip", (2, 1030, 18), (x, 0, 188), steel, edge=.7))
    for y in (-514, 514):
        frame.append(box("Shelf return lip", (550, 2, 18), (0, y, 188), steel, edge=.7))
    for y in (-575, 575):
        frame.append(box("Rear pegboard upright", (28, 28, 650), (334, y, 1095), steel))
    frame_obj = merge(frame, "frame")
    tabletop = merge([
        box("40 mm structural top", (696, 1200, 40), (2, 0, 778), topmat),
        box("Black PVC front edge", (4, 1200, 40), (-348, 0, 778), pvc, edge=1),
        box("2 mm ESD mat", (696, 1196, 2), (0, 0, 799), esd, edge=.5),
    ], "tabletop")
    backing = [pegboard(boardmat)]
    for z in (828, 1412):
        backing.append(box("Pegboard horizontal surround", (16, 1200, 16), (333, 0, z), steel))
    for y in (-592, 592):
        backing.append(box("Pegboard vertical surround", (16, 16, 568), (333, y, 1120), steel))
    # Open-ended spanners are concave solids, not painted silhouettes.
    wrench = [(-13, -93), (-20, -80), (-17, -58), (-7, -50), (-7, 50), (-18, 63),
              (-21, 82), (-13, 98), (-8, 94), (-10, 77), (0, 70), (10, 77), (8, 94),
              (13, 98), (21, 82), (18, 63), (7, 50), (7, -50), (17, -58), (20, -80),
              (13, -93), (8, -89), (10, -73), (0, -65), (-10, -73), (-8, -89)]
    for y, z, angle in ((-440, 1180, -8), (-300, 1160, 12)):
        backing.append(flat_tool("Open-ended steel spanner", wrench, (312, y, z), 5, toolsteel, angle))
        backing.append(rod("Pegboard hook", (333, y, z + 105), (309, y, z + 105), 2, toolsteel))
    for y, color in ((-110, red), (20, blue)):
        backing.append(rod("Screwdriver blade", (311, y, 1050), (311, y, 1200), 3, toolsteel, 16))
        backing.append(box("Screwdriver grip", (22, 24, 92), (311, y, 1240), color, edge=5))
        backing.append(rod("Screwdriver hanging hook", (333, y, 1295), (300, y, 1295), 2, toolsteel))
    backing.append(box("Hammer handle", (22, 22, 225), (309, 205, 1122.5), blue, edge=4))
    backing.append(box("Hammer steel head", (34, 125, 36), (307, 205, 1240), toolsteel, edge=3))
    backing.append(rod("Hammer hook", (333, 205, 1270), (297, 205, 1270), 2, toolsteel))
    for side in (-1, 1):
        backing.append(beam("Pliers jaws", (309, 395, 1190), (309, 395 + side * 17, 1238), 10, 8, toolsteel))
        backing.append(beam("Pliers red handle", (309, 395, 1190), (309, 395 + side * 27, 1100), 15, 12, red))
    backing.append(rod("Pliers steel pivot", (300, 395, 1190), (318, 395, 1190), 8, toolsteel, 24))
    backing.append(rod("Pliers hook", (333, 395, 1260), (298, 395, 1260), 2, toolsteel))
    for y, length in ((-480, 85), (-400, 100), (-310, 120)):
        backing.append(rod("Hex key long leg", (311, y, 950), (311, y, 950 + length), 3, toolsteel))
        backing.append(rod("Hex key short leg", (311, y, 950), (311, y + 28, 950), 3, toolsteel))
    board_obj = merge(backing, "pegboard_tools")
    collision = props.box("COL tabletop", (700, 1200, 42), (0, 0, 779), bevel_mm=0)
    complete(folder, [record("frame", frame_obj, 45),
                      record("tabletop", tabletop, 26, PLASTIC, .55, [collision], "frame"),
                      record("pegboard_tools", board_obj, 9, parent="frame")],
             {"bench_xyz": [700, 1200, 800], "top_substrate_thickness": 40, "esd_mat_thickness": 2,
              "finished_top_z": 800, "top_substrate_z": [758, 798], "pegboard_yz": [1200, 600],
              "pegboard_z": [820, 1420], "pegboard_holes": 288, "pegboard_hole_diameter": 6,
              "pegboard_pitch": 50, "lower_shelf_top_z": 199},
             "Finished ESD working surface is Z800: the exact 40 mm top lies at Z758..798, "
             "with a separate 2 mm grey-green mat above it. Black PVC is the -X front edge. "
             "The 1200 x 600 rear pegboard is on +X at Z820..1420, with 288 real through-holes "
             "and steel spanners, screwdrivers, hammer, pliers and hex keys facing the robot. "
             "Overall bbox includes the pegboard; 800 mm describes the bench working height. "
             "The task explicitly requests only ONE tabletop collision box; legs, lower shelf, "
             "feet and pegboard are visual-only and intentionally omitted from collision. "
             "Its top is exactly Z800, including the mat, with nominal XY 700 x 1200. "
             "Keep tools/background behind the robot manipulation workspace.")


def build_floor():
    folder = start("factory_floor")
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(__file__), "--floor-artwork-worker"], check=True)
    material = props.pbr_material("Epoxy #8C9196 / 1 m repeat / analytic safety paint", "#8C9196", .38, 0,
                                  grime=.12)
    tree, bsdf = material.node_tree, material.node_tree.nodes["Principled BSDF"]
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "FloorTileUV"
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.name = "One metre epoxy tile"
    texture.image = bpy.data.images.load(str(folder / "epoxy_1m_source.png"), check_existing=True)
    texture.extension = "REPEAT"
    tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
    position = tree.nodes.new("ShaderNodeNewGeometry")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(position.outputs["Position"], separate.inputs[0])
    x, y, z = separate.outputs

    def calc(op, *args):
        return props._math(tree, op, *args)

    def band(value, lower, upper):
        return calc("MULTIPLY", calc("GREATER_THAN", value, lower), calc("LESS_THAN", value, upper))

    # Continuous 75 mm line, with an L-shaped return. Coordinates are metres.
    line_a = calc("MULTIPLY", band(x, -1.4375, -1.3625), band(y, -2.3375, 2.1))
    line_b = calc("MULTIPLY", band(y, -2.3375, -2.2625), band(x, -1.4375, 1.8))
    line_mask = calc("MAXIMUM", line_a, line_b)
    zone_mask = calc("MULTIPLY", band(x, 1.0, 2.2), band(y, -2.3, -1.3))
    phase = calc("PINGPONG", calc("MULTIPLY", calc("ADD", x, y), math.sqrt(2) / .2), 1)
    stripe_mask = calc("GREATER_THAN", phase, .5)
    top = calc("GREATER_THAN", z, -.0001)
    yellow = props._linear_color("#E8BB24")
    stripe = tree.nodes.new("ShaderNodeMixRGB")
    stripe.inputs[1].default_value = yellow
    stripe.inputs[2].default_value = props._linear_color("#202326")
    tree.links.new(stripe_mask, stripe.inputs[0])
    zone = tree.nodes.new("ShaderNodeMixRGB")
    tree.links.new(calc("MULTIPLY", zone_mask, top), zone.inputs[0])
    tree.links.new(texture.outputs["Color"], zone.inputs[1])
    tree.links.new(stripe.outputs[0], zone.inputs[2])
    line = tree.nodes.new("ShaderNodeMixRGB")
    tree.links.new(calc("MULTIPLY", line_mask, top), line.inputs[0])
    tree.links.new(zone.outputs[0], line.inputs[1])
    line.inputs[2].default_value = yellow
    tree.links.new(line.outputs[0], bsdf.inputs["Base Color"])
    rough = tree.nodes.new("ShaderNodeTexNoise")
    rough.inputs["Scale"].default_value = 7
    tree.links.new(uv.outputs["UV"], rough.inputs["Vector"])
    variation = tree.nodes.new("ShaderNodeMapRange")
    variation.inputs["To Min"].default_value = .28
    variation.inputs["To Max"].default_value = .48
    tree.links.new(rough.outputs["Fac"], variation.inputs["Value"])
    tree.links.new(variation.outputs["Result"], bsdf.inputs["Roughness"])
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .08
    bump.inputs["Distance"].default_value = .00003
    tree.links.new(texture.outputs["Color"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    floor = box("floor", (6000, 6000, 10), (0, 0, -5), material, edge=.5)
    layer = floor.data.uv_layers.new(name="FloorTileUV")
    for loop in floor.data.loops:
        v = floor.data.vertices[loop.vertex_index].co
        layer.data[loop.index].uv = (v.x + 3, v.y + 3)
    plane = {"type": "plane", "pos_m": [0, 0, 0], "normal": [0, 0, 1], "size_m": [3, 3, .01]}
    complete(folder, [record("floor", floor, 720, {"specular": .35, "shininess": .4, "reflectance": .03},
                              .65, collision_primitives=[plane])],
             {"surface_xy": [6000, 6000], "surface_z": 0, "visual_thickness_below_surface": 10,
              "substrate_repeat_xy": [1000, 1000], "repeat_count_xy": [6, 6],
              "safety_line_width": 75, "hatched_zone_xy": [1200, 1000],
              "hatched_zone_bounds_xy": [[1000, 2200], [-2300, -1300]], "hatch_angle_deg": 45},
             "The contact surface and origin are Z0. A 10 mm visual-only slab extends down to Z-10 "
             "so the visual has a finite bbox; it adds no solid collision. Exactly ONE analytic "
             "MuJoCo plane is declared in floor.collision_primitives, with position, unit normal "
             "and MuJoCo size. The collision OBJ list is deliberately empty because a plane is "
             "not a convex volume; loaders must create type=plane from this explicit metadata. "
             "epoxy_1m_source.png is repeated exactly six times per axis in FloorTileUV; the "
             "unique baked 2048 atlas includes this 1 m substrate repetition, dragged scuffs, "
             "the globally placed 75 mm yellow L safety line and the yellow/black 45-degree zone. "
             "Warning markings are intentionally not periodic. Source gloss varies .28..48; "
             "MuJoCo uses the part's uniform highlight approximation.", origin="floor_surface_center")


def build_fence():
    folder = start("safety_fence")
    yellow = paint("Safety yellow powdercoat", "#E4B52D")
    dark = props.pbr_material("Dark steel welded wire", "#343A3C", .48, .75, .10, .15, "brushed")
    zinc = props.pbr_material("Fasteners / zinc", "#BFC3C7", .30, 1, .04, .12, "brushed")
    objects = []
    for y in (-480, 480):
        objects.append(box("Yellow 40 x 40 post", (40, 40, 1800), (0, y, 900), yellow))
    for z in (215, 1785):
        objects.append(box("Yellow panel cross rail", (30, 920, 30), (0, 0, z), yellow))
    objects.append(box("200 mm sheet-metal kick plate", (2, 920, 200), (0, 0, 100), yellow, edge=.5))
    vertical, horizontal = [], []
    for y in range(-450, 451, 50):
        vertical.append({"axis": "Z", "center_mm": [1.5, y, 1000], "length_mm": 1540,
                         "diameter_mm": 3})
        objects.append(rod("Real 3 mm vertical wire", (1.5, y, 230), (1.5, y, 1770), 1.5, dark, vertices=24))
    for z in range(250, 1751, 50):
        horizontal.append({"axis": "Y", "center_mm": [-1.5, 0, z], "length_mm": 920,
                           "diameter_mm": 3})
        objects.append(rod("Real 3 mm horizontal wire", (-1.5, -460, z), (-1.5, 460, z), 1.5, dark, vertices=24))
    for y in (-450, 450):
        for z in (95, 215, 1785):
            objects.append(rod("Panel mounting bolt", (-17, y, z), (-20, y, z), 4, zinc, 16, .3))
    obj = merge(objects, "panel")
    complete(folder, [record("panel", obj, 24)],
             {"nominal_panel_yz_including_posts": [1000, 1800], "post_section_xy": [40, 40],
              "kick_plate_height": 200, "kick_plate_thickness": 2, "rod_diameter": 3,
              "rod_center_pitch": 50, "vertical_rods": len(vertical), "horizontal_rods": len(horizontal)},
             "A single 1000 mm wide by 1800 mm high panel includes the two yellow 40 x 40 posts. "
             "The clear mesh spans Y-460..460 and Z230..1770 above a 200 mm sheet-metal kick plate. "
             "All 19 vertical and 31 horizontal wires are real closed 24-sided 3 mm diameter rods "
             "on exact 50 mm centre pitch, with small rounded end edges. Orthogonal wire planes "
             "are separated by 3 mm and tangent at crossings, as welded mesh; no alpha texture "
             "or painted transparency is used. The rod manifest records nominal manufacturing "
             "dimensions. Background only: NO collision geometry or primitives.", rod_manifest=vertical + horizontal)


def library_stock(folder, prop_id, name, position, scale, parent="frame", variant="A"):
    """Append the existing PBR mesh, preserve its atlas, and copy exact existing pixels."""
    library = OUTPUT / prop_id
    source = json.loads((library / "prop.json").read_text())
    part = next(p for p in source["parts"] if p["name"] == "body")
    source_name = part.get("source_visual_object", "body")
    with bpy.data.libraries.load(str(library / "source.blend"), link=False) as (available, loaded):
        if source_name not in available.objects:
            raise ValueError(f"Missing library visual {prop_id}/{source_name}")
        loaded.objects = [source_name]
    obj = loaded.objects[0]
    bpy.context.collection.objects.link(obj)
    obj.name = name
    obj.hide_set(False)
    obj.hide_render = False
    obj.scale = (scale,) * 3
    obj.location = Vector(position) / 1000
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    texture = part["texture"]
    selected = next((v for v in source["variants"] if v["name"] == variant), None)
    if selected:
        texture = selected.get("textures", {}).get("body", texture)
    filename = f"{prop_id}_{variant}_albedo.png"
    shutil.copyfile(library / texture, folder / filename)
    digest = hashlib.sha256((library / texture).read_bytes()).hexdigest()
    if variant not in ("A", "blue"):
        # Preserve the appended source PBR graph, switching only its existing
        # variant artwork image. The final atlas is reused without rebaking.
        for material in obj.data.materials:
            if material is None:
                continue
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image and node.image.name.startswith("label_A"):
                    node.image = bpy.data.images.load(str(library / f"label_{variant}.png"), check_existing=True)
    return record(name, obj, part["mass_kg"] * scale ** 3,
                  PLASTIC if prop_id == "small_tote" else PAPER, part["friction"], parent=parent,
                  texture=filename, texture_px=part["texture_px"],
                  library_instance={"prop_id": prop_id, "part_name": "body", "variant": variant,
                                    "uniform_scale": scale, "position_mm": position,
                                    "texture_sha256": digest})


def build_rack():
    folder = start("background_rack")
    blue = paint("Blue rack uprights #1E4C9A", "#1E4C9A")
    orange = paint("Orange rack beams #E86A10", "#E86A10")
    zinc = props.pbr_material("Galvanised shelf deck", "#BFC3C7", .4, 1, .06, .15, "brushed")
    steel = props.pbr_material("Rack footplate and anchors", "#454B52", .5, .8, .10, .15)
    objects = []
    for x in (-365, 365):
        for y in (-865, 865):
            objects.append(box("Blue upright", (70, 70, 2392), (x, y, 1204), blue))
            objects.append(box("Anchored upright footplate", (70, 70, 8), (x, y, 4), steel, edge=1))
            for offset in (-20, 20):
                objects.append(props.hexagonal_prism("Base anchor head", 12, 6,
                                                      (x + offset, y, 11), material=zinc))
    for level in (650, 1450):
        for x in (-365, 365):
            objects.append(box("Orange load beam", (50, 1660, 80), (x, 0, level - 40), orange))
            for y in (-815, 815):
                objects.append(box("Orange beam end connector", (58, 55, 130), (x, y, level - 35), orange))
        for y in (-865, 865):
            objects.append(box("Blue depth tie", (660, 35, 40), (0, y, level - 20), blue))
        for x in (-290, -145, 0, 145, 290):
            objects.append(box("Galvanised deck slat", (140, 1640, 5), (x, 0, level - 2.5), zinc, edge=.8))
    for y in (-865, 865):
        for za, zb in ((150, 850), (850, 1550), (1550, 2250)):
            objects.append(beam("Blue side diagonal brace", (-330, y, za), (330, y, zb), 20, 8, blue))
        objects.append(box("Top depth tie", (660, 35, 40), (0, y, 2360), blue))
    objects.append(box("Orange top rear tie", (40, 1660, 60), (365, 0, 2350), orange))
    frame = merge(objects, "frame")
    parts = [record("frame", frame, 125)]
    for layer, z in enumerate((650, 1450)):
        for i, y in enumerate((-475, 475)):
            parts.append(library_stock(folder, "small_tote", f"tote_{layer}_{i}", [-130, y, z], 2,
                                       variant="blue"))
        for i, y in enumerate((-530, 0, 530)):
            variant = ("A", "B", "C")[i]
            parts.append(library_stock(folder, "parcel_box_m", f"carton_{layer}_{i}", [230, y, z], 2.2,
                                       variant=variant))
        parts.append(library_stock(folder, "parcel_box_m", f"carton_{layer}_stack", [230, 0, z + 176],
                                   2.2, variant="B"))
        parts.append(library_stock(folder, "parcel_box_s", f"small_carton_{layer}", [-130, 0, z],
                                   2.2, variant="A"))
    complete(folder, parts,
             {"rack_xyz": [800, 1800, 2400], "shelf_top_z": [650, 1450], "upright_section_xy": [70, 70],
              "load_beam_height": 80, "tote_instances": 4, "carton_instances": 10},
             "Two loaded levels, blue #1E4C9A uprights and depth/diagonal braces, orange #E86A10 "
             "load beams and beam connectors, galvanised deck slats and anchored footplates. "
             "The rack is 800 X x 1800 Y x 2400 Z; the prompt's 2400 x 1800 x 800 uses H/W/D. "
             "All four totes append Task 1 small_tote PBR geometry, original UVs and byte-identical "
             "blue atlas; the ten cartons append Task 4 parcel_box_m/s PBR geometry and reuse "
             "byte-identical QF Express A/B/C atlases. These match the task's explicit Task 4/1 "
             "reuse instruction; section 3.11 alternatively mentions Task 5 cartons. Stock is "
             "uniformly scaled 2x for totes and 2.2x for cartons to suit a full-size background "
             "rack, without modifying any library asset. Each stock item is its own fixed part "
             "with its original nonoverlapping atlas; repeated items never stack UV islands "
             "inside one part. Per-instance lineage and texture SHA256 are included. Background "
             "only: NO collision geometry or primitives. The descriptive stock masses scale "
             "with volume; these are not manipulated benchmark instances.")


def render_exports(prop_id):
    folder = start(prop_id)
    data = json.loads((folder / "prop.json").read_text())
    visuals = []
    for part in data["parts"]:
        bpy.ops.wm.obj_import(filepath=str(folder / part["visual"]), forward_axis="Y", up_axis="Z")
        obj = bpy.context.selected_objects[0]
        obj.name = part["name"]
        material = props.pbr_material(part["name"] + " delivered atlas", "#FFFFFF", .5, 0)
        tree, bsdf = material.node_tree, material.node_tree.nodes["Principled BSDF"]
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(folder / part["texture"]), check_existing=True)
        tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
        obj.data.materials.clear()
        obj.data.materials.append(material)
        visuals.append(obj)
    props.render_preview(visuals, folder / "preview_export.png")
    log("RENDER_EXPORTS", prop_id)


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", choices=IDS, nargs="+", default=list(IDS))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--geometry-only", action="store_true")
    modes.add_argument("--export-only", action="store_true")
    modes.add_argument("--render-exports", action="store_true")
    parser.add_argument("--reuse-bakes", action="store_true")
    ARGS = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    builders = dict(zip(IDS, (build_workbench, build_floor, build_fence, build_rack), strict=True))
    for prop_id in ARGS.props:
        if ARGS.render_exports:
            render_exports(prop_id)
        elif ARGS.export_only:
            bpy.ops.wm.open_mainfile(filepath=str(QA / f"{prop_id}_geometry.blend"))
            export_current()
        else:
            builders[prop_id]()


if __name__ == "__main__":
    main()
    props.finalize_assets()
