"""Build only benchmark_props.md sections 1, 2 and 3.10, in millimetres.

Blender -b --factory-startup --python-exit-code 1 -P this_file -- [--props ID ...]
--geometry-only stores resumable geometry; --export-only finishes those sources.
The optional DIN kit deliberately has no collision or clip-on behaviour.
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
QA = HERE / "_qa" / "task10"
OUTPUT = HERE.parents[1] / "assets" / "props"
IDS = ("bearing_housing_c010", "bearing_housing_c020", "bearing_housing_c030",
       "bearing_6204", "dowel_pin", "pin_fixture", "din_rail_kit")


def artwork_worker():
    """Original fictional laser marking, projected onto the upper bearing face."""
    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo

    path = Path(sys.argv[-1])
    size = 2048
    image = Image.new("RGB", (size, size), "white")
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 82)
    text = "NVK 6204-2RS"
    radius = 21.75 / 47 * size
    # The top of each character faces radially outward, as on a laser-marked race.
    for index, char in enumerate(text):
        theta = math.radians(155 - index * 12)
        tile = Image.new("RGB", (120, 126), "white")
        ImageDraw.Draw(tile).text((60, 63), char, font=font, anchor="mm", fill="#41474B")
        tile = tile.rotate(math.degrees(theta) - 90, resample=Image.Resampling.BICUBIC, expand=True,
                           fillcolor="white")
        x = size / 2 + radius * math.cos(theta)
        y = size / 2 - radius * math.sin(theta)
        image.paste(tile, (round(x - tile.width / 2), round(y - tile.height / 2)))
    info = PngInfo()
    info.add(b"sRGB", b"\x00")
    image.save(path, pnginfo=info)


if __name__ == "__main__" and "--artwork-worker" in sys.argv:
    artwork_worker()
    raise SystemExit(0)

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

import prop_lib as props
from build_task06_shaped_kitting import boolean, finish, regular_polygon, wall_polygons

METAL = {"specular": .70, "shininess": .70, "reflectance": .05}
BLACK_STEEL = {"specular": .50, "shininess": .50, "reflectance": .02}
PLASTIC = {"specular": .30, "shininess": .40, "reflectance": 0.0}
COMMON = (
    "Metres, Z up, front -X, bottom-centre origin, applied transforms and a shared prop frame. "
    "Triangulated visual OBJ with normals, 30-degree smoothing, nonoverlapping 66-degree Smart UVs. "
    "Albedo: CPU Cycles 64-sample DIFFUSE COLOR * (0.4 + 0.6 * AO), RGB8 sRGB, no direct light "
    "or highlights. Full Principled PBR and marking images packed in source.blend. "
    "Preview: 1024 square, CPU Cycles 64, neutral grey, three lights, front-left above. "
    "Collision coordinates retain 12 decimal places in metres. Simulation may vary instance "
    "colour by 0.97-1.03. Suggested worktable position [0.400, 0, 0.800] m; positioning and "
    "reachable grasp poses are left to the scene, not baked into these local meshes. "
)


def log(*items):
    print("TASK10", *items, flush=True)


def hide(objects):
    for obj in objects:
        obj.hide_render = True
        obj.hide_set(True)


def start(prop_id):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    bpy.context.preferences.filepaths.save_version = 0
    folder = OUTPUT / prop_id
    folder.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    log("START", prop_id, bpy.app.version_string)
    return folder


def box(name, size, center, material=None, edge=.4, segments=3):
    return props.box(name, size, center, bevel_mm=edge, segments=segments, material=material)


def mesh(name, vertices, faces, material=None, edge=0):
    return props._mesh_object(name, vertices, faces, (0, 0, 0), edge, 3, material)


def merge(objects, name):
    obj = props.join_objects(objects, name) if len(objects) > 1 else objects[0]
    obj.name = name
    return obj


def solid_lathe(name, rings, count, material=None, apothem=False):
    """A capped solid of revolution. rings are (radius_mm, z_mm)."""
    scale = 1 / math.cos(math.pi / count) if apothem else 1
    vertices = [(r * scale * math.cos(2 * math.pi * i / count),
                 r * scale * math.sin(2 * math.pi * i / count), z)
                for r, z in rings for i in range(count)]
    faces = [tuple(reversed(range(count))), tuple(range((len(rings) - 1) * count, len(rings) * count))]
    for k in range(len(rings) - 1):
        for i in range(count):
            j = (i + 1) % count
            faces.append((k * count + i, k * count + j, (k + 1) * count + j, (k + 1) * count + i))
    return mesh(name, vertices, faces, material)


def ring_lathe(name, profile, count, materials, seal_edges=(), marking_edge=None):
    vertices = [(r * math.cos(2 * math.pi * i / count),
                 r * math.sin(2 * math.pi * i / count), z)
                for r, z in profile for i in range(count)]
    faces, material_indices = [], []
    for k in range(len(profile)):
        next_k = (k + 1) % len(profile)
        for i in range(count):
            j = (i + 1) % count
            faces.append((k * count + i, k * count + j, next_k * count + j, next_k * count + i))
            material_indices.append(1 if k in seal_edges else 2 if k == marking_edge else 0)
    obj = mesh(name, vertices, faces)
    for material in materials:
        obj.data.materials.append(material)
    for polygon, index in zip(obj.data.polygons, material_indices, strict=True):
        polygon.material_index = index
    return obj


def extrude_xz(name, profile, depth, y, material, edge=.15):
    n = len(profile)
    vertices = [(x, y + delta, z) for delta in (-depth / 2, depth / 2) for x, z in profile]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    faces.extend((i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n))
    obj = mesh(name, vertices, faces, material, edge)
    # Blender's bevel miters can protrude slightly beyond an acute profile tip.
    # Preserve the specified stock envelope while retaining the rounded edges.
    lower = np.min(vertices, axis=0) / 1000
    upper = np.max(vertices, axis=0) / 1000
    for vertex in obj.data.vertices:
        vertex.co = np.clip(vertex.co, lower, upper)
    return obj


def part(name, visual, collisions, mass, friction=.35, material=METAL, size=1024,
         texture=None, **extra):
    visual.name = name
    finish(visual)
    visual["rigid_part"] = name
    props.uv_unwrap(visual)
    for loop in visual.data.uv_layers["UVMap"].data:
        loop.uv = (.0001 + .9998 * loop.uv.x, .0001 + .9998 * loop.uv.y)
    for index, obj in enumerate(collisions):
        obj.name = f"COL__{name}__{index:03d}"
    hide(collisions)
    return {"name": name, "source_visual_object": visual.name,
            "source_collision_objects": [p.name for p in collisions],
            "mass_kg": mass, "friction": friction, "mujoco_material": dict(material),
            "texture_px": size, "parent": None, "joint": None,
            "bake_texture": texture or f"{name}_albedo.png", **extra}


def complete(folder, parts, dimensions, notes, extra=None):
    visuals = [bpy.data.objects[p["source_visual_object"]] for p in parts]
    tier = "medium" if folder.name == "din_rail_kit" else "small"
    data = {"id": folder.name, "task": "precision_insertion", "units": "m",
            "origin": "bottom_center", "bbox_mm": [round(v, 5) for v in props.bbox_mm(visuals)],
            "triangles": props.triangle_count(visuals), "budget_class": tier,
            "source": "source.blend", "preview": "preview.png", "parts": parts,
            "variants": [], "dimensions_mm": dimensions, "notes": notes + " " + COMMON,
            "build_blender_version": bpy.app.version_string, **(extra or {})}
    assert data["triangles"] <= (50000 if tier == "medium" else 10000), data["triangles"]
    bpy.context.scene["task10_manifest"] = json.dumps(data)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in visuals
            obj.hide_set(obj not in visuals)
    props._select(visuals)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(QA / f"{folder.name}_geometry.blend"), check_existing=False)
    (QA / f"{folder.name}_preflight.json").write_text(json.dumps(data, indent=2) + "\n")
    log("GEOMETRY", folder.name, data["bbox_mm"], data["triangles"])
    if not ARGS.geometry_only:
        export_current()


def bearing_housing(clearance_mm):
    """Parametric 70 X x 110 Y x 40 housing; clearance is diametral, in mm."""
    if clearance_mm <= 0 or clearance_mm > 1:
        raise ValueError("diametral clearance must be between 0 and 1 mm")
    prop_id = f"bearing_housing_c{round(clearance_mm * 100):03d}"
    folder = start(prop_id)
    radius = (47 + clearance_mm) / 2
    entry = math.tan(math.radians(30))  # fixed axial 1 mm, 30 degrees from the bore axis
    aluminium = props.pbr_material("Milled aluminium C4C9CE", "#C4C9CE", .30, 1,
                                   wear=.06, grime=.12, machining="milled")
    visual = box("Housing block", (70, 110, 40), (0, 0, 20), aluminium)
    bore = solid_lathe("Stepped precision bore cutter",
                       [(20, -1), (20, 26), (radius, 26), (radius, 39),
                        (radius + entry, 40), (radius + entry, 41)], 192, apothem=True)
    boolean(visual, bore)
    for y in (-42, 42):
        cutter = solid_lathe("Counterbored mounting hole",
                             [(4.5, -1), (4.5, 31), (7.5, 31), (7.5, 39.7),
                              (7.8, 40), (7.8, 41)], 64, apothem=True)
        cutter.data.transform(Matrix.Translation((0, props.mm(y), 0)))
        boolean(visual, cutter)
    lower = props.ring_segments("Lower bore and seat", 20, 31, 26, count=48, location_mm=(0, 0, 13))
    upper = props.ring_segments("Upper bore wall", radius, 31, 13, count=48,
                                location_mm=(0, 0, 32.5))
    chamfer = props.ring_segments("30 degree entry", radius, 31, 1, count=48,
                                  location_mm=(0, 0, 39.5), top_inner_radius_mm=radius + entry)
    outer = [box("Y peripheral fill", (70, 24, 40), (0, y, 20), edge=0) for y in (-43, 43)]
    outer += [box("X peripheral fill", (4, 62, 40), (x, 0, 20), edge=0) for x in (-33, 33)]
    spec = part("body", visual, upper + chamfer + lower + outer, 0.62)
    spec["collision_bands"] = {"upper": [0, 48], "chamfer": [48, 96], "lower": [96, 144],
                                "periphery": [144, 148]}
    complete(folder, [spec], {"block_xyz": [70, 110, 40], "upper_bore": 47 + clearance_mm,
                             "upper_depth": 14, "lower_bore": 40, "seat_z": 26,
                             "seat_radial_width": radius - 20, "entry_axial_depth": 1,
                             "entry_angle_from_axis_deg": 30, "mount_hole": 9,
                             "counterbore": 15, "counterbore_depth": 9,
                             "mount_centres_y": [-42, 42]},
             "Static table fixture. Diametral clearance c=" + str(clearance_mm) + " mm. "
             "The 1 x 30 degree entry is 1 mm axially (Z39-40), radial expansion tan(30) mm. "
             "The nominal 3.5 mm seat grows by c/2 to preserve both exact bore diameters. "
             "Required collision layout is 48 upper + 48 tapered entry + 48 lower sectors, "
             "outer circumradius 31 mm, plus exactly four peripheral boxes (148 total). "
             "As mandated by that layout, mounting holes are visual only and the diagonal "
             "spaces between the R31 ring and the four boxes are unfilled; central insertion "
             "surfaces and seat remain exact. No hull closes the bore.",
             {"precision_mm": .01, "bore_diameter_mm": 47 + clearance_mm,
              "lower_bore_diameter_mm": 40, "clearance_mm": clearance_mm,
              "fixed": True, "collision_outer_radius_mm": 31,
              "collision_apothem_rule": True})


def bearing_6204():
    folder = start("bearing_6204")
    steel = props.pbr_material("Bearing steel", "#A0A6AC", .20, 1, grime=.10, machining="turned")
    seal = props.pbr_material("2RS rubber", "#111111", .90, 0, grime=.12)
    etched = props.pbr_material("NVK laser etched outer race", "#A0A6AC", .20, 1,
                                grime=.10, machining="turned")
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(__file__), "--artwork-worker",
                    str(folder / "nvk_marking.png")], check=True)
    tree = etched.node_tree
    image = tree.nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(folder / "nvk_marking.png"))
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "MarkUV"
    tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
    bsdf = tree.nodes["Principled BSDF"]
    original = bsdf.inputs["Base Color"].links[0].from_socket
    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1
    tree.links.new(original, mix.inputs[1])
    tree.links.new(image.outputs["Color"], mix.inputs[2])
    tree.links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    # One watertight cross-section: steel races plus recessed, grooved 2RS faces.
    profile = [(10.6, 0), (13.8, 0), (14.3, .5), (14.8, .5), (14.9, .56),
               (15.0, .5), (19.6, .5), (19.7, .56), (19.8, .5), (20.3, .5),
               (20.8, 0), (22.9, 0), (23.5, .6), (23.5, 13.4), (22.9, 14),
               (20.8, 14), (20.3, 13.5), (19.8, 13.5), (19.7, 13.44),
               (19.6, 13.5), (15.0, 13.5), (14.9, 13.44), (14.8, 13.5),
               (14.3, 13.5), (13.8, 14), (10.6, 14), (10, 13.4), (10, .6)]
    visual = ring_lathe("body", profile, 128, [steel, seal, etched],
                        seal_edges=tuple(range(2, 9)) + tuple(range(16, 23)), marking_edge=14)
    marking = visual.data.uv_layers.new(name="MarkUV")
    for loop in visual.data.loops:
        v = visual.data.vertices[loop.vertex_index].co * 1000
        marking.data[loop.index].uv = (.5 + v.x / 47, .5 + v.y / 47)
    collision = solid_lathe("One analytic 48 sided chamfered prism",
                            [(22.9, 0), (23.5, .6), (23.5, 13.4), (22.9, 14)], 48)
    spec = part("body", visual, [collision], .106, friction=.25)
    complete(folder, [spec], {"outer_diameter": 47, "inner_diameter": 20, "width": 14,
                             "chamfer": .6, "seal_recess": .5},
             "NVK 6204-2RS, two black recessed rubber seals; orange E25B12 seal variant. "
             "One analytic convex 48-sided prism with vertices on the nominal OD47 circle, "
             "0.6 mm bevel rings at both ends; collision deliberately fills the ID20 visual bore. "
             "A 48-sided prism requires at least 96 vertices (192 with both bevel rings), "
             "which cannot meet the universal 64-vertex/file rule. prop_lib therefore divides "
             "that same convex solid into exact convex subpieces without changing its union "
             "or enlarging its diameter. This is the closest faithful validator-compatible choice.",
             {"precision_mm": .01, "outer_diameter_mm": 47, "inner_diameter_mm": 20,
              "width_mm": 14, "collision_sides": 48, "collision_nominal_prisms": 1,
              "seal_material": seal.name, "marking": "NVK 6204-2RS"})


def dowel_pin():
    folder = start("dowel_pin")
    steel = props.pbr_material("Hardened dowel steel", "#8F969D", .25, 1,
                               grime=.10, machining="turned")
    lead = .5 * math.tan(math.radians(15))
    # R1 is the rolling radius around the upper rim; a full spherical R1 cap
    # cannot span a radius-5 shank. A short 15-degree lead blends into that rim.
    rings = [(5 - lead, 0), (5, .5), (5, 38.5), (5 - lead, 39)]
    fillet_radius = 1
    fillet_centre_r = 5 - lead - fillet_radius
    rings.extend((fillet_centre_r + math.cos(math.radians(a)), 39 + math.sin(math.radians(a)))
                 for a in (15, 30, 45, 60, 75, 90))
    visual = solid_lathe("body", rings, 96, steel)
    collision = props.cylinder("Inscribed 32 sided dowel prism", 5, 40, (0, 0, 20),
                                bevel_mm=0, vertices=32)
    complete(folder, [part("body", visual, [collision], .025, friction=.30)],
             {"diameter": 10, "length": 40, "lead_axial": .5, "lead_angle_from_axis_deg": 15,
              "ball_end_rolling_radius": 1},
             "OD10 x 40 hardened pin; one 32-sided collision prism, circumradius exactly 5 mm. "
             "The requested R1 spherical end cannot span an OD10 shank: a spherical cap covering "
             "the full face needs radius >=5 mm. Closest faithful choice: R1 rounded ball-nose "
             "rim with a small central flat, following the 0.5 mm axial x 15 degree lead. "
             "The opposite end has the specified lead chamfer. Collision ignores end dressing "
             "as prescribed and keeps the nominal full-length prism.",
             {"precision_mm": .01, "diameter_mm": 10, "length_mm": 40, "collision_sides": 32})


def pin_fixture():
    folder = start("pin_fixture")
    steel = props.pbr_material("Blackened fixture steel", "#2E3135", .45, .8,
                               grime=.16, machining="milled")
    specs = []
    for clearance in (.05, .10, .20):
        tag = f"c{round(clearance * 100):03d}"
        radius = (10 + clearance) / 2
        visual = box(f"Plate {tag}", (80, 120, 20), (0, 0, 10), steel)
        collisions = []
        for y in (-30, 30):
            cutter = solid_lathe("Chamfered through hole", [(radius + .5, -1),
                                 (radius + .5, 0), (radius, .5), (radius, 19.5),
                                 (radius + .5, 20), (radius + .5, 21)], 128, apothem=True)
            cutter.data.transform(Matrix.Translation((0, props.mm(y), 0)))
            boolean(visual, cutter)
            outline = regular_polygon(radius / math.cos(math.pi / 32), 32, center=(0, y))
            bounds = (-40, 40, -60, 0) if y < 0 else (-40, 40, 0, 60)
            polygons = wall_polygons(outline, bounds, (0, y), merge=False)
            assert len(polygons) == 32
            collisions.extend(props.extruded_profile("Exact hole sector", p, 20, (0, 0, 10),
                                                      bevel_mm=0) for p in polygons)
        name = "body" if tag == "c005" else f"body_{tag}"
        spec = part(name, visual, collisions, 1.45, friction=.30, material=BLACK_STEEL,
                    clearance_mm=clearance, bore_diameter_mm=10 + clearance)
        specs.append(spec)
    # Alternative geometry remains in the source, hidden and at the same prop origin.
    complete(folder, [specs[0]], {"plate_xyz": [80, 120, 20], "hole_pitch": 60,
                                "hole_centres_xy": [[0, -30], [0, 30]], "entry_chamfer": .5},
             "Fixed blackened plate, two through holes on a 60 mm Y pitch. Default c=0.05 mm; "
             "c=0.10 and c=0.20 are replacement-geometry variants in the variants[].parts "
             "records, all in the identical frame. Load one complete variant, never all three. "
             "Each hole has 32 convex sectors with exact design apothems. To meet the 64-piece "
             "limit, sectors extend to the rectangular half-plate perimeter, integrating the "
             "peripheral box fill without extra pieces. Entry chamfers are visual; collision "
             "retains the exact cylindrical through-hole dimensions over the full height.",
             {"precision_mm": .01, "bore_diameter_mm": 10.05, "clearance_mm": .05,
              "hole_pitch_mm": 60, "fixed": True, "variant_source_parts": specs,
              "collision_apothem_rule": True})


def capsule_profile(width, length, centre_x=0, centre_y=0, count=32):
    r = width / 2
    straight = (length - width) / 2
    return [(centre_x + r * math.cos(a), centre_y + offset + r * math.sin(a))
            for offset, start_angle in ((straight, 0), (-straight, math.pi))
            for a in (start_angle + math.pi * i / (count // 2) for i in range(count // 2 + 1))]


def terminal_block(materials):
    grey, clip_mat, yellow, brass, steel, white = materials
    profile = [(-23.5, 14), (-23.5, 38), (-20, 43), (-14, 43), (-12, 48.5),
               (12, 48.5), (14, 43), (20, 43), (23.5, 38), (23.5, 14),
               (17, 14), (15, 17), (-15, 17), (-17, 14)]
    body = extrude_xz("Terminal moulding", profile, 5.2, 0, grey, .23)
    for x in (-16.5, 16.5):
        cutter = props.cylinder("Screw well", 2.2, 9, (x, 0, 42), vertices=24, bevel_mm=0)
        boolean(body, cutter)
    for x in (-23, 23):
        cutter = props.cylinder("Wire entry tunnel", 1.85, 10, (x, 0, 32),
                                vertices=24, bevel_mm=0, rotation_deg=(0, 90, 0))
        boolean(body, cutter)
    items = [body]
    foot = [(-18.5, 6), (-15, 6), (-15, 6.4), (-17.7, 6.4), (-17.7, 7.7),
            (17.7, 7.7), (17.7, 10), (16, 14.5), (-16, 14.5), (-18.5, 10)]
    items.append(extrude_xz("Fixed clip foot with under-flange hook", foot, 4.6, 0, clip_mat, .10))
    latch = [(15, 6), (18.5, 6), (18.5, 10), (21.5, 11.5), (21.5, 12.8),
             (17.7, 11.5), (17.7, 6.4), (15, 6.4)]
    items.append(extrude_xz("Sliding clip latch geometry", latch, 4.2, 0, yellow, .10))
    for x in (-16.5, 16.5):
        screw = props.cylinder("Recessed slotted terminal screw", 1.75, 1.5, (x, 0, 41.5),
                               vertices=24, bevel_mm=.12, material=steel)
        boolean(screw, box("Screwdriver slot", (3.9, .55, .7), (x, 0, 42.1), edge=0))
        items.append(screw)
        items.append(box("Brass conductor clamp", (5, 3.8, 3.2), (x, 0, 34), brass, .12))
    items.append(box("Marker strip", (10, 3.6, .28), (0, 0, 48.2), white, .08))
    return merge(items, "terminal_01")


def din_rail_kit():
    folder = start("din_rail_kit")
    zinc = props.pbr_material("Passivated zinc rail", "#C9C08A", .30, 1,
                              wear=.08, grime=.12, machining="brushed")
    grey = props.pbr_material("IMC Controls grey nylon", "#9E9E9E", .55, 0, wear=.10, grime=.15)
    clip_mat = props.pbr_material("Grey clip foot", "#73797D", .55, 0, wear=.10, grime=.15)
    yellow = props.pbr_material("Latch yellow", "#D8A923", .55, 0, wear=.08, grime=.12)
    brass = props.pbr_material("Brass electrical clamp", "#B08D57", .35, 1, machining="brushed")
    steel = props.pbr_material("Bright terminal screws", "#BFC3C7", .30, 1, machining="turned")
    white = props.pbr_material("Off white marker", "#E9E8DC", .65, 0, grime=.12)
    rail_profile = [(-17.5, 6.5), (-12.5, 6.5), (-12.5, 0), (12.5, 0), (12.5, 6.5),
                    (17.5, 6.5), (17.5, 7.5), (11.5, 7.5), (11.5, 1),
                    (-11.5, 1), (-11.5, 7.5), (-17.5, 7.5)]
    rail = extrude_xz("rail", rail_profile, 250, 0, zinc, .10)
    for index, y in enumerate(range(-100, 101, 25)):
        # Aligned slots of length equal to pitch touch. This 1 mm lateral stagger
        # preserves every requested slot dimension and pitch, with positive webs.
        profile = capsule_profile(5.3, 25, centre_x=(-1 if index % 2 else 1), centre_y=y)
        cutter = props.extruded_profile("25 x 5.3 through slot", profile, 4, (0, 0, .5), bevel_mm=0)
        boolean(rail, cutter)
    specs = [part("rail", rail, [], .09, size=2048)]
    terminal = terminal_block((grey, clip_mat, yellow, brass, steel, white))
    base_spec = part("terminal_01", terminal, [], .012, friction=.40, material=PLASTIC,
                     texture="terminal_block_albedo.png")
    terminal.name = "TERMINAL_TEMPLATE"
    # Copy UVs as well as geometry: the eight blocks use one identical baked atlas.
    for index in range(8):
        y = (index - 3.5) * 5.35
        obj = terminal if index == 0 else terminal.copy()
        if index:
            obj.data = terminal.data.copy()
            bpy.context.collection.objects.link(obj)
        # The prototype first moves only after all copies have their own data.
        if index == 0:
            obj = terminal.copy()
            obj.data = terminal.data.copy()
            bpy.context.collection.objects.link(obj)
        obj.data.transform(Matrix.Translation((0, props.mm(y), 0)))
        obj.name = f"terminal_{index + 1:02d}"
        spec = dict(base_spec, name=obj.name, source_visual_object=obj.name)
        specs.append(spec)
    bpy.data.objects.remove(terminal, do_unlink=True)
    cover_profile = [(-23.5, 14), (-23.5, 38), (-20, 43), (-14, 43), (-12, 48.5),
                     (12, 48.5), (14, 43), (20, 43), (23.5, 38), (23.5, 14)]
    cover = extrude_xz("end_cover", cover_profile, 1.5, 22.7, grey, .15)
    specs.append(part("end_cover", cover, [], .003, friction=.40, material=PLASTIC))
    clamp_profile = [(-20, 7.7), (-20, 23), (-14, 28), (14, 28), (20, 23), (20, 7.7),
                     (17.7, 7.7), (17.7, 6), (15, 6), (15, 6.4), (18.5, 6.4),
                     (18.5, 10), (-18.5, 10), (-18.5, 6.4), (-15, 6.4), (-15, 6),
                     (-17.7, 6), (-17.7, 7.7)]
    clamp = extrude_xz("End clamp housing", clamp_profile, 9.5, -28, clip_mat, .20)
    screw = props.cylinder("End clamp adjustment screw", 3.1, 2, (0, -28, 28),
                           vertices=32, bevel_mm=.18, material=steel)
    boolean(screw, box("Clamp screw slot", (6.5, .8, 1), (0, -28, 29), edge=0))
    specs.append(part("end_clamp", merge([clamp, screw], "end_clamp"), [], .016,
                      friction=.40, material=PLASTIC))
    complete(folder, specs, {"rail_xyz": [35, 250, 7.5], "rail_thickness": 1,
                             "slot_length_width": [25, 5.3], "slot_pitch": 25,
                             "slot_lateral_stagger": 1, "terminal_xyz": [47, 5.2, 42.5],
                             "terminal_count": 8},
             "Geometry and textures only: no collision proxies, joints, springs or clip-on "
             "behaviour are modelled. TS35 x 7.5 x 250 rail, 1 mm stock, true through slots; "
             "eight generic IMC Controls screw terminals with wire tunnels, brass clamps, "
             "recessed slotted screws, fixed under-flange hooks and sliding latch feet; "
             "one end cover and one screw end clamp. Slots are exactly 25 x 5.3 at 25 mm "
             "longitudinal pitch, alternately offset +/-1 mm laterally. Equal-length, "
             "collinear slots at that pitch would touch and create nonmanifold web edges; "
             "the stagger is the closest faithful way to retain both dimensions and pitch. "
             "The terminals share a single baked atlas and retain distinct meshes.",
             {"geometry_only": True, "clip_on_behaviour_modelled": False})


def precise_collision(name, objects, folder, shaft=None):
    """Reuse prop_lib's exact hull subdivision; avoid six-decimal OBJ rounding."""
    files = []
    for obj in objects:
        world = props._world_mesh(obj)
        try:
            hulls = props._split_hulls(props._hull([v.co for v in world.vertices]))
        finally:
            bpy.data.meshes.remove(world)
        for bm in hulls:
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
            bm.verts.ensure_lookup_table()
            bm.verts.index_update()
            points = np.asarray([tuple(v.co) for v in bm.verts], dtype=float)
            if shaft:
                # Eliminate float32 overshoot so an exported shaft can never
                # exceed its nominal design circle, even at sub-nanometre scale.
                for point in points:
                    r = math.hypot(*point[:2])
                    limit = shaft[0]
                    if len(shaft) > 1:
                        z = point[2] * 1000
                        limit -= max(0, .6 - z, z - 13.4)
                    limit = props.mm(limit) - 2e-12
                    if r > limit:
                        point[:2] *= limit / r
            faces = [[v.index for v in f.verts] for f in bm.faces]
            bm.free()
            filename = f"{name}_col_{len(files):02d}.obj"
            lines = ["# Convex prop_lib hull, metres, precision coordinates"]
            lines.extend("v " + " ".join(f"{v:.12f}" for v in p) for p in points)
            lines.extend("f " + " ".join(str(v + 1) for v in f) for f in faces)
            (folder / filename).write_text("\n".join(lines) + "\n")
            files.append(filename)
    return files


def bake(obj, size, path, cache):
    if path.name in cache or (ARGS.reuse_bakes and path.exists()):
        cache.add(path.name)
        return
    log("BAKE", path.name, size)
    for other in bpy.context.scene.objects:
        if other.type == "MESH":
            other.hide_render = other != obj
    obj.hide_set(False)
    obj.hide_render = False
    props.bake_albedo([obj], size, path)
    cache.add(path.name)


def export_current():
    data = json.loads(bpy.context.scene["task10_manifest"])
    folder = OUTPUT / data["id"]
    cache = set()

    def export(spec):
        obj = bpy.data.objects[spec["source_visual_object"]]
        spec["visual"] = props.export_obj(obj, folder / f"{spec['name']}.obj")
        texture = spec["bake_texture"]
        bake(obj, spec["texture_px"], folder / texture, cache)
        spec["texture"] = texture
        shaft = (23.5, .6) if data["id"] == "bearing_6204" else (5,) if data["id"] == "dowel_pin" else None
        spec["collision"] = precise_collision(spec["name"],
            [bpy.data.objects[n] for n in spec["source_collision_objects"]], folder, shaft)

    for spec in data["parts"]:
        export(spec)
    if data["id"] == "bearing_6204":
        seal = bpy.data.materials[data["seal_material"]]
        rgb_node = next(n for n in seal.node_tree.nodes if n.type == "RGB")
        original = tuple(rgb_node.outputs[0].default_value)
        rgb_node.outputs[0].default_value = props._linear_color("#E25B12")
        bake(bpy.data.objects[data["parts"][0]["source_visual_object"]], 1024,
             folder / "body_albedo_v1.png", cache)
        rgb_node.outputs[0].default_value = original
        data["variants"] = [{"name": "black_2rs", "textures": {"body": "body_albedo.png"}},
                            {"name": "orange_2rs", "textures": {"body": "body_albedo_v1.png"}}]
    if data["id"] == "pin_fixture":
        data["variants"] = []
        for spec in data.pop("variant_source_parts"):
            if spec["clearance_mm"] == .05:
                spec = dict(data["parts"][0])
            else:
                export(spec)
            spec["name"] = "body"
            obj = bpy.data.objects[spec["source_visual_object"]]
            data["variants"].append({"name": f"c{round(spec['clearance_mm'] * 100):03d}",
                "clearance_mm": spec["clearance_mm"], "bore_diameter_mm": spec["bore_diameter_mm"],
                "triangles": props.triangle_count([obj]), "parts": [spec],
                "textures": {"body": spec["texture"]}})
    visuals = [bpy.data.objects[p["source_visual_object"]] for p in data["parts"]]
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in visuals
            obj.hide_set(obj not in visuals)
    props.render_preview(visuals, folder / "preview.png")
    props.write_prop_json(data, folder / "prop.json")
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(HERE / "check_task10_precision_insertion.py"),
                    "--prepare-collisions", "--props", data["id"]], check=True)
    bpy.context.scene["task10_delivered"] = json.dumps(data)
    for image in bpy.data.images:
        if image.source == "FILE":
            image.use_fake_user = True
    props._select(visuals)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    log("COMPLETE", data["id"], data["triangles"], sum(len(p["collision"]) for p in data["parts"]))


def render_exports(prop_id):
    """Inspect the delivered meshes and baked atlases, including geometry variants."""
    folder = OUTPUT / prop_id
    data = json.loads((folder / "prop.json").read_text())
    for variant in data["variants"] or [{"name": "default", "textures": {}}]:
        props.reset_scene()
        visuals = []
        for spec in variant.get("parts", data["parts"]):
            bpy.ops.wm.obj_import(filepath=str(folder / spec["visual"]), forward_axis="Y", up_axis="Z")
            obj = bpy.context.selected_objects[0]
            material = props.pbr_material(spec["name"] + " delivered atlas", "#FFFFFF", .45, 0)
            tree, bsdf = material.node_tree, material.node_tree.nodes["Principled BSDF"]
            texture = variant.get("textures", {}).get(spec["name"], spec["texture"])
            node = tree.nodes.new("ShaderNodeTexImage")
            node.image = bpy.data.images.load(str(folder / texture), check_existing=True)
            tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
            bsdf.inputs["Metallic"].default_value = .35 if spec["mujoco_material"]["specular"] >= .7 else 0
            obj.data.materials.clear()
            obj.data.materials.append(material)
            visuals.append(obj)
        target = QA / "delivered_previews" / prop_id
        target.mkdir(parents=True, exist_ok=True)
        props.render_preview(visuals, target / f"{variant['name']}.png")
        if prop_id == "din_rail_kit":
            props.render_preview(visuals[1:], target / "terminal_detail.png")
        log("RENDER_EXPORTS", prop_id, variant["name"])


def audit_sources(prop_ids):
    results = []
    for prop_id in prop_ids:
        folder = OUTPUT / prop_id
        data = json.loads((folder / "prop.json").read_text())
        bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
        specs = data["parts"] + [p for v in data["variants"] for p in v.get("parts", [])]
        for spec in specs:
            obj = bpy.data.objects[spec["source_visual_object"]]
            assert np.allclose(obj.matrix_world, np.eye(4), rtol=0, atol=1e-9), obj.name
            assert obj.data.uv_layers.get("UVMap"), obj.name
            assert all(len(p.vertices) == 3 for p in obj.data.polygons), obj.name
            for material in obj.data.materials:
                assert material.node_tree and any(n.type == "BSDF_PRINCIPLED"
                                                  for n in material.node_tree.nodes), material.name
            assert props.triangle_count([obj]) == len(obj.data.polygons), obj.name
        assert all(i.packed_file for i in bpy.data.images if i.source == "FILE"), prop_id
        scene = bpy.context.scene
        assert scene.cycles.samples == 64 and scene.cycles.device == "CPU"
        assert scene.unit_settings.system == "METRIC" and scene.unit_settings.scale_length == 1
        results.append({"prop": prop_id, "status": "PASS", "parts_and_variants_checked": len(specs),
                        "transforms_applied": True, "pbr_retained": True, "images_packed": True})
    QA.mkdir(parents=True, exist_ok=True)
    (QA / "source_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    log("SOURCE_CHECKS", len(results), "PASS")


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--reuse-bakes", action="store_true")
    parser.add_argument("--render-exports", action="store_true")
    parser.add_argument("--audit-sources", action="store_true")
    ARGS = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if ARGS.geometry_only and ARGS.export_only:
        parser.error("--geometry-only and --export-only are mutually exclusive")
    if ARGS.audit_sources:
        audit_sources(ARGS.props)
        return
    for prop_id in ARGS.props:
        if ARGS.render_exports:
            render_exports(prop_id)
        elif ARGS.export_only:
            bpy.ops.wm.open_mainfile(filepath=str(QA / f"{prop_id}_geometry.blend"))
            export_current()
        elif prop_id.startswith("bearing_housing_"):
            bearing_housing(int(prop_id.rsplit("c", 1)[1]) / 100)
        else:
            globals()[prop_id]()


if __name__ == "__main__":
    main()
    props.finalize_assets()
