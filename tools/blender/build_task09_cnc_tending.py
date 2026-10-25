"""Build only benchmark_props.md section 3.9, using the common prop pipeline.

Blender -b --factory-startup --python-exit-code 1 -P this_file -- [--props ID ...]
--geometry-only saves resumable sources; --export-only exports those sources.
--render-exports inspects the delivered OBJ/albedo assets rather than source PBR.
Panel devices are transformed instances of the existing Task 2 library, never rebuilt.
Geometry arguments are millimetres; every delivered OBJ uses the common metre frame.
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
OUTPUT = HERE.parents[1] / "assets" / "props"
QA = HERE / "_qa" / "task09"
IDS = ("cnc_front", "part_tray_6", "round_blank", "machined_part")


def artwork_worker():
    """Original fictional screen, legends and generic safety art; no fetched art."""
    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo

    folder = Path(json.load(sys.stdin)["folder"])

    def font(size):
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)

    def save(image, filename):
        info = PngInfo()
        info.add(b"sRGB", b"\x00")
        image.save(folder / filename, pnginfo=info)

    if folder.name == "cnc_front":
        image = Image.new("RGB", (1200, 900), "#102331")
        d = ImageDraw.Draw(image)
        d.rectangle((0, 0, 1200, 142), fill="#1F5FA8")
        d.text((46, 43), "IMC-CNC 850", font=font(78), fill="white")
        d.text((46, 176), "AUTO   /   PROGRAM O0085", font=font(43), fill="#CAD7DD")
        for y, text in ((282, "X   +100.000"), (378, "Y      +0.000"), (474, "Z   +1150.000")):
            d.text((48, y), text, font=font(68), fill="#E9F1F2")
        d.rounded_rectangle((820, 285, 1134, 605), 10, outline="#7293A0", width=5)
        d.ellipse((858, 368, 1096, 567), outline="#66B9D1", width=9)
        d.line((980, 325, 980, 586), fill="#426E82", width=4)
        d.line((839, 465, 1117, 465), fill="#426E82", width=4)
        d.rectangle((36, 645, 1164, 749), fill="#244E44")
        d.text((63, 669), "READY  /  DOOR INTERLOCK", font=font(48), fill="#94E2B4")
        d.text((47, 803), "F  240 mm/min     S  1200 rpm", font=font(40), fill="#A4BCC8")
        save(image, "screen_ui.png")

        image = Image.new("RGB", (1500, 3500), "#343A40")
        d = ImageDraw.Draw(image)

        def panel_text(y_mm, z_mm, text, size, color="white"):
            # From -X, screen-right is -Y. Image top is +Z.
            d.text(((-290 - y_mm) * 10, (1250 - z_mm) * 10), text,
                   anchor="mm", font=font(size), fill=color)

        panel_text(-365, 1233, "IMC / MACHINE CONTROL", 66, "#CDD9DF")
        panel_text(-405, 976, "CYCLE START", 59)
        panel_text(-325, 976, "FEED HOLD", 59)
        panel_text(-365, 916, "MAN     MODE     AUTO", 64, "#D5D9DD")
        d.line((80, 2860, 1420, 2860), fill="#64717A", width=5)
        save(image, "panel_print.png")

        image = Image.new("RGB", (1200, 360), "#1F5FA8")
        d = ImageDraw.Draw(image)
        d.text((54, 24), "IMC", font=font(270), fill="#F1F5F8")
        d.text((640, 96), "COMPACT", font=font(59), fill="#C9E0F4")
        d.text((640, 184), "MACHINING", font=font(59), fill="#C9E0F4")
        save(image, "fictional_imc.png")

        image = Image.new("RGB", (900, 500), "#E8BE33")
        d = ImageDraw.Draw(image)
        d.polygon([(170, 52), (42, 290), (298, 290)], fill="#1D2429")
        d.polygon([(170, 90), (78, 267), (263, 267)], fill="#F0CB36")
        d.rectangle((157, 143, 183, 215), fill="#1D2429")
        d.ellipse((157, 230, 183, 256), fill="#1D2429")
        d.text((340, 104), "CAUTION", font=font(90), fill="#1D2429")
        d.text((340, 236), "MOVING PARTS", font=font(52), fill="#1D2429")
        d.text((55, 366), "KEEP DOOR CLOSED DURING CYCLE", font=font(40), fill="#1D2429")
        save(image, "generic_warning.png")
    else:
        for label in ("IN", "OUT"):
            image = Image.new("RGB", (1700, 2500), "white")
            d = ImageDraw.Draw(image)
            # XY planar label map: image right is +X, image up is +Y.
            # Top-rim text is on the unobstructed -X strip, away from all holes.
            stamp = Image.new("RGB", (720, 220), "white")
            sd = ImageDraw.Draw(stamp)
            sd.text((360, 110), label, anchor="mm", font=font(190), fill="#353E44")
            # From -X, +Y is screen-left: the text top points into +X,
            # and its reading direction is -Y.
            stamp = stamp.rotate(-90, expand=True)
            image.paste(stamp, (38, 890))
            save(image, f"tray_{label.lower()}_etch.png")


if __name__ == "__main__" and "--artwork-worker" in sys.argv:
    artwork_worker()
    raise SystemExit(0)

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

import prop_lib as props
from build_task06_shaped_kitting import boolean, finish, regular_polygon, wall_polygons
from build_task02_panel_operation import revolve
from build_task05_carton_carry import grain, multiply_color

PAINT = {"specular": .30, "shininess": .30, "reflectance": 0.0}
METAL = {"specular": .70, "shininess": .70, "reflectance": .05}
STEEL = {"specular": .50, "shininess": .50, "reflectance": .02}
CLEAR = {"specular": .45, "shininess": .75, "reflectance": .03}
COMMON = (
    "Metres, Z up, front -X, applied transforms and a common prop frame. "
    "Triangulated OBJ, 30-degree smoothing and weighted normals. Smart UV 66 degrees, "
    "nonoverlapping islands, 0.01 margins. CPU Cycles 64-sample DIFFUSE COLOR * "
    "(0.4 + 0.6 * AO), RGB8 sRGB, excluding direct illumination and highlights. "
    "Full Principled PBR and original printing images are packed in source.blend. "
    "Preview: CPU Cycles 64, 512 square, neutral grey, three area lights, front-left above. "
    "Per-instance colour multipliers of 0.97-1.03 may be applied by simulation. "
)


def log(*args):
    print("TASK09", *args, flush=True)


def hide(objects):
    for obj in objects:
        obj.hide_render = True
        obj.hide_set(True)


def start(prop_id):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    folder = OUTPUT / prop_id
    folder.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    log("START", prop_id, bpy.app.version_string)
    return folder


def box(name, size, center, material=None, edge=1, segments=3):
    return props.box(name, size, center, bevel_mm=edge, segments=segments, material=material)


def merge(objects, name):
    obj = props.join_objects(objects, name) if len(objects) > 1 else objects[0]
    obj.name = name
    return obj


def mapped_uv(obj, name, mapping):
    layer = obj.data.uv_layers.get(name) or obj.data.uv_layers.new(name=name)
    for loop in obj.data.loops:
        point = obj.data.vertices[loop.vertex_index].co * 1000
        layer.data[loop.index].uv = mapping(point)


def image_ink(material, image_path, uv_name="PrintUV"):
    tree = material.node_tree
    node = tree.nodes.new("ShaderNodeTexImage")
    node.name = "Artwork"
    node.image = bpy.data.images.load(str(image_path), check_existing=True)
    node.extension = "EXTEND"
    node.image.colorspace_settings.name = "sRGB"
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_name
    tree.links.new(uv.outputs["UV"], node.inputs["Vector"])
    multiply_color(material, node.outputs["Color"], "Original fictional artwork")
    return node


def printed_front(name, size_yz, center, path, base="#FFFFFF", metallic=0):
    material = props.pbr_material(name + " ink", base, .5, metallic, grime=.12)
    image_ink(material, path)
    obj = box(name, (.15, *size_yz), center, material, edge=.04, segments=2)
    y, z = center[1:]
    mapped_uv(obj, "PrintUV", lambda p: (.5 - (p.y - y) / size_yz[0],
                                        .5 + (p.z - z) / size_yz[1]))
    return obj


def art(folder):
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(__file__), "--artwork-worker"],
                   input=json.dumps({"folder": str(folder)}), text=True, check=True)


def record(name, visual, collisions, mass, material=METAL, friction=.35,
           parent=None, joint=None, size=2048, rgba=None, **extra):
    for i, obj in enumerate(collisions):
        obj.name = f"COL__{name}__{i:02d}"
    result = {"name": name, "source_visual_object": visual.name,
              "source_collision_objects": [obj.name for obj in collisions],
              "mass_kg": mass, "friction": friction, "mujoco_material": dict(material),
              "parent": parent, "joint": joint, "texture_px": size, **extra}
    if rgba is not None:
        result["rgba"], result["texture_px"] = rgba, None
    hide(collisions)
    return result


def complete(folder, parts, dimensions, notes, variants=None, extra=None):
    visuals = []
    for spec in parts:
        obj = bpy.data.objects[spec["source_visual_object"]]
        finish(obj)
        obj["rigid_part"] = spec["name"]
        if spec["texture_px"] and not spec.get("library_instance"):
            props.uv_unwrap(obj)
        if spec["texture_px"] and not spec.get("library_instance"):
            # Keep quantized atlas borders strictly inside [0,1].
            for loop in obj.data.uv_layers["UVMap"].data:
                loop.uv = (.0001 + .9998 * loop.uv.x, .0001 + .9998 * loop.uv.y)
        visuals.append(obj)
    data = {"id": folder.name, "task": "cnc_tending", "units": "m",
            "origin": "floor_front_face_center" if folder.name == "cnc_front" else "bottom_center",
            "bbox_mm": [round(v, 4) for v in props.bbox_mm(visuals)],
            "triangles": props.triangle_count(visuals),
            "budget_class": "large" if folder.name == "cnc_front" else
                            "medium" if folder.name == "part_tray_6" else "small",
            "source": "source.blend", "preview": "preview.png", "parts": parts,
            "variants": variants or [], "dimensions_mm": dimensions,
            "notes": notes + " " + COMMON, "build_blender_version": bpy.app.version_string,
            **(extra or {})}
    assert data["triangles"] <= {"small": 10000, "medium": 50000, "large": 150000}[data["budget_class"]]
    bpy.context.scene["task09_manifest"] = json.dumps(data)
    bpy.context.scene["bake_recipe"] = "CPU Cycles 64 DIFFUSE COLOR * (0.4 + 0.6 * AO), RGB8 sRGB"
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in visuals
            obj.hide_set(obj not in visuals)
    bpy.context.preferences.filepaths.save_version = 0
    props._select(visuals)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(QA / f"{folder.name}_geometry.blend"), check_existing=False)
    (QA / f"{folder.name}_preflight.json").write_text(json.dumps(data, indent=2) + "\n")
    log("GEOMETRY", folder.name, data["bbox_mm"], data["triangles"])
    if not ARGS.geometry_only:
        export_current()


def bake(obj, size, path):
    if ARGS.reuse_bakes and path.exists():
        log("REUSE_BAKE", path.name)
        return
    log("BAKE", path.name, size)
    for other in bpy.context.scene.objects:
        if other.type == "MESH":
            other.hide_render = other != obj
    obj.hide_set(False)
    obj.hide_render = False
    props.bake_albedo([obj], size, path)


def preview_extras(data, visuals, folder):
    if data["id"] == "cnc_front":
        moving = [bpy.data.objects[p["source_visual_object"]] for p in data["parts"]
                  if p["name"] in ("door", "door_window")]
        for obj in moving:
            obj.location.y = .46
        props.render_preview(visuals, folder / "preview_open.png")
        for obj in moving:
            obj.location.y = 0
        panel = [bpy.data.objects[p["source_visual_object"]] for p in data["parts"]
                 if p["name"] in ("panel", "screen") or p.get("library_instance")]
        props.render_preview(panel, folder / "preview_panel.png")
    if data["id"] == "part_tray_6":
        node = bpy.data.materials[data["variant_material"]].node_tree.nodes["Artwork"]
        original = node.image
        node.image = bpy.data.images.load(str(folder / "tray_out_etch.png"), check_existing=True)
        props.render_preview(visuals, folder / "preview_out.png")
        node.image = original


def export_current():
    data = json.loads(bpy.context.scene["task09_manifest"])
    folder = OUTPUT / data["id"]
    for spec in data["parts"]:
        name = spec["name"]
        obj = bpy.data.objects[spec["source_visual_object"]]
        if spec.get("library_instance"):
            texture = spec["library_texture"]
            shutil.copyfile(OUTPUT / "panel_devices" / texture, folder / texture)
        elif spec["texture_px"]:
            texture = f"{name}_albedo.png"
            bake(obj, spec["texture_px"], folder / texture)
        else:
            texture = None
        if data["id"] == "part_tray_6":
            node = bpy.data.materials[data["variant_material"]].node_tree.nodes["Artwork"]
            original = node.image
            node.image = bpy.data.images.load(str(folder / "tray_out_etch.png"), check_existing=True)
            bake(obj, spec["texture_px"], folder / "body_albedo_v1.png")
            node.image = original
        spec["visual"] = props.export_obj(obj, folder / f"{name}.obj")
        spec["texture"] = texture
        spec["collision"] = props.export_collision(
            name, [bpy.data.objects[n] for n in spec["source_collision_objects"]], folder)
        hide([bpy.data.objects[n] for n in spec["source_collision_objects"]])
    visuals = [bpy.data.objects[p["source_visual_object"]] for p in data["parts"]]
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in visuals
            obj.hide_set(obj not in visuals)
    props.render_preview(visuals, folder / "preview.png")
    preview_extras(data, visuals, folder)
    props.write_prop_json(data, folder / "prop.json")
    # A rigidly rotated source cylinder can acquire non-coplanar cap coordinates
    # in six-decimal OBJ. Re-hull each existing proxy's unchanged points only.
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(HERE / "check_task09_cnc_tending.py"),
                    "--repair-collisions", "--props", data["id"]], check=True)
    bpy.context.scene["task09_delivered"] = json.dumps(data)
    for image in bpy.data.images:
        if image.source == "FILE":
            image.use_fake_user = True
    props._select(visuals)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    log("COMPLETE", data["id"], data["triangles"], sum(len(p["collision"]) for p in data["parts"]))


def coolant_material(material):
    """Sparse dried-coolant streaks in the exported colour, plus source roughness."""
    tree = material.node_tree
    coord = tree.nodes.new("ShaderNodeNewGeometry")
    stretch = tree.nodes.new("ShaderNodeVectorMath")
    stretch.operation = "MULTIPLY"
    stretch.inputs[1].default_value = (1, 1.8, .12)
    tree.links.new(coord.outputs["Position"], stretch.inputs[0])
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 85
    noise.inputs["Detail"].default_value = 3
    tree.links.new(stretch.outputs[0], noise.inputs["Vector"])
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = .42
    ramp.color_ramp.elements[0].color = (.68, .77, .72, 1)
    ramp.color_ramp.elements[1].position = .57
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    tree.links.new(noise.outputs["Fac"], ramp.inputs[0])
    multiply_color(material, ramp.outputs[0], "Dried emulsion streaks")
    return material


def serrated_material():
    material = props.pbr_material("Diagonal serrated hardened jaw plates", "#BFC3C7", .38, 1,
                                  grime=.18)
    tree = material.node_tree
    geom = tree.nodes.new("ShaderNodeNewGeometry")
    xyz = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(geom.outputs["Position"], xyz.inputs[0])
    phase = props._math(tree, "ADD", xyz.outputs["Y"], xyz.outputs["Z"])
    sine = props._math(tree, "SINE", props._math(tree, "MULTIPLY", phase, 2 * math.pi / .0018))
    dark = props._math(tree, "ADD", .8, props._math(tree, "MULTIPLY", sine, .2))
    multiply_color(material, dark, "1.8 mm diagonal jaw serrations")
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = .00018
    bump.inputs["Strength"].default_value = .45
    tree.links.new(sine, bump.inputs["Height"])
    tree.links.new(bump.outputs[0], tree.nodes["Principled BSDF"].inputs["Normal"])
    return material


def library_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((OUTPUT / "panel_devices").iterdir()) if p.is_file()}


def install_panel(parts, folder):
    """Append prototypes and transform copies, retaining source UVs and bakes."""
    library = json.loads((OUTPUT / "panel_devices" / "prop.json").read_text())
    sources = {p["name"]: p for p in library["parts"]}
    needed = ("bezel", "btn_green", "btn_yellow", "estop_plate", "estop", "selector_knob")
    names = [*needed, *(n for key in needed for n in sources[key]["source_collision_objects"])]
    with bpy.data.libraries.load(str(OUTPUT / "panel_devices" / "source.blend"), link=False) as (src, dst):
        assert set(names) <= set(src.objects), "Task 2 source must include its original prototypes"
        dst.objects = names
    prototypes = {obj.name: obj for obj in dst.objects}
    for obj in prototypes.values():
        bpy.context.collection.objects.link(obj)
    hide(prototypes.values())
    # Columns: device +X -> -Y; +Y -> +Z; outward +Z -> -X.
    rotation = Matrix(((0, 0, -1), (-1, 0, 0), (0, 1, 0)))
    placements, exclusions = [], []

    def install(template, name, y, z, parent="panel"):
        source = sources[template]
        transform = rotation.to_4x4()
        transform.translation = Vector((-.040, y / 1000, z / 1000))
        obj = prototypes[template].copy()
        obj.data = obj.data.copy()
        obj.name = name
        bpy.context.collection.objects.link(obj)
        obj.data.transform(transform)
        obj.hide_render = False
        obj.hide_set(False)
        # Preserve device-local procedural coordinates after applying transforms.
        frame = bpy.data.objects.new(name + " material coordinate frame", None)
        frame.matrix_world = transform
        bpy.context.collection.objects.link(frame)
        for i, original in enumerate(obj.data.materials):
            material = original.copy()
            obj.data.materials[i] = material
            for node in material.node_tree.nodes:
                if node.type == "TEX_COORD":
                    node.object = frame
        collisions = []
        for source_name in source["source_collision_objects"]:
            col = prototypes[source_name].copy()
            col.data = col.data.copy()
            bpy.context.collection.objects.link(col)
            col.data.transform(transform)
            collisions.append(col)
        joint = json.loads(json.dumps(source["joint"]))
        if joint:
            joint["parent"] = parent
            joint["axis"] = list(rotation @ Vector(joint["axis"]))
            joint["anchor_m"] = [-.040, y / 1000, z / 1000]
            if "latch" in joint:
                joint["latch"]["release_axis"] = [-1, 0, 0]
            exclusions.extend([[name, "body"], [name, "panel"], [name, parent]])
        spec = record(name, obj, collisions, source["mass_kg"], source["mujoco_material"],
                      source["friction"], parent=parent, joint=joint, size=source["texture_px"],
                      library_instance={"prop_id": "panel_devices", "part": template},
                      library_texture=source["texture"],
                      mounting_position_m=[-.040, y / 1000, z / 1000],
                      local_to_prop_rotation=[list(row) for row in rotation])
        if template == "selector_knob":
            spec["recommended_initial_qpos"] = source["recommended_initial_qpos"]
        parts.append(spec)
        return name

    for name, template, y in (("cycle_start", "btn_green", -405),
                              ("feed_hold", "btn_yellow", -325)):
        parent = install("bezel", name + "_bezel", y, 1000)
        install(template, name, y, 1000, parent)
        placements.append({"name": name, "library_part": template,
                           "position_m": [-.040, y / 1000, 1.0], "press_direction": [1, 0, 0]})
    parent = install("estop_plate", "estop_plate", -365, 1180)
    install("estop", "estop", -365, 1180, parent)
    install("selector_knob", "mode_selector", -365, 940)
    placements += [{"name": "estop", "library_part": "estop", "position_m": [-.04, -.365, 1.18],
                    "press_direction": [1, 0, 0]},
                   {"name": "mode_selector", "library_part": "selector_knob",
                    "position_m": [-.04, -.365, .940], "press_direction": [1, 0, 0],
                    "rotation_axis": [-1, 0, 0]}]
    return placements, exclusions


def build_cnc():
    folder = start("cnc_front")
    art(folder)
    before = library_hashes()
    (QA / "panel_devices_sha256_before.json").write_text(json.dumps(before, indent=2) + "\n")
    light = props.pbr_material("Light grey 1.5 mm powder coated sheet #DADDE0", "#DADDE0", .6, 0,
                               wear=.045, grime=.15)
    dark = props.pbr_material("Dark grey trim #3B3F44", "#3B3F44", .6, 0, wear=.045, grime=.15)
    blue = props.pbr_material("IMC blue accent #1F5FA8", "#1F5FA8", .6, 0, wear=.04, grime=.15)
    metal = coolant_material(props.pbr_material("Milled aluminium table with coolant", "#C9CDD1", .3, 1,
                                               grime=.18, machining="milled"))
    cast = coolant_material(props.pbr_material("Blue grey cast iron vise #4A5A6A", "#4A5A6A", .55, .35,
                                              wear=.045, grime=.18))
    steel = props.pbr_material("Brushed machine steel", "#BFC3C7", .3, 1, grime=.15, machining="brushed")
    black = props.pbr_material("Dark rubber slide seals", "#22262A", .8, 0, grime=.15)
    jaw_mat = serrated_material()
    body, body_col = [], []
    # Exactly four front sheets leave the required 480 x 420 clear opening.
    front = [((1.5, 900, 680), (.75, 0, 460)),
             ((1.5, 900, 380), (.75, 0, 1410)),
             ((1.5, 210, 420), (.75, -345, 1010)),
             ((1.5, 210, 420), (.75, 345, 1010))]
    for i, (size, center) in enumerate(front):
        body.append(box(f"Folded front sheet {i}", size, center, light, edge=.3))
        body_col.append(box(f"Front opening wall {i}", size, center, edge=0))
    for name, size, center in (
        ("left side", (698.5, 1.5, 1480), (350.75, -449.25, 860)),
        ("right side", (698.5, 1.5, 1480), (350.75, 449.25, 860)),
        ("back", (1.5, 897, 1477), (699.25, 0, 860)),
        ("roof", (698.5, 897, 1.5), (350.75, 0, 1599.25)),
        ("sump floor", (698.5, 897, 1.5), (350.75, 0, 120.75)),
    ):
        body.append(box(name, size, center, light, edge=.3))
        body_col.append(box(name + " collision", size, center, edge=0))
    body.append(box("50 mm inset kick plinth", (600, 800, 120), (350, 0, 60), dark, edge=2))
    body_col.append(box("Inset kick collision", (600, 800, 120), (350, 0, 60), edge=2, segments=1))
    body.append(box("Blue upper accent band", (1.5, 896, 66), (-.75, 0, 1475), blue, edge=.3))
    body.append(printed_front("Fictional IMC branding", (180, 54), (-1.58, -295, 1475),
                              folder / "fictional_imc.png"))
    body.append(printed_front("Generic safety warning", (99, 55), (-.09, 345, 1060),
                              folder / "generic_warning.png"))
    # Shallow folded seam strips and an interior guide column give sheet-metal depth.
    for y in (-437, 437):
        body.append(box("Front edge return", (16, 6, 1470), (9.5, y, 860), dark, edge=1))
    for z in (780, 1240):
        body.append(box("Door slide track", (9, 880, 10), (-4.5, 0, z), dark, edge=1))
        body_col.append(box("Door track collision", (9, 880, 10), (-4.5, 0, z), edge=.5, segments=1))
    for y in (-150, 150):
        body.append(box("Table support", (160, 40, 150), (300, y, 695), dark, edge=2))
    body.append(box("Spindle column", (80, 190, 700), (320, 0, 1090), dark, edge=2))
    body.append(box("Spindle guide rail", (45, 24, 640), (265, -67, 1120), steel, edge=1))
    body.append(box("Spindle guide rail", (45, 24, 640), (265, 67, 1120), steel, edge=1))
    # Chip slope spans the inside of the sill, below the workholding surface.
    slope = box("Inward sloping chip apron", (155, 476, 1.5), (83, 0, 775), metal, edge=.3)
    slope.data.transform(Matrix.Translation(Vector((.083, 0, .775))) @
                         Matrix.Rotation(math.radians(14), 4, "Y") @
                         Matrix.Translation(Vector((-.083, 0, -.775))))
    body.append(slope)
    rng = random.Random(9)
    # Low-relief solid curls: aluminium swarf is visual only, away from seating faces.
    for i in range(48):
        x, y = rng.uniform(190, 390), rng.choice((-1, 1)) * rng.uniform(70, 180)
        radius, a = rng.uniform(1.3, 3.3), rng.uniform(0, math.tau)
        profile = []
        for j in range(7):
            theta = a + j * math.pi / 7
            profile.append((x + radius * math.cos(theta), y + radius * math.sin(theta)))
        for j in reversed(range(7)):
            theta = a + j * math.pi / 7
            profile.append((x + (radius - .65) * math.cos(theta), y + (radius - .65) * math.sin(theta)))
        chip = props.extruded_profile(f"Curled aluminium chip {i}", profile, .35,
                                     (0, 0, 800.21), bevel_mm=.08, segments=2, material=steel)
        body.append(chip)
    body_obj = merge(body, "body")
    parts = [record("body", body_obj, body_col, 180, PAINT, .5)]

    # Door: a real frame hole, not an opaque rectangle with an alpha texture.
    frame = box("door", (25, 500, 440), (-17.5, 0, 1010), light, edge=2)
    boolean(frame, box("Door observation aperture cutter", (40, 380, 300), (-17.5, 0, 1010), edge=1))
    door_bits = [frame]
    door_col = []
    for size, center in (((25, 60, 440), (-17.5, -220, 1010)),
                         ((25, 60, 440), (-17.5, 220, 1010)),
                         ((25, 380, 70), (-17.5, 0, 825)),
                         ((25, 380, 70), (-17.5, 0, 1195))):
        door_col.append(box("Door frame collision", size, center, edge=1, segments=1))
    # A 1.5 mm seal ring sits inside the frame's existing aperture, behind glass.
    seal = box("Window gasket", (6, 383, 303), (-12.5, 0, 1010), black, edge=.5)
    boolean(seal, box("Gasket clear opening", (12, 377, 297), (-12.5, 0, 1010), edge=.5))
    door_bits.append(seal)
    handle_mat = grain(props.pbr_material("Handled stainless bar, fingerprints", "#BFC3C7", .35, 1,
                                          grime=.18, machining="brushed"), 120, .1, .005, (1, 2, .15))
    door_bits.append(props.cylinder("Door handle D20 x 180", 10, 180, (-60, -220, 990),
                                    bevel_mm=.5, vertices=48, material=handle_mat))
    door_col.append(props.cylinder("Handle collision", 10, 180, (-60, -220, 990),
                                   bevel_mm=0, vertices=24))
    for z in (915, 1065):
        door_bits.append(props.cylinder("Handle standoff", 6, 25, (-42.5, -220, z), bevel_mm=.4,
                                        vertices=32, material=steel, rotation_deg=(0, 90, 0)))
    door_obj = merge(door_bits, "door")
    joint = {"type": "slide", "parent": "body", "axis": [0, 1, 0],
             "anchor_m": [-.0175, 0, 1.01], "range": [0, .46], "damping": 8, "frictionloss": 3}
    window_mass = .38 * .30 * .005 * 1200
    parts.append(record("door", door_obj, door_col, 4.0 - window_mass, PAINT, .5,
                        parent="body", joint=joint, assembly_mass_kg=4.0))
    glass = props.pbr_material("Clear 5 mm polycarbonate observation window", "#D9E6F2", .1, 0)
    bsdf = glass.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Transmission Weight"].default_value = .85
    bsdf.inputs["IOR"].default_value = 1.58
    bsdf.inputs["Alpha"].default_value = .25
    window = box("door_window", (5, 380, 300), (-21.5, 0, 1010), glass, edge=.3)
    parts.append(record("door_window", window,
                        [box("Window collision", (5, 380, 300), (-21.5, 0, 1010), edge=0)],
                        window_mass, CLEAR, .4, parent="door", rgba=[.85, .90, .95, .25]))

    table = box("T-slot work table", (400, 400, 30), (220, 0, 785), metal, edge=1)
    for y in (-105, 0, 105):
        boolean(table, box("T slot neck", (404, 8, 8), (220, y, 798), edge=0))
        boolean(table, box("T slot undercut", (404, 16, 6), (220, y, 792), edge=0))
    parts.append(record("table", table,
                        [box("Table collision", (400, 400, 30), (220, 0, 785), edge=.5, segments=1)],
                        22, METAL, .35, parent="body"))

    vise_bits = [box("Vise base", (140, 90, 20), (100, 0, 810), cast, edge=1)]
    vise_col = [box("Vise base collision", (140, 90, 20), (100, 0, 810), edge=.5, segments=1)]
    for y in (-35, 35):
        vise_bits.append(box("Vise side rail", (140, 20, 30), (100, y, 835), cast, edge=1))
        vise_col.append(box("Vise side rail collision", (140, 20, 30), (100, y, 835), edge=.5, segments=1))
    vise_bits.append(props.cylinder("Vise leadscrew", 6, 110, (100, 0, 837), bevel_mm=.3,
                                    vertices=32, material=steel, rotation_deg=(0, 90, 0)))
    parts.append(record("vise_body", merge(vise_bits, "vise_body"), vise_col, 2.8, STEEL, .3, parent="table"))
    fixed = merge([box("Fixed jaw casting", (22, 90, 30), (134, 0, 865), cast, edge=.5),
                   box("Fixed serrated plate", (3, 90, 30), (121.5, 0, 865), jaw_mat, edge=.3)], "fixed_jaw")
    parts.append(record("fixed_jaw", fixed,
                        [box("Fixed jaw collision", (25, 90, 30), (132.5, 0, 865), edge=.4, segments=1)],
                        .48, STEEL, .3, parent="vise_body"))
    moving = merge([box("Moving jaw casting", (22, 90, 30), (46, 0, 865), cast, edge=.5),
                    box("Moving serrated plate", (3, 90, 30), (58.5, 0, 865), jaw_mat, edge=.3)], "vise_jaw")
    jaw_joint = {"type": "slide", "parent": "vise_body", "axis": [1, 0, 0],
                 "anchor_m": [.060, 0, .865], "range": [0, .020], "damping": 2, "frictionloss": 1}
    parts.append(record("vise_jaw", moving,
                        [box("Moving jaw collision", (25, 90, 30), (47.5, 0, 865), edge=.4, segments=1)],
                        .48, STEEL, .3, parent="vise_body", joint=jaw_joint))
    parallels, parallel_col = [], []
    # The 80 mm length runs along Y, parallel to the jaw faces. Putting it
    # along X would intersect the moving jaw throughout its closing stroke.
    for x in (90, 110):
        parallels.append(box("80 x 8 x 10 parallel", (8, 80, 10), (x, 0, 855), steel, edge=.3))
        parallel_col.append(box("Parallel collision", (8, 80, 10), (x, 0, 855), edge=.2, segments=1))
    parts.append(record("parallels", merge(parallels, "parallels"), parallel_col, .10, STEEL, .3,
                        parent="vise_body"))

    spindle = [box("Static spindle head 130 x 160 x 250", (130, 160, 250), (195, 0, 1365), light, edge=2),
               props.cylinder("D100 spindle nose", 50, 45, (195, 0, 1217.5), bevel_mm=1,
                              vertices=64, material=steel),
               props.cylinder("Tool holder", 17.5, 30, (195, 0, 1180), bevel_mm=.5,
                              vertices=48, material=dark),
               props.cylinder("D10 end mill, tip Z1150", 5, 15, (195, 0, 1157.5), bevel_mm=.3,
                              vertices=32, material=steel)]
    parts.append(record("spindle", merge(spindle, "spindle"),
                        [box("Spindle head collision", (130, 160, 250), (195, 0, 1365), edge=2, segments=1),
                         props.cylinder("Spindle nose collision", 50, 45, (195, 0, 1217.5),
                                        bevel_mm=0, vertices=32)],
                        18, STEEL, .3, parent="body"))

    panel_material = props.pbr_material("Printed CNC operating panel", "#FFFFFF", .6, 0, grime=.15)
    image_ink(panel_material, folder / "panel_print.png")
    panel = box("panel", (40, 150, 350), (-20, -365, 1075), panel_material, edge=2)
    mapped_uv(panel, "PrintUV", lambda p: ((-290 - p.y) / 150, (p.z - 900) / 350))
    parts.append(record("panel", panel,
                        [box("Panel collision", (40, 150, 350), (-20, -365, 1075), edge=2, segments=1)],
                        1.2, PAINT, .5, parent="body"))
    screen = printed_front("screen", (120, 90), (-40.8, -365, 1097), folder / "screen_ui.png")
    parts.append(record("screen", screen, [], .08, PAINT, .5, parent="panel", emission=.5))
    placements, exclusions = install_panel(parts, folder)
    assert before == library_hashes(), "Task 2 library was modified"
    complete(folder, parts,
             {"enclosure_xyz": [700, 900, 1600], "sheet_thickness": 1.5,
              "kick_height": 120, "kick_inset": 50, "front_x": 0,
              "opening_y": [-240, 240], "opening_z": [800, 1220],
              "door_frame_xyz": [25, 500, 440], "door_front_x": -30,
              "door_window_xyz": [5, 380, 300], "handle_diameter": 20,
              "handle_center_x_y": [-60, -220], "handle_z": [900, 1080],
              "table_top_z": 800, "table_x": [20, 420], "table_y": [-200, 200],
              "vise_body_x": [30, 170], "vise_body_y": [-45, 45], "vise_body_z": [800, 850],
              "fixed_jaw_face_x": 120, "moving_jaw_face_open_x": 60,
              "moving_jaw_face_clamped_x": 80, "jaw_plate_yz": [90, 30], "jaw_plate_z": [850, 880],
              "parallel_xyz": [8, 80, 10], "parallel_centres_x": [90, 110],
              "part_bottom_z": 860, "part_center_xy": [100, 0],
              "spindle_head_x": [130, 260], "end_mill_diameter": 10, "end_mill_tip_z": 1150,
              "panel_y": [-440, -290], "panel_z": [900, 1250], "panel_projection": 40,
              "screen_yz": [120, 90], "screen_z": [1052, 1142]},
             "Only the compact front section is modelled. Door/frame/handle plus the fixed glass "
             "child have a combined moving mass of exactly 4 kg: frame 3.316 kg, glass 0.684 kg. "
             "The untextured door_window inherits the door slide; do not add a second joint. "
             "The 180 mm handle bar centreline is X=-60 mm, 30 mm proud of the X=-30 door face; "
             "its outer surface reaches X=-70, so the complete closed prop is 770 X, not 700. "
             "A 2 mm general edge radius cannot fit 1.5 mm sheet: thin-stock edges use R0.3, "
             "while solid enclosure trim, the plinth, door and panel use R2. "
             "The prescribed 120 x 90 screen and 60 mm emergency disc at Z1180 cannot share the "
             "top of a 150 mm-wide panel without overlap. The screen is immediately below the "
             "emergency disc, Z1052-1142; emergency, button and panel positions are preserved. "
             "All seven panel library components are appended, copied and rigidly transformed from "
             "the existing Task 2 source; original 1024 atlases are copied byte-for-byte, not rebaked. "
             "Press direction is +X; the two-position selector rotates about outward -X, inheriting "
             "its detents and recommended initial MAN angle. All main machine atlases are 2048. "
             "The specified one-piece table collider fills visual T slots. The specified two spindle "
             "proxies cover head and nose, omitting the narrow holder/cutter. Specified door proxies "
             "omit small handle standoffs. Coolant streaks and solid curled aluminium chips are visual. "
             "Jaw serrations use a 1.8 mm diagonal colour pattern and 0.18 mm source-PBR relief. "
             "Jaw mass, damping=2 and frictionloss=1 were unspecified benchmark choices. Automatic "
             "clamping, interlocks, button springs, e-stop latch and selector detents are simulator "
             "contracts in metadata, not a demonstrated machining controller. "
             "The parallels' 80 mm length runs along Y, parallel to the jaw faces, at X90 and "
             "X110; this leaves both parallels clear of the complete jaw stroke. "
             "No workpiece is fused to this prop: place either separate bar at [0.100,0,0.860]. "
             "The suggested X=340 world placement puts the workpiece at X440 but the handle at X280, "
             "outside the stated X300-480 ideal reach. Use a phased scene: X360 machine placement "
             "puts the initial handle at X300 and part at X460; the complete Y460 door stroke takes "
             "its handle through the gap between the two hands' preferred Y bands. The emergency "
             "centre Z1180 is also above the Z800-1000 preferred band. Exact geometry is retained; "
             "scene-level reach and regrasp planning remain necessary.",
             extra={"door_assembly_mass_kg": 4.0, "device_placements": placements,
                    "panel_library_sha256": before, "collision_exclusions": exclusions,
                    "workpiece_pose_m": [.100, 0, .860],
                    "recommended_world_translation_m": [.360, 0, 0],
                    "jaw_controller": {"trigger": "cycle_start", "open_qpos_m": 0,
                                       "clamp_qpos_m": .020, "part_diameter_m": .040},
                    "preview_open": "preview_open.png", "preview_panel": "preview_panel.png"})


def build_tray():
    folder = start("part_tray_6")
    art(folder)
    material = props.pbr_material("Brushed aluminium laser etched IN OUT", "#BFC4C9", .3, 1,
                                  grime=.15, machining="brushed")
    ink = image_ink(material, folder / "tray_in_etch.png", "EtchUV")
    bump = material.node_tree.nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = .00006
    bump.inputs["Strength"].default_value = .3
    material.node_tree.links.new(ink.outputs["Color"], bump.inputs["Height"])
    material.node_tree.links.new(bump.outputs[0], material.node_tree.nodes["Principled BSDF"].inputs["Normal"])
    tray = box("body", (170, 250, 30), (0, 0, 15), material, edge=1)
    for x in (-35, 35):
        for y in (-75, 0, 75):
            cutter = revolve("D44 blind pocket with 1 x 45 entry chamfer",
                              [(0, 10), (22, 10), (22, 29), (23, 30), (23, 33), (0, 33)], sectors=64)
            cutter.data.transform(Matrix.Translation(Vector((x / 1000, y / 1000, 0))))
            boolean(tray, cutter)
    mapped_uv(tray, "EtchUV", lambda p: ((p.x + 85) / 170, (p.y + 125) / 250))
    collisions = [box("10 mm blind pocket floor", (170, 250, 10), (0, 0, 5), edge=.5, segments=1)]
    for x in (-35, 35):
        for y in (-75, 0, 75):
            bounds = (x - 25, x + 25, y - 25, y + 25)
            outline = regular_polygon(22 / math.cos(math.pi / 8), 8, math.pi / 8, (x, y))
            walls = wall_polygons(outline, bounds, (x, y), merge=False)
            assert len(walls) == 8
            collisions.extend(props.extruded_profile(f"Octagonal pocket wall {x} {y} {i}", wall,
                                                     20, (0, 0, 20), bevel_mm=0)
                              for i, wall in enumerate(walls))
    # Fill the material outside the six 50 x 50 pocket-wall cells with
    # seven boxes. Overlap at X/Y stripe intersections is allowed for proxies.
    for lo, hi in ((-85, -60), (-10, 10), (60, 85)):
        collisions.append(box("Solid X stripe", (hi - lo, 250, 20), ((lo + hi) / 2, 0, 20), edge=0))
    for lo, hi in ((-125, -100), (-50, -25), (25, 50), (100, 125)):
        collisions.append(box("Solid Y stripe", (170, hi - lo, 20), (0, (lo + hi) / 2, 20), edge=0))
    assert len(collisions) == 56
    complete(folder, [record("body", tray, collisions, .9)],
             {"xyz": [170, 250, 30], "pocket_diameter": 44, "pocket_depth": 20,
              "pocket_entry_chamfer": 1, "pocket_centres_xy": [[x, y] for x in (-35, 35) for y in (-75, 0, 75)],
              "collision_pocket_apothem": 22, "collision_walls_per_pocket": 8},
             "Six real 64-sided circular blind pockets, 20 mm deep, with 1 x 45 degree entry chamfers. "
             "Collision is one 10 mm floor plus eight convex wall sectors per pocket, each within "
             "a 50 x 50 mm cell; seven additional boxes fill all remaining material between cells. "
             "The union is the complete plate minus exactly six circumscribed octagonal pockets, "
             "apothem 22 mm; 56 pieces total, no hull across any pocket. The specifically required "
             "8-wall, apothem-22 proxy has a maximum 1.813 mm excess radial clearance at octagon "
             "vertices relative to a round R22 bore, exceeding the general 1 mm proxy target; the "
             "task-specific eight-wall prescription is preserved. IN and OUT are laser-etch albedo "
             "variants on identical UVs, with 0.06 mm etch relief in source PBR. MuJoCo only shows "
             "the colour mark, because it cannot render the PBR relief. This 0.9 kg tray is static, "
             "not a workpiece for single-hand pickup.",
             variants=[{"name": "IN", "textures": {"body": "body_albedo.png"}},
                       {"name": "OUT", "textures": {"body": "body_albedo_v1.png"},
                        "preview": "preview_out.png"}],
             extra={"variant_material": material.name, "static": True})


def build_bar(prop_id):
    folder = start(prop_id)
    if prop_id == "round_blank":
        side = grain(props.pbr_material("Mill finish 6061 bar #BCC1C6", "#BCC1C6", .5, 1,
                                        grime=.12), 950, .08, .012, (1, 1, .018))
        end = grain(props.pbr_material("Rough saw-cut end faces", "#BCC1C6", .65, 1,
                                       grime=.12, machining="brushed"),
                    1800, .14, .04, (.03, 1, .03))
        obj = props.cylinder("body", 20, 60, (0, 0, 30), bevel_mm=.4, segments=3,
                             vertices=96, material=side)
        obj.data.materials.append(end)
        for polygon in obj.data.polygons:
            if abs(polygon.normal.z) > .98:
                polygon.material_index = 1
        collisions = [props.cylinder("24-sided blank collision", 20, 60, (0, 0, 30),
                                     bevel_mm=0, vertices=24)]
        mass, notes = .204, "D40 x 60 mm 6061 blank; side mill finish roughness 0.5, longitudinal " \
                            "grain, and separate rough saw-cut end faces. R0.4 softened saw edges. " \
                            "One 24-sided inscribed cylinder collider, nominal diameter preserved."
        dimensions = {"diameter": 40, "height": 60, "edge_radius": .4}
    else:
        material = props.pbr_material("Bright turned 6061, concentric tool feed", "#D7DCE0", .25, 1,
                                      grime=.10, machining="turned")
        # Continuous closed profile includes both the blind bore and every 1 mm chamfer.
        obj = revolve("body", [(0, 0), (19, 0), (20, 1), (20, 39), (19, 40),
                               (16, 40), (15, 41), (15, 59), (14, 60), (7, 60),
                               (6, 59), (6, 46), (5, 45), (0, 45)], material, sectors=96)
        collisions = [props.cylinder("D40 x 40 base collision", 20, 40, (0, 0, 20),
                                     bevel_mm=0, vertices=32),
                      props.cylinder("D30 x 20 step collision", 15, 20, (0, 0, 50),
                                     bevel_mm=0, vertices=32)]
        mass, notes = .170, "D40 x 60 envelope; upper D30 step is 20 mm high; D12 blind bore " \
                            "ends at Z45 (15 mm deep). Every profile corner has a 1 x 45 degree " \
                            "chamfer, including bore mouth/bottom and step root. Bright turned " \
                            "finish roughness 0.25 with concentric feed marks. Exactly two nominal " \
                            "cylinder proxies D40 x 40 and D30 x 20, each 32 sides; per section 3.9 " \
                            "the bore is visual and filled by collision, and small chamfers are " \
                            "approximated. The specified 0.170 kg is authoritative rather than " \
                            "recomputed from the stepped/hollow geometric aluminium volume."
        dimensions = {"diameter": 40, "height": 60, "step_diameter": 30, "step_height": 20,
                      "hole_diameter": 12, "hole_depth": 15, "chamfer": 1}
    complete(folder, [record("body", obj, collisions, mass, size=1024)], dimensions, notes,
             extra={"mass_kg": mass, "friction": .35})


def render_exports(prop_id):
    folder = OUTPUT / prop_id
    data = json.loads((folder / "prop.json").read_text())
    for variant in data["variants"] or [{"name": "default", "textures": {}}]:
        props.reset_scene()
        visuals = []
        for spec in data["parts"]:
            bpy.ops.wm.obj_import(filepath=str(folder / spec["visual"]), forward_axis="Y", up_axis="Z")
            obj = bpy.context.selected_objects[0]
            obj.name = spec["name"]
            material = props.pbr_material(spec["name"] + " delivered albedo", "#FFFFFF", .5, 0)
            tree, bsdf = material.node_tree, material.node_tree.nodes["Principled BSDF"]
            texture = variant.get("textures", {}).get(spec["name"], spec["texture"])
            if texture:
                node = tree.nodes.new("ShaderNodeTexImage")
                node.image = bpy.data.images.load(str(folder / texture), check_existing=True)
                tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
                bsdf.inputs["Metallic"].default_value = .35 if spec["mujoco_material"]["specular"] >= .7 else 0
            else:
                bsdf.inputs["Base Color"].default_value = spec["rgba"]
                bsdf.inputs["Alpha"].default_value = spec["rgba"][3]
                bsdf.inputs["Transmission Weight"].default_value = .8
            obj.data.materials.clear()
            obj.data.materials.append(material)
            visuals.append(obj)
        target = QA / "delivered_previews" / prop_id
        target.mkdir(parents=True, exist_ok=True)
        props.render_preview(visuals, target / f"{variant['name']}.png")
        if prop_id == "cnc_front":
            for obj in visuals:
                if obj.name in ("door", "door_window"):
                    obj.location.y = .46
            props.render_preview(visuals, target / "door_open.png")
            panel = [obj for obj in visuals if obj.name in ("panel", "screen") or
                     any(p["name"] == obj.name and p.get("library_instance") for p in data["parts"])]
            props.render_preview(panel, target / "panel.png")
        log("RENDER_EXPORTS", prop_id, variant["name"])


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", choices=IDS, nargs="+", default=list(IDS))
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--reuse-bakes", action="store_true")
    parser.add_argument("--render-exports", action="store_true")
    ARGS = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    for prop_id in ARGS.props:
        if ARGS.render_exports:
            render_exports(prop_id)
        elif ARGS.export_only:
            bpy.ops.wm.open_mainfile(filepath=str(QA / f"{prop_id}_geometry.blend"))
            export_current()
        elif prop_id == "cnc_front":
            build_cnc()
        elif prop_id == "part_tray_6":
            build_tray()
        else:
            build_bar(prop_id)


if __name__ == "__main__":
    main()
    props.finalize_assets()
