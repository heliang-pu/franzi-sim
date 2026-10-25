"""Task 08 palletizing props; dimensions are millimetres, exports are metres.

Blender -b --factory-startup --python-exit-code 1 -P this_file -- [--props ID ...]
--geometry-only saves resumable inspection sources; --export-only resumes them.
--render-exports renders delivered OBJ/albedo assets, including every variant.
The isolated --artwork-worker entry point runs in the supplied Pillow Python.
No Task 5 asset is rebuilt; its modelling helpers are reused without mutation.
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
QA = HERE / "_qa" / "task08"
CARTONS = {"pallet_carton_160x80x80": (80, 160, 80, .35, "SKU-A"),
           "pallet_carton_120x80x80": (80, 120, 80, .25, "SKU-B")}
IDS = ("mini_pallet", "scissor_lift_table", *CARTONS, "pattern_sign")


def stacking_plans():
    """Real XY footprints, always within the 300 X by 400 Y pallet."""
    def grid(xs, ys, dims, sku, first=1):
        return [{"number": first + i, "sku": sku, "center_xy_mm": [x, y],
                 "size_xy_mm": list(dims)}
                for i, (x, y) in enumerate((x, y) for y in ys for x in xs)]
    return [
        {"name": "plan_01", "title": "COMPACT / SKU-B", "description": "Tight 2 x 2 cross stack",
         "layers": [grid([-40, 40], [-60, 60], (80, 120), "SKU-B"),
                    grid([-60, 60], [-40, 40], (120, 80), "SKU-B", 5)]},
        {"name": "plan_02", "title": "MIXED / SKU-A + SKU-B", "description": "Mixed-length 2 x 2 cross stack",
         "layers": [[{"number": i + 1, "sku": sku, "center_xy_mm": [x, y],
                      "size_xy_mm": [80, long]} for i, (x, y, long, sku) in enumerate(
                          [(-40, -80, 160, "SKU-A"), (40, -60, 120, "SKU-B"),
                           (-40, 80, 160, "SKU-A"), (40, 60, 120, "SKU-B")])],
                    grid([-60], [-40, 40], (160, 80), "SKU-A", 5)
                    + grid([80], [-40, 40], (120, 80), "SKU-B", 7)]},
        {"name": "plan_03", "title": "CLEARANCE / SKU-B", "description": "Spaced 2 x 2 cross stack",
         "layers": [grid([-45, 45], [-70, 70], (80, 120), "SKU-B"),
                    grid([-70, 70], [-45, 45], (120, 80), "SKU-B", 5)]},
    ]


def artwork_worker():
    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo

    request = json.load(sys.stdin)
    folder = Path(request["folder"])

    def font(size):
        for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                     "/System/Library/Fonts/Helvetica.ttc"):
            if Path(path).is_file():
                return ImageFont.truetype(path, size)
        return ImageFont.load_default(size=size)

    def save(image, filename):
        info = PngInfo()
        info.add(b"sRGB", b"\x00")
        image.save(folder / filename, pnginfo=info)

    if request["kind"] == "pallet":
        image = Image.new("RGB", (900, 600), "white")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((24, 24, 876, 576), 32, outline="#24211C", width=24)
        for y, text, size in ((122, "NOVA", 135), (300, "HT / 26", 112), (466, "FICT-008", 94)):
            draw.text((450, y), text, anchor="mm", fill="#24211C", font=font(size))
        save(image, "fictional_fumigation_stamp.png")
    elif request["kind"] == "carton":
        image = Image.new("RGB", (600, 1100), "white")
        draw = ImageDraw.Draw(image)
        draw.polygon([(300, 80), (90, 360), (214, 360), (214, 930),
                      (386, 930), (386, 360), (510, 360)], fill="#16191C")
        save(image, "orientation_arrow.png")
    elif request["kind"] == "sign":
        for index, plan in enumerate(stacking_plans(), 1):
            image = Image.new("RGB", (1680, 2376), "#F9F8F2")
            draw = ImageDraw.Draw(image)
            blue, ink, grey = "#1B4963", "#172C38", "#647581"
            draw.rectangle((0, 0, 1680, 244), fill=blue)
            draw.text((82, 60), f"STACKING PLAN  /  P08-{index:02d}", font=font(70), fill="white")
            draw.text((82, 159), plan["title"], font=font(43), fill="#E6ECED")
            draw.text((82, 282), plan["description"], font=font(43), fill=ink)
            draw.text((82, 346), "PALLET 300 x 400 mm   |   4 CARTONS / LAYER", font=font(34), fill=grey)
            for layer_index, layer in enumerate(plan["layers"]):
                y0 = 450 + layer_index * 820
                draw.text((88, y0), f"LAYER {layer_index + 1}", font=font(52), fill=blue)
                draw.text((810, y0 + 8), "0 deg" if layer_index == 0 else "90 deg", font=font(40), fill=ink)
                # A correctly scaled 300 X by 400 Y top view, +Y up.
                left, top, scale = 590, y0 + 85, 1.45
                right, bottom = left + 300 * scale, top + 400 * scale
                draw.rectangle((left, top, right, bottom), fill="#E6E5DE", outline=grey, width=5)
                draw.text((right + 32, top + 36), "+Y", font=font(30), fill=grey)
                draw.line((right + 40, top + 150, right + 40, top + 88), fill=grey, width=5)
                draw.polygon([(right + 40, top + 77), (right + 29, top + 96),
                              (right + 51, top + 96)], fill=grey)
                for item in layer:
                    x, y = item["center_xy_mm"]
                    dx, dy = item["size_xy_mm"]
                    cx, cy = left + (x + 150) * scale, top + (200 - y) * scale
                    box = (cx - dx * scale / 2 + 2, cy - dy * scale / 2 + 2,
                           cx + dx * scale / 2 - 2, cy + dy * scale / 2 - 2)
                    fill = "#DCAC68" if item["sku"] == "SKU-A" else "#A9C9DA"
                    draw.rounded_rectangle(box, 6, fill=fill, outline=blue, width=4)
                    draw.text((cx, cy - 13), str(item["number"]), font=font(52), fill=ink, anchor="mm")
                    draw.text((cx, cy + 35), item["sku"], font=font(24), fill=ink, anchor="mm")
                draw.text((88, y0 + 685), "Top view / numbered placement order", font=font(31), fill=grey)
            draw.rectangle((0, 2185, 1680, 2376), fill=blue)
            draw.text((84, 2226), "NOVA Logistics", font=font(70), fill="white")
            draw.text((87, 2321), "FICTIONAL TRAINING PLAN  /  GRIP THE 80 mm SIDE", font=font(26), fill="white")
            save(image, f"plan_{index:02d}.png")
    else:
        raise ValueError(request["kind"])


if __name__ == "__main__" and "--artwork-worker" in sys.argv:
    artwork_worker()
    raise SystemExit(0)

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

import prop_lib as props
from build_task05_carton_carry import (
    bevel, boolean, combine, finish, grain, handling_material, image_material,
    kraft_material, move, multiply_color, tape_material, tape_ribbon, wave_patch,
)

WOOD = {"specular": .10, "shininess": .15, "reflectance": 0.0}
PLASTIC = {"specular": .25, "shininess": .30, "reflectance": 0.0}
PAINT = {"specular": .30, "shininess": .30, "reflectance": 0.0}
PAPER = {"specular": .05, "shininess": .10, "reflectance": 0.0}
CLEAR = {"specular": .45, "shininess": .75, "reflectance": .03}
COMMON_NOTES = (
    "Metres, Z up, front -X, common bottom-centre prop frame, applied transforms. "
    "Triangulated OBJ with normals, 30-degree split smoothing and weighted normals; "
    "nonoverlapping Smart UV 66-degree atlases with 0.01 island margins. "
    "Each opaque rigid part has one RGB8 sRGB albedo, CPU Cycles 64 DIFFUSE COLOR * "
    "(0.4 + 0.6 * AO), excluding direct illumination and highlights. "
    "Full procedural Principled PBR and original artwork are packed in source.blend. "
    "512-square previews: CPU Cycles 64, neutral grey, three lights, front-left above. "
    "Instance colour variation may be multiplied by 0.97-1.03 in simulation. "
)


def artwork(folder, kind):
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(__file__), "--artwork-worker"],
                   input=json.dumps({"folder": str(folder), "kind": kind}), text=True, check=True)


def start(prop_id):
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    folder = OUTPUT / prop_id
    folder.mkdir(parents=True, exist_ok=True)
    print("TASK08_START", prop_id, bpy.app.version_string, flush=True)
    return folder


def box(name, dims, center, mat=None, radius=1):
    return props.box(name, dims, center, bevel_mm=radius, segments=3, material=mat)


def hide(objects):
    for obj in objects:
        obj.hide_render = True
        obj.hide_set(True)


def uv_map(obj, name, mapping):
    layer = obj.data.uv_layers.get(name) or obj.data.uv_layers.new(name=name)
    for loop in obj.data.loops:
        xyz = tuple(v * 1000 for v in obj.data.vertices[loop.vertex_index].co)
        layer.data[loop.index].uv = mapping(xyz)


def ink_material(material, path, uv_name, name):
    result = material.copy()
    result.name = name
    tree = result.node_tree
    node = tree.nodes.new("ShaderNodeTexImage")
    node.name = "Artwork"
    node.image = bpy.data.images.load(str(path), check_existing=True)
    node.image.colorspace_settings.name = "sRGB"
    node.extension = "EXTEND"
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_name
    tree.links.new(uv.outputs["UV"], node.inputs["Vector"])
    multiply_color(result, node.outputs["Color"], name)
    return result


def part(name, visual, collisions, mass, friction, material, size=2048, rgba=None, parent=None):
    result = {"name": name, "source_visual_object": visual.name,
              "source_collision_objects": [obj.name for obj in collisions],
              "mass_kg": mass, "friction": friction, "mujoco_material": material,
              "texture_px": size, "parent": parent, "joint": None}
    if rgba is not None:
        result["rgba"], result["texture_px"] = rgba, None
    return result


def complete(folder, parts, variants, dimensions, notes, extra=None, alternates=None):
    alternates = alternates or []
    all_specs = parts + alternates
    for spec in all_specs:
        obj = bpy.data.objects[spec["source_visual_object"]]
        finish(obj)
        obj["rigid_part"] = spec["name"]
        if spec["texture_px"]:
            props.uv_unwrap(obj)
            for loop in obj.data.uv_layers["UVMap"].data:
                loop.uv = (.0005 + .999 * loop.uv.x, .0005 + .999 * loop.uv.y)
        hide([bpy.data.objects[name] for name in spec["source_collision_objects"]])
    visuals = [bpy.data.objects[spec["source_visual_object"]] for spec in parts]
    for spec in alternates:
        hide([bpy.data.objects[spec["source_visual_object"]]])
    budget = "small" if folder.name in CARTONS else "large" if folder.name == "scissor_lift_table" else "medium"
    data = {"id": folder.name, "task": "palletizing", "units": "m", "origin": "bottom_center",
            "bbox_mm": [round(v, 5) for v in props.bbox_mm(visuals)],
            "triangles": props.triangle_count(visuals), "budget_class": budget,
            "source": "source.blend", "preview": "preview.png", "parts": parts, "variants": variants,
            "dimensions_mm": dimensions, "notes": notes + " " + COMMON_NOTES,
            "build_blender_version": bpy.app.version_string, **(extra or {})}
    assert data["triangles"] <= {"small": 10000, "medium": 50000, "large": 150000}[budget], data["triangles"]
    for spec in alternates:
        assert props.triangle_count([bpy.data.objects[spec["source_visual_object"]]]) <= 50000
    manifest = {"data": data, "alternates": alternates}
    scene = bpy.context.scene
    scene["task08_manifest"] = json.dumps(manifest)
    scene["prop_id"] = folder.name
    scene["bake_recipe"] = "CPU Cycles 64 DIFFUSE COLOR * (0.4 + 0.6 * AO), RGB8 sRGB"
    scene["preview_recipe"] = "CPU Cycles 64, 512 square, neutral grey, three lights, front-left"
    scene["dimensions_mm"] = json.dumps(dimensions)
    QA.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    props._select(visuals)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(QA / f"{folder.name}_geometry.blend"), check_existing=False)
    (QA / f"{folder.name}_preflight.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("TASK08_GEOMETRY", folder.name, data["bbox_mm"], data["triangles"], flush=True)
    if not ARGS.geometry_only:
        export_current()


def bake_one(obj, size, target):
    if ARGS.reuse_bakes and target.name not in ARGS.rebake_textures and target.is_file():
        header = target.read_bytes()[:26]
        if header[:8] == b"\x89PNG\r\n\x1a\n" and int.from_bytes(header[16:20], "big") == size and int.from_bytes(header[20:24], "big") == size:
            print("TASK08_REUSE_BAKE", target.name, flush=True)
            return
    for other in bpy.context.scene.objects:
        if other.type == "MESH":
            other.hide_render = other != obj
    obj.hide_set(False)
    obj.hide_render = False
    print("TASK08_BAKE", target.name, size, flush=True)
    props.bake_albedo([obj], size, target)


def export_current():
    manifest = json.loads(bpy.context.scene["task08_manifest"])
    data, alternates = manifest["data"], manifest["alternates"]
    folder = OUTPUT / data["id"]
    if data["id"] == "mini_pallet":
        for material in bpy.data.materials:
            if material.name.startswith("Pine "):
                wood_growth_rings(material, "bottom runner" in material.name)
    exported = []
    for spec in data["parts"] + alternates:
        name = spec["name"]
        obj = bpy.data.objects[spec["source_visual_object"]]
        texture = spec.get("texture_filename", f"{name}_albedo.png") if spec["texture_px"] else None
        if texture:
            bake_one(obj, spec["texture_px"], folder / texture)
        for variant in spec.get("texture_variants", []):
            material = bpy.data.materials[variant["material"]]
            node = material.node_tree.nodes["Artwork"]
            old = node.image
            node.image = bpy.data.images.load(str(folder / variant["source_image"]), check_existing=True)
            node.image.use_fake_user = True
            bake_one(obj, spec["texture_px"], folder / variant["texture"])
            node.image = old
        collision_files = props.export_collision(name, [bpy.data.objects[n] for n in spec["source_collision_objects"]], folder)
        hide([bpy.data.objects[n] for n in spec["source_collision_objects"]])
        result = {key: spec[key] for key in ("name", "mass_kg", "friction", "mujoco_material", "texture_px",
                                           "parent", "joint", "source_visual_object", "source_collision_objects")}
        result.update(visual=props.export_obj(obj, folder / f"{name}.obj"), texture=texture,
                      collision=collision_files)
        if "rgba" in spec:
            result["rgba"] = spec["rgba"]
        exported.append(result)
    count = len(data["parts"])
    data["parts"] = exported[:count]
    if alternates:
        alt = exported[count]
        variant = next(v for v in data["variants"] if v["name"] == "blue_hdpe")
        obj = bpy.data.objects[alt["source_visual_object"]]
        override = {key: alt[key] for key in ("visual", "texture", "collision", "mujoco_material", "source_visual_object")}
        override["collision"] = data["parts"][0]["collision"]
        override.update(triangles=props.triangle_count([obj]), bbox_mm=props.bbox_mm([obj]))
        variant["part_overrides"] = {"body": override}
    visuals = [bpy.data.objects[p["source_visual_object"]] for p in data["parts"]]
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in visuals
    for obj in visuals:
        obj.hide_set(False)
        obj.hide_render = False
    props.render_preview(visuals, folder / "preview.png")
    if alternates:
        alt = bpy.data.objects[alternates[0]["source_visual_object"]]
        visuals[0].hide_render = True
        alt.hide_set(False)
        alt.hide_render = False
        props.render_preview([alt], folder / "preview_blue_hdpe.png")
        hide([alt])
        visuals[0].hide_render = False
    if data["id"] == "pattern_sign":
        material = bpy.data.materials[data["artwork_material"]]
        node = material.node_tree.nodes["Artwork"]
        original = node.image
        for index in (2, 3):
            node.image = bpy.data.images.load(str(folder / f"plan_{index:02d}.png"), check_existing=True)
            node.image.use_fake_user = True
            props.render_preview(visuals, folder / f"preview_plan_{index:02d}.png")
        node.image = original
    props.write_prop_json(data, folder / "prop.json")
    bpy.context.scene["task08_delivered"] = json.dumps(data)
    for image in bpy.data.images:
        if image.source == "FILE":
            image.use_fake_user = True
    bpy.ops.file.pack_all()
    props._select(visuals)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    print("TASK08_COMPLETE", data["id"], data["bbox_mm"], data["triangles"],
          sum(len(p["collision"]) for p in data["parts"]), flush=True)


def wood_growth_rings(material, along_y=False):
    """Distinct wavy latewood lines, in addition to submillimetre fibres."""
    tree = material.node_tree
    if tree.nodes.get("Pine wavy growth rings"):
        return material
    coords = tree.nodes.new("ShaderNodeNewGeometry")
    mapping = tree.nodes.new("ShaderNodeVectorMath")
    mapping.operation = "MULTIPLY"
    mapping.inputs[1].default_value = (1, .055, .5) if along_y else (.055, 1, .5)
    tree.links.new(coords.outputs["Position"], mapping.inputs[0])
    wave = tree.nodes.new("ShaderNodeTexWave")
    wave.name = "Pine wavy growth rings"
    wave.wave_type = "BANDS"
    wave.bands_direction = "X" if along_y else "Y"
    wave.inputs["Scale"].default_value = 48
    wave.inputs["Distortion"].default_value = 6.5
    wave.inputs["Detail"].default_value = 3
    wave.inputs["Detail Scale"].default_value = 1.3
    tree.links.new(mapping.outputs[0], wave.inputs["Vector"])
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[1])
    for index, (position, shade) in enumerate(((0, 1.03), (.61, 1.0), (.82, .79), (.94, .65), (1, .86))):
        element = ramp.color_ramp.elements[0] if index == 0 else ramp.color_ramp.elements.new(position)
        element.position = position
        element.color = (shade, shade, shade, 1)
    tree.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    multiply_color(material, ramp.outputs["Color"], "Pine earlywood and dark latewood")
    return material


def pine_material(name, color, along_y=False):
    material = props.pbr_material(name, color, .75, 0, wear=.025, grime=.18)
    grain(material, 900, .24, .055, (1, .018, 1) if along_y else (.018, 1, 1))
    return wood_growth_rings(material, along_y)


def build_pallet():
    folder = start("mini_pallet")
    artwork(folder, "pallet")
    pine = [pine_material(f"Pine top board {i} with elongated grain", color) for i, color in enumerate(
        ("#C49A6C", "#C69C6E", "#C1986B", "#C59B6D", "#C2996C"))]
    bottom = pine_material("Pine bottom runner, grain along Y", "#BD9569", along_y=True)
    iron = props.pbr_material("Dark slightly oxidised nail heads", "#36342E", .62, .65, grime=.18)
    objects, collisions = [], []
    for i, yy in enumerate((-168, -84, 0, 84, 168)):
        objects.append(box(f"Top board {i:02d} 300x64x12", (300, 64, 12), (0, yy, 74), pine[i], 1))
        collisions.append(box(f"COL top board {i:02d}", (300, 64, 12), (0, yy, 74), radius=0))
        for xx in (-120, 0, 120):
            for dy in (-20, 20):
                objects.append(props.cylinder("Flush dark nail head", 1.75, .6, (xx, yy + dy, 79.73),
                                               bevel_mm=.10, segments=2, vertices=16, material=iron))
    for xx in (-120, 0, 120):
        for yy in (-170, 0, 170):
            block = box(f"Pine block x{xx:+d} y{yy:+d} 60x60x56", (60, 60, 56), (xx, yy, 40), pine[2], 1)
            if xx == -120 and yy == 170:
                material = ink_material(pine[2], folder / "fictional_fumigation_stamp.png", "StampUV",
                                        "Fictional NOVA HT stamp, not an official certification")
                block.data.materials.append(material)
                for face in block.data.polygons:
                    if face.normal.x < -.9:
                        face.material_index = 1
                uv_map(block, "StampUV", lambda p: (.5 - (p[1] - yy) / 48, .5 + (p[2] - 40) / 32))
            objects.append(block)
    for xx in (-118, 0, 118):
        objects.append(box(f"Bottom runner x{xx:+d} 64x400x12", (64, 400, 12), (xx, 0, 6), bottom, 1))
    collisions.append(box("COL specified combined blocks and bottom runners", (300, 400, 68), (0, 0, 34), radius=0))
    wood = combine(objects, "body")

    hdpe = grain(props.pbr_material("Blue HDPE #1F4E8C, worn moulded ribs", "#1F4E8C", .55, 0,
                                    wear=.06, grime=.16), 1450, .06, .025)
    plastic = []
    for i, yy in enumerate((-168, -84, 0, 84, 168)):
        panel = box(f"HDPE grid top panel {i:02d}", (300, 64, 12), (0, yy, 74), hdpe, 0)
        for xx in (-120, -80, -40, 0, 40, 80, 120):
            for dy in (-15, 15):
                cutter = props.box("Actual rounded through-grid cell", (32, 18, 16), (xx, yy + dy, 74),
                                   bevel_mm=1.4, segments=2)
                boolean(panel, cutter)
        bevel(panel, .7)
        plastic.append(panel)
    for xx in (-120, 0, 120):
        for yy in (-170, 0, 170):
            plastic.append(box("HDPE moulded support block", (60, 60, 56), (xx, yy, 40), hdpe, 1.2))
    for xx in (-118, 0, 118):
        plastic.append(box("HDPE bottom runner", (64, 400, 12), (xx, 0, 6), hdpe, 1))
    plastic = combine(plastic, "body_hdpe")
    alt = part("body_hdpe", plastic, [], 2.5, .50, PLASTIC)
    alt["texture_filename"] = "body_albedo_v1.png"
    # Both variants deliberately use the exact specified six proxy boxes.
    variants = [{"name": "pine", "textures": {"body": "body_albedo.png"}},
                {"name": "blue_hdpe", "textures": {"body": "body_albedo_v1.png"},
                 "part_overrides": {"body": {"visual": "body_hdpe.obj"}},
                 "preview": "preview_blue_hdpe.png", "mass_kg": 2.5}]
    complete(folder, [part("body", wood, collisions, 2.5, .50, WOOD)], variants,
             {"nominal_xyz": [300, 400, 80], "top_board_xyz": [300, 64, 12], "top_board_y": [-168, -84, 0, 84, 168],
              "top_board_gap": 20, "block_xyz": [60, 60, 56], "block_x": [-120, 0, 120],
              "block_y": [-170, 0, 170], "bottom_board_xyz": [64, 400, 12], "bottom_board_x": [-118, 0, 118]},
             "Five pine deckboards, nine support blocks and three Y-grain bottom runners. Elongated wood grain, "
             "3-percent board variation, worn edges, cavity dirt, thirty dark nail heads and an original fictional "
             "NOVA / HT / FICT-008 stamp. Nail heads stand 0.03 mm proud of the nominal 80 mm deck. "
             "The blue HDPE variant replaces the body mesh, not just the colour: seventy real rounded through-holes "
             "form five grid panels. Apply variants[].part_overrides by part name and then variants[].textures; "
             "the same common frame and six prescribed collision boxes are retained. SPEC EXCEPTION: the required "
             "single lower 300x400x68 proxy fills the fork openings, and the top proxies fill HDPE grid cells. "
             "These deliberately coarse, task-specific proxies override the general 1 mm concavity rule. "
             "This pallet may be fixed to the lift; it is not a fork-entry collision model.",
             {"fixed_to_lift_allowed": True, "static_mount": "scissor_lift_table at z=0.720 m",
              "recommended_placement_m": {"origin_x": .40, "origin_y": 0, "support_z": .720},
              "variant_contract": "Merge part_overrides into the named base part; then merge textures. Unspecified fields inherit."},
             [alt])


def aim(obj, start_mm, end_mm):
    start_v, end_v = Vector(start_mm), Vector(end_mm)
    transform = (Matrix.Translation((start_v + end_v) * .0005)
                 @ (end_v - start_v).to_track_quat("Z", "Y").to_matrix().to_4x4())
    obj.data.transform(transform)
    obj.data.update()
    return obj


def strut(name, a, b, material):
    length = (Vector(b) - Vector(a)).length
    obj = box(name, (30, 15, length), (0, 0, 0), material, 0)
    boolean(obj, box("Open 2 mm steel tube wall", (26, 11, length + 4), (0, 0, 0), radius=0))
    bevel(obj, .6)
    return aim(obj, a, b)


def build_lift():
    folder = start("scissor_lift_table")
    yellow = grain(props.pbr_material("Yellow steel #E3B505 powder coat", "#E3B505", .60, 0,
                                      wear=.06, grime=.17), 1700, .055, .02)
    black = props.pbr_material("Black scissor steel #222222", "#222222", .45, .8,
                               wear=.04, grime=.16, machining="brushed")
    zinc = props.pbr_material("Brushed pivot and hydraulic chrome", "#BFC3C7", .30, 1,
                              grime=.13, machining="brushed")
    hazard = yellow.copy()
    hazard.name = "Original yellow-black diagonal table-edge hazard stripes"
    tree = hazard.node_tree
    position = tree.nodes.new("ShaderNodeNewGeometry")
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(position.outputs["Position"], split.inputs[0])
    phase = props._math(tree, "DIVIDE", props._math(tree, "ADD", split.outputs["Z"],
                       props._math(tree, "ADD", split.outputs["X"], split.outputs["Y"])), .044)
    mask = props._math(tree, "LESS_THAN", props._math(tree, "FRACT", phase), .5)
    factor = props._math(tree, "ADD", .025, props._math(tree, "MULTIPLY", mask, .975))
    multiply_color(hazard, factor, "44 mm diagonal warning repeat")
    top = box("500Y x400X x40 top, top z720", (400, 500, 40), (0, 0, 700), yellow, 2)
    top.data.materials.append(hazard)
    for face in top.data.polygons:
        if abs(face.normal.z) < .25:
            face.material_index = 1
    base = box("520Y x420X x60 base", (420, 520, 60), (0, 0, 30), yellow, 2)
    objects = [base, top]
    for yy in (-174, 174):
        objects.append(strut("Black scissor ascending 30x15 tube", (-156, yy - 8.5, 78), (156, yy - 8.5, 662), black))
        objects.append(strut("Black scissor descending 30x15 tube", (156, yy + 8.5, 78), (-156, yy + 8.5, 662), black))
        for xx in (-156, 156):
            for zz in (73, 667):
                objects.append(box("Yellow pinned scissor mounting clevis", (44, 42, 26), (xx, yy, zz), yellow, 1.2))
                objects.append(props.cylinder("Zinc mounting pivot", 7, 48, (xx, yy, 78 if zz == 73 else 662),
                                               bevel_mm=.5, vertices=24, material=zinc, rotation_deg=(90, 0, 0)))
        objects.append(props.cylinder("Scissor central pin", 10, 50, (0, yy, 370), bevel_mm=.5,
                                       vertices=32, material=zinc, rotation_deg=(90, 0, 0)))
        for sy in (-1, 1):
            objects.append(props.cylinder("Black central pivot retaining washer", 16, 2,
                                           (0, yy + sy * 24, 370), bevel_mm=.4, vertices=32,
                                           material=black, rotation_deg=(90, 0, 0)))
    # A transverse tie links the two ascending arms; the ram is pinned to it
    # and to the base, rather than floating in the open central bay.
    tie_x = -156 + (454 - 78) * 312 / 584
    objects.append(box("Black transverse scissor tie beam", (30, 350, 20), (tie_x, -8.5, 454), black, 1))
    objects.append(box("Yellow hydraulic base clevis", (40, 44, 48), (-126, 0, 84), yellow, 1))
    objects.append(box("Yellow hydraulic tie-beam clevis", (34, 36, 30), (tie_x, 0, 444), yellow, 1))
    for xx, zz in ((-126, 95), (tie_x, 444)):
        objects.append(props.cylinder("Hydraulic anchored transverse pin", 6, 48, (xx, 0, zz),
                                       bevel_mm=.5, vertices=24, material=zinc, rotation_deg=(90, 0, 0)))
    a, b = Vector((-126, 0, 95)), Vector((tie_x, 0, 444))
    direction = (b - a).normalized()
    mid = a + direction * 250
    objects.append(aim(props.cylinder("Hydraulic black cylinder barrel", 17, 250, bevel_mm=1,
                                      vertices=32, material=black), a, mid))
    objects.append(aim(props.cylinder("Chrome hydraulic piston rod", 8, (b - mid).length,
                                      bevel_mm=.5, vertices=32, material=zinc), mid, b))
    for xx in (-180, 180):
        for yy in (-225, 225):
            objects.append(props.cylinder("Recessed hexagonal base fixing bolt", 6, 3, (xx, yy, 59.3),
                                           bevel_mm=.5, vertices=6, material=zinc))
    label_path = folder / "lift_rating_label.png"
    props.make_label_png(label_path, 1120, 280,
                         {"brand": "NOVA Logistics", "title": "NOVA Logistics / LIFT-08",
                          "lines": ["TOP 720 mm   |   SETUP 620 - 760 mm", "STATIC BENCHMARK PROP / KEEP CLEAR"],
                          "background": "#F1EACB", "foreground": "#16191C"})
    label_mat = image_material("Fictional lift setup label", label_path, "LabelUV")
    label = box("Lift setup nameplate", (.10, 112, 28), (-200.025, 0, 700), label_mat, .025)
    uv_map(label, "LabelUV", lambda p: (.5 - p[1] / 112, .5 + (p[2] - 700) / 28))
    objects.append(label)
    visual = combine(objects, "body")
    collisions = [box("COL top", (400, 500, 40), (0, 0, 700), radius=0),
                  box("COL base", (420, 520, 60), (0, 0, 30), radius=0)]
    complete(folder, [part("body", visual, collisions, 42.0, .50, PAINT)], [],
             {"top_xyz": [400, 500, 40], "top_surface_z": 720, "base_xyz": [420, 520, 60],
              "scissor_tube_cross_section": [30, 15], "scissor_pairs": 2, "pallet_top_z": 800},
             "Static yellow steel scissor table, two separated pairs of black 30x15x2 hollow steel tubes, "
             "zinc pins/retainers, hydraulic barrel and chrome rod. Diagonal black/yellow stripes on all four "
             "vertical top edges; surface grain, restrained edge wear and cavity dirt. Top is z=720 mm, so a "
             "nominal 80 mm pallet reaches 800 mm. Base dimensions use the same Y-first convention as the top: "
             "520Y x420X x60Z. Exactly two specified collision boxes: arms, hydraulic equipment and mounting "
             "clevises are visual only. The 620-760 mm top-surface range is configuration metadata, not an "
             "animated joint; changing height requires regenerating the linkage pose. No live movable mechanism "
             "is claimed. Mass 42 kg is an estimate because the spec gives none.",
             {"static": True, "adjustable_height_mm": [620, 760], "initial_top_height_mm": 720,
              "adjustable_height_m": [.620, .760], "height_reference": "top surface above base origin",
              "recommended_placement_m": {"origin_x": .40, "origin_y": 0, "support_z": 0}})


def build_carton(prop_id):
    x, y, height, mass, sku = CARTONS[prop_id]
    folder = start(prop_id)
    artwork(folder, "carton")
    kraft, tape = kraft_material(), tape_material()
    medium = props.pbr_material("Exposed wavy single-wall corrugated medium", "#C19A68", .92, 0, grime=.14)
    core_y, top = y - .30, height - .30
    body = box("4 mm FEFCO 0201 shell and major flaps", (x, core_y, top), (0, 0, top / 2), kraft, 0)
    boolean(body, box("Actual internal air", (x - 8, core_y - 8, top - 8), (0, 0, top / 2), radius=0))
    for z in (2, top - 2):
        boolean(body, box("1.5 mm major flap centre seam", (1.5, core_y - 8, 6), (0, 0, z), radius=0))
    for sx in (-1, 1):
        for sy in (-1, 1):
            boolean(body, props.cylinder("Compressed paper corner", 1.8, top * .53,
                                         (sx * (x / 2 + 1.12), sy * (core_y / 2 - .65), top * .63),
                                         bevel_mm=.15, vertices=16))
    patches = []
    for sy in (-1, 1):
        for sx in (-1, 1):
            lo, hi = (25, x / 2 - 2) if sx == 1 else (-x / 2 + 2, -25)
            boolean(body, box("Open 4 mm corrugated flap cut edge", (hi - lo, 1.4, 3.3),
                              ((lo + hi) / 2, sy * core_y / 2, top - 2), radius=0))
            patches.append(wave_patch("Actual wavy exposed flap medium", lo, hi,
                                       sy * (core_y / 2 - .35), top, medium))
        boolean(body, box("Top flap fold score", (x - 2, .32, .24),
                          (0, sy * (core_y / 2 - 4.6), top), radius=.04))
    label_center = (-x / 2 + .06, 0, height * .69)
    boolean(body, props.box("Label seating skewed two degrees", (.5, 94.3, 30.3), label_center,
                            bevel_mm=.04, rotation_deg=(2, 0, 0)))
    bevel(body, .4)
    icon_path = folder / "handling_icons.png"
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(HERE / "build_task05_carton_carry.py"), "--artwork-worker"],
                   input=json.dumps({"path": str(icon_path)}), text=True, check=True)
    printed = handling_material(kraft, icon_path)
    body.data.materials.append(printed)
    layer = body.data.uv_layers.new(name="HandlingUV")
    for face in body.data.polygons:
        front = face.normal.x < -.65 and face.center.x < props.mm(-x / 2 + 1)
        side = abs(face.normal.y) > .65 and abs(face.center.y) > props.mm(core_y / 2 - 1)
        if (front or side) and props.mm(3) < face.center.z < props.mm(top - 3):
            face.material_index = len(body.data.materials) - 1
        width = 90 if front else 68
        for loop_index in face.loop_indices:
            p = body.data.vertices[body.data.loops[loop_index].vertex_index].co * 1000
            horizontal = -p.y if front else p.x * (1 if face.normal.y > 0 else -1)
            layer.data[loop_index].uv = (.5 + horizontal / width, .5 + (p.z - 20) / (width * 460 / 1500))
    label_path = folder / "nova_sku_label.png"
    props.make_label_png(label_path, 1504, 480,
                         {"brand": "NOVA Logistics", "title": sku,
                          "lines": [f"NOVA Logistics / {int(mass * 1000)} g", "PALLETIZING / GRIP 80 mm SIDE"],
                          "barcode": True, "seed": sku, "background": "#F4F1E7", "foreground": "#17191B"})
    label_mat = image_material(f"Fictional NOVA {sku} label", label_path, "LabelUV")
    label = props.box(f"NOVA {sku} label skewed two degrees", (.12, 94, 30), label_center,
                      bevel_mm=.025, rotation_deg=(2, 0, 0), material=label_mat)
    unskew = Matrix.Rotation(math.radians(-2), 3, "X")
    label_uv = label.data.uv_layers.new(name="LabelUV")
    center = Vector([props.mm(v) for v in label_center])
    for loop in label.data.loops:
        local = (unskew @ (label.data.vertices[loop.vertex_index].co - center)) * 1000
        label_uv.data[loop.index].uv = (.5 - local.y / 94, .5 + local.z / 30)
    ribbon = tape_ribbon(core_y, top, tape)
    arrow_mat = ink_material(tape, folder / "orientation_arrow.png", "OrientationUV", "Black +Y orientation arrow on top tape")
    ribbon.data.materials.append(arrow_mat)
    for face in ribbon.data.polygons:
        if face.normal.z > .9:
            face.material_index = len(ribbon.data.materials) - 1
    uv_map(ribbon, "OrientationUV", lambda p: (.5 + p[0] / 30, .5 + p[1] / 55))
    objects = [body, *patches, label, ribbon]
    minor_depth = x / 2 - 4
    for end in (-1, 1):
        for z in (6.02, top - 6.02):
            objects.append(box("Internal 4 mm minor flap", (x - 8.1, minor_depth, 4),
                               (0, end * (core_y / 2 - 4.05 - minor_depth / 2), z), kraft, .3))
    for i, yy in enumerate((-.38, -.28, .29, .40)):
        objects.append(props.box("Fine tape wrinkle", (18, .5, .08), (-8 + i * 5, yy * core_y, top + .19),
                                 bevel_mm=.025, material=tape, rotation_deg=(0, 0, (-1) ** i * 11)))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1)
    bubble = bpy.context.object
    bubble.scale = [props.mm(v) for v in (3.5, 1.6, .12)]
    props._finish(bubble, "Small trapped tape air bubble", (5, .13 * core_y, top + .18), 0, 3, tape)
    objects.append(bubble)
    visual = combine(objects, "body")
    collider = box("COL nominal carton single box", (x, y, height), (0, 0, height / 2), radius=0)
    complete(folder, [part("body", visual, [collider], mass, .55, PAPER, 1024)], [],
             {"nominal_xyz": [x, y, height], "size_convention": "Y x X x height", "wall": 4,
              "major_flap_gap": 1.5, "tape_width": 48, "tape_end_fold": 50,
              "label_yz": [94, 30], "label_skew_deg": 2},
             f"Task 5 carton style scaled to {y}Y x{x}X x{height}Z: actual hollow 4 mm FEFCO 0201 shell, "
             "four inner minor flaps, 1.5 mm major-flap seams, exposed real wavy cut-edge flutes and compressed "
             "corners. Glossy 48 mm brown tape turns down 50 mm at each end, with fine wrinkles and a shallow "
             f"air bubble. Original generic arrows, umbrella and glass ink; fictional NOVA Logistics / {sku} "
             "label skewed 2 degrees. A large black top orientation arrow points +Y, printed on the sealing tape. "
             "Tape, label and bubble allowances stay inside the nominal box envelope. Exactly one full nominal "
             "box collider, ignoring the interior and flap gaps. Paper highlight parameters apply to the single "
             "export material; tape gloss survives in the source PBR. Task 8 specified masses replace Task 5's "
             "heavier mass variants. Grip the 80 mm X span with one hand; do not squeeze the 120/160 mm Y span.",
             {"sku": sku, "grasp_axis": [1, 0, 0], "grasp_width_mm": 80,
              "orientation_arrow_axis": [0, 1, 0], "single_hand_payload_kg": mass,
              "recommended_placement_m": {"origin_x": .40, "origin_y": .16, "support_z": .85}})


def build_sign():
    folder = start("pattern_sign")
    artwork(folder, "sign")
    base_mat = grain(props.pbr_material("Dark blue powder-coated sign base", "#264C68", .60, 0,
                                        wear=.06, grime=.17), 1700, .055, .02)
    base = box("220Y x60X x20 sign base", (60, 220, 20), (0, 0, 10), base_mat, 1)
    boolean(base, box("Real acrylic insertion slot", (3.3, 212, 11), (0, 0, 15), radius=.3))
    base = combine([base], "base")
    rgba = [.85, .90, .95, .25]
    clear_mat = props.pbr_material("Clear acrylic, uniform rgba and no texture", "#D9E6F2", .14, 0)
    bsdf = clear_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Transmission Weight"].default_value = .92
    bsdf.inputs["IOR"].default_value = 1.49
    bsdf.inputs["Alpha"].default_value = .25
    clear_mat.diffuse_color = (*rgba[:3], rgba[3])
    acrylic = combine([box("210Y x297Z x3X clear acrylic", (3, 210, 297), (0, 0, 158.5), clear_mat, .45)], "acrylic")
    # The sheet sits in the centre plane of the transparent acrylic holder.
    sheet_mat = image_material("Original NOVA stacking-plan artwork", folder / "plan_01.png", "PlanUV")
    image_node = next(n for n in sheet_mat.node_tree.nodes if n.type == "TEX_IMAGE")
    image_node.name = "Artwork"
    sheet = box("196Y x283Z stacking-plan sheet", (.12, 196, 283), (0, 0, 158.5), sheet_mat, .025)
    uv_map(sheet, "PlanUV", lambda p: (.5 - p[1] / 196, .5 + (p[2] - 158.5) / 283))
    sheet = combine([sheet], "plan_sheet")
    sheet_spec = part("plan_sheet", sheet, [], .01, .5, PAPER, parent="base")
    sheet_spec["texture_variants"] = [{"material": sheet_mat.name, "source_image": f"plan_{i:02d}.png",
                                        "texture": f"plan_sheet_albedo_v{i - 1}.png"} for i in (2, 3)]
    variants = [{"name": plan["name"], "textures": {"plan_sheet": "plan_sheet_albedo.png" if i == 0 else f"plan_sheet_albedo_v{i}.png"},
                 "stacking_plan": plan, "preview": "preview.png" if i == 0 else f"preview_plan_{i + 1:02d}.png"}
                for i, plan in enumerate(stacking_plans())]
    collider = box("COL single thin acrylic sign proxy", (3, 210, 297), (0, 0, 158.5), radius=0)
    complete(folder, [part("base", base, [], .45, .5, PAINT),
                      part("acrylic", acrylic, [collider], .22, .4, CLEAR, rgba=rgba, parent="base"), sheet_spec],
             variants, {"acrylic_xyz": [3, 210, 297], "base_xyz": [60, 220, 20],
                        "acrylic_bottom_z": 10, "insertion_depth": 10, "sheet_yz": [196, 283]},
             "A4-size 210Y x297Z x3X clear acrylic inserted 10 mm into an actual slot in the 220Y x60X x20Z base. "
             "Separate untextured acrylic part uses uniform rgba [0.85,0.90,0.95,0.25]; the opaque paper sheet "
             "is a separate part, never an alpha texture. Three original numbered 2x2 layer-one plans and "
             "90-degree-rotated layer-two cartons, with fictional NOVA Logistics footer. Compact SKU-B, mixed "
             "SKU-A/B and clearance SKU-B plans have distinct packing footprints. All eight numbered cartons "
             "in each plan fit within a 300X x400Y pallet; mixed layer 2 is rearranged after rotating individual "
             "cartons because a rigid 90-degree rotation of the 320 mm long layer would overhang the 300 mm "
             "pallet. SPEC CONFLICT: one convex box cannot cover both the 60 mm deep base and 3 mm thin sign "
             "without large empty-space collisions. The single requested box therefore follows the acrylic; "
             "the out-of-workspace stationary base is visual-only. Slot and paper are not collision geometry. "
             "Masses are modelling estimates. The base is oriented with its 220 mm width along robot-left Y.",
             {"static": True, "artwork_material": sheet_mat.name, "stacking_plans": stacking_plans(),
              "recommended_placement_m": {"origin_x": .56, "origin_y": .35, "support_z": .80}})


def render_exports(prop_id):
    folder = OUTPUT / prop_id
    data = json.loads((folder / "prop.json").read_text())
    choices = [(None, "preview_export.png")]
    if prop_id == "mini_pallet":
        choices.append((data["variants"][1], "preview_export_blue_hdpe.png"))
    elif prop_id == "pattern_sign":
        choices.extend((v, f"preview_export_{v['name']}.png") for v in data["variants"][1:])
    for variant, preview in choices:
        props.reset_scene()
        bpy.ops.outliner.orphans_purge(do_recursive=True)
        objects = []
        for base_part in data["parts"]:
            spec = dict(base_part)
            if variant:
                spec.update(variant.get("part_overrides", {}).get(spec["name"], {}))
                spec["texture"] = variant.get("textures", {}).get(spec["name"], spec["texture"])
            previous = set(bpy.data.objects)
            bpy.ops.wm.obj_import(filepath=str(folder / spec["visual"]), forward_axis="Y", up_axis="Z")
            for obj in set(bpy.data.objects) - previous:
                material = props.pbr_material(spec["name"] + " exported single material", "#FFFFFF", .65, 0)
                tree = material.node_tree
                bsdf = tree.nodes["Principled BSDF"]
                bsdf.inputs["Specular IOR Level"].default_value = spec["mujoco_material"]["specular"]
                bsdf.inputs["Roughness"].default_value = 1 - spec["mujoco_material"]["shininess"]
                if spec["texture"]:
                    node = tree.nodes.new("ShaderNodeTexImage")
                    node.image = bpy.data.images.load(str(folder / spec["texture"]), check_existing=True)
                    uv = tree.nodes.new("ShaderNodeUVMap")
                    uv.uv_map = obj.data.uv_layers.active.name
                    tree.links.new(uv.outputs["UV"], node.inputs["Vector"])
                    tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
                else:
                    bsdf.inputs["Base Color"].default_value = (*spec["rgba"][:3], 1)
                    bsdf.inputs["Alpha"].default_value = spec["rgba"][3]
                obj.data.materials.clear()
                obj.data.materials.append(material)
                objects.append(obj)
        bpy.context.view_layer.update()
        props.render_preview(objects, folder / preview)
        print("TASK08_EXPORT_PREVIEW", prop_id, preview, flush=True)


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", choices=IDS, nargs="+", default=list(IDS))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--geometry-only", action="store_true")
    modes.add_argument("--export-only", action="store_true")
    modes.add_argument("--render-exports", action="store_true")
    parser.add_argument("--reuse-bakes", action="store_true", help="Resume an export with existing valid-size PNG bakes")
    parser.add_argument("--rebake-textures", nargs="*", default=[], help="Regenerate these PNGs even with --reuse-bakes")
    ARGS = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    for prop_id in ARGS.props:
        if ARGS.export_only:
            bpy.ops.wm.open_mainfile(filepath=str(QA / f"{prop_id}_geometry.blend"))
            export_current()
        elif ARGS.render_exports:
            render_exports(prop_id)
        elif prop_id == "mini_pallet":
            build_pallet()
        elif prop_id == "scissor_lift_table":
            build_lift()
        elif prop_id in CARTONS:
            build_carton(prop_id)
        else:
            build_sign()


if __name__ == "__main__":
    main()
    props.finalize_assets()
