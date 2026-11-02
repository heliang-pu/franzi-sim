"""Build the section 6.1 handle tote without changing the original small tote.

Usage: blender -b --factory-startup -t 6 -P tools/blender/build_task18_handle_tote.py --
All modelling dimensions are millimetres; exports share the bottom-centre metre frame.
The private --artwork worker uses PROP_PYTHON and only the bundled DejaVu Sans font.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VARIANTS = (
    ("blue", "#1F5FA8", "BIN-A07"),
    ("grey", "#9AA0A6", "BIN-A07"),
    ("red", "#C62828", "NG"),
    ("yellow", "#F9A825", "RECHECK"),
    ("green", "#2E7D32", "OK"),
)


def artwork(folder: Path) -> None:
    """Draw exact 22 mm cap-height text on a 90 x 40 mm white card."""
    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo

    font = ImageFont.truetype(str(HERE / "fonts" / "DejaVuSans-Bold.ttf"), 400)
    records = {}
    for name, _, title in VARIANTS:
        left, top, right, bottom = font.getbbox(title)
        mask = Image.new("L", (right - left, bottom - top), 0)
        ImageDraw.Draw(mask).text((-left, -top), title, font=font, fill=255)
        height = 440  # 20 pixels/mm; physical glyph height, not font em size.
        natural_width = round(mask.width * height / mask.height)
        width = min(1720, natural_width)  # 2 mm paper margin each side.
        mask = mask.resize((width, height), Image.Resampling.LANCZOS)
        image = Image.new("RGB", (1800, 800), "#FAFAF5")
        image.paste("#171B20", ((1800 - width) // 2, 180), mask)
        info = PngInfo()
        info.add(b"sRGB", b"\x00")
        image.save(folder / f"label_{name}_source.png", pnginfo=info)
        records[name] = {"text": title, "glyph_height_mm": 22,
                         "glyph_width_mm": width / 20,
                         "horizontal_scale": width / natural_width}
    (folder / "label_metrics.json").write_text(json.dumps(records, indent=2) + "\n")


if "--artwork" in sys.argv:
    artwork(Path(sys.argv[sys.argv.index("--artwork") + 1]))
    raise SystemExit(0)

import bmesh
import bpy
from mathutils import Vector

import prop_lib as props
from build_task01_tote_transfer import (
    DRAFT, WALL, bevel_edges, boolean, finish_mesh, loft, move_mesh,
    plastic_material, rounded_rectangle, rr_prism, rr_ring,
)
from build_task07_lidded_tote import grip_smudges, rounded_strip

FOLDER = props.ROOT / "assets" / "props" / "handle_tote"
QA = HERE / "_qa" / "task18"


def outer(z: float) -> tuple[float, float]:
    """Original tote shell inside its six-millimetre outward rim."""
    return 188 - 2 * (100 - z) * DRAFT, 288 - 2 * (100 - z) * DRAFT


def build_body(material: bpy.types.Material) -> bpy.types.Object:
    """Reuse task 01's rounded ring loft, fused ribs and underside waffle."""
    body = loft("Tapered shell with integral rim", [
        rr_ring(*outer(0), 4, 0), rr_ring(*outer(96), 4, 96),
        rr_ring(200, 300, 7, 96), rr_ring(200, 300, 7, 100),
        rr_ring(188 - 2 * WALL, 288 - 2 * WALL, 1, 100),
        rr_ring(outer(4)[0] - 2 * WALL, outer(4)[1] - 2 * WALL, 1, 4),
    ], material, bevel=0.65)
    for side in (-1, 1):
        # Space the front ribs outside the new 90 mm label; same count/section.
        for y in ((-105, -55, 55, 105) if side == -1 else (-105, -35, 35, 105)):
            rings = []
            for z in (10, 90):
                face = outer(z)[0] / 2
                a, b = sorted((side * (face - 0.5), side * (face + 5)))
                rings.append([(a, y - 1.5, z), (b, y - 1.5, z),
                              (b, y + 1.5, z), (a, y + 1.5, z)])
            boolean(body, loft("Long-wall rib", rings, material, bevel=0.5), "UNION")
        for x in (-60, 60):
            rings = []
            for z in (10, 90):
                face = outer(z)[1] / 2
                a, b = sorted((side * (face - 0.5), side * (face + 5)))
                rings.append([(x - 1.5, a, z), (x + 1.5, a, z),
                              (x + 1.5, b, z), (x - 1.5, b, z)])
            boolean(body, loft("End-wall rib", rings, material, bevel=0.5), "UNION")
        hole = rr_prism("80 x 20 cosmetic handle opening", 80, 20, 36, 8)
        move_mesh(hole, (0, side * 143, 58), (90, 0, 0))
        boolean(body, hole)
    for x in (-63, -21, 21, 63):
        for y in (-115, -69, -23, 23, 69, 115):
            boolean(body, rr_prism("3 mm underside waffle recess", 34, 38, 4, 2,
                                   location=(x, y, 1), bevel=0.3))
    for side in (-1, 1):
        # Bevel before union: the broad faces remain exactly vertical/14 mm apart.
        handle = props.box("Solid handle block", (120, 14, 30), (0, side * 143, 85),
                           bevel_mm=1, segments=3, material=material)
        boolean(body, handle, "UNION")
    # Finish after all Booleans: cutting an already twice-bevelled shell can
    # produce nonmanifold slivers and cause a subsequent handle union to lose faces.
    bevel_edges(body, 0.32, angle=45)
    return body


def add_card(body: bpy.types.Object) -> tuple[bpy.types.Object, bpy.types.ShaderNodeTexImage]:
    """One real thin card, following the -X wall draft, with independent print UV."""
    center = (-outer(50)[0] / 2 - 0.12, 0, 50)
    paper = props.pbr_material("White variant label", "#FAFAF5", 0.8, 0, grime=0.03)
    card = props.box("90 x 40 mm front label card", (0.30, 90, 40), center,
                     rotation_deg=(0, -2, 0), bevel_mm=0.08, material=paper)
    layer = card.data.uv_layers.new(name="LabelUV")
    angle = math.radians(2)
    for loop in card.data.loops:
        v = card.data.vertices[loop.vertex_index].co * 1000
        height = (v.z - center[2]) * math.cos(angle) - (v.x - center[0]) * math.sin(angle)
        layer.data[loop.index].uv = (0.5 - v.y / 90, 0.5 + height / 40)
    tree = paper.node_tree
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "LabelUV"
    image = tree.nodes.new("ShaderNodeTexImage")
    image.name = "Variant label artwork"
    image.extension = "EXTEND"
    tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
    rgb = next(node for node in tree.nodes if node.type == "RGB")
    for link in list(rgb.outputs[0].links):
        tree.links.new(image.outputs["Color"], link.to_socket)
    return props.join_objects([body, card], "body"), image


def collisions() -> list[bpy.types.Object]:
    """Eleven convex pieces, rounded at the shell/rim corners."""
    pieces = [props.extruded_profile("Floor collision",
              rounded_rectangle(*outer(4), 4, segments=4), 4, (0, 0, 2), bevel_mm=0)]
    for side in (-1, 1):
        for axis in ("X", "Y"):
            rings = []
            for z in (4, 100):
                ox, oy = (v / 2 for v in outer(z))
                inner = (ox if axis == "X" else oy) - WALL
                rings.append([(x, y, z) for x, y in rounded_strip(ox, oy, inner, 4, axis, side)])
            pieces.append(loft(f"Wall {axis}{side:+d} collision", rings))
    for side in (1, -1):
        pieces.append(props.box(f"Handle {side:+d} collision", (120, 14, 30),
                                (0, side * 143, 85), bevel_mm=0))
    # The wall tops cover the inner 3 mm. Four clipped, rounded outer strips
    # cover the flange, retaining the cavity and avoiding square R7 corners.
    for side in (-1, 1):
        for axis in ("X", "Y"):
            inner = (94 if axis == "X" else 144) - 0.2
            polygon = rounded_strip(100, 150, inner, 7, axis, side)
            pieces.append(props.extruded_profile(f"Rim {axis}{side:+d} collision", polygon,
                                                4, (0, 0, 98), bevel_mm=0))
    return pieces


def preview(body: bpy.types.Object, path: Path, texture: Path, reverse: bool = False) -> None:
    """Inspect the delivered baked material at 1024 px using the library lighting."""
    scene = bpy.data.scenes.new("Task18 export preview")
    scene.render.engine = "CYCLES"
    scene.cycles.device, scene.cycles.samples = "CPU", 64
    scene.cycles.use_adaptive_sampling = False
    scene.render.threads_mode, scene.render.threads = "FIXED", 6
    scene.render.resolution_x = scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform, scene.view_settings.look = "AgX", "None"
    world = bpy.data.worlds.new("Neutral grey task18 preview")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.18, 0.18, 0.18, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45
    visual = body.copy()
    visual.data = body.data.copy()
    scene.collection.objects.link(visual)
    mat = props.pbr_material("Delivered atlas preview", "#FFFFFF", 0.55, 0)
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(str(texture), check_existing=False)
    uv = mat.node_tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    mat.node_tree.links.new(uv.outputs["UV"], node.inputs["Vector"])
    mat.node_tree.links.new(node.outputs["Color"], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    visual.data.materials.clear()
    visual.data.materials.append(mat)
    for face in visual.data.polygons:
        face.material_index = 0
    center, span = Vector((0, 0, 0.05)), 0.3
    camera_data = bpy.data.cameras.new("Front-left camera")
    camera = bpy.data.objects.new("Front-left camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector((-1.8, -1.35 if reverse else 1.35, 1.25)) * span
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type, camera_data.ortho_scale = "ORTHO", span * 1.55
    scene.camera = camera
    for name, offset, power, size in (
        ("Key", (-1.5, 1.2, 2.1), 120, 1.4),
        ("Fill", (-0.8, -1.8, 0.9), 55, 1.6),
        ("Rim", (1.2, 0.7, 1.7), 85, 1.1),
    ):
        lamp = bpy.data.lights.new(name, "AREA")
        lamp.energy, lamp.shape, lamp.size = power * span * span, "DISK", size * span
        obj = bpy.data.objects.new(name, lamp)
        scene.collection.objects.link(obj)
        obj.location = center + Vector(offset) * span
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
    floor = props.box("Preview floor", (3000, 3000, 1), (0, 0, -1.4), bevel_mm=0,
                       material=props.pbr_material("Neutral grey floor", "#777777", 0.8, 0))
    for collection in list(floor.users_collection):
        collection.objects.unlink(floor)
    scene.collection.objects.link(floor)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.scenes.remove(scene)
    bpy.data.worlds.remove(world)


def finalize_isolated() -> None:
    """Scope the shared compactor's global scans to this prop, via a directory link."""
    original_root = props.ROOT
    with tempfile.TemporaryDirectory(prefix="finalize_", dir=QA) as temporary:
        root = Path(temporary)
        scoped = root / "assets" / "props"
        scoped.mkdir(parents=True)
        (scoped / "handle_tote").symlink_to(FOLDER, target_is_directory=True)
        try:
            props.ROOT = root
            props.finalize_assets()
        finally:
            props.ROOT = original_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=["handle_tote"], default=["handle_tote"])
    parser.add_argument("--bake-variants", nargs="+", choices=[v[0] for v in VARIANTS],
                        default=[v[0] for v in VARIANTS], help="Rebake only these colours; reuse completed others")
    parser.add_argument("--reuse-bakes", action="store_true",
                        help="Resume after interruption; geometry, UVs and artwork must be unchanged")
    parser.add_argument("--previews-only", action="store_true", help="Re-render exported atlases from source")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    FOLDER.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    if args.previews_only:
        bpy.ops.wm.open_mainfile(filepath=str(FOLDER / "source.blend"))
        body = bpy.data.objects["body"]
        for i, (name, _, _) in enumerate(VARIANTS):
            texture = FOLDER / ("body_albedo.png" if i == 0 else f"body_albedo_v{i}.png")
            preview(body, FOLDER / ("preview.png" if i == 0 else f"preview_{name}.png"), texture)
        preview(body, FOLDER / "preview_reverse.png", FOLDER / "body_albedo.png", reverse=True)
        props._DELIVERED_PROPS.add("handle_tote")
        finalize_isolated()
        return
    props.reset_scene()
    bpy.context.scene.render.threads_mode, bpy.context.scene.render.threads = "FIXED", 6
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(Path(__file__).resolve()),
                    "--artwork", str(FOLDER)], check=True)
    pp = plastic_material("Handle tote PP", "#1F5FA8", 0.55, tote=True)
    grip_smudges(pp, (0, -143, 85), radius=52)
    body, label_node = add_card(build_body(pp))
    finish_mesh(body, weighted_normals=True)
    mesh_check = bmesh.new()
    mesh_check.from_mesh(body.data)
    assert all(edge.is_manifold for edge in mesh_check.edges), "Visual must be closed before baking"
    mesh_check.free()
    props.uv_unwrap(body)
    # Preserve interior atlas margins under compact UV coordinate rounding.
    for loop in body.data.uv_layers["UVMap"].data:
        loop.uv = (0.0005 + 0.999 * loop.uv.x, 0.0005 + 0.999 * loop.uv.y)
    assert props.triangle_count([body]) <= 50_000
    rgb = next(node for node in pp.node_tree.nodes if node.type == "RGB")
    variants = []
    for i, (name, color, title) in enumerate(VARIANTS):
        rgb.outputs[0].default_value = props._linear_color(color)
        pp.diffuse_color = props._linear_color(color)
        label_node.image = bpy.data.images.load(str(FOLDER / f"label_{name}_source.png"), check_existing=True)
        texture = "body_albedo.png" if i == 0 else f"body_albedo_v{i}.png"
        preview_path = FOLDER / ("preview.png" if i == 0 else f"preview_{name}.png")
        if args.reuse_bakes or name not in args.bake_variants:
            if not (FOLDER / texture).is_file() or not preview_path.is_file():
                raise FileNotFoundError(f"Missing completed bake/preview for {name}")
        else:
            props.bake_albedo([body], 2048, FOLDER / texture)
            preview(body, preview_path, FOLDER / texture)
        variants.append({"name": name, "textures": {"body": texture},
                         "base_color": color, "label_text": title, "label_face": "-X"})
        print("TASK18_VARIANT", name, flush=True)
    preview(body, FOLDER / "preview_reverse.png", FOLDER / "body_albedo.png", reverse=True)
    rgb.outputs[0].default_value = props._linear_color(VARIANTS[0][1])
    pp.diffuse_color = props._linear_color(VARIANTS[0][1])
    label_node.image = bpy.data.images.load(str(FOLDER / "label_blue_source.png"), check_existing=True)
    pieces = collisions()
    files = props.export_collision("body", pieces, FOLDER)
    assert len(files) == 11
    for obj in pieces:
        obj.hide_render = True
        obj.hide_set(True)
        obj.display_type = "WIRE"
        obj["role"] = "convex_collision_proxy"
    props.export_obj(body, FOLDER / "body.obj")
    data = {
        "id": "handle_tote", "task": "line_replenish", "used_by": ["line_replenish", "tote_sorting"],
        "units": "m", "origin": "bottom_center", "static": False, "budget_class": "medium",
        "bbox_mm": [round(v, 6) for v in props.bbox_mm([body])], "triangles": props.triangle_count([body]),
        "source": "source.blend", "preview": "preview.png",
        "parts": [{"name": "body", "visual": "body.obj", "texture": "body_albedo.png", "texture_px": 2048,
                   "collision": files, "mass_kg": 0.45, "friction": 0.40,
                   "mujoco_material": {"specular": 0.25, "shininess": 0.30, "reflectance": 0.0},
                   "parent": None, "joint": None}],
        "variants": variants,
        "dimensions_mm": {"outside_xyz": [200, 300, 100], "wall_top_xy": [188, 288],
                          "wall_bottom_xy": list(outer(0)), "wall_normal_thickness": 3,
                          "draft_degrees": 2, "floor_nominal": 4, "underside_recess": 3,
                          "rim_outward_width": 6, "rim_thickness": 4,
                          "rib_width_projection": [3, 5], "ribs_long_side_each": 4, "ribs_short_side_each": 2,
                          "front_rib_y": [-105, -55, 55, 105],
                          "handle_centers": [[0, 143, 85], [0, -143, 85]],
                          "handle_length": 120, "handle_thickness": 14, "handle_height": 30,
                          "handle_xyz": [120, 14, 30], "handle_bevel": 1,
                          "handle_z_range": [70, 100], "handle_abs_y_faces": [136, 150],
                          "handle_opening_width_height_radius": [80, 20, 8], "handle_opening_z_range": [48, 68],
                          "label_width_height": [90, 40], "label_center_z": 50, "label_glyph_height": 22},
        "grasp": {"points_mm": [[0, 143, 85], [0, -143, 85]], "width_mm": 14},
        "collision_roles": dict(zip(files, ["floor", "wall_nx", "wall_ny", "wall_px", "wall_py",
                                             "handle_py", "handle_ny", "rim_nx", "rim_ny", "rim_px", "rim_py"], strict=True)),
        "notes": (
            "Task18 section 6.1: two solid 120 X x 14 Y x 30 Z handles at Y +/-143, Z85, "
            "flat vertical grip faces at |Y|136 and 150; tops flush at Z100; 1 mm three-segment bevels. "
            "Two real rounded 80 x 20 openings at Z48..68 are cosmetic; collision intentionally fills them. "
            "Original small_tote ring-loft geometry approach, ribs, 4 mm floor and 3 mm underside waffle retained. "
            "The exact 200 x 300 outer rim plus 6 mm outward flange implies 188 x 288 wall top and "
            "181.016 x 281.016 wall base; inherited approximate 193 x 293 base is incompatible. "
            "Front inner ribs move from Y +/-35 to +/-55 to clear the 90 x 40 white card. "
            "Card on -X follows the 2-degree draft; all variant glyphs are exactly 22 mm high. "
            "BIN-A07 and RECHECK use horizontally condensed bundled DejaVu Sans Bold to fit an 86 mm ink width. "
            "Eleven convex collision pieces preserve the cavity and flat handle faces; cosmetic openings, "
            "underside recesses, ribs and thin label are omitted from collision per the task decomposition. "
            "Handles use nominal box collisions (bevel error below 1 mm); shell/rim proxies follow rounded corners. "
            "Mass 0.45 kg without payload; simulation adds task payload. No buttons, lamps, conveyor or joints. "
            "No single fixed placement is prescribed for this movable prop; use the scene's cart/lane support transform. "
            "Metres, Z up, front -X, bottom-centre origin, transforms applied. Smart UV 66 degrees; normals smoothed "
            "at 30 degrees; baked CPU Cycles 64-sample diffuse color times (0.4 + 0.6 AO), no direct light/highlights. "
            "Source retains packed Principled PBR. Previews render the delivered atlas at 1024 square, "
            "three-point lighting and grey background; final compaction delivers 512-square previews. "
            f"Blender {bpy.app.version_string}."
        ),
    }
    props.write_prop_json(data, FOLDER / "prop.json")
    bpy.context.scene["prop_id"] = "handle_tote"
    bpy.context.scene["task18_manifest"] = json.dumps(data)
    props._select([body])
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(FOLDER / "source.blend"), check_existing=False)
    print("TASK18_COMPLETE", data["bbox_mm"], data["triangles"], len(files), flush=True)
    finalize_isolated()


if __name__ == "__main__":
    main()
