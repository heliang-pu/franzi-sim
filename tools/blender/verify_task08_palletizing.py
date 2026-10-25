"""Task 08 acceptance, beyond the shared validator.

Supplied environment Python: this_file checks every variant, packing and scope.
Headless Blender: -P this_file -- --blender-sources checks retained geometry/PBR.
All reports and temporary variant manifests stay under tools/blender/.
"""

import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
QA = HERE / "_qa" / "task08"
IDS = ("mini_pallet", "scissor_lift_table", "pallet_carton_160x80x80",
       "pallet_carton_120x80x80", "pattern_sign")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(actual, expected, message, tolerance=.03):
    import numpy as np
    require(np.allclose(actual, expected, atol=tolerance, rtol=0),
            f"{message}: {actual} != {expected}")


def verify_plans(data):
    summary = []
    for plan in data["stacking_plans"]:
        numbers = []
        layers = plan["layers"]
        require(len(layers) == 2 and all(len(layer) == 4 for layer in layers), "four cartons per layer")
        for index, layer in enumerate(layers):
            for item in layer:
                x, y = item["center_xy_mm"]
                dx, dy = item["size_xy_mm"]
                require(abs(x) + dx / 2 <= 150 and abs(y) + dy / 2 <= 200, "no pallet overhang")
                long = 160 if item["sku"] == "SKU-A" else 120
                near([dx, dy], [80, long] if index == 0 else [long, 80], "layer orientation")
                numbers.append(item["number"])
            for a, first in enumerate(layer):
                for second in layer[a + 1:]:
                    centers = [abs(v - w) for v, w in zip(first["center_xy_mm"], second["center_xy_mm"])]
                    half_sizes = [(v + w) / 2 for v, w in zip(first["size_xy_mm"], second["size_xy_mm"])]
                    require(any(c >= h - 1e-6 for c, h in zip(centers, half_sizes)), "no carton overlap")
        require(sorted(numbers) == list(range(1, 9)), "unique placement numbers 1-8")
        require(sorted(item["sku"] for item in layers[0]) == sorted(item["sku"] for item in layers[1]),
                "same carton inventory on both layers")
        require(len({item["center_xy_mm"][0] for item in layers[0]}) == 2, "layer 1 has two columns")
        summary.append({"name": plan["name"], "cartons_per_layer": 4, "overhang_mm": 0,
                        "layer_2_carton_rotation_deg": 90, "numbered_boxes": numbers})
    require(len(summary) == 3, "three plans")
    return summary


def external_checks():
    import numpy as np
    from PIL import Image
    from validate_props import print_table, read_obj, validate_prop

    results = []
    checks = {}
    for prop_id in IDS:
        folder = ROOT / "assets" / "props" / prop_id
        data = json.loads((folder / "prop.json").read_text())
        require(data["task"] == "palletizing", "task classification")
        expected = {"mini_pallet": 6, "scissor_lift_table": 2, "pattern_sign": 1}.get(prop_id, 1)
        require(sum(len(p["collision"]) for p in data["parts"]) == expected, "specified collision count")
        checks[prop_id] = {"collision_count": expected}
        for variant in data["variants"]:
            variant_data = copy.deepcopy(data)
            for part in variant_data["parts"]:
                part.update(variant.get("part_overrides", {}).get(part["name"], {}))
                part["texture"] = variant.get("textures", {}).get(part["name"], part["texture"])
            points, count = [], 0
            for part in variant_data["parts"]:
                mesh = read_obj(folder / part["visual"])
                points.append(mesh.vertices)
                count += len(mesh.faces)
            variant_data["triangles"] = count
            variant_data["bbox_mm"] = (np.ptp(np.concatenate(points), axis=0) * 1000).tolist()
            if prop_id == "mini_pallet":
                require(len(variant_data["parts"][0]["collision"]) == 6, "HDPE must inherit all six proxies")
                near(variant_data["bbox_mm"], [300, 400, 80], "variant nominal envelope", .05)
            target = QA / "variants" / variant["name"] / prop_id
            target.mkdir(parents=True, exist_ok=True)
            # Hard links keep validation entirely under the allowed directory,
            # without copying large .blend files or writing through any link.
            for source in folder.iterdir():
                if source.is_file() and source.name != "prop.json":
                    link = target / source.name
                    if link.exists():
                        link.unlink()
                    os.link(source, link)
            (target / "prop.json").write_text(json.dumps(variant_data, indent=2) + "\n")
            result = validate_prop(target)
            result.prop_id = f"{prop_id}[{variant['name']}]"
            results.append(result)
        for part in data["parts"]:
            if part["texture"]:
                with Image.open(folder / part["texture"]) as image:
                    require(image.mode == "RGB", "opaque RGB albedo, no alpha-dependent geometry")
        if prop_id.startswith("pallet_carton"):
            expected_mass = .35 if "160" in prop_id else .25
            near([data["parts"][0]["mass_kg"], data["parts"][0]["friction"]], [expected_mass, .55], "carton dynamics", 1e-9)
            require(data["grasp_width_mm"] == 80 and expected_mass <= .5, "single-hand limits")
        if prop_id == "scissor_lift_table":
            require(data["adjustable_height_mm"] == [620, 760], "lift range")
            near(data["dimensions_mm"]["top_surface_z"] + 80, 800, "pallet top at robot work height")
        if prop_id == "pattern_sign":
            acrylic = next(p for p in data["parts"] if p["name"] == "acrylic")
            require(acrylic["texture"] is None and acrylic["texture_px"] is None, "untextured clear acrylic")
            near(acrylic["rgba"], [.85, .90, .95, .25], "acrylic rgba", 1e-9)
            checks[prop_id]["plans"] = verify_plans(data)
            hashes = [hashlib.sha256((folder / v["textures"]["plan_sheet"]).read_bytes()).hexdigest()
                      for v in data["variants"]]
            require(len(set(hashes)) == 3, "three different baked plan textures")
    print_table(results)
    require(all(not result.errors for result in results), "all variant validator rows pass")
    baseline = json.loads((HERE / "task08_existing_files.json").read_text())
    changed = [name for name, digest in baseline.items()
               if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    require(not changed, f"previous-session files changed: {changed}")
    checks["scope"] = {"existing_files_unchanged": len(baseline), "roller_conveyor_rebuilt": False}
    checks["mujoco_load"] = mujoco_checks()
    make_contact_sheet()
    (HERE / "task08_export_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"TASK08_EXTERNAL_PASS: {len(results)} variant rows; {len(baseline)} existing files unchanged")


def mujoco_checks():
    """Compile the actual OBJ/PNG payloads, without claiming a robot episode."""
    import xml.etree.ElementTree as ET
    import mujoco

    reports = []
    output = QA / "mujoco"
    output.mkdir(parents=True, exist_ok=True)
    for prop_id in IDS:
        folder = ROOT / "assets" / "props" / prop_id
        data = json.loads((folder / "prop.json").read_text())
        choices = [("default", {})] + [(v["name"], v) for v in data["variants"][1:]]
        for label, variant in choices:
            xml = ET.Element("mujoco", model=f"{prop_id}_{label}")
            ET.SubElement(xml, "compiler", angle="radian")
            asset = ET.SubElement(xml, "asset")
            world = ET.SubElement(xml, "worldbody")
            body = ET.SubElement(world, "body", name="prop")
            for base_part in data["parts"]:
                part = copy.deepcopy(base_part)
                part.update(variant.get("part_overrides", {}).get(part["name"], {}))
                part["texture"] = variant.get("textures", {}).get(part["name"], part["texture"])
                name = part["name"]
                ET.SubElement(asset, "mesh", name=f"visual_{name}", file=str(folder / part["visual"]))
                material = {"name": f"material_{name}", **{k: str(v) for k, v in part["mujoco_material"].items()}}
                if part["texture"]:
                    ET.SubElement(asset, "texture", name=f"texture_{name}", type="2d", file=str(folder / part["texture"]))
                    material["texture"] = f"texture_{name}"
                else:
                    material["rgba"] = " ".join(str(v) for v in part["rgba"])
                ET.SubElement(asset, "material", **material)
                ET.SubElement(body, "geom", name=f"visual_{name}", type="mesh", mesh=f"visual_{name}",
                              material=f"material_{name}", contype="0", conaffinity="0", group="2", mass="0")
                for index, filename in enumerate(part["collision"]):
                    collision_name = f"collision_{name}_{index}"
                    ET.SubElement(asset, "mesh", name=collision_name, file=str(folder / filename))
                    ET.SubElement(body, "geom", name=collision_name, type="mesh", mesh=collision_name,
                                  group="3", rgba="1 0 0 0", friction=f"{part['friction']} 0.005 0.0001")
            text = ET.tostring(xml, encoding="unicode")
            (output / f"{prop_id}_{label}.xml").write_text(text + "\n")
            model = mujoco.MjModel.from_xml_string(text)
            state = mujoco.MjData(model)
            mujoco.mj_forward(model, state)
            reports.append({"prop": prop_id, "variant": label, "meshes": model.nmesh,
                            "textures": model.ntex, "geoms": model.ngeom, "compiled": True})
    (HERE / "task08_mujoco_checks.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"TASK08_MUJOCO_LOAD_PASS: {len(reports)} configurations, MuJoCo {mujoco.__version__}")
    return reports


def make_contact_sheet():
    from PIL import Image, ImageDraw, ImageFont

    items = [("mini_pallet", "preview.png", "mini_pallet / pine"),
             ("mini_pallet", "preview_blue_hdpe.png", "mini_pallet / blue HDPE"),
             ("scissor_lift_table", "preview.png", "scissor_lift_table / 720 mm"),
             ("pallet_carton_160x80x80", "preview.png", "SKU-A / 160Y x 80X x 80Z"),
             ("pallet_carton_120x80x80", "preview.png", "SKU-B / 120Y x 80X x 80Z"),
             ("pattern_sign", "preview.png", "pattern_sign / plan 01"),
             ("pattern_sign", "preview_plan_02.png", "pattern_sign / plan 02"),
             ("pattern_sign", "preview_plan_03.png", "pattern_sign / plan 03")]
    sheet = Image.new("RGB", (1920, 1040), "#E6E8EB")
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default(size=22)
    draw = ImageDraw.Draw(sheet)
    for index, (prop_id, preview, label) in enumerate(items):
        x, y = index % 4 * 480, index // 4 * 520
        with Image.open(ROOT / "assets" / "props" / prop_id / preview) as image:
            sheet.paste(image.convert("RGB").resize((480, 480), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 16, y + 491), label, font=font, fill="#223143")
    sheet.save(HERE / "task08_previews.png")


def source_checks():
    import bmesh
    import bpy
    import numpy as np
    from mathutils import Vector

    def points(obj, group=None):
        vertices = obj.data.vertices
        if group:
            index = obj.vertex_groups[group].index
            vertices = [v for v in vertices if any(g.group == index and g.weight > .5 for g in v.groups)]
        return np.asarray([tuple(v.co) for v in vertices]) * 1000

    def bounds(obj, group=None):
        p = points(obj, group)
        return np.array([p.min(axis=0), p.max(axis=0)])

    def groups(obj, prefix):
        return [group.name for group in obj.vertex_groups if group.name.startswith(prefix)]

    report = {}
    for prop_id in IDS:
        folder = ROOT / "assets" / "props" / prop_id
        data = json.loads((folder / "prop.json").read_text())
        bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
        objects = {p["name"]: bpy.data.objects[p["source_visual_object"]] for p in data["parts"]}
        variants = [bpy.data.objects[v["part_overrides"]["body"]["source_visual_object"]]
                    for v in data["variants"] if "part_overrides" in v]
        all_objects = list(objects.values()) + variants
        for obj in all_objects:
            near(np.asarray(obj.matrix_world), np.eye(4), "applied transforms", 1e-7)
            require(all(len(p.vertices) == 3 for p in obj.data.polygons), "triangulated source mesh")
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            require(all(e.is_manifold for e in bm.edges), f"{prop_id}/{obj.name}: closed manifold components")
            require(all(f.calc_area() > 1e-16 for f in bm.faces), "nonzero source faces")
            bm.free()
            for slot in {p.material_index for p in obj.data.polygons}:
                require(obj.data.materials[slot].node_tree.nodes.get("Principled BSDF") is not None, "retained PBR")
        for part in data["parts"]:
            for name in part["source_collision_objects"]:
                proxy = bpy.data.objects[name]
                near(np.asarray(proxy.matrix_world), np.eye(4), "common collision frame", 1e-7)
                require(proxy.hide_render and proxy.hide_get(), "hidden collision source")
        require(bpy.context.scene.cycles.samples == 64 and bpy.context.scene.cycles.device == "CPU", "64 sample CPU bake")
        require(all(image.packed_file for image in bpy.data.images if image.source == "FILE"), "all image sources packed")
        summary = {"parts": len(data["parts"]), "applied_transforms": True, "manifold_components": True,
                   "pbr_packed": True, "cycles_samples": 64}
        if prop_id == "mini_pallet":
            wood = objects["body"]
            require(any(m and m.node_tree.nodes.get("Pine wavy growth rings") for m in wood.data.materials),
                    "retained visible wavy wood-grain shader")
            require(len(groups(wood, "Top board")) == 5, "five deckboards")
            require(len(groups(wood, "Pine block")) == 9, "nine blocks")
            require(len(groups(wood, "Bottom runner")) == 3, "three bottom runners")
            require(len(groups(wood, "Flush dark nail head")) == 30, "thirty nail heads")
            for group, yy in zip(groups(wood, "Top board"), (-168, -84, 0, 84, 168)):
                bb = bounds(wood, group)
                near(bb[1] - bb[0], [300, 64, 12], "deckboard dimensions")
                near(bb.mean(axis=0), [0, yy, 74], "deckboard positions and 20 mm gaps")
            for group in groups(wood, "Pine block"):
                bb = bounds(wood, group)
                near(bb[1] - bb[0], [60, 60, 56], "block dimensions")
                require(round(bb.mean(axis=0)[0]) in (-120, 0, 120), "block X grid")
                require(round(bb.mean(axis=0)[1]) in (-170, 0, 170), "block Y grid")
            for group, xx in zip(groups(wood, "Bottom runner"), (-118, 0, 118)):
                bb = bounds(wood, group)
                near(bb[1] - bb[0], [64, 400, 12], "bottom runner dimensions")
                near(bb.mean(axis=0), [xx, 0, 6], "bottom runner positions")
            hdpe = variants[0]
            near(np.ptp(points(hdpe), axis=0), [300, 400, 80], "HDPE dimensions")
            hole_count = 0
            for yy in (-168, -84, 0, 84, 168):
                for xx in (-120, -80, -40, 0, 40, 80, 120):
                    for dy in (-15, 15):
                        hit, p, _, _ = hdpe.ray_cast(Vector((xx / 1000, (yy + dy) / 1000, .10)), Vector((0, 0, -1)))
                        require(not hit or p.z < .0681, "real 12 mm through-grid hole")
                        hole_count += 1
            hit, p, _, _ = hdpe.ray_cast(Vector((.020, .015, .10)), Vector((0, 0, -1)))
            require(hit and abs(p.z - .08) < .0001, "solid HDPE grid rib")
            summary.update(top_boards=5, blocks=9, bottom_runners=3, nail_heads=30, hdpe_real_holes=hole_count)
        elif prop_id == "scissor_lift_table":
            obj = objects["body"]
            top_group = groups(obj, "500Y")[0]
            bb = bounds(obj, top_group)
            near(bb[1] - bb[0], [400, 500, 40], "lift top dimensions")
            near([bb[0, 2], bb[1, 2]], [680, 720], "lift top Z limits")
            bb = bounds(obj, groups(obj, "520Y")[0])
            near(bb[1] - bb[0], [420, 520, 60], "lift base dimensions")
            require(len(groups(obj, "Black scissor ascending")) == 2, "two ascending scissor arms")
            require(len(groups(obj, "Black scissor descending")) == 2, "two descending scissor arms")
            summary.update(scissor_pairs=2, top_z_mm=720, adjustable_height_mm=data["adjustable_height_mm"])
        elif prop_id.startswith("pallet_carton"):
            obj = objects["body"]
            nominal = [80, 160 if "160" in prop_id else 120, 80]
            near(np.ptp(points(obj), axis=0), nominal, "carton XYZ")
            for start, direction in [((-42, 0, 16), (1, 0, 0)), ((0, -nominal[1] / 2 - 2, 16), (0, 1, 0))]:
                direction = Vector(direction)
                hit, p, _, _ = obj.ray_cast(Vector(start) * .001, direction)
                require(hit, "outer carton wall")
                hit, q, _, _ = obj.ray_cast(p + direction * .00002, direction)
                require(hit, "inner carton wall")
                near((q - p).length * 1000, 4, "actual 4 mm walls")
            require(len(groups(obj, "Internal 4 mm minor flap")) == 4, "four inner minor flaps")
            require(len(groups(obj, "Actual wavy exposed flap medium")) == 4, "four actual flute patches")
            require(len(groups(obj, "Fine tape wrinkle")) == 4, "tape wrinkles")
            require(len(groups(obj, "Small trapped tape air bubble")) == 1, "tape bubble")
            require(obj.data.uv_layers.get("OrientationUV") is not None, "top arrow mapping")
            require(any(m and "orientation arrow" in m.name for m in obj.data.materials), "top arrow ink material")
            summary.update(sku=data["sku"], wall_mm=4, mass_kg=data["parts"][0]["mass_kg"], grip_mm=80)
        else:
            near(np.ptp(points(objects["acrylic"]), axis=0), [3, 210, 297], "A4 acrylic")
            near(bounds(objects["acrylic"])[0, 2], 10, "10 mm acrylic insertion")
            near(np.ptp(points(objects["base"]), axis=0), [60, 220, 20], "sign base")
            for material in objects["acrylic"].data.materials:
                require(not any(n.type == "TEX_IMAGE" for n in material.node_tree.nodes), "no acrylic image texture")
            packed_names = {Path(image.filepath).name for image in bpy.data.images
                            if image.source == "FILE" and image.packed_file}
            require({"plan_01.png", "plan_02.png", "plan_03.png"} <= packed_names,
                    "all three original plan artworks packed in source")
            summary["plans"] = verify_plans(data)
        report[prop_id] = summary
        print("TASK08_SOURCE_PASS", prop_id, json.dumps(summary), flush=True)
    (HERE / "task08_source_checks.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    if "--props" in sys.argv:
        IDS = tuple(sys.argv[sys.argv.index("--props") + 1:])
    if "--blender-sources" in sys.argv:
        source_checks()
    else:
        external_checks()
