"""Independent Task 06 checks of exported geometry, UVs and MuJoCo seating.

Run with /Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -B.
This is an asset integration smoke test, not a robot kitting success-rate test.
"""

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.dont_write_bytecode = True

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from validate_props import read_obj
from verify_task01_tote_transfer import convex_contains, section_hits
from verify_task05_exports import check_uv_overlap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "assets" / "props"
IDS = ("foam_kit_tray", "hex_blank", "flanged_bushing", "l_bracket", "spur_gear")
MASS = (.6, .149, .128, .120, .029)
REPORT = {}


def record(prop_id, name, passed, evidence):
    REPORT.setdefault(prop_id, {})[name] = {"pass": bool(passed), "evidence": evidence}


def load(prop_id):
    folder = ROOT / prop_id
    metadata = json.loads((folder / "prop.json").read_text())
    visual = read_obj(folder / "body.obj").mesh()
    collisions = [read_obj(folder / name).mesh() for name in metadata["parts"][0]["collision"]]
    return metadata, visual, collisions


def occupied(pieces, point):
    return any(convex_contains(piece, point) for piece in pieces)


def generic_checks():
    for prop_id, mass in zip(IDS, MASS, strict=True):
        metadata, visual, collisions = load(prop_id)
        record(prop_id, "visual_watertight", visual.is_watertight, bool(visual.is_watertight))
        record(prop_id, "visual_consistent_winding", visual.is_winding_consistent,
               bool(visual.is_winding_consistent))
        record(prop_id, "specified_mass", abs(metadata["parts"][0]["mass_kg"] - mass) < 1e-9, mass)
        friction = .8 if prop_id == "foam_kit_tray" else .3
        record(prop_id, "specified_friction", metadata["parts"][0]["friction"] == friction, friction)
        try:
            result = check_uv_overlap(ROOT / prop_id / "body.obj")
            record(prop_id, "nonoverlapping_export_uv", True, result)
        except AssertionError as exc:
            record(prop_id, "nonoverlapping_export_uv", False, str(exc))
        with Image.open(ROOT / prop_id / "body_albedo.png") as image:
            pixels = np.asarray(image)
        record(prop_id, "baked_albedo_has_surface_detail", len(np.unique(pixels.reshape(-1, 3), axis=0)) > 20,
               {"range": np.ptp(pixels.astype(float), axis=(0, 1)).tolist()})
        expected = 64 if prop_id == "foam_kit_tray" else (1 if prop_id == "hex_blank" else 2)
        record(prop_id, "specified_collision_piece_count", len(collisions) == expected, len(collisions))


def tray_checks():
    prop_id = "foam_kit_tray"
    metadata, visual, pieces = load(prop_id)
    for name, point, floor in (
        ("hex", (55.173, -89.827), 18), ("bushing", (55.173, 90.173), 13),
        ("l", (-73, -85.173), 18), ("gear", (-54.827, 90.173), 33),
        ("hex_notch", (34, -89.827), 18), ("bushing_notch", (27.5, 90.173), 13),
        ("l_notch", (-82.5, -89.827), 18), ("gear_notch", (-86.5, 90.173), 33),
    ):
        hits = section_hits(visual, (*point, 0), 2)
        record(prop_id, name + "_visual_floor", bool(hits) and abs(max(hits) - floor) < .025, hits)
        state = [occupied(pieces, (*point, floor + 1)), occupied(pieces, (*point, floor - .2))]
        record(prop_id, name + "_collision_open_with_floor", state == [False, True], state)
    for label, point, axis, center, expected in (
        ("bushing_counterbore_43", (55.173, 90, 40), 1, 90, [-21.5, 21.5]),
        ("bushing_body_33", (55.173, 90, 25), 1, 90, [-16.5, 16.5]),
        ("gear_seat_51", (-54.827, 90, 38), 1, 90, [-25.5, 25.5]),
        ("l_outer_63", (-73, -90, 30), 1, -90, [-31.5, 31.5]),
        ("l_outer_43", (-55, -118, 30), 0, -55, [-21.5, 21.5]),
        ("l_long_leg_7", (-55, -68, 30), 0, -55, [-21.5, -14.5]),
        ("l_short_leg_7", (-40, -90, 30), 1, -90, [-31.5, -24.5]),
    ):
        hits = [hit - center for hit in section_hits(visual, point, axis) if abs(hit - center) < 40]
        record(prop_id, label, len(hits) == 2 and np.allclose(hits, expected, atol=.035), hits)
    shoulder = section_hits(visual, (55.173, 109, 0), 2)
    record(prop_id, "bushing_counterbore_floor_z38", bool(shoulder) and abs(max(shoulder) - 38) < .025,
           shoulder)
    upper_open = not occupied(pieces, (55, 109, 40))
    lower_closed = occupied(pieces, (55, 109, 25))
    record(prop_id, "stepped_collision_does_not_close_counterbore", upper_open and lower_closed,
           [upper_open, lower_closed])
    hex_hits = section_hits(visual, (55, -89.827, 30), 0)
    record(prop_id, "hex_back_flat_plus15", any(abs(v - 70) < .025 for v in hex_hits), hex_hits)
    fill = [(0, 0, 40), (110, -170, 40), (-110, 170, 40), (0, 130, 40),
            (-129, 0, 47), (129, 0, 47), (0, -189, 47), (0, 189, 47)]
    state = [occupied(pieces, point) for point in fill]
    record(prop_id, "inter_pocket_foam_and_four_PP_walls_collide", all(state), state)
    world = metadata["recommended_placement_m"]["pocket_centers_world_xy"]
    record(prop_id, "all_four_placement_centers_in_robot_reach",
           all(.300 <= x <= .480 and .030 <= abs(y) <= .450 for x, y in world), world)
    record(prop_id, "all_round_steps_have_16_boundary_segments",
           all(metadata["collision_layout"][name]["boundary_segments"] == 16
               for name in ("bushing_body", "bushing_counterbore", "gear")), metadata["collision_layout"])


def part_checks():
    _, hex_mesh, _ = load("hex_blank")
    hits = section_hits(hex_mesh, (0, .173, 15), 0)
    record("hex_blank", "actual_across_flats_27", len(hits) == 2 and np.allclose(hits, [-13.5, 13.5], atol=.01),
           hits)
    end = section_hits(hex_mesh, (0, .173, .5), 0)
    expected = 13.5 - .5 * math.tan(math.radians(30))
    record("hex_blank", "end_chamfer_30_degrees", len(end) == 2 and np.allclose(end, [-expected, expected], atol=.03),
           end)
    _, bushing, _ = load("flanged_bushing")
    bore = section_hits(bushing, (0, 0, 0), 2)
    record("flanged_bushing", "actual_bore_is_through", not bore, bore)
    for z, outer, label in ((12, 15, "body_OD30_ID20"), (27.5, 20, "flange_OD40_ID20")):
        hits = section_hits(bushing, (0, 0, z), 0)
        record("flanged_bushing", label, len(hits) == 4 and np.allclose(hits, [-outer, -10, 10, outer], atol=.02), hits)
    _, bracket, _ = load("l_bracket")
    for y in (-12, 15):
        hits = section_hits(bracket, (-18, y, 20), 0)
        record("l_bracket", f"long_leg_slot_{y}_through", not hits, hits)
        width = [h for h in section_hits(bracket, (-18, y, 20), 2) if 10 < h < 30]
        record("l_bracket", f"long_leg_slot_{y}_width_6_5",
               len(width) == 2 and np.allclose(width, [16.75, 23.25], atol=.025), width)
    slots = [h for h in section_hits(bracket, (-18, 0, 20), 1) if -25 < h < 25]
    record("l_bracket", "two_slots_actual_length12", len(slots) == 4 and
           np.allclose(slots, [-18, -6, 9, 21], atol=.025), slots)
    short = section_hits(bracket, (7, -28, 20), 1)
    record("l_bracket", "short_leg_hole_through", not short, short)
    fillet = section_hits(bracket, (0, -24, 20), 0)
    expected = -12 - math.sqrt(4 ** 2 - 2 ** 2)
    record("l_bracket", "R4_inner_fillet", len(fillet) == 2 and abs(fillet[-1] - expected) < .035, fillet)
    metadata, gear, _ = load("spur_gear")
    vertices = gear.vertices * 1000
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    rb = 22.5 * math.cos(math.radians(20))
    selected = vertices[((np.abs(vertices[:, 2] - .3) < .002) |
                         (np.abs(vertices[:, 2] - 11.7) < .002)) & (radius > rb + .05) & (radius < 23.99)]
    r = np.linalg.norm(selected[:, :2], axis=1)
    angle = np.arctan2(selected[:, 1], selected[:, 0])
    tooth = np.round(angle / (2 * math.pi / 30))
    relative = np.abs(angle - tooth * 2 * math.pi / 30)
    t = np.sqrt((r / rb) ** 2 - 1)
    expected_angle = math.pi / 60 + math.tan(math.radians(20)) - math.radians(20) - t + np.arctan(t)
    error = np.abs(relative - expected_angle)
    record("spur_gear", "exported_flanks_satisfy_true_involute_equation",
           len(error) >= 240 and float(error.max()) < .00008,
           {"flank_vertices": len(error), "max_angular_error_rad": float(error.max())})
    tooth_ids = sorted(set((tooth.astype(int) % 30).tolist()))
    record("spur_gear", "30_distinct_involute_teeth", tooth_ids == list(range(30)), tooth_ids)
    record("spur_gear", "actual_tip_and_root_circles", abs(radius.max() - 24) < .002 and
           bool(np.any(abs(radius - 20.625) < .002)), [float(radius.max()), 20.625])
    hits = section_hits(gear, (0, 0, 0), 2)
    record("spur_gear", "bore_is_through_18mm_height", not hits, hits)
    bore = [v for v in section_hits(gear, (0, 0, 6), 1) if abs(v) < 8]
    record("spur_gear", "actual_bore_diameter8", len(bore) == 2 and np.allclose(bore, [-4, 4], atol=.02), bore)
    key = [v for v in section_hits(gear, (4.6, 0, 6), 1) if abs(v) < 5]
    record("spur_gear", "keyway_actual_width3", len(key) == 2 and np.allclose(key, [-1.5, 1.5], atol=.015), key)
    hub = section_hits(gear, (8.173, .173, 0), 2)
    face = section_hits(gear, (17.173, .173, 0), 2)
    record("spur_gear", "face12_and_one_sided_hub18", np.allclose(hub, [0, 18], atol=.025) and
           np.allclose(face, [0, 12], atol=.025), {"hub": hub, "face": face})
    steel = next(v for v in metadata["variants"] if v["name"] == "steel")
    record("spur_gear", "steel_variant_has_mass_and_material_override",
           steel["masses_kg"]["body"] == .161454 and steel["mujoco_materials"]["body"]["specular"] == .7,
           steel)
    with Image.open(ROOT / "spur_gear" / "body_albedo.png") as image:
        pom_pixels = np.asarray(image).astype(float)
    with Image.open(ROOT / "spur_gear" / "body_albedo_v1.png") as image:
        steel_pixels = np.asarray(image).astype(float)
    delta = float(np.abs(pom_pixels - steel_pixels).mean())
    record("spur_gear", "steel_texture_differs_from_POM", delta > 15, delta)


def integration_checks():
    mjcf = ET.Element("mujoco", model="task06_asset_seating_smoke")
    ET.SubElement(mjcf, "compiler", angle="radian", inertiafromgeom="true", inertiagrouprange="0 1")
    ET.SubElement(mjcf, "option", timestep=".001", gravity="0 0 -9.81", integrator="implicitfast")
    assets, world = ET.SubElement(mjcf, "asset"), ET.SubElement(mjcf, "worldbody")
    locations = {"foam_kit_tray": (0, 0, 0), "hex_blank": (.055, -.09, .0182),
                 "flanged_bushing": (.055, .09, .0132), "l_bracket": (-.055, -.09, .0182),
                 "spur_gear": (-.055, .09, .0332)}
    for prop_id in IDS:
        metadata, _, _ = load(prop_id)
        part = metadata["parts"][0]
        folder = ROOT / prop_id
        ET.SubElement(assets, "mesh", name=prop_id + "_visual", file=str(folder / "body.obj"))
        ET.SubElement(assets, "texture", name=prop_id + "_texture", type="2d",
                      file=str(folder / part["texture"]))
        ET.SubElement(assets, "material", name=prop_id + "_material", texture=prop_id + "_texture",
                      texuniform="false", **{key: str(value) for key, value in part["mujoco_material"].items()})
        body = ET.SubElement(world, "body", name=prop_id, pos=" ".join(map(str, locations[prop_id])))
        if prop_id != "foam_kit_tray":
            ET.SubElement(body, "freejoint", name=prop_id + "_free")
        ET.SubElement(body, "geom", name=prop_id + "_visual", type="mesh", mesh=prop_id + "_visual",
                      material=prop_id + "_material", contype="0", conaffinity="0", group="2", mass="0")
        for i, filename in enumerate(part["collision"]):
            name = prop_id + f"_col_{i:02d}"
            ET.SubElement(assets, "mesh", name=name, file=str(folder / filename))
            ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, group="0",
                          mass=str(part["mass_kg"] / len(part["collision"])),
                          friction=f"{part['friction']} .005 .0001", rgba=".5 .5 .5 0")
    path = HERE / "_qa" / "task06" / "seated_parts.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(mjcf)
    ET.ElementTree(mjcf).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_min = min((c.dist for c in data.contact), default=0)
    record("integration", "all_OBJ_and_albedos_compile_in_MuJoCo", model.nmesh == 76,
           {"meshes": model.nmesh, "textures": model.ntex, "bodies": model.nbody,
            "initial_min_contact_distance_m": initial_min})
    record("integration", "correct_seated_poses_do_not_penetrate", initial_min >= -.00005, initial_min)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    for prop_id, floor in (("hex_blank", .018), ("flanged_bushing", .013),
                           ("l_bracket", .018), ("spur_gear", .033)):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prop_id)
        position = data.xpos[index]
        expected = np.asarray(locations[prop_id])
        expected[2] = floor
        record("integration", prop_id + "_rests_on_correct_floor",
               bool(np.all(np.abs(position - expected) < .0003)),
               {"rest_position_m": position.tolist(), "target_m": expected.tolist()})
    inverted = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "flanged_bushing_free")
    address = model.jnt_qposadr[joint]
    inverted.qpos[address:address + 7] = [.055, .09, .043, 0, 1, 0, 0]
    mujoco.mj_forward(model, inverted)
    bush_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "flanged_bushing")
    tray_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foam_kit_tray")
    distances = [float(c.dist) for c in inverted.contact
                 if {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])} == {bush_id, tray_id}]
    record("integration", "flange_down_cannot_reach_bushing_floor",
           bool(distances) and min(distances) < -.001,
           {"contacts": len(distances), "max_penetration_m": -min(distances) if distances else 0})
    # Both colour variants must be loadable, with the steel mass applied.
    gear_body = world.find("body[@name='spur_gear']")
    for geom in gear_body.findall("geom"):
        if geom.get("group") == "0":
            geom.set("mass", str(.161454 / 2))
    assets.find("texture[@name='spur_gear_texture']").set("file", str(ROOT / "spur_gear" / "body_albedo_v1.png"))
    steel_model = mujoco.MjModel.from_xml_string(ET.tostring(mjcf, encoding="unicode"))
    gear_id = mujoco.mj_name2id(steel_model, mujoco.mjtObj.mjOBJ_BODY, "spur_gear")
    record("integration", "steel_variant_loads_with_correct_mass", abs(steel_model.body_mass[gear_id] - .161454) < 1e-9,
           float(steel_model.body_mass[gear_id]))


def contact_sheet():
    width, height = 512, 550
    sheet = Image.new("RGB", (width * 3, height * 2), "#EEEEEE")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    paths = [(name, ROOT / name / "preview.png") for name in IDS]
    steel_preview = ROOT / "spur_gear" / "preview_steel.png"
    if steel_preview.is_file():
        paths.append(("spur_gear / steel variant", steel_preview))
    for i, (label, path) in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert("RGB").resize((width, width), Image.Resampling.LANCZOS)
            sheet.paste(image, ((i % 3) * width, (i // 3) * height))
        draw.text(((i % 3) * width + 16, (i // 3) * height + 520), label, fill="#222222", font=font)
    sheet.save(HERE / "task06_previews.png")


def main():
    generic_checks()
    tray_checks()
    part_checks()
    try:
        integration_checks()
    except Exception as exc:
        record("integration", "MuJoCo_smoke", False, str(exc))
    contact_sheet()
    (HERE / "task06_geometry_checks.json").write_text(json.dumps(REPORT, indent=2) + "\n")
    failures = []
    for prop_id, checks in REPORT.items():
        print(f"{prop_id}: {sum(check['pass'] for check in checks.values())}/{len(checks)} checks passed")
        failures += [(prop_id, name, check["evidence"]) for name, check in checks.items() if not check["pass"]]
    for failure in failures:
        print("FAIL", *failure)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
