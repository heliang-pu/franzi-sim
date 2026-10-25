"""Task 9 semantic checks supplementing the shared geometric validator.

Run with the prescribed inovance-fineart Python. --repair-collisions performs
only the established post-OBJ-rounding convex-topology repair, without moving
points or changing any other task's files. The default checks all four props.
"""

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

from validate_props import read_obj

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "assets" / "props"
QA = HERE / "_qa" / "task09"
IDS = ("cnc_front", "part_tray_6", "round_blank", "machined_part")


def repair_collisions(prop_id):
    folder = ROOT / prop_id
    data = json.loads((folder / "prop.json").read_text())
    repaired, checked = [], 0
    for part in data["parts"]:
        for reference in part["collision"]:
            checked += 1
            path = folder / reference
            original = read_obj(path).mesh()
            if original.is_convex:
                continue
            hull = original.convex_hull
            ratio = abs(hull.volume / original.volume)
            assert original.is_watertight and .998 <= ratio <= 1.002, (path, ratio)
            assert len(hull.vertices) <= 64 and hull.is_convex, path
            old_points = {tuple(np.rint(v * 1e6).astype(int)) for v in original.vertices}
            assert all(tuple(np.rint(v * 1e6).astype(int)) in old_points for v in hull.vertices)
            lines = ["# Convex topology from unchanged micrometre OBJ points; metres"]
            lines += ["v " + " ".join(f"{value:.6f}" for value in v) for v in hull.vertices]
            lines += ["vn " + " ".join(f"{value:.9f}" for value in n) for n in hull.face_normals]
            lines += ["f " + " ".join(f"{int(v) + 1}//{i + 1}" for v in face)
                      for i, face in enumerate(hull.faces)]
            path.write_text("\n".join(lines) + "\n")
            final = read_obj(path).mesh()
            assert final.is_convex and final.is_watertight, path
            repaired.append({"file": reference, "hull_volume_ratio": float(ratio)})
    report = {"prop": prop_id, "checked": checked, "repaired": repaired}
    (QA / f"{prop_id}_collision_quantization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{prop_id}: {checked} collision proxies checked; {len(repaired)} quantized topologies repaired")


def run_checks(prop_id):
    folder = ROOT / prop_id
    data = json.loads((folder / "prop.json").read_text())
    parts = {p["name"]: p for p in data["parts"]}
    checks = []

    def check(value, description):
        assert value, f"{prop_id}: {description}"
        checks.append(description)

    def close(a, b, description, atol=.0011):
        check(np.allclose(a, b, atol=atol, rtol=0), description)

    def bounds(part, collision=False):
        paths = parts[part]["collision"] if collision else [parts[part]["visual"]]
        points = np.concatenate([read_obj(folder / name).vertices for name in paths]) * 1000
        return points.min(axis=0), points.max(axis=0)

    def files(part):
        return [read_obj(folder / name).mesh() for name in parts[part]["collision"]]

    if prop_id == "cnc_front":
        close(data["bbox_mm"], [770, 900, 1600], "closed complete bbox includes proud handle")
        check(data["origin"] == "floor_front_face_center", "floor/front origin is explicit")
        close(bounds("body")[1], [700, 450, 1600], "enclosure rear/side/top dimensions")
        close(bounds("body")[0][1:], [-450, 0], "enclosure width and floor")
        joint = parts["door"]["joint"]
        close(joint["axis"], [0, 1, 0], "door slides +Y")
        close(joint["range"], [0, .46], "door 460 mm travel", atol=1e-9)
        check(joint["damping"] == 8 and joint["frictionloss"] == 3, "door damping/frictionloss")
        close(parts["door"]["mass_kg"] + parts["door_window"]["mass_kg"], 4,
              "complete moving door assembly is 4 kg", atol=1e-9)
        check(parts["door_window"]["parent"] == "door" and parts["door_window"]["joint"] is None,
              "window inherits door motion without a duplicate joint")
        check(parts["door_window"]["texture"] is None, "window uses geom alpha, no alpha texture")
        close(parts["door_window"]["rgba"], [.85, .90, .95, .25], "window RGBA")
        low, high = bounds("door_window")
        close(high - low, [5, 380, 300], "window is 380 x 300 x 5")
        door_proxies = files("door")
        check(len(door_proxies) == 5, "door four frame walls plus handle")
        for i, expected in enumerate(([-30, -250, 790], [-30, 190, 790],
                                      [-30, -190, 790], [-30, -190, 1160])):
            close(door_proxies[i].bounds[0] * 1000, expected, f"door frame wall {i} position")
        close(door_proxies[-1].bounds * 1000, [[-70, -230, 900], [-50, -210, 1080]],
              "handle diameter, height and 30 mm proud centreline")
        joint = parts["vise_jaw"]["joint"]
        close(joint["axis"], [1, 0, 0], "vise jaw slides +X")
        close(joint["range"], [0, .02], "vise jaw 20 mm travel", atol=1e-9)
        close(bounds("vise_jaw")[1][0], 60, "open moving jaw face X60")
        close(bounds("vise_jaw")[1][0] + joint["range"][1] * 1000, 80, "clamped jaw face X80")
        close(bounds("fixed_jaw")[0][0], 120, "fixed jaw face X120")
        close(bounds("vise_body")[0], [30, -45, 800], "vise body low bound")
        close(bounds("vise_body")[1], [170, 45, 850], "vise body high bound")
        close(bounds("table")[1][2], 800, "table top Z800")
        close(bounds("parallels")[1][2], 860, "parallels seat workpiece at Z860")
        for i, parallel in enumerate(files("parallels")):
            close(np.ptp(parallel.vertices, axis=0) * 1000, [8, 80, 10],
                  f"parallel {i + 1} runs parallel to jaw faces")
            check(parallel.bounds[0, 0] > .080 and parallel.bounds[1, 0] < .120,
                  f"parallel {i + 1} clears complete jaw stroke")
        close(bounds("spindle")[0][2], 1150, "end mill tip Z1150")
        close(bounds("panel"), [[-40, -440, 900], [0, -290, 1250]], "panel 40 mm projection and extents")
        close(bounds("screen")[1][1:] - bounds("screen")[0][1:], [120, 90], "screen 120 x 90")
        for name, z in (("cycle_start", 1.0), ("feed_hold", 1.0), ("estop", 1.18)):
            close(parts[name]["joint"]["axis"], [1, 0, 0], name + " press direction +X")
            close(parts[name]["mounting_position_m"][2], z, name + " specified height", atol=1e-9)
        close(parts["mode_selector"]["joint"]["axis"], [-1, 0, 0], "mode selector outward axis")
        check("detents" in parts["mode_selector"]["joint"] and "latch" in parts["estop"]["joint"],
              "selector detents and emergency latch retained")
        library = ROOT / "panel_devices"
        current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(library.iterdir()) if p.is_file()}
        check(current == data["panel_library_sha256"], "every Task 2 library file remains unchanged")
        lib_parts = {p["name"]: p for p in json.loads((library / "prop.json").read_text())["parts"]}
        for part in parts.values():
            if not part.get("library_instance"):
                continue
            template = lib_parts[part["library_instance"]["part"]]
            original = read_obj(library / template["visual"])
            placed = read_obj(folder / part["visual"])
            rotation = np.asarray(part["local_to_prop_rotation"])
            # Avoid the local Accelerate BLAS's spurious floating-point status
            # warnings on a tall Nx3 matrix with a signed-permutation rotation.
            expected = np.einsum("ij,kj->ik", original.vertices, rotation,
                                 optimize=False) + np.asarray(part["mounting_position_m"])
            close(placed.vertices, expected, part["name"] + " is a rigidly placed library mesh", atol=1.1e-6)
            check((library / template["texture"]).read_bytes() == (folder / part["texture"]).read_bytes(),
                  part["name"] + " atlas copied exactly from library")
        # Four front proxies leave precisely the specified unobstructed aperture.
        fronts = files("body")[:4]
        for y in (-239, 0, 239):
            for z in (801, 1000, 1219):
                check(not any(np.all([.00075, y / 1000, z / 1000] > m.bounds[0]) and
                              np.all([.00075, y / 1000, z / 1000] < m.bounds[1]) for m in fronts),
                      f"front opening clear at Y{y} Z{z}")
    elif prop_id == "part_tray_6":
        close(data["bbox_mm"], [170, 250, 30], "tray exact overall dimensions")
        proxies = files("body")
        check(len(proxies) == 56, "one floor plus 6 x 8 pocket walls and seven filler boxes")
        close(proxies[0].bounds[1][2] * 1000, 10, "blind pocket floor Z10")
        centres = [(x, y) for x in (-35, 35) for y in (-75, 0, 75)]
        for i, (x, y) in enumerate(centres):
            walls = proxies[1 + i * 8:9 + i * 8]
            distances = []
            for wall in walls:
                face_distances = np.einsum("ij,ij->i", wall.face_normals,
                                           wall.triangles_center - np.asarray([x / 1000, y / 1000, .02]))
                side = np.abs(wall.face_normals[:, 2]) < 1e-8
                # The pocket plane has the most negative signed distance.
                # Sector-join planes pass through the pole before rounding;
                # micrometre OBJ rounding can move those by a few micrometres.
                distances.append(-min(face_distances[side]) * 1000)
            close(distances, [22] * 8, f"pocket {i + 1} has eight exact apothem-22 walls", atol=.001)
            for wall in walls:
                close(wall.bounds[:, 2] * 1000, [10, 30], f"pocket {i + 1} wall depth")
            for filler in proxies[49:]:
                point = np.asarray([x / 1000, y / 1000, .02])
                check(not np.all((point > filler.bounds[0]) & (point < filler.bounds[1])),
                      f"pocket {i + 1} is not filled by a solid-material box")
        check({v["name"] for v in data["variants"]} == {"IN", "OUT"}, "IN and OUT variants exist")
        check((folder / "body_albedo.png").read_bytes() != (folder / "body_albedo_v1.png").read_bytes(),
              "IN and OUT atlases differ")
        check(data["static"] and parts["body"]["mass_kg"] == .9, "tray static with 0.9 kg mass")
    else:
        close(data["bbox_mm"], [40, 40, 60], "bar exact envelope")
        check(parts["body"]["friction"] == .35, "bar friction 0.35")
        check(parts["body"]["mass_kg"] == (.204 if prop_id == "round_blank" else .170), "specified bar mass")
        proxies = files("body")
        check(len(proxies) == (1 if prop_id == "round_blank" else 2), "specified cylinder proxy count")
        if prop_id == "round_blank":
            check(len(proxies[0].vertices) == 48, "blank collider is a 24-sided cylinder")
        else:
            close(proxies[0].bounds * 1000, [[-20, -20, 0], [20, 20, 40]], "D40 x 40 base proxy")
            close(proxies[1].bounds * 1000, [[-15, -15, 40], [15, 15, 60]], "D30 x 20 step proxy")
            vertices = read_obj(folder / parts["body"]["visual"]).vertices * 1000
            radius = np.hypot(vertices[:, 0], vertices[:, 1])
            check(np.any(np.isclose(radius, 6, atol=.001) & np.isclose(vertices[:, 2], 59, atol=.001)),
                  "actual D12 bore below 1 mm mouth chamfer")
            check(np.any((radius < .001) & np.isclose(vertices[:, 2], 45, atol=.001)), "blind bore depth 15 mm")
    for part in parts.values():
        obj = read_obj(folder / part["visual"])
        check(np.all([len(face) == 3 for face in obj.faces]), part["name"] + " triangulated")
    result = {"prop": prop_id, "passed": len(checks), "checks": checks}
    print(f"{prop_id}: {len(checks)} semantic checks PASS")
    return result


def mujoco_smoke(prop_id):
    """Compile delivered meshes/textures and evaluate initial/endpoint poses."""
    # MuJoCo's default diagnostic log, if needed, must remain in the allowed tree.
    os.chdir(QA)
    import mujoco

    folder = ROOT / prop_id
    data = json.loads((folder / "prop.json").read_text())
    root = ET.Element("mujoco", model="task09_" + prop_id)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 0")
    assets, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
    bodies, joint_names = {}, []
    for part in data["parts"]:
        parent = bodies[part["parent"]] if part["parent"] else world
        body = ET.SubElement(parent, "body", name=part["name"])
        bodies[part["name"]] = body
        joint = part["joint"]
        if joint:
            name = part["name"] + "_joint"
            joint_names.append(name)
            attrs = {"name": name, "type": joint["type"],
                     "axis": " ".join(map(str, joint["axis"])),
                     "pos": " ".join(map(str, joint["anchor_m"])),
                     "range": " ".join(map(str, joint["range"])),
                     "damping": str(joint["damping"]), "frictionloss": str(joint["frictionloss"])}
            if joint.get("spring"):
                attrs["stiffness"] = str(joint["spring"]["stiffness"])
                attrs["springref"] = str(joint["spring"]["reference"])
            ET.SubElement(body, "joint", **attrs)
        visual_name = part["name"] + "_visual"
        ET.SubElement(assets, "mesh", name=visual_name, file=str(folder / part["visual"]))
        attrs = {"mesh": visual_name, "type": "mesh", "contype": "0", "conaffinity": "0",
                 "mass": "0", "group": "2"}
        if part["texture"]:
            texture_name = part["name"] + "_texture"
            ET.SubElement(assets, "texture", name=texture_name, type="2d", file=str(folder / part["texture"]))
            ET.SubElement(assets, "material", name=part["name"] + "_mat", texture=texture_name,
                          **{k: str(v) for k, v in part["mujoco_material"].items()})
            attrs["material"] = part["name"] + "_mat"
        else:
            attrs["rgba"] = " ".join(map(str, part["rgba"]))
        ET.SubElement(body, "geom", **attrs)
        for i, reference in enumerate(part["collision"]):
            name = part["name"] + f"_collision_{i}"
            ET.SubElement(assets, "mesh", name=name, file=str(folder / reference))
            ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, group="3",
                          mass=str(part["mass_kg"] / len(part["collision"])),
                          friction=f"{part['friction']} 0.005 0.0001", rgba="0.3 0.5 0.8 0.1")
    path = QA / f"{prop_id}_load_smoke.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    assert np.isfinite(state.geom_xpos).all()
    if prop_id == "cnc_front":
        assert model.njnt == 6, model.njnt
        for name, q in (("door_joint", .46), ("vise_jaw_joint", .02)):
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            state.qpos[model.jnt_qposadr[index]] = q
        mujoco.mj_forward(model, state)
        assert np.isfinite(state.geom_xpos).all()
        door = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "door")
        window = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "door_window")
        assert np.allclose(state.xpos[door], [0, .46, 0])
        assert np.allclose(state.xpos[window], [0, .46, 0])
    mujoco.mj_step(model, state)
    assert np.isfinite(state.qpos).all() and np.isfinite(state.qvel).all()
    result = {"prop": prop_id, "mujoco_version": mujoco.__version__, "meshes": model.nmesh,
              "joints": model.njnt, "initial_and_endpoint_forward": "PASS", "one_step": "PASS"}
    print(f"{prop_id}: MuJoCo {mujoco.__version__}, {model.nmesh} meshes, "
          f"{model.njnt} joints, compile/forward/one-step PASS")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", choices=IDS, nargs="+", default=list(IDS))
    parser.add_argument("--repair-collisions", action="store_true")
    parser.add_argument("--mujoco", action="store_true")
    args = parser.parse_args()
    if args.repair_collisions:
        for prop_id in args.props:
            repair_collisions(prop_id)
    else:
        results = [run_checks(prop_id) for prop_id in args.props]
        (QA / "semantic_validation.json").write_text(json.dumps(results, indent=2) + "\n")
        if args.mujoco:
            smoke = [mujoco_smoke(prop_id) for prop_id in args.props]
            (QA / "mujoco_validation.json").write_text(json.dumps(smoke, indent=2) + "\n")


if __name__ == "__main__":
    main()
