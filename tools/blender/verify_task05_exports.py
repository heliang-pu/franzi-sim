"""Load Task 05's actual OBJ/PNG exports in MuJoCo and smoke-test metadata.

Run with the specified inovance-fineart Python. This is an asset-load and
joint/mass integration check, not a robot grasp/carry benchmark evaluation.
Diagnostic MJCF and JSON stay under tools/blender/.
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.dont_write_bytecode = True
import mujoco
import numpy as np

from validate_props import read_obj

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IDS = ("carton_250x200x150", "carton_300x200x200", "carton_220x180x120", "roller_conveyor")


def signed_twice_area(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def overlap_area(first, second):
    """Convex clipping of two UV triangles; shared boundaries have zero area."""
    polygon = list(first)
    sign = 1 if signed_twice_area(*second) >= 0 else -1
    for index in range(3):
        a, b = second[index], second[(index + 1) % 3]
        if not polygon:
            return 0.0
        clipped = []
        previous = polygon[-1]
        previous_distance = sign * signed_twice_area(a, b, previous)
        for current in polygon:
            distance = sign * signed_twice_area(a, b, current)
            if (distance >= 0) != (previous_distance >= 0):
                fraction = previous_distance / (previous_distance - distance)
                clipped.append((previous[0] + fraction * (current[0] - previous[0]),
                                previous[1] + fraction * (current[1] - previous[1])))
            if distance >= 0:
                clipped.append(current)
            previous, previous_distance = current, distance
        polygon = clipped
    if len(polygon) < 3:
        return 0.0
    return abs(sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
                   - polygon[i][1] * polygon[(i + 1) % len(polygon)][0]
                   for i in range(len(polygon)))) / 2


def check_uv_overlap(path):
    """Grid broad phase plus exact triangle clipping, without optional GIS libraries."""
    obj = read_obj(path)
    triangles = obj.uv[obj.uv_faces]
    lo, hi = triangles.min(axis=1), triangles.max(axis=1)
    lower = np.floor(lo * 64).astype(int)
    upper = np.floor(hi * 64).astype(int)
    bins, checked, boundary_contacts = {}, 0, 0
    maximum_boundary_area = 0.0
    uv_area = 0.0
    for index, triangle in enumerate(triangles):
        triangle = [tuple(point) for point in triangle]
        area = abs(signed_twice_area(*triangle)) / 2
        uv_area += area
        if area <= 1e-12:
            continue
        keys = [(x, y) for x in range(lower[index, 0], upper[index, 0] + 1)
                for y in range(lower[index, 1], upper[index, 1] + 1)]
        candidates = set()
        for key in keys:
            candidates.update(bins.get(key, ()))
        for other in candidates:
            if np.any(np.minimum(hi[index], hi[other]) - np.maximum(lo[index], lo[other]) <= 1e-12):
                continue
            shared = set(obj.uv_faces[index]) & set(obj.uv_faces[other])
            if len(shared) == 2:
                a, b = [obj.uv[vertex] for vertex in shared]
                c = obj.uv[next(vertex for vertex in obj.uv_faces[index] if vertex not in shared)]
                d = obj.uv[next(vertex for vertex in obj.uv_faces[other] if vertex not in shared)]
                if signed_twice_area(a, b, c) * signed_twice_area(a, b, d) <= 0:
                    continue
            checked += 1
            overlap = overlap_area(triangle, triangles[other])
            # OBJ writes UVs to six decimal places. Adjacent projected edges
            # can cross by that rounding width; do not mistake a <0.004-pixel
            # boundary strip for overlapping islands. The negative control in
            # task05_uv_regression.log rejects the old stacked primitive UVs.
            def perimeter(points):
                return sum(math.dist(points[i], points[(i + 1) % 3]) for i in range(3))

            quantization_area = 2e-6 * min(perimeter(triangle), perimeter(triangles[other]))
            require(overlap <= max(1e-10, quantization_area),
                    f"{path.name}: overlapping UV faces {index}/{other}, area {overlap}")
            if overlap > 1e-10:
                boundary_contacts += 1
                maximum_boundary_area = max(maximum_boundary_area, overlap)
        for key in keys:
            bins.setdefault(key, []).append(index)
    return {"triangle_area": uv_area, "exact_pairs_checked": checked, "overlaps": 0,
            "obj_rounding_boundary_contacts": boundary_contacts,
            "maximum_boundary_area_px2": maximum_boundary_area * 2048 ** 2}


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def numbers(values):
    return " ".join(f"{value:.12g}" for value in values)


def make_mjcf(folder, data, mass=None, remove_stop=False):
    conveyor = data["id"] == "roller_conveyor"
    root = ET.Element("mujoco", model=data["id"])
    ET.SubElement(root, "compiler", angle="radian", autolimits="false", inertiafromgeom="true")
    ET.SubElement(root, "option", timestep="0.001", gravity="0 0 0" if conveyor else "0 0 -9.81")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    if not conveyor:
        ET.SubElement(world, "geom", name="support_table", type="plane", size="1 1 .01",
                      friction=".55 .005 .0001")
    bodies = {}
    parts = [p for p in data["parts"] if not (remove_stop and p["name"] == "end_stop")]
    for part in parts:
        name = part["name"]
        visual_name, texture_name, material_name = f"mesh_{name}", f"texture_{name}", f"material_{name}"
        ET.SubElement(asset, "mesh", name=visual_name, file=str(folder / part["visual"]))
        ET.SubElement(asset, "texture", name=texture_name, type="2d", file=str(folder / part["texture"]))
        ET.SubElement(asset, "material", name=material_name, texture=texture_name,
                      **{key: str(value) for key, value in part["mujoco_material"].items()})
        parent = bodies[part["parent"]] if part["parent"] else world
        body = ET.SubElement(parent, "body", name=name, pos="0 0 0" if conveyor else "0 0 .001")
        bodies[name] = body
        if not conveyor:
            ET.SubElement(body, "freejoint", name="carton_free")
        joint = part["joint"]
        if joint:
            attributes = {"name": f"joint_{name}", "type": joint["type"], "axis": numbers(joint["axis"]),
                          "pos": numbers(joint["anchor_m"]), "damping": str(joint["damping"]),
                          "frictionloss": str(joint["frictionloss"]),
                          "limited": str(joint.get("limited", True)).lower()}
            if joint.get("limited", True):
                attributes["range"] = numbers(joint["range"])
            ET.SubElement(body, "joint", **attributes)
        ET.SubElement(body, "geom", name=f"visual_{name}", type="mesh", mesh=visual_name,
                      material=material_name, mass="0", contype="0", conaffinity="0", group="2")
        for index, reference in enumerate(part["collision"]):
            col_name = f"collision_{name}_{index:02d}"
            ET.SubElement(asset, "mesh", name=col_name, file=str(folder / reference))
            part_mass = mass if mass is not None else part["mass_kg"]
            ET.SubElement(body, "geom", name=col_name, type="mesh", mesh=col_name,
                          mass=str(part_mass / len(part["collision"])),
                          friction=f"{part['friction']} .005 .0001", rgba="0 0 0 0", group="3")
    return ET.tostring(root, encoding="unicode")


def imported_visual_bbox(model, state, parts, subtract_z=0):
    vertices = []
    for part in parts:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"visual_{part['name']}")
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        local = model.mesh_vert[start:start + count].astype(np.float64)
        rotation = state.geom_xmat[geom].reshape(3, 3)
        require(np.isfinite(local).all() and np.isfinite(rotation).all(), "finite imported vertices/pose")
        world = np.einsum("ij,kj->ik", local, rotation) + state.geom_xpos[geom]
        vertices.append(world)
    vertices = np.concatenate(vertices)
    vertices[:, 2] -= subtract_z
    return np.array([vertices.min(axis=0), vertices.max(axis=0)]) * 1000


def verify(prop_id):
    folder = ROOT / "assets" / "props" / prop_id
    data = json.loads((folder / "prop.json").read_text())
    uv_checks = {}
    for part in data["parts"]:
        if part.get("atlas_copy_from"):
            # Source audit verifies the clones' UVs are identical to roller_00.
            continue
        uv_checks[part["name"]] = check_uv_overlap(folder / part["visual"])
    output = HERE / "_qa" / "task05"
    output.mkdir(parents=True, exist_ok=True)
    xml = make_mjcf(folder, data)
    xml_path = output / f"{prop_id}_smoke.xml"
    xml_path.write_text(xml + "\n")
    model = mujoco.MjModel.from_xml_string(xml)
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    conveyor = prop_id == "roller_conveyor"
    bbox = imported_visual_bbox(model, state, data["parts"], 0 if conveyor else .001)
    require(np.allclose(np.ptp(bbox, axis=0), data["bbox_mm"], atol=.025, rtol=0),
            "MuJoCo-imported common-frame visual bbox")
    if conveyor:
        require(model.njnt == 10 and model.nq == 10 and model.nv == 10, "ten independent hinges")
        require(not bool(model.jnt_limited.any()), "all ten hinges are unlimited")
        require(np.allclose(model.jnt_axis, np.tile([0, 1, 0], (10, 1))), "hinge axes are Y")
        require(np.allclose(model.dof_damping, .001), "hinge damping")
        require(np.allclose(model.dof_frictionloss, 0), "free hinge friction loss")
        visual_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"visual_roller_{i:02d}")
                        for i in range(10)]
        initial_centers = state.geom_xpos[visual_geoms].copy()
        state.qpos[:] = 8 * math.pi
        state.qvel[:] = 3
        mujoco.mj_forward(model, state)
        for _ in range(50):
            mujoco.mj_step(model, state)
        require(np.isfinite(state.qpos).all() and np.isfinite(state.qvel).all(), "finite dynamics")
        require(bool((state.qpos > 8 * math.pi).all()), "all rollers move beyond four full turns")
        require(bool(((state.qvel > 0) & (state.qvel < 3)).all()), "damping decreases angular speed")
        require(np.allclose(state.geom_xpos[visual_geoms], initial_centers, atol=2.5e-5, rtol=0),
                "rollers rotate around their own common-frame axes without translation")
        removed_xml = make_mjcf(folder, data, remove_stop=True)
        removed = mujoco.MjModel.from_xml_string(removed_xml)
        require(removed.njnt == 10 and removed.nbody == model.nbody - 1, "removable stop integration")
        (output / "roller_conveyor_stop_removed_smoke.xml").write_text(removed_xml + "\n")
        details = {"unlimited_hinges": 10, "stepped_ms": 50, "start_angle_rad": 8 * math.pi,
                   "final_angle_range_rad": [float(state.qpos.min()), float(state.qpos.max())],
                   "final_speed_range_rad_s": [float(state.qvel.min()), float(state.qvel.max())],
                   "stop_removed_load": "PASS"}
    else:
        checked_masses = []
        for variant in data["variants"]:
            mass = variant["part_masses_kg"]["body"]
            variant_model = mujoco.MjModel.from_xml_string(make_mjcf(folder, data, mass=mass))
            body = mujoco.mj_name2id(variant_model, mujoco.mjtObj.mjOBJ_BODY, "body")
            require(abs(variant_model.body_mass[body] - mass) < 1e-9, "mass variant in simulator")
            checked_masses.append(mass)
        for _ in range(100):
            mujoco.mj_step(model, state)
        require(np.isfinite(state.qpos).all(), "finite free-carton dynamics")
        require(-.002 < state.qpos[2] < .002, "carton settles on its nominal box collider")
        details = {"mass_variants_loaded_kg": checked_masses, "stepped_ms": 100,
                   "settled_root_z_m": float(state.qpos[2])}
    return {"id": prop_id, "status": "PASS", "mujoco_version": mujoco.__version__,
            "uv_atlases": uv_checks,
            "mjcf": str(xml_path.relative_to(ROOT)),
            "imported_bbox_mm": np.ptp(bbox, axis=0).tolist(), **details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    args = parser.parse_args()
    report = []
    for prop_id in args.props:
        try:
            result = verify(prop_id)
        except Exception as exc:
            result = {"id": prop_id, "status": "FAIL", "error": str(exc)}
        print("TASK05_MUJOCO_EXPORT_AUDIT", json.dumps(result), flush=True)
        report.append(result)
    (HERE / "task05_export_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(result["status"] == "PASS" for result in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
