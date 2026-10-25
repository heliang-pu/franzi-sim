"""Independent geometry/placement acceptance checks for the two task-2 props.

Run with the prescribed inovance-fineart Python. Also applies the shared
validator to all complete console variants, including blue texture choices.
Only task-2 files under tools/blender and assets/props are written.
"""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
from trimesh.proximity import closest_point_naive

from prop_schema import referenced_files
from validate_props import read_obj, validate_prop


ROOT = Path(__file__).resolve().parents[2] / "assets" / "props"
TOOLS = Path(__file__).resolve().parent
UV_CACHE = {}


def uv_overlap_pixels(obj, size=1024):
    """Rasterize strict triangle interiors; shared UV edges do not count."""
    triangles = obj.uv[obj.uv_faces]
    key = hashlib.sha256(np.round(triangles, 6).tobytes()).hexdigest()
    if key in UV_CACHE:
        return UV_CACHE[key]
    occupied = np.zeros((size, size), dtype=bool)
    overlap = np.zeros_like(occupied)
    def cross2(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    for triangle in triangles * size:
        a, b, c = triangle
        denominator = cross2(b - a, c - a)
        if abs(denominator) < 1e-12:
            continue
        lower = np.maximum(np.floor(triangle.min(0)).astype(int), 0)
        upper = np.minimum(np.ceil(triangle.max(0)).astype(int), size - 1)
        if (upper < lower).any():
            continue
        y, x = np.mgrid[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
        p = np.stack((x + 0.5, y + 0.5), axis=-1)
        u = cross2(b - p, c - p) / denominator
        v = cross2(c - p, a - p) / denominator
        w = 1 - u - v
        inside = (u > 1e-5) & (v > 1e-5) & (w > 1e-5)
        region = np.s_[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
        overlap[region] |= occupied[region] & inside
        occupied[region] |= inside
    result = int(overlap.sum())
    UV_CACHE[key] = result
    return result


def ray_hits(mesh, origin_mm, direction):
    origin = np.asarray(origin_mm, dtype=float) / 1000
    direction = np.asarray(direction, dtype=float)
    triangles = mesh.triangles
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, e2)
    determinant = np.einsum("ij,ij->i", e1, h)
    good = np.abs(determinant) > 1e-15
    inverse = np.zeros_like(determinant)
    inverse[good] = 1 / determinant[good]
    s = origin - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inverse * np.einsum("ij,j->i", q, direction)
    t = inverse * np.einsum("ij,ij->i", e2, q)
    good &= (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
    return np.unique(np.round(t[good] * 1000, 4))


def main():
    report = {}

    def check(group, name, passed, evidence):
        report.setdefault(group, {})[name] = {"pass": bool(passed), "evidence": evidence}

    library_folder = ROOT / "panel_devices"
    library = json.loads((library_folder / "prop.json").read_text())
    parts = {p["name"]: p for p in library["parts"]}
    expected = {"bezel", "btn_green", "btn_red", "btn_yellow", "selector_knob", "estop",
                "pilot_green", "pilot_amber", "pilot_red", "legend_plate"}
    check("panel_devices", "required_components", expected <= parts.keys(), sorted(parts))
    for name, part in parts.items():
        obj = read_obj(library_folder / part["visual"])
        model = obj.mesh()
        check("panel_devices", name + "_closed_and_oriented",
              model.is_watertight and model.is_winding_consistent,
              {"watertight": bool(model.is_watertight), "oriented": bool(model.is_winding_consistent)})
        overlap = uv_overlap_pixels(obj)
        check("panel_devices", name + "_nonoverlapping_uv_interiors", overlap == 0, overlap)
    dimensions = {"bezel": [29.5, 29.5, 3], "btn_green": [24, 24, 8], "btn_red": [24, 24, 8],
                  "btn_yellow": [24, 24, 8], "selector_knob": [27, 32, 20],
                  "pilot_green": [24, 24, 6], "pilot_amber": [24, 24, 6], "pilot_red": [24, 24, 6],
                  "legend_plate": [30, 10, 1]}
    for name, nominal in dimensions.items():
        model = read_obj(library_folder / parts[name]["visual"]).mesh()
        extent = np.ptp(model.vertices, axis=0) * 1000
        check("panel_devices", name + "_dimensions", np.allclose(extent, nominal, atol=0.025), extent.tolist())
    for name in ("btn_green", "btn_red", "btn_yellow"):
        part, joint = parts[name], parts[name]["joint"]
        check("panel_devices", name + "_return_and_travel",
              joint["type"] == "slide" and joint["axis"] == [0, 0, -1]
              and joint["range"] == [0, 0.004] and joint["spring"]["reference"] == 0
              and joint["spring"]["stiffness"] > 0 and part["mass_kg"] == 0.01 and part["friction"] == 0.8,
              joint)
    for name in ("btn_green", "btn_red", "btn_yellow", "estop"):
        model = read_obj(library_folder / parts[name]["visual"]).mesh()
        proxy = read_obj(library_folder / parts[name]["collision"][0])
        _, distances, _ = closest_point_naive(model, proxy.vertices)
        maximum = float(distances.max() * 1000)
        check("panel_devices", name + "_proxy_boundary_within_1_mm", maximum <= 1.01, maximum)
    joint = parts["selector_knob"]["joint"]
    check("panel_devices", "selector_endpoint_detents",
          joint["type"] == "hinge" and np.allclose(joint["range"], [-np.pi / 4, np.pi / 4])
          and joint["detents"]["positions_rad"] == joint["range"], joint)
    joint = parts["estop"]["joint"]
    check("panel_devices", "estop_latch_and_separate_fixed_disc",
          joint["type"] == "slide" and joint["axis"] == [0, 0, -1] and joint["range"] == [0, 0.005]
          and joint["latch"]["hold_at_m"] == 0.005 and parts["estop_plate"]["joint"] is None,
          joint)
    check("panel_devices", "all_pilot_emissions",
          all(parts["pilot_" + color]["emission"] == 0.8 for color in ("green", "amber", "red")), 0.8)
    legends = {v["text"] for v in library["variants"]}
    check("panel_devices", "six_legend_textures",
          legends == {"START", "STOP", "AUTO", "MAN", "CYCLE START", "FEED HOLD"}
          and len({v["textures"]["legend_plate"] for v in library["variants"]}) == 6, sorted(legends))

    folder = ROOT / "control_console"
    data = json.loads((folder / "prop.json").read_text())
    layouts = {v["name"]: v for v in data["variants"] if "parts" in v}
    expected_layouts = {
        "A": [("lamp_green", -60, 150), ("lamp_amber", 0, 150), ("lamp_red", 60, 150),
              ("mode", -90, 70), ("start", -20, 70), ("stop", 40, 70), ("emergency", 110, 110)],
        "B": [("lamp_green", 60, 150), ("lamp_amber", 0, 150), ("lamp_red", -60, 150),
              ("mode", 90, 70), ("start", 20, 70), ("stop", -40, 70), ("emergency", -110, 110)],
        "C": [("lamp_green", -60, 170), ("lamp_amber", 0, 170), ("lamp_red", 60, 170),
              ("mode", -110, 110), ("start", -50, 110), ("stop", 10, 110), ("emergency", 100, 110)],
    }
    n = np.array([-1, 0, 2]) / np.sqrt(5)
    for name, variant in layouts.items():
        group = "control_console_" + name
        actual = [(d["name"], d["u_mm"], d["v_mm"]) for d in variant["devices"]]
        check(group, "all_specified_device_centres", actual == expected_layouts[name], actual)
        part_map = {p["name"]: p for p in variant["parts"]}
        exported = {part: read_obj(folder / record["visual"]) for part, record in part_map.items()}
        models = {part: obj.mesh() for part, obj in exported.items()}
        for part, model in models.items():
            check(group, part + "_closed_and_oriented", model.is_watertight and model.is_winding_consistent,
                  {"watertight": bool(model.is_watertight), "oriented": bool(model.is_winding_consistent)})
            overlap = uv_overlap_pixels(exported[part])
            check(group, part + "_nonoverlapping_uv_interiors", overlap == 0, overlap)
        check(group, "six_shell_colliders", len(part_map["body"]["collision"]) == 6,
              len(part_map["body"]["collision"]))
        for index, reference in enumerate(part_map["body"]["collision"]):
            proxy = read_obj(folder / reference)
            _, distances, _ = closest_point_naive(models["body"], proxy.vertices)
            maximum = float(distances.max() * 1000)
            check(group, f"shell_proxy_{index}_boundary_within_1_mm", maximum <= 1.01, maximum)
        mass = sum(p["mass_kg"] for p in variant["parts"])
        check(group, "four_kg_complete_assembly", abs(mass - 4) < 1e-10, mass)
        for device in variant["devices"]:
            key, u, v = device["name"], device["u_mm"], device["v_mm"]
            surface = np.array([-106 + v * 2 / np.sqrt(5), u, 60 + v / np.sqrt(5)])
            position = (surface + 1.5 * n) / 1000
            rot = np.array(device["rotation_matrix"])
            check(group, key + "_3d_pose", np.allclose(position, device["position_m"], atol=1e-10)
                  and np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) and np.linalg.det(rot) > 0.999999,
                  device["position_m"])
            for part in ("body", "panel"):
                hits = ray_hits(models[part], surface, n)
                near = hits[(hits > -5) & (hits < 5)]
                check(group, key + "_real_hole_in_" + part, len(near) == 0, near.tolist())
            radii = []
            for angle in (0, np.pi / 64):
                direction = np.cos(angle) * np.array([0, -1, 0]) + np.sin(angle) * np.array([2, 0, 1]) / np.sqrt(5)
                hits = ray_hits(models["panel"], surface + 0.75 * n, direction)
                radii.extend(abs(hits[(abs(hits) > 0.01) & (abs(hits) < 15)]).tolist())
            check(group, key + "_22_3_mm_bore_size",
                  len(radii) == 4 and min(radii) >= 11.145 and max(radii) <= 11.17, radii)
            joint = part_map[key]["joint"]
            if joint:
                axis = n if joint["type"] == "hinge" else -n
                check(group, key + "_transformed_joint", np.allclose(joint["axis"], axis, atol=1e-7)
                      and np.allclose(joint["anchor_m"], position, atol=1e-10), joint)
                world = position + np.array(variant["recommended_world_translation_m"])
                check(group, key + "_robot_workspace",
                      0.3 <= world[0] <= 0.48 and 0.03 <= abs(world[1]) <= 0.45 and 0.8 <= world[2] <= 1.0,
                      world.tolist())
        # Probe away from all bores, bevels, labels and the rear connector.
        point = np.array([-106 + 40 * 2 / np.sqrt(5), 130, 60 + 40 / np.sqrt(5)])
        hits = ray_hits(models["body"], point, n)
        near = hits[(hits > -5) & (hits < 5)]
        check(group, "roof_is_1_5_mm_normal_thickness",
              len(near) == 2 and np.allclose(near, [-1.5, 0], atol=0.005), near.tolist())
        wall_hits = ray_hits(models["body"], (0, 50, 20), (1, 0, 0))
        check(group, "front_back_sheet_thickness",
              len(wall_hits) == 4 and np.allclose(np.diff(wall_hits)[[0, 2]], [1.5, 1.5], atol=0.005),
              wall_hits.tolist())
        corner_hits = ray_hits(models["body"], (0, 159, 20), (1, 0, 0))
        expected_corner = 97 + np.sqrt(3**2 - 2**2)
        check(group, "R3_plan_corners", len(corner_hits) == 2
              and np.allclose(corner_hits, [-6 - expected_corner, -6 + expected_corner], atol=0.04),
              corner_hits.tolist())
        # Exercise generic validation on every variant, not just JSON's default.
        qa = TOOLS / "_qa" / "task02" / name / "control_console"
        qa.mkdir(parents=True, exist_ok=True)
        assembled = copy.deepcopy(data)
        assembled.update(parts=variant["parts"], bbox_mm=variant["bbox_mm"],
                         triangles=variant["triangles"], preview=variant["preview"],
                         variants=variant["colour_variants"])
        assembled.pop("panel_frame", None)
        for reference in referenced_files(assembled):
            destination = qa / reference
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination.unlink()
            os.link(folder / reference, destination)
        (qa / "prop.json").write_text(json.dumps(assembled, indent=2) + "\n")
        result = validate_prop(qa)
        check(group, "shared_validator_all_meshes_and_colour_variants", not result.errors, result.errors)

    output = TOOLS / "task02_geometry_checks.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    failures = []
    for group, checks in report.items():
        passed = sum(item["pass"] for item in checks.values())
        print(f"{group}: {passed}/{len(checks)} acceptance checks passed")
        failures.extend((group, key, item["evidence"]) for key, item in checks.items() if not item["pass"])
    for failure in failures:
        print("FAIL", *failure)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
