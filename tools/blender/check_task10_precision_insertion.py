"""Trimesh face/axis measurements and Task 10 checks, supplementary to validate_props.

Use the prescribed inovance-fineart Python. --prepare-collisions repairs only
convex face topology after Blender's float32 hull construction, preserving points.
The upper bore check intersects *every* collision piece with Z32.5 mm, then
measures the minimum distance between the axis and the actual triangle/plane
intersection segments. Measuring the unrestricted axis over Z0-40 would return
20 mm because of the deliberately smaller lower bore; that is a different bore.
"""

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import trimesh

from prop_schema import referenced_files
from validate_props import print_table, read_obj, validate_prop

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "assets" / "props"
QA = HERE / "_qa" / "task10"
IDS = ("bearing_housing_c010", "bearing_housing_c020", "bearing_housing_c030",
       "bearing_6204", "dowel_pin", "pin_fixture", "din_rail_kit")


def load(path):
    result = trimesh.load_mesh(path, process=True)
    assert isinstance(result, trimesh.Trimesh), path
    # OBJ import can split a vertex at every face normal / UV seam. Geometry
    # measurements and solidity checks must weld those coincident positions.
    result.merge_vertices(merge_tex=True, merge_norm=True)
    return result


def prepare(prop_id):
    folder = ROOT / prop_id
    data = json.loads((folder / "prop.json").read_text())
    references = {ref for ref in referenced_files(data) if "_col_" in ref}
    repairs = []
    for reference in sorted(references):
        path = folder / reference
        original = load(path)
        assert original.is_watertight and original.is_winding_consistent, path
        if original.is_convex:
            continue
        hull = original.convex_hull
        ratio = abs(hull.volume / original.volume)
        assert .99999 <= ratio <= 1.00001 and len(hull.vertices) <= 64, (path, ratio)
        old_points = {tuple(p) for p in original.vertices}
        assert all(tuple(p) in old_points for p in hull.vertices), path
        lines = ["# Trimesh convex topology; unchanged precision metre coordinates"]
        lines += ["v " + " ".join(f"{v:.12f}" for v in p) for p in hull.vertices]
        lines += ["vn " + " ".join(f"{v:.12f}" for v in normal) for normal in hull.face_normals]
        lines += ["f " + " ".join(f"{v + 1}//{i + 1}" for v in face)
                  for i, face in enumerate(hull.faces)]
        path.write_text("\n".join(lines) + "\n")
        assert load(path).is_convex, path
        repairs.append({"file": reference, "hull_to_original_volume": ratio})
    QA.mkdir(parents=True, exist_ok=True)
    (QA / f"{prop_id}_collision_topology.json").write_text(
        json.dumps({"checked": len(references), "repairs": repairs}, indent=2) + "\n")
    print(f"{prop_id}: {len(references)} collision files, {len(repairs)} topology repairs; coordinates unchanged")


def distance_to_segment(a, b, centre):
    a, b = a - centre, b - centre
    direction = b - a
    denominator = float(np.dot(direction, direction))
    t = np.clip(-float(np.dot(a, direction)) / denominator, 0, 1) if denominator > 1e-30 else 0
    return float(np.linalg.norm(a + t * direction))


def face_axis_distance(meshes, z_mm, centre_mm=(0, 0)):
    """Minimum axis-to-face distance in a horizontal slice, measured in mm.

    No vertex-radius shortcut: triangle edges are intersected with the slice,
    so this finds the flat's apothem, rather than the larger circumradius.
    """
    centre = np.asarray(centre_mm, dtype=float)
    minimum = math.inf
    for obj in meshes:
        triangles = np.asarray(obj.triangles) * 1000
        for triangle in triangles:
            if triangle[:, 2].min() > z_mm + 1e-7 or triangle[:, 2].max() < z_mm - 1e-7:
                continue
            points = []
            for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
                da, db = a[2] - z_mm, b[2] - z_mm
                if abs(da) < 1e-7:
                    points.append(a[:2])
                if da * db < 0:
                    points.append((a + (-da / (db - da)) * (b - a))[:2])
            for i, a in enumerate(points):
                for b in points[i + 1:]:
                    minimum = min(minimum, distance_to_segment(a, b, centre))
    assert math.isfinite(minimum), f"No faces intersect Z={z_mm} mm"
    return minimum


def near(actual, expected, tolerance=.01):
    assert np.allclose(actual, expected, rtol=0, atol=tolerance), (actual, expected, tolerance)


def fixture_variant_validation(data, folder):
    """Use the unmodified shared validator on each replacement-geometry manifest."""
    results = []
    for variant in data["variants"]:
        manifest = copy.deepcopy(data)
        manifest["parts"] = variant["parts"]
        manifest["triangles"] = variant["triangles"]
        manifest["clearance_mm"] = variant["clearance_mm"]
        manifest["bore_diameter_mm"] = variant["bore_diameter_mm"]
        manifest["variants"] = []
        destination = QA / "variants" / variant["name"] / "pin_fixture"
        destination.mkdir(parents=True, exist_ok=True)
        for reference in referenced_files(manifest):
            target = destination / reference
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            os.link(folder / reference, target)
        (destination / "prop.json").write_text(json.dumps(manifest, indent=2) + "\n")
        result = validate_prop(destination)
        result.prop_id = f"pin_fixture/{variant['name']}"
        results.append(result)
    return results


def run(prop_id):
    folder = ROOT / prop_id
    data = json.loads((folder / "prop.json").read_text())
    parts = data["parts"]
    meshes = [load(folder / filename) for part in parts for filename in part["collision"]]
    record = {"prop": prop_id, "status": "PASS"}
    if prop_id.startswith("bearing_housing_"):
        target = data["bore_diameter_mm"] / 2
        assert data["precision_mm"] == .01 and len(meshes) == 148
        near(data["bbox_mm"], [70, 110, 40], .0001)
        upper = face_axis_distance(meshes, 32.5)
        lower = face_axis_distance(meshes, 13)
        chamfer = face_axis_distance(meshes, 39.5)
        near(upper, target)
        near(lower, 20)
        near(chamfer, target + .5 * math.tan(math.radians(30)))
        # Each of the 48 specified pieces must independently have the right flat.
        individual = [face_axis_distance([m], 32.5) for m in meshes[:48]]
        near(individual, np.full(48, target))
        for index, (start, stop, low, high) in enumerate(((0, 48, 26, 39), (48, 96, 39, 40),
                                                       (96, 144, 0, 26))):
            band = meshes[start:stop]
            near([m.bounds[:, 2] * 1000 for m in band], np.tile([low, high], (48, 1)), .00001)
            for m in band:
                radii = np.linalg.norm(m.vertices[:, :2], axis=1) * 1000
                near(radii.max(), 31, .00001)
        visual = load(folder / parts[0]["visual"])
        near(face_axis_distance([visual], 32.5), target)
        near(face_axis_distance([visual], 13), 20)
        record.update(target_upper_radius_mm=target, measured_upper_radius_mm=upper,
                      error_mm=abs(upper - target), measured_lower_radius_mm=lower,
                      measured_entry_mid_radius_mm=chamfer,
                      worst_upper_segment_error_mm=float(np.max(np.abs(np.asarray(individual) - target))),
                      collision_pieces=len(meshes), measure_z_mm=32.5)
    elif prop_id == "bearing_6204":
        points = np.concatenate([m.vertices for m in meshes]) * 1000
        r = np.linalg.norm(points[:, :2], axis=1)
        assert r.max() <= 23.5 + 1e-9, r.max()
        near(r.max(), 23.5, .00001)
        at_nominal = points[np.abs(r - 23.5) < .00001]
        angles = np.mod(np.arctan2(at_nominal[:, 1], at_nominal[:, 0]), 2 * math.pi)
        lattice = angles * 48 / (2 * math.pi)
        near(lattice, np.round(lattice), .00001)
        assert len(np.unique(np.mod(np.round(lattice).astype(int), 48))) == 48
        near(parts[0]["mass_kg"], .106, 1e-12)
        # The exact split union must still have the unsplit prism's analytic volume.
        area_factor = 24 * math.sin(2 * math.pi / 48)
        expected = area_factor * (23.5 ** 2 * 12.8 + 2 * .6 / 3 * (22.9 ** 2 + 22.9 * 23.5 + 23.5 ** 2))
        near(sum(abs(m.volume) for m in meshes) * 1e9, expected, .01)
        assert all(len(m.vertices) <= 64 for m in meshes)
        record.update(max_collision_radius_mm=float(r.max()), exact_union_volume_mm3=expected,
                      collision_pieces=len(meshes), mass_kg=.106, circle_sides=48)
    elif prop_id == "dowel_pin":
        assert len(meshes) == 1 and len(meshes[0].vertices) == 64
        points = meshes[0].vertices * 1000
        radii = np.linalg.norm(points[:, :2], axis=1)
        assert radii.max() <= 5 + 1e-9
        near(radii, np.full(64, 5), .00001)
        near(np.ptp(points, axis=0), [10, 10, 40], .00001)
        near(parts[0]["mass_kg"], .025, 1e-12)
        record.update(max_collision_radius_mm=float(radii.max()), circle_sides=32,
                      collision_pieces=1, mass_kg=.025)
    elif prop_id == "pin_fixture":
        rows = []
        assert [v["clearance_mm"] for v in data["variants"]] == [.05, .10, .20]
        for variant in data["variants"]:
            collisions = [load(folder / f) for p in variant["parts"] for f in p["collision"]]
            assert len(collisions) == 64
            target = variant["bore_diameter_mm"] / 2
            measured = [face_axis_distance(collisions, 10, (0, y)) for y in (-30, 30)]
            near(measured, [target, target])
            for start, y in ((0, -30), (32, 30)):
                individual = [face_axis_distance([m], 10, (0, y)) for m in collisions[start:start + 32]]
                near(individual, np.full(32, target))
            visual = load(folder / variant["parts"][0]["visual"])
            near([face_axis_distance([visual], 10, (0, y)) for y in (-30, 30)], [target, target])
            rows.append({"variant": variant["name"], "target_radius_mm": target,
                         "measured_radii_mm": measured, "collision_pieces": 64})
        record["variants"] = rows
    else:
        assert data["geometry_only"] and not data["clip_on_behaviour_modelled"]
        assert not meshes and len(parts) == 11
        near(data["bbox_mm"], [47, 250, 48.5], .001)
        rail = load(folder / parts[0]["visual"])
        near(np.ptp(rail.vertices, axis=0) * 1000, [35, 250, 7.5], .001)
        for terminal in parts[1:9]:
            obj = load(folder / terminal["visual"])
            near(np.ptp(obj.vertices, axis=0) * 1000, [47, 5.2, 42.5], .002)
        record.update(terminal_count=8, collision_pieces=0, clip_on_behaviour_modelled=False)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    parser.add_argument("--prepare-collisions", action="store_true")
    args = parser.parse_args()
    QA.mkdir(parents=True, exist_ok=True)
    if args.prepare_collisions:
        for prop_id in args.props:
            prepare(prop_id)
        return 0
    records = [run(prop_id) for prop_id in args.props]
    variant_results = []
    if "pin_fixture" in args.props:
        folder = ROOT / "pin_fixture"
        variant_results = fixture_variant_validation(json.loads((folder / "prop.json").read_text()), folder)
        print_table(variant_results)
        assert not any(r.errors for r in variant_results), "Fixture variant validation failed"
    print("\nHousing upper bore: all collision piece faces intersected at Z32.5 mm")
    print("Prop                 | Target radius mm | Measured mm    | Error mm    | Status")
    print("---------------------+------------------+----------------+-------------+-------")
    for row in records:
        if "target_upper_radius_mm" in row:
            print(f"{row['prop']:20s} | {row['target_upper_radius_mm']:16.6f} | "
                  f"{row['measured_upper_radius_mm']:14.9f} | {row['error_mm']:11.9f} | PASS")
    (QA / "precision_checks.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"\n{len(records)}/{len(records)} semantic checks passed; housing tolerance 0.01 mm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
