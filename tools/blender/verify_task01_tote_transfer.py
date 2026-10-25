"""Task-1 geometry acceptance probes beyond validate_props.py.

Run with the prescribed inovance-fineart Python. No optional ray/rtree backend
is needed: vertical and horizontal section intersections are numpy-only.
"""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

from validate_props import read_obj

ROOT = Path(__file__).resolve().parents[2] / "assets" / "props"


def section_hits(mesh, origin_mm, axis):
    """Intersect an infinite axis line, returning unique hit coordinates in mm."""
    origin = np.asarray(origin_mm, dtype=float) / 1000
    direction = np.eye(3)[axis]
    triangles = mesh.triangles
    edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(determinant) > 1e-15
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1 / determinant[valid]
    s = origin - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * np.einsum("ij,j->i", q, direction)
    t = inverse * np.einsum("ij,ij->i", edge2, q)
    valid &= (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
    return np.unique(np.round((origin[axis] + t[valid]) * 1000, 5)).tolist()


def convex_contains(mesh, point_mm):
    point = np.asarray(point_mm) / 1000
    return bool(np.all(np.einsum("ij,ij->i", point - mesh.triangles_center, mesh.face_normals) <= 1e-7))


def main():
    report = {}
    for prop_id in ("small_tote", "flange_part", "abs_housing"):
        folder = ROOT / prop_id
        data = json.loads((folder / "prop.json").read_text())
        visual = read_obj(folder / "body.obj").mesh()
        collision = [read_obj(folder / name).mesh() for name in data["parts"][0]["collision"]]
        checks = {}

        def check(name, condition, evidence):
            checks[name] = {"pass": bool(condition), "evidence": evidence}

        check("visual_watertight", visual.is_watertight, bool(visual.is_watertight))
        check("visual_winding_consistent", visual.is_winding_consistent, bool(visual.is_winding_consistent))
        expected_mass = {"small_tote": 0.45, "flange_part": 0.09, "abs_housing": 0.04}[prop_id]
        check("mass", abs(data["parts"][0]["mass_kg"] - expected_mass) < 1e-9,
              data["parts"][0]["mass_kg"])
        if prop_id == "small_tote":
            check("nine_separate_collision_pieces", len(collision) == 9, len(collision))
            hits = section_hits(visual, (0.173, 0, 30), 2)
            check("open_top_and_four_mm_floor", len(hits) == 2 and abs(hits[-1] - 4) < 0.03, hits)
            hits = section_hits(visual, (0.173, 0, 72.5), 1)
            check("handle_is_a_real_hole_on_positive_y", len(hits) >= 2 and max(hits) < 0, hits)
            cavity = any(convex_contains(piece, (0, 0, 30)) for piece in collision)
            check("collision_cavity_remains_open", not cavity, {"cavity_occupied": cavity})
            floor = any(convex_contains(piece, (0, 0, 2)) for piece in collision)
            check("collision_floor_present", floor, {"floor_occupied": floor})
            walls = [any(convex_contains(piece, p) for piece in collision)
                     for p in ((91.0, 0, 50), (-91.0, 0, 50), (0, 141.0, 50), (0, -141.0, 50))]
            check("all_four_collision_walls_present", all(walls), walls)
        elif prop_id == "flange_part":
            for name, xy in (("centre_bore", (0, 0)), ("bolt_0", (22.5, 0)),
                             ("bolt_1", (0, 22.5)), ("bolt_2", (-22.5, 0)), ("bolt_3", (0, -22.5))):
                hits = section_hits(visual, (*xy, 0), 2)
                check(name + "_through", not hits, hits)
            hits = section_hits(visual, (12, 0.173, 0), 2)
            check("hub_total_height_20", len(hits) == 2 and abs(hits[-1] - 20) < 0.02, hits)
            hits = section_hits(visual, (26, 8.173, 0), 2)
            check("disc_thickness_12", len(hits) == 2 and abs(hits[-1] - 12) < 0.02, hits)
            check("two_cylinder_colliders", len(collision) == 2 and all(len(m.vertices) == 64 for m in collision),
                  [len(m.vertices) for m in collision])
        else:
            for i, (x, y) in enumerate(((-15.5, -28), (-15.5, 28), (15.5, -28), (15.5, 28))):
                hits = section_hits(visual, (x + 0.083, y + 0.047, 0), 2)
                check(f"screw_bore_{i}", len(hits) == 2 and 21 < hits[-1] < 22.1, hits)
            hits = section_hits(visual, (0, 0.173, 10.5), 0)
            check("opposed_tabs_extend_to_48_mm", len(hits) == 2 and abs(hits[-1] - hits[0] - 48) < 0.03, hits)
            hits = section_hits(visual, (0, 0.173, 28), 0)
            check("parting_groove_below_30_mm_top", len(hits) == 2 and 44.4 < hits[-1] - hits[0] < 44.6, hits)
            check("one_nominal_box_collider", len(collision) == 1 and len(collision[0].vertices) == 8,
                  [len(m.vertices) for m in collision])
        report[prop_id] = checks
    path = Path(__file__).with_name("task01_geometry_checks.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    failures = [(prop_id, name, item["evidence"]) for prop_id, checks in report.items()
                for name, item in checks.items() if not item["pass"]]
    for prop_id, checks in report.items():
        print(f"{prop_id}: {sum(item['pass'] for item in checks.values())}/{len(checks)} geometry checks passed")
    for failure in failures:
        print("FAIL", *failure)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
