"""Task-3 acceptance probes on the actual exported OBJ geometry.

Run with the prescribed inovance-fineart Python after validate_props.py.
Independent section measurements check holes, thicknesses and T-slot undercuts;
metadata alone is not evidence of those features.
"""

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image

from validate_props import read_obj
from verify_task01_tote_transfer import convex_contains, section_hits

ROOT = Path(__file__).resolve().parents[2] / "assets" / "props"


def main():
    report = {}
    for prop_id in ("esd_tray", "tray_rail_set"):
        folder = ROOT / prop_id
        data = json.loads((folder / "prop.json").read_text())
        parts = {part["name"]: part for part in data["parts"]}
        visuals = {name: read_obj(folder / part["visual"]).mesh() for name, part in parts.items()}
        collisions = {name: [read_obj(folder / path).mesh() for path in part["collision"]]
                      for name, part in parts.items()}
        checks = {}

        def check(name, condition, evidence):
            checks[name] = {"pass": bool(condition), "evidence": evidence}

        for name, mesh in visuals.items():
            check(name + "_visual_watertight", mesh.is_watertight, bool(mesh.is_watertight))
            check(name + "_visual_winding", mesh.is_winding_consistent, bool(mesh.is_winding_consistent))
            with Image.open(folder / parts[name]["texture"]) as image:
                pixels = np.asarray(image)
            yellow = (pixels[:, :, 0] > 100) & (pixels[:, :, 1] > 65) & (pixels[:, :, 2] < 85)
            if name in ("body", "stop"):
                check(name + "_yellow_marking_baked", int(yellow.sum()) > 100, int(yellow.sum()))

        all_vertices = np.concatenate([mesh.vertices for mesh in visuals.values()]) * 1000
        expected = [200, 300, 50] if prop_id == "esd_tray" else [246, 700, 50]
        measured = np.ptp(all_vertices, axis=0)
        check("nominal_bbox", np.allclose(measured, expected, atol=.01), measured.tolist())
        origin = [(all_vertices[:, 0].min() + all_vertices[:, 0].max()) / 2,
                  (all_vertices[:, 1].min() + all_vertices[:, 1].max()) / 2, all_vertices[:, 2].min()]
        check("shared_bottom_centre_origin", np.allclose(origin, 0, atol=.01), origin)
        check("exactly_five_collision_pieces", sum(map(len, collisions.values())) == 5,
              {name: len(pieces) for name, pieces in collisions.items()})

        if prop_id == "esd_tray":
            mesh, pieces = visuals["body"], collisions["body"]
            check("mass_035_kg", parts["body"]["mass_kg"] == .35, parts["body"]["mass_kg"])
            check("friction_025", parts["body"]["friction"] == .25, parts["body"]["friction"])
            hits = section_hits(mesh, (.173, .281, 0), 2)
            check("three_mm_floor_and_open_top", len(hits) == 2 and np.allclose(hits, [0, 3], atol=.01), hits)
            hits = section_hits(mesh, (.173, .281, 12), 0)
            check("three_mm_long_walls", len(hits) == 4 and np.allclose(hits, [-100, -97, 97, 100], atol=.01), hits)
            hits = section_hits(mesh, (60.173, .281, 32), 1)
            check("three_mm_short_walls", len(hits) == 4 and np.allclose(hits, [-150, -147, 147, 150], atol=.01), hits)
            hits = section_hits(mesh, (.173, 0, 32), 1)
            check("both_short_wall_handles_are_through_holes", not hits, hits)
            for side in (-1, 1):
                hits = section_hits(mesh, (0, side * 148.5, 32), 0)
                inner_hits = [value for value in hits if abs(value) < 40]
                check(f"handle_{side}_width_70", len(inner_hits) == 2 and
                      np.allclose(inner_hits, [-35, 35], atol=.02), inner_hits)
                hits = section_hits(mesh, (0, side * 148.5, 32), 2)
                check(f"handle_{side}_height_22", any(abs(v - 21) < .02 for v in hits) and
                      any(abs(v - 43) < .02 for v in hits), hits)
            hits = section_hits(mesh, (98, 0, 12), 1)
            r8 = 142 + math.sqrt(8 ** 2 - 6 ** 2)
            check("outer_corner_R8", len(hits) == 2 and np.allclose(hits, [-r8, r8], atol=.04), hits)
            hits = section_hits(mesh, (96, 0, 12), 1)
            check("inner_corner_R5_for_3mm_wall", len(hits) == 4 and
                  np.allclose(hits[1:3], [-145, 145], atol=.04), hits)
            check("collision_cavity_is_open", not any(convex_contains(p, (0, 0, 25)) for p in pieces),
                  [convex_contains(p, (0, 0, 25)) for p in pieces])
            points = [(0, 0, 1.5), (-98.5, 0, 15), (98.5, 0, 15), (0, -148.5, 15), (0, 148.5, 15)]
            occupied = [any(convex_contains(p, point) for p in pieces) for point in points]
            check("floor_and_all_four_walls_collide", all(occupied), occupied)
            check("rounded_floor_proxy_under_64_vertices", len(pieces[0].vertices) <= 64,
                  len(pieces[0].vertices))
        else:
            frame, stop = visuals["frame"], visuals["stop"]
            frame_cols = collisions["frame"]
            rails = sorted([mesh for mesh in frame_cols if mesh.extents[1] > .6],
                           key=lambda mesh: mesh.bounds[0, 0])
            bars = sorted([mesh for mesh in frame_cols if mesh.extents[0] > .2],
                          key=lambda mesh: mesh.bounds[0, 1])
            gap = (rails[1].bounds[0, 0] - rails[0].bounds[1, 0]) * 1000 if len(rails) == 2 else -1
            check("two_700x20x20_rail_proxies", len(rails) == 2 and
                  all(np.allclose(p.extents * 1000, [20, 700, 20], atol=.01) for p in rails),
                  [p.extents.tolist() for p in rails])
            check("206_mm_clear_inside_gap", abs(gap - 206) < .01, gap)
            hits = section_hits(frame, (0, .173, 37.5), 0)
            left, right = [v for v in hits if v < 0], [v for v in hits if v > 0]
            real_gap = min(right) - max(left) if left and right else -1
            check("206_mm_gap_in_visual_mesh", abs(real_gap - 206) < .01, real_gap)
            check("two_246_mm_cross_bars", len(bars) == 2 and
                  all(np.allclose(p.extents * 1000, [246, 20, 20], atol=.01) for p in bars),
                  [p.extents.tolist() for p in bars])
            centres = [p.centroid[1] * 1000 for p in bars]
            check("cross_bars_50_mm_inboard", np.allclose(centres, [-300, 300], atol=.01), centres)
            for side in (-1, 1):
                hits = section_hits(frame, (0, side * 300 + 7, 16), 0)
                check(f"cross_bar_{side}_visual_length_246", len(hits) == 2 and
                      np.allclose(hits, [-123, 123], atol=.01), hits)
                hits = section_hits(frame, (side * 120, 0, 36), 1)
                check(f"rail_{side}_visual_length_700", len(hits) == 2 and
                      np.allclose(hits, [-350, 350], atol=.01), hits)
                for z, width, label in ((39, 6, "mouth"), (37.5, 8.5, "undercut")):
                    hits = section_hits(frame, (side * 113, .173, z), 0)
                    near = [value - side * 113 for value in hits if abs(value - side * 113) < 7]
                    check(f"rail_{side}_real_T_slot_{label}", len(near) == 2 and
                          np.allclose(near, [-width / 2, width / 2], atol=.02), near)
                hits = section_hits(frame, (side * 113, .173, 30), 2)
                check(f"rail_{side}_slot_depth_5", bool(hits) and abs(max(hits) - 35) < .02, hits)
                hits = section_hits(frame, (side * 113 + .173, 0, 30), 1)
                check(f"rail_{side}_core_bore_through", not hits, hits)
            components = stop.split(only_watertight=False)
            dimensions = sorted([np.round(mesh.extents * 1000, 4).tolist() for mesh in components],
                                key=lambda size: size[1])
            check("40x40x30_block_and_5mm_bumper", len(dimensions) == 2 and
                  np.allclose(dimensions, [[40, 5, 30], [40, 40, 30]], atol=.02), dimensions)
            hits = section_hits(stop, (.173, -300, 35), 1)
            check("bumper_faces_positive_Y_at_negative_end", bool(hits) and abs(max(hits) + 275) < .01, hits)
            check("frame_contact_friction_025", parts["frame"]["friction"] == .25,
                  parts["frame"]["friction"])
            check("stop_is_fixed_child", parts["stop"]["parent"] == "frame" and
                  parts["stop"]["joint"] is None, [parts["stop"]["parent"], parts["stop"]["joint"]])
        report[prop_id] = checks

    path = Path(__file__).with_name("task03_geometry_checks.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    failures = []
    for prop_id, checks in report.items():
        print(f"{prop_id}: {sum(item['pass'] for item in checks.values())}/{len(checks)} geometry checks passed")
        failures.extend((prop_id, name, item["evidence"]) for name, item in checks.items() if not item["pass"])
    for failure in failures:
        print("FAIL", *failure)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
