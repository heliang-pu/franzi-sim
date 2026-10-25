"""Re-hull individual Task 07 proxies after OBJ coordinate quantization.

This is the same narrow repair used by Task 02: vertices never move, and a
volume change above 0.2 percent is rejected. No whole-part convex hulls.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

from convexify_task02_collisions import collision_files
from validate_props import read_obj

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "assets" / "props"
IDS = ("lidded_tote", "station_plate", "drawer_cabinet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    args = parser.parse_args()
    destination = HERE / "task07_convexity_repairs.json"
    report = json.loads(destination.read_text()) if destination.exists() else {}
    for prop_id in args.props:
        folder = ROOT / prop_id
        data = json.loads((folder / "prop.json").read_text())
        files, repaired = collision_files(data), []
        for reference in files:
            path = folder / reference
            original = read_obj(path).mesh()
            if original.is_convex:
                continue
            hull = original.convex_hull
            ratio = abs(float(hull.volume / original.volume))
            if not original.is_watertight or not .998 <= ratio <= 1.002:
                raise ValueError(f"Refusing a substantive shape change: {reference}, ratio={ratio}")
            if not hull.is_convex or len(hull.vertices) > 64:
                raise ValueError(f"Invalid hull: {reference}")
            existing = {tuple(np.rint(v * 1e6).astype(int)) for v in original.vertices}
            assert all(tuple(np.rint(v * 1e6).astype(int)) in existing for v in hull.vertices)
            lines = ["# Convex topology of existing quantized vertices; metres, common prop frame"]
            lines += ["v " + " ".join(f"{v:.6f}" for v in point) for point in hull.vertices]
            lines += ["vn " + " ".join(f"{v:.9f}" for v in normal) for normal in hull.face_normals]
            lines += ["f " + " ".join(f"{int(v) + 1}//{i + 1}" for v in face)
                      for i, face in enumerate(hull.faces)]
            path.write_text("\n".join(lines) + "\n")
            result = read_obj(path).mesh()
            assert result.is_convex and result.is_watertight and result.is_winding_consistent
            repaired.append({"file": reference, "hull_volume_ratio": ratio, "vertices": len(result.vertices)})
        report[prop_id] = {"checked": len(files),
                           "repaired": repaired or report.get(prop_id, {}).get("repaired", [])}
        print("TASK07_CONVEX", prop_id, len(files), "checked;", len(repaired), "repaired", flush=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
