"""Repair proxy topology AFTER Blender's one-micrometre OBJ quantization.

Rotating a cylinder makes its cap coordinates non-coplanar after rounding.
Re-hull only each existing convex piece, using its exported double-precision
points. Never hull a whole part or move vertices. A volume change over 0.2%
is rejected, so this cannot silently replace a concave shape by a solid hull.
Run with the prescribed inovance-fineart Python, also called by the builder.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

from validate_props import read_obj


ROOT = Path(__file__).resolve().parents[2] / "assets" / "props"


def collision_files(data):
    result = set()

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "collision" and isinstance(child, list):
                    result.update(child)
                elif key != "notes":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=["panel_devices", "control_console"],
                        default=["panel_devices", "control_console"])
    args = parser.parse_args()
    report_path = Path(__file__).with_name("task02_convexity_repairs.json")
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for prop_id in args.props:
        folder = ROOT / prop_id
        metadata = json.loads((folder / "prop.json").read_text())
        repaired = []
        for reference in collision_files(metadata):
            path = folder / reference
            original = read_obj(path).mesh()
            if original.is_convex:
                continue
            hull = original.convex_hull
            ratio = abs(hull.volume / original.volume)
            if not original.is_watertight or not 0.998 <= ratio <= 1.002:
                raise ValueError(f"Refusing nontrivial geometry change for {path.name}: volume ratio {ratio}")
            if len(hull.vertices) > 64 or not hull.is_convex:
                raise ValueError(f"Hull exceeds the proxy budget or is not convex: {path.name}")
            # Every hull point must be an existing one-micrometre export point.
            original_points = {tuple(np.rint(v * 1e6).astype(int)) for v in original.vertices}
            if not all(tuple(np.rint(v * 1e6).astype(int)) in original_points for v in hull.vertices):
                raise ValueError("Convex repair unexpectedly introduced vertices")
            lines = ["# Convex topology from the existing quantized proxy points; metres"]
            lines.extend("v " + " ".join(f"{value:.6f}" for value in v) for v in hull.vertices)
            lines.extend("vn " + " ".join(f"{value:.9f}" for value in n) for n in hull.face_normals)
            lines.extend("f " + " ".join(f"{int(v) + 1}//{index + 1}" for v in face)
                         for index, face in enumerate(hull.faces))
            path.write_text("\n".join(lines) + "\n")
            result = read_obj(path).mesh()
            if not result.is_convex or not result.is_watertight:
                raise ValueError(f"Quantized proxy still fails convexity: {path.name}")
            repaired.append({"file": reference, "hull_volume_ratio": float(ratio),
                             "vertices": len(result.vertices)})
        previous = report.get(prop_id, {}).get("repaired", [])
        report[prop_id] = {"checked": len(collision_files(metadata)), "repaired": repaired or previous}
        print(f"{prop_id}: {len(collision_files(metadata))} convex proxies checked; "
              f"{len(repaired)} quantized topologies repaired")
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
