"""Check task18 exported grip geometry, label art, and protected-file hashes.

Run: "$PROP_PYTHON" -B tools/blender/check_task18_handle_tote.py
No Blender dependency; inspect compacted, simulation-facing OBJ files.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
from validate_props import read_obj, validate_prop

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "assets" / "props" / "handle_tote"
QA = ROOT / "tools" / "blender" / "_qa" / "task18"


def ray_hits(triangles: np.ndarray, origin: list[float], direction: list[float]) -> np.ndarray:
    """Vectorised Moller-Trumbore hits, returned as positive ray distances in mm."""
    origin, direction = np.asarray(origin), np.asarray(direction)
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    valid = np.abs(det) > 1e-10
    inverse = np.zeros_like(det)
    inverse[valid] = 1 / det[valid]
    t = origin - triangles[:, 0]
    u = np.einsum("ij,ij->i", t, p) * inverse
    q = np.cross(t, e1)
    v = q @ direction * inverse
    distance = np.einsum("ij,ij->i", e2, q) * inverse
    mask = valid & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (distance >= 0)
    return np.unique(np.round(distance[mask], 5))


def main() -> None:
    data = json.loads((FOLDER / "prop.json").read_text())
    validation = validate_prop(FOLDER)
    assert not validation.errors, validation.errors
    visual = read_obj(FOLDER / "body.obj")
    assert visual.mesh().is_watertight, "Visual shell/card must be closed"
    triangles = visual.vertices[visual.faces] * 1000
    points = visual.vertices * 1000
    np.testing.assert_allclose(points.min(0), [-100, -150, 0], atol=0.01)
    np.testing.assert_allclose(points.max(0), [100, 150, 100], atol=0.01)
    assert data["grasp"] == {"points_mm": [[0, 143, 85], [0, -143, 85]], "width_mm": 14}
    assert data["parts"][0]["mass_kg"] == 0.45
    assert data["parts"][0]["friction"] == 0.40
    assert data["parts"][0]["joint"] is None
    assert len(data["parts"][0]["collision"]) == 11
    grip_checks = []
    for side in (-1, 1):
        for x in (-40, 0, 40):
            for z in (75, 85, 95):
                outside = ray_hits(triangles, [x, side * 160, z], [0, -side, 0])
                inside = ray_hits(triangles, [x, side * 120, z], [0, side, 0])
                np.testing.assert_allclose(outside[:2], [10, 24], atol=0.01)
                np.testing.assert_allclose(inside[:2], [16, 30], atol=0.01)
                grip_checks.append({"side": side, "x": x, "z": z,
                                    "outer_abs_y_mm": 160 - float(outside[0]),
                                    "inner_abs_y_mm": 120 + float(inside[0])})
        for x in (-30, 0, 30):
            hits = ray_hits(triangles, [x, side * 160, 58], [0, -side, 0])
            assert not (hits < 30).any(), ("Cosmetic opening blocked", side, x, hits)
        hits = ray_hits(triangles, [42, side * 160, 58], [0, -side, 0])
        assert (hits < 30).any(), "Opening exceeds nominal width"
    cavity = ray_hits(triangles, [0, 0, 120], [0, 0, -1])
    np.testing.assert_allclose(cavity[0], 116, atol=0.01)
    collision_checks = []
    for file, role in data["collision_roles"].items():
        obj = read_obj(FOLDER / file)
        mesh = obj.mesh()
        assert mesh.is_convex and mesh.is_watertight
        assert len(mesh.vertices) <= 64
        bounds = obj.vertices * 1000
        if role.startswith("handle_"):
            side = 1 if role.endswith("py") else -1
            np.testing.assert_allclose(bounds.max(0) - bounds.min(0), [120, 14, 30], atol=0.01)
            np.testing.assert_allclose((bounds.max(0) + bounds.min(0)) / 2, [0, side * 143, 85], atol=0.01)
        collision_checks.append({"file": file, "role": role, "vertices": len(mesh.vertices),
                                 "bbox_mm": (bounds.max(0) - bounds.min(0)).tolist()})
    expected = [("blue", "#1F5FA8", "BIN-A07"), ("grey", "#9AA0A6", "BIN-A07"),
                ("red", "#C62828", "NG"), ("yellow", "#F9A825", "RECHECK"), ("green", "#2E7D32", "OK")]
    assert [(v["name"], v["base_color"], v["label_text"]) for v in data["variants"]] == expected
    for name, _, _ in expected:
        with Image.open(FOLDER / f"label_{name}_source.png") as image:
            pixels = np.asarray(image.convert("RGB"))
        ys, xs = np.where(pixels.min(2) < 100)
        assert abs((ys.max() - ys.min() + 1) / 20 - 22) < 0.1
        assert (xs.max() - xs.min() + 1) / 20 <= 86.1
    hashes = json.loads((QA / "protected_hashes.json").read_text())
    for file, digest in hashes.items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest, f"Protected file changed: {file}"
    result = {"status": "PASS", "bbox_mm": np.ptp(points, axis=0).tolist(),
              "triangles": len(visual.faces), "visual_watertight": bool(visual.mesh().is_watertight),
              "grip_checks": grip_checks, "cosmetic_openings": "both clear",
              "cavity_floor_z_mm": 120 - float(cavity[0]), "collisions": collision_checks,
              "labels": "all five variants match; cap height 22 mm", "protected_files_unchanged": len(hashes)}
    (QA / "geometry_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
