"""Verify Task 12 preservation, exported textures and pocket/notch contrast.

Run with the same Python as validate_props.py, after the material-only update.
The baseline is captured before the update in _qa/task12/baseline_hashes.json.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from validate_props import read_obj, validate_prop

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QA = HERE / "_qa" / "task12"
FOLDER = ROOT / "assets" / "props" / "foam_kit_tray"


def texture_sample(obj, image, point):
    """Interpolate the exported OBJ's UVs at an actual horizontal surface."""
    triangles = obj.vertices[obj.faces]
    flat = np.max(np.abs(triangles[:, :, 2] - point[2]), axis=1) < 1e-7
    indices = np.flatnonzero(flat)
    triangles = triangles[flat, :, :2]
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    q = np.asarray(point[:2]) - triangles[:, 0]
    determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    valid = np.abs(determinant) > 1e-14
    u = np.divide(q[:, 0] * b[:, 1] - q[:, 1] * b[:, 0], determinant,
                  out=np.full(len(q), np.inf), where=valid)
    v = np.divide(a[:, 0] * q[:, 1] - a[:, 1] * q[:, 0], determinant,
                  out=np.full(len(q), np.inf), where=valid)
    inside = np.flatnonzero(valid & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6))
    if not len(inside):
        raise AssertionError(f"No horizontal exported surface at {point}")
    i = inside[0]
    uv = np.array([1 - u[i] - v[i], u[i], v[i]]) @ obj.uv[obj.uv_faces[indices[i]]]
    x = int(np.clip(uv[0] * image.width, 1, image.width - 2))
    y = int(np.clip((1 - uv[1]) * image.height, 1, image.height - 2))
    return np.asarray(image)[y - 1:y + 2, x - 1:x + 2, :3].mean(axis=(0, 1))


def main():
    baseline = json.loads((QA / "baseline_hashes.json").read_text())
    permitted = {str((FOLDER / filename).relative_to(ROOT))
                 for filename in ("source.blend", "prop.json", "preview.png")}
    changed = []
    for relative, digest in baseline.items():
        target = ROOT / relative
        assert target.is_file(), f"Removed original asset: {relative}"
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            changed.append(relative)
    assert set(changed) <= permitted, f"Unexpected asset changes: {set(changed) - permitted}"
    current_files = {str(p.relative_to(ROOT)) for p in (ROOT / "assets" / "props").rglob("*") if p.is_file()}
    added = current_files - baseline.keys()
    assert added == {str((FOLDER / "body_albedo_v1.png").relative_to(ROOT))}, added
    data = json.loads((FOLDER / "prop.json").read_text())
    variants = {v["name"]: v for v in data["variants"]}
    assert set(variants) == {"black", "light_grey"}
    for part in data["parts"]:
        assert variants["black"]["textures"][part["name"]] == part["texture"]
    assert variants["light_grey"]["foam_base_color_srgb"] == "#A9AFB6"
    assert variants["light_grey"]["pocket_ao_strength"] == .8
    result = validate_prop(FOLDER)
    assert not result.errors, result.errors
    image_sizes = {}
    for target, expected in ((FOLDER / "body_albedo_v1.png", (1024, 1024)),
                             (FOLDER / "preview.png", (512, 512)),
                             (HERE / "_qa" / "foam_kit_tray_light_top.png", (640, 480))):
        with Image.open(target) as image:
            assert image.size == expected, (target, image.size)
            image.verify()
        image_sizes[str(target.relative_to(ROOT))] = list(expected)
    obj = read_obj(FOLDER / "body.obj")
    probes = (
        ("hex_blank", (.090, -.090, .043), (.055, -.090, .018), (.040, -.090, .018)),
        ("flanged_bushing", (.090, .090, .043), (.055, .090, .013), (.0335, .090, .013)),
        ("l_bracket", (-.045, -.090, .043), (-.073, -.090, .018), (-.0825, -.090, .018)),
        ("spur_gear", (-.015, .090, .043), (-.055, .090, .033), (-.0805, .090, .033)),
    )
    contrast = {}
    weights = np.array([.2126, .7152, .0722])
    with Image.open(FOLDER / "body_albedo_v1.png") as image:
        for name, top, floor, notch in probes:
            values = {label: float(texture_sample(obj, image, point) @ weights)
                      for label, point in (("top", top), ("floor", floor), ("notch", notch))}
            assert values["floor"] < .90 * values["top"], (name, values)
            assert values["notch"] < .90 * values["top"], (name, values)
            contrast[name] = {key + "_srgb_luma_255": round(value, 2) for key, value in values.items()}
    audit = {"status": "PASS", "existing_files_checked": len(baseline),
             "changed_existing_assets": changed, "visual_and_all_64_collisions_unchanged": True,
             "added_assets": sorted(added),
             "black_albedo_unchanged": True, "other_props_unchanged": True,
             "variants": list(variants), "image_sizes": image_sizes,
             "exported_atlas_contrast": contrast, "validator": result.row()}
    (QA / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
