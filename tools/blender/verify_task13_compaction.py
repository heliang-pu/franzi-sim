"""Focused compaction regressions and full-delivery audit; no asset rebuilding."""

import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from compact_props import ASSETS, QA, deduplicate, inventory, round_obj, save_png
from prop_schema import referenced_files, schema_errors
from validate_props import read_obj, validate_prop


def regressions():
    checks = []
    with tempfile.TemporaryDirectory(prefix="regression_", dir=QA) as temporary:
        root = Path(temporary)
        folder = root / "_test_cube"
        shutil.copytree(ASSETS / "_test_cube", folder)
        data = json.loads((folder / "prop.json").read_text())
        assert not validate_prop(folder).errors
        shared = root / "shared"
        shared.mkdir()
        shutil.copy2(folder / "body_albedo.png", shared / "body_albedo.png")
        changed = copy.deepcopy(data)
        changed["parts"][0]["texture"] = "../shared/body_albedo.png"
        (folder / "prop.json").write_text(json.dumps(changed))
        assert not validate_prop(folder).errors
        checks.append("sibling PNG reference resolves and validates")
        changed["parts"][0]["texture"] = "../../escaped.png"
        assert schema_errors(changed)
        checks.append("multi-parent traversal rejected")
        (shared / "body_albedo.png").unlink()
        (shared / "body_albedo.png").symlink_to(ASSETS / "_test_cube/body_albedo.png")
        changed["parts"][0]["texture"] = "../shared/body_albedo.png"
        (folder / "prop.json").write_text(json.dumps(changed))
        assert any("unsafe file" in error for error in validate_prop(folder).errors)
        checks.append("symlink escaping props root rejected")
        (shared / "body_albedo.png").unlink()
        (folder / "prop.json").write_text(json.dumps(data))
        shutil.copy2(folder / "body_albedo.png", folder / "copy_albedo.png")
        changed = copy.deepcopy(data)
        changed["variants"] = [{"name": "same", "textures": {"body": "copy_albedo.png"}}]
        (folder / "prop.json").write_text(json.dumps(changed))
        removed = deduplicate(root)
        assert len(removed) == 1 and not (folder / "copy_albedo.png").exists()
        assert not validate_prop(folder).errors
        checks.append("duplicate variant texture redirected before deletion")
        opaque = root / "opaque.png"
        Image.new("RGBA", (16, 16), (23, 45, 67, 255)).save(opaque)
        save_png(opaque)
        with Image.open(opaque) as image:
            assert image.mode == "RGB" and image.getpixel((0, 0)) == (23, 45, 67)
        transparent = root / "transparent.png"
        Image.new("RGBA", (16, 16), (23, 45, 67, 128)).save(transparent)
        save_png(transparent)
        with Image.open(transparent) as image:
            assert image.mode == "RGBA" and image.getpixel((0, 0)) == (23, 45, 67, 128)
        checks.append("unused alpha removed; real alpha and RGB pixels preserved")
        tiny = root / "tiny.obj"
        tiny.write_text("v 0.000001 0.000001 0\nv 0.000004 0.000001 0\n"
                        "v 0.000001 0.000004 0\nvt 0.123456 0.234567\n"
                        "vt 0.5 0.2\nvt 0.1 0.7\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1\n")
        original = read_obj(tiny)
        result = round_obj(tiny, root)
        first_bytes = tiny.read_bytes()
        round_obj(tiny, root)
        assert tiny.read_bytes() == first_bytes
        final = read_obj(tiny)
        assert result["higher_precision_vertices"] == 3
        assert np.array_equal(original.faces, final.faces)
        assert np.array_equal(original.uv_faces, final.uv_faces)
        assert (final.mesh().area_faces > 1e-18).all()
        assert np.linalg.norm(original.vertices - final.vertices, axis=1).max() <= .000005
        checks.append("fine triangle preserved with documented precision exception")
        diagonal = root / "diagonal.obj"
        diagonal.write_text("v 0.000005 0.000005 0.000005\nv 0.1 0 0\nv 0 .1 0\nf 1 2 3\n")
        original = read_obj(diagonal)
        result = round_obj(diagonal, root)
        assert result["higher_precision_vertices"] == 1
        assert np.linalg.norm(original.vertices - read_obj(diagonal).vertices, axis=1).max() <= .000005
        checks.append("Euclidean movement bound enforced rather than per-axis bound")
        first_bytes = diagonal.read_bytes()
        round_obj(diagonal, root)
        assert diagonal.read_bytes() == first_bytes
        checks.append("compaction is byte-idempotent, including promoted precision")
    return checks


def audit():
    before = json.loads((QA / "inventory_before.json").read_text())
    changes = json.loads((QA / "rounding_audit.json").read_text())
    assert max(row["max_vertex_change_mm"] for row in changes) <= .005000001
    assert not list(ASSETS.rglob("*.blend1"))
    assert not list((QA.parents[1]).rglob("*.blend1"))
    references = 0
    shared_refs = 0
    for path in sorted(ASSETS.glob("*/prop.json")):
        data = json.loads(path.read_text())
        assert not schema_errors(data), path
        for reference in referenced_files(data):
            target = (path.parent / reference).resolve()
            assert target.is_file() and target.is_relative_to(ASSETS), (path, reference)
            references += 1
            shared_refs += reference.startswith("../")
        if any(row["higher_precision_vertices"] and row["file"].split("/")[0] == path.parent.name
               for row in changes):
            assert "Task 13 compact OBJ exception:" in data["notes"], path
    unchanged_sources = 0
    for relative, info in before["files"].items():
        if Path(relative).name == "source.blend":
            assert hashlib.sha256((ASSETS / relative).read_bytes()).hexdigest() == info["sha256"], relative
            unchanged_sources += 1
    for path in ASSETS.rglob("*.png"):
        with Image.open(path) as image:
            assert image.mode in ("RGB", "RGBA", "P"), path
            if path.name.startswith("preview"):
                assert image.size == (512, 512), path
            elif "albedo" in path.name and path.parent.name != "factory_floor":
                assert max(image.size) <= 1024, path
    current = inventory()
    size = sum(current["categories"].values()) - current["categories"]["source.blend"]
    assert size <= 90_000_000, size
    return {"references_checked": references, "shared_references": shared_refs,
            "source_blends_unchanged": unchanged_sources, "public_bytes": size,
            "obj_files": len(changes), "max_vertex_change_mm_bound": .005,
            "max_recorded_single_pass_change_mm": max(row["max_vertex_change_mm"] for row in changes),
            "precision_exception_files": sum(row["higher_precision_vertices"] > 0 for row in changes)}


if __name__ == "__main__":
    QA.mkdir(parents=True, exist_ok=True)
    result = {"regressions": regressions(), "audit": audit()}
    (QA / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
