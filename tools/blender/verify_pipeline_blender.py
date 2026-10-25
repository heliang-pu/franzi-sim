"""Headless integration checks of reusable helpers; no benchmark props built."""

import json
import math
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bmesh
import bpy
import numpy as np
import prop_lib as props

QA = Path(__file__).resolve().parent / "_qa"


def mesh_volume(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    assert all(edge.is_manifold for edge in bm.edges), obj.name
    volume = abs(bm.calc_volume(signed=True))
    bm.free()
    return volume


def obj_arrays(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        words = line.split()
        if words and words[0] == "v":
            vertices.append([float(v) for v in words[1:4]])
        elif words and words[0] == "f":
            assert len(words) == 4, "non-triangulated export"
            faces.append([int(v.split("/")[0]) - 1 for v in words[1:]])
    return np.asarray(vertices), faces


def main():
    QA.mkdir(parents=True, exist_ok=True)
    scene = props.reset_scene()
    assert len(scene.objects) == 0 and scene.cycles.device == "CPU"
    assert scene.unit_settings.scale_length == 1 and props.mm(100) == 0.1
    results = {"blender": bpy.app.version_string, "checks": []}
    materials = [
        props.pbr_material(kind, "#C9CDD1", 0.3, 1, 0.2, 0.15, kind)
        for kind in ("turned", "milled", "brushed")
    ]
    assert all(mat.node_tree.nodes.get("Principled BSDF") for mat in materials)
    objects = [
        props.box("asymmetric", (23, 41, 17), (21, 37, 52), bevel_mm=0),
        props.rounded_box("rounded", (45, 70, 30), 4, (0, 0, 15)),
        props.cylinder("shaft", 5, 40, vertices=32),
        props.tube("tube", 30, 20, 14, bevel_mm=0, vertices=48),
        props.hexagonal_prism("hex", 27, 30),
        props.extruded_profile("concave L", [(0, 0), (60, 0), (60, 4), (4, 4), (4, 40), (0, 40)], 12),
    ]
    assert all(mesh_volume(obj) > 0 for obj in objects)
    props.uv_unwrap(objects[1], "smart")
    props.export_obj(objects[1], QA / "smart_uv.obj")
    results["checks"].append("all seven primitive types; manifold geometry; all machining groups")
    asymmetric = objects[0]
    asymmetric.rotation_euler.z = math.pi / 2
    asymmetric.location = (0.012, -0.006, 0.003)
    bpy.context.view_layer.update()
    props.export_obj(asymmetric, QA / "asymmetric.obj")
    vertices, _ = obj_arrays(QA / "asymmetric.obj")
    # World transform is Z+90deg, then (12,-6,3) mm. Detect axis swaps,
    # handedness changes, lost translation and accidental part recentering.
    assert np.allclose(vertices.min(axis=0), [-0.0455, 0.0035, 0.0465], atol=1e-7)
    assert np.allclose(vertices.max(axis=0), [-0.0045, 0.0265, 0.0635], atol=1e-7)
    assert math.isclose(asymmetric.location.x, 0.012, abs_tol=1e-8), "export mutated the source transform"
    results["checks"].append("asymmetric rotated/translated OBJ preserves the prop frame and source")

    sectors = props.ring_segments("bore", 23.55, 31, 13, count=48, location_mm=(0, 0, 32.5))
    errors = []
    for sector in sectors:
        assert mesh_volume(sector) > 0
        inner_faces = [
            face
            for face in sector.data.polygons
            if abs(face.normal.z) < 1e-5 and face.normal.dot(face.center) < -0.001
        ]
        assert len(inner_faces) == 1, (
            sector.name,
            [(tuple(f.normal), tuple(f.center)) for f in sector.data.polygons],
        )
        errors.append(abs(abs(inner_faces[0].normal.dot(inner_faces[0].center)) * 1000 - 23.55))
    assert max(errors) < 0.01
    results["bore_apothem_error_mm"] = max(errors)
    tapered = props.ring_segments("entry", 23.55, 31, 1, count=48, top_inner_radius_mm=24.12735)
    assert all(mesh_volume(piece) > 0 for piece in tapered)
    results["checks"].append("48-sector bore and tapered entry; apothem correct within 0.01 mm")

    oversize = props.cylinder("48-sided solid", 23.5, 14, (11, 17, 7), vertices=48, bevel_mm=0)
    original_volume = mesh_volume(oversize)
    files = props.export_collision("split", [oversize], QA)
    assert len(files) >= 2
    total_volume, maximum = 0, 0
    for filename in files:
        vertices, faces = obj_arrays(QA / filename)
        assert len(vertices) <= 64
        maximum = max(maximum, len(vertices))
        mesh = bpy.data.meshes.new("Roundtrip hull")
        mesh.from_pydata(vertices, [], faces)
        obj = bpy.data.objects.new("Roundtrip hull", mesh)
        total_volume += mesh_volume(obj)
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
    assert abs(total_volume / original_volume - 1) < 1e-4
    results["collision_split"] = {
        "pieces": len(files),
        "max_vertices": maximum,
        "relative_volume_error": abs(total_volume / original_volume - 1),
    }
    results["checks"].append("oversize convex hull subdivided under 64 vertices without volume loss")

    props.reset_scene()
    # Deliberately exercise the Pillow-free path even though Pillow is present.
    props._label_blender(
        QA / "label_fallback.png",
        256,
        256,
        {"text": "TEST", "checker": True, "brand": "QF Express", "barcode": "fake", "zone": "A"},
    )
    image = bpy.data.images.load(str(QA / "label_fallback.png"))
    pixels = np.asarray(image.pixels[:]).reshape(256, 256, 4)
    assert image.size[:] == (256, 256) and pixels[:, :, :3].std() > 0.1
    assert np.linalg.norm(pixels[240, 80, :3] - pixels[240, 112, :3]) > 0.3, (
        "fallback lost its checker pattern"
    )
    assert len(bpy.context.scene.objects) == 0, "label fallback leaked into the source scene"
    results["checks"].append("Pillow-free text/checker/zone/barcode colour bake and scene restoration")

    metal = props.pbr_material("Metal test", "#C9CDD1", 0.3, 1)
    blue = props.pbr_material("Blue test", "#285575", 0.4, 0)
    left = props.box("metal", (20, 20, 20), (-30, 0, 10), material=metal)
    right = props.box("blue", (20, 20, 20), (30, 0, 10), material=blue)
    props.uv_unwrap(left, "smart")
    props.uv_unwrap(right, "cube")
    props.bake_albedo([left, right], 1024, QA / "multi_albedo.png")
    assert metal.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value == 1
    assert all(n.name != "Temporary bake target" for n in metal.node_tree.nodes)
    props.export_obj(left, QA / "multi_metal.obj")
    props.export_obj(right, QA / "multi_blue.obj")
    results["checks"].append("multi-object atlas; pure-metal diffuse colour; PBR graph restoration")
    (QA / "blender_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print("BLENDER_CHECKS_PASS", json.dumps(results))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - propagate Blender script failures to the shell.
        traceback.print_exc()
        sys.exit(1)
