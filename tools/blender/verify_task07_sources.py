"""Check Task 07 source geometry and closed-to-open convex proxy sweeps.

Headless Blender: -P this_file -- [--geometry-only]
The geometry-only mode reads the resumable QA blends; otherwise it checks the
delivered source.blend files. It does not change the delivered props.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bmesh
import bpy
import numpy as np

import prop_lib as props

IDS = ("lidded_tote", "station_plate", "drawer_cabinet")


def hull(obj):
    mesh = props._world_mesh(obj)
    bm = props._hull([v.co for v in mesh.vertices])
    bpy.data.meshes.remove(mesh)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    vertices = np.array([tuple(v.co) for v in bm.verts])
    faces = [[v.index for v in f.verts] for f in bm.faces]
    bm.free()
    return vertices, faces


def directions(vertices, faces):
    normals, edges = [], []
    for face in faces:
        a, b, c = vertices[face[:3]]
        normals.append(np.cross(b - a, c - a))
        for i, j in zip(face, face[1:] + face[:1], strict=True):
            edges.append(vertices[j] - vertices[i])

    def unique(values):
        values = np.array(values)
        lengths = np.linalg.norm(values, axis=1)
        values = values[lengths > 1e-12] / lengths[lengths > 1e-12, None]
        # Opposite axes give the same separation test.
        for row in values:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        return np.unique(np.round(values, 7), axis=0)

    return unique(normals), unique(edges)


def penetration(a, b):
    av, af = a
    bv, bf = b
    if np.min(np.minimum(av.max(axis=0), bv.max(axis=0)) -
              np.maximum(av.min(axis=0), bv.min(axis=0))) <= 0:
        return 0.0
    an, ae = directions(av, af)
    bn, be = directions(bv, bf)
    axes = np.vstack((an, bn, np.cross(ae[:, None], be[None, :]).reshape(-1, 3)))
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-9] / lengths[lengths > 1e-9, None]
    pa, pb = av @ axes.T, bv @ axes.T
    overlap = np.minimum(pa.max(axis=0), pb.max(axis=0)) - np.maximum(pa.min(axis=0), pb.min(axis=0))
    return max(0, float(overlap.min()))


def transformed(shape, joint, position):
    vertices, faces = shape
    if joint["type"] == "slide":
        return vertices + np.asarray(joint["axis"]) * position, faces
    axis = np.asarray(joint["axis"])
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + math.sin(position) * cross + (1 - math.cos(position)) * cross @ cross
    anchor = np.asarray(joint["anchor_m"])
    return (vertices - anchor) @ rotation.T + anchor, faces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    report = {}
    for prop_id in IDS:
        folder = props.ROOT / "assets" / "props" / prop_id
        source = HERE / "_qa" / "task07" / f"{prop_id}_geometry.blend" if args.geometry_only else folder / "source.blend"
        bpy.ops.wm.open_mainfile(filepath=str(source))
        manifest = json.loads(bpy.context.scene["task07_manifest"])
        checks = {}

        def check(name, condition, evidence):
            checks[name] = {"pass": bool(condition), "evidence": evidence}

        scene = bpy.context.scene
        check("metric_CPU_Cycles_64", scene.unit_settings.system == "METRIC" and
              scene.unit_settings.scale_length == 1 and scene.render.engine == "CYCLES" and
              scene.cycles.device == "CPU" and scene.cycles.samples == 64,
              [scene.unit_settings.system, scene.cycles.device, scene.cycles.samples])
        collision_shapes = {}
        for part in manifest["parts"]:
            name = part["name"]
            obj = bpy.data.objects[part["visual_object"]]
            check(name + "_identity_transform", np.allclose(np.array(obj.matrix_world), np.eye(4), atol=1e-7),
                  np.array(obj.matrix_world).tolist())
            check(name + "_triangulated", all(len(p.vertices) == 3 for p in obj.data.polygons), len(obj.data.polygons))
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bad = sum(not edge.is_manifold for edge in bm.edges)
            bm.free()
            check(name + "_closed_visual_components", bad == 0, {"nonmanifold_edges": bad})
            mats = [slot.material for slot in obj.material_slots]
            check(name + "_PBR_retained", all(m and m.node_tree.nodes.get("Principled BSDF") for m in mats),
                  [m.name if m else None for m in mats])
            uv = obj.data.uv_layers.get("UVMap")
            check(name + "_bake_atlas", uv is not None and uv.active_render,
                  [layer.name for layer in obj.data.uv_layers])
            collisions = [bpy.data.objects[n] for n in part["collision_objects"]]
            check(name + "_proxies_hidden", all(o.hide_render for o in collisions), len(collisions))
            check(name + "_proxy_frames", all(np.allclose(np.array(o.matrix_world), np.eye(4), atol=1e-7)
                                                for o in collisions), len(collisions))
            collision_shapes[name] = [hull(o) for o in collisions]
        for part in manifest["parts"]:
            joint = part["joint"]
            if not joint:
                continue
            name = part["name"]
            positions = list(np.linspace(*joint["range"], 23 if joint["type"] == "hinge" else 19))
            if joint["type"] == "hinge":
                positions += [math.radians(1), math.radians(3)]
            worst, location = 0.0, None
            for position in positions:
                for i, moving in enumerate(collision_shapes[name]):
                    moved = transformed(moving, joint, float(position))
                    obstacles = [shape for other, shapes in collision_shapes.items() if other != name for shape in shapes]
                    for j, stationary in enumerate(obstacles):
                        depth = penetration(moved, stationary)
                        if depth > worst:
                            worst, location = depth, [float(position), i, j]
            check(name + "_sweep_clear", worst <= .00015,
                  {"samples": len(positions), "max_penetration_mm": worst * 1000, "at": location})
        if prop_id == "lidded_tote":
            lid = next(p for p in manifest["parts"] if p["name"] == "lid")
            joint = lid["joint"]
            check("exact_hinge_contract", np.allclose(joint["anchor_m"], [.102, 0, .152]) and
                  joint["axis"] == [0, 1, 0] and np.allclose(joint["range"], [0, math.radians(110)]) and
                  joint["damping"] == .05 and joint["frictionloss"] == .4, joint)
            check("catch_contract_in_notes", "0.3 N m" in manifest["notes"] and "0..3 degrees" in manifest["notes"],
                  manifest["interaction"])
        report[prop_id] = checks
        print("TASK07_SOURCE", prop_id, sum(item["pass"] for item in checks.values()), "/", len(checks), flush=True)
    destination = HERE / ("task07_geometry_checks.json" if args.geometry_only else "task07_source_checks.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    failures = [(prop_id, name, item["evidence"]) for prop_id, checks in report.items()
                for name, item in checks.items() if not item["pass"]]
    for failure in failures:
        print("FAIL", *failure, flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} Task 07 source checks failed")


if __name__ == "__main__":
    main()
