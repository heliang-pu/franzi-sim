"""Read-only geometric/kinematic acceptance checks for the Task 05 sources."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CARTONS = {
    "carton_250x200x150": (200, 250, 150),
    "carton_300x200x200": (200, 300, 200),
    "carton_220x180x120": (180, 220, 120),
}
IDS = (*CARTONS, "roller_conveyor")


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def near(actual, expected, description, tolerance=.025):
    require(np.allclose(actual, expected, atol=tolerance, rtol=0),
            f"{description}: {np.asarray(actual).tolist()} != {np.asarray(expected).tolist()}")


def points(obj, group=None):
    if group is None:
        vertices = obj.data.vertices
    else:
        index = obj.vertex_groups[group].index
        vertices = [v for v in obj.data.vertices
                    if any(g.group == index and g.weight > .5 for g in v.groups)]
    return np.asarray([tuple(v.co) for v in vertices]) * 1000


def bounds(obj, group=None):
    vertices = points(obj, group)
    return np.array([vertices.min(axis=0), vertices.max(axis=0)])


def size(obj, group=None):
    return np.ptp(bounds(obj, group), axis=0)


def groups(obj, prefix):
    return [group.name for group in obj.vertex_groups if group.name.startswith(prefix)]


def ray_wall_thickness(obj, start, direction):
    direction = Vector(direction)
    hit, location, _, _ = obj.ray_cast(Vector([v / 1000 for v in start]), direction)
    require(hit, "outer wall ray hit")
    hit, inside, _, _ = obj.ray_cast(location + direction * .00002, direction)
    require(hit, "inner wall ray hit")
    return (inside - location).length * 1000


def audit_common(data):
    objects = {part["name"]: bpy.data.objects[part["source_visual_object"]] for part in data["parts"]}
    vertices = np.concatenate([points(obj) for obj in objects.values()])
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    near(hi - lo, data["bbox_mm"], "bbox metadata")
    near([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]], [0, 0, 0], "bottom-centre origin")
    for part in data["parts"]:
        obj = objects[part["name"]]
        near(np.asarray(obj.matrix_world), np.eye(4), "applied visual transforms", 1e-7)
        require(all(len(face.vertices) == 3 for face in obj.data.polygons), "triangulated source")
        require(obj.data.uv_layers.get("UVMap") is not None, "export UV atlas")
        atlas = np.asarray([tuple(loop.uv) for loop in obj.data.uv_layers["UVMap"].data])
        require(atlas.min() >= 0 and atlas.max() <= 1, "UV atlas entirely inside [0,1]")
        for slot in {p.material_index for p in obj.data.polygons}:
            mat = obj.data.materials[slot]
            require(mat is not None and mat.node_tree.nodes.get("Principled BSDF") is not None,
                    "retained source PBR")
        require(len(part["source_collision_objects"]) == len(part["collision"]), "proxy provenance")
        for name in part["source_collision_objects"]:
            proxy = bpy.data.objects[name]
            near(np.asarray(proxy.matrix_world), np.eye(4), "applied proxy transforms", 1e-7)
            require(proxy.hide_get() and proxy.hide_render, "hidden source proxies")
    require(bpy.context.scene.cycles.device == "CPU", "CPU Cycles")
    require(bpy.context.scene.cycles.samples == 64, "64 samples")
    require(all(image.packed_file for image in bpy.data.images if image.source == "FILE"), "packed images")
    return objects


def audit_carton(data, objects):
    x, y, height = CARTONS[data["id"]]
    obj = objects["body"]
    near(size(obj), [x, y, height], "nominal XYZ / Y-first identifier")
    require(len(data["parts"]) == 1 and len(data["parts"][0]["collision"]) == 1, "one box collider")
    collider = bpy.data.objects[data["parts"][0]["source_collision_objects"][0]]
    near(bounds(collider), [[-x / 2, -y / 2, 0], [x / 2, y / 2, height]], "nominal box proxy")
    require(len(collider.data.vertices) == 8, "carton box has eight vertices")
    near(ray_wall_thickness(obj, (-x / 2 - 2, 0, height * .20), (1, 0, 0)), 4, "X wall thickness")
    near(ray_wall_thickness(obj, (0, -y / 2 - 2, height * .20), (0, 1, 0)), 4, "Y wall thickness")
    near(data["mass_variants_kg"], [.5, 1, 1.5], "three mass variants")
    near([variant["part_masses_kg"]["body"] for variant in data["variants"]], [.5, 1, 1.5],
         "variant part masses")
    near(data["parts"][0]["friction"], .55, "paper friction")
    require(data["squeeze_axis"] == [0, 1, 0], "Y squeeze axis")
    minor_flaps = groups(obj, "Internal 4 mm minor flap")
    require(len(minor_flaps) == 4, "four minor flaps")
    for group in minor_flaps:
        near(size(obj, group), [x - 8.1, x / 2 - 4, 4], "minor flap board and depth")
    require(len(groups(obj, "Visible wavy single-wall flap medium")) == 4, "four real cut flute patches")
    require(len(groups(obj, "Fine tape wrinkle")) == 4, "four tape wrinkles")
    require(len(groups(obj, "Trapped air bubble")) == 1, "tape bubble")
    tape = groups(obj, "48 mm brown tape")[0]
    near(size(obj, tape), [48, y, 50.15], "tape width, overall Y and 50 mm end fold")
    label = groups(obj, "NOVA SKU label")[0]
    unskew = np.asarray(Matrix.Rotation(math.radians(-2), 3, "X"))
    near(np.ptp(points(obj, label) @ unskew.T, axis=0), [.12, 112, 43], "2-degree physical SKU label")
    require(obj.data.uv_layers.get("HandlingUV") is not None, "handling ink UVs")
    require(obj.data.uv_layers.get("LabelUV") is not None, "SKU label UVs")
    shell = points(obj, groups(obj, "4 mm FEFCO 0201")[0])
    for sign in (-1, 1):
        seam = (np.abs(shell[:, 0] - sign * .75) < .003) & (shell[:, 2] > height - 4.4)
        require(bool(seam.any()), "1.5 mm top flap gap")
    return {"nominal_xyz_mm": [x, y, height], "wall_mm": 4, "mass_variants_kg": [.5, 1, 1.5],
            "minor_flaps": 4, "visible_flute_patches": 4}


def audit_conveyor(data, objects):
    near(data["bbox_mm"], [500, 440, 70], "fitted assembly envelope")
    frame = objects["frame"]
    offset = np.asarray(data["functional_frame_offset_m"]) * 1000
    channels = groups(frame, "40x20x2 real C-channel side frame")
    require(len(channels) == 2, "two real C-channels")
    rail_bounds = sorted([bounds(frame, group) for group in channels], key=lambda b: b[0, 1])
    for group in channels:
        near(size(frame, group), [480, 20, 40], "40x20 C-channel over 480 mm")
    near(rail_bounds[1][0, 1] - rail_bounds[0][1, 1], 320, "320 mm clear width")
    # Sample the unfilled groove, between adjacent roller axles and away from feet.
    hit, location, _, _ = frame.ray_cast(Vector((.01, .17, .020)), Vector((0, 0, 1)))
    require(hit, "C-channel inner top flange")
    near(location.z * 1000, 38, "C-channel retains 18x36 mm empty groove / 2 mm top wall")
    feet = groups(frame, "40x40 L-foot on tabletop")
    require(len(feet) == 4, "four L-feet")
    for group in feet:
        near(size(frame, group), [40, 40, 40], "L-foot geometry")
        near(bounds(frame, group)[0, 2], 0, "feet seat on table")
    frame_part = data["parts"][0]
    require(len(frame_part["collision"]) == 36, "concave frame/foot and fixed-axle decomposition")
    static_bounds = [bounds(bpy.data.objects[name]) for name in frame_part["source_collision_objects"]]
    centers = []
    reference_uv = objects["roller_00"].data.uv_layers["UVMap"]
    reference_uv = np.asarray([tuple(loop.uv) for loop in reference_uv.data])
    for index in range(10):
        name = f"roller_{index:02d}"
        obj = objects[name]
        part = next(p for p in data["parts"] if p["name"] == name)
        bbox = bounds(obj)
        center = bbox.mean(axis=0)
        centers.append(center)
        near(size(obj), [30, 316, 30], "D30 x 316 roller")
        near(center, [-225 + index * 50 + offset[0], 0, 35], "roller centre")
        near(bbox[1, 2], 50, "roller top above table")
        require(len(part["collision"]) == 1, "one collider per roller")
        proxy = bpy.data.objects[part["source_collision_objects"][0]]
        vertices = points(proxy)
        require(len(vertices) == 32, "16-sided roller cylinder has 32 vertices")
        near(np.linalg.norm(vertices[:, [0, 2]] - center[[0, 2]], axis=1), 15,
             "collider vertices lie on design circle")
        require(len(np.unique(np.round(vertices[:, 1], 3))) == 2, "cylinder has two end rings")
        joint = part["joint"]
        require(joint["type"] == "hinge" and joint["axis"] == [0, 1, 0], "Y hinge")
        near(np.asarray(joint["anchor_m"]) * 1000, center, "common-frame hinge anchor")
        require(joint["limited"] is False and joint["continuous"] is True, "unlimited rotation")
        near([joint["damping"], joint["frictionloss"]], [.001, 0], "free hinge damping/friction", 1e-9)
        near([tuple(loop.uv) for loop in obj.data.uv_layers["UVMap"].data], reference_uv,
             "identical reusable roller UVs", 1e-7)
        for fixed in static_bounds:
            overlap = np.minimum(fixed[1], bbox[1]) - np.maximum(fixed[0], bbox[0])
            require(not bool((overlap > .01).all()), "roller initial clearance from fixed frame proxies")
    near(np.diff(np.asarray(centers)[:, 0]), np.full(9, 50), "50 mm roller pitch")
    stop = objects["end_stop"]
    near(size(stop), [20, 320, 30], "stop dimensions")
    near(bounds(stop)[1, 2] - 50, 20, "stop top 20 mm above rollers")
    require(bounds(stop)[1, 0] <= bounds(objects["roller_00"])[0, 0] + .025,
            "outboard stop avoids first roller")
    stop_part = next(p for p in data["parts"] if p["name"] == "end_stop")
    require(stop_part["removable"] and stop_part["joint"] is None, "separate removable stop")
    require(data["flow_axis"] == [-1, 0, 0], "flow toward -X")
    require(next(v for v in data["variants"] if v["name"] == "stop_removed")["disabled_parts"]
            == ["end_stop"], "explicit stop removal variant")
    return {"rollers": 10, "roller_pitch_mm": 50, "roller_top_mm": 50, "frame_top_mm": 40,
            "inside_width_mm": 320, "frame_length_mm": 480, "fitted_length_mm": 500,
            "unlimited_hinges": 10, "initial_roller_frame_clearance": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    report = []
    for prop_id in args.props:
        folder = ROOT / "assets" / "props" / prop_id
        try:
            data = json.loads((folder / "prop.json").read_text())
            bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
            objects = audit_common(data)
            details = audit_carton(data, objects) if prop_id in CARTONS else audit_conveyor(data, objects)
            result = {"id": prop_id, "status": "PASS", **details}
        except Exception as exc:
            result = {"id": prop_id, "status": "FAIL", "error": str(exc)}
        report.append(result)
        print("TASK05_SOURCE_AUDIT", json.dumps(result), flush=True)
    (HERE / "task05_source_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(result["status"] != "PASS" for result in report):
        raise RuntimeError("Task 05 source audit failed")


if __name__ == "__main__":
    main()
