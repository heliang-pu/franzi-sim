"""Read-only dimensional/source audit for Task 04, run in headless Blender."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import bpy
import bmesh
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IDS = ("belt_conveyor", "sort_bin", "parcel_box_s", "parcel_box_m", "parcel_pouch")
REPORT = []


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def near(actual, expected, description, tolerance=.03):
    require(np.allclose(actual, expected, atol=tolerance, rtol=0),
            f"{description}: {np.asarray(actual).tolist()} != {np.asarray(expected).tolist()}")


def points(obj, group=None):
    if group is None:
        vertices = obj.data.vertices
    else:
        index = obj.vertex_groups[group].index
        vertices = [vertex for vertex in obj.data.vertices
                    if any(item.group == index and item.weight > .5 for item in vertex.groups)]
    return np.asarray([tuple(vertex.co) for vertex in vertices]) * 1000


def groups(obj, prefix):
    result = [group.name for group in obj.vertex_groups if group.name.startswith(prefix)]
    require(bool(result), f"{obj.name}: missing feature {prefix}")
    return result


def bounds(obj, group=None):
    verts = points(obj, group)
    return np.array([verts.min(axis=0), verts.max(axis=0)])


def size(obj, group=None):
    return np.ptp(bounds(obj, group), axis=0)


def center(obj, group=None):
    return bounds(obj, group).mean(axis=0)


def audit_common(data):
    objects = {part["name"]: bpy.data.objects[part["source_visual_object"]] for part in data["parts"]}
    all_points = np.concatenate([points(obj) for obj in objects.values()])
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    near(upper - lower, data["bbox_mm"], "declared bbox")
    near([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, lower[2]], [0, 0, 0],
         "floor/bottom-centre origin")
    for part in data["parts"]:
        visual = objects[part["name"]]
        near(np.asarray(visual.matrix_world), np.eye(4), "visual applied transforms", 1e-7)
        require(all(len(face.vertices) == 3 for face in visual.data.polygons), "source triangulation")
        require(visual.data.uv_layers.get("UVMap") is not None, "export UV atlas")
        require(len(part["collision"]) == len(part["source_collision_objects"]), "proxy provenance")
        for name in part["source_collision_objects"]:
            proxy = bpy.data.objects[name]
            require(proxy.hide_render and proxy.hide_get(), "source proxies must be hidden")
            near(np.asarray(proxy.matrix_world), np.eye(4), "proxy applied transforms", 1e-7)
        # Boolean cutters can leave an unused empty slot; inspect assigned slots.
        used_slots = {polygon.material_index for polygon in visual.data.polygons}
        for slot in used_slots:
            material = visual.data.materials[slot]
            require(material is not None, "all used face materials must be assigned")
            require(material.node_tree.nodes.get("Principled BSDF") is not None,
                    "PBR source material must be retained")
    require(bpy.context.scene.cycles.samples == 64, "64-sample bake setting")
    require(bpy.context.scene.cycles.device == "CPU", "CPU bake setting")
    return objects


def audit_conveyor(data, objects):
    frame, belt = objects["frame"], objects["belt"]
    offset = np.asarray(data["functional_frame_offset_m"]) * 1000
    near(size(belt), [150, 760, 3], "belt dimensions")
    near(bounds(belt)[1, 2], 800, "belt top")
    drums = groups(frame, "D50 end drum")
    require(len(drums) == 2, "two drums")
    drum_centers = [center(frame, group) for group in drums]
    near(abs(drum_centers[0][1] - drum_centers[1][1]), 760, "drum centre distance")
    for group in drums:
        near(size(frame, group), [170, 50, 50], "D50 drum")
    rails = groups(frame, "Stainless guide")
    require(len(rails) == 2, "two guides")
    rail_bounds = sorted([bounds(frame, group) for group in rails], key=lambda value: value[0, 0])
    near(rail_bounds[1][0, 0] - rail_bounds[0][1, 0], 160, "guide inner gap")
    for group in rails:
        near(size(frame, group), [2, 760, 30], "guide strip")
        near(bounds(frame, group)[1, 2], 830, "guide top")
    for group in groups(frame, "Side beam"):
        near(size(frame, group), [40, 800, 40], "4040 side beam")
        near(bounds(frame, group)[1, 2], 790, "beam top")
    feet = groups(frame, "D50 x 20 levelling foot")
    require(len(feet) == 4, "four levelling feet")
    for group in feet:
        near(size(frame, group), [50, 50, 20], "levelling foot")
        near(bounds(frame, group)[0, 2], 0, "feet on floor")
    require(len(groups(frame, "40x40 vertical leg")) == 4, "four legs")
    require(len(groups(frame, "Guide upright bracket")) == 8, "200 mm bracket pitch")
    end = groups(frame, "Stainless 160x40x3 end plate")[0]
    near(size(frame, end), [160, 3, 40], "end plate")
    near(bounds(frame, end)[1, 2], 840, "end plate top")
    gear = groups(frame, "100x80x80 reduction gearbox")[0]
    near(size(frame, gear), [100, 80, 80], "gearbox")
    gear_center = center(frame, gear) - offset
    require(gear_center[0] > 0 and gear_center[1] < 0, "motor at +X, -Y end")
    eye = groups(frame, "Photo-eye 30x20x12")[0]
    near(size(frame, eye), [12, 30, 20], "photo-eye")
    near((center(frame, eye) - offset)[2], 830, "photo-eye optical height")
    reflector = groups(frame, "Reflector 30x30x3")[0]
    near(size(frame, reflector), [3, 30, 30], "reflector")
    require(sum(len(part["collision"]) for part in data["parts"]) == 8, "eight conveyor proxies")
    require(data["flow_axis"] == [0, -1, 0], "flow +Y to -Y")


def audit_bin(data, objects):
    body, sign = objects["body"], objects["sign"]
    shell = groups(body, "200x160x150 open")[0]
    near(size(body, shell), [200, 160, 150], "bin body")
    near(size(sign), [3, 120, 80], "sign")
    post = groups(body, "D20 sign post")[0]
    near(size(body, post), [20, 20, 299], "post starts on 1 mm floor")
    near(bounds(body, post)[1, 2], 300, "post top")
    require(sum(part["mass_kg"] for part in data["parts"]) == .8, "static effective mass")
    require(sum(len(part["collision"]) for part in data["parts"]) == 6, "six bin proxies")
    for variant in data["variants"]:
        require(set(variant["textures"]) == {"sign"}, "only sign texture changes")
    lip = bpy.data.objects[data["parts"][0]["source_collision_objects"][-1]]
    lip_points = points(lip)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(lip_points.T))
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    angle = math.degrees(math.acos(abs(normal[2])))
    near(angle, 30, "entry ramp angle from horizontal")


def audit_parcel(data, objects):
    visual = objects["body"]
    pouch = data["id"] == "parcel_pouch"
    dims = data["dimensions_mm"]["nominal_xyz"]
    near(size(visual), dims, "parcel outside dimensions")
    require(len(data["parts"]) == 1 and len(data["parts"][0]["collision"]) == 1, "one box proxy")
    collision = bpy.data.objects[data["parts"][0]["source_collision_objects"][0]]
    near(size(collision), [76, 196, 33] if pouch else dims, "parcel box collision")
    angle = math.radians(-data["dimensions_mm"]["label_skew_deg"])
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                         [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    label_group = groups(visual, "Shipping label")[0]
    label = points(visual, label_group) @ rotation.T
    near(np.ptp(label, axis=0), [60, 90 if pouch else 100, .16], "physical label dimensions")
    require(visual.data.uv_layers.get("LabelUV") is not None, "independent source image UV")
    require([variant["name"] for variant in data["variants"]] == ["A", "B", "C"], "zone variants")
    if pouch:
        seam = groups(visual, "Continuous 2 mm heat-sealed")[0]
        near(size(visual, seam), [80, 200, .8], "heat seal flange")
    else:
        require(len(groups(visual, "Exposed wavy corrugated")) == 2, "two cut flute edges")
        require(len(groups(visual, "Tape fine wrinkle")) == 4, "physical tape wrinkles")
        require(.15 <= data["parts"][0]["mass_kg"] <= .5, "single-hand parcel mass")
        # A closed inner cavity with outward normals can pass watertightness
        # and winding checks, yet add its air volume instead of subtracting it.
        bm = bmesh.new()
        bm.from_mesh(visual.data)
        volume = bm.calc_volume(signed=True)
        bm.free()
        outer = float(np.prod(dims)) * 1e-9
        inner = float(np.prod(np.asarray(dims) - 6)) * 1e-9
        require(.8 < volume / (outer - inner) < 1.2, "3 mm carton wall / inner-cavity orientation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    failed = False
    for prop_id in args.props:
        folder = ROOT / "assets" / "props" / prop_id
        data = json.loads((folder / "prop.json").read_text())
        bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
        try:
            objects = audit_common(data)
            if prop_id == "belt_conveyor":
                audit_conveyor(data, objects)
            elif prop_id == "sort_bin":
                audit_bin(data, objects)
            else:
                audit_parcel(data, objects)
            result = {"id": prop_id, "status": "PASS"}
        except Exception as exc:
            failed = True
            result = {"id": prop_id, "status": "FAIL", "error": str(exc)}
        REPORT.append(result)
        print("TASK04_SOURCE_AUDIT", json.dumps(result), flush=True)
    (HERE / "task04_source_checks.json").write_text(json.dumps(REPORT, indent=2) + "\n")
    if failed:
        raise RuntimeError("Task 04 source audit failed")


if __name__ == "__main__":
    main()
