"""Validator regressions, exact UV overlap checks and MuJoCo loading/settling.

Use the benchmark's external Python, after build_task00_pipeline.py and
verify_pipeline_blender.py. Temporary failure fixtures stay inside tools/blender.
"""

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from xml.sax.saxutils import quoteattr

sys.dont_write_bytecode = True

import mujoco
import numpy as np
import trimesh
from PIL import Image
from prop_schema import TRIANGLE_BUDGETS, schema_errors
from validate_props import read_obj, validate_prop

HERE = Path(__file__).resolve().parent
QA = HERE / "_qa"
CUBE = HERE.parents[1] / "assets/props/_test_cube"


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def area(polygon):
    return abs(sum(cross(a, b) for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True))) / 2


def intersection_area(first, second):
    """Sutherland-Hodgman clipping; detects actual area, not shared boundaries."""
    clipped = list(first)
    if cross(second[1] - second[0], second[2] - second[0]) < 0:
        second = second[::-1]
    for start, end in zip(second, np.roll(second, -1, axis=0), strict=True):
        if not clipped:
            return 0
        input_points, clipped = clipped, []
        previous = input_points[-1]
        previous_distance = cross(end - start, previous - start)
        for current in input_points:
            current_distance = cross(end - start, current - start)
            if (current_distance >= 0) != (previous_distance >= 0):
                clipped.append(
                    previous
                    + (current - previous) * previous_distance / (previous_distance - current_distance)
                )
            if current_distance >= 0:
                clipped.append(current)
            previous, previous_distance = current, current_distance
    return area(clipped) if len(clipped) >= 3 else 0


def check_uv(paths):
    triangles = np.concatenate([obj.uv[obj.uv_faces] for obj in (read_obj(p) for p in paths)])
    lower, upper = triangles.min(axis=1), triangles.max(axis=1)
    for i, triangle in enumerate(triangles):
        assert area(triangle) > 1e-12, f"degenerate UV triangle {i} in {paths}"
        candidates = np.flatnonzero(
            np.all(np.minimum(upper[i], upper) - np.maximum(lower[i], lower) > 1e-9, axis=1)
        )
        for j in candidates[candidates > i]:
            overlap = intersection_area(triangle, triangles[j])
            assert overlap < 1e-10, f"UV area overlap {overlap} between triangles {i},{j} in {paths}"
    return {"meshes": [p.name for p in paths], "triangles": len(triangles), "overlapping_pairs": 0}


def validator_regressions():
    checks = []
    with tempfile.TemporaryDirectory(prefix="validator_cases_", dir=QA) as temporary:
        folder = Path(temporary) / "_test_cube"
        shutil.copytree(CUBE, folder)
        metadata = json.loads((folder / "prop.json").read_text())
        collision_path = folder / metadata["parts"][0]["collision"][0]
        original_collision = collision_path.read_text()

        def write(data):
            (folder / "prop.json").write_text(json.dumps(data))

        def reject(name, data, message):
            write(data)
            errors = validate_prop(folder).errors
            assert any(message in error for error in errors), (name, errors)
            checks.append(name)
            write(metadata)

        assert not validate_prop(folder).errors
        checks.append("valid cube accepted")
        missing = copy.deepcopy(metadata)
        missing["parts"][0]["texture"] = "missing.png"
        reject("missing referenced file", missing, "missing/unsafe file")
        wrong_bbox = copy.deepcopy(metadata)
        wrong_bbox["bbox_mm"][0] += 2
        reject("bbox out of 1 mm tolerance", wrong_bbox, "bbox differs")
        within = copy.deepcopy(metadata)
        within["bbox_mm"][0] += 0.8
        write(within)
        assert not validate_prop(folder).errors
        checks.append("0.8 mm accepted for regular props")
        within["precision_mm"] = 0.01
        reject("0.8 mm rejected for precision props", within, "bbox differs")
        bad_count = copy.deepcopy(metadata)
        bad_count["triangles"] += 1
        reject("triangle metadata mismatch", bad_count, "triangles metadata")
        with patch.dict(TRIANGLE_BUDGETS, {"small": 100}):
            errors = validate_prop(folder).errors
            assert any("triangle budget exceeded" in error for error in errors)
        checks.append("triangle budget enforced against actual mesh")
        polygon = read_obj(collision_path).mesh()
        index = np.argmax(polygon.vertices.sum(axis=1))
        polygon.vertices[index] = [0, 0, 0.05]
        collision_path.write_text(polygon.export(file_type="obj"))
        reject("concave closed collision rejected", metadata, "nonconvex")
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.04)
        collision_path.write_text(sphere.export(file_type="obj"))
        reject("collision above 64 vertices", metadata, "vertices > 64")
        collision_path.write_text(original_collision)
        image_path = folder / metadata["parts"][0]["texture"]
        with Image.open(image_path) as image:
            resized = image.resize((512, 512))
        image_bytes = image_path.read_bytes()
        resized.save(image_path)
        reject("wrong PNG size", metadata, "expected 1024x1024")
        image_path.write_bytes(image_bytes)
        variant = copy.deepcopy(metadata)
        variant["variants"] = [{"name": "missing", "textures": {"body": "variant_missing.png"}}]
        reject("missing variant file", variant, "missing/unsafe file")
        variant["variants"][0]["textures"]["body"] = "variant_wrong.png"
        resized.save(folder / "variant_wrong.png")
        reject("variant PNG size enforced", variant, "expected 1024x1024")
        bad_joint = copy.deepcopy(metadata)
        child = copy.deepcopy(bad_joint["parts"][0])
        child.update(name="lid", collision=[], parent="body", joint={"type": "hinge"})
        bad_joint["parts"].append(child)
        reject("incomplete joint", bad_joint, "joint anchor_m")
        child["joint"].update(
            axis=[0, 0, 1],
            anchor_m=[0, 0, 0.1],
            range=[0, 1.92],
            damping=0.05,
            frictionloss=0.4,
            spring={"stiffness": 3, "reference": 0},
        )
        assert not schema_errors(bad_joint)
        checks.append("complete parent/joint/spring schema accepted")
        child["joint"]["axis"] = [0, 0, 2]
        assert any("axis must be a unit vector" in error for error in schema_errors(bad_joint))
        checks.append("joint unit axis enforced")
        cycle = copy.deepcopy(bad_joint)
        cycle["parts"][0]["parent"] = "lid"
        assert any("cyclic parent" in error for error in schema_errors(cycle))
        checks.append("cyclic parent chain rejected")
        unsafe = copy.deepcopy(metadata)
        unsafe["parts"][0]["texture"] = "../../outside.png"
        assert any("invalid texture filename" in error for error in schema_errors(unsafe))
        checks.append("unsafe file reference rejected")
    return checks


def check_bake_colors():
    with Image.open(QA / "multi_albedo.png") as image:
        atlas = np.asarray(image)
    results = {}
    for filename, expected in (("multi_metal.obj", [201, 205, 209]), ("multi_blue.obj", [40, 85, 117])):
        obj = read_obj(QA / filename)
        positions = obj.vertices[obj.faces]
        normals = np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0])
        top = np.flatnonzero(normals[:, 2] > 0.00005)[0]
        u, v = obj.uv[obj.uv_faces[top]].mean(axis=0)
        pixel = atlas[min(1023, int((1 - v) * 1024)), min(1023, int(u * 1024)), :3].astype(int)
        # Nearby geometry can darken the AO term slightly. Colour-only output
        # stays close to (and no brighter than) the source sRGB base colour.
        assert np.all(pixel <= np.asarray(expected) + 2) and np.all(
            pixel >= np.asarray(expected) * 0.93 - 2
        ), (filename, pixel, expected)
        results[filename] = pixel.tolist()
    return results


def check_mujoco():
    def path(name):
        return quoteattr(str(CUBE / name))

    xml = f"""<mujoco model="prop_pipeline_proof">
      <option timestep="0.002"/>
      <asset>
        <mesh name="visual" file={path("body.obj")}/>
        <mesh name="collision" file={path("body_col_00.obj")}/>
        <texture name="albedo" type="2d" file={path("body_albedo.png")}/>
        <material name="paint" texture="albedo" specular="0.25" shininess="0.3"/>
      </asset>
      <worldbody>
        <geom type="plane" size="1 1 0.01"/>
        <body pos="0 0 0.02">
          <freejoint/>
          <geom name="visual" type="mesh" mesh="visual" material="paint" contype="0" conaffinity="0" mass="0"/>
          <geom name="collision" type="mesh" mesh="collision" mass="0.1" friction="0.4 0.005 0.0001"/>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nmesh == 2 and model.ntex == 1
    assert model.mesh_texcoord.shape[0] > 0
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual")
    mesh_id = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    vertices = (
        model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    )
    assert np.allclose(np.ptp(vertices, axis=0), 0.1, atol=1e-6)
    assert np.allclose(vertices.min(axis=0), [-0.05, -0.05, 0.02], atol=1e-6)
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and abs(data.qpos[2]) < 0.001, data.qpos
    return {
        "version": mujoco.__version__,
        "meshes": model.nmesh,
        "textures": model.ntex,
        "steps": 500,
        "settled_body_z_m": float(data.qpos[2]),
    }


def main():
    QA.mkdir(parents=True, exist_ok=True)
    results = {
        "validator_regressions": validator_regressions(),
        "uv_checks": [
            check_uv([CUBE / "body.obj"]),
            check_uv([QA / "smart_uv.obj"]),
            check_uv([QA / "multi_metal.obj", QA / "multi_blue.obj"]),
        ],
        "bake_srgb_samples": check_bake_colors(),
        "mujoco": check_mujoco(),
    }
    (QA / "external_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print("EXTERNAL_CHECKS_PASS", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
