"""Load the delivered Task 07 OBJ assets in MuJoCo and check joint behaviour.

Run with /Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -B.
The in-memory fixture is a geometry/joint smoke test, not a robot benchmark.
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco
import numpy as np

from validate_props import read_obj

ROOT = HERE.parents[1] / "assets" / "props"
IDS = ("lidded_tote", "station_plate", "drawer_cabinet")


def numbers(values):
    return " ".join(format(float(value), ".12g") for value in values)


def compile_prop(folder, data):
    root = ET.Element("mujoco", model=data["id"])
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    elements = {}
    for part in data["parts"]:
        name = part["name"]
        parent = elements[part["parent"]] if part["parent"] else world
        body = ET.SubElement(parent, "body", name=name)
        elements[name] = body
        visual = read_obj(folder / part["visual"])
        lo, hi = visual.vertices.min(axis=0), visual.vertices.max(axis=0)
        center = (lo + hi) / 2
        extents = hi - lo
        # Conservative positive fixture inertia; prescribed mass is kept exact.
        inertia = part["mass_kg"] / 12 * (np.sum(extents ** 2) - extents ** 2)
        ET.SubElement(body, "inertial", pos=numbers(center), mass=str(part["mass_kg"]),
                      diaginertia=numbers(np.maximum(inertia, 1e-9)))
        joint = part["joint"]
        if joint:
            attributes = {"name": name + "_joint", "type": joint["type"],
                          "pos": numbers(joint["anchor_m"]), "axis": numbers(joint["axis"]),
                          "range": numbers(joint["range"]), "limited": "true", "damping": str(joint["damping"]),
                          "frictionloss": str(joint["frictionloss"])}
            for key in ("solimpfriction", "solreffriction"):
                if key in joint:
                    attributes[key] = numbers(joint[key])
            ET.SubElement(body, "joint", **attributes)
        for index, reference in enumerate([part["visual"], *part["collision"]]):
            mesh_name = name + "_" + str(index)
            ET.SubElement(assets, "mesh", name=mesh_name, file=str(folder / reference))
            ET.SubElement(body, "geom", name=mesh_name, type="mesh", mesh=mesh_name, mass="0",
                          contype="0" if index == 0 else "1", conaffinity="0" if index == 0 else "1",
                          friction=f"{part['friction']} .005 .0001", group="2" if index == 0 else "3")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=IDS, default=list(IDS))
    args = parser.parse_args()
    report = {}
    for prop_id in args.props:
        folder = ROOT / prop_id
        data = json.loads((folder / "prop.json").read_text())
        checks = {}

        def check(name, condition, evidence):
            checks[name] = {"pass": bool(condition), "evidence": evidence}

        model, state = compile_prop(folder, data)
        expected_joints = sum(p["joint"] is not None for p in data["parts"])
        check("all_OBJ_meshes_compile_in_MuJoCo", model.nmesh == sum(1 + len(p["collision"]) for p in data["parts"]),
              {"meshes": model.nmesh, "joints": model.njnt, "mujoco": mujoco.__version__})
        check("joint_count", model.njnt == expected_joints, model.njnt)
        mujoco.mj_forward(model, state)
        for part in data["parts"]:
            name, joint = part["name"], part["joint"]
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            check(name + "_mass", abs(model.body_mass[body_id] - part["mass_kg"]) < 1e-9,
                  float(model.body_mass[body_id]))
            visual = read_obj(folder / part["visual"]).mesh()
            check(name + "_export_manifold", visual.is_watertight and visual.is_winding_consistent,
                  {"watertight": bool(visual.is_watertight), "consistent_winding": bool(visual.is_winding_consistent)})
            if not joint:
                continue
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name + "_joint")
            qadr, dadr = model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id]
            check(name + "_joint_contract", np.allclose(model.jnt_range[joint_id], joint["range"]) and
                  np.allclose(model.jnt_axis[joint_id], joint["axis"]) and
                  abs(model.dof_damping[dadr] - joint["damping"]) < 1e-12 and
                  abs(model.dof_frictionloss[dadr] - joint["frictionloss"]) < 1e-12,
                  {"range": model.jnt_range[joint_id].tolist(), "axis": model.jnt_axis[joint_id].tolist(),
                   "damping": float(model.dof_damping[dadr]), "frictionloss": float(model.dof_frictionloss[dadr])})
            if joint["type"] == "slide":
                mujoco.mj_resetData(model, state)
                mujoco.mj_forward(model, state)
                initial = state.xpos[body_id].copy()
                state.qpos[qadr] = joint["range"][1]
                mujoco.mj_forward(model, state)
                displacement = state.xpos[body_id] - initial
                check(name + "_180mm_minus_X", np.allclose(displacement, [-.18, 0, 0], atol=1e-9),
                      displacement.tolist())
            else:
                drifts = []
                for degrees in (3, 30, 60, 90, 109):
                    mujoco.mj_resetData(model, state)
                    initial = math.radians(degrees)
                    state.qpos[qadr] = initial
                    for _ in range(1000):
                        mujoco.mj_step(model, state)
                    drifts.append(abs(float(state.qpos[qadr]) - initial))
                check(name + "_friction_holds_under_gravity", max(drifts) < math.radians(.05),
                      {"seconds_each": 2, "angles_deg": [3, 30, 60, 90, 109],
                       "drift_deg": [math.degrees(drift) for drift in drifts],
                       "fixture": "prescribed mass and conservative box inertia; no robot or catch actuator"})
                check(name + "_no_unrequested_full_range_spring", model.jnt_stiffness[joint_id] == 0,
                      float(model.jnt_stiffness[joint_id]))
        report[prop_id] = checks
        print("TASK07_MUJOCO", prop_id, sum(c["pass"] for c in checks.values()), "/", len(checks), flush=True)
    (HERE / "task07_export_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    failures = [(prop_id, name, value["evidence"]) for prop_id, checks in report.items()
                for name, value in checks.items() if not value["pass"]]
    for failure in failures:
        print("FAIL", *failure)
    if failures:
        raise RuntimeError(f"{len(failures)} exported asset checks failed")


if __name__ == "__main__":
    main()
