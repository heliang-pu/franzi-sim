"""Scene check shared by the industrial benchmark scenes (scenes/<task>/check.py).

It builds the scene in a plain Scene (no expert) and settles it for 4 s with
the arms hanging while the head pitches down. It then checks:
- the Scene audits: no contact deeper than 1 mm, no joint past its limit;
- that every free prop rests (drift between 0.5 s and the end of the settle);
- that each of the task's KEY_POINTS is reachable by its arm, with the shared
  IK at the planning margin, as a top-down grasp, a 45-degree forward grasp
  (sim.tabletop.grasp_rotation), or either;
and renders the overview and the four policy cameras into preview.png. The
report goes to report.json beside scene.xml.

A task module provides NAME, INSTRUCTION, build() -> (root, report) and
KEY_POINTS: [{"name", "side", "xyz", "grasp": "top" | "forward" | "any"}].
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot_kinematics import Kinematics
from sim.camera_rig import POLICY_CAMERAS
from sim.episode import Scene
from sim.tabletop import grasp_rotation

SETTLE_S = 4.0
VIEWS = ("overview", "portrait_front", *POLICY_CAMERAS)


def reachable(kin, point):
    rotations = {"top": [None], "forward": [grasp_rotation(point["side"])]}
    rotations["any"] = rotations["top"] + rotations["forward"]
    found = []
    for rotation in rotations[point.get("grasp", "any")]:
        try:
            kin.ik(point["side"], point["xyz"], np.zeros(7), rotation=rotation)
            found.append("top" if rotation is None else "forward")
        except RuntimeError:
            pass
    return found


def check(task, output):
    output.mkdir(parents=True, exist_ok=False)

    class TaskScene(Scene):
        def build_root(self):
            return task.build()

    scene = TaskScene(output)
    m, d = scene.m, scene.d
    free = [
        m.body(m.jnt_bodyid[j]).name
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    report = {
        "scene": task.NAME,
        "instruction": task.INSTRUCTION,
        "bodies": m.nbody,
        "geoms": m.ngeom,
        "meshes": m.nmesh,
        "textures": m.ntex,
    }
    start = None
    error = None
    try:
        for step in range(int(SETTLE_S / m.opt.timestep)):
            scene.physics_step()
            if step == int(0.5 / m.opt.timestep):
                start = {b: d.xpos[m.body(b).id].copy() for b in free}
    except RuntimeError as exc:
        error = str(exc)
    report["settle_error"] = error
    report["max_penetration_mm"] = round(scene.maxdepth * 1000, 3)
    report["deepest_contact"] = scene.deepest
    report["max_limit_error"] = round(scene.maxlimit, 5)
    if start:
        report["free_drift_mm"] = {
            b: round(1000 * float(np.linalg.norm(d.xpos[m.body(b).id] - start[b])), 2)
            for b in free
        }
    kin = Kinematics()
    report["key_points"] = {
        p["name"]: {
            "side": p["side"],
            "xyz": p["xyz"],
            "grasp": p.get("grasp", "any"),
            "reachable_as": reachable(kin, p),
        }
        for p in task.KEY_POINTS
    }
    renderer = mujoco.Renderer(m, height=480, width=640)
    tiles = []
    for cam in VIEWS:
        renderer.update_scene(d, camera=cam)
        tiles.append((cam, Image.fromarray(renderer.render())))
    sheet = Image.new("RGB", (1920, 1000), (20, 27, 38))
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    draw = ImageDraw.Draw(sheet)
    for k, (cam, image) in enumerate(tiles):
        x, y = 640 * (k % 3), 40 + 480 * (k // 3)
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + 6), cam, font=font, fill="white")
    draw.text((10, 10), f"{task.NAME}: {task.INSTRUCTION}", font=font, fill="white")
    sheet.save(output / "preview.png")
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    return report


def main(task):
    parser = argparse.ArgumentParser(description=f"Check the {task.NAME} scene.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(task, args.output)
    unreachable = [n for n, k in report["key_points"].items() if not k["reachable_as"]]
    drift = max((report.get("free_drift_mm") or {"none": 0}).values())
    print(
        json.dumps(
            {
                "scene": report["scene"],
                "settle_error": report["settle_error"],
                "max_penetration_mm": report["max_penetration_mm"],
                "max_free_drift_mm": drift,
                "unreachable": unreachable,
            },
            ensure_ascii=False,
        )
    )
