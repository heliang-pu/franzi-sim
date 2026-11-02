"""Scene check shared by the industrial benchmark scenes (scenes/<task>/check.py).

It builds the scene in a plain Scene (no expert) and settles it for 4 s with
the arms hanging while the head pitches down. It then checks:
- the Scene audits: no contact deeper than 1 mm, no joint past its limit;
- that every free prop rests (drift between 0.5 s and the end of the settle);
- that each of the task's KEY_POINTS is reachable by its arm, with the shared
  IK at the planning margin, as a top-down grasp, a 45-degree forward grasp
  (sim.tabletop.grasp_rotation), or either, solved from the hanging arm, the
  default posture or the arm's key points solved before it;
- that each key point with a width is graspable: the object fits the jaws, and
  the arm posed there by the IK, its gripper open as an expert opens it
  (sim.tabletop.grip_open), touches nothing but the robot in the settled scene
  (a static collision test in a copy of the state);
and renders the overview and the four policy cameras into preview.png. The
report goes to report.json beside scene.xml.

A task module provides NAME, INSTRUCTION, build() -> (root, report) and
KEY_POINTS: [{"name", "side", "xyz", "grasp": "top" | "forward" | "any"}], xyz
being the grasp site's target (the open pads' centre). Optional keys: "yaw"
turns the top-down hand about z (0 closes the jaws along x), "psi" is the
45-degree grasp's azimuth, and "width" (m), the object's size across the jaws,
asks for the grasp probe. "ignore" lists bodies the task will have moved
out of the way by then (a lid or a door it opens first). Points where the
hand is meant to touch something (buttons, push faces) carry no width. Reach
alone once let palletizing and parcel sorting pass with objects wider than
the jaws.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from robot_kinematics import Kinematics
from sim.build_task import ARM_JOINTS
from sim.camera_rig import POLICY_CAMERAS
from sim.episode import Scene, overlay_font
from sim.gripper_linkage import set_closure
from sim.tabletop import JAW_SPAN, READY, grasp_rotation, grip_open, rot_z

SETTLE_S = 4.0
VIEWS = ("overview", "portrait_front", *POLICY_CAMERAS)
MIN_ROOM = 0.004  # least room per side between the open jaws and the object
GRAZE = 0.0002  # contacts shallower than this are the posed hand grazing


def orientations(point):
    """(label, hand rotation) pairs to try for a key point."""
    top = ("top", rot_z(point.get("yaw", 0.0)))
    forward = ("forward", grasp_rotation(point["side"], point.get("psi", 0.0)))
    return {"top": [top], "forward": [forward], "any": [top, forward]}[
        point.get("grasp", "any")
    ]


def reachable(kin, point, starts):
    """{label: arm joint values} for each orientation the IK reaches, from the
    first of starts (present arm poses, in order) that solves it."""
    found = {}
    for label, rotation in orientations(point):
        for start in starts:
            try:
                found[label] = kin.ik(
                    point["side"], point["xyz"], start, rotation=rotation
                )
                break
            except RuntimeError:
                pass
    return found


def probe(m, d, side, q, width, ignore=()):
    """Contacts of arm `side`, posed at joint values q with its gripper open
    for width, with anything that is not the robot or in ignore (body names):
    [{"body", "depth_mm"}]."""
    test = mujoco.MjData(m)
    test.qpos[:] = d.qpos
    for joint, value in zip(ARM_JOINTS, q, strict=True):
        test.qpos[m.joint(f"{side}_{joint}_joint").qposadr[0]] = value
    set_closure(m, test, side, max(grip_open(width), 0.0))
    mujoco.mj_forward(m, test)

    def arm(body):
        name = m.body(body).name
        return name.startswith(side) and "front" not in name

    def scene(body):
        return body == 0 or m.body(body).name.endswith("_part")

    deepest = {}
    for c in test.contact[: test.ncon]:
        b1, b2 = (int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2))
        other = b2 if arm(b1) else b1 if arm(b2) else None
        if other is None or not scene(other) or -c.dist < GRAZE:
            continue
        name = m.body(other).name
        if name in ignore:
            continue
        deepest[name] = max(deepest.get(name, 0.0), -float(c.dist))
    return [
        {"body": name, "depth_mm": round(1000 * depth, 2)}
        for name, depth in sorted(deepest.items(), key=lambda kv: -kv[1])
    ]


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
    # The nominal layout's instruction: seeded scenes word it per episode.
    instruction = scene.setup["task"]["instruction"]
    report = {
        "scene": task.NAME,
        "instruction": instruction,
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
    # The IK starts from the hanging arm, then from the default posture
    # (tabletop.start_home) and from the side's key points solved so far, as an
    # expert chains its moves. With the IK's seeds in the new joint convention
    # (2026-10-25), shaped_kitting's gear pocket solves only from the default
    # posture, and the release 2 cm below it only from the pocket.
    home = {s: kin.ik(s, READY[s], np.zeros(7)) for s in ("left", "right")}
    solved = {"left": [], "right": []}
    report["key_points"] = {}
    for p in task.KEY_POINTS:
        side = p["side"]
        found = reachable(kin, p, [np.zeros(7), home[side], *solved[side]])
        solved[side] += found.values()
        entry = {
            "side": p["side"],
            "xyz": p["xyz"],
            "grasp": p.get("grasp", "any"),
            "reachable_as": list(found),
        }
        if "width" in p:
            # The first reachable orientation, in the order orientations() tries.
            label, q = next(iter(found.items()), (None, None))
            contacts = (
                probe(m, d, p["side"], q, p["width"], p.get("ignore", ()))
                if label
                else []
            )
            fits = p["width"] + 2 * MIN_ROOM <= JAW_SPAN
            entry["grasp_probe"] = {
                "width_mm": round(1000 * p["width"], 1),
                "fits_jaws": fits,
                "as": label,
                "contacts": contacts,
                "ok": bool(label) and fits and not contacts,
            }
        report["key_points"][p["name"]] = entry
    renderer = mujoco.Renderer(m, height=480, width=640)
    tiles = []
    for cam in VIEWS:
        renderer.update_scene(d, camera=cam)
        tiles.append((cam, Image.fromarray(renderer.render())))
    sheet = Image.new("RGB", (1920, 1000), (20, 27, 38))
    font = overlay_font(18)
    draw = ImageDraw.Draw(sheet)
    for k, (cam, image) in enumerate(tiles):
        x, y = 640 * (k % 3), 40 + 480 * (k // 3)
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + 6), cam, font=font, fill="white")
    draw.text((10, 10), f"{task.NAME}: {instruction}", font=font, fill="white")
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
    points = report["key_points"]
    unreachable = [n for n, k in points.items() if not k["reachable_as"]]
    ungraspable = [
        n for n, k in points.items() if not k.get("grasp_probe", {"ok": True})["ok"]
    ]
    drift = max((report.get("free_drift_mm") or {"none": 0}).values())
    print(
        json.dumps(
            {
                "scene": report["scene"],
                "settle_error": report["settle_error"],
                "max_penetration_mm": report["max_penetration_mm"],
                "max_free_drift_mm": drift,
                "unreachable": unreachable,
                "ungraspable": ungraspable,
            },
            ensure_ascii=False,
        )
    )
