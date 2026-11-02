"""Isaac Sim check of a scene exported by isaac/export.py, the counterpart of
sim/bench_check.py:

    ISAAC_PYTHON=<Isaac Sim python> isaac/run.sh isaac.check <export folder>

It builds <folder>/scene.usdc with isaac/stage.py, opens it, and:
- settles the scene for 4 s with the arms hanging while the head pitches down,
  as the MuJoCo check does, and measures each free prop's drift (0.5 s to the
  end), the deepest contact, the robot's joint tracking and how far the jointed
  props' joints moved;
- renders the overview, the front portrait and the four policy cameras into
  check/preview.png, and stacks the MuJoCo check's preview above it in
  check/compare.png;
- poses the robot in the export's test posture and compares every robot body's
  world pose with MuJoCo's, which checks the joint frames, welds and chains;
and writes check/report.json.
"""

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from isaac import runtime

SETTLE_S, DRIFT_FROM_S = 4.0, 0.5
SIZE = (640, 480)
FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def font(size):
    from PIL import ImageFont

    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def minimum_jerk(x):
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def settle(rt, robot):
    """sim/bench_check.py's settle: zero posture, fingers open, head pitched
    down over its ramp; returns the settle's measurements."""
    steps = round(SETTLE_S / rt.dt)
    mark = round(DRIFT_FROM_S / rt.dt)
    start_free = start_props = None
    zero = {name: 0.0 for name in rt.index}
    wall = time.perf_counter()
    for step in range(steps):
        t = step * rt.dt
        targets = dict(zero)
        targets["head_pitch_joint"] = robot["head_pitch"] * minimum_jerk(
            min(t / robot["head_ramp_s"], 1.0)
        )
        rt.set_targets(targets)
        rt.step()
        if step + 1 == mark:
            start_free, start_props = rt.free_positions(), rt.prop_joints()
    wall = time.perf_counter() - wall
    end_free, end_props, q = rt.free_positions(), rt.prop_joints(), rt.joint_positions()
    arms = [f"{s}_{j}_joint" for s in ("left", "right") for j in robot["arm_joints"]]
    depth, pair = rt.deepest_contact()
    return {
        "realtime_factor": round(SETTLE_S / wall, 3),
        "max_penetration_mm": round(depth * 1000, 3),
        "deepest_contact": pair,
        "free_drift_mm": {
            name: round(
                1000 * float(np.linalg.norm(end_free[name] - start_free[name])), 2
            )
            for name in end_free
        },
        "prop_joint_travel": {
            name: round(float(np.abs(end_props[name] - start_props[name]).max()), 5)
            for name in end_props
            if end_props[name].size
        },
        "arm_tracking_mrad": round(1000 * max(abs(q[j]) for j in arms), 3),
        "head_pitch_rad": round(q["head_pitch_joint"], 4),
    }


def parity(rt, model):
    """Every robot body's world pose in the test posture, against MuJoCo's."""
    rt.set_positions(model["test"]["qpos"])
    rt.update_kinematics()
    links = rt.link_poses()
    worst_p, worst_r = (0.0, None), (0.0, None)
    from isaac.stage import Pose

    for name, info in rt.paths["bodies"].items():
        p, q = links[Path(info["link"]).name]
        pose = Pose.of(p, q) * Pose.of(info["pos"], info["quat"])
        ref = model["test"]["bodies"][name]
        dp = float(np.linalg.norm(pose.p - np.asarray(ref["xpos"])))
        R = pose.R.T @ Pose.of(ref["xpos"], ref["xquat"]).R
        dr = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
        if dp > worst_p[0]:
            worst_p = (dp, name)
        if dr > worst_r[0]:
            worst_r = (dr, name)
    return {
        "bodies": len(rt.paths["bodies"]),
        "max_position_error_mm": round(worst_p[0] * 1000, 4),
        "worst_position_body": worst_p[1],
        "max_rotation_error_deg": round(worst_r[0], 4),
        "worst_rotation_body": worst_r[1],
    }


def gain_check(rt):
    """Largest relative gap between the gains PhysX parsed from the USD drives
    and the SI gains stage.py meant."""
    gaps = []
    for parsed, meant in zip(rt.parsed_gains, rt.gains):
        mask = meant > 0
        gaps.append(float(np.max(np.abs(parsed[mask] - meant[mask]) / meant[mask])))
    return {"stiffness": round(gaps[0], 6), "damping": round(gaps[1], 6)}


def sheet(images, title, badge):
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (1920, 1000), (20, 27, 38))
    draw = ImageDraw.Draw(canvas)
    label = font(18)
    for k, (name, rgb) in enumerate(images.items()):
        x, y = 640 * (k % 3), 40 + 480 * (k // 3)
        canvas.paste(Image.fromarray(rgb), (x, y))
        draw.text((x + 8, y + 6), name, font=label, fill="white")
    draw.text((10, 10), title, font=label, fill="white")
    draw.text((1700, 10), badge, font=label, fill=(128, 218, 233))
    return canvas


def compare(folder, isaac_sheet):
    from PIL import Image, ImageDraw

    reference = folder / "mujoco" / "preview.png"
    if not reference.exists():
        return None
    top = Image.open(reference).convert("RGB")
    ImageDraw.Draw(top).text((1700, 10), "MuJoCo", font=font(18), fill=(255, 210, 120))
    canvas = Image.new("RGB", (1920, top.height + isaac_sheet.height))
    canvas.paste(top, (0, 0))
    canvas.paste(isaac_sheet, (0, top.height))
    return canvas


def check(folder):
    import importlib.metadata

    from PIL import Image

    from isaac import stage

    out = folder / "check"
    out.mkdir(exist_ok=False)
    model = json.loads((folder / "model.json").read_text())
    robot = model["robot"]
    views = ["overview", "portrait_front", *robot["policy_cameras"]]
    t0 = time.perf_counter()
    usd = stage.build(folder)
    t1 = time.perf_counter()
    runtime.log(
        f"built {usd.name} ({usd.stat().st_size / 1e6:.1f} MB) in {t1 - t0:.1f} s"
    )
    rt = runtime.Runtime(folder, views=views, size=SIZE)
    t2 = time.perf_counter()
    runtime.log(f"loaded in {t2 - t1:.1f} s: {len(rt.index)} robot dofs")
    task = model["task"]
    report = {
        "scene": task["scene"],
        "instruction": task["instruction"],
        "isaacsim": importlib.metadata.version("isaacsim"),
        "usd_mb": round(usd.stat().st_size / 1e6, 1),
        "build_s": round(t1 - t0, 1),
        "load_s": round(t2 - t1, 1),
        "robot_dofs": len(rt.index),
        "coupled_dofs": len(rt.coupled),
        "articulations": list(rt.paths["articulations"]),
        "free_bodies": rt.free_names,
        "parsed_gain_gap": gain_check(rt),
    }
    report["settle"] = settle(rt, robot)
    runtime.log(f"settled: {json.dumps(report['settle'])}")
    t3 = time.perf_counter()
    images = rt.render()
    report["first_render_s"] = round(time.perf_counter() - t3, 1)
    runtime.log(f"rendered {len(images)} views in {report['first_render_s']} s")
    for name, rgb in images.items():
        Image.fromarray(rgb).save(out / f"{name}.png")
    preview = sheet(images, f"{task['scene']}: {task['instruction']}", "Isaac Sim 5.1")
    preview.save(out / "preview.png")
    stacked = compare(folder, preview)
    if stacked is not None:
        stacked.save(out / "compare.png")
    report["parity"] = parity(rt, model)
    mujoco_report = folder / "mujoco" / "report.json"
    if mujoco_report.exists():
        ref = json.loads(mujoco_report.read_text())
        report["mujoco"] = {
            k: ref.get(k)
            for k in ("settle_error", "max_penetration_mm", "free_drift_mm")
        }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Check an exported scene in Isaac Sim."
    )
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    app = runtime.start_app()
    try:
        report = check(args.folder.resolve())
        summary = {
            "scene": report["scene"],
            "max_drift_mm": max(report["settle"]["free_drift_mm"].values(), default=0),
            "max_penetration_mm": report["settle"]["max_penetration_mm"],
            "parity_mm": report["parity"]["max_position_error_mm"],
            "realtime_factor": report["settle"]["realtime_factor"],
        }
        print("ISAAC_CHECK", json.dumps(summary), flush=True)
    except Exception:
        # SimulationApp.close() ends the process before Python prints it.
        traceback.print_exc()
        print("ISAAC_CHECK_FAILED", flush=True)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
