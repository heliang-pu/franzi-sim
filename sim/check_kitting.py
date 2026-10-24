"""Acceptance checks for the sort-and-kit scene across seeds.

For each seed it runs the expert headless (run_kitting.run) and keeps the run.
For the first seeds it also counts, from segmentation renders, the unoccluded
pixels of every part and of every cell frame in the head and body cameras: once
the head gaze has settled with the arms still hanging, and again at the ready
pose. The first seeds also get preview sheets (overview plus the four policy
cameras at the ready pose). Writes report.json; the per-run files are as
run_kitting writes them.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim import kitting_task as kt
from sim.camera_rig import POLICY_CAMERAS
from sim.run_kitting import KitScene, run

VIEW_CAMERAS = ("head_d435", "body_d435")
FRAME_BARS = ("near", "far", "inner", "outer")


def pixel_counts(scene, renderer):
    """Unoccluded pixels of each part and each cell frame, per camera."""
    groups = {n: [scene.m.geom(n + "_part").id] for n in scene.parts}
    for c in scene.cells:
        groups[c] = [scene.m.geom(f"{c}_frame_{b}").id for b in FRAME_BARS]
    renderer.enable_segmentation_rendering()
    counts = {}
    for cam in VIEW_CAMERAS:
        renderer.update_scene(scene.d, camera=cam)
        seg = renderer.render()
        ids = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        counts[cam] = {n: int(np.isin(ids, g).sum()) for n, g in groups.items()}
    renderer.disable_segmentation_rendering()
    return counts


def sheet(scene, renderer, wide, title):
    """Overview (1280 x 720) above the four policy cameras (640 x 480 each)."""
    canvas = Image.new("RGB", (1280, 720 + 960), (20, 27, 38))
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    draw = ImageDraw.Draw(canvas)
    wide.update_scene(scene.d, camera="overview")
    canvas.paste(Image.fromarray(wide.render()), (0, 0))
    draw.rectangle((0, 0, 1280, 40), fill=(20, 27, 38))
    draw.text((16, 9), title, font=font, fill="white")
    for i, cam in enumerate(POLICY_CAMERAS):
        renderer.update_scene(scene.d, camera=cam)
        x, y = (i % 2) * 640, 720 + (i // 2) * 480
        canvas.paste(Image.fromarray(renderer.render()), (x, y))
        draw.rectangle((x, y, x + 640, y + 34), fill=(20, 27, 38))
        draw.text((x + 12, y + 7), cam, font=font, fill="white")
    return canvas


def views(output, seed, kin, preview):
    """Settling and visibility for seed: part drift while the gaze settles, and
    pixel counts then and at the ready pose. Optionally saves a preview."""
    spec = kt.sample_episode(seed, kin)
    output.mkdir(parents=True, exist_ok=False)
    scene = KitScene(output, spec)
    renderer = mujoco.Renderer(scene.m, 480, 640)
    wide = mujoco.Renderer(scene.m, 720, 1280) if preview else None
    try:
        scene.stage("settle", "Hold still while the head gaze settles.")
        scene.move(seconds=3.5)
        drift = {
            p["name"]: 1000
            * float(
                np.linalg.norm(
                    np.subtract(scene.object_info(p["name"])["xyz"][:2], p["xy"])
                )
            )
            for p in spec["parts"]
        }
        result = {
            "max_xy_drift_after_settle_mm": max(drift.values()),
            "max_speed_after_settle_m_s": max(
                scene.object_info(n)["velocity_m_s"] for n in scene.parts
            ),
            "settled": pixel_counts(scene, renderer),
        }
        scene.stage("ready", "Raise both open grippers behind the kit trays.")
        scene.move(
            spec["ready"],
            grips={s: kt.grip_open(0.04) for s in kt.SIDES},
            seconds=3.0,
            rotations={s: kt.grasp_rotation(s) for s in kt.SIDES},
            speed=kt.SPEED,
        )
        scene.move(seconds=0.5)
        result["ready"] = pixel_counts(scene, renderer)
        if preview:
            title = f"seed {seed} | ready pose | {spec['instruction']}"
            sheet(scene, renderer, wide, title).save(
                output.parent / f"preview_seed{seed:03d}.png"
            )
    finally:
        renderer.close()
        if wide:
            wide.close()
        scene.close()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=20, help="expert episodes")
    p.add_argument("--views", type=int, default=6, help="seeds with pixel counts")
    p.add_argument("--previews", type=int, default=3, help="seeds with a sheet")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    kin = kt.kinematics(kt.SAMPLE_MARGIN)
    episodes = []
    for seed in range(args.seeds):
        r = run(args.output / "episodes" / f"seed_{seed:03d}", seed, kin=kin)
        episodes.append(
            {
                "seed": seed,
                "success": r["success"],
                "progress": r["progress"],
                "error": r["error"],
                "draws": r["draws"],
                "duration_s": r["duration_s"],
                "wall_time_s": r["wall_time_s"],
                "max_penetration_mm": r["physics"]["max_penetration_mm"],
                "max_joint_limit_error_rad": r["physics"]["max_joint_limit_error"],
                "max_offset_from_cell_mm": max(
                    r["offset_from_assigned_cell_mm"].values()
                ),
                "stray_contact_ticks": r["stray_contact_ticks"],
            }
        )
        print(json.dumps(episodes[-1]), flush=True)
    seen = {}
    for seed in range(args.views):
        seen[seed] = views(
            args.output / "views" / f"seed_{seed:03d}", seed, kin, seed < args.previews
        )
        print(f"views seed {seed} done", flush=True)
    low = [
        {"seed": seed, "pose": pose, "camera": cam, "object": name, "pixels": n}
        for seed, v in seen.items()
        for pose in ("settled", "ready")
        for cam, counts in v[pose].items()
        for name, n in counts.items()
        if n < 150
    ]
    report = {
        "episodes": len(episodes),
        "successes": sum(e["success"] for e in episodes),
        "progress_total": sum(e["progress"] for e in episodes),
        "parts_total": 6 * len(episodes),
        "duration_s": [
            min(e["duration_s"] for e in episodes),
            max(e["duration_s"] for e in episodes),
        ],
        "max_penetration_mm": max(e["max_penetration_mm"] for e in episodes),
        "max_joint_limit_error_rad": max(
            e["max_joint_limit_error_rad"] for e in episodes
        ),
        "max_offset_from_cell_mm": max(
            (e["max_offset_from_cell_mm"] for e in episodes if e["success"]),
            default=None,
        ),
        "stray_contact_ticks": sum(e["stray_contact_ticks"] for e in episodes),
        "draws": [e["draws"] for e in episodes],
        "max_xy_drift_after_settle_mm": max(
            v["max_xy_drift_after_settle_mm"] for v in seen.values()
        ),
        "min_pixels": {
            pose: {
                cam: min(min(v[pose][cam].values()) for v in seen.values())
                for cam in VIEW_CAMERAS
            }
            for pose in ("settled", "ready")
        },
        "below_150_pixels": low,
        "per_episode": episodes,
        "views": seen,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("per_episode", "views")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
