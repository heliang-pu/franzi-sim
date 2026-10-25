"""Acceptance checks shared by the object scenes (sim/tabletop), across seeds.

For each seed it runs the scene's expert headless and keeps the run. For the
first seeds it also counts, from segmentation renders, the unoccluded pixels
of every prop in the head and body cameras: once the head gaze has settled
with the arms still hanging, and again at the ready pose. The first seeds also
get preview sheets (overview plus the four policy cameras at the ready pose).
Writes report.json; the per-run files are as the scene's run.py writes them.
"""

import argparse
import functools
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim import tabletop as tt
from sim.camera_rig import POLICY_CAMERAS

VIEW_CAMERAS = ("head_d435", "body_d435")


def prop_groups(m, props, plain=()):
    """Geom ids per prop: its <prop>_visual* geoms, and each plain geom (one
    named geom each)."""
    names = [m.geom(g).name for g in range(m.ngeom)]
    found = {name: [names.index(name)] for name in plain}
    for prop in props:
        found[prop] = [
            g for g, n in enumerate(names) if n.startswith(prop) and "_visual" in n
        ]
    return found


def pixel_counts(scene, renderer, groups):
    """Unoccluded pixels of each group, per camera."""
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


def views(output, spec, scene_cls, props, plain, preview):
    """Settling and visibility for spec: object drift while the gaze settles,
    and pixel counts then and at the ready pose. Optionally saves a preview."""
    output.mkdir(parents=True, exist_ok=False)
    scene = scene_cls(output, spec)
    renderer = mujoco.Renderer(scene.m, 480, 640)
    wide = mujoco.Renderer(scene.m, 720, 1280) if preview else None
    groups = prop_groups(scene.m, props, plain)
    try:
        start = {n: np.array(scene.xyz(n)[:2]) for n in scene.OBJECTS}
        scene.stage("settle", "Hold still while the head gaze settles.")
        scene.move(seconds=3.5)
        result = {
            "max_xy_drift_after_settle_mm": max(
                1000 * float(np.linalg.norm(scene.xyz(n)[:2] - start[n]))
                for n in scene.OBJECTS
            ),
            "max_speed_after_settle_m_s": max(
                scene.object_info(n)["velocity_m_s"] for n in scene.OBJECTS
            ),
            "settled": pixel_counts(scene, renderer, groups),
        }
        scene.stage("ready", "Raise both open grippers.")
        tt.go_ready(scene, {s: tt.grip_open(0.06) for s in tt.SIDES})
        scene.move(seconds=0.5)
        result["ready"] = pixel_counts(scene, renderer, groups)
        if preview:
            title = f"seed {spec['seed']} | ready pose | {spec['instruction']}"
            sheet(scene, renderer, wide, title).save(
                output.parent / f"preview_seed{spec['seed']:03d}.png"
            )
    finally:
        renderer.close()
        if wide:
            wide.close()
        scene.close()
    return result


def main(
    description,
    sample_episode,
    run,
    scene_cls,
    props,
    plain=(),
    fields=None,
    variants=None,
):
    """The check CLI: run(output, seed, kin=...) runs one episode and returns
    its summary; fields(summary) adds per-episode report fields. With variants,
    --variant picks one, passed to sample_episode and run as variant=."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=20, help="expert episodes")
    p.add_argument("--views", type=int, default=4, help="seeds with pixel counts")
    p.add_argument("--previews", type=int, default=2, help="seeds with a sheet")
    if variants:
        p.add_argument("--variant", choices=variants, default=variants[0])
    args = p.parse_args()
    if variants:
        sample_episode = functools.partial(sample_episode, variant=args.variant)
        run = functools.partial(run, variant=args.variant)
    args.output.mkdir(parents=True, exist_ok=False)
    kin = tt.kinematics(tt.SAMPLE_MARGIN)
    episodes = []
    for seed in range(args.seeds):
        r = run(args.output / "episodes" / f"seed_{seed:03d}", seed, kin=kin)
        episodes.append(
            {
                "seed": seed,
                "success": r["success"],
                "progress": r["progress"],
                "milestones": len(r["milestones"]),
                "error": r["error"],
                "draws": r["draws"],
                "duration_s": r["duration_s"],
                "wall_time_s": r["wall_time_s"],
                "max_penetration_mm": r["physics"]["max_penetration_mm"],
                "max_joint_limit_error_rad": r["physics"]["max_joint_limit_error"],
                **(fields(r) if fields else {}),
            }
        )
        print(json.dumps(episodes[-1]), flush=True)
    seen = {}
    for seed in range(args.views):
        spec = sample_episode(seed, kin)
        seen[seed] = views(
            args.output / "views" / f"seed_{seed:03d}",
            spec,
            scene_cls,
            props,
            plain,
            seed < args.previews,
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
        "milestones_total": sum(e["milestones"] for e in episodes),
        "duration_s": [
            min(e["duration_s"] for e in episodes),
            max(e["duration_s"] for e in episodes),
        ],
        "max_penetration_mm": max(e["max_penetration_mm"] for e in episodes),
        "max_joint_limit_error_rad": max(
            e["max_joint_limit_error_rad"] for e in episodes
        ),
        "draws": [e["draws"] for e in episodes],
        "max_xy_drift_after_settle_mm": max(
            (v["max_xy_drift_after_settle_mm"] for v in seen.values()), default=None
        ),
        "min_pixels": {
            pose: {
                cam: min(min(v[pose][cam].values()) for v in seen.values())
                for cam in VIEW_CAMERAS
            }
            for pose in ("settled", "ready")
        }
        if seen
        else None,
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
