"""Convert recorded expert runs (scenes.pick_place.run, scenes.kitting.run or a
RoboSyn scene's run, with --record) into raw VLA episodes.

    python -m data.convert_expert_run outputs/expert_pinocchio --output outputs/<new_episode>
    python -m data.convert_expert_run outputs/<runs>/seed_* --output outputs/<new_folder>

With several runs, --output is a new folder holding one episode per run, named
after the run; failed runs are skipped unless --allow-failed.

The expert commands MuJoCo in URDF radians and gripper closure fractions. The
episode uses the SDK convention of vla_episode.py, so a policy trained on it
commands the robot through the stock SDK unchanged:

- joints: robots.franzi.kinematics.Kinematics().to_sdk, q_sdk = sign * q_urdf + offset;
- grippers: 1000 * (1 - closure), rounded and clipped to 0..1000 as simbridge's
  GripperReadPosition reports it (simbridge.robot.GripperMap), for the measured
  closure (state) and the commanded one (action) alike;
- action[k] is the target the expert applies at tick k and holds for one tick;
  setpoint[k] is action[k-1], the target its servos track when state[k] is
  measured (the measured joints at k = 0);
- robot_s is the trace's simulated time. Each camera frame was rendered at the
  same physics step as the state, so camera_seq = k + 1; there is no wall clock,
  so the UTC fields are -1;
- the episode instruction is the run's own (summary.json "instruction", one per
  scenes.kitting.run seed), else KITTING_TASK; --task overrides both. Subtask texts
  come from trace.json. The camera videos are copied unchanged. Ground-truth
  object state (trace.json "objects", summary.json placements) stays out of
  frames.npz.

Runs with the full log (sim/episode.py, 2026-10-25) give schema-2 episodes:
- state_full and action_full: the 16 above, then the waist and head joints
  through their franzi_sdk.yaml unit mappings (simbridge.robot.load_robot);
- velocity_full and effort_full with each joint's SDK sign; the grippers' as
  position units/s (-1000 times the closure rate) and squeeze force (N);
- eef_pose and camera_pose as logged, the cameras in vla_episode.CAMERAS order;
- extras/: the recording's depth and segmentation videos and segmentation.json,
  renamed to the SDK camera names, and the world state for evaluation only
  (eval.npz: object_pose, object_velocity, prop_qpos and their names;
  eval_contacts.json).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from data.vla_episode import (
    CAMERAS,
    FPS,
    HEIGHT,
    KITTING_TASK,
    WIDTH,
    EpisodeWriter,
    video_info,
)
from robots.franzi.kinematics import ROBOT_CONFIG, Kinematics
from simbridge.robot import GripperMap, load_robot


def to_sdk(kin, values):
    """(n, 16) URDF joints + closures -> SDK joints + gripper positions."""
    values = np.asarray(values, float)
    grips = [
        [GripperMap.position_from_closure(g) for g in col] for col in values.T[14:]
    ]
    return np.c_[
        kin.to_sdk("left", values[:, :7]), kin.to_sdk("right", values[:, 7:14]), *grips
    ]


def full_to_sdk(kin, robot, values):
    """(n, 22) FULL_NAMES positions (URDF rad, closures) -> SDK units."""
    values = np.asarray(values, float)
    return np.c_[
        to_sdk(kin, values[:, :16]),
        robot.units["waist"].to_sdk(values[:, 16:20]),
        robot.units["head"].to_sdk(values[:, 20:22]),
    ]


def rates_to_sdk(robot, values, gripper_scale):
    """(n, 22) FULL_NAMES rates or efforts -> SDK signs; the gripper columns
    times gripper_scale."""
    values = np.asarray(values, float)
    return np.c_[
        robot.units["left"].rate_to_sdk(values[:, :7]),
        robot.units["right"].rate_to_sdk(values[:, 7:14]),
        gripper_scale * values[:, 14:16],
        robot.units["waist"].rate_to_sdk(values[:, 16:20]),
        robot.units["head"].rate_to_sdk(values[:, 20:22]),
    ]


def full_rows(kin, data, sim_cameras):
    """Per-tick FULL_ARRAYS rows of a run's full log, or [] without one."""
    if "state_full" not in data.files:
        return []
    robot = load_robot()
    order = [list(data["camera_names"]).index(sim) for sim in sim_cameras]
    columns = {
        "state_full": full_to_sdk(kin, robot, data["state_full"]),
        "action_full": full_to_sdk(kin, robot, data["action_full"]),
        "velocity_full": rates_to_sdk(robot, data["velocity_full"], -1000.0),
        "effort_full": rates_to_sdk(robot, data["effort_full"], 1.0),
        "eef_pose": data["eef_pose"],
        "camera_pose": data["camera_pose"][:, order],
    }
    n = len(data["state_full"])
    return [{key: value[k] for key, value in columns.items()} for k in range(n)]


def convert(run, output, task=None, allow_failed=False, kin=None):
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text())
    task = task or summary.get("instruction") or KITTING_TASK
    if not summary["success"] and not allow_failed:
        raise ValueError(f"{run}: the expert failed ({summary['error']})")
    contract = summary["contract"]
    if (contract["state_dim"], contract["action_dim"], contract["control_hz"]) != (
        16,
        16,
        FPS,
    ):
        raise ValueError(f"{run}: contract {contract} is not 16-D at {FPS} Hz")
    data = np.load(run / "trajectory.npz")
    trace = json.loads((run / "trace.json").read_text())
    n = len(data["timestamp"])
    if len(trace) != n or [s["phase"] for s in trace] != data["subtask"].tolist():
        raise ValueError(f"{run}: trace.json and trajectory.npz disagree")
    streams = yaml.safe_load(ROBOT_CONFIG.read_text())["cameras"]["streams"]
    videos = {}
    for camera in CAMERAS:
        sim = streams[camera]["sim"]
        if sim not in contract["cameras"]:
            raise ValueError(
                f"{run}: no {sim} video (cameras {contract['cameras']}); runs from "
                "before the four-camera rig cannot be converted"
            )
        videos[camera] = run / f"{sim}.mp4"
        if video_info(videos[camera]) != (n, FPS, WIDTH, HEIGHT):
            raise ValueError(f"{videos[camera]}: {video_info(videos[camera])}")

    kin = kin or Kinematics()
    state = to_sdk(kin, data["state"])
    action = to_sdk(kin, data["action"])
    setpoint = np.r_[state[:1, :14], action[:-1, :14]]
    sims = [streams[camera]["sim"] for camera in CAMERAS]
    full = full_rows(kin, data, sims)
    writer = EpisodeWriter(output, encode=False)
    for k in range(n):
        writer.add(
            {
                "state": state[k],
                "action": action[k],
                "setpoint": setpoint[k],
                "arm_source": [1, 1],
                "robot_s": data["timestamp"][k],
                "state_utc_ms": -1,
                "camera_seq": [k + 1] * 4,
                "camera_utc_ms": [-1] * 4,
                "camera_host_ms": [-1] * 4,
                "tick_late_s": 0.0,
                "subtask": trace[k]["subtask"],
                **(full[k] if full else {}),
            }
        )
    for camera, path in videos.items():
        writer.copy_video(camera, path)
    for camera, sim in zip(CAMERAS, sims, strict=True):
        for kind in ("depth", "segmentation"):
            source = run / f"{kind}_{sim}.mkv"
            if source.exists():
                writer.copy_extra(f"{kind}_{camera}.mkv", source)
    if (run / "segmentation.json").exists():
        writer.copy_extra("segmentation.json", run / "segmentation.json")
    if full:
        writer.save_extra(
            "eval.npz",
            **{
                key: data[key]
                for key in (
                    "object_pose",
                    "object_velocity",
                    "prop_qpos",
                    "object_names",
                    "prop_joint_names",
                )
            },
        )
    if (run / "contacts.json").exists():
        writer.copy_extra("eval_contacts.json", run / "contacts.json")
    meta = {
        "source": "expert_run",
        "complete": True,
        "error": None,
        "task": task,
        "source_run": {
            "path": str(run),
            "task_name": summary.get("task"),
            "seed": summary.get("seed"),
            "success": summary["success"],
            "duration_s": summary["duration_s"],
            # Isaac runs keep MuJoCo as the kinematic mirror (isaac/engine.py).
            "simulator": summary.get(
                "simulator", f"MuJoCo {summary['mujoco_version']}"
            ),
            "mujoco_version": summary["mujoco_version"],
            "source_urdf_sha256": summary["source_urdf_sha256"],
            "videos": {c: p.name for c, p in videos.items()},
        },
        "conversion": {
            "joints": "robots.franzi.kinematics.Kinematics().to_sdk",
            "gripper": "simbridge.robot.GripperMap.position_from_closure",
            "setpoint": "action of the previous tick",
            "full": "waist and head: simbridge.robot.load_robot() unit maps; rates "
            "and efforts: their signs, grippers x -1000 (rate) and N (squeeze)"
            if full
            else None,
        },
        "clock": "sim",
        "images": "rendered at the state's physics step",
    }
    return writer.close(meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", type=Path, nargs="+", help="outputs/<expert run> ...")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new episode folder; with several runs, a new folder of episodes",
    )
    parser.add_argument("--task", help="episode instruction (default: the run's own)")
    parser.add_argument("--allow-failed", action="store_true")
    args = parser.parse_args()
    if len(args.runs) == 1:
        info = convert(args.runs[0], args.output, args.task, args.allow_failed)
        print(
            json.dumps({k: info[k] for k in ("frames", "task", "subtasks")}, indent=2)
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    kin = Kinematics()
    frames, skipped = [], []
    for run in args.runs:
        success = json.loads((run / "summary.json").read_text())["success"]
        if not success and not args.allow_failed:
            skipped.append(run.name)
            continue
        info = convert(run, args.output / run.name, args.task, args.allow_failed, kin)
        frames.append(info["frames"])
        print(f"{run.name}: {info['frames']} frames, {info['task']}", flush=True)
    summary = {"episodes": len(frames), "frames": sum(frames), "skipped": skipped}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
