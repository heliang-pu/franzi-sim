"""Convert recorded expert runs (sim.run_expert or sim.run_kitting, with --record)
into raw VLA episodes.

    python -m data.convert_expert_run outputs/expert_pinocchio --output outputs/<new_episode>
    python -m data.convert_expert_run outputs/<runs>/seed_* --output outputs/<new_folder>

With several runs, --output is a new folder holding one episode per run, named
after the run; failed runs are skipped unless --allow-failed.

The expert commands MuJoCo in URDF radians and gripper closure fractions. The
episode uses the SDK convention of vla_episode.py, so a policy trained on it
commands the robot through the stock SDK unchanged:

- joints: robot_kinematics.Kinematics().to_sdk, q_sdk = sign * q_urdf + offset;
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
  sim.run_kitting seed), else KITTING_TASK; --task overrides both. Subtask texts
  come from trace.json. The camera videos are copied unchanged. Ground-truth
  object state (trace.json "objects", summary.json placements) is not carried over.
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
from robot_kinematics import ROBOT_CONFIG, Kinematics
from simbridge.robot import GripperMap


def to_sdk(kin, values):
    """(n, 16) URDF joints + closures -> SDK joints + gripper positions."""
    values = np.asarray(values, float)
    grips = [
        [GripperMap.position_from_closure(g) for g in col] for col in values.T[14:]
    ]
    return np.c_[
        kin.to_sdk("left", values[:, :7]), kin.to_sdk("right", values[:, 7:14]), *grips
    ]


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
            }
        )
    for camera, path in videos.items():
        writer.copy_video(camera, path)
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
            "mujoco_version": summary["mujoco_version"],
            "source_urdf_sha256": summary["source_urdf_sha256"],
            "videos": {c: p.name for c, p in videos.items()},
        },
        "conversion": {
            "joints": "robot_kinematics.Kinematics().to_sdk",
            "gripper": "simbridge.robot.GripperMap.position_from_closure",
            "setpoint": "action of the previous tick",
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
