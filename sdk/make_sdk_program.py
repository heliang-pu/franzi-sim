"""Write the data that sdk_acceptance.py sends, in SDK joint coordinates.

Like taught poses on a real controller, the program is plain data: joint
vectors (rad, controller MDH convention from config/robot/franzi_sdk.yaml) and
gripper positions (0 closed .. 1000 open). Built once from the MuJoCo scene:

- poses: the expert's collision-free "ready" pose and small variations; every
  programmed move is swept along its joint-space line and rejected if any
  robot geom would touch the world or the other arm (kinematic check only);
- fault probe: a left-hand pose above a free spot of the table and an IK
  descent ending 5 cm below its surface, to exercise penetration/stall handling;
- task replay: the privileged IK expert's 25 Hz joint targets from a recorded
  run, segmented by phase, for an open-loop full-task attempt through the SDK.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from scenes.pick_place.run import PickPlaceScene
from sim.build_task import ROOT, TABLE_Z
from simbridge.robot import load_robot


class Sweep:
    """Kinematic clearance of joint-space lines, on a scratch MjData."""

    def __init__(self, scene):
        self.scene = scene
        self.m = scene.m
        self.d = mujoco.MjData(scene.m)
        mujoco.mj_forward(self.m, self.d)
        self.baseline = self.pairs()

    def pairs(self):
        robot = self.scene.robot_bodies
        out = set()
        for c in self.d.contact[: self.d.ncon]:
            a, b = (int(self.m.geom_bodyid[g]) for g in c.geom)
            names = [self.m.body(a).name, self.m.body(b).name]
            cross_arm = all(n.startswith(("left", "right")) for n in names) and (
                names[0][:4] != names[1][:4]
            )
            if (a in robot) != (b in robot) or cross_arm:
                out.add(tuple(sorted(self.m.geom(int(g)).name for g in c.geom)))
        return out

    def check(self, start, goal, label, samples=60):
        s = self.scene
        for k in range(samples + 1):
            q = start + (goal - start) * k / samples
            self.d.qpos[s.qids["left"]] = q[:7]
            self.d.qpos[s.qids["right"]] = q[7:]
            mujoco.mj_forward(self.m, self.d)
            new = self.pairs() - self.baseline
            if new:
                raise SystemExit(f"{label}: contact at {k}/{samples}: {sorted(new)}")


class Clamp:
    """Keep targets inside the SDK limits (sim range and archive controller
    range); the expert plans against the URDF range, which is a little wider
    for some joints. Every clamp is counted and reported in the program."""

    def __init__(self, robot):
        self.robot = robot
        self.count = 0
        self.max_rad = 0.0
        self.joints = set()

    def __call__(self, side, q_sdk):
        lim = self.robot.arms[side].sdk_limits
        out = np.clip(q_sdk, lim[:, 0] + 1e-6, lim[:, 1] - 1e-6)
        moved = np.abs(out - q_sdk)
        if moved.max() > 0:
            self.count += 1
            self.max_rad = max(self.max_rad, float(moved.max()))
            self.joints.update(f"{side} j{j + 1}" for j in np.flatnonzero(moved))
        return out

    def report(self):
        return {
            "clamped_vectors": self.count,
            "max_change_deg": round(float(np.rad2deg(self.max_rad)), 4),
            "joints": sorted(self.joints),
        }


def retime(robot, points, rate, margin=0.9):
    """Slow a phase uniformly until its 25 Hz path respects the SDK velocity and
    acceleration limits (the expert plans without them). The geometric path is
    unchanged; returns the resampled points and the slow-down factor."""
    q = np.asarray(points, float)
    vmax = np.r_[robot.arms["left"].vmax, robot.arms["right"].vmax] * margin
    amax = np.r_[robot.arms["left"].amax, robot.arms["right"].amax] * margin
    if len(q) < 3:
        return points, 1.0
    v = np.diff(q, axis=0) * rate
    a = np.diff(v, axis=0) * rate
    k = max(1.0, np.max(np.abs(v) / vmax), np.sqrt(np.max(np.abs(a) / amax)))
    if k == 1.0:
        return points, 1.0
    t = np.arange(len(q)) * k
    new_t = np.arange(0.0, t[-1] + 1e-9, 1.0)
    out = np.array([np.interp(new_t, t, col) for col in q.T]).T
    out[-1] = q[-1]
    return out.round(6).tolist(), round(float(k), 3)


def sdk(robot, q_sim, clamp):
    left = clamp("left", robot.arms["left"].to_sdk(q_sim[:7]))
    right = clamp("right", robot.arms["right"].to_sdk(q_sim[7:]))
    return np.r_[left, right].round(6).tolist()


def to_sim(robot, q_sdk):
    q = np.asarray(q_sdk, float)
    return np.r_[robot.arms["left"].to_sim(q[:7]), robot.arms["right"].to_sim(q[7:])]


def left_sim(robot, clamp, q_sim):
    left = robot.arms["left"]
    return left.to_sim(clamp("left", left.to_sdk(q_sim)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--expert-run",
        type=Path,
        default=ROOT / "outputs/expert_wrist_block",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "config/sdk_acceptance/program.json"
    )
    args = parser.parse_args()
    robot = load_robot()
    clamp = Clamp(robot)
    data = np.load(args.expert_run / "trajectory.npz")
    summary = json.loads((args.expert_run / "summary.json").read_text())
    if not summary["success"]:
        raise SystemExit(f"{args.expert_run} is not a successful expert run")
    phases = []
    for i, name in enumerate(data["subtask"]):
        if not phases or phases[-1]["name"] != name:
            phases.append({"name": str(name), "points": [], "gripper": None})
        action = data["action"][i]
        phases[-1]["points"].append(sdk(robot, action[:14], clamp))
        phases[-1]["gripper"] = [
            robot.grippers[s].position_from_closure(action[14 + k])
            for k, s in enumerate(("left", "right"))
        ]
    for phase in phases:
        phase["points"], phase["slowdown"] = retime(robot, phase["points"], 25)
    ready = np.asarray(next(p for p in phases if p["name"] == "ready")["points"][-1])
    pose_a = ready
    pose_b = ready.copy()
    pose_b[[0, 7]] += 0.15  # shoulder pitch
    pose_b[[3, 10]] += 0.20  # elbow flexion
    single = pose_b[:7].copy()
    single[2] += 0.30  # left shoulder yaw
    scratch = ROOT / "outputs" / ".sdk_program_scene"
    scratch.mkdir(parents=True, exist_ok=True)
    scene = PickPlaceScene(scratch, record=False)
    above = scene.ik("left", [0.28, 0.45, TABLE_Z + 0.12])
    # A straight descent as IK waypoints 2 cm apart, ending 5 cm below the
    # surface, so the first contact is the fingertips on the free table spot.
    press = [above]
    for z in np.linspace(TABLE_Z + 0.12, TABLE_Z + 0.0275 - 0.05, 9)[1:]:
        press.append(scene.ik("left", [0.28, 0.45, z], near=press[-1]))
    sweep = Sweep(scene)
    zero = np.zeros(14)
    a, b = to_sim(robot, pose_a), to_sim(robot, pose_b)
    s1 = to_sim(robot, np.r_[single, pose_b[7:]])
    finish = to_sim(robot, phases[-1]["points"][-1])
    probe_above = np.r_[left_sim(robot, clamp, above), finish[7:]]
    for start, goal, label in [
        (zero, a, "home -> pose_a"),
        (a, b, "pose_a -> pose_b"),
        (b, s1, "pose_b -> single"),
        (s1, a, "single -> pose_a"),
        (a, zero, "pose_a -> home"),
        (finish, probe_above, "task finish -> probe above"),
    ]:
        sweep.check(start, goal, label)
    scene.close()
    left = robot.arms["left"]
    program = {
        "joint_convention": robot.convention,
        "robot_config_sha256": hashlib.sha256(robot.path.read_bytes()).hexdigest(),
        "pose_a": {"left": pose_a[:7].tolist(), "right": pose_a[7:].tolist()},
        "pose_b": {"left": pose_b[:7].tolist(), "right": pose_b[7:].tolist()},
        "single_arm_target": single.tolist(),
        "gripper_positions": [750, 250, 1000],
        "out_of_range_elbow_rad": -0.5,
        "limit_clamping": None,
        "sweep_checked": "joint-space lines between these poses clear the world "
        "and the other arm (kinematic check in MuJoCo)",
        "fault_probe": {
            "side": 0,
            "above": clamp("left", left.to_sdk(above)).round(6).tolist(),
            "press_path": [
                clamp("left", left.to_sdk(q)).round(6).tolist() for q in press[1:]
            ],
            "note": "grasp site 5 cm below the table surface at a free spot",
        },
        "task": {
            "source_run": str(args.expert_run.resolve().relative_to(ROOT)),
            "source_success": summary["success"],
            "source_duration_s": summary["duration_s"],
            "rate_hz": 25,
            "phases": phases,
            "note": "open-loop replay of privileged IK expert targets; not FineART",
        },
    }
    program["limit_clamping"] = clamp.report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(program, indent=1))
    print(f"wrote {args.output}: {len(phases)} phases, {len(data['action'])} samples")


if __name__ == "__main__":
    main()
