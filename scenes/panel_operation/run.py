"""Panel operation episode: the scene and a scripted IK expert.

PanelScene is sim.tabletop.ObjectScene for the control console, which has no
free objects: the selector (a hinge) and the START, STOP and emergency-stop
caps (slides) are read from their joints. The milestones: the selector at
AUTO, then START pressed while it is. Success also needs the right hand off
both at the end, STOP never pressed and the emergency stop never touched (the
contact audit lets the hands touch jointed parts, so the scene tracks that
itself). Nothing is lifted, so HELD and CARRY_PHASES stay empty. The expert
reads the console's pose and the selector's angle from the simulator
(privileged): no welds, no pose writes, no contact-mask changes, no robot
qpos/qvel writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.panel_operation import task as pt
from sim import tabletop as tt
from sim.episode import minimum_jerk

SIDE = "right"
# The episode's subtasks (user rule 2026-10-25: one action of one arm on one
# object), the trace's subtask field; s.stage's names stay the fine phases.
# The left arm holds its default posture throughout.
TURN_SELECTOR = "Turn the selector from MAN to AUTO with the right hand."
PRESS_START = "Press the green START button with the right hand."
ARM_BACK = "Move the right arm back to its start pose."
# Device: the console joint that reads it (rad for the selector, m of travel
# for the caps).
DEVICES = {
    "selector": "console_mode",
    "start": "console_start",
    "stop": "console_stop",
    "emergency": "console_emergency",
}


class PanelScene(tt.ObjectScene):
    MILESTONES = ("selector_auto", "start_pressed")

    def __init__(self, output, spec, record=False):
        super().__init__(output, spec, record)
        m = self.m
        self.device_qadr = {k: int(m.joint(j).qposadr[0]) for k, j in DEVICES.items()}
        self.device_body = {k: int(m.joint(j).bodyid[0]) for k, j in DEVICES.items()}
        self.console = m.body("console_part").id
        # The caps' deepest travel so far (START: how far the press went).
        self.peak = {k: self.device(k) for k in ("start", "stop", "emergency")}
        self.estop_log = None
        self.grip_side = None  # the expert's grip on the lever (task.grip_angle)

    def build_root(self):
        return pt.build(self.spec)

    def device(self, name):
        return float(self.d.qpos[self.device_qadr[name]])

    def panel(self):
        return {
            "selector_deg": float(np.degrees(self.device("selector"))),
            **{
                f"{k}_travel_mm": 1000 * self.device(k)
                for k in ("start", "stop", "emergency")
            },
        }

    def console_pose(self):
        """The console base's position and yaw (rad), as the simulator has it."""
        R = self.d.xmat[self.console].reshape(3, 3)
        return self.d.xpos[self.console].copy(), float(np.arctan2(R[1, 0], R[0, 0]))

    def robot_on(self, device, pairs):
        """Robot bodies touching device's part."""
        body = self.device_body[device]
        return set(pairs[(pairs == body).any(1)].ravel().tolist()) & self.robot_bodies

    def gripping(self):
        """Both right jaws on the selector's lever."""
        return self.robot_on("selector", self.contact_bodies()).issuperset(
            self.jaws[SIDE]
        )

    def conditions(self, pairs):
        auto = self.device("selector") >= pt.AUTO_MIN
        return {
            "selector_auto": auto,
            "start_pressed": auto and self.device("start") >= pt.PRESSED,
        }

    def track_objects(self):
        super().track_objects()
        for k in self.peak:
            self.peak[k] = max(self.peak[k], self.device(k))
        touched = self.robot_on("emergency", self.contact_bodies())
        if touched:
            if self.estop_log is None:
                self.estop_log = {
                    "bodies": sorted(self.body_names[b] for b in touched),
                    "first_s": round(self.clock(), 3),
                    "phase": self.phase,
                    "physics_steps": 0,
                }
            self.estop_log["physics_steps"] += 1

    def violations(self):
        """What the task forbids, so far: the emergency stop touched by the
        robot, STOP pressed."""
        found = []
        if self.estop_log:
            found.append("emergency stop touched")
        if self.peak["stop"] >= pt.PRESSED:
            found.append("STOP pressed")
        return found

    def complete(self):
        pairs = self.contact_bodies()
        released = not (
            self.robot_on("selector", pairs) or self.robot_on("start", pairs)
        )
        return (
            self.status(pairs)["selector_auto"]
            and self.reached["start_pressed"]
            and released
            and not self.violations()
        )

    def sample(self):
        record = super().sample()
        record["panel"] = self.panel()
        return record


def follow(s, side, path, seconds=1.0, grips=None):
    """Scene.move along a joint path (waypoints from side's current goal): timed
    by joint travel at tt.SPEED and stretched to the controller's rates, as
    move(speed=) times its IK lines; the other arm holds its goal."""
    paths = {k: np.array([s.goal[k], s.goal[k]]) for k in s.goal}
    paths[side] = np.asarray(path, float)
    arcs = {
        k: np.r_[0, np.cumsum(np.abs(np.diff(p, axis=0)).max(1))]
        for k, p in paths.items()
    }
    seconds = max(seconds, 1.875 * max(a[-1] for a in arcs.values()) / tt.SPEED)
    seconds = s.stretch(paths, arcs, seconds)
    start = s.grip.copy()
    end = start | (grips or {})
    n = int(seconds / s.m.opt.timestep)
    for i in range(n):
        p = minimum_jerk((i + 1) / n)
        if s.step_number % 20 == 0:
            for k, path_k in paths.items():
                s.goal[k] = s.along(path_k, arcs[k], p)
                s.grip[k] = start[k] + (end[k] - start[k]) * p
        s.physics_step()


def expert(s):
    tt.start_home(s)
    home_q = s.goal[SIDE].copy()  # the default posture's joints
    pos, yaw = s.console_pose()
    knob = s.device("selector")
    body = s.body()
    # The sampler's grip (task.choose_grip): its side, and its arm joints as
    # the seed of the IK from the measured state.
    s.grip_side = s.spec["grip"]["side"]
    poses = pt.key_poses(pos, yaw, knob, s.grip_side, s.spec["press_angle"])
    grip_q = pt.fit_grip(s.kin, poses, s.spec["grip"]["joints"], body)
    normal = poses["normal"]
    grip, turned = poses["grip_rotation"], poses["turned_rotation"]
    press = poses["press_rotation"]

    def line(target, seconds, rotation, **kw):
        s.move(
            {SIDE: target},
            seconds=seconds,
            linear=True,
            rotations={SIDE: rotation},
            speed=tt.SPEED,
            **kw,
        )

    def turn(point, start, end, seconds, **kw):
        path = pt.wrist_turn(s.kin, SIDE, s.goal[SIDE], point, start, end, body)
        follow(s, SIDE, path, seconds, **kw)

    approach = pt.approach_path(s.kin, s.goal[SIDE], poses, grip_q, body)
    s.stage("approach", TURN_SELECTOR)
    follow(s, SIDE, approach, 3.0, grips={SIDE: tt.grip_open(pt.LEVER)})
    s.stage("align", TURN_SELECTOR)
    line(poses["grip"], 2.0, grip)
    s.stage("grasp", TURN_SELECTOR)
    s.move(grips={SIDE: tt.grip_closed(pt.LEVER, pt.SQUEEZE)}, seconds=1.0)
    s.move(seconds=0.5)
    if not s.gripping():
        raise RuntimeError("no bilateral grip on the selector")
    s.stage("turn", TURN_SELECTOR)
    turn(poses["grip"], grip, turned, 3.0)
    s.move(seconds=0.4)
    if s.device("selector") < pt.AUTO_MIN:
        raise RuntimeError(
            f"the selector stopped at {np.degrees(s.device('selector')):.1f} deg"
        )
    s.stage("release", TURN_SELECTOR)
    s.move(grips={SIDE: tt.grip_open(pt.LEVER)}, seconds=1.0)
    s.move(seconds=0.3)
    s.stage("retreat", TURN_SELECTOR)
    line(poses["lift"], 1.0, turned)
    s.stage("reorient", PRESS_START)
    turn(poses["lift"], turned, press, 2.0)
    s.stage("reach", PRESS_START)
    line(poses["prepress"], 2.0, press, grips={SIDE: pt.PRESS_CLOSURE})
    s.stage("press", PRESS_START)
    line(poses["press"], 1.5, press)
    goal, deepest = np.array(poses["press"]), s.device("start")
    for _ in range(int(pt.PRESS_REACH / 0.001)):
        depth = s.device("start")
        if depth >= pt.PRESS_TO:
            break
        # Closed loop: the cap follows the fingertips once they touch it, so
        # the steps shrink as the travel nears PRESS_TO.
        goal = goal - np.clip(pt.PRESS_TO - depth + 0.0003, 0.0003, 0.001) * normal
        line(goal, 0.4, press)
        deepest = max(deepest, s.device("start"))
    s.move(seconds=0.4)
    if max(deepest, s.device("start")) < pt.PRESSED:
        raise RuntimeError(f"START went down only {1000 * deepest:.2f} mm")
    s.stage("back_off", PRESS_START)
    line(poses["backoff"], 1.5, press)
    s.stage("return", ARM_BACK)
    # Back over the selector, then down the approach in reverse (task.press_path).
    back = pt.return_line(s.kin, s.goal[SIDE], poses, approach, body)
    if np.abs(back[-1] - approach[-1]).max() > pt.REJOIN:
        raise RuntimeError("the arm came back too far from the approach's end")
    follow(s, SIDE, back, 1.5)
    s.move(joints={SIDE: approach[-1]}, seconds=0.5, speed=tt.SPEED)
    opening = 1 - tt.HOME_OPENING / tt.JAW_SPAN
    follow(s, SIDE, approach[::-1], 3.0, grips={SIDE: opening})
    s.stage("finish", ARM_BACK)
    s.move(joints={SIDE: home_q}, seconds=0.5, speed=tt.SPEED)
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None, engine=None):
    """One expert episode for seed into output (must not exist). engine runs it
    on another simulator (isaac.episode.isaac_scene for Isaac Sim)."""
    spec = pt.sample_episode(seed, kin)
    return tt.run_episode(
        PanelScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {
            "panel": s.panel(),
            "peak_travel_mm": {k: 1000 * v for k, v in s.peak.items()},
            "violations": s.violations(),
            "emergency_stop_contact": s.estop_log,
            "grip_side": s.grip_side,
        },
        engine=engine,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record", action="store_true")
    args = p.parse_args()
    result = run(args.output, args.seed, args.record)
    print(
        json.dumps({k: result[k] for k in ("success", "error", "progress")}), flush=True
    )


if __name__ == "__main__":
    main()
