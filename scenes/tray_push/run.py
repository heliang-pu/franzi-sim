"""Tray push episode: the scene and a scripted IK expert.

TrayScene is sim.tabletop.ObjectScene with task.py's tray and the two loose
parts in it: the episode framework's audits and recordings, plus object
tracking, the milestones (the tray pushed half way, then resting against the
stop) and the success test, which also needs both parts still lying in the
tray. Nothing is lifted, so HELD is empty. The expert reads the tray's pose
from the simulator (privileged): no welds, no object pose writes, no
contact-mask changes, no robot qpos/qvel writes.

An episode is two single-arm subtasks (the trace's subtask; its phase is the
finer stage): the left hand pushes the tray to the stop and lets go, then the
left arm goes back to its start pose. The right arm holds its start pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.tray_push import task as kt
from sim import tabletop as tt

# The subtasks, the trace's subtask texts (user rule 2026-10-25: one action of
# one arm on one object each).
PUSH_TRAY = "Push the tray along the rails to the end stop with the left hand."
ARM_BACK = "Move the left arm back to its start pose."


class TrayScene(tt.ObjectScene):
    OBJECTS = ("tray", *kt.PARTS)
    MILESTONES = ("pushed_halfway", "at_stop")

    def __init__(self, output, spec, record=False):
        super().__init__(output, spec, record)
        self.stop = self.m.body("rails_stop_part").id

    def build_root(self):
        return kt.build(self.spec)

    def in_tray_frame(self, name):
        """A body's position in the tray's frame (its base centre), m."""
        bid = self.ids["tray"]
        frame = self.d.xmat[bid].reshape(3, 3)
        return frame.T @ (self.xyz(name) - self.d.xpos[bid])

    def in_tray(self, name, pairs):
        """Lying on the tray's floor inside its walls, upright, still, clear of
        the robot."""
        x, y, z = self.in_tray_frame(name)
        return (
            abs(x) < kt.INNER[0]
            and abs(y) < kt.INNER[1]
            and abs(z - kt.FLOOR) < 0.003
            and self.settled(name, pairs)
        )

    def tray_y(self):
        """The tray's centre along the rails (their frame), m."""
        return kt.to_rails(self.spec["rails"], self.xyz("tray"))[1]

    def stop_gap(self):
        """How far (m) the tray's centre is from where the stop holds it."""
        return self.tray_y() - kt.TRAY_STOP_Y

    def on_stop(self):
        return self.stop in self.touching("tray", self.contact_bodies())

    def conditions(self, pairs):
        y = self.tray_y()
        return {
            "pushed_halfway": y < (self.spec["tray_y"] + kt.TRAY_STOP_Y) / 2,
            "at_stop": abs(y - kt.TRAY_STOP_Y) < kt.STOP_TOLERANCE
            and self.settled("tray", pairs),
        }

    def complete(self):
        pairs = self.contact_bodies()
        return self.status(pairs)["at_stop"] and all(
            self.in_tray(n, pairs) for n in kt.PARTS
        )


def seat(s, move, goal):
    """The rest of the way onto the stop: one slow move of the measured gap
    (privileged) and PRELOAD, then 1 mm steps until the tray touches the stop.
    Returns the hand's last goal."""
    rails = s.spec["rails"]
    goal = np.asarray(goal, float) - kt.along(rails, s.stop_gap() + kt.PRELOAD)
    move(goal, 2.0)
    for _ in range(3):
        if s.on_stop():
            break
        goal = goal - kt.along(rails, 0.001)
        move(goal, 0.5)
    s.move(seconds=0.5)
    if not s.on_stop() and s.stop_gap() > kt.STOP_TOLERANCE:
        raise RuntimeError(f"the tray stopped {s.stop_gap() * 1000:.1f} mm short")
    return goal


def expert(s):
    """From the default posture, PUSH_TRAY: close the fingers over the tray's
    +y wall, lower them just behind it, push the tray along the rails to EARLY
    short of the stop, seat it on the stop and draw back; then ARM_BACK: rise,
    go straight part of the way to READY and move in joint space back to the
    start pose's joints. Only the left arm moves. The tray's pose is read once
    it has settled (privileged)."""
    tt.start_home(s)
    home = s.goal["left"].copy()  # the start pose's joints, to go back to
    rails = s.spec["rails"]
    poses = kt.key_poses(rails, s.tray_y(), s.spec["push_dx"])
    rotation = {"left": kt.hand(rails)}

    def move(target, seconds=2.0, **kw):
        # Straight lines, the hand straight down, its jaws across the rails.
        s.move(
            {"left": target},
            seconds=seconds,
            linear=True,
            rotations=rotation,
            speed=tt.SPEED,
            **kw,
        )

    s.stage("approach", PUSH_TRAY)
    move(poses["above"], 2.5, grips={"left": kt.CLOSED})
    s.stage("lower", PUSH_TRAY)
    move(poses["touch"], 1.5)
    s.stage("push", PUSH_TRAY)
    move(poses["push"], kt.PUSH_SECONDS)
    if s.stop_gap() > kt.EARLY + 0.01:
        raise RuntimeError(f"the tray did not follow the push ({s.stop_gap():.3f} m)")
    s.stage("seat", PUSH_TRAY)
    back = seat(s, move, poses["push"]) + kt.along(rails, kt.BACK)
    s.stage("release", PUSH_TRAY)
    move(back, 1.0)
    s.stage("retreat", ARM_BACK)
    clear = back + [0, 0, kt.CLEAR]
    move(clear, 1.5)
    s.stage("return", ARM_BACK)
    move(kt.homeward(clear), 2.0)
    s.stage("finish", ARM_BACK)
    s.move(
        joints={"left": home},
        grips={"left": kt.HOME_GRIP},
        seconds=2.0,
        speed=tt.SPEED,
    )
    s.move(seconds=1.0)


def summary(s):
    """Extra summary.json fields: where the tray started and ended along the
    rails (their frame) and how far the parts slid in it (mm, tray frame)."""
    shift = {
        n: [
            round(1000 * float(v), 1)
            for v in s.in_tray_frame(n)[:2] - s.spec["parts"][n]["xy"]
        ]
        for n in kt.PARTS
    }
    return {
        "tray_start_y": s.spec["tray_y"],
        "tray_final_y": s.tray_y(),
        "stop_gap_mm": round(1000 * s.stop_gap(), 2),
        "part_shift_mm": shift,
    }


def run(output, seed, record=False, kin=None, engine=None):
    """One expert episode for seed into output (must not exist). engine runs it
    on another simulator (isaac.episode.isaac_scene for Isaac Sim)."""
    spec = kt.sample_episode(seed, kin)
    return tt.run_episode(
        TrayScene, expert, spec, output, record, extra=summary, engine=engine
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
