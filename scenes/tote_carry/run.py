"""Tote carry episode: the scene and a scripted IK expert.

CarryScene is sim.tabletop.ObjectScene with task.py's tote: the episode
framework's audits and recordings, plus object tracking, the milestones (the
tote lifted by both hands, carried over the conveyor, set on its rollers) and
the success test. While it is carried over the conveyor (CARRY_PHASES), the
held tote may touch nothing but the robot. The expert reads the tote's pose
from the simulator (privileged): no welds, no object pose writes, no
contact-mask changes, no robot qpos/qvel writes.

Both hands hold one rigid tote, so every move that carries it is timed evenly
over its waypoints (no speed=, which times each arm by its own joint travel),
as many for both hands: they then advance together. The lift and the set-down
are straight lines (1 cm waypoints); the carry over the conveyor turns the tote
square to the rollers on the way, a path of 1 cm or half-degree waypoints that
CarryScene.move_path plays. Both stretch the moves to the controller's rates
as Scene.move does. The free hands come from the default posture and go back
to it in joint space.

The episode splits into five subtasks (the trace's subtask; its phase keeps
the fine stage; user rule 2026-10-25: a subtask is one action of one arm on
one object, or of both hands on one object together, data.subtask_check).
Three are both hands on the one tote: grip its end walls (approach, align,
grasp), carry it onto the conveyor (lift, carry, lower) and let go of it
(release, retreat off the walls). Then each arm goes back to the default
posture on its own, left then right (finish): no two-arm returns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.tote_carry import task as kt
from sim import tabletop as tt
from sim.episode import minimum_jerk


class CarryScene(tt.ObjectScene):
    OBJECTS = ("tote",)
    HELD = ("tote",)
    MILESTONES = ("tote_lifted", "over_conveyor", "on_conveyor")
    CARRY_PHASES = ("carry",)

    def build_root(self):
        return kt.build(self.spec)

    def tote_pose(self):
        """The tote's base centre (x, y, z) and yaw, measured (privileged)."""
        R = self.d.xmat[self.ids["tote"]].reshape(3, 3)
        return [
            *(float(v) for v in self.xyz("tote")),
            float(np.arctan2(R[1, 0], R[0, 0])),
        ]

    def held_by_both(self, pairs):
        others = self.touching("tote", pairs)
        return all(others.issuperset(self.jaws[s]) for s in tt.SIDES)

    def conditions(self, pairs):
        x, y, z = self.xyz("tote")
        tx, ty = self.spec["target"]
        held = self.held_by_both(pairs)
        at_target = abs(x - tx) < 0.03 and abs(y - ty) < 0.03
        return {
            "tote_lifted": held and z > kt.BENCH_Z + 0.05,
            "over_conveyor": held and at_target and z > kt.ROLLER_TOP + 0.01,
            "on_conveyor": (
                at_target
                and abs(z - kt.ROLLER_TOP) < 0.004
                and self.settled("tote", pairs)
            ),
        }

    def complete(self):
        return self.status()["on_conveyor"] and self.reached["over_conveyor"]

    def move_path(self, paths, seconds):
        """Move the arms along joint paths {side: waypoints} with as many
        waypoints each, timed evenly over them (the hands keep step) on
        Scene.move's minimum-jerk profile, stretched to the controller's rates
        as it does."""
        seconds = self.stretch(paths, {}, seconds)
        n = int(seconds / self.m.opt.timestep)
        for i in range(n):
            p = minimum_jerk((i + 1) / n)
            if self.step_number % 20 == 0:
                for side, path in paths.items():
                    self.goal[side] = self.along(path, None, p)
            self.physics_step()


GRIP = "Grip the tote's end walls with both hands."
CARRY = "Carry the tote onto the roller conveyor with both hands."
LET_GO = "Let go of the tote with both hands."
BACK = "Move the {} arm back to its start pose."


def check_walls(s):
    if not s.held_by_both(s.contact_bodies()):
        raise RuntimeError("no bilateral grasp of both hands on the tote's end walls")


def expert(s):
    tt.start_home(s)
    home = {side: s.goal[side].copy() for side in tt.SIDES}
    tote = s.tote_pose()
    target = s.spec["target"]
    poses = kt.key_poses(tote, target)
    turned = {side: poses[side]["rotation"] for side in tt.SIDES}
    square = {side: poses[side]["place_rotation"] for side in tt.SIDES}
    over_walls = {
        side: kt.pregrasp_joints(s.kin, side, poses, s.body()) for side in tt.SIDES
    }
    opened = {side: tt.grip_open(kt.GRIP_W) for side in tt.SIDES}
    closed = {side: tt.grip_closed(kt.GRIP_W, kt.SQUEEZE) for side in tt.SIDES}

    def move(key, seconds, rotations, **kw):
        """Both hands along straight lines to their key poses, evenly timed."""
        s.move(
            {side: poses[side][key] for side in tt.SIDES},
            seconds=seconds,
            linear=True,
            rotations=rotations,
            **kw,
        )

    s.stage("approach", GRIP)
    s.move(joints=over_walls, grips=opened, seconds=3.0, speed=tt.SPEED)
    s.stage("align", GRIP)
    move("grasp", 2.0, turned)
    s.stage("grasp", GRIP)
    s.move(grips=closed, seconds=1.0)
    s.move(seconds=0.5)
    check_walls(s)
    s.stage("lift", CARRY)
    start = float(s.xyz("tote")[2])
    move("lift", 2.0, turned)
    if s.xyz("tote")[2] < start + 0.8 * kt.LIFT_H:
        raise RuntimeError("lift failed for the tote")
    check_walls(s)
    s.stage("carry", CARRY)
    carry = kt.carry_targets(tote, target)
    s.move_path(
        {
            side: kt.chain(s.kin, side, s.goal[side], carry[side], s.body())
            for side in tt.SIDES
        },
        3.0,
    )
    s.stage("lower", CARRY)
    move("place", 1.5, square)
    # Re-measured (privileged): the rest of the way down to SET_DOWN over the
    # rollers, so the tote is let go without a drop.
    gap = float(s.xyz("tote")[2]) - kt.ROLLER_TOP - kt.SET_DOWN
    if gap > 0:
        s.move(
            {side: poses[side]["place"] - [0, 0, gap] for side in tt.SIDES},
            seconds=1.0,
            linear=True,
            rotations=square,
        )
    s.stage("release", LET_GO)
    s.move(grips=opened, seconds=1.0)
    s.move(seconds=0.5)
    s.stage("retreat", LET_GO)
    move("retreat", 1.5, square)
    for side in tt.SIDES:  # one arm at a time, the other holding
        s.stage("finish", BACK.format(side))
        s.move(
            joints={side: home[side]},
            grips={side: 1 - tt.HOME_OPENING / tt.JAW_SPAN},
            seconds=3.0,
            speed=tt.SPEED,
        )
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None, engine=None):
    """One expert episode for seed into output (must not exist). engine runs it
    on another simulator (isaac.episode.isaac_scene for Isaac Sim)."""
    spec = kt.sample_episode(seed, kin)
    return tt.run_episode(
        CarryScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {
            "tote_offset_mm": 1000
            * float(np.linalg.norm(s.xyz("tote")[:2] - s.spec["target"])),
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
