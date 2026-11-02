"""Precision insertion episode: the scene and a scripted IK expert.

InsertionScene is sim.tabletop.ObjectScene with task.py's two loose parts:
the episode framework's audits and recordings, plus object tracking, the
milestones (the bearing lifted and the pin out of its hole, the bearing on
the shaft, both parts home) and the success test: the bearing seated on the
shoulder and the pin standing in its target hole, both centred, upright, still
and clear of the robot.

The arms work one after the other, top-down with the jaws along y
(task.BEARING_YAW, PIN_YAW), in single-arm subtasks (user rule
2026-10-25: one arm on one object per subtask, the other arm's commands
unchanged meanwhile; data.subtask_check checks it): the left arm picks up the
bearing, slides it onto the shaft and moves back to its start pose; then the
right arm picks the pin out of its hole, inserts it into the empty one and
moves back. Each subtask's text is the trace's subtask, its fine stages
(approach, align, grasp, lift; carry, centre, insert, release; retreat,
return) the phase.

The expert reads the parts' poses from the simulator (privileged): the arm
swings its open hand over its part in joint space, grasps the part where it
lies, carries it over its seat, centres it there from its measured pose in
the hand (one or two rounds bring it within 0.1 mm of the axis), lowers it in
three straight moves (4 mm into the lead-in or the hole, 2 mm over the seat,
then, measured again, 0.5 mm over it) and lets go, then rises 7 cm and swings
back to its start joints. A part caught on the rim (pushed up in its hand) is
backed up, centred again and tried once more. No welds, no object pose writes,
no contact-mask changes, no robot qpos/qvel writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.precision_insertion import task as pt
from sim import tabletop as tt

PART = {side: part for part, (side, *_) in pt.PARTS.items()}
WIDTH = {side: width for _, (side, width, *_) in pt.PARTS.items()}
HAND = {side: pt.hand(part) for side, part in PART.items()}  # top-down, turned
# The part's lowest point over its seat, from its base centre: the ring on the
# shaft's shoulder (r 12 mm) and the pin's foot (r 5 mm). On PhysX a held part
# pivots 2-4 degrees about the line between the pads (cnc_tending, 2026-10-25);
# at 3 degrees the bearing's ring sits 0.6 mm below its centre.
FOOT = {"bearing": 0.012, "pin": 0.005}
# Success: the part's base within SEAT_XY of its seat's axis and SEAT_DZ of its
# height, tilted less than SEAT_TILT (on the shaft or in the hole it can tilt
# about 4 and 3 degrees).
SEAT_XY = 0.001
SEAT_DZ = 0.0015
SEAT_TILT = np.radians(2.0)
# Centring over the seat: rounds of measured corrections until the part's
# base is within CENTRE_TOL of the seat's axis.
CENTRE_TOL = 0.0001
CENTRE_ROUNDS = 8
# Each reading averages the part's place over CENTRE_TICKS control ticks with
# the hands held: on PhysX the held pin swings about the line between the pads
# (task version 4, Isaac seed 0: up to 0.8 mm at its foot, period 0.32 s, not
# dying out), and single readings scattered 0.1-0.6 mm, so centring took all
# eight rounds. Sixteen ticks are two periods.
CENTRE_TICKS = 16
# A part pushed up in its hand this far on the way in is caught on its seat's
# rim. Held parts creep down in the grasp instead (MuJoCo's soft friction: the
# bearing 0.13-0.32 mm/s, by its yaw, the pin 0.14 mm/s; 1.1-2.8 and 1.2 mm
# while held), so every height is aimed from the part's measured pose.
PUSH_TOL = 0.0005
RISE = {"bearing": 0.05, "pin": 0.03}  # the least lift that counts
# Each arm's subtasks (the trace's subtask texts): pick, place, move back.
SUBTASKS = {
    "left": (
        "Pick up the bearing with the left hand.",
        "Slide the bearing onto the shaft with the left hand.",
        "Move the left arm back to its start pose.",
    ),
    "right": (
        "Pick the dowel pin out of its hole with the right hand.",
        "Insert the dowel pin into the empty hole with the right hand.",
        "Move the right arm back to its start pose.",
    ),
}


class InsertionScene(tt.ObjectScene):
    OBJECTS = ("bearing", "pin")
    HELD = ("bearing", "pin")
    MILESTONES = (
        "bearing_lifted",
        "pin_out",
        "bearing_on_shaft",
        "bearing_seated",
        "pin_seated",
    )
    # Carried through the air: there a held part touching anything but the
    # robot is a collision. The lift (the bearing leaves the bench, the pin
    # its hole), the insertion (the bore runs down the shaft, the pin into its
    # hole) and the release touch the fixtures by design.
    CARRY_PHASES = ("carry", "centre")

    def build_root(self):
        return pt.build(self.spec)

    def tilt(self, name):
        """Angle (rad) between the part's axis and the vertical."""
        return float(np.arccos(np.clip(self.d.xmat[self.ids[name]][8], -1, 1)))

    def off_axis(self, name, xy):
        return float(np.linalg.norm(self.xyz(name)[:2] - np.asarray(xy)))

    def seated(self, name, xy, z, pairs):
        return (
            self.off_axis(name, xy) < SEAT_XY
            and abs(self.xyz(name)[2] - z) < SEAT_DZ
            and self.tilt(name) < SEAT_TILT
            and self.settled(name, pairs, upright=False)
        )

    def low(self, name):
        """The height (m) of the part's lowest point over its seat."""
        return float(self.xyz(name)[2]) - FOOT[name] * np.sin(self.tilt(name))

    def in_hand(self, name, pairs):
        side = pt.PARTS[name][0]
        return self.touching(name, pairs).issuperset(self.jaws[side])

    def conditions(self, pairs):
        seat = pt.goals(self.spec)
        bearing, pin = self.xyz("bearing"), self.xyz("pin")
        return {
            "bearing_lifted": self.in_hand("bearing", pairs)
            and bearing[2] > pt.SHAFT_TOP,
            "pin_out": self.in_hand("pin", pairs) and pin[2] > pt.PLATE_TOP,
            "bearing_on_shaft": self.off_axis("bearing", seat["bearing"]) < SEAT_XY
            and bearing[2] < pt.SHAFT_TOP - 0.005,
            "bearing_seated": self.seated("bearing", seat["bearing"], pt.SEAT_Z, pairs),
            "pin_seated": self.seated("pin", seat["pin"], pt.BENCH_Z, pairs),
        }

    def complete(self):
        now = self.status()
        return now["bearing_seated"] and now["pin_seated"]


def measure(s, sides):
    """Each side's part's base (x, y), averaged over CENTRE_TICKS control
    ticks with the hands held."""
    total = {side: np.zeros(2) for side in sides}
    for _ in range(CENTRE_TICKS):
        s.move(seconds=0.04)  # one control tick
        for side in sides:
            total[side] += s.xyz(PART[side])[:2]
    return {side: place / CENTRE_TICKS for side, place in total.items()}


def centre(s, move, goals, seats):
    """Shift each hand's goal until its part's base sits within CENTRE_TOL of
    its seat's axis, from the measured part poses (measure()); returns the
    goals."""
    goals = {side: np.array(goal, float) for side, goal in goals.items()}
    for _ in range(CENTRE_ROUNDS):
        seen = measure(s, goals)
        error = {side: np.asarray(seats[side], float) - seen[side] for side in goals}
        print(
            "  centre: "
            + ", ".join(
                f"{PART[side]} {1000 * np.linalg.norm(e):.3f} mm"
                for side, e in error.items()
            ),
            flush=True,
        )
        if max(np.linalg.norm(e) for e in error.values()) < CENTRE_TOL:
            break
        for side, e in error.items():
            goals[side][:2] += e
        move(goals, 0.5)
    return goals


def hold_offset(s, side):
    """How far the part's lowest point hangs below its hand's grasp site (m)."""
    return float(s.d.site_xpos[s.sites[side]][2]) - s.low(PART[side])


class Caught(RuntimeError):
    """A part caught on its seat's rim: pushed up in its hand. goals are the
    hand goals the move went to."""

    def __init__(self, message, goals):
        super().__init__(message)
        self.goals = goals


def lower(s, move, goals, base_z, seconds):
    """Lower (or raise) each hand so that its part's lowest point reaches
    base_z[part], from the part's measured pose; raises Caught if a part was
    pushed up in its hand (caught on the shaft or the plate)."""
    hold = {side: hold_offset(s, side) for side in goals}
    goals = {
        side: goal - [0, 0, s.low(PART[side]) - base_z[PART[side]]]
        for side, goal in goals.items()
    }
    move(goals, seconds)
    for side in goals:
        pushed = hold[side] - hold_offset(s, side)
        if pushed > PUSH_TOL:
            raise Caught(
                f"the {PART[side]} caught on the way in ({1000 * pushed:.2f} mm)",
                goals,
            )
    return goals


def insert(s, move, goals, seats, tries=2):
    """Each part into its lead-in, then 2 mm over its seat and, measured again,
    0.5 mm over it, from the hover; a part caught on its seat's rim is backed
    up to the hover, centred again and tried again."""
    for attempt in range(1, tries + 1):
        try:
            goals = lower(s, move, goals, pt.ENGAGE, 1.2)
            goals = lower(s, move, goals, pt.NEAR, 2.0)
            return lower(s, move, goals, pt.HOME, 0.6)
        except Caught as caught:
            if attempt == tries:
                raise
            print(f"  retry: {caught}", flush=True)
            goals = lower(s, move, caught.goals, pt.HOVER, 1.0)
            goals = centre(s, move, goals, seats)
    return goals


def work(s, side, start, start_grip):
    """side's three subtasks on its part; the other arm holds. start and
    start_grip are its start pose's joints and gripper closure, which it
    returns to."""
    part, width = PART[side], WIDTH[side]
    seat = {side: pt.goals(s.spec)[part]}
    # The part's base where it lies, from the simulator (privileged).
    poses = pt.key_poses(part, [float(v) for v in s.xyz(part)], seat[side])
    pick, place, back = SUBTASKS[side]

    def move(targets, seconds=2.0, **kw):
        s.move(
            targets,
            seconds=seconds,
            linear=True,
            rotations=HAND,
            speed=tt.SPEED,
            **kw,
        )

    s.stage("approach", pick)
    # In joint space (task.swing), within the controller's rates: straight from
    # the start pose, the IK waypoints' corners stretch the move to up to 25 s.
    s.move(
        joints={side: pt.swing(s.kin, side, s.goal[side], poses["above"], HAND[side])},
        grips={side: tt.grip_open(width)},
        seconds=2.5,
    )
    s.stage("align", pick)
    move({side: poses["grasp"]}, 2.0)
    s.stage("grasp", pick)
    s.move(grips={side: tt.grip_closed(width)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, part, side)
    s.stage("lift", pick)
    tt.lift(s, move, {side: poses["lift"]}, {part: RISE[part]}, seconds=1.5)
    s.stage("carry", place)
    move({side: poses["over"]}, 2.0)
    s.stage("centre", place)
    goals = lower(s, move, {side: poses["over"]}, pt.HOVER, 1.0)
    s.move(seconds=0.2)
    goals = centre(s, move, goals, seat)
    s.stage("insert", place)
    goals = insert(s, move, goals, seat)
    s.stage("release", place)
    s.move(grips={side: tt.grip_open(width)}, seconds=1.0)
    s.move(seconds=0.5)
    s.stage("retreat", back)
    move({side: goals[side] + [0, 0, pt.RETREAT]}, 1.5)
    s.stage("return", back)
    pt.sag(s.kin, side, s.goal[side], start)
    s.move(joints={side: start}, grips={side: start_grip}, seconds=2.5)


def expert(s):
    tt.start_home(s)
    start = {side: s.goal[side].copy() for side in tt.SIDES}
    grip = dict(s.grip)
    for side in ("left", "right"):
        work(s, side, start[side], grip[side])
    s.move(seconds=1.0)  # both parts home, both arms in their start pose


def run(output, seed, record=False, kin=None, engine=None):
    """One expert episode for seed into output (must not exist). engine runs it
    on another simulator (isaac.episode.isaac_scene for Isaac Sim)."""
    spec = pt.sample_episode(seed, kin)
    return tt.run_episode(
        InsertionScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {
            "pin_target": s.spec["pin_target"],
            "seat_errors_mm": {
                name: round(1000 * s.off_axis(name, xy), 3)
                for name, xy in pt.goals(s.spec).items()
            },
            "tilts_deg": {n: round(float(np.degrees(s.tilt(n))), 3) for n in s.OBJECTS},
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
