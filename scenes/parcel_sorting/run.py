"""Parcel sorting episode: the scene and a scripted IK expert.

SortScene is sim.tabletop.ObjectScene with task.py's four parcels: the episode
framework's audits and recordings, plus object tracking, the milestones (each
parcel picked up by its arm, then lying in the bin of its zone) and the
success test, every parcel in the bin of its zone. The expert reads the
parcels' poses from the simulator and their zones from the episode spec
(privileged): no welds, no object pose writes, no contact-mask changes, no
robot qpos/qvel writes.

One arm moves at a time, in single-arm subtasks (user, 2026-10-25: an
episode splits at its subtask boundaries; the trace's subtask is the coarse
text, its phase the fine stage): the arm that goes first picks and places its
two parcels in the order the spec draws and returns to its start pose, then
the other arm does the same, while the idle arm's joint goals and gripper stay
as they are. A pick is a joint-space move to a staging pose over the parcel
(elbow down), straight down and in along the 45-degree approach, close and
straight up off the belt. A place is straight up to the carry height, turned
there in place to the bin's azimuth, straight back off the belt and across
over the bin, straight down until the parcel's lowest corner is 5 mm over the
bin floor, levelled and set down to 1 mm, let go, and straight up again. A
return is a joint-space move back to the default posture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.parcel_sorting import task as kt
from sim import tabletop as tt

NAMES = tuple(f"parcel{k}" for k in range(len(kt.PARCELS)))
# Gripper closures: open around a parcel, closed on it, and the default
# posture's (tt.start_home), which each arm's return restores.
OPENED = 1 - kt.OPEN / tt.JAW_SPAN
CLOSED = tt.grip_closed(kt.PARCEL_W, kt.SQUEEZE)
HOME_GRIP = 1 - tt.HOME_OPENING / tt.JAW_SPAN


class SortScene(tt.ObjectScene):
    OBJECTS = NAMES
    HELD = NAMES
    MILESTONES = tuple(
        f"{zone}_{what}" for zone in kt.BINS for what in ("picked", "sorted")
    )
    # A held parcel may touch nothing but the robot from the end of its first
    # lift until it is set down on the bin floor.
    CARRY_PHASES = ("carry", "lower")

    def build_root(self):
        return kt.build(self.spec)

    def in_bin(self, name, zone, pairs):
        """Lying flat on the floor of zone's bin, inside its walls, still and
        clear of the robot."""
        x, y, z = self.xyz(name)
        bx, by = self.spec["bins"][zone]
        return (
            abs(x - bx) < kt.BIN_INNER[0]
            and abs(y - by) < kt.BIN_INNER[1]
            and abs(z - kt.BIN_FLOOR) < 0.003
            and self.settled(name, pairs)
        )

    def conditions(self, pairs):
        now = {}
        for name, parcel in zip(NAMES, self.spec["parcels"], strict=True):
            zone = parcel["zone"]
            held = self.touching(name, pairs).issuperset(self.jaws[kt.SIDE_OF[zone]])
            lifted = self.xyz(name)[2] > kt.BELT_Z + kt.LIFT + 0.005
            now[f"{zone}_picked"] = held and lifted
            now[f"{zone}_sorted"] = self.in_bin(name, zone, pairs)
        return now

    def complete(self):
        now = self.status()
        return all(now[f"{zone}_sorted"] for zone in kt.BINS)


def parcel_yaw(s, name):
    """A parcel's measured yaw off lying across the belt, folded into +-90
    degrees (a box turned half a turn grasps the same)."""
    R = s.d.xmat[s.ids[name]].reshape(3, 3)
    return kt.folded(np.arctan2(R[1, 0], R[0, 0]) - kt.ACROSS)


def place_target(s, name, site, gap, drop=None):
    """Where to move the hand, now at site, so the held parcel's lowest corner
    comes gap over the bin floor and (with drop) its centre over drop (x, y):
    site shifted by how far the parcel, measured now (privileged), is off that.
    A held parcel creeps down and tilts in the jaws (0.27 mm and 0.1 degree a
    second); lowered by the planned poses, a parcel carried 19 s sat on the bin
    floor before it was let go."""
    R = s.d.xmat[s.ids[name]].reshape(3, 3)
    base = np.asarray(s.xyz(name), float)
    half = (kt.PARCEL_W / 2, kt.PARCEL_L / 2)
    low = min(
        (base + R @ [u * half[0], v * half[1], 0])[2] for u in (-1, 1) for v in (-1, 1)
    )
    shift = np.array([0.0, 0.0, kt.BIN_FLOOR + gap - low])
    if drop is not None:
        shift[:2] = np.asarray(drop) - base[:2]
    return np.asarray(site, float) + shift


def level_target(s, name, site, rotation, gap):
    """The hand pose (site, rotation) that levels the held parcel, turning the
    hand, now at site with rotation, about the grasp site by the parcel's
    measured tilt (privileged), and brings its base gap over the bin floor.
    Let go with its lowest corner 1 mm over the floor but tilted 1.8 degrees, a
    parcel's other end fell 4 mm and landed 0.64 mm deep."""
    R = s.d.xmat[s.ids[name]].reshape(3, 3)
    up = R[:, 2]
    axis = np.cross(up, [0.0, 0.0, 1.0])
    sin, cos = np.linalg.norm(axis), up[2]
    turn = np.eye(3)
    if sin > 1e-6:
        k = axis / sin
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        turn = np.eye(3) + sin * K + (1 - cos) * K @ K
    base = np.asarray(s.xyz(name), float)
    site = np.asarray(site, float)
    want = [base[0], base[1], kt.BIN_FLOOR + gap]
    return want - turn @ (base - site), turn @ np.asarray(rotation, float)


def check_held(s, name, side, z, what):
    """Both jaws of side on the parcel, whose base is within 1 cm of z."""
    if s.object_info(name)["jaw_contacts"][side] != [1, 2]:
        raise RuntimeError(f"{side}: lost the {name} while {what}")
    if abs(s.xyz(name)[2] - z) > 0.01:
        raise RuntimeError(f"{side}: the {name} slipped in the hand while {what}")


def to(s, side, poses, key, rotation, seconds=None):
    """Side's hand straight to poses[key]; the other arm holds."""
    s.move(
        {side: poses[key]},
        seconds=seconds or kt.SECONDS[key],
        linear=True,
        rotations={side: rotation},
        speed=tt.SPEED,
    )


def pick(s, side, k):
    """One subtask: pick up parcel k with side's hand (approach, align, grasp,
    lift) from its measured pose. Returns the parcel's key poses."""
    parcel = s.spec["parcels"][k]
    name, zone = NAMES[k], parcel["zone"]
    poses = kt.key_poses(
        side,
        [float(v) for v in s.xyz(name)],
        parcel_yaw(s, name),
        s.spec["place_psi"][zone],
        parcel["drop"],
    )
    text = f"Pick up parcel {zone} with the {side} hand."
    # The hand turns from the default posture as part of its first pick.
    s.stage("approach", text)
    s.move(
        joints={side: kt.stage_pose(s.kin, side, poses, s.body())},
        grips={side: OPENED},
        seconds=3.0,
        speed=tt.SPEED,
    )
    s.stage("align", text)
    to(s, side, poses, "pregrasp", poses["pick"])
    to(s, side, poses, "grasp", poses["pick"])
    s.stage("grasp", text)
    s.move(grips={side: CLOSED}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, name, side)
    s.stage("lift", text)
    start = float(s.xyz(name)[2])
    to(s, side, poses, "lift", poses["pick"])
    if s.xyz(name)[2] < start + 0.6 * kt.LIFT:
        raise RuntimeError(f"{side}: lift failed for the {name}")
    return poses


def place(s, side, k, poses):
    """One subtask: put the held parcel k into the bin of its zone with side's
    hand (carry, lower, set_down, release, rise)."""
    parcel = s.spec["parcels"][k]
    name, zone = NAMES[k], parcel["zone"]
    put = poses["put"]
    text = f"Put parcel {zone} into bin {zone} with the {side} hand."
    s.stage("carry", text)
    to(s, side, poses, "raise", poses["pick"])
    for q in kt.turn_poses(s.kin, side, poses, s.goal[side], s.body()):
        s.move(joints={side: q}, seconds=0.5, speed=tt.SPEED)
    to(s, side, poses, "back", put)
    to(s, side, poses, "over", put)
    check_held(s, name, side, kt.CARRY_Z, "carrying it")
    poses["place"] = place_target(s, name, poses["over"], kt.RELEASE, parcel["drop"])
    s.stage("lower", text)
    to(s, side, poses, "place", put)
    check_held(s, name, side, kt.BIN_FLOOR + kt.RELEASE, "lowering it")
    poses["set"], level = level_target(s, name, poses["place"], put, kt.SET)
    s.stage("set_down", text)
    to(s, side, poses, "set", level)
    s.stage("release", text)
    s.move(grips={side: OPENED}, seconds=1.0)
    s.move(seconds=0.5)
    s.stage("rise", text)
    to(s, side, poses, "over", put, seconds=2.0)


def back(s, side):
    """One subtask: side's arm back to its start pose, the default posture
    (a joint-space move), its jaws as they started."""
    s.stage("return", f"Move the {side} arm back to its start pose.")
    s.move(
        joints={side: kt.home(s.kin, side)},
        grips={side: HOME_GRIP},
        seconds=3.0,
        speed=tt.SPEED,
    )


def expert(s):
    order = kt.sequence(s.spec)
    tt.start_home(s)
    for i, (side, k) in enumerate(order):
        place(s, side, k, pick(s, side, k))
        if i + 1 == len(order) or order[i + 1][0] != side:
            back(s, side)
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None, engine=None):
    """One expert episode for seed into output (must not exist). engine runs it
    on another simulator (isaac.episode.isaac_scene for Isaac Sim)."""
    spec = kt.sample_episode(seed, kin)
    return tt.run_episode(
        SortScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {
            "zones": {n: p["zone"] for n, p in zip(NAMES, s.spec["parcels"])},
            "sequence": [[side, NAMES[k]] for side, k in kt.sequence(s.spec)],
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
