"""Shaped kitting episode: the scene and a scripted IK expert.

ShapedKitScene is sim.tabletop.ObjectScene with task.py's four loose parts:
the episode framework's audits and recordings, plus object tracking, the
milestones (each part lifted by its arm, then kitted) and the success test:
every part seated in its own pocket, centred, upright, at the pocket's yaw
(the hex blank's flats, the bracket's legs), still and clear of the robot.

The episode splits into single-arm subtasks (user rule 2026-10-25): each
part, in the drawn order, is picked up and then put into its pocket by the
arm that owns it while the other arm holds still, and each arm goes back to
its start pose in a subtask of its own after its last part. A subtask's text
(the trace's subtask) names the arm and the part; its fine stages are the
trace's phase. The hand works top-down with the jaws along x, the right hand
turned 180 degrees (task.HAND_YAW): it turns at READY in its first pick and
back in its return. The expert reads the parts' poses from the simulator
(privileged): it closes on each where it lies, turned to its yaw, lifts it,
turns it in place to the pocket's yaw, centres it over its pocket from its
measured pose in the hand, lowers it with the fingers in the pocket's notches
until the part's lowest point is 3 mm over the floor, sets it down to 0.5 mm
and lets go. No welds, no object pose writes, no contact-mask changes, no
robot qpos/qvel writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.shaped_kitting import task as kt
from sim import tabletop as tt

# Kitted: the part's origin within POCKET_XY of its pocket's centre (the
# pockets leave 4 mm per side), its base within POCKET_DZ of the floor, and a
# hex blank or bracket within POCKET_YAW of the pocket's yaw (the pocket lets
# the hex turn freely and the bracket about 7 degrees).
POCKET_XY = 0.005
POCKET_DZ = 0.003
POCKET_YAW = np.radians(10)
# The jaws open this much wider than the part to let go in the pocket: 3 mm
# per side keeps the fingers inside the notches.
RELEASE_OPEN = 0.006
# Centring a carried part over its pocket: the hand may leave the notches'
# centre line by NOTCH_SLACK across the jaws (the fingers have 2.5 mm per side
# there) and by SLIDE_SLACK along them.
NOTCH_SLACK = 0.0015
SLIDE_SLACK = 0.003
# A part whose grasp point would still miss its pocket's by more than this
# after the hand's shift slipped in the hand: stop rather than scrape a wall.
MISS_MAX = 0.002
CENTRE_MIN = 0.0002  # smaller corrections over the pocket need no move
# The carried descent stops with the part's lowest point SET_GAP (3 mm) over
# the pocket floor; the release then sets it down to TOUCH_GAP before the jaws
# open. Let go from 2.6 mm, the 29 g gear landed 0.66 mm deep (the audit stops
# at 1 mm).
TOUCH_GAP = 0.0005
TURN_MIN = np.radians(0.2)  # smaller yaw differences need no turn


def yaw_of(xmat):
    R = np.asarray(xmat).reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


class ShapedKitScene(tt.ObjectScene):
    OBJECTS = tuple(kt.PARTS)
    HELD = tuple(kt.PARTS)
    MILESTONES = tuple(f"{n}_{k}" for n in kt.PARTS for k in ("lifted", "kitted"))
    # Carried through the air: there a held part touching anything but the
    # robot is a collision. That covers the turns at carry height, the carry
    # over the tray, the centring over the pocket and the descent into it,
    # which stops 3 mm over its floor, so a part touching the foam there
    # scraped a pocket wall. The lift (the part leaves the bench) and the
    # release (the part is set down on the floor in the jaws) touch the bench
    # and the foam by design.
    CARRY_PHASES = ("turn", "transfer", "centre", "insert")

    def __init__(self, output, spec, record=False):
        super().__init__(output, spec, record)
        m = self.m

        def vertices(g):
            start = m.mesh_vertadr[m.geom_dataid[g]]
            return m.mesh_vert[start : start + m.mesh_vertnum[m.geom_dataid[g]]].copy()

        # Each part's collision pieces with their mesh vertices (geom frame).
        self.pieces = {
            name: [
                (g, vertices(g))
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == bid and m.geom_contype[g] | m.geom_conaffinity[g]
            ]
            for name, bid in self.ids.items()
        }

    def build_root(self):
        return kt.build(self.spec)

    def lowest(self, name):
        """The height (m) of the part's lowest collision vertex."""
        d = self.d
        return min(
            float((d.geom_xpos[g][2] + v @ d.geom_xmat[g].reshape(3, 3)[2]).min())
            for g, v in self.pieces[name]
        )

    def tilt(self, name):
        """Angle (rad) between the part's axis and the vertical."""
        return float(np.arccos(np.clip(self.d.xmat[self.ids[name]][8], -1, 1)))

    def yaw(self, name):
        """The part's yaw off its pocket's, folded by its symmetry (0 for a
        round part)."""
        psi = kt.tray_pose(self.spec)[1]
        return kt.fold(yaw_of(self.d.xmat[self.ids[name]]) - psi, kt.SYMMETRY[name])

    def kitted(self, name, pairs):
        x, y, z = self.xyz(name)
        cx, cy = kt.pocket_xy(kt.tray_pose(self.spec), kt.PARTS[name][0])
        return (
            np.hypot(x - cx, y - cy) < POCKET_XY
            and abs(z - kt.FLOOR_Z) < POCKET_DZ
            and abs(self.yaw(name)) < POCKET_YAW
            and self.settled(name, pairs)
        )

    def conditions(self, pairs):
        now = {}
        for name, (_, _, side) in kt.PARTS.items():
            held = self.touching(name, pairs).issuperset(self.jaws[side])
            lifted = self.xyz(name)[2] > self.rest_z[name] + 0.03
            now[f"{name}_lifted"] = held and lifted
            now[f"{name}_kitted"] = self.kitted(name, pairs)
        return now

    def complete(self):
        now = self.status()
        return all(now[f"{name}_kitted"] for name in kt.PARTS)


def measured(s, name):
    """A part's measured base (x, y, z) and yaw."""
    return [float(v) for v in s.xyz(name)], yaw_of(s.d.xmat[s.ids[name]])


def in_hand(s, side, name):
    """The measured grasp point of part name (task.grasp_xy at its pose) less
    side's grasp site, in x and y (m)."""
    point = kt.grasp_xy(name, s.xyz(name)[:2], yaw_of(s.d.xmat[s.ids[name]]))
    return point - s.d.site_xpos[s.sites[side]][:2]


def aim(s, side, name, poses):
    """Grasp-site targets over the part's pocket ("over"), where it lets go
    ("release") and where the hand then rises to ("clear"), from its measured
    pose in the hand: its grasp point over its pocket's, the hand within
    NOTCH_SLACK across the jaws and SLIDE_SLACK along them (the tray's y and
    x) of the notches' centre line, and the part's lowest point SET_GAP over
    the pocket floor (parts creep down in the jaws under MuJoCo, about 0.2
    mm/s, and pivot 2-4 degrees under PhysX)."""
    offset = in_hand(s, side, name)
    R = tt.rot_z(kt.tray_pose(s.spec)[1])[:2, :2]
    slack = np.array([SLIDE_SLACK, NOTCH_SLACK])
    shift = R @ np.clip(-R.T @ offset, -slack, slack)
    site = s.d.site_xpos[s.sites[side]]
    low = s.lowest(name)
    print(
        f"  {name}: in hand {1000 * offset[0]:+.2f} {1000 * offset[1]:+.2f} mm,"
        f" base {1000 * (s.xyz(name)[2] - site[2]):+.2f} mm"
        f" (lowest {1000 * (low - s.xyz(name)[2]):+.2f}),"
        f" yaw {np.degrees(s.yaw(name)):+.2f}, tilt {np.degrees(s.tilt(name)):.2f}"
        " deg",
        flush=True,
    )
    miss = float(np.linalg.norm(offset + shift))
    if miss > MISS_MAX:
        raise RuntimeError(f"the {name} sits {1000 * miss:.1f} mm off in the hand")
    xy = np.asarray(poses["over"][:2]) + shift
    drop = low - (kt.FLOOR_Z + kt.SET_GAP)
    return {
        "over": [*xy, poses["over"][2]],
        "release": [*xy, float(site[2] - drop)],
        "clear": [*xy, poses["clear"][2]],
    }


def set_down(s, side, name, goals):
    """The grasp-site target straight below the release goal that brings the
    part's measured lowest point TOUCH_GAP over its pocket floor."""
    site = s.d.site_xpos[s.sites[side]]
    drop = s.lowest(name) - (kt.FLOOR_Z + TOUCH_GAP)
    return [*goals["release"][:2], float(site[2] - drop)]


class Arm:
    """One arm's straight-line moves and in-place turns at the scene speed; the
    other arm's joint goals and gripper stay as they are (Scene.move with one
    target), so every subtask moves one arm only. It keeps the hand's yaw (the
    hand is top-down throughout) and the target it last moved to."""

    def __init__(self, s, side):
        self.s, self.side = s, side
        self.yaw = 0.0  # tt.start_home's hand
        self.at = tt.READY[side]
        self.home = None  # the start pose's joints, once taken
        self.chain = []  # joint goals along the first (180-degree) turn

    def move(self, target, seconds=2.0, grip=None, speed=tt.SPEED, jump=None):
        """A straight line to target, refused (RuntimeError) if its chained IK
        would jump branches (task.JUMP): the hand would swing off the line."""
        s = self.s
        kt.chained(
            s.kin,
            self.side,
            s.goal[self.side],
            target,
            self.yaw,
            jump or kt.JUMP["line"],
        )
        self.s.move(
            {self.side: target},
            grips=None if grip is None else {self.side: grip},
            seconds=seconds,
            linear=True,
            rotations={self.side: tt.rot_z(self.yaw)},
            speed=speed,
        )
        self.at = target

    def grip(self, closure, seconds):
        self.s.move(grips={self.side: closure}, seconds=seconds)

    def turn(self, stage, instruction, target, yaw):
        """Turn the hand in place at target to yaw, high above everything, in
        chained steps of at most task.TURN_STEP (a stage of its own when it
        turns more than TURN_MIN). A turn of several steps (the 180-degree
        turn at READY, where every joint stays clear of its limits) runs at
        the controller's rate limits alone, not at tt.SPEED: 7 s instead of
        10."""
        steps = kt.turn_yaws(self.yaw, yaw)
        speed = tt.SPEED if len(steps) == 1 else None
        goals = [self.s.goal[self.side].copy()]
        if abs(yaw - self.yaw) > TURN_MIN:
            self.s.stage(stage, instruction)
            for step in steps:
                self.yaw = step
                self.move(
                    target,
                    1.2 if len(steps) == 1 else 0.3,
                    speed=speed,
                    jump=kt.JUMP["turn"],
                )
                goals.append(self.s.goal[self.side].copy())
        self.yaw = yaw
        return goals


def pick(s, arm, name):
    """Subtask: pick the part up with its arm, turned to its yaw. Returns its
    key poses. The right hand first turns 180 degrees at READY (task.HAND_YAW)."""
    width, _, squeeze, _ = kt.GRIP[name]
    text = f"Pick up the {kt.LABEL[name]} with the {arm.side} hand."
    poses = kt.key_poses(name, *measured(s, name), kt.tray_pose(s.spec))
    if abs(kt.HAND_YAW[arm.side] - arm.yaw) > np.pi / 2:
        arm.chain = arm.turn("orient", text, arm.at, kt.HAND_YAW[arm.side])
    s.stage("approach", text)
    arm.move(poses["above"], 2.5, grip=tt.grip_open(width))
    arm.turn("orient", text, poses["above"], poses["grasp_yaw"])
    s.stage("align", text)
    arm.move(poses["grasp"], 2.0)
    s.stage("grasp", text)
    arm.grip(tt.grip_closed(width, squeeze), 1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, name, arm.side)
    s.stage("lift", text)
    start = float(s.xyz(name)[2])
    arm.move(poses["above"], 2.0)
    if s.xyz(name)[2] < start + 0.05:
        raise RuntimeError(f"lift failed for the {name}")
    return poses


def place(s, arm, name, poses):
    """Subtask: turn the part to its pocket's yaw, carry it over its pocket,
    centre it from its measured pose in the hand, lower it into the pocket, set
    it down, let go and raise the hand clear of it."""
    width = kt.GRIP[name][0]
    text = f"Put the {kt.LABEL[name]} into its pocket with the {arm.side} hand."
    arm.turn("turn", text, poses["above"], poses["hand_yaw"])
    goals = aim(s, arm.side, name, poses)
    s.stage("transfer", text)
    for target in poses["via"]:
        arm.move(target, 2.0)
    arm.move(goals["over"], 2.5)
    last, goals = goals, aim(s, arm.side, name, poses)
    if np.linalg.norm(np.subtract(goals["over"], last["over"])) > CENTRE_MIN:
        s.stage("centre", text)
        arm.move(goals["over"], 0.6)
        goals = aim(s, arm.side, name, poses)
    s.stage("insert", text)
    arm.move(goals["release"], 2.0)
    s.stage("release", text)
    arm.move(set_down(s, arm.side, name, goals), 0.8)
    arm.grip(1 - (width + RELEASE_OPEN) / tt.JAW_SPAN, 0.8)
    s.move(seconds=0.4)
    s.stage("retreat", text)
    arm.move(goals["clear"], 1.5)


def go_home(s, arm):
    """Subtask: the arm back to its start pose: via task.RETURN_VIA to READY,
    the hand turned to its HAND_YAW there, a joint move onto the pose its
    first turn ended on (the start pose for the left hand) and, for the right
    hand, that turn's recorded steps backwards, which end on the start pose's
    joints (task.SWEEP_DRIFT says why)."""
    side = arm.side
    text = f"Move the {side} arm back to its start pose."
    s.stage("return", text)
    arm.move(kt.RETURN_VIA[side], 2.0, grip=1 - tt.HOME_OPENING / tt.JAW_SPAN)
    arm.move(tt.READY[side], 2.5)
    arm.turn("orient", text, tt.READY[side], kt.HAND_YAW[side])
    s.stage("return", text)
    onto = arm.chain[-1] if arm.chain else arm.home
    kt.sweep(s.kin, side, s.goal[side], onto)
    s.move(joints={side: onto}, seconds=1.0, speed=tt.SPEED)
    if arm.chain:
        s.stage("orient", text)
        for q in reversed(arm.chain[:-1]):
            s.move(joints={side: q}, seconds=0.3)
    arm.yaw, arm.at = 0.0, tt.READY[side]


def expert(s):
    """Each part in the drawn order: picked and placed by the arm that owns
    it, while the other arm holds; each arm goes back to its start pose right
    after its last part."""
    arms = {side: Arm(s, side) for side in tt.SIDES}
    order = s.spec["order"]
    tt.start_home(s)
    for side, arm in arms.items():
        arm.home = s.goal[side].copy()
    for k, name in enumerate(order):
        arm = arms[kt.PARTS[name][2]]
        poses = pick(s, arm, name)
        place(s, arm, name, poses)
        if all(kt.PARTS[n][2] != arm.side for n in order[k + 1 :]):
            go_home(s, arm)
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None, engine=None):
    """One expert episode for seed into output (must not exist). engine runs it
    on another simulator (isaac.episode.isaac_scene for Isaac Sim)."""
    spec = kt.sample_episode(seed, kin)
    return tt.run_episode(
        ShapedKitScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {"order": s.spec["order"]},
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
