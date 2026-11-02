"""Palletizing episode: the scene and a scripted IK expert.

PalletScene is sim.tabletop.ObjectScene with task.py's four cartons: the
episode framework's audits and recordings, plus object tracking, the
milestones (each of plan_04's layer-1 cells holding a carton) and the success
test. The expert fills the cells one at a time in task.py's order, near cells
first. Each carton is three single-arm subtasks (user rule 2026-10-25: an
episode splits cleanly at them, data.subtask_check): pick it up, stack it in
its slot, move the arm back (after its first carton to its ready pose, the
hand square at READY, after its second to its start pose); the other arm
holds its goals and jaws meanwhile. The trace's subtask is that text, its phase the
fine stage. The expert reads the cartons' poses from the simulator (privileged):
no welds, no object pose writes, no contact-mask changes, no robot qpos/qvel
writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.palletizing import task as pt
from sim import tabletop as tt


class PalletScene(tt.ObjectScene):
    OBJECTS = pt.CARTONS
    HELD = pt.CARTONS
    MILESTONES = tuple(f"slot_{n}" for n in pt.SLOTS)
    # A carried carton may touch only the robot: while it turns square over
    # the stand, over the placed ones and while it is lowered past them (6 mm
    # from its column's neighbour, 40 mm across), down to 3 mm over the deck.
    # The last 2.5 mm ("set") are left out: PhysX reports pairs within 0.5 mm
    # as touching (isaac/engine.py).
    CARRY_PHASES = ("square", "carry", "lower")

    def build_root(self):
        return pt.build(self.spec)

    def yaw(self, name):
        """The carton's turn (rad) from the plan's orientation, long side along
        x, folded into ±90 degrees."""
        R = self.d.xmat[self.ids[name]].reshape(3, 3)
        long_side = R[:, 1]  # the prop's y, its 120 mm side, along -x unturned
        turn = np.arctan2(long_side[1], long_side[0]) - np.pi
        return float((turn + np.pi / 2) % np.pi - np.pi / 2)

    def in_slot(self, name, slot, pairs):
        """In the cell, square, flat on the deck, still and clear of the robot."""
        x, y, z = self.xyz(name)
        cx, cy = pt.slots(self.spec)[slot]
        return (
            abs(x - cx) <= pt.SLOT_TOL
            and abs(y - cy) <= pt.SLOT_TOL
            and abs(z - pt.DECK_Z) < pt.SEAT_TOL
            and abs(self.yaw(name)) <= pt.YAW_TOL
            and self.settled(name, pairs)
        )

    def conditions(self, pairs):
        return {
            f"slot_{n}": any(self.in_slot(c, n, pairs) for c in self.OBJECTS)
            for n in pt.SLOTS
        }

    def complete(self):
        return all(self.status().values())

    def placements(self):
        """Each cell's carton (the nearest one's name if it is in the cell, else
        None), its offset from the cell centre (mm) and turn (deg)."""
        pairs = self.contact_bodies()
        out = {}
        for n, (cx, cy) in pt.slots(self.spec).items():
            name = min(
                self.OBJECTS,
                key=lambda c: np.hypot(self.xyz(c)[0] - cx, self.xyz(c)[1] - cy),
            )
            x, y, z = self.xyz(name)
            out[str(n)] = {
                "carton": name if self.in_slot(name, n, pairs) else None,
                "nearest": name,
                "offset_mm": [round(1000 * (x - cx), 2), round(1000 * (y - cy), 2)],
                "height_mm": round(1000 * (z - pt.DECK_Z), 2),
                "yaw_deg": round(float(np.degrees(self.yaw(name))), 2),
            }
        return out


OPENED = 1 - (pt.BOX_W + 2 * pt.ROOM) / tt.JAW_SPAN
CLOSED = tt.grip_closed(pt.BOX_W, pt.SQUEEZE)
HOME = 1 - tt.HOME_OPENING / tt.JAW_SPAN  # the default posture's jaws
SECONDS = {"approach": 2.5, "carry": 2.0, "set": 1.0, "square": 0.3}


def label(name):
    """The carton's name in the subtask texts, by its place on its stand:
    "left inner carton" (nearer the robot's midline) or "left outer carton"."""
    side = "left" if "left" in name else "right"
    return f"{side} {'inner' if name.endswith('0') else 'outer'} carton"


def line(s, side, target, rotation, seconds, **kw):
    """One hand on a straight line to target; the other holds its goal."""
    s.move(
        {side: target},
        seconds=seconds,
        linear=True,
        rotations={side: rotation},
        speed=tt.SPEED,
        **kw,
    )


def carton_poses(s, job):
    """task.key_poses for the job's carton as it stands (privileged)."""
    name, slot = job["carton"], job["slot"]
    base = [float(v) for v in s.xyz(name)]
    place_at = pt.place_xy(s.spec, slot)
    return pt.key_poses(job["side"], base, s.yaw(name), place_at, pt.row_of(slot))


def held_offset(s, side, name):
    """Where the carried carton is in the hand, measured (privileged): its base
    centre's x, y and its lowest base corner's z, from the grasp site."""
    R = s.d.xmat[s.ids[name]].reshape(3, 3)
    base = np.array(s.xyz(name), float)
    low = base[2] + (pt.BASE_CORNERS @ R.T)[:, 2].min()
    return np.array([base[0], base[1], low]) - s.d.site_xpos[s.sites[side]]


def set_down(s, side, name, slot, height):
    """The grasp-site target putting the carried carton's base centre over its
    place in slot and its lowest corner height (m) over the deck, from where
    the carton sits in the hand now. Under MuJoCo it creeps down the pads (4.5
    mm over a 15 s carry); under PhysX it can pivot about the pads instead."""
    return np.array([*pt.place_xy(s.spec, slot), pt.DECK_Z + height]) - held_offset(
        s, side, name
    )


def pick(s, job):
    """Subtask: one hand turns in place at READY to its carton's yaw, takes the
    carton and lifts it off the stand. Turned on the approach line instead,
    the hand's first waypoint jumped and the rate limiter stretched the
    approach to about 8 s."""
    side, name = job["side"], job["carton"]
    text = f"Pick up the {label(name)} with the {side} hand."
    poses = carton_poses(s, job)
    s.stage("turn", text)
    s.move(
        {side: tt.READY[side]},
        grips={side: OPENED},
        rotations={side: poses["pick"]},
        seconds=1.5,
        speed=tt.SPEED,
    )
    for stage, key, rotation in pt.PICK_MOVES:
        s.stage(stage, text)
        line(s, side, poses[key], poses[rotation], SECONDS.get(stage, 1.5))
        if stage == "align":
            s.stage("grasp", text)
            s.move(grips={side: CLOSED}, seconds=1.0)
            s.move(seconds=0.5)
            tt.check_grasp(s, name, side)
            start = float(s.xyz(name)[2])
    if s.xyz(name)[2] < start + 0.8 * pt.LIFT_H:
        raise RuntimeError(f"{side}: lift failed for {name}")
    return poses


def stack(s, job, poses):
    """Subtask: carry the carton to its cell, set it down and let go. Each
    move down is aimed by the carton as it is held then (set_down)."""
    side, name, slot = job["side"], job["carton"], job["slot"]
    text = f"Stack the {label(name)} in slot {slot} with the {side} hand."
    heights = {"low": pt.LOW, "place": pt.SET, "set": pt.RELEASE}
    for stage, key, rotation in pt.STACK_MOVES:
        s.stage(stage, text)
        if stage == "square":  # turned in place over the stand, step by step
            for turn in poses["turns"]:
                line(s, side, poses[key], turn, SECONDS["square"])
            continue
        target = poses[key]
        if key in heights:
            target = set_down(s, side, name, slot, heights[key])
        line(s, side, target, poses[rotation], SECONDS.get(stage, 1.5))
    s.stage("release", text)
    s.move(grips={side: OPENED}, seconds=1.0)
    s.move(seconds=0.3)


def back(s, side, poses, joints, home):
    """Subtask: the hand rises clear of the cartons, leaves the pallet
    outboard and returns to READY, then to joints: the arm's start pose (home:
    its last carton, the jaws as they started) or its ready pose, from which
    its next carton planned."""
    where = "start" if home else "ready"
    text = f"Move the {side} arm back to its {where} pose."
    ((stage, key, rotation),) = pt.BACK_MOVES
    s.stage(stage, text)
    line(s, side, poses[key], poses[rotation], 1.5)
    s.stage("leave", text)
    line(s, side, poses["exit"], poses["put"], 1.5)
    line(s, side, tt.READY[side], poses["put"], 2.5)
    s.stage("home" if home else "ready", text)
    grips = {side: HOME if home else OPENED}
    s.move(joints={side: joints}, grips=grips, seconds=1.0, speed=tt.SPEED)


def expert(s):
    jobs = s.spec["steps"]
    tt.start_home(s)
    start = {side: s.goal[side].copy() for side in tt.SIDES}
    # The ready pose between an arm's cartons (task.ready_pose): the default
    # posture's joints with the hand turned square in place.
    ready = {
        side: s.ik(side, tt.READY[side], rotation=pt.grasp_rotation(side))
        for side in tt.SIDES
    }
    for k, job in enumerate(jobs):
        side = job["side"]
        poses = pick(s, job)
        stack(s, job, poses)
        last = all(later["side"] != side for later in jobs[k + 1 :])
        back(s, side, poses, start[side] if last else ready[side], home=last)
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None, engine=None):
    """One expert episode for seed into output (must not exist). engine runs it
    on another simulator (isaac.episode.isaac_scene for Isaac Sim)."""
    spec = pt.sample_episode(seed, kin)
    return tt.run_episode(
        PalletScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {"first": s.spec["first"], "placements": s.placements()},
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
