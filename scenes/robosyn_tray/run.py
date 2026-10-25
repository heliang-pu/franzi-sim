"""Two-handed tray carry episode: the scene and a scripted IK expert.

TrayScene is sim.tabletop.ObjectScene with task.py's objects: the episode
framework's audits and recordings, plus object tracking, the task's milestones
(each ingredient on the tray, the tray lifted by both hands, the tray set down
on its target with both ingredients still on it) and the success test. The
expert reads object poses from the simulator (privileged): no welds, no object
pose writes, no contact-mask changes, no robot qpos/qvel writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.robosyn_tray import task as kt
from sim import tabletop as tt


class TrayScene(tt.ObjectScene):
    OBJECTS = ("tray", *kt.FOODS)
    HELD = (*kt.FOODS, "tray")
    MILESTONES = ("apple_on_tray", "tomato_on_tray", "tray_lifted", "tray_on_target")

    def build_root(self):
        return kt.build(self.spec)

    def on_tray(self, name, pairs):
        """Inside the tray's rim, resting on its plate, still."""
        R = self.d.xmat[self.ids["tray"]].reshape(3, 3)
        x, y, z = R.T @ (self.xyz(name) - self.xyz("tray"))
        return (
            abs(x) < kt.RIM_HALF[0] - 0.035
            and abs(y) < kt.RIM_HALF[1] - 0.035
            and abs(z - 2 * kt.PLATE[2]) < 0.004
            and self.settled(name, pairs)
        )

    def held_by_both(self, pairs):
        others = self.touching("tray", pairs)
        return all(others.issuperset(self.jaws[s]) for s in tt.SIDES)

    def conditions(self, pairs):
        tray = self.xyz("tray")
        foods = {f"{n}_on_tray": self.on_tray(n, pairs) for n in kt.FOODS}
        return {
            **foods,
            "tray_lifted": tray[2] > kt.TABLE_Z + 0.02 and self.held_by_both(pairs),
            "tray_on_target": (
                np.linalg.norm(tray[:2] - self.spec["target"]) < 0.02
                and abs(tray[2] - kt.TABLE_Z) < 0.003
                and self.settled("tray", pairs)
                and all(foods.values())
            ),
        }

    def complete(self):
        return self.status()["tray_on_target"]


def base(s, name):
    return [float(v) for v in s.xyz(name)]


def check_handles(s):
    pairs = s.contact_bodies()
    if not s.held_by_both(pairs):
        raise RuntimeError("no bilateral grasp of both hands on the tray's handles")


def expert(s):
    spec = s.spec
    move = tt.mover(s)
    widths = {n: kt.food_width(n) for n in kt.FOODS}
    side_of = kt.SIDE_OF
    post = 2 * kt.POST[0]
    s.stage("ready", "Raise both open grippers.")
    tt.go_ready(s, {side_of[n]: tt.grip_open(widths[n]) for n in kt.FOODS})
    tray = base(s, "tray")[:2]
    load = {
        n: tt.pick_place(
            side_of[n], base(s, n), kt.GRASP_Z, widths[n], kt.on_tray(tray, n), kt.LIFT
        )
        for n in kt.FOODS
    }

    def both(key):
        return {side_of[n]: load[n][key] for n in kt.FOODS}

    s.stage(
        "approach",
        "Reach for the apple with the left hand and the tomato with the right.",
    )
    move(both("pregrasp"))
    s.stage("align", "Lower the open grippers around the apple and the tomato.")
    move(both("grasp"), 1.5)
    s.stage("grasp", "Close both hands.")
    s.move(grips={side_of[n]: tt.grip_closed(widths[n]) for n in kt.FOODS}, seconds=1.0)
    s.move(seconds=0.5)
    for n in kt.FOODS:
        tt.check_grasp(s, n, side_of[n])
    s.stage("lift", "Lift the apple and the tomato.")
    tt.lift(s, move, both("lift"), {n: 0.04 for n in kt.FOODS})
    s.stage("over_tray", "Carry them over the tray.")
    move(both("above"))
    s.stage("load", "Set them down on the tray.")
    move(both("place"))
    s.stage("release", "Let go of both.")
    s.move(grips={side_of[n]: tt.grip_open(widths[n]) for n in kt.FOODS}, seconds=1.0)
    s.stage("back_off", "Back both hands away from the tray.")
    move(both("retreat"), 1.5)
    # Both hands carry the tray by its handles.
    carry = kt.carry_poses(base(s, "tray")[:2], spec["target"])

    def hands(key):
        return {side: carry[side][key] for side in tt.SIDES}

    s.stage("to_handles", "Reach for the tray's two handles.")
    move(hands("pregrasp"), grips={side: tt.grip_open(post) for side in tt.SIDES})
    s.stage("align_handles", "Bring the open grippers around the handles.")
    move(hands("grasp"), 1.5)
    s.stage("grip_handles", "Close both hands on the handles.")
    s.move(
        grips={side: tt.grip_closed(post, kt.POST_SQUEEZE) for side in tt.SIDES},
        seconds=1.0,
    )
    s.move(seconds=0.5)
    check_handles(s)
    s.stage("lift_tray", "Lift the tray with both hands.")
    tt.lift(s, move, hands("up"), {"tray": kt.CARRY_LIFT * 0.6}, seconds=2.0)
    s.stage("carry_tray", f"Carry the tray to the {spec['side']}.")
    move(hands("over"), 3.0)
    s.stage("lower_tray", "Set the tray down.")
    move(hands("down"), 2.0)
    s.stage("let_go", "Let go of the handles.")
    s.move(grips={side: tt.grip_open(post) for side in tt.SIDES}, seconds=1.0)
    s.stage("back_away", "Back both hands away from the tray.")
    move(hands("retreat"), 1.5)
    s.stage("finish", "Return both arms; the loaded tray rests on its new place.")
    move(tt.READY, 3.0)
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None):
    """One expert episode for seed into output (must not exist)."""
    spec = kt.sample_episode(seed, kin)
    return tt.run_episode(
        TrayScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {
            "side": s.spec["side"],
            "tray_offset_mm": 1000
            * float(np.linalg.norm(s.xyz("tray")[:2] - s.spec["target"])),
        },
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
