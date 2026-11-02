"""Basket episode: the scene and a scripted IK expert.

BasketScene is sim.tabletop.ObjectScene with task.py's objects: the episode
framework's audits and recordings, plus object tracking, the task's milestones
(the milk in the basket, the basket lifted by its handle, the basket set down
on its target with the milk inside, the milk on the counter) and the success
test. The expert reads object poses from the simulator (privileged): no welds,
no object pose writes, no contact-mask changes, no robot qpos/qvel writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.robosyn_basket import task as kt
from sim import tabletop as tt


class BasketScene(tt.ObjectScene):
    OBJECTS = ("basket", "milk")
    HELD = ("milk", "basket")
    MILESTONES = ("milk_in_basket", "basket_lifted", "basket_moved", "milk_on_counter")

    def build_root(self):
        return kt.build(self.spec)

    def in_basket(self, pairs):
        R = self.d.xmat[self.ids["basket"]].reshape(3, 3)
        x, y, z = R.T @ (self.xyz("milk") - self.xyz("basket"))
        dx, dy = kt.INSIDE["milk"]
        return (
            abs(x - dx) < 0.04
            and abs(y - dy) < 0.05
            and abs(z - kt.DIMS["basket"]["floor_z"]) < 0.008
            and self.settled("milk", pairs)
        )

    def on_counter(self, pairs):
        milk = self.xyz("milk")
        return (
            np.linalg.norm(milk[:2] - kt.counter(self.spec["side"])) < 0.03
            and abs(milk[2] - (kt.TABLE_Z + 2 * kt.COUNTER_HALF[2])) < 0.004
            and self.settled("milk", pairs)
        )

    def conditions(self, pairs):
        basket = self.xyz("basket")
        inside = self.in_basket(pairs)
        held = self.touching("basket", pairs).issuperset(self.jaws[self.spec["side"]])
        there = (
            np.linalg.norm(basket[:2] - self.spec["target"]) < kt.TARGET_TOL
            and abs(basket[2] - kt.TABLE_Z) < 0.004
            and self.settled("basket", pairs)
        )
        return {
            "milk_in_basket": inside,
            "basket_lifted": basket[2] > kt.TABLE_Z + 0.03 and held,
            "basket_moved": there and inside,
            "milk_on_counter": there and self.on_counter(pairs),
        }

    def complete(self):
        return self.status()["milk_on_counter"]


def base(s, name):
    return [float(v) for v in s.xyz(name)]


def expert(s):
    spec = s.spec
    side = spec["side"]
    other = "right" if side == "left" else "left"
    upright = {k: tt.grasp_rotation(k) for k in tt.SIDES}
    move = tt.mover(s)
    w = kt.width("milk")
    handle_w = kt.DIMS["basket"]["handle_width"]
    s.stage("ready", "Raise both open grippers.")
    tt.go_ready(s, {k: tt.grip_open(w) for k in tt.SIDES})
    # The left hand packs the milk.
    basket = base(s, "basket")[:2]
    pack = tt.pick_place(
        "left",
        base(s, "milk"),
        kt.GRASP_Z["milk"],
        w,
        kt.in_basket(basket, "milk"),
        kt.LIFT,
    )
    s.stage("approach", "Reach for the milk carton with the left hand.")
    move({"left": pack["pregrasp"]})
    s.stage("align", "Lower the open left gripper around the milk carton.")
    move({"left": pack["grasp"]}, 1.5)
    s.stage("grasp", "Close the left hand on the milk carton.")
    s.move(grips={"left": tt.grip_closed(w)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, "milk", "left")
    s.stage("lift", "Lift the milk carton over the basket's rim.")
    tt.lift(s, move, {"left": pack["lift"]}, {"milk": 0.08}, seconds=2.0)
    s.stage("over_basket", "Carry the milk carton over the basket.")
    move({"left": pack["above"]})
    s.stage("pack", "Lower the milk carton into the basket.")
    move({"left": pack["place"]})
    s.stage("release", "Let go of the milk carton.")
    s.move(grips={"left": tt.grip_open(w)}, seconds=1.0)
    s.stage("rise", "Raise the left hand out of the basket.")
    move({"left": pack["rise"]}, 1.5)
    # The named side's hand carries the basket by its handle, its approach
    # turned 90 degrees; the other arm goes back to the ready point.
    carry = kt.carry_poses(base(s, "basket")[:2], spec["target"], side)
    turned = {side: carry["rotation"]}

    def turn_move(targets, seconds=2.0, **kw):
        s.move(
            targets,
            seconds=seconds,
            linear=True,
            rotations=turned,
            speed=tt.SPEED,
            **kw,
        )

    high = carry["pregrasp"] + [0, 0, kt.TURN_HIGH]
    s.stage("to_handle", f"Bring the {side} hand over the basket's handle.")
    move(
        {side: high, other: tt.READY[other]}, 3.0, grips={side: tt.grip_open(handle_w)}
    )
    s.stage("turn", "Turn the hand to take the handle from the side.")
    s.move({side: high}, seconds=2.0, rotations=turned, speed=tt.SPEED)
    s.stage("to_handle_2", "Lower the turned hand beside the handle.")
    turn_move({side: carry["pregrasp"]}, 2.0)
    s.stage("align_handle", "Bring the open gripper around the handle.")
    turn_move({side: carry["grasp"]}, 1.5)
    s.stage("grip_handle", "Close the hand on the handle.")
    s.move(grips={side: tt.grip_closed(handle_w, kt.HANDLE_SQUEEZE)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, "basket", side)
    s.stage("lift_basket", "Lift the basket by its handle.")
    tt.lift(
        s,
        turn_move,
        {side: carry["lift"]},
        {"basket": kt.BASKET_LIFT * 0.6},
        seconds=2.0,
    )
    s.stage("carry", f"Carry the basket to the {side}.")
    turn_move({side: carry["above"]}, 3.0)
    s.stage("set_down", "Set the basket down.")
    turn_move({side: carry["place"]}, 2.0)
    s.stage("let_go", "Let go of the handle.")
    s.move(grips={side: tt.grip_open(handle_w)}, seconds=1.0)
    s.stage("back_away", "Back the hand away from the handle and up.")
    turn_move({side: carry["retreat"]}, 1.5)
    clear = carry["retreat"] + [0, 0, kt.CLEAR]
    turn_move({side: clear}, 1.5)
    s.stage("turn_back", "Turn the hand upright again.")
    s.move({side: clear}, seconds=2.0, rotations=upright, speed=tt.SPEED)
    # The same hand takes the milk out onto the counter on that side.
    counter = [*kt.counter(side), kt.TABLE_Z + 2 * kt.COUNTER_HALF[2]]
    unpack = tt.pick_place(
        side, base(s, "milk"), kt.GRASP_Z["milk"], w, counter, kt.LIFT
    )
    s.stage("to_milk", "Reach into the basket for the milk carton.")
    move(
        {side: unpack["pregrasp"] + [0, 0, kt.HOVER]},
        2.0,
        grips={side: tt.grip_open(w)},
    )
    move({side: unpack["pregrasp"]}, 1.0)
    s.stage("align_2", "Lower the open gripper around the milk carton.")
    move({side: unpack["grasp"]}, 1.5)
    s.stage("grasp_2", "Close the hand on the milk carton.")
    s.move(grips={side: tt.grip_closed(w)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, "milk", side)
    s.stage("lift_2", "Lift the milk carton out of the basket.")
    tt.lift(s, move, {side: unpack["lift"]}, {"milk": 0.08}, seconds=2.0)
    s.stage("to_counter", "Carry the milk carton over the counter.")
    move({side: unpack["above"]})
    s.stage("unpack", "Set the milk carton on the counter.")
    move({side: unpack["place"]})
    s.stage("release_2", "Let go of the milk carton.")
    s.move(grips={side: tt.grip_open(w)}, seconds=1.0)
    s.stage("back_off", "Back the hand away.")
    move({side: unpack["retreat"]}, 1.5)
    s.stage(
        "finish",
        f"Return both arms; the basket stands on the {side}, the milk on the counter.",
    )
    move(tt.READY, 3.0)
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None):
    """One expert episode for seed into output (must not exist)."""
    spec = kt.sample_episode(seed, kin)
    return tt.run_episode(
        BasketScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {
            "side": s.spec["side"],
            "basket_offset_mm": 1000
            * float(np.linalg.norm(s.xyz("basket")[:2] - s.spec["target"])),
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
