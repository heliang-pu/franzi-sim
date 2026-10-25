"""RoboSyn drawer episode: the scene and a scripted IK expert.

DrawerScene is sim.tabletop.ObjectScene with task.py's objects: the episode
framework's audits and recordings, plus object tracking, the task's milestones
(drawer open, object on the hand-over pad, object in the drawer, drawer shut
on it, button pressed after that) and the success test. The expert reads
object and drawer poses from the simulator (privileged): no welds, no object
pose writes, no contact-mask changes, no robot qpos/qvel writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.robosyn_drawer import task as kt
from sim import tabletop as tt


class DrawerScene(tt.ObjectScene):
    OBJECTS = kt.ITEMS
    MILESTONES = (
        "drawer_open",
        "item_on_pad",
        "item_in_drawer",
        "drawer_closed",
        "button_pressed",
    )

    def __init__(self, output, spec, record=False):
        self.HELD = (spec["item"],)
        if spec["variant"] == "retrieve":
            self.MILESTONES = (
                "drawer_open",
                "item_on_pad",
                "item_home",
                "drawer_closed",
                "button_pressed",
            )
        self.deepest_press = 0.0
        super().__init__(output, spec, record)
        self.drawer_body = self.m.body("drawer_part").id
        self.drawer_qadr = int(self.m.joint("drawer_part_joint").qposadr[0])
        self.cabinet_body = self.m.body("drawer_cabinet_part").id
        self.button_q = int(self.m.joint("button_part_joint").qposadr[0])
        self.button_body = self.m.body("button_base_part").id

    def build_root(self):
        return kt.build(self.spec)

    def drawer_q(self):
        return float(self.d.qpos[self.drawer_qadr])

    def button_depth(self):
        return -float(self.d.qpos[self.button_q])

    def in_drawer(self, name, pairs):
        """Inside the drawer's cavity, on its floor, still."""
        R = self.d.xmat[self.drawer_body].reshape(3, 3)
        x, y, z = R.T @ (self.xyz(name) - self.d.xpos[self.drawer_body])
        dims = kt.DIMS["drawer"]
        return (
            abs(x) < 0.08 - 0.02
            and abs(y) < dims["cavity_half_y"] - 0.02
            and abs(z - dims["floor_z"]) < 0.006
            and self.settled(name, pairs)
        )

    def on_pad(self, name, pairs):
        item = self.xyz(name)
        return (
            np.linalg.norm(item[:2] - self.spec["pad"]["xy"]) < 0.03
            and abs(item[2] - kt.PAD_TOP) < 0.004
            and self.settled(name, pairs)
        )

    def at_home(self, name, pairs):
        item = self.xyz(name)
        return (
            np.linalg.norm(item[:2] - self.spec["items"][name]["home"]) < 0.02
            and abs(item[2] - kt.TABLE_Z) < 0.004
            and self.settled(name, pairs)
        )

    def conditions(self, pairs):
        item = self.spec["item"]
        if self.spec["variant"] == "retrieve":
            now = {
                "drawer_open": self.drawer_q() >= kt.PULL - 0.01,
                "item_on_pad": self.on_pad(item, pairs),
                "item_home": self.at_home(item, pairs),
            }
            now["drawer_closed"] = self.drawer_q() <= 0.005 and now["item_home"]
            now["button_pressed"] = (
                self.button_depth() >= kt.PRESSED and now["drawer_closed"]
            )
            return now
        now = {
            "drawer_open": self.drawer_q() >= kt.PULL - 0.01,
            "item_on_pad": self.on_pad(item, pairs),
            "item_in_drawer": self.in_drawer(item, pairs),
        }
        now["drawer_closed"] = self.drawer_q() <= 0.005 and now["item_in_drawer"]
        now["button_pressed"] = (
            self.button_depth() >= kt.PRESSED and now["drawer_closed"]
        )
        return now

    def sample(self):
        record = super().sample()
        record["drawer_q_m"] = self.drawer_q()
        record["button_depth_mm"] = 1000 * self.button_depth()
        return record

    def track_objects(self):
        super().track_objects()
        self.deepest_press = max(self.deepest_press, self.button_depth())

    def complete(self):
        """The final state: the object in the shut drawer, the button pressed
        after shutting and back up, everything still and clear of the robot."""
        now = self.status()
        return (
            now["drawer_closed"]
            and self.reached["button_pressed"]
            and self.button_depth() < 0.002
        )


def base(s, name):
    """An object's measured base (its body origin, x, y, z)."""
    return [float(v) for v in s.xyz(name)]


def check_knob(s):
    pairs = s.contact_bodies()
    others = set(pairs[(pairs == s.drawer_body).any(1)].ravel().tolist())
    if not set(s.jaws["left"]) <= others:
        raise RuntimeError("left: no bilateral grasp on the drawer knob")


def expert(s):
    if s.spec["variant"] == "retrieve":
        return expert_retrieve(s)
    spec = s.spec
    item, w = spec["item"], kt.width(spec["item"])
    noun = kt.NAMES[item]
    knob_w = kt.DIMS["drawer"]["knob_width"]
    cabinet = [float(v) for v in s.d.xpos[s.cabinet_body][:2]]
    move = tt.mover(s)
    s.stage("ready", "Raise both open grippers.")
    tt.go_ready(s, {"left": tt.grip_open(knob_w), "right": tt.grip_open(w)})
    # The left hand opens the drawer while the right carries the object over.
    opening = kt.drawer_moves(cabinet, s.drawer_q(), kt.PULL)
    to_pad = tt.pick_place(
        "right", base(s, item), kt.PICK_Z, w, kt.on_pad(spec["pad"]["xy"]), kt.LIFT
    )
    s.stage(
        "approach",
        f"Reach for the drawer's knob with the left hand, the {noun} with the right.",
    )
    move({"left": opening["pregrasp"], "right": to_pad["pregrasp"]})
    s.stage("align", f"Bring the open grippers around the knob and the {noun}.")
    move({"left": opening["grasp"], "right": to_pad["grasp"]}, 1.5)
    s.stage("grasp", f"Close the left hand on the knob and the right on the {noun}.")
    s.move(
        grips={
            "left": tt.grip_closed(knob_w, kt.KNOB_SQUEEZE),
            "right": tt.grip_closed(w, kt.SQUEEZE),
        },
        seconds=1.0,
    )
    s.move(seconds=0.5)
    check_knob(s)
    tt.check_grasp(s, item, "right")
    s.stage("open", f"Pull the drawer open; lift the {noun}.")
    start = float(s.xyz(item)[2])
    move({"left": opening["slid"], "right": to_pad["lift"]}, 2.5)
    if s.drawer_q() < kt.PULL - 0.01:
        raise RuntimeError(f"the drawer opened only {s.drawer_q():.3f} m")
    if s.xyz(item)[2] < start + kt.LIFT * 0.6:
        raise RuntimeError(f"lift failed for the {item}")
    s.stage(
        "hand_over", f"Let go of the knob; carry the {noun} over the hand-over pad."
    )
    move({"right": to_pad["above"]}, grips={"left": tt.grip_open(knob_w)})
    s.stage("set_down", f"Back the left hand off; set the {noun} on the pad.")
    move({"left": opening["retreat"], "right": to_pad["place"]})
    s.stage("release", f"Let go of the {noun} on the pad.")
    s.move(grips={"right": tt.grip_open(w)}, seconds=1.0)
    s.stage("clear", "Back the right hand away from the pad.")
    move({"right": to_pad["retreat"]}, 1.5)
    # The left hand takes it from the pad into the drawer; the right goes to
    # the button.
    store = tt.pick_place(
        "left",
        base(s, item),
        kt.STORE_Z,
        w,
        kt.in_drawer(cabinet, s.drawer_q()),
        kt.LIFT,
        kt.CARRY,
    )
    press = tt.press_poses(kt.button_top(s.d.xpos[s.button_body]), "right")
    s.stage("take", f"Reach for the {noun} on the pad with the left hand.")
    move(
        {"left": store["pregrasp"], "right": press["above"]},
        grips={"left": tt.grip_open(w), "right": 0.95},
    )
    s.stage("align_2", f"Lower the open left gripper around the {noun}.")
    move({"left": store["grasp"]}, 1.5)
    s.stage("grasp_2", f"Close the left hand on the {noun}.")
    s.move(grips={"left": tt.grip_closed(w, kt.SQUEEZE)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, item, "left")
    s.stage("lift_2", f"Lift the {noun} off the pad.")
    tt.lift(s, move, {"left": store["lift"]}, {item: kt.LIFT * 0.6})
    s.stage("carry", f"Carry the {noun} over the open drawer.")
    move({"left": store["above"]})
    s.stage("store", f"Lower the {noun} into the drawer.")
    move({"left": store["place"]})
    s.stage("release_2", f"Let go of the {noun} in the drawer.")
    s.move(grips={"left": tt.grip_open(w)}, seconds=1.0)
    s.stage("leave_drawer", "Raise the left hand out of the drawer.")
    move({"left": store["rise"]}, 1.5)
    # The left hand pushes the drawer shut.
    closing = kt.drawer_moves(cabinet, s.drawer_q(), 0.0)
    s.stage("to_knob", "Reach for the drawer's knob again.")
    move({"left": closing["pregrasp"]}, grips={"left": tt.grip_open(knob_w)})
    s.stage("align_3", "Bring the open left gripper around the knob.")
    move({"left": closing["grasp"]}, 1.5)
    s.stage("grasp_3", "Close the left hand on the knob.")
    s.move(grips={"left": tt.grip_closed(knob_w, kt.KNOB_SQUEEZE)}, seconds=1.0)
    s.move(seconds=0.5)
    check_knob(s)
    s.stage("close", "Push the drawer shut.")
    move({"left": closing["slid"]}, 2.5)
    if s.drawer_q() > 0.006:
        raise RuntimeError(f"the drawer stayed {s.drawer_q():.3f} m open")
    s.stage("let_go", "Let go of the knob and back away.")
    s.move(grips={"left": tt.grip_open(knob_w)}, seconds=1.0)
    move({"left": closing["retreat"]}, 1.5)
    s.stage("press", "Press the button with the right hand.")
    tt.press(s, move, "right", press, s.button_depth, kt.PRESS, kt.PRESSED)
    s.stage("release_button", "Let the button spring back up.")
    move({"right": press["above"]}, 1.0)
    s.stage("finish", f"Return both arms; the {noun} stays in the shut drawer.")
    move(tt.READY, 3.0, grips={"left": tt.grip_open(w), "right": tt.grip_open(w)})
    s.move(seconds=1.0)


def expert_retrieve(s):
    """The left hand opens the drawer and sets the beaker on the pad; the right
    hand stands it in its slot while the left shuts the drawer; the right hand
    presses the button."""
    spec = s.spec
    item, w = spec["item"], kt.width(spec["item"])
    noun = kt.NAMES[item]
    knob_w = kt.DIMS["drawer"]["knob_width"]
    cabinet = [float(v) for v in s.d.xpos[s.cabinet_body][:2]]
    move = tt.mover(s)
    s.stage("ready", "Raise both open grippers.")
    tt.go_ready(s, {"left": tt.grip_open(knob_w), "right": tt.grip_open(w)})
    opening = kt.drawer_moves(cabinet, s.drawer_q(), kt.PULL)
    s.stage("approach", "Reach for the drawer's knob with the left hand.")
    move({"left": opening["pregrasp"]})
    s.stage("align", "Bring the open left gripper around the knob.")
    move({"left": opening["grasp"]}, 1.5)
    s.stage("grasp", "Close the left hand on the knob.")
    s.move(grips={"left": tt.grip_closed(knob_w, kt.KNOB_SQUEEZE)}, seconds=1.0)
    s.move(seconds=0.5)
    check_knob(s)
    s.stage("open", "Pull the drawer open.")
    move({"left": opening["slid"]}, 2.5)
    if s.drawer_q() < kt.PULL - 0.01:
        raise RuntimeError(f"the drawer opened only {s.drawer_q():.3f} m")
    s.stage("let_go", "Let go of the knob and back away.")
    s.move(grips={"left": tt.grip_open(w)}, seconds=1.0)
    move({"left": opening["retreat"]}, 1.5)
    take = tt.pick_place(
        "left", base(s, item), kt.STORE_Z, w, kt.on_pad(spec["pad"]["xy"]), kt.CARRY
    )
    s.stage("reach_in", f"Reach into the drawer for the {noun}.")
    move({"left": take["pregrasp"]})
    s.stage("align_2", f"Lower the open left gripper around the {noun}.")
    move({"left": take["grasp"]}, 1.5)
    s.stage("grasp_2", f"Close the left hand on the {noun}.")
    s.move(grips={"left": tt.grip_closed(w, kt.SQUEEZE)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, item, "left")
    s.stage("lift_2", f"Lift the {noun} out of the drawer.")
    tt.lift(s, move, {"left": take["lift"]}, {item: 0.08})
    s.stage("hand_over", f"Carry the {noun} over the hand-over pad.")
    move({"left": take["above"]})
    s.stage("set_down", f"Set the {noun} on the pad.")
    move({"left": take["place"]})
    s.stage("release", f"Let go of the {noun} on the pad.")
    s.move(grips={"left": tt.grip_open(w)}, seconds=1.0)
    s.stage("clear", "Back the left hand away from the pad.")
    move({"left": take["retreat"]}, 1.5)
    home = [*spec["items"][item]["home"], kt.TABLE_Z]
    back = tt.pick_place("right", base(s, item), kt.PICK_Z, w, home, kt.LIFT)
    closing = kt.drawer_moves(cabinet, s.drawer_q(), 0.0)
    s.stage(
        "take",
        f"Reach for the {noun} on the pad with the right hand and for the knob with the left.",
    )
    move(
        {"left": closing["pregrasp"], "right": back["pregrasp"]},
        grips={"left": tt.grip_open(knob_w)},
    )
    s.stage("align_3", f"Bring the open grippers around the knob and the {noun}.")
    move({"left": closing["grasp"], "right": back["grasp"]}, 1.5)
    s.stage("grasp_3", f"Close the left hand on the knob and the right on the {noun}.")
    s.move(
        grips={
            "left": tt.grip_closed(knob_w, kt.KNOB_SQUEEZE),
            "right": tt.grip_closed(w, kt.SQUEEZE),
        },
        seconds=1.0,
    )
    s.move(seconds=0.5)
    check_knob(s)
    tt.check_grasp(s, item, "right")
    s.stage("close", f"Push the drawer shut; lift the {noun} off the pad.")
    start = float(s.xyz(item)[2])
    move({"left": closing["slid"], "right": back["lift"]}, 2.5)
    if s.drawer_q() > 0.006:
        raise RuntimeError(f"the drawer stayed {s.drawer_q():.3f} m open")
    if s.xyz(item)[2] < start + kt.LIFT * 0.6:
        raise RuntimeError(f"lift failed for the {item}")
    s.stage("return", f"Let go of the knob; carry the {noun} back over its spot.")
    move({"right": back["above"]}, grips={"left": tt.grip_open(knob_w)})
    s.stage("stand", f"Back the left hand off; stand the {noun} in its spot.")
    move({"left": closing["retreat"], "right": back["place"]})
    s.stage("release_3", f"Let go of the {noun}.")
    s.move(grips={"right": tt.grip_open(w)}, seconds=1.0)
    s.stage("clear_2", "Back the right hand away.")
    move({"right": back["retreat"]}, 1.5)
    press = tt.press_poses(kt.button_top(s.d.xpos[s.button_body]), "right")
    s.stage("to_button", "Bring the closed right hand over the button.")
    move({"right": press["above"]}, 2.0, grips={"right": 0.95})
    s.stage("press", "Press the button with the right hand.")
    tt.press(s, move, "right", press, s.button_depth, kt.PRESS, kt.PRESSED)
    s.stage("release_button", "Let the button spring back up.")
    move({"right": press["above"]}, 1.0)
    s.stage(
        "finish",
        f"Return both arms; the {noun} stands in its spot and the drawer is shut.",
    )
    move(tt.READY, 3.0, grips={"left": tt.grip_open(w), "right": tt.grip_open(w)})
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None, variant="store"):
    """One expert episode for seed into output (must not exist)."""
    spec = kt.sample_episode(seed, kin, variant=variant)
    return tt.run_episode(
        DrawerScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {
            "item": s.spec["item"],
            "deepest_press_mm": 1000 * s.deepest_press,
            "final_drawer_q_m": s.drawer_q(),
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record", action="store_true")
    p.add_argument("--variant", choices=kt.VARIANTS, default="store")
    args = p.parse_args()
    result = run(args.output, args.seed, args.record, variant=args.variant)
    print(
        json.dumps({k: result[k] for k in ("success", "error", "progress")}), flush=True
    )


if __name__ == "__main__":
    main()
