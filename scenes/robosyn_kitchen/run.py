"""RoboSyn kitchen episode: the scene and a scripted IK expert.

KitchenScene is sim.tabletop.ObjectScene with task.py's objects: the episode
framework's audits and recordings, plus object tracking, the task's milestones
(lid off, ingredient in the pot, the other on the plate, lid back on, button
pressed after that) and the success test. The expert reads object poses from
the simulator (privileged): no welds, no object pose writes, no contact-mask
changes, no robot qpos/qvel writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.robosyn_kitchen import task as kt
from sim import tabletop as tt


class KitchenScene(tt.ObjectScene):
    OBJECTS = ("pot", "lid", "plate", *kt.FOODS)
    HELD = ("lid", *kt.FOODS)
    MILESTONES = ("lid_off", "food_in_pot", "food_on_plate", "lid_on", "button_pressed")

    def __init__(self, output, spec, record=False):
        self.deepest_press = 0.0
        super().__init__(output, spec, record)
        self.button_q = int(self.m.joint("button_part_joint").qposadr[0])
        self.button_body = self.m.body("button_base_part").id

    def build_root(self):
        return kt.build(self.spec)

    def button_depth(self):
        return -float(self.d.qpos[self.button_q])

    def lid_on(self, pairs):
        lid, pot = self.xyz("lid"), self.xyz("pot")
        return (
            np.linalg.norm(lid[:2] - pot[:2]) < 0.012
            and abs(lid[2] - (pot[2] + kt.DIMS["pot"]["rim_z"])) < 0.004
            and self.settled("lid", pairs)
        )

    def lid_off(self, pairs):
        lid = self.xyz("lid")
        return (
            np.linalg.norm(lid[:2] - self.spec["rest"]["xy"]) < 0.03
            and abs(lid[2] - kt.REST_TOP) < 0.004
            and self.settled("lid", pairs)
        )

    def in_pot(self, name, pairs):
        food, pot = self.xyz(name), self.xyz("pot")
        return (
            np.linalg.norm(food[:2] - pot[:2]) < kt.DIMS["pot"]["inner_radius"] - 0.02
            and food[2] < pot[2] + kt.DIMS["pot"]["floor_z"] + 0.012
            and self.settled(name, pairs)
        )

    def on_plate(self, name, pairs):
        food, plate = self.xyz(name), self.xyz("plate")
        return (
            np.linalg.norm(food[:2] - plate[:2]) < 0.045
            and food[2] < plate[2] + kt.DIMS["plate"]["floor_z"] + 0.012
            and self.settled(name, pairs)
        )

    def conditions(self, pairs):
        now = {
            "lid_off": self.lid_off(pairs),
            "food_in_pot": self.in_pot(self.spec["into_pot"], pairs),
            "food_on_plate": self.on_plate(self.spec["onto_plate"], pairs),
        }
        # The lid counts as on once it covers the ingredient (it starts on the
        # empty pot), the press once the covered pot holds it.
        now["lid_on"] = self.lid_on(pairs) and now["food_in_pot"]
        now["button_pressed"] = self.button_depth() >= kt.PRESSED and now["lid_on"]
        return now

    def sample(self):
        record = super().sample()
        record["button_depth_mm"] = 1000 * self.button_depth()
        return record

    def track_objects(self):
        super().track_objects()
        self.deepest_press = max(self.deepest_press, self.button_depth())

    def complete(self):
        """The final state: the pot covered with its ingredient inside, the
        other on the plate, the button pressed after covering and back up,
        everything still and clear of the robot."""
        now = self.status()
        return (
            now["lid_on"]
            and now["food_on_plate"]
            and self.reached["button_pressed"]
            and self.button_depth() < 0.002
        )


def base(s, name):
    """An object's measured base (its body origin, x, y, z)."""
    return [float(v) for v in s.xyz(name)]


def expert(s):
    if s.spec["variant"] == "swap":
        return expert_swap(s)
    spec = s.spec
    first, second = spec["into_pot"], spec["onto_plate"]
    knob, w1, w2 = kt.knob_width(), kt.food_width(first), kt.food_width(second)
    move = tt.mover(s)
    s.stage("ready", "Raise both open grippers.")
    tt.go_ready(s, {"left": tt.grip_open(knob), "right": tt.grip_open(w1)})
    # Lid off with the left hand while the right picks up the pot's ingredient.
    rest = kt.lid_on_rest(spec["rest"]["xy"])
    lid = tt.pick_place("left", base(s, "lid"), kt.KNOB_Z, knob, rest, kt.LID_LIFT)
    food = tt.pick_place(
        "right", base(s, first), kt.GRASP_Z, w1, kt.in_pot(base(s, "pot")), kt.LIFT
    )
    s.stage(
        "approach",
        f"Reach for the lid's knob with the left hand, the {first} with the right.",
    )
    move({"left": lid["pregrasp"], "right": food["pregrasp"]})
    s.stage("align", f"Lower the open grippers around the knob and the {first}.")
    move({"left": lid["grasp"], "right": food["grasp"]}, 1.5)
    s.stage("grasp", f"Close the left hand on the knob and the right on the {first}.")
    s.move(
        grips={
            "left": tt.grip_closed(knob, kt.LID_SQUEEZE),
            "right": tt.grip_closed(w1),
        },
        seconds=1.0,
    )
    s.move(seconds=0.5)
    tt.check_grasp(s, "lid", "left")
    tt.check_grasp(s, first, "right")
    s.stage("lift", f"Lift the lid off the pot and the {first} off the table.")
    tt.lift(
        s,
        move,
        {"left": lid["lift"], "right": food["lift"]},
        {"lid": 0.04, first: 0.06},
    )
    s.stage("uncover", "Carry the lid over its rest beside the pot.")
    move({"left": lid["above"]})
    s.stage("set_lid", f"Set the lid on its rest; bring the {first} over the open pot.")
    move({"left": lid["place"], "right": food["above"]})
    s.stage("release_lid", "Let go of the lid.")
    s.move(grips={"left": tt.grip_open(knob)}, seconds=1.0)
    s.stage("into_pot", f"Back away from the lid; lower the {first} into the pot.")
    move({"left": lid["retreat"], "right": food["place"]})
    s.stage("release_food", f"Let go of the {first} in the pot.")
    s.move(grips={"right": tt.grip_open(w1)}, seconds=1.0)
    s.stage("leave_pot", "Raise the right hand out of the pot.")
    move({"right": food["rise"]}, 1.5)
    # Lid back with the left hand while the right serves the other ingredient.
    lid = tt.pick_place(
        "left",
        base(s, "lid"),
        kt.KNOB_Z,
        knob,
        kt.lid_on_pot(base(s, "pot")),
        kt.LID_LIFT,
    )
    serve = tt.pick_place(
        "right",
        base(s, second),
        kt.GRASP_Z,
        w2,
        kt.on_plate(base(s, "plate")),
        kt.LIFT,
        0.06,
    )
    s.stage("approach_2", f"Reach for the lid's knob again and for the {second}.")
    move(
        {"left": lid["pregrasp"], "right": serve["pregrasp"]},
        grips={"right": tt.grip_open(w2)},
    )
    s.stage("align_2", f"Lower the open grippers around the knob and the {second}.")
    move({"left": lid["grasp"], "right": serve["grasp"]}, 1.5)
    s.stage(
        "grasp_2", f"Close the left hand on the knob and the right on the {second}."
    )
    s.move(
        grips={
            "left": tt.grip_closed(knob, kt.LID_SQUEEZE),
            "right": tt.grip_closed(w2),
        },
        seconds=1.0,
    )
    s.move(seconds=0.5)
    tt.check_grasp(s, "lid", "left")
    tt.check_grasp(s, second, "right")
    s.stage("lift_2", f"Lift the lid and the {second}.")
    tt.lift(
        s,
        move,
        {"left": lid["lift"], "right": serve["lift"]},
        {"lid": 0.04, second: 0.06},
    )
    s.stage("carry", f"Carry the lid over the pot and the {second} over the plate.")
    move({"left": lid["above"], "right": serve["above"]})
    s.stage("cover", f"Put the lid on the pot and the {second} on the plate.")
    move({"left": lid["place"], "right": serve["place"]})
    s.stage("release", f"Let go of the lid and the {second}.")
    s.move(grips={"left": tt.grip_open(knob), "right": tt.grip_open(w2)}, seconds=1.0)
    s.stage("back_off", "Back both hands away.")
    move({"left": lid["retreat"], "right": serve["retreat"]}, 1.5)
    # The button, with the left hand's closed fingertips.
    poses = tt.press_poses(kt.button_top(s.d.xpos[s.button_body]), "left")
    s.stage(
        "to_button", "Bring the closed left hand over the button; return the right arm."
    )
    move(
        {"left": poses["above"], "right": tt.READY["right"]}, 2.5, grips={"left": 0.95}
    )
    s.stage("press", "Press the button.")
    tt.press(s, move, "left", poses, s.button_depth, kt.PRESS, kt.PRESSED)
    s.stage("release_button", "Let the button spring back up.")
    move({"left": poses["above"]}, 1.0)
    s.stage(
        "finish",
        f"Return both arms; the {first} stays in the covered pot and the {second} "
        "on the plate.",
    )
    move(tt.READY, 3.0, grips={"left": tt.grip_open(knob), "right": tt.grip_open(w2)})
    s.move(seconds=1.0)


def cover_and_press(s, move, knob, closing_grip):
    """The left hand puts the lid back on the pot from its rest and presses
    the button; the right arm returns to the ready point meanwhile."""
    lid = tt.pick_place(
        "left",
        base(s, "lid"),
        kt.KNOB_Z,
        knob,
        kt.lid_on_pot(base(s, "pot")),
        kt.LID_LIFT,
    )
    s.stage(
        "approach_lid", "Reach for the lid's knob on its rest; return the right arm."
    )
    move(
        {"left": lid["pregrasp"], "right": tt.READY["right"]},
        grips={"left": tt.grip_open(knob)},
    )
    s.stage("align_lid", "Lower the open left gripper around the knob.")
    move({"left": lid["grasp"]}, 1.5)
    s.stage("grasp_lid", "Close the left hand on the knob.")
    s.move(grips={"left": tt.grip_closed(knob, kt.LID_SQUEEZE)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, "lid", "left")
    s.stage("lift_lid", "Lift the lid off its rest.")
    tt.lift(s, move, {"left": lid["lift"]}, {"lid": 0.04})
    s.stage("carry_lid", "Carry the lid over the pot.")
    move({"left": lid["above"]})
    s.stage("cover", "Put the lid on the pot.")
    move({"left": lid["place"]})
    s.stage("release_lid", "Let go of the lid.")
    s.move(grips={"left": tt.grip_open(knob)}, seconds=1.0)
    s.stage("back_off", "Back the left hand away from the lid.")
    move({"left": lid["retreat"]}, 1.5)
    poses = tt.press_poses(kt.button_top(s.d.xpos[s.button_body]), "left")
    s.stage("to_button", "Bring the closed left hand over the button.")
    move({"left": poses["above"]}, 2.5, grips={"left": 0.95})
    s.stage("press", "Press the button.")
    tt.press(s, move, "left", poses, s.button_depth, kt.PRESS, kt.PRESSED)
    s.stage("release_button", "Let the button spring back up.")
    move({"left": poses["above"]}, 1.0)
    s.stage("finish", "Return both arms; the pot stays covered.")
    move(tt.READY, 3.0, grips={"left": tt.grip_open(knob), "right": closing_grip})
    s.move(seconds=1.0)


def expert_swap(s):
    """Lid off (left), the served ingredient out of the pot onto the plate and
    the other one in (right), lid on and the button (left)."""
    spec = s.spec
    out_item, in_item = spec["onto_plate"], spec["into_pot"]
    knob, w_out, w_in = kt.knob_width(), kt.food_width(out_item), kt.food_width(in_item)
    move = tt.mover(s)
    s.stage("ready", "Raise both open grippers.")
    tt.go_ready(s, {"left": tt.grip_open(knob), "right": tt.grip_open(w_out)})
    rest = kt.lid_on_rest(spec["rest"]["xy"])
    lid = tt.pick_place("left", base(s, "lid"), kt.KNOB_Z, knob, rest, kt.LID_LIFT)
    s.stage("approach", "Reach for the lid's knob with the left hand.")
    move({"left": lid["pregrasp"]})
    s.stage("align", "Lower the open left gripper around the knob.")
    move({"left": lid["grasp"]}, 1.5)
    s.stage("grasp", "Close the left hand on the knob.")
    s.move(grips={"left": tt.grip_closed(knob, kt.LID_SQUEEZE)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, "lid", "left")
    s.stage("lift", "Lift the lid off the pot.")
    tt.lift(s, move, {"left": lid["lift"]}, {"lid": 0.04})
    s.stage("uncover", "Carry the lid over its rest beside the pot.")
    move({"left": lid["above"]})
    out = kt.from_pot("right", base(s, out_item), w_out, kt.on_plate(base(s, "plate")))
    s.stage(
        "set_lid",
        f"Set the lid on its rest; bring the right hand over the {out_item} in the pot.",
    )
    move({"left": lid["place"], "right": out["pregrasp"]})
    s.stage("release_lid", "Let go of the lid.")
    s.move(grips={"left": tt.grip_open(knob)}, seconds=1.0)
    s.stage(
        "reach_in",
        f"Back away from the lid; lower the open right gripper around the {out_item}.",
    )
    move({"left": lid["retreat"], "right": out["grasp"]})
    s.stage("grasp_out", f"Close the right hand on the {out_item}.")
    s.move(grips={"right": tt.grip_closed(w_out)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, out_item, "right")
    s.stage("lift_out", f"Lift the {out_item} out of the pot.")
    tt.lift(s, move, {"right": out["lift"]}, {out_item: 0.08})
    s.stage("serve", f"Carry the {out_item} over the plate.")
    move({"right": out["above"]})
    s.stage("place_plate", f"Set the {out_item} on the plate.")
    move({"right": out["place"]})
    s.stage("release_plate", f"Let go of the {out_item}.")
    s.move(grips={"right": tt.grip_open(w_out)}, seconds=1.0)
    s.stage("leave_plate", "Back the right hand away from the plate.")
    move({"right": out["retreat"]}, 1.5)
    food = tt.pick_place(
        "right", base(s, in_item), kt.GRASP_Z, w_in, kt.in_pot(base(s, "pot")), kt.LIFT
    )
    s.stage("approach_in", f"Reach for the {in_item} with the right hand.")
    move({"right": food["pregrasp"]}, grips={"right": tt.grip_open(w_in)})
    s.stage("align_in", f"Lower the open right gripper around the {in_item}.")
    move({"right": food["grasp"]}, 1.5)
    s.stage("grasp_in", f"Close the right hand on the {in_item}.")
    s.move(grips={"right": tt.grip_closed(w_in)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, in_item, "right")
    s.stage("lift_in", f"Lift the {in_item}.")
    tt.lift(s, move, {"right": food["lift"]}, {in_item: 0.06})
    s.stage("to_pot", f"Carry the {in_item} over the pot.")
    move({"right": food["above"]})
    s.stage("into_pot", f"Lower the {in_item} into the pot.")
    move({"right": food["place"]})
    s.stage("release_food", f"Let go of the {in_item} in the pot.")
    s.move(grips={"right": tt.grip_open(w_in)}, seconds=1.0)
    s.stage("leave_pot", "Raise the right hand out of the pot.")
    move({"right": food["rise"]}, 1.5)
    cover_and_press(s, move, knob, tt.grip_open(w_in))


def run(output, seed, record=False, kin=None, variant="cook"):
    """One expert episode for seed into output (must not exist)."""
    spec = kt.sample_episode(seed, kin, variant=variant)
    return tt.run_episode(
        KitchenScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {"deepest_press_mm": 1000 * s.deepest_press},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record", action="store_true")
    p.add_argument("--variant", choices=kt.VARIANTS, default="cook")
    args = p.parse_args()
    result = run(args.output, args.seed, args.record, variant=args.variant)
    print(
        json.dumps({k: result[k] for k in ("success", "error", "progress")}), flush=True
    )


if __name__ == "__main__":
    main()
