"""Lab bench episode: the scene and a scripted IK expert.

LabScene is sim.tabletop.ObjectScene with task.py's objects and the stirrer's
own logic: a knob press (3 mm in, released below 1 mm) toggles it, it only
starts with the beaker on its plate, and its indicator turns green while it
runs. The milestones are the beaker on the stirrer, the stirrer started, the
stirrer stopped after its run, and the beaker on the tray. The expert reads
object and knob states from the simulator (privileged): no welds, no object
pose writes, no contact-mask changes, no robot qpos/qvel writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.robosyn_lab import task as kt
from sim import tabletop as tt


class LabScene(tt.ObjectScene):
    OBJECTS = ("beaker",)
    HELD = ("beaker",)
    MILESTONES = ("beaker_on_stirrer", "stirrer_on", "stirrer_off", "beaker_on_tray")

    def __init__(self, output, spec, record=False):
        self.running, self.knob_down = False, False
        self.started_at, self.ran_s, self.presses = None, 0.0, 0
        super().__init__(output, spec, record)
        side = spec["side"]
        self.knob_q = int(self.m.joint(f"mixer_knob_{side}_part_joint").qposadr[0])
        self.led = self.m.geom("mixer_led").id
        self.mixer = self.m.body("mixer_part").id

    def build_root(self):
        return kt.build(self.spec)

    def knob_depth(self):
        return float(self.d.qpos[self.knob_q])

    def on_plate(self, pairs):
        beaker = self.xyz("beaker")
        target = kt.plate(self.d.xpos[self.mixer][:2])
        return (
            np.linalg.norm(beaker[:2] - target[:2]) < 0.025
            and abs(beaker[2] - target[2]) < 0.004
            and self.settled("beaker", pairs)
        )

    def on_tray(self, pairs):
        beaker = self.xyz("beaker")
        target = kt.on_tray(self.spec["tray"]["xy"])
        return (
            np.linalg.norm(beaker[:2] - target[:2]) < 0.03
            and abs(beaker[2] - target[2]) < 0.004
            and self.settled("beaker", pairs)
        )

    def track_objects(self):
        super().track_objects()
        depth = self.knob_depth()
        if not self.knob_down and depth >= kt.PRESSED:
            self.knob_down = True
            self.presses += 1
            if self.running:
                self.running = False
                self.ran_s = float(self.d.time) - self.started_at
            elif self.on_plate(self.contact_bodies()):
                self.running, self.started_at = True, float(self.d.time)
            rgba = kt.LED_ON if self.running else kt.LED_OFF
            self.m.geom_rgba[self.led] = [float(v) for v in rgba.split()]
        elif self.knob_down and depth <= kt.RELEASED:
            self.knob_down = False

    def conditions(self, pairs):
        return {
            "beaker_on_stirrer": self.on_plate(pairs),
            "stirrer_on": self.running,
            "stirrer_off": not self.running and self.ran_s >= kt.RUN - 0.5,
            "beaker_on_tray": self.on_tray(pairs) and self.ran_s >= kt.RUN - 0.5,
        }

    def sample(self):
        record = super().sample()
        record["knob_depth_mm"] = 1000 * self.knob_depth()
        record["stirring"] = self.running
        return record

    def complete(self):
        now = self.status()
        return now["beaker_on_tray"] and not self.running and not self.knob_down


def base(s, name):
    return [float(v) for v in s.xyz(name)]


def press_knob(s, move, side, poses):
    s.stage("press", f"Press the {side} knob.")
    tt.press(s, move, side, poses, s.knob_depth, kt.PRESS, kt.PRESSED)
    s.stage("release_knob", "Let the knob spring back.")
    move({side: poses["above"]}, 1.0)


def expert(s):
    spec = s.spec
    side = spec["side"]
    w = kt.width()
    move = tt.mover(s)
    mixer = [float(v) for v in s.d.xpos[s.mixer][:2]]
    poses = tt.press_poses(kt.knob_top(mixer, side), side)
    s.stage("ready", "Raise both open grippers.")
    tt.go_ready(s, {"left": tt.grip_open(w), "right": tt.grip_open(w)})
    load = tt.pick_place(
        "right", base(s, "beaker"), kt.GRASP_Z, w, kt.plate(mixer), kt.LIFT
    )
    s.stage("approach", "Reach for the beaker with the right hand.")
    move({"right": load["pregrasp"]})
    s.stage("align", "Lower the open right gripper around the beaker.")
    move({"right": load["grasp"]}, 1.5)
    s.stage("grasp", "Close the right hand on the beaker.")
    s.move(grips={"right": tt.grip_closed(w, kt.SQUEEZE)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, "beaker", "right")
    s.stage("lift", "Lift the beaker.")
    tt.lift(s, move, {"right": load["lift"]}, {"beaker": 0.06})
    s.stage("to_stirrer", "Carry the beaker over the stirrer.")
    move({"right": load["above"]})
    s.stage("place", "Set the beaker on the stirrer's plate.")
    move({"right": load["place"]})
    s.stage("release", "Let go of the beaker.")
    s.move(grips={"right": tt.grip_open(w)}, seconds=1.0)
    s.stage("back_off", "Back the right hand away.")
    move({"right": load["retreat"]}, 1.5)
    s.stage("to_knob", f"Bring the closed {side} hand over the {side} knob.")
    # For the left knob the right hand first goes back to the ready point:
    # backed off only, its wrist met the left hand's over the knob (seed 1).
    away = {"right": tt.READY["right"]} if side == "left" else {}
    move({side: poses["above"], **away}, 2.5, grips={side: 0.95})
    press_knob(s, move, side, poses)
    if not s.running:
        raise RuntimeError("the stirrer did not start")
    s.stage("stir", "Let the stirrer run.")
    s.move(seconds=kt.RUN)
    press_knob(s, move, side, poses)
    if s.running:
        raise RuntimeError("the stirrer did not stop")
    unload = tt.pick_place(
        "right",
        base(s, "beaker"),
        kt.GRASP_Z,
        w,
        kt.on_tray(spec["tray"]["xy"]),
        kt.LIFT,
    )
    s.stage("approach_2", "Reach for the beaker on the stirrer.")
    move(
        {"right": unload["pregrasp"], "left": tt.READY["left"]},
        grips={"right": tt.grip_open(w)},
    )
    s.stage("align_2", "Lower the open right gripper around the beaker.")
    move({"right": unload["grasp"]}, 1.5)
    s.stage("grasp_2", "Close the right hand on the beaker.")
    s.move(grips={"right": tt.grip_closed(w, kt.SQUEEZE)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, "beaker", "right")
    s.stage("lift_2", "Lift the beaker off the stirrer.")
    tt.lift(s, move, {"right": unload["lift"]}, {"beaker": 0.06})
    s.stage("to_tray", "Carry the beaker over the tray.")
    move({"right": unload["above"]})
    s.stage("place_2", "Set the beaker on the tray.")
    move({"right": unload["place"]})
    s.stage("release_2", "Let go of the beaker.")
    s.move(grips={"right": tt.grip_open(w)}, seconds=1.0)
    s.stage("back_off_2", "Back the right hand away.")
    move({"right": unload["retreat"]}, 1.5)
    s.stage("finish", "Return both arms; the stirred beaker stands on the tray.")
    move(tt.READY, 3.0, grips={"left": tt.grip_open(w), "right": tt.grip_open(w)})
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None):
    """One expert episode for seed into output (must not exist)."""
    spec = kt.sample_episode(seed, kin)
    return tt.run_episode(
        LabScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {
            "side": s.spec["side"],
            "ran_s": s.ran_s,
            "presses": s.presses,
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
