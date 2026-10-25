"""Tote transfer episode: the scene and a scripted IK expert.

ToteScene is sim.tabletop.ObjectScene with task.py's two loose parts: the
episode framework's audits and recordings, plus object tracking, the
milestones (the named part lifted out, then lying in the grey tote) and the
success test, which also needs the other part still in the blue tote. The
expert reads the part's pose from the simulator (privileged): no welds, no
object pose writes, no contact-mask changes, no robot qpos/qvel writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenes.tote_transfer import task as kt
from sim import tabletop as tt


class ToteScene(tt.ObjectScene):
    OBJECTS = tuple(kt.PARTS)
    MILESTONES = ("lifted_out", "in_grey_tote")

    def __init__(self, output, spec, record=False):
        self.HELD = (spec["target"],)
        super().__init__(output, spec, record)

    def build_root(self):
        return kt.build(self.spec)

    def in_tote(self, name, tote, pairs):
        """Lying flat on the tote's floor, inside its walls, still."""
        x, y, z = self.xyz(name)
        return (
            abs(x - tote[0]) < kt.INNER[0]
            and abs(y - tote[1]) < kt.INNER[1]
            and abs(z - kt.FLOOR_Z) < 0.003
            and self.settled(name, pairs)
        )

    def conditions(self, pairs):
        target = self.spec["target"]
        held = self.touching(target, pairs).issuperset(self.jaws["left"])
        return {
            "lifted_out": held and self.xyz(target)[2] > kt.RIM_Z,
            "in_grey_tote": self.in_tote(target, kt.DEST, pairs),
        }

    def complete(self):
        pairs = self.contact_bodies()
        other = next(n for n in kt.PARTS if n != self.spec["target"])
        return self.status(pairs)["in_grey_tote"] and self.in_tote(
            other, kt.SOURCE, pairs
        )


def expert(s):
    spec = s.spec
    target = spec["target"]
    part = kt.PARTS[target][3]
    width = kt.PARTS[target][2]
    base = [float(v) for v in s.xyz(target)]
    poses = kt.key_poses(base, spec["parts"][target]["yaw"], width, spec["drop"])
    rotation = {"left": poses["rotation"]}

    def move(key, seconds=2.0, **kw):
        s.move(
            {"left": poses[key]},
            seconds=seconds,
            linear=True,
            rotations=rotation,
            speed=tt.SPEED,
            **kw,
        )

    s.stage("ready", "Raise the open left gripper.")
    s.move({"left": kt.READY}, grips={"left": tt.grip_open(width)}, seconds=3.0)
    s.stage("approach", f"Move the left hand over the {part} in the blue tote.")
    move("above", 2.5)
    s.stage("align", f"Lower the open gripper around the {part}.")
    move("grasp", 2.0)
    s.stage("grasp", f"Close the gripper on the {part}.")
    s.move(grips={"left": tt.grip_closed(width)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, target, "left")
    s.stage("lift", f"Lift the {part} out of the blue tote.")
    start = float(s.xyz(target)[2])
    move("lift", 2.0)
    if s.xyz(target)[2] < start + 0.1:
        raise RuntimeError(f"lift failed for the {target}")
    s.stage("transfer", "Carry it over the grey tote.")
    move("over", 2.5)
    s.stage("lower", "Lower it into the grey tote.")
    move("release", 2.0)
    s.stage("release", f"Let go of the {part}.")
    s.move(grips={"left": tt.grip_open(width)}, seconds=1.0)
    s.move(seconds=0.5)
    s.stage("retreat", "Raise the left hand out of the tote.")
    move("clear", 1.5)
    s.stage("finish", "Return the left arm; the part lies in the grey tote.")
    s.move({"left": kt.READY}, seconds=2.5, linear=True)
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None):
    """One expert episode for seed into output (must not exist)."""
    spec = kt.sample_episode(seed, kin)
    return tt.run_episode(
        ToteScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {"target": s.spec["target"]},
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
