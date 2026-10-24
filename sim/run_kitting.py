"""Dual-arm sort-and-kit episode: the kitting scene and a scripted IK expert.

KitScene is run_expert.Scene with kitting_task's objects: the same model
overrides, gravity feedforward, head gaze, per-step penetration and joint-limit
audits and recordings, plus per-part tracking, the colour-matched success test
and episode.json beside scene.xml. The expert follows kitting_task's key poses
round by round and reads part poses from the simulator (privileged): no welds,
no object pose writes, no contact-mask changes, no robot qpos/qvel writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from sim import kitting_task as kt
from sim.build_task import ROOT
from sim.camera_rig import POLICY_CAMERAS
from sim.run_expert import Scene


class KitScene(Scene):
    def __init__(self, output, spec, record=False):
        self.spec = spec
        self.parts = {p["name"]: p for p in spec["parts"]}
        self.cells = {c["name"]: c for c in spec["cells"]}
        self.round = 0
        super().__init__(output, record)
        self.kin = kt.kinematics()
        (output / "episode.json").write_text(json.dumps(spec, indent=2))
        self.part_ids = {n: self.m.body(n + "_part").id for n in self.parts}
        self.part_dofs = {
            n: int(self.m.joint(n + "_part_joint").dofadr[0]) for n in self.parts
        }
        self.jaws = {
            s: [self.m.body(f"{s}finger{i}_Link").id for i in (1, 2)] for s in kt.SIDES
        }
        self.rest_z = {n: float(self.d.xpos[b][2]) for n, b in self.part_ids.items()}
        self.lift_evidence = {
            n: {
                "peak_height_m": 0.0,
                "bilateral_physics_steps": 0,
                "airborne_bilateral_steps": 0,
            }
            for n in self.parts
        }

    def build_root(self):
        return kt.build(self.spec)

    def contact_bodies(self):
        """Body ids of this step's contact pairs, shape (ncon, 2)."""
        return self.m.geom_bodyid[self.d.contact.geom]

    def touching(self, name, pairs):
        bid = self.part_ids[name]
        return set(pairs[(pairs == bid).any(1)].ravel().tolist()) - {bid}

    def object_info(self, name, pairs=None):
        pairs = self.contact_bodies() if pairs is None else pairs
        others = self.touching(name, pairs)
        bid, dof = self.part_ids[name], self.part_dofs[name]
        return {
            "xyz": self.d.xpos[bid].tolist(),
            "quat": self.d.xquat[bid].tolist(),
            "velocity_m_s": float(np.linalg.norm(self.d.qvel[dof : dof + 3])),
            "touching_robot": bool(others & self.robot_bodies),
            "jaw_contacts": {
                s: [i for i, b in enumerate(self.jaws[s], 1) if b in others]
                for s in kt.SIDES
            },
        }

    def sample(self):
        pairs = self.contact_bodies()
        return {
            "t": float(self.d.time),
            "phase": self.phase,
            "subtask": self.instruction,
            "round": self.round,
            "state": self.state().tolist(),
            "action": self.command().tolist(),
            "progress": self.progress(),
            "objects": {n: self.object_info(n, pairs) for n in self.parts},
            "max_penetration_mm": self.maxdepth * 1000,
        }

    def track_objects(self):
        pairs = self.contact_bodies()
        for name, bid in self.part_ids.items():
            e = self.lift_evidence[name]
            z = float(self.d.xpos[bid][2])
            e["peak_height_m"] = max(e["peak_height_m"], z)
            others = self.touching(name, pairs)
            if any(others.issuperset(self.jaws[s]) for s in kt.SIDES):
                e["bilateral_physics_steps"] += 1
                if z > self.rest_z[name] + 0.04:
                    e["airborne_bilateral_steps"] += 1

    def placements(self):
        """Cell each part rests in: a cell of its own colour within CELL_TOL of
        the cell centre, the part upright, seated on the tray, still and clear
        of the robot. None otherwise; a cell holding two parts counts for
        neither."""
        pairs = self.contact_bodies()
        found = {}
        for name, part in self.parts.items():
            bid, dof = self.part_ids[name], self.part_dofs[name]
            x, y, z = self.d.xpos[bid]
            settled = (
                self.d.xmat[bid][8] > np.cos(np.radians(15))
                and abs(z - (kt.KIT_TOP + kt.HEIGHT / 2)) < 0.003
                and np.linalg.norm(self.d.qvel[dof : dof + 3]) < 0.025
                and not self.touching(name, pairs) & self.robot_bodies
            )
            found[name] = next(
                (
                    c["name"]
                    for c in self.cells.values()
                    if settled
                    and c["color"] == part["color"]
                    and abs(x - c["xy"][0]) <= kt.CELL_TOL
                    and abs(y - c["xy"][1]) <= kt.CELL_TOL
                ),
                None,
            )
        taken = [c for c in found.values() if c]
        return {n: c if c and taken.count(c) == 1 else None for n, c in found.items()}

    def progress(self):
        return sum(c is not None for c in self.placements().values())

    def complete(self):
        return self.progress() == len(self.parts)


def measured(s, name):
    """A part's measured xy and grasp azimuth: its yaw folded into half its
    symmetry period either side of 0, so the fingers close on two faces; 0 for
    a cylinder."""
    bid = s.part_ids[name]
    xy = [float(v) for v in s.d.xpos[bid][:2]]
    period = kt.TYPES[s.parts[name]["shape"]]["symmetry"]
    if not period:
        return xy, 0.0
    R = s.d.xmat[bid].reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return xy, float((yaw + period / 2) % period - period / 2)


def expert(s):
    spec = s.spec
    count = len(spec["rounds"])
    upright = {side: kt.grasp_rotation(side) for side in kt.SIDES}
    opened = {side: kt.grip_open(0.04) for side in kt.SIDES}
    s.stage("ready", "Raise both open grippers behind the kit trays.")
    s.move(spec["ready"], grips=opened, seconds=3.0, rotations=upright, speed=kt.SPEED)
    for r, pair in enumerate(spec["rounds"], 1):
        s.round = r
        part = {side: s.parts[pair[side]] for side in kt.SIDES}
        cell = {side: s.cells[spec["assignment"][pair[side]]] for side in kt.SIDES}
        poses = {
            side: kt.key_poses(part[side], cell[side], *measured(s, pair[side]))
            for side in kt.SIDES
        }
        rot = {side: poses[side]["rotation"] for side in kt.SIDES}
        width = {side: kt.TYPES[part[side]["shape"]]["width"] for side in kt.SIDES}
        left, right = kt.describe(part["left"]), kt.describe(part["right"])
        to = {
            key: {side: poses[side][key] for side in kt.SIDES} for key in poses["left"]
        }
        s.stage(
            "approach",
            f"Round {r} of {count}: reach for the {left} on the left "
            f"and the {right} on the right.",
        )
        # Straight in from the last retreat: a joint-space move here swept the
        # hand through a part already in the tray (seed 2, round 3).
        s.move(
            to["pregrasp"],
            grips={side: kt.grip_open(width[side]) for side in kt.SIDES},
            seconds=2.0,
            linear=True,
            rotations=rot,
            speed=kt.SPEED,
        )
        s.stage("align", f"Lower both open grippers around the {left} and the {right}.")
        s.move(to["grasp"], seconds=1.5, linear=True, rotations=rot, speed=kt.SPEED)
        s.stage("grasp", "Close both grippers on the parts.")
        s.move(
            grips={side: kt.grip_closed(width[side]) for side in kt.SIDES}, seconds=1.0
        )
        s.move(seconds=0.5)
        for side in kt.SIDES:
            if s.object_info(pair[side])["jaw_contacts"][side] != [1, 2]:
                raise RuntimeError(f"{side}: no bilateral grasp on {pair[side]}")
        s.stage("lift", "Lift both parts clear of the supply mats.")
        s.move(to["lift"], seconds=1.5, linear=True, rotations=rot, speed=kt.SPEED)
        for side in kt.SIDES:
            if s.object_info(pair[side])["xyz"][2] < s.rest_z[pair[side]] + 0.06:
                raise RuntimeError(f"{side}: lift failed for {pair[side]}")
        s.stage(
            "transfer",
            f"Carry the {left} and the {right} over cells of their own colors.",
        )
        s.move(to["above"], seconds=2.0, linear=True, rotations=rot, speed=kt.SPEED)
        s.stage("place", "Set both parts down in their tray cells.")
        s.move(to["place"], seconds=1.5, linear=True, rotations=rot, speed=kt.SPEED)
        s.stage("release", "Open both grippers and leave the parts in the cells.")
        s.move(
            grips={side: kt.grip_open(width[side]) for side in kt.SIDES}, seconds=1.0
        )
        s.stage("retract", "Back both grippers away from the trays.")
        s.move(to["retreat"], seconds=1.5, linear=True, rotations=rot, speed=kt.SPEED)
    s.round = 0
    s.stage("finish", "Return both arms; every part must rest in a cell of its color.")
    s.move(
        spec["ready"],
        grips=opened,
        seconds=3.0,
        linear=True,
        rotations=upright,
        speed=kt.SPEED,
    )
    s.move(seconds=1.0)


def stray_contacts(scene):
    """Control ticks at which the robot touched a part that is not one of the
    current round's two (any part, outside the rounds)."""
    count = 0
    for x in scene.trace:
        mine = set(scene.spec["rounds"][x["round"] - 1].values()) if x["round"] else ()
        count += sum(
            o["touching_robot"] and n not in mine for n, o in x["objects"].items()
        )
    return count


def run(output, seed, record=False, kin=None):
    """One expert episode for seed into output (must not exist); writes the run
    files and returns the summary."""
    output.mkdir(parents=True, exist_ok=False)
    spec = kt.sample_episode(seed, kin)
    scene = KitScene(output, spec, record)
    error = None
    start = time.monotonic()
    try:
        expert(scene)
    except (RuntimeError, ValueError) as e:
        error = str(e)
        print("STOP:", error, flush=True)
    finally:
        scene.close()
    placed = scene.placements()
    offsets = {}
    for name, cell in spec["assignment"].items():
        xy = np.array(scene.object_info(name)["xyz"][:2])
        offsets[name] = 1000 * float(np.linalg.norm(xy - scene.cells[cell]["xy"]))
    success = (
        error is None
        and scene.complete()
        and scene.stable_frames >= 15
        and all(
            e["airborne_bilateral_steps"] >= 100 for e in scene.lift_evidence.values()
        )
    )
    result = {
        "success": bool(success),
        "controller": "privileged IK expert (kitting v0), not FineART inference",
        "host": "local Mac",
        "mujoco_version": mujoco.__version__,
        "task": spec["task"],
        "seed": spec["seed"],
        "draws": spec["draws"],
        "instruction": spec["instruction"],
        "duration_s": float(scene.d.time),
        "wall_time_s": time.monotonic() - start,
        "error": error,
        "progress": sum(c is not None for c in placed.values()),
        "placements": placed,
        "offset_from_assigned_cell_mm": offsets,
        "stray_contact_ticks": stray_contacts(scene),
        "final_objects": {n: scene.object_info(n) for n in scene.parts},
        "lift_evidence": scene.lift_evidence,
        "final_stable_control_frames": scene.stable_frames,
        "physics": {
            "max_penetration_mm": scene.maxdepth * 1000,
            "deepest_geoms": scene.deepest,
            "max_joint_limit_error": scene.maxlimit,
        },
        "contract": {
            "action_dim": 16,
            "state_dim": 16,
            "units": "14 arm joint radians plus two gripper closure fractions",
            "cameras": list(POLICY_CAMERAS),
            "control_hz": 25,
            "physics_hz": 500,
            "language": "task = instruction; per tick: phase, subtask, round",
        },
        "collision_setup": scene.setup,
        "camera_rig": scene.camera_metadata,
        "source_urdf_sha256": hashlib.sha256(
            (ROOT / "assets/franzi_source/wheel_robot_4.0.urdf").read_bytes()
        ).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    (output / "trace.json").write_text(json.dumps(scene.trace, indent=2))
    if scene.trace:
        np.savez_compressed(
            output / "trajectory.npz",
            state=[x["state"] for x in scene.trace],
            action=[x["action"] for x in scene.trace],
            timestamp=[x["t"] for x in scene.trace],
            # subtask holds the phase names, as run_expert writes it, so
            # data.convert_expert_run reads kitting runs too.
            subtask=[x["phase"] for x in scene.trace],
            instruction=[x["subtask"] for x in scene.trace],
            round=[x["round"] for x in scene.trace],
            progress=[x["progress"] for x in scene.trace],
            task=spec["instruction"],
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record", action="store_true")
    args = p.parse_args()
    result = run(args.output, args.seed, args.record)
    print(
        json.dumps({k: result[k] for k in ("success", "error", "progress")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
