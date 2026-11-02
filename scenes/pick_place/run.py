"""Dual-arm pick and place: PickPlaceScene and the scripted IK expert.

PickPlaceScene is sim.episode.Scene with task.py's two parts: their grasp
evidence and the success test (each part on its pad, seated, still and clear of
the robot). The expert reads part poses only for its checks. No object pose
writes after reset, welds, attachment constraints, contact-mask changes, or
simulator-state inputs to a learned policy. This is an expert demo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from robot_kinematics import URDF
from scenes.pick_place.task import GOALS, PAD_DROP, SOURCES, build
from sim import tabletop as tt
from sim.build_task import FINGERS, TABLE_Z
from sim.camera_rig import POLICY_CAMERAS
from sim.episode import PhysicsError, Scene

# Closure on the 45 mm parts: the jaws 42.9 mm apart, a 2 mm squeeze.
GRIP = 0.548


class PickPlaceScene(Scene):
    def __init__(self, output, record=False):
        super().__init__(output, record)
        self.lift_evidence = {
            s: {
                "peak_height_m": 0.0,
                "bilateral_physics_steps": 0,
                "airborne_bilateral_steps": 0,
            }
            for s in self.qids
        }

    def build_root(self):
        return build()

    def jaw_contacts(self, side):
        target = self.m.body(side + "_part").id
        jaws = {self.m.body(f).id: i for i, f in enumerate(FINGERS[side], 1)}
        result = set()
        for c in self.d.contact:
            b = [int(self.m.geom_bodyid[g]) for g in c.geom]
            if target in b:
                result.update(jaws[v] for v in b if v in jaws)
        return sorted(result)

    def object_info(self, side):
        j = self.m.joint(side + "_part_joint")
        da = int(j.dofadr[0])
        bid = self.m.body(side + "_part").id
        touched = False
        for c in self.d.contact:
            pair = {int(self.m.geom_bodyid[g]) for g in c.geom}
            if bid in pair and pair & self.robot_bodies:
                touched = True
        return {
            "xyz": self.d.xpos[bid].tolist(),
            "velocity_m_s": float(np.linalg.norm(self.d.qvel[da : da + 3])),
            "touching_robot": touched,
            "jaw_contacts": self.jaw_contacts(side),
        }

    def sample(self):
        """One trace record: the measured state and the command about to apply."""
        return {
            "t": self.clock(),
            "phase": self.phase,
            "subtask": self.instruction,
            "state": self.state().tolist(),
            "action": self.command().tolist(),
            "objects": {s: self.object_info(s) for s in self.goal},
            "max_penetration_mm": self.maxdepth * 1000,
        }

    def track_objects(self):
        """Grasp evidence, updated after every physics step."""
        for side in self.goal:
            info = self.object_info(side)
            e = self.lift_evidence[side]
            e["peak_height_m"] = max(e["peak_height_m"], info["xyz"][2])
            if len(info["jaw_contacts"]) == 2:
                e["bilateral_physics_steps"] += 1
                if info["xyz"][2] > TABLE_Z + 0.10:
                    e["airborne_bilateral_steps"] += 1

    def complete(self):
        infos = {s: self.object_info(s) for s in SOURCES}
        return all(
            np.linalg.norm(np.array(infos[s]["xyz"])[:2] - GOALS[s]) < 0.025
            and abs(infos[s]["xyz"][2] - (TABLE_Z + 0.006 + 0.0275)) < 0.004
            and infos[s]["velocity_m_s"] < 0.025
            and not infos[s]["touching_robot"]
            for s in SOURCES
        )


def expert(scene):
    """From the default posture, each arm in turn picks its part, puts it on
    its pad and moves back: one arm and one object per subtask, the other arm
    holding (user, 2026-10-25). The subtask text is the trace's subtask, the
    stage name its finer phase."""
    s = scene
    tt.start_home(s)
    opened = 1 - tt.HOME_OPENING / tt.JAW_SPAN
    for side, color in (("left", "blue"), ("right", "orange")):
        source, goal = SOURCES[side], GOALS[side]
        pick = f"Pick up the {color} box with the {side} hand."
        s.stage("approach", pick)
        s.move({side: [*source, TABLE_Z + 0.16]}, seconds=2.0)
        s.stage("align", pick)
        s.move({side: [*source, TABLE_Z + 0.0275 + PAD_DROP]}, seconds=2.0, linear=True)
        s.stage("grasp", pick)
        s.move(grips={side: GRIP}, seconds=1.5)
        s.move(seconds=0.5)
        if len(s.jaw_contacts(side)) != 2:
            raise RuntimeError(f"{side}: bilateral grasp not established")
        s.stage("lift", pick)
        s.move({side: [*source, TABLE_Z + 0.15]}, seconds=2.5, linear=True)
        if s.object_info(side)["xyz"][2] < TABLE_Z + 0.10:
            raise RuntimeError(f"{side}: lift failed")
        place = f"Put the {color} box on the {color} pad with the {side} hand."
        s.stage("transfer", place)
        s.move({side: [*goal, TABLE_Z + 0.15]}, seconds=3.0, linear=True)
        s.stage("place", place)
        s.move(
            {side: [*goal, TABLE_Z + 0.006 + 0.0275 + PAD_DROP]},
            seconds=2.5,
            linear=True,
        )
        s.stage("release", place)
        s.move(grips={side: opened}, seconds=1.5)
        back = f"Move the {side} arm back to its start pose."
        s.stage("retract", back)
        s.move({side: [*goal, TABLE_Z + 0.15]}, seconds=2.0, linear=True)
        s.stage("finish", back)
        # Slow enough that the swing back from the far pad does not push the
        # upright thigh, which rests on its lower joint limit, past that limit.
        s.move({side: tt.READY[side]}, seconds=3.0)
    s.move(seconds=1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--record", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = PickPlaceScene(args.output, args.record)
    error, void = None, False
    start = time.monotonic()
    try:
        expert(scene)
    except (RuntimeError, ValueError) as e:
        error, void = str(e), isinstance(e, PhysicsError)
        print("VOID:" if void else "STOP:", error, flush=True)
    finally:
        scene.close()
    objects = {side: scene.object_info(side) for side in SOURCES}
    success = (
        error is None
        and scene.complete()
        and scene.stable_frames >= 15
        and all(
            scene.lift_evidence[s]["airborne_bilateral_steps"] >= 100 for s in SOURCES
        )
        and not scene.collisions()
    )
    result = {
        "success": bool(success),
        "controller": "privileged IK expert, not FineART inference",
        "host": "local Mac",
        "mujoco_version": mujoco.__version__,
        "duration_s": scene.clock(),
        "wall_time_s": time.monotonic() - start,
        "error": error,
        "void": void,
        "final_objects": objects,
        "lift_evidence": scene.lift_evidence,
        "final_stable_control_frames": scene.stable_frames,
        "physics": {
            "max_penetration_mm": scene.maxdepth * 1000,
            "deepest_geoms": scene.deepest,
            "max_joint_limit_error": scene.maxlimit,
            "collisions": scene.collisions(),
        },
        "contract": {
            "action_dim": 16,
            "state_dim": 16,
            "units": "14 arm joint radians plus two gripper closure fractions",
            "cameras": list(POLICY_CAMERAS),
            "control_hz": 25,
            "physics_hz": 500,
            "full": scene.full_contract(),
        },
        "collision_setup": scene.setup,
        "camera_rig": scene.camera_metadata,
        "source_urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    (args.output / "trace.json").write_text(json.dumps(scene.trace, indent=2))
    if scene.trace:
        np.savez_compressed(
            args.output / "trajectory.npz",
            state=[x["state"] for x in scene.trace],
            action=[x["action"] for x in scene.trace],
            timestamp=[x["t"] for x in scene.trace],
            subtask=[x["phase"] for x in scene.trace],
            **scene.full_arrays(),
        )
    scene.write_full(args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
