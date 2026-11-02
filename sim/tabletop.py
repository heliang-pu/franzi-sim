"""Tabletop scenes that move free objects with both hands: the 45-degree
grasp, grip widths, key poses, an object-tracking Scene and the run files.

A scene's task.py samples its layout and builds its objects (bodies named
<name>_part, see sim/props); its run.py subclasses ObjectScene with the task's
conditions and scripts its expert from these key poses; run_episode() runs it
and writes the files every scene writes (scene.xml, episode.json, summary.json,
trace.json, trajectory.npz and, with record, the videos).
"""

from __future__ import annotations

import hashlib
import json
import socket
import time

import mujoco
import numpy as np

from robot_kinematics import URDF, Kinematics
from sim.build_task import FINGERS
from sim.camera_rig import POLICY_CAMERAS
from sim.episode import PhysicsError, Scene
from sim.gripper_linkage import contact_drop

SIDES = ("left", "right")
JAW_SPAN = 0.095  # jaw gap at closure 0; closure g moves each finger 0.0475 g
CLEAR = 0.012  # finger clearance per side while open
# Closure past contact. The kitting parts (50-60 g) hold at 2.1 mm (~1.6 N);
# objects of 90-130 g get 4 mm. Heavier ones pass more (8 mm saturates the
# finger servos' 6 N).
SQUEEZE = 0.004
PRE = 0.08  # pre-grasp and retreat distance along the approach axis
# The fingertips beyond the pads' centre along the fingers, open or closed
# (the finger moves as one piece): beyond the grasp site with the jaws open,
# TIP + contact_drop(width) closed. Measured on wheel_robot_26.8.16_3's finger
# meshes: 13.4 mm at every opening (the 17 mm before was wheel_robot_4.0's).
TIP = 0.0134
SET_GAP = 0.003  # objects are let go this far above their support
# The two-part expert's ready point, reached fingers down and then turned into
# the 45-degree grasp in place (swung straight there, a fingertip brushed the
# table, which the gripper linkage turned into a jaw driven shut).
READY = {"left": [0.15, 0.37, 0.99], "right": [0.15, -0.37, 0.99]}
# The jaws in the default posture (start_home): open, 5 mm per finger off the
# open limit. Raised from hanging with the fingers on that limit, a finger
# overshot it by 2.3 mm (the 2 mm audit stops the episode).
HOME_OPENING = 0.085
# The expert keeps 10 mrad inside the joint limits and the samplers 20 mrad,
# so measured object poses that differ a little from the sampled ones plan.
LIMIT_MARGIN = 0.01
SAMPLE_MARGIN = 0.02
SPEED = 0.6  # peak joint command rate (rad/s), as in the kitting expert
STILL = 0.025  # m/s
UPRIGHT = np.cos(np.radians(8))


def kinematics(margin=LIMIT_MARGIN):
    """The shared IK with a scene limit margin."""
    return Kinematics(margin=margin)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def grasp_rotation(side, psi=0.0):
    """Hand frame for the 45-degree grasp at azimuth psi: approach forward and
    down, rolled so the fingers close horizontally and the wrist D405 stays on
    top of the hand (the kitting scene's grasp)."""
    roll = -np.pi / 2 if side == "left" else np.pi / 2
    return rot_z(psi) @ rot_y(-np.pi / 4) @ rot_z(roll)


def grip_open(width):
    """Open closure for an object of width (m). The hands never open fully:
    open fingers sit on their joint limits, and gravity along the tilted
    slides pushes them past."""
    return 1 - (width + 2 * CLEAR) / JAW_SPAN


def grip_closed(width, squeeze=SQUEEZE):
    return 1 - (width - squeeze) / JAW_SPAN


def pick_place(side, base, grasp_z, width, target, lift, above=None, psi=0.0):
    """Grasp-site targets (world, m) for moving an object whose base is at base
    (x, y, z) to target (the base's x, y, z there), grasped grasp_z above its
    base, with the hand rotation. above is how high the base is carried over
    the target (default: lift). The hand stops contact_drop(width) short along
    the approach, so the dropping pads close at the grasp point."""
    rotation = grasp_rotation(side, psi)
    axis = rotation @ [0, 0, 1]  # opposite the approach direction
    short = contact_drop(width) * axis
    grasp = np.asarray(base, float) + [0, 0, grasp_z] + short
    place = np.asarray(target, float) + [0, 0, grasp_z + SET_GAP] + short
    up = np.array([0, 0, lift])
    over = np.array([0, 0, lift if above is None else above])
    return {
        "rotation": rotation,
        "pregrasp": grasp + PRE * axis,
        "grasp": grasp,
        "lift": grasp + up,
        "above": place + over,
        "place": place,
        "rise": place + over,
        "retreat": place + PRE * axis,
    }


def press_poses(top, side):
    """Grasp-site targets for pressing with the closed fingertips at top (x,
    y, z of what they press): 5 cm over it, and touching it."""
    rotation = grasp_rotation(side)
    axis = rotation @ [0, 0, 1]
    reach = TIP + contact_drop(0.0)  # closed fingertips beyond the grasp site
    tip = np.asarray(top, float) + reach * axis
    return {
        "rotation": rotation,
        "above": tip + [0, 0, 0.05],
        "touch": tip + [0, 0, 0.004],
    }


def ready_pose(kin, side):
    """Arm pose at the ready point as go_ready reaches it."""
    down = kin.ik(side, READY[side], np.zeros(7))
    return kin.ik(side, READY[side], down, rotation=grasp_rotation(side))


def plan(kin, side, targets):
    """Raises RuntimeError where a key pose of side's path (from the ready
    pose through targets and back to READY), or a 1 cm waypoint between them,
    has no IK solution inside the limits."""
    rotation = grasp_rotation(side)
    q = ready_pose(kin, side)
    for target in [*targets, READY[side]]:
        q = kin.line(side, q, target, rotation=rotation)[-1]


class ObjectScene(Scene):
    """sim.episode.Scene with free objects: per-object state and contacts,
    grasp evidence for the objects the hands lift, and the task's milestones,
    each reached once its condition held. A scene sets OBJECTS (free bodies
    <name>_part with freejoints <name>_part_joint), HELD (the lifted ones) and
    MILESTONES, and defines conditions() and complete(). In the phases named
    in CARRY_PHASES, a held object (both jaws of a hand on it) that touches
    anything but the robot is logged as a collision too (carried_collisions)."""

    OBJECTS: tuple = ()
    HELD: tuple = ()
    MILESTONES: tuple = ()
    CARRY_PHASES: tuple = ()

    def __init__(self, output, spec, record=False):
        self.spec = spec
        self.reached = {k: False for k in self.MILESTONES}
        super().__init__(output, record)
        self.kin = kinematics()
        (output / "episode.json").write_text(json.dumps(spec, indent=2))
        self.ids = {n: self.m.body(n + "_part").id for n in self.OBJECTS}
        self.free = {
            n: int(self.m.joint(n + "_part_joint").dofadr[0]) for n in self.OBJECTS
        }
        self.jaws = {s: [self.m.body(f).id for f in FINGERS[s]] for s in SIDES}
        self.rest_z = {n: float(self.d.xpos[b][2]) for n, b in self.ids.items()}
        self.lift_evidence = {
            n: {
                "peak_height_m": 0.0,
                "bilateral_physics_steps": 0,
                "airborne_bilateral_steps": 0,
            }
            for n in self.HELD
        }
        self.carried_log = {}

    def conditions(self, pairs):
        """The task's conditions now, {milestone: bool}."""
        raise NotImplementedError

    def contact_bodies(self):
        """Body ids of this step's contact pairs, shape (ncon, 2)."""
        return self.m.geom_bodyid[self.d.contact.geom]

    def touching(self, name, pairs):
        bid = self.ids[name]
        return set(pairs[(pairs == bid).any(1)].ravel().tolist()) - {bid}

    def xyz(self, name):
        return self.d.xpos[self.ids[name]]

    def object_info(self, name, pairs=None):
        pairs = self.contact_bodies() if pairs is None else pairs
        others = self.touching(name, pairs)
        bid, dof = self.ids[name], self.free[name]
        return {
            "xyz": self.d.xpos[bid].tolist(),
            "quat": self.d.xquat[bid].tolist(),
            "velocity_m_s": float(np.linalg.norm(self.d.qvel[dof : dof + 3])),
            "touching_robot": bool(others & self.robot_bodies),
            "jaw_contacts": {
                s: [i for i, b in enumerate(self.jaws[s], 1) if b in others]
                for s in SIDES
            },
        }

    def settled(self, name, pairs, upright=True):
        """Still, clear of the robot and (by default) upright."""
        bid, dof = self.ids[name], self.free[name]
        return (
            (not upright or self.d.xmat[bid][8] > UPRIGHT)
            and np.linalg.norm(self.d.qvel[dof : dof + 3]) < STILL
            and not self.touching(name, pairs) & self.robot_bodies
        )

    def status(self, pairs=None):
        """The conditions now; each milestone stays reached once it held."""
        pairs = self.contact_bodies() if pairs is None else pairs
        now = {k: bool(v) for k, v in self.conditions(pairs).items()}
        for k, v in now.items():
            self.reached[k] = self.reached[k] or v
        return now

    def progress(self):
        return int(sum(self.reached.values()))

    def sample(self):
        pairs = self.contact_bodies()
        return {
            "t": self.clock(),
            "phase": self.phase,
            "subtask": self.instruction,
            "state": self.state().tolist(),
            "action": self.command().tolist(),
            "conditions": self.status(pairs),
            "progress": self.progress(),
            "objects": {n: self.object_info(n, pairs) for n in self.OBJECTS},
            "max_penetration_mm": self.maxdepth * 1000,
        }

    def track_objects(self):
        pairs = self.contact_bodies()
        for name in self.HELD:
            e = self.lift_evidence[name]
            z = float(self.xyz(name)[2])
            e["peak_height_m"] = max(e["peak_height_m"], z)
            others = self.touching(name, pairs)
            if any(others.issuperset(self.jaws[s]) for s in SIDES):
                e["bilateral_physics_steps"] += 1
                if z > self.rest_z[name] + 0.03:
                    e["airborne_bilateral_steps"] += 1
                if self.phase in self.CARRY_PHASES:
                    for other in others - self.robot_bodies:
                        key = (name, self.body_names[other] or "world")
                        hit = self.carried_log.setdefault(
                            key,
                            {
                                "object": name,
                                "touched": key[1],
                                "first_s": round(self.clock(), 3),
                                "phase": self.phase,
                                "physics_steps": 0,
                            },
                        )
                        hit["physics_steps"] += 1

    def carried_collisions(self):
        """Held objects touching anything but the robot while carried."""
        return sorted(self.carried_log.values(), key=lambda h: h["first_s"])


def start_home(s):
    """The robot's default posture, which a recorded episode starts in (user,
    2026-10-25: both grippers held up in front, the head down): both hands at
    READY, fingers down, jaws HOME_OPENING apart, the head at the workstation
    gaze. It is reached before the recording (Scene.prepare), the hands raised
    straight up first: swung there in joint space, both fingertips brushed
    tote_transfer's bench (0.17 mm, the contact audit). The arms end on the
    IK's joints from the hanging arm, the posture samplers plan from."""
    s.stage("home", "Take the default posture: grippers held up, head down.")
    grip = 1 - HOME_OPENING / JAW_SPAN
    s.prepare(READY, grips={side: grip for side in SIDES}, seconds=3.0, lift=True)


def go_ready(s, grips):
    """To the ready point as ready_pose plans it: fingers down, then the hands
    turned in place."""
    upright = {side: grasp_rotation(side) for side in SIDES}
    s.move(READY, grips=grips, seconds=3.0, speed=SPEED)
    s.move(READY, grips=grips, seconds=1.5, rotations=upright, speed=SPEED)


def mover(s):
    """Straight-line moves in the 45-degree grasp at the scene speed:
    move(targets, seconds=2.0, grips=None)."""
    rot = {side: grasp_rotation(side) for side in SIDES}

    def move(targets=None, seconds=2.0, **kw):
        s.move(targets, seconds=seconds, linear=True, rotations=rot, speed=SPEED, **kw)

    return move


def check_grasp(s, name, side):
    if s.object_info(name)["jaw_contacts"][side] != [1, 2]:
        raise RuntimeError(f"{side}: no bilateral grasp on the {name}")


def lift(s, move, targets, rises, seconds=1.5):
    """Move to targets, then check that each object rose at least its rise (m)
    from where it was."""
    start = {n: float(s.xyz(n)[2]) for n in rises}
    move(targets, seconds)
    for n, rise in rises.items():
        if s.xyz(n)[2] < start[n] + rise:
            raise RuntimeError(f"lift failed for the {n}")


def press(s, move, side, poses, depth, target, pressed):
    """Press with side's closed fingertips from poses["touch"] (press_poses),
    2 mm lower at a time until depth() (the pressed thing's travel, m, read
    from the simulator: privileged) reaches target. At the 45-degree hand the
    pads' lowest corner, not the fingertip axis, meets the button first, so a
    fixed target pressed under 1 mm. Raises RuntimeError if the travel never
    reached pressed."""
    move({side: poses["touch"]}, 1.0)
    goal, deepest = np.array(poses["touch"]), depth()
    for _ in range(12):
        if depth() >= target:
            break
        goal = goal - [0, 0, 0.002]
        move({side: goal}, 0.3)
        deepest = max(deepest, depth())
    s.move(seconds=0.4)
    if max(deepest, depth()) < pressed:
        raise RuntimeError("the button did not go down")


def run_episode(scene_cls, expert, spec, output, record=False, extra=None, engine=None):
    """One expert episode for spec into output (must not exist): writes the
    run files and returns the summary. Success needs no error, the scene's
    complete() held for 15 control ticks at the end, 100 airborne steps of
    bilateral grasp on every held object, and no collision (the contact audit)
    or violation (Scene.violation, logged instead of ending the episode only
    with stop_on_violation off). "completed" is the same test without the
    collisions and violations: the task done, whatever it touched (a policy
    evaluation's lenient count). A run whose simulation broke down
    (PhysicsError) is marked void: redraw it rather than count it.
    extra(scene) adds summary fields. engine(scene_cls) runs the scene on
    another simulator (isaac.episode)."""
    output.mkdir(parents=True, exist_ok=False)
    scene = (engine(scene_cls) if engine else scene_cls)(output, spec, record)
    error, void = None, False
    start = time.monotonic()
    try:
        expert(scene)
    except (RuntimeError, ValueError) as e:
        error, void = str(e), isinstance(e, PhysicsError)
        print("VOID:" if void else "STOP:", error, flush=True)
    finally:
        scene.close()
    final = scene.status()
    collisions, carried = scene.collisions(), scene.carried_collisions()
    violations = scene.violations()
    completed = (
        error is None
        and scene.complete()
        and scene.stable_frames >= 15
        and all(
            e["airborne_bilateral_steps"] >= 100 for e in scene.lift_evidence.values()
        )
    )
    success = completed and not collisions and not carried and not violations
    result = {
        "success": bool(success),
        "completed": bool(completed),
        "controller": f"privileged IK expert ({spec['task']} v{spec['version']}), "
        "not FineART inference",
        "host": socket.gethostname(),
        "simulator": scene.SIMULATOR,
        "mujoco_version": mujoco.__version__,
        "task": spec["task"],
        "seed": spec["seed"],
        "draws": spec["draws"],
        "instruction": spec["instruction"],
        "duration_s": scene.clock(),
        # Simulated before the recording (Scene.prepare), not in the trace.
        "prepare_s": scene.t0,
        "wall_time_s": time.monotonic() - start,
        "error": error,
        "void": void,
        "progress": scene.progress(),
        "milestones": scene.reached,
        "final_conditions": final,
        **(extra(scene) if extra else {}),
        "final_objects": {n: scene.object_info(n) for n in scene.OBJECTS},
        "lift_evidence": scene.lift_evidence,
        "final_stable_control_frames": scene.stable_frames,
        "physics": {
            "max_penetration_mm": scene.maxdepth * 1000,
            "deepest_geoms": scene.deepest,
            "max_joint_limit_error": scene.maxlimit,
            # The contact audit (Scene.audit_contact, CARRY_PHASES): a run
            # with any collision is not a success.
            "collisions": collisions,
            "carried_collisions": carried,
            # Penetrations and limit errors that didn't end the episode
            # (stop_on_violation off); any is a failure too.
            "violations": violations,
        },
        "contract": {
            "action_dim": 16,
            "state_dim": 16,
            "units": "14 arm joint radians plus two gripper closure fractions",
            "cameras": list(POLICY_CAMERAS),
            "control_hz": 25,
            "physics_hz": 500,
            "language": "task = instruction; per tick: phase, subtask",
            "full": scene.full_contract(),
        },
        "collision_setup": scene.setup,
        "camera_rig": scene.camera_metadata,
        "source_urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    (output / "trace.json").write_text(json.dumps(scene.trace, indent=2))
    if scene.trace:
        np.savez_compressed(
            output / "trajectory.npz",
            state=[x["state"] for x in scene.trace],
            action=[x["action"] for x in scene.trace],
            timestamp=[x["t"] for x in scene.trace],
            # subtask holds the phase names, as the other scenes write it, so
            # data.convert_expert_run reads these runs too.
            subtask=[x["phase"] for x in scene.trace],
            instruction=[x["subtask"] for x in scene.trace],
            progress=[x["progress"] for x in scene.trace],
            task=spec["instruction"],
            **scene.full_arrays(),
        )
    scene.write_full(output)
    return result
