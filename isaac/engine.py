"""MuJoCo scenes with their physics on Isaac Sim 5.1: isaac_scene(cls).

isaac_scene() turns a sim.tabletop.ObjectScene subclass into one whose physics
runs on PhysX, so the scene's expert, conditions and success test run
unchanged and closed-loop on Isaac Sim (user decision 2026-10-25):
- The MuJoCo model is built and compiled as in every run, then exported and
  authored into the episode's USD stage in <run>/isaac/ (isaac/export.py,
  isaac/stage.py) and run by isaac/runtime.py.
- Every physics step sets the Isaac joint targets from the scene's goal and
  grip and the head's gaze ramp, as sim.episode.Scene sets MuJoCo's ctrl, and
  steps PhysX. Isaac's state is then copied into the MuJoCo data, a kinematic
  mirror: the robot's and the props' joints, the free bodies' poses and
  velocities, then mj_kinematics. All the expert and the scene read (xpos,
  xquat, xmat, qvel, joint values) is Isaac's.
- Contacts come from PhysX's contact report, mapped back to MuJoCo bodies and
  geoms, for contact_bodies(), the grasp evidence, the 1 mm penetration audit
  and the contact audit (Scene.audit_contact, the carried objects' check), so
  success means the same as on MuJoCo. check_physics() runs on the mirror.
- With record, the overview and the policy cameras render on RTX into the same
  videos a MuJoCo run writes. Without DISPLAY only the four policy cameras
  render and record: the overview (expert.mp4) and the mosaic (cameras.mp4)
  are for people to watch, and a dataset needs neither.
- Each episode's light comes from its seed (isaac/stage.py, RANDOM_LIGHT).

Import it only inside a running Isaac Sim app (runtime.start_app()).
"""

import mujoco
import numpy as np

from isaac import export, runtime, stage
from robots.franzi.camera_rig import POLICY_CAMERAS
from robots.franzi.scene import ARM_JOINTS, FINGER_JOINTS
from sim.episode import Video

# PhysX reports pairs up to the 4 mm contact offset apart; MuJoCo's contacts
# (margin 0) touch. A pair touches here within 0.5 mm.
TOUCH = 0.0005
# Record the overview and mosaic videos too (isaac.episode --no-display).
DISPLAY = True


class IsaacEngine:
    LABEL = "ISAAC SIM 5.1"
    SIMULATOR = "Isaac Sim 5.1 (PhysX TGS on the CPU at 500 Hz, RTX)"
    # sim.episode's depth and segmentation videos come from MuJoCo's renderer,
    # which an Isaac run does not have; RTX depth is not wired in yet.
    RECORD_DEPTH = False

    def __init__(self, output, spec, record=False):
        # The MuJoCo scene without MuJoCo's renderer: Isaac records the videos.
        super().__init__(output, spec, False)
        self.record = record
        folder = output / "isaac"
        folder.mkdir()
        mujoco.mj_forward(self.m, self.d)
        entry = {k: spec.get(k) for k in ("task", "seed", "instruction")}
        model, arrays = export.write_model(self.m, self.d, folder, entry)
        stage.build(folder, model, arrays)
        self.display = record and DISPLAY
        views = [*(["overview"] if self.display else []), *POLICY_CAMERAS]
        views = views if record else []
        self.rt = runtime.Runtime(folder, views=views)
        m = self.m
        self.qadr, self.vadr = {}, {}
        for name in [*self.rt.index, *self.rt.prop_dofs]:
            joint = m.joint(name)
            self.qadr[name], self.vadr[name] = joint.qposadr[0], joint.dofadr[0]
        self.free_adr = {}
        for name in self.rt.free_names:
            j = m.body_jntadr[m.body(name).id]
            self.free_adr[name] = (m.jnt_qposadr[j], m.jnt_dofadr[j])
        # MuJoCo geom ids by the collider names stage.py records.
        self.geom_ids = {m.geom(g).name or f"geom{g}": g for g in range(m.ngeom)}
        self.arm_names = {s: [f"{s}_{j}_joint" for j in ARM_JOINTS] for s in self.goal}
        # The finger that closes with a positive slide first (FINGER_JOINTS).
        self.finger_names = {s: list(FINGER_JOINTS[s]) for s in self.goal}
        if record:
            if self.display:
                self.videos["demo"] = Video(output / "expert.mp4", 1280, 720)
                self.videos["cameras"] = Video(output / "cameras.mp4", 1280, 960)
            for cam in POLICY_CAMERAS:
                self.videos[cam] = Video(output / (cam + ".mp4"), 640, 480)
        self.mirror()

    def mirror(self):
        """Copy Isaac's state into the MuJoCo data and update its kinematics."""
        d = self.d
        for name, value in self.rt.joint_positions().items():
            d.qpos[self.qadr[name]] = value
        for name, value in self.rt.joint_velocities().items():
            d.qvel[self.vadr[name]] = value
        for name, (value, rate) in self.rt.prop_state().items():
            d.qpos[self.qadr[name]] = value
            d.qvel[self.vadr[name]] = rate
        for name, (p, quat, v, w) in self.rt.free_state().items():
            a, dof = self.free_adr[name]
            d.qpos[a : a + 3] = p
            d.qpos[a + 3 : a + 7] = quat
            d.qvel[dof : dof + 3] = v
            # MuJoCo's free-joint angular velocity is in the body frame.
            d.qvel[dof + 3 : dof + 6] = stage.rot(quat).T @ w
        d.time = self.step_number * self.rt.dt
        mujoco.mj_kinematics(self.m, d)
        # The camera poses log_full() records.
        mujoco.mj_comPos(self.m, d)
        mujoco.mj_camlight(self.m, d)

    def physics_step(self):
        # sim.episode.Scene.physics_step, with PhysX for mj_step.
        if self.recording and self.step_number % 20 == 0:
            self.trace.append(self.sample())
            self.log_full()
            self.stable_frames = self.stable_frames + 1 if self.complete() else 0
            if self.record:
                self.render()
        targets = dict(self.body_goal)  # the torso servos (turn())
        targets["head_yaw_joint"], targets["head_pitch_joint"] = self.head_command()
        for side in self.goal:
            targets.update(zip(self.arm_names[side], self.goal[side]))
            first, second = self.finger_names[side]
            targets[first] = self.grip[side] * 0.0475
            targets[second] = -self.grip[side] * 0.0475
        self.rt.set_targets(targets)
        # A recorded tick renders once, in render(). Before the recording the
        # last step of each tick renders too (Runtime.step), so that RTX's
        # denoiser and caches are warm for the first frame.
        warm = self.record and not self.recording
        self.rt.step(sync=warm and (self.step_number + 1) % 20 == 0)
        self.step_number += 1
        self.mirror()
        self.check_physics()  # on the mirror: a non-finite state, a flung object
        self.audit()
        self.track_objects()

    def log_full(self):
        """sim.episode.Scene.log_full on the mirror, with Isaac's efforts and
        contact forces: MuJoCo's actuator forces and contacts are not computed
        there. Efforts read as in MuJoCo, the servo force plus the gravity
        feedforward, here PhysX's measured joint force plus the mirror's
        gravity and Coriolis terms (Isaac's robot links ignore gravity)."""
        m, d = self.m, self.d
        mujoco.mj_comVel(m, d)
        bias = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, bias)
        d.qfrc_applied[:] = 0
        for side in self.dofs:
            d.qfrc_applied[self.dofs[side]] = bias[self.dofs[side]]
        d.qfrc_applied[self.held] = bias[self.held]
        efforts = self.rt.joint_efforts()
        for u in range(m.nu):
            name = m.joint(m.actuator_trnid[u][0]).name
            if name in efforts:
                d.actuator_force[u] = efforts[name]
        super().log_full()
        forces = {}
        for (b1, n1), (b2, n2), force in self.rt.contact_forces:
            if b1 is None or b2 is None:
                continue
            if b1 in self.robot_bodies and b2 in self.robot_bodies:
                continue
            pair = tuple(
                sorted((m.body(b1).name or "world", m.body(b2).name or "world"))
            )
            forces[pair] = forces.get(pair, 0.0) + force
        self.contacts[-1] = [[*pair, round(f, 4)] for pair, f in sorted(forces.items())]

    def audit(self):
        """sim.episode.Scene's audits: no contact deeper than 1 mm, no joint
        more than 0.002 past its limit (Scene.violation), and the contact audit
        (audit_contact) on the pairs that touch (within TOUCH, as
        contact_bodies() counts)."""
        for (b1, first), (b2, second), depth in self.rt.contacts:
            if depth > self.maxdepth:
                self.maxdepth, self.deepest = depth, [first, second]
            if depth > 0.001:
                self.violation(
                    ("penetration", first, second),
                    f"Penetration {depth * 1000:.3f} mm: {[first, second]}",
                    depth * 1000,
                )
            if depth > -TOUCH and b1 is not None and b2 is not None:
                g1, g2 = self.geom_ids[first], self.geom_ids[second]
                self.audit_contact(g1, g2, max(depth, 0.0))
        m, d = self.m, self.d
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                q = float(d.qpos[m.jnt_qposadr[j]])
                lo, hi = m.jnt_range[j]
                err = max(lo - q, q - hi, 0.0)
                self.maxlimit = max(self.maxlimit, err)
                if err > 0.002:
                    name = m.joint(j).name
                    self.violation(
                        ("joint_limit", name),
                        f"Joint limit violation: {name} {err:.6f}",
                        err,
                    )

    def contact_bodies(self):
        """Body ids of this step's touching pairs, shape (n, 2), as MuJoCo's."""
        pairs = [
            (first[0], second[0])
            for first, second, depth in self.rt.contacts
            if depth > -TOUCH and first[0] is not None and second[0] is not None
        ]
        return np.array(pairs, dtype=int).reshape(-1, 2)

    def capture(self):
        return self.rt.render()

    def render(self):
        """sim.episode.Scene.render, or without DISPLAY only the policy
        cameras' videos (RECORD_DEPTH is off here)."""
        if self.display:
            return super().render()
        images = self.capture()
        for cam in POLICY_CAMERAS:
            self.videos[cam].write(images[cam])
        self.frames += 1


def isaac_scene(cls):
    """cls (a sim.tabletop.ObjectScene subclass) with its physics on Isaac Sim."""
    return type(f"Isaac{cls.__name__}", (IsaacEngine, cls), {})
