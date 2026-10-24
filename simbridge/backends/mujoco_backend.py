"""MuJoCo adapter around scenes.pick_place.run.PickPlaceScene, the
pick-and-place scene on sim.episode.Scene (the derived scene and its audits).

Scene.physics_step keeps every rule of the expert: position-actuator ctrl plus
gravity feedforward through qfrc_applied, a per-step penetration (> 1 mm) and
joint-limit (> 0.002) audit, the head gaze ramp, no pose writes.
"""

import time

import mujoco
import numpy as np

from scenes.pick_place.run import PickPlaceScene
from sim.build_task import ARM_JOINTS
from simbridge.backends.base import (
    EFFORT_SIMULATED,
    BackendState,
    Capabilities,
    SimBackend,
)
from simbridge.status import PhysicsFault


class SharedRenderer:
    """One GL and MuJoCo render context for every image size. Each context holds
    its own shadow map (8192^2 here, about 256 MB), so a mujoco.Renderer per
    size multiplied that. The steps are mujoco.Renderer's: the model's offscreen
    buffer, with a viewport per request, so the pixels are the same."""

    def __init__(self, m):
        self.m = m
        self.limit = (m.vis.global_.offwidth, m.vis.global_.offheight)
        self.gl = mujoco.GLContext(*self.limit)
        self.gl.make_current()
        self.scene = mujoco.MjvScene(m, maxgeom=10000)
        self.option = mujoco.MjvOption()
        self.context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.context)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

    def render(self, d, camera_id, width, height):
        if width > self.limit[0] or height > self.limit[1]:
            raise ValueError(f"{width}x{height} exceeds the offscreen buffer")
        self.gl.make_current()
        self.camera.fixedcamid = camera_id
        mujoco.mjv_updateScene(
            self.m,
            d,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        rect = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(rect, self.scene, self.context)
        rgb = np.empty((height, width, 3), np.uint8)
        mujoco.mjr_readPixels(rgb, None, rect, self.context)
        return np.flipud(rgb)

    def close(self):
        self.context.free()
        self.gl.free()


class MujocoBackend(SimBackend):
    name = "mujoco"
    label = f"MuJoCo {mujoco.__version__}"
    implementation_status = "implemented"

    def load(self, robot, options, output):
        # Scene.physics_step audits with fixed thresholds; refuse a config that
        # asks for different ones rather than silently ignoring it.
        if robot.safety["max_penetration_m"] != 0.001 or (
            robot.safety["max_joint_limit_overshoot_rad"] != 0.002
        ):
            raise ValueError("MuJoCo Scene audits are fixed at 1 mm and 0.002 rad")
        self.robot = robot
        self.options = options
        self.scene = PickPlaceScene(output, record=False)
        m = self.m = self.scene.m
        self.d = self.scene.d
        for side in ("left", "right"):
            expected = tuple(f"{side}_{j}_joint" for j in ARM_JOINTS)
            if robot.arms[side].sim_joints != expected:
                raise ValueError(f"{side}: robot config joint order differs from Scene")
            fingers = tuple(f"{side}finger{i}_joint" for i in (1, 2))
            if robot.grippers[side].sim_joints != fingers:
                raise ValueError(f"{side}: gripper joints differ from Scene")
        joints = {j for u in robot.units.values() for j in u.sim_joints}
        joints |= {j for g in robot.grippers.values() for j in g.sim_joints}
        self.qadr = {j: int(m.joint(j).qposadr[0]) for j in joints}
        self.vadr = {j: int(m.joint(j).dofadr[0]) for j in joints}
        for u in robot.units.values():
            u.check_model_ranges([m.joint(j).range for j in u.sim_joints], "MuJoCo")
        self.body_names = [m.body(i).name for i in range(m.nbody)]
        self.robot_bodies = self.scene.robot_bodies
        self.renderer = None
        cams = {m.camera(i).name for i in range(m.ncam)}
        self.policy = tuple(
            c.sim_name for c in robot.cameras.values() if c.sim_name in cams
        )
        self.display = tuple(sorted(cams - set(self.policy)))
        self.sites = {s: m.site(s + "_grasp").id for s in ("left", "right")}

    def capabilities(self):
        return Capabilities(
            joint_position_targets=True,
            joint_velocity_feedback=True,
            joint_effort=EFFORT_SIMULATED,
            rgb_cameras=self.policy + self.display,
            contact_query=True,
            penetration_audit=True,
            loop_modes=("framework", "simulator"),
            render_thread="physics thread (one shared render context, created there)",
            notes=(
                "effort = actuator force + gravity feedforward (qfrc_applied)",
                "finger actuators kp 1500, force range +-6 N (Scene override)",
            ),
        )

    @property
    def dt(self):
        return float(self.m.opt.timestep)

    def reset(self):
        raise NotImplementedError("Scene is built fresh for every session")

    def apply(self, targets):
        s = self.scene
        for side, unit in self.robot.arms.items():
            s.goal[side] = np.array([targets[j] for j in unit.sim_joints])
        for side, g in self.robot.grippers.items():
            # Scene drives both fingers from one closure fraction; the mapping
            # layer sends mirrored targets, so this inversion is exact.
            s.grip[side] = g.closure([targets[j] for j in g.sim_joints])

    def step(self):
        try:
            self.scene.physics_step()
        except RuntimeError as exc:
            text = str(exc)
            kind = "penetration" if text.startswith("Penetration") else "joint_limit"
            raise PhysicsFault(kind, text, step=self.scene.step_number) from exc

    def read(self):
        m, d = self.m, self.d
        effort = d.qfrc_actuator + d.qfrc_applied
        contacts = []
        for c in d.contact[: d.ncon]:
            a, b = (int(m.geom_bodyid[g]) for g in c.geom)
            contacts.append(
                {
                    "a": self.body_names[a],
                    "b": self.body_names[b],
                    "a_robot": a in self.robot_bodies,
                    "b_robot": b in self.robot_bodies,
                    "depth": max(0.0, -float(c.dist)),
                }
            )
        return BackendState(
            sim_time=float(d.time),
            step=self.scene.step_number,
            q={j: float(d.qpos[a]) for j, a in self.qadr.items()},
            dq={j: float(d.qvel[a]) for j, a in self.vadr.items()},
            effort={j: float(effort[a]) for j, a in self.vadr.items()},
            effort_kind=EFFORT_SIMULATED,
            contacts=contacts,
            max_penetration_m=float(self.scene.maxdepth),
        )

    def render(self, camera, width, height):
        if camera not in self.policy + self.display:
            raise KeyError(f"no MuJoCo camera {camera!r}")
        if self.renderer is None:
            self.renderer = SharedRenderer(self.m)
        # Scene.physics_step only runs mj_forward on trace ticks; camera poses
        # must match the state that was just stepped.
        mujoco.mj_forward(self.m, self.d)
        return self.renderer.render(self.d, self.m.camera(camera).id, width, height)

    def task_state(self):
        return {side: self.scene.object_info(side) for side in ("left", "right")}

    def tracked_frames(self):
        mujoco.mj_kinematics(self.m, self.d)
        frames = {
            f"{s}_grasp": self.d.site_xpos[i].tolist() for s, i in self.sites.items()
        }
        for cam in self.policy:
            frames[cam] = self.d.site_xpos[self.m.site(cam).id].tolist()
        return frames

    def run_owned_loop(self, hooks):
        """MuJoCo stands in for an engine that owns its loop (like a viewer or
        Isaac Sim's app loop): it paces itself and calls the bridge's hooks."""
        tick = time.monotonic()
        while not hooks.should_stop():
            if hooks.halted():
                hooks.idle()
                continue
            hooks.pre_step()
            try:
                self.step()
            except PhysicsFault as exc:
                hooks.on_fault(exc)
            hooks.post_step()
            tick += self.dt
            delay = tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:
                tick = time.monotonic()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.scene.close()
