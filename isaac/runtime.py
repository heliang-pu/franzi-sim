"""Run a stage built by isaac/stage.py in Isaac Sim 5.1, with the robot driven as
sim/episode.py drives it in MuJoCo.

start_app() must run before anything imports pxr or omni (isaac.stage
included). Runtime opens folder/scene.usdc and steps PhysX at MuJoCo's 500 Hz,
TGS on the CPU:
- joint values are MuJoCo's (qpos); PhysX counts them from qpos0;
- before every physics step the gripper linkage's coupled joints get their
  targets from the measured finger slides, through MuJoCo's polynomials;
- the robot's links ignore gravity (stage.py), so it needs no feedforward;
- render() renders the cameras it was given and returns their RGB images.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# The app settings the speed check rendered with on the pro6000 (driver 580,
# 2026-10-24): one GPU, since a stale Vulkan ICD lists it twice. The longer set
# it borrowed from isaac-language-grasp (DLSS off, no viewport window, the RTX
# data window) gave no camera frames in 900 s here.
#
# Zero frame delay, as Isaac's isaacsim.exp.base.zero_delay.kit sets it: each
# update waits for its render, so one render delivers the frame of the current
# state. Without it the frame arrived two to four renders later, every one of
# them redrawing all cameras (render probes 13 and 15, 2026-10-25: 1.00 renders
# per frame against 2.94, frames as fresh).
#
# The headless app's default viewport (1280x720) renders every update unless
# its updates are off, and nothing reads it. Off, tray_push seed 0 ran 9 %
# faster beside the 200-episode batch, with the physics the same bit for bit,
# the frames as close to a run with it on as two such runs are to each other
# (1.4 grey levels on average: RTX differs from run to run) and the frame
# check passed (offset 0, ratio 4.3; 2026-10-27).
APP_CONFIG = {
    "headless": True,
    "renderer": "RaytracedLighting",
    "multi_gpu": False,
    "disable_viewport_updates": True,
    "extra_args": [
        "--/app/hydraEngine/waitIdle=true",
        "--/app/updateOrder/checkForHydraRenderComplete=1000",
    ],
}
# The first renders after a driver or shader change compile RTX shaders for
# minutes and hand out empty frames meanwhile.
PRIME_TIMEOUT_S = 900
# Extra renders of the same state before render() returns: frames as settled as
# a still scene's, for checking a recording against (isaac.episode --settle).
SETTLE = 0


def start_app():
    sys.argv = sys.argv[:1]
    from isaacsim import SimulationApp

    app = SimulationApp(dict(APP_CONFIG))
    import carb

    # Partial opacity (glass, MuJoCo's rgba alpha) renders only with this on;
    # otherwise the CNC door's window is opaque.
    carb.settings.get_settings().set_bool(
        "/rtx/raytracing/fractionalCutoutOpacity", True
    )
    return app


def log(message):
    print(f"[isaac] {message}", flush=True)


def polynomial(coef, x):
    return sum(c * x**k for k, c in enumerate(coef))


class Runtime:
    def __init__(self, folder, views=(), size=(640, 480)):
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation, RigidPrim
        from isaacsim.sensors.camera import Camera
        from omni.physx import get_physx_simulation_interface
        from pxr import UsdGeom

        self.folder = Path(folder)
        self.model = json.loads((self.folder / "model.json").read_text())
        self.paths = json.loads((self.folder / "paths.json").read_text())
        context = omni.usd.get_context()
        context.open_stage(str(self.folder / "scene.usdc"))
        app = omni.kit.app.get_app()
        for _ in range(5):
            app.update()
        self.stage = context.get_stage()
        self.dt = float(self.paths["timestep"])
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.dt,
            rendering_dt=self.dt,
            physics_prim_path=self.paths["physics_scene"],
            backend="numpy",
            device="cpu",
        )
        physics = self.world.get_physics_context()
        physics.enable_gpu_dynamics(False)
        physics.set_solver_type("TGS")
        self.robot = Articulation(prim_paths_expr=self.paths["robot"], name="franzi")
        self.props = {
            name: Articulation(prim_paths_expr=path, name=f"prop_{k}")
            for k, (name, path) in enumerate(self.paths["articulations"].items())
            if path != self.paths["robot"]
        }
        self.free_names = list(self.paths["free_bodies"])
        self.free = (
            RigidPrim(
                prim_paths_expr=list(self.paths["free_bodies"].values()), name="free"
            )
            if self.free_names
            else None
        )
        self.size = tuple(size)
        self.cameras = {}
        for k, name in enumerate(views):
            path = self.paths["cameras"][name]
            camera = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
            vertical = camera.GetVerticalApertureAttr().Get()
            camera.GetHorizontalApertureAttr().Set(float(vertical * size[0] / size[1]))
            product = rep.create.render_product(path, self.size, force_new=True)
            self.cameras[name] = Camera(
                prim_path=path,
                name=f"camera_{k}",
                resolution=self.size,
                render_product_path=product.path,
            )
        self.world.reset()
        self.robot.initialize()
        for view in self.props.values():
            view.initialize()
        if self.free is not None:
            self.free.initialize()
        for camera in self.cameras.values():
            camera.initialize()
            # Keep every frame. By default a Camera keeps one only once the
            # simulation time has moved a rendering step past the last one it
            # kept, so it skipped frames rendered at an unchanged time (render()
            # after a sync step, the settle renders).
            camera._frequency = -1
        self.physx = get_physx_simulation_interface()
        self.setup_robot()
        # The jointed props' joints by MuJoCo name: (view, dof index, qpos0).
        prims = {dof["prim"]: (name, dof) for name, dof in self.paths["dofs"].items()}
        self.prop_dofs = {}
        for view in self.props.values():
            for i, prim in enumerate(view.dof_names):
                name, dof = prims[prim]
                self.prop_dofs[name] = (view, i, dof["qpos0"])
        self.steps = 0
        self.deepest = (0.0, None)
        self.contacts, self.contact_forces = [], []
        self.collider_info = {}

    def setup_robot(self):
        names = list(self.robot.dof_names)
        dofs = {k: v for k, v in self.paths["dofs"].items() if v["robot"]}
        self.index = {k: names.index(v["prim"]) for k, v in dofs.items()}
        n = len(names)
        self.qpos0 = np.zeros(n)
        kp, kd, effort = np.zeros(n), np.zeros(n), np.full(n, 1e6)
        for name, dof in dofs.items():
            i = self.index[name]
            self.qpos0[i] = dof["qpos0"]
            kp[i], kd[i] = dof["stiffness"], dof["damping"]
            if dof["max_force"] is not None:
                effort[i] = dof["max_force"]
        # The gains PhysX parsed from the USD drives, before they are set in SI:
        # a check of stage.py's per-degree units.
        parsed = self.robot.get_gains()
        self.parsed_gains = (np.asarray(parsed[0])[0], np.asarray(parsed[1])[0])
        self.gains = (kp, kd)
        self.robot.set_gains(kps=kp[None], kds=kd[None])
        self.robot.set_max_efforts(effort[None])
        self.coupled = [
            (self.index[name], self.index[dof["leader"]], dof["polycoef"])
            for name, dof in dofs.items()
            if "leader" in dof
        ]
        self.target = np.zeros(n)

    def set_targets(self, values):
        """Joint targets by MuJoCo joint name, in MuJoCo's values."""
        for name, value in values.items():
            i = self.index[name]
            self.target[i] = value - self.qpos0[i]

    def set_positions(self, values):
        """Teleport the robot's joints (MuJoCo values) and hold them there; only
        for checks, never during an episode."""
        q = np.asarray(self.robot.get_joint_positions())[0].copy()
        for name, value in values.items():
            if name in self.index:
                i = self.index[name]
                q[i] = value - self.qpos0[i]
        self.target = q.copy()
        self.robot.set_joint_positions(q[None])
        self.robot.set_joint_velocities(np.zeros_like(q)[None])
        self.robot.set_joint_position_targets(q[None])

    def step(self, sync=False):
        """One physics step. sync=True takes it through Kit's update loop,
        which renders as well: the engine gives it to the last step of each
        tick before the recording, to warm RTX's denoiser and caches."""
        q = np.asarray(self.robot.get_joint_positions())[0]
        for follower, leader, coef in self.coupled:
            self.target[follower] = polynomial(coef, q[leader])
        self.robot.set_joint_position_targets(self.target[None])
        # With rendering_dt equal to physics_dt, step(render=True) is one Kit
        # update that steps the physics once.
        self.world.step(render=sync)
        self.steps += 1
        self.read_contacts()

    def collider(self, handle):
        """(MuJoCo body id, geom name) of a contact report's collider; body id
        None for a collider stage.py did not make."""
        info = self.collider_info.get(handle)
        if info is None:
            from pxr import PhysicsSchemaTools

            path = str(PhysicsSchemaTools.intToSdfPath(handle))
            body, name = self.paths["colliders"].get(path, (None, path))
            info = self.collider_info[handle] = (body, name)
        return info

    def read_contacts(self):
        """This step's contact pairs as (collider, collider, depth), depth the
        deepest point's penetration (negative while apart: PhysX reports pairs
        within the 4 mm contact offset), and in contact_forces each pair's
        summed normal force (N, from the points' impulses)."""
        headers, data = self.physx.get_contact_report()
        self.contacts, self.contact_forces = [], []
        for h in headers:
            if not h.num_contact_data:  # a pair that just lost contact
                continue
            points = [
                data[k]
                for k in range(
                    h.contact_data_offset, h.contact_data_offset + h.num_contact_data
                )
            ]
            depth = max(-float(p.separation) for p in points)
            impulse = sum(
                abs(
                    p.impulse.x * p.normal.x
                    + p.impulse.y * p.normal.y
                    + p.impulse.z * p.normal.z
                )
                for p in points
            )
            pair = (self.collider(int(h.collider0)), self.collider(int(h.collider1)))
            self.contacts.append((*pair, depth))
            self.contact_forces.append((*pair, impulse / self.dt))
            if depth > self.deepest[0]:
                self.deepest = (depth, pair)

    def deepest_contact(self):
        depth, pair = self.deepest
        return depth, [name for _, name in pair] if pair else None

    def joint_positions(self):
        """The robot's joints by MuJoCo name, in MuJoCo's values."""
        q = np.asarray(self.robot.get_joint_positions())[0]
        return {name: float(q[i] + self.qpos0[i]) for name, i in self.index.items()}

    def joint_velocities(self):
        qd = np.asarray(self.robot.get_joint_velocities())[0]
        return {name: float(qd[i]) for name, i in self.index.items()}

    def joint_efforts(self):
        """The robot's joint forces as PhysX measures them (projected on each
        joint, N m or N), by MuJoCo name. The links ignore gravity, so these
        hold no gravity load of the robot's own."""
        tau = np.asarray(self.robot.get_measured_joint_efforts())[0]
        return {name: float(tau[i]) for name, i in self.index.items()}

    def free_positions(self):
        if self.free is None:
            return {}
        positions = np.asarray(self.free.get_world_poses()[0])
        return dict(zip(self.free_names, positions))

    def free_state(self):
        """Each free body's (position, (w, x, y, z), linear and angular velocity
        in the world frame)."""
        if self.free is None:
            return {}
        positions, quats = (np.asarray(a) for a in self.free.get_world_poses())
        velocities = np.asarray(self.free.get_velocities())
        return {
            name: (positions[k], quats[k], velocities[k, :3], velocities[k, 3:])
            for k, name in enumerate(self.free_names)
        }

    def prop_joints(self):
        return {
            name: np.asarray(view.get_joint_positions())[0]
            for name, view in self.props.items()
        }

    def prop_state(self):
        """The jointed props' joints by MuJoCo name: (MuJoCo value, velocity)."""
        q = {k: np.asarray(v.get_joint_positions())[0] for k, v in self.props.items()}
        qd = {k: np.asarray(v.get_joint_velocities())[0] for k, v in self.props.items()}
        names = {id(v): k for k, v in self.props.items()}
        out = {}
        for name, (view, i, qpos0) in self.prop_dofs.items():
            key = names[id(view)]
            out[name] = (float(q[key][i] + qpos0), float(qd[key][i]))
        return out

    def update_kinematics(self):
        """Link poses from the joint positions just set, without a dynamics step."""
        view = self.world.physics_sim_view
        if hasattr(view, "update_articulations_kinematic"):
            view.update_articulations_kinematic()
        else:
            self.world.step(render=False)

    def link_poses(self):
        """World poses of the robot's rigid bodies by prim name, as (p, (w, x, y, z))."""
        view = self.robot._physics_view
        names = view.shared_metatype.link_names
        transforms = np.asarray(view.get_link_transforms())[0]
        return {
            name: (t[:3], np.r_[t[6], t[3:6]])  # PhysX quaternions are (x, y, z, w)
            for name, t in zip(names, transforms)
        }

    def render(self):
        """RGB frames of the cameras for the current physics state, waiting
        while RTX compiles its shaders. Each render is a Kit update, which
        takes the stepped state to the renderer. It renders until every
        camera's latest frame was rendered at the current simulation time (a
        frame of an earlier state carries an earlier time): once with
        APP_CONFIG's zero frame delay, two to four times without it, where a
        fixed two renders gave frames one control tick (40 ms) late at random
        (lag probes 10-12, 2026-10-25)."""
        now = self.world.current_time
        start = last = time.perf_counter()
        while True:
            self.world.render()
            frames = {n: c.get_current_frame() for n, c in self.cameras.items()}
            images = {n: np.asarray(f["rgb"]) for n, f in frames.items()}
            if all(
                frames[n]["rendering_time"] > now - self.dt / 2
                and f.ndim == 3
                and f.shape[:2] == self.size[::-1]
                for n, f in images.items()
            ):
                for _ in range(SETTLE):
                    self.world.render()
                if SETTLE:
                    images = {
                        n: np.asarray(c.get_current_frame()["rgb"])
                        for n, c in self.cameras.items()
                    }
                return {n: f[..., :3].astype(np.uint8) for n, f in images.items()}
            clock = time.perf_counter()
            if clock - last > 30:
                shapes = {n: f.shape for n, f in images.items()}
                times = {n: f["rendering_time"] for n, f in frames.items()}
                log(
                    f"waiting for camera frames at {now:.3f} s "
                    f"({clock - start:.0f} s): {shapes} {times}"
                )
                last = clock
            if clock - start > PRIME_TIMEOUT_S:
                raise RuntimeError(f"no camera frames after {PRIME_TIMEOUT_S} s")
