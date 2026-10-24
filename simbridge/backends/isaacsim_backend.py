"""Isaac Sim adapter. DRAFT: NEVER EXECUTED.

Written on macOS arm64, where Isaac Sim cannot run, against the documented
Isaac Sim 4.5 / 5.x standalone Python API (``isaacsim.*`` namespaces). No line
of this file has run; treat every engine call as an assumption until the
bring-up checks pass on a Linux + NVIDIA workstation. Isaac Sim 6.x is not
handled (new URDFImporter class, isaacsim.core.prims moved to "deprecated").

Scene source: ``assets/derived/pybullet/{franzi_pybullet.urdf, scene.json}``,
written by sdk/convert_pybullet_assets.py from the compiled MuJoCo scene: MuJoCo's
colliders (boxes, 49 convex hulls per wrist, convex finger hulls), contact-pair
matrix, friction, actuator gains and force limits, cameras and grasp sites. The
source URDF is NOT imported: its collisions are whole CAD meshes, which PhysX
would turn into one hull per link (filling the jaw opening), so contacts could
not be compared with MuJoCo. The derived URDF holds absolute mesh paths of the
checkout that generated it; load() rewrites that prefix into
``<output>/franzi_isaacsim.urdf`` (the meshes must exist in this checkout).

Assumptions, in the order a first bring-up should check them:
 1. SimulationApp (headless) is created in load(), on the main thread, before
    any omni/isaacsim import. app.close() is deferred to atexit because it can
    end the interpreter before server.py writes server_result.json.
 2. URDF import via _urdf.ImportConfig + "URDFParseAndImportFile":
    merge_fixed_joints=False keeps the sensor links as camera parents,
    convex_decomp=False gives one convex hull per collision mesh (as MuJoCo),
    self_collision=True plus USD collision groups rebuilt from scene.json give
    MuJoCo's pair matrix. Links are looked up by name anywhere under the
    returned prim, so a nested 5.x asset layout is tolerated.
 3. PhysX cooks at most 255 hull vertices (hullVertexLimit raised from 64): 8
    wrist hulls have 373-1740 vertices and are approximated. CPU pipeline only;
    GPU convexes are limited to 64 vertices.
 4. Drives through isaacsim.core.prims.Articulation in SI units per radian.
    MuJoCo's passive joint damping is folded into drive damping, so PhysX's
    force clamp (+-6 N fingers) also bounds the damping term. URDF velocity
    limits (3.14) become PhysX max joint velocities, which MuJoCo lacks.
 5. Arm gravity + Coriolis feedforward through set_joint_efforts, assumed to be
    added to the drive force (MuJoCo: qfrc_applied = qfrc_bias).
 6. Penetration audit from the PhysX contact report (PhysxContactReportAPI,
    threshold 0). Each point's ``separation`` comes from the narrow phase of the
    step that just ran (pre-solve, like MuJoCo's contact.dist after mj_step).
    PhysX cannot report post-solve depth, pairs that produced no contact point
    (filtered, tunnelled, beyond contactOffset), or depth deeper than its
    manifold points. Points within contactOffset have positive separation, so
    "touching" uses options.touch_tolerance_m.
 7. Cameras are USD camera prims authored under the URDF sensor links with the
    camera_rig convention (scene.json axes_in_link; USD and MuJoCo cameras both
    look along -Z with +Y up), read by isaacsim.sensors.camera.Camera after
    world.render(). Whether that frame already shows the current state
    (annotator latency) is unverified; see options.render_updates.
 8. Owned loop: rendering_dt == physics_dt, so every world.step() is exactly
    one physics step; a SimulationManager PRE_PHYSICS_STEP callback carries
    hooks.pre_step and the loop fails fast if it does not fire once per step.
"""

import atexit
import json
import re
import time
from pathlib import Path

import numpy as np

from simbridge import ROOT
from simbridge.backends.base import (
    EFFORT_SIMULATED,
    BackendState,
    Capabilities,
    SimBackend,
)
from simbridge.status import PhysicsFault

DERIVED = ROOT / "assets/derived/pybullet"
SENSOR_LINKS = (
    "head_d435_Link",
    "body_d435_Link",
    "left_wrist_d405_Link",
    "right_wrist_d405_Link",
    "MID360_Link",
    "2Dlidar_Link",
)


def _quat_wxyz(R):
    """Rotation matrix (columns = child axes in the parent) -> unit (w, x, y, z)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    k = int(np.argmax([t, R[0, 0], R[1, 1], R[2, 2]]))
    if k == 0:
        s = 2 * np.sqrt(1 + t)
        q = [
            s / 4,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ]
    elif k == 1:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [
            (R[2, 1] - R[1, 2]) / s,
            s / 4,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        ]
    elif k == 2:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            s / 4,
            (R[1, 2] + R[2, 1]) / s,
        ]
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            s / 4,
        ]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def _matrix_xyzw(q):
    """Unit quaternion (x, y, z, w), the tensor-API order -> rotation matrix."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rewrite_urdf(text, generated_urdf, staging):
    """Point the derived URDF's absolute mesh paths at this checkout and make
    every name a legal USD prim name. A link or mesh file starting with a digit
    (2Dlidar_Link) makes the importer fail the whole import ("Used null prim",
    seen with the source project's Isaac URDF converter), so such links
    and joints get a "lidar_" prefix and such meshes a prefixed symlink."""
    old = str(generated_urdf).rsplit("/assets/derived/", 1)[0]
    text = text.replace(old + "/", str(ROOT) + "/")
    names = set(re.findall(r'<(?:link|joint)\s+name="([^"]+)"', text))
    renamed = {n: f"lidar_{n}" for n in names if n[0].isdigit()}
    for old_name, new_name in renamed.items():
        text = re.sub(
            rf'(name|link)="{re.escape(old_name)}"', rf'\1="{new_name}"', text
        )
    for ref in sorted(set(re.findall(r'filename="([^"]+)"', text))):
        if Path(ref).name[0].isdigit():
            alias = staging / f"lidar_{Path(ref).name}"
            if not alias.exists():
                alias.symlink_to(ref)
            text = text.replace(f'filename="{ref}"', f'filename="{alias}"')
    return text, renamed


def _prim_name(name):
    return "".join(c if c.isalnum() else "_" for c in name)


class IsaacSimBackend(SimBackend):
    name = "isaacsim"
    label = "Isaac Sim 4.5/5.x (draft, never executed)"
    implementation_status = "draft (never executed)"

    def load(self, robot, options, output):
        self.robot, self.options, self.output = robot, options, output
        self.scene = json.loads((DERIVED / "scene.json").read_text())
        from isaacsim import SimulationApp  # must precede every other omni import

        self.app = SimulationApp(
            {
                "headless": bool(options.get("headless", True)),
                "renderer": options.get("renderer", "RayTracedLighting"),
            }
        )
        atexit.register(self.app.close)
        from isaacsim.core.api import World

        self._dt = float(self.scene["physics"]["timestep_s"])
        # rendering_dt == physics_dt: one world.step() is one physics step.
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self._dt,
            rendering_dt=self._dt,
            backend="numpy",
            device="cpu",
        )
        ctx = self.world.get_physics_context()
        ctx.enable_gpu_dynamics(False)
        ctx.set_solver_type("TGS")
        ctx.set_gravity(float(self.scene["physics"]["gravity"][2]))
        self.stage = self.world.stage
        self._import_robot()
        self._author_world()
        self._author_physics()
        self._author_cameras()
        from isaacsim.core.prims import Articulation
        from isaacsim.sensors.camera import Camera

        self.view = Articulation(prim_paths_expr=self.root_path, name="franzi")
        # No pose arguments: the wrapper keeps the pose authored in _author_cameras.
        self.cams = {
            n: (Camera(prim_path=p, name=f"cam_{n}", resolution=r), r)
            for n, (p, r) in self.cam_paths.items()
        }
        self.world.reset()  # starts the timeline; runs one unaudited warm-up physics step
        self.view.initialize()
        for cam, _ in self.cams.values():
            cam.initialize()
        self._configure_drives()
        self._step, self._rendered_step, self._contacts = 0, -1, []
        self.maxdepth, self.deepest, self.maxlimit = 0.0, None, 0.0

    # ----- scene construction (USD, before the timeline starts) --------------
    def _import_robot(self):
        import omni.kit.commands
        from isaacsim.asset.importer.urdf import _urdf
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        urdf = self.output / "franzi_isaacsim.urdf"
        text, renamed = _rewrite_urdf(
            (DERIVED / "franzi_pybullet.urdf").read_text(),
            self.scene["urdf"],
            self.output,
        )
        urdf.write_text(text)
        self.sensor_links = tuple(renamed.get(n, n) for n in SENSOR_LINKS)
        cfg = _urdf.ImportConfig()
        cfg.merge_fixed_joints = False
        cfg.fix_base = True
        cfg.self_collision = True
        cfg.convex_decomp = False
        cfg.import_inertia_tensor = True
        cfg.distance_scale = 1.0
        cfg.make_default_prim = False
        cfg.create_physics_scene = False
        cfg.set_default_drive_type(1)  # UrdfJointTargetType.JOINT_DRIVE_POSITION
        ok, path = omni.kit.commands.execute(
            "URDFParseAndImportFile", urdf_path=str(urdf), import_config=cfg
        )
        if not ok or not path:
            raise RuntimeError(f"URDF import failed for {urdf}")
        self.robot_path = str(path)
        prim = self.stage.GetPrimAtPath(self.robot_path)
        base = self.scene["base"]["position"]
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*base))
        self.links, self.root_path = {}, None
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                self.links[p.GetName()] = str(p.GetPath())
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                self.root_path = str(p.GetPath())
            # MuJoCo gains are forces; a PhysX "acceleration" drive would scale
            # them by the joint inertia.
            for kind in ("angular", "linear"):
                if p.HasAPI(UsdPhysics.DriveAPI, kind):
                    UsdPhysics.DriveAPI(p, kind).CreateTypeAttr().Set("force")
            # Fixed root joint: body0 empty means frame 0 is in world coordinates,
            # so it must follow the base height or PhysX pulls base_link to z=0.
            if (
                p.IsA(UsdPhysics.FixedJoint)
                and not UsdPhysics.Joint(p).GetBody0Rel().GetTargets()
            ):
                UsdPhysics.Joint(p).GetLocalPos0Attr().Set(Gf.Vec3f(*base))
        missing = (set(self.scene["robot_links"]) | set(self.sensor_links)) - set(
            self.links
        )
        if missing or self.root_path is None:
            raise RuntimeError(
                f"imported robot lacks links {sorted(missing)} or an articulation root"
            )

    def _author_world(self):
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid

        self.paths = {name: path for name, path in self.links.items()}
        for name, g in self.scene["world_geoms"].items():
            shape = g.get(
                "realised_as", g
            )  # the floor plane is realised as a box below z=0
            path = f"/World/scene/{_prim_name(name)}"
            FixedCuboid(
                path,
                name=_prim_name(name),
                position=np.add(g["pos"], shape.get("offset", 0.0)),
                orientation=np.roll(g["quat_xyzw"], 1),
                size=1.0,
                scale=2 * np.asarray(shape["half_extents"]),
                color=np.asarray(g["rgba"][:3]),
            )
            self.paths[name] = path
        self.parts = {}
        for name, b in self.scene["free_bodies"].items():
            path = f"/World/scene/{name}"
            self.parts[name.split("_")[0]] = DynamicCuboid(
                path,
                name=name,
                position=np.asarray(b["position"]),
                orientation=np.roll(b["orientation_xyzw"], 1),
                size=1.0,
                scale=2 * np.asarray(b["geom"]["half_extents"]),
                mass=float(b["mass_kg"]),
                color=np.asarray(b["geom"]["rgba"][:3]),
            )
            self.paths[name] = path
        self.names = {path: name for name, path in self.paths.items()}

    def _author_physics(self):
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

        opt, stage = self.options, self.stage
        for p in Usd.PrimRange(stage.GetPrimAtPath(self.robot_path)):
            if p.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Set("convexHull")
                api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(p)
                api.CreateHullVertexLimitAttr().Set(
                    int(opt.get("hull_vertex_limit", 255))
                )
        art = PhysxSchema.PhysxArticulationAPI.Apply(
            stage.GetPrimAtPath(self.root_path)
        )
        art.CreateEnabledSelfCollisionsAttr().Set(True)
        art.CreateSolverPositionIterationCountAttr().Set(
            int(opt.get("solver_position_iterations", 64))
        )
        art.CreateSolverVelocityIterationCountAttr().Set(
            int(opt.get("solver_velocity_iterations", 4))
        )
        mass = opt.get(
            "sensor_link_mass_kg", 0.001
        )  # MuJoCo drops these URDF inertials
        if mass is not None:
            for link in self.sensor_links:
                api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(self.links[link]))
                api.CreateMassAttr().Set(float(mass))
                api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1e-7, 1e-7, 1e-7))
        bodies = [self.links[n] for n in self.scene["robot_links"]] + [
            self.paths[n] for n in self.scene["free_bodies"]
        ]
        for path in bodies:
            report = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path))
            report.CreateThresholdAttr().Set(0.0)
        for name in self.scene["free_bodies"]:
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(
                stage.GetPrimAtPath(self.paths[name])
            )
            rb.CreateSolverPositionIterationCountAttr().Set(
                int(opt.get("solver_position_iterations", 64))
            )
        # MuJoCo combines sliding friction with max(); torsional/rolling terms are not mapped.
        materials = {}
        frictions = {
            n: v["mujoco_friction"][0] for n, v in self.scene["robot_links"].items()
        }
        frictions |= {
            n: v["mujoco_friction"][0] for n, v in self.scene["world_geoms"].items()
        }
        frictions |= {
            n: v["geom"]["mujoco_friction"][0]
            for n, v in self.scene["free_bodies"].items()
        }
        for name, mu in frictions.items():
            if mu not in materials:
                mat = UsdShade.Material.Define(
                    stage, f"/World/physics_materials/mu_{len(materials)}"
                )
                api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                api.CreateStaticFrictionAttr().Set(float(mu))
                api.CreateDynamicFrictionAttr().Set(float(mu))
                api.CreateRestitutionAttr().Set(0.0)
                PhysxSchema.PhysxMaterialAPI.Apply(
                    mat.GetPrim()
                ).CreateFrictionCombineModeAttr().Set("max")
                materials[mu] = mat
            binding = UsdShade.MaterialBindingAPI.Apply(
                stage.GetPrimAtPath(self.paths[name])
            )
            binding.Bind(
                materials[mu], UsdShade.Tokens.weakerThanDescendants, "physics"
            )
        # MuJoCo's contype/conaffinity matrix as USD collision groups; a group
        # filters every class it does not collide with (itself included).
        classes = self.scene["collision_filter"]["classes"]
        for c in classes:
            group = UsdPhysics.CollisionGroup.Define(
                stage, f"/World/collision_groups/{c['name']}"
            )
            includes = group.GetCollidersCollectionAPI().CreateIncludesRel()
            for member in c["members"]:
                includes.AddTarget(Sdf.Path(self.paths[member]))
            filtered = group.CreateFilteredGroupsRel()
            for other in classes:
                if other["name"] not in c["collides_with"]:
                    filtered.AddTarget(
                        Sdf.Path(f"/World/collision_groups/{other['name']}")
                    )

    def _author_cameras(self):
        from pxr import Gf, UsdGeom

        sizes = {c.sim_name: (c.width, c.height) for c in self.robot.cameras.values()}
        sizes[self.options.get("display_camera", "overview")] = tuple(
            self.options.get("display_size", (960, 720))
        )
        znear = float(
            self.scene["render"]["znear_m"]
        )  # 45 mm: past each camera's own housing
        self.cam_paths = {}
        for name, (w, h) in sizes.items():
            spec = self.scene["cameras"][name]
            if spec["role"] == "policy":
                parent, pos, axes = (
                    self.links[spec["urdf_link"]],
                    spec["pos_in_link"],
                    spec["axes_in_link"],
                )
            else:
                parent, pos, axes = (
                    "/World",
                    spec["pos_in_parent"],
                    spec["axes_in_parent"],
                )
            path = f"{parent}/simbridge_{name}"
            cam = UsdGeom.Camera.Define(self.stage, path)
            xf = UsdGeom.Xformable(cam)
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(*_quat_wxyz(axes))
            )
            # Pinhole from fovy with square pixels: only aperture/focal ratios matter.
            focal = 10.0
            vertical = 2 * focal * np.tan(np.deg2rad(spec["fovy_deg"]) / 2)
            cam.CreateFocalLengthAttr().Set(focal)
            cam.CreateVerticalApertureAttr().Set(float(vertical))
            cam.CreateHorizontalApertureAttr().Set(float(vertical * w / h))
            cam.CreateClippingRangeAttr().Set(Gf.Vec2f(znear, 100.0))
            self.cam_paths[name] = (path, (w, h))

    def _configure_drives(self):
        v = self.view
        names = list(v.dof_names)
        self.dof = {j: i for i, j in enumerate(names)}
        acts = self.scene["actuators"]
        if set(acts) != set(names):
            raise RuntimeError(
                f"DOFs differ from MuJoCo actuators: {sorted(set(acts) ^ set(names))}"
            )
        n = len(names)
        kp, kd, fmax, armature = (np.zeros(n) for _ in range(4))
        for j, a in acts.items():
            i = self.dof[j]
            if a["kind"] == "position":
                kp[i], kd[i] = a["gain"], a["mujoco_joint_damping"]
            else:  # MuJoCo velocity actuator (wheels, target 0): kv plus passive damping
                kd[i] = a["gain"] + a["mujoco_joint_damping"]
            fmax[i], armature[i] = a["force_limit"], a["mujoco_armature"]
        v.set_gains(kps=kp[None], kds=kd[None])
        v.set_max_efforts(fmax[None])
        v.set_armatures(armature[None])
        limits = np.asarray(v.get_dof_limits())[0]
        self.lo, self.hi = limits[:, 0], limits[:, 1]
        for u in self.robot.units.values():
            for j, (lo, hi) in zip(u.sim_joints, u.sim_limits):
                if not np.allclose(limits[self.dof[j]], [lo, hi], atol=1e-4):
                    raise ValueError(
                        f"{j}: PhysX limits {limits[self.dof[j]]} differ from the URDF"
                    )
        self.arm_idx = np.array(
            [self.dof[j] for u in self.robot.arms.values() for j in u.sim_joints]
        )
        self.head_pitch = self.dof["head_pitch_joint"]
        self.targets = np.zeros(
            n
        )  # legs, waist, head yaw, steering held at 0 (upright)
        gravity = np.asarray(v.get_generalized_gravity_forces())
        if gravity.shape[-1] != n:
            raise RuntimeError(
                f"gravity forces have {gravity.shape[-1]} entries for {n} DOFs (base not fixed?)"
            )

    # ----- SimBackend ----------------------------------------------------------
    def capabilities(self):
        return Capabilities(
            joint_position_targets=True,
            joint_velocity_feedback=True,
            joint_effort=EFFORT_SIMULATED,
            rgb_cameras=tuple(self.cams),
            contact_query=True,
            penetration_audit=True,
            loop_modes=("framework", "simulator"),
            render_thread="main thread only (SimulationApp/Kit), never inside a physics callback",
            notes=(
                "DRAFT: never executed; every capability is an untested implementation attempt",
                "effort = PhysX measured joint effort (drive + feedforward + limit terms)",
                "finger drives kp 1500, max force 6 N; MuJoCo passive damping folded into drive damping",
                "contacts are one entry per body pair (deepest point), touching if separation <= tolerance",
            ),
        )

    @property
    def dt(self):
        return self._dt

    def reset(self):
        raise NotImplementedError("the stage is built fresh for every session")

    def apply(self, targets):
        for joint, value in targets.items():
            self.targets[self.dof[joint]] = value

    def _before_physics(self):
        # Head gaze ramp as in sim.episode.Scene: the joint moves, never the camera.
        x = min(self._step * self._dt / 1.5, 1.0)
        self.targets[self.head_pitch] = 0.50 * x * x * (3 - 2 * x)
        v = self.view
        v.set_joint_position_targets(self.targets[None])
        bias = np.asarray(v.get_generalized_gravity_forces())[0]
        bias = bias + np.asarray(v.get_coriolis_and_centrifugal_forces())[0]
        effort = np.zeros_like(self.targets)
        effort[self.arm_idx] = bias[self.arm_idx]
        v.set_joint_efforts(effort[None])

    def _after_physics(self):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        self._step += 1
        headers, data = get_physx_simulation_interface().get_contact_report()
        tolerance = float(self.options.get("touch_tolerance_m", 0.0005))
        contacts, deepest = [], (0.0, None)
        for h in headers:
            if h.num_contact_data == 0:  # CONTACT_LOST events carry no points
                continue
            sep = min(
                data[i].separation
                for i in range(
                    h.contact_data_offset, h.contact_data_offset + h.num_contact_data
                )
            )
            a, b = (
                str(PhysicsSchemaTools.intToSdfPath(x)) for x in (h.actor0, h.actor1)
            )
            pair = [self.names.get(a, a), self.names.get(b, b)]
            if -sep > deepest[0]:
                deepest = (-sep, pair)
            if sep <= tolerance:
                contacts.append(
                    {
                        "a": pair[0],
                        "b": pair[1],
                        "a_robot": a.startswith(self.robot_path + "/"),
                        "b_robot": b.startswith(self.robot_path + "/"),
                        "depth": max(0.0, -float(sep)),
                    }
                )
        self._contacts = contacts
        if deepest[0] > self.maxdepth:
            self.maxdepth, self.deepest = deepest
        if deepest[0] > float(self.robot.safety["max_penetration_m"]):
            raise PhysicsFault(
                "penetration",
                f"Penetration {deepest[0] * 1000:.3f} mm: {deepest[1]}",
                step=self._step,
            )
        q = np.asarray(self.view.get_joint_positions())[0]
        with np.errstate(invalid="ignore"):
            err = np.nan_to_num(np.maximum(self.lo - q, q - self.hi), nan=0.0).clip(
                min=0.0
            )
        i = int(np.argmax(err))
        self.maxlimit = max(self.maxlimit, float(err[i]))
        if err[i] > float(self.robot.safety["max_joint_limit_overshoot_rad"]):
            name = self.view.dof_names[i]
            raise PhysicsFault(
                "joint_limit",
                f"Joint limit violation: {name} {err[i]:.6f}",
                step=self._step,
            )

    def step(self):
        self._before_physics()
        self.world.step(render=False)
        self._after_physics()

    def read(self):
        v = self.view
        q, dq, tau = (
            np.asarray(x)[0]
            for x in (
                v.get_joint_positions(),
                v.get_joint_velocities(),
                v.get_measured_joint_efforts(),
            )
        )
        return BackendState(
            sim_time=self._step * self._dt,
            step=self._step,
            q={j: float(q[i]) for j, i in self.dof.items()},
            dq={j: float(dq[i]) for j, i in self.dof.items()},
            effort={j: float(tau[i]) for j, i in self.dof.items()},
            effort_kind=EFFORT_SIMULATED,
            contacts=list(self._contacts),
            max_penetration_m=float(self.maxdepth),
        )

    def render(self, camera, width, height):
        if camera not in self.cams:
            raise KeyError(f"no Isaac Sim camera {camera!r}")
        if self._rendered_step != self._step:
            for _ in range(int(self.options.get("render_updates", 1))):
                self.world.render()  # Kit update with physics stepping disabled
            self._rendered_step = self._step
        cam, (w, h) = self.cams[camera]
        rgb = np.asarray(cam.get_rgba())[..., :3]
        if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(
                f"{camera}: unexpected frame {rgb.shape} {rgb.dtype} (no frame rendered yet?)"
            )
        if (width, height) != (w, h):
            if width * h != height * w:
                raise ValueError(
                    f"{camera}: renders {w}x{h}; {width}x{height} changes the aspect ratio"
                )
            from PIL import Image

            rgb = np.asarray(
                Image.fromarray(rgb).resize((width, height), Image.BILINEAR)
            )
        return np.ascontiguousarray(rgb)

    def task_state(self):
        out = {}
        for side, part in self.parts.items():
            body = f"{side}_part"
            pos, _ = part.get_world_pose()
            mine = [c for c in self._contacts if body in (c["a"], c["b"])]
            jaws = {f"{side}finger{i}_Link": i for i in (1, 2)}
            out[side] = {
                "xyz": np.asarray(pos, float).tolist(),
                "velocity_m_s": float(np.linalg.norm(part.get_linear_velocity())),
                "touching_robot": any(c["a_robot"] or c["b_robot"] for c in mine),
                "jaw_contacts": sorted(
                    {jaws[n] for c in mine for n in (c["a"], c["b"]) if n in jaws}
                ),
            }
        return out

    def tracked_frames(self):
        # The tensor view is a private attribute in 4.5/5.x; rows are x y z qx qy qz qw.
        poses = np.asarray(self.view._physics_view.get_link_transforms())[0]

        def pose(link):
            row = poses[self.view.get_link_index(link)]
            return row[:3], _matrix_xyzw(row[3:])

        frames = {}
        for site, s in self.scene["sites"].items():
            p, R = pose(s["body"])
            frames[site] = (p + R @ np.asarray(s["pos"])).tolist()
        for c in self.robot.cameras.values():
            frames[c.sim_name] = pose(self.scene["cameras"][c.sim_name]["urdf_link"])[
                0
            ].tolist()
        return frames

    def run_owned_loop(self, hooks):
        """Isaac Sim steps through world.step(); its PRE_PHYSICS_STEP event runs
        hooks.pre_step. Audit and post_step run after world.step() returns, on
        the main thread outside any physics callback, so renders stay legal and
        a fault freezes physics at once. Kit updates (render=True) every trace
        tick keep the app and viewport alive."""
        from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager

        fired = []

        def pre_physics(*_):  # assumed signature (step_dt, context)
            fired.append(1)
            hooks.pre_step()
            self._before_physics()

        cid = SimulationManager.register_callback(
            pre_physics, event=IsaacEvents.PRE_PHYSICS_STEP
        )
        every = max(1, round(1 / (self.robot.control["trace_hz"] * self._dt)))
        rtf = float(self.options.get("realtime_factor", 1.0))
        tick = time.monotonic()
        try:
            while self.app.is_running() and not hooks.should_stop():
                if hooks.halted():
                    hooks.idle()
                    continue
                fired.clear()
                due = (self._step + 1) % every == 0
                self.world.step(render=due)
                if len(fired) != 1:
                    raise RuntimeError(
                        f"PRE_PHYSICS_STEP fired {len(fired)} times for one world.step()"
                    )
                try:
                    self._after_physics()
                except PhysicsFault as exc:
                    hooks.on_fault(exc)
                if due:
                    self._rendered_step = self._step
                hooks.post_step()
                if rtf > 0:
                    tick += self._dt / rtf
                    delay = tick - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -0.25:
                        tick = time.monotonic()
        finally:
            SimulationManager.deregister_callback(cid)

    def close(self):
        # SimulationApp.close() runs at interpreter exit (atexit, registered in load).
        if getattr(self, "world", None) is not None:
            self.world.stop()
