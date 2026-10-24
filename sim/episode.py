"""The episode every scene runs: Scene, the local MuJoCo dual-arm workstation
with per-physics-step contact auditing.

Scene compiles the derived robot scene with its cameras, applies the model
overrides, drives the robot through position-actuator ctrl plus gravity
feedforward, audits penetration and joint limits after every physics step,
samples the 25 Hz trace and records the videos. A scene subclass
(scenes/<name>/run.py) brings its objects and its task.

No object pose writes after reset, welds, attachment constraints, contact-mask
changes, or simulator-state inputs to a learned policy.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot_kinematics import BODY_JOINTS, Kinematics, controller_rates
from sim.build_task import ARM_JOINTS, build, fmt
from sim.camera_rig import POLICY_CAMERAS, assert_mounts, install_cameras
from sim.gripper_linkage import dependent_joints

TORSO_SERVOS = (
    "calf_pitch_joint",
    "thigh_pitch_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
)
HEAD_SERVOS = ("head_yaw_joint", "head_pitch_joint")
# Share of the controller's arm rate limits, at its default speed ratio, that the
# 25 Hz arm goals may use, so a replay on the real robot at that ratio is not
# clipped (user decision 2026-10-24).
RATE_HEADROOM = 0.9


def minimum_jerk(x):
    """Quintic progress (0 to 1) with zero velocity and acceleration at both ends."""
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def add_camera(parent, name, eye, lookat, fovy=45):
    eye, lookat = np.asarray(eye), np.asarray(lookat)
    z = eye - lookat
    z = z / np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return ET.SubElement(
        parent,
        "camera",
        name=name,
        pos=fmt(eye),
        xyaxes=fmt(np.r_[x, y]),
        fovy=str(fovy),
    )


class Video:
    def __init__(self, path, w, h, fps=25):
        self.container = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = w
        self.stream.height = h
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "20", "preset": "fast"}

    def write(self, rgb):
        for packet in self.stream.encode(
            av.VideoFrame.from_ndarray(rgb, format="rgb24")
        ):
            self.container.mux(packet)

    def close(self):
        for p in self.stream.encode():
            self.container.mux(p)
        self.container.close()


class Scene:
    def __init__(self, output, record=False):
        root, self.setup = self.build_root()
        world = root.find("worldbody")
        add_camera(world, "overview", [1.85, -1.70, 1.85], [0.10, 0, 1.0], 48)
        add_camera(world, "portrait_front", [1.35, 0, 1.62], [0.02, 0, 1.43], 35)
        add_camera(world, "portrait_side", [1.1, -0.9, 1.7], [0.02, 0, 1.43], 35)
        self.camera_metadata = install_cameras(root)
        self.xml = ET.tostring(root, encoding="unicode")
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        # URDF camera sites lie inside the lens enclosure/front visor. A 45 mm
        # render near plane removes only that camera's own housing; it does not
        # change camera extrinsics or any contact/physics geometry.
        self.m.vis.map.znear = 0.045 / self.m.stat.extent
        mapping = root.find("visual/map")
        if mapping is None:
            mapping = ET.SubElement(root.find("visual"), "map")
        mapping.set("znear", str(float(self.m.vis.map.znear)))
        self.xml = ET.tostring(root, encoding="unicode")
        (output / "scene.xml").write_text(self.xml)
        self.d = mujoco.MjData(self.m)
        self.output = output
        self.qids = {
            s: np.array([self.m.joint(f"{s}_{j}_joint").qposadr[0] for j in ARM_JOINTS])
            for s in ["left", "right"]
        }
        self.dofs = {
            s: np.array([self.m.joint(f"{s}_{j}_joint").dofadr[0] for j in ARM_JOINTS])
            for s in ["left", "right"]
        }
        self.aids = {
            s: np.array([self.m.actuator(f"{s}_{j}_joint").id for j in ARM_JOINTS])
            for s in ["left", "right"]
        }
        self.fingers = {
            s: np.array([self.m.actuator(f"{s}finger{i}_joint").id for i in [1, 2]])
            for s in ["left", "right"]
        }
        self.fqa = {
            s: np.array(
                [self.m.joint(f"{s}finger{i}_joint").qposadr[0] for i in [1, 2]]
            )
            for s in ["left", "right"]
        }
        self.sites = {s: self.m.site(s + "_grasp").id for s in ["left", "right"]}
        # Leg, waist and head servos also get gravity feedforward: on kp alone the
        # torso sagged up to 7.5 mrad per joint (1.3 degrees in all) with the arms
        # forward, and the head 3.8 mrad, where the real servos hold their pose.
        # So do the gripper linkage's coupled joints: the real lead screw does not
        # back-drive, so only the finger slides' own load reaches the finger servos.
        linkage = [j for s in ("left", "right") for j in dependent_joints(s)]
        self.held = np.array(
            [self.m.joint(j).dofadr[0] for j in TORSO_SERVOS + HEAD_SERVOS]
            + [self.m.joint(j).dofadr[0] for j in linkage]
        )
        # IK and FK come from the shared host-side kinematics, the same code a
        # real-robot host and simbridge use; it matches this model's FK to 1e-15 m.
        self.kin = Kinematics()
        rates = controller_rates()
        share = rates["default_ratio"] * RATE_HEADROOM
        self.max_rate = rates["velocity"] * share
        self.max_accel = rates["acceleration"] * share
        self.body_qpos = {j: self.m.joint(j).qposadr[0] for j in BODY_JOINTS}
        self.goal = {s: np.zeros(7) for s in self.qids}
        self.grip = {s: 0.0 for s in self.qids}
        self.step_number = 0
        self.frames = 0
        self.phase = "reset"
        self.instruction = "Initialize the fixed-base workstation."
        self.trace = []
        self.maxdepth = 0.0
        self.deepest = None
        self.maxlimit = 0.0
        self.stable_frames = 0
        # Grasp evidence per object, kept by the scene's track_objects().
        self.lift_evidence = {}
        # Scene objects are the bodies named <name>_part; the rest is the robot.
        self.robot_bodies = {
            i
            for i in range(1, self.m.nbody)
            if not self.m.body(i).name.endswith("_part")
        }
        for side in self.qids:
            for aid in self.fingers[side]:
                self.m.actuator_gainprm[aid, 0] = 1500
                self.m.actuator_biasprm[aid, 1] = -1500
                self.m.actuator_forcerange[aid] = [-6, 6]
        # Initial values are the source model's zero posture; the table is in
        # front of the robot, clear of the resting hands.
        mujoco.mj_forward(self.m, self.d)
        # The source leg/waist servos (kp 6000 N m/rad, 5 N m s/rad damping) are
        # nearly undamped (zeta ~0.01), so the upper body rocked at ~1.5 Hz after
        # every arm move. Add servo velocity feedback: zeta 0.7 at this posture.
        for name in TORSO_SERVOS:
            dof = self.m.joint(name).dofadr[0]
            unit, column = np.zeros(self.m.nv), np.zeros(self.m.nv)
            unit[dof] = 1
            mujoco.mj_mulM(self.m, self.d, column, unit)
            aid = self.m.actuator(name).id
            kp = self.m.actuator_gainprm[aid, 0]
            self.m.actuator_biasprm[aid, 2] = -1.4 * np.sqrt(kp * column[dof])
        assert_mounts(self.m, self.d, self.camera_metadata)
        self.record = record
        self.renderer = (
            mujoco.Renderer(self.m, height=480, width=640) if record else None
        )
        self.videos = {}
        if record:
            self.videos["demo"] = Video(output / "expert.mp4", 1280, 720)
            self.videos["cameras"] = Video(output / "cameras.mp4", 1280, 960)
            for cam in POLICY_CAMERAS:
                self.videos[cam] = Video(output / (cam + ".mp4"), 640, 480)

    def build_root(self):
        """The derived scene tree and build report: the robot, its table and the
        lights. A scene subclass returns its task.py's scene."""
        return build()

    def body(self):
        return {j: float(self.d.qpos[a]) for j, a in self.body_qpos.items()}

    def ik(self, side, target, near=None, rotation=None):
        current = self.d.qpos[self.qids[side]]
        return self.kin.ik(
            side, target, current, self.body(), near=near, rotation=rotation
        )

    def state(self):
        return np.r_[
            self.d.qpos[self.qids["left"]],
            self.d.qpos[self.qids["right"]],
            [
                (self.d.qpos[self.fqa[s]][0] - self.d.qpos[self.fqa[s]][1]) / 0.095
                for s in ["left", "right"]
            ],
        ]

    def command(self):
        return np.r_[
            self.goal["left"], self.goal["right"], self.grip["left"], self.grip["right"]
        ]

    def physics_step(self):
        # Save the measured observation and the command about to be applied at
        # the same 25 Hz control boundary, before advancing the physics.
        if self.step_number % 20 == 0:
            mujoco.mj_forward(self.m, self.d)
            self.trace.append(self.sample())
            self.stable_frames = self.stable_frames + 1 if self.complete() else 0
            if self.record:
                self.render()
        for side in self.goal:
            self.d.ctrl[self.aids[side]] = self.goal[side]
            self.d.ctrl[self.fingers[side]] = [
                self.grip[side] * 0.0475,
                -self.grip[side] * 0.0475,
            ]
        # Fixed workstation gaze: move the actual head joint, never rotate a
        # camera independently of its URDF mounting frame. A minimum-jerk ramp
        # over 3.3 s stays within the controller's head limits (head_config.json:
        # 30 deg/s, 40 deg/s2, 50 deg/s3); the 1.5 s smoothstep it replaces
        # reached 76 deg/s2.
        gaze_phase = min(float(self.d.time) / 3.3, 1.0)
        gaze_ramp = gaze_phase**3 * (10 - 15 * gaze_phase + 6 * gaze_phase**2)
        self.d.ctrl[self.m.actuator("head_pitch_joint").id] = 0.50 * gaze_ramp
        # Gravity torque feedforward is applied through forces, never pose or
        # velocity writes. It is separate from object contact and joint servos.
        self.d.qfrc_applied[:] = 0
        for side in self.dofs:
            self.d.qfrc_applied[self.dofs[side]] = self.d.qfrc_bias[self.dofs[side]]
        self.d.qfrc_applied[self.held] = self.d.qfrc_bias[self.held]
        mujoco.mj_step(self.m, self.d)
        self.step_number += 1
        for c in self.d.contact:
            depth = max(0.0, -float(c.dist))
            if depth > self.maxdepth:
                self.maxdepth = depth
                self.deepest = [self.m.geom(int(g)).name for g in c.geom]
            if depth > 0.001:
                raise RuntimeError(
                    f"Penetration {depth * 1000:.3f} mm: {[self.m.geom(int(g)).name for g in c.geom]}"
                )
        for j in range(self.m.njnt):
            if self.m.jnt_limited[j]:
                q = float(self.d.qpos[self.m.jnt_qposadr[j]])
                lo, hi = self.m.jnt_range[j]
                err = max(lo - q, q - hi, 0.0)
                self.maxlimit = max(self.maxlimit, err)
                if err > 0.002:
                    raise RuntimeError(
                        f"Joint limit violation: {self.m.joint(j).name} {err:.6f}"
                    )
        self.track_objects()

    def sample(self):
        """One trace record: the measured state and the command about to apply.
        A scene subclass adds its objects."""
        return {
            "t": float(self.d.time),
            "phase": self.phase,
            "subtask": self.instruction,
            "state": self.state().tolist(),
            "action": self.command().tolist(),
            "max_penetration_mm": self.maxdepth * 1000,
        }

    def track_objects(self):
        """Grasp evidence, updated after every physics step; a scene subclass
        keeps it in lift_evidence."""

    def complete(self):
        """Whether the scene's task is done; stable_frames counts the control
        ticks it has held. The robot alone has no task."""
        return False

    def move(
        self,
        targets=None,
        grips=None,
        seconds=2.0,
        linear=False,
        rotations=None,
        speed=None,
    ):
        # Joint path per arm. With linear=True it runs through IK waypoints 1 cm
        # apart on the Cartesian line, so the hand cannot swing off the line near
        # the edge of its reach; the timing profile is the same either way.
        # rotations holds a hand orientation per arm (default world-aligned).
        paths = {s: np.array([self.goal[s], self.goal[s]]) for s in self.goal}
        for s, xyz in (targets or {}).items():
            rotation = (rotations or {}).get(s)
            if linear:
                paths[s] = self.kin.line(
                    s, self.goal[s], xyz, self.body(), rotation=rotation
                )
            else:
                paths[s] = np.array([self.goal[s], self.ik(s, xyz, rotation=rotation)])
        # With speed (rad/s), waypoints are timed by joint travel instead of
        # evenly: arc is the cumulative largest joint step, so the hand slows
        # only where the arm turns fast, and the move is stretched until no
        # joint's peak rate (1.875x the mean of the quintic) exceeds speed.
        arcs = {}
        if speed:
            for s, path in paths.items():
                arcs[s] = np.r_[0, np.cumsum(np.abs(np.diff(path, axis=0)).max(1))]
            seconds = max(seconds, 1.875 * max(a[-1] for a in arcs.values()) / speed)
        seconds = self.stretch(paths, arcs, seconds)
        gs = self.grip.copy()
        ge = gs | (grips or {})
        n = int(seconds / self.m.opt.timestep)
        for i in range(n):
            p = minimum_jerk((i + 1) / n)
            if self.step_number % 20 == 0:
                for s, path in paths.items():
                    self.goal[s] = self.along(path, arcs.get(s), p)
                    self.grip[s] = gs[s] + (ge[s] - gs[s]) * p
            self.physics_step()

    @staticmethod
    def along(path, arc, p):
        """Point at progress p (0 to 1) on a joint path: evenly over its
        waypoints, or by arc (cumulative joint travel) when given."""
        if arc is not None and arc[-1] > 0:
            u = np.interp(p * arc[-1], arc, np.arange(len(path)))
        else:
            u = p * (len(path) - 1)
        k = min(int(u), len(path) - 2)
        return path[k] + (path[k + 1] - path[k]) * (u - k)

    def stretch(self, paths, arcs, seconds):
        """The move's duration, lengthened until the arm goals it commands at the
        25 Hz control ticks stay within max_rate and max_accel. The move starts
        at rest. Acceleration counts with its square root, as it scales with
        1/T^2 on smooth paths; the 2% minimum step ends the loop where waypoint
        corners make it scale with 1/T."""
        dt = 20 * self.m.opt.timestep
        while True:
            n = int(seconds / self.m.opt.timestep)
            ticks = [i for i in range(n) if (self.step_number + i) % 20 == 0]
            if not ticks:
                return seconds
            worst = 0.0
            for s, path in paths.items():
                q = np.array(
                    [path[0], path[0]]
                    + [
                        self.along(path, arcs.get(s), minimum_jerk((i + 1) / n))
                        for i in ticks
                    ]
                )
                rate = np.abs(np.diff(q, axis=0)) / dt / self.max_rate
                accel = np.abs(np.diff(q, 2, axis=0)) / dt**2 / self.max_accel
                worst = max(worst, rate.max(), np.sqrt(accel.max()))
            if worst <= 1:
                return seconds
            seconds *= max(worst, 1.02)

    def stage(self, name, instruction):
        self.phase = name
        self.instruction = instruction
        print(f"{self.d.time:6.2f}s {name}: {instruction}", flush=True)

    def render(self):
        images = {}
        for cam in ["overview", *POLICY_CAMERAS]:
            self.renderer.update_scene(self.d, camera=cam)
            images[cam] = self.renderer.render().copy()
        canvas = Image.new("RGB", (1280, 720), (20, 27, 38))
        canvas.paste(Image.fromarray(images["overview"]).resize((960, 720)), (0, 0))
        canvas.paste(
            Image.fromarray(images["left_wrist_d405"]).resize((320, 240)), (960, 70)
        )
        canvas.paste(
            Image.fromarray(images["right_wrist_d405"]).resize((320, 240)), (960, 390)
        )
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        font = ImageFont.truetype(font_path, 21)
        small = ImageFont.truetype(font_path, 17)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 1280, 65), fill=(20, 27, 38))
        draw.text(
            (18, 8),
            f"LOCAL MUJOCO | DUAL-ARM EXPERT | {self.d.time:05.2f}s | {self.phase.upper()}",
            font=font,
            fill="white",
        )
        draw.text((18, 36), self.instruction, font=small, fill=(128, 218, 233))
        draw.text((972, 324), "LEFT WRIST / D405", font=small, fill="white")
        draw.text((972, 644), "RIGHT WRIST / D405", font=small, fill="white")
        draw.text(
            (972, 676),
            f"Peak depth: {self.maxdepth * 1000:.3f} mm",
            font=small,
            fill=(142, 231, 180),
        )
        self.videos["demo"].write(np.asarray(canvas))
        mosaic = Image.new("RGB", (1280, 960))
        for i, cam in enumerate(POLICY_CAMERAS):
            x, y = (i % 2) * 640, (i // 2) * 480
            mosaic.paste(Image.fromarray(images[cam]), (x, y))
            md = ImageDraw.Draw(mosaic)
            md.rectangle((x, y, x + 640, y + 32), fill=(20, 27, 38))
            md.text(
                (x + 12, y + 5),
                f"{cam.upper()} | {self.d.time:.2f}s",
                font=small,
                fill="white",
            )
        self.videos["cameras"].write(np.asarray(mosaic))
        for cam in POLICY_CAMERAS:
            self.videos[cam].write(images[cam])
        self.frames += 1

    def close(self):
        for video in self.videos.values():
            video.close()
        if self.renderer:
            self.renderer.close()
