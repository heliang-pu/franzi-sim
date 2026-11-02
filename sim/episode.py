"""The episode every scene runs: Scene, the local MuJoCo dual-arm workstation
with per-physics-step contact auditing.

Scene compiles the derived robot scene with its cameras, applies the model
overrides, drives the robot through position-actuator ctrl plus gravity
feedforward, audits penetration and joint limits after every physics step,
samples the 25 Hz trace and records the videos. A scene subclass
(scenes/<name>/run.py) brings its objects and its task.

Besides the arms and grippers, the waist and head servos take goals (turn(),
look()), and every control tick logs the full robot and world state
(full_arrays(): FULL_NAMES state, action, velocity and effort, the hands' and
cameras' poses, free objects, prop joints and contacts); a recording adds
each policy camera's depth and segmentation (user request 2026-10-25: control
the waist, collect everything). With the waist and head left alone, the
physics is bit-identical to before.

No object pose writes after reset, welds, attachment constraints, contact-mask
changes, or simulator-state inputs to a learned policy: the world state is
logged for evaluation and analysis only.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot_kinematics import BODY_JOINTS, Kinematics, controller_rates
from sim.build_task import (
    ARM_JOINTS,
    BASE_JOINTS,
    BASE_LIFT,
    FINGER_JOINTS,
    build,
    fmt,
)
from sim.camera_rig import POLICY_CAMERAS, assert_mounts, install_cameras
from sim.gripper_linkage import dependent_joints

TORSO_SERVOS = (
    "calf_pitch_joint",
    "thigh_pitch_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
)
HEAD_SERVOS = ("head_yaw_joint", "head_pitch_joint")
BODY_SERVOS = TORSO_SERVOS + HEAD_SERVOS
# The full robot state and action, by name: the 16-D data contract (left arm,
# right arm, left and right gripper closure) followed by the waist and head
# servos. FULL_NAMES[:16] is the 16-D contract; VIEWS name the training subsets.
FULL_NAMES = (
    [f"left_{j}" for j in ARM_JOINTS]
    + [f"right_{j}" for j in ARM_JOINTS]
    + ["left_gripper", "right_gripper"]
    + [j.removesuffix("_joint") for j in BODY_SERVOS]
)
VIEWS = {
    "arms16": list(range(16)),
    "waist17": list(range(16)) + [FULL_NAMES.index("waist_yaw")],
}
# Share of the controller's arm rate limits, at its default speed ratio, that the
# 25 Hz arm goals may use, so a replay on the real robot at that ratio is not
# clipped (user decision 2026-10-24).
RATE_HEADROOM = 0.9
# The waist and head servos' limits in the controller image (waist_config.json,
# head_config.json: 30 deg/s, 40 deg/s2, 50 deg/s3) and the waist's default
# speed ratio (joint_vel_ratio 30 %): turn() keeps within RATE_HEADROOM of them
# at that ratio, as move() does for the arms. Not verified on hardware.
BODY_LIMITS = {
    "velocity": np.radians(30),
    "acceleration": np.radians(40),
    "jerk": np.radians(50),
}
BODY_SPEED_RATIO = 0.30
# The mobile base's limits: the chassis SDK's velocity range (publish_vehicle_
# control: about 0.5 m/s along x and y, 0.2 rad/s about z) and accelerations
# chosen here (the SDK documents none). drive() keeps within RATE_HEADROOM.
BASE_LIMITS = {
    "velocity": (0.5, 0.2),  # m/s, rad/s
    "acceleration": (0.5, 0.5),  # m/s2, rad/s2
}
# The base's three wheel units, a swerve drive (user, 2026-10-25: the wheels
# steer and roll, not slide like mecanum wheels). Each steers along its own
# velocity and rolls at its speed over the wheel radius; before each drive the
# wheels turn to where it starts, at most STEER_RATE (rad/s, chosen here: the
# SDK documents no steering limit). The base still moves on its planar joints:
# the wheels follow its motion and carry no load.
WHEEL_UNITS = ("left_front", "right_front", "rear")
STEER_RATE = 1.5
# Each drive's steering is planned at these progress points, within
# STEER_LIMIT of zero (the steering joints' range is 3.14 rad).
STEER_PROGRESS = np.linspace(0, 1, 41)
STEER_LIMIT = 3.0
# The workstation gaze: the head pitches to GAZE_PITCH over GAZE_SECONDS.
GAZE_PITCH = 0.50
GAZE_SECONDS = 3.3
# prepare() holds until no arm, torso or head joint moves faster than this
# (rad/s) for HOLD_TICKS control ticks, within PREPARE_LIMIT seconds.
STILL_RATE = 0.005
HOLD_TICKS = 12
PREPARE_LIMIT = 6.0
# Depth (mm) and segmentation (instance id) frames: 16-bit, lossless.
DEPTH_SCALE = 1000.0
# The simulation broke down (PhysicsError) when MuJoCo raises one of these
# warnings (a bad state, which mj_step answers by resetting it silently, or a
# full contact or constraint arena, which drops contacts), when the state is
# not finite, or when a free object moves faster than FLUNG (m/s, rad/s): no
# object here falls more than 1.3 m, which gives 5 m/s.
BROKEN_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_CONTACTFULL,
    mujoco.mjtWarning.mjWARN_CNSTRFULL,
)
FLUNG = (10.0, 200.0)
# Text in recordings and preview sheets: Arial on macOS, DejaVu Sans on Linux
# (the pro6000), as simbridge/recorder.py.
OVERLAY_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


class PhysicsError(RuntimeError):
    """The simulation itself broke down (Scene.check_physics): the episode is
    void, neither the expert's nor a policy's failure. RoboDojo likewise drops
    a seed whose PhysX state broke and draws another; runs mark it "void"."""


def overlay_font(size):
    for path in OVERLAY_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


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


class LosslessVideo(Video):
    """16-bit single-channel frames (depth in mm, segmentation ids) as FFV1 in
    Matroska, which keeps every value exactly."""

    def __init__(self, path, w, h, fps=25):
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("ffv1", rate=fps)
        self.stream.width = w
        self.stream.height = h
        self.stream.pix_fmt = "gray16le"

    def write(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint16)
        for packet in self.stream.encode(
            av.VideoFrame.from_ndarray(frame, format="gray16le")
        ):
            self.container.mux(packet)


def quat(xmat):
    """A 3x3 rotation (flattened) as a w, x, y, z quaternion."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, float).ravel())
    return q


class Scene:
    # The simulator, in the video header and summary.json; isaac/episode.py
    # runs the same scenes on Isaac Sim.
    LABEL = "LOCAL MUJOCO"
    SIMULATOR = f"MuJoCo {mujoco.__version__}"
    # A recording also writes each policy camera's depth and segmentation.
    RECORD_DEPTH = True
    # A contact deeper than 1 mm or a joint 0.002 past its limit ends the
    # episode (RuntimeError). With this off they are only logged (violation()):
    # a policy evaluation's lenient count (isaac.policy --lenient).
    stop_on_violation = True

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
        # Each hand's fingers, the one that closes with a positive slide first
        # (FINGER_JOINTS): a grip g commands +-0.0475 g.
        self.fingers = {
            s: np.array([self.m.actuator(j).id for j in FINGER_JOINTS[s]])
            for s in ["left", "right"]
        }
        self.fqa = {
            s: np.array([self.m.joint(j).qposadr[0] for j in FINGER_JOINTS[s]])
            for s in ["left", "right"]
        }
        self.fdof = {
            s: np.array([self.m.joint(j).dofadr[0] for j in FINGER_JOINTS[s]])
            for s in ["left", "right"]
        }
        self.cam_ids = [self.m.camera(c).id for c in POLICY_CAMERAS]
        self.sites = {s: self.m.site(s + "_grasp").id for s in ["left", "right"]}
        # The waist and head servos: the torso holds its zero posture and the
        # head runs the workstation gaze until a scene turns or looks.
        self.body_aids = {j: self.m.actuator(j).id for j in BODY_SERVOS}
        self.body_qadr = {j: self.m.joint(j).qposadr[0] for j in BODY_SERVOS}
        self.body_dofs = {j: self.m.joint(j).dofadr[0] for j in BODY_SERVOS}
        self.body_goal = {j: 0.0 for j in TORSO_SERVOS}
        self.head_goal = None
        # The mobile base (build(mobile=True)): its joints' goals, x, y and yaw
        # from where it starts.
        joint_names = {self.m.joint(j).name for j in range(self.m.njnt)}
        self.mobile = all(j in joint_names for j in BASE_JOINTS)
        if self.mobile:
            self.base_qadr = np.array([self.m.joint(j).qposadr[0] for j in BASE_JOINTS])
            self.base_dofs = np.array([self.m.joint(j).dofadr[0] for j in BASE_JOINTS])
            self.base_aids = np.array([self.m.actuator(j).id for j in BASE_JOINTS])
            self.base_goal = np.zeros(3)
            self.base_last = np.zeros(3)
            steer = [self.m.joint(f"{u}_steering_joint") for u in WHEEL_UNITS]
            wheel = [self.m.joint(f"{u}_wheel_joint") for u in WHEEL_UNITS]
            self.steer_qadr = np.array([j.qposadr[0] for j in steer])
            self.wheel_dofs = np.array([j.dofadr[0] for j in wheel])
            self.steer_aids = np.array([self.m.actuator(j.name).id for j in steer])
            self.wheel_aids = np.array([self.m.actuator(j.name).id for j in wheel])
            # Each unit's steering axis in the base frame, about whose centre
            # (the yaw joint) the base turns: from the start pose, since the
            # units' bodies sit in base_link's turned frame (BASE_ROT), where
            # pos[:2] put the front wheels 0.14 m behind the centre. steering()
            # takes each unit to steer about +z and roll along +x at 0 (to the
            # URDF's 1.5708 rad right angles: 3.7 urad off).
            start = mujoco.MjData(self.m)
            mujoco.mj_kinematics(self.m, start)
            axes = start.xaxis[[j.id for j in steer + wheel]]
            assert np.allclose(axes, [[0, 0, 1]] * 3 + [[0, 1, 0]] * 3, atol=1e-4)
            centre = start.xanchor[self.m.joint("base_yaw").id]
            self.unit_xy = (start.xanchor[[j.id for j in steer]] - centre)[:, :2]
            self.wheel_radius = float(self.m.geom("left_front_wheel_Link").size[0])
            self.steer_goal = np.zeros(3)
            self.wheel_rate = np.zeros(3)
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
        # Control ticks are recorded (trace, full log, frames) from the start,
        # or from the end of prepare(), which also moves the clock's zero.
        self.recording = True
        self.t0 = 0.0
        self.trace = []
        self.maxdepth = 0.0
        self.deepest = None
        self.maxlimit = 0.0
        self.violation_log = {}
        self.stable_frames = 0
        # Grasp evidence per object, kept by the scene's track_objects().
        self.lift_evidence = {}
        # Scene objects are the bodies named <name>_part; the rest is the robot.
        self.robot_bodies = {
            i
            for i in range(1, self.m.nbody)
            if not self.m.body(i).name.endswith("_part")
        }
        # The contact audit (user request 2026-10-25: no collisions). The robot
        # may touch the floor with its base and wheels, and movable things
        # (free objects, buttons, lids, doors) with its hands: each wrist_roll
        # link and everything it carries. Any other contact of a robot body,
        # including one arm on the other, is a collision; collisions() lists
        # them, and a run with any is not a success.
        hands = {self.m.body(f"{s}_wrist_roll_Link").id for s in ("left", "right")}
        self.hand_bodies = {
            b for b in self.robot_bodies if any(self.descends(b, h) for h in hands)
        }
        self.base_bodies = {
            b
            for b in self.robot_bodies
            if self.m.body(b).name == "base_link"
            or "wheel" in self.m.body(b).name
            or "steering" in self.m.body(b).name
        }
        self.floor_geoms = {
            g
            for g in range(self.m.ngeom)
            if self.m.geom(g).name == "floor"
            or self.m.body(int(self.m.geom_bodyid[g])).name == "floor_part"
        }
        self.collision_log = {}
        # The world state logged every control tick (evaluation only): free
        # objects, the props' own joints (doors, lids, buttons, rollers) and
        # contacts that involve anything but the robot alone.
        self.body_names = [self.m.body(b).name for b in range(self.m.nbody)]
        joints = range(self.m.njnt)
        self.free_objects = [
            (
                self.body_names[self.m.jnt_bodyid[j]],
                self.m.jnt_qposadr[j],
                self.m.jnt_dofadr[j],
            )
            for j in joints
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        self.prop_joints = [
            (self.m.joint(j).name, self.m.jnt_qposadr[j])
            for j in joints
            # int(): a numpy integer is never `in` a tuple of these enums.
            if int(self.m.jnt_type[j])
            in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            and self.body_names[self.m.jnt_bodyid[j]].endswith("_part")
        ]
        self.full = {}
        self.contacts = []
        # Segmentation instance ids: 0 nothing, 1 the robot, then one per body.
        self.segment_names = ["background", "robot"]
        segment = np.zeros(self.m.nbody, np.uint16)
        for b in range(self.m.nbody):
            if b in self.robot_bodies:
                segment[b] = 1
            else:
                segment[b] = len(self.segment_names)
                self.segment_names.append(self.body_names[b] or "world")
        self.geom_segment = segment[self.m.geom_bodyid]
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
            if self.RECORD_DEPTH:
                for cam in POLICY_CAMERAS:
                    for kind in ("depth", "segmentation"):
                        self.videos[f"{kind}_{cam}"] = LosslessVideo(
                            output / f"{kind}_{cam}.mkv", 640, 480
                        )

    def build_root(self):
        """The derived scene tree and build report: the robot, its table and the
        lights. A scene subclass returns its task.py's scene."""
        return build()

    def body(self):
        return {j: float(self.d.qpos[a]) for j, a in self.body_qpos.items()}

    def ik(self, side, target, near=None, rotation=None):
        current = self.d.qpos[self.qids[side]]
        target, rotation = self.to_base(target, rotation)
        return self.kin.ik(
            side, target, current, self.body(), near=near, rotation=rotation
        )

    def base_pose(self):
        """The mobile base's measured x, y (m) and yaw (rad) from its start."""
        return self.d.qpos[self.base_qadr].copy()

    def to_base(self, xyz, rotation=None):
        """A world target (grasp-site position, hand rotation) in the frame the
        shared kinematics works in: the robot where it starts. Unchanged on a
        fixed base."""
        if not self.mobile:
            return xyz, rotation
        x, y, yaw = self.base_pose()
        c, s = np.cos(yaw), np.sin(yaw)
        back = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])  # Rz(-yaw)
        local = back @ (np.asarray(xyz, float) - [x, y, BASE_LIFT])
        rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
        return local, back @ rotation

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

    def head_command(self):
        """The head servos' goals (yaw, pitch): look()'s, or until then the
        fixed workstation gaze. That moves the actual head joint, never a camera
        apart from its URDF mounting frame: a minimum-jerk ramp to 0.50 rad of
        pitch over 3.3 s, within the controller's head limits (head_config.json:
        30 deg/s, 40 deg/s2, 50 deg/s3); the 1.5 s smoothstep it replaced reached
        76 deg/s2."""
        if self.head_goal is not None:
            return self.head_goal
        gaze_phase = min(float(self.d.time) / GAZE_SECONDS, 1.0)
        gaze_ramp = gaze_phase**3 * (10 - 15 * gaze_phase + 6 * gaze_phase**2)
        return 0.0, GAZE_PITCH * gaze_ramp

    def state_full(self):
        """FULL_NAMES positions: state() then the waist and head servos (rad)."""
        return np.r_[
            self.state(), [self.d.qpos[self.body_qadr[j]] for j in BODY_SERVOS]
        ]

    def command_full(self):
        """FULL_NAMES goals: command() then the waist and head servos' goals."""
        return np.r_[
            self.command(),
            [self.body_goal[j] for j in TORSO_SERVOS],
            self.head_command(),
        ]

    def log_full(self):
        """Append one control tick to the full log (see full_arrays())."""
        m, d = self.m, self.d
        closure_rate = [
            (d.qvel[self.fdof[s][0]] - d.qvel[self.fdof[s][1]]) / 0.095
            for s in ("left", "right")
        ]
        body = [self.body_dofs[j] for j in BODY_SERVOS]
        squeeze = [
            (
                d.actuator_force[self.fingers[s][0]]
                - d.actuator_force[self.fingers[s][1]]
            )
            / 2
            for s in ("left", "right")
        ]
        rows = {
            "state_full": self.state_full(),
            "action_full": self.command_full(),
            "velocity_full": np.r_[
                d.qvel[self.dofs["left"]],
                d.qvel[self.dofs["right"]],
                closure_rate,
                d.qvel[body],
            ],
            # Joint torque as the motors give it: the servo plus the gravity
            # feedforward; the grippers' squeeze force (N).
            "effort_full": np.r_[
                d.actuator_force[self.aids["left"]] + d.qfrc_applied[self.dofs["left"]],
                d.actuator_force[self.aids["right"]]
                + d.qfrc_applied[self.dofs["right"]],
                squeeze,
                d.actuator_force[[self.body_aids[j] for j in BODY_SERVOS]]
                + d.qfrc_applied[body],
            ],
            "finger_qpos": d.qpos[np.r_[self.fqa["left"], self.fqa["right"]]],
            "eef_pose": [
                np.r_[d.site_xpos[self.sites[s]], quat(d.site_xmat[self.sites[s]])]
                for s in ("left", "right")
            ],
            "camera_pose": [
                np.r_[d.cam_xpos[c], quat(d.cam_xmat[c])] for c in self.cam_ids
            ],
            "object_pose": np.reshape(
                [d.qpos[a : a + 7] for _, a, _ in self.free_objects], (-1, 7)
            ),
            "object_velocity": np.reshape(
                [d.qvel[v : v + 6] for _, _, v in self.free_objects], (-1, 6)
            ),
            "prop_qpos": [d.qpos[a] for _, a in self.prop_joints],
        }
        if self.mobile:
            # In the base's own frame, as the chassis SDK takes them: forward,
            # left, and turning; the command is the goal's rate over this tick.
            yaw = d.qpos[self.base_qadr[2]]
            c, s = np.cos(yaw), np.sin(yaw)
            back = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            rows["base_pose"] = d.qpos[self.base_qadr]
            rows["base_goal"] = self.base_goal.copy()
            rows["base_velocity"] = back @ d.qvel[self.base_dofs]
            rows["base_command"] = (
                back @ (self.base_goal - self.base_last) / (20 * m.opt.timestep)
            )
            self.base_last = self.base_goal.copy()
            rows["wheel_steer"] = d.qpos[self.steer_qadr]
            rows["wheel_rate"] = d.qvel[self.wheel_dofs]
        for key, row in rows.items():
            self.full.setdefault(key, []).append(np.asarray(row, float))
        forces, f6 = {}, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = (int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            if b1 in self.robot_bodies and b2 in self.robot_bodies:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            pair = tuple(
                sorted((self.body_names[b1] or "world", self.body_names[b2] or "world"))
            )
            forces[pair] = forces.get(pair, 0.0) + float(f6[0])
        self.contacts.append(
            [[*pair, round(f, 4)] for pair, f in sorted(forces.items())]
        )

    def full_arrays(self):
        """The full 25 Hz log, one row per trace sample, for trajectory.npz:
        state_full, action_full, velocity_full, effort_full (FULL_NAMES; arm,
        waist and head joints in URDF radians, rad/s and N m, gripper closure
        0-1, its rate and squeeze force in N), finger_qpos (left 1, 2, right 1,
        2; m), eef_pose (left, right grasp site: x, y, z, qw, qx, qy, qz in the
        world), camera_pose (POLICY_CAMERAS, MuJoCo camera frame: looking along
        -z, y up), and the world state for evaluation only: object_pose and
        object_velocity (free bodies: freejoint qpos and qvel) and prop_qpos
        (the props' hinge and slide joints). Empty when nothing was logged (a
        simulator port that steps the physics itself)."""
        if not self.full:
            return {}
        arrays = {k: np.stack(v) for k, v in self.full.items()}
        arrays.update(
            full_names=np.array(FULL_NAMES),
            camera_names=np.array(POLICY_CAMERAS),
            object_names=np.array([n for n, _, _ in self.free_objects], dtype=str),
            prop_joint_names=np.array([n for n, _ in self.prop_joints], dtype=str),
        )
        return arrays

    def write_full(self, output):
        """contacts.json (per tick: [body, body, summed normal force N] for
        every pair that is not the robot alone) and, with a recording,
        segmentation.json (the instance ids of the segmentation videos)."""
        (output / "contacts.json").write_text(json.dumps(self.contacts))
        if self.record and self.RECORD_DEPTH:
            (output / "segmentation.json").write_text(
                json.dumps(dict(enumerate(self.segment_names)), indent=1)
            )

    def full_contract(self):
        """The summary.json description of the full log (None without one)."""
        if not self.full:
            return None
        contract = {
            "names": FULL_NAMES,
            "dim": len(FULL_NAMES),
            "views": VIEWS,
            "units": "arm, waist and head joints in URDF rad (rad/s, N m); grippers "
            "closure 0 open to 1 closed (1/s, squeeze N)",
            "arrays": "trajectory.npz: state_full, action_full, velocity_full, "
            "effort_full, finger_qpos, eef_pose, camera_pose; world state for "
            "evaluation only: object_pose, object_velocity, prop_qpos, contacts.json",
            "body_limits": {
                "velocity_deg_s": 30,
                "acceleration_deg_s2": 40,
                "jerk_deg_s3": 50,
                "speed_ratio": BODY_SPEED_RATIO,
            },
        }
        if self.mobile:
            contract["base"] = {
                "joints": BASE_JOINTS,
                "arrays": "base_pose (x, y m, yaw rad from the start, world), "
                "base_goal (the same, commanded), base_velocity and base_command "
                "(forward and left m/s, turning rad/s, in the base frame); "
                "wheel_steer (rad) and wheel_rate (rad/s) of the left front, right "
                "front and rear units, measured",
                "limits": {
                    "velocity_m_s_rad_s": BASE_LIMITS["velocity"],
                    "acceleration_m_s2_rad_s2": BASE_LIMITS["acceleration"],
                },
            }
        if self.record and self.RECORD_DEPTH:
            contract["depth"] = (
                "depth_<camera>.mkv: FFV1 gray16le, millimetres along the optical "
                "axis, one frame per trace sample"
            )
            contract["segmentation"] = (
                "segmentation_<camera>.mkv: FFV1 gray16le instance ids "
                "(segmentation.json): 0 nothing, 1 the robot, then one per body"
            )
        return contract

    def physics_step(self):
        # Save the measured observation and the command about to be applied at
        # the same 25 Hz control boundary, before advancing the physics.
        if self.recording and self.step_number % 20 == 0:
            mujoco.mj_forward(self.m, self.d)
            self.trace.append(self.sample())
            self.log_full()
            self.stable_frames = self.stable_frames + 1 if self.complete() else 0
            if self.record:
                self.render()
        for side in self.goal:
            self.d.ctrl[self.aids[side]] = self.goal[side]
            self.d.ctrl[self.fingers[side]] = [
                self.grip[side] * 0.0475,
                -self.grip[side] * 0.0475,
            ]
        for j in TORSO_SERVOS:
            self.d.ctrl[self.body_aids[j]] = self.body_goal[j]
        for j, goal in zip(HEAD_SERVOS, self.head_command(), strict=True):
            self.d.ctrl[self.body_aids[j]] = goal
        if self.mobile:
            self.d.ctrl[self.base_aids] = self.base_goal
            self.d.ctrl[self.steer_aids] = self.steer_goal
            self.d.ctrl[self.wheel_aids] = self.wheel_rate
        # Gravity torque feedforward is applied through forces, never pose or
        # velocity writes. It is separate from object contact and joint servos.
        self.d.qfrc_applied[:] = 0
        for side in self.dofs:
            self.d.qfrc_applied[self.dofs[side]] = self.d.qfrc_bias[self.dofs[side]]
        self.d.qfrc_applied[self.held] = self.d.qfrc_bias[self.held]
        mujoco.mj_step(self.m, self.d)
        self.step_number += 1
        self.check_physics()
        for c in self.d.contact:
            depth = max(0.0, -float(c.dist))
            if depth > self.maxdepth:
                self.maxdepth = depth
                self.deepest = [self.m.geom(int(g)).name for g in c.geom]
            if depth > 0.001:
                names = [self.m.geom(int(g)).name for g in c.geom]
                self.violation(
                    ("penetration", *names),
                    f"Penetration {depth * 1000:.3f} mm: {names}",
                    depth * 1000,
                )
            if depth > 0.0:
                self.audit_contact(int(c.geom[0]), int(c.geom[1]), depth)
        for j in range(self.m.njnt):
            if self.m.jnt_limited[j]:
                q = float(self.d.qpos[self.m.jnt_qposadr[j]])
                lo, hi = self.m.jnt_range[j]
                err = max(lo - q, q - hi, 0.0)
                self.maxlimit = max(self.maxlimit, err)
                if err > 0.002:
                    name = self.m.joint(j).name
                    self.violation(
                        ("joint_limit", name),
                        f"Joint limit violation: {name} {err:.6f}",
                        err,
                    )
        self.track_objects()

    def violation(self, key, message, size):
        """A contact over 1 mm deep or a joint over 0.002 past its limit: ends
        the episode, or with stop_on_violation off is logged once per key (the
        geoms, or the joint) with its first message, time and phase, the
        physics steps it lasted and its largest size (mm, or the limit error)."""
        if self.stop_on_violation:
            raise RuntimeError(message)
        hit = self.violation_log.get(key)
        if hit is None:
            hit = self.violation_log[key] = {
                "kind": key[0],
                "first": message,
                "first_s": round(self.clock(), 3),
                "phase": self.phase,
                "physics_steps": 0,
                "max": 0.0,
            }
        hit["physics_steps"] += 1
        hit["max"] = max(hit["max"], round(size, 6))

    def violations(self):
        """The run's logged violations (violation()), first ones first."""
        return sorted(self.violation_log.values(), key=lambda h: h["first_s"])

    def check_physics(self):
        """Raise PhysicsError if the step broke the simulation (BROKEN_WARNINGS,
        a non-finite state, a free object flung away)."""
        d = self.d
        for w in BROKEN_WARNINGS:
            if d.warning[w].number:
                raise PhysicsError(
                    f"{mujoco.mjtWarning(w).name} (info {d.warning[w].lastinfo})"
                )
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            raise PhysicsError("non-finite state")
        for name, _, dof in self.free_objects:
            speed = np.linalg.norm(d.qvel[dof : dof + 3])
            spin = np.linalg.norm(d.qvel[dof + 3 : dof + 6])
            if speed > FLUNG[0] or spin > FLUNG[1]:
                raise PhysicsError(f"{name} flung at {speed:.1f} m/s, {spin:.0f} rad/s")

    def sample(self):
        """One trace record: the measured state and the command about to apply.
        A scene subclass adds its objects."""
        return {
            "t": self.clock(),
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
        joints=None,
    ):
        # Joint path per arm. With linear=True it runs through IK waypoints 1 cm
        # apart on the Cartesian line, so the hand cannot swing off the line near
        # the edge of its reach; the timing profile is the same either way.
        # rotations holds a hand orientation per arm (default world-aligned);
        # joints, an arm's joint goal in place of a target.
        paths = {s: np.array([self.goal[s], self.goal[s]]) for s in self.goal}
        for s, q in (joints or {}).items():
            paths[s] = np.array([self.goal[s], q])
        for s, xyz in (targets or {}).items():
            rotation = (rotations or {}).get(s)
            if linear:
                xyz_k, rotation_k = self.to_base(xyz, rotation)
                paths[s] = self.kin.line(
                    s, self.goal[s], xyz_k, self.body(), rotation=rotation_k
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

    @staticmethod
    def body_seconds(delta, ratio):
        """The shortest minimum-jerk move of delta (rad) whose peak velocity,
        acceleration and jerk (1.875, 5.77 and 60 times delta over T, T^2, T^3)
        stay within RATE_HEADROOM of BODY_LIMITS at speed ratio ratio."""
        share, delta = ratio * RATE_HEADROOM, abs(delta)
        return max(
            1.875 * delta / (BODY_LIMITS["velocity"] * share),
            np.sqrt(10 / np.sqrt(3) * delta / (BODY_LIMITS["acceleration"] * share)),
            np.cbrt(60 * delta / (BODY_LIMITS["jerk"] * share)),
        )

    def hold_arms(self, seconds, grips, update):
        """Run seconds with the arm goals held, calling update(p) with the
        minimum-jerk progress p at every control tick; grips blend as in move()."""
        gs = self.grip.copy()
        ge = gs | (grips or {})
        n = int(seconds / self.m.opt.timestep)
        for i in range(n):
            p = minimum_jerk((i + 1) / n)
            if self.step_number % 20 == 0:
                update(p)
                for s in self.grip:
                    self.grip[s] = gs[s] + (ge[s] - gs[s]) * p
            self.physics_step()

    def turn(self, waist_yaw, seconds=0.0, grips=None):
        """Turn the waist to waist_yaw (rad) while the arms hold their joint
        goals, so the hands and what they hold turn with the torso; it takes
        at least seconds and as long as BODY_LIMITS at BODY_SPEED_RATIO need."""
        start = self.body_goal["waist_yaw_joint"]
        seconds = max(seconds, self.body_seconds(waist_yaw - start, BODY_SPEED_RATIO))

        def update(p):
            self.body_goal["waist_yaw_joint"] = start + (waist_yaw - start) * p

        self.hold_arms(seconds, grips, update)

    def look(self, yaw, pitch, seconds=0.0):
        """Point the head (yaw, pitch rad) while the arms hold, within the head
        limits at the full rate, as the workstation gaze does."""
        y0, p0 = self.head_command()
        delta = max(abs(yaw - y0), abs(pitch - p0))
        seconds = max(seconds, self.body_seconds(delta, 1.0))

        def update(p):
            self.head_goal = (y0 + (yaw - y0) * p, p0 + (pitch - p0) * p)

        self.hold_arms(seconds, None, update)

    @staticmethod
    def base_seconds(distance, angle):
        """The shortest minimum-jerk base move over distance (m) and angle (rad)
        whose peak velocity and acceleration (1.875 and 5.77 times the travel
        over T and T^2) stay within RATE_HEADROOM of BASE_LIMITS."""
        (v, w), (a, b) = BASE_LIMITS["velocity"], BASE_LIMITS["acceleration"]
        h = RATE_HEADROOM
        return max(
            1.875 * distance / (v * h),
            1.875 * angle / (w * h),
            np.sqrt(10 / np.sqrt(3) * distance / (a * h)),
            np.sqrt(10 / np.sqrt(3) * angle / (b * h)),
        )

    def unit_velocity(self, velocity):
        """Each wheel unit's velocity (x, y, base frame, m/s) for a base
        velocity (forward, left m/s and turning rad/s, base frame)."""
        xy = self.unit_xy
        return velocity[:2] + velocity[2] * np.c_[-xy[:, 1], xy[:, 0]]

    def steering(self, start, goal):
        """Each unit's steering angles along a drive from start to goal (x, y,
        yaw), at progress STEER_PROGRESS, or None where they cannot stay within
        STEER_LIMIT. A unit steers along its own velocity, or the other way with
        the wheel rolling back: for the whole drive, the way that keeps its
        angles in range and starts nearest its present angle."""
        lines = []
        for p in STEER_PROGRESS:
            yaw = start[2] + (goal[2] - start[2]) * p
            c, s = np.cos(yaw), np.sin(yaw)
            back = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])  # Rz(-yaw)
            v = self.unit_velocity(back @ (goal - start))
            lines.append(np.arctan2(v[:, 1], v[:, 0]))
        # A wheel's line has period pi: it rolls either way along it.
        lines = np.unwrap(np.array(lines), axis=0, period=np.pi)
        paths = []
        for unit, line in enumerate(lines.T):
            near = np.round((self.steer_goal[unit] - line[0]) / np.pi)
            options = [line + k * np.pi for k in (near, near - 1, near + 1)]
            fit = [o for o in options if np.abs(o).max() <= STEER_LIMIT]
            if not fit:
                return None
            paths.append(min(fit, key=lambda o: abs(o[0] - self.steer_goal[unit])))
        return np.array(paths).T

    def drive(self, x, y, yaw, seconds=0.0, grips=None):
        """Drive the mobile base to x, y (m) and yaw (rad) from its start, on a
        straight line while it turns, as its swerve drive can; the arms hold their
        joint goals, so what the hands hold goes along. The wheels first turn to
        where the motion starts, then steer (steering()) and roll with it. A
        drive whose steering would leave the joint range (turning round while
        going far) goes in halves, the wheels turning again in between. It
        takes at least seconds and as long as BASE_LIMITS need."""
        start, goal = self.base_goal.copy(), np.array([x, y, yaw], float)
        if np.abs(goal - start).max() < 1e-9:
            self.hold_arms(seconds, grips, lambda p: None)
            return
        seconds = max(
            seconds,
            self.base_seconds(
                float(np.linalg.norm(goal[:2] - start[:2])), abs(goal[2] - start[2])
            ),
        )
        paths = self.steering(start, goal)
        if paths is None:
            half = (start + goal) / 2
            self.drive(*half, seconds / 2, grips)
            self.drive(*goal, seconds / 2, grips)
            return
        first = self.steer_goal.copy()
        turn = float(np.abs(paths[0] - first).max())
        if turn > 1e-3:

            def steer(p):
                self.steer_goal = first + (paths[0] - first) * p

            self.hold_arms(1.875 * turn / STEER_RATE, grips, steer)
        last, tick = start.copy(), 20 * self.m.opt.timestep

        def update(p):
            nonlocal last
            self.base_goal = start + (goal - start) * p
            c, s = np.cos(self.base_goal[2]), np.sin(self.base_goal[2])
            back = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            v = self.unit_velocity(back @ (self.base_goal - last) / tick)
            self.steer_goal = np.array(
                [np.interp(p, STEER_PROGRESS, path) for path in paths.T]
            )
            ahead = np.c_[np.cos(self.steer_goal), np.sin(self.steer_goal)]
            self.wheel_rate = (v * ahead).sum(1) / self.wheel_radius
            last = self.base_goal.copy()

        self.hold_arms(seconds, grips, update)
        self.wheel_rate = np.zeros(3)

    def prepare(self, targets, grips=None, rotations=None, seconds=3.0, lift=False):
        """Take the starting posture before the recording: move the arms to
        targets, let the head finish its gaze, and hold until the robot is
        still. Nothing is recorded meanwhile; the recording and its clock start
        after it (user request 2026-10-25: an episode starts in the robot's
        default posture, without the motion into it). The servos move the
        robot there; no state is written. lift=True first raises each hand, as
        it hangs, to its target's height above where it hangs (or the least
        step toward its target that the arm reaches), so that it does not swing
        through what stands in front of the robot; the arms still end on the
        joints the IK gives from where they started."""
        self.recording = False
        if lift:
            # In joint space: a straight line from the straight hanging arm
            # starts beside a singularity, where its 1 cm waypoints jumped and
            # stretch() made the lift take 45 s.
            for joints in self.lift_path(targets, rotations):
                self.move(grips=grips, seconds=seconds, joints=joints)
        else:
            self.move(targets, grips=grips, rotations=rotations, seconds=seconds)
        dofs = np.r_[
            self.dofs["left"],
            self.dofs["right"],
            [self.body_dofs[j] for j in BODY_SERVOS],
        ]
        # Checked at control ticks, so the recording starts on one (t = 0).
        still, limit = 0, self.d.time + PREPARE_LIMIT
        while self.d.time < GAZE_SECONDS or still < HOLD_TICKS or self.step_number % 20:
            if self.d.time > limit:
                raise RuntimeError("the robot did not come to rest before recording")
            self.physics_step()
            if self.step_number % 20 == 0:
                fast = np.abs(self.d.qvel[dofs]).max() > STILL_RATE
                still = 0 if fast else still + 1
        self.head_goal = self.head_command()
        self.t0 = float(self.d.time)
        self.recording = True

    def lift_path(self, targets, rotations=None):
        """The arm goals of prepare(lift=True) from the arms' goals now: each
        hand raised as it hangs to its target's height, then the IK of its
        target from where the arm is ([{side: raised}, {side: final}]). The
        raise goes straight above where the hand hangs, or the least step
        toward its target that the arm reaches: wheel_robot_26.8.16_3's hand
        hangs under the shoulder, where the arm cannot hold it up at the ready
        height."""
        rotations = rotations or {}
        final = {
            s: self.ik(s, xyz, rotation=rotations.get(s)) for s, xyz in targets.items()
        }
        raised = {}
        for s in targets:
            xyz, rotation = self.kin.fk(s, self.goal[s], self.body())
            for step in np.linspace(0.0, 1.0, 11):
                xy = (1 - step) * xyz[:2] + step * np.asarray(targets[s][:2])
                try:
                    raised[s] = self.ik(
                        s, [*xy, targets[s][2]], near=self.goal[s], rotation=rotation
                    )
                    break
                except RuntimeError:
                    if step == 1.0:
                        raise
        return [raised, final]

    def clock(self):
        """Seconds since the recording started."""
        return float(self.d.time) - self.t0

    def descends(self, body, ancestor):
        """Whether body is ancestor or lies below it in the body tree."""
        while body:
            if body == ancestor:
                return True
            body = int(self.m.body_parentid[body])
        return False

    def audit_contact(self, g1, g2, depth):
        """Log a touching contact if it is a collision (see hand_bodies)."""
        m = self.m
        b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
        r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
        if not (r1 or r2):
            return
        if not (r1 and r2):
            r, o, og = (b1, b2, g2) if r1 else (b2, b1, g1)
            if r in self.base_bodies and og in self.floor_geoms:
                return
            if r in self.hand_bodies and m.body_weldid[o] != 0:
                return
            pair = (m.body(r).name, m.body(o).name if o else m.geom(og).name)
        else:
            pair = tuple(sorted((m.body(b1).name, m.body(b2).name)))
        hit = self.collision_log.get(pair)
        if hit is None:
            hit = self.collision_log[pair] = {
                "bodies": list(pair),
                "first_s": round(self.clock(), 3),
                "phase": self.phase,
                "physics_steps": 0,
                "max_depth_mm": 0.0,
            }
        hit["physics_steps"] += 1
        hit["max_depth_mm"] = max(hit["max_depth_mm"], round(depth * 1000, 3))

    def collisions(self):
        """The run's collisions (audit_contact), first ones first."""
        return sorted(self.collision_log.values(), key=lambda h: h["first_s"])

    def stage(self, name, instruction):
        self.phase = name
        self.instruction = instruction
        print(f"{self.clock():6.2f}s {name}: {instruction}", flush=True)

    def capture(self):
        """The overview and the policy cameras' frames, by camera name."""
        images = {}
        for cam in ["overview", *POLICY_CAMERAS]:
            self.renderer.update_scene(self.d, camera=cam)
            images[cam] = self.renderer.render().copy()
        return images

    def capture_depth(self):
        """Each policy camera's depth (mm along the optical axis) and
        segmentation (instance ids, see segment_names) as 16-bit frames."""
        frames = {}
        for cam in POLICY_CAMERAS:
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.d, camera=cam)
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()
            self.renderer.enable_segmentation_rendering()
            self.renderer.update_scene(self.d, camera=cam)
            seg = self.renderer.render()
            self.renderer.disable_segmentation_rendering()
            hit = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
            ids = np.where(hit, self.geom_segment[np.clip(seg[..., 0], 0, None)], 0)
            frames[cam] = (
                np.clip(np.rint(depth * DEPTH_SCALE), 0, 65535).astype(np.uint16),
                ids.astype(np.uint16),
            )
        return frames

    def render(self):
        images = self.capture()
        canvas = Image.new("RGB", (1280, 720), (20, 27, 38))
        canvas.paste(Image.fromarray(images["overview"]).resize((960, 720)), (0, 0))
        canvas.paste(
            Image.fromarray(images["left_wrist_d405"]).resize((320, 240)), (960, 70)
        )
        canvas.paste(
            Image.fromarray(images["right_wrist_d405"]).resize((320, 240)), (960, 390)
        )
        font, small = overlay_font(21), overlay_font(17)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 1280, 65), fill=(20, 27, 38))
        draw.text(
            (18, 8),
            f"{self.LABEL} | DUAL-ARM EXPERT | {self.clock():05.2f}s | {self.phase.upper()}",
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
                f"{cam.upper()} | {self.clock():.2f}s",
                font=small,
                fill="white",
            )
        self.videos["cameras"].write(np.asarray(mosaic))
        for cam in POLICY_CAMERAS:
            self.videos[cam].write(images[cam])
        if self.RECORD_DEPTH:
            for cam, (depth, ids) in self.capture_depth().items():
                self.videos[f"depth_{cam}"].write(depth)
                self.videos[f"segmentation_{cam}"].write(ids)
        self.frames += 1

    def close(self):
        for video in self.videos.values():
            video.close()
        if self.renderer:
            self.renderer.close()
