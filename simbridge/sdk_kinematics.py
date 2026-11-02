"""The SDK's kinematics endpoints, answered by the shared host-side kinematics.

Fk/FkTool/IK and FKb/FKbTool/IKb/IKbTool come from robot_kinematics.Kinematics
(Pinocchio on the source URDF, and the expert's IK), so the project has one FK
and IK ("real follows sim": on the real robot these calls would run the
controller's own kinematics, which the host does not use). This module only
converts between the SDK's conventions and that model. From the vendor manual,
Python client and C++ headers:

- joints: SDK values, q_sdk = sign * q_urdf + offset (the active convention);
- frames: Fk gives the flange (MDH frame A7) in the arm's MDH base frame A0, and
  FKb gives it in base frame B (world-aligned at the base, robot_kinematics). A0
  in the parent link of the arm's first joint and A7 in wrist_roll_Link come
  from check_sdk_kinematics.py (config/robot/kinematic_crosscheck.json; with
  wheel_robot_26.8.16_3 both are those links' own frames, to 1e-6); the tool is
  the flange times flan2tool, the identity at start-up;
- transforms: 4x4, column-major 16-vectors, translations in m, at the float32
  precision of the controller's TransformMatrix4x4;
- poses: [x, y, z, rz, ry, rx] with R = Rx(rx) Ry(ry) Rz(rz)
  (SciPy from_euler("zyx", [rz, ry, rx])), read back with ry in [-pi/2, pi/2].

What is inferred rather than read from a controller is listed under
``kinematics`` in config/robot/franzi_sdk.yaml.
"""

import json
import threading
import xml.etree.ElementTree as ET

import numpy as np

from simbridge import ROOT
from simbridge.status import Rejected, Sdk


def to_wire(T):
    """4x4 -> column-major 16 floats at the controller's float32 precision."""
    return np.asarray(T, np.float32).T.reshape(-1).astype(float).tolist()


def from_wire(values):
    """Column-major 16 numbers -> 4x4 rigid transform, read as float32 like the
    controller. The rotation is snapped to the nearest proper rotation."""
    try:
        m = np.array([float(v) for v in values], np.float32)
    except (TypeError, ValueError):
        raise Rejected(Sdk.PARAM_INVALID, "transform must be 16 numbers") from None
    if m.shape != (16,) or not np.all(np.isfinite(m)):
        raise Rejected(Sdk.PARAM_INVALID, "transform must be 16 finite numbers")
    T = m.astype(float).reshape(4, 4).T
    R = T[:3, :3]
    if (
        np.abs(T[3] - [0, 0, 0, 1]).max() > 1e-6
        or np.abs(R.T @ R - np.eye(3)).max() > 1e-3
        or np.linalg.det(R) < 0
    ):
        raise Rejected(Sdk.PARAM_INVALID, "transform is not rigid (column-major?)")
    u, _, vt = np.linalg.svd(R)
    T[:3, :3] = u @ vt
    return T


def homogeneous(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def inverse(T):
    R, p = T[:3, :3], T[:3, 3]
    return homogeneous(R.T, -R.T @ p)


def matrix_from_pose(pose):
    """[x, y, z, rz, ry, rx] -> 4x4 with R = Rx(rx) Ry(ry) Rz(rz)."""
    x, y, z, rz, ry, rx = (float(v) for v in pose)
    cz, sz, cy, sy, cx, sx = (f(a) for a in (rz, ry, rx) for f in (np.cos, np.sin))
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return homogeneous(Rx @ Ry @ Rz, [x, y, z])


def pose_from_matrix(T):
    """4x4 -> [x, y, z, rz, ry, rx], ry in [-pi/2, pi/2] (as SciPy's
    as_euler("zyx"); at gimbal lock rx is set to 0, as SciPy does)."""
    R = np.asarray(T, float)[:3, :3]
    ry = float(np.arcsin(np.clip(R[0, 2], -1.0, 1.0)))
    if np.hypot(R[0, 0], R[0, 1]) > 1e-9:
        rz = float(np.arctan2(-R[0, 1], R[0, 0]))
        rx = float(np.arctan2(-R[1, 2], R[2, 2]))
    else:
        rz, rx = float(np.arctan2(R[1, 0], R[1, 1])), 0.0
    return [*np.asarray(T, float)[:3, 3].tolist(), rz, ry, rx]


class SdkKinematics:
    """SDK frames and joint values around robot_kinematics.Kinematics, plus the
    session's flange -> tool transforms. Raises ImportError where Pinocchio is
    missing (the services then report the endpoints as unsupported)."""

    def __init__(self, robot):
        from robot_kinematics import URDF, Kinematics

        self.robot = robot
        self.cfg = robot.raw["kinematics"]
        # The endpoints emulate the controller's IK, which knows no planning
        # margin: solutions may sit anywhere inside the limits MoveJ accepts,
        # which keep robot.yaml's position margin from the URDF bounds too
        # (wheel_robot_26.8.16_3's wrist roll stops at 1.57, inside the
        # controller's 1.5708). Host-side planning (sdk.sdk_cartesian, the
        # experts) keeps the larger default.
        self.kin = Kinematics(margin=robot.position_margin + 0.0001)
        self.lock = threading.Lock()  # Kinematics keeps one Pinocchio data
        self.waist = robot.units["waist"]
        model = self.kin.model
        # The arm's base link (the first joint's parent: left_j0_Link) and the
        # grasp point in the wrist_roll frame, a fixed transform through the
        # gripper link.
        self.parent, self.grasp_in_wrist = {}, {}
        parents = {
            j.get("name"): j.find("parent").get("link")
            for j in ET.parse(URDF).getroot().findall("joint")
        }
        for side in robot.arms:
            self.parent[side] = model.getFrameId(
                parents[robot.arms[side].sim_joints[0]]
            )
            frame = model.frames[self.kin.frames[side]]
            self.grasp_in_wrist[side] = homogeneous(
                frame.placement.rotation, frame.placement.translation
            )
        self.flan2tool = {side: np.eye(4) for side in robot.arms}
        evidence = json.loads(
            (ROOT / robot.raw["joint_convention"]["evidence"]).read_text()
        )
        self.a0_in_parent, self.flange_in_wrist = {}, {}
        for side in robot.arms:
            f = evidence["arms"][side]["selected"]["frames"]
            base, flange = f["mdh_base_in_parent"], f["mdh_flange_in_child_link"]
            self.a0_in_parent[side] = homogeneous(base["R"], base["p_m"])
            self.flange_in_wrist[side] = homogeneous(flange["R"], flange["p_m"])

    def _body(self, torso_sdk):
        if torso_sdk is None:
            return None  # upright, as robot_kinematics assumes
        return dict(zip(self.waist.sim_joints, self.waist.to_sim(torso_sdk)))

    def _world(self, side, q_sdk, body):
        """World poses of wrist_roll_Link and of the arm's base link at SDK
        joint values."""
        q = self.robot.arms[side].to_sim(q_sdk)
        with self.lock:
            p, R = self.kin.fk(side, q, body)
            base = self.kin.data.oMf[self.parent[side]]
            base = homogeneous(base.rotation, base.translation + self.kin.base)
        return homogeneous(R, p) @ inverse(self.grasp_in_wrist[side]), base

    def _solve(self, side, wrist, seed_sdk, body):
        """SDK joints putting wrist_roll_Link at wrist (4x4, world)."""
        grasp = wrist @ self.grasp_in_wrist[side]
        current = self.robot.arms[side].to_sim(seed_sdk)
        try:
            with self.lock:
                q = self.kin.ik(
                    side, grasp[:3, 3], current, body, rotation=grasp[:3, :3]
                )
        except RuntimeError as exc:
            raise Rejected(Sdk.UNREACHABLE, str(exc)) from None
        return self.robot.arms[side].to_sdk(q)

    def _tool(self, side, T, tool):
        return T @ self.flan2tool[side] if tool else T

    def _flange(self, side, T, tool):
        return T @ inverse(self.flan2tool[side]) if tool else T

    # ----- arm frame A0 (moves with the torso, so the waist does not matter) ---
    def fk(self, side, q_sdk, tool=False):
        wrist, base = self._world(side, q_sdk, None)
        T = inverse(base @ self.a0_in_parent[side]) @ wrist
        return self._tool(side, T @ self.flange_in_wrist[side], tool)

    def ik(self, side, target, seed_sdk, tool=True):
        """SDK joints putting the tool (or flange) at target, a 4x4 in A0;
        robot_kinematics.ik seeded with seed_sdk."""
        _, base = self._world(side, seed_sdk, None)
        flange = base @ self.a0_in_parent[side] @ self._flange(side, target, tool)
        wrist = flange @ inverse(self.flange_in_wrist[side])
        return self._solve(side, wrist, seed_sdk, None)

    # ----- base frame B = URDF base_link (world-aligned, fixed base) ------------
    def fkb(self, side, q_sdk, torso_sdk, tool=False):
        wrist, _ = self._world(side, q_sdk, self._body(torso_sdk))
        T = wrist @ self.flange_in_wrist[side]
        T[:3, 3] -= self.kin.base
        return self._tool(side, T, tool)

    def ikb(self, side, target, torso_sdk, seed_sdk, tool=False):
        flange = self._flange(side, target, tool).copy()
        flange[:3, 3] += self.kin.base
        wrist = flange @ inverse(self.flange_in_wrist[side])
        return self._solve(side, wrist, seed_sdk, self._body(torso_sdk))

    def describe(self):
        return {
            "model": "robot_kinematics.Kinematics (shared host kinematics)",
            "flan2tool": {s: to_wire(T) for s, T in self.flan2tool.items()},
            "inferred": list(self.cfg["inferred"]),
        }
