"""Shared host-side kinematics: one FK and IK for the expert, simbridge and the robot.

"Real follows sim": Cartesian goals are solved here, and the simulation and the
real controller both receive joint commands only, so the controller's own IK never
picks a pose. FK and Jacobians come from Pinocchio on the source URDF, with each
{side}_grasp tool point and base_link placed as in the simulation; FK equals the
MuJoCo scene's to 1e-15 m and the controller image's MDH table to 1e-14, and the
user found the URDF to match the real robot. Joint limits are the controller-
aligned ranges the scene uses. Needs NumPy, SciPy, PyYAML and Pinocchio (pin), as
in the inovance-fineart env; no physics engine, so simbridge services and a
real-robot host can import it.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "assets/franzi_source"
# Controller limits read from the controller image, the SDK <-> URDF joint mapping
# and the tool point, shared with simbridge.
ROBOT_CONFIG = ROOT / "config/robot/franzi_sdk.yaml"
ARM_JOINTS = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_pitch",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
]
# Leg and waist joints between the fixed base and the shoulders.
BODY_JOINTS = (
    "calf_pitch_joint",
    "thigh_pitch_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
)


def controller_ranges():
    """Source-model joint ranges clipped to the stored controller limits.

    Each range becomes the intersection with the controller's (mapped
    q_urdf = (q_sdk - offset) / sign). Sides where the controller is tighter are
    pulled in by the SDK's position margin, so a plan never sits on a bound the
    SDK layer rejects. Returns {joint: {urdf_rad, controller_rad, aligned_rad}}.
    """
    config = yaml.safe_load(ROBOT_CONFIG.read_text())
    margin = config["limits"]["position_margin_rad"]
    source = ET.parse(SOURCE / "franzi.xml").getroot()
    joints = {j.get("name"): j for j in source.iter("joint")}
    out = {}
    for unit in config["units"].values():
        mapping = unit["mapping"]["controller_mdh"]
        for name, limits, sign, offset in zip(
            unit["sim_joints"],
            unit["controller_limits_deg"],
            mapping["sign"],
            mapping["offset_rad"],
        ):
            lo, hi = map(float, joints[name].get("range").split())
            c_lo, c_hi = sorted((np.radians(v) - offset) / sign for v in limits)
            new = (
                c_lo + margin if c_lo > lo else lo,
                c_hi - margin if c_hi < hi else hi,
            )
            out[name] = {
                "urdf_rad": [lo, hi],
                "controller_rad": [float(c_lo), float(c_hi)],
                "aligned_rad": [float(v) for v in new],
            }
    return out


class Kinematics:
    """FK and IK of each arm's grasp point in the world frame (fixed base).

    margin (rad) keeps IK solutions that far inside the aligned limits. A plan
    that rests on a limit can overshoot it by more than the scene's 2 mrad audit
    while the servo tracks (the kitting task plans with 10 mrad)."""

    def __init__(self, margin=0.0001):
        self.margin = margin
        self.model = pin.buildModelFromUrdf(str(SOURCE / "wheel_robot_4.0.urdf"))
        source = ET.parse(SOURCE / "franzi.xml").getroot()
        base = source.find(".//body[@name='base_link']")
        self.base = np.array(base.get("pos").split(), float)  # URDF root in world
        config = yaml.safe_load(ROBOT_CONFIG.read_text())
        tcp = config["frames"]["tcp"]["position_in_wrist_roll_link_m"]
        ranges = controller_ranges()
        self.neutral = pin.neutral(self.model)
        self.body_q = {
            j: self.model.joints[self.model.getJointId(j)].idx_q for j in BODY_JOINTS
        }
        self.frames, self.idx_q, self.idx_v, self.limits, self.mapping = (
            {},
            {},
            {},
            {},
            {},
        )
        for side in ("left", "right"):
            names = [f"{side}_{j}_joint" for j in ARM_JOINTS]
            joints = [self.model.joints[self.model.getJointId(n)] for n in names]
            self.idx_q[side] = np.array([j.idx_q for j in joints])
            self.idx_v[side] = np.array([j.idx_v for j in joints])
            placement = pin.SE3(np.eye(3), np.array(tcp[side], float))
            self.frames[side] = self.model.addFrame(
                pin.Frame(
                    f"{side}_grasp", joints[-1].id, 0, placement, pin.FrameType.OP_FRAME
                )
            )
            self.limits[side] = np.array([ranges[n]["aligned_rad"] for n in names])
            unit = config["units"][side]["mapping"]["controller_mdh"]
            self.mapping[side] = (
                np.array(unit["sign"], float),
                np.array(unit["offset_rad"], float),
            )
        self.data = self.model.createData()

    def config(self, side, q, body=None):
        """Full Pinocchio configuration: the arm pose q, leg and waist joints from
        body (default upright), everything else neutral."""
        full = self.neutral.copy()
        for name, value in (body or {}).items():
            full[self.body_q[name]] = value
        full[self.idx_q[side]] = q
        return full

    def fk(self, side, q, body=None):
        """Grasp point position and frame (p, R) in the world."""
        pin.framesForwardKinematics(self.model, self.data, self.config(side, q, body))
        M = self.data.oMf[self.frames[side]]
        return M.translation + self.base, M.rotation.copy()

    def ik(self, side, target, current, body=None, near=None, rotation=None):
        """Arm joints putting the grasp point at target (world, m) with its frame
        at rotation (default world-aligned), margin inside the aligned limits.

        current is the present arm pose, which breaks ties between solutions. near
        is the previous waypoint's solution when chaining a line: the first
        feasible candidate from it is kept, so the arm stays on its branch.
        Raises RuntimeError when no candidate converges.
        """
        vr = self.limits[side]
        target = np.asarray(target, float)
        goal = np.eye(3) if rotation is None else np.asarray(rotation, float)
        current = np.asarray(current if near is None else near, float)
        full = self.config(side, current, body)
        qi, vi, frame = self.idx_q[side], self.idx_v[side], self.frames[side]
        # The arm has one redundant DOF (elbow swivel). Chained waypoints get a
        # tiny pull toward the previous solution; without it the swivel jumped
        # between waypoints and the arm visibly wobbled along straight moves.
        swivel = 0.002 if near is not None else 0.0
        cache = {}

        def evaluate(q):
            # Residual and its exact Jacobian from one FK pass. Position rows are
            # scaled by 10; the rotation error is the rotation vector, whose
            # gradient stays nonzero at a pi error.
            key = q.tobytes()
            if key not in cache:
                full[qi] = q
                pin.framesForwardKinematics(self.model, self.data, full)
                M = self.data.oMf[frame]
                J = pin.computeFrameJacobian(
                    self.model, self.data, full, frame, pin.LOCAL_WORLD_ALIGNED
                )[:, vi]
                error = goal.T @ M.rotation
                cache.clear()
                cache[key] = (
                    np.r_[
                        (M.translation + self.base - target) * 10,
                        pin.log3(error),
                        (q - current) * swivel,
                    ],
                    np.vstack(
                        [
                            J[:3] * 10,
                            pin.Jlog3(error) @ M.rotation.T @ J[3:],
                            np.eye(len(q)) * swivel,
                        ]
                    ),
                )
            return cache[key]

        sign = 1 if side == "left" else -1
        seeds = [
            current,
            np.array([0.2, 0.5 * sign, -0.3 * sign, -1.6, 1.4 * sign, 1.0, 0.6 * sign]),
            np.array([0.5, 1.0 * sign, 0, -2, 0, 1.0, 0]),
            np.array([-0.4, 0.3 * sign, 0, -1.2, 0, 1.0, 0]),
        ]
        candidates = []
        inset = max(0.001, 2 * self.margin)  # seeds start inside the bounds
        for seed in seeds:
            try:
                res = least_squares(
                    lambda q: evaluate(q)[0],
                    np.clip(seed, vr[:, 0] + inset, vr[:, 1] - inset),
                    jac=lambda q: evaluate(q)[1],
                    bounds=(vr[:, 0] + self.margin, vr[:, 1] - self.margin),
                    max_nfev=220,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
            except ValueError:
                # A seed whose rotation error reaches pi has no finite Jlog3.
                continue
            err = np.linalg.norm(res.fun[:3]) / 10
            ori = np.linalg.norm(res.fun[3:6])
            if err < 0.0002 and ori < 0.003:
                candidates.append((np.linalg.norm(res.x - current), res.x))
                # A waypoint continues from its predecessor: keep that branch.
                if near is not None and seed is seeds[0]:
                    break
        if not candidates:
            raise RuntimeError(f"{side} IK infeasible at {target.tolist()}")
        return min(candidates, key=lambda c: c[0])[1]

    def line(self, side, start, target, body=None, rotation=None, step=0.01):
        """Joint waypoints from arm pose start to the grasp point at target along
        the Cartesian line, IK every step metres, each chained to the last."""
        begin = self.fk(side, start, body)[0]
        count = int(np.ceil(np.linalg.norm(np.subtract(target, begin)) / step))
        path = [np.asarray(start, float)]
        for point in np.linspace(begin, target, max(1, count) + 1)[1:]:
            path.append(
                self.ik(side, point, path[-1], body, near=path[-1], rotation=rotation)
            )
        return np.array(path)

    def to_sdk(self, side, q):
        """URDF arm joints (rad) to the vendor SDK's joint values (rad)."""
        sign, offset = self.mapping[side]
        return sign * np.asarray(q, float) + offset

    def from_sdk(self, side, q_sdk):
        """Vendor SDK joint values (rad) to URDF arm joints (rad)."""
        sign, offset = self.mapping[side]
        return (np.asarray(q_sdk, float) - offset) / sign
