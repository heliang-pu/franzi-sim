"""Shared host-side kinematics: one FK and IK for the expert, simbridge and the robot.

"Real follows sim": Cartesian goals are solved here, and the simulation and the
real controller both receive joint commands only, so the controller's own IK never
picks a pose. FK and Jacobians come from Pinocchio on the source URDF
(wheel_robot_26.8.16_3 since 2026-10-25, exported in the controller's joint
convention), with each {side}_grasp tool point and the URDF root placed as in
the simulation (robots/franzi/model.py). Joint limits are the controller-
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

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "assets/urdf"
URDF = SOURCE / "wheel_robot_26.8.16_3.urdf"
# The base frame B: world-aligned, BASE_HEIGHT above the floor under the base
# centre, where the previous URDF's root was. The new root lies on its back
# (x up, y forward, z left) on the calf axis, ROOT_IN_BASE in B. Pinocchio
# works in B; the world is B plus base (Kinematics.base).
BASE_HEIGHT = 0.0872648
ROOT_IN_BASE = (
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float),
    np.array([0, 0, 0.24785]),
)
# Controller limits read from the controller image, the SDK <-> URDF joint mapping
# and the tool point, shared with simbridge.
ROBOT_CONFIG = HERE / "config/franzi_sdk.yaml"
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
# Default IK planning margin inside the aligned limits (user decision 2026-10-24).
# The pick-and-place approach otherwise parks both wrist pitches 0.065 deg from
# the controller's -60 deg limit, where tracking overshoot on the real robot
# could trip its soft limit.
PLANNING_MARGIN = float(np.radians(2.0))


def controller_ranges():
    """Source-model joint ranges clipped to the stored controller limits.

    Each range becomes the intersection with the controller's (mapped
    q_urdf = (q_sdk - offset) / sign). Sides where the controller is tighter are
    pulled in by the SDK's position margin, so a plan never sits on a bound the
    SDK layer rejects. Returns {joint: {urdf_rad, controller_rad, aligned_rad}}.
    """
    config = yaml.safe_load(ROBOT_CONFIG.read_text())
    margin = config["limits"]["position_margin_rad"]
    joints = {j.get("name"): j for j in ET.parse(URDF).getroot().findall("joint")}
    out = {}
    for unit in config["units"].values():
        mapping = unit["mapping"]["controller_mdh"]
        for name, limits, sign, offset in zip(
            unit["sim_joints"],
            unit["controller_limits_deg"],
            mapping["sign"],
            mapping["offset_rad"],
        ):
            limit = joints[name].find("limit")
            lo, hi = float(limit.get("lower")), float(limit.get("upper"))
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


def controller_rates():
    """Arm joint rate limits stored in the controller image, in ARM_JOINTS order:
    velocity (rad/s) and acceleration (rad/s^2) at a 100% speed ratio, and the
    controller's default speed ratio (0 to 1). Not verified on hardware."""
    config = yaml.safe_load(ROBOT_CONFIG.read_text())
    rates = config["limits"]["controller_archive_rates_deg"]
    return {
        "velocity": np.radians(rates["velocity"]),
        "acceleration": np.radians(rates["acceleration"]),
        "default_ratio": rates["default_velocity_ratio_percent"] / 100,
    }


def root_joints():
    """The URDF's movable joints that Pinocchio hangs off its universe: those
    whose parent link reaches the root link through fixed joints only."""
    joints = ET.parse(URDF).getroot().findall("joint")
    into = {j.find("child").get("link"): j for j in joints}
    names = []
    for joint in joints:
        if joint.get("type") == "fixed":
            continue
        link = joint.find("parent").get("link")
        while link in into and into[link].get("type") == "fixed":
            link = into[link].find("parent").get("link")
        if link not in into:
            names.append(joint.get("name"))
    return names


class Kinematics:
    """FK and IK of each arm's grasp point in the world frame (fixed base).

    margin (rad) keeps IK solutions that far inside the aligned limits, by
    default PLANNING_MARGIN. A plan that rests on a limit can overshoot it while
    the servo tracks. simbridge's endpoints, which emulate the controller's own
    IK, pass 0.0001; the kitting task passes 10 and 20 mrad, because its inner
    cells put the shoulder roll on a limit."""

    def __init__(self, margin=PLANNING_MARGIN):
        self.margin = margin
        self.model = pin.buildModelFromUrdf(str(URDF))
        # Into B: everything hung off the root moves with it. The joints come
        # from the URDF tree: Isaac's Pinocchio 2.7 can't index model.parents
        # inside Kit, and calls a frame's parentJoint parent.
        root = pin.SE3(*ROOT_IN_BASE)
        for name in root_joints():
            j = self.model.getJointId(name)
            self.model.jointPlacements[j] = root * self.model.jointPlacements[j]
        for f in range(self.model.nframes):
            frame = self.model.frames[f]
            parent = (
                frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
            )
            if parent == 0:
                frame.placement = root * frame.placement
                self.model.frames[f] = frame
        self.base = np.array([0.0, 0.0, BASE_HEIGHT])  # B in the world
        config = yaml.safe_load(ROBOT_CONFIG.read_text())
        tcp = config["frames"]["tcp"]
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
            # The tool point in its gripper link, which is welded to the wrist roll.
            gripper = self.model.getFrameId(tcp["parent_link"][side])
            placement = self.model.frames[gripper].placement * pin.SE3(
                np.array(tcp["rotation"][side], float),
                np.array(tcp["position_m"][side], float),
            )
            self.frames[side] = self.model.addFrame(
                pin.Frame(
                    f"{side}_grasp",
                    joints[-1].id,
                    gripper,
                    placement,
                    pin.FrameType.OP_FRAME,
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

    def _residual(self, side, target, goal, current, chained, body):
        """The IK's residual and exact Jacobian at q, from one FK pass."""
        full = self.config(side, current, body)
        qi, vi, frame = self.idx_q[side], self.idx_v[side], self.frames[side]
        # The arm has one redundant DOF (elbow swivel). Chained waypoints get a
        # tiny pull toward the previous solution; without it the swivel jumped
        # between waypoints and the arm visibly wobbled along straight moves.
        swivel = 0.002 if chained else 0.0
        cache = {}

        def evaluate(q):
            # Position rows are scaled by 10; the rotation error is the rotation
            # vector, whose gradient stays nonzero at a pi error.
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

        return evaluate

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
        evaluate = self._residual(side, target, goal, current, near is not None, body)

        # The fixed seeds in wheel_robot_26.8.16_3's convention: wheel_robot_4.0's
        # times its SDK signs [-1, 1, -1, -1, -1, -1, 1], the same arm postures.
        # Left in the old convention, their elbows lay outside the new range,
        # clipped onto its bound, and alone they solved 21-36% of the reachable
        # targets on a grid (these: 44-98%).
        sign = 1 if side == "left" else -1
        seeds = [
            current,
            np.array(
                [-0.2, 0.5 * sign, 0.3 * sign, 1.6, -1.4 * sign, -1.0, 0.6 * sign]
            ),
            np.array([-0.5, 1.0 * sign, 0, 2.0, 0, -1.0, 0]),
            np.array([0.4, 0.3 * sign, 0, 1.2, 0, -1.0, 0]),
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

    def follow(self, side, target, rotation, previous, body=None, step=0.1):
        """Arm joints for one target of a stream a learned policy sends (sla.serve):
        from the previous solution, the pose nearest to target (world, m) and
        rotation that the margin-limited joints reach within step (rad) of it.
        Unlike ik() it never raises or changes branch, since a policy's target can
        lie out of reach and the arm must not jump. Without the step bound, expert
        hand paths replayed this way flipped a wrist by up to 0.6 rad in one tick
        near a singularity (the expert's largest step is 0.034), and one such flip
        later put an upper arm into the torso. Returns the joints, the position
        error (m) and the rotation error (rad)."""
        vr = self.limits[side]
        goal = np.asarray(rotation, float)
        previous = np.asarray(previous, float)
        evaluate = self._residual(
            side, np.asarray(target, float), goal, previous, True, body
        )
        inset = max(0.001, 2 * self.margin)
        start = np.clip(previous, vr[:, 0] + inset, vr[:, 1] - inset)
        try:
            res = least_squares(
                lambda q: evaluate(q)[0],
                start,
                jac=lambda q: evaluate(q)[1],
                bounds=(
                    np.maximum(vr[:, 0] + self.margin, start - step),
                    np.minimum(vr[:, 1] - self.margin, start + step),
                ),
                max_nfev=50,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
        except ValueError:
            # A rotation error of pi has no finite Jlog3: hold the arm.
            return previous.copy(), float("inf"), float("inf")
        return (
            res.x,
            float(np.linalg.norm(res.fun[:3]) / 10),
            float(np.linalg.norm(res.fun[3:6])),
        )

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
