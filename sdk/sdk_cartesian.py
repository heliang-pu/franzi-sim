"""Host-side Cartesian goals for the stock SDK ("real follows sim").

Goals are solved on this host with the shared kinematics (robot_kinematics:
Pinocchio on the source URDF, the expert's IK) and sent as joint commands in
SDK coordinates: MoveJ for a point-to-point move, the joint teleop queue for a
straight line. A simulator and the real controller therefore receive the same
numbers; neither the controller's IK nor MoveL or Cartesian teleop is used.

    from sdk.stock_sdk import connect
    from sdk.sdk_cartesian import CartesianArm

    arm, sensors = connect("config/connections/sim_mujoco.yaml")
    cart = CartesianArm(arm)
    p, R = cart.pose("left")
    cart.move({"left": p + [0, 0, 0.05]})          # MoveJ, waits for arrival
    cart.line({"left": p + [0.04, 0, 0.05]})       # straight line, teleop queue

Targets are grasp points ({side}_grasp, frames.tcp in
config/robot/franzi_sdk.yaml) in the simulation's world frame: base_link sits at
the franzi.xml offset (``CartesianArm.kin.base``) with world-aligned axes. The
grasp frame keeps its current rotation unless ``rotations`` gives a 3x3 per
side. The measured waist joints enter every solve. With raise_on_error=True the
stock client may raise on a rejected call before this module sees the status.
"""

import time

import numpy as np
import yaml

from robot_kinematics import ROBOT_CONFIG, Kinematics

SIDES = {"left": 0, "right": 10}
WAIST, DUAL = 20, 15


class CartesianError(RuntimeError):
    """A goal that has no IK solution, or a command the SDK rejected."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class CartesianArm:
    """Cartesian goals for a stock ``ArmClient``, sent as SDK joint commands."""

    def __init__(self, arm, kinematics=None, timeout_s=60.0, poll_s=0.04):
        self.arm = arm
        self.kin = kinematics or Kinematics()
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        waist = yaml.safe_load(ROBOT_CONFIG.read_text())["units"]["waist"]
        mapping = waist["mapping"]["controller_mdh"]
        self.waist_joints = waist["sim_joints"]
        self.waist_sign = np.asarray(mapping["sign"], float)
        self.waist_offset = np.asarray(mapping["offset_rad"], float)

    # ----- state ------------------------------------------------------------------
    def joints(self, side):
        """Measured arm joints of side, in URDF radians."""
        status, q = self.arm.GetJointState(SIDES[side])
        if status != 0 or len(q) != 7:
            raise CartesianError(f"GetJointState({SIDES[side]}) failed", status)
        return self.kin.from_sdk(side, q)

    def body(self):
        """Measured leg and waist joints as robot_kinematics body values."""
        status, q = self.arm.GetJointState(WAIST)
        if status != 0 or len(q) != 4:
            raise CartesianError(f"GetJointState({WAIST}) failed", status)
        q = (np.asarray(q, float) - self.waist_offset) / self.waist_sign
        return dict(zip(self.waist_joints, q))

    def pose(self, side):
        """Grasp point (p, R) of side in the world, from the measured joints."""
        return self.kin.fk(side, self.joints(side), self.body())

    # ----- goals --------------------------------------------------------------------
    def solve(self, targets, rotations=None):
        """SDK joints putting each side's grasp point at its target: {side: q}."""
        body, rotations = self.body(), rotations or {}
        goals = {}
        for side, target in targets.items():
            current = self.joints(side)
            rotation = rotations.get(side)
            if rotation is None:
                rotation = self.kin.fk(side, current, body)[1]
            try:
                q = self.kin.ik(side, target, current, body, rotation=rotation)
            except RuntimeError as exc:
                raise CartesianError(str(exc)) from None
            goals[side] = self.kin.to_sdk(side, q)
        return goals

    def move(self, targets, rotations=None, wait=True):
        """One MoveJ to the solved joints (both arms together when both are
        given). Returns the SDK joint goals; waits for arrival unless told not
        to, raising if the motion ends in a fault (-1) or a stall (-3)."""
        goals = self.solve(targets, rotations)
        side = self._side(goals)
        point = np.concatenate([goals[s] for s in SIDES if s in goals]).tolist()
        self._check(self.arm.MoveJ([point], side=side), "MoveJ")
        if wait:
            self.wait(side)
        return goals

    def line(self, targets, rotations=None, speed=0.05, rate=25.0, wait=True):
        """Straight lines of the grasp points, streamed through the joint teleop
        queue at ``rate`` Hz: robot_kinematics.line waypoints ``speed / rate``
        apart, each side resampled to the same count, and slowed down if a joint
        would exceed 0.9 of the SDK's current velocity limit (GetJointVelLimit,
        which includes the velocity ratio). Returns the SDK joint paths and the
        slow-down factor."""
        body, rotations = self.body(), rotations or {}
        paths = {}
        for side, target in targets.items():
            start = self.joints(side)
            rotation = rotations.get(side)
            if rotation is None:
                rotation = self.kin.fk(side, start, body)[1]
            try:
                paths[side] = self.kin.line(
                    side, start, target, body, rotation=rotation, step=speed / rate
                )
            except RuntimeError as exc:
                raise CartesianError(str(exc)) from None
        count = max(len(p) for p in paths.values())
        slowdown = 1.0
        for side, path in paths.items():
            path = _resample(path, count)
            limit = np.deg2rad(np.asarray(self.arm.GetJointVelLimit(SIDES[side])))
            peak = np.max(np.abs(np.diff(path, axis=0)) * rate / limit)
            slowdown = max(slowdown, peak / 0.9)
        count = int(np.ceil((count - 1) * slowdown)) + 1
        paths = {s: self.kin.to_sdk(s, _resample(p, count)) for s, p in paths.items()}
        side = self._side(paths)
        dual = side == DUAL
        if dual:
            self._check(self.arm.TeleJointDual(), "TeleJointDual")
        else:
            self._check(self.arm.TeleJoint(side), "TeleJoint")
            (single,) = paths.values()
        try:
            tick = time.monotonic()
            for k in range(1, count):  # point 0 is where the arm already is
                if dual:
                    status = self.arm.PushJointTeleopQueueDual(
                        paths["left"][k].tolist(), paths["right"][k].tolist()
                    )
                else:
                    status = self.arm.PushJointTeleopQueue(side, single[k].tolist())
                self._check(status, "PushJointTeleopQueue")
                tick += 1.0 / rate
                time.sleep(max(0.0, tick - time.monotonic()))
            if wait:
                self.wait(side)
        finally:
            if dual:
                self.arm.StopTeleJointDual()
            else:
                self.arm.StopTeleJoint(side)
        return {"paths": paths, "slowdown": slowdown, "points": count}

    def wait(self, side=DUAL):
        """Poll GetMotionStatus until the motion leaves 1 (running)."""
        deadline = time.monotonic() + self.timeout_s
        while True:
            code = self.arm.GetMotionStatus(side)
            if code != 1:
                if code != 0:
                    raise CartesianError(f"motion ended with status {code}", code)
                return code
            if time.monotonic() > deadline:
                raise CartesianError("motion did not finish in time", code)
            time.sleep(self.poll_s)

    # ----- helpers ------------------------------------------------------------------
    @staticmethod
    def _side(goals):
        if set(goals) == set(SIDES):
            return DUAL
        if len(goals) != 1 or next(iter(goals)) not in SIDES:
            raise CartesianError(f"targets must name left, right or both: {goals}")
        return SIDES[next(iter(goals))]

    @staticmethod
    def _check(status, method):
        if status != 0:
            raise CartesianError(f"{method} rejected with status {status}", status)


def _resample(path, count):
    """Joint path linearly resampled to count points (same start and end)."""
    path = np.asarray(path, float)
    if len(path) == count:
        return path
    old = np.linspace(0.0, 1.0, len(path))
    new = np.linspace(0.0, 1.0, count)
    return np.stack([np.interp(new, old, path[:, j]) for j in range(path.shape[1])], 1)
