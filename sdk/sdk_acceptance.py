"""Acceptance program written only against the unmodified V1R1C9 SDK.

    python -m sdk.sdk_acceptance --connection config/connections/sim_mujoco.yaml \
        --output outputs/<new_run> [--task] [--fault-probe]

The same file drives every backend; only the connection profile changes. It
records every SDK call (action log) and the feedback it returned, and groups
its checks by what they establish: communication compatibility, control
semantics, the controller's kinematics interface, physical behaviour and
full-task execution. It never sees ground
truth: whether the task succeeded is judged afterwards from the simulator
trace (verify_backends.py), not by this program.

Motions here are tuned for simulation and have not been validated on the
physical robot, so the program refuses non-loopback hosts.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sdk.stock_sdk import ROOT, connect, read_profile

CAMERAS = ("head", "waist", "wrist_left", "wrist_right")
UNSUPPORTED_PROBES = {
    "GetJointJerkLimit": (0,),
    "MoveL": ([[0.3, 0.2, 0.9, 0.0, 0.0, 0.0]], 0),
    "SetMdh": (0, [[0.0] * 4] * 7),
    "GetMdh": (0,),
    "FkLink": (0, [0.0] * 7, 7),
    "GetArmEulerAngle": (0,),
    "GripperSetForce": (0, 50),
    "GetRobotState": (),
    "StartImpedance": (0,),
    "TeleCartesian": (0,),
}
# Cartesian goal of the kinematics stage: both flanges this far up in frame B.
LIFT_M = 0.03


def brief(value):
    """JSON-friendly summary of an SDK return value for the action log."""
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {k: brief(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [brief(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {k: brief(getattr(value, k)) for k in value.__dataclass_fields__}
    return value


class Session:
    def __init__(self, arm, sensors, out):
        self.arm = arm
        self.sensors = sensors
        self.out = out
        self.start = time.time()
        self.calls = []
        self.checks = []

    def call(self, method, *args, client="arm", expect_error=False, **kwargs):
        target = self.arm if client == "arm" else self.sensors
        t0 = time.time()
        error = result = None
        try:
            result = getattr(target, method)(*args, **kwargs)
        except Exception as exc:  # logged, re-raised unless expected
            error = f"{type(exc).__name__}: {exc}"
            if not expect_error:
                self._log(method, args, kwargs, result, error, t0)
                raise
        self._log(method, args, kwargs, result, error, t0)
        return error if expect_error else result

    def _log(self, method, args, kwargs, result, error, t0):
        self.calls.append(
            {
                "t_s": round(t0 - self.start, 4),
                "method": method,
                "args": brief(list(args)),
                "kwargs": brief(kwargs),
                "result": brief(result),
                "error": error,
                "latency_ms": round((time.time() - t0) * 1000, 2),
            }
        )

    def check(self, category, name, passed, **evidence):
        self.checks.append(
            {"category": category, "name": name, "passed": bool(passed), **evidence}
        )
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {category}: {name}", flush=True)
        return bool(passed)

    def joints(self, side=15):
        return np.asarray(self.call("GetJointState", side)[1], float)

    def wait_motion(self, side=15, timeout=60.0):
        """Poll GetMotionStatus until it leaves 1; return code, seconds, codes seen."""
        t0 = time.monotonic()
        seen = []
        while time.monotonic() - t0 < timeout:
            code = self.call("GetMotionStatus", side)
            if not seen or seen[-1] != code:
                seen.append(code)
            if code != 1:
                return code, time.monotonic() - t0, seen
            time.sleep(0.04)
        return 1, timeout, seen

    def wait_gripper(self, side, timeout=20.0):
        t0 = time.monotonic()
        state = None
        while time.monotonic() - t0 < timeout:
            state = self.call("GripperReadGripState", side)["value"]
            if state != 0:
                break
            time.sleep(0.04)
        return state, self.call("GripperReadPosition", side)["value"]

    def stream(self, start, goal, count, dual=True, side=0, rate=25.0):
        """Teleop points from start to goal on an absolute wall-clock schedule
        (as the vendor's arm_joint_stream.py demo paces its pushes)."""
        codes = []
        tick = time.monotonic()
        for k in range(1, count + 1):
            q = start + (goal - start) * k / count
            if dual:
                codes.append(
                    self.call(
                        "PushJointTeleopQueueDual", q[:7].tolist(), q[7:].tolist()
                    )
                )
            else:
                codes.append(self.call("PushJointTeleopQueue", side, q.tolist()))
            tick += 1 / rate
            time.sleep(max(0.0, tick - time.monotonic()))
        return codes


def core(s, program):
    c = "communication"
    version = s.call("GetVersion")
    s.check(c, "GetVersion returns a string", isinstance(version, str), version=version)
    s.check(c, "IsReady true at start", s.call("IsReady"))
    s.check(c, "AxisCount is the 20-axis whole body", s.call("AxisCount") == 20)
    q0 = s.joints()
    s.check(c, "dual-arm joint state has 14 values", q0.shape == (14,), q=q0.tolist())
    waist, head = s.call("GetJointState", 20), s.call("GetJointState", 30)
    s.check(
        c, "waist 4 and head 2 joint values", len(waist[1]) == 4 and len(head[1]) == 2
    )
    state = s.call("ReadJointState", 0)
    s.check(
        c,
        "ReadJointState position/velocity/torque, torque source labelled",
        all(len(state[k]) == 7 for k in ("position", "velocity", "torque"))
        and "torque_source" in state,
        torque_source=state.get("torque_source"),
    )
    s.check(c, "estop inactive at start", not s.call("IsEmergencyStopActive"))
    s.check(c, "unit state Enabled (40)", s.call("GetMechUnitState", 0) == (0, 40))
    seqs = {}
    for name in CAMERAS:
        image = s.call("get_image", name, client="sensor", timeout_sec=30)
        seqs[name] = image.seq
        now_ms = time.time() * 1000
        pixels = np.frombuffer(image.data, np.uint8)
        good = (
            image.code == 0
            and (image.width, image.height, image.format) == (640, 480, "RGBA8")
            and pixels.size == 640 * 480 * 4
        )
        if good:
            rgba = pixels.reshape(480, 640, 4)
            Image.fromarray(rgba).save(s.out / f"{name}.png")
        s.check(
            c,
            f"{name} image 640x480 RGBA8, UTC timestamp, non-blank",
            good and abs(now_ms - image.timestamp) < 10_000 and pixels.std() > 5,
            seq=image.seq,
            timestamp_utc_ms=image.timestamp,
            client_lag_ms=round(now_ms - image.timestamp, 1),
        )
    second = s.call("get_image", "head", client="sensor", timeout_sec=30)
    s.check(
        c,
        "camera seq increases per frame",
        second.seq > seqs["head"],
        seq=[seqs["head"], second.seq],
    )
    s.check(
        c,
        "unknown camera -> code -1",
        s.call("get_image", "nope", client="sensor").code == -1,
    )
    chassis = s.call("get_chassis_state", client="sensor")
    s.check(c, "fixed chassis reports IDLE (1)", chassis.status == 1)
    from module_communication.messages import NavigatePoseControlMsg, VehicleControlMsg

    s.check(
        c,
        "fixed chassis refuses a nonzero velocity",
        not s.call(
            "publish_vehicle_control", VehicleControlMsg(linear_x=0.1), client="sensor"
        ),
    )
    err = s.call(
        "send_nav_goal",
        NavigatePoseControlMsg(map_name="m"),
        client="sensor",
        expect_error=True,
    )
    s.check(
        c, "navigation explicitly unsupported", err and "UNSUPPORTED" in err, error=err
    )
    for name, args in UNSUPPORTED_PROBES.items():
        err = s.call(name, *args, expect_error=True)
        s.check(
            c, f"{name} explicitly unsupported", err and "UNSUPPORTED" in err, error=err
        )
    err = s.call("rpc", "MoveJ", {"side": 0}, expect_error=True)
    s.check(
        c,
        "malformed request (missing joint_path) -> protocol error",
        err and "INVALID_REQUEST" in err,
        error=err,
    )

    c = "control_semantics"
    limit100 = np.asarray(s.call("GetJointVelLimit", 0))
    s.check(
        c,
        "SetJointVelocityRatio(15, [50, 50]) accepted",
        s.call("SetJointVelocityRatio", 15, [50, 50]) == 0,
    )
    s.check(
        c,
        "GetJointVelocityRatio feedback",
        s.call("GetJointVelocityRatio", 15) == [50.0, 50.0],
    )
    limit50 = np.asarray(s.call("GetJointVelLimit", 0))
    s.check(
        c,
        "effective velocity limit (deg/s) scales with ratio",
        np.allclose(limit50, limit100 / 2),
        deg_s=limit50.tolist(),
    )
    s.check(c, "ratio 0 rejected (-3)", s.call("SetJointVelocityRatio", 0, [0]) == -3)
    s.check(
        c, "ratio reset to 100", s.call("SetJointVelocityRatio", 15, [100, 100]) == 0
    )
    s.check(
        c,
        "disable left (lifecycle 2)",
        s.call("SetEnableState", 0, 2) == 0 and not s.call("GetEnableState", 0),
    )
    s.check(
        c,
        "disabled unit reports Disabled (20)",
        s.call("GetMechUnitState", 0) == (0, 20),
    )
    s.check(c, "motion on a disabled unit rejected (-5)", s.call("Home", 0) == -5)
    s.check(
        c,
        "re-enable left (lifecycle 1)",
        s.call("SetMechUnitLifecycle", 0, 1) == 0 and s.call("GetEnableState", 0),
    )
    pose_a = np.r_[program["pose_a"]["left"], program["pose_a"]["right"]]
    bad = pose_a.copy()
    bad[3] = program["out_of_range_elbow_rad"]
    s.check(
        c,
        "out-of-range target rejected (-3)",
        s.call("MoveJ", [bad.tolist()], 15) == -3,
    )
    s.check(
        c,
        "wrong vector length rejected (-3)",
        s.call("MoveJ", [pose_a[:13].tolist()], 15) == -3,
    )
    nan = pose_a.copy()
    nan[0] = float("nan")
    s.check(
        c, "non-finite target rejected (-3)", s.call("MoveJ", [nan.tolist()], 15) == -3
    )
    s.check(c, "invalid side rejected (-3)", s.call("GetJointState", 5)[0] == -3)
    s.check(
        c,
        "teleop push without a session -> mode conflict (-10)",
        s.call("PushJointTeleopQueue", 0, pose_a[:7].tolist()) == -10,
    )

    p = "physical_behaviour"
    t0 = time.monotonic()
    accepted = s.call("MoveJ", [pose_a.tolist()], 15)
    first = s.call("GetMotionStatus", 15)
    code, seconds, seen = s.wait_motion(15)
    s.check(
        c,
        "MoveJ returns 0 = accepted, not arrived",
        accepted == 0 and first == 1,
        first_status=first,
    )
    s.check(
        c,
        "GetMotionStatus 1 -> 0 on arrival",
        code == 0 and seen[0] == 1,
        statuses=seen,
        seconds=round(seconds, 3),
    )
    q = s.joints()
    desired = np.asarray(s.call("GetDesireJointPos", 15))
    s.check(
        p,
        "dual MoveJ reaches target (< 0.01 rad)",
        np.max(np.abs(q - pose_a)) < 0.01,
        max_error_rad=float(np.max(np.abs(q - pose_a))),
        seconds=round(time.monotonic() - t0, 3),
    )
    s.check(
        c,
        "desired joint position equals the target",
        np.allclose(desired, pose_a, atol=1e-6),
    )
    for side, sl in ((0, slice(0, 7)), (10, slice(7, 14))):
        st = s.call("ReadJointState", side)
        torque = np.asarray(st["torque"])
        s.check(
            p,
            f"side {side} holds against gravity (|torque| > 0.1 N m)",
            torque.size == 7
            and np.all(np.isfinite(torque))
            and np.max(np.abs(torque)) > 0.1,
            torque=torque.round(3).tolist(),
            source=st.get("torque_source"),
        )
    pose_b = np.r_[program["pose_b"]["left"], program["pose_b"]["right"]]
    s.check(c, "TeleJointDual session starts", s.call("TeleJointDual") == 0)
    s.check(
        c,
        "MoveJ during teleop -> mode conflict (-10)",
        s.call("MoveJ", [pose_b.tolist()], 15) == -10,
    )
    s.check(c, "Home during teleop -> mode conflict (-10)", s.call("Home") == -10)
    codes = s.stream(pose_a, pose_b, 25)
    code, seconds, _ = s.wait_motion(15)
    q = s.joints()
    s.check(c, "25 streamed dual teleop points accepted", all(x == 0 for x in codes))
    s.check(
        p,
        "dual teleop stream tracked (< 0.01 rad)",
        code == 0 and np.max(np.abs(q - pose_b)) < 0.01,
        max_error_rad=float(np.max(np.abs(q - pose_b))),
    )
    s.check(c, "StopTeleJointDual", s.call("StopTeleJointDual") == 0)
    s.check(c, "single-arm TeleJoint(0)", s.call("TeleJoint", 0) == 0)
    single = np.asarray(program["single_arm_target"])
    right_before = s.joints(10)
    codes = s.stream(pose_b[:7], single, 20, dual=False, side=0)
    code, _, _ = s.wait_motion(0)
    s.check(c, "single-arm queue points accepted", all(x == 0 for x in codes))
    s.check(
        p,
        "single-arm teleop tracked, other arm still",
        code == 0
        and np.max(np.abs(s.joints(0) - single)) < 0.01
        and np.max(np.abs(s.joints(10) - right_before)) < 0.005,
    )
    s.check(c, "StopTeleJoint(0)", s.call("StopTeleJoint", 0) == 0)
    s.check(
        c,
        "MoveJ accepted (start of soft-stop test)",
        s.call("MoveJ", [pose_a.tolist()], 15) == 0,
    )
    s.check(
        c,
        "TeleJoint while a trajectory runs -> mode conflict (-10)",
        s.call("TeleJoint", 0) == -10,
    )
    time.sleep(0.25)
    s.check(c, "SoftStop(True) accepted", s.call("SoftStop", True))
    s.check(
        c,
        "soft stop: IsReady false, estop active",
        not s.call("IsReady") and s.call("IsEmergencyStopActive"),
    )
    code, _, _ = s.wait_motion(15, timeout=20)
    stopped = s.joints()
    s.check(
        p,
        "soft stop brakes the motion short of its target",
        code == 0 and np.max(np.abs(stopped - pose_a)) > 0.02,
        remaining_rad=float(np.max(np.abs(stopped - pose_a))),
    )
    s.check(c, "motion during soft stop rejected (-5)", s.call("Home") == -5)
    s.check(
        c, "enable during soft stop rejected (-8)", s.call("SetEnableState", 0, 1) == -8
    )
    s.check(
        c,
        "gripper during soft stop rejected (-5)",
        s.call("GripperSetPosition", 0, 500) == -5,
    )
    s.check(
        c, "SoftStop(False) releases", s.call("SoftStop", False) and s.call("IsReady")
    )
    s.check(
        c,
        "motion resumes after release",
        s.call("MoveJ", [pose_a.tolist()], 15) == 0 and s.wait_motion(15)[0] == 0,
    )

    for side in (0, 10):
        s.check(
            c,
            f"gripper {side} initialise (no-op in simulation)",
            s.call("GripperInitialize", side) == 0
            and s.call("GripperReadInitState", side)["value"] == 1,
        )
        readings = {}
        for target in program["gripper_positions"]:
            s.call("GripperSetPosition", side, target)
            state, position = s.wait_gripper(side)
            readings[target] = (state, position)
        ok = all(abs(pos - t) <= 25 and st == 1 for t, (st, pos) in readings.items())
        s.check(
            p,
            f"gripper {side} position feedback tracks 750/250/1000 in air",
            ok,
            readings={str(k): v for k, v in readings.items()},
        )
    s.check(c, "Home accepted", s.call("Home") == 0)
    code, _, _ = s.wait_motion(15)
    s.check(
        p,
        "Home reaches the SDK zero pose (< 0.01 rad)",
        code == 0 and np.max(np.abs(s.joints())) < 0.01,
        max_abs_rad=float(np.max(np.abs(s.joints()))),
    )


def matrix(values):
    """Vendor TransformMatrix4x4: column-major 16 -> 4x4."""
    return np.asarray(values, float).reshape(4, 4).T


def wire(T):
    return np.asarray(T, float).T.reshape(-1).tolist()


def axis_rotations(rz, ry, rx):
    """Rz(rz), Ry(ry), Rx(rx)."""
    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )
    Ry = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )
    return Rz, Ry, Rx


def pose_rotation(pose):
    """[x, y, z, rz, ry, rx] -> R = Rx(rx) Ry(ry) Rz(rz)
    (sdk_manual/cartesian_pose_convention.md)."""
    Rz, Ry, Rx = axis_rotations(*pose[3:6])
    return Rx @ Ry @ Rz


def kinematics(s, program):
    """The SDK's kinematics endpoints: FK/IK and pose consistency, the tool and
    base frames, and one Cartesian goal reached through IKb + MoveJ. Every
    number is the SDK's own; whether it matches the simulated robot is scored
    from the trace (verify_backends.py)."""
    c, p = "kinematics", "physical_behaviour"
    for side in (0, 10):
        status, flan = s.call("GetFlan2tool", side)
        R = matrix(flan)[:3, :3] if len(flan) == 16 else np.zeros((3, 3))
        s.check(
            c,
            f"GetFlan2tool({side}) is a rigid transform",
            status == 0 and np.allclose(R.T @ R, np.eye(3), atol=1e-5),
            flan2tool=flan,
        )
    pose_a = np.r_[program["pose_a"]["left"], program["pose_a"]["right"]]
    s.check(
        c,
        "MoveJ to pose_a for the kinematics checks",
        s.call("MoveJ", [pose_a.tolist()], 15) == 0 and s.wait_motion(15)[0] == 0,
    )
    goals, targets = [], []
    for side in (0, 10):
        q = s.joints(side).tolist()
        fk = s.call("Fk", side, q)[1]
        tool, flan = s.call("FkTool", side, q)[1], s.call("GetFlan2tool", side)[1]
        s.check(
            c,
            f"side {side}: FkTool = Fk flan2tool",
            np.allclose(matrix(tool), matrix(fk) @ matrix(flan), atol=1e-6),
        )
        status, pose = s.call("GetCurCartesianPose", side)
        T = matrix(tool)
        # Read moments after FkTool: the arm may still settle (motion status 0
        # allows 0.02 rad/s), hence 0.1 mm / 1 mrad.
        s.check(
            c,
            f"side {side}: GetCurCartesianPose matches FkTool of the measured joints",
            status == 0
            and np.allclose(pose[:3], T[:3, 3], atol=1e-4)
            and np.allclose(pose_rotation(pose), T[:3, :3], atol=1e-3),
            pose=pose,
        )
        status, q_ik = s.call("IK", side, tool)
        s.check(
            c,
            f"side {side}: IK of the current tool pose returns the current joints",
            status == 0 and np.max(np.abs(np.subtract(q_ik, q))) < 1e-4,
            max_difference_rad=float(np.max(np.abs(np.subtract(q_ik, q))))
            if status == 0
            else None,
        )
        waist = s.call("GetJointState", 20)[1]
        fkb = s.call("FKb", side, q)[1]
        s.check(
            c,
            f"side {side}: FKb with empty torso_joints uses the measured waist",
            np.allclose(fkb, s.call("FKb", side, q, waist)[1], atol=1e-5),
        )
        status, q_b = s.call("IKb", side, fkb)
        s.check(
            c,
            f"side {side}: IKb(FKb(q)) returns q",
            status == 0 and np.max(np.abs(np.subtract(q_b, q))) < 1e-4,
        )
        target = matrix(fkb)
        target[2, 3] += LIFT_M
        status, q_up = s.call("IKb", side, wire(target))
        s.check(c, f"side {side}: IKb of the flange {LIFT_M} m higher", status == 0)
        goals.extend(q_up if status == 0 else q)
        targets.append(target)
    moved = s.call("MoveJ", [goals], 15) == 0 and s.wait_motion(15)[0] == 0
    errors = []
    for side, target in zip((0, 10), targets):
        fkb = matrix(s.call("FKb", side, s.joints(side).tolist())[1])
        errors.append(float(np.linalg.norm(fkb[:3, 3] - target[:3, 3])))
    s.check(
        p,
        f"IKb + MoveJ raise both flanges {LIFT_M} m in B (FKb error < 1 mm)",
        moved and max(errors) < 0.001,
        error_m=errors,
    )
    q = s.joints(0).tolist()
    tool = np.eye(4)
    # Large angles, so that the pose convention check below means something.
    tool[:3, :3] = pose_rotation([0, 0, 0, 0.5, -0.4, 0.3])
    tool[0, 3] = 0.245  # along the flange x axis, toward the fingers
    status = s.call("SetFlan2tool", 0, wire(tool))
    back = s.call("GetFlan2tool", 0)[1]
    fk, ft = matrix(s.call("Fk", 0, q)[1]), s.call("FkTool", 0, q)[1]
    s.check(
        c,
        "SetFlan2tool round-trips; FkTool and IK use the new tool",
        status == 0
        and np.allclose(back, wire(tool), atol=1e-6)
        and np.allclose(matrix(ft), fk @ tool, atol=1e-6)
        and np.max(np.abs(np.subtract(s.call("IK", 0, ft)[1], q))) < 1e-4,
    )
    status, pose = s.call("GetCurCartesianPose", 0)
    T = matrix(s.call("FkTool", 0, s.joints(0).tolist())[1])
    Rz, Ry, Rx = axis_rotations(*pose[3:6])
    s.check(
        c,
        "pose convention with a rotated tool: R = Rx Ry Rz (Rz Ry Rx would differ)",
        status == 0
        and np.allclose(pose[:3], T[:3, 3], atol=1e-4)
        and np.allclose(pose_rotation(pose), T[:3, :3], atol=1e-3)
        and np.abs(Rz @ Ry @ Rx - T[:3, :3]).max() > 0.05,
        pose=pose,
    )
    s.check(
        c,
        "flan2tool reset to identity",
        s.call("SetFlan2tool", 0, wire(np.eye(4))) == 0,
    )
    far = np.eye(4)
    far[:3, 3] = [2.0, 0.0, 0.0]
    s.check(c, "unreachable IK target -> -6", s.call("IK", 0, wire(far))[0] == -6)
    s.check(
        c, "non-rigid transform rejected (-3)", s.call("IK", 0, [1.0] * 16)[0] == -3
    )
    s.check(c, "Fk with 6 joints rejected (-3)", s.call("Fk", 0, q[:6])[0] == -3)
    s.check(c, "Fk on side 15 rejected (-3)", s.call("Fk", 15, q)[0] == -3)
    s.check(
        c,
        "Home after the kinematics checks",
        s.call("Home") == 0 and s.wait_motion(15)[0] == 0,
    )


def task(s, program):
    """Open-loop replay of the expert's joint targets with SDK gripper commands."""
    c = "task_execution"
    rate = program["task"]["rate_hz"]
    phases = program["task"]["phases"]
    log = []
    s.check(c, "TeleJointDual for the task", s.call("TeleJointDual") == 0)
    current = s.joints()
    grip = [None, None]
    for phase in phases:
        entry = {"phase": phase["name"], "started_s": round(time.time() - s.start, 3)}
        if phase["gripper"] != grip:
            for k, side in enumerate((0, 10)):
                s.call("GripperSetPosition", side, phase["gripper"][k])
            entry["grip_states"] = [s.wait_gripper(side)[0] for side in (0, 10)]
            grip = list(phase["gripper"])
        rejected = 0
        tick = time.monotonic()
        for point in phase["points"]:
            q = np.asarray(point)
            if np.max(np.abs(q - current)) > 1e-9:
                rejected += (
                    s.call("PushJointTeleopQueueDual", q[:7].tolist(), q[7:].tolist())
                    != 0
                )
                current = q
            tick += 1 / rate
            time.sleep(max(0.0, tick - time.monotonic()))
        entry["motion_status"], entry["settle_s"], _ = s.wait_motion(15, timeout=60)
        entry["rejected_points"] = int(rejected)
        entry["grip_state_after"] = [
            s.call("GripperReadGripState", side)["value"] for side in (0, 10)
        ]
        log.append(entry)
        print(f"  task phase {phase['name']}: {entry}", flush=True)
    s.check(c, "StopTeleJointDual after the task", s.call("StopTeleJointDual") == 0)
    lift = next((e for e in log if e["phase"] == "lift"), None)
    s.check(
        c,
        "SDK reports both grippers holding (2) after the lift",
        lift is not None and lift["grip_state_after"] == [2, 2],
        lift=lift,
    )
    s.check(
        c,
        "every phase settled (GetMotionStatus 0) and no point rejected",
        all(e["motion_status"] == 0 and e["rejected_points"] == 0 for e in log),
    )
    return log


def fault_probe(s, program):
    """Drive the left hand into the table. A backend must either stop on a
    > 1 mm penetration (fault) or report the stalled motion (-3); both are
    explicit. Run last: a physics fault ends the simulation session."""
    c = "fault_handling"
    probe = program["fault_probe"]
    s.check(
        c,
        "MoveJ above the probe spot",
        s.call("MoveJ", [probe["above"]], 0) == 0 and s.wait_motion(0)[0] == 0,
    )
    s.check(c, "descent MoveJ accepted", s.call("MoveJ", probe["press_path"], 0) == 0)
    code, seconds, seen = s.wait_motion(0, timeout=60)
    alarm = s.call("GetCurAlarm")
    outcome = {
        "final_motion_status": code,
        "statuses": seen,
        "seconds": round(seconds, 2),
        "alarm": alarm,
    }
    if code == -1:
        s.check(
            c,
            "physics fault latched and reported (status -1, alarm, not ready)",
            alarm["has_alarm"] and not s.call("IsReady"),
            **outcome,
        )
        s.check(c, "motion after fault rejected (-9)", s.call("Home") == -9)
        s.check(
            c,
            "re-enable after fault rejected (-9)",
            s.call("SetEnableState", 0, 1) == -9,
        )
        outcome["result"] = "fault"
    elif code == -3:
        s.check(
            c,
            "stalled motion reported (-3), no fault",
            not alarm["has_alarm"] and s.call("IsReady"),
            **outcome,
        )
        s.check(
            c,
            "recovers by moving back up",
            s.call("MoveJ", [probe["above"]], 0) == 0 and s.wait_motion(0)[0] == 0,
        )
        outcome["result"] = "stall"
    else:
        s.check(
            c, "pressing into the table is reported (fault or stall)", False, **outcome
        )
        outcome["result"] = "unreported"
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--connection", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--program", type=Path, default=ROOT / "config/sdk_acceptance/program.json"
    )
    parser.add_argument(
        "--task", action="store_true", help="open-loop full-task replay"
    )
    parser.add_argument(
        "--fault-probe", action="store_true", help="press into the table last"
    )
    args = parser.parse_args()
    profile = read_profile(args.connection)
    if profile["host"] not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(
            "refusing a non-loopback host: these motions are simulation-only"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    program = json.loads(args.program.read_text())
    arm, sensors = connect(
        profile,
        auto_heartbeat=False,
        auto_reconnect=False,
        raise_on_error=False,
        timeout_ms=30_000,
    )
    s = Session(arm, sensors, args.output)
    result = {
        "connection": profile,
        "program": str(args.program),
        "joint_convention": program["joint_convention"],
    }
    error = None
    try:
        core(s, program)
        kinematics(s, program)
        if args.task:
            result["task_log"] = task(s, program)
        if args.fault_probe:
            result["fault_probe"] = fault_probe(s, program)
    except Exception as exc:  # noqa: BLE001 - recorded in verification.json
        error = f"{type(exc).__name__}: {exc}"
        print("ERROR", error, flush=True)
    finally:
        sensors.close()
        arm.shutdown()
    summary = {}
    for check in s.checks:
        cat = summary.setdefault(check["category"], {"passed": 0, "failed": 0})
        cat["passed" if check["passed"] else "failed"] += 1
    result.update({"error": error, "summary_by_category": summary, "checks": s.checks})
    (args.output / "verification.json").write_text(json.dumps(result, indent=2))
    (args.output / "sdk_calls.json").write_text(json.dumps(s.calls, indent=1))
    print(json.dumps({"error": error, "summary_by_category": summary}, indent=2))
    if error or any(not c["passed"] for c in s.checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
