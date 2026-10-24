"""V1R1C9 request validation and response shapes for the two SDK ports.

Queries read the runtime's latest snapshot; commands run on the physics thread
through ``runtime.call``. A method missing from ``handlers`` raises Unsupported,
which becomes an explicit protocol error naming the method and the backend.
"""

import time

import numpy as np

from simbridge.control import precheck_push
from simbridge.sdk_kinematics import SdkKinematics, from_wire, pose_from_matrix, to_wire
from simbridge.status import InvalidRequest, Rejected, Sdk, Unsupported

SIDES = {0: ("left",), 10: ("right",), 15: ("left", "right")}
UNIT_STATE = {"fault": 0, "disabled": 20, "enabled": 40, "running": 50}


def ok(value=None, **fields):
    out = {"status": 0, **fields}
    if value is not None:
        out["value"] = value
    return out


def status(code, message):
    return {"status": int(code), "err_msg": message}


class ArmService:
    """``arm_sdk.*`` routes on the upper-body port (vendor default 8000)."""

    def __init__(self, runtime, robot, backend):
        self.runtime = runtime
        self.robot = robot
        self.backend = backend
        self.version = f"V1R1C9/simbridge-{backend.name}"
        names = [
            "GetVersion", "IsReady", "AxisCount", "SoftStop", "GetEmergencyStopState",
            "SetEnableState", "SetMechUnitLifecycle", "GetEnableState",
            "GetMechUnitState", "GetCurAlarm", "GetJointState", "GetCurJointVel",
            "ReadJointState", "GetMotionStatus", "GetDesireJointPos",
            "GetDesireJointVel", "GetJointVelLimit", "GetJointAccLimit",
            "GetJointVelocityRatio", "SetJointVelocityRatio", "MoveJ", "Home",
            "TeleJoint", "TeleJointDual", "StopTeleJoint", "StopTeleJointDual",
            "PushJointTeleopQueue", "PushJointTeleopQueueDual", "EeGripperInitialize",
            "EeGripperReadInitState", "EeGripperOpen", "EeGripperClose",
            "EeGripperSetPosition", "EeGripperReadPosition", "EeGripperReadGripState",
            "Fk", "FkTool", "FKb", "FKbTool", "IK", "IKb", "IKbTool", "GetFlan2tool",
            "SetFlan2tool", "GetCurCartesianPose", "GetCurCartesianPoseDual",
        ]  # fmt: skip
        self.handlers = {n: getattr(self, "_" + n) for n in names}
        try:  # robot_kinematics needs Pinocchio, which not every server env has
            self.kin, self.kin_missing = SdkKinematics(robot), None
        except ImportError as exc:
            self.kin, self.kin_missing = None, f"{type(exc).__name__}: {exc}"

    def dispatch(self, name, args):
        self.runtime.touch(name)
        handler = self.handlers.get(name)
        if handler is None:
            raise Unsupported(
                f"arm_sdk.{name} is not provided by the {self.backend.name} "
                "simulation backend"
            )
        request = args[0] if args else {}
        if not isinstance(request, dict):
            raise InvalidRequest("expected one request map")
        return handler(request)

    # ----- helpers ---------------------------------------------------------
    def _snap(self):
        return self.runtime.snapshot()

    def _arms(self, side):
        units = SIDES.get(int(side))
        if units is None:
            raise Rejected(Sdk.PARAM_INVALID, f"side {side} is not an arm (0/10/15)")
        return units

    def _command(self, fn):
        try:
            return self.runtime.call(fn)
        except TimeoutError as exc:
            return status(Sdk.NOT_READY, str(exc))

    def _joint_values(self, side, key):
        snap = self._snap()
        side = int(side)
        if side in (20, 30):
            unit = self.robot.unit_by_side(side)
            return snap["units"][unit.name][key]
        values = []
        for u in self._arms(side):
            values.extend(snap["units"][u][key])
        return values

    def _guard(self, fn, request):
        try:
            return fn(request)
        except Rejected as exc:
            return status(exc.code, exc.message)

    # ----- core / system ----------------------------------------------------
    def _GetVersion(self, request):
        return self.version

    def _IsReady(self, request):
        snap = self._snap()
        return ok(
            bool(snap["physics_running"] and not snap["estop"] and not snap["fault"])
        )

    def _AxisCount(self, request):
        return ok(self.robot.axis_count)

    def _SoftStop(self, request):
        active = bool(request["active"])
        return self._command(lambda ctl, now: ctl.soft_stop(active, now))

    def _GetEmergencyStopState(self, request):
        return ok(estop_active=bool(self._snap()["estop"]))

    def _lifecycle(self, unit_value, cmd):
        unit_value = int(unit_value)
        if unit_value in (20, 30):
            return status(
                Sdk.UNSUPPORTED, "waist/head are read-only in this simulation"
            )
        units = SIDES.get(unit_value)
        if units is None or unit_value == 15:
            return status(Sdk.PARAM_INVALID, f"mechanical unit {unit_value} not 0/10")
        return self._command(lambda ctl, now: ctl.lifecycle(units, int(cmd), now))

    def _SetEnableState(self, request):
        return self._lifecycle(request["robot_type"], request["lifecycle_cmd"])

    def _SetMechUnitLifecycle(self, request):
        return self._lifecycle(request["component_type"], request["lifecycle_cmd"])

    def _GetEnableState(self, request):
        def run(request):
            arms = self._snap()["arms"]
            return ok(all(arms[u]["enabled"] for u in self._arms(request["side"])))

        return self._guard(run, request)

    def _GetMechUnitState(self, request):
        unit = int(request["component_type"])
        snap = self._snap()
        if unit not in (0, 10):
            return status(Sdk.UNSUPPORTED, "waist/head lifecycle is not simulated")
        arm = snap["arms"][SIDES[unit][0]]
        if snap["fault"]:
            state = UNIT_STATE["fault"]
        elif not arm["enabled"]:
            state = UNIT_STATE["disabled"]
        elif arm["motion_status"] == 1:
            state = UNIT_STATE["running"]
        else:
            state = UNIT_STATE["enabled"]
        return ok(state)

    def _GetCurAlarm(self, request):
        alarms = [
            {
                "module_name": f"simbridge/{self.backend.name}",
                "alarm_code": 0x7F000001,
                "level": 1,
                "timestamp_us": f["utc_ms"] * 1000,
                "custom_info": f"{f['kind']}: {f['message']} (sim_time {f['sim_time']:.4f}s)",
                "source_id": -1,
            }
            for f in self.runtime.faults
        ]
        return ok(
            has_alarm=bool(alarms), alarms=alarms, code_space="simulation-defined"
        )

    # ----- state ------------------------------------------------------------
    def _GetJointState(self, request):
        return self._guard(lambda r: ok(self._joint_values(r["side"], "q")), request)

    def _GetCurJointVel(self, request):
        return self._guard(lambda r: ok(self._joint_values(r["side"], "dq")), request)

    def _ReadJointState(self, request):
        def run(request):
            snap = self._snap()
            units = self._arms(request["side"])
            torque, source = [], snap["effort_kind"]
            if all(snap["units"][u]["effort"] is not None for u in units):
                for u in units:
                    torque.extend(snap["units"][u]["effort"])
            return ok(
                position=self._joint_values(request["side"], "q"),
                velocity=self._joint_values(request["side"], "dq"),
                torque=torque,
                torque_source=source,
                sim_time=snap["sim_time"],
            )

        return self._guard(run, request)

    def _GetMotionStatus(self, request):
        def run(request):
            codes = [
                self._snap()["arms"][u]["motion_status"]
                for u in self._arms(request["side"])
            ]
            for code in (-1, -3, 1):
                if code in codes:
                    return ok(code)
            return ok(0)

        return self._guard(run, request)

    def _desired(self, request, key):
        def run(request):
            values = []
            for u in self._arms(request["side"]):
                values.extend(self._snap()["arms"][u][key])
            return ok(values)

        return self._guard(run, request)

    def _GetDesireJointPos(self, request):
        return self._desired(request, "setpoint")

    def _GetDesireJointVel(self, request):
        return self._desired(request, "velocity")

    # ----- limits -------------------------------------------------------------
    def _rate_limit(self, request, attr):
        def run(request):
            values = []
            arms = self._snap()["arms"]
            for u in self._arms(request["side"]):
                limit = getattr(self.robot.arms[u], attr)
                # V1R1C9 reports deg/s and deg/s^2, scaled by the velocity ratio.
                values.extend(np.rad2deg(limit * arms[u]["ratio"] / 100).tolist())
            return ok(values)

        return self._guard(run, request)

    def _GetJointVelLimit(self, request):
        return self._rate_limit(request, "vmax")

    def _GetJointAccLimit(self, request):
        return self._rate_limit(request, "amax")

    def _GetJointVelocityRatio(self, request):
        def run(request):
            arms = self._snap()["arms"]
            return ok([arms[u]["ratio"] for u in self._arms(request["side"])])

        return self._guard(run, request)

    def _SetJointVelocityRatio(self, request):
        def run(request):
            units = self._arms(request["side"])
            ratios = [float(x) for x in request["ratio"]]
            return self._command(lambda ctl, now: ctl.set_ratio(units, ratios, now))

        return self._guard(run, request)

    # ----- motion -------------------------------------------------------------
    def _MoveJ(self, request):
        def run(request):
            if int(request.get("side", 15)) in (20, 30):
                return status(Sdk.UNSUPPORTED, "waist/head motion is not simulated")
            units = self._arms(request.get("side", 15))
            path = [list(map(float, p)) for p in request["joint_path"]]
            return self._command(lambda ctl, now: ctl.move_j(units, path, now))

        return self._guard(run, request)

    def _Home(self, request):
        def run(request):
            if int(request.get("side", 15)) in (20, 30):
                return status(Sdk.UNSUPPORTED, "waist/head motion is not simulated")
            units = self._arms(request.get("side", 15))
            path = [[0.0] * (7 * len(units))]
            return self._command(
                lambda ctl, now: ctl.move_j(units, path, now, home=True)
            )

        return self._guard(run, request)

    def _TeleJoint(self, request):
        def run(request):
            units = self._arms(request["side"])
            if len(units) != 1:
                return status(
                    Sdk.PARAM_INVALID, "TeleJoint takes one arm; use TeleJointDual"
                )
            return self._command(lambda ctl, now: ctl.tele_start(units, now))

        return self._guard(run, request)

    def _TeleJointDual(self, request):
        return self._command(lambda ctl, now: ctl.tele_start(("left", "right"), now))

    def _StopTeleJoint(self, request):
        def run(request):
            units = self._arms(request["side"])
            return self._command(lambda ctl, now: ctl.tele_stop(units, now))

        return self._guard(run, request)

    def _StopTeleJointDual(self, request):
        return self._command(lambda ctl, now: ctl.tele_stop(("left", "right"), now))

    def _PushJointTeleopQueue(self, request):
        def run(request):
            units = self._arms(request["side"])
            if len(units) != 1:
                return status(Sdk.PARAM_INVALID, "single-arm queue takes side 0 or 10")
            points = {units[0]: [float(x) for x in request["joints"]]}
            return self._stream(points)

        return self._guard(run, request)

    def _PushJointTeleopQueueDual(self, request):
        points = {
            "left": [float(x) for x in request["left"]],
            "right": [float(x) for x in request["right"]],
        }
        return self._stream(points)

    def _stream(self, points):
        """Answer at once from the snapshot; the physics thread re-checks."""
        stamp = time.monotonic()
        rejected = precheck_push(self.robot, points, self._snap())
        if rejected:
            return rejected
        self.runtime.submit(
            lambda ctl, now: ctl.tele_push(points, now, stamp, replied=True)
        )
        return ok()

    # ----- kinematics: robot_kinematics in SDK conventions (sdk_kinematics.py) --
    def _model(self):
        if self.kin is None:
            raise Unsupported(
                "arm_sdk kinematics endpoints need robot_kinematics (Pinocchio), "
                f"which this server's Python cannot import ({self.kin_missing})"
            )
        return self.kin

    def _one_arm(self, side):
        units = SIDES.get(int(side))
        if units is None or len(units) != 1:
            raise Rejected(Sdk.PARAM_INVALID, f"side {side} is not one arm (0/10)")
        return units[0]

    @staticmethod
    def _numbers(values, count, what):
        try:
            out = np.array([float(v) for v in values])
        except (TypeError, ValueError):
            raise Rejected(Sdk.PARAM_INVALID, f"{what} must be numbers") from None
        if out.shape != (count,) or not np.all(np.isfinite(out)):
            raise Rejected(Sdk.PARAM_INVALID, f"{what} must be {count} finite numbers")
        return out

    @staticmethod
    def _transform(field):
        if not isinstance(field, dict):
            raise Rejected(Sdk.PARAM_INVALID, "transform must be {value: [16]}")
        return from_wire(field.get("value", ()))

    def _torso(self, request):
        values = request.get("torso_joints") or []
        if len(values) == 0:
            # Vendor: an empty list makes the controller read the waist unit.
            return np.asarray(self._snap()["units"]["waist"]["q"], float)
        return self._numbers(values, 4, "torso_joints")

    def _measured(self, side):
        return np.asarray(self._snap()["units"][side]["q"], float)

    def _kinematics(self, request, fn):
        """Run fn(request, side, model) -> 4x4 or joints, as FkRsp or IkRsp."""
        model = self._model()

        def run(request):
            out = fn(request, self._one_arm(request["side"]), model)
            if out.shape == (4, 4):
                return {"status": 0, "out_tf": {"value": to_wire(out)}}
            return {"status": 0, "out_joint_pos": out.tolist()}

        return self._guard(run, request)

    def _fk(self, request, tool):
        def fn(request, side, model):
            q = self._numbers(request["cur_joints"], 7, "cur_joints")
            return model.fk(side, q, tool=tool)

        return self._kinematics(request, fn)

    def _Fk(self, request):
        return self._fk(request, tool=False)

    def _FkTool(self, request):
        return self._fk(request, tool=True)

    def _fkb(self, request, tool):
        def fn(request, side, model):
            q = self._numbers(request["cur_joints"], 7, "cur_joints")
            return model.fkb(side, q, self._torso(request), tool=tool)

        return self._kinematics(request, fn)

    def _FKb(self, request):
        return self._fkb(request, tool=False)

    def _FKbTool(self, request):
        return self._fkb(request, tool=True)

    def _IK(self, request):
        def fn(request, side, model):
            target = self._transform(request["base2tool"])
            return model.ik(side, target, self._measured(side), tool=True)

        return self._kinematics(request, fn)

    def _ikb(self, request, tool):
        def fn(request, side, model):
            target = self._transform(request["base2ee"])
            torso = self._torso(request)
            return model.ikb(side, target, torso, self._measured(side), tool=tool)

        return self._kinematics(request, fn)

    def _IKb(self, request):
        return self._ikb(request, tool=False)

    def _IKbTool(self, request):
        return self._ikb(request, tool=True)

    def _GetFlan2tool(self, request):
        return self._kinematics(request, lambda r, side, m: m.flan2tool[side])

    def _SetFlan2tool(self, request):
        model = self._model()

        def run(request):
            side = self._one_arm(request["side"])
            model.flan2tool[side] = self._transform(request["flan2tool"])
            return ok()

        return self._guard(run, request)

    def _pose(self, model, side):
        return pose_from_matrix(model.fk(side, self._measured(side), tool=True))

    def _GetCurCartesianPose(self, request):
        model = self._model()
        return self._guard(
            lambda r: ok(self._pose(model, self._one_arm(r["side"]))), request
        )

    def _GetCurCartesianPoseDual(self, request):
        model = self._model()
        return ok(left=self._pose(model, "left"), right=self._pose(model, "right"))

    # ----- end effector ---------------------------------------------------------
    def _gripper_side(self, request):
        units = SIDES.get(int(request["side"]))
        if units is None or len(units) != 1:
            raise Rejected(Sdk.PARAM_INVALID, "gripper side must be 0 or 10")
        return units[0]

    def _set_gripper(self, request, position):
        def run(request):
            side = self._gripper_side(request)
            value = int(request["value"]) if position is None else position
            return self._command(lambda ctl, now: ctl.gripper(side, value, now))

        return self._guard(run, request)

    def _EeGripperInitialize(self, request):
        def run(request):
            self._gripper_side(request)
            return ok(note="simulated gripper needs no homing; nothing was calibrated")

        return self._guard(run, request)

    def _EeGripperReadInitState(self, request):
        def run(request):
            self._gripper_side(request)
            return ok(1)  # 1 = initialised: the simulated gripper needs no homing

        return self._guard(run, request)

    def _EeGripperOpen(self, request):
        return self._set_gripper(request, 1000)

    def _EeGripperClose(self, request):
        return self._set_gripper(request, 0)

    def _EeGripperSetPosition(self, request):
        return self._set_gripper(request, None)

    def _EeGripperReadPosition(self, request):
        def run(request):
            g = self._snap()["grippers"][self._gripper_side(request)]
            return ok(g["position"], target=g["target"])

        return self._guard(run, request)

    def _EeGripperReadGripState(self, request):
        def run(request):
            return ok(
                self._snap()["grippers"][self._gripper_side(request)]["grip_state"]
            )

        return self._guard(run, request)


class SensorService:
    """module_communication routes (vendor default port 8100)."""

    def __init__(self, runtime, robot, backend):
        self.runtime = runtime
        self.robot = robot
        self.backend = backend
        self.cameras = set(backend.capabilities().rgb_cameras)

    def dispatch(self, name, args):
        self.runtime.touch(name)
        if name == "get_image":
            return self.image(str(args[0]))
        if name == "get_chassis_state":
            # Fixed base: IDLE is the true state; there is no map.
            return {
                "status": 1,
                "map_name": "",
                "timestamp": time.time_ns() // 1_000_000,
            }
        if name == "publish_vehicle_control":
            msg = args[0] if args else {}
            # The base is welded to the world in this experiment; only a zero
            # command is honoured, anything else reports failure.
            return all(
                abs(float(msg.get(k, 0))) < 1e-9
                for k in ("linear_x", "linear_y", "angular_z")
            )
        raise Unsupported(
            f"{name} (navigation/localisation) is not provided: the simulated base is fixed"
        )

    def image(self, name):
        spec = self.robot.cameras.get(name)
        if spec is None:
            return {"code": -1, "err_msg": f"unknown camera {name!r}"}
        if spec.sim_name not in self.cameras:
            return {
                "code": -2,
                "err_msg": f"{self.backend.name} renders no {name} image",
            }
        rgb, entry = self.runtime.render(name, timeout=30)
        rgba = np.concatenate(
            [rgb, np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)], axis=2
        )
        return {
            "code": 0,
            "err_msg": "",
            "seq": entry["seq"],
            "width": spec.width,
            "height": spec.height,
            "timestamp": entry["utc_ms"],
            "format": "RGBA8",
            "data": rgba.tobytes(),
        }
