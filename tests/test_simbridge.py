"""Engine-free tests of the bridge layers. Run from the project root:

    /Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m pytest tests -q

``KinematicPlant`` below is a test double that tracks targets perfectly; it is
not a simulator backend and is not registered in simbridge.backends.
"""

import importlib.util
import json
import threading
import time

import numpy as np
import pytest

from sdk.check_sdk_kinematics import (
    compose,
    fk_mdh,
    fk_urdf,
    machine_params_mdh,
    pose_from_vector,
    urdf_chain,
)
from sdk.stock_sdk import UPPER, connect
from simbridge import ROOT
from simbridge.backends.base import (
    EFFORT_UNAVAILABLE,
    BackendState,
    Capabilities,
    SimBackend,
)
from simbridge.control import Controller, Segment
from simbridge.protocol import ARM_PREFIX, SENSOR_ROUTES, Endpoint, arm_catalog
from simbridge.robot import load_robot
from simbridge.runtime import Runtime
from simbridge.sdk_kinematics import (
    SdkKinematics,
    from_wire,
    homogeneous,
    matrix_from_pose,
    pose_from_matrix,
    to_wire,
)
from simbridge.services import ArmService, SensorService
from simbridge.status import Rejected, Unsupported

ROBOT = load_robot()
CROSSCHECK = json.loads((ROOT / "config/robot/kinematic_crosscheck.json").read_text())
# The SDK kinematics endpoints answer from robot_kinematics, which needs Pinocchio
# (the inovance-fineart env); elsewhere those tests are skipped.
PINOCCHIO = importlib.util.find_spec("pinocchio") is not None
KIN = SdkKinematics(ROBOT) if PINOCCHIO else None
needs_pinocchio = pytest.mark.skipif(not PINOCCHIO, reason="needs Pinocchio")
IMAGE = (
    ROOT / "sdk_review/controller_image_probe_v2/files/image/inocfg/inbc_for_humanoid"
)
# The vendor SDK and the controller-image extract live in sdk_review/, which is
# not in the repository: tests that need them skip in a fresh clone.
VENDOR_SDK = UPPER.is_dir()
needs_vendor_sdk = pytest.mark.skipif(
    not VENDOR_SDK, reason="vendor SDK (sdk_review/) not in this checkout"
)
needs_image = pytest.mark.skipif(
    not IMAGE.exists(), reason="controller image (sdk_review/) not in this checkout"
)


# ----- robot mapping and kinematics ------------------------------------------
def test_arm_mapping_matches_crosscheck():
    for side in ("left", "right"):
        selected = CROSSCHECK["arms"][side]["selected"]
        unit = ROBOT.raw["units"][side]["mapping"]["controller_mdh"]
        assert unit["sign"] == selected["signs"]
        assert np.allclose(unit["offset_rad"], selected["offsets_rad"])
    waist = ROBOT.raw["units"]["waist"]["mapping"]["controller_mdh"]
    assert waist["sign"] == CROSSCHECK["waist"]["selected"]["signs"]


@needs_image
def test_mdh_reproduces_urdf_with_selected_mapping():
    machine = (
        ROOT / "sdk_review/controller_image_probe_v2/files/image/inocfg/"
        "inbc_for_humanoid/machine_params.json"
    )
    for side in ("left", "right"):
        unit = ROBOT.arms[side]
        frames = CROSSCHECK["arms"][side]["selected"]["frames"]
        R_b = np.array(frames["mdh_base_in_parent"]["R"])
        p_b = np.array(frames["mdh_base_in_parent"]["p_m"])
        R_f = np.array(frames["mdh_flange_in_child_link"]["R"])
        p_f = np.array(frames["mdh_flange_in_child_link"]["p_m"])
        Q = np.random.default_rng(3).uniform(
            unit.sim_limits[:, 0], unit.sim_limits[:, 1], (50, 7)
        )
        R_u, p_u = fk_urdf(urdf_chain(ROBOT.urdf, unit.sim_joints), Q)
        # Tool transform = inverse of the flange expressed in the child link.
        R_t, p_t = R_f.T, -R_f.T @ p_f
        R, p = fk_mdh(machine_params_mdh(machine, f"{side}_arm"), unit.to_sdk(Q))
        R, p = compose(R_b, p_b, R, p, R_t, p_t)
        assert np.max(np.linalg.norm(p - p_u, axis=1)) < 1e-9
        assert np.max(np.abs(R - R_u)) < 1e-9


def test_elbow_sign_and_limits():
    left = ROBOT.arms["left"]
    # URDF elbow flexion is negative; controller flexion is positive.
    assert left.to_sdk(np.r_[0, 0, 0, -1.0, 0, 0, 0])[3] == pytest.approx(1.0)
    lo, hi = np.rad2deg(left.sdk_limits[3])
    # URDF bound on one side, controller bound (minus the margin) on the other.
    assert lo == pytest.approx(-7.39, abs=0.01) and hi == pytest.approx(
        147.943, abs=1e-3
    )
    q = np.linspace(-1, 1, 7)
    assert np.allclose(left.to_sim(left.to_sdk(q)), q)


def test_gripper_semantics():
    g = ROBOT.grippers["left"]
    assert g.closure_from_position(1000) == 0 and g.closure_from_position(0) == 1
    targets = g.finger_targets(0.5)
    assert g.closure(list(targets.values())) == pytest.approx(0.5)
    assert g.position_from_closure(0.548) == 452


def test_pose_from_vector_roundtrip():
    R, p = pose_from_vector(np.r_[0.3, -0.2, 0.1, 1, 2, 3])
    assert np.allclose(R @ R.T, np.eye(3)) and np.allclose(p, [1, 2, 3])


# ----- the SDK's kinematics endpoints (robot_kinematics in SDK conventions) ------
def test_wire_format_is_column_major_float32():
    T = homogeneous(
        matrix_from_pose([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])[:3, :3], [1, 2, 3]
    )
    values = to_wire(T)
    assert values[12:15] == [1.0, 2.0, 3.0] and values[3] == values[7] == 0.0
    assert values == np.float32(values).astype(float).tolist()
    assert np.allclose(from_wire(values), T, atol=1e-6)
    for bad in ([1.0] * 16, values[:15], [float("nan")] + values[1:], T.reshape(-1)):
        with pytest.raises(Rejected) as exc:
            from_wire(bad)  # the last one is row-major: translation in row 4
        assert exc.value.code == -3


def test_pose_convention_is_extrinsic_zyx():
    rng = np.random.default_rng(5)
    for _ in range(200):
        x, rz, ry, rx = rng.normal(size=3), *rng.uniform(-3, 3, 3)
        T = matrix_from_pose([*x, rz, ry, rx])
        cz, sz, cy, sy, cx, sx = (
            np.cos(rz),
            np.sin(rz),
            np.cos(ry),
            np.sin(ry),
            np.cos(rx),
            np.sin(rx),
        )
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        assert np.allclose(T[:3, :3], Rx @ Ry @ Rz) and np.allclose(T[:3, 3], x)
        back = pose_from_matrix(T)
        assert -np.pi / 2 <= back[4] <= np.pi / 2
        assert np.allclose(matrix_from_pose(back), T)
    locked = pose_from_matrix(matrix_from_pose([0, 0, 0, 0.7, np.pi / 2, 0.0]))
    assert np.allclose(locked[3:], [0.7, np.pi / 2, 0.0])
    scipy = pytest.importorskip("scipy.spatial.transform")
    angles = [0.3, -1.1, 2.0]
    R = scipy.Rotation.from_euler("zyx", angles).as_matrix()  # vendor's reference
    assert np.allclose(matrix_from_pose([0, 0, 0, *angles])[:3, :3], R)


@needs_pinocchio
@needs_image
def test_sdk_fk_equals_the_controller_mdh_model():
    """Fk through the shared kinematics equals the controller image's own MDH
    chain at the same SDK joints: frames, mapping and units all line up."""
    rng = np.random.default_rng(11)
    for side in ("left", "right"):
        unit = ROBOT.arms[side]
        table = machine_params_mdh(IMAGE / "machine_params.json", f"{side}_arm")
        for q in rng.uniform(unit.sdk_limits[:, 0], unit.sdk_limits[:, 1], (50, 7)):
            R, p = fk_mdh(table, q)
            T = KIN.fk(side, q)
            assert np.allclose(T[:3, :3], R[0], atol=1e-12)
            assert np.allclose(T[:3, 3], p[0], atol=1e-12)


@needs_pinocchio
def test_fkb_is_the_shared_kinematics_flange_in_base_link():
    rng = np.random.default_rng(12)
    waist = ROBOT.units["waist"]
    for side in ("left", "right"):
        unit = ROBOT.arms[side]
        for _ in range(20):
            q = rng.uniform(unit.sdk_limits[:, 0], unit.sdk_limits[:, 1])
            torso = rng.uniform(waist.sdk_limits[:, 0], waist.sdk_limits[:, 1])
            body = dict(zip(waist.sim_joints, waist.to_sim(torso)))
            grasp, R_wrist = KIN.kin.fk(side, unit.to_sim(q), body)
            flange = KIN.flange_in_wrist[side]
            T = KIN.fkb(side, q, torso)
            wrist = grasp - R_wrist @ KIN.tcp[side]
            assert np.allclose(T[:3, :3], R_wrist @ flange[:3, :3], atol=1e-12)
            assert np.allclose(
                T[:3, 3] + KIN.kin.base, wrist + R_wrist @ flange[:3, 3], atol=1e-12
            )


@needs_pinocchio
def test_ik_round_trip_inside_the_aligned_limits():
    rng = np.random.default_rng(2)
    for side in ("left", "right"):
        lo, hi = ROBOT.arms[side].sdk_limits.T
        for _ in range(25):
            q = rng.uniform(lo + 0.05, hi - 0.05)
            target = from_wire(to_wire(KIN.fk(side, q, tool=True)))
            assert np.max(np.abs(KIN.ik(side, target, q) - q)) < 1e-4
            seed = np.clip(q + rng.normal(scale=0.1, size=7), lo, hi)
            reached = KIN.fk(side, KIN.ik(side, target, seed), tool=True)
            assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 1e-5
            assert np.abs(reached[:3, :3] - target[:3, :3]).max() < 1e-4
            base = KIN.fkb(side, q, np.zeros(4))
            back = KIN.ikb(side, from_wire(to_wire(base)), np.zeros(4), q)
            assert np.max(np.abs(back - q)) < 1e-4
    with pytest.raises(Rejected) as exc:
        KIN.ik("left", homogeneous(np.eye(3), [2.0, 0, 0]), np.zeros(7))
    assert exc.value.code == -6


@needs_pinocchio
def test_kinematics_follow_the_physical_joints_in_any_convention():
    raw = load_robot(convention="urdf_passthrough")
    passthrough = SdkKinematics(raw)
    q_urdf = np.array([0.3, 0.2, -0.4, -1.2, 0.5, 0.3, -0.2])
    left = ROBOT.arms["left"]
    for unit, other in zip(ROBOT.units.values(), raw.units.values()):
        physical = [np.sort(u.to_sim(u.sdk_limits.T).T, 1) for u in (unit, other)]
        assert np.allclose(*physical), f"{unit.name}: same limits in both conventions"
    assert np.allclose(
        passthrough.fk("left", q_urdf), KIN.fk("left", left.to_sdk(q_urdf))
    )
    target = KIN.fk("left", left.to_sdk(q_urdf), tool=True)
    assert np.allclose(passthrough.ik("left", target, q_urdf), q_urdf, atol=1e-5)


# ----- control semantics -------------------------------------------------------
def measured(ctl):
    return {
        "units": {
            n: {"q": a.setpoint.copy(), "dq": a.velocity.copy()}
            for n, a in ctl.arms.items()
        },
        "grippers": {
            s: {
                "closure": ctl.grip_setpoint[s],
                "closure_rate": 0.0,
                "external_contact_jaws": 0,
            }
            for s in ctl.grip_setpoint
        },
    }


def run(ctl, seconds, dt=0.002):
    peak_v = peak_a = 0.0
    last = {n: a.velocity.copy() for n, a in ctl.arms.items()}
    for _ in range(int(seconds / dt)):
        ctl.tick(dt)
        for n, a in ctl.arms.items():
            peak_v = max(peak_v, float(np.max(np.abs(a.velocity) / a.vmax)))
            peak_a = max(
                peak_a, float(np.max(np.abs(a.velocity - last[n]) / dt / a.amax))
            )
            last[n] = a.velocity.copy()
    return peak_v, peak_a


def test_segment_respects_limits_and_synchronises():
    vmax, amax = np.ones(3), 2 * np.ones(3)
    seg = Segment(np.zeros(3), np.array([1.0, 0.2, -0.5]), vmax, amax)
    t = np.linspace(0, seg.duration, 2001)
    q = np.array([seg.sample(x)[0] for x in t])
    v = np.gradient(q, t, axis=0)
    assert np.all(np.abs(v) <= vmax * 1.01)
    assert np.allclose(q[-1], [1.0, 0.2, -0.5])
    # All joints reach the goal at the same instant.
    frac = (q - q[0]) / np.array([1.0, 0.2, -0.5])
    assert np.allclose(frac[:, 0], frac[:, 1]) and np.allclose(frac[:, 0], frac[:, 2])


def test_movej_accepted_is_not_arrived():
    ctl = Controller(ROBOT)
    target = np.r_[[0.2, 0.3, 0.0, 0.5, 0.0, 0.0, 0.0] * 2]
    target[8] = -0.3
    assert ctl.move_j(("left", "right"), [target.tolist()], 0.0) == {"status": 0}
    assert ctl.motion_status("left", measured(ctl), 0.0) == 1
    peak_v, peak_a = run(ctl, 3.0)
    assert ctl.motion_status("left", measured(ctl), 3.0) == 0
    assert np.allclose(ctl.arms["right"].setpoint, target[7:])
    assert peak_v <= 1.0001 and peak_a <= 1.0001


def test_rejections_have_sdk_codes():
    ctl = Controller(ROBOT)
    ok = [0.0] * 7
    assert ctl.move_j(("left",), [[0, 0, 0, -0.5, 0, 0, 0]], 0)["status"] == -3
    assert ctl.move_j(("left",), [[0.0] * 6], 0)["status"] == -3
    assert ctl.move_j(("left",), [[float("nan")] + [0.0] * 6], 0)["status"] == -3
    assert ctl.tele_push({"left": ok}, 0, 0.0)["status"] == -10
    assert ctl.tele_start(("left",), 0)["status"] == 0
    assert ctl.move_j(("left",), [ok], 0)["status"] == -10
    assert ctl.set_ratio(("left",), [0], 0)["status"] == -3
    assert ctl.gripper("left", 1001, 0)["status"] == -3
    assert ctl.lifecycle(("right",), 2, 0)["status"] == 0
    assert ctl.move_j(("right",), [ok], 0)["status"] == -5
    assert ctl.lifecycle(("right",), 7, 0)["status"] == -3


def test_soft_stop_brakes_then_gates():
    ctl = Controller(ROBOT)
    ctl.move_j(("left",), [[1.0, 0, 0, 0, 0, 0, 0]], 0)
    run(ctl, 0.5)
    moving = ctl.arms["left"].velocity.copy()
    assert np.max(np.abs(moving)) > 0.1
    ctl.soft_stop(True, 0.5)
    _, peak_a = run(ctl, 1.0)
    assert peak_a <= 1.0001 and not np.any(ctl.arms["left"].velocity)
    assert abs(ctl.arms["left"].setpoint[0] - 1.0) > 0.05
    assert ctl.move_j(("left",), [[0.0] * 7], 1.5)["status"] == -5
    assert ctl.lifecycle(("left",), 1, 1.5)["status"] == -8
    ctl.soft_stop(False, 1.5)
    assert ctl.move_j(("left",), [[0.0] * 7], 1.5)["status"] == 0


def test_teleop_stream_follows_a_smooth_stream_without_stopping():
    ctl = Controller(ROBOT)
    ctl.tele_start(("left",), 0)
    ideal, setpoints, speeds = [], [], []
    for k in range(1, 26):
        x = k / 25
        q = np.zeros(7)
        q[0] = 0.3 * x * x * (3 - 2 * x)  # 0.3 rad in 1 s, within the limits
        assert ctl.tele_push({"left": q.tolist()}, k * 0.04, k * 0.04)["status"] == 0
        for _ in range(20):
            ctl.tick(0.002)
            setpoints.append(ctl.arms["left"].setpoint[0])
            speeds.append(ctl.arms["left"].velocity[0])
    t = np.arange(1, 501) * 0.002 - 0.04  # one point of buffering
    x = np.clip(t, 0, 1)
    ideal = 0.3 * x * x * (3 - 2 * x)
    assert np.max(np.abs(np.array(setpoints) - ideal)) < 0.005
    assert min(speeds[150:350]) > 0.2, "no stop-and-go between points"
    for _ in range(500):
        ctl.tick(0.002)
    assert ctl.arms["left"].setpoint[0] == pytest.approx(0.3)


def test_teleop_keeps_client_pacing_when_points_arrive_in_a_burst():
    """A slow simulator receives a 25 Hz stream in bursts; playback still
    spends 40 ms of simulated time per point, as the client intended."""
    ctl = Controller(ROBOT)
    ctl.tele_start(("left",), 0)
    for k in range(1, 26):
        q = np.zeros(7)
        q[0] = 0.01 * k
        ctl.tele_push({"left": q.tolist()}, 0.0, k * 0.04)
    for _ in range(250):  # 0.5 s of simulated time
        ctl.tick(0.002)
    assert ctl.arms["left"].setpoint[0] == pytest.approx(0.125, abs=0.015)
    assert ctl.arms["left"].velocity[0] == pytest.approx(0.25, abs=0.02)


def test_teleop_pacing_survives_a_stall_and_catch_up_burst():
    """A slow reply stalls a 25 Hz client, whose absolute schedule then sends
    the missed points back to back; the playback keeps 40 ms per point."""
    ctl = Controller(ROBOT)
    ctl.tele_start(("left",), 0)
    stamps = [k * 0.04 for k in range(1, 11)]
    stamps += [0.59, 0.591, 0.592, 0.593, 0.594]  # stall, then points 11-15
    stamps += [0.6 + k * 0.04 for k in range(1, 26)]
    speeds = []
    for k, stamp in enumerate(stamps, start=1):
        q = np.zeros(7)
        q[0] = 0.01 * k  # 0.25 rad/s at 40 ms per point
        ctl.tele_push({"left": q.tolist()}, stamp, stamp)
        for _ in range(20):
            ctl.tick(0.002)
            speeds.append(ctl.arms["left"].velocity[0])
    # After the start-up transient: 0.25 rad/s through the burst and after it.
    assert max(speeds[200:]) < 0.27, "the burst must not be replayed faster"


def test_fault_latches():
    ctl = Controller(ROBOT)
    ctl.latch_fault({"kind": "penetration", "message": "x", "sim_time": 1.0}, 1.0)
    assert ctl.move_j(("left",), [[0.0] * 7], 1.0)["status"] == -9
    assert ctl.lifecycle(("left",), 3, 1.0)["status"] == -9
    assert ctl.motion_status("left", measured(ctl), 1.0) == -1


# ----- protocol, services and runtime over TCP with the stock client ------------
class KinematicPlant(SimBackend):
    """Test double: joints jump to their targets; no physics, no cameras."""

    name = "kinematic-test-double"
    label = "test double"

    def load(self, robot, options, output):
        self.q = {j: 0.0 for u in robot.units.values() for j in u.sim_joints}
        self.q.update({j: 0.0 for g in robot.grippers.values() for j in g.sim_joints})
        self.steps = 0

    def capabilities(self):
        return Capabilities(
            True, True, EFFORT_UNAVAILABLE, (), False, False, ("framework",)
        )

    @property
    def dt(self):
        return 0.002

    def reset(self):
        pass

    def apply(self, targets):
        self.q.update(targets)

    def step(self):
        self.steps += 1

    def read(self):
        return BackendState(
            self.steps * self.dt, self.steps, dict(self.q), {j: 0.0 for j in self.q}, {},
            EFFORT_UNAVAILABLE, [], 0.0,
        )  # fmt: skip

    def render(self, camera, width, height):
        raise NotImplementedError

    def close(self):
        pass


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not VENDOR_SDK:
        pytest.skip("vendor SDK (sdk_review/) not in this checkout")
    out = tmp_path_factory.mktemp("bridge")
    backend = KinematicPlant()
    backend.load(ROBOT, {}, out)
    runtime = Runtime(ROBOT, backend, {"idle_timeout_s": 600}, out)
    arm = Endpoint("arm", "127.0.0.1", 0, arm_catalog(UPPER / "core/client"),
                   ArmService(runtime, ROBOT, backend).dispatch, prefix=ARM_PREFIX).start()  # fmt: skip
    sensor = Endpoint("sensor", "127.0.0.1", 0, SENSOR_ROUTES,
                      SensorService(runtime, ROBOT, backend).dispatch).start()  # fmt: skip
    loop = threading.Thread(target=runtime.run, daemon=True)
    loop.start()
    client, sensors = connect(
        {"host": "127.0.0.1", "arm_port": arm.port, "sensor_port": sensor.port},
        auto_heartbeat=False, auto_reconnect=False, raise_on_error=False,
    )  # fmt: skip
    yield client, sensors, runtime, arm
    sensors.close()
    client.shutdown()
    runtime.stop_event.set()
    loop.join(5)
    arm.close()
    sensor.close()


@needs_vendor_sdk
def test_catalog_covers_the_vendor_client():
    names = arm_catalog(UPPER / "core/client")
    assert {"MoveJ", "EeGripperSetPosition", "MoveL", "GetCurAlarm", "Fk"} <= set(names)
    assert len(names) > 100


def test_stock_client_round_trip(server):
    client, _sensors, runtime, _ = server
    assert client.GetVersion().endswith("kinematic-test-double")
    assert client.AxisCount() == 20
    assert client.MoveJ([[0.1] + [0.0] * 13], side=15) == 0
    deadline = time.monotonic() + 10
    while client.GetMotionStatus(15) != 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert client.GetJointState(0)[1][0] == pytest.approx(0.1)
    sim_q = runtime.backend.q["left_shoulder_pitch_joint"]
    assert sim_q == pytest.approx(-0.1), "SDK j1 maps to minus URDF shoulder pitch"
    state = client.ReadJointState(0)
    assert state["torque"] == [] and state["torque_source"] == EFFORT_UNAVAILABLE


def test_unsupported_and_malformed_raise(server):
    client, sensors, _, _ = server
    from core.libs.inbc_rpc.exceptions import InbcRpcRemoteError

    with pytest.raises(InbcRpcRemoteError, match="UNSUPPORTED"):
        client.GetJointJerkLimit(0)
    with pytest.raises(InbcRpcRemoteError, match="UNSUPPORTED"):
        client.SetMdh(0, [[0.0] * 4] * 7)
    with pytest.raises(InbcRpcRemoteError, match="INVALID_REQUEST"):
        client.rpc("MoveJ", {"side": 0})
    assert client.MoveJ([[0.0] * 7], side=30) == -4
    assert client.GetJointState(5)[0] == -3
    image = sensors.get_image("head")
    assert image.code == -2, "a backend without cameras must not fake an image"
    assert sensors.get_image("nope").code == -1


def test_heartbeat_frame_is_echoed(server):
    import socket
    import struct

    _, _, _, arm = server
    with socket.create_connection(("127.0.0.1", arm.port), timeout=2) as sock:
        header = struct.pack("<BB2xIQI", 39, 0, 0, 0, 0)
        sock.sendall(header)
        assert sock.recv(20) == header


def test_core_layers_import_no_engine():
    """Protocol, services, control, robot, runtime and the server entry point
    must not pull in any physics engine; adapters load only when selected."""
    import subprocess
    import sys

    code = (
        "import sys, simbridge.server, simbridge.protocol, simbridge.services, "
        "simbridge.control, simbridge.robot, simbridge.runtime\n"
        "print([m for m in ('mujoco', 'pybullet', 'isaacsim', 'omni', 'genesis') "
        "if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"


def test_precheck_push_uses_only_the_snapshot():
    from simbridge.control import precheck_push

    arm = {"enabled": True, "mode": "teleop", "teleop_queue": 0}
    snap = {"fault": None, "estop": False, "arms": {"left": dict(arm)}}
    ok_point = {"left": [0.0] * 7}
    assert precheck_push(ROBOT, ok_point, snap) is None
    assert (
        precheck_push(ROBOT, {"left": [0, 0, 0, -0.5, 0, 0, 0]}, snap)["status"] == -3
    )
    idle = {**snap, "arms": {"left": {**arm, "mode": "idle"}}}
    assert precheck_push(ROBOT, ok_point, idle)["status"] == -10
    assert precheck_push(ROBOT, ok_point, {**snap, "estop": True})["status"] == -5
    assert precheck_push(ROBOT, ok_point, {**snap, "fault": {"x": 1}})["status"] == -9
    full = {**snap, "arms": {"left": {**arm, "teleop_queue": 500}}}
    assert precheck_push(ROBOT, ok_point, full)["status"] == -3


def test_stock_client_teleop_stream(server):
    client, _sensors, runtime, _ = server
    assert client.Home() == 0
    deadline = time.monotonic() + 10
    while client.GetMotionStatus(15) != 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert client.TeleJointDual() == 0
    for k in range(1, 11):
        q = [0.01 * k] + [0.0] * 6
        assert client.PushJointTeleopQueueDual(q, q) == 0
        time.sleep(0.02)
    deadline = time.monotonic() + 10
    while client.GetMotionStatus(15) != 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert client.GetJointState(15)[1][0] == pytest.approx(0.1, abs=1e-6)
    assert client.StopTeleJointDual() == 0
    assert client.PushJointTeleopQueueDual([0.0] * 7, [0.0] * 7) == -10
    dropped = [
        e for e in runtime.controller.events if e["event"] == "TeleopPointDropped"
    ]
    assert not dropped


@needs_pinocchio
def test_stock_client_kinematics(server):
    from core.libs.inbc_rpc.exceptions import InbcRpcRemoteError

    client, _sensors, _runtime, _ = server
    assert client.Home() == 0
    deadline = time.monotonic() + 10
    while client.GetMotionStatus(15) != 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    q = [0.2, 0.3, 0.1, 1.0, 0.1, 0.2, 0.3]
    status, fk = client.Fk(0, q)
    assert status == 0 and np.allclose(fk, to_wire(KIN.fk("left", q)))
    assert client.FkTool(0, q)[1] == fk, "identity flan2tool at start-up"
    status, solution = client.IK(0, fk)
    reached = KIN.fk("left", solution)
    assert status == 0 and np.allclose(reached, from_wire(fk), atol=1e-5)
    status, fkb = client.FKb(10, q, [0.0] * 4)
    assert status == 0 and fkb == client.FKb(10, q)[1], "empty torso = measured waist"
    status, solution = client.IKb(10, fkb)
    assert status == 0 and np.allclose(
        KIN.fkb("right", solution, np.zeros(4)), from_wire(fkb), atol=1e-5
    )
    status, pose = client.GetCurCartesianPose(0)
    assert status == 0 and np.allclose(
        matrix_from_pose(pose), KIN.fk("left", np.zeros(7))
    )
    dual = client.GetCurCartesianPoseDual()
    assert dual[0] == 0 and set(dual[1]) == {"left", "right"}
    tool = to_wire(homogeneous(np.eye(3), [0.245, 0.0, 0.0]))
    assert client.SetFlan2tool(0, tool) == 0 and client.GetFlan2tool(0)[1] == tool
    assert np.allclose(
        client.FkTool(0, q)[1], to_wire(KIN.fk("left", q) @ from_wire(tool))
    )
    assert client.SetFlan2tool(0, to_wire(np.eye(4))) == 0
    far = to_wire(homogeneous(np.eye(3), [2.0, 0.0, 0.0]))
    assert client.IK(0, far) == (-6, [])
    assert client.IK(0, [1.0] * 16)[0] == -3
    assert client.Fk(15, q)[0] == -3 and client.Fk(0, q[:6])[0] == -3
    for removed in ("FkLink", "GetMdh", "GetArmEulerAngle"):
        args = (0, q, 7) if removed == "FkLink" else (0,)
        with pytest.raises(InbcRpcRemoteError, match="UNSUPPORTED"):
            getattr(client, removed)(*args)


def test_kinematics_endpoints_unsupported_without_pinocchio(monkeypatch):
    """A server whose Python lacks Pinocchio (the Isaac Sim env) still starts and
    reports the kinematics endpoints as unsupported, never as empty success."""
    from simbridge import services

    def missing(robot):
        raise ImportError("No module named 'pinocchio'")

    monkeypatch.setattr(services, "SdkKinematics", missing)

    class Stub:
        name = "stub"

        def touch(self, method):
            pass

    service = services.ArmService(Stub(), ROBOT, Stub())
    with pytest.raises(Unsupported, match="Pinocchio"):
        service.dispatch("Fk", [{"side": 0, "cur_joints": [0.0] * 7}])


@needs_pinocchio
def test_host_cartesian_goals_move_and_line(server):
    """sdk_cartesian solves on the host and sends SDK joint commands; the
    joints the bridge applies put the grasp points where they were asked."""
    from sdk.sdk_cartesian import CartesianArm, CartesianError

    client, _sensors, runtime, _ = server
    program = json.loads((ROOT / "config/sdk_acceptance/program.json").read_text())
    pose_a = program["pose_a"]["left"] + program["pose_a"]["right"]
    assert client.MoveJ([pose_a], side=15) == 0
    cart = CartesianArm(client)
    cart.wait()

    p0, R0 = cart.pose("left")
    goal = p0 + [0.0, 0.0, 0.05]
    cart.move({"left": goal})
    p1, R1 = cart.pose("left")
    assert np.linalg.norm(p1 - goal) < 1e-4 and np.allclose(R1, R0, atol=1e-3)
    left = ROBOT.arms["left"].sim_joints
    applied = [runtime.backend.q[j] for j in left]  # URDF joints in the plant
    assert np.linalg.norm(cart.kin.fk("left", applied)[0] - goal) < 1e-4

    starts = {s: cart.pose(s)[0] for s in ("left", "right")}
    targets = {s: p + [-0.04, 0.0, 0.0] for s, p in starts.items()}
    result = cart.line(targets, speed=0.05)
    for side, target in targets.items():
        assert np.linalg.norm(cart.pose(side)[0] - target) < 1e-4
        a, b = starts[side], target
        for q in result["paths"][side]:
            p = cart.kin.fk(side, cart.kin.from_sdk(side, q))[0]
            t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
            assert np.linalg.norm(a + t * (b - a) - p) < 1e-4, "off the line"
    assert client.GetMotionStatus(15) == 0

    before = client.GetJointState(15)[1]
    with pytest.raises(CartesianError):
        cart.move({"left": [2.0, 0.0, 1.0]})
    assert client.GetJointState(15)[1] == before, "nothing sent for a failed solve"
