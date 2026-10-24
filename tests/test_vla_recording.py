"""Engine-free tests of the VLA recorder, the expert conversion mapping and the
LeRobot feature set. Run from the project root:

    /Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m pytest tests -q

The recorder test drives the stock SDK clients against an in-process simbridge
runtime whose backend is a test double: joints jump to their targets and each
camera image is one grey level that encodes the physics step it was rendered at.
The runtime runs at 0.1x real time, as config/backends/mujoco_sim_clock.yaml.
"""

import importlib.util
import json
import threading

import numpy as np
import pytest

from data.record_sdk_episode import STREAM, Recorder
from data.vla_episode import (
    ARRAYS,
    CAMERAS,
    LEROBOT_FEATURES,
    EpisodeWriter,
    camera_frames,
    read_episode,
)
from sdk.stock_sdk import UPPER, connect
from simbridge.backends.base import (
    EFFORT_UNAVAILABLE,
    BackendState,
    Capabilities,
    SimBackend,
)
from simbridge.protocol import ARM_PREFIX, SENSOR_ROUTES, Endpoint, arm_catalog
from simbridge.robot import GripperMap, load_robot
from simbridge.runtime import Runtime
from simbridge.services import ArmService, SensorService

ROBOT = load_robot()


def grey(step):
    return 20 + step % 200


class CameraPlant(SimBackend):
    """Test double with the four policy cameras; no physics."""

    name = "camera-test-double"
    label = "test double"

    def load(self, robot, options, output):
        self.q = {j: 0.0 for u in robot.units.values() for j in u.sim_joints}
        self.q.update({j: 0.0 for g in robot.grippers.values() for j in g.sim_joints})
        self.steps = 0

    def capabilities(self):
        cameras = tuple(spec.sim_name for spec in ROBOT.cameras.values())
        return Capabilities(
            True, True, EFFORT_UNAVAILABLE, cameras, False, False, ("framework",)
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
        return np.full((height, width, 3), grey(self.steps), np.uint8)

    def close(self):
        pass


@pytest.fixture
def bridge(tmp_path):
    if not UPPER.is_dir():
        pytest.skip("vendor SDK (sdk_review/) not in this checkout")
    backend = CameraPlant()
    backend.load(ROBOT, {}, tmp_path)
    options = {"idle_timeout_s": 600, "realtime_factor": 0.1}
    runtime = Runtime(ROBOT, backend, options, tmp_path)
    arm = Endpoint("arm", "127.0.0.1", 0, arm_catalog(UPPER / "core/client"),
                   ArmService(runtime, ROBOT, backend).dispatch, prefix=ARM_PREFIX).start()  # fmt: skip
    sensor = Endpoint("sensor", "127.0.0.1", 0, SENSOR_ROUTES,
                      SensorService(runtime, ROBOT, backend).dispatch).start()  # fmt: skip
    loop = threading.Thread(target=runtime.run, daemon=True)
    loop.start()
    profile = {"host": "127.0.0.1", "arm_port": arm.port, "sensor_port": sensor.port}
    client, sensors = connect(profile, auto_heartbeat=False, auto_reconnect=False)
    sensors.close()
    yield client, profile
    client.shutdown()
    runtime.stop_event.set()
    loop.join(5)
    arm.close()
    sensor.close()


class Ramp:
    """Streams left j1 up by 0.01 rad per tick and half-closes the left
    gripper at tick 3."""

    def __init__(self, ticks):
        self.ticks = ticks

    def describe(self):
        return {"name": "ramp", "ticks": self.ticks}

    def start(self, rec):
        assert rec.arm.TeleJointDual() == 0

    def step(self, rec, obs):
        k = rec.tick
        if k >= self.ticks:
            return False
        rec.subtask = "close" if k >= 3 else "open"
        if k == 3:
            assert rec.arm.GripperSetPosition(0, 600) == 0
        left = [0.01 * (k + 1)] + [0.0] * 6
        assert rec.arm.PushJointTeleopQueueDual(left, [0.0] * 7) == 0
        return True

    def stop(self, rec):
        rec.arm.StopTeleJointDual()


def test_recorder_round_trip_through_the_stock_sdk(bridge, tmp_path):
    client, profile = bridge
    out = tmp_path / "episode"
    info = Recorder(client, profile, out, "test task").run(Ramp(12))
    assert info["complete"] and info["clock"] == "sim" and info["images"] == "fresh"
    assert info["reads"]["state"] == "GetJointState, dual"
    assert info["subtasks"] == ["open", "close"]
    _, a = read_episode(out)
    assert info["frames"] == 12
    # Ticks are due on whole periods of simulated time; the state is read at
    # the first physics step at or after that.
    due = a["robot_s"] - a["tick_late_s"]
    assert np.allclose(due / 0.04, np.round(due / 0.04), atol=1e-6)
    assert np.allclose(np.diff(due), 0.04, atol=1e-9)
    assert (a["tick_late_s"] > -1e-9).all() and (a["tick_late_s"] < 0.0041).all()
    # The action is the point pushed at that tick, held per arm and gripper.
    assert np.allclose(a["action"][:, 0], 0.01 * np.arange(1, 13))
    assert np.allclose(a["action"][:, 1:14], 0.0)
    assert a["action"][:3, 14].tolist() == [1000] * 3
    assert a["action"][3:, 14].tolist() == [600] * 9
    assert (a["action"][:, 15] == 1000).all() and (a["arm_source"] == STREAM).all()
    assert a["state"][0, 14:].tolist() == [1000, 1000]
    # Once the stream is under way the servo setpoint is the point pushed two
    # ticks earlier (simbridge plays one point behind, interpolating).
    assert np.abs(a["setpoint"][-4:, 0] - a["action"][-6:-2, 0]).max() < 0.004
    assert (np.diff(a["camera_seq"], axis=0) == 1).all()
    # Each image is rendered within a few physics steps of the state read.
    steps = np.round(a["robot_s"] / 0.002).astype(int)
    for camera in CAMERAS:
        means = [frame.mean() for frame in camera_frames(out, camera)]
        assert len(means) == 12
        lag = [
            min(abs(m - grey(s + d)) for d in range(4)) for m, s in zip(means, steps)
        ]
        assert max(lag) < 3
    calls = [c["method"] for c in json.loads((out / "commands.json").read_text())]
    assert calls[0] == "TeleJointDual" and calls[-1] == "StopTeleJointDual"
    assert calls.count("PushJointTeleopQueueDual") == 12


def test_only_accepted_commands_become_the_action(tmp_path):
    rec = Recorder(None, {}, tmp_path / "unused", "task", clock="wall", images="fresh")
    rec.targets = {"left": np.zeros(7), "right": np.zeros(7)}
    rec.grips = {"left": 1000.0, "right": 1000.0}
    rec.sources = {"left": 0, "right": 0}
    path = [[0.1] * 14, [0.2] * 7 + [0.3] * 7]
    rec.command("MoveJ", {"joint_path": path, "side": None}, 0, None, 0.0)
    assert rec.targets["left"].tolist() == [0.2] * 7
    assert rec.targets["right"].tolist() == [0.3] * 7
    rec.command("MoveJ", {"joint_path": [[0.5] * 7], "side": 10}, 0, None, 0.0)
    assert rec.targets["left"].tolist() == [0.2] * 7
    assert rec.targets["right"].tolist() == [0.5] * 7
    rec.command("PushJointTeleopQueue", {"side": 0, "target_joints": [0.4] * 7}, -3,
                None, 0.0)  # fmt: skip
    assert rec.targets["left"].tolist() == [0.2] * 7  # rejected: unchanged
    rec.command("PushJointTeleopQueue", {"side": 0, "target_joints": [0.4] * 7}, 0,
                None, 0.0)  # fmt: skip
    assert rec.targets["left"].tolist() == [0.4] * 7
    assert rec.sources == {"left": 1, "right": 2}
    rec.command("Home", {"side": None}, 0, None, 0.0)
    assert rec.targets["left"].tolist() == [0.0] * 7 and rec.sources["left"] == 2
    rec.command("GripperClose", {"side": 10, "timeout_ms": 500}, 0, None, 0.0)
    rec.command("GripperSetPosition", {"side": 0, "position": 250}, 0, None, 0.0)
    rec.command("GripperOpen", {"side": 0}, None, "ArmSdkError: -5", 0.0)
    assert rec.grips == {"left": 250.0, "right": 0.0}
    assert len(rec.commands) == 8


def test_episode_round_trip(tmp_path):
    writer = EpisodeWriter(tmp_path / "ep")
    for k in range(3):
        row = {key: np.full(shape, k) for key, (_, shape) in ARRAYS.items()}
        row["subtask"] = "a" if k < 2 else "b"
        writer.add(row, {c: np.full((480, 640, 3), 60 * k, np.uint8) for c in CAMERAS})
    info = writer.close({"task": "t", "complete": True})
    loaded, a = read_episode(tmp_path / "ep")
    assert loaded["frames"] == info["frames"] == 3
    assert loaded["subtasks"] == ["a", "b"] and loaded["task"] == "t"
    assert a["subtask_index"].tolist() == [0, 0, 1]
    assert a["state"][2].tolist() == [2.0] * 16
    means = [f.mean() for f in camera_frames(tmp_path / "ep", "waist")]
    assert np.allclose(means, [0, 60, 120], atol=2)


@pytest.mark.skipif(
    importlib.util.find_spec("pinocchio") is None, reason="needs Pinocchio"
)
def test_expert_conversion_matches_what_the_sdk_reports():
    """The converter's URDF -> SDK mapping equals simbridge's for the same
    simulated joints, so converted expert data and SDK recordings agree."""
    from data.convert_expert_run import to_sdk
    from robot_kinematics import Kinematics

    rng = np.random.default_rng(0)
    values = np.c_[rng.uniform(-1.5, 1.5, (50, 14)), rng.uniform(0, 1, (50, 2))]
    expected = np.c_[
        ROBOT.arms["left"].to_sdk(values[:, :7]),
        ROBOT.arms["right"].to_sdk(values[:, 7:14]),
        [[GripperMap.position_from_closure(g) for g in row] for row in values[:, 14:]],
    ]
    assert np.array_equal(to_sdk(Kinematics(), values), expected)


def test_lerobot_features_keep_extras_out_of_policy_inputs():
    """LeRobot feeds every observation.* key to a policy as state and every
    action* key as action: only the state, the cameras and the action may."""
    policy = {"observation.state", "action"} | {
        f"observation.images.{c}" for c in CAMERAS
    }
    for key in LEROBOT_FEATURES:
        if key.startswith(("observation", "action")):
            assert key in policy
    assert policy <= set(LEROBOT_FEATURES)
