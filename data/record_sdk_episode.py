"""Record one VLA training episode through the unmodified V1R1C9 SDK.

    python -m data.record_sdk_episode --connection config/connections/sim_mujoco.yaml \
        --output outputs/<new_episode> [--driver replay|passive]

The same code records the simulator and the real robot: only the connection
profile changes. Each 25 Hz tick reads GetJointState, GripperReadPosition for
both grippers, GetDesireJointPos and the four get_image cameras, then lets the
driver act. The driver commands the robot only through ``Recorder.arm``: it
forwards every call to the stock ArmClient and logs it, and an accepted MoveJ,
Home, PushJointTeleopQueue(Dual) or GripperSetPosition/Open/Close becomes the
episode's action. The output is a raw episode (vla_episode.py); turn it into a
LeRobot dataset with export_lerobot.py.

Clocks. On the real robot ticks follow the host's monotonic clock, and camera
workers keep the latest frame of each camera ("latest" images). The simulator
renders each 640x480 camera on its physics thread (about 19 ms), so four
cameras cannot keep 25 Hz in real time. When ReadJointState reports a
``sim_time`` (simbridge does, the controller does not), ticks follow simulated
time and every tick waits for four fresh renders, which share the state's
physics step. That needs a server slowed to 0.1x real time
(config/backends/mujoco_sim_clock.yaml): see the README.

Drivers: ``replay`` streams the expert's task from config/sdk_acceptance/
program.json (simulation-tuned, so loopback hosts only); ``passive`` sends no
command and only observes. Other drivers (teleop, a policy) implement
start/step/stop like these and are passed to ``Recorder.run``.
"""

import argparse
import inspect
import ipaddress
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data.vla_episode import CAMERAS, FPS, HEIGHT, KITTING_TASK, WIDTH, EpisodeWriter
from sdk.stock_sdk import ROOT, connect, load, read_profile

SIDES = {0: ("left",), 10: ("right",), 15: ("left", "right")}
# arm_source codes of commanded actions (vla_episode.ARM_SOURCES).
STREAM, MOVEJ = 1, 2


def utc_ms():
    return time.time_ns() // 1_000_000


def brief(value):
    """JSON-friendly copy of an SDK argument or return value."""
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {k: brief(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [brief(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else str(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def status_of(result):
    if isinstance(result, dict):
        return result.get("status")
    return result


class RecordedArm:
    """The stock ArmClient as the driver sees it: calls are forwarded unchanged,
    logged, and accepted commands update the recorder's action."""

    def __init__(self, arm, recorder):
        self._arm = arm
        self._recorder = recorder

    def __getattr__(self, name):
        target = getattr(self._arm, name)
        if not callable(target):
            return target

        def call(*args, **kwargs):
            bound = inspect.signature(target).bind(*args, **kwargs)
            bound.apply_defaults()
            t0 = time.monotonic()
            result = error = None
            try:
                result = target(*args, **kwargs)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self._recorder.command(name, bound.arguments, result, error, t0)
            return result

        return call


class Cameras:
    """Stock ModuleCommunicationClients, one per camera by default so the four
    get_image requests run concurrently.

    ``grab()`` returns {camera: (rgb, seq, utc_ms, host_ms)}. Fresh mode fetches
    all cameras now and waits; latest mode returns what background workers
    fetched last (each at most at the control rate) and never blocks a tick.
    """

    def __init__(self, profile, latest, shared=False, timeout=30.0):
        _, Client, _, _ = load()
        self.timeout = timeout
        self.clients = {}
        for camera in CAMERAS:
            if shared and self.clients:
                self.clients[camera] = next(iter(self.clients.values()))
                continue
            client = Client(profile["host"], int(profile["sensor_port"]))
            client.connect(timeout_sec=float(profile.get("connect_timeout_s", 5.0)))
            self.clients[camera] = client
        self.pool = ThreadPoolExecutor(len(CAMERAS))
        self.latest = latest
        self.frames = {}
        self.errors = {}
        self.lock = threading.Lock()
        self.running = threading.Event()
        self.workers = []
        if latest:
            self.running.set()
            for camera in CAMERAS:
                worker = threading.Thread(target=self._worker, args=(camera,))
                worker.daemon = True
                worker.start()
                self.workers.append(worker)
            deadline = time.monotonic() + timeout
            while len(self.frames) < len(CAMERAS):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no first image from {self.errors}")
                time.sleep(0.01)

    def fetch(self, camera):
        res = self.clients[camera].get_image(camera, timeout_sec=self.timeout)
        host = utc_ms()
        if res.code != 0:
            raise RuntimeError(f"get_image({camera}) code {res.code}: {res.err_msg}")
        channels = {"RGBA8": 4, "RGB8": 3}.get(res.format)
        if channels is None or (res.width, res.height) != (WIDTH, HEIGHT):
            raise RuntimeError(
                f"get_image({camera}): {res.width}x{res.height} {res.format}, "
                f"expected {WIDTH}x{HEIGHT} RGBA8 or RGB8"
            )
        pixels = np.frombuffer(res.data, np.uint8).reshape(HEIGHT, WIDTH, channels)
        return np.ascontiguousarray(pixels[:, :, :3]), res.seq, res.timestamp, host

    def _worker(self, camera):
        period = 1.0 / FPS
        while self.running.is_set():
            t0 = time.monotonic()
            try:
                frame = self.fetch(camera)
                with self.lock:
                    self.frames[camera] = frame
            except Exception as exc:  # noqa: BLE001 - kept, retried, surfaced
                self.errors[camera] = f"{type(exc).__name__}: {exc}"
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def grab(self):
        if not self.latest:
            futures = {c: self.pool.submit(self.fetch, c) for c in CAMERAS}
            return {c: f.result(timeout=self.timeout) for c, f in futures.items()}
        with self.lock:
            frames = dict(self.frames)
        stale = [c for c, f in frames.items() if utc_ms() - f[3] > 1000]
        if stale:
            raise RuntimeError(f"no image for 1 s from {stale}: {self.errors}")
        return frames

    def close(self):
        self.running.clear()
        for worker in self.workers:
            worker.join(timeout=self.timeout)
        self.pool.shutdown()
        for client in set(self.clients.values()):
            client.close()


class Clock:
    """The robot's clock: host monotonic seconds, or simbridge's sim_time."""

    def __init__(self, arm, mode):
        self.arm = arm
        self.mode = mode
        self.origin = time.monotonic()

    def now(self):
        if self.mode == "sim":
            return float(self.arm.ReadJointState(0)["sim_time"])
        return time.monotonic() - self.origin

    def wait_until(self, due):
        """Robot time once it has reached ``due``."""
        while (now := self.now()) < due - 1e-6:
            if self.mode == "wall":
                time.sleep(due - now)
            else:
                time.sleep(0.002)
        return now


class Recorder:
    def __init__(
        self, arm, profile, output, task, clock="auto", images="auto", shared=False
    ):
        self.raw = arm
        self.arm = RecordedArm(arm, self)
        self.profile = profile
        self.task = task
        self.subtask = task
        if clock == "auto":
            clock = "sim" if "sim_time" in arm.ReadJointState(0) else "wall"
        if images == "auto":
            images = "fresh" if clock == "sim" else "latest"
        self.clock = Clock(arm, clock)
        self.images = images
        self.shared = shared
        self.output = Path(output)
        self.commands = []
        self.tick = -1
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.reads = {}

    # ----- actions -----------------------------------------------------------
    def command(self, name, arguments, result, error, t0):
        """Log one driver call; accepted commands become the action."""
        status = status_of(result)
        self.commands.append(
            {
                "tick": self.tick,
                "host_s": round(t0 - self.t0, 4),
                "utc_ms": utc_ms(),
                "method": name,
                "args": brief(dict(arguments)),
                "result": brief(result),
                "error": error,
                "latency_ms": round((time.monotonic() - t0) * 1000, 2),
            }
        )
        if error is not None or status != 0:
            return
        a = arguments
        with self.lock:
            if name in ("MoveJ", "Home"):
                side = 15 if a.get("side") is None else int(a["side"])
                goal = a["joint_path"][-1] if name == "MoveJ" else [0.0] * 14
                for i, unit in enumerate(SIDES[side]):
                    self.targets[unit] = np.asarray(goal[7 * i : 7 * i + 7], float)
                    self.sources[unit] = MOVEJ
            elif name == "PushJointTeleopQueue":
                (unit,) = SIDES[int(a["side"])]
                self.targets[unit] = np.asarray(a["target_joints"], float)
                self.sources[unit] = STREAM
            elif name == "PushJointTeleopQueueDual":
                for unit in ("left", "right"):
                    self.targets[unit] = np.asarray(a[f"{unit}_joints"], float)
                    self.sources[unit] = STREAM
            elif name in ("GripperSetPosition", "GripperOpen", "GripperClose"):
                (unit,) = SIDES[int(a["side"])]
                value = {"GripperOpen": 1000, "GripperClose": 0}.get(name)
                self.grips[unit] = float(a["position"] if value is None else value)

    # ----- observations ------------------------------------------------------
    def _joints(self, method, mode):
        """14 arm values from one side-15 call ("dual") or from the left and
        right calls ("per_arm"); the SDK documents side 15 only for some reads."""
        read = getattr(self.raw, method)
        parts = [(15, 14)] if mode == "dual" else [(0, 7), (10, 7)]
        values = []
        for side, count in parts:
            value = read(side)
            status, q = value if method == "GetJointState" else (0, value)
            if status != 0 or len(q) != count:
                raise RuntimeError(f"{method}({side}) returned status {status}, {q}")
            values.extend(q)
        return np.asarray(values, float)

    def probe(self):
        """Choose how the state and the servo setpoint are read, once."""
        for method in ("GetJointState", "GetDesireJointPos"):
            self.reads[method] = "unavailable"
            for mode in ("dual", "per_arm"):
                try:
                    self._joints(method, mode)
                except Exception as exc:  # noqa: BLE001 - try the next mode
                    failure = exc
                    continue
                self.reads[method] = mode
                break
            if self.reads["GetJointState"] == "unavailable":
                raise failure

    def observe(self):
        state_ms = utc_ms()
        q = self._joints("GetJointState", self.reads["GetJointState"])
        grips = []
        for side in (0, 10):
            res = self.raw.GripperReadPosition(side)
            if res.get("status") != 0:
                raise RuntimeError(f"GripperReadPosition({side}) returned {res}")
            grips.append(float(res["value"]))
        setpoint = np.full(14, np.nan)
        if self.reads["GetDesireJointPos"] != "unavailable":
            setpoint = self._joints(
                "GetDesireJointPos", self.reads["GetDesireJointPos"]
            )
        return {
            "state": np.r_[q, grips],
            "setpoint": setpoint,
            "state_utc_ms": state_ms,
            "images": self.cameras.grab(),
        }

    # ----- episode -----------------------------------------------------------
    def run(self, driver, max_frames=None, cameras=None):
        """Record until the driver's step returns False (that tick is not
        recorded) or max_frames. The episode is written also when a read, the
        driver or Ctrl-C ends it early, marked incomplete; returns episode.json.
        ``cameras`` replaces the stock-SDK Cameras (tests)."""
        period = 1.0 / FPS
        version = self.raw.GetVersion()
        self.probe()
        self.cameras = cameras or Cameras(
            self.profile, self.images == "latest", self.shared
        )
        try:
            writer = EpisodeWriter(self.output)
        except BaseException:
            self.cameras.close()
            raise
        started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        error, slips, wall = None, 0, []
        try:
            first = self.observe()
            self.targets = {"left": first["state"][:7], "right": first["state"][7:14]}
            self.grips = {"left": first["state"][14], "right": first["state"][15]}
            self.sources = {"left": 0, "right": 0}
            driver.start(self)
            anchor, k0 = self.clock.now(), 0
            if self.clock.mode == "sim":
                # Ticks on whole periods of simulated time, like the server's trace.
                anchor = math.ceil(anchor / period - 1e-6) * period
            while max_frames is None or len(wall) < max_frames:
                due = anchor + (self.tick + 1 - k0) * period
                now = self.clock.wait_until(due)
                late = now - due
                if late > period:  # restart the schedule rather than burst
                    anchor, k0, slips = now, self.tick + 1, slips + 1
                obs = self.observe()
                self.tick += 1
                if not driver.step(self, obs):
                    break
                wall.append(time.monotonic())
                with self.lock:
                    action = np.r_[
                        self.targets["left"],
                        self.targets["right"],
                        self.grips["left"],
                        self.grips["right"],
                    ]
                    sources = [self.sources["left"], self.sources["right"]]
                images = {c: obs["images"][c][0] for c in CAMERAS}
                seq, stamp, host = zip(*(obs["images"][c][1:] for c in CAMERAS))
                row = {
                    "state": obs["state"],
                    "action": action,
                    "setpoint": obs["setpoint"],
                    "arm_source": sources,
                    "robot_s": now,
                    "state_utc_ms": obs["state_utc_ms"],
                    "camera_seq": seq,
                    "camera_utc_ms": stamp,
                    "camera_host_ms": host,
                    "tick_late_s": late,
                    "subtask": self.subtask,
                }
                writer.add(row, images)
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - saved, incomplete
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                driver.stop(self)
            except Exception as exc:  # noqa: BLE001 - recorded, episode still saved
                error = error or f"driver stop: {type(exc).__name__}: {exc}"
            self.cameras.close()
        timing, warnings = self.timing(writer.rows, wall, slips)
        meta = {
            "source": "sdk_recorder",
            "complete": error is None,
            "error": error,
            "task": self.task,
            "driver": driver.describe(),
            "connection": {
                k: self.profile.get(k)
                for k in ("label", "host", "arm_port", "sensor_port")
            },
            "sdk_version": version,
            "reads": {
                "state": f"GetJointState, {self.reads['GetJointState']}",
                "setpoint": f"GetDesireJointPos, {self.reads['GetDesireJointPos']}",
            },
            "clock": self.clock.mode,
            "images": self.images,
            "started_utc": started,
            "timing": timing,
            "warnings": warnings,
        }
        return writer.close(meta, self.commands)

    def timing(self, rows, wall, slips):
        """Summary of how evenly the episode was sampled, plus warnings."""
        if len(wall) < 2:
            return {"frames": len(wall)}, ["fewer than two frames"]
        a = {key: np.asarray(values) for key, values in rows.items()}
        robot = np.diff(a["robot_s"])
        gaps = np.diff(wall)
        repeats = (np.diff(a["camera_seq"], axis=0) == 0).sum(axis=0)
        transfer = a["camera_host_ms"] - a["camera_utc_ms"]
        age = a["state_utc_ms"][:, None] - a["camera_utc_ms"]

        def spread(values):
            return {
                c: [float(np.median(v)), float(v.max())]
                for c, v in zip(CAMERAS, values.T)
            }

        timing = {
            "frames": len(wall),
            "robot_s": [float(a["robot_s"][0]), float(a["robot_s"][-1])],
            "robot_period_s": [float(robot.min()), float(robot.max())],
            "late_ticks": int((a["tick_late_s"] > 0.25 / FPS).sum()),
            "max_late_s": float(a["tick_late_s"].max()),
            "schedule_restarts": slips,
            "wall_s": float(wall[-1] - wall[0]),
            "wall_period_s": [
                float(gaps.min()),
                float(np.median(gaps)),
                float(gaps.max()),
            ],
            "robot_per_wall_s": float(robot.sum() / gaps.sum()),
            "repeated_images": dict(zip(CAMERAS, map(int, repeats))),
            # [median, max] in ms; age < 0: the image was taken after the state.
            "image_transfer_ms": spread(transfer),
            "image_age_at_state_read_ms": spread(age),
        }
        warnings = []
        if slips or timing["late_ticks"]:
            warnings.append(
                f"{timing['late_ticks']} ticks started over 10 ms late and the "
                f"schedule restarted {slips} times: frames are not evenly spaced"
            )
        if repeats.any():
            warnings.append(f"repeated camera frames: {timing['repeated_images']}")
        streamed = (a["arm_source"] == STREAM).any()
        if self.clock.mode == "sim" and streamed and gaps.min() < 0.25:
            warnings.append(
                "simbridge plays a teleop point over the wall-clock gap between "
                "pushes when it is under teleop_max_gap_s (0.2 s): with ticks "
                f"{gaps.min():.3f} s apart the robot moved slower than the 25 Hz "
                "targets; use config/backends/mujoco_sim_clock.yaml"
            )
        return timing, warnings


class Passive:
    """Observes only; the action holds the first measured state."""

    def __init__(self, seconds):
        self.frames = round(seconds * FPS)

    def describe(self):
        return {"name": "passive", "frames": self.frames}

    def start(self, rec):
        pass

    def step(self, rec, obs):
        return rec.tick < self.frames

    def stop(self, rec):
        pass


class ProgramReplay:
    """The expert's task as SDK joint targets: one PushJointTeleopQueueDual
    point per tick and the phase's gripper positions at each phase start
    (config/sdk_acceptance/program.json, made by sdk/make_sdk_program.py). Subtask
    texts come from the program's source expert run."""

    def __init__(self, program_path, until=None):
        self.path = Path(program_path)
        program = json.loads(self.path.read_text())
        self.source = program["task"]["source_run"]
        texts = {}
        trace = ROOT / self.source / "trace.json"
        if trace.exists():
            texts = {s["phase"]: s["subtask"] for s in json.loads(trace.read_text())}
        self.points = []
        for phase in program["task"]["phases"]:
            text = texts.get(phase["name"], phase["name"])
            for q in phase["points"]:
                self.points.append((phase["name"], text, phase["gripper"], q))
            if phase["name"] == until:
                break
        else:
            if until is not None:
                raise ValueError(f"no phase {until!r} in {program_path}")
        self.grip = None
        self.phases = list(dict.fromkeys(p[0] for p in self.points))

    def describe(self):
        return {
            "name": "replay",
            "program": str(self.path.relative_to(ROOT)),
            "source_run": self.source,
            "phases": self.phases,
            "points": len(self.points),
        }

    def start(self, rec):
        q = np.asarray(self.points[0][3])
        measured = rec.targets["left"].tolist() + rec.targets["right"].tolist()
        if np.max(np.abs(np.asarray(measured) - q)) > 0.05:
            raise RuntimeError("the arms are not at the program's start pose")
        if rec.arm.TeleJointDual() != 0:
            raise RuntimeError("TeleJointDual refused")

    def step(self, rec, obs):
        if rec.tick >= len(self.points):
            return False
        _, text, gripper, q = self.points[rec.tick]
        rec.subtask = text
        if gripper != self.grip:
            for side, position in zip((0, 10), gripper):
                if rec.arm.GripperSetPosition(side, position) != 0:
                    raise RuntimeError(f"GripperSetPosition({side}) refused")
            self.grip = list(gripper)
        if rec.arm.PushJointTeleopQueueDual(q[:7], q[7:]) != 0:
            raise RuntimeError(f"tick {rec.tick}: PushJointTeleopQueueDual refused")
        return True

    def stop(self, rec):
        rec.arm.StopTeleJointDual()


def loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--connection", required=True, help="config/connections/*.yaml")
    parser.add_argument("--output", type=Path, required=True, help="new episode folder")
    parser.add_argument("--driver", choices=["replay", "passive"], default="replay")
    parser.add_argument(
        "--program", type=Path, default=ROOT / "config/sdk_acceptance/program.json"
    )
    parser.add_argument("--until", help="replay: last program phase (default: all)")
    parser.add_argument("--seconds", type=float, default=10.0, help="passive length")
    parser.add_argument("--task", default=KITTING_TASK, help="episode instruction")
    parser.add_argument("--clock", choices=["auto", "wall", "sim"], default="auto")
    parser.add_argument("--images", choices=["auto", "fresh", "latest"], default="auto")
    parser.add_argument(
        "--one-camera-connection",
        action="store_true",
        help="fetch the cameras in turn over one sensor connection",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists")
    profile = read_profile(args.connection)
    if args.driver == "replay":
        if not loopback(profile["host"]):
            raise SystemExit(
                "the replay driver's motions are simulation-tuned: loopback hosts only"
            )
        driver = ProgramReplay(args.program, args.until)
    else:
        driver = Passive(args.seconds)
    arm, sensors = connect(profile)
    sensors.close()  # Cameras opens its own connections
    try:
        recorder = Recorder(
            arm,
            profile,
            args.output,
            args.task,
            args.clock,
            args.images,
            args.one_camera_connection,
        )
        print(
            f"recording {args.output} from {profile.get('label', profile['host'])}: "
            f"clock {recorder.clock.mode}, images {recorder.images}",
            flush=True,
        )
        info = recorder.run(driver)
    finally:
        arm.shutdown()
    summary = {k: info[k] for k in ("complete", "error", "frames", "clock", "images")}
    print(json.dumps({**summary, "timing": info["timing"]}, indent=2))
    for warning in info["warnings"]:
        print("WARNING:", warning)
    if not info["complete"]:
        raise SystemExit(f"episode incomplete: {info['error']}")


if __name__ == "__main__":
    main()
