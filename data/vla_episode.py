"""Raw VLA episodes: the one schema shared by the SDK recorder and the expert
converter. export_lerobot.py turns any set of them into a LeRobot v3.0 dataset.

One folder per episode:

    episode.json        source, conventions, task and subtask texts, timing summary
    frames.npz          one row per 25 Hz control tick (ARRAYS)
    commands.json       every SDK call the driver made (SDK recordings only)
    cameras/<name>.mp4  one H.264 frame per row, 640x480 RGB, named as in get_image

Convention, identical for the simulator and the real robot, so a policy trained
on either drives the robot through the stock SDK unchanged:

- state[k], measured at tick k: 14 arm joints in SDK (controller MDH) radians,
  left j1..j7 then right j1..j7, as GetJointState reports them
  (q_sdk = sign * q_urdf + offset, config/robot/franzi_sdk.yaml), then the left
  and right gripper positions as GripperReadPosition reports them (0 closed,
  1000 open).
- action[k], decided after observing tick k: per arm the last accepted joint
  target (a PushJointTeleopQueue point, or a MoveJ's last waypoint), and per
  side the last accepted gripper position, in the same units. Targets are held
  until the next command; before the first one they are the measured state.
- setpoint[k]: the joint target the servos track when state[k] is read
  (GetDesireJointPos); the gap to action shows the command latency.
- arm_source[k], per arm: what action[k] comes from (ARM_SOURCES).

Ground-truth object state never enters an episode.
"""

import json
import queue
import shutil
import threading
from pathlib import Path

import av
import numpy as np

SCHEMA = "inovance-franzi-vla-episode/1"
ROBOT = "inovance_franzi_dual_arm"  # config/robot/franzi_sdk.yaml name
FPS = 25
WIDTH, HEIGHT = 640, 480
CAMERAS = ("head", "waist", "wrist_left", "wrist_right")
JOINT_NAMES = [f"{side}_j{i}" for side in ("left", "right") for i in range(1, 8)]
STATE_NAMES = JOINT_NAMES + ["left_gripper", "right_gripper"]
ARM_SOURCES = {
    0: "no command yet: holds the measured joints of the first tick",
    1: "streamed joint target (PushJointTeleopQueue[Dual], or the expert's 25 Hz target)",
    2: "MoveJ or Home: the last waypoint",
}
KITTING_TASK = (
    "Pick up the blue and orange parts with both arms at the same time "
    "and place them on the target pads."
)
ARRAYS = {
    "state": (np.float64, (16,)),
    "action": (np.float64, (16,)),
    "setpoint": (np.float64, (14,)),
    "arm_source": (np.int8, (2,)),
    # Robot clock at the state read: simulated time, or host monotonic seconds.
    "robot_s": (np.float64, ()),
    # Host UTC milliseconds at the state read; -1 where there is no wall clock.
    "state_utc_ms": (np.int64, ()),
    # Per camera in CAMERAS order: get_image seq and timestamp (UTC ms), and the
    # host UTC ms when the image arrived; -1 where unknown.
    "camera_seq": (np.int64, (4,)),
    "camera_utc_ms": (np.int64, (4,)),
    "camera_host_ms": (np.int64, (4,)),
    # Seconds the tick started after its scheduled robot-clock time.
    "tick_late_s": (np.float64, ()),
    "subtask_index": (np.int16, ()),
}
VIDEO_FEATURE = {
    "dtype": "video",
    "shape": (HEIGHT, WIDTH, 3),
    "names": ["height", "width", "channels"],
}
# The LeRobot feature set (export_lerobot.py). LeRobot feeds every observation.*
# key to a policy as state, so the setpoint, the action source and the
# timestamps live under command.* and timing.* instead.
LEROBOT_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
    "action": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
    **{f"observation.images.{c}": VIDEO_FEATURE for c in CAMERAS},
    "command.setpoint": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
    "command.arm_source": {"dtype": "int64", "shape": (2,), "names": ["left", "right"]},
    "timing.robot_s": {"dtype": "float64", "shape": (1,), "names": None},
    "timing.state_utc_ms": {"dtype": "int64", "shape": (1,), "names": None},
    "timing.camera_seq": {"dtype": "int64", "shape": (4,), "names": list(CAMERAS)},
    "timing.camera_utc_ms": {"dtype": "int64", "shape": (4,), "names": list(CAMERAS)},
    "subtask": {"dtype": "string", "shape": (1,), "names": None},
}


class Video:
    """H.264 writer on its own thread, so encoding never stalls a control tick.
    Same settings as sim/run_expert.py's policy-camera videos."""

    def __init__(self, path, fps=FPS, depth=50):
        self.container = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = WIDTH, HEIGHT
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "20", "preset": "fast"}
        self.frames = 0
        self.error = None
        self.queue = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while (rgb := self.queue.get()) is not None:
            try:
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in self.stream.encode(frame):
                    self.container.mux(packet)
                self.frames += 1
            except Exception as exc:  # noqa: BLE001 - raised by write/close
                self.error = exc

    def write(self, rgb):
        if self.error:
            raise self.error
        self.queue.put(rgb)

    def close(self):
        self.queue.put(None)
        self.thread.join()
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        if self.error:
            raise self.error


class EpisodeWriter:
    """Accumulates rows and camera frames; ``close`` writes the folder."""

    def __init__(self, folder, encode=True):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        (self.folder / "cameras").mkdir()
        self.rows = {key: [] for key in ARRAYS}
        self.subtasks = []
        self.videos = {}
        if encode:
            self.videos = {c: Video(self.folder / f"cameras/{c}.mp4") for c in CAMERAS}

    def subtask_index(self, text):
        if text not in self.subtasks:
            self.subtasks.append(text)
        return self.subtasks.index(text)

    def add(self, row, images=None):
        """One tick: ``row`` holds every ARRAYS key except subtask_index plus
        ``subtask`` (text); ``images`` maps each camera to HxWx3 uint8 RGB."""
        for key, (dtype, shape) in ARRAYS.items():
            if key == "subtask_index":
                value = self.subtask_index(row["subtask"])
            else:
                value = row[key]
            value = np.asarray(value, dtype)
            if value.shape != shape:
                raise ValueError(f"{key}: shape {value.shape}, expected {shape}")
            self.rows[key].append(value)
        if self.videos:
            for camera in CAMERAS:
                self.videos[camera].write(images[camera])

    def copy_video(self, camera, source):
        shutil.copyfile(source, self.folder / f"cameras/{camera}.mp4")

    def close(self, meta, commands=None):
        """Finish the videos and write frames.npz and episode.json; ``meta`` is
        merged into the episode.json written here."""
        error = None
        for video in self.videos.values():
            try:
                video.close()
            except Exception as exc:  # noqa: BLE001 - every video gets closed
                error = error or exc
        arrays = {
            key: np.stack(values) if values else np.zeros((0, *shape), dtype)
            for key, (dtype, shape), values in zip(
                ARRAYS, ARRAYS.values(), self.rows.values()
            )
        }
        np.savez_compressed(self.folder / "frames.npz", **arrays)
        if commands is not None:
            (self.folder / "commands.json").write_text(json.dumps(commands, indent=1))
        info = {
            "schema": SCHEMA,
            "robot": ROBOT,
            "fps": FPS,
            "frames": len(self.rows["state"]),
            "cameras": list(CAMERAS),
            "image_size": [WIDTH, HEIGHT],
            "state_names": STATE_NAMES,
            "action_names": STATE_NAMES,
            "conventions": {
                "joints": "SDK controller MDH radians, q_sdk = sign * q_urdf + offset "
                "(config/robot/franzi_sdk.yaml); left j1..j7, right j1..j7",
                "gripper": "GripperReadPosition / GripperSetPosition units: "
                "0 closed .. 1000 open",
                "action": "last accepted command per arm and gripper side, held "
                "until the next one (see vla_episode.py)",
                "arm_source": ARM_SOURCES,
            },
            "subtasks": self.subtasks,
            **meta,
        }
        (self.folder / "episode.json").write_text(json.dumps(info, indent=2))
        if error:
            raise error
        return info


def read_episode(folder):
    """(episode.json dict, {name: array}) of a raw episode folder."""
    folder = Path(folder)
    info = json.loads((folder / "episode.json").read_text())
    if info.get("schema") != SCHEMA:
        raise ValueError(f"{folder}: schema {info.get('schema')!r}, need {SCHEMA}")
    with np.load(folder / "frames.npz") as data:
        arrays = {key: data[key] for key in ARRAYS}
    for key, (_, shape) in ARRAYS.items():
        if arrays[key].shape != (info["frames"], *shape):
            raise ValueError(f"{folder}: {key} has shape {arrays[key].shape}")
    return info, arrays


def camera_frames(folder, camera):
    """Decoded HxWx3 uint8 RGB frames of one camera video, in order."""
    with av.open(str(Path(folder) / f"cameras/{camera}.mp4")) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def video_info(path):
    """(frame count, fps, width, height) of an MP4, from its stream header."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return (
            stream.frames,
            float(stream.average_rate),
            stream.codec_context.width,
            stream.codec_context.height,
        )
