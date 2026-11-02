"""Raw VLA episodes: the one schema shared by the SDK recorder and the expert
converter. export_lerobot.py turns any set of them into a LeRobot v3.0 dataset.

One folder per episode:

    episode.json        source, conventions, task and subtask texts, timing summary
    frames.npz          one row per 25 Hz control tick (ARRAYS, and FULL_ARRAYS
                        when episode.json says "full")
    commands.json       every SDK call the driver made (SDK recordings only)
    cameras/<name>.mp4  one H.264 frame per row, 640x480 RGB, named as in get_image
    extras/             schema 2, optional, listed in episode.json "extras": each
                        camera's depth (depth_<name>.mkv, mm) and segmentation
                        (segmentation_<name>.mkv, ids in segmentation.json), FFV1
                        gray16le, one frame per row; and the world state for
                        evaluation only (eval.npz, eval_contacts.json)

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

Schema 2 (2026-10-25, user request: collect everything) adds the full robot
when the source has it (expert runs; SDK recordings so far do not): FULL_NAMES
(the 16 above, then the waist's calf, thigh, waist pitch and waist yaw, and the
head's yaw and pitch, in SDK units of config/robot/franzi_sdk.yaml) as
state_full, action_full, velocity_full and effort_full; each hand's grasp-site
pose and each camera's pose in the world (eef_pose, camera_pose). VIEWS name the
training subsets: arms16 (the 16 above), waist17 (plus waist yaw), full22.

Ground-truth object state never enters frames.npz or a policy input: schema 2
keeps it in extras/eval.npz for evaluation only.
"""

import json
import queue
import shutil
import threading
from pathlib import Path

import av
import numpy as np

SCHEMA = "inovance-franzi-vla-episode/2"
READABLE = ("inovance-franzi-vla-episode/1", SCHEMA)
ROBOT = "inovance_franzi_dual_arm"  # config/robot/franzi_sdk.yaml name
FPS = 25
WIDTH, HEIGHT = 640, 480
CAMERAS = ("head", "waist", "wrist_left", "wrist_right")
JOINT_NAMES = [f"{side}_j{i}" for side in ("left", "right") for i in range(1, 8)]
STATE_NAMES = JOINT_NAMES + ["left_gripper", "right_gripper"]
# The waist and head units' joints, in config/robot/franzi_sdk.yaml order.
BODY_NAMES = [
    "calf_pitch",
    "thigh_pitch",
    "waist_pitch",
    "waist_yaw",
    "head_yaw",
    "head_pitch",
]
FULL_NAMES = STATE_NAMES + BODY_NAMES
VIEWS = {
    "arms16": list(range(16)),
    "waist17": list(range(16)) + [FULL_NAMES.index("waist_yaw")],
    "full22": list(range(len(FULL_NAMES))),
}
POSE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]
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
# Schema 2, optional: every row of an episode has all of them or none.
# Velocities and efforts take each joint's SDK sign; the grippers' are the
# position rate (units/s, 1000 per full stroke) and the squeeze force (N).
FULL_ARRAYS = {
    "state_full": (np.float64, (len(FULL_NAMES),)),
    "action_full": (np.float64, (len(FULL_NAMES),)),
    "velocity_full": (np.float64, (len(FULL_NAMES),)),
    "effort_full": (np.float64, (len(FULL_NAMES),)),
    # Left then right grasp site, and each camera in CAMERAS order:
    # x, y, z, qw, qx, qy, qz in the world (the robot base frame, fixed).
    "eef_pose": (np.float64, (2, 7)),
    "camera_pose": (np.float64, (len(CAMERAS), 7)),
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


def lerobot_features(view="arms16", full=False):
    """LEROBOT_FEATURES with the policy state and action of view (VIEWS) and,
    with full, the schema-2 arrays under robot.*, which no policy reads unless
    configured to (arms16 without full is LEROBOT_FEATURES)."""
    names = [FULL_NAMES[i] for i in VIEWS[view]]
    vector = {"dtype": "float32", "shape": (len(names),), "names": names}
    features = dict(LEROBOT_FEATURES)
    features["observation.state"] = vector
    features["action"] = dict(vector)
    if full:
        joints = {"dtype": "float32", "shape": (len(FULL_NAMES),), "names": FULL_NAMES}
        hands = [f"{s}_{p}" for s in ("left", "right") for p in POSE_NAMES]
        cameras = [f"{c}_{p}" for c in CAMERAS for p in POSE_NAMES]
        features.update(
            {
                "robot.state_full": joints,
                "robot.action_full": dict(joints),
                "robot.velocity_full": dict(joints),
                "robot.effort_full": dict(joints),
                "robot.eef_pose": {"dtype": "float32", "shape": (14,), "names": hands},
                "robot.camera_pose": {
                    "dtype": "float32",
                    "shape": (len(cameras),),
                    "names": cameras,
                },
            }
        )
    return features


class Video:
    """H.264 writer on its own thread, so encoding never stalls a control tick.
    Same settings as sim/episode.py's policy-camera videos."""

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
        self.full = None  # whether rows carry FULL_ARRAYS: set by the first row
        self.full_rows = {key: [] for key in FULL_ARRAYS}
        self.extras = []
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
        ``subtask`` (text), and every FULL_ARRAYS key or none of them;
        ``images`` maps each camera to HxWx3 uint8 RGB."""
        full = any(key in row for key in FULL_ARRAYS)
        if self.full is None:
            self.full = full
        elif full != self.full:
            raise ValueError("rows must all carry FULL_ARRAYS or all omit them")
        arrays = {**ARRAYS, **(FULL_ARRAYS if full else {})}
        for key, (dtype, shape) in arrays.items():
            if key == "subtask_index":
                value = self.subtask_index(row["subtask"])
            else:
                value = row[key]
            value = np.asarray(value, dtype)
            if value.shape != shape:
                raise ValueError(f"{key}: shape {value.shape}, expected {shape}")
            (self.rows if key in ARRAYS else self.full_rows)[key].append(value)
        if self.videos:
            for camera in CAMERAS:
                self.videos[camera].write(images[camera])

    def copy_video(self, camera, source):
        shutil.copyfile(source, self.folder / f"cameras/{camera}.mp4")

    def copy_extra(self, name, source):
        """Copy a file into extras/ under name and list it in episode.json."""
        (self.folder / "extras").mkdir(exist_ok=True)
        shutil.copyfile(source, self.folder / "extras" / name)
        self.extras.append(name)

    def save_extra(self, name, **arrays):
        """Write arrays as extras/<name> (npz) and list it in episode.json."""
        (self.folder / "extras").mkdir(exist_ok=True)
        np.savez_compressed(self.folder / "extras" / name, **arrays)
        self.extras.append(name)

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
        if self.full:
            arrays.update({key: np.stack(v) for key, v in self.full_rows.items()})
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
            "full": bool(self.full),
            "full_names": FULL_NAMES if self.full else None,
            "extras": sorted(self.extras),
            "conventions": {
                "joints": "SDK controller MDH radians, q_sdk = sign * q_urdf + offset "
                "(config/robot/franzi_sdk.yaml); left j1..j7, right j1..j7",
                "gripper": "GripperReadPosition / GripperSetPosition units: "
                "0 closed .. 1000 open",
                "action": "last accepted command per arm and gripper side, held "
                "until the next one (see vla_episode.py)",
                "arm_source": ARM_SOURCES,
                "full": "FULL_NAMES in SDK units (waist and head: their "
                "franzi_sdk.yaml unit mappings); velocity_full and effort_full "
                "take each joint's sign, grippers in position units/s and squeeze "
                "N; eef_pose and camera_pose: x, y, z, qw, qx, qy, qz in the world",
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
    if info.get("schema") not in READABLE:
        raise ValueError(f"{folder}: schema {info.get('schema')!r}, need {READABLE}")
    info.setdefault("full", False)
    info.setdefault("extras", [])
    wanted = {**ARRAYS, **(FULL_ARRAYS if info["full"] else {})}
    with np.load(folder / "frames.npz") as data:
        arrays = {key: data[key] for key in wanted}
    for key, (_, shape) in wanted.items():
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
