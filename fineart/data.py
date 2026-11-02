"""The scenes' LeRobot v2.1 datasets as FineART training samples.

A sample is one recorded tick: the four policy cameras' frames and the 16-D
state before the action, the next CHUNK actions (the last one repeated past the
episode's end and flagged in action_is_pad), the episode's instruction as the
task and the tick's subtask text. Its memory row holds the subtask SUBTASK_EVERY
ticks back (NO_MEMORY before that), which only fineart.recipe's memory recipe
reads. The frames come straight from the AV1 videos (a keyframe every 2 frames,
so a seek decodes at most two) as uint8 [3, H, W]; train.py scales them to
[0, 1] on the GPU.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

from fineart.protocol import NO_MEMORY, SUBTASK_EVERY

CAMERAS = ("head", "waist", "wrist_left", "wrist_right")
CHUNK = 50
FPS = 25
ROBOT = "inovance_franzi_dual_arm"
HOLDOUT = 4  # last episodes of each dataset kept for the validation loss


def vectors(table, name):
    """A fixed-size list column as a float32 [rows, width] array."""
    values = table.column(name).combine_chunks()
    width = values.type.list_size
    return values.flatten().to_numpy().reshape(-1, width).astype(np.float32)


def info(root):
    meta = json.loads((Path(root) / "meta/info.json").read_text())
    assert meta["codebase_version"] == "v2.1" and meta["fps"] == FPS, root
    return meta


class Episodes(Dataset):
    """Every tick of the datasets under roots (each a lerobot_v21 folder). The
    last `holdout` episodes of each dataset are the validation split."""

    def __init__(self, roots, split, holdout=HOLDOUT):
        assert split in ("train", "validation")
        self.episodes = []
        rows = []
        for root in map(Path, roots):
            meta = info(root)
            tasks = {}
            for line in (root / "meta/tasks.jsonl").read_text().splitlines():
                row = json.loads(line)
                tasks[row["task_index"]] = row["task"]
            count = meta["total_episodes"]
            chosen = (
                range(count - holdout)
                if split == "train"
                else range(count - holdout, count)
            )
            for index in chosen:
                chunk = index // meta["chunks_size"]
                table = pq.read_table(
                    root
                    / meta["data_path"].format(
                        episode_chunk=chunk, episode_index=index
                    ),
                    columns=[
                        "observation.state",
                        "action",
                        "subtask",
                        "task_index",
                        "frame_index",
                    ],
                )
                frames = table.column("frame_index").to_numpy()
                assert (frames == np.arange(len(frames))).all(), (root, index)
                task = set(table.column("task_index").to_numpy().tolist())
                assert len(task) == 1, (root, index)
                names, codes = np.unique(
                    table.column("subtask").to_pylist(), return_inverse=True
                )
                videos = {
                    camera: str(
                        root
                        / meta["video_path"].format(
                            episode_chunk=chunk,
                            video_key=f"observation.images.{camera}",
                            episode_index=index,
                        )
                    )
                    for camera in CAMERAS
                }
                rows.append(
                    np.stack(
                        [
                            np.full(len(frames), len(self.episodes)),
                            np.arange(len(frames)),
                        ],
                        1,
                    )
                )
                self.episodes.append(
                    {
                        "state": vectors(table, "observation.state"),
                        "action": vectors(table, "action"),
                        "subtasks": names.tolist(),
                        "subtask": codes.astype(np.int16),
                        "task": tasks[task.pop()],
                        "videos": videos,
                        "scene": root.parent.name,
                        "episode_index": index,
                    }
                )
        self.rows = np.concatenate(rows).astype(np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        episode, frame = (int(v) for v in self.rows[i])
        data = self.episodes[episode]
        n = len(data["state"])
        ahead = np.arange(frame, frame + CHUNK)
        back = frame - SUBTASK_EVERY
        memory = data["subtasks"][data["subtask"][back]] if back >= 0 else NO_MEMORY
        sample = {
            "observation.state": torch.from_numpy(data["state"][frame].copy()),
            "action": torch.from_numpy(data["action"][np.minimum(ahead, n - 1)]),
            "action_is_pad": torch.from_numpy(ahead >= n),
            "timestamp": torch.tensor(frame / FPS, dtype=torch.float32),
            "index": torch.tensor(i),
            "task": data["task"],
            "robot_type": ROBOT,
            "language_persistent": [
                {
                    "role": "assistant",
                    "content": data["subtasks"][data["subtask"][frame]],
                    "style": "subtask",
                    "timestamp": 0.0,
                    "camera": None,
                    "tool_calls": None,
                },
                {
                    "role": "assistant",
                    "content": memory,
                    "style": "memory",
                    "timestamp": 0.0,
                    "camera": None,
                    "tool_calls": None,
                },
            ],
        }
        for camera in CAMERAS:
            decoder = VideoDecoder(data["videos"][camera], seek_mode="approximate")
            sample[f"observation.images.{camera}"] = decoder[frame]
        return sample


def scene_roots(collection):
    """The lerobot_v21 folders of a collection folder (<scene>/lerobot_v21)."""
    return sorted(
        str(p)
        for p in Path(collection).glob("*/lerobot_v21")
        if (p / "meta/info.json").exists()
    )
