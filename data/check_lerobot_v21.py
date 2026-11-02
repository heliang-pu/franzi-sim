"""Load a v2.1 dataset of this project with the LeRobot that reads v2.1, the way
openpi's data loader does, and compare what it returns with the dataset's files
read directly (pyarrow, PyAV). It is the acceptance check of the v2.1 export.

Run it with openpi's pinned LeRobot (commit 0cf8648, datasets 3.6.0): openpi's
own venv, or the reference venv of docs/vla_data.md. It imports nothing from
this project:

    <venv>/bin/python data/check_lerobot_v21.py outputs/<dataset> [...]

LeRobotDatasetMetadata has to load the tasks, episodes and aggregated stats,
and LeRobotDataset the parquet files, passing its timestamp check. At five
frames of every episode, LeRobotDataset has to return the stored state, the
50-step action chunk openpi asks for (delta_timestamps) with its padding, the
subtask, the task, and each camera image within 3 levels of PyAV's decoding:
0 with LeRobot's pyav backend, while torchcodec rounds RGB differently.
"""

import json
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.common.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

HORIZON = 50  # openpi's action_horizon for pi0 and pi0.5
TOLERANCE = 3


def frame_at(path, t, fps):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(t / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time >= t - 0.5 / fps:
                return frame.to_ndarray(format="rgb24")
    raise ValueError(f"{path}: no frame at {t:.3f} s")


def check(root):
    root = Path(root)
    meta = LeRobotDatasetMetadata("local/check", root=root)
    dataset = LeRobotDataset(
        "local/check",
        root=root,
        delta_timestamps={"action": [t / meta.fps for t in range(HORIZON)]},
    )
    numeric = [k for k, ft in meta.features.items() if ft["dtype"] != "string"]
    assert sorted(meta.stats) == sorted(numeric), sorted(meta.stats)
    worst, start = 0, 0
    for e in range(meta.total_episodes):
        n = meta.episodes[e]["length"]
        rows = pq.read_table(root / meta.get_data_file_path(e)).to_pydict()
        for k in sorted({0, 1, n // 2, n - 2, n - 1}):
            item = dataset[start + k]
            assert (item["episode_index"].item(), item["frame_index"].item()) == (e, k)
            assert abs(item["timestamp"].item() - k / meta.fps) < 1e-4
            state = torch.tensor(rows["observation.state"][k])
            assert torch.equal(item["observation.state"], state)
            chunk = [rows["action"][min(k + j, n - 1)] for j in range(HORIZON)]
            assert torch.equal(item["action"], torch.tensor(chunk))
            assert item["action_is_pad"].tolist() == [
                k + j >= n for j in range(HORIZON)
            ]
            assert item["subtask"] == rows["subtask"][k]
            assert item["task"] == meta.tasks[rows["task_index"][k]]
            for key in meta.video_keys:
                height, width, _ = meta.shapes[key]
                image = item[key]
                assert image.dtype == torch.float32
                assert tuple(image.shape) == (3, height, width)
                path = root / meta.get_video_file_path(e, key)
                want = frame_at(path, k / meta.fps, meta.fps).astype(int)
                got = np.rint(image.numpy().transpose(1, 2, 0) * 255).astype(int)
                worst = max(worst, int(np.abs(got - want).max()))
        start += n
    assert worst <= TOLERANCE, f"{root}: images {worst} levels from PyAV's"
    return {
        "lerobot_codebase_version": CODEBASE_VERSION,
        "episodes": meta.total_episodes,
        "frames": len(dataset),
        "video_backend": dataset.video_backend,
        "max_pixel_difference": worst,
    }


if __name__ == "__main__":
    print(json.dumps({path: check(path) for path in sys.argv[1:]}, indent=2))
