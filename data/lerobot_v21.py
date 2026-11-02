"""LeRobot dataset format v2.1, written and read without LeRobot.

v2.1 is what openpi's pinned LeRobot (commit 0cf8648, 2025-05) reads, and with
it openpi, kai0, pistar and Spatial-Forcing in this workspace. The LeRobot of
the lerobot env reads and writes v3.0 only; data/convert_lerobot.py converts
between the two.

    meta/info.json              codebase_version "v2.1", features, path templates
    meta/tasks.jsonl            {"task_index", "task"} per task
    meta/episodes.jsonl         {"episode_index", "tasks", "length"} per episode
    meta/episodes_stats.jsonl   {"episode_index", "stats": {feature: {stat: value}}}
    meta/stats.json             aggregated stats (LeRobot v2.1 aggregates its own)
    data/chunk-{c:03d}/episode_{e:06d}.parquet        one file per episode
    videos/chunk-{c:03d}/{key}/episode_{e:06d}.mp4   one per episode and camera

- The parquet schema metadata holds the Hugging Face features in the datasets
  3.x form ("Sequence"). datasets 4, which the lerobot env has, writes "List",
  which datasets 3.x cannot parse; openpi pins datasets 3.6.0.
- Stats are LeRobot v2.1's (compute_stats.py): min, max, mean, std and count of
  every feature but strings. Image stats use about n**0.75 of the n frames
  (sample_indices), every k-th pixel (downsample), per channel in [0, 1] with
  shape (3, 1, 1), and count the sampled frames.
- The stats add the q01..q99 quantiles that current LeRobot keeps for quantile
  normalization (pi0.5), computed exactly rather than from its histograms. v2.1
  readers ignore them; a v3.0 conversion keeps them.
"""

import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
INFO = "meta/info.json"
TASKS = "meta/tasks.jsonl"
EPISODES = "meta/episodes.jsonl"
EPISODES_STATS = "meta/episodes_stats.jsonl"
STATS = "meta/stats.json"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def data_path(root, episode):
    chunk = episode // CHUNK_SIZE
    return Path(root) / DATA_PATH.format(episode_chunk=chunk, episode_index=episode)


def video_path(root, episode, key):
    path = VIDEO_PATH.format(
        episode_chunk=episode // CHUNK_SIZE, video_key=key, episode_index=episode
    )
    return Path(root) / path


def hf_feature(ft):
    """One feature as datasets 3.x serialises it (LeRobot v2.1's
    get_hf_features_from_features)."""
    if ft["dtype"] == "image":
        return {"_type": "Image"}
    value = {"dtype": ft["dtype"], "_type": "Value"}
    shape = tuple(ft["shape"])
    if shape == (1,):
        return value
    if len(shape) == 1:
        return {"feature": value, "length": shape[0], "_type": "Sequence"}
    raise NotImplementedError(f"{len(shape)}-D feature {ft}")


def arrow_type(ft):
    if ft["dtype"] == "image":
        return pa.struct({"bytes": pa.binary(), "path": pa.string()})
    if ft["dtype"] == "string":
        value = pa.string()
    else:
        value = pa.from_numpy_dtype(np.dtype(ft["dtype"]))
    shape = tuple(ft["shape"])
    if shape == (1,):
        return value
    if len(shape) == 1:
        return pa.list_(value, shape[0])
    raise NotImplementedError(f"{len(shape)}-D feature {ft}")


def schema(features):
    """Arrow schema of the parquet columns (every non-video feature), with the
    features in the datasets 3.x form as its "huggingface" metadata."""
    columns = {k: ft for k, ft in features.items() if ft["dtype"] != "video"}
    hf = {"info": {"features": {k: hf_feature(ft) for k, ft in columns.items()}}}
    return pa.schema(
        [(k, arrow_type(ft)) for k, ft in columns.items()],
        metadata={"huggingface": json.dumps(hf)},
    )


def table(columns, features):
    """An arrow table from {feature: n rows}: numpy arrays, or lists of str."""
    target = schema(features)
    arrays = []
    for field in target:
        values = columns[field.name]
        if pa.types.is_fixed_size_list(field.type):
            flat = pa.array(np.asarray(values).reshape(-1), field.type.value_type)
            arrays.append(pa.FixedSizeListArray.from_arrays(flat, field.type.list_size))
        else:
            arrays.append(pa.array(values, field.type))
    return pa.Table.from_arrays(arrays, schema=target)


def sample_indices(n):
    """The frames LeRobot computes image stats on: about n**0.75 of them, at
    least 100 (or all n), at most 10000, evenly spread."""
    count = max(min(n, 100), min(int(n**0.75), 10_000))
    return np.round(np.linspace(0, n - 1, count)).astype(int).tolist()


def downsample(chw):
    """LeRobot's auto_downsample_height_width: every k-th pixel of images whose
    longer side is 300 px or more, k = longer side // 150."""
    longer = max(chw.shape[1:])
    if longer < 300:
        return chw
    k = longer // 150
    return chw[:, ::k, ::k]


def feature_stats(values, axis=0, keepdims=False):
    values = np.asarray(values)
    quantiles = np.quantile(
        values, list(QUANTILES.values()), axis=axis, keepdims=keepdims
    )
    return {
        "min": np.min(values, axis=axis, keepdims=keepdims),
        "max": np.max(values, axis=axis, keepdims=keepdims),
        "mean": np.mean(values, axis=axis, keepdims=keepdims),
        "std": np.std(values, axis=axis, keepdims=keepdims),
        "count": np.array([len(values)]),
        **dict(zip(QUANTILES, quantiles)),
    }


def column_stats(columns, features):
    """Stats of every column but strings; scalars keep shape (1,)."""
    return {
        key: feature_stats(values, keepdims=np.ndim(values) == 1)
        for key, values in columns.items()
        if features[key]["dtype"] != "string"
    }


def image_stats(frames):
    """Stats of sampled frames, each 3xHxW uint8 already downsampled."""
    stats = feature_stats(np.stack(frames), axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else v[0] / 255.0 for k, v in stats.items()}


def aggregate(stats_list):
    """LeRobot's aggregate_stats: the overall min and max, the count-weighted
    mean and pooled std; each quantile keeps the lowest (q01..q50) or highest
    (q90, q99) episode value, as current LeRobot does."""
    out = {}
    for key in dict.fromkeys(k for stats in stats_list for k in stats):
        group = [stats[key] for stats in stats_list if key in stats]
        mean = np.stack([s["mean"] for s in group]).astype(np.float64)
        std = np.stack([s["std"] for s in group]).astype(np.float64)
        count = np.stack([s["count"] for s in group])
        weight = count.reshape(-1, *[1] * (mean.ndim - 1))
        total = count.sum(axis=0)
        average = (mean * weight).sum(axis=0) / total
        spread = std**2 + (mean - average) ** 2
        out[key] = {
            "min": np.min(np.stack([s["min"] for s in group]), axis=0),
            "max": np.max(np.stack([s["max"] for s in group]), axis=0),
            "mean": average,
            "std": np.sqrt((spread * weight).sum(axis=0) / total),
            "count": total,
        }
        for q in QUANTILES:
            if all(q in s for s in group):
                values = np.stack([s[q] for s in group])
                low = int(q[1:]) <= 50
                out[key][q] = values.min(axis=0) if low else values.max(axis=0)
    return out


def serialize(stats):
    return {
        k: {s: np.asarray(v).tolist() for s, v in ft.items()} for k, ft in stats.items()
    }


def video_info(path):
    """The "info" of a video feature, as LeRobot v2.1's get_video_info."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        pix_fmt = stream.pix_fmt
        if any(s in pix_fmt for s in ("gray", "depth", "monochrome")):
            channels = 1
        elif "rgba" in pix_fmt or "yuva" in pix_fmt:
            channels = 4
        else:
            channels = 3
        return {
            "video.height": stream.height,
            "video.width": stream.width,
            "video.codec": stream.codec.canonical_name,
            "video.pix_fmt": pix_fmt,
            "video.is_depth_map": False,
            "video.fps": int(stream.base_rate),
            "video.channels": channels,
            "has_audio": bool(container.streams.audio),
        }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def read(root):
    """(info, tasks, episodes, per-episode stats) of a v2.1 dataset: tasks a
    list of texts by task_index, the others lists by episode_index."""
    root = Path(root)
    info = json.loads((root / INFO).read_text())
    if info["codebase_version"] != CODEBASE_VERSION:
        raise ValueError(f"{root}: codebase_version {info['codebase_version']}")
    tasks = sorted(read_jsonl(root / TASKS), key=lambda t: t["task_index"])
    episodes = sorted(read_jsonl(root / EPISODES), key=lambda e: e["episode_index"])
    stats = sorted(read_jsonl(root / EPISODES_STATS), key=lambda e: e["episode_index"])
    for name, rows, key in (
        ("tasks", tasks, "task_index"),
        ("episodes", episodes, "episode_index"),
        ("episode stats", stats, "episode_index"),
    ):
        if [r[key] for r in rows] != list(range(len(rows))):
            raise ValueError(f"{root}: {name} are not numbered 0..{len(rows) - 1}")
    return info, [t["task"] for t in tasks], episodes, [s["stats"] for s in stats]


class Writer:
    """A new v2.1 dataset, one episode at a time.

    ``features`` are the dataset's own; DEFAULT_FEATURES are appended. Write an
    episode's videos to ``video_path(writer.root, episode, key)`` before ``add``;
    ``close`` writes the metadata, video info from episode 0 as LeRobot v2.1.
    """

    def __init__(self, root, fps, features, robot_type=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.fps = fps
        self.robot_type = robot_type
        features = {**features, **DEFAULT_FEATURES}
        self.features = {k: dict(ft) for k, ft in features.items()}
        self.video_keys = [k for k, ft in features.items() if ft["dtype"] == "video"]
        self.tasks = []
        self.episodes = []
        self.stats = []
        self.frames = 0

    def task_index(self, task):
        if task not in self.tasks:
            self.tasks.append(task)
        return self.tasks.index(task)

    def defaults(self, n, task):
        """The DEFAULT_FEATURES columns of the next episode: n frames of one task."""
        k = np.arange(n)
        return {
            "timestamp": (k / self.fps).astype(np.float32),
            "frame_index": k,
            "episode_index": np.full(n, len(self.episodes)),
            "index": self.frames + k,
            "task_index": np.full(n, self.task_index(task)),
        }

    def add(self, rows, tasks, stats):
        """Write the next episode: ``rows`` an arrow table of every non-video
        column (DEFAULT_FEATURES included), ``tasks`` its task texts, ``stats``
        its per-feature stats (videos included)."""
        episode = len(self.episodes)
        target = schema(self.features)
        path = data_path(self.root, episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(rows.select(target.names).cast(target), path)
        self.episodes.append(
            {"episode_index": episode, "tasks": list(tasks), "length": rows.num_rows}
        )
        self.stats.append(stats)
        self.frames += rows.num_rows

    def close(self, stats=None):
        """Write meta/; ``stats`` (serialized) replaces the aggregate for
        stats.json."""
        if not self.episodes:
            raise ValueError(f"{self.root}: no episodes")
        for key in self.video_keys:
            self.features[key]["info"] = video_info(video_path(self.root, 0, key))
        n = len(self.episodes)
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": n,
            "total_frames": self.frames,
            "total_tasks": len(self.tasks),
            "total_videos": n * len(self.video_keys),
            "total_chunks": (n - 1) // CHUNK_SIZE + 1,
            "chunks_size": CHUNK_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{n}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if self.video_keys else None,
            "features": {
                k: {**ft, "shape": list(ft["shape"])} for k, ft in self.features.items()
            },
        }
        write_json(self.root / INFO, info)
        write_jsonl(
            self.root / TASKS,
            [{"task_index": i, "task": t} for i, t in enumerate(self.tasks)],
        )
        write_jsonl(self.root / EPISODES, self.episodes)
        write_jsonl(
            self.root / EPISODES_STATS,
            [
                {"episode_index": e, "stats": serialize(s)}
                for e, s in enumerate(self.stats)
            ],
        )
        write_json(self.root / STATS, stats or serialize(aggregate(self.stats)))
        return info
