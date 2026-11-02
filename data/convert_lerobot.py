"""Convert a LeRobot dataset between format v2.1 and v3.0, into a new folder.

    /Users/puheliang/miniforge3/envs/lerobot/bin/python -m data.convert_lerobot \
        <dataset> --output <new_dataset>

The input's meta/info.json codebase_version sets the direction. The input is
not changed and nothing goes to the Hugging Face Hub. Runs in the lerobot env.

- v2.1 -> v3.0 is LeRobot's own converter
  (lerobot.scripts.convert_dataset_v21_to_v30), run step by step from the input
  to the output folder: its convert_dataset() renames the input folder and
  pushes to the Hub. It concatenates the videos by stream copy; for H.264 and
  HEVC, FFmpeg's concat demuxer repeats the parameter sets in every keyframe,
  which decodes to the same frames.
- v3.0 -> v2.1, which LeRobot lacks, writes with data/lerobot_v21.py. Each
  episode's rows go to their own parquet file, with the features in the
  datasets 3.x form. Each episode's video is cut out of its file by stream copy
  from its from_timestamp, so episodes must start on a keyframe, as LeRobot
  writes them. Tasks, episodes and per-episode stats (every key, quantiles
  included) go back to jsonl.
- Other files in meta/, such as inovance_sources.json, are copied, and so is
  extras/ (the episodes' depth, segmentation and evaluation files).

Each conversion is checked against its input: every non-video value and stat,
each episode's video packets (byte for byte, H.264 and HEVC ones without their
parameter sets), and frames decoded from both sides; a v3.0 output is decoded
by LeRobot itself.
"""

import argparse
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data import lerobot_v21 as v21

V21_META = {"info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"}
V30_META = {"info.json", "tasks.parquet", "episodes", "stats.json"}
# LeRobot decodes with torchcodec and the system FFmpeg. On the same packets its
# RGB comes out about one level brighter than PyAV's, and at most 3 levels
# apart (measured 2026-10-24).
DECODER_TOLERANCE = 3
# NAL unit types of the parameter sets (H.264 SPS, PPS; HEVC VPS, SPS, PPS).
PARAMETER_SETS = {"h264": {7, 8}, "hevc": {32, 33, 34}}


def read_v30(root):
    """(info, tasks by task_index, meta/episodes rows by episode_index)."""
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError(f"{root}: codebase_version {info['codebase_version']}")
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    tasks.sort(key=lambda t: t["task_index"])
    episodes = [
        row
        for path in sorted((root / "meta/episodes").glob("*/*.parquet"))
        for row in pq.read_table(path).to_pylist()
    ]
    episodes.sort(key=lambda e: e["episode_index"])
    for name, rows, key in (
        ("tasks", tasks, "task_index"),
        ("episodes", episodes, "episode_index"),
    ):
        if [r[key] for r in rows] != list(range(len(rows))):
            raise ValueError(f"{root}: {name} are not numbered 0..{len(rows) - 1}")
    return info, [t["task"] for t in tasks], episodes


def row_stats(row):
    """{feature: {stat: array}} from a meta/episodes row's stats/<f>/<s> columns."""
    stats = {}
    for key, value in row.items():
        if key.startswith("stats/") and value is not None:
            feature, _, stat = key.removeprefix("stats/").rpartition("/")
            stats.setdefault(feature, {})[stat] = np.asarray(value)
    return stats


def same_stats(row, stats):
    """A meta/episodes row's stats equal an episodes_stats.jsonl entry."""
    ours = row_stats(row)
    return ours.keys() == stats.keys() and all(
        ours[f].keys() == stats[f].keys()
        and all(np.array_equal(ours[f][s], np.asarray(stats[f][s])) for s in stats[f])
        for f in stats
    )


def v30_path(root, info, row, key=None):
    """An episode's data file, or with ``key`` its video file."""
    if key is None:
        chunk, file = row["data/chunk_index"], row["data/file_index"]
        return root / info["data_path"].format(chunk_index=chunk, file_index=file)
    chunk, file = row[f"videos/{key}/chunk_index"], row[f"videos/{key}/file_index"]
    return root / info["video_path"].format(
        video_key=key, chunk_index=chunk, file_index=file
    )


def episode_rows(root, info, row, cache):
    """An episode's rows of its v3.0 data file; ``cache`` keeps the last file."""
    path = v30_path(root, info, row)
    if cache.get("path") != path:
        cache.update(path=path, table=pq.read_table(path))
    table = cache["table"]
    rows = table.filter(pc.equal(table["episode_index"], row["episode_index"]))
    if rows.num_rows != row["length"]:
        raise ValueError(
            f"{path}: episode {row['episode_index']} has {rows.num_rows} rows"
        )
    return rows


def video_window(row, key):
    return row[f"videos/{key}/from_timestamp"], row[f"videos/{key}/to_timestamp"]


def without_parameter_sets(data, codec):
    """An H.264 or HEVC packet without its parameter-set NAL units (MP4 packets
    carry 4-byte NAL lengths, as FFmpeg writes them); other codecs unchanged."""
    if codec not in PARAMETER_SETS:
        return data
    kept, i = [], 0
    while i < len(data):
        size = int.from_bytes(data[i : i + 4], "big")
        unit = data[i + 4 : i + 4 + size]
        i += 4 + size
        kind = unit[0] & 0x1F if codec == "h264" else unit[0] >> 1 & 0x3F
        if kind not in PARAMETER_SETS[codec]:
            kept.append(unit)
    return b"".join(kept)


def packets(path, fps, start=0.0, end=float("inf")):
    """(seconds from ``start``, payload) of the packets shown in [start, end);
    H.264 and HEVC payloads without their parameter sets."""
    half = 0.5 / fps
    found = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = stream.codec.canonical_name
        container.seek(int(start / stream.time_base), stream=stream)
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            t = float(packet.pts * stream.time_base)
            if t >= end - half:
                break
            if t >= start - half:
                found.append((t - start, without_parameter_sets(bytes(packet), codec)))
    return found


def same_packets(ours, theirs, n):
    """Equal payloads and display times, n of them."""
    assert len(ours) == len(theirs) == n, (len(ours), len(theirs), n)
    assert [p for _, p in ours] == [p for _, p in theirs]
    assert np.allclose([t for t, _ in ours], [t for t, _ in theirs], atol=1e-6)


def frame_at(path, t, fps):
    """The decoded RGB frame shown at ``t`` seconds."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(t / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time >= t - 0.5 / fps:
                return frame.to_ndarray(format="rgb24")
    raise ValueError(f"{path}: no frame at {t:.3f} s")


def cut(source, target, start, end, fps):
    """Copy the packets shown in [start, end) seconds into a new video whose
    timestamps start at 0; returns the packet count."""
    half = 0.5 / fps
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        av.open(str(source)) as src,
        av.open(str(target), "w", options={"movflags": "faststart"}) as dst,
    ):
        stream = src.streams.video[0]
        out = dst.add_stream_from_template(stream, opaque=True)
        out.time_base = stream.time_base
        src.seek(int(start / stream.time_base), stream=stream)
        for packet in src.demux(stream):
            if packet.pts is None:
                continue
            t = float(packet.pts * stream.time_base)
            if t >= end - half:
                break
            if t < start - half:
                continue
            if count == 0:
                if not packet.is_keyframe:
                    raise ValueError(f"{source}: no keyframe at {start:.3f} s")
                offset = packet.pts
            packet.pts -= offset
            packet.dts -= offset
            packet.stream = out
            dst.mux(packet)
            count += 1
    return count


def copy_meta(src, dst, structural):
    for path in sorted((src / "meta").iterdir()):
        if path.name in structural:
            continue
        if path.is_dir():
            shutil.copytree(path, dst / "meta" / path.name)
        else:
            shutil.copy2(path, dst / "meta" / path.name)
    # The episodes' depth, segmentation and evaluation files (export_lerobot):
    # indexed by episode, which both versions keep.
    if (src / "extras").is_dir():
        shutil.copytree(src / "extras", dst / "extras")


def to_v21(src, dst):
    info, tasks, episodes = read_v30(src)
    features = {
        key: {k: v for k, v in ft.items() if k not in ("fps", "info")}
        for key, ft in info["features"].items()
        if key not in v21.DEFAULT_FEATURES
    }
    writer = v21.Writer(dst, info["fps"], features, info.get("robot_type"))
    writer.tasks = tasks
    for key in writer.video_keys:
        for row in episodes:
            target = v21.video_path(dst, row["episode_index"], key)
            source = v30_path(src, info, row, key)
            count = cut(source, target, *video_window(row, key), info["fps"])
            if count != row["length"]:
                raise ValueError(f"{source}: {count} frames for {row['length']} rows")
    cache = {}
    for row in episodes:
        writer.add(episode_rows(src, info, row, cache), row["tasks"], row_stats(row))
    stats = src / "meta/stats.json"
    writer.close(json.loads(stats.read_text()) if stats.exists() else None)
    copy_meta(src, dst, V30_META)


def check_v21(src, dst):
    """The v2.1 output against its v3.0 input."""
    info, tasks, episodes = read_v30(src)
    info21, tasks21, episodes21, stats21 = v21.read(dst)
    assert tasks21 == tasks
    assert (info21["total_episodes"], info21["total_frames"]) == (
        info["total_episodes"],
        info["total_frames"],
    )
    fps = info["fps"]
    keys = [k for k, ft in info["features"].items() if ft["dtype"] == "video"]
    cache = {}
    for row, row21, stats in zip(episodes, episodes21, stats21):
        e, n = row["episode_index"], row["length"]
        assert (row21["length"], row21["tasks"]) == (n, list(row["tasks"]))
        assert same_stats(row, stats), e
        rows = episode_rows(src, info, row, cache)
        written = pq.read_table(v21.data_path(dst, e))
        assert written.to_pylist() == rows.select(written.column_names).to_pylist(), e
        for key in keys:
            start, end = video_window(row, key)
            ours = v21.video_path(dst, e, key)
            theirs = v30_path(src, info, row, key)
            same_packets(packets(ours, fps), packets(theirs, fps, start, end), n)
            for t in (0, (n - 1) / fps):
                assert np.array_equal(
                    frame_at(ours, t, fps), frame_at(theirs, start + t, fps)
                ), (ours, t)
    return {"episodes": len(episodes), "frames": info["total_frames"]}


def to_v30(src, dst):
    from lerobot.datasets.utils import (
        DEFAULT_DATA_FILE_SIZE_IN_MB,
        DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    )
    from lerobot.scripts import convert_dataset_v21_to_v30 as lerobot_v30

    v21.read(src)
    dst.mkdir(parents=True)
    lerobot_v30.convert_info(
        src, dst, DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB
    )
    lerobot_v30.convert_tasks(src, dst)
    data = lerobot_v30.convert_data(src, dst, DEFAULT_DATA_FILE_SIZE_IN_MB)
    videos = lerobot_v30.convert_videos(src, dst, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    lerobot_v30.convert_episodes_metadata(src, dst, data, videos)
    # It writes meta/episodes through datasets 4, whose "List" features the
    # LeRobot of openpi_with_subtask (33cad37, datasets 3.6.0) cannot parse.
    # LeRobot's own v3.0 writer leaves those parquet files without them.
    for path in sorted((dst / "meta/episodes").glob("*/*.parquet")):
        pq.write_table(pq.read_table(path).replace_schema_metadata(None), path)
    copy_meta(src, dst, V21_META | {"stats.json"})


def check_v30(src, dst, repo_id):
    """The v3.0 output against its v2.1 input, read back through LeRobot."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info21, tasks21, episodes21, stats21 = v21.read(src)
    info, tasks, episodes = read_v30(dst)
    assert tasks == tasks21
    dataset = LeRobotDataset(repo_id, root=dst)
    assert dataset.meta.total_episodes == len(episodes21)
    assert len(dataset) == info21["total_frames"]
    fps, keys = info["fps"], dataset.meta.video_keys
    frames = dataset.hf_dataset.data.table
    worst, start = 0, 0
    for row21, stats, row in zip(episodes21, stats21, episodes):
        e, n = row21["episode_index"], row21["length"]
        assert (row["length"], list(row["tasks"])) == (n, row21["tasks"])
        assert same_stats(row, stats), e
        rows = pq.read_table(v21.data_path(src, e))
        loaded = frames.slice(start, n).select(rows.column_names)
        assert loaded.to_pylist() == rows.to_pylist(), e
        for key in keys:
            theirs = v21.video_path(src, e, key)
            window = video_window(row, key)
            ours = v30_path(dst, info, row, key)
            same_packets(packets(ours, fps, *window), packets(theirs, fps), n)
        # First and last frames: PyAV on both sides must agree exactly (the
        # episode's from_timestamp is right), LeRobot's decoder within tolerance.
        for k in (0, n - 1):
            item = dataset[start + k]
            for key in keys:
                want = frame_at(v21.video_path(src, e, key), k / fps, fps)
                t = video_window(row, key)[0] + k / fps
                assert np.array_equal(
                    frame_at(v30_path(dst, info, row, key), t, fps), want
                )
                got = np.rint(item[key].numpy().transpose(1, 2, 0) * 255).astype(int)
                worst = max(worst, int(np.abs(got - want).max()))
        start += n
    assert worst <= DECODER_TOLERANCE, worst
    return {
        "episodes": len(episodes),
        "frames": len(dataset),
        "lerobot_decoder_max_difference": worst,
    }


def convert(src, dst, repo_id):
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    version = json.loads((src / "meta/info.json").read_text())["codebase_version"]
    if version == "v2.1":
        to_v30(src, dst)
        report = check_v30(src, dst, repo_id)
    elif version == "v3.0":
        to_v21(src, dst)
        report = check_v21(src, dst)
    else:
        raise ValueError(
            f"{src}: codebase_version {version}; only v2.1 and v3.0 convert"
        )
    written = json.loads((dst / "meta/info.json").read_text())["codebase_version"]
    return {
        "input": str(src),
        "output": str(dst),
        "from": version,
        "to": written,
        **report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", type=Path, help="a LeRobot v2.1 or v3.0 dataset")
    parser.add_argument("--output", type=Path, required=True, help="new dataset folder")
    parser.add_argument(
        "--repo-id",
        default="local/converted",
        help="the repo_id LeRobot loads a v3.0 output under for the check",
    )
    args = parser.parse_args()
    print(json.dumps(convert(args.dataset, args.output, args.repo_id), indent=2))


if __name__ == "__main__":
    main()
