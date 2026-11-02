"""Write raw VLA episodes (vla_episode.py) as one LeRobot dataset, format v2.1
(the default) or v3.0.

    /Users/puheliang/miniforge3/envs/lerobot/bin/python -m data.export_lerobot \
        --episodes outputs/<raw_episode> [...] --output outputs/<new_dataset> [--format v3.0]

Runs in the lerobot conda env; the recorder and the expert converter run in
inovance-fineart and do not need it.

- v2.1 is what openpi's pinned LeRobot reads. data/lerobot_v21.py writes it
  without LeRobot. Each episode's camera videos are re-encoded with LeRobot's
  defaults: AV1 (libsvtav1, preset 12), a keyframe every 2 frames, CRF 30.
  --vcodec libx264 suits decoders without AV1 (decord).
- v3.0 is written by the lerobot env's LeRobot (LeRobotDataset.create) with its
  own AV1 encoding.
- data.convert_lerobot converts a dataset to the other version.

SDK recordings and converted expert runs share one feature set, so they can be
mixed in one dataset (features: vla_episode.lerobot_features; only the state,
the action and the four cameras are policy inputs). ``timestamp`` is LeRobot's
frame_index / 25; the measured robot time is ``timing.robot_s``. ``task`` is
the episode instruction and ``subtask`` the per-step one.

--view picks the policy state and action (vla_episode.VIEWS): arms16 (the
default, the 16-D contract), waist17 (plus waist yaw) or full22; the last two
need schema-2 episodes with the full robot. --full adds those arrays under
robot.* (by default when every episode has them). Each episode's extras/
(depth and segmentation videos, evaluation-only world state) is copied to
extras/episode_<index>/ beside the dataset, listed in meta/inovance_extras.json:
LeRobot video features decode as 8-bit RGB, so 16-bit depth stays out of them.
"""

import argparse
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np

from data import lerobot_v21 as v21
from data.vla_episode import (
    CAMERAS,
    FPS,
    HEIGHT,
    ROBOT,
    VIEWS,
    WIDTH,
    camera_frames,
    lerobot_features,
    read_episode,
    video_info,
)

# LeRobot's VideoEncoderConfig defaults; the 2-frame GOP keeps random access
# cheap when training decodes single frames.
ENCODERS = {
    "libsvtav1": {"g": "2", "crf": "30", "preset": "12"},
    "libx264": {"g": "2", "crf": "30"},
}


def check(folder, include_incomplete):
    info, arrays = read_episode(folder)
    if not info["complete"] and not include_incomplete:
        raise ValueError(f"{folder}: incomplete episode ({info['error']})")
    if (info["fps"], info["cameras"], info["robot"]) != (FPS, list(CAMERAS), ROBOT):
        raise ValueError(f"{folder}: not a {ROBOT} episode at {FPS} Hz")
    for camera in CAMERAS:
        found = video_info(Path(folder) / f"cameras/{camera}.mp4")
        if found != (info["frames"], FPS, WIDTH, HEIGHT):
            raise ValueError(f"{folder}: {camera} video is {found}")
    if np.isnan(arrays["setpoint"]).any():
        print(f"warning: {folder}: no servo setpoint (NaN in command.setpoint)")
    return info, arrays


def policy_vectors(a, view):
    """The (state, action) a policy sees in view: the 16-D arrays for arms16,
    else the view's columns of the full robot."""
    if view == "arms16":
        return a["state"], a["action"]
    return a["state_full"][:, VIEWS[view]], a["action_full"][:, VIEWS[view]]


def columns(info, a, view="arms16", full=False):
    """Every non-video lerobot_features(view, full) column of one raw episode."""
    state, action = policy_vectors(a, view)
    robot = {}
    if full:
        robot = {
            f"robot.{key}": a[key].reshape(len(a[key]), -1).astype(np.float32)
            for key in (
                "state_full",
                "action_full",
                "velocity_full",
                "effort_full",
                "eef_pose",
                "camera_pose",
            )
        }
    return {
        "observation.state": state.astype(np.float32),
        "action": action.astype(np.float32),
        **robot,
        "command.setpoint": a["setpoint"].astype(np.float32),
        "command.arm_source": a["arm_source"].astype(np.int64),
        "timing.robot_s": a["robot_s"],
        "timing.state_utc_ms": a["state_utc_ms"],
        "timing.camera_seq": a["camera_seq"],
        "timing.camera_utc_ms": a["camera_utc_ms"],
        "subtask": [info["subtasks"][i] for i in a["subtask_index"]],
    }


def frames(folder, info, a, view="arms16", full=False):
    cols = columns(info, a, view, full)
    videos = {c: camera_frames(folder, c) for c in CAMERAS}
    for k in range(info["frames"]):
        yield {
            **{
                key: v[k] if isinstance(v, list) or v.ndim > 1 else v[k : k + 1]
                for key, v in cols.items()
            },
            **{f"observation.images.{c}": next(videos[c]) for c in CAMERAS},
            "task": info["task"],
        }
    for camera, video in videos.items():
        if next(video, None) is not None:
            raise ValueError(f"{folder}: {camera} video has extra frames")


def encode(folder, camera, path, n, vcodec):
    """Re-encode one camera video; returns the image stats of its frames."""
    sample = v21.sample_indices(n)
    wanted, picked, count = set(sample), {}, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream(vcodec, rate=FPS, options=ENCODERS[vcodec])
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "yuv420p"
        for rgb in camera_frames(folder, camera):
            if count in wanted:
                picked[count] = v21.downsample(rgb.transpose(2, 0, 1))
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = count
            container.mux(stream.encode(frame))
            count += 1
        container.mux(stream.encode())
    if count != n:
        raise ValueError(f"{folder}: {camera} video has {count} frames, not {n}")
    return v21.image_stats([picked[k] for k in sample])


def first_frame(path):
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_ndarray(format="rgb24")


def sources(episodes, loaded):
    keep = ("source", "complete", "clock", "images", "driver", "source_run")
    return [
        {
            "episode_index": index,
            "raw_episode": str(folder),
            "frames": info["frames"],
            **{k: info[k] for k in keep if k in info},
            "warnings": info.get("warnings", []),
        }
        for index, (folder, (info, _)) in enumerate(zip(episodes, loaded))
    ]


def write_sources(output, episodes, loaded, view="arms16", full=False):
    conventions = json.loads((Path(episodes[0]) / "episode.json").read_text())
    (output / "meta/inovance_sources.json").write_text(
        json.dumps(
            {
                "conventions": conventions["conventions"],
                "view": view,
                "full": full,
                "episodes": sources(episodes, loaded),
            },
            indent=2,
        )
    )
    copy_extras(output, episodes, loaded)


def copy_extras(output, episodes, loaded):
    """Each episode's extras/ to extras/episode_<index>/, listed in
    meta/inovance_extras.json (empty when no episode has any)."""
    listing = {}
    for index, (folder, (info, _)) in enumerate(zip(episodes, loaded)):
        if not info["extras"]:
            continue
        dst = output / f"extras/episode_{index:06d}"
        dst.mkdir(parents=True)
        for name in info["extras"]:
            shutil.copy2(Path(folder) / "extras" / name, dst / name)
        listing[f"episode_{index:06d}"] = info["extras"]
    (output / "meta/inovance_extras.json").write_text(
        json.dumps(
            {
                "about": "extras/episode_<index>/ per episode: depth_<camera>.mkv "
                "(mm) and segmentation_<camera>.mkv (ids, segmentation.json), FFV1 "
                "gray16le, one frame per dataset frame; eval.npz and "
                "eval_contacts.json: world state for evaluation only, never a "
                "policy input",
                "episodes": listing,
            },
            indent=2,
        )
    )


def export_v21(episodes, output, loaded, vcodec, view="arms16", full=False):
    # SVT-AV1 prints a banner per encoder (4 per episode); keep its errors only.
    # Its 3.0 release, in PyAV 15, runs preset 12 as 10.
    os.environ.setdefault("SVT_LOG", "1")
    features = lerobot_features(view, full)
    writer = v21.Writer(output, FPS, features, robot_type=ROBOT)
    for index, (folder, (info, a)) in enumerate(zip(episodes, loaded)):
        n = info["frames"]
        keys = {c: f"observation.images.{c}" for c in CAMERAS}
        with ThreadPoolExecutor(len(CAMERAS)) as pool:
            jobs = {
                key: pool.submit(
                    encode, folder, c, v21.video_path(output, index, key), n, vcodec
                )
                for c, key in keys.items()
            }
            images = {key: job.result() for key, job in jobs.items()}
        cols = {**columns(info, a, view, full), **writer.defaults(n, info["task"])}
        stats = {**v21.column_stats(cols, writer.features), **images}
        writer.add(v21.table(cols, writer.features), [info["task"]], stats)
        print(f"episode {index}: {n} frames from {folder}", flush=True)
    writer.close()
    write_sources(output, episodes, loaded, view, full)
    return verify_v21(output, episodes, loaded, view)


def verify_v21(output, folders, loaded, view="arms16"):
    """Reload the parquet files as LeRobot v2.1 does (datasets.load_dataset)
    and compare each episode's first and last frames; check every video's
    header and that its first frame is close to the raw camera's."""
    import datasets

    info, tasks, episodes, _ = v21.read(output)
    with tempfile.TemporaryDirectory() as cache:
        data = datasets.load_dataset(
            "parquet", data_dir=str(output / "data"), split="train", cache_dir=cache
        )
        assert len(data) == info["total_frames"] == sum(i["frames"] for i, _ in loaded)
        start = 0
        for e, (folder, (raw, a)) in enumerate(zip(folders, loaded)):
            assert episodes[e]["length"] == raw["frames"]
            assert episodes[e]["tasks"] == [raw["task"]]
            state, action = policy_vectors(a, view)
            for k in (0, raw["frames"] - 1):
                row = data[start + k]
                assert np.allclose(row["observation.state"], state[k])
                assert np.allclose(row["action"], action[k])
                assert row["timing.camera_seq"] == a["camera_seq"][k].tolist()
                assert row["subtask"] == raw["subtasks"][a["subtask_index"][k]]
                assert (row["frame_index"], row["index"]) == (k, start + k)
                assert tasks[row["task_index"]] == raw["task"]
            # The re-encode stays within 2 levels on average (40.8 dB PSNR or
            # better on the 2026-10-24 scenes); another camera's image would not.
            for camera in CAMERAS:
                path = v21.video_path(output, e, f"observation.images.{camera}")
                assert video_info(path) == (raw["frames"], FPS, WIDTH, HEIGHT), path
                ours = first_frame(path).astype(int)
                theirs = first_frame(Path(folder) / f"cameras/{camera}.mp4")
                assert np.abs(ours - theirs).mean() < 4, path
            start += raw["frames"]
    return {
        "root": str(output),
        "codebase_version": info["codebase_version"],
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "tasks": tasks,
        "features": list(info["features"]),
    }


def export_v30(episodes, output, loaded, repo_id, view="arms16", full=False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id,
        fps=FPS,
        features=lerobot_features(view, full),
        root=output,
        robot_type=ROBOT,
        use_videos=True,
        streaming_encoding=True,
    )
    for index, (folder, (info, a)) in enumerate(zip(episodes, loaded)):
        for frame in frames(folder, info, a, view, full):
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"episode {index}: {info['frames']} frames from {folder}", flush=True)
    dataset.finalize()
    write_sources(output, episodes, loaded, view, full)
    return verify_v30(output, repo_id, loaded, view)


def verify_v30(output, repo_id, loaded, view="arms16"):
    """Reload the dataset and compare each episode's first and last frames."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=output)
    assert len(dataset) == sum(info["frames"] for info, _ in loaded)
    start = 0
    for info, a in loaded:
        state, action = policy_vectors(a, view)
        for k in (0, info["frames"] - 1):
            item = dataset[start + k]
            assert np.allclose(item["observation.state"].numpy(), state[k])
            assert np.allclose(item["action"].numpy(), action[k])
            assert item["timing.camera_seq"].tolist() == a["camera_seq"][k].tolist()
            assert item["subtask"] == info["subtasks"][a["subtask_index"][k]]
            for camera in CAMERAS:
                assert item[f"observation.images.{camera}"].shape == (3, HEIGHT, WIDTH)
        start += info["frames"]
    on_disk = json.loads((Path(output) / "meta/info.json").read_text())
    return {
        "root": str(output),
        "codebase_version": on_disk["codebase_version"],
        "episodes": dataset.meta.total_episodes,
        "frames": len(dataset),
        "tasks": dataset.meta.tasks.index.tolist(),
        "features": list(dataset.meta.features),
    }


def export(
    episodes,
    output,
    repo_id,
    include_incomplete=False,
    version="v2.1",
    vcodec="libsvtav1",
    view="arms16",
    full=None,
):
    """full: add the robot.* arrays (None: when every episode has them)."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    loaded = [check(folder, include_incomplete) for folder in episodes]
    have_full = all(info["full"] for info, _ in loaded)
    if (view != "arms16" or full) and not have_full:
        raise ValueError(f"view {view} / full needs the full robot in every episode")
    full = have_full if full is None else full
    if version == "v2.1":
        return export_v21(episodes, output, loaded, vcodec, view, full)
    return export_v30(episodes, output, loaded, repo_id, view, full)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new dataset folder")
    parser.add_argument("--format", choices=("v2.1", "v3.0"), default="v2.1")
    parser.add_argument(
        "--vcodec", choices=tuple(ENCODERS), default="libsvtav1", help="v2.1 only"
    )
    parser.add_argument(
        "--repo-id", default="local/inovance_franzi_kitting", help="v3.0 only"
    )
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--view", choices=tuple(VIEWS), default="arms16")
    parser.add_argument(
        "--full",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="add robot.* (default: when every episode has the full robot)",
    )
    args = parser.parse_args()
    report = export(
        args.episodes,
        args.output,
        args.repo_id,
        args.include_incomplete,
        args.format,
        args.vcodec,
        args.view,
        args.full,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
