"""Write raw VLA episodes (vla_episode.py) as one LeRobot v3.0 dataset.

    /Users/puheliang/miniforge3/envs/lerobot/bin/python -m data.export_lerobot \
        --episodes outputs/<raw_episode> [...] --output outputs/<new_dataset>

Runs in the lerobot conda env (LeRobot 0.5.2, dataset format v3.0); the
recorder and the expert converter run in inovance-fineart and do not need it.
SDK recordings and converted expert runs share one feature set, so they can be
mixed in one dataset (features: vla_episode.LEROBOT_FEATURES; only the 16-D
state and the four cameras are policy inputs). ``timestamp`` is LeRobot's
frame_index / 25; the measured robot time is ``timing.robot_s``. ``task`` is
the episode instruction and ``subtask`` the per-step one. Videos are
re-encoded with LeRobot's defaults (AV1).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from data.vla_episode import (
    CAMERAS,
    FPS,
    HEIGHT,
    LEROBOT_FEATURES,
    ROBOT,
    WIDTH,
    camera_frames,
    read_episode,
    video_info,
)


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


def frames(folder, info, a):
    videos = {c: camera_frames(folder, c) for c in CAMERAS}
    for k in range(info["frames"]):
        yield {
            "observation.state": a["state"][k].astype(np.float32),
            "action": a["action"][k].astype(np.float32),
            **{f"observation.images.{c}": next(videos[c]) for c in CAMERAS},
            "command.setpoint": a["setpoint"][k].astype(np.float32),
            "command.arm_source": a["arm_source"][k].astype(np.int64),
            "timing.robot_s": a["robot_s"][k : k + 1],
            "timing.state_utc_ms": a["state_utc_ms"][k : k + 1],
            "timing.camera_seq": a["camera_seq"][k],
            "timing.camera_utc_ms": a["camera_utc_ms"][k],
            "subtask": info["subtasks"][a["subtask_index"][k]],
            "task": info["task"],
        }
    for camera, video in videos.items():
        if next(video, None) is not None:
            raise ValueError(f"{folder}: {camera} video has extra frames")


def export(episodes, output, repo_id, include_incomplete=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    loaded = [check(folder, include_incomplete) for folder in episodes]
    dataset = LeRobotDataset.create(
        repo_id,
        fps=FPS,
        features=LEROBOT_FEATURES,
        root=output,
        robot_type=ROBOT,
        use_videos=True,
        streaming_encoding=True,
    )
    sources = []
    for index, (folder, (info, a)) in enumerate(zip(episodes, loaded)):
        for frame in frames(folder, info, a):
            dataset.add_frame(frame)
        dataset.save_episode()
        keep = ("source", "complete", "clock", "images", "driver", "source_run")
        sources.append(
            {
                "episode_index": index,
                "raw_episode": str(folder),
                "frames": info["frames"],
                **{k: info[k] for k in keep if k in info},
                "warnings": info.get("warnings", []),
            }
        )
        print(f"episode {index}: {info['frames']} frames from {folder}", flush=True)
    dataset.finalize()
    conventions = json.loads((Path(episodes[0]) / "episode.json").read_text())
    (output / "meta/inovance_sources.json").write_text(
        json.dumps(
            {"conventions": conventions["conventions"], "episodes": sources}, indent=2
        )
    )
    return verify(output, repo_id, loaded)


def verify(output, repo_id, loaded):
    """Reload the dataset and compare each episode's first and last frames."""
    dataset = LeRobotDataset(repo_id, root=output)
    assert len(dataset) == sum(info["frames"] for info, _ in loaded)
    start = 0
    for info, a in loaded:
        for k in (0, info["frames"] - 1):
            item = dataset[start + k]
            assert np.allclose(item["observation.state"].numpy(), a["state"][k])
            assert np.allclose(item["action"].numpy(), a["action"][k])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new dataset folder")
    parser.add_argument("--repo-id", default="local/inovance_franzi_kitting")
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()
    report = export(args.episodes, args.output, args.repo_id, args.include_incomplete)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
