"""Engine-free tests of the LeRobot v2.1 export and the v2.1 <-> v3.0
conversion (data/lerobot_v21.py, data/export_lerobot.py,
data/convert_lerobot.py). They need pyarrow, datasets and LeRobot, so they run
in the lerobot env and are skipped in inovance-fineart:

    /Users/puheliang/miniforge3/envs/lerobot/bin/python -m pytest tests/test_lerobot_formats.py -q

Two short raw episodes, with a different task each, every value numbered by
its frame and every camera image one grey level per frame and camera.
H.264 and then AV1 leave such flat frames up to 4 levels darker (RGB <-> YUV
rounding), so the levels are 12 apart between frames and 24 between cameras.
"""

import json

import numpy as np
import pytest

from data.vla_episode import ARRAYS, CAMERAS, EpisodeWriter

pytest.importorskip("pyarrow")
pytest.importorskip("datasets")
pytest.importorskip("lerobot.scripts.convert_dataset_v21_to_v30")

LENGTHS = (6, 9)


def grey(episode, camera, k):
    return 16 + 24 * CAMERAS.index(camera) + 12 * k + 6 * episode


@pytest.fixture(scope="module")
def v21_dataset(tmp_path_factory):
    from data.export_lerobot import export

    root = tmp_path_factory.mktemp("formats")
    folders = []
    for e, n in enumerate(LENGTHS):
        writer = EpisodeWriter(root / f"raw_{e}")
        for k in range(n):
            row = {
                key: np.full(shape, 100 * e + k) for key, (_, shape) in ARRAYS.items()
            }
            row["subtask"] = f"step {k // 3}"
            images = {
                c: np.full((480, 640, 3), grey(e, c, k), np.uint8) for c in CAMERAS
            }
            writer.add(row, images)
        writer.close({"task": f"task {e}", "complete": True, "error": None})
        folders.append(root / f"raw_{e}")
    export(folders, root / "v21", "local/test")
    return root


def test_v21_export_layout_and_values(v21_dataset):
    import pyarrow.parquet as pq

    from data import lerobot_v21 as v21
    from data.convert_lerobot import frame_at

    root = v21_dataset / "v21"
    info, tasks, episodes, stats = v21.read(root)
    assert (info["total_episodes"], info["total_frames"]) == (2, sum(LENGTHS))
    assert info["total_videos"] == 2 * len(CAMERAS) and info["total_chunks"] == 1
    assert tasks == ["task 0", "task 1"]
    assert [(ep["length"], ep["tasks"]) for ep in episodes] == [
        (6, ["task 0"]),
        (9, ["task 1"]),
    ]
    for e, n in enumerate(LENGTHS):
        rows = pq.read_table(v21.data_path(root, e))
        hf = json.loads(rows.schema.metadata[b"huggingface"])["info"]["features"]
        assert hf["observation.state"]["_type"] == "Sequence"  # datasets 3.x form
        assert "List" not in json.dumps(hf)
        values = rows.to_pydict()
        k = np.arange(n)
        assert values["observation.state"] == [[100.0 * e + i] * 16 for i in k]
        assert values["timestamp"] == (k / 25).astype(np.float32).tolist()
        assert values["index"] == (k + sum(LENGTHS[:e])).tolist()
        assert values["task_index"] == [e] * n
        assert values["subtask"] == [f"step {i // 3}" for i in k]
        assert stats[e]["observation.state"]["q50"] == [100.0 * e + (n - 1) / 2] * 16
        for camera in CAMERAS:
            key = f"observation.images.{camera}"
            assert stats[e][key]["count"] == [n]  # all frames sampled below 100
            assert np.array(stats[e][key]["mean"]).shape == (3, 1, 1)
            path = v21.video_path(root, e, key)
            assert v21.video_info(path)["video.codec"] == "av1"
            for i in (0, n - 1):
                frame = frame_at(path, i / 25, 25).astype(int)
                assert np.abs(frame - grey(e, camera, i)).mean() <= 5, (path, i)


def test_v21_v30_round_trip_is_lossless(v21_dataset):
    """v2.1 -> v3.0 (LeRobot's converter) -> v2.1 gives back the same files:
    metadata, stats, parquet schemas and values, and video packets."""
    import av
    import pyarrow.parquet as pq

    from data import lerobot_v21 as v21
    from data.convert_lerobot import convert

    first, v30, back = (v21_dataset / name for name in ("v21", "v30", "back"))
    assert convert(first, v30, "local/test")["to"] == "v3.0"
    # No datasets 4 features where LeRobot 33cad37 (datasets 3.6.0) parses them.
    for path in (v30 / "meta/episodes").glob("*/*.parquet"):
        assert b"huggingface" not in (pq.read_schema(path).metadata or {})
    assert convert(v30, back, "local/test")["to"] == "v2.1"
    assert v21.read(first) == v21.read(back)
    for name in (v21.STATS, "meta/inovance_sources.json"):
        assert json.loads((first / name).read_text()) == json.loads(
            (back / name).read_text()
        )

    def packets(root, e, camera):
        path = v21.video_path(root, e, f"observation.images.{camera}")
        with av.open(str(path)) as container:
            return [(p.pts, bytes(p)) for p in container.demux(video=0) if p.size]

    for e, n in enumerate(LENGTHS):
        a = pq.read_table(v21.data_path(first, e))
        b = pq.read_table(v21.data_path(back, e))
        assert a.schema.equals(b.schema, check_metadata=True) and a.equals(b)
        for camera in CAMERAS:
            ours = packets(first, e, camera)
            assert ours == packets(back, e, camera) and len(ours) == n
