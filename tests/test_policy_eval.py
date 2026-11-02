"""Engine-free tests of the policy evaluation's plumbing: the server's wire
format (fineart.protocol) and isaac.evaluate's success-rate report. Run from the
project root with the rest of the suite."""

import numpy as np

from fineart.protocol import address, pack, unpack
from isaac.evaluate import report


def test_arrays_survive_the_wire_format():
    for array in (
        np.arange(800, dtype=np.float32).reshape(50, 16),
        np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8),
    ):
        out = unpack(pack(array))
        assert out.dtype == array.dtype and out.shape == array.shape
        assert np.array_equal(out, array)


def test_server_addresses():
    assert address("18780") == ("127.0.0.1", 18780)
    assert address("10.0.0.2:9000") == ("10.0.0.2", 9000)


def test_report_counts_rates_without_void_episodes():
    def run(scene, seed, termination):
        return {
            "scene": scene,
            "seed": seed,
            "success": termination == "success",
            "completed": termination in ("success", "completed"),
            "termination": termination,
            "error": None,
            "progress": 0,
            "collisions": [],
            "violations": [],
            "duration_s": 10.0 + seed,
        }

    results = [
        run("a", 0, "success"),
        run("a", 1, "time_limit"),
        run("a", 2, "void"),
        run("b", 0, "error"),
        run("b", 1, "completed"),
    ]
    out = report(results, ["a", "b"])
    a, b = out["scenes"]["a"], out["scenes"]["b"]
    assert (a["success_rate"], a["episodes"], a["void"]) == (0.5, 2, 1)
    assert a["completion_rate"] == 0.5
    assert a["terminations"] == {"success": 1, "time_limit": 1, "void": 1}
    assert a["success_duration_s"] == [10.0, 10.0]
    assert [f["seed"] for f in a["failed"]] == [1]
    assert b["success_rate"] == 0.0 and b["success_duration_s"] is None
    # A task done while touching something counts only toward completion.
    assert (b["completion_rate"], b["completions"]) == (0.5, 1)
    assert [f["seed"] for f in b["failed"]] == [0, 1]
    assert out["mean_success_rate"] == 0.25
    assert out["mean_completion_rate"] == 0.5
