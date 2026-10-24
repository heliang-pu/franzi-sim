"""Validate a retained local expert run and its recorded camera streams."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import av
import numpy as np

from sim.build_task import GOALS, TABLE_Z


def verify(folder):
    result = json.loads((folder / "summary.json").read_text())
    trace = json.loads((folder / "trace.json").read_text())
    assert result["success"] and result["error"] is None
    assert result["physics"]["max_penetration_mm"] <= 1
    assert result["physics"]["max_joint_limit_error"] <= 0.002
    assert result["final_stable_control_frames"] >= 15
    for side in ("left", "right"):
        evidence = result["lift_evidence"][side]
        assert evidence["airborne_bilateral_steps"] >= 100
        assert evidence["peak_height_m"] > TABLE_Z + 0.10
        final = result["final_objects"][side]
        assert not final["touching_robot"]
        assert final["velocity_m_s"] < 0.025
        assert np.linalg.norm(np.array(final["xyz"][:2]) - GOALS[side]) < 0.025
        assert abs(final["xyz"][2] - (TABLE_Z + 0.006 + 0.0275)) < 0.004
    times = np.array([x["t"] for x in trace])
    assert np.allclose(np.diff(times), 1 / 25, atol=1e-8)
    for key in ("state", "action"):
        values = np.array([x[key] for x in trace])
        assert values.shape == (len(trace), 16) and np.isfinite(values).all()
    root = ET.parse(folder / "scene.xml").getroot()
    assert root.find(".//weld") is None
    assert len(root.findall(".//freejoint")) == 2
    for side in ("left", "right"):
        wrist = root.find(f".//body[@name='{side}_wrist_roll_Link']")
        assert wrist.find(f"geom[@name='{side}_wrist_roll_Link']") is None
        assert sum(g.get("type") == "mesh" for g in wrist.findall("geom")) >= 40
        for finger in (1, 2):
            g = root.find(f".//geom[@name='{side}finger{finger}_Link']")
            assert g.get("type") == "mesh" and int(g.get("contype")) != 0
    videos = {}
    names = ["expert", *result["contract"]["cameras"]]
    if (folder / "cameras.mp4").exists():
        names.append("cameras")
    for name in names:
        with av.open(folder / (name + ".mp4")) as container:
            stream = container.streams.video[0]
            assert stream.frames == len(trace)
            assert float(stream.average_rate) == 25
            frame = next(container.decode(video=0))
            assert frame.width == (1280 if name in {"expert", "cameras"} else 640)
            videos[name] = {"frames": stream.frames, "fps": 25}
    return {
        "verified": True,
        "control_samples": len(trace),
        "videos": videos,
        "max_penetration_mm": result["physics"]["max_penetration_mm"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    report = verify(args.folder)
    (args.folder / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
