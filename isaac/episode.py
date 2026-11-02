"""Run a scene's scripted expert closed-loop on Isaac Sim 5.1:

    ISAAC_PYTHON=<Isaac Sim python> isaac/run.sh isaac.episode <scene> --output outputs/<new_run> --seed 0 [--record]

The scene's run.run(output, seed, record, engine=...) runs as on MuJoCo, with
isaac.engine.isaac_scene as the engine, and writes the same run files
(episode.json, summary.json, trace.json, trajectory.npz and, with --record,
the videos), plus the episode's USD stage in <run>/isaac/. Every benchmark
scene's run() takes engine. --settle N renders every recorded frame N more
times: a reference that a recording of the same seed is checked against
(data.frame_lag.frame_offset; the physics does not depend on rendering).
--no-display records only the four policy cameras' videos, without the
overview (expert.mp4) and the mosaic (cameras.mp4).

Isaac Sim's Python needs MuJoCo, PyAV and Pinocchio for this (pin 2.7.0:
Pinocchio 3's wheels need numpy 2, and Isaac Sim 5.1 has numpy 1.26).
"""

import argparse
import importlib
import json
import traceback
from pathlib import Path

from isaac import runtime


def main():
    parser = argparse.ArgumentParser(description="Run a scene's expert on Isaac Sim.")
    parser.add_argument("scene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--settle",
        type=int,
        default=0,
        help="extra renders per recorded frame: a reference to check recordings by",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="record only the policy cameras (no expert.mp4, no cameras.mp4)",
    )
    args = parser.parse_args()
    runtime.SETTLE = args.settle
    app = runtime.start_app()
    try:
        from isaac import engine
        from isaac.engine import isaac_scene

        engine.DISPLAY = not args.no_display

        scene = importlib.import_module(f"scenes.{args.scene}.run")
        result = scene.run(
            args.output.resolve(), args.seed, args.record, engine=isaac_scene
        )
        summary = {k: result[k] for k in ("success", "error", "progress", "duration_s")}
        print("ISAAC_EPISODE", json.dumps(summary), flush=True)
    except Exception:
        # SimulationApp.close() ends the process before Python prints it.
        traceback.print_exc()
        print("ISAAC_EPISODE_FAILED", flush=True)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
