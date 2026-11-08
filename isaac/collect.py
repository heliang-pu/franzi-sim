"""Collect a scene's expert demonstrations on Isaac Sim straight into a LeRobot
v2.1 dataset:

    ISAAC_PYTHON=<Isaac Sim python> python -m isaac.collect <scene> --seeds 0-19 \\
        --output outputs/<new_dir> [--parallel 8] [--fill] [--display 2]

Runs in the project env, on the pro6000. Each seed is one recorded episode in
its own Isaac Sim process (isaac/run.sh isaac.episode <scene> --record), up to
--parallel at once. With the zero frame delay (isaac/runtime.py, 2026-10-25)
8 at once recorded at 0.80x real time in all (tote_transfer seeds 0-7: 243 s
in 304 s), the CPU then the limit (load about 45 on 24 cores, the GPU busy
26-97 %); RAM allows about 10. Each episode's light comes from its seed
(isaac/stage.py).

--fill replaces every failed seed with the next unused one after the list (at
most a fifth of the list more), so that as many succeed as were listed.
--display N records the overview and mosaic videos only for the first N
seeds; the others record just the four policy cameras, which is all the
dataset and the frame check use.

Then the first successful seed runs again with settled frames (--settle), and
its recorded frames must match the reference's at offset 0
(data.frame_lag.frame_offset): the frames show their own tick. Otherwise
nothing is exported. The successful runs become raw episodes
(data.convert_expert_run) and one LeRobot v2.1 dataset (data.export_lerobot;
data.convert_lerobot makes a v3.0 copy in a lerobot env). <output> holds runs/
(the runs and their logs), check/ (the reference run), raw/ (the raw
episodes), lerobot_v21/ (the dataset) and collect.json (each seed's result,
errors and timings, and the frame check).
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from data import convert_expert_run, export_lerobot
from data.frame_lag import frame_offset
from sim import ROOT

# Extra renders per frame in the reference run, and how many times the next
# best offset's distance must exceed offset 0's (seeds 0 and 9 of tote
# transfer v3 gave 3.6 to 4.2, 2026-10-25).
SETTLE = 4
MIN_RATIO = 2.0
FILL_SHARE = 0.2  # --fill's spare seeds, as a share of the listed ones


def seed_list(text):
    """'0-19' or '0,4,7-9' -> [0, ..., 19] or [0, 4, 7, 8, 9]."""
    seeds = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        seeds += range(int(first), int(last or first) + 1)
    return seeds


def record(scene, seed, runs, settle=0, display=True):
    """One recorded Isaac episode; its result for collect.json."""
    run, log = runs / f"seed_{seed:03d}", runs / f"seed_{seed:03d}.log"
    start = time.monotonic()
    with open(log, "w") as out:
        command = [str(ROOT / "isaac/run.sh"), "isaac.episode", scene]
        command += ["--output", str(run), "--seed", str(seed), "--record"]
        command += ["--settle", str(settle)]
        command += [] if display else ["--no-display"]
        code = subprocess.run(
            command, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT, check=False
        ).returncode
    summary = run / "summary.json"
    result = json.loads(summary.read_text()) if summary.exists() else {}
    print(f"seed {seed}: {'ok' if result.get('success') else 'failed'}", flush=True)
    return {
        "seed": seed,
        "run": str(run),
        "success": bool(result.get("success")),
        "error": result.get("error")
        if summary.exists()
        else f"no summary (exit {code}), see {log.name}",
        "simulator": result.get("simulator"),
        "duration_s": result.get("duration_s"),
        "wall_s": round(time.monotonic() - start, 1),
    }


def record_all(scene, seeds, runs, parallel, fill, display):
    """Record seeds, parallel at once (the first display of them with the
    display videos). With fill, each failed seed is replaced by the next
    unused one after the list, at most FILL_SHARE of the list in all."""
    spare = max(seeds) + 1
    spares = iter(range(spare, spare + max(1, int(len(seeds) * FILL_SHARE))))
    shown = set(seeds[:display])
    results = []
    with ThreadPoolExecutor(parallel) as pool:

        def submit(seed):
            return pool.submit(record, scene, seed, runs, display=seed in shown)

        pending = {submit(s) for s in seeds}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
                seed = (
                    next(spares, None) if fill and not results[-1]["success"] else None
                )
                if seed is not None:
                    pending.add(submit(seed))
    return sorted(results, key=lambda r: r["seed"])


def check_frames(scene, result, folder):
    """The frame check of one successful run against its settled reference."""
    folder.mkdir()
    reference = record(scene, result["seed"], folder, settle=SETTLE, display=False)
    check = {"seed": result["seed"], "reference": reference["run"]}
    try:
        offset, ratio, distances = frame_offset(result["run"], reference["run"])
    except (ValueError, OSError) as e:  # a failed or diverging reference run
        return check | {"passed": False, "error": str(e)}
    return check | {
        "offset": offset,
        "ratio": ratio,
        "distances": distances,
        "passed": offset == 0 and ratio >= MIN_RATIO,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect expert data on Isaac Sim.")
    parser.add_argument("scene")
    parser.add_argument("--seeds", type=seed_list, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new folder")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument(
        "--fill", action="store_true", help="replace failed seeds with new ones"
    )
    parser.add_argument(
        "--display",
        type=int,
        default=None,
        help="record the overview and mosaic videos only for the first N seeds",
    )
    args = parser.parse_args()
    if not os.environ.get("ISAAC_PYTHON"):
        raise SystemExit("set ISAAC_PYTHON to Isaac Sim's interpreter")
    output = args.output.resolve()
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    display = len(args.seeds) if args.display is None else args.display
    results = record_all(
        args.scene, args.seeds, runs, args.parallel, args.fill, display
    )
    recorded = time.monotonic() - start
    succeeded = [r for r in results if r["success"]]
    check = (
        check_frames(args.scene, succeeded[0], output / "check") if succeeded else None
    )
    episodes = []
    if check and check["passed"]:
        for result in succeeded:
            episodes.append(output / "raw" / Path(result["run"]).name)
            convert_expert_run.convert(result["run"], episodes[-1])
    elif check:
        print(f"frame check failed, nothing exported: {json.dumps(check)}", flush=True)
    dataset = None
    if episodes:
        dataset = export_lerobot.export(
            episodes, output / "lerobot_v21", f"local/{args.scene}_isaac"
        )
    report = {
        "scene": args.scene,
        "parallel": args.parallel,
        "seeds": args.seeds,
        "succeeded": len(succeeded),
        "exported": len(episodes),
        "frame_check": check,
        "recording_wall_s": round(recorded, 1),
        "simulated_s": round(sum(r["duration_s"] or 0 for r in results), 1),
        "total_wall_s": round(time.monotonic() - start, 1),
        "dataset": dataset,
        "runs": results,
    }
    (output / "collect.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, default=str))


if __name__ == "__main__":
    main()
