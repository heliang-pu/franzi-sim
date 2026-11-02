"""Evaluate a policy on the benchmark scenes closed-loop on Isaac Sim and
count its success rate:

    ISAAC_PYTHON=<Isaac Sim python> python -m isaac.evaluate --output outputs/<new_dir> \\
        [--scenes tote_transfer,tray_push] [--seeds 10000-10019] [--server 127.0.0.1:18780] \\
        [--parallel 8] [--data <collection>] [--display 1] [--stop-on-violation]

Runs in the project env, on the pro6000, with fineart.serve already serving
the policy. Each (scene, seed) is one episode in its own Isaac Sim process
(isaac/run.sh isaac.policy), up to --parallel at once. The seeds default to
10000-10019: the training batches drew from 0 up (rand200 used 0-239). Each
scene's time limit is LIMIT_FACTOR times its longest demonstration in --data,
rounded up to 5 s (DEFAULT_LIMITS without it).

The episodes run with isaac.policy --lenient (unless --stop-on-violation): a
penetration over 1 mm or a joint past its limit is logged instead of ending
the episode. The success rate is the benchmark's (any violation or collision is
a failure, as when the episode stops at once); the completion rate counts the
episodes whose task got done whatever they touched (the summary's
"completed"), to tell a policy that can't do a task from one that does it
roughly.

<output> holds <scene>/seed_<n>/ (each run, its log beside it) and report.json:
per scene the success and completion rates, each termination (success,
completed, time_limit, incomplete, error, void) and the seeds that failed and
why; overall the means of the scenes' rates. Void episodes (the simulation
broke down before any violation) are left out of the rates, as the experts'
are redrawn.
"""

import argparse
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client
from pathlib import Path

from fineart.protocol import AUTHKEY, PORT, SUBTASK_EVERY, address

ROOT = Path(__file__).resolve().parents[1]

SCENES = (
    "tote_transfer",
    "panel_operation",
    "tray_push",
    "parcel_sorting",
    "tote_carry",
    "shaped_kitting",
    "lidded_tote",
    "palletizing",
    "cnc_tending",
    "precision_insertion",
)
LIMIT_FACTOR = 1.5
# 1.5x the longest of each scene's 200 Isaac demonstrations (2026-10-28).
DEFAULT_LIMITS = {
    "tote_transfer": 80,
    "panel_operation": 95,
    "tray_push": 50,
    "parcel_sorting": 195,
    "tote_carry": 60,
    "shaped_kitting": 230,
    "lidded_tote": 170,
    "palletizing": 275,
    "cnc_tending": 295,
    "precision_insertion": 90,
}


def limits(data):
    """Each scene's time limit from its dataset's longest episode."""
    if data is None:
        return dict(DEFAULT_LIMITS)
    out = {}
    for scene in SCENES:
        episodes = Path(data) / scene / "lerobot_v21/meta/episodes.jsonl"
        longest = max(
            json.loads(line)["length"] for line in episodes.read_text().splitlines()
        )
        out[scene] = 5 * math.ceil(LIMIT_FACTOR * longest / 25 / 5)
    return out


def episode(scene, seed, folder, server, limit, args, display):
    run, log = folder / f"seed_{seed}", folder / f"seed_{seed}.log"
    start = time.monotonic()
    with open(log, "w") as out:
        command = [str(ROOT / "isaac/run.sh"), "isaac.policy", scene]
        command += ["--output", str(run), "--seed", str(seed), "--server", server]
        command += ["--limit", str(limit), "--horizon", str(args.horizon)]
        command += ["--subtask-every", str(args.subtask_every)]
        command += [] if args.stop_on_violation else ["--lenient"]
        command += ["--display"] if display else []
        code = subprocess.run(
            command, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT, check=False
        ).returncode
    summary = run / "summary.json"
    result = json.loads(summary.read_text()) if summary.exists() else {}
    termination = result.get("termination", "crash")
    print(f"{scene} seed {seed}: {termination}", flush=True)
    calls = result.get("policy", {}).get("calls", [])
    return {
        "scene": scene,
        "seed": seed,
        "run": str(run),
        "success": bool(result.get("success")),
        "completed": bool(result.get("completed")),
        "termination": termination,
        "error": result.get("error")
        if summary.exists()
        else f"no summary (exit {code}), see {log.name}",
        "duration_s": result.get("duration_s"),
        "progress": result.get("progress"),
        "milestones": result.get("milestones"),
        "collisions": result.get("physics", {}).get("collisions"),
        "violations": result.get("physics", {}).get("violations"),
        "policy_calls": len(calls),
        "policy_seconds": round(sum(c.get("seconds", 0) for c in calls), 1),
        "wall_s": round(time.monotonic() - start, 1),
    }


def report(results, scenes):
    rows = {}
    for scene in scenes:
        mine = [r for r in results if r["scene"] == scene]
        counted = [r for r in mine if r["termination"] != "void"]
        ok = [r for r in counted if r["success"]]
        done = [r for r in counted if r["completed"]]
        terminations = {}
        for r in mine:
            terminations[r["termination"]] = terminations.get(r["termination"], 0) + 1
        rows[scene] = {
            "success_rate": len(ok) / len(counted) if counted else None,
            "successes": len(ok),
            "completion_rate": len(done) / len(counted) if counted else None,
            "completions": len(done),
            "episodes": len(counted),
            "void": len(mine) - len(counted),
            "terminations": terminations,
            "success_duration_s": [
                min(r["duration_s"] for r in ok),
                max(r["duration_s"] for r in ok),
            ]
            if ok
            else None,
            "failed": [
                {
                    k: r[k]
                    for k in (
                        "seed",
                        "termination",
                        "error",
                        "progress",
                        "collisions",
                        "violations",
                    )
                }
                for r in counted
                if not r["success"]
            ],
        }

    def mean(key):
        rates = [row[key] for row in rows.values() if row[key] is not None]
        return sum(rates) / len(rates) if rates else None

    return {
        "mean_success_rate": mean("success_rate"),
        "mean_completion_rate": mean("completion_rate"),
        "scenes": rows,
    }


def main():
    from isaac.collect import seed_list  # the data tools, which report() doesn't need

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--seeds", type=seed_list, default=seed_list("10000-10019"))
    parser.add_argument(
        "--server", default=str(PORT), help="host:port of fineart.serve"
    )
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument(
        "--data", type=Path, help="collection folder for the time limits"
    )
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--subtask-every", type=int, default=SUBTASK_EVERY)
    parser.add_argument(
        "--stop-on-violation",
        action="store_true",
        help="end an episode at its first violation (no completion rate)",
    )
    parser.add_argument(
        "--display", type=int, default=1, help="seeds per scene with expert.mp4"
    )
    args = parser.parse_args()
    if not os.environ.get("ISAAC_PYTHON"):
        raise SystemExit("set ISAAC_PYTHON to Isaac Sim's interpreter")
    scenes = [s for s in args.scenes.split(",") if s]
    assert set(scenes) <= set(SCENES), scenes
    with Client(address(args.server), authkey=AUTHKEY) as connection:
        connection.send({"op": "info"})
        info = connection.recv()
    args.output.mkdir(parents=True, exist_ok=False)
    limit = limits(args.data)
    started = time.time()
    jobs = []
    for scene in scenes:
        (args.output / scene).mkdir()
        for i, seed in enumerate(args.seeds):
            jobs.append((scene, seed, i < args.display))
    with ThreadPoolExecutor(args.parallel) as pool:
        futures = [
            pool.submit(
                episode,
                scene,
                seed,
                args.output / scene,
                args.server,
                limit[scene],
                args,
                display,
            )
            for scene, seed, display in jobs
        ]
        results = [f.result() for f in futures]
    out = {
        "policy": info,
        "seeds": args.seeds,
        "limits_s": {s: limit[s] for s in scenes},
        "horizon": args.horizon,
        "subtask_every": args.subtask_every,
        "lenient": not args.stop_on_violation,
        "wall_s": round(time.time() - started, 1),
        **report(results, scenes),
        "episodes": results,
    }
    (args.output / "report.json").write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                s: [r["success_rate"], r["completion_rate"]]
                for s, r in out["scenes"].items()
            }
        ),
        flush=True,
    )
    print(
        "mean success rate",
        out["mean_success_rate"],
        "mean completion rate",
        out["mean_completion_rate"],
        flush=True,
    )


if __name__ == "__main__":
    main()
