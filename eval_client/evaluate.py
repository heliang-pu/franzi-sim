"""Evaluate a policy on the benchmark scenes closed-loop and count its success
rate:

    ISAAC_PYTHON=<Isaac Sim python> python -m eval_client.evaluate --policy <name> --output outputs/<new_dir> \\
        [--scenes tote_transfer,tray_push] [--seeds 10000-10019] [--server ws://127.0.0.1:18780] \\
        [--parallel 8] [--data <collection>] [--display 1] [--set horizon=50] [--stop-on-violation] [--engine mujoco]

Runs in the project env, on the pro6000, with policy.serve already serving the
policy. Each (scene, seed) is one episode in its own simulator process
(isaac/run.sh eval_client.episode, or the project env's Python for MuJoCo), up
to --parallel at once, all against the one server. The seeds default to
10000-10019: the training batches drew from 0 up (rand200 used 0-239). Each
scene's time limit is LIMIT_FACTOR times its longest demonstration in --data,
rounded up to 5 s (DEFAULT_LIMITS without it).

The episodes run --lenient (unless --stop-on-violation): a penetration over
1 mm or a joint past its limit is logged instead of ending the episode. The
success rate is the benchmark's (any violation or collision is a failure, as
when the episode stops at once); the completion rate counts the episodes whose
task got done whatever they touched (the summary's "completed"), to tell a
policy that can't do a task from one that does it roughly.

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
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from policy.serve import PORT, load_config

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


def episode(scene, seed, folder, limit, args, display):
    run, log = folder / f"seed_{seed}", folder / f"seed_{seed}.log"
    start = time.monotonic()
    with open(log, "w") as out:
        if args.engine == "isaac":
            command = [str(ROOT / "isaac/run.sh"), "eval_client.episode", scene]
        else:
            command = [sys.executable, "-m", "eval_client.episode", scene]
        command += ["--policy", args.policy, "--engine", args.engine]
        command += ["--output", str(run), "--seed", str(seed)]
        command += ["--server", args.server, "--limit", str(limit)]
        for item in args.set:
            command += ["--set", item]
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


def describe(server):
    """The served policy's describe(), which also shows the server is up."""
    from eval_client.episode import PolicyClient

    client = PolicyClient(server, "evaluate")
    try:
        return client.call(func_name="describe")
    finally:
        client.close()


def main():
    from isaac.collect import seed_list  # the data tools, which report() doesn't need

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="policy/<name>/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--seeds", type=seed_list, default=seed_list("10000-10019"))
    parser.add_argument("--server", default=f"ws://127.0.0.1:{PORT}")
    parser.add_argument("--engine", choices=("isaac", "mujoco"), default="isaac")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument(
        "--data", type=Path, help="collection folder for the time limits"
    )
    parser.add_argument(
        "--stop-on-violation",
        action="store_true",
        help="end an episode at its first violation (no completion rate)",
    )
    parser.add_argument(
        "--display", type=int, default=1, help="seeds per scene with expert.mp4"
    )
    args = parser.parse_args()
    if args.engine == "isaac" and not os.environ.get("ISAAC_PYTHON"):
        raise SystemExit("set ISAAC_PYTHON to Isaac Sim's interpreter")
    scenes = [s for s in args.scenes.split(",") if s]
    assert set(scenes) <= set(SCENES), scenes
    config = load_config(ROOT / "policy" / args.policy / "deploy.yml", args.set)
    info = describe(args.server)
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
                episode, scene, seed, args.output / scene, limit[scene], args, display
            )
            for scene, seed, display in jobs
        ]
        results = [f.result() for f in futures]
    out = {
        "policy": info,
        "policy_name": args.policy,
        "config": config,
        "engine": args.engine,
        "seeds": args.seeds,
        "limits_s": {s: limit[s] for s in scenes},
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
