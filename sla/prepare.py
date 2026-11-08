"""Write what SLA's fine-tuning reads about a collection, once:

    python -m sla.prepare --data <isaac collection> --output <assets>/<asset id>

- norm_stats.json: the normalization of the model's state (sla.eef.model_state)
  and of the relative chunks (the actions), with SLA's own RunningStats. Every 10th
  start of each episode, as fineart.relative samples its relative chunks;
- labels/<scene>.npy and labels/texts.json: each frame's prompt (sla.coarse), by the
  dataset's global frame index;
- frames.json: each dataset's frames, for the sampling weights (sla.train).

The joints go through the same FK as training and the server (sla.eef). The object
states stay out of it all: this reads only the state, action and subtask columns.
"""

import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize

from sla import coarse
from sla.eef import Hands, model_state, relative

CHUNK = 50
FPS = 25
EVERY = 10
worker = {}


def vectors(table, name):
    values = table.column(name).combine_chunks()
    width = values.type.list_size
    return values.flatten().to_numpy().reshape(-1, width).astype(np.float64)


def start_worker():
    worker["hands"] = Hands()


def episode(job):
    """One episode's hand poses, relative chunks and prompts."""
    path, scene, instruction = job
    hands = worker["hands"]
    table = pq.read_table(
        path, columns=["observation.state", "action", "subtask", "index"]
    )
    state = hands.pose(vectors(table, "observation.state"))
    target = hands.pose(vectors(table, "action"))
    chunks = []
    for t in range(0, len(state), EVERY):
        ahead = np.minimum(np.arange(t, t + CHUNK), len(target) - 1)
        chunks.append(relative(state[t], target[ahead]))
    subtasks = table.column("subtask").to_pylist()
    if scene in coarse.LONG_SCENES:
        # A marker's neighbour is the next run of expert text, not the next frame.
        runs = [
            i for i in range(len(subtasks)) if i == 0 or subtasks[i] != subtasks[i - 1]
        ]
        texts = coarse.coarsen(scene, [subtasks[i] for i in runs])
        stops = [*runs[1:], len(subtasks)]
        prompts = [
            text
            for text, begin, stop in zip(texts, runs, stops, strict=True)
            for _ in range(stop - begin)
        ]
    else:
        prompts = [instruction] * len(subtasks)
    return {
        "state": state,
        "actions": np.concatenate(chunks),
        "prompts": prompts,
        "index": table.column("index").to_numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=16)
    args = parser.parse_args()
    (args.output / "labels").mkdir(parents=True, exist_ok=True)
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    texts, frames = {}, {}
    with Pool(args.processes, initializer=start_worker) as pool:
        for root in sorted(args.data.glob("*/lerobot_v21")):
            scene = root.parent.name
            meta = json.loads((root / "meta/info.json").read_text())
            assert meta["fps"] == FPS, root
            tasks = {}
            for line in (root / "meta/tasks.jsonl").read_text().splitlines():
                row = json.loads(line)
                tasks[row["task_index"]] = row["task"]
            jobs = []
            for index in range(meta["total_episodes"]):
                path = root / meta["data_path"].format(
                    episode_chunk=index // meta["chunks_size"], episode_index=index
                )
                task = pq.read_table(path, columns=["task_index"]).column("task_index")
                jobs.append((path, scene, tasks[int(task[0].as_py())]))
            ids = np.full(meta["total_frames"], -1, np.int16)
            seen = Counter()
            for result in pool.imap_unordered(episode, jobs):
                stats["state"].update(model_state(result["state"]))
                stats["actions"].update(result["actions"])
                prompts = result["prompts"]
                code = [texts.setdefault(p, len(texts)) for p in prompts]
                ids[result["index"]] = code
                seen.update(prompts)
                angle = np.linalg.norm(result["state"][:, [3, 4, 5]], axis=1).max()
                angle = max(
                    angle,
                    np.linalg.norm(result["state"][:, [10, 11, 12]], axis=1).max(),
                )
                seen["<largest rotation, deg>"] = max(
                    seen["<largest rotation, deg>"], int(np.degrees(angle))
                )
            assert (ids >= 0).all(), scene
            np.save(args.output / "labels" / f"{scene}.npy", ids)
            frames[scene] = int(meta["total_frames"])
            print(f"== {scene}: {meta['total_frames']} frames", flush=True)
            for text, count in seen.most_common():
                print(f"  {count:8d}  {text}", flush=True)
    normalize.save(args.output, {k: s.get_statistics() for k, s in stats.items()})
    (args.output / "labels/texts.json").write_text(
        json.dumps(sorted(texts, key=texts.get), indent=1)
    )
    (args.output / "frames.json").write_text(json.dumps(frames, indent=1))
    print("SLA_PREPARED", len(texts), "prompts", flush=True)


if __name__ == "__main__":
    main()
