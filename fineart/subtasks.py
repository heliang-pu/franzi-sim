"""How often a checkpoint names the current subtask: on frames spread evenly
over the held-out episodes (fineart.data's validation split), the subtask the
policy generates from the task, the four frames and the state (as fineart.serve
asks for it), against the frame's label. A match is the exact text. Frames near
a subtask's start or end are ambiguous, so a perfect model scores below 100%.
A checkpoint with the memory recipe (fineart.recipe) is given the label of
SUBTASK_EVERY ticks back as its memory, as in training.

--chain asks as the closed loop does (isaac.policy): every SUBTASK_EVERY frames
(3 s) from each episode's start, the memory being the previous answer. On the
expert's own frames, an episode then shows whether the answers follow the
subtasks through a long task. Each episode's summary gives the share of answers
that match, how many of the subtasks met at the asked frames were named while
they lasted, and the longest run of wrong answers in seconds.

    python -m fineart.subtasks --data <collection> --model <checkpoint> \
        --tokenizer <paligemma_tokenizer> --output <new .json> \
        [--per-episode 10 | --chain] [--scenes palletizing,cnc_tending]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.utils.constants import MESSAGES_RENDERED

from fineart import recipe
from fineart.data import CAMERAS, FPS, ROBOT, Episodes, scene_roots
from fineart.protocol import NO_MEMORY, SUBTASK_EVERY
from fineart.serve import SUBTASK_TOKENS
from fineart.train import load


def language(sample, style):
    (row,) = [r for r in sample["language_persistent"] if r["style"] == style]
    return row["content"]


def ask(policy, pre, prompt, sample, memory):
    """The subtask the policy generates for a sample, as fineart.serve asks."""
    obs = {
        "observation.state": sample["observation.state"],
        "task": sample["task"],
        "robot_type": ROBOT,
    }
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        obs[key] = sample[key].float() / 255
    messages = recipe.subtask_messages(prompt, sample["task"], memory)
    query = obs | {MESSAGES_RENDERED: messages}
    with torch.inference_mode():
        return policy.select_message(
            pre(query), max_new_tokens=SUBTASK_TOKENS, suppress_loc_tokens=True
        ).strip()


def chain_summary(rows):
    """One episode's chained answers (--chain)."""
    runs = []  # [label, named] per run of equal labels over the asked frames
    for row in rows:
        if not runs or runs[-1][0] != row["label"]:
            runs.append([row["label"], False])
        runs[-1][1] |= row["match"]
    wrong = longest = 0
    for row in rows:
        wrong = 0 if row["match"] else wrong + 1
        longest = max(longest, wrong)
    return {
        "scene": rows[0]["scene"],
        "episode": rows[0]["episode"],
        "queries": len(rows),
        "accuracy": float(np.mean([r["match"] for r in rows])),
        "subtasks": len(runs),
        "named": sum(named for _, named in runs),
        "longest_wrong_s": longest * SUBTASK_EVERY / FPS,
        "last_match": rows[-1]["match"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-episode", type=int, default=10)
    parser.add_argument(
        "--chain", action="store_true", help="ask as the closed loop does"
    )
    parser.add_argument("--scenes", help="comma-separated (default: all ten)")
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    policy, pre, _ = load(args.model, args.tokenizer, "cuda:0", False)
    prompt = recipe.load(args.model)
    memory_recipe = recipe.uses_memory(prompt)
    validation = Episodes(scene_roots(args.data), "validation")
    scenes = set(args.scenes.split(",")) if args.scenes else None
    start = time.monotonic()
    rows, episodes, offset = [], [], 0
    for episode in validation.episodes:
        n = len(episode["state"])
        if scenes is not None and episode["scene"] not in scenes:
            offset += n
            continue
        if args.chain:
            frames = range(0, n, SUBTASK_EVERY)
        else:
            frames = np.linspace(0, n - 1, args.per_episode, dtype=int)
        answer, asked = None, []
        for frame in frames:
            sample = validation[offset + int(frame)]
            memory = answer if args.chain else language(sample, "memory")
            answer = ask(policy, pre, prompt, sample, memory)
            label = language(sample, "subtask")
            asked.append(
                {
                    "scene": episode["scene"],
                    "episode": episode["episode_index"],
                    "frame": int(frame),
                    "memory": (memory or NO_MEMORY) if memory_recipe else None,
                    "label": label,
                    "predicted": answer,
                    "match": answer == label,
                }
            )
        rows += asked
        if args.chain:
            episodes.append(chain_summary(asked))
            print(json.dumps(episodes[-1]), flush=True)
        offset += n
    per_scene = {}
    for row in rows:
        per_scene.setdefault(row["scene"], []).append(row["match"])
    report = {
        "model": str(args.model),
        "memory": memory_recipe,
        "chain": args.chain,
        "frames": len(rows),
        "accuracy": float(np.mean([r["match"] for r in rows])),
        "per_scene": {s: float(np.mean(v)) for s, v in sorted(per_scene.items())},
        "seconds": round(time.monotonic() - start, 1),
        "episodes": episodes if args.chain else None,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps({k: report[k] for k in ("frames", "accuracy", "per_scene")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
