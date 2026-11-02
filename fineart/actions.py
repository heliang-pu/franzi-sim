"""How far a checkpoint's actions are from the expert's, open loop: on frames
spread evenly over a few episodes of each scene, held-out ones (fineart.data's
validation split) and as many training ones for comparison, the chunk the
policy predicts for the frame's own subtask label (the low-level prompt alone,
as fineart.serve asks once it knows the subtask) against the recorded chunk.
The flow matching's noise is seeded by the frame's position in the list (plus
--seed), so every checkpoint gets the same noise; another --seed shows how much
the chunks vary with the noise alone.

The .npz holds the predicted and recorded chunks and the states in the
datasets' SDK units, with each frame's split, scene, episode and frame index;
the joint errors are printed. The hand errors need the shared kinematics
(Pinocchio), which this env lacks: the project env computes them from the .npz.

    python -m fineart.actions --data <collection> --model <checkpoint> \
        --tokenizer <paligemma_tokenizer> --output <new .npz> \
        [--episodes 4] [--per-episode 10] [--seed 0]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fineart.data import CAMERAS, CHUNK, ROBOT, Episodes, scene_roots
from fineart.serve import action_prompt
from fineart.subtasks import language
from fineart.train import load

APPLIED = 25  # actions of a chunk isaac.policy applies before asking again


def chosen(split, count):
    """count episodes of each scene in a split, spread evenly over it."""
    by_scene = {}
    for i, episode in enumerate(split.episodes):
        by_scene.setdefault(episode["scene"], []).append(i)
    picked = []
    for _, indices in sorted(by_scene.items()):
        spread = np.rint(np.linspace(0, len(indices) - 1, min(count, len(indices))))
        picked += [indices[k] for k in spread.astype(int)]
    return picked


def predict(policy, pre, post, prompt, sample, seed):
    """The policy's chunk for a sample, given its subtask label (SDK units)."""
    obs = {"observation.state": sample["observation.state"], "robot_type": ROBOT}
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        obs[key] = sample[key].float() / 255
    text = prompt.replace("${task}", sample["task"])
    text = text.replace("${subtask}", language(sample, "subtask"))
    with torch.inference_mode():
        torch.manual_seed(seed)
        actions = post(policy.predict_action_chunk(pre(obs | {"task": text})))
    return actions.float().cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4, help="per scene and split")
    parser.add_argument("--per-episode", type=int, default=10)
    parser.add_argument(
        "--seed", type=int, default=0, help="added to every frame's seed"
    )
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    policy, pre, post = load(args.model, args.tokenizer, "cuda:0", False)
    prompt = action_prompt(args.model)
    roots = scene_roots(args.data)
    start = time.monotonic()
    rows = {
        k: [] for k in ("predicted", "recorded", "state", "split", "scene", "episode")
    }
    rows["frame"] = []
    for name in ("validation", "train"):
        split = Episodes(roots, name)
        offsets = np.cumsum([0] + [len(e["state"]) for e in split.episodes])
        for e in chosen(split, args.episodes):
            episode = split.episodes[e]
            last = len(episode["state"]) - CHUNK  # whole chunks only
            for frame in np.linspace(0, last, args.per_episode, dtype=int):
                sample = split[int(offsets[e] + frame)]
                seed = args.seed + len(rows["frame"])
                rows["predicted"].append(
                    predict(policy, pre, post, prompt, sample, seed)
                )
                rows["recorded"].append(sample["action"].numpy())
                rows["state"].append(sample["observation.state"].numpy())
                rows["split"].append(name)
                rows["scene"].append(episode["scene"])
                rows["episode"].append(episode["episode_index"])
                rows["frame"].append(int(frame))
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    np.savez_compressed(args.output, model=str(args.model), **arrays)
    error = np.abs(arrays["predicted"] - arrays["recorded"])[:, :APPLIED]
    for name in ("validation", "train"):
        mask = arrays["split"] == name
        print(
            json.dumps(
                {
                    "split": name,
                    "frames": int(mask.sum()),
                    "arm_mrad_mean": round(1000 * float(error[mask, :, :14].mean()), 2),
                    "gripper_mean": round(float(error[mask, :, 14:].mean()), 1),
                }
            ),
            flush=True,
        )
    print(f"{time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
