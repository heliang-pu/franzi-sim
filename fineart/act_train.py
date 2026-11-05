"""Train ACT on scenes' demonstrations: a model of another kind than FineART-VLA,
to tell a weak fine-tune from data or an evaluation nothing could succeed on
(user request 2026-11-03: "换个act什么的模型试试").

The model is LeRobot's ACT unchanged, with the paper's settings: a ResNet-18 on
each camera's full frame, a transformer, the CVAE (its latent is zero at
inference) and the L1 loss. It reads no language, so one model serves one scene,
and where a scene's instruction says what to do (tote_transfer names one of two
parts), --variants lists the words that tell the instructions apart and the
model gets a one-hot of the word as an extra encoder token. Without it the model
can't know the target and goes for either part.
Its chunk is fineart.data's 50 actions, the paper's 2 s at 25 Hz. The state and
the actions are normalized by their mean and std over the training episodes, the
frames by ImageNet's for the backbone. An arm that holds still through a scene
has a std of about zero, which would turn a tiny error of its measured state
into a huge input, hence STD_FLOOR.

A checkpoint holds the weights, the config and normalization.json; the newest
also the optimizer state, for --resume. --init starts from another run's
checkpoint with a new optimizer, keeping the weights that fit. A model that
learned a scene without --variants ignores a token added later (flipping it moved
the planned hand 0.1 mm: it had memorized each training layout's target), so
--init-prefix model.backbone takes over only the vision. The validation reports
the L1 on frames of the held-out and the training episodes, and the error of the
first 25 actions in fineart.actions's units.

    python -m fineart.act_train --data <collection> --scenes tote_transfer \
        --output <run folder> --steps 100000
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from torch.utils.data import DataLoader, Dataset, Subset

from fineart.data import CAMERAS, CHUNK, Episodes

IMAGES = [f"observation.images.{camera}" for camera in CAMERAS]
VARIANT = "observation.environment_state"
KEYS = [*IMAGES, "observation.state", "action", "action_is_pad"]
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
# The joints in radians, then the grippers in SDK units (0 closed .. 1000 open).
STD_FLOOR = np.r_[np.full(14, 0.01), np.full(2, 10.0)]
APPLIED = 25  # actions of a chunk isaac.policy applies before asking again


def one_hot(task, variants):
    """Which of the variants' words an instruction names."""
    hits = [word in task for word in variants]
    assert sum(hits) == 1, (task, variants)
    return torch.tensor(hits, dtype=torch.float32)


class Samples(Dataset):
    """Episodes' samples without the texts, which ACT doesn't read."""

    def __init__(self, episodes, variants):
        self.episodes = episodes
        self.variants = variants

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, i):
        sample = self.episodes[i]
        out = {key: sample[key] for key in KEYS}
        if self.variants:
            out[VARIANT] = one_hot(sample["task"], self.variants)
        return out


def fit(dataset):
    """Each vector's mean and floored std over the episodes."""
    stats = {}
    for name in ("state", "action"):
        values = np.concatenate([e[name] for e in dataset.episodes])
        stats[name] = {
            "mean": values.mean(0).tolist(),
            "std": np.maximum(values.std(0), STD_FLOOR).tolist(),
        }
    return stats


def tensors(stats, device):
    """(mean, std) of the state and of the action, and of the frames."""
    out = {
        name: tuple(
            torch.tensor(stats[name][k], dtype=torch.float32, device=device)
            for k in ("mean", "std")
        )
        for name in ("state", "action")
    }
    out["image"] = tuple(
        torch.tensor(v, dtype=torch.float32, device=device).view(3, 1, 1)
        for v in (IMAGE_MEAN, IMAGE_STD)
    )
    return out


def prepare(raw, scales, device):
    """A batch of uint8 frames and raw vectors, normalized on the GPU."""
    batch = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
    mean, std = scales["image"]
    for key in IMAGES:
        batch[key] = (batch[key].float().div_(255) - mean) / std
    mean, std = scales["state"]
    batch["observation.state"] = (batch["observation.state"] - mean) / std
    if "action" in batch:
        mean, std = scales["action"]
        batch["action"] = (batch["action"] - mean) / std
    return batch


def configure(device, variants):
    inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (16,))}
    for key in IMAGES:
        inputs[key] = PolicyFeature(FeatureType.VISUAL, (3, 480, 640))
    if variants:
        inputs[VARIANT] = PolicyFeature(FeatureType.ENV, (len(variants),))
    return ACTConfig(
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        input_features=inputs,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (16,))},
        device=device,
        push_to_hub=False,
    )


def load(model, device):
    """The policy and its normalization from a checkpoint folder."""
    policy = ACTPolicy.from_pretrained(str(model))
    policy.to(device)
    policy.config.device = device
    stats = json.loads((Path(model) / "normalization.json").read_text())
    return policy, stats


def warm_start(policy, source, device, prefix=""):
    """Copies a checkpoint's weights whose names start with prefix into a new
    policy: those of the same shape, and the existing rows of the encoder's token
    embeddings, which --variants lengthens by one. Returns the names left at their
    fresh values."""
    old = load(source, device)[0].state_dict()
    fresh = []
    with torch.no_grad():
        for name, value in policy.state_dict().items():
            if not name.startswith(prefix):
                fresh.append(name)
            elif name in old and old[name].shape == value.shape:
                value.copy_(old[name])
            elif name in old and old[name].shape[1:] == value.shape[1:]:
                value[: len(old[name])] = old[name]
                fresh.append(name)
            else:
                fresh.append(name)
    return fresh


def save(policy, optimizer, stats, target, step, args):
    target.mkdir(parents=True)
    policy.save_pretrained(target)
    (target / "normalization.json").write_text(json.dumps(stats, indent=2))
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict()}, target / "optimizer.pt"
    )
    report = {
        "step": step,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    (target / "training_state.json").write_text(json.dumps(report, indent=2))
    # As fineart.train: a run killed mid-save can still resume.
    for old in target.parent.glob("step_*/optimizer.pt"):
        if old.parent != target:
            old.unlink()


@torch.no_grad()
def validate(policy, loader, scales, device):
    """The chunk at zero latent against the recorded one: its L1 in normalized
    units, and the first APPLIED actions' mean errors in mrad and SDK units."""
    _, std = scales["action"]
    l1, arm, grip, count = 0.0, 0.0, 0.0, 0
    for raw in loader:
        batch = prepare(raw, scales, device)
        error = (policy.predict_action_chunk(batch) - batch["action"]).abs()
        valid = ~batch["action_is_pad"].unsqueeze(-1)
        l1 += float((error * valid).sum() / valid.sum() / 16)
        first = (error * std)[:, :APPLIED]
        valid = valid[:, :APPLIED]
        arm += float((first[..., :14] * valid).sum() / valid.sum() / 14)
        grip += float((first[..., 14:] * valid).sum() / valid.sum() / 2)
        count += 1
    policy.train()
    return {
        "l1": l1 / count,
        "arm_mrad_mean": 1000 * arm / count,
        "gripper_mean": grip / count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True, help="folder of <scene>/lerobot_v21"
    )
    parser.add_argument("--scenes", required=True, help="comma-separated")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--validate-every", type=int, default=2500)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument(
        "--variants",
        default="",
        help="comma-separated words the instructions differ by",
    )
    parser.add_argument("--init", type=Path, help="a checkpoint to start from")
    parser.add_argument(
        "--init-prefix", default="", help="only --init's weights with this name prefix"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="only the first N training episodes (an overfit test)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="from the newest checkpoint in --output"
    )
    args = parser.parse_args()
    device = "cuda:0"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    roots = [args.data / scene / "lerobot_v21" for scene in args.scenes.split(",")]
    dataset = Episodes(roots, "train")
    validation = Episodes(roots, "validation")
    if args.episodes:
        dataset.rows = dataset.rows[dataset.rows[:, 0] < args.episodes]
        dataset.episodes = dataset.episodes[: args.episodes]
    variants = [word for word in args.variants.split(",") if word]
    start = 0
    fresh = None
    newest = sorted(args.output.glob("step_*/optimizer.pt"))
    if args.resume and newest:
        state = torch.load(newest[-1], map_location="cpu", weights_only=False)
        policy, stats = load(newest[-1].parent, device)
        start = state["step"]
    else:
        assert not newest, f"{args.output} already holds checkpoints: --resume?"
        stats = fit(dataset) | {"variants": variants}
        policy = ACTPolicy(configure(device, variants)).to(device)
        if args.init:
            fresh = warm_start(policy, args.init, device, args.init_prefix)
            if not args.init_prefix:  # its outputs mean something on its scales only
                saved = json.loads((args.init / "normalization.json").read_text())
                stats = saved | {"variants": variants}
    policy.train()
    random.seed(args.seed + start)
    np.random.seed(args.seed + start)
    torch.manual_seed(args.seed + start)
    scales = tensors(stats, device)
    config = policy.config
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=config.optimizer_lr,
        weight_decay=config.optimizer_weight_decay,
        fused=True,
    )
    if start:
        optimizer.load_state_dict(state["optimizer"])
    loader = DataLoader(
        Samples(dataset, variants),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )

    def check(split):
        picked = np.linspace(0, len(split) - 1, args.validation_samples, dtype=int)
        return DataLoader(
            Subset(Samples(split, variants), picked.tolist()),
            batch_size=args.batch,
            num_workers=4,
        )

    checks = {"validation": check(validation), "train": check(dataset)}
    log_file = args.output / "metrics.jsonl"

    def log(value):
        value["wall_time"] = time.time()
        with log_file.open("a") as handle:
            handle.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    log(
        {
            "event": "loaded",
            "datasets": [str(root) for root in roots],
            "train_frames": len(dataset),
            "train_episodes": len(dataset.episodes),
            "validation_frames": len(validation),
            "validation_episodes": len(validation.episodes),
            "parameters": sum(p.numel() for p in policy.parameters()),
            "steps_per_epoch": len(loader),
            "planned_steps": args.steps,
            "planned_epochs": args.steps / len(loader),
            "start_step": start,
            "init": str(args.init) if args.init else None,
            "fresh_weights": fresh,
            "stats": stats,
        }
    )

    def batches():
        while True:
            yield from loader

    def evaluate(step):
        values = {k: validate(policy, v, scales, device) for k, v in checks.items()}
        log({"event": "validation", "step": step, **values})

    iterator = batches()
    started = time.monotonic()
    for step in range(start + 1, args.steps + 1):
        batch = prepare(next(iterator), scales, device)
        optimizer.zero_grad(set_to_none=True)
        loss, values = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}: {values}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        if step <= start + 3 or step % 100 == 0:
            log(
                {
                    "event": "train",
                    "step": step,
                    "epoch": step / len(loader),
                    "loss": float(loss),
                    **values,
                    "grad_norm": float(grad),
                    "elapsed_s": time.monotonic() - started,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
            )
        due = step % args.save_every == 0 or step == args.steps
        if step % args.validate_every == 0 or due:
            evaluate(step)
        if due:
            save(policy, optimizer, stats, args.output / f"step_{step:06d}", step, args)
            log({"event": "checkpoint", "step": step})
    log(
        {
            "event": "complete",
            "steps": args.steps,
            "elapsed_s": time.monotonic() - started,
        }
    )


if __name__ == "__main__":
    main()
