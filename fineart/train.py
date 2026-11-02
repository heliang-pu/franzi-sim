"""Fine-tune FineART-VLA on the benchmark's demonstrations.

The loss, knowledge insulation, the FAST action tokens and the subtask recipe
are the pinned upstream implementation; this script only feeds it the scenes'
episodes (fineart.data), schedules the learning rate (linear warmup, then a
cosine down to a tenth; --schedule-from starts it again on a resumed run),
validates on the held-out episodes and saves checkpoints. Each checkpoint
holds the weights and the processors; only the newest keeps the optimizer
state (about 26 GB), for --resume. --accumulate splits each step's batch into
micro-batches of --batch, for a GPU with less memory free. After
fineart-so101-sim's training/train.py. One GPU, or several under torchrun.

    python -m fineart.train --data <collection> --model <initialization> \
        --tokenizer <paligemma_tokenizer> --output <run folder> --steps 30000
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
from lerobot.utils.collate import lerobot_collate_fn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers.initialization import no_init_weights

from fineart.data import CAMERAS, Episodes, scene_roots

IMAGES = [f"observation.images.{camera}" for camera in CAMERAS]


def load(model, tokenizer, device, training, checkpointing=True, freeze_vision=False):
    """The policy and its processors from a checkpoint folder. freeze_vision
    keeps the vision tower (SigLIP) fixed."""
    config = PreTrainedConfig.from_pretrained(str(model))
    if freeze_vision:
        config.freeze_vision_encoder = True
    config.device = "cpu"
    config.dtype = torch.bfloat16
    config.compile_model = False
    config.use_compiled_text_ce = False
    config.use_compiled_vision = False
    config.gradient_checkpointing = training and checkpointing
    config.pretrained_path = str(model.resolve())
    config.text_tokenizer_name = str(tokenizer.resolve())
    config.push_to_hub = False
    with no_init_weights():
        policy = FineARTVLAPolicy.from_pretrained(
            str(model), config=config, strict=True
        )
    policy.to(device)
    config.device = device
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=str(model),
        preprocessor_overrides={
            "fineart_vla_text_tokenizer": {"tokenizer_name": str(tokenizer.resolve())},
            "action_tokenizer_processor": {
                "paligemma_tokenizer_name": str(tokenizer.resolve())
            },
            "device_processor": {"device": device},
        },
    )
    policy.train(training)
    return policy, pre, post


def to_device(raw, device):
    """Moves the uint8 frames to the GPU and scales them to [0, 1] there."""
    for key in IMAGES:
        raw[key] = raw[key].to(device, non_blocking=True).float().div_(255)
    return raw


def save(policy, optimizer, source, target, step, args):
    target.mkdir(parents=True, exist_ok=True)
    config = policy.config
    dtype = config.dtype
    config.dtype = "bfloat16"
    policy.save_pretrained(target)
    config.dtype = dtype
    for file in source.glob("policy_*processor*"):
        if file.is_file():
            shutil.copy2(file, target / file.name)
    for name in ("normalization_provenance.json", "processor_check.json"):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    shutil.copytree(
        source / "action_tokenizer", target / "action_tokenizer", dirs_exist_ok=True
    )
    for old in target.parent.glob("step_*/optimizer.pt"):
        old.unlink()
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        target / "optimizer.pt",
    )
    digest = hashlib.file_digest(
        (target / "model.safetensors").open("rb"), "sha256"
    ).hexdigest()
    report = {
        "step": step,
        "base_checkpoint": str(source),
        "model_sha256": digest,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    (target / "training_state.json").write_text(json.dumps(report, indent=2))


@torch.no_grad()
def validate(policy, pre, loader, rank):
    policy.eval()
    states = (
        torch.cuda.get_rng_state(),
        torch.get_rng_state(),
        random.getstate(),
        np.random.get_state(),
    )
    torch.manual_seed(19000 + rank)
    random.seed(19000 + rank)
    np.random.seed(19000 + rank)
    values = []
    for raw in loader:
        loss, _ = policy(pre(to_device(raw, policy.config.device)))
        values.append(float(loss))
    torch.cuda.set_rng_state(states[0])
    torch.set_rng_state(states[1])
    random.setstate(states[2])
    np.random.set_state(states[3])
    policy.train()
    total = torch.tensor([sum(values), len(values)], device=policy.config.device)
    if dist.is_initialized():
        dist.all_reduce(total)
    return float(total[0] / total[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True, help="folder of <scene>/lerobot_v21"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--accumulate",
        type=int,
        default=1,
        help="micro-batches of --batch per step (validation: one batch of their sum)",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument(
        "--schedule-from",
        type=int,
        default=0,
        help="step the warmup and the cosine count from",
    )
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument(
        "--resume", action="store_true", help="--model is a checkpoint of this run"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="no validation or checkpoints"
    )
    parser.add_argument(
        "--no-checkpointing", action="store_true", help="keep activations"
    )
    parser.add_argument(
        "--spike", type=float, default=1e3, help="log the top gradients above this norm"
    )
    parser.add_argument(
        "--freeze-vision", action="store_true", help="keep the vision tower fixed"
    )
    parser.add_argument(
        "--max-skips",
        type=int,
        default=50,
        help="non-finite steps skipped before stopping",
    )
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device(device))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    policy, pre, _ = load(
        args.model,
        args.tokenizer,
        device,
        True,
        not args.no_checkpointing,
        args.freeze_vision,
    )
    source = (
        Path(
            json.loads((args.model / "training_state.json").read_text())[
                "base_checkpoint"
            ]
        )
        if args.resume
        else args.model
    )
    roots = scene_roots(args.data)
    dataset = Episodes(roots, "train")
    validation = Episodes(roots, "validation")
    sampler = DistributedSampler(
        dataset, world, rank, seed=args.seed, shuffle=True, drop_last=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=lerobot_collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    picked = np.linspace(
        0, len(validation) - 1, args.validation_samples, dtype=int
    ).tolist()[rank::world]
    val_loader = DataLoader(
        Subset(validation, picked),
        batch_size=args.batch * args.accumulate,
        num_workers=4,
        collate_fn=lerobot_collate_fn,
    )
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        fused=True,
    )
    start = 0
    if args.resume:
        state = torch.load(
            args.model / "optimizer.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
    wrapped = (
        DistributedDataParallel(
            policy,
            device_ids=[local_rank],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
        if world > 1
        else policy
    )
    log_file = args.output / "metrics.jsonl"

    def log(value):
        if rank == 0:
            value["wall_time"] = time.time()
            with log_file.open("a") as handle:
                handle.write(json.dumps(value) + "\n")
            print(json.dumps(value), flush=True)

    log(
        {
            "event": "loaded",
            "datasets": roots,
            "train_frames": len(dataset),
            "train_episodes": len(dataset.episodes),
            "validation_frames": len(validation),
            "validation_episodes": len(validation.episodes),
            "world_size": world,
            "global_batch": args.batch * world * args.accumulate,
            "parameters": sum(p.numel() for p in policy.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in policy.parameters() if p.requires_grad
            ),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "steps_per_epoch": len(loader) // args.accumulate,
            "planned_steps": args.steps,
            "planned_epochs": args.steps * args.accumulate / len(loader),
            "start_step": start,
        }
    )
    if not args.smoke:
        log(
            {
                "event": "validation",
                "step": start,
                "loss": validate(policy, pre, val_loader, rank),
            }
        )
    epoch = start * args.accumulate // len(loader)
    sampler.set_epoch(epoch)
    iterator = iter(loader)

    def draw():
        nonlocal epoch, iterator
        try:
            return next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            return next(iterator)

    started = time.monotonic()
    skipped = 0
    for step in range(start + 1, args.steps + 1):
        local = step - args.schedule_from
        if local <= args.warmup:
            factor = local / max(args.warmup, 1)
        else:
            span = args.steps - args.schedule_from - args.warmup
            progress = (local - args.warmup) / max(span, 1)
            factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for micro in range(args.accumulate):
            batch = pre(to_device(draw(), device))
            if step == start + 1 and micro == 0:
                log(
                    {
                        "event": "processor_contract",
                        "text_target_tokens": int((batch["text_labels"] != -100).sum()),
                        "action_samples": int(batch["predict_actions"].sum()),
                        "tensor_keys": {
                            k: list(v.shape)
                            for k, v in batch.items()
                            if isinstance(v, torch.Tensor)
                        },
                    }
                )
            # DDP reduces the gradients on the last micro-batch only.
            last = micro == args.accumulate - 1
            with contextlib.nullcontext() if last or world == 1 else wrapped.no_sync():
                loss, values = wrapped(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}: {values}")
                (loss / args.accumulate).backward()
            for k, v in values.items():
                if isinstance(v, (int, float, torch.Tensor)):
                    v = float(v) / args.accumulate
                    metrics[k] = metrics.get(k, 0.0) + v
                else:
                    metrics[k] = v
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        grad = torch.nn.utils.get_total_norm(grads)
        finite = bool(torch.isfinite(grad))
        if not finite or grad > args.spike:
            # Which parameters carry it, before clipping.
            norms = sorted(
                (
                    (float(p.grad.detach().float().norm()), name)
                    for name, p in policy.named_parameters()
                    if p.grad is not None
                ),
                key=lambda x: -x[0] if math.isfinite(x[0]) else -math.inf,
            )
            bad = [name for value, name in norms if not math.isfinite(value)]
            log(
                {
                    "event": "grad_spike",
                    "step": step,
                    "grad_norm": float(grad),
                    "nonfinite_params": bad[:8],
                    "top": norms[:8],
                    **{k: v for k, v in metrics.items() if isinstance(v, float)},
                }
            )
        if not finite:
            skipped += 1
            if skipped > args.max_skips:
                raise RuntimeError(f"{skipped} steps skipped on non-finite gradients")
            optimizer.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grads_with_norm_(policy.parameters(), 1.0, grad)
        optimizer.step()
        if step <= start + 3 or step % 10 == 0:
            log(
                {
                    "event": "train",
                    "step": step,
                    "epoch": step * args.accumulate / len(loader),
                    **metrics,
                    "grad_norm": float(grad),
                    "skipped_steps": skipped,
                    "lr": args.lr * factor,
                    "elapsed_s": time.monotonic() - started,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
            )
        due = step % args.save_every == 0 or step == args.steps
        if not args.smoke and (step % args.validate_every == 0 or due):
            log(
                {
                    "event": "validation",
                    "step": step,
                    "loss": validate(policy, pre, val_loader, rank),
                }
            )
        if not args.smoke and due:
            if world > 1:
                dist.barrier()
            if rank == 0:
                save(
                    policy,
                    optimizer,
                    source,
                    args.output / f"step_{step:06d}",
                    step,
                    args,
                )
                log({"event": "checkpoint", "step": step})
            if world > 1:
                dist.barrier()
    log(
        {
            "event": "complete",
            "steps": args.steps,
            "smoke": args.smoke,
            "elapsed_s": time.monotonic() - started,
        }
    )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
