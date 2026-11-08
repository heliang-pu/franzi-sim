"""Fine-tune SLA on the scenes' datasets, through its own scripts/train_align.py:

    python -m sla.train --data <isaac collection> --assets <assets>/<asset id> \\
        --base <SLA_base>/params --teacher <omnivggt> --output <new_dir> --exp <name>

Run it from a folder holding checkpoints/dinov2/, where SLA's teacher looks for its
patch embedding (the teacher's own weights then overwrite it). PYTHONPATH needs this
repository and SLA's checkout.

The settings are SLA's post-training (external_config/train_robotwin.py) unless a
flag says otherwise: its base weights, 25% of the samples without the state, the
flow and 3D alignment losses, EMA 0.999. A dataset's share of the samples follows
the square root of its frames: uniform over frames gave tray_push 3.2% and
cnc_tending 21.7% in FineART's runs, and equal shares would starve the long scenes'
many steps.

--memory-frames > 1 trains with MEM's video memory (sla.mem): each camera sees that
many frames, --memory-stride ticks apart. It adds no parameter, so --base can be a
memoryless checkpoint's params (the paper's weaker MEM-Posttrain-Only).

--resume restores the checkpoint into this run's shardings (resharded()), so a run
saved with one --fsdp can go on with another, DDP (--fsdp 1) included.
"""

import argparse
import json
from pathlib import Path

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

from sla import mem
from sla.policy import FranziDataConfig

NAME = "franzi"
ALIGN = {
    "align_enabled": True,
    "align_target_model": "auto",
    "vla_layers_align": 12,
    "vla_adapt_layers_align": False,
    "vggt_layers_align": -1,
    "vggt_adapt_layers_align": False,
    "pooling_func": "bilinear",
    "use_vggt_pe": True,
    "use_vlm_norm": True,
    "align_loss_coeff": 0.5,
    "ignore_img_padding_area": True,
    "use_camera_params": False,
    "align_device_mode": "shared",
}


def model():
    return pi0_config.Pi0Config(pi05=True)


def asset_config(assets):
    assets = Path(assets).resolve()
    return _config.AssetsConfig(assets_dir=str(assets.parent), asset_id=assets.name)


def inference_config(assets):
    """What sla.serve needs of a config: the model and the transforms."""
    return _config.TrainConfig(
        name=NAME,
        exp_name="serve",
        model=model(),
        data=FranziDataConfig(
            repo_id="franzi", assets=asset_config(assets), use_quantile_norm=True
        ),
    )


def train_config(args):
    assets = args.assets.resolve()
    frames = json.loads((assets / "frames.json").read_text())
    if args.scenes:
        frames = {scene: frames[scene] for scene in args.scenes.split(",")}
    sources = tuple(
        _config.MixedDataSourceConfig(
            name=scene,
            # Sampled in proportion to frames times weight.
            weight=count**-0.5,
            data_factory=FranziDataConfig(
                repo_id=str((args.data / scene / "lerobot_v21").resolve()),
                scene=scene,
                labels=str(assets / "labels"),
                use_quantile_norm=True,
            ),
        )
        for scene, count in frames.items()
    )
    return _config.TrainConfig(
        name=NAME,
        exp_name=args.exp,
        model=model(),
        data=_config.MixedLeRobotDataConfig(
            assets=asset_config(assets),
            use_quantile_norm=True,
            fast_local_loading=True,
            sources=sources,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.base)),
        vggt_weight_path=str(args.teacher),
        align_vggt_devices=args.teacher_devices,
        align_feature_batch_size=args.teacher_batch,
        state_dropout_prob=0.25,
        freeze_filter=model().get_freeze_filter(),
        ema_decay=0.999,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup,
            peak_lr=args.lr,
            decay_steps=args.steps,
            decay_lr=args.lr / 10,
        ),
        num_train_steps=args.steps,
        save_interval=args.save_every,
        keep_period=args.keep_every,
        log_interval=args.log_every,
        batch_size=args.batch,
        gradient_accumulation_steps=args.accumulate,
        num_workers=args.workers,
        fsdp_devices=args.fsdp,
        checkpoint_base_dir=str(args.output),
        wandb_enabled=False,
        resume=args.resume,
        seed=args.seed,
        **ALIGN,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="<SLA_base>/params")
    parser.add_argument("--teacher", type=Path, required=True, help="OmniVGGT folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--scenes", help="comma-separated, default all of frames.json")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fsdp", type=int, default=1, help="devices a model shards to")
    parser.add_argument("--teacher-devices", type=int, default=1)
    parser.add_argument("--teacher-batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument(
        "--keep-every", type=int, default=5000, help="older checkpoints kept"
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--memory-frames", type=int, default=1, help="MEM's frames (sla.mem); 1 is off"
    )
    parser.add_argument("--memory-stride", type=int, default=mem.STRIDE, help="ticks")
    args = parser.parse_args()
    if args.memory_frames > 1:
        mem.install(args.memory_frames, args.memory_stride)
    # Imported here: it starts by setting JAX's allocator through the environment.
    from scripts import train_align

    resharded(train_align)
    train_align.main(train_config(args))


def resharded(train_align):
    """SLA's resume, restoring into the run's own shardings. Its restore targets are
    bare shapes and orbax, given no restore arguments, keeps the sharding each array
    was saved with: a run saved with FSDP 8 failed under DDP at its first step (pjit's
    sharding mismatch). The targets now carry the run's shardings, and orbax gets
    restore arguments built from them."""
    import jax
    import orbax.checkpoint as ocp
    from openpi.shared import array_typing as at

    checkpoints = train_align._checkpoints
    init = train_align.init_train_state

    def init_train_state(config, init_rng, mesh, *, resume):
        state, shardings = init(config, init_rng, mesh, resume=resume)
        if resume:
            state = jax.tree.map(
                lambda s, sharding: jax.ShapeDtypeStruct(
                    s.shape, s.dtype, sharding=sharding, weak_type=s.weak_type
                ),
                state,
                shardings,
            )
        return state, shardings

    def restore_state(checkpoint_manager, state, data_loader, step=None):
        del data_loader
        with at.disable_typechecking():
            train_state, params = checkpoints._split_params(state)
            items = {"train_state": train_state, "params": {"params": params}}
            restored = checkpoint_manager.restore(
                step,
                items=items,
                restore_kwargs={
                    name: {
                        "restore_args": ocp.checkpoint_utils.construct_restore_args(
                            item
                        )
                    }
                    for name, item in items.items()
                },
            )
        return checkpoints._merge_params(restored["train_state"], restored["params"])

    train_align.init_train_state = init_train_state
    checkpoints.restore_state = restore_state


if __name__ == "__main__":
    main()
