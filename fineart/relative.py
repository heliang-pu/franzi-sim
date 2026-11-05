"""Relative actions (user decision 2026-11-02), and the tool that gives a
checkpoint or initialization folder them.

The open-loop check (docs/fineart.md) found a chunk's first action about 15 mm
from where the hand is: the actions are absolute joint targets, and the prompt
gives the policy the state in 256 bins only. With relative actions the policy
predicts each arm joint's chunk less the state the chunk starts from, as
openpi's DeltaActions does. LeRobot's RelativeActionsProcessorStep subtracts the
measured state before normalizing, and the postprocessor's
AbsoluteActionsProcessorStep adds it back after unnormalizing. Every FineART
processor pair carries both steps, switched off. The grippers' closures stay
absolute: their names hold "gripper", the steps' exclude_joints.

The action statistics become those of the relative chunks, as LeRobot's
compute_relative_action_stats takes them: every CHUNK-step chunk of the training
split that stays inside its episode, less its first frame's state. Only every
STRIDE-th start is used here. The state statistics stay.

    python -m fineart.relative --data <collection> --model <checkpoint or initialization> \
        --tokenizer <paligemma_tokenizer> --output <new folder>

writes a copy of the folder as an initialization: the weights are hard-linked,
and there is no optimizer state or training_state.json, so fineart.train starts
a run from it without --resume. It then checks the processors on samples, the
source folder's beside the new ones: the actions back within 1e-5 after the
round trip, and the FAST tokens per chunk with their reconstruction error.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
from safetensors.numpy import load_file, save_file

from fineart.data import CAMERAS, CHUNK, Episodes, scene_roots
from fineart.init import statistics

EXCLUDE = ["gripper"]
STRIDE = 10  # chunk starts per episode for the statistics: every 10th frame
SKIPPED = ("optimizer.pt", "training_state.json")
REWRITTEN = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "normalization_provenance.json",
    "processor_check.json",
)
NORMALIZERS = ("normalizer_processor", "unnormalizer_processor")
SWITCHES = {
    "policy_preprocessor.json": "relative_actions_processor",
    "policy_postprocessor.json": "absolute_actions_processor",
}


def mask_for(names):
    """Which action dimensions become relative, as the processor step decides."""
    step = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=EXCLUDE, action_names=names
    )
    return np.array(step._build_mask(len(names)))


def relative_rows(split, mask, stride=STRIDE):
    """Every stride-th whole chunk of each episode less its first state, as rows."""
    rows = []
    for episode in split.episodes:
        state, action = episode["state"], episode["action"]
        starts = np.arange(0, len(action) - CHUNK + 1, stride)
        chunks = action[starts[:, None] + np.arange(CHUNK)]
        chunks[..., mask] -= state[starts][:, None][..., mask]
        rows.append(chunks.reshape(-1, action.shape[1]))
    return np.concatenate(rows)


def processors(model, tokenizer):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(str(model))
    return make_pre_post_processors(
        config,
        pretrained_path=str(model),
        preprocessor_overrides={
            "fineart_vla_text_tokenizer": {"tokenizer_name": str(tokenizer.resolve())},
            "action_tokenizer_processor": {
                "paligemma_tokenizer_name": str(tokenizer.resolve())
            },
            "device_processor": {"device": "cpu"},
        },
    )


def fast_report(pre, processed):
    """FAST code tokens per chunk and the reconstruction error (normalized units)."""
    step = next(
        s for s in pre.steps if type(s).__name__ == "ActionTokenizerProcessorStep"
    )
    actions = processed["action"].float().numpy()
    errors = []
    for chunk in actions:
        tokens = step.action_tokenizer(chunk[None])
        back = step.action_tokenizer.decode(
            tokens, time_horizon=chunk.shape[0], action_dim=chunk.shape[1]
        )
        errors.append(float(np.sqrt(np.mean((np.asarray(back)[0] - chunk) ** 2))))
    codes = processed["action.code_token_mask"].sum(1).float()
    return {
        "code_tokens_mean": round(float(codes.mean()), 1),
        "code_tokens_max": int(codes.max()),
        "tokens_max": int(processed["action.token_mask"].sum(1).max()),
        "reconstruction_rmse": round(float(np.sqrt(np.mean(np.square(errors)))), 5),
    }


def check(target, source, tokenizer, train, samples=64):
    """Build target's processors and source's, run both on the same samples spread
    over the training split; writes processor_check.json."""
    from lerobot.utils.collate import lerobot_collate_fn

    indices = np.linspace(0, len(train) - 1, samples, dtype=int).tolist()

    def batch():
        raw = lerobot_collate_fn([train[i] for i in indices])
        for camera in CAMERAS:
            key = f"observation.images.{camera}"
            raw[key] = raw[key].float() / 255
        return raw

    report = {"samples": len(indices)}
    for name, folder in (("relative", target), ("absolute", source)):
        pre, post = processors(folder, tokenizer)
        raw = batch()
        action = raw["action"].clone()
        processed = pre(raw)
        back = post(processed["action"])
        torch.testing.assert_close(back, action, atol=1e-5, rtol=1e-5)
        report[name] = {
            "max_action_roundtrip_error": float((back - action).abs().max()),
            "normalized_action_abs_max": float(processed["action"].abs().max()),
            **fast_report(pre, processed),
        }
    (target / "processor_check.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True, help="folder of <scene>/lerobot_v21"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    assert not target.exists(), f"{target} exists"
    config = json.loads((args.model / "config.json").read_text())
    assert not config.get("use_relative_actions"), f"{args.model} is relative already"
    names = config["action_feature_names"]
    mask = mask_for(names)
    assert mask.tolist() == [True] * 14 + [False] * 2, mask
    train = Episodes(scene_roots(args.data), "train")
    rows = relative_rows(train, mask)
    stats = statistics(rows)

    target.mkdir(parents=True)
    for path in args.model.iterdir():
        if path.name.startswith(".") or path.name in SKIPPED + REWRITTEN:
            continue
        if path.suffix == ".safetensors" and "normalizer" in path.name:
            continue  # rewritten below
        if path.is_dir():
            shutil.copytree(path, target / path.name)
        else:
            os.link(path.resolve(), target / path.name)
    config.update(use_relative_actions=True, relative_exclude_joints=EXCLUDE)
    (target / "config.json").write_text(json.dumps(config, indent=2))
    for filename, switch in SWITCHES.items():
        processor = json.loads((args.model / filename).read_text())
        found = 0
        for step in processor["steps"]:
            if step["registry_name"] == switch:
                step["config"]["enabled"] = True
                if "exclude_joints" in step["config"]:
                    assert step["config"]["action_names"] == names, filename
                    step["config"]["exclude_joints"] = EXCLUDE
                found += 1
            if step["registry_name"] in NORMALIZERS:
                tensors = load_file(args.model / step["state_file"])
                for key, value in stats.items():
                    assert f"action.{key}" in tensors, (filename, key)
                    tensors[f"action.{key}"] = value
                save_file(tensors, target / step["state_file"])
        assert found == 1, (filename, switch)
        (target / filename).write_text(json.dumps(processor, indent=2))
    provenance = json.loads((args.model / "normalization_provenance.json").read_text())
    provenance.update(
        action=(
            f"QUANTILES q01/q99 over relative chunks: every {STRIDE}th {CHUNK}-step"
            " chunk of the training episodes less its first state, the arm joints"
            " relative and the grippers' closures absolute"
        ),
        action_q01=stats["q01"].tolist(),
        action_q99=stats["q99"].tolist(),
        relative_from=str(args.model.resolve()),
        relative_rows=len(rows),
    )
    (target / "normalization_provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    report = check(target, args.model, args.tokenizer, train)
    print(json.dumps({"output": str(target), **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
