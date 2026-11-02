"""Write the FineART initialization for the benchmark.

The base checkpoint's weights stay untouched (linked, not copied). Its features
become the 16-D contract (both arms' joints, then both grippers' closure) and
the four policy cameras, and its normalization is fitted on the training split:
- state: MIN_MAX over the training states with a 2% margin (at least 0.01), so
  every state the policy is shown maps inside [-1, 1]; FineART writes the state
  into its prompt in 256 bins over that range, and a value below -1 becomes
  bin -1;
- action: QUANTILES (q01 and q99 to -1 and 1), as in the base checkpoint.
The other statistics (min, max, mean, std, q10, q50, q90) are stored too.

The base is lerobot/FineART-VLA-subtask-midtrained (user choice 2026-10-29):
FineART's bimanual mid-training checkpoint (151 tasks, 1718 h, an ALOHA-style
robot), not the single-arm super_poulain fine-tune the first run started from.
Its recipe, its FAST action tokenizer and its text length stay as they are:
- the recipe trains the subtask from the task (30%) and the actions on the
  subtask text alone (70%);
- its FAST tokenizer (scale 10) reconstructs these actions to an RMSE of 0.0069
  in normalized units (limit 0.1; refitted on them, 0.0069 too, fewer tokens),
  and the VLM was trained on its tokens;
- automatic FAST fitting is off, and a chunk over MAX_ACTION_TOKENS is cut
  rather than stopping the training.

It then builds the processors from the new folder and checks them on samples:
the states inside [-1, 1], the actions back to within 1e-5 after normalizing
and unnormalizing, and the action tokens (FAST) within the configured maximum.
After fineart-so101-sim's prepare_sim_init.py.

    python -m fineart.init --data <collection> --base <FineART-VLA-subtask-midtrained> \
        --tokenizer <paligemma_tokenizer> --output <new folder>
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from fineart.data import CAMERAS, Episodes, info, scene_roots

QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
STATE_MARGIN = 0.02  # of each state dimension's training range
STATE_MARGIN_MIN = 0.01  # rad, or closure
# FAST tokens of one 50-step chunk with the base's tokenizer: 32 on average, at
# most 58 in 256 training chunks. The base pads every chunk to 384.
MAX_ACTION_TOKENS = 128


def statistics(values):
    stats = {
        "min": values.min(0),
        "max": values.max(0),
        "mean": values.mean(0),
        "std": values.std(0),
        "count": np.array([len(values)]),
    }
    for q in QUANTILES:
        stats[f"q{round(q * 100):02d}"] = np.quantile(values, q, axis=0)
    return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def features(names):
    inputs = {"observation.state": {"type": "STATE", "shape": [len(names)]}}
    for camera in CAMERAS:
        inputs[f"observation.images.{camera}"] = {
            "type": "VISUAL",
            "shape": [3, 480, 640],
        }
    return inputs, {"action": {"type": "ACTION", "shape": [len(names)]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True, help="folder of <scene>/lerobot_v21"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    assert not target.exists(), f"{target} exists"
    roots = scene_roots(args.data)
    assert len(roots) == 10, roots
    names = info(roots[0])["features"]["action"]["names"]
    for root in roots:
        meta = info(root)["features"]
        assert meta["action"]["names"] == names == meta["observation.state"]["names"], (
            root
        )
    train = Episodes(roots, "train")
    stats = {
        "observation.state": statistics(
            np.concatenate([e["state"] for e in train.episodes])
        ),
        "action": statistics(np.concatenate([e["action"] for e in train.episodes])),
    }
    low, high = stats["observation.state"]["min"], stats["observation.state"]["max"]
    margin = np.maximum((high - low) * STATE_MARGIN, STATE_MARGIN_MIN).astype(
        np.float32
    )
    inputs, outputs = features(names)

    target.mkdir(parents=True)
    for path in args.base.iterdir():
        if path.name == "model.safetensors":
            (target / path.name).symlink_to(path.resolve())
        elif path.name.startswith("."):
            continue
        elif path.is_dir():
            shutil.copytree(path, target / path.name)
        else:
            shutil.copy2(path, target / path.name)
    config = json.loads((target / "config.json").read_text())
    config.update(
        input_features=inputs,
        output_features=outputs,
        action_feature_names=names,
        max_action_tokens=MAX_ACTION_TOKENS,
        auto_fit_fast_tokenizer=False,
        normalization_mapping={
            "VISUAL": "IDENTITY",
            "STATE": "MIN_MAX",
            "ACTION": "QUANTILES",
        },
        pretrained_path=str(target),
        repo_id=None,
    )
    (target / "config.json").write_text(json.dumps(config, indent=2))
    for prefix, registry in (
        ("pre", "normalizer_processor"),
        ("post", "unnormalizer_processor"),
    ):
        path = target / f"policy_{prefix}processor.json"
        processor = json.loads(path.read_text())
        for step in processor["steps"]:
            if step["registry_name"] == "relative_actions_processor":
                step["config"]["action_names"] = names
            if step["registry_name"] == "action_tokenizer_processor":
                step["config"]["max_action_tokens"] = MAX_ACTION_TOKENS
                step["config"]["allow_truncation"] = True
            if step["registry_name"] != registry:
                continue
            step["config"]["norm_map"].update(STATE="MIN_MAX", ACTION="QUANTILES")
            step["config"]["features"] = (
                {**inputs, **outputs} if prefix == "pre" else outputs
            )
            tensors = {
                f"{feature}.{name}": value
                for feature, values in stats.items()
                for name, value in values.items()
            }
            tensors["observation.state.min"] = low - margin
            tensors["observation.state.max"] = high + margin
            save_file(tensors, target / step["state_file"])
        path.write_text(json.dumps(processor, indent=2))
    base = hashlib.file_digest(
        (args.base / "model.safetensors").open("rb"), "sha256"
    ).hexdigest()
    provenance = {
        "base_checkpoint": str(args.base.resolve()),
        "base_model_sha256": base,
        "datasets": roots,
        "fit_split": "train",
        "train_episodes": len(train.episodes),
        "train_frames": len(train),
        "action_names": names,
        "cameras": list(CAMERAS),
        "state": f"MIN_MAX over the training states, margin {STATE_MARGIN} of the range (at least {STATE_MARGIN_MIN})",
        "action": "QUANTILES q01/q99 over the training actions, absolute joint targets and closures",
        "state_min": (low - margin).tolist(),
        "state_max": (high + margin).tolist(),
        "action_q01": stats["action"]["q01"].tolist(),
        "action_q99": stats["action"]["q99"].tolist(),
    }
    (target / "normalization_provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    print(
        json.dumps(check(target, args.tokenizer, train, low - margin, high + margin)),
        flush=True,
    )


def check(target, tokenizer, train, state_min, state_max, samples=64):
    """Build the processors from target and run them on samples spread over the
    training split; writes processor_check.json."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.collate import lerobot_collate_fn

    config = PreTrainedConfig.from_pretrained(str(target))
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=str(target),
        preprocessor_overrides={
            "fineart_vla_text_tokenizer": {"tokenizer_name": str(tokenizer.resolve())},
            "action_tokenizer_processor": {
                "paligemma_tokenizer_name": str(tokenizer.resolve())
            },
            "device_processor": {"device": "cpu"},
        },
    )
    indices = np.linspace(0, len(train) - 1, samples, dtype=int).tolist()
    raw = lerobot_collate_fn([train[i] for i in indices])
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        raw[key] = raw[key].float() / 255
    state, action = raw["observation.state"].clone(), raw["action"].clone()
    processed = pre(raw)
    low, high = torch.from_numpy(state_min), torch.from_numpy(state_max)
    expected = 2 * (state - low) / (high - low) - 1
    torch.testing.assert_close(processed["observation.state"], expected)
    assert bool((processed["observation.state"].abs() <= 1).all())
    back = post(processed["action"])
    torch.testing.assert_close(back, action, atol=1e-5, rtol=1e-5)
    tokens = {
        key: list(value.shape)
        for key, value in processed.items()
        if isinstance(value, torch.Tensor)
    }
    report = {
        "samples": len(indices),
        "state_inside_unit_range": True,
        "max_action_roundtrip_error": float((back - action).abs().max()),
        "tensors": tokens,
    }
    if "action.token_mask" in processed:
        report["action_tokens_per_sample_max"] = int(
            processed["action.token_mask"].sum(1).max()
        )
    if "text_labels" in processed:
        report["text_target_tokens"] = int((processed["text_labels"] != -100).sum())
    (target / "processor_check.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
