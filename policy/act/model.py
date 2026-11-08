"""An ACT checkpoint (fineart.act_train) as a policy, in the fineart env:

    python -m policy.serve --config policy/act/deploy.yml --set checkpoint=<step folder> [--set ensemble=0.01 --set horizon=1]

ACT reads no language: the task is used only for the checkpoint's --variants
word (fineart.act_train), and the subtask logged and shown in the run's videos
is the task. Its chunk is deterministic (zero latent), so the seed is ignored.

ensemble is the paper's temporal ensemble with that coefficient (LeRobot's
ACTTemporalEnsembler, one per episode): each tick's action averages what every
chunk so far said about that tick, the oldest weighing most. It needs one query
per tick, so horizon 1, and refuses anything else. The chunk's first action is
then the ensembled one.
"""

import threading
import time
from pathlib import Path

import torch
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

from fineart.act_train import IMAGES, VARIANT, load, one_hot, prepare, tensors
from fineart.data import CAMERAS, CHUNK
from policy import contract
from policy.session import Sessions


class Model(Sessions):
    def __init__(self, config):
        super().__init__()
        self.checkpoint = Path(config["checkpoint"]).resolve()
        self.ensemble = config.get("ensemble")
        self.device = config.get("device", "cuda:0")
        self.policy, stats = load(self.checkpoint, self.device)
        self.policy.eval()
        self.scales = tensors(stats, self.device)
        self.variants = stats.get("variants")
        self.gpu = threading.Lock()
        self.requests = 0

    def describe(self):
        config = self.policy.config
        return super().describe() | {
            "policy": "ACT",
            "checkpoint": str(self.checkpoint),
            "chunk": config.chunk_size,
            "inputs": sorted(config.input_features),
            "variants": self.variants,
            "ensemble": self.ensemble,
            "requests": self.requests,
        }

    def start(self, meta):
        ensembler = None
        if self.ensemble is not None:
            ensembler = ACTTemporalEnsembler(self.ensemble, CHUNK)
        return {"ensembler": ensembler, "tick": None}

    def act(self, state, obs):
        tick, ensembler = contract.info(obs, "tick"), state["ensembler"]
        if ensembler and state["tick"] is not None and tick != state["tick"] + 1:
            raise ValueError("the temporal ensemble needs one query per tick")
        state["tick"] = tick
        vector = contract.state_vector(obs).astype("float32")
        raw = {"observation.state": torch.from_numpy(vector).unsqueeze(0)}
        frames = contract.frames(obs, CAMERAS)
        for camera, key in zip(CAMERAS, IMAGES, strict=True):
            frame = torch.from_numpy(frames[camera].copy())
            raw[key] = frame.permute(2, 0, 1).unsqueeze(0)
        if self.variants:
            raw[VARIANT] = one_hot(obs["instruction"], self.variants).unsqueeze(0)
        with self.gpu, torch.inference_mode():
            start = time.monotonic()
            batch = prepare(raw, self.scales, self.device)
            actions = self.policy.predict_action_chunk(batch)
            mean, std = self.scales["action"]
            actions = actions * std + mean
            if ensembler is not None:
                actions[:, 0] = ensembler.update(actions)
            actions = actions.float().cpu().numpy()[0]
            seconds = time.monotonic() - start
            self.requests += 1
        return actions, {"subtask": obs["instruction"], "seconds": seconds}
