"""A FineART-VLA checkpoint as a policy, in the fineart env:

    python -m policy.serve --config policy/fineart/deploy.yml --set checkpoint=<step folder> --set tokenizer=<paligemma_tokenizer>

Each episode's subtask is generated again every subtask_every ticks (3 s) and
between those conditions the actions unchanged. A generation sees the previous
one's subtask as the memory (None at first), which a checkpoint with the memory
recipe reads (fineart.recipe); it runs select_message on the prompt the
checkpoint's recipe trains it on. The actions are then predicted on the prompt
the recipe's low-level stream trained: the subtask alone for the bimanual
base's recipe, "<task> Subtask: <subtask>" for super_poulain's. The episode's
seed and the tick seed the flow matching's noise (seed * 100000 + tick), so a
query's reply depends on the episode alone.
"""

import threading
import time
from pathlib import Path

import torch
from lerobot.utils.constants import MESSAGES_RENDERED

from fineart import recipe
from fineart.data import CAMERAS, ROBOT
from fineart.protocol import SUBTASK_EVERY
from fineart.serve import SUBTASK_TOKENS, action_prompt
from fineart.train import load
from policy import contract
from policy.session import Sessions


class Model(Sessions):
    def __init__(self, config):
        super().__init__()
        self.checkpoint = Path(config["checkpoint"]).resolve()
        self.subtask_every = config.get("subtask_every") or SUBTASK_EVERY
        device = config.get("device", "cuda:0")
        self.policy, self.pre, self.post = load(
            self.checkpoint, Path(config["tokenizer"]), device, False
        )
        self.policy.reset()
        self.prompt = action_prompt(self.checkpoint)
        self.recipe = recipe.load(self.checkpoint)
        self.gpu = threading.Lock()
        self.requests = 0

    def describe(self):
        config = self.policy.config
        return super().describe() | {
            "policy": "FineART-VLA",
            "checkpoint": str(self.checkpoint),
            "chunk": config.chunk_size,
            "inputs": sorted(config.input_features),
            "action_names": config.action_feature_names,
            "action_prompt": self.prompt,
            "memory": recipe.uses_memory(self.recipe),
            "subtask_every": self.subtask_every,
            "requests": self.requests,
        }

    def start(self, meta):
        return {"seed": meta["seed"], "subtask": None, "memory": None, "asked": None}

    def act(self, state, obs):
        tick, task = contract.info(obs, "tick"), obs["instruction"]
        if state["asked"] is None or tick - state["asked"] >= self.subtask_every:
            state["memory"], state["subtask"], state["asked"] = (
                state["subtask"],
                None,
                tick,
            )
        vector = torch.from_numpy(contract.state_vector(obs).astype("float32"))
        sample = {"observation.state": vector, "task": task, "robot_type": ROBOT}
        frames = contract.frames(obs, CAMERAS)
        for camera in CAMERAS:
            frame = torch.from_numpy(frames[camera].copy())
            sample[f"observation.images.{camera}"] = (
                frame.permute(2, 0, 1).float() / 255
            )
        subtask, generated = state["subtask"], False
        with self.gpu, torch.inference_mode():
            start = time.monotonic()
            torch.manual_seed(state["seed"] * 100000 + tick)
            if not subtask:
                messages = recipe.subtask_messages(self.recipe, task, state["memory"])
                subtask = self.policy.select_message(
                    self.pre(sample | {MESSAGES_RENDERED: messages}),
                    max_new_tokens=SUBTASK_TOKENS,
                    suppress_loc_tokens=True,
                ).strip()
                generated = True
            prompt = self.prompt.replace("${task}", task)
            batch = self.pre(sample | {"task": prompt.replace("${subtask}", subtask)})
            actions = self.post(self.policy.predict_action_chunk(batch))
            actions = actions.float().cpu().numpy()[0]
            seconds = time.monotonic() - start
            self.requests += 1
        state["subtask"] = subtask
        return actions, {
            "subtask": subtask,
            "generated": generated,
            "memory": state["memory"] if generated else None,
            "seconds": seconds,
        }
