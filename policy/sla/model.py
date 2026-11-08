"""An SLA checkpoint (sla/, docs/sla.md) as a policy, in SLA's env:

    python -m policy.serve --config policy/sla/deploy.yml --set checkpoint=<checkpoint> --set assets=<assets>/<asset id> \\
        [--set scheduler=<scheduler checkpoint>] [--set zero_noise=true] [--set memory_frames=6]

The observation's 16-D joint state becomes SLA's hand poses and its chunk of
relative hand poses becomes 16-D joint targets through the shared kinematics
(sla.eef, one Hands per thread), each row from the one before, as sla.serve does.

The prompt: in a short scene the instruction; in a long scene the scheduler's
coarse step (sla.scheduler), one schedule per episode. Either is asked again
every subtask_every ticks and kept between. The episode's seed and the tick
seed the flow's noise (seed * 100000 + tick); zero_noise starts it at zero.

memory_frames > 1 serves a checkpoint trained with MEM (sla.mem): the model keeps
the cameras' frames of the ticks it is shown and stacks the history sla.mem.history
picks, the frames before the episode's start being its first. It asks to be
shown every observe_every ticks, memory_stride by default, which covers queries
on multiples of the stride (horizons 25 and 50); a query whose history it hasn't
seen fails, and observe_every 1 covers any horizon.
"""

import threading
import time
from pathlib import Path

import numpy as np

from fineart.protocol import SUBTASK_EVERY
from policy import contract
from policy.session import Sessions
from sla import mem
from sla.eef import hands


class Model(Sessions):
    def __init__(self, config):
        super().__init__()
        import jax  # SLA's env only
        from openpi.policies import policy_config
        from openpi.shared import normalize

        from sla.train import inference_config, model

        frames = int(config.get("memory_frames") or 1)
        stride = int(config.get("memory_stride") or mem.STRIDE)
        self.memory = (frames, stride) if frames > 1 else None
        if self.memory:
            mem.install(frames, stride)
            self.observe_every = int(config.get("observe_every") or stride)
        self.jax = jax
        self.checkpoint = Path(config["checkpoint"]).resolve()
        self.assets = Path(config["assets"]).resolve()
        self.subtask_every = config.get("subtask_every") or SUBTASK_EVERY
        self.zero_noise = bool(config.get("zero_noise"))
        self.config = model()
        self.policy = policy_config.create_trained_policy(
            inference_config(self.assets),
            self.checkpoint,
            norm_stats=normalize.load(self.assets),
        )
        self.scheduler = None
        if config.get("scheduler"):
            from sla.scheduler import load

            self.scheduler = load(Path(config["scheduler"]))
        self.gpu = threading.Lock()
        self.requests = 0

    def describe(self):
        return super().describe() | {
            "policy": "SLA",
            "checkpoint": str(self.checkpoint),
            "assets": str(self.assets),
            "scheduler": self.scheduler is not None,
            "zero_noise": self.zero_noise,
            "memory": None
            if self.memory is None
            else {
                "frames": self.memory[0],
                "stride": self.memory[1],
                "cameras": list(mem.CAMERAS),
            },
            "subtask_every": self.subtask_every,
            "requests": self.requests,
        }

    def start(self, meta):
        return {
            "seed": meta["seed"],
            "subtask": None,
            "asked": None,
            "schedule": None,
            "seen": {},
        }

    def see(self, state, obs):
        if self.memory is None:
            return
        tick = contract.info(obs, "tick")
        frames = contract.frames(obs, mem.CAMERAS)
        # Copies: a decoded frame is a view of its whole message.
        state["seen"][tick] = {c: np.array(f) for c, f in frames.items()}
        oldest = tick - (self.memory[0] - 1) * self.memory[1]
        for old in [t for t in state["seen"] if 0 < t < oldest]:
            del state["seen"][old]

    def act(self, state, obs):
        tick, task = contract.info(obs, "tick"), obs["instruction"]
        frames = contract.frames(obs)
        if state["asked"] is None or tick - state["asked"] >= self.subtask_every:
            state["subtask"], state["asked"] = None, tick
        subtask, generated = state["subtask"], False
        if not subtask:
            generated, subtask = True, task
            if self.scheduler and self.scheduler.handles(task):
                if state["schedule"] is None:
                    state["schedule"] = self.scheduler.episode(task)
                with self.gpu:
                    subtask = state["schedule"].step(frames)
        images = {c: frames[c] for c in mem.CAMERAS}
        if self.memory is not None:
            picks = mem.history(range(tick + 1), *self.memory)
            missing = [t for t in picks if t not in state["seen"]]
            if missing:
                raise ValueError(
                    f"no frames of ticks {missing} for the history: observe_every "
                    "must divide the query ticks"
                )
            images = {c: np.stack([state["seen"][t][c] for t in picks]) for c in images}
        # The state as the old path's float32 request carried it.
        vector = contract.state_vector(obs).astype(np.float32).astype(np.float64)
        noise = None
        if self.zero_noise:
            shape = (self.config.action_horizon, self.config.action_dim)
            noise = np.zeros(shape, np.float32)
        with self.gpu:
            start = time.monotonic()
            self.policy._rng = self.jax.random.key(state["seed"] * 100000 + tick)
            poses = self.policy.infer(
                {"images": images, "state": vector, "prompt": subtask}, noise=noise
            )["actions"]
            seconds = time.monotonic() - start
            self.requests += 1
        # Outside the lock, beside other episodes' inference (one Hands per thread).
        rows, errors = hands().joints(poses, vector)
        state["subtask"] = subtask
        info = {"subtask": subtask, "generated": generated, "seconds": seconds}
        return rows.astype(np.float32), info | {"ik_errors": errors}
