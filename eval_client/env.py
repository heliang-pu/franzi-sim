"""XPolicyLab's TASK_ENV over a running scene episode, on MuJoCo or Isaac Sim:
the simulator's side of the policy interface (policy/contract.py).

At each control tick (25 Hz) the policy is shown the four policy cameras'
frames of that tick, the very frames the run's videos record (one render per
tick, Frames), and the measured 16-D state in the datasets' SDK units (joints,
then gripper positions 0 closed .. 1000 open, rounded as convert_expert_run
rounds them). take_action applies one action as the arms' joint targets and the
grippers' closures and runs the tick's 20 physics steps. The episode ends when
the scene's complete() has held for 15 ticks, or at the time limit.
"""

from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml

from policy import contract

ROOT = Path(__file__).resolve().parents[1]
# Each dataset camera's simulated camera.
STREAMS = yaml.safe_load((ROOT / "robots/franzi/config/franzi_sdk.yaml").read_text())[
    "cameras"
]["streams"]
SIDES = ("left", "right")


def to_sdk(kin, state):
    """A 16-D state (URDF joints, closures) in the datasets' SDK units: the
    grippers as simbridge.robot.GripperMap.position_from_closure."""
    grips = [float(np.clip(round(1000 * (1 - c)), 0, 1000)) for c in state[14:]]
    return np.r_[kin.to_sdk("left", state[:7]), kin.to_sdk("right", state[7:14]), grips]


def apply(s, action):
    """Sets one tick's targets from a 16-D action in SDK units."""
    action = np.asarray(action, float)
    if action.shape != (16,) or not np.isfinite(action).all():
        raise RuntimeError(f"invalid policy action {action.tolist()}")
    s.goal["left"] = s.kin.from_sdk("left", action[:7])
    s.goal["right"] = s.kin.from_sdk("right", action[7:14])
    for side, position in zip(SIDES, action[14:], strict=True):
        s.grip[side] = float(np.clip(1 - position / 1000, 0, 1))


class Frames:
    """One render per tick: the policy's observation and the recording share
    the frames captured at a physics step."""

    captured_at = None

    def capture(self):
        if self.captured_at != self.step_number:
            self.captured = super().capture()
            self.captured_at = self.step_number
        return self.captured


class TaskEnv:
    def __init__(self, s, scene, seed, limit, config, policy_info):
        self.s = s
        self.scene = scene
        self.seed = seed
        self.limit = limit
        self.config = config
        self.policy_info = policy_info
        self.episode = f"{scene}/{seed}/{uuid4().hex[:8]}"
        self.tick = 0
        self.calls = []

    def meta(self):
        return {
            "episode": self.episode,
            "seed": self.seed,
            "scene": self.scene,
            "instruction": self.s.spec["instruction"],
        }

    def get_obs(self):
        s = self.s
        frames = s.capture()
        return contract.observation(
            s.spec["instruction"],
            to_sdk(s.kin, s.state()),
            {name: frames[STREAMS[name]["sim"]] for name in contract.CAMERAS.values()},
            self.episode,
            self.seed,
            self.tick,
        )

    def take_action(self, action):
        apply(self.s, contract.action_vector(action))
        for _ in range(20):
            self.s.physics_step()
        self.tick += 1

    def is_episode_end(self):
        return self.s.stable_frames >= 15 or self.s.clock() >= self.limit

    def note(self, reply):
        """Logs a query and shows its subtask in the run's videos."""
        info = reply.get("info") or {}
        self.calls.append({"tick": self.tick, **info})
        self.s.instruction = info.get("subtask")
