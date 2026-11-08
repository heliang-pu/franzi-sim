"""A recorded run's own commands served as a policy: the check of the evaluation
path, since a replay of an expert run on its own seed must succeed again.

    python -m policy.serve --config policy/replay/deploy.yml --set trajectory=outputs/<run>/trajectory.npz

The commands go through the datasets' SDK units and back, as a dataset's
actions do (the grippers rounded to 0 .. 1000). A query at tick t returns the
50 commands from t, the last repeated past the run's end. Runs in the project
env (the shared kinematics' SDK map).
"""

from pathlib import Path

import numpy as np

from eval_client.env import to_sdk
from policy import contract
from policy.session import Sessions
from robots.franzi.kinematics import Kinematics

CHUNK = 50


class Model(Sessions):
    def __init__(self, config):
        super().__init__()
        self.trajectory = Path(config["trajectory"]).resolve()
        kin = Kinematics()
        with np.load(self.trajectory) as data:
            self.actions = np.array([to_sdk(kin, a) for a in data["action"]])

    def describe(self):
        return super().describe() | {
            "policy": "replay",
            "trajectory": str(self.trajectory),
            "ticks": len(self.actions),
        }

    def act(self, state, obs):
        tick = contract.info(obs, "tick")
        ahead = np.minimum(np.arange(tick, tick + CHUNK), len(self.actions) - 1)
        return self.actions[ahead], {"subtask": "replay"}
