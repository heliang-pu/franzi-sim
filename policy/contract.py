"""What the simulator and a policy exchange: XPolicyLab's standard observation
and action dictionaries, filled as the benchmark's datasets are. NumPy only, so
both sides import it: the simulator's Python and every policy's env.

An observation:
    instruction        the episode's instruction
    additional_info    frequency (control ticks per second), episode (its id,
                       which keys the policy's per-episode state when episodes
                       share a server), seed and tick (for seeded noise)
    vision/<camera>    color: uint8 [480, 640, 3] RGB, the tick's own frame
    state/<part>       the measured 16-D state in the datasets' SDK units

The cameras are XPolicyLab's names; the body D435, which XPolicyLab has no name
for, is cam_waist after the datasets' "waist". The parts are XPolicyLab's
dual-arm keys: each arm's 7 joints (SDK radians) and each gripper's position
(0 closed .. 1000 open). An action is the same parts as targets, one per tick;
a chunk is a list of them.
"""

import numpy as np

FREQUENCY = 25
CAMERAS = {
    "cam_head": "head",
    "cam_waist": "waist",
    "cam_left_wrist": "wrist_left",
    "cam_right_wrist": "wrist_right",
}
# The 16-D vector's parts in the datasets' order: the arms, then the grippers.
PARTS = (
    ("left_arm_joint_state", slice(0, 7)),
    ("right_arm_joint_state", slice(7, 14)),
    ("left_ee_joint_state", slice(14, 15)),
    ("right_ee_joint_state", slice(15, 16)),
)


def observation(instruction, state, frames, episode, seed, tick):
    """An observation from the 16-D state and the frames by dataset camera."""
    state = np.asarray(state, float)
    return {
        "instruction": instruction,
        "additional_info": {
            "frequency": FREQUENCY,
            "episode": episode,
            "seed": int(seed),
            "tick": int(tick),
        },
        "vision": {cam: {"color": frames[name]} for cam, name in CAMERAS.items()},
        "state": {part: state[where] for part, where in PARTS},
    }


def state_vector(obs):
    """The observation's 16-D state, float64."""
    return np.concatenate(
        [np.asarray(obs["state"][part], float).reshape(-1) for part, _ in PARTS]
    )


def frames(obs, cameras=None):
    """The observation's frames by dataset camera name."""
    names = {name: cam for cam, name in CAMERAS.items()}
    return {name: obs["vision"][names[name]]["color"] for name in (cameras or names)}


def info(obs, key):
    return obs["additional_info"][key]


def actions(chunk):
    """A chunk [n, 16] as XPolicyLab's action dictionaries."""
    chunk = np.asarray(chunk)
    if chunk.ndim != 2 or chunk.shape[1] != 16:
        raise ValueError(f"a chunk is [n, 16], not {list(chunk.shape)}")
    return [{part: row[where] for part, where in PARTS} for row in chunk]


def action_vector(action):
    """One action dictionary as the 16-D vector, float64."""
    return np.concatenate(
        [np.asarray(action[part], float).reshape(-1) for part, _ in PARTS]
    )
