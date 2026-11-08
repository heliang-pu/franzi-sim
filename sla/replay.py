"""What the end-effector route costs: an expert run's commands as hand poses, solved
back into joints the way sla.serve solves SLA's, for isaac.policy --replay.

    python -m sla.replay <run>/trajectory.npz --output <new_file>.npz

The hand paths stay the expert's; the elbow swivel is the IK's, which a joint-space
expert chose otherwise. A replay of the result that fails on a scene's success test,
or on the arm touching something, is a limit of any policy in hand poses there,
however well it learns.
"""

import argparse
from pathlib import Path

import numpy as np

from sla.eef import OPEN, Hands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hands = Hands()
    with np.load(args.trajectory) as data:
        action = data["action"].astype(np.float64)
    kin = hands.kin
    rows = np.c_[
        kin.to_sdk("left", action[:, :7]),
        kin.to_sdk("right", action[:, 7:14]),
        (1 - action[:, 14:]) * OPEN,
    ]
    solved, errors = hands.joints(hands.pose(rows), rows[0])
    out = np.c_[
        kin.from_sdk("left", solved[:, :7]),
        kin.from_sdk("right", solved[:, 7:14]),
        1 - solved[:, 14:] / OPEN,
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, action=out)
    print(
        "SLA_REPLAY",
        {
            "ticks": len(out),
            "largest_ik_error_m_rad": errors,
            "largest_joint_change_from_expert_rad": float(
                np.abs(out[:, :14] - action[:, :14]).max()
            ),
            "largest_joint_step_rad": float(np.abs(np.diff(out[:, :14], axis=0)).max()),
            "expert_largest_joint_step_rad": float(
                np.abs(np.diff(action[:, :14], axis=0)).max()
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
