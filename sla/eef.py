"""The robot's joint data as SLA's hand poses, and hand poses as joint targets.

A row of SLA's data holds both hands, [x, y, z, rx, ry, rz, gripper] each:
- the position of the grasp point in the frame of SLA's pre-training corpus: +x
  forward, +y left, +z up, the origin midway between the arm bases. The world
  frame has those axes, so only the origin differs;
- the rotation of the grasp frame as a rotation vector (fingers down with the jaws
  along x is zero);
- the gripper 0 closed .. 1 open, as its unified_gripper.

The model's state shows each rotation as the frame's first two columns instead
(model_state, 20-D). A rotation vector wraps at pi, and the data cross it: around
either hand's mean orientation the right hand turns up to 161 degrees, and the left
one in tote_transfer nearly a full turn, so a tiny turn there would flip the state
between the ends of its range. The actions keep SLA's world-frame rotation vectors,
far from the wrap.

A row of ours is the datasets' 16-D row: both arms' joints in SDK units, then the
gripper positions 0 closed .. 1000 open.

Actions are relative as in SLA's transforms: position minus the chunk's first
state, rotation R_action R_state^T, the gripper absolute.
"""

import threading

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from robots.franzi.kinematics import ARM_JOINTS, Kinematics

SIDES = ("left", "right")
OPEN = 1000.0  # the datasets' gripper position when fully open
DIM = 14  # SLA's row
STATE = 20  # the model's state: [x, y, z, two frame columns, gripper] per hand
JOINTS = 16  # ours


_threads = threading.local()


def hands():
    """This thread's Hands. SLA's mixed dataset reads a batch's samples in several
    threads of each loader process, and Pinocchio's data isn't thread-safe: with one
    Hands per process the threads overwrote each other's FK, which then returned
    frames that weren't rotations (the first 16-card runs died on one) and, more
    often, quietly wrong poses."""
    if not hasattr(_threads, "hands"):
        _threads.hands = Hands()
    return _threads.hands


class Hands:
    """One arm's FK and IK at a time: use one per thread (hands()), as Pinocchio's
    data is shared."""

    def __init__(self):
        self.kin = Kinematics()
        pin.framesForwardKinematics(self.kin.model, self.kin.data, self.kin.neutral)
        bases = [
            self.kin.data.oMi[
                self.kin.model.getJointId(f"{side}_{ARM_JOINTS[0]}_joint")
            ].translation
            for side in SIDES
        ]
        self.origin = np.mean(bases, axis=0) + self.kin.base

    def pose(self, row):
        """A 16-D row (or rows) as hand poses."""
        row = np.asarray(row, float)
        if row.ndim > 1:
            return np.stack([self.pose(r) for r in row])
        hands = []
        for i, side in enumerate(SIDES):
            q = self.kin.from_sdk(side, row[7 * i : 7 * i + 7])
            p, R = self.kin.fk(side, q)
            hands += [
                p - self.origin,
                Rotation.from_matrix(R).as_rotvec(),
                [row[14 + i] / OPEN],
            ]
        return np.concatenate(hands)

    def joints(self, poses, state):
        """16-D rows for hand poses in a chunk's order, each row's IK starting from
        the row before and the first from the state's joints. Returns the rows and
        each arm's largest position (m) and rotation (rad) error over the chunk."""
        poses = np.asarray(poses, float)
        state = np.asarray(state, float)
        rows = np.empty((len(poses), JOINTS))
        errors = {}
        for i, side in enumerate(SIDES):
            q = self.kin.from_sdk(side, state[7 * i : 7 * i + 7])
            worst = np.zeros(2)
            for t, pose in enumerate(poses[:, 7 * i : 7 * i + 7]):
                q, position, rotation = self.kin.follow(
                    side,
                    pose[:3] + self.origin,
                    Rotation.from_rotvec(pose[3:6]).as_matrix(),
                    q,
                )
                worst = np.maximum(worst, (position, rotation))
                rows[t, 7 * i : 7 * i + 7] = self.kin.to_sdk(side, q)
                rows[t, 14 + i] = np.clip(pose[6], 0, 1) * OPEN
            errors[side] = worst.tolist()
        return rows, errors


def relative(state, poses):
    """SLA's action for absolute hand poses [T, 14] from a state [14]: what its
    DeltaActions makes of them, without openpi."""
    state, poses = np.asarray(state, float), np.array(poses, float)
    for i in (0, 7):
        poses[:, i : i + 3] -= state[i : i + 3]
        now = Rotation.from_rotvec(state[i + 3 : i + 6])
        poses[:, i + 3 : i + 6] = (
            Rotation.from_rotvec(poses[:, i + 3 : i + 6]) * now.inv()
        ).as_rotvec()
    return poses


def model_state(state):
    """A 14-D hand row (or rows) as the model's 20-D state."""
    state = np.asarray(state, float)
    hands = []
    for i in (0, 7):
        frame = Rotation.from_rotvec(state[..., i + 3 : i + 6]).as_matrix()
        hands += [
            state[..., i : i + 3],
            frame[..., :, 0],
            frame[..., :, 1],
            state[..., i + 6 : i + 7],
        ]
    return np.concatenate(hands, axis=-1)


def rotvec_state(state):
    """The model's 20-D state (or states) back as a 14-D hand row, the two columns
    made orthonormal first."""
    state = np.asarray(state, float)
    hands = []
    for i in (0, 10):
        x = state[..., i + 3 : i + 6]
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        y = state[..., i + 6 : i + 9]
        y = y - np.sum(x * y, axis=-1, keepdims=True) * x
        y = y / np.linalg.norm(y, axis=-1, keepdims=True)
        frame = np.stack([x, y, np.cross(x, y)], axis=-1)
        hands += [
            state[..., i : i + 3],
            Rotation.from_matrix(frame.reshape(-1, 3, 3))
            .as_rotvec()
            .reshape(*state.shape[:-1], 3),
            state[..., i + 9 : i + 10],
        ]
    return np.concatenate(hands, axis=-1)
