"""Tote transfer: the left arm moves one part from the blue source tote into
the grey destination tote.

Both totes stand on the workbench at x = 0.37 with their long side along y.
The source tote spans y 0.12–0.42; the destination spans y −0.21–0.09, so the
left arm reaches both (top-down grasps reach x ≤ 0.38 at part height and y down
to about 0.0 at drop height). The aluminium flange and the plastic housing lie
loose on the source tote's floor; the instruction names the one to move.

The hand works top-down, its jaws closing along x. Its housing spans ±81 mm
along x from 40 to 80 mm above the fingertips, so the parts lie on the totes'
centre line (11 mm from each wall), and its wrist camera block reaches 93 mm
outboard (+y) about 90 mm above the closed fingertips, so the housing part
lies no further out than y 0.31 and parts are let go just above the grey
tote's floor, where that block passes over the rim.

A seed fixes which part moves, the parts' places and yaw, the spot in the grey
tote and the wording. Every draw is IK-checked along the expert's straight
moves and redrawn until it plans.
"""

import numpy as np

from sim import prop_assets as pa
from sim import tabletop as tt
from sim.build_task import TABLE_Z
from sim.build_task import build as robot_scene
from sim.gripper_linkage import contact_drop
from sim.workcell import industrial_cell

NAME = "tote_transfer"
SOURCE = (0.37, 0.27)
DEST = (0.37, -0.06)
FLOOR_Z = TABLE_Z + 0.004  # the tote's 4 mm floor
RIM_Z = TABLE_Z + 0.10
INNER = (0.0905, 0.1405)  # half the inner floor (draft walls narrow it)
# part: (prop, nominal xy, grip width along x, name in the instruction)
PARTS = {
    "flange": ("flange_part", (0.37, 0.19), 0.060, "aluminium flange"),
    "housing": ("abs_housing", (0.37, 0.31), 0.045, "plastic housing"),
}
DROP_XY = (0.37, 0.03)
# The closed pads' centre above a part's base: the fingertips reach 17 mm
# below it, 2 mm off the tote floor. It leaves the flange's 12 mm disc 8 mm of
# pad.
GRASP_Z = 0.019
# A part's base over the grey tote's floor when let go. Parts falling 10 mm
# sank 0.8 mm into the floor on landing (the audit stops at 1 mm); from 6 mm
# the wrist camera block still clears the grey tote's rim by 3 mm.
RELEASE = 0.006
CARRY_Z = 1.02  # grasp site height while over the rims
READY = tt.READY["left"]
JITTER = 0.005
DROP_JITTER = 0.01
YAW = np.radians(5)
INSTRUCTIONS = (
    "Move the {part} from the blue tote into the grey tote.",
    "Put the {part} into the grey tote.",
    "Take the {part} out of the blue bin and drop it into the grey bin.",
)
# Grasp-site positions (open jaws) for the scene check.
KEY_POINTS = [
    {
        "name": f"grasp_{name}",
        "side": "left",
        "xyz": [*xy, FLOOR_Z + GRASP_Z + contact_drop(width)],
        "grasp": "top",
    }
    for name, (_, xy, width, _) in PARTS.items()
] + [
    {
        "name": f"release_{name}",
        "side": "left",
        "xyz": [*DROP_XY, FLOOR_Z + RELEASE + GRASP_Z + contact_drop(width)],
        "grasp": "top",
    }
    for name, (_, _, width, _) in PARTS.items()
]


def hand(yaw):
    """Top-down hand turned yaw about z (the shared IK's default is yaw 0)."""
    return tt.rot_z(yaw)


def key_poses(base, yaw, width, drop):
    """Grasp-site targets (world, m) for moving the part whose base is at base
    (x, y, z) to drop (x, y) in the grey tote. The hand stops
    contact_drop(width) above the grasp point, so the dropping pads close
    there."""
    above = GRASP_Z + contact_drop(width)  # grasp site over the part's base
    grasp = np.asarray(base, float) + [0, 0, above]
    release = np.array([*drop, FLOOR_Z + RELEASE + above])
    return {
        "rotation": hand(yaw),
        "above": [grasp[0], grasp[1], CARRY_Z],
        "grasp": grasp,
        "lift": [grasp[0], grasp[1], CARRY_Z],
        "over": [release[0], release[1], CARRY_Z],
        "release": release,
        "clear": [release[0], release[1], CARRY_Z],
    }


def episode_poses(spec):
    part = spec["parts"][spec["target"]]
    return key_poses(
        [*part["xy"], FLOOR_Z], part["yaw"], PARTS[spec["target"]][2], spec["drop"]
    )


def plan(kin, spec):
    """Raises RuntimeError where a key pose, or a 1 cm waypoint between them,
    has no IK solution inside the limits."""
    poses = episode_poses(spec)
    q = kin.ik("left", READY, np.zeros(7))
    q = kin.line("left", q, poses["above"], rotation=poses["rotation"])[-1]
    for key in ("grasp", "lift", "over", "release", "clear"):
        q = kin.line("left", q, poses[key], rotation=poses["rotation"])[-1]
    kin.line("left", q, READY)


def nominal():
    """The layout bench_check builds: no jitter, the flange to move."""
    return {
        "target": "flange",
        "parts": {n: {"xy": list(p[1]), "yaw": 0.0} for n, p in PARTS.items()},
        "drop": list(DROP_XY),
        "instruction": INSTRUCTIONS[0].format(part=PARTS["flange"][3]),
    }


def draw(rng):
    target = str(rng.choice(list(PARTS)))
    parts = {
        n: {
            "xy": [float(v) for v in np.asarray(p[1]) + rng.uniform(-1, 1, 2) * JITTER],
            "yaw": float(rng.uniform(-1, 1) * YAW),
        }
        for n, p in PARTS.items()
    }
    drop = np.asarray(DROP_XY) + rng.uniform(-1, 1, 2) * DROP_JITTER
    wording = INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))]
    return {
        "target": target,
        "parts": parts,
        "drop": [float(v) for v in drop],
        "instruction": wording.format(part=PARTS[target][3]),
    }


def sample_episode(seed, kin=None, attempts=100):
    kin = kin or tt.kinematics(tt.SAMPLE_MARGIN)
    rng = np.random.default_rng(seed)
    for attempt in range(1, attempts + 1):
        spec = draw(rng)
        try:
            plan(kin, spec)
        except RuntimeError:
            continue
        return {
            "task": NAME,
            "version": 1,
            "seed": int(seed),
            "draws": attempt,
            **spec,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def build(spec=None):
    """The scene for spec (default: the nominal layout)."""
    spec = spec or nominal()
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    pa.add(
        root, "small_tote", "source_tote", pos=(*SOURCE, TABLE_Z), variant_name="blue"
    )
    pa.add(root, "small_tote", "dest_tote", pos=(*DEST, TABLE_Z), variant_name="grey")
    for name, (prop, _, _, _) in PARTS.items():
        part = spec["parts"][name]
        half = part["yaw"] / 2
        pa.add(
            root,
            prop,
            name,
            pos=(*part["xy"], FLOOR_Z + 0.0005),
            quat=(np.cos(half), 0, 0, np.sin(half)),
            free=True,
        )
    report["task"] = {
        "name": NAME,
        "instruction": spec["instruction"],
        "key_points": KEY_POINTS,
    }
    return root, report
