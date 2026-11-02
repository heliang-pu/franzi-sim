"""Tote carry: both arms take the tote by its two end walls, lift it off the
workbench and set it on the roller conveyor behind it.

The 300 × 200 × 100 mm blue tote (0.45 kg) stands on the workbench (top at
BENCH_Z, 0.90 m) at about x = 0.31 with its long side along y, so its end
walls, with their handle holes, face the two hands at y = ±0.145. Each hand
takes its side's wall top, the jaws closing across the wall and its outward
rim. The roller conveyor runs along y behind the tote on a 50 mm riser
(rollers at 1.00 m) with its end stop fitted at y = -0.25. With the waist
fixed the hands cannot cross the midline, so the tote keeps its centre near
y = 0. It is lifted LIFT_H (its base 30 mm over the rollers while it crosses
the conveyor's frame), carried over its spot on the conveyor while it turns
square to the rollers, and set down there.

The expert's hand is tilted TILT (25 degrees) forward from fingers-down, not
the 45-degree grasp the key points are checked with (sim.bench_check): at 45
degrees the wrist is 17 cm behind the grasp point, and with arm-trunk
collisions on (2026-10-25) the wrist pitch link sank 14 mm into the torso at
the grasp and 19 mm during the lift. 30 degrees kept it clear for the tote
within 8 mm of its nominal place, and 25 degrees for the wider layouts of
version 5: there the hand nearer the body comes 5 cm closer to the chest, which
cost the 30-degree hand up to 24 mm of torso clearance (yaw 8 degrees, y 30 mm
over) and IK at the corners. The steeper hand still reaches tote centres out
to x 0.65 at the place height (the spots go to 0.62).

The arms reach the walls on the elbow-out branch REACH_SEED leads the IK to.
The IK from the default posture finds an elbow-low branch, along which the
lift passed near a wrist singularity: joint steps up to 0.2 rad per 1 cm
waypoint, and for seed 1 no IK at all; from REACH_SEED every 1 cm step stays
under 0.06 rad.

The pads grip the wall's inner face and the rim's outer face (150 mm from
the tote's centre line). The inner face leans out 2 degrees (141.0 mm at the
top), so the hand is rolled by that draft and the inner pad lies flat on it:
square to the wall, the pad touched it only along its lower edge, and the tote
crept 1.7 degrees about that line while carried. GRIP_Y is the middle of the
two pads at the grasp point and GRIP_W their distance; on WALL_Y and WALL_W,
the inner pad stopped 1 mm short of the wall and only the outer pads gripped.

A seed draws (version 5, user request 2026-10-27: wider randomization):
- the tote's yaw on the bench, up to YAW either way, and its x anywhere the
  turned tote still fits between the bench's front edge and the riser (the
  bench in front of the riser is 230 mm deep, the tote 200 mm: at 8 degrees
  that leaves 1 cm), and its y within TOTE_Y;
- its spot on the conveyor, x in TARGET_X and y in TARGET_Y, away from where
  a floor edge would lie on a roller's crest (CREST_CLEAR); it goes down
  square to the rollers. Turned 0.5-3 degrees with an edge on a crest, the
  tote drew a sideways contact 1.4-7.6 mm deep that flung it (MuJoCo's convex
  collision between the thin floor and the 16-sided roller), at any spot along
  the rollers 50 mm apart; square to them it never did;
- one of the wordings.
Every draw is IK-checked along the expert's moves and redrawn until it plans.

Until 2026-10-25 this was carton_carry: a 250 mm wide carton, which the 95 mm
jaws cannot hold, so the hands had to squeeze it with their wrist housings.
"""

import numpy as np

from sim import prop_assets as pa
from sim import tabletop as tt
from sim.build_task import build as robot_scene
from sim.gripper_linkage import contact_drop
from sim.workcell import BENCH_FRONT_X, BENCH_Z, block, industrial_cell

NAME = "tote_carry"
INSTRUCTION = (
    "Carry the tote with both hands and set it on the roller conveyor behind it."
)
INSTRUCTIONS = (
    INSTRUCTION,
    "Lift the blue tote with both hands and put it on the roller conveyor.",
    "Pick up the tote by its end walls and place it on the conveyor behind it.",
    "Move the blue tote from the workbench onto the roller conveyor.",
)
TOTE = (0.31, 0.0)
WALL_Y = 0.1438  # the grip line: wall's inner face at 137.6 mm, rim's at 150 mm
WALL_W = 0.0124  # the wall and its outward rim, across the jaws
GRASP_Z = 0.090  # the grasp point over the tote's base, 10 mm under the rim
RISER = {"center": (0.655, 0.0), "size": (0.45, 0.52), "top": BENCH_Z + 0.05}
ROLLER_TOP = RISER["top"] + 0.05
# The rollers' axes lie at y = ROLLER_Y0 + ROLLER_PITCH k (k 0-9), centred on
# y = 0.01; the end stop at y = -0.25.
ROLLER_Y0, ROLLER_PITCH = -0.215, 0.05
# The nominal spot (bench_check's key points): its floor edges fall 15.6 mm
# from the nearest roller axes.
TARGET = (0.56, 0.01)
LIFT_H = 0.13  # the tote's base over the bench while carried
ALONG_Y = (np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4))  # the conveyor's flow
# The expert's grasp (see the module docstring).
TILT = np.radians(25)
DRAFT = np.radians(2)  # the tote's walls
GRIP_Y = 0.1452
GRIP_W = 0.0091
# 6 mm past contact: each finger servo presses 4.5 N, so the two pads of a
# hand hold 10.8 N against the 2.2 N of the tote each hand carries.
SQUEEZE = 0.006
APPROACH = 0.08  # the open hands come down this far onto the walls, and go
# back up along the hand this far after letting go
# The tote is let go this far over the rollers, re-measured (privileged): on
# PhysX a held part does not sag in the jaws as under MuJoCo's soft friction,
# and at the planned set-down (SET_GAP 3 mm) Isaac's tote still hung 2.2 mm
# over the rollers, MuJoCo's 0.6 mm.
SET_DOWN = 0.0005
# An IK seed on the elbow-out branch at the walls (wheel_robot_26.8.16_3; the
# right arm's is its mirror image: shoulder roll and yaw, wrist yaw and roll
# change sign).
MIRROR = np.array([1, -1, -1, 1, -1, 1, -1])
REACH_SEED = {
    "left": np.array([-1.9, 2.25, -1.3, 2.4, -2.1, 0.05, 1.3]),
    "right": np.array([-1.9, 2.25, -1.3, 2.4, -2.1, 0.05, 1.3]) * MIRROR,
}
# The draws (see the module docstring).
YAW = np.radians(8)  # the tote's yaw on the bench, either way
TOTE_Y = 0.03  # its centre's y, either way
# Its footprint on the bench: the floor's half sizes, and the walls' outer
# half sizes at the riser's top (they lean out 2 degrees).
FLOOR = (0.09065, 0.14065)
WALLS = (0.0923, 0.1423)
OVERHANG = 0.005  # a floor corner may hang this far past the bench's front
RISER_GAP = 0.004  # the walls stay this far off the riser
# On the rollers (from x 0.497; at 0.55 the floor's near edge hangs 38 mm over
# the frame, 10 mm above it); the hands reach tote centres to 0.65 there.
TARGET_X = (0.55, 0.62)
# Its y: the tote's ends 30 mm off the end stop and off the conveyor's far end.
TARGET_Y = (-0.06, 0.07)
CREST_CLEAR = 0.005  # a floor edge stays this far off a roller's axis
TURN_STEP = np.radians(0.5)  # the carry's waypoints: 1 cm or this apart


def poses(side):
    """Grasp-site targets for one hand: its end wall picked up at the tote's
    start and set down at the target on the rollers."""
    sign = 1 if side == "left" else -1
    return tt.pick_place(
        side,
        [TOTE[0], TOTE[1] + sign * WALL_Y, BENCH_Z],
        GRASP_Z,
        WALL_W,
        [TARGET[0], TARGET[1] + sign * WALL_Y, ROLLER_TOP],
        LIFT_H,
    )


KEY_POINTS = [
    {
        "name": f"{key}_{side}",
        "side": side,
        "xyz": [round(float(v), 4) for v in poses(side)[key]],
        "grasp": "forward",
        "width": WALL_W,
    }
    for side in tt.SIDES
    for key in ("grasp", "place")
]


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def hand(side, yaw=0.0):
    """The expert's hand for the tote at yaw: tilted TILT forward from
    fingers-down, its jaws closing across the end walls, rolled by the walls'
    draft so the pads lie flat on the inner faces (with TILT 45 degrees and no
    draft this is sim.tabletop.grasp_rotation(side, yaw))."""
    sign = 1 if side == "left" else -1
    roll = -np.pi / 2 if side == "left" else np.pi / 2
    return tt.rot_z(yaw) @ rot_x(-sign * DRAFT) @ tt.rot_y(-TILT) @ tt.rot_z(roll)


def grip(side, x, y, base_z, yaw):
    """One hand's grasp-site target and rotation for the tote whose base centre
    is at (x, y, base_z) with yaw. The hand stops contact_drop(GRIP_W) short
    along itself, so the dropping pads close on the grip line."""
    sign = 1 if side == "left" else -1
    rotation = hand(side, yaw)
    wall = np.array([x, y, base_z + GRASP_Z]) + tt.rot_z(yaw) @ [0, sign * GRIP_Y, 0]
    return wall + contact_drop(GRIP_W) * (rotation @ [0, 0, 1]), rotation


def key_poses(tote, target):
    """Per hand, the grasp-site targets (world, m) for carrying the tote whose
    base centre is at tote (x, y, z) with yaw tote[3] to target (x, y) on the
    rollers, set down square to them: rotation, pregrasp, grasp and lift at the
    tote's yaw, then place_rotation, over, place and retreat at yaw 0."""
    x, y, z, yaw = (float(v) for v in tote)
    out = {}
    for side in tt.SIDES:
        grasp, rotation = grip(side, x, y, z, yaw)
        over, square = grip(side, *target, BENCH_Z + LIFT_H, 0.0)
        place = grip(side, *target, ROLLER_TOP + tt.SET_GAP, 0.0)[0]
        out[side] = {
            "rotation": rotation,
            "pregrasp": grasp + [0, 0, APPROACH],
            "grasp": grasp,
            "lift": grip(side, x, y, BENCH_Z + LIFT_H, yaw)[0],
            "place_rotation": square,
            "over": over,
            "place": place,
            "retreat": place + APPROACH * (square @ [0, 0, 1]),
        }
    return out


def carry_targets(tote, target):
    """Per hand, the (grasp-site target, rotation) waypoints carrying the tote
    LIFT_H over the bench from over its start to over target while it turns
    from its yaw to 0: 1 cm or TURN_STEP apart, as many for both hands."""
    x, y, _, yaw = (float(v) for v in tote)
    shift = np.hypot(target[0] - x, target[1] - y)
    n = max(1, int(np.ceil(shift / 0.01)), int(np.ceil(abs(yaw) / TURN_STEP)))
    return {
        side: [
            grip(
                side,
                x + u * (target[0] - x),
                y + u * (target[1] - y),
                BENCH_Z + LIFT_H,
                yaw * (1 - u),
            )
            for u in np.linspace(0, 1, n + 1)[1:]
        ]
        for side in tt.SIDES
    }


def chain(kin, side, q, waypoints, body=None):
    """Joint waypoints from arm pose q through waypoints [(target, rotation)],
    each IK chained to the last, as Kinematics.line chains a straight move."""
    path = [np.asarray(q, float)]
    for target, rotation in waypoints:
        path.append(
            kin.ik(side, target, path[-1], body, near=path[-1], rotation=rotation)
        )
    return np.array(path)


def episode_tote(spec):
    return [*spec["tote"]["xy"], BENCH_Z, spec["tote"]["yaw"]]


def pregrasp_joints(kin, side, poses, body=None):
    """The arm over its end wall, on the elbow-out branch (REACH_SEED). The
    expert goes there from the default posture in joint space: on a straight
    line from READY the arm undid its turn into the hand on the way (shoulder
    and wrist yaw swung 0.8-1.4 rad back and forth) and the move stretched to
    11 s."""
    rotation = poses[side]["rotation"]
    return kin.ik(
        side, poses[side]["pregrasp"], REACH_SEED[side], body, rotation=rotation
    )


def plan(kin, spec):
    """Raises RuntimeError where a key pose, or a waypoint between them, has
    no IK solution inside the limits, from over the end walls to the retreat.
    The moves from the default posture and back to it run in joint space."""
    tote = episode_tote(spec)
    poses = key_poses(tote, spec["target"])
    carry = carry_targets(tote, spec["target"])
    for side in tt.SIDES:
        p = poses[side]
        q = pregrasp_joints(kin, side, poses)
        for key in ("grasp", "lift"):
            q = kin.line(side, q, p[key], rotation=p["rotation"])[-1]
        q = chain(kin, side, q, carry[side])[-1]
        for key in ("place", "retreat"):
            q = kin.line(side, q, p[key], rotation=p["place_rotation"])[-1]


def nominal():
    """The layout bench_check builds: no jitter, the first wording."""
    return {
        "tote": {"xy": list(TOTE), "yaw": 0.0},
        "target": list(TARGET),
        "instruction": INSTRUCTION,
    }


def x_range(yaw):
    """Where the tote's centre may stand along x at yaw: its floor on the
    bench (a corner at most OVERHANG past the front edge), its walls RISER_GAP
    off the riser."""
    c, s = np.cos(abs(yaw)), np.sin(abs(yaw))
    riser = RISER["center"][0] - RISER["size"][0] / 2
    return (
        BENCH_FRONT_X - OVERHANG + FLOOR[0] * c + FLOOR[1] * s,
        riser - RISER_GAP - (WALLS[0] * c + WALLS[1] * s),
    )


def clear_of_crests(y):
    """Whether the tote square to the rollers at y has both floor edges at
    least CREST_CLEAR off every roller's axis."""
    for edge in (y + FLOOR[1], y - FLOOR[1]):
        phase = (edge - ROLLER_Y0) % ROLLER_PITCH
        if min(phase, ROLLER_PITCH - phase) < CREST_CLEAR:
            return False
    return True


def draw(rng):
    yaw = rng.uniform(-YAW, YAW)
    x = rng.uniform(*x_range(yaw))
    y = rng.uniform(-TOTE_Y, TOTE_Y)
    target_y = rng.uniform(*TARGET_Y)
    while not clear_of_crests(target_y):
        target_y = rng.uniform(*TARGET_Y)
    return {
        "tote": {"xy": [float(x), float(y)], "yaw": float(yaw)},
        "target": [float(rng.uniform(*TARGET_X)), float(target_y)],
        "instruction": INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))],
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
            # 2: subtasks of both hands on the tote (labels only); 3: each arm
            # goes back on its own; 4: wheel_robot_26.8.16_3, the elbow-out
            # branch, the tote let go 0.5 mm over the rollers, re-measured; 5:
            # the wider layouts, the spot on the conveyor drawn, the tote set
            # down square to the rollers, the hand at 25 degrees
            "version": 5,
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
    block(root, "riser", RISER["center"], RISER["size"], BENCH_Z, RISER["top"])
    pa.add(
        root,
        "roller_conveyor",
        "outbound",
        pos=(*RISER["center"], RISER["top"]),
        quat=ALONG_Y,
        variant_name="stop_fitted",
    )
    half = spec["tote"]["yaw"] / 2
    pa.add(
        root,
        "small_tote",
        "tote",
        pos=(*spec["tote"]["xy"], BENCH_Z + 0.0005),
        quat=(np.cos(half), 0, 0, np.sin(half)),
        free=True,
        variant_name="blue",
    )
    report["task"] = {
        "name": NAME,
        "instruction": spec["instruction"],
        "key_points": KEY_POINTS,
    }
    return root, report
