"""Tray push: the left arm pushes the parts tray along its rails until it rests
against the stop, without spilling the parts.

The rail set (two rails, their cross bars and the stop) stands on the workbench
(top at BENCH_Z, 0.90 m), nominally centred at x = 0.37, y = 0 with the rails
along y; a seed shifts it by -3..+3 cm in x and -2..+6 cm in y and turns it by
-8..+5 degrees, kept on the bench. Everything below is in the rails' frame (x
across them, away from the robot; y along them). The tray (200 mm deep,
between rails 206 mm apart) slides on the bench top; its collision geoms get
contact priority so that its own 0.25 friction governs the sliding. It starts
centred at y = 0.0-0.135, its +y end clear of the cross bar at y = +0.30, and
stops at y = −0.125 against the block at the −y end. The flange and the plastic
housing lie loose anywhere on its floor.

The left hand pushes the tray's +y end wall with its closed fingers (closure
0.95), straight down with the jaws across the rails (along x at yaw 0, as in
the default posture): the fingers' -y faces, 10.5 mm ahead of the grasp site,
press flat on the wall's outer face, and the push goes across the finger
slides. The push point lies 55-75 mm beyond the tray's centre line, away from
the robot, so the fingers (±14.4 mm along x) stay 6-26 mm clear of the wall's
70 mm finger hole and 13 mm inside the far rail. On the robot model of
2026-10-25 (URDF wheel_robot_26.8.16_3) the hand straight down no longer
reaches the push's end (y 0.035, 60 mm over the bench) on the robot's side of
the hole, x 0.31; at x 0.40-0.46 it does. The grasp site is 56 mm above the
bench, which puts the finger hulls' tips 27 mm above it (7 mm over the rails'
cross bar at the start) and the linkage that reaches 13 mm past the fingers
over the wall 11 mm above the wall top. The push ends EARLY short of the stop,
and a last slow move of the measured remaining gap seats the tray: on Isaac,
which has no contact priority, the tray slides with the bench's 0.6. The way
back runs straight HOMEWARD of the way to READY, then in joint space to the
start pose's joints. The gripper geometry this rests on is in FINGER_FACE,
FINGER_HALF and FINGERTIP.

A seed fixes the rail set's pose, the tray's start, where the two parts lie in
it and their yaw, the push point and the wording. Every draw is IK-checked
along the expert's moves and redrawn until it plans (92% of seeds 0-39 on the
first draw).
"""

import numpy as np

from robots.franzi.gripper_linkage import contact_drop
from robots.franzi.scene import build as robot_scene
from sim import prop_assets as pa
from sim import tabletop as tt
from sim.workcell import BENCH_FRONT_X, BENCH_Z, industrial_cell

NAME = "tray_push"
# The rail set's nominal pose on the bench: its centre (x, y) and yaw. The
# tray, the stop and the push live in the rails' frame: x across the rails (+x
# away from the robot), y along them (+y toward the tray's start).
RAILS = (0.37, 0.0, 0.0)
# A seed's shift of the rail set across the bench (x), along it (y) and its
# turn about z. The hand's reach bounds them: it must get to the push's end
# near the stop, low by the robot's midline, with the jaws across the rails.
# Turned counterclockwise the wrist roll runs out (+10 degrees planned nowhere,
# +5 only with the rails 3 cm toward +y); clockwise it reaches to -15 degrees
# with the rails 2-4 cm toward the robot, but turned that far the rails' -y
# end leaves the bench unless they move away from it, where the hand no longer
# reaches: the turn and the shift are drawn together, on the bench (on_bench).
RAILS_DX = (-0.03, 0.03)
RAILS_DY = (-0.02, 0.06)
RAILS_YAW = np.radians((-8.0, 5.0))
RAILS_HALF = (0.123, 0.35)  # half the rail set's footprint (tray_rail_set)
TRAY = (0.0, 0.12)  # nominal start of the tray's base centre, rails' frame
# A seed's start of the tray's centre along the rails: its +y end stays 5 mm
# off the rails' cross bar (y 0.29-0.31) at the top.
START_Y = (0.0, 0.135)
TRAY_STOP_Y = -0.125  # the tray's centre against the stop's bumper (y -0.275)
TRAY_HALF = (0.100, 0.150)  # outside, along x and y
INNER = (0.0965, 0.1465)  # inside the walls' colliders, from the centre
FLOOR = 0.003  # the floor's top above the tray's base
TRAY_FLOOR_Z = BENCH_Z + FLOOR
STOP_TOLERANCE = 0.002  # the tray's centre this close to TRAY_STOP_Y
# part: (prop, nominal place in the tray (x, y from its centre), footprint
# radius, name). A seed puts each part anywhere on the tray's floor, PART_GAP
# off the walls and the other part.
PARTS = {
    "flange": ("flange_part", (-0.025, -0.05), 0.030, "aluminium flange"),
    "housing": ("abs_housing", (0.02, 0.06), 0.0424, "plastic housing"),
}
PART_GAP = 0.01  # least room between a part's footprint and a wall or the other
READY = tt.READY["left"]
HOME_GRIP = 1 - tt.HOME_OPENING / tt.JAW_SPAN
# The push, the hand straight down (the IK's default rotation, jaws along x).
CLOSED = 0.95  # the pushing fingers' closure
# Gripper geometry no shared helper gives, measured on the finger meshes at
# CLOSED with the hand straight down; re-measure it for another gripper:
FINGER_FACE = 0.0105  # the fingers' faces across the jaws from the grasp site
FINGER_HALF = 0.0144  # the fingers' half extent along the jaws
# The closed fingertips below the grasp site, as tt.press_poses takes them
# (29.4 mm, where the finger hulls reach).
FINGERTIP = tt.TIP + contact_drop(tt.JAW_SPAN * (1 - CLOSED))
CROSS_BAR_TOP = 0.020  # the rails' cross bars over the bench (tray_rail_set)
FINGER_HOLE = 0.035  # half the width of the tray's end-wall finger hole (esd_tray)
# Grasp site over the bench while pushing: the fingertips 7 mm over the cross
# bar, so the linkage reaching past the fingers passes 11 mm over the wall top.
PUSH_HEIGHT = CROSS_BAR_TOP + 0.007 + FINGERTIP
# Push point beyond the tray's centre line (away from the robot): the fingers
# 6-26 mm clear of the finger hole (filled in the tray's collider) and at least
# 13 mm inside the far rail (x 0.103).
PUSH_DX = tuple(FINGER_HOLE + FINGER_HALF + c for c in (0.006, 0.026))
GAP = 0.006  # the fingers come down this far behind the wall
EARLY = 0.012  # the push ends this short of the stop; the seat goes the rest
PUSH_SECONDS = 4.0  # the push takes at least this (move() stretches it)
PRELOAD = 0.001  # the seat's goal past the stop
RISE = 0.06  # over the wall before lowering
BACK = 0.03  # drawn back off the wall before rising
CLEAR = 0.08
# The way back runs straight this far toward READY, then in joint space to the
# start pose's joints: straight on to READY the arm turned up to 0.24 rad per
# 1 cm near READY, and the rate limits stretched that move to 39 s.
HOMEWARD = 0.65
# A larger joint step between the 1 cm waypoints of a straight move is a jump
# to another IK branch (2.3 rad seen with the jaws turned along y): redrawn.
MAX_STEP = 0.2
INSTRUCTIONS = (
    "Push the parts tray along the rails until it touches the end stop.",
    "Slide the tray down its rails to the end stop without spilling the parts.",
    "Push the ESD tray against the stop at the end of the rails.",
    "Move the parts tray along the rails until it rests against the stop.",
)
INSTRUCTION = INSTRUCTIONS[0]


def to_world(rails, x, y, z=0.0):
    """A point of the rails' frame (x across, y along, z) in the world."""
    rx, ry, yaw = rails
    c, s = np.cos(yaw), np.sin(yaw)
    return [float(rx + c * x - s * y), float(ry + s * x + c * y), float(z)]


def to_rails(rails, point):
    """A world point's (x, y) in the rails' frame."""
    rx, ry, yaw = rails
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = point[0] - rx, point[1] - ry
    return float(c * dx + s * dy), float(-s * dx + c * dy)


def on_bench(rails):
    """The rail set's footprint lies on the bench, 5 mm in from its edge."""
    corners = [
        (sx * RAILS_HALF[0], sy * RAILS_HALF[1]) for sx in (-1, 1) for sy in (-1, 1)
    ]
    return min(to_world(rails, x, y)[0] for x, y in corners) >= BENCH_FRONT_X + 0.005


def along(rails, d):
    """A world offset of d (m) along the rails (+y of their frame)."""
    return np.array([-np.sin(rails[2]) * d, np.cos(rails[2]) * d, 0.0])


def hand(rails):
    """The pushing hand: straight down, its jaws across the rails."""
    return tt.rot_z(rails[2])


def key_poses(rails, tray_y, push_dx):
    """Grasp-site targets (world, m) for pushing the tray, whose base centre is
    tray_y along the rails, to the stop at push_dx beyond its centre line.
    "seat" is where the stop holds the wall, PRELOAD further."""
    z = BENCH_Z + PUSH_HEIGHT
    start = tray_y + TRAY_HALF[1] + FINGER_FACE + GAP
    stop = TRAY_STOP_Y + TRAY_HALF[1] + FINGER_FACE  # the face at the stopped wall
    local = {
        "above": (start, z + RISE),
        "touch": (start, z),
        "push": (stop + EARLY, z),
        "seat": (stop - PRELOAD, z),
        "back": (stop + BACK, z),
        "clear": (stop + BACK, z + CLEAR),
    }
    return {k: to_world(rails, push_dx, y, h) for k, (y, h) in local.items()}


def homeward(clear):
    """Where the straight way back from clear hands over to the joint move."""
    return [float(v) for v in np.add(clear, np.subtract(READY, clear) * HOMEWARD)]


# The push points for the scene check: with the tray at its nominal start and
# at the stop, the hand straight down (yaw 0).
KEY_POINTS = [
    {
        "name": name,
        "side": "left",
        "xyz": key_poses(RAILS, TRAY[1], float(np.mean(PUSH_DX)))[key],
        "grasp": "top",
    }
    for name, key in (("push_start", "touch"), ("push_end", "seat"))
]


def plan(kin, spec):
    """Raises RuntimeError where a key pose, or a 1 cm waypoint between them,
    has no IK solution inside the limits, from the default posture through the
    push and part of the way back (the rest is a joint move to the start pose),
    the hand straight down."""
    rails = spec["rails"]
    poses = key_poses(rails, spec["tray_y"], spec["push_dx"])
    q = kin.ik("left", READY, np.zeros(7))
    keys = ("above", "touch", "push", "seat", "back", "clear")
    for target in [*(poses[k] for k in keys), homeward(poses["clear"])]:
        path = kin.line("left", q, target, rotation=hand(rails))
        if np.abs(np.diff(path, axis=0)).max() > MAX_STEP:
            raise RuntimeError("the IK branch jumps on the way")
        q = path[-1]


def nominal():
    """The layout bench_check builds: no jitter."""
    return {
        "rails": list(RAILS),
        "tray_y": TRAY[1],
        "parts": {n: {"xy": list(p[1]), "yaw": 0.0} for n, p in PARTS.items()},
        "push_dx": float(np.mean(PUSH_DX)),
        "instruction": INSTRUCTION,
    }


def draw(rng):
    while True:  # the rail set's pose, redrawn until it lies on the bench
        rails = [
            RAILS[0] + float(rng.uniform(*RAILS_DX)),
            RAILS[1] + float(rng.uniform(*RAILS_DY)),
            float(rng.uniform(*RAILS_YAW)),
        ]
        if on_bench(rails):
            break
    return {
        "rails": rails,
        "tray_y": float(rng.uniform(*START_Y)),
        "parts": place_parts(rng),
        "push_dx": float(rng.uniform(*PUSH_DX)),
        "instruction": INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))],
    }


def place_parts(rng):
    """Each part anywhere on the tray's floor, PART_GAP inside the walls and
    off the other part (redrawn until apart); the flange turns freely, the
    housing's box repeats every half turn."""
    (a, (_, _, ra, _)), (b, (_, _, rb, _)) = PARTS.items()
    while True:
        parts = {}
        for name, radius in ((a, ra), (b, rb)):
            room = np.subtract(INNER, radius + PART_GAP)
            spread = np.pi if name == "flange" else np.pi / 2
            parts[name] = {
                "xy": [float(v) for v in rng.uniform(-room, room)],
                "yaw": float(rng.uniform(-spread, spread)),
            }
        gap = np.linalg.norm(np.subtract(parts[a]["xy"], parts[b]["xy"]))
        if gap > ra + rb + PART_GAP:
            return parts


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
            # 2: the trace's subtasks are single-arm actions; 3: the robot
            # model of 2026-10-25, pushing beyond the tray's centre line; 4: the
            # rail set moves and turns on the bench, the parts lie anywhere.
            "version": 4,
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
    rails = spec["rails"]

    def turned(yaw):
        return (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))

    pa.add(
        root,
        "tray_rail_set",
        "rails",
        pos=to_world(rails, 0.0, 0.0, BENCH_Z),
        quat=turned(rails[2]),
    )
    pa.add(
        root,
        "esd_tray",
        "tray",
        pos=to_world(rails, TRAY[0], spec["tray_y"], BENCH_Z + 0.0005),
        quat=turned(rails[2]),
        free=True,
    )
    pa.set_geoms(root, "tray_body_col", priority="1")
    for name, (prop, _, _, _) in PARTS.items():
        part = spec["parts"][name]
        x, y = part["xy"]
        pa.add(
            root,
            prop,
            name,
            pos=to_world(rails, TRAY[0] + x, spec["tray_y"] + y, TRAY_FLOOR_Z + 0.001),
            quat=turned(rails[2] + part["yaw"]),
            free=True,
        )
    report["task"] = {
        "name": NAME,
        "instruction": spec["instruction"],
        "key_points": KEY_POINTS,
    }
    return root, report
