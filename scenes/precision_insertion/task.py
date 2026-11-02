"""Precision insertion: the left arm slides the 6204 bearing onto its shaft
until it seats on the shoulder, and the right arm moves the dowel pin from the
fixture's one hole into the other. Both fits have 1.0 mm diametral clearance;
only the shaft has a lead-in in its collider (1 mm at 30 degrees: 1.08 mm of
radial capture), the pin holes' chamfers are visual (0.5 mm of capture).

Both fixtures are fixed to the workbench (top at BENCH_Z, 0.90 m) at
x = 0.37: the shaft fixture at y = +0.18 (shaft D19 under the bearing's 20 mm
bore, shoulder top 34 mm above the bench) and the pin fixture at y = −0.18
(c100: two 11 mm holes for the 10 mm pin, at y −0.21 and −0.15). The bearing
lies flat on the bench to the left; the pin stands in one hole, resting on the
bench through the plate. Every move is top-down and the part stays in the
hand until it is home: the fingertips reach past the pads (TIP), which kept
them from following the bearing into the housing bore of the first version
(0.2 mm clearance, 2026-10-24), and its 0.05-0.1 mm fits were beyond
position-controlled, camera-guided hands. Both hands close along y
(BEARING_YAW, PIN_YAW): on the robot model wheel_robot_26.8.16_3 (2026-10-25)
the top-down hand with the jaws along x no longer reaches the shaft's seat,
nor the inner hole at the scene check's margin.

The arms work one after the other (user rule 2026-10-25: one arm, one object
per subtask; the other arm holds still): the left arm picks the bearing,
slides it onto the shaft and goes back to its start pose, then the right arm
does the same with the pin.

A seed fixes where the bearing lies and its yaw, where the shaft fixture and
the pin fixture stand on the bench and how each is turned (the shaft is round;
turning the pin plate moves its holes about its centre), which hole the pin
starts in (the other is its target) and its yaw, and the wording (RANGES).
Every draw is checked for room (clear(): the bearing's open hand clear of the
shaft fixture) and IK-checked along each arm's path (the joint-space swing from
the start pose over its part, the straight moves, the swing back), and redrawn
until it plans. build() with no spec is the nominal layout the scene check
builds.
"""

import numpy as np

from sim import prop_assets as pa
from sim import tabletop as tt
from sim.build_task import build as robot_scene
from sim.gripper_linkage import contact_drop
from sim.workcell import BENCH_Z, industrial_cell

NAME = "precision_insertion"
INSTRUCTION = (
    "Slide the bearing onto the shaft and move the dowel pin into the empty hole."
)
SHAFT = (0.37, 0.18)
BEARING = (0.36, 0.33)
FIXTURE = (0.37, -0.18)
HOLE_DY = (-0.03, 0.03)  # the pin fixture's holes along y from its centre
HOLES = [(FIXTURE[0], FIXTURE[1] + dy) for dy in HOLE_DY]
SEAT_Z = BENCH_Z + 0.034  # the shoulder the bearing's inner ring seats on
SHAFT_TOP = BENCH_Z + 0.054
PLATE_TOP = BENCH_Z + 0.020
BEARING_D, PIN_D = 0.047, 0.010
BEARING_H, PIN_L = 0.014, 0.040
# The fingertips reach tt.TIP past the grasp site with the jaws open, and
# contact_drop(width) lower closed on a part.
GRASP_Z = tt.TIP + 0.002  # closed pads' centre over the support: fingertips 2 mm up
PIN_HOLD = PLATE_TOP - BENCH_Z + GRASP_Z  # pads' centre over the pin's foot
# Each hand turned about the vertical (rad, top-down; 0 closes the jaws along
# x), so both close along y with the wrist D405 on the far side (+x). With the
# jaws along x the left hand reaches the seat only down to a grasp site 0.988 m
# high; it needs 0.964. The bearing's hand turned +45 or +60 degrees, a swing
# sank below SWING_FLOOR in 12 and 11 of 20 seeds' first draws; +90 planned
# but swung 6.4 s each way against -90's 3.6-4.7 s. The pin's hand with the
# jaws along x misses pin_home (0.952 m over the inner hole) at the scene
# check's 2 degree margin; turned +90 it plans every seed on the first draw
# and swings in 4.4-4.8 s (-90: 6.5-6.9 s).
BEARING_YAW = np.radians(-90)
PIN_YAW = np.radians(90)
# How each part is held: (side, width across the jaws, closed pads' centre
# above the part's base, hand yaw).
PARTS = {
    "bearing": ("left", BEARING_D, GRASP_Z, BEARING_YAW),
    "pin": ("right", PIN_D, PIN_HOLD, PIN_YAW),
}
# The parts' bases on the expert's path (m): carried 20 mm over the shaft top
# and the plate, centred 4 and 3 mm above them, 4 mm into the shaft and the
# hole, brought 2 mm above the shoulder and the bench and let go 0.5 mm above
# them (a part falling 10 mm trips the 1 mm penetration audit; on PhysX a held
# part does not creep down onto its support as on MuJoCo, and drops from 1 mm
# and more sank 0.5-1.2 mm into it).
CARRY = {"bearing": SHAFT_TOP + 0.020, "pin": PLATE_TOP + 0.020}
HOVER = {"bearing": SHAFT_TOP + 0.004, "pin": PLATE_TOP + 0.003}
ENGAGE = {"bearing": SHAFT_TOP - 0.004, "pin": PLATE_TOP - 0.004}
NEAR = {"bearing": SEAT_Z + 0.002, "pin": BENCH_Z + 0.002}
HOME = {"bearing": SEAT_Z + 0.0005, "pin": BENCH_Z + 0.0005}
# The open hand swings in joint space from the start pose to SWING over its
# grasp, then comes straight down; after letting go it rises RETREAT straight
# up and swings back to the start pose's joints. Straight from the start pose,
# the 1 cm IK waypoints' corners (the shoulder roll reverses on the way) made
# Scene.move's rate limits stretch the 21 cm approach to 8.6-25.4 s; the swing
# takes 3.6-4.4 s at the controller's rates. Joint interpolation sags below the
# straight line (to 24 mm over the bench at x 0.24 for a swing to 7 cm over the
# grasp, on the previous model), so plan() keeps the swings' grasp site
# SWING_FLOOR over the bench: from RETREAT the swing back stays 78-88 mm over
# it. A straight 12 cm retreat took up to 6.3 s, 7 cm 1.5-1.7 s.
SWING = 0.12
RETREAT = 0.07
SWING_FLOOR = 0.07
READY = tt.READY
# Seeded variation (task version 4, 2026-10-27), uniform over these ranges (m;
# yaws over a full or half turn). On the left the bearing and the shaft fixture
# take one band each, either way round, so the bearing lies inside or outside
# the shaft (the open hand needs about 12 cm along y from the fixture: drawn
# over one region, three draws in four met it; with bands 0.10-0.18 and 0.26-
# 0.32 one in thirteen, with these one in fifty). Each box is where the part's
# features stay inside both D435 images at the start pose (bearing rim, shaft
# top, both holes at any plate yaw and the pin top) and its arm reaches it at
# the seat heights: the near side from the head camera's bottom edge (x 0.33-
# 0.35), the outer side from the body camera's (bearing y 0.32, plate -0.26),
# the far side from reach (the shaft seat x 0.39 at y 0.10 and 0.43 at 0.26,
# the holes 0.39 at y -0.13). clear() and plan() redraw the rest.
BANDS = {"inner": (0.10, 0.17), "outer": (0.27, 0.32)}  # y, left of the midline
RANGES = {
    ("bearing", "inner"): (0.34, 0.41),  # x
    ("bearing", "outer"): (0.35, 0.44),
    ("shaft", "inner"): (0.34, 0.39),
    ("shaft", "outer"): (0.35, 0.43),
    "fixture_x": (0.35, 0.38),
    "fixture_y": (-0.26, -0.14),
}
# The fixtures' footprints (half sizes along their own x, y) and the shaft
# column's radius at its shoulder; the bearing's open hand near the bench with
# its jaws along y (measured: within 10.5 x 50 mm of the bearing's axis below
# 40 mm over the bench, 16 x 69 mm below 60), and the room kept between them.
SHAFT_BASE = (0.035, 0.055)
SHAFT_R = 0.012
HAND_LOW = (0.0105, 0.050)
HAND_HIGH = (0.0162, 0.0692)
ROOM = 0.008
INSTRUCTIONS = (
    "Slide the bearing onto the shaft and move the dowel pin into the empty hole.",
    "Put the bearing on the shaft down to its shoulder and the pin in the other hole.",
    "Fit the bearing over the shaft, then transfer the dowel pin to the free hole.",
    "Seat the bearing on the shaft and insert the pin into the fixture's empty hole.",
)


def site(z, width):
    """The open hand's grasp-site height for closed pads' centre at z."""
    return round(z + contact_drop(width), 4)


KEY_POINTS = [
    {
        "name": "grasp_bearing",
        "side": "left",
        "xyz": [*BEARING, site(BENCH_Z + GRASP_Z, BEARING_D)],
        "grasp": "top",
        "yaw": float(BEARING_YAW),
        "width": BEARING_D,
    },
    {
        "name": "above_shaft",
        "side": "left",
        "xyz": [*SHAFT, site(SHAFT_TOP + 0.005 + GRASP_Z, BEARING_D)],
        "grasp": "top",
        "yaw": float(BEARING_YAW),
        "width": BEARING_D,
    },
    {
        "name": "seat_bearing",
        "side": "left",
        "xyz": [*SHAFT, site(SEAT_Z + GRASP_Z, BEARING_D)],
        "grasp": "top",
        "yaw": float(BEARING_YAW),
        "width": BEARING_D,
    },
    {
        "name": "grasp_pin",
        "side": "right",
        "xyz": [*HOLES[0], site(BENCH_Z + PIN_HOLD, PIN_D)],
        "grasp": "top",
        "yaw": float(PIN_YAW),
        "width": PIN_D,
    },
    {
        "name": "above_hole2",
        "side": "right",
        "xyz": [*HOLES[1], site(PLATE_TOP + 0.005 + PIN_HOLD, PIN_D)],
        "grasp": "top",
        "yaw": float(PIN_YAW),
        "width": PIN_D,
    },
    {
        "name": "pin_home",
        "side": "right",
        "xyz": [*HOLES[1], site(BENCH_Z + PIN_HOLD, PIN_D)],
        "grasp": "top",
        "yaw": float(PIN_YAW),
        "width": PIN_D,
    },
]


def hole_xy(spec, k):
    """Hole k's centre (x, y) on the spec's pin fixture, turned fixture_yaw."""
    yaw = spec.get("fixture_yaw", 0.0)
    return [
        float(v)
        for v in np.asarray(spec["fixture"]) + tt.rot_z(yaw)[:2, 1] * HOLE_DY[k]
    ]


def grasp_site(part, base_z):
    """Grasp-site height (open pads' centre) that puts the closed pads where
    they hold part with its base at base_z."""
    _, width, hold, _ = PARTS[part]
    return base_z + hold + contact_drop(width)


def hand(part):
    """The top-down hand's rotation for part."""
    return tt.rot_z(PARTS[part][3])


def key_poses(part, start, goal):
    """Grasp-site targets (world, m) for moving part from its base at start
    (x, y, z) to its seat at goal (x, y). The hand stops contact_drop(width)
    above the grasp point, so the dropping pads close there."""
    start = np.asarray(start, float)
    grasp = [*start[:2], grasp_site(part, start[2])]

    def at(xy, base_z):
        return np.array([*xy, grasp_site(part, base_z)])

    home = at(goal, HOME[part])
    return {
        "above": at(start[:2], start[2]) + [0, 0, SWING],
        "grasp": np.array(grasp),
        "lift": at(start[:2], CARRY[part]),
        "over": at(goal, CARRY[part]),
        "hover": at(goal, HOVER[part]),
        "engage": at(goal, ENGAGE[part]),
        "near": at(goal, NEAR[part]),
        "home": home,
        "clear": home + [0, 0, RETREAT],
    }


def goals(spec):
    """Each part's seat (x, y): the shaft and the pin's target hole."""
    return {"bearing": spec["shaft"], "pin": hole_xy(spec, spec["pin_target"])}


def nominal_starts(spec):
    """Each part's base (x, y, z) as the spec lays it out."""
    return {
        "bearing": [*spec["bearing"]["xy"], BENCH_Z],
        "pin": [*hole_xy(spec, 1 - spec["pin_target"]), BENCH_Z],
    }


ORDER = ("grasp", "lift", "over", "hover", "engage", "near", "home", "clear")


def sag(kin, side, start, end):
    """Raises RuntimeError if the grasp site sinks below SWING_FLOOR over the
    bench on the joint-space move from arm joints start to end."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    for u in np.linspace(0, 1, 41):
        z = kin.fk(side, start + (end - start) * u)[0][2]
        if z < BENCH_Z + SWING_FLOOR:
            raise RuntimeError(f"{side} swing sinks to {z:.3f} m")


def swing(kin, side, start, target, rotation=None):
    """The arm joints at target (grasp site, hand rotation) nearest start,
    which the expert swings to in joint space (sag() checks the way)."""
    end = kin.ik(side, target, start, rotation=rotation)
    sag(kin, side, start, end)
    return end


def plan(kin, spec):
    """Raises RuntimeError where a key pose of either arm's path, or a 1 cm
    waypoint between them, has no IK solution inside the limits, or where a
    swing from or back to the start pose sinks too low."""
    starts, ends = nominal_starts(spec), goals(spec)
    for part, (side, *_) in PARTS.items():
        poses, rotation = key_poses(part, starts[part], ends[part]), hand(part)
        start = kin.ik(side, READY[side], np.zeros(7))  # tt.start_home's joints
        q = swing(kin, side, start, poses["above"], rotation)
        for key in ORDER:
            q = kin.line(side, q, poses[key], rotation=rotation)[-1]
        sag(kin, side, q, start)


def nominal():
    """The layout the scene check builds: no jitter, the pin in the first
    hole."""
    return {
        "bearing": {"xy": list(BEARING), "yaw": 0.0},
        "shaft": list(SHAFT),
        "shaft_yaw": 0.0,
        "fixture": list(FIXTURE),
        "fixture_yaw": 0.0,
        "pin_target": 1,
        "pin_yaw": 0.0,
        "instruction": INSTRUCTION,
    }


def draw(rng):
    def turn(span=2 * np.pi):
        return float(rng.uniform(-span / 2, span / 2))

    inside = bool(rng.integers(2))  # the bearing in the inner band
    place = {}
    for part, band in (("bearing", "inner"), ("shaft", "outer")):
        band = band if inside else ("outer" if band == "inner" else "inner")
        place[part] = [
            float(rng.uniform(*RANGES[part, band])),
            float(rng.uniform(*BANDS[band])),
        ]
    return {
        "bearing": {"xy": place["bearing"], "yaw": turn()},
        "shaft": place["shaft"],
        # Both footprints look the same turned half a turn: the shaft's base,
        # and the pin plate with its holes swapped (the start hole is drawn).
        "shaft_yaw": turn(np.pi),
        "fixture": [
            float(rng.uniform(*RANGES["fixture_x"])),
            float(rng.uniform(*RANGES["fixture_y"])),
        ],
        "fixture_yaw": turn(np.pi),
        "pin_target": int(rng.integers(2)),
        "pin_yaw": turn(),
        "instruction": INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))],
    }


def overlap(c1, half1, yaw1, c2, half2, yaw2):
    """Whether two rectangles (centre, half sizes, yaw) overlap: no separating
    axis among their edges' normals."""
    a1, a2 = tt.rot_z(yaw1)[:2, :2].T, tt.rot_z(yaw2)[:2, :2].T
    d = np.subtract(c2, c1)
    for axis in (*a1, *a2):
        r1 = sum(h * abs(axis @ u) for h, u in zip(half1, a1, strict=True))
        r2 = sum(h * abs(axis @ u) for h, u in zip(half2, a2, strict=True))
        if abs(axis @ d) > r1 + r2:
            return False
    return True


def clear(spec):
    """Raises RuntimeError where the bearing, or its open hand at the grasp
    (jaws along y), comes within ROOM of the shaft fixture's base or of its
    shaft column."""
    bearing, shaft, yaw = spec["bearing"]["xy"], spec["shaft"], spec["shaft_yaw"]
    base = [h + ROOM for h in SHAFT_BASE]
    disc = (BEARING_D / 2, BEARING_D / 2)
    column = (SHAFT_R + ROOM, SHAFT_R + ROOM)
    if (
        overlap(bearing, HAND_LOW, 0.0, shaft, base, yaw)
        or overlap(bearing, disc, 0.0, shaft, base, yaw)
        or overlap(bearing, HAND_HIGH, 0.0, shaft, column, 0.0)
    ):
        raise RuntimeError("the bearing's hand meets the shaft fixture")


def sample_episode(seed, kin=None, attempts=100):
    kin = kin or tt.kinematics(tt.SAMPLE_MARGIN)
    rng = np.random.default_rng(seed)
    for attempt in range(1, attempts + 1):
        spec = draw(rng)
        try:
            clear(spec)
            plan(kin, spec)
        except RuntimeError:
            continue
        return {
            "task": NAME,
            # 2: one arm at a time (single-arm subtasks); 3: wheel_robot_26.8.16_3,
            # the hands turned, let go 0.5 mm over the seats; 4: wider layouts
            # (RANGES), the fixtures turned.
            "version": 4,
            "seed": int(seed),
            "draws": attempt,
            **spec,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def yaw_quat(yaw):
    return (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))


def build(spec=None):
    """The scene for spec (default: the nominal layout)."""
    spec = spec or nominal()
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    pa.add(
        root,
        "bearing_shaft_fixture",
        "shaft",
        pos=(*spec["shaft"], BENCH_Z),
        quat=yaw_quat(spec.get("shaft_yaw", 0.0)),
    )
    pa.add(
        root,
        "bearing_6204",
        "bearing",
        pos=(*spec["bearing"]["xy"], BENCH_Z + 0.0005),
        quat=yaw_quat(spec["bearing"]["yaw"]),
        free=True,
    )
    pa.add(
        root,
        "pin_fixture",
        "fixture",
        pos=(*spec["fixture"], BENCH_Z),
        quat=yaw_quat(spec.get("fixture_yaw", 0.0)),
        variant_name="c100",
    )
    pa.add(
        root,
        "dowel_pin",
        "pin",
        pos=(*hole_xy(spec, 1 - spec["pin_target"]), BENCH_Z + 0.0005),
        quat=yaw_quat(spec["pin_yaw"]),
        free=True,
    )
    report["task"] = {
        "name": NAME,
        "instruction": spec["instruction"],
        "key_points": KEY_POINTS,
    }
    return root, report
