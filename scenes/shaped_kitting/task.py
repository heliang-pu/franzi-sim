"""Shaped kitting: put each part into the foam pocket of its shape, in the
orientation the pocket allows (the bushing flange up, the gear hub up, the
hex aligned with its pocket's flats).

The foam kit tray lies on the workbench (top at BENCH_Z, 0.90 m), nominally
centred at x = 0.40, y = 0, with its pockets at x 0.345 and 0.455, y ±0.09;
they are cut 4 mm per side larger than the parts (1.5 mm until 2026-10-25).
The left arm owns the bushing and gear pockets (y +0.09) and their parts,
which lie on the left of the tray (nominally at y 0.25); the right arm owns
the hex blank and the L bracket on the right. Every part lies upright as its
pocket holds it, so the hand takes it top-down, its jaws turned to the part's
yaw, and only moves it. At this bench height top-down grasps reach the far
pockets too (about x 0.48).

Each pocket has a finger notch on either side along x, 26 mm wide and 5 mm
deeper than the pocket, so the fingers (21 mm wide) follow a part down to the
pocket floor: the hand lowers every part to SET_GAP over the floor, then sets
it down and lets go. The jaws hold the bushing by its flange, the gear by its
tip circle, the hex blank across its flats and the L bracket by its long leg
(4.25 mm thick, which the notches straddle), 17.9 mm from the bracket's
origin.

A seed fixes the order the four parts are moved in (one arm at a time), the
tray's place and yaw on the bench (TRAY_SHIFT, TRAY_YAW), each part's place
anywhere in its arm's supply area beside the tray (SUPPLY_X, SUPPLY_Y) in the
body camera's view and clear of the tray and of the other part's open
fingers, each part's yaw (the round parts and the hex blank any, the bracket
within 30 degrees) and the wording. Every draw is IK-checked along the
expert's straight moves and turns and redrawn until it plans.
"""

import numpy as np

from sim import prop_assets as pa
from sim import tabletop as tt
from sim.build_task import build as robot_scene
from sim.gripper_linkage import contact_drop
from sim.workcell import BENCH_Z, industrial_cell

NAME = "shaped_kitting"
INSTRUCTION = "Put each part into the foam pocket that matches its shape."
TRAY = (0.40, 0.0)
FOAM_TOP = BENCH_Z + 0.043
FLOOR_Z = BENCH_Z + 0.023  # every pocket's floor (prop.json floor_z_mm)
GRASP_Z = 0.019  # closed pads' centre over a part's base (see GRIP)
# A part's lowest point over the pocket floor where the carried descent ends.
# Held parts creep down in the jaws under MuJoCo (about 0.2 mm/s) and pivot
# 2-4 degrees about the pad line under PhysX, so the hand aims from their
# measured lowest point (run.aim); at 2 mm, nominal, they reached the floor in
# the jaws.
SET_GAP = 0.003
# part: (prop, xy, arm); GRIP says how the jaws hold it
PARTS = {
    "bushing": ("flanged_bushing", (0.36, 0.25), "left"),
    "gear": ("spur_gear", (0.44, 0.25), "left"),
    "hex": ("hex_blank", (0.36, -0.25), "right"),
    "bracket": ("l_bracket", (0.44, -0.25), "right"),
}
# Pocket centres in the tray frame, and in the world for the nominal tray.
POCKET = {
    p["part"]: (p["center_yx_mm"][1] / 1000, p["center_yx_mm"][0] / 1000)
    for p in pa.spec("foam_kit_tray")["pockets"]
}
POCKETS = {prop: (TRAY[0] + x, TRAY[1] + y) for prop, (x, y) in POCKET.items()}
# How the jaws hold each part: (width across the jaws, the grasp point in the
# part frame (x, y), squeeze past contact, closed pads' centre over the part's
# base). The finger meshes reach 13.4 mm below the closed pads' centre, so at
# GRASP_Z their tips stay 5.6 mm over the base; the gear's 12 mm rim is held
# 3 mm lower. The bracket is held by its long leg, whose collider spans x -20
# to -15.75 mm. Squeeze grows with mass: the gear is 29 g, the others
# 120-149 g; the bracket's leg is so thin that the fingers reach their closed
# stop 4.25 mm past contact.
GRIP = {
    "bushing": (0.040, (0.0, 0.0), 0.006, GRASP_Z),
    "gear": (0.048, (0.0, 0.0), 0.003, GRASP_Z - 0.003),
    "hex": (0.027, (0.0, 0.0), 0.006, GRASP_Z),
    "bracket": (0.00425, (-0.017875, 0.0), 0.00425, GRASP_Z),
}
# Each part's height (its pocket's floor to its top, seated), for the retreat.
HEIGHT = {"bushing": 0.030, "gear": 0.018, "hex": 0.030, "bracket": 0.040}
# The hand's yaw over the pockets (the jaws close along x either way, as the
# finger notches need): the right hand works turned 180 degrees. On the
# wheel_robot_26.8.16_3 arm the top-down right hand at yaw 0 meets its wrist
# roll's -90 degree stop at 0.985 m over the bracket's pocket (x 0.327), below
# the carry heights; turned, it reaches past 1.06 m. The left hand reaches
# 1.005 m over the gear's pocket (x 0.345) at yaw 0, enough with ENTRY_Z.
HAND_YAW = {"left": 0.0, "right": np.pi}
# The 180-degree turn goes in chained IK steps at READY: in one joint-space
# move the hand swung 243 mm; in 15-degree steps it stays within 2.9 mm.
TURN_STEP = np.radians(15)
# The largest joint step a straight line may take between its 1 cm waypoints,
# and a turn between its steps (the 15-degree turn steps take up to 21
# degrees). Over a hex pocket moved near the midline the chained IK flipped
# the right shoulder roll by 187 degrees in one waypoint of the 3 cm descent,
# and the hand swung onto the bench; plan_arm rejects such layouts.
JUMP = {"line": 0.25, "turn": 0.6}
# Each arm goes back to its start pose by a joint move at READY onto the pose
# its first turn started from or ended on, then (the right hand) back along
# that turn's recorded steps. Over its work the right arm's elbow swivel
# drifts by up to 213 degrees (shoulder pitch and yaw together), and turned
# back by chained IK steps from there, steps jumped up to 104 degrees (in most
# layouts more than 30). The joint move keeps the hand within SWEEP_DRIFT of
# READY and its fingertips SWEEP_CLEAR over the bench (at most 86 mm and 59 mm
# over 40 layouts).
SWEEP_DRIFT = 0.12
SWEEP_CLEAR = 0.04
# Yaw symmetry period (rad; 0: round, any yaw) and the start-yaw jitter.
SYMMETRY = {"bushing": 0.0, "gear": 0.0, "hex": np.pi / 3, "bracket": 2 * np.pi}
YAW = {
    "bushing": np.pi,
    "gear": np.pi,
    "hex": np.pi / 6,  # its 60-degree symmetry: any yaw
    "bracket": np.radians(30),
}
LABEL = {
    "bushing": "flanged bushing",
    "gear": "spur gear",
    "hex": "hex blank",
    "bracket": "L bracket",
}
# The layout's randomization (task version 4). The tray's centre moves within
# TRAY_SHIFT of TRAY (x, y ranges, m) and it turns within TRAY_YAW (rad).
# Turned toward +yaw or pushed toward the robot, the gear's pocket comes into
# the left hand's wrist-roll stop (a +4 degree turn moves it 6 mm toward the
# body and 4 mm toward the midline); outward in x the far pockets leave the
# reach at the far corner. The parts lie anywhere in their arm's supply area,
# x SUPPLY_X and |y| SUPPLY_Y (top-down grasps reach x 0.30-0.48 there at any
# yaw within 30 degrees), clear of the tray by MARGIN and of each other's open
# fingers (footprint circles of RADIUS; the bracket's around its origin).
TRAY_SHIFT = ((-0.010, 0.015), (-0.03, 0.03))
TRAY_YAW = (np.radians(-6), np.radians(2))
TRAY_HALF = (0.130, 0.190)  # the tray's outer half-sizes (prop.json 260 x 380)
SUPPLY_X = (0.33, 0.47)
SUPPLY_Y = (0.21, 0.35)
RADIUS = {"bushing": 0.020, "gear": 0.024, "hex": 0.0156, "bracket": 0.036}
# What the cameras see of the bench, from the start posture (the head at its
# 0.50 rad gaze), their image corners projected onto z 0.92: the body D435's
# view widens from |y| 0.288 (left) and 0.253 (right) at x 0.24 by 0.589 per
# metre of x (it looks a little left), the head D435's starts at x 0.307. At
# (0.32, -0.35) a hex blank showed 0 pixels in the body camera; every part
# keeps its whole footprint inside that view (in_view), and SUPPLY_X starts at
# 0.33, where the head camera sees most of the bracket and all of the rest.
BODY_VIEW = (0.24, 0.288, -0.253, 0.589)  # x0, left y, right y at x0, widening
FINGER = (0.012, 0.0105)  # an open finger's thickness out of its pad, half-width
MARGIN = 0.005
# A lifted part's base height over the supply, where the hand turns it, and
# over its pocket before the descent: 12 mm over the foam and 5 mm over the
# shell's rim (BENCH_Z + 0.050), which the carry crosses 20-25 mm higher. No
# carry passes over another pocket. Over the bracket's pocket the turned
# right hand's wrist pitch reaches its -60 degree stop from 1.00 m up (4.9
# degrees inside it at 0.95 m), so the entry stays low.
CARRY_Z = BENCH_Z + 0.09
ENTRY_Z = BENCH_Z + 0.055
# Waypoints (grasp site, world): the bracket's carry passes 7 cm outboard of
# its pocket at 1.06 m before it descends to the entry. Straight from the
# supply, the line ran along that wrist pitch stop and Scene.move's rate
# limits stretched its kinks to 13.5 s; via the waypoint it takes 4 s. Each
# arm's return passes (0.30, +-0.30, 1.10): straight from the left pockets to
# READY the left wrist left its roll stop for its yaw stop and took 20-26 s,
# via the waypoint 9.4-9.8 s.
TRANSFER_VIA = {"bracket": (-0.07, 1.06)}  # y offset toward its arm, site z
RETURN_VIA = {"left": (0.30, 0.30, 1.10), "right": (0.30, -0.30, 1.10)}
# After letting go, the fingertips rise this far over the part's top.
CLEARANCE = 0.015
INSTRUCTIONS = (
    INSTRUCTION,
    "Place the four parts in their matching pockets of the grey foam tray.",
    (
        "Kit the bushing, the gear, the hex blank and the L bracket into the "
        "foam tray, each in the pocket of its shape."
    ),
    "Sort the parts into the foam tray: every part goes into the pocket cut for it.",
)


def site(base_z, width, grasp_z=GRASP_Z):
    """The open hand's grasp-site height for a part resting at base_z."""
    return round(base_z + grasp_z + contact_drop(width), 4)


def grasp_xy(name, xy, yaw):
    """Where the jaws close on part name lying at xy with yaw."""
    offset = np.asarray(GRIP[name][1])
    return np.asarray(xy, float) + tt.rot_z(yaw)[:2, :2] @ offset


def key_point(label, name, xy, base_z):
    """A grasp-site key point (sim.bench_check) for part name, its base at
    (xy, base_z), the hand at its arm's HAND_YAW."""
    width, _, _, grasp_z = GRIP[name]
    side = PARTS[name][2]
    return {
        "name": label,
        "side": side,
        "xyz": [*grasp_xy(name, xy, 0.0).tolist(), site(base_z, width, grasp_z)],
        "grasp": "top",
        "yaw": HAND_YAW[side],
        "width": width,
    }


# The grasps, each part over its pocket, and each part let go in its pocket,
# where the open fingers stand in the pocket's notches.
KEY_POINTS = (
    [key_point(f"grasp_{n}", n, p[1], BENCH_Z) for n, p in PARTS.items()]
    + [
        key_point(f"pocket_{p[0]}", n, POCKETS[p[0]], FOAM_TOP + SET_GAP)
        for n, p in PARTS.items()
    ]
    + [
        key_point(f"release_{p[0]}", n, POCKETS[p[0]], FLOOR_Z + SET_GAP)
        for n, p in PARTS.items()
    ]
)


def fold(yaw, period):
    """yaw (rad) folded into half a symmetry period either side of 0 (0 for a
    round part)."""
    if not period:
        return 0.0
    return float((yaw + period / 2) % period - period / 2)


def tray_pose(spec):
    """The tray's centre (x, y) and yaw in spec (nominal before version 4)."""
    tray = spec.get("tray") or {"xy": TRAY, "yaw": 0.0}
    return np.asarray(tray["xy"], float), float(tray["yaw"])


def pocket_xy(tray, prop):
    """The world centre of prop's pocket in the tray at tray (xy, yaw)."""
    xy, yaw = tray
    return xy + tt.rot_z(yaw)[:2, :2] @ np.asarray(POCKET[prop])


def key_poses(name, base, yaw, tray):
    """Grasp-site targets (world, m) and hand yaws for moving part name, whose
    base lies at base (x, y, z) with yaw (rad), into its pocket in the tray at
    tray (xy, yaw): the hand closes on it turned to its yaw (folded by its
    symmetry about the pocket's) from its arm's HAND_YAW, lifts it to CARRY_Z,
    turns it to the pocket's yaw, carries it down to ENTRY_Z over its pocket,
    lowers it to SET_GAP over the pocket floor and, having let go, rises until
    its fingertips clear the part by CLEARANCE. The hand stops
    contact_drop(width) above the grasp point, so the dropping pads close
    there."""
    width, _, _, grasp_z = GRIP[name]
    prop, _, side = PARTS[name]
    above = grasp_z + contact_drop(width)  # grasp site over the part's base
    psi = tray[1]
    g = grasp_xy(name, base[:2], yaw)
    c = grasp_xy(name, pocket_xy(tray, prop), psi)
    over = ENTRY_Z + above
    clear = FLOOR_Z + HEIGHT[name] + CLEARANCE + tt.TIP + contact_drop(width)
    hand = HAND_YAW[side] + psi
    via = (
        [[c[0], c[1] + dy, z] for dy, z in [TRANSFER_VIA[name]]]
        if name in TRANSFER_VIA
        else []
    )
    return {
        "hand_yaw": hand,
        "grasp_yaw": hand + fold(yaw - psi, SYMMETRY[name]),
        "above": [g[0], g[1], CARRY_Z + above],
        "grasp": [g[0], g[1], base[2] + above],
        "via": via,
        "over": [c[0], c[1], over],
        "release": [c[0], c[1], FLOOR_Z + SET_GAP + above],
        "clear": [c[0], c[1], max(over, clear)],
    }


def corners(centre, half, yaw):
    """The corners of a rectangle of half-sizes half about centre, turned yaw."""
    R = tt.rot_z(yaw)[:2, :2]
    unit = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
    return np.asarray(centre, float) + (unit * half) @ R.T


def overlap(a, b):
    """Whether convex polygons a and b (corner arrays) overlap (separating
    axes)."""
    for poly in (a, b):
        for k in range(len(poly)):
            edge = poly[(k + 1) % len(poly)] - poly[k]
            axis = np.array([-edge[1], edge[0]])
            pa_, pb = a @ axis, b @ axis
            if pa_.max() < pb.min() or pb.max() < pa_.min():
                return False
    return True


def near(poly, point, radius):
    """Whether a circle (point, radius) touches convex polygon poly."""
    point = np.asarray(point, float)
    inside = all(
        np.cross(poly[(k + 1) % len(poly)] - poly[k], point - poly[k]) >= 0
        for k in range(len(poly))
    )
    if inside:
        return True
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        u = np.clip(np.dot(point - a, b - a) / np.dot(b - a, b - a), 0, 1)
        if np.linalg.norm(a + u * (b - a) - point) < radius:
            return True
    return False


def fingers(name, xy, yaw, tray):
    """The open fingers' footprints (two rectangles) when the hand closes on
    part name lying at xy with yaw, as the expert opens them (task GRIP width
    plus tt.CLEAR per side), widened by MARGIN."""
    width = GRIP[name][0]
    hand = HAND_YAW[PARTS[name][2]] + tray[1] + fold(yaw - tray[1], SYMMETRY[name])
    g = grasp_xy(name, xy, yaw)
    R = tt.rot_z(hand)[:2, :2]
    half = (FINGER[0] / 2 + MARGIN, FINGER[1] + MARGIN)
    out = width / 2 + tt.CLEAR + FINGER[0] / 2
    return [corners(g + R @ [s * out, 0.0], half, hand) for s in (-1, 1)]


def in_view(name, xy):
    """Whether part name at xy lies wholly in the body D435's view of the
    bench (BODY_VIEW), MARGIN inside its edges."""
    x0, left, right, widen = BODY_VIEW
    x, y = xy
    room = RADIUS[name] + MARGIN
    return y + room <= left + widen * (x - x0) and y - room >= right - widen * (x - x0)


def clear_layout(spec):
    """Whether spec's parts lie clear: each in the body camera's view, off the
    tray (by MARGIN) and of the other part on its side, and each one's open
    fingers clear of the tray and of that other part."""
    tray = tray_pose(spec)
    rim = corners(tray[0], np.add(TRAY_HALF, MARGIN), tray[1])
    parts = spec["parts"]
    for side in tt.SIDES:
        mine_ = [n for n, p in PARTS.items() if p[2] == side]
        for name in mine_:
            other = next(n for n in mine_ if n != name)
            xy, oxy = parts[name]["xy"], parts[other]["xy"]
            if not in_view(name, xy):
                return False
            if np.linalg.norm(np.subtract(xy, oxy)) < (
                RADIUS[name] + RADIUS[other] + 2 * MARGIN
            ):
                return False
            if near(rim, xy, RADIUS[name]):
                return False
            for box in fingers(name, xy, parts[name]["yaw"], tray):
                if overlap(box, rim) or near(box, oxy, RADIUS[other]):
                    return False
    return True


def turn_yaws(start, end, step=TURN_STEP):
    """The hand yaws of an in-place turn from start to end (rad), in steps of
    at most step: [end] for a small turn."""
    n = max(1, int(np.ceil(abs(end - start) / step - 1e-9)))
    return [start + (end - start) * k / n for k in range(1, n + 1)]


def mine(spec, side):
    """side's parts in the order they are moved."""
    return [name for name in spec["order"] if PARTS[name][2] == side]


def chained(kin, side, q, target, yaw, jump):
    """kin.line's waypoints from arm pose q to target with the hand at yaw;
    raises RuntimeError if consecutive waypoints differ by more than jump
    (rad) in any joint, a jump to another IK branch."""
    path = kin.line(side, q, target, rotation=tt.rot_z(yaw))
    step = float(np.abs(np.diff(path, axis=0)).max(initial=0.0))
    if step > jump:
        raise RuntimeError(
            f"{side} IK jumps {np.degrees(step):.0f} deg on the way to "
            f"{np.round(target, 3).tolist()}"
        )
    return path


def sweep(kin, side, qa, qb):
    """Raises RuntimeError if the joint move from arm pose qa to qb (minimum
    jerk) takes the grasp site more than SWEEP_DRIFT from where it starts or
    the fingertips lower than SWEEP_CLEAR over the bench."""
    start = kin.fk(side, qa)[0]
    reach = tt.TIP + contact_drop(0.0)  # the fingertips below the site, at most
    fingers = np.array([[s * tt.HOME_OPENING / 2, 0.0, -reach] for s in (-1, 1)])
    for u in np.linspace(0, 1, 21):
        site, R = kin.fk(side, qa + (qb - qa) * (10 * u**3 - 15 * u**4 + 6 * u**5))
        low = (site + fingers @ R.T)[:, 2].min()
        if np.linalg.norm(site - start) > SWEEP_DRIFT or low < BENCH_Z + SWEEP_CLEAR:
            raise RuntimeError(f"{side} joint move sweeps the hand to {site.round(3)}")


def plan_arm(kin, spec, side):
    """Raises RuntimeError where a key pose, or a 1 cm waypoint between them,
    or a step of an in-place turn has no IK solution inside the limits, or
    where consecutive ones jump branches (JUMP), or where the joint move back
    onto the start sweeps too far (sweep), on side's path from the default
    posture (tt.start_home) through its two parts and back, as the expert
    moves it."""
    q, yaw = kin.ik(side, tt.READY[side], np.zeros(7)), 0.0
    start = q

    def turn(target, end):
        nonlocal q, yaw
        for y in turn_yaws(yaw, end):
            q = chained(kin, side, q, target, y, JUMP["turn"])[-1]
        yaw = end

    def line(target):
        nonlocal q
        q = chained(kin, side, q, target, yaw, JUMP["line"])[-1]

    turn(tt.READY[side], HAND_YAW[side])
    turned = q  # where the first turn ends, which the return goes back along
    tray = tray_pose(spec)
    for name in mine(spec, side):
        part = spec["parts"][name]
        poses = key_poses(name, [*part["xy"], BENCH_Z], part["yaw"], tray)
        line(poses["above"])
        turn(poses["above"], poses["grasp_yaw"])
        line(poses["grasp"])
        line(poses["above"])
        turn(poses["above"], poses["hand_yaw"])
        for target in poses["via"]:
            line(target)
        for key in ("over", "release", "clear"):
            line(poses[key])
    line(RETURN_VIA[side])
    line(tt.READY[side])
    turn(tt.READY[side], HAND_YAW[side])
    sweep(kin, side, q, turned if HAND_YAW[side] else start)


def plan(kin, spec):
    """plan_arm for both arms."""
    for side in tt.SIDES:
        plan_arm(kin, spec, side)


def nominal():
    """The layout bench_check builds: the tray at TRAY, the parts on their
    nominal spots (PARTS), no yaw."""
    return {
        "order": list(PARTS),
        "tray": {"xy": list(TRAY), "yaw": 0.0},
        "parts": {n: {"xy": list(p[1]), "yaw": 0.0} for n, p in PARTS.items()},
        "instruction": INSTRUCTION,
    }


def draw(rng):
    """One layout: the tray's pose, then the parts drawn over their supply
    areas until they lie clear (clear_layout)."""
    names = list(PARTS)
    order = [names[i] for i in rng.permutation(len(names))]
    shift = [rng.uniform(*r) for r in TRAY_SHIFT]
    tray = {
        "xy": [float(TRAY[0] + shift[0]), float(TRAY[1] + shift[1])],
        "yaw": float(rng.uniform(*TRAY_YAW)),
    }
    wording = INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))]
    while True:
        parts = {}
        for name, (_, _, side) in PARTS.items():
            sign = 1 if side == "left" else -1
            parts[name] = {
                "xy": [
                    float(rng.uniform(*SUPPLY_X)),
                    float(sign * rng.uniform(*SUPPLY_Y)),
                ],
                "yaw": float(rng.uniform(-1, 1) * YAW[name]),
            }
        spec = {"order": order, "tray": tray, "parts": parts, "instruction": wording}
        if clear_layout(spec):
            return spec


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
            # 2: one arm at a time, one subtask per pick, place and return;
            # 3: the wheel_robot_26.8.16_3 arm, the right hand turned 180 degrees;
            # 4: the tray and the parts placed over the bench (TRAY_SHIFT ...)
            "version": 4,
            "seed": int(seed),
            "draws": attempt,
            **spec,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def build(spec=None):
    """The scene for spec (default: the nominal layout)."""
    spec = spec or nominal()
    tray = tray_pose(spec)
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    # Light-grey foam: the cameras can't make out pockets in the black one.
    pa.add(
        root,
        "foam_kit_tray",
        "kit_tray",
        pos=(*tray[0], BENCH_Z),
        quat=(np.cos(tray[1] / 2), 0, 0, np.sin(tray[1] / 2)),
        variant_name="light_grey",
    )
    for name, (prop, _, _) in PARTS.items():
        part = spec["parts"][name]
        half = part["yaw"] / 2
        pa.add(
            root,
            prop,
            name,
            pos=(*part["xy"], BENCH_Z + 0.0005),
            quat=(np.cos(half), 0, 0, np.sin(half)),
            free=True,
        )
    report["task"] = {
        "name": NAME,
        "instruction": spec["instruction"],
        "key_points": KEY_POINTS,
        "pockets_world_xy": {prop: pocket_xy(tray, prop).tolist() for prop in POCKET},
    }
    return root, report
