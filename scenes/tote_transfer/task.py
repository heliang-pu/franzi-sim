"""Tote transfer: the left arm moves one part from the blue source tote into
the grey destination tote.

Both totes stand on the workbench (top at BENCH_Z, 0.90 m) with their long
side along y, the blue one on the robot's left, the grey one nearer its
midline, a few centimetres apart; each episode moves them (TOTE_SHIFT, GAP).
The aluminium flange and the plastic housing lie loose anywhere on the blue
tote's floor where the hand can grasp them and both D435s see them
(VIEW_X_MIN, VIEW_Y_EDGE); the instruction names the one to move, and it goes
anywhere on the grey tote's floor that the hand reaches and the cameras see.

The hand works top-down. On wheel_robot_26.8.16_3 the wrist roll bends the
hand within the jaws' plane, so with the default posture's hand (yaw 0,
wrist camera outboard) the arm keeps its elbow behind the shoulder and its
forearm level, and the wrist yaw and roll reach their limits before the hand
is 5 cm above the default posture's height: it cannot carry a part over a
rim. Turned in place at READY to a yaw in HAND_YAWS, in TURN_STEP chained IK
steps (the hand stays within 3 mm of READY), the arm holds its elbow out to
the side, and up as the hand reaches forward, which reaches both totes at
carry height. The jaws close across the housing's 45 mm width, so its yaw
sets the hand's (HOUSING_YAW keeps it in HAND_YAWS); the round flange takes
any yaw and is grasped at a hand yaw drawn from HAND_YAWS. The transfer goes
in joint space; the way home retraces the transfer and the approach.

Below a rim the hand spans HAND_U along its jaws and HAND_V across them; its
wrist D405 hangs 46–93 mm beside the jaws' plane, its lowest point 81 mm above
the grasp site, 18 mm above a rim when the hand grasps or lets go at a tote's
floor. So a part may lie wherever its own footprint and the hand's, turned to
the grasp's yaw, keep WALL_GAP from the walls, and the other part keeps clear
of the jaws.

A seed fixes the totes' places, which part moves, the parts' places and yaw,
the hand's yaw, the spot in the grey tote and the wording. Every draw is
IK-checked along the expert's moves and redrawn until it plans.
"""

import numpy as np

from sim import prop_assets as pa
from sim import tabletop as tt
from sim.build_task import build as robot_scene
from sim.gripper_linkage import contact_drop
from sim.workcell import BENCH_Z, industrial_cell

NAME = "tote_transfer"
# 2: bench 0.90 m; 3: default posture start; 4: subtasks; 5: wheel_robot_26.8.16_3
# (the hand turned half a turn at READY, carried lower, let go 0.5 mm up);
# 6: the totes move, the parts lie anywhere on the floor at any yaw the hand
# takes, the hand's yaw follows the part, and the drop spans the reachable floor.
VERSION = 6
# The totes' nominal centres (the scene check's layout), 3 cm apart.
SOURCE = (0.37, 0.28)
DEST = (0.37, -0.05)
# Each episode moves the grey tote up to TOTE_SHIFT (x, y), the blue one as
# far in x, and sets the gap between their rims (GAP): the blue tote's y
# follows the grey one's. The grey tote stays near the blue one because the
# left arm reaches into it only near its +y wall (DROP_LINE).
TOTE_SHIFT = (0.02, 0.015)
GAP = (0.01, 0.04)
TOTE_HALF = (0.10, 0.15)  # half the outer rim
FLOOR_Z = BENCH_Z + 0.004  # the tote's 4 mm floor
RIM_Z = BENCH_Z + 0.10
INNER = (0.0905, 0.1405)  # half the inner floor (draft walls narrow it)
# part: (prop, nominal xy, grip width, name in the instruction)
PARTS = {
    "flange": ("flange_part", (0.37, 0.20), 0.060, "aluminium flange"),
    "housing": ("abs_housing", (0.37, 0.32), 0.045, "plastic housing"),
}
# Each part's footprint: the flange's radius, the housing's half size (its
# 45 mm width, which the jaws close across, along its own x) and the radius
# that holds either.
FLANGE_R = 0.030
HOUSING_HALF = (0.0225, 0.035)
REACH_R = {"flange": FLANGE_R, "housing": float(np.hypot(*HOUSING_HALF))}
DROP_XY = (0.37, 0.03)
# The hand below a rim (grasp frame, measured on the colliders): half its
# span along the jaws, open for the grasp and for letting go, and across them.
HAND_U = {"grasp": {"flange": 0.0757, "housing": 0.0682}}
HAND_U["release"] = {"flange": 0.0697, "housing": 0.0622}
HAND_V = 0.0272
# What reaches below the parts' tops as the hand closes on one, which the
# other part must keep WALL_GAP clear of: the open fingers, FINGER_REACH past
# the part's sides along the jaws (12 mm clearance, 12 mm finger) and
# FINGER_V across them.
FINGER_REACH = 0.024
FINGER_V = 0.023
WALL_GAP = 0.006  # hand and parts from the walls, and the other part from the jaws
# The hand's yaws the arm carries out the task in (rad, about z; 0 closes the
# jaws along x with the wrist camera outboard): the default posture's hand
# turned 90 to 105 or 165 to 270 degrees clockwise (seen from above). Between
# them the way back to RISE ends on another branch, or the joint-space
# transfer tilts the hand past TRANSFER_TILT; nearer the default yaw the arm
# cannot carry the part over a rim.
HAND_YAWS = ((np.radians(-270), np.radians(-165)), (np.radians(-105), np.radians(-90)))
HAND_YAW = -np.pi  # the nominal layout's
# The housing's yaw: it lies at any yaw whose grasp falls in HAND_YAWS (the
# jaws close across it either way round), 120 of its 180 distinct degrees.
HOUSING_YAW = (np.radians(-105), np.radians(15))
# Where the hand works. The left arm reaches into the grey tote near the
# robot's midline only so far: the straight descent to let go stays on one
# IK branch for drops with y at least DROP_LINE[0] + DROP_LINE[1] (x - 0.37)
# (1 cm inside the boundary measured over the hand yaws), beyond which its
# waypoints jumped to another branch.
DROP_LINE = (0.015, 0.75)
# A straight move whose 1 cm IK waypoints differ by more than JUMP (rad) has
# switched branches: the arm would swing between them.
JUMP = 0.25
# What the head and body D435s see at the start posture, at part height on a
# tote's floor (the bench projected into both 640 x 480 images): the head
# camera's lower image edge runs at x 0.307, the body camera's left edge along
# y = 0.325 + 0.59 (x - 0.30). Every part, and the moving one where it is let
# go, lies wholly 5 mm inside both: farther out the body camera kept 12-58 of
# a part's 500-900 pixels.
VIEW_X_MIN = 0.312
VIEW_Y_EDGE = (0.320, 0.59)
# The closed pads' centre above a part's base: the fingertips reach tt.TIP
# below it, 5.6 mm off the tote floor, and 6.4 mm of pad hold the flange's
# 12 mm disc.
GRASP_Z = 0.019
# The carried descent into the grey tote stops with the part's measured lowest
# point SET_GAP over the floor, out of reach of the part's creep in the jaws
# (MuJoCo) and its pivot about the pads' line (PhysX, 2-4 degrees); the
# set-down then brings that point to RELEASE over the floor, where the jaws
# open (by RELEASE_OPEN, 6 mm per side).
SET_GAP = 0.006
RELEASE = 0.0005
RELEASE_OPEN = 0.012
# Grasp site height while over the rims: the part's base 21-24 mm over them.
CARRY_Z = BENCH_Z + 0.155
LIFT_MIN = 0.08  # the lift raises the part at least this far, or it slipped
READY = tt.READY["left"]
TURN_STEP = np.radians(15)
# The hand rises here, over READY, after turning; its housing stays 12 cm
# short of the nearest source tote's near wall. On the way home it comes back
# here near the joints it rose here with, goes down onto the turn's last
# joints and turns back along the turn's path.
RISE = (READY[0], READY[1], CARRY_Z)
# The transfer from over the part to over the drop goes in joint space: a
# straight line there rides joint limits, whose kinks stretched it to 9 s
# (2.5 s in joint space). The sampler keeps the hand on course: its grasp site
# at most TRANSFER_DIP below CARRY_Z and its axis within TRANSFER_TILT of
# vertical. The way home retraces the transfer and the approach (a joint-space
# move from over the drop straight to RISE passed 2.5 mm from the source
# tote's wall) and must come back to RISE within RETURN_SLACK (rad) of the
# joints the arm rose there with.
TRANSFER_DIP = 0.01
TRANSFER_TILT = np.radians(8)
RETURN_SLACK = 0.2
PLACE_TRIES = 200  # rejection draws per place
GRASP_TRIES = 20  # draws of the moving part's yaw, place and drop per layout
INSTRUCTIONS = (
    "Move the {part} from the blue tote into the grey tote.",
    "Put the {part} into the grey tote.",
    "Take the {part} out of the blue bin and drop it into the grey bin.",
)
# Grasp-site positions (open jaws) for the scene check (the nominal layout).
KEY_POINTS = [
    {
        "name": f"grasp_{name}",
        "side": "left",
        "xyz": [*xy, FLOOR_Z + GRASP_Z + contact_drop(width)],
        "grasp": "top",
        "yaw": HAND_YAW,
        "width": width,
    }
    for name, (_, xy, width, _) in PARTS.items()
] + [
    {
        "name": f"release_{name}",
        "side": "left",
        "xyz": [*DROP_XY, FLOOR_Z + RELEASE + GRASP_Z + contact_drop(width)],
        "grasp": "top",
        "yaw": HAND_YAW,
        "width": width,
    }
    for name, (_, _, width, _) in PARTS.items()
]


def hand(yaw):
    """Top-down hand turned yaw about z (the shared IK's default is yaw 0)."""
    return tt.rot_z(yaw)


def release_open(width):
    """The closure the jaws open to when they let go of a part of width."""
    return 1 - (width + RELEASE_OPEN) / tt.JAW_SPAN


def in_bands(yaw):
    return any(lo - 1e-9 <= yaw <= hi + 1e-9 for lo, hi in HAND_YAWS)


def housing_hand_yaw(psi):
    """The hand yaw in HAND_YAWS that closes the jaws across a housing lying at
    yaw psi (the jaws close either way round: any psi + k pi)."""
    for k in range(-3, 3):
        if in_bands(psi + k * np.pi):
            return float(psi + k * np.pi)
    raise ValueError(f"no hand yaw grasps a housing at {np.degrees(psi):.1f} deg")


def draw_hand_yaw(rng):
    """A hand yaw uniform over HAND_YAWS."""
    spans = np.array([hi - lo for lo, hi in HAND_YAWS])
    u = rng.uniform(0, spans.sum())
    for (lo, _), span in zip(HAND_YAWS, spans, strict=True):
        if u <= span:
            return float(lo + u)
        u -= span
    return float(HAND_YAWS[-1][1])


def turn_yaws(start, end, step=TURN_STEP):
    """Hand yaws from start (excluded) to end (included), at most step apart."""
    n = max(1, int(np.ceil(abs(end - start) / step - 1e-9)))
    return [start + (end - start) * (i + 1) / n for i in range(n)]


def turn_path(kin, start, yaw):
    """The left arm's joints turning the hand in place at READY from the default
    posture's yaw (0, at the arm pose start) to yaw (clockwise, in HAND_YAWS),
    each step chained to the last as a straight move chains its waypoints.
    Turned the other way the chain jumps (2.1 rad in one step); in one
    joint-space move the hand swung 24 cm, out over the source tote."""
    path, q = [], np.asarray(start, float)
    for step in turn_yaws(0.0, yaw):
        q = kin.ik("left", READY, q, near=q, rotation=hand(step))
        path.append(q)
    return path


def joint_path(kin, start, end, n=21):
    """Grasp-site positions and frames along the joint-space move from start
    to end (Scene.move with a joint goal interpolates linearly)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    return [kin.fk("left", start + s * (end - start)) for s in np.linspace(0, 1, n)]


def check_transfer(kin, start, end):
    """Raises RuntimeError if the joint-space transfer from start to end lets
    the held part dip or tilt (TRANSFER_DIP, TRANSFER_TILT)."""
    for p, R in joint_path(kin, start, end):
        if p[2] < CARRY_Z - TRANSFER_DIP or R[2, 2] < np.cos(TRANSFER_TILT):
            raise RuntimeError(f"the transfer dips or tilts the hand at {p.round(3)}")


def transfer_goal(kin, lifted, poses):
    """The arm pose over the drop that the transfer moves to in joint space:
    the IK solution nearest the pose the part was lifted with."""
    return kin.ik("left", poses["over"], lifted, rotation=poses["rotation"])


def key_poses(base, yaw, width, drop):
    """Grasp-site targets (world, m) for moving the part whose base is at base
    (x, y, z) to drop (x, y) in the grey tote, the hand at yaw. The hand stops
    contact_drop(width) above the grasp point, so the dropping pads close
    there. "release" is where the expert plans to let go; it lowers the part
    by its measured lowest point instead."""
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
        [*part["xy"], FLOOR_Z],
        spec["hand_yaw"],
        PARTS[spec["target"]][2],
        spec["drop"],
    )


def line(kin, q, target, rotation):
    """kin.line's waypoints from arm pose q to target, as Scene.move(linear=True)
    runs them; raises RuntimeError where they jump (JUMP)."""
    path = kin.line("left", q, target, rotation=rotation)
    if len(path) > 1 and np.abs(np.diff(path, axis=0)).max() > JUMP:
        raise RuntimeError(f"the line to {np.round(target, 3).tolist()} jumps")
    return path


def plan(kin, spec):
    """Raises RuntimeError where a pose of the expert's path, or a 1 cm
    waypoint of its straight moves, has no IK solution inside the limits or
    jumps branch (line), where the transfer would let the part dip or tilt
    (check_transfer), or where the way home would end on another branch: from
    the default posture the turn at READY, up to RISE, the pick, the transfer,
    the set-down, back over the part and to RISE."""
    poses = episode_poses(spec)
    turned = poses["rotation"]
    start = kin.ik("left", READY, np.zeros(7))
    q = turn_path(kin, start, spec["hand_yaw"])[-1]
    rise = line(kin, q, RISE, turned)[-1]
    q = rise
    for key in ("above", "grasp", "lift"):
        q = line(kin, q, poses[key], turned)[-1]
    lifted = q
    over = transfer_goal(kin, lifted, poses)
    check_transfer(kin, lifted, over)
    q = over
    for key in ("release", "clear"):
        q = line(kin, q, poses[key], turned)[-1]
    check_transfer(kin, q, lifted)
    back = line(kin, lifted, RISE, turned)[-1]
    if np.abs(back - rise).max() > RETURN_SLACK:
        raise RuntimeError("the way back to RISE ends on another branch")


def rect(centre, half, yaw):
    """The corners (4 x 2) of a rectangle of half sizes half (along its own x,
    y) turned yaw about z, centred at centre."""
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[u, v] for u in (-half[0], half[0]) for v in (-half[1], half[1])])
    return np.asarray(centre, float) + local @ np.array([[c, s], [-s, c]])


def inside(points, tote, gap=WALL_GAP):
    """Whether every point lies gap or more inside the tote's inner floor."""
    d = np.abs(np.asarray(points, float) - np.asarray(tote, float))
    return bool(np.all(d <= np.asarray(INNER) - gap))


def part_inside(name, xy, yaw, tote, gap=WALL_GAP):
    """Whether part name at xy and yaw keeps gap from the tote's walls."""
    if name == "flange":
        return inside([xy], tote, gap + FLANGE_R)
    return inside(rect(xy, HOUSING_HALF, yaw), tote, gap)


def footprint(name, xy, yaw):
    """Points around part name's outline at xy and yaw."""
    if name == "flange":
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        return np.asarray(xy, float) + FLANGE_R * np.c_[np.cos(a), np.sin(a)]
    return rect(xy, HOUSING_HALF, yaw)


def in_view(name, xy, yaw):
    """Whether part name at xy and yaw lies wholly in both D435 images
    (VIEW_X_MIN, VIEW_Y_EDGE)."""
    p = footprint(name, xy, yaw)
    edge = VIEW_Y_EDGE[0] + VIEW_Y_EDGE[1] * (p[:, 0] - 0.30)
    return bool(np.all(p[:, 0] >= VIEW_X_MIN) and np.all(p[:, 1] <= edge))


def hand_inside(name, xy, yaw, tote, use):
    """Whether the hand, turned yaw at xy and open for use ("grasp" or
    "release") on part name, keeps WALL_GAP from the tote's walls below its
    rim."""
    return inside(rect(xy, (HAND_U[use][name], HAND_V), yaw), tote)


def cover(name, xy, yaw):
    """Points covering part name's footprint at xy and yaw: outline and inside."""
    if name == "flange":
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ring = np.c_[np.cos(a), np.sin(a)]
        return np.asarray(xy, float) + np.vstack(
            [[0, 0], 0.5 * FLANGE_R * ring, FLANGE_R * ring]
        )
    u, v = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 7))
    local = np.c_[u.ravel() * HOUSING_HALF[0], v.ravel() * HOUSING_HALF[1]]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(xy, float) + local @ np.array([[c, s], [-s, c]])


def clear_of_jaws(target, xy, yaw, other, other_xy, other_yaw):
    """Whether the other part lies clear of the hand closing on target at xy
    with yaw (FINGER_REACH, FINGER_V, WALL_GAP), and apart from the target."""
    d = cover(other, other_xy, other_yaw) - np.asarray(xy, float)
    c, s = np.cos(yaw), np.sin(yaw)
    u, v = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    reach = PARTS[target][2] / 2 + FINGER_REACH + WALL_GAP
    if np.any((np.abs(u) < reach) & (np.abs(v) < FINGER_V + WALL_GAP)):
        return False
    gap = np.hypot(*np.subtract(other_xy, xy))
    return bool(gap >= REACH_R[target] + REACH_R[other] + WALL_GAP)


def nominal():
    """The layout bench_check builds: no shifts, the flange to move."""
    return {
        "target": "flange",
        "totes": {"source": list(SOURCE), "dest": list(DEST)},
        "parts": {n: {"xy": list(p[1]), "yaw": 0.0} for n, p in PARTS.items()},
        "hand_yaw": HAND_YAW,
        "drop": list(DROP_XY),
        "instruction": INSTRUCTIONS[0].format(part=PARTS["flange"][3]),
    }


def draw_totes(rng):
    dest = np.asarray(DEST) + rng.uniform(-1, 1, 2) * TOTE_SHIFT
    gap = rng.uniform(*GAP)
    source = np.array(
        [
            SOURCE[0] + rng.uniform(-1, 1) * TOTE_SHIFT[0],
            dest[1] + 2 * TOTE_HALF[1] + gap,
        ]
    )
    return source, dest


def draw_yaw(rng, name):
    if name == "flange":
        return float(rng.uniform(-np.pi, np.pi))
    return float(rng.uniform(*HOUSING_YAW))


def draw_place(rng, tote, ok):
    """A point uniform on the tote's inner floor for which ok(xy) holds, or
    None after PLACE_TRIES draws."""
    for _ in range(PLACE_TRIES):
        xy = np.asarray(tote) + rng.uniform(-1, 1, 2) * INNER
        if ok(xy):
            return [float(v) for v in xy]
    return None


def draw(rng):
    """One layout, or None where no place was found: the totes, then the
    moving part's yaw, the hand's, its place and its drop together (up to
    GRASP_TRIES times), then the other part."""
    source, dest = draw_totes(rng)
    target = str(rng.choice(list(PARTS)))
    other = next(n for n in PARTS if n != target)
    yaws = {other: draw_yaw(rng, other)}
    for _ in range(GRASP_TRIES):
        yaws[target] = draw_yaw(rng, target)
        if target == "flange":
            hand_yaw = draw_hand_yaw(rng)
        else:
            hand_yaw = housing_hand_yaw(yaws["housing"])
        xy = draw_place(
            rng,
            source,
            lambda p, hand_yaw=hand_yaw: (
                part_inside(target, p, yaws[target], source)
                and hand_inside(target, p, hand_yaw, source, "grasp")
                and in_view(target, p, yaws[target])
            ),
        )
        # The part turns with the hand: it is let go at its yaw.
        drop = xy and draw_place(
            rng,
            dest,
            lambda p, hand_yaw=hand_yaw: (
                p[1] >= DROP_LINE[0] + DROP_LINE[1] * (p[0] - 0.37)
                and part_inside(target, p, yaws[target], dest)
                and hand_inside(target, p, hand_yaw, dest, "release")
                and in_view(target, p, yaws[target])
            ),
        )
        if drop:
            break
    else:
        return None
    other_xy = draw_place(
        rng,
        source,
        lambda p: (
            part_inside(other, p, yaws[other], source)
            and clear_of_jaws(target, xy, hand_yaw, other, p, yaws[other])
            and in_view(other, p, yaws[other])
        ),
    )
    if other_xy is None:
        return None
    places = {target: xy, other: other_xy}
    wording = INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))]
    return {
        "target": target,
        "totes": {
            "source": [float(v) for v in source],
            "dest": [float(v) for v in dest],
        },
        "parts": {n: {"xy": places[n], "yaw": yaws[n]} for n in PARTS},
        "hand_yaw": hand_yaw,
        "drop": drop,
        "instruction": wording.format(part=PARTS[target][3]),
    }


def sample_episode(seed, kin=None, attempts=100):
    kin = kin or tt.kinematics(tt.SAMPLE_MARGIN)
    rng = np.random.default_rng(seed)
    for attempt in range(1, attempts + 1):
        spec = draw(rng)
        if spec is None:
            continue
        try:
            plan(kin, spec)
        except RuntimeError:
            continue
        return {
            "task": NAME,
            "version": VERSION,
            "seed": int(seed),
            "draws": attempt,
            **spec,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def build(spec=None):
    """The scene for spec (default: the nominal layout)."""
    spec = spec or nominal()
    totes = spec["totes"]
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    pa.add(
        root,
        "small_tote",
        "source_tote",
        pos=(*totes["source"], BENCH_Z),
        variant_name="blue",
    )
    pa.add(
        root,
        "small_tote",
        "dest_tote",
        pos=(*totes["dest"], BENCH_Z),
        variant_name="grey",
    )
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
