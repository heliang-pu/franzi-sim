"""Parcel sorting: each arm takes the parcels on its half of the belt and puts
each into the bin whose zone letter matches the parcel's label.

The belt conveyor runs along y in front of the robot on a 100 mm plinth, so the
belt top is at 0.90 m and its centre at x = 0.525. The four 140 × 75 × 60 mm
parcels lie across the belt, centred on it, so that the 45-degree grasp
(sim.tabletop.grasp_rotation), its jaws closing along y, takes them across
their 75 mm width with the guide rails clear of the fingers, 55 mm over the
parcel's base (GRASP_Z). At this height that grasp reaches past x 0.55. The
belt is still in this version. Four zone bins (200 × 240 × 100 mm, rims at
0.80 m) stand in front of the belt at x = 0.30: A (outer) and B (inner) on the
left, C (inner) and D (outer) on the right, their zone signs just above the
back rims, below the parcels' path. Two parcels lie on each half of the belt,
and each arm puts its two into its own side's bins by their labels, setting
each down on the floor with the gripper inside the bin.

An episode (sample_episode) draws which of its side's two zones each parcel
carries, which arm goes first and the order in which each arm takes its two,
where each parcel lies along and across the belt and how it is turned (up to
12 degrees off lying across it, and a half turn round for half of them), where
the bin row stands, the spot in each bin, how far the hand turns over the
inner bins, and the wording (the ranges and what limits them are with
Y_INNER below). The expert (run.py) moves one arm at a time (user,
2026-10-25: an episode splits into single-arm subtasks): the first arm picks
and places its two parcels and returns to its start pose, then the other arm,
while the idle arm holds still. Per parcel: a joint-space move to a staging
pose over the parcel, straight in along the 45-degree approach, straight up
off the belt to CARRY_Z, turned there in place to the bin's azimuth, straight
back off the belt and on over the bin, straight down into it, set down and let
go, then straight up. plan() checks every draw the way the expert moves (IK at
every 1 cm waypoint, joint jumps, the time a parcel is held, the hand's height
on the joint-space moves, the turns' drift, where the elbow lies over each
parcel, the carried parcel's clearance from those still on the belt) and
sample_episode redraws until it plans.

Until 2026-10-25 the parcels lay along the belt (the 45-degree jaws would have
had to span 140-200 mm), a pouch only 4.5 mm proud of the guide rail could not
be picked, the 158 mm wide, 150 mm deep bins left the gripper housing and the
wrist camera block no room, and the three bins shared B at the midline, where
neither arm reaches the bin's centre.
"""

from types import SimpleNamespace

import numpy as np
import pinocchio as pin

from robot_kinematics import controller_rates
from sim import prop_assets as pa
from sim import tabletop as tt
from sim.build_task import ARM_JOINTS
from sim.build_task import build as robot_scene
from sim.episode import RATE_HEADROOM, Scene
from sim.gripper_linkage import contact_drop
from sim.workcell import block, industrial_cell, stand

NAME = "parcel_sorting"
INSTRUCTION = "Sort each parcel into the bin whose zone letter matches its label."
CONVEYOR = (0.6475, 0.0)  # origin; belt centre x 0.525
PLINTH_Z = 0.10
BELT_Z = 0.80 + PLINTH_Z
BELT_X = 0.525
PARCEL = "parcel_box_s"
PARCEL_W = 0.075  # across the jaws
PARCEL_L = 0.140  # along the belt's width (x)
ACROSS = np.pi / 2  # the parcel prop's yaw on the belt: its long side along x
# The grasp point over a parcel's base, 5 mm under its top. At 50 mm the
# gripper's link 5, which swings in with the jaws, met the parcel's top edge
# 0.8 mm deep as the pads touched; from 55 mm it stays 3.5 mm clear.
GRASP_Z = 0.055
BIN_X, BIN_BASE_Z = 0.30, 0.70  # bins are 100 mm deep: rims at 0.80
BIN_FLOOR = BIN_BASE_Z + 0.001
BIN_INNER = (0.099, 0.119)  # half the bin's inner floor (1 mm walls)
BINS = {"A": 0.375, "B": 0.125, "C": -0.125, "D": -0.375}
# The bin body's centre in the sort_bin prop frame (its origin is the centre of
# the whole bottom bounding box, sign and entry lip included).
BIN_OFFSET = pa.spec("sort_bin")["functional_frame_offset_m"]
# The nominal layout, (zone, y on the belt): each arm's two parcels go to its
# two bins, and neither to the bin nearest to it. Parcels 0 and 1 are the left
# arm's, 2 and 3 the right's.
PARCELS = [("B", 0.30), ("A", 0.12), ("D", -0.12), ("C", -0.30)]
SIDE_ZONES = {"left": ("A", "B"), "right": ("C", "D")}
SIDE_OF = {zone: side for side, zones in SIDE_ZONES.items() for zone in zones}
SLOTS = {"left": (0, 1), "right": (2, 3)}
# The jaws' gap around a parcel while open: OPEN_ROOM per side (the 75 mm
# parcel leaves no room for grip_open's 12 mm in the 95 mm jaws, tt.JAW_SPAN);
# each finger stays 4.5 mm off its open limit.
OPEN_ROOM = 0.0055
OPEN = PARCEL_W + 2 * OPEN_ROOM
# Closure past contact: the 250 g parcels get 8 mm, which saturates the finger
# servos' 6 N, as for the RoboSyn lid and the loaded tray.
SQUEEZE = 0.008
LIFT = 0.03  # first lift off the belt, before the carry
# A carried parcel's base, 20 mm over the guide rails (0.93) and 10 mm under
# the tops of the parcels still on the belt: it backs off the belt BACK
# straight toward the robot (its centre to x 0.40) before it heads for its bin,
# and the sampler checks that it passes the other parcel by MIN_GAP. Carried
# 40 mm over those parcels (base at 1.00), the hand over an outer bin is so
# close to the shoulder that the IK drifted onto the elbow-up branch; the kinked
# joint path then stretched the 4.9 s move to 24 s under the controller's rate
# limits, and the parcel crept 5 mm down in the jaws meanwhile.
CARRY_Z = BELT_Z + 0.05
BACK = 0.125
MIN_GAP = 0.010
# A parcel's lowest corner over the bin floor at the end of the carry (RELEASE)
# and its base when let go (SET): the expert lowers it to RELEASE, then sets it
# down, levelled, onto the floor in a phase where it may touch it. Let go 3.9
# mm over the floor, a parcel landed 0.80 mm deep (the audit stops at 1 mm);
# let go 1 mm over it but tilted 1.8 degrees, 0.64 mm. A held parcel creeps
# down and tilts in the jaws under MuJoCo (0.27 mm and 0.1 degree a second) and
# pivots 2-4 degrees about the pads under PhysX, so the expert measures it
# before each of the two moves (run.place_target, run.level_target); set down
# onto the floor rather than over it, a parcel the jaws let pivot is levelled
# by the floor itself.
RELEASE, SET = 0.005, 0.0
# The hand's azimuth over each bin (sim.tabletop.grasp_rotation's psi). Over
# the inner bins B and C, at x 0.30 and 0.125 off the midline, the 45-degree
# hand put its wrist into the torso's hull over the bin (12-26 mm; 3 mm with
# the 2026-10-25 URDF's 12 mm longer gripper link) and onto the thigh at the
# release; turned 20 degrees toward the midline, the wrist swings outboard and
# keeps 28 mm or more off the trunk (43.7 mm over the bin, 51.8 mm at the
# release). The outer bins keep psi 0 (turned more than 10 degrees, their
# release does not plan).
PLACE_PSI = {"A": 0.0, "B": -np.radians(20), "C": np.radians(20), "D": 0.0}
# The spot in a bin: the parcel's centre DROP_DX beyond the bin's centre in x,
# which keeps the hand 22 mm or more off the bin's walls (8-20 mm at the
# centre). The turned parcels in B and C come within 10 mm of the far wall,
# the others 14 mm.
DROP_DX = {"A": 0.015, "B": 0.010, "C": 0.010, "D": 0.015}
# The layout's randomization (sample_episode, version 4; version 3 jittered the
# parcels 4 and 8 mm and 3 degrees and the release spot 10 mm in y):
# - Along the belt each side's inner parcel lies Y_INNER off the midline, its
#   outer one Y_OUTER, their centres at least MIN_PITCH apart: the belt's ends
#   (an outer parcel turned 12 degrees keeps its corners 3 mm inside them), the
#   gripper housing's 85 mm either side of the grasp, and the reach by the
#   midline.
# - Each parcel's yaw is up to YAW_SPREAD off lying across the belt, and half
#   of them lie a half turn round (label upside down, the same grasp). The
#   guide rails, 160 mm apart, leave a 140 mm parcel turned 12 degrees 3.7 mm
#   per side.
# - Across the belt each parcel lies as far off the centre as its yaw leaves
#   room, RAIL_ROOM from both guide rails.
# - The bin row (all four bins and their stands) moves up to BIN_SHIFT along
#   y. Moved 5 mm away from the robot, a parcel set down in a turned inner bin
#   met the conveyor's side beam (BEAM_X), which overhangs the bins' far walls;
#   moved 5 mm toward it, with a 15-degree turn over bin B, the wrist came
#   within 1.9 mm of the torso's hull.
# - The spot in each bin moves up to DROP_SPREAD (x, y) about its nominal one,
#   the hand's housing 17 mm or more off the bin's walls.
# - Over the inner bins the hand (and so the parcel set down) turns INNER_PSI
#   toward the midline: under 20 degrees the wrist nears the torso, past 30
#   degrees the hand nears the bin's near wall.
Y_INNER, Y_OUTER, MIN_PITCH = (0.08, 0.17), (0.25, 0.32), 0.14
YAW_SPREAD = np.radians(12)
RAIL_ROOM = 0.003
RAIL_GAP = 0.160  # between the guide rails' inner faces, centred on BELT_X
BIN_SHIFT = 0.010
DROP_SPREAD = (0.002, 0.020)
INNER_PSI = (np.radians(20), np.radians(30))
# The conveyor's near side beam, whose face toward the robot (x) overhangs the
# bins' far walls at 0.85-0.89 m, and the least room a parcel set down keeps
# from it.
BEAM_X = CONVEYOR[0] - 0.2505
BEAM_ROOM = 0.005
# The hanging arm (the URDF zero posture), from which tt.ready_pose and
# tt.start_home solve their IK too, and one of the two start postures of the
# staging pose's IK (stage_pose).
HANGING = np.zeros(7)
# The least distance (m) of the elbow off the midline at a staging pose: at
# 0.105 the elbow sat in the torso's hull, from 0.235 it kept 27 mm off it.
ELBOW_OUT = 0.18
MAX_STEP = 0.2  # rad: the largest joint step between 1 cm waypoints a plan takes
# The staging pose over each parcel (stage_pose), and how low the grasp site may
# dip on a joint-space move: SWING_BELT_Z over the belt (beyond its near guide
# rail at RAIL_X), 70 mm over the parcels' tops, keeps the fingertips, which
# reach about 25 mm under the grasp site in the 45-degree grasp, 40 mm clear of
# them; SWING_Z elsewhere 12 cm over the bins' rims (at 0.95 the swing from
# the default posture failed by 0.1 mm in a third of the corner layouts, over
# the bins' near edge). A turn in place goes in steps of TURN_STEP, each moving
# the site at most TURN_DRIFT off its point.
STAGE_UP = 0.08
RAIL_X = 0.445
SWING_BELT_Z, SWING_Z = 1.03, 0.92
TURN_DRIFT = 0.015
TURN_STEP = np.radians(10)
# The seconds the expert asks of each straight move (Scene.move stretches them
# to the joint speed and the controller's rate limits), and the longest a move
# holding a parcel may take: the parcel creeps in the jaws meanwhile.
SECONDS = {
    "pregrasp": 1.5,
    "grasp": 1.5,
    "lift": 1.0,
    "raise": 1.0,
    "back": 1.5,
    "over": 2.0,
    "place": 2.5,
    "set": 1.0,
}
HELD_SECONDS = 8.0
# What Scene.stretch reads of a scene: the 500 Hz step, a move starting on a
# control tick, and the arm rate limits Scene derives from the controller's.
_RATES = controller_rates()
_TIMING = SimpleNamespace(
    m=SimpleNamespace(opt=SimpleNamespace(timestep=0.002)),
    step_number=0,
    along=Scene.along,
    max_rate=_RATES["velocity"] * _RATES["default_ratio"] * RATE_HEADROOM,
    max_accel=_RATES["acceleration"] * _RATES["default_ratio"] * RATE_HEADROOM,
)
INSTRUCTIONS = (
    INSTRUCTION,
    "Put every parcel from the conveyor into the bin of the zone on its label.",
    "Read each parcel's label and drop it into the bin with the same zone letter.",
    "Move the four parcels off the belt into the zone bins their labels name.",
)


def rotation(side, psi):
    """The 45-degree grasp at azimuth psi: a parcel's yaw (its offset from
    lying across the belt) on the belt, so the jaws close on its two long
    faces, and PLACE_PSI over its bin."""
    return tt.grasp_rotation(side, psi)


def folded(yaw):
    """A parcel's yaw off lying across the belt, folded into +-90 degrees: a
    box turned half a turn is grasped the same way."""
    return float((yaw + np.pi / 2) % np.pi - np.pi / 2)


def key_poses(side, base, yaw, psi, drop):
    """Grasp-site targets (world, m) for moving the parcel whose base (its
    bottom centre) is at base with yaw (folded) into a bin, its centre to drop
    (x, y), the hand at azimuth psi there. The hand stops contact_drop short
    along the approach, so the
    dropping pads close GRASP_Z over the base. It takes the parcel with the
    "pick" rotation, lifts it straight up, raises it to CARRY_Z, turns there
    to the bin's "put" rotation, backs it off the belt, carries it over the bin
    and lowers it straight down to RELEASE over the floor ("place"), then sets
    it down to SET ("set"); the expert corrects both for the parcel's creep in
    the jaws."""
    pick, put = rotation(side, yaw), rotation(side, psi)
    drop_short = contact_drop(PARCEL_W)
    grasp = np.asarray(base, float) + [0, 0, GRASP_Z] + drop_short * pick[:, 2]
    place = np.array([*drop, BIN_FLOOR + RELEASE + GRASP_Z]) + drop_short * put[:, 2]
    carry = CARRY_Z + GRASP_Z + drop_short * pick[2, 2]
    return {
        "pick": pick,
        "put": put,
        "pick_psi": float(yaw),
        "put_psi": float(psi),
        "stage": grasp + tt.PRE * pick[:, 2] + [0, 0, STAGE_UP],
        "pregrasp": grasp + tt.PRE * pick[:, 2],
        "grasp": grasp,
        "lift": grasp + [0, 0, LIFT],
        "raise": np.array([grasp[0], grasp[1], carry]),
        "back": np.array([grasp[0] - BACK, grasp[1], carry]),
        "over": np.array([place[0], place[1], carry]),
        "place": place,
        "set": place - [0, 0, RELEASE - SET],
    }


def parcel_poses(spec, k):
    p = spec["parcels"][k]
    side = SIDE_OF[p["zone"]]
    base = [*p["xy"], BELT_Z]
    psi = spec["place_psi"][p["zone"]]
    return side, key_poses(side, base, folded(p["yaw"]), psi, p["drop"])


def elbow(kin, side, q, body=None):
    """The elbow's position (world, m) at arm joints q: its joint's origin."""
    pin.forwardKinematics(kin.model, kin.data, kin.config(side, q, body))
    joint = kin.model.getJointId(f"{side}_{ARM_JOINTS[3]}_joint")
    return kin.data.oMi[joint].translation + kin.base


def outboard(side, xyz):
    """How far (m) a point lies off the midline on side's own side."""
    return xyz[1] if side == "left" else -xyz[1]


def stage_pose(kin, side, poses, body=None):
    """Arm joints STAGE_UP over the pregrasp, hand in the "pick" rotation, elbow
    down and out: of the IK solutions found from the hanging arm and from the
    default posture, the one whose elbow lies farther outboard. A joint-space
    move ends here, and straight moves go on down to the parcel.
    - Chained along straight moves from the default posture instead, the IK
      sometimes kept the arm's elbow-up branch (shoulder pitch 1.5 rad) all the
      way to the parcel, then jumped 2.4 rad on the way into an outer bin.
    - Since the shared IK's seeds moved to wheel_robot_26.8.16_3's convention
      (2026-10-25), the solution nearest the hanging arm swings the elbow in
      (0.105 m off the midline, 94 mm into the torso's hull) at a third of the
      staging poses of the wider layouts; the one from the default posture kept
      it out (0.235 m or more) at all 160 of seeds 0-39."""
    found = []
    for start in (HANGING, home(kin, side)):
        try:
            found.append(
                kin.ik(side, poses["stage"], start, body, rotation=poses["pick"])
            )
        except RuntimeError:
            continue
    if not found:
        raise RuntimeError(f"{side}: no IK solution at the staging pose")
    return max(found, key=lambda q: outboard(side, elbow(kin, side, q, body)))


def turn_poses(kin, side, poses, q, body=None):
    """Arm joints turning the hand in place at "raise" from q (the "pick"
    azimuth) to the bin's "put" azimuth, in steps of at most TURN_STEP: each
    the IK solution nearest the last. Turned in one joint-space move, a turn of
    up to 42 degrees (yaw on the belt and the inner bins' azimuth) drifted the
    hand up to 36 mm off its point."""
    a, b = poses["pick_psi"], poses["put_psi"]
    steps = max(1, int(np.ceil(abs(b - a) / TURN_STEP - 1e-9)))
    path = []
    for i in range(1, steps + 1):
        hand = rotation(side, a + (b - a) * i / steps)
        q = kin.ik(side, poses["raise"], q, body, rotation=hand)
        path.append(q)
    return path


def home(kin, side):
    """Arm joints of the default posture (tt.start_home): READY, fingers down."""
    return kin.ik(side, tt.READY[side], HANGING)


def move_seconds(path, seconds):
    """How long Scene.move takes along a joint path asked for seconds at
    tt.SPEED: timed by joint travel, then stretched until its 25 Hz goals keep
    to the controller's rate limits (Scene.stretch, on a stand-in scene)."""
    arc = np.r_[0, np.cumsum(np.abs(np.diff(path, axis=0)).max(1))]
    seconds = max(seconds, 1.875 * arc[-1] / tt.SPEED)
    return Scene.stretch(_TIMING, {"arm": path}, {"arm": arc}, seconds)


def straight(kin, side, q, key, poses, rotation, body=None, held=False):
    """The last joints of the expert's straight move from q to poses[key]
    (Scene.move(linear=True)). Raises RuntimeError where a 1 cm waypoint has no
    IK solution, the arm jumps more than MAX_STEP between two, or (held) the
    move would take longer than HELD_SECONDS."""
    path = kin.line(side, q, poses[key], body, rotation=rotation)
    step = np.abs(np.diff(path, axis=0)).max()
    if step > MAX_STEP:
        raise RuntimeError(f"{side}: a {step:.2f} rad joint step to {key}")
    if held and move_seconds(path, SECONDS[key]) > HELD_SECONDS:
        raise RuntimeError(f"{side}: the move to {key} takes too long")
    return path[-1]


def footprint(xy, yaw, grow=0.0):
    """Corners of a parcel's footprint (x, y) at yaw off lying across the belt."""
    c, s = np.cos(yaw), np.sin(yaw)
    half = (PARCEL_L / 2 + grow, PARCEL_W / 2 + grow)
    return [
        (xy[0] + c * u - s * v, xy[1] + s * u + c * v)
        for u, v in ((half[0], half[1]), (half[0], -half[1]), (-half[0], -half[1]))
        + ((-half[0], half[1]),)
    ]


def apart(a, b):
    """Whether two convex polygons (corner lists) are disjoint: a separating
    axis among their edges' normals."""
    for poly in (a, b):
        for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1], strict=True):
            n = (y0 - y1, x1 - x0)
            pa = [n[0] * x + n[1] * y for x, y in a]
            pb = [n[0] * x + n[1] * y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return True
    return False


def check_beam(spec, k):
    """Raises RuntimeError where parcel k, set down at its spot and turned to
    its bin's azimuth, reaches within BEAM_ROOM of the conveyor's side beam."""
    p = spec["parcels"][k]
    psi = abs(spec["place_psi"][p["zone"]])
    half = PARCEL_L / 2 * np.cos(psi) + PARCEL_W / 2 * np.sin(psi)
    if p["drop"][0] + half > BEAM_X - BEAM_ROOM:
        raise RuntimeError(f"parcel {k} would pass under the conveyor's beam")


def check_clear(spec, k, others, poses):
    """Raises RuntimeError where parcel k, carried from "raise" (turning there)
    back off the belt and over its bin, comes within MIN_GAP of a parcel in
    others, still on the belt: its base passes under their tops."""
    p = spec["parcels"]
    psi = spec["place_psi"][p[k]["zone"]]
    short = contact_drop(PARCEL_W) * poses["put"][:2, 2]
    yaw = folded(p[k]["yaw"])
    yaws = [yaw + (psi - yaw) * t for t in np.linspace(0, 1, 5)]
    centres = [(p[k]["xy"], yaw) for yaw in yaws]
    for a, b in (("raise", "back"), ("back", "over")):
        for t in np.linspace(0, 1, 25):
            site = poses[a][:2] + (poses[b][:2] - poses[a][:2]) * t
            centres.append((site - short, psi))
    for other in others:
        rest = footprint(p[other]["xy"], p[other]["yaw"])
        for xy, yaw in centres:
            if not apart(footprint(xy, yaw, MIN_GAP), rest):
                raise RuntimeError(f"parcel {k} passes parcel {other} too close")


def swept(kin, side, q0, q1, body=None, n=24):
    """Grasp-site positions along the joint-space move from q0 to q1."""
    return np.array(
        [kin.fk(side, q0 + (q1 - q0) * t, body)[0] for t in np.linspace(0, 1, n + 1)]
    )


def check_swing(kin, side, q0, q1, body=None):
    """Raises RuntimeError where the hand dips on a joint-space move between
    poses over the belt or the bins: under SWING_Z, or under SWING_BELT_Z over
    the belt (x beyond its near guide rail)."""
    for x, _, z in swept(kin, side, q0, q1, body):
        if z < (SWING_BELT_Z if x > RAIL_X else SWING_Z):
            raise RuntimeError(f"{side}: the hand dips to z {z:.3f} at x {x:.3f}")


def check_turn(kin, side, q0, q1, body=None):
    """Raises RuntimeError where a turn in place moves the hand more than
    TURN_DRIFT off its point."""
    path = swept(kin, side, q0, q1, body)
    drift = np.linalg.norm(path - path[0], axis=1).max()
    if drift > TURN_DRIFT:
        raise RuntimeError(f"{side}: the turn in place drifts {drift * 1000:.0f} mm")


def sequence(spec):
    """The parcels in the order the expert moves them, [(side, k)]: the first
    side's two in its order, then the other side's."""
    first = spec["first"]
    sides = (first, next(side for side in tt.SIDES if side != first))
    return [(side, k) for side in sides for k in spec["order"][side]]


def plan(kin, spec):
    """Raises RuntimeError where either arm's path, the expert's (run.py), has
    no IK solution inside the limits at a key pose or a 1 cm waypoint of its
    straight moves, jumps between two waypoints, takes too long with a parcel
    in hand, swings the hand low on a joint-space move, or carries a parcel too
    close past one still on the belt. Per parcel: a joint-space move to the
    staging pose over the parcel, straight down and in to it, straight up off
    the belt and to the carry height, turned there in place, straight back off
    the belt and over the bin, down to the release and up again; after an arm's
    two parcels a joint-space move back to its start pose. The arms take turns
    (sequence), so each arm's path is planned on its own."""
    order = sequence(spec)
    for side in tt.SIDES:
        q = home(kin, side)
        for k in spec["order"][side]:
            _, poses = parcel_poses(spec, k)
            later = order[order.index((side, k)) + 1 :]
            check_clear(spec, k, [j for _, j in later], poses)
            check_beam(spec, k)
            q1 = stage_pose(kin, side, poses)
            if outboard(side, elbow(kin, side, q1)) < ELBOW_OUT:
                raise RuntimeError(f"{side}: the elbow swings in over parcel {k}")
            check_swing(kin, side, q, q1)
            q = q1
            for key in ("pregrasp", "grasp"):
                q = straight(kin, side, q, key, poses, poses["pick"])
            for key in ("lift", "raise"):
                q = straight(kin, side, q, key, poses, poses["pick"], held=True)
            for q1 in turn_poses(kin, side, poses, q):
                check_turn(kin, side, q, q1)
                q = q1
            for key in ("back", "over", "place", "set"):
                q = straight(kin, side, q, key, poses, poses["put"], held=True)
            q = straight(kin, side, q, "over", poses, poses["put"])
        check_swing(kin, side, q, home(kin, side))


def nominal():
    """The layout bench_check builds: PARCELS, no jitter."""
    return {
        "parcels": [
            {
                "zone": zone,
                "xy": [BELT_X, y],
                "yaw": 0.0,
                "drop": [BIN_X + DROP_DX[zone], BINS[zone]],
            }
            for zone, y in PARCELS
        ],
        "bins": {zone: [BIN_X, y] for zone, y in BINS.items()},
        "place_psi": dict(PLACE_PSI),
        "first": "left",
        "order": {side: list(SLOTS[side]) for side in tt.SIDES},
        "instruction": INSTRUCTION,
    }


def rail_room(yaw):
    """How far (m) a parcel at yaw may lie off the belt's centre line in x and
    keep RAIL_ROOM from both guide rails."""
    yaw = abs(folded(yaw))
    half = PARCEL_L / 2 * np.cos(yaw) + PARCEL_W / 2 * np.sin(yaw)
    return RAIL_GAP / 2 - RAIL_ROOM - half


def draw(rng):
    shift = rng.uniform(-1, 1) * BIN_SHIFT
    bins = {zone: [BIN_X, float(y + shift)] for zone, y in BINS.items()}
    place_psi = {
        zone: float(np.sign(psi) * rng.uniform(*INNER_PSI)) if psi else 0.0
        for zone, psi in PLACE_PSI.items()
    }
    parcels = [None] * len(PARCELS)
    order = {}
    for side in tt.SIDES:
        sign = 1.0 if side == "left" else -1.0
        inner = rng.uniform(*Y_INNER)
        outer = rng.uniform(max(Y_OUTER[0], inner + MIN_PITCH), Y_OUTER[1])
        zones = rng.permutation(SIDE_ZONES[side])
        for k, zone in zip(SLOTS[side], zones, strict=True):
            zone = str(zone)
            # PARCELS lists each side's outer slot first for the left arm and
            # its inner one first for the right: |y| tells them apart.
            y = sign * (outer if abs(PARCELS[k][1]) > 0.2 else inner)
            yaw = rng.uniform(-1, 1) * YAW_SPREAD + np.pi * rng.integers(2)
            x = BELT_X + rng.uniform(-1, 1) * rail_room(yaw)
            dx, dy = (rng.uniform(-1, 1) * d for d in DROP_SPREAD)
            parcels[k] = {
                "zone": zone,
                "xy": [float(x), float(y)],
                "yaw": float(yaw),
                "drop": [
                    bins[zone][0] + DROP_DX[zone] + float(dx),
                    bins[zone][1] + float(dy),
                ],
            }
        order[side] = [int(k) for k in rng.permutation(SLOTS[side])]
    first = tt.SIDES[int(rng.integers(len(tt.SIDES)))]
    wording = INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))]
    return {
        "parcels": parcels,
        "bins": bins,
        "place_psi": place_psi,
        "first": first,
        "order": order,
        "instruction": wording,
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
            # 2: one arm at a time, one subtask per pick, place and return
            # (user, 2026-10-25), and the first side drawn. 3: the robot model
            # wheel_robot_26.8.16_3, parcels set down onto the bin floor. 4: the
            # wider layout ranges (user, 2026-10-27), the bin row and the inner
            # bins' azimuth drawn.
            "version": 4,
            "seed": int(seed),
            "draws": attempt,
            **spec,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def key_points():
    """Grasp-site positions (open jaws) of the nominal layout's grasps and
    releases, for the scene check."""
    spec = nominal()
    points = []
    for k, (zone, _) in enumerate(PARCELS):
        side, p = parcel_poses(spec, k)
        points += [
            {
                "name": f"{key}_{k}_{zone}",
                "side": side,
                "xyz": [round(float(v), 4) for v in p[key]],
                "grasp": "forward",
                "psi": round(float(psi), 4),
                "width": PARCEL_W,
            }
            for key, psi in (("grasp", 0.0), ("place", PLACE_PSI[zone]))
        ]
    return points


KEY_POINTS = key_points()


def build(spec=None):
    """The scene for spec (default: the nominal layout)."""
    spec = spec or nominal()
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root, bench=False)
    block(root, "plinth", CONVEYOR, (0.52, 0.84), 0.0, PLINTH_Z)
    pa.add(root, "belt_conveyor", "conveyor", pos=(*CONVEYOR, PLINTH_Z))
    for zone, (x, y) in spec["bins"].items():
        stand(root, f"bin_stand_{zone}", (x, y), (0.18, 0.22), top_z=BIN_BASE_Z)
        pa.add(
            root,
            "sort_bin",
            f"bin_{zone}",
            pos=(x - BIN_OFFSET[0], y - BIN_OFFSET[1], BIN_BASE_Z),
            variant_name=zone,
        )
    for k, p in enumerate(spec["parcels"]):
        half = (ACROSS + p["yaw"]) / 2
        pa.add(
            root,
            PARCEL,
            f"parcel{k}",
            pos=(*p["xy"], BELT_Z + 0.0005),
            quat=(np.cos(half), 0, 0, np.sin(half)),
            free=True,
            variant_name=p["zone"],
        )
    report["task"] = {
        "name": NAME,
        "instruction": spec["instruction"],
        "key_points": KEY_POINTS,
        "parcels": [
            {"body": f"parcel{k}_part", "zone": p["zone"]}
            for k, p in enumerate(spec["parcels"])
        ],
    }
    return root, report
