"""Lidded tote: lift the tote's lid, take the hex blank out, set it on the
station plate, and close the lid.

The tote stands on the workbench (top at BENCH_Z, 0.90 m) at x = 0.445,
y = 0.12, hinged at the back (x = 0.547, z = 1.052). Its hinge friction, with
prop.json's solreffriction/solimpfriction for static holding, holds the lid
wherever it is left, so one arm can open it, pick and close it in turn. The
hex blank lies near the tote's front wall (x = 0.395; the wall's inner face is
at 0.356 at the floor), turned 30 degrees so that two of its flats face y. The
hand takes it top-down with its jaws along y: along x the housing (about 81 mm
either side of the jaws) would meet the front wall, and at x = 0.38 the jaws
along y left it about 1 mm. The station plate lies to the left at y = 0.40.
The left arm does it all; the pull tab points are pinched from the front, so
they carry no grasp probe, and the hex's probe ignores the lid, open by then.

The expert pinches the 60 mm pull tab across its width, the jaws along the
hinge axis: the pinch then turns like a second hinge, so the hand need not
turn with the lid while it carries the tab round the arc, 0 to 110 degrees.
Turned with the lid, the hand reached only about 50 degrees from the front,
and pitched towards the robot it met the torso. On wheel_robot_26.8.16_3 no
single pitch reaches the whole arc, so the hand pitches by the lid's angle
(beta_at: 85 degrees below the horizontal on the shut lid to 60 from 50
degrees on). The pads close 1 mm beyond the tab's tip (PINCH_OUT), where no
turn of the lid brings the fingertips (FINGER_REACH) to its front skirt. The
lid is swung open past the point where its weight balances (84 degrees), to
OPEN_TO, so it stays open without the hinge friction (PhysX has none: there
it falls back onto its 110-degree stop), and it is closed from wherever it
stands.

Every hand orientation is one of a family: jaws along the hinge axis, the
wrist camera on top, fingers from straight down (the hex blank, the camera
over the tote: turned the other way it met the front rim) to the pinch.
Turning between them only pitches the hand about the jaw axis. Every move is
a chain of IK waypoints, each solved from the last (lines 1 cm apart, the
arcs likewise, the turns in place 3 degrees apart), so the whole episode
stays on one IK branch; the arm goes home the way it came. The expert runs
these same chains, from the arm's present goal.

A seed fixes the tote's place and turn, the hex blank's place and turn and
the flange's place in it, the plate's place, the spot on it and the
wording. Every draw is IK-checked along the expert's moves, the lid's arcs
in both directions included, and redrawn until it plans.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from robots.franzi.gripper_linkage import contact_drop
from robots.franzi.scene import build as robot_scene
from sim import prop_assets as pa
from sim import tabletop as tt
from sim.workcell import BENCH_Z, industrial_cell

NAME = "lidded_tote"
INSTRUCTION = "Open the tote, take the hex blank out and put it on the station plate, then close the lid."
TOTE = np.array([0.445, 0.12, BENCH_Z])
PLATE = (0.36, 0.40)
PLATE_HALF = (0.075, 0.10)  # the station plate's 150 x 200 mm top
PLATE_TOP = BENCH_Z + 0.010
FLOOR_Z = BENCH_Z + 0.004
RIM_Z = BENCH_Z + 0.150
# The hinge (prop frame): the lid opens about +y, its front rising towards +x.
HINGE_REL = np.array([0.102, 0.0, 0.152])
HINGE = TOTE + HINGE_REL
PULL_TAB = TOTE + [-0.110363, 0.0, 0.161937]
HEX = (0.395, 0.13)
HEX_YAW = np.radians(30)  # two flats face y, across the jaws
HEX_FLATS = 0.027
HEX_CORNER = 0.0156  # the blank's corners from its axis (27 mm across flats)
# The top-down hand on the hex: jaws along y, the wrist camera over the tote.
JAWS_ALONG_Y = -np.pi / 2
# The closed pads' centre over a part's base: the fingertips (tt.TIP past the
# pads) 2 mm above it.
GRASP_Z = round(tt.TIP + 0.002, 6)
SET_GAP = 0.003
FLANGE = (0.40, 0.21)  # the other part in the tote, which stays there
# The lid's range (prop.json) and the angles the expert uses. Past BALANCE the
# lid's weight opens it further (its centre of mass, 104 mm in front of the
# hinge and 11 mm above it, is then behind the hinge).
LID_RANGE = (0.0, 1.9198621771937625)
BALANCE = np.radians(84)
OPEN_TO = np.radians(105)
OPENED = np.radians(90)  # the lid counts as open from here
CLOSED = 0.03  # rad: the lid counts as shut below this
BACK_OFF = np.radians(25)  # open-jawed travel down the arc, off the tab
# The pull tab (prop frame, lid shut): 60 mm along y, standing out 18 mm from
# the front skirt's outer face (x = -0.102) at the roof's top (z = 0.165),
# 3 mm thick and pointing 10 degrees down.
TAB_WIDTH = 0.060
TAB_ROOT = np.array([-0.102, 0.0, 0.165])
TAB_DOWN = np.radians(10)
TAB_DIR = np.array([-np.cos(TAB_DOWN), 0.0, -np.sin(TAB_DOWN)])
TAB_NORMAL = np.array([-np.sin(TAB_DOWN), 0.0, np.cos(TAB_DOWN)])
# Gripper geometry (measured on the present finger hulls; re-measure for a new
# gripper): the farthest point of a finger from the closed pads' centre in the
# plane of the pads, 13.4 mm on along the fingers and 10.5 mm to the side.
FINGER_REACH = 0.017
# The pinch point along the tab from the skirt's face, 1 mm past the tab's
# tip: FINGER_REACH and 2 mm, so no turn of the lid between the pads brings a
# fingertip to the skirt. At 15 mm the fingertips' corners grazed it, up to
# 0.8 mm deep, with the lid turned about 40 degrees against the hand.
PINCH_OUT = round(FINGER_REACH + 0.002, 6)
PINCH = TAB_ROOT + PINCH_OUT * TAB_DIR - 0.0015 * TAB_NORMAL
PINCH_RADIUS = float(np.hypot(*(PINCH - HINGE_REL)[[0, 2]]))
# The pinching hand's fingers below the horizontal, by lid angle: BETA_SHUT on
# the shut lid, rising evenly to BETA_OPEN at BETA_UNTIL and held from there
# (wheel_robot_26.8.16_3: no one pitch reaches the whole arc; near the shut lid
# only 75-90 degrees reach the tab clear of the torso, from 40 degrees only
# 30-70).
BETA_SHUT = np.radians(85)
BETA_OPEN = np.radians(60)
BETA_UNTIL = np.radians(50)
# The open hand comes down onto the shut lid's tab from this far above: from
# PRE back along its fingers its forearm met the torso.
PINCH_RISE = 0.05
PINCH_SQUEEZE = 0.006  # 4.5 N a pad; the lid needs about 3 N at the tab
HEX_SQUEEZE = 0.006  # the 149 g blank
# Gripper geometry (measured, like FINGER_REACH): the jaw blocks reach this
# far from the grasp site along the jaws, open for the 27 mm blank (59 mm; 47
# mm closed on it), from the pads' height to 26 mm above the site.
JAW_REACH = 0.060
# The carry, in the tote's frame (its left wall's inner face at y 0.136, the
# rim's outer edge at 0.150, the blank's corners 15.6 mm from its axis): the
# grasp site CARRY_Z high over the tote (the blank's base 15 mm over the rim;
# the top-down hand reaches no higher over the hex), the blank rising inside
# the left wall up to LIFT_Y, crossing the rim at CARRY_Z until the jaw blocks
# are past its edge (RIM_Y), then down on a slant, which keeps them outside,
# to PLACE_HIGH over its place. The rim point was first set by the blank
# alone (y 0.178): rising back from a place near the tote, the open jaw block
# grazed the rim (seed 3, 0.03 mm).
CARRY_Z = BENCH_Z + 0.20
LIFT_Y = 0.112
OVER_RIM_Y = 0.172  # the blank's corners 6 mm past the rim's edge
RIM_Y = round(0.150 + JAW_REACH + 0.008, 6)
RIM_PASS_Z = CARRY_Z - 0.015  # the top-down hand reaches the rim point no higher
PLACE_HIGH = 0.035
# The blank first rises straight up this far, its base 10 mm over a flange
# beside it, before slanting towards the tote's left wall.
RAISE = 0.03
TURN_Z = BENCH_Z + 0.15  # the hand turns between the pinch and top-down here
# The set-down: the hex's lowest point this far above the plate when the jaws
# open, measured just before (under PhysX a carried part turns 2-4 degrees
# about the pads instead of creeping).
RELEASE_GAP = 0.0005
TURN_STEP = np.radians(3)  # rotation between the waypoints of a turn in place
# Back over the hex blank's place from the plate, the arm steps onto the
# joints it turned down with there (a joint-space move: the hand stays put
# while the elbow swivels) before the closing plays the opening backwards;
# this far apart at most (rad, any joint).
REJOIN = 0.7
# The reach to the tab. The arm swings its forearm from pointing forwards
# (at READY, the jaws along x) to pointing inwards (the jaws along the hinge,
# the elbow out in front), so part of it is a joint-space move: the hand rises
# RISE straight up from READY, fingers down, swings joint-space to STAGE (in
# the tote's frame) before the point over the tab and goes on in a line. A
# line all the way from READY (its wrist roll on its bound from midway) was
# stretched to 38 s by the rate limiter's kinks; swung from READY's height,
# the hand swept the lid's front edge with the tote 20-30 mm nearer and turned
# (up to 16 mm deep). Raised first, the swing keeps 40 mm off the tote and the
# trunk over seeds 0-39; the hand rises no higher than about 0.13 fingers down.
RISE = 0.12
STAGE = np.array([-0.06, 0.08, 0.0])
READY = tt.READY["left"]
# Seeded variation (version 4, user request 2026-10-27: positions by
# centimetres and turns by tens of degrees; m or rad, ranges as (low, high),
# HEX_TURN and SPOT_RANGE as half-widths about the centre).
# The tote's shift from TOTE and its turn. Past them the plan fails: with
# the tote 60 mm nearer the robot or turned -15 degrees, the IK chain from
# READY to over the tab changes branch; 40 mm farther (or 20 mm farther, 40
# mm to the right and turned -10 degrees), the tab on the lid's 110-degree
# stop, where PhysX leaves the lid, is out of reach; turned +25 degrees and
# 40 mm nearer, the arc. Farther along y than 40 mm the plate rarely fits
# beside the tote (-70 mm plans).
TOTE_SHIFT = ((-0.04, 0.02), (-0.04, 0.04))
TOTE_TURN = (np.radians(-10), np.radians(20))
# The hex blank in the tote's frame: x from where the head D435 still sees
# about 100 px of it over the front wall with the lid open (the tote at its
# far shift and turned 15-20 degrees; 12 px at -0.056, where the hand's front
# keeps 5 mm off the wall; the body D435 never sees into the tote's front
# half) to where its wrist camera (93 mm behind) keeps 15 mm off the back
# wall; y as far as the open jaw blocks (59 mm out along the jaws) keep 5 mm
# off the side walls. Its turn about HEX_YAW (flats every 60 degrees), which
# the jaws follow: at 30 degrees the arm's IK fails over most of the tote.
HEX_X = (-0.044, -0.020)
HEX_Y = (-0.072, 0.072)
HEX_TURN = np.radians(20)
# The flange beside the blank, on the side with more room: its centre at
# least FLANGE_GAP from the blank's along y (the open finger beside the blank
# keeps 5 mm off its rim), its rim 5 mm off the walls.
FLANGE_GAP = 0.075
FLANGE_Y = 0.101
FLANGE_X = (-0.050, 0.050)
# The station plate's shift on the bench from PLATE, and the hex's spot on it
# around its centre (10 and 15 mm inside the success test's margins). The
# plate's edge keeps 2 mm off the tote's base, the spot keeps the open jaw
# blocks RIM_CLEAR past the tote's rim and lies at least SPOT_X from the
# robot: the head D435 sees the bench from about x 0.30 at the plate's height
# (the blank set down at 0.28 showed 4-16 px, from 0.32 on 400-510), and in
# the default posture the left arm hides the plate from the body D435.
PLATE_SHIFT = ((-0.03, 0.03), (-0.02, 0.04))
SPOT_RANGE = (0.045, 0.065)
RIM_CLEAR = 0.015
SPOT_X = 0.32
INSTRUCTIONS = (
    INSTRUCTION,
    "Lift the tote's lid, move the hex blank onto the station plate and shut the lid.",
    "Take the hex blank out of the lidded tote, place it on the plate and close the tote.",
    "Open the lid, put the hex blank from the tote on the station plate, then close the lid again.",
)
# The largest joint step allowed between 1 cm waypoints (rad): a bigger one is
# an IK branch change, which would swing the hand off its path.
MAX_STEP = 0.15


def tab_at(angle):
    """The pull tab's position with the lid opened by angle (rad)."""
    rel = PULL_TAB - HINGE
    c, s = np.cos(angle), np.sin(angle)
    return HINGE + [rel[0] * c + rel[2] * s, 0.0, -rel[0] * s + rel[2] * c]


def site(base_z):
    """The open hand's grasp-site height for the hex resting at base_z."""
    return round(base_z + GRASP_Z + contact_drop(HEX_FLATS), 4)


KEY_POINTS = [
    {
        "name": "pull_tab_closed",
        "side": "left",
        "xyz": [round(float(v), 4) for v in PULL_TAB],
        "grasp": "any",
    },
    {
        "name": "pull_tab_open_60",
        "side": "left",
        "xyz": [round(float(v), 4) for v in tab_at(np.radians(60))],
        "grasp": "any",
    },
    {
        "name": "grasp_hex",
        "side": "left",
        "xyz": [*HEX, site(FLOOR_Z)],
        "grasp": "top",
        "yaw": JAWS_ALONG_Y,
        "width": HEX_FLATS,
        "ignore": ["tote_lid_part"],
    },
    {
        "name": "place_on_plate",
        "side": "left",
        "xyz": [*PLATE, site(PLATE_TOP + SET_GAP)],
        "grasp": "top",
        "yaw": JAWS_ALONG_Y,
        "width": HEX_FLATS,
    },
]


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def tote_frame(tote):
    """The tote's origin (base centre, world) and rotation for tote (spec's
    {"xy", "yaw"})."""
    return np.array([*tote["xy"], BENCH_Z]), tt.rot_z(tote["yaw"])


def lid_point(tote, angle, point):
    """World position of point (prop frame, lid shut) with the lid opened by
    angle (rad)."""
    origin, R = tote_frame(tote)
    return origin + R @ (HINGE_REL + rot_y(angle) @ (np.asarray(point) - HINGE_REL))


def hand(tote, beta):
    """The family of hand rotations: jaws closing along the tote's hinge axis,
    the wrist camera on top, fingers pointing forward and beta below the
    horizontal (pi/2: straight down, the camera towards the hinge)."""
    return tt.rot_z(tote["yaw"]) @ rot_y(beta - np.pi / 2) @ tt.rot_z(-np.pi / 2)


def beta_at(angle):
    """The pinching hand's pitch with the lid at angle (rad)."""
    share = min(max(angle / BETA_UNTIL, 0.0), 1.0)
    return BETA_SHUT + (BETA_OPEN - BETA_SHUT) * share


def pinch_rotation(tote, angle):
    return hand(tote, beta_at(angle))


def pinch_site(tote, angle):
    """The open hand's grasp site that closes the pads on the pull tab with
    the lid at angle: contact_drop short of the pinch point along the fingers."""
    approach = -pinch_rotation(tote, angle)[:, 2]
    return lid_point(tote, angle, PINCH) - contact_drop(TAB_WIDTH) * approach


def pinch_above(tote):
    """Where the open hand starts down onto the shut lid's tab."""
    return pinch_site(tote, 0.0) + [0.0, 0.0, PINCH_RISE]


def arc_angles(start, end, step=0.01):
    """Lid angles from start to end (rad, start excluded) that move the pinch
    point at most step (m) apart."""
    n = max(1, int(np.ceil(abs(end - start) * PINCH_RADIUS / step)))
    return np.linspace(start, end, n + 1)[1:]


def chain(kin, q, targets, body=None):
    """Arm joint waypoints through targets [(position, rotation)], each IK
    chained to the last. Raises RuntimeError where one has no solution or the
    arm changes IK branch (a joint step over MAX_STEP)."""
    path = []
    for position, rotation in targets:
        nxt = kin.ik("left", position, q, body, near=q, rotation=rotation)
        if np.abs(nxt - q).max() > MAX_STEP:
            raise RuntimeError(f"IK branch change at {np.round(position, 3)}")
        path.append(nxt)
        q = nxt
    return np.array(path)


def arc(kin, q, tote, start, end, body=None):
    """Joint waypoints carrying the pinch point round the hinge from lid
    angle start to end, the hand pitched as beta_at says."""
    return chain(
        kin,
        q,
        [
            (pinch_site(tote, a), pinch_rotation(tote, a))
            for a in arc_angles(start, end)
        ],
        body,
    )


def line(kin, q, target, rotation, body=None, step=0.01):
    """Joint waypoints to target on the straight line from the hand's present
    place (FK of q), every step metres, at rotation."""
    begin = kin.fk("left", q, body)[0]
    n = max(1, int(np.ceil(np.linalg.norm(np.subtract(target, begin)) / step)))
    points = np.linspace(begin, target, n + 1)[1:]
    return chain(kin, q, [(p, rotation) for p in points], body)


def turn(kin, q, start, end, body=None):
    """Joint waypoints turning the hand in place from rotation start to end
    (the grasp site held where FK of q puts it), TURN_STEP apart."""
    position = kin.fk("left", q, body)[0]
    rotations = Rotation.from_matrix(np.array([start, end]))
    angle = np.linalg.norm((rotations[0].inv() * rotations[1]).as_rotvec())
    n = max(1, int(np.ceil(angle / TURN_STEP)))
    steps = Slerp([0.0, 1.0], rotations)(np.linspace(0, 1, n + 1)[1:])
    return chain(kin, q, [(position, r) for r in steps.as_matrix()], body)


def close_start(angle):
    """Where the open hand leaves the arc after letting go of the tab of a lid
    opened to angle (and where it rejoins it to close the lid)."""
    return min(OPEN_TO, angle) - BACK_OFF


def hand_yaw(hex_yaw, tote_yaw=0.0):
    """The top-down hand's yaw that closes the jaws on two flats of a hex
    blank turned hex_yaw (flats every 60 degrees), nearest JAWS_ALONG_Y in
    the tote's frame (its housing then lies along the side walls)."""
    turn = (hex_yaw - tote_yaw - HEX_YAW + np.pi / 6) % (np.pi / 3) - np.pi / 6
    return JAWS_ALONG_Y + tote_yaw + turn


def hex_poses(tote, base, hex_yaw, spot):
    """Grasp-site targets (world) for moving the hex blank whose base is at
    base (x, y, z) to spot (x, y) on the plate, and the hand's rotation. The
    blank rises inside the tote towards its left wall ("lift"), crosses the
    rim CARRY_Z high ("rim") and comes down onto the plate on a slant
    ("high"): carried straight across at CARRY_Z, the top-down hand's wrist
    pitch reached its limit past y 0.35 and the arm swung 0.2 rad per cm
    (wheel_robot_26.8.16_3). The expert re-measures the place target before
    it sets the blank down."""
    above = GRASP_Z + contact_drop(HEX_FLATS)
    grasp = np.asarray(base, float) + [0, 0, above]
    place = np.array([*spot, PLATE_TOP + RELEASE_GAP + above])
    origin, R = tote_frame(tote)
    rel = R.T @ (grasp - origin)  # the blank in the tote's frame

    def in_tote(x_rel, y_rel, z):
        return [*(origin + R @ [x_rel, y_rel, 0.0])[:2], z]

    rim_x = (rel[0] + (R.T @ (place - origin))[0]) / 2
    return {
        "rotation": tt.rot_z(hand_yaw(hex_yaw, tote["yaw"])),
        "above": [grasp[0], grasp[1], TURN_Z],
        "grasp": grasp,
        "raise": grasp + [0, 0, RAISE],
        "lift": in_tote(rel[0], LIFT_Y, CARRY_Z),
        "over_rim": in_tote((rel[0] + rim_x) / 2, OVER_RIM_Y, CARRY_Z),
        "rim": in_tote(rim_x, RIM_Y, RIM_PASS_Z),
        "high": [place[0], place[1], place[2] + PLACE_HIGH],
        "place": place,
    }


def reach_path(kin, home, tote, body=None):
    """The way from home to over the tab, (rise, stage, path): the line rise
    straight up from home's place (fingers down), a joint-space move from its
    end to the joints stage, with the hand STAGE (tote frame) before the point
    over the tab, and the line path to that point. stage and path come from
    one IK chain planned from home (the hand turned at READY, the line to over
    the tab, and back from there to the stage), so the line is the one the
    closing replays backwards."""
    begin = kin.fk("left", home, body)[0]
    rise = line(kin, home, begin + [0.0, 0.0, RISE], np.eye(3), body)
    shut = pinch_rotation(tote, 0.0)
    turned = turn(kin, home, np.eye(3), shut, body)[-1]
    above = line(kin, turned, pinch_above(tote), shut, body)[-1]
    stage = pinch_above(tote) + tt.rot_z(tote["yaw"]) @ STAGE
    back = line(kin, above, stage, shut, body)
    return rise, back[-1], np.vstack([back[-2::-1], above])


def closing(kin, done, tote, angle, body=None):
    """The way back, {move: joint path}, from the opening's paths done
    ({move: (start joints, path)}), with the lid standing at angle: back over
    the tab and the lid the way the hand came (its turn, its line and the
    back-off arc played backwards, up the arc to the tab), round the arc to
    shut (the opening's arc backwards; past OPEN_TO, an arc of its own first),
    and off the tab and home the same way. The arm thus ends on the joints it
    started from, whatever branch the carry drifted to (a closing chained on
    its own ended up to 1.1 rad from them with the tote turned)."""

    def back(name):
        start, path = done[name]
        return np.vstack([path[-2::-1], start])

    lid_end = done["open_lid"][1][-1]
    out = {
        "tilt_to_lid": back("turn_down"),
        "reach_lid": back("over_hex"),
        "up_arc": back("back_off"),
    }
    close = back("open_lid")
    if angle > OPEN_TO:
        past = arc(kin, lid_end, tote, OPEN_TO, angle, body)
        out["up_arc"] = np.vstack([out["up_arc"], past])
        close = np.vstack([past[-2::-1], lid_end, close])
    out["close_lid"] = close
    out["back_away"] = back("align_tab")
    out["return"] = back("reach_tab")
    return out


def plan(kin, spec, home=None):
    """The expert's path as joint waypoints, {move: (start joints, path)},
    chained from home (default: the default posture's joints). Raises
    RuntimeError where a waypoint has no IK solution inside the limits, the
    arm changes IK branch, or the hand, back over the hex blank's place, is
    further than REJOIN from the joints it turned down with there. The closing
    (closing()) is planned with the lid where the opening leaves it (OPEN_TO,
    held by the hinge friction) and on its open stop (where it falls without
    that friction, as in PhysX)."""
    tote = spec["tote"]
    shut = pinch_rotation(tote, 0.0)
    home = kin.ik("left", READY, np.zeros(7)) if home is None else home
    done = {}

    def run(name, start, path):
        done[name] = (start, path)
        return path[-1]

    # Open: over the tab (up, joint-space to the stage, then a line), down
    # onto it, round the arc and off it.
    rise, stage, approach = reach_path(kin, home, tote)
    q = run("rise", home, rise)
    done["reach_joint"] = (q, stage[None])
    q = run("reach_tab", stage, approach)
    q = run("align_tab", q, line(kin, q, pinch_site(tote, 0.0), shut))
    q = run("open_lid", q, arc(kin, q, tote, 0.0, OPEN_TO))
    q = run("back_off", q, arc(kin, q, tote, OPEN_TO, close_start(OPEN_TO)))
    # The hex blank onto the plate, and back over its place the way it came.
    poses = hex_poses(
        tote, [*spec["hex"]["xy"], FLOOR_Z], spec["hex"]["yaw"], spec["spot"]
    )
    down, held = poses["rotation"], pinch_rotation(tote, close_start(OPEN_TO))
    q = run("over_hex", q, line(kin, q, poses["above"], held))
    q = run("turn_down", q, turn(kin, q, held, down))
    for key in ("grasp", "raise", "lift", "over_rim", "rim", "high", "place"):
        q = run(key, q, line(kin, q, poses[key], down))
    for key in ("place", "high", "rim", "over_rim"):
        start, path = done[key]
        q = run(key + "_back", q, np.vstack([path[-2::-1], start]))
    q = run("above_back", q, line(kin, q, poses["above"], down))
    if np.abs(q - done["turn_down"][1][-1]).max() > REJOIN:
        raise RuntimeError("back over the hex, the arm is too far from its joints")
    for angle in (OPEN_TO, LID_RANGE[1]):
        tag = "" if angle == OPEN_TO else "_stop"
        for name, path in closing(kin, done, tote, angle).items():
            done[name + tag] = (None, path)
    return done


def nominal():
    """The layout bench_check builds: no jitter."""
    return {
        "tote": {"xy": [float(v) for v in TOTE[:2]], "yaw": 0.0},
        "hex": {"xy": list(HEX), "yaw": float(HEX_YAW)},
        "flange": {"xy": list(FLANGE)},
        "plate": list(PLATE),
        "spot": list(PLATE),
        "instruction": INSTRUCTION,
    }


def fits(spec):
    """Whether a drawn layout keeps the plate off the tote's base, the open
    jaw blocks at the spot past the tote's rim and the spot in the head
    camera's view (SPOT_X)."""
    origin, R = tote_frame(spec["tote"])
    corners = np.array(
        [[x, y, 0] for x in (-0.0888, 0.0888) for y in (-0.1388, 0.1388)]
    )
    base_y = (origin + corners @ R.T)[:, 1].max()
    if base_y > spec["plate"][1] - PLATE_HALF[1] - 0.002:
        return False
    if spec["spot"][0] < SPOT_X:
        return False
    spot_rel = R.T @ (np.r_[spec["spot"], BENCH_Z] - origin)
    return spot_rel[1] - JAW_REACH - RIM_CLEAR > 0.150


def draw_once(rng):
    """One layout: the tote's place and turn, the blank's place and turn in
    it, the flange beside it, the plate's place, the spot on the plate and the
    wording."""
    yaw = float(rng.uniform(*TOTE_TURN))
    xy = TOTE[:2] + [rng.uniform(*axis) for axis in TOTE_SHIFT]
    origin, R = tote_frame({"xy": xy, "yaw": yaw})

    def world(x_rel, y_rel):
        return [float(v) for v in (origin + R @ [x_rel, y_rel, 0.0])[:2]]

    hex_rel = (rng.uniform(*HEX_X), rng.uniform(*HEX_Y))
    side = 1.0 if hex_rel[1] < 0 else -1.0  # towards the wider space
    flange_y = rng.uniform(*sorted((hex_rel[1] + side * FLANGE_GAP, side * FLANGE_Y)))
    plate = PLATE + np.array([rng.uniform(*axis) for axis in PLATE_SHIFT])
    spot = plate + rng.uniform(-1, 1, 2) * SPOT_RANGE
    return {
        "tote": {"xy": [float(v) for v in xy], "yaw": yaw},
        "hex": {
            "xy": world(*hex_rel),
            "yaw": float(HEX_YAW + yaw + rng.uniform(-1, 1) * HEX_TURN),
        },
        "flange": {"xy": world(rng.uniform(*FLANGE_X), flange_y)},
        "plate": [float(v) for v in plate],
        "spot": [float(v) for v in spot],
        "instruction": INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))],
    }


def draw(rng):
    """draw_once() until fits() holds (a geometric redraw, not a planning
    one)."""
    while True:
        spec = draw_once(rng)
        if fits(spec):
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
            # 2: one subtask text per action of the left arm; 3: the
            # wheel_robot_26.8.16_3 model, chained turns, pitch by lid angle;
            # 4: wider layouts (the tote turned, the plate moved)
            "version": 4,
            "seed": int(seed),
            "draws": attempt,
            **spec,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def build(spec=None):
    """The scene for spec (default: the nominal layout)."""
    spec = spec or nominal()
    tote = spec["tote"]
    half = tote["yaw"] / 2
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    pa.add(
        root,
        "lidded_tote",
        "tote",
        pos=(*tote["xy"], BENCH_Z),
        quat=(np.cos(half), 0, 0, np.sin(half)),
    )
    pa.add(root, "station_plate", "plate", pos=(*spec.get("plate", PLATE), BENCH_Z))
    half = spec["hex"]["yaw"] / 2
    pa.add(
        root,
        "hex_blank",
        "hex",
        pos=(*spec["hex"]["xy"], FLOOR_Z + 0.0005),
        quat=(np.cos(half), 0, 0, np.sin(half)),
        free=True,
    )
    pa.add(
        root,
        "flange_part",
        "flange",
        pos=(*spec["flange"]["xy"], FLOOR_Z + 0.0005),
        free=True,
    )
    report["task"] = {
        "name": NAME,
        "instruction": spec["instruction"],
        "key_points": KEY_POINTS,
    }
    return root, report
