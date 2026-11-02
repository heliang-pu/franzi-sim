"""CNC tending: open the machine door, take the finished part out of the vise
into the tray, load a raw blank into the vise, close the door and press CYCLE
START.

The machine front faces the robot at x = 0.36, shifted to y = +0.25 so that its
door handle (y 0.03 when closed) and vise (y 0.25) are the left arm's, while
its control panel (CYCLE START at y -0.155) and the parts tray on the stand to
the right are the right arm's. Moving parts between vise and tray therefore
takes a hand-over. The finished part rests on the vise's parallels with the
jaw open; the tray holds three blanks in its near row, its far row is free.
The mode selector starts at MAN, the machine's recommended initial angle,
which sim.prop_assets applies. The machine stands on the floor (no workbench):
its table and the tray stand keep the 0.80 m working height. The grasp probes
of the vise and the right hand's stand key point ignore the door, which the
task opens first.

The parts change hands on a small steel hand-over stand at HANDOVER, between
the robot and the machine, left of the panel (added 2026-10-25 with the
expert). In the air the two grippers cannot hold a 60 mm part at once: each
needs about 37 mm of its height for its tilted pads, and over 324 static
configurations (both hands closed on the part, both arms reaching it) the best
still had the arms 3.4 mm into each other.

Arm-trunk collisions (2026-10-25) shape the grasps. Near the robot's midline a
45-degree hand pointing straight ahead puts its wrist inside the torso's hull:
at the closed door's handle (y 0.03) every pitch from 45 to 70 degrees and
every height on the bar did, by 20-100 mm. So the left hand works the handle
and the stand turned 45 degrees outward (psi -45: wrist out to +y), closing
its jaws diagonally on the bar (the door is pulled half by the pads' normal
force, half by their grip), and the right hand works the stand turned 55
degrees the other way and carries along ROUTE in front of the panel so; in
the vise, over the tray and at the button the hands point (nearly) ahead. They
turn only in place, high, through IK waypoints (turn_path).

Parts are held with the 45-degree grasp: the blank 40 mm over its base, the
finished part on its D30 step, 57 mm over its base, which keeps the hand 17 mm
higher over the vise jaws and over the near-row blanks when it goes into the
tray's far row.

A seed fixes where the fixtures stand (Layout: the machine with its door,
vise and panel, the tray on its stand and its turn, the hand-over stand and its
turn), which blank is loaded, which far-row pocket takes the finished part
(named in the instruction), where the parts lie (the finished part along the
vise jaws, the blanks in their pockets, every part's yaw) and the wording.
Every draw is IK-checked along the expert's straight moves and in-place turns
and redrawn until it plans.

Constants that rest on this gripper's geometry, to re-measure after a gripper
change: PRESS_REACH (measured), FINISHED_Z, DOOR_CLEAR_X, SETTLE, and the
azimuths, ROUTE, STAND_VIA and HOME_VIA (their clearances). SWIVEL_ROLL rests
on the arm's kinematics. All were checked on wheel_robot_26.8.16_3 (version 4).
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from sim import prop_assets as pa
from sim import tabletop as tt
from sim.build_task import ARM_JOINTS, TABLE_Z
from sim.build_task import build as robot_scene
from sim.gripper_linkage import contact_drop
from sim.workcell import industrial_cell, stand

NAME = "cnc_tending"
INSTRUCTION = "Swap the finished part in the vise for a raw blank, close the door and press CYCLE START."
MACHINE = np.array([0.36, 0.25, 0.0])
WORKPIECE = MACHINE + [0.10, 0.0, 0.86]
VISE_FLOOR = WORKPIECE[2]  # the parallels' top
TRAY_XY = (0.37, -0.345)  # the tray and its stand, nominal
TRAY_STAND_SIZE = (0.22, 0.27)
TRAY_TOP = TABLE_Z + 0.03
POCKET_FLOOR = TRAY_TOP - 0.02
# The pockets in the tray's frame: rows POCKET_DX before and behind its
# centre, POCKET_DY along them (Layout turns them with the tray).
POCKET_DX, POCKET_DY = 0.035, (-0.075, 0.0, 0.075)
NEAR_ROW = [(TRAY_XY[0] - POCKET_DX, TRAY_XY[1] + dy) for dy in POCKET_DY]
FAR_ROW = [(TRAY_XY[0] + POCKET_DX, TRAY_XY[1] + dy) for dy in POCKET_DY]
# The far-row pockets by name, from the robot's view (y grows to its left).
POCKETS = ("right", "middle", "left")
HANDLE = MACHINE + [-0.06, -0.22, 0.99]
BUTTON_FACE = MACHINE + [-0.049, -0.405, 1.0]  # CYCLE START's green cap at rest
HANDLE_D = 0.020  # the door handle bar
PART_D = 0.040  # the finished part and the round blanks
# Grasp heights over a part's base and the widths there: the blank's D40 body
# and the finished part's D30 step (gripper geometry: its D40 shoulder, 40 mm
# up, stays 2 mm below the closed pads, which reach 15 mm under their centre).
BLANK_Z, BLANK_W = 0.040, PART_D
FINISHED_Z, FINISHED_W = 0.057, 0.030
# The hand-over stand: a steel top at TABLE_Z between the robot and the
# machine. Its front edge stays DOOR_GAP short of the door (front face at
# DOOR_FRONT_X), which slides over y 0-0.96 just above it (bottom at z 0.79).
HANDOVER = (0.29, 0.04, TABLE_Z)  # a part's base there, nominal
HANDOVER_SIZE = (0.07, 0.08)
DOOR_FRONT_X, DOOR_GAP = 0.33, 0.005
# The machine's right side (y; its panel is at y -0.19 to -0.04), which the
# tray stand's edge keeps MACHINE_GAP from (both nominal).
MACHINE_SIDE_Y, MACHINE_GAP = -0.20, 0.01
# Hand azimuths (sim.tabletop.grasp_rotation's psi). At the handle and the
# stand the hands turn outward (see the docstring). Checked statically along
# the carries: the left hand at -45 keeps its wrist 47 mm off the torso at
# the stand (-30: 3 mm), the right at +55 35 mm (+45: 14 mm, +30: 23 mm in).
HANDLE_PSI = np.radians(-45)
STAND_PSI = {"left": np.radians(-45), "right": np.radians(55)}
# In-place turns go through IK waypoints this far apart (slerp): a turn made
# as one joint-space move swung the right hand 5 cm into the panel.
TURN_STEP = np.radians(5)
# The door: grasped on the bar BAR_Z high, pulled open to OPEN (its travel
# ends at 0.46; from 0.36 the vise is reachable, the hand in the vise then 7 cm
# off the door's edge) and pushed back to 0. The hand comes to the bar and
# leaves it DOOR_PRE along its approach: enough for the fingertips to clear the
# bar, and the arm stays clear of its reach limit (backed off 8 cm at 0.40
# open, it took 5.4 s).
BAR_Z = 0.97
OPEN = 0.38
DOOR_PRE = 0.05
OPEN_MIN = 0.36
CLOSED = 0.005
# The hand pushes the bar to this opening, 4 mm into the closed stop: aimed at
# 0, the door stopped 4.3 mm open (the arm and the pads give a little under
# the door's friction).
SHUT = -0.004
# The hand reaches and leaves the handle on its +y side (along the turned
# approach), so between the handle and the vise it passes behind the bar at
# this grasp-site x (gripper geometry: the bar's front is at x 0.29, the open
# hand's front about 5 cm ahead of the site): swept past at the vise's x, a
# finger pushed the door shut.
DOOR_CLEAR_X = 0.17
BAR_SQUEEZE = 0.006  # closure past contact on the bar (about 4.5 N a finger)
PART_SQUEEZE = 0.005
# A part's base rises this far above the parallels before it leaves the vise
# (jaw plates' top 20 mm above them), and the axis leaves the machine at
# OUT_X before the hand turns.
VISE_LIFT = 0.045
OUT_X = 0.30
# Parts are carried and turned with their base at CARRY_Z: above the near-row
# blanks (top 0.87) and the stand; the hand descends straight onto a support.
CARRY_Z = 0.925
STAND_LIFT = CARRY_Z - HANDOVER[2]
# The right hand's route between the stand and the tray (a part's base xy at
# CARRY_Z) in front of the panel (front x 0.32, y -0.19 to -0.04, from z 0.90
# up), held at its stand azimuth: carrying a part its gripper stays 13 mm off
# the panel, and the empty hand leaving the stand, jaws open, 5 mm (seeds 0-19
# on wheel_robot_26.8.16_3). At the last point, clear of panel and torso, it
# turns to point ahead (it does so in place, as every turn).
ROUTE = ((0.25, -0.02), (0.24, -0.26))
# The right arm's elbow swivel. Its first turn, at the ready point, picks the
# swivel that every later straight move keeps (each 1 cm waypoint continues
# from the last). Solved from the default posture, its shoulder roll came out
# at -1.87 rad, and the line to the stand ran it to -1.69, beside the shoulder
# singularity at -pi/2 (shoulder pitch and yaw axes aligned): the two traded
# 0.15 rad in one waypoint, and the move stretched from 5.5 s to 25 s (seed 0:
# 211 s in all). Seeded with the roll at SWIVEL_ROLL (any seed from -2.9 to
# -1.3 lands on the same pose, roll -2.06), the roll stays between -3.0 and
# -1.96 rad and the move takes 4 s (seed 0: 185 s in all).
SWIVEL_ROLL = {"right": -2.0}
# The empty right hand comes to the stand through this point (grasp site xy,
# at the height above the stand), in the corridor it later leaves by: turned
# out to its stand azimuth, its forward jaw points at the panel's left corner,
# and on the straight line from the ready point it passed 1.5 mm from it (a
# PhysX contact is reported within 0.5 mm); this way it keeps 13 mm, and the
# arm 29 mm off the torso (statically). Points behind the stand at its own y
# put the forearm into the torso.
STAND_VIA = (0.24, -0.02)
# Setting a part down. Every target comes from how the hand holds the part
# now (measured), since the grip differs by engine: in MuJoCo the part creeps
# up to 4 mm down the jaws on a carry, under PhysX it does not, but it pivots
# about the line between the two pads (patch friction at their centres gives
# no torque about it), hanging 2-4 degrees off vertical and swinging when the
# hand moves.
# - Onto the stand (flat): lowered as a carry phase to SETTLE over its set-down
#   point (base SET_GAP over the support), then set_* the rest (lowered in one
#   carried move, the finished part met the stand 0.16 s early in MuJoCo).
# - Into the tray pocket (2 mm radial clearance) and between the vise jaws:
#   lowered (carried) until its base hovers HOVER over the rim, held SETTLE_S
#   so the swing dies down, re-centred on its measured base (carried), then
#   inserted slowly in INSERT_S (the part may touch the walls there). Lowered
#   straight in under PhysX, the finished part swung from 3 to 12 degrees, its
#   base 3.8 mm off centre, and wedged against the pocket's wall (1.0 mm).
# - Before letting go, the part's lowest point is brought to within GAP_MAX of
#   its support. Let go at the planned 3 mm, the finished part dropped onto
#   the stand under PhysX (1.2 mm deep); in MuJoCo the creep had closed the
#   gap to 0.0-1.8 mm.
SETTLE = 0.010
HOVER = 0.012
SETTLE_S = 0.8
INSERT_S = 2.5
GAP_MAX = 0.0005
PART_R = 0.020  # both parts' bottoms (D40)
TRAY_RIM = TRAY_TOP
JAW_TOP = VISE_FLOOR + 0.020  # the vise jaw plates' top
# Where a part sits in the vise when loaded: 5 mm off the fixed jaw's face
# (15 mm off the open moving jaw), room for a blank hanging a few degrees off
# vertical under PhysX.
LOAD_DX = -0.005
# Cycle start: the right hand's closed fingers press it along +x, the hand
# 45 degrees down and turned 30 degrees out (psi +30 keeps the wrist outside
# the torso's hull, 0 put it 4 mm in). PRESS_REACH (gripper geometry,
# measured on the finger meshes): the closed fingers' most forward (+x) point
# from the grasp site, world frame, at that rotation and closure PRESS_GRIP.
# Re-measured on the wheel_robot_26.8.16_3 gripper: unchanged.
PRESS_PSI = np.radians(30)
PRESS_GRIP = 0.95
PRESS_REACH = np.array([0.0269, 0.0053, -0.0134])
PRESS_BACK = 0.03  # approach and retreat this far in front of the cap
PRESS_TOUCH = -0.004  # the closed loop starts with the fingers 4 mm short
PRESS_STEP = 0.001
PRESS_DEPTH = 0.003  # the cap's 4 mm travel: stop pressing at 3 mm
PRESSED = 0.002  # counts as pressed
RELEASED = 0.001
# The left hand's waiting pose (grasp site) while the right one works at the
# stand (the right hand waits at its ready point).
WAIT = {"left": [0.20, 0.24, 1.02]}
# Each arm goes back to its start joints in joint space from here: from the
# wait pose the left hand swung 10 mm into the shut door, which under PhysX
# (no door friction) it dragged 0.11 m open; from here it stays 85 mm clear.
# The right hand comes from the press: its line to (0.10, -0.37) ran the wrist
# pitch onto its limit margin and stretched from 3.3 s to 8.7 s; to this
# point it keeps 0.29 rad of room and 29 mm off the panel, and the joint move
# home 60 mm off everything (statically).
HOME_VIA = {"left": [0.10, 0.37, 1.02], "right": [0.16, -0.25, 1.00]}
READY = tt.READY
# Randomization (version 5, 2026-10-27): each draw moves the fixtures from the
# nominal layout above, uniformly within these ranges (m, rad), and places the
# parts. The cameras and arms set the ranges (seeds 0-39, the head and body
# D435s' object pixels at the default posture).
# - The machine, with its door, vise and panel: MACHINE_SHIFT_X and
#   MACHINE_SHIFT_Y. Closer than 1 cm the hand-over stand, which keeps its
#   distance from the door, would be pushed back out of the body D435's view.
# - The tray on its stand: TRAY_SHIFT_X away from the robot (3 cm closer, the
#   head D435 kept 15% of the near-row blanks, and turned none of one), up to
#   TRAY_SHIFT_Y to the right (its stand's edge keeps MACHINE_GAP from the
#   machine's side, so it moves left only with the machine; further right the
#   body D435 loses the middle blank), turned up to TRAY_YAW. The right hand's
#   azimuth over the tray follows the turn, so its grasps there are the
#   nominal ones turned with the tray.
# - The hand-over stand: DOOR_GAP short of the door, turned or not, but at most
#   STAND_BACK behind its nominal place (the body D435 sees 27% of it there,
#   6% at 4 cm), STAND_SIDE sideways, turned up to STAND_YAW. The parts go on
#   its centre, the hands keep their azimuths. Back and left bring the right
#   wrist to the torso, statically 29 mm off in the nominal layout, 0.7 mm
#   nearer per mm back and 0.55 per mm left (3 cm back and 1.5 cm left it
#   touched): beyond STAND_FREE_BACK, each mm back takes STAND_TRADE mm off
#   its left range, which keeps the corners 6 mm off. Turned out to its stand
#   azimuth, the right hand's forward jaw reaches 42 mm ahead of the stand's
#   centre and 55 mm to the right, beside the panel's left edge (8 cm right
#   of the centre): the stand stays within STAND_PANEL_SLACK of that distance
#   (2.6 cm nearer and 1.5 cm back, the jaw met the panel's corner at 2.9 mm).
# - The finished part VISE_JITTER along the vise jaws (the parallels are 80 mm
#   long), the blanks POCKET_JITTER in their pockets (apothem 22 mm, radius
#   20), every part at any yaw: they are round (32-sided colliders).
MACHINE_SHIFT_X, MACHINE_SHIFT_Y = (-0.01, 0.02), (-0.02, 0.02)
TRAY_SHIFT_X, TRAY_SHIFT_Y, TRAY_YAW = (0.0, 0.04), 0.03, np.radians(15)
STAND_BACK, STAND_SIDE, STAND_YAW = 0.03, (-0.02, 0.015), np.radians(20)
STAND_FREE_BACK, STAND_TRADE, STAND_PANEL_SLACK = 0.02, 1.2, 0.015
VISE_JITTER = 0.015
POCKET_JITTER = 0.0015
INSTRUCTIONS = (
    (
        "Swap the finished part in the vise for a raw blank: put the finished "
        "part in the {pocket} pocket of the tray's back row, then close the "
        "door and press CYCLE START."
    ),
    (
        "Open the machine, move the finished part to the back-row {pocket} "
        "pocket of the tray, load a blank into the vise, close the door and "
        "start the cycle."
    ),
    (
        "Unload the finished part into the tray's {pocket} back pocket, load a "
        "new blank into the vise, shut the door and press the green CYCLE "
        "START button."
    ),
)


def yaw_quat(yaw):
    """MJCF quaternion (w x y z) of a turn by yaw about z."""
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


def turned(yaw, v):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def half_extent(size, yaw):
    """Half extents (x, y) of a size (x, y) rectangle turned by yaw."""
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    hx, hy = size[0] / 2, size[1] / 2
    return np.array([c * hx + s * hy, s * hx + c * hy])


class Layout:
    """Where an episode's fixtures stand (spec keys machine [x, y], tray [x,
    y, yaw] and stand [x, y, yaw]; nominal without them) and the points that
    follow from them."""

    def __init__(self, spec):
        mx, my = spec.get("machine", MACHINE[:2])
        self.shift = np.array([mx - MACHINE[0], my - MACHINE[1], 0.0])
        self.machine = MACHINE + self.shift
        self.workpiece = WORKPIECE + self.shift
        self.handle = HANDLE + self.shift
        self.button_face = BUTTON_FACE + self.shift
        self.door_clear_x = DOOR_CLEAR_X + self.shift[0]
        self.out_x = OUT_X + self.shift[0]
        tx, ty, tyaw = spec.get("tray", (*TRAY_XY, 0.0))
        self.tray = np.array([tx, ty], float)
        self.tray_yaw = float(tyaw)
        self.near_row = [self.pocket(-POCKET_DX, dy) for dy in POCKET_DY]
        self.far_row = [self.pocket(POCKET_DX, dy) for dy in POCKET_DY]
        sx, sy, syaw = spec.get("stand", (*HANDOVER[:2], 0.0))
        self.handover = (float(sx), float(sy), HANDOVER[2])
        self.stand_yaw = float(syaw)
        self.stand_set = (float(sx), float(sy), HANDOVER[2] + tt.SET_GAP)
        # Through the via and along the route the right hand passes in front
        # of the panel: the stand's end of both moves with the stand (never
        # further forward than the door, and kept beside the panel: see
        # STAND_PANEL_SLACK), the tray's end, where the hand turns, keeps its
        # distance from the panel's right side. (Moved left with the machine
        # as well, the via took the right wrist to 1.3 mm from the torso.)
        moved = np.array([sx - HANDOVER[0], sy - HANDOVER[1]])
        self.stand_via = tuple(np.add(STAND_VIA, moved))
        toward = np.minimum(self.shift[:2], 0.0)
        self.route = (tuple(np.add(ROUTE[0], moved)), tuple(np.add(ROUTE[1], toward)))

    def pocket(self, dx, dy):
        return tuple(float(v) for v in self.tray + turned(self.tray_yaw, (dx, dy)))

    def bar(self, q):
        """The handle bar's axis at grasp height with the door open q (m)."""
        return np.array([self.handle[0], self.handle[1] + q, BAR_Z])


def hand(side, psi=0.0):
    return tt.grasp_rotation(side, psi)


def site(side, point, width, psi=0.0):
    """Grasp-site target that puts the closed pads' centre at point: backed
    off along the approach by the pads' drop, as sim.tabletop.pick_place."""
    axis = hand(side, psi) @ [0, 0, 1]
    return np.asarray(point, float) + contact_drop(width) * axis


def door_poses(lay, q0, q1):
    """Left-hand targets for moving the door from q0 to q1 by its handle."""
    rotation = hand("left", HANDLE_PSI)
    axis = rotation @ [0, 0, 1]
    grasp = site("left", lay.bar(q0), HANDLE_D, HANDLE_PSI)
    slid = site("left", lay.bar(q1), HANDLE_D, HANDLE_PSI)
    pregrasp, retreat = grasp + DOOR_PRE * axis, slid + DOOR_PRE * axis
    return {
        "rotation": rotation,
        "behind": [lay.door_clear_x, *pregrasp[1:]],
        "pregrasp": pregrasp,
        "grasp": grasp,
        "slid": slid,
        "retreat": retreat,
        "clear": [lay.door_clear_x, *retreat[1:]],
    }


def part_grip(name):
    """(grasp height over the base, width) of a part."""
    return (FINISHED_Z, FINISHED_W) if name == "finished" else (BLANK_Z, BLANK_W)


def pick_poses(side, base, name, psi, lift):
    """Targets for taking part name, standing at base, straight up off its
    support: above (the pads' centre lift over the grasp), grasp, and the
    rotation."""
    z, width = part_grip(name)
    grasp = site(side, np.asarray(base, float) + [0, 0, z], width, psi)
    return {
        "rotation": hand(side, psi),
        "grasp": grasp,
        "above": grasp + [0, 0, lift],
    }


def place_poses(side, base, name, psi, lift):
    """Targets for setting part name down with its base at base (SET_GAP
    over its support) straight from above."""
    z, width = part_grip(name)
    place = site(side, np.asarray(base, float) + [0, 0, z + tt.SET_GAP], width, psi)
    return {
        "rotation": hand(side, psi),
        "place": place,
        "above": place + [0, 0, lift],
    }


def carry_site(side, xy, name, psi):
    """Grasp site holding part name with its base at xy, CARRY_Z."""
    z, width = part_grip(name)
    return site(side, [*xy, CARRY_Z + z], width, psi)


def vise_take(lay, base):
    """Left-hand targets for taking the finished part (base on the parallels)
    out of the vise and the machine, hand pointing ahead."""
    z, width = part_grip("finished")
    rotation = hand("left")
    axis = rotation @ [0, 0, 1]
    grasp = site("left", np.asarray(base, float) + [0, 0, z], width)
    lift = grasp + [0, 0, VISE_LIFT]
    return {
        "rotation": rotation,
        "pregrasp": grasp + tt.PRE * axis,
        "grasp": grasp,
        "lift": lift,
        "out": lift + [lay.out_x - base[0], 0, CARRY_Z - VISE_FLOOR - VISE_LIFT],
    }


def vise_load(lay, base):
    """Left-hand targets for lowering a blank onto the parallels at base."""
    z, width = part_grip("blank")
    rotation = hand("left")
    place = site("left", np.asarray(base, float) + [0, 0, z + tt.SET_GAP], width)
    above = place + [0, 0, CARRY_Z - VISE_FLOOR]
    return {
        "rotation": rotation,
        "entry": above + [lay.out_x - base[0], 0, 0],
        "above": above,
        "place": place,
        # Up until the open fingers clear the blank's top, then out behind
        # the door's handle bar.
        "rise": place + [0, 0, 0.06],
        "out": [lay.door_clear_x, place[1], place[2] + 0.06],
    }


def press_poses(lay):
    """Right-hand targets for pressing CYCLE START along +x."""
    rotation = hand("right", PRESS_PSI)
    face = np.asarray(lay.button_face, float) - PRESS_REACH
    return {
        "rotation": rotation,
        "back": face + [-PRESS_BACK, 0, 0],
        "touch": face + [PRESS_TOUCH, 0, 0],
    }


def turn_site(side, part_point, name, psi):
    """Grasp site holding part name at part_point (its pads' centre) with the
    hand turned to psi: hands turn about the part they hold."""
    return site(side, part_point, part_grip(name)[1], psi)


def swivel_seed(side, q):
    """The IK seed for the arm's first turn: q with its shoulder roll at
    SWIVEL_ROLL, or None (no seed) for an arm without one."""
    if side not in SWIVEL_ROLL:
        return None
    seed = np.array(q, float)
    seed[ARM_JOINTS.index("shoulder_roll")] = SWIVEL_ROLL[side]
    return seed


def turn_path(kin, side, q, start, target, r0, r1, body=None):
    """Arm joint waypoints that turn the hand from rotation r0 to r1 (slerp,
    TURN_STEP apart) while the grasp site moves straight from start to
    target, each IK solution chained to the last from arm pose q."""
    ends = Rotation.from_matrix(np.array([r0, r1]))
    angle = (ends[1] * ends[0].inv()).magnitude()
    count = max(1, int(np.ceil(angle / TURN_STEP)))
    slerp = Slerp([0.0, 1.0], ends)
    start, target = np.asarray(start, float), np.asarray(target, float)
    path = [np.asarray(q, float)]
    for t in np.linspace(0, 1, count + 1)[1:]:
        rotation = slerp([t]).as_matrix()[0]
        point = start + (target - start) * t
        path.append(
            kin.ik(side, point, path[-1], body, near=path[-1], rotation=rotation)
        )
    return np.array(path)


def plan_paths(spec):
    """Each arm's moves in the expert's order, from the ready pose: ("line",
    target, rotation) for a straight move, ("turn", target, rotation) for a
    turn in place (target None: about the grasp site) and ("joints", None,
    rotation) for one in joint space. The arms end with a joint-space move to
    the ready pose."""
    lay = Layout(spec)
    blank = spec["blank"]
    vise = spec["vise"]
    fin_base = [*vise, VISE_FLOOR]
    load_base = [vise[0] + LOAD_DX, vise[1], VISE_FLOOR]
    stand = lay.handover
    lift = CARRY_Z - stand[2]
    blank_base = [*spec["blanks"][blank], POCKET_FLOOR]
    drop_base = [*lay.far_row[spec["drop"]], POCKET_FLOOR]
    opening = door_poses(lay, 0.0, OPEN)
    closing = door_poses(lay, OPEN, SHUT)
    take = vise_take(lay, fin_base)
    load = vise_load(lay, load_base)
    ls, rs, ts = STAND_PSI["left"], STAND_PSI["right"], lay.tray_yaw
    put_fin = place_poses("left", stand, "finished", ls, lift)
    get_fin = pick_poses("right", stand, "finished", rs, lift)
    into_tray = place_poses("right", drop_base, "finished", ts, CARRY_Z - POCKET_FLOOR)
    get_blank = pick_poses("right", blank_base, "blank", ts, CARRY_Z - POCKET_FLOOR)
    put_blank = place_poses("right", stand, "blank", rs, lift)
    take_blank = pick_poses("left", stand, "blank", ls, lift)
    press = press_poses(lay)
    l0, r0 = hand("left"), hand("right", ts)
    fin_out = [lay.out_x, vise[1], CARRY_Z + FINISHED_Z]  # pads' centre, part out
    fin_turn = [*lay.route[-1], CARRY_Z + FINISHED_Z]
    blank_turn = [*lay.route[-1], CARRY_Z + BLANK_Z]
    blank_out = [lay.out_x, vise[1], CARRY_Z + BLANK_Z]
    left = [
        ("turn", None, opening["rotation"]),
        ("line", opening["pregrasp"], opening["rotation"]),
        ("line", opening["grasp"], opening["rotation"]),
        ("line", opening["slid"], opening["rotation"]),
        ("line", opening["retreat"], opening["rotation"]),
        ("line", opening["clear"], opening["rotation"]),
        ("turn", None, l0),
        ("line", [lay.door_clear_x, vise[1], take["pregrasp"][2]], l0),
        ("line", take["pregrasp"], l0),
        ("line", take["grasp"], l0),
        ("line", take["lift"], l0),
        ("line", take["out"], l0),
        ("turn", turn_site("left", fin_out, "finished", ls), put_fin["rotation"]),
        ("line", put_fin["above"], put_fin["rotation"]),
        ("line", put_fin["place"], put_fin["rotation"]),
        ("line", put_fin["above"], put_fin["rotation"]),
        ("line", WAIT["left"], put_fin["rotation"]),
        ("line", take_blank["above"], take_blank["rotation"]),
        ("line", take_blank["grasp"], take_blank["rotation"]),
        ("line", take_blank["above"], take_blank["rotation"]),
        (
            "line",
            carry_site("left", (lay.out_x, vise[1]), "blank", ls),
            take_blank["rotation"],
        ),
        ("turn", turn_site("left", blank_out, "blank", 0.0), l0),
        ("line", load["above"], l0),
        ("line", load["place"], l0),
        ("line", load["rise"], l0),
        ("line", load["out"], l0),
        ("turn", None, closing["rotation"]),
        ("line", closing["behind"], closing["rotation"]),
        ("line", closing["pregrasp"], closing["rotation"]),
        ("line", closing["grasp"], closing["rotation"]),
        ("line", closing["slid"], closing["rotation"]),
        ("line", closing["retreat"], closing["rotation"]),
        ("line", HOME_VIA["left"], closing["rotation"]),
    ]
    right = [
        ("joints", None, get_fin["rotation"]),
        ("line", [*lay.stand_via, get_fin["above"][2]], get_fin["rotation"]),
        ("line", get_fin["above"], get_fin["rotation"]),
        ("line", get_fin["grasp"], get_fin["rotation"]),
        ("line", get_fin["above"], get_fin["rotation"]),
        *[
            ("line", carry_site("right", xy, "finished", rs), get_fin["rotation"])
            for xy in lay.route
        ],
        ("turn", turn_site("right", fin_turn, "finished", ts), r0),
        ("line", into_tray["above"], r0),
        ("line", into_tray["place"], r0),
        ("line", into_tray["above"], r0),
        ("line", get_blank["above"], r0),
        ("line", get_blank["grasp"], r0),
        ("line", get_blank["above"], r0),
        ("line", carry_site("right", lay.route[-1], "blank", ts), r0),
        ("turn", turn_site("right", blank_turn, "blank", rs), put_blank["rotation"]),
        *[
            ("line", carry_site("right", xy, "blank", rs), put_blank["rotation"])
            for xy in lay.route[-2::-1]
        ],
        ("line", put_blank["above"], put_blank["rotation"]),
        ("line", put_blank["place"], put_blank["rotation"]),
        ("line", put_blank["above"], put_blank["rotation"]),
        *[
            ("line", carry_site("right", xy, "blank", rs), put_blank["rotation"])
            for xy in lay.route
        ],
        ("turn", None, press["rotation"]),
        ("line", press["back"], press["rotation"]),
        ("line", press["touch"] + [PRESS_DEPTH + 0.004, 0, 0], press["rotation"]),
        ("line", press["back"], press["rotation"]),
        ("line", HOME_VIA["right"], press["rotation"]),
    ]
    return {"left": left, "right": right}


def plan(kin, spec):
    """Raises RuntimeError where a key pose, or a 1 cm waypoint of a straight
    move between them, has no IK solution inside the limits."""
    for side, moves in plan_paths(spec).items():
        q = kin.ik(side, READY[side], np.zeros(7))
        here, now = kin.fk(side, q)
        for kind, target, rotation in moves:
            if kind == "joints":  # a joint-space move to a new hand rotation
                seed = swivel_seed(side, q)
                q = kin.ik(side, here, q, near=seed, rotation=rotation)
            elif kind == "turn":
                end = here if target is None else np.asarray(target, float)
                q = turn_path(kin, side, q, here, end, now, rotation)[-1]
                here = end
            else:
                q = kin.line(side, q, target, rotation=rotation)[-1]
                here = np.asarray(target, float)
            now = rotation


def nominal():
    """The layout bench_check builds: fixtures in place, no jitter, the middle
    blank and pocket."""
    return {
        "machine": [float(v) for v in MACHINE[:2]],
        "tray": [*TRAY_XY, 0.0],
        "stand": [*HANDOVER[:2], 0.0],
        "blank": "blank1",
        "drop": 1,
        "vise": [float(WORKPIECE[0]), float(WORKPIECE[1])],
        "blanks": {
            f"blank{k}": [float(v) for v in xy] for k, xy in enumerate(NEAR_ROW)
        },
        "yaws": {},
        "instruction": INSTRUCTIONS[0].format(pocket=POCKETS[1]),
    }


def draw(rng):
    """One layout: fixtures moved within the ranges above (each clear of the
    machine and its door by construction), then the parts placed."""
    shift = np.array([rng.uniform(*MACHINE_SHIFT_X), rng.uniform(*MACHINE_SHIFT_Y)])
    tray_yaw = rng.uniform(-1, 1) * TRAY_YAW
    # The tray stand's left edge MACHINE_GAP short of the machine's side.
    tray_y = MACHINE_SIDE_Y + shift[1] - MACHINE_GAP
    tray_y -= half_extent(TRAY_STAND_SIZE, tray_yaw)[1]
    tray = [
        TRAY_XY[0] + rng.uniform(*TRAY_SHIFT_X),
        tray_y - rng.uniform(0, 1) * TRAY_SHIFT_Y,
        tray_yaw,
    ]
    # The hand-over stand: sideways first (within STAND_PANEL_SLACK of its
    # distance from the panel), then back from the door (front edge DOOR_GAP
    # short of it) by at most STAND_BACK, less as far as it went left.
    stand_yaw = rng.uniform(-1, 1) * STAND_YAW
    side = rng.uniform(max(STAND_SIDE[0], shift[1] - STAND_PANEL_SLACK), STAND_SIDE[1])
    front = DOOR_FRONT_X + shift[0] - DOOR_GAP
    front -= half_extent(HANDOVER_SIZE, stand_yaw)[0]
    most = min(STAND_BACK, STAND_FREE_BACK + (STAND_SIDE[1] - side) / STAND_TRADE)
    back = max(0.0, front - (HANDOVER[0] - most))
    stand = [front - rng.uniform(0, 1) * back, HANDOVER[1] + side, stand_yaw]
    machine = MACHINE[:2] + shift
    lay = Layout({"machine": machine, "tray": tray, "stand": stand})
    drop = int(rng.integers(len(FAR_ROW)))
    blank = f"blank{int(rng.integers(len(NEAR_ROW)))}"
    vise = [
        float(lay.workpiece[0]),
        float(lay.workpiece[1] + rng.uniform(-1, 1) * VISE_JITTER),
    ]
    blanks = {
        f"blank{k}": [
            float(v) for v in np.asarray(xy) + rng.uniform(-1, 1, 2) * POCKET_JITTER
        ]
        for k, xy in enumerate(lay.near_row)
    }
    yaws = {name: float(rng.uniform(0, 2 * np.pi)) for name in ("finished", *blanks)}
    wording = INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))]
    return {
        "machine": [float(v) for v in machine],
        "tray": [float(v) for v in tray],
        "stand": [float(v) for v in stand],
        "blank": blank,
        "drop": drop,
        "vise": vise,
        "blanks": blanks,
        "yaws": yaws,
        "instruction": wording.format(pocket=POCKETS[drop]),
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
            # 2: single-arm subtasks (user rule 2026-10-25), one arm moving at a
            # time; 3: set-downs robust on PhysX too (hover, re-centre, seat);
            # 4: the robot model of URDF wheel_robot_26.8.16_3; 5: the fixtures
            # move and turn, parts at any yaw (layout randomization).
            "version": 5,
            "seed": int(seed),
            "draws": attempt,
            **spec,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def _round(v):
    return [round(float(x), 4) for x in v]


# Grasp-site targets (open pads' centre) for the scene check: the expert's
# grasps and placements in the nominal layout.
_NOMINAL = Layout({})
_FIN_BASE = [*WORKPIECE[:2], VISE_FLOOR]
KEY_POINTS = [
    {
        "name": "door_handle_closed",
        "side": "left",
        "xyz": _round(door_poses(_NOMINAL, 0.0, 0.0)["grasp"]),
        "grasp": "forward",
        "psi": float(HANDLE_PSI),
        "width": HANDLE_D,
    },
    {
        "name": f"door_handle_open_{OPEN:.2f}",
        "side": "left",
        "xyz": _round(door_poses(_NOMINAL, OPEN, OPEN)["grasp"]),
        "grasp": "forward",
        "psi": float(HANDLE_PSI),
        "width": HANDLE_D,
    },
    {
        "name": "vise_part",
        "side": "left",
        "xyz": _round(vise_take(_NOMINAL, _FIN_BASE)["grasp"]),
        "grasp": "forward",
        "width": FINISHED_W,
        "ignore": ["cnc_door_part", "cnc_door_window_part"],
    },
    {
        "name": "handover_left",
        "side": "left",
        "xyz": _round(
            pick_poses("left", HANDOVER, "blank", STAND_PSI["left"], 0)["grasp"]
        ),
        "grasp": "forward",
        "psi": float(STAND_PSI["left"]),
        "width": BLANK_W,
    },
    {
        "name": "handover_right",
        "side": "right",
        "xyz": _round(
            pick_poses("right", HANDOVER, "blank", STAND_PSI["right"], 0)["grasp"]
        ),
        "grasp": "forward",
        "psi": float(STAND_PSI["right"]),
        "width": BLANK_W,
        # Turned toward the door, the jaw block reaches its closed front: the
        # door is open whenever the right hand works at the stand.
        "ignore": ["cnc_door_part", "cnc_door_window_part"],
    },
    {
        "name": "tray_far_row",
        "side": "right",
        "xyz": _round(
            place_poses("right", [*FAR_ROW[2], POCKET_FLOOR], "finished", 0.0, 0)[
                "place"
            ]
        ),
        "grasp": "forward",
        "width": FINISHED_W,
    },
    {
        "name": "tray_blank",
        "side": "right",
        "xyz": _round(
            pick_poses("right", [*NEAR_ROW[2], POCKET_FLOOR], "blank", 0.0, 0)["grasp"]
        ),
        "grasp": "forward",
        "width": BLANK_W,
    },
    {
        "name": "cycle_start",
        "side": "right",
        "xyz": _round(press_poses(_NOMINAL)["touch"]),
        "grasp": "forward",
        "psi": float(PRESS_PSI),
    },
]


def build(spec=None):
    """The scene for spec (default: the nominal layout)."""
    spec = spec or nominal()
    lay = Layout(spec)
    yaws = spec.get("yaws", {})
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root, bench=False)
    pa.add(root, "cnc_front", "cnc", pos=tuple(lay.machine))
    tray_quat = yaw_quat(lay.tray_yaw)
    stand(root, "tray_stand", lay.tray, TRAY_STAND_SIZE, top_z=TABLE_Z).set(
        "quat", " ".join(f"{v:.9g}" for v in tray_quat)
    )
    stand(root, "handover_stand", lay.handover[:2], HANDOVER_SIZE, top_z=TABLE_Z).set(
        "quat", " ".join(f"{v:.9g}" for v in yaw_quat(lay.stand_yaw))
    )
    pa.add(
        root,
        "part_tray_6",
        "tray",
        pos=(*lay.tray, TABLE_Z),
        quat=tray_quat,
        variant_name="IN",
    )
    for name, xy in spec["blanks"].items():
        pa.add(
            root,
            "round_blank",
            name,
            pos=(*xy, POCKET_FLOOR + 0.0005),
            quat=yaw_quat(yaws.get(name, 0.0)),
            free=True,
        )
    pa.add(
        root,
        "machined_part",
        "finished",
        pos=(*spec["vise"], VISE_FLOOR + 0.0005),
        quat=yaw_quat(yaws.get("finished", 0.0)),
        free=True,
    )
    report["task"] = {
        "name": NAME,
        "instruction": spec["instruction"],
        "key_points": KEY_POINTS,
    }
    return root, report
