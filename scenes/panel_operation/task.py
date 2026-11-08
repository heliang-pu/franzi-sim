"""Panel operation: turn the mode selector to AUTO, then press the green START
button, keeping clear of the emergency stop.

The control console (layout A) stands on the workbench (top at BENCH_Z, 0.90
m) with its sloped panel facing the robot; its front edge is at x ≈ 0.30, so
the devices sit at x 0.36–0.40 and z 0.99–1.02. The selector (y −0.09) and
START (y −0.02) are the right arm's; STOP (y +0.04) and the emergency stop
(y +0.11) lie on the left. The selector starts at MAN (−45 degrees, the
panel library's recommended initial angle); AUTO is +45. Key points are 10 mm
out from each device's mount along the panel normal. They are pressed or
turned, not grasped, so they carry no grasp probe.

The expert works with the right hand's fingers pointing into the panel (along
its normal, 26.6 degrees off vertical). It grips the selector's 9 × 14 × 32 mm
lever (collider 6–20 mm out from the mount; the knob's 6 mm base has none)
with the jaws across it, fingertips 7.5 mm out, and turns the hand about the
panel normal. The arm's last joint doesn't turn about the fingers (since the
URDF wheel_robot_26.8.16_3 it turns about the jaw plane's normal, before about
the jaw axis), so the turn is the whole wrist: its three axes meet in a point
that, with the fingers along the normal, lies on the knob's axis 0.256 m out
(1.7 mm off it), so the shoulder and elbow hold still while the wrist turns
the hand (wrist_turn, solved by FK); the wrist's pitch and roll then stay
within the forearm's angle to the knob's axis, and where the 88 degree turn
sweeps them decides which elbow swivel keeps the wrist inside its limits. On
the current model, for the grip at knob + 90 degrees, that is the elbow out to
the side, 5-16 cm below the shoulder (forearm 65-79 degrees off the axis, up
to 0.24 rad of slack); on the old model it was the raised elbow, above the
shoulder (40-45 degrees off). Elsewhere the wrist pitch reaches its ±60 degree
limit within the turn, and with the elbow low and inside the forearm lies in
the torso. The sampler finds the posture by itself (choose_grip): of the IK
solutions at the grip from a fixed draw of random arm poses, it takes the one
with the most slack through the turn whose wrist turns alone, whose elbow and
wrist stay out of the torso and whose approach plans, and the spec carries it
to the expert, which solves its IK from there. On the old model that posture
was about 3 rad of shoulder pitch and yaw from the default one, and a
joint-space move between them cut 13-46 mm into the torso hull, so the hand
reaches the selector on one straight line from the default posture, turning
on the way, with its IK guided into the grip's posture (guided_line), and
comes back from START over the selector and down the approach in reverse. Of the two grips across the lever, the one at knob
+ 90 degrees is favoured by 0.1 rad of slack (on the old model the other's
raised elbow kept 0.26 rad against 0.21 through the turn, but the approach
could not reach it: it ended 2.5 rad off, the arm 102 mm inside the torso
hull). The turn stops TURN_TO, 2 degrees short of AUTO's stop. START is
pressed with the closed fingertips along the normal, closed-loop to PRESS_TO
of its 4 mm travel, at the hand angle the sampler picks for the most joint
slack (choose_press; the new model's wrist roll, ±90 degrees, is the joint that
binds there). The left arm stays home.

The console's own contact exclusions (layout A's collision_exclusions: the
caps and the selector against their mounts and legend plates) are added here:
prop_assets.add reads only the prop-level list, which this prop leaves empty,
and without them the START cap rests on its legend plate and went down 0.07
mm under 10 N (4.0 mm under 2 N with them). Freed from its plate, the
emergency stop (no spring) slid to its 5 mm stop under its own weight within
5 s, though that is below its 0.12 N friction (MuJoCo's soft friction creeps):
its joint gets the static-friction solver settings the lidded tote's prop
notes give, and moves 0.1 mm in 60 s. PhysX has no joint friction, so there it
slides to its stop anyway; the success test asks only that the robot never
touches it.

A seed draws the console's place on the bench (x −4 to +6 cm, y −10 to +5
cm, turned −5 to +35 degrees, its base on the bench; see X_RANGE), the
selector's start within 5 degrees of MAN, the console's colour and the
wording. Every draw is IK-checked along the expert's path and redrawn until it
plans.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robots.franzi.gripper_linkage import contact_drop
from robots.franzi.scene import ARM_JOINTS
from robots.franzi.scene import build as robot_scene
from sim import prop_assets as pa
from sim import tabletop as tt
from sim.workcell import BENCH_FRONT_X, BENCH_Z, industrial_cell

NAME = "panel_operation"
PROP = "control_console"
CONSOLE = np.array([0.41, 0.0, BENCH_Z])
MAN = -np.pi / 4  # the selector's start: its MAN detent
AUTO = np.pi / 4
FRAME = pa.spec(PROP)["panel_frame"]
NORMAL = np.array(FRAME["outward_normal"])  # console frame, out of the panel
UP = np.array(FRAME["v_direction"])  # up the slope
ACROSS = np.array([0.0, 1.0, 0.0])  # along the panel, to the robot's left
# The selector counts as at AUTO within 5 degrees of its stop; the expert turns
# it to 2 degrees short, so the hand never pushes it into the stop.
AUTO_MIN = AUTO - np.radians(5)
TURN_TO = AUTO - np.radians(2)
# START (4 mm travel) counts as pressed from 2.5 mm; the expert presses to 3.
PRESSED = 0.0025
PRESS_TO = 0.003
# Gripping the lever: its width across the jaws, and the fingertips' height
# over the selector's mount (over the knob's 6 mm base, on the lever's 6-20 mm,
# 6.5 mm over the selector's legend plate). The site stops contact_drop further
# out, so the dropping pads close there. 8 mm of squeeze saturates the finger
# servos' 6 N: the lever turns on the pads' normal forces alone (friction there
# gives no torque about the knob's axis), against the knob's 0.025 N m
# friction and its damping.
LEVER = 0.009
TIP_OUT = 0.0075
GRIP_OUT = TIP_OUT + tt.TIP + contact_drop(LEVER)
SQUEEZE = 0.008
# Pressing: the closed fingertips (closure PRESS_CLOSURE, 4.75 mm apart; with
# the current fingers a 19 × 15 mm footprint centred on the 23 mm cap) touch
# the cap's top, 9 mm out, and the closed loop goes on from there, up to
# PRESS_REACH past the touch.
PRESS_CLOSURE = 0.95
CAP_TOP = 0.009
PRESS_OUT = CAP_TOP + tt.TIP + contact_drop(tt.JAW_SPAN * (1 - PRESS_CLOSURE))
PRESS_REACH = 0.012
# Pre-grip and pre-press distance along the normal. At 80 mm (tt.PRE) the new
# model's wrist roll (±90 degrees) ran into its bound on the line from the
# selector to START in 7 of 20 layouts, and the IK jumped to another arm
# configuration; at 40 mm every layout plans on its first draw.
APPROACH = 0.04
# After letting go, the hand backs off this far (fingertips 22 mm over the
# lever's top) and turns back in place there.
LIFT = 0.03
TURN_STEP = np.radians(2)  # IK spacing of in-place turns
# The grip pose is searched from SEARCH_SEEDS random arm poses (a fixed draw):
# each IK solution at the grip is judged by its joint slack through a turn at
# TURN_PROBE spacing, the grip at knob + 90 degrees favoured by SIDE_MARGIN.
SEARCH_SEEDS = 80
TURN_PROBE = np.radians(8)
SIDE_MARGIN = 0.1
# A candidate whose shoulder or elbow moves more than ARM_HOLD (rad) in that
# turn is dropped (the wrist cannot turn the hand alone there, and the IK swung
# the arm instead; the raised elbow's arm moves 6 mrad), and so is one with its
# elbow or wrist joint inside TORSO_BOX (world, m; the torso's collision hull
# spans x -0.119..0.129, y ±0.19, z 1.066..1.557). With the elbow low and
# inside, the wrist can make the turn too, and that pose won on slack; its
# forearm lay 82 mm deep in the torso hull.
ARM_HOLD = 0.05
TORSO_BOX = (np.array([-0.12, -0.19, 1.07]), np.array([0.13, 0.19, 1.56]))
# The approach line's IK starts GUIDE of the way toward its interpolation to
# the grip posture, the rest from the previous waypoint. An approach that ends
# more than APPROACH_REACH (rad) from its target pose went into another
# configuration (0.14 for the raised elbow; 2.4-2.6 for the flipped shoulder
# and for the other side's raised elbow, which ran 102 mm into the torso).
GUIDE = 0.3
APPROACH_REACH = 0.3
# An approach whose joint travel (its largest joint step, summed over the
# waypoints) exceeds APPROACH_ARC (rad) winds through other configurations:
# the elbow-out approaches take 2.2-2.4 rad, and one to the other side's raised
# elbow took 9.5 and put the wrist D405's housing 1 mm into the torso.
APPROACH_ARC = 5.0
# The hand presses START at the angle (see hand) of PRESS_ANGLES around the
# turned hand's that keeps the most joint slack from the turn back over the
# selector to the end of the return. Pressing at the grip's angle, as on the
# old model, held the new model's wrist roll on its bound in half the layouts.
PRESS_ANGLES = np.radians(np.arange(-120, 31, 15))
# A planned path is continuous when no waypoint moves a joint more than this.
STEP_MAX = 0.2
# The hand comes back from START over the selector and down the approach in
# reverse. The line back must end within REJOIN (rad) of the approach's last
# pose. One straight line from START to READY instead wound through up to 10
# rad of joint travel at 24-31 degrees of yaw (31 s), and with START at the
# midline its wrist brushed the torso.
REJOIN = 0.15
READY = tt.READY["right"]
# Where the console stands: CONSOLE moved by X_RANGE and Y_RANGE (m) and turned
# YAW_RANGE about z. Toward the robot the head D435 crops the console's front
# from about 5 cm (its devices stay in view), and from 9 cm no grip plans; away
# from it the arm plans to 7 cm. In y the expert plans from -12 to +9 cm; at
# +5 cm START sits 3 cm left of the midline. Turned to face the robot's left
# (yaw < 0) the right forearm has to come from inside: from -10 degrees every
# approach jumps to the elbow-low branch, 59-64 mm inside the torso hull.
# Turned toward the right arm it plans to +40 degrees, the e-stop then 8 cm
# nearer the robot.
X_RANGE = (-0.04, 0.06)
Y_RANGE = (-0.10, 0.05)
YAW_RANGE = np.radians((-5, 35))
# The console's base (its bounding box) stays this far behind the bench's front
# edge; a draw that would overhang it is drawn again.
BENCH_MARGIN = 0.01
KNOB_JITTER = np.radians(5)
COLOURS = {"grey": "A", "blue": "blue"}  # layout A in either body colour
INSTRUCTIONS = (
    (
        "Turn the selector to AUTO, then press the green START button. Keep clear "
        "of the emergency stop."
    ),
    (
        "Switch the mode selector from MAN to AUTO and press START, without "
        "touching the emergency stop."
    ),
    (
        "Set the machine to AUTO with the selector switch, then push the green "
        "START button. Do not touch the E-stop."
    ),
    (
        "Put the mode switch on AUTO and start the machine with the green button; "
        "leave the emergency stop alone."
    ),
)
INSTRUCTION = INSTRUCTIONS[0]
# The lidded tote's static hold (its prop notes): MuJoCo's soft friction
# defaults let a joint creep below its frictionloss.
STATIC_FRICTION = {"solreffriction": "0.002 1", "solimpfriction": "0.9999 0.9999 0.001"}


def device_point(part, lift=0.010):
    return [
        round(float(v), 4)
        for v in CONSOLE + pa.joint_anchor(PROP, part) + lift * NORMAL
    ]


KEY_POINTS = [
    {
        "name": "selector",
        "side": "right",
        "xyz": device_point("mode", 0.012),
        "grasp": "any",
    },
    {
        "name": "start_button",
        "side": "right",
        "xyz": device_point("start"),
        "grasp": "any",
    },
    {
        "name": "stop_button",
        "side": "left",
        "xyz": device_point("stop"),
        "grasp": "any",
    },
    {
        "name": "emergency_stop",
        "side": "left",
        "xyz": device_point("emergency", 0.02),
        "grasp": "any",
    },
]


def hand(yaw, theta):
    """Hand frame with the fingers pointing into the panel (the grasp frame's z
    along the outward normal) and the jaws closing at theta from the panel's
    up-slope direction toward +y; the console turned yaw about z."""
    return tt.rot_z(yaw) @ np.c_[UP, ACROSS, NORMAL] @ tt.rot_z(theta)


def grip_angle(knob, side=1):
    """Hand angle (see hand) gripping the lever at knob with the jaws across it:
    side 1 at knob + 90 degrees, -1 at knob - 90 (the hand turned half round)."""
    return knob + side * np.pi / 2


def key_poses(pos, yaw, knob, side=1, press_angle=None):
    """Grasp-site targets (world, m) and hand rotations for the console whose
    base centre is at pos, turned yaw about z, with the selector at knob (rad),
    gripped from side (grip_angle), START pressed at press_angle (default the
    grip's)."""
    turn = tt.rot_z(yaw)
    normal = turn @ NORMAL

    def out(part, distance):
        return (
            np.asarray(pos, float)
            + turn @ pa.joint_anchor(PROP, part)
            + (distance * normal)
        )

    grip, press = out("mode", GRIP_OUT), out("start", PRESS_OUT)
    return {
        "normal": normal,
        "yaw": yaw,
        "side": side,
        "grip_rotation": hand(yaw, grip_angle(knob, side)),
        "turned_rotation": hand(yaw, grip_angle(TURN_TO, side)),
        "press_rotation": hand(
            yaw, grip_angle(knob, side) if press_angle is None else press_angle
        ),
        "pregrip": grip + APPROACH * normal,
        "grip": grip,
        "lift": grip + LIFT * normal,
        "prepress": press + APPROACH * normal,
        "press": press,
        "backoff": press + APPROACH * normal,
    }


def wrist_angles(kin, side, q, rotation, body=None):
    """The last three arm joints (the wrist) that turn the hand to rotation with
    the first four held at q's: least squares on the hand's rotation error, by
    FK, from q's wrist."""
    lo = kin.limits[side][4:, 0] + kin.margin
    hi = kin.limits[side][4:, 1] - kin.margin
    rotation = np.asarray(rotation, float)

    def error(wrist):
        hand_now = kin.fk(side, np.r_[q[:4], wrist], body)[1]
        return Rotation.from_matrix(rotation.T @ hand_now).as_rotvec()

    start = np.clip(q[4:], lo + 1e-6, hi - 1e-6)
    return least_squares(
        error, start, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12
    ).x


def wrist_turn(kin, side, q, point, start, end, body=None, step=TURN_STEP):
    """Arm joint waypoints from q turning the hand in place at point (world) from
    rotation start to end with the wrist, every step radians: the wrist's axes
    meet near the fingers' axis, so the shoulder and elbow can hold while the
    wrist turns the hand (wrist_angles). Each waypoint's IK starts there and
    corrects what the arm must add (the grasp site lies 1 mm off the wrist's
    axis)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    turn = Rotation.from_matrix(start.T @ end).as_rotvec()
    count = max(1, int(np.ceil(np.linalg.norm(turn) / step)))
    path = [np.asarray(q, float)]
    for k in range(1, count + 1):
        rotation = start @ Rotation.from_rotvec(turn * k / count).as_matrix()
        prev = path[-1]
        guess = np.r_[prev[:4], wrist_angles(kin, side, prev, rotation, body)]
        path.append(kin.ik(side, point, guess, body, near=guess, rotation=rotation))
    return np.array(path)


def slack(kin, side, path):
    """The path's least joint distance to the aligned limits (rad)."""
    lo, hi = kin.limits[side].T
    return float(np.minimum(path - lo, hi - path).min())


def joint_centres(kin, side, q, body=None):
    """World positions (m) of the arm's elbow and wrist joints at q, by the
    shared kinematics' FK."""
    pin.forwardKinematics(kin.model, kin.data, kin.config(side, q, body))
    return [
        kin.data.oMi[kin.model.getJointId(f"{side}_{joint}_joint")].translation
        + kin.base
        for joint in (ARM_JOINTS[3], ARM_JOINTS[5])
    ]


def in_torso(kin, side, path, body=None):
    """Whether an elbow or wrist joint of the path lies inside TORSO_BOX."""
    lo, hi = TORSO_BOX
    return any(
        np.all((p > lo) & (p < hi))
        for q in path
        for p in joint_centres(kin, side, q, body)
    )


def grip_candidates(kin, pos, yaw, knob, body=None):
    """[(score, key poses, arm joints)] at the grip, best first: the distinct IK
    solutions from SEARCH_SEEDS random arm poses on either side of the lever,
    scored by their least joint slack through the turn (the side at knob + 90
    degrees given SIDE_MARGIN), with the arm held (ARM_HOLD) and out of the
    torso (TORSO_BOX). No joint values are assumed: the search finds the
    raised elbow by itself."""
    lo, hi = kin.limits["right"].T
    seeds = np.random.default_rng(0).uniform(lo, hi, (SEARCH_SEEDS, 7))
    found = []
    for side in (1, -1):
        poses = key_poses(pos, yaw, knob, side)
        grip, turned = poses["grip_rotation"], poses["turned_rotation"]
        seen = []
        for seed in seeds:
            try:
                q = kin.ik("right", poses["grip"], seed, body, rotation=grip)
            except RuntimeError:
                continue
            if any(np.abs(q - other).max() < 0.05 for other in seen):
                continue
            seen.append(q)
            try:
                path = wrist_turn(
                    kin, "right", q, poses["grip"], grip, turned, body, TURN_PROBE
                )
            except RuntimeError:
                continue
            if np.abs(path[:, :4] - q[:4]).max() > ARM_HOLD:
                continue
            if in_torso(kin, "right", path[[0, -1]], body):
                continue
            score = slack(kin, "right", path) + (SIDE_MARGIN if side == 1 else 0)
            found.append((score, poses, q))
    return sorted(found, key=lambda c: -c[0])


def choose_grip(kin, pos, yaw, knob, home, body=None):
    """The best grip candidate whose approach from home (the default posture)
    plans: (key poses, arm joints at the grip)."""
    for _, poses, q in grip_candidates(kin, pos, yaw, knob, body):
        try:
            approach_path(kin, home, poses, q, body)
        except RuntimeError:
            continue
        return poses, q
    raise RuntimeError("no grip on the selector plans")


def fit_grip(kin, poses, joints, body=None):
    """The arm joints at the grip, by IK from the chosen grip's joints."""
    seed = np.asarray(joints, float)
    grip = poses["grip_rotation"]
    return kin.ik("right", poses["grip"], seed, body, near=seed, rotation=grip)


def guided_line(kin, side, q0, q1, target, start, end, body=None, step=0.01):
    """Arm joint waypoints moving the grasp site from where q0 puts it straight
    to target, every step metres, while the hand turns from rotation start to
    end. Each waypoint's IK starts GUIDE of the way from the previous waypoint
    toward the joint interpolation from q0 to q1 (a pose at target), so the arm
    moves into q1's configuration on the way instead of where the IK drifts."""
    begin = kin.fk(side, q0, body)[0]
    count = max(1, int(np.ceil(np.linalg.norm(np.subtract(target, begin)) / step)))
    start = np.asarray(start, float)
    turn = Rotation.from_matrix(start.T @ np.asarray(end, float)).as_rotvec()
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    path = [q0]
    for k in range(1, count + 1):
        t = k / count
        point = begin + (np.asarray(target, float) - begin) * t
        rotation = start @ Rotation.from_rotvec(turn * t).as_matrix()
        guess = GUIDE * (q0 + (q1 - q0) * t) + (1 - GUIDE) * path[-1]
        path.append(kin.ik(side, point, guess, body, near=guess, rotation=rotation))
    return np.array(path)


def approach_path(kin, q, poses, grip_q, body=None):
    """Joint waypoints from q (the default posture, hand down at READY) to the
    pre-grip pose in grip_q's configuration, on guided_line. Raises
    RuntimeError if it ends APPROACH_REACH or more from that pose, or takes an
    elbow or wrist joint through the torso."""
    grip = poses["grip_rotation"]
    target = kin.line("right", grip_q, poses["pregrip"], body, rotation=grip)[-1]
    path = guided_line(kin, "right", q, target, poses["pregrip"], np.eye(3), grip, body)
    if np.abs(path[-1] - target).max() > APPROACH_REACH:
        raise RuntimeError("the approach ends in another arm configuration")
    if np.abs(np.diff(path, axis=0)).max(1).sum() > APPROACH_ARC:
        raise RuntimeError("the approach winds through other arm configurations")
    if in_torso(kin, "right", path, body):
        raise RuntimeError("the approach takes the arm through the torso")
    return path


def continuous(path):
    """Whether no waypoint of path moves a joint more than STEP_MAX."""
    return len(path) < 2 or np.abs(np.diff(path, axis=0)).max() <= STEP_MAX


def press_path(kin, q, poses, approach, body=None):
    """Joint paths [(stage, waypoints)] from q, the open hand backed off the
    turned selector at poses["lift"], through the press and back over the
    selector: the hand turned in place to poses["press_rotation"], the closed
    fingertips over START, pressed PRESS_REACH past the touch (the most the
    closed loop may go), backed off, and on a straight line back to the
    pre-grip pose, turning about the panel normal to the grip's angle, its IK
    guided into approach's last pose, from where the expert replays approach in
    reverse. Raises RuntimeError where the IK fails or jumps, or the line back
    ends REJOIN or more from that pose."""
    press, turned = poses["press_rotation"], poses["turned_rotation"]
    deepest = poses["press"] - PRESS_REACH * poses["normal"]
    out = [
        ("reorient", wrist_turn(kin, "right", q, poses["lift"], turned, press, body))
    ]
    for stage, target in (
        ("reach", poses["prepress"]),
        ("press", deepest),
        ("back_off", poses["backoff"]),
    ):
        out.append(
            (stage, kin.line("right", out[-1][1][-1], target, body, rotation=press))
        )
    back = return_line(kin, out[-1][1][-1], poses, approach, body)
    out.append(("return", back))
    if not all(continuous(path) for _, path in out):
        raise RuntimeError("the arm jumps between IK branches on the way to START")
    if np.abs(back[-1] - approach[-1]).max() > REJOIN:
        raise RuntimeError("the arm comes back too far from the approach's end")
    return out


def return_line(kin, q, poses, approach, body=None):
    """Joint waypoints from q, the hand backed off START, back to the pre-grip
    pose over the selector (press_path)."""
    press, grip = poses["press_rotation"], poses["grip_rotation"]
    return guided_line(
        kin, "right", q, approach[-1], poses["pregrip"], press, grip, body
    )


def choose_press(kin, q, poses, approach, body=None):
    """The press angle (see hand) of PRESS_ANGLES around the turned hand's
    whose press_path from q keeps the most joint slack."""
    yaw, turned = poses["yaw"], grip_angle(TURN_TO, poses["side"])
    best = None
    for delta in PRESS_ANGLES:
        angle = turned + delta
        trial = dict(poses, press_rotation=hand(yaw, angle))
        try:
            paths = press_path(kin, q, trial, approach, body)
        except RuntimeError:
            continue
        score = min(slack(kin, "right", path) for _, path in paths)
        if best is None or score > best[0]:
            best = (score, float(angle))
    if best is None:
        raise RuntimeError("START cannot be pressed from the turned selector")
    return best[1]


def home_pose(kin):
    """The arm joints start_home leaves the right arm on."""
    return kin.ik("right", READY, np.zeros(7))


def plan(kin, spec):
    """Raises RuntimeError where a key pose of the expert's path, or a waypoint
    between them, has no IK solution inside the limits or jumps between IK
    branches, for spec's console, grip (choose_grip) and press angle
    (choose_press)."""
    console = spec["console"]
    poses = key_poses(
        console["xyz"],
        console["yaw"],
        spec["selector"],
        spec["grip"]["side"],
        spec["press_angle"],
    )
    grip_q = fit_grip(kin, poses, spec["grip"]["joints"])
    approach, q = lift_pose(kin, poses, grip_q, home_pose(kin))
    press_path(kin, q, poses, approach)


def lift_pose(kin, poses, grip_q, home, body=None):
    """The approach's joint waypoints, and the arm joints with the open hand
    backed off the turned selector, as the expert gets there."""
    grip, turned = poses["grip_rotation"], poses["turned_rotation"]
    approach = approach_path(kin, home, poses, grip_q, body)
    q = kin.line("right", approach[-1], poses["grip"], body, rotation=grip)[-1]
    q = wrist_turn(kin, "right", q, poses["grip"], grip, turned, body)[-1]
    return approach, kin.line("right", q, poses["lift"], body, rotation=turned)[-1]


def nominal():
    """The layout bench_check builds: no jitter, the selector at MAN."""
    return {
        "console": {"xyz": [float(v) for v in CONSOLE], "yaw": 0.0},
        "selector": MAN,
        "colour": "grey",
        "instruction": INSTRUCTION,
    }


def on_bench(xy, yaw):
    """Whether the console's base at xy (its bottom centre, m), turned yaw,
    stays BENCH_MARGIN behind the bench's front edge."""
    half = np.array(pa.spec(PROP)["bbox_mm"][:2]) / 2000
    corners = np.array(
        [[sx * half[0], sy * half[1]] for sx in (-1, 1) for sy in (-1, 1)]
    )
    x = xy[0] + corners[:, 0] * np.cos(yaw) - corners[:, 1] * np.sin(yaw)
    return bool(x.min() >= BENCH_FRONT_X + BENCH_MARGIN)


def draw(rng):
    while True:
        xy = CONSOLE[:2] + [rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)]
        yaw = rng.uniform(*YAW_RANGE)
        if on_bench(xy, yaw):
            break
    return {
        "console": {
            "xyz": [float(xy[0]), float(xy[1]), BENCH_Z],
            "yaw": float(yaw),
        },
        "selector": float(MAN + rng.uniform(0, 1) * KNOB_JITTER),
        "colour": str(rng.choice(list(COLOURS))),
        "instruction": INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))],
    }


def sample_episode(seed, kin=None, attempts=100):
    kin = kin or tt.kinematics(tt.SAMPLE_MARGIN)
    rng = np.random.default_rng(seed)
    for attempt in range(1, attempts + 1):
        spec = draw(rng)
        console = spec["console"]
        home = home_pose(kin)
        try:
            poses, q = choose_grip(
                kin, console["xyz"], console["yaw"], spec["selector"], home
            )
            # The expert's grip: its side and, as the IK seed, its arm joints;
            # and the hand's angle for pressing START.
            spec["grip"] = {"side": poses["side"], "joints": [float(v) for v in q]}
            approach, lift = lift_pose(kin, poses, q, home)
            spec["press_angle"] = choose_press(kin, lift, poses, approach)
            plan(kin, spec)
        except RuntimeError:
            continue
        return {
            "task": NAME,
            # 2: one subtask text per single-arm action; 3: the new robot model
            # (URDF wheel_robot_26.8.16_3); 4: the console's place widened
            "version": 4,
            "seed": int(seed),
            "draws": attempt,
            **spec,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def fix_console(root, names):
    """The console's contact exclusions (layout A's list) between its bodies
    names (prop_assets.add's {part: body}), unless already there, and the
    emergency stop's static friction."""
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    have = {
        frozenset((e.get("body1"), e.get("body2"))) for e in contact.iter("exclude")
    }
    for a, b in pa.variant(PROP, "A")["collision_exclusions"]:
        if frozenset((names[a], names[b])) not in have:
            ET.SubElement(contact, "exclude", body1=names[a], body2=names[b])
    for joint in root.iter("joint"):
        if joint.get("name") == "console_emergency":
            joint.attrib.update(STATIC_FRICTION)


def build(spec=None):
    """The scene for spec (default: the nominal layout)."""
    spec = spec or nominal()
    console = spec["console"]
    half = console["yaw"] / 2
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    names = pa.add(
        root,
        PROP,
        "console",
        pos=tuple(console["xyz"]),
        quat=(np.cos(half), 0, 0, np.sin(half)),
        variant_name=COLOURS[spec["colour"]],
        initial={"mode": spec["selector"]},
    )
    fix_console(root, names)
    report["task"] = {
        "name": NAME,
        "instruction": spec["instruction"],
        "key_points": KEY_POINTS,
    }
    return root, report
