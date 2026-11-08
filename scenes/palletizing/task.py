"""Palletizing: stack the cartons from the two side stands onto the pallet in
the pattern on the sign (layer 1 of plan_04: four 120 × 80 mm cartons in a
2 × 2 block, long side along x, the two columns 40 mm apart).

A scissor lift on a 100 mm plinth holds the pallet's top deck at 0.90 m in
front of the robot (pallet centre x = 0.40). Each side stand (top 0.90 m)
carries two cartons for the arm on its side, long side along x and 40 mm
apart, and each arm fills the pallet's half on its side (y = ±0.06). The
cartons are taken with sim.tabletop.grasp_rotation's hand tilted 30 degrees
from vertical instead of 45, its jaws closing along y across their 80 mm
width; the 40 mm gaps on the stands and in plan_04 are where the fingers go.
The sign stands behind the pallet, facing the robot.

The expert (run.py) fills the cells one at a time, each arm its near cell
first: the first arm, then the other, then the first arm's far cell, then the
other's; after each carton the arm goes back to READY, in the joints its
first turn reached (ready_pose()). Two hands in the same row would collide
(their housings span about ±85 mm along the jaws; the columns' centres are
120 mm apart). Its moves, which the sampler IK-checks, are key_poses() in the
order of PLACE_MOVES, from ready_pose() and back to READY:
- The grasp point is 5 mm under a carton's top: 15 mm under it, the
  gripper's parallelogram links swung into the carton's top edges as the jaws
  closed (5 mm deep when the pads met the faces).
- At the near cells (x 0.34) the wrist hangs over the chest. Tilted 45
  degrees, the hand needs the wrist pitch at its 60-degree limit there: the
  IK chain rode that limit and could reach a near cell in a branch with the
  wrist 0-2 mm from the torso instead of 10-19 (old robot model), and carried
  in from straight above it came within 1 mm. Tilted 35 degrees it kept 39 mm
  on the old model, but only 3.6 mm on wheel_robot_26.8.16_3, whose gripper
  sits 12 mm further from the wrist; at 30 degrees it keeps 50 mm (kinematic
  dry run of the expert's IK chain).
- A carton for a near cell is carried in 12 cm outboard and slid in 3 cm
  above the deck, and the hand rises straight up after letting go: backed off
  along the grasp axis at 45 degrees, the wrist went into the torso.
- Far cartons are set 6 mm beyond their cell, so the cartons of a column
  stand 6 mm apart instead of touching, and none is lowered against another.

A seed fixes the cartons' places and yaw on the stands, which arm starts, the
order each arm takes its two cartons in (so which carton goes to which cell)
and the wording. Every draw is IK-checked along the expert's straight moves
and redrawn until it plans.

Until 2026-10-25 the cartons stood with their 120 mm side along y, wider than
the 95 mm jaws, the deck was at 0.80 m, and plan_01 packed them without gaps.
"""

import numpy as np

from robots.franzi.gripper_linkage import contact_drop
from robots.franzi.scene import build as robot_scene
from sim import prop_assets as pa
from sim import tabletop as tt
from sim.workcell import block, industrial_cell, stand

NAME = "palletizing"
INSTRUCTIONS = (
    "Stack the four cartons on the pallet as the sign shows.",
    "Palletize the four cartons in the pattern on the sign.",
    "Put the cartons on the pallet following the layer plan on the sign.",
    "Load the pallet with the four boxes as shown on the pattern sign.",
)
INSTRUCTION = INSTRUCTIONS[0]
LIFT = (0.46, 0.0)  # its base starts at x 0.25, clear of the robot's front wheels
PLINTH_Z = 0.10
DECK_Z = 0.90
PALLET = (0.40, 0.0, DECK_Z - 0.08)
BOX = "pallet_carton_120x80x80"
BOX_W = 0.080  # across the jaws
# Gripper-dependent (re-measure for another gripper): the closed pads' centre
# over a carton's base, 5 mm under its top. At 15 mm under it the linkage's
# parallelogram links met the carton's top edges as the jaws closed (above).
GRASP_Z = 0.075
# Closure past contact: the 250 g carton gets 6 mm (4.5 N per finger of the
# finger servos' 6 N).
SQUEEZE = 0.006
# Room per side between the open pads and a carton: the fingers stay 2 mm off
# their open limit (tt.grip_open's 12 mm per side would put them past it).
ROOM = 0.0055
LIFT_H = 0.10  # a carried carton's base over the stand or deck: 20 mm over the tops
LOW = 0.03  # the base over the deck where a carton starts down
# The carton's lowest corner over the deck where "lower" stops and where the
# hand lets go ("set"), each re-measured in the hand (run.py). Let go 3 mm up,
# a carton dropped that far under PhysX, where nothing creeps in the pads as
# in MuJoCo; on the cnc stand such a drop penetrated 1.18 mm.
SET = 0.003
RELEASE = 0.0005
OUT = 0.12  # near cells: carried in this far outboard of the cell, then slid in
# The hand's rise after letting go: the fingertips, tt.TIP past the pads,
# end about 2 cm over the carton tops.
RISE = 0.04
EXIT = 0.20  # the hands leave the pallet this far outboard of their cells
TILT = np.radians(30)  # the hand's approach from vertical (see above)
# A carton taken turned is turned square in place over its stand in steps of
# at most SQUARE_STEP: in one step the arm's joints changed up to 0.42 rad at
# 15 degrees and swung the hand between the two poses; in 3-degree steps each
# changes at most 0.09 rad. The sampler redraws a layout whose planned joint
# path jumps more than JUMP between two waypoints: where the nearby IK branch
# meets a joint limit, the next waypoint lands on another (0.45-1.01 rad).
SQUARE_STEP = np.radians(3)
JUMP = 0.35
STAND_Z = DECK_Z
STANDS = {"left": (0.37, 0.37), "right": (0.37, -0.37)}
SUPPLY = {"left": [(0.37, 0.31), (0.37, 0.43)], "right": [(0.37, -0.31), (0.37, -0.43)]}
CARTONS = tuple(f"carton_{side}{k}" for side in tt.SIDES for k in range(2))
# A carton's base corners in its prop frame (80 mm along x, 120 mm along y).
BASE_CORNERS = np.array(
    [[x, y, 0.0] for x in (-0.04, 0.04) for y in (-0.06, 0.06)], dtype=float
)
SIGN_XY = (0.76, 0.0)
PLAN = next(
    p for p in pa.spec("pattern_sign")["stacking_plans"] if p["name"] == "plan_04"
)
# Layer 1's cells by their number on the sign: 1 near right, 2 far right, 3 near
# left, 4 far left (world x, y of the carton's centre).
SLOTS = {
    b["number"]: (
        PALLET[0] + b["center_xy_mm"][0] / 1000,
        PALLET[1] + b["center_xy_mm"][1] / 1000,
    )
    for b in PLAN["layers"][0]
}
TARGETS = list(SLOTS.values())
CELL_SHIFT = 0.006  # far cartons are set this much beyond their cell (+x)
# Success: each cell holds a carton within SLOT_TOL of its centre (x and y),
# within YAW_TOL of the plan's orientation, flat on the deck (SEAT_TOL), still
# and clear of the robot.
SLOT_TOL = 0.015
YAW_TOL = np.radians(8)
SEAT_TOL = 0.003
# The sampled layout (version 4, 2026-10-27: wide enough that 200 episodes
# differ). The lift, its plinth, the pallet and the sign move together by up
# to PALLET_SHIFT (x, y); each supply stand moves along x by up to
# STAND_SHIFT; each stand's two cartons turn together about their centre by up
# to PAIR_YAW, so the 40 mm finger gap between them stays, and slide as far
# as the stand top leaves them (EDGE inside it); then each carton moves on its
# own by up to JITTER and turns by up to YAW, the gap staying at least 36 mm
# where the fingers go in. What limits each range: see sample notes below.
PALLET_SHIFT = (0.02, 0.005)
STAND_SHIFT = 0.02
STAND_SIZE = (0.20, 0.26)
PAIR_YAW = np.radians(15)
PAIR_SIZE = (0.12, 0.20)  # the two cartons and their gap, along x and y
EDGE = 0.005
JITTER = (0.002, 0.002)
YAW = np.radians(2)


def side_of(slot):
    return "left" if SLOTS[slot][1] > 0 else "right"


def row_of(slot):
    return "far" if SLOTS[slot][0] > PALLET[0] else "near"


def side_slots(side):
    """{"far": slot, "near": slot} of the side's half of the pallet."""
    return {row_of(n): n for n in SLOTS if side_of(n) == side}


def slots(spec):
    """The cells' centres (x, y) for spec's pallet, which may have moved."""
    dx, dy = spec.get("pallet", (0.0, 0.0))
    return {n: (x + dx, y + dy) for n, (x, y) in SLOTS.items()}


def place_xy(spec, slot):
    """Where the expert sets a carton's centre for slot."""
    x, y = slots(spec)[slot]
    return (x + (CELL_SHIFT if row_of(slot) == "far" else 0.0), y)


def grasp_rotation(side, psi=0.0):
    """sim.tabletop.grasp_rotation's 45-degree hand (jaws along y turned psi
    about z, the wrist D405 on top), turned about the jaw axis to TILT from
    vertical."""
    steeper = tt.rot_z(psi) @ tt.rot_y(np.pi / 4 - TILT) @ tt.rot_z(-psi)
    return steeper @ tt.grasp_rotation(side, psi)


def key_poses(side, base, yaw, place_at, row):
    """Grasp-site targets (world, m) and hand rotations for moving the carton
    whose base centre is at base (x, y, z), turned yaw from the plan's
    orientation, to place_at (x, y) in a cell of row ("near" or "far"). The
    hand (grasp_rotation) takes the carton turned with it ("pick"), turns it
    square in place over the stand and carries it square ("put"); each
    grasp-site target stops contact_drop(BOX_W) short along the approach, so
    the dropping pads close at the grasp point."""
    pick, put = grasp_rotation(side, yaw), grasp_rotation(side)
    drop = contact_drop(BOX_W)
    grasp = np.asarray(base, float) + [0, 0, GRASP_Z] + drop * (pick @ [0, 0, 1])
    place = np.array([*place_at, DECK_Z + SET + GRASP_Z])
    place = place + drop * (put @ [0, 0, 1])
    outboard = 1.0 if side == "left" else -1.0
    out = OUT * outboard if row == "near" else 0.0
    steps = max(1, int(np.ceil(abs(yaw) / SQUARE_STEP)))
    return {
        "pick": pick,
        "put": put,
        # "square": the hand's turns, from the pick to put, in place at "lift"
        "turns": [
            grasp_rotation(side, yaw * (1 - i / steps)) for i in range(1, steps + 1)
        ],
        "side": side,
        "base": np.asarray(base, float),
        "pregrasp": grasp + tt.PRE * (pick @ [0, 0, 1]),
        "grasp": grasp,
        "lift": grasp + [0, 0, LIFT_H],
        "over": place + [0, out, LIFT_H],
        "low": place + [0, 0, LOW - SET],
        "place": place,
        "set": place + [0, 0, RELEASE - SET],
        "rise": place + [0, 0, RISE],
        "exit": place + [0, EXIT * outboard, LIFT_H],
    }


# The expert's straight moves for one carton, (stage, key, rotation), by
# subtask: pick it up (the grip closes after "align"), stack it ("square"
# turns it in place over the stand; the grip opens after "set"), move the arm
# back (then "leave" goes to "exit" and READY).
PICK_MOVES = (
    ("approach", "pregrasp", "pick"),
    ("align", "grasp", "pick"),
    ("lift", "lift", "pick"),
)
STACK_MOVES = (
    ("square", "lift", "put"),
    ("carry", "over", "put"),
    ("carry", "low", "put"),
    ("lower", "place", "put"),
    ("set", "set", "put"),
)
BACK_MOVES = (("retreat", "rise", "put"),)
PLACE_MOVES = PICK_MOVES + STACK_MOVES + BACK_MOVES


def down_pose(kin, side):
    """The default posture's arm joints: fingers down at READY (tt.start_home)."""
    return kin.ik(side, tt.READY[side], np.zeros(7))


def ready_pose(kin, side):
    """The arm's ready pose at READY, where it waits between its cartons: the
    default posture's joints with the hand turned square in place (as
    tt.ready_pose)."""
    return kin.ik(
        side, tt.READY[side], down_pose(kin, side), rotation=grasp_rotation(side)
    )


def episode_poses(spec, step):
    job = spec["steps"][step]
    carton = spec["cartons"][job["carton"]]
    base = [*carton["xy"], STAND_Z]
    place_at = place_xy(spec, job["slot"])
    return key_poses(job["side"], base, carton["yaw"], place_at, row_of(job["slot"]))


def moves(poses):
    """The expert's straight moves for one carton as (stage, target, rotation),
    the square turn in its steps, then the hand's way back to READY."""
    out = []
    for stage, key, rotation in PLACE_MOVES:
        if stage == "square":
            out += [(stage, poses[key], turn) for turn in poses["turns"]]
        else:
            out.append((stage, poses[key], poses[rotation]))
    out.append(("leave", poses["exit"], poses["put"]))
    out.append(("leave", np.asarray(tt.READY[poses["side"]]), poses["put"]))
    return out


def plan(kin, spec):
    """Raises RuntimeError where a key pose of the expert's path, or a 1 cm
    waypoint between them, has no IK solution inside the limits, or where its
    joints jump more than JUMP between two waypoints. Each carton starts from
    the arm's default posture (its first) or its ready pose (its second), the
    hand turned in place at READY to the carton's yaw ("turn")."""
    seen = set()
    for step in range(len(spec["steps"])):
        poses = episode_poses(spec, step)
        side = poses["side"]
        start = ready_pose(kin, side) if side in seen else down_pose(kin, side)
        seen.add(side)
        q = kin.ik(side, tt.READY[side], start, rotation=poses["pick"])
        for stage, target, rotation in moves(poses):
            path = kin.line(side, q, target, rotation=rotation)
            if np.abs(np.diff(path, axis=0)).max() > JUMP:
                raise RuntimeError(f"{side} {stage}: the joints jump on the way")
            q = path[-1]


def steps(first, order):
    """The four placements in order: the arm `first` and then the other fill
    their near cells, then their far cells, each taking its supply cartons in
    order (indices into SUPPLY)."""
    other = "right" if first == "left" else "left"
    return [
        {
            "side": side,
            "carton": f"carton_{side}{order[side][k]}",
            "slot": side_slots(side)[row],
        }
        for k, row in enumerate(("near", "far"))
        for side in (first, other)
    ]


def nominal():
    """The layout bench_check builds: nothing moved, the left arm first."""
    return {
        "first": "left",
        "pallet": [0.0, 0.0],
        "stands": {side: list(STANDS[side]) for side in tt.SIDES},
        "cartons": {
            f"carton_{side}{k}": {"xy": list(xy), "yaw": 0.0}
            for side in tt.SIDES
            for k, xy in enumerate(SUPPLY[side])
        },
        "steps": steps("left", {side: (0, 1) for side in tt.SIDES}),
        "instruction": INSTRUCTION,
    }


def pair_room(theta):
    """How far (x, y) a stand's carton pair turned theta may slide from the
    stand's centre and stay EDGE inside its top."""
    c, s = np.cos(theta), np.sin(abs(theta))
    half = (
        np.array(
            [PAIR_SIZE[0] * c + PAIR_SIZE[1] * s, PAIR_SIZE[0] * s + PAIR_SIZE[1] * c]
        )
        / 2
    )
    return np.maximum(np.asarray(STAND_SIZE) / 2 - half - EDGE, 0.0)


def draw(rng):
    first = str(rng.choice(tt.SIDES))
    pallet = rng.uniform(-1, 1, 2) * PALLET_SHIFT
    stands, cartons = {}, {}
    for side in tt.SIDES:
        stand = np.array(STANDS[side]) + [rng.uniform(-1, 1) * STAND_SHIFT, 0.0]
        theta = float(rng.uniform(-1, 1) * PAIR_YAW)
        centre = stand + rng.uniform(-1, 1, 2) * pair_room(theta)
        turn = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        # The inner carton (k = 0, nearer the robot's midline) and the outer
        # one, half their nominal pitch either way along the pair's turned y.
        half = abs(SUPPLY[side][1][1] - SUPPLY[side][0][1]) / 2
        inward = -half if side == "left" else half
        for k, along in enumerate((inward, -inward)):
            xy = centre + turn @ [0.0, along] + rng.uniform(-1, 1, 2) * JITTER
            cartons[f"carton_{side}{k}"] = {
                "xy": [float(v) for v in xy],
                "yaw": theta + float(rng.uniform(-1, 1) * YAW),
            }
        stands[side] = [float(v) for v in stand]
    order = {side: tuple(int(k) for k in rng.permutation(2)) for side in tt.SIDES}
    return {
        "first": first,
        "pallet": [float(v) for v in pallet],
        "stands": stands,
        "cartons": cartons,
        "steps": steps(first, order),
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
            # 2: single-arm subtasks (user rule 2026-10-25); 3: the robot model
            # wheel_robot_26.8.16_3, the hand at 30 degrees, let go 0.5 mm up;
            # 4: the wide layout sampler (2026-10-27)
            "version": 4,
            "seed": int(seed),
            "draws": attempt,
            **spec,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def key_points():
    """Grasp-site targets of the nominal layout for the scene check: each supply
    carton's grasp and each cell's place. bench_check probes them with the
    45-degree grasp (the tilted hand nearest to the expert's 30 degrees), so
    the targets back off along its axis."""
    drop = contact_drop(BOX_W)
    spec = nominal()
    points = []
    for name, carton in spec["cartons"].items():
        side = "left" if "left" in name else "right"
        base = [*carton["xy"], STAND_Z + GRASP_Z]
        points.append((f"grasp_{name}", side, base))
    for slot in SLOTS:
        base = [*place_xy(spec, slot), DECK_Z + SET + GRASP_Z]
        points.append((f"place_slot{slot}", side_of(slot), base))
    return [
        {
            "name": name,
            "side": side,
            "xyz": [
                round(float(v), 4)
                for v in np.add(base, drop * (tt.grasp_rotation(side) @ [0, 0, 1]))
            ],
            "grasp": "forward",
            "width": BOX_W,
        }
        for name, side, base in points
    ]


KEY_POINTS = key_points()


def build(spec=None):
    """The scene for spec (default: the nominal layout)."""
    spec = spec or nominal()
    shift = np.array(spec.get("pallet", (0.0, 0.0)), float)
    stands = spec.get("stands", STANDS)
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root, bench=False)
    lift = np.add(LIFT, shift)
    block(root, "lift_plinth", lift, (0.42, 0.52), 0.0, PLINTH_Z)
    pa.add(root, "scissor_lift_table", "lift", pos=(*lift, PLINTH_Z))
    pallet = (*np.add(PALLET[:2], shift), PALLET[2])
    pa.add(root, "mini_pallet", "pallet", pos=pallet, variant_name="pine")
    for side in tt.SIDES:
        stand(root, f"supply_{side}", stands[side], STAND_SIZE, top_z=STAND_Z)
    for name, carton in spec["cartons"].items():
        half = (np.pi / 2 + carton["yaw"]) / 2  # the long side along x, turned
        pa.add(
            root,
            BOX,
            name,
            pos=(*carton["xy"], STAND_Z + 0.0005),
            quat=(np.cos(half), 0, 0, np.sin(half)),
            free=True,
        )
    sign = np.add(SIGN_XY, shift)  # the sign stays behind the pallet
    stand(root, "sign_stand", sign, (0.14, 0.26), top_z=DECK_Z)
    pa.add(
        root,
        "pattern_sign",
        "sign",
        pos=(*sign, DECK_Z),
        variant_name=PLAN["name"],
    )
    cells = slots(spec)
    report["task"] = {
        "name": NAME,
        "instruction": spec["instruction"],
        "key_points": KEY_POINTS,
        "targets_xy": list(cells.values()),
        "slots": {str(n): list(xy) for n, xy in cells.items()},
    }
    return root, report
