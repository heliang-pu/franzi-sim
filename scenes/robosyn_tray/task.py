"""Two-handed tray carry with RoboSynChallenge ingredients: episode sampling,
key poses and the scene.

A serving tray with an upright handle at each end lies in the middle of the
table, RoboSyn's apple on the left and its tomato on the right. Each hand puts
its ingredient on the tray (at the same time), then both hands take the
handles, lift the tray 4.5 cm, carry it 13 cm to the side the instruction names,
set it down and let go. The ingredients ride along on the tray: friction only.

A seed fixes the layout (jitter of the tray and the ingredients, which side
the tray goes to) and the wording of the instruction. Every draw is checked
with the shared IK (each key pose, and each 1 cm waypoint of the straight
moves) and redrawn until it plans.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from sim import props as pp
from sim import robosyn
from sim import tabletop as tt
from sim.build_task import ROOT, TABLE_Z, fmt
from sim.build_task import build as build_base

DERIVED = Path(__file__).resolve().parent / "assets/derived"
FOODS = ("apple", "tomato")
PROPS = robosyn.prepare(DERIVED, FOODS)
DIMS = {name: prop["dims"] for name, prop in PROPS.items()}
# The tray (m): a plate with a low rim, so a hand placing an ingredient 3 cm
# above it clears the rim, and a 22 mm post at each end for the fingers.
PLATE = (0.10, 0.235, 0.004)  # half sizes
RIM_HALF = (0.10, 0.195)  # rim walls stand inside these half lengths
RIM = (0.0025, 0.008)  # rim half thickness and height above the plate
POST = (0.011, 0.025)  # handle radius and half height
POST_Y = 0.215  # handles' distance from the tray centre
TRAY_XY = (0.40, 0.0)
SPOT_Y = 0.10  # ingredients on the tray, either side of its centre: the two
# hands, placing together, stay 20 cm apart (each housing spans about 85 mm)
FOOD_XY = {"apple": (0.38, 0.38), "tomato": (0.38, -0.38)}
SIDE_OF = {"apple": "left", "tomato": "right"}
SHIFT = 0.13  # how far the tray is carried, left or right
CARRY_LIFT = 0.045  # the tray counts as airborne from 3 cm (ObjectScene)
JITTER = 0.006
MASS = {"tray": 0.35, "apple": 0.13, "tomato": 0.09}
GRASP_Z = 0.030
LIFT = 0.06
POST_Z = 2 * PLATE[2] + POST[1]  # handle grasp point above the tray's base
POST_SQUEEZE = 0.008  # saturates the finger servos' 6 N, for the loaded tray
INSTRUCTIONS = (
    (
        "Put the apple and the tomato on the tray, then carry the tray to the {side} "
        "with both hands."
    ),
    (
        "Load the tray with the apple and the tomato and move it {side} using both "
        "handles."
    ),
    (
        "Place both ingredients on the tray, lift it by its two handles and set it "
        "down further {side}."
    ),
)
SURFACE = {"friction": "0.7 0.005 0.0001", "solref": ".004 1", "solimp": ".95 .99 .001"}
CONTACT = {
    "friction": "1.2 .01 .0001",
    "condim": "4",
    "solref": ".004 1",
    "solimp": ".95 .99 .001",
}
WOOD, HANDLE, MARK = "0.72 0.56 0.38 1", "0.10 0.10 0.11 1", "0.80 0.79 0.75 1"


def food_width(name):
    return DIMS[name]["size"][1]


def handle(tray, side):
    """A handle's grasp point base (x, y, table z) for the tray at tray (x, y);
    its grasp point is POST_Z above."""
    sign = 1 if side == "left" else -1
    return [tray[0], tray[1] + sign * POST_Y, TABLE_Z]


def on_tray(tray, name):
    sign = 1 if SIDE_OF[name] == "left" else -1
    return [tray[0], tray[1] + sign * SPOT_Y, TABLE_Z + 2 * PLATE[2]]


def target(spec):
    """Where the tray goes (x, y)."""
    sign = 1 if spec["side"] == "left" else -1
    x, y = spec["tray"]["xy"]
    return [x, y + sign * SHIFT]


def carry_poses(tray, goal):
    """Per hand: the key poses for carrying the tray from tray to goal (x, y)
    by its handles: pregrasp, grasp, up, over, down, retreat."""
    poses = {}
    for side in tt.SIDES:
        at = tt.pick_place(
            side,
            handle(tray, side),
            POST_Z,
            2 * POST[0],
            handle(goal, side),
            CARRY_LIFT,
        )
        down = at["place"] - [0, 0, tt.SET_GAP]
        poses[side] = {
            "rotation": at["rotation"],
            "pregrasp": at["pregrasp"],
            "grasp": at["grasp"],
            "up": at["lift"],
            "over": down + [0, 0, CARRY_LIFT],
            "down": down,
            "retreat": down + tt.PRE * (at["rotation"] @ [0, 0, 1]),
        }
    return poses


def episode_poses(spec):
    tray = spec["tray"]["xy"]
    load = {
        name: tt.pick_place(
            SIDE_OF[name],
            [*spec["foods"][name]["xy"], TABLE_Z],
            GRASP_Z,
            food_width(name),
            on_tray(tray, name),
            LIFT,
        )
        for name in FOODS
    }
    return load, carry_poses(tray, target(spec))


def plan(kin, spec):
    load, carry = episode_poses(spec)
    for name in FOODS:
        side = SIDE_OF[name]
        tt.plan(
            kin,
            side,
            [
                load[name][k]
                for k in ("pregrasp", "grasp", "lift", "above", "place", "retreat")
            ]
            + [
                carry[side][k]
                for k in ("pregrasp", "grasp", "up", "over", "down", "retreat")
            ],
        )


def draw(rng):
    def jitter(xy):
        return [float(v) for v in np.asarray(xy) + rng.uniform(-1, 1, 2) * JITTER]

    side = str(rng.choice(("left", "right")))
    wording = INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))]
    return {
        "tray": {"xy": jitter(TRAY_XY)},
        "foods": {n: {"xy": jitter(FOOD_XY[n])} for n in FOODS},
        "side": side,
        "instruction": wording.format(side=side),
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
            "task": "robosyn_tray",
            "version": 1,
            "seed": int(seed),
            "draws": attempt,
            **spec,
            "target": target(spec),
            "ready": tt.READY,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def add_tray(world, xy):
    """The tray body tray_part: plate, rim and two handle posts, as boxes and
    cylinders."""
    body = pp.free_body(
        world, "tray", [*xy, TABLE_Z], 0.0, MASS["tray"], [0.2, 0.47, 0.06]
    )
    geoms = [
        ("plate", "box", [0, 0, PLATE[2]], PLATE, WOOD),
        (
            "rim_near",
            "box",
            [-RIM_HALF[0] + RIM[0], 0, 2 * PLATE[2] + RIM[1]],
            [RIM[0], RIM_HALF[1], RIM[1]],
            WOOD,
        ),
        (
            "rim_far",
            "box",
            [RIM_HALF[0] - RIM[0], 0, 2 * PLATE[2] + RIM[1]],
            [RIM[0], RIM_HALF[1], RIM[1]],
            WOOD,
        ),
        (
            "rim_left",
            "box",
            [0, RIM_HALF[1] - RIM[0], 2 * PLATE[2] + RIM[1]],
            [RIM_HALF[0], RIM[0], RIM[1]],
            WOOD,
        ),
        (
            "rim_right",
            "box",
            [0, -RIM_HALF[1] + RIM[0], 2 * PLATE[2] + RIM[1]],
            [RIM_HALF[0], RIM[0], RIM[1]],
            WOOD,
        ),
        ("post_left", "cylinder", [0, POST_Y, 2 * PLATE[2] + POST[1]], POST, HANDLE),
        ("post_right", "cylinder", [0, -POST_Y, 2 * PLATE[2] + POST[1]], POST, HANDLE),
    ]
    for name, kind, pos, size, rgba in geoms:
        ET.SubElement(
            body,
            "geom",
            name=f"tray_{name}_visual",
            type=kind,
            pos=fmt(pos),
            size=fmt(size),
            rgba=rgba,
            **CONTACT,
        )


def build(spec):
    """Derived scene for spec: build_task's robot, table and lights plus the
    tray, the two ingredients and the marks of the tray's place on either
    side."""
    root, report = build_base()
    root.set("model", "franzi_robosyn_tray")
    asset, world = root.find("asset"), root.find("worldbody")
    x, y = spec["tray"]["xy"]
    for side, sign in (("left", 1), ("right", -1)):
        # Drawn only: a pale strip under where the tray's plate would land.
        ET.SubElement(
            world,
            "geom",
            name=f"mark_{side}",
            type="box",
            pos=fmt([x, y + sign * SHIFT, TABLE_Z + 0.0003]),
            size=fmt([PLATE[0] + 0.01, PLATE[1] + 0.01, 0.0003]),
            rgba=MARK,
            contype="0",
            conaffinity="0",
        )
    add_tray(world, spec["tray"]["xy"])
    for name in FOODS:
        pos = [*spec["foods"][name]["xy"], TABLE_Z + 0.0005]
        body = pp.free_body(world, name, pos, 0.0, MASS[name], DIMS[name]["size"])
        pp.add_geoms(
            body, name, *pp.mesh_assets(asset, name, PROPS[name], DERIVED), CONTACT
        )
    report["task"] = {k: spec[k] for k in ("task", "version", "seed", "instruction")}
    return root, report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    spec = sample_episode(args.seed)
    root, report = build(spec)
    (ROOT / "outputs").mkdir(exist_ok=True)
    ET.ElementTree(root).write(
        ROOT / "outputs/robosyn_tray_task.xml", encoding="unicode"
    )
    (ROOT / "outputs/robosyn_tray_episode.json").write_text(json.dumps(spec, indent=2))
    print(json.dumps(spec, indent=2))
