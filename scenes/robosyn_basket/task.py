"""Basket task with RoboSynChallenge props: episode sampling, key poses and
the scene.

RoboSyn's woven basket stands in the middle of the table, its handle arching
along y, and RoboSyn's milk carton stands on the left. A shopping run:

1. the left hand puts the milk into the basket;
2. the hand on the side the instruction names takes the handle from that side,
   lifts the basket 5 cm, carries it 13 cm that way and sets it down (the milk
   rides in it: friction only);
3. the same hand takes the milk out onto the counter mat on that side.

One thing goes in, 4 cm ahead of the basket's centre: the oval's corners close
in on the gripper housing (about 85 mm either side of the grasp point and 65
mm behind it) anywhere else, and the handle arches over the centre line.

A seed fixes the layout (jitter of every prop, the side) and the wording of the
instruction. Every draw is checked with the shared IK (each key pose, and each
1 cm waypoint of the straight moves) and redrawn until it plans.
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
ITEMS = ("milk",)
PROP_OF = {"milk": "milk", "basket": "basket"}
PROPS = robosyn.prepare(DERIVED, tuple(PROP_OF.values()))
DIMS = {name: PROPS[prop]["dims"] for name, prop in PROP_OF.items()}
SIDE_OF = {"milk": "left"}
# Layout (world, m).
BASKET_XY = (0.42, 0.0)
START = {"milk": (0.37, 0.31)}
INSIDE = {"milk": (-0.04, 0.0)}
COUNTER_Y = 0.43  # the counter mats, either side
COUNTER_X = 0.37
COUNTER_HALF = (0.06, 0.05, 0.002)
JITTER = 0.006
SHIFT = 0.13  # how far the basket is carried, left or right
MASS = {"basket": 0.15, "milk": 0.06}
GRASP_Z = {"milk": 0.060}  # high on the carton's straight sides
LIFT = 0.13  # things clear the basket's 112 mm rim by 1.8 cm
BASKET_LIFT = 0.05
HANDLE_SQUEEZE = 0.008  # saturates the finger servos' 6 N
# After letting go of the handle the hand rises this far before it turns back
# into the upright grasp: turned on the spot, a finger swept through the handle.
CLEAR = 0.12
HOVER = 0.06  # turned back, it comes down onto the milk from this far above
# The hand turns between the upright grasp and the handle's turned one only on
# the spot, this far above the handle's pre-grasp: turning while it moved, it
# swept the basket over (seed 0) and knocked the milk down (seed 1).
TURN_HIGH = 0.10
TARGET_TOL = (
    0.04  # the hanging basket swings about the handle bar, so it lands 1.6-2.8 cm off
)
INSTRUCTIONS = (
    (
        "Put the milk carton in the basket, carry the basket to the {side} by its "
        "handle and take the milk out onto the counter there."
    ),
    (
        "Shop for milk: pack it into the basket, move the basket {side} and unpack "
        "the milk onto the {side} counter."
    ),
    (
        "Load the milk into the basket, lift the basket by the handle, set it down "
        "on the {side} and put the milk on the counter beside it."
    ),
)
CONTACT = {
    "friction": "1.2 .01 .0001",
    "condim": "4",
    "solref": ".004 1",
    "solimp": ".95 .99 .001",
}
# The handle's torsional friction 0.03 m keeps the hanging basket from
# swinging about the fingers' closing axis.
HANDLE_CONTACT = {**CONTACT, "friction": "1.2 .03 .0001"}


def width(name):
    """Width where the fingers close (along y)."""
    return DIMS[name]["size"][1]


def counter(side):
    sign = 1 if side == "left" else -1
    return [COUNTER_X, sign * COUNTER_Y]


def handle_psi(side):
    """The handle runs along y, so a hand takes it from its own side: approach
    turned 90 degrees (toward -y for the left hand), fingers closing along x."""
    return -np.pi / 2 if side == "left" else np.pi / 2


def handle_base(basket):
    """The handle's grasp point base (x, y, table z) for the basket at basket;
    the grasp point is DIMS["basket"]["handle_mid_z"] above."""
    return [basket[0], basket[1], TABLE_Z]


def in_basket(basket, name):
    dx, dy = INSIDE[name]
    return [basket[0] + dx, basket[1] + dy, TABLE_Z + DIMS["basket"]["floor_z"]]


def target(spec):
    sign = 1 if spec["side"] == "left" else -1
    x, y = spec["basket"]["xy"]
    return [x, y + sign * SHIFT]


def carry_poses(basket, goal, side):
    return tt.pick_place(
        side,
        handle_base(basket),
        DIMS["basket"]["handle_mid_z"],
        DIMS["basket"]["handle_width"],
        handle_base(goal),
        BASKET_LIFT,
        psi=handle_psi(side),
    )


def episode_poses(spec):
    basket = spec["basket"]["xy"]
    pack = {
        n: tt.pick_place(
            SIDE_OF[n],
            [*spec["items"][n]["xy"], TABLE_Z],
            GRASP_Z[n],
            width(n),
            in_basket(basket, n),
            LIFT,
        )
        for n in ITEMS
    }
    side = spec["side"]
    goal = target(spec)
    unpack = tt.pick_place(
        side,
        in_basket(goal, "milk"),
        GRASP_Z["milk"],
        width("milk"),
        [*counter(side), TABLE_Z + 2 * COUNTER_HALF[2]],
        LIFT,
    )
    return pack, carry_poses(basket, goal, side), unpack


def plan(kin, spec):
    """Each hand's path, every key pose and 1 cm waypoint IK-checked: the left
    packs the milk; the named side's hand carries the basket (its approach
    turned) and unpacks the milk."""
    pack, carry, unpack = episode_poses(spec)
    side = spec["side"]
    steps = ("pregrasp", "grasp", "lift", "above", "place", "rise")
    upright = tt.grasp_rotation(side)
    q = tt.ready_pose(kin, side)
    if side == "left":
        for target_ in (pack["milk"][k] for k in steps):
            q = kin.line(side, q, target_, rotation=upright)[-1]
    else:
        tt.plan(kin, "left", [pack["milk"][k] for k in steps])
    high = carry["pregrasp"] + [0, 0, TURN_HIGH]
    q = kin.line(side, q, high, rotation=upright)[-1]
    q = kin.ik(side, high, q, rotation=carry["rotation"])
    q = kin.line(side, q, carry["pregrasp"], rotation=carry["rotation"])[-1]
    for k in ("grasp", "lift", "above", "place", "retreat"):
        q = kin.line(side, q, carry[k], rotation=carry["rotation"])[-1]
    clear = carry["retreat"] + [0, 0, CLEAR]
    q = kin.line(side, q, clear, rotation=carry["rotation"])[-1]
    q = kin.ik(side, clear, q, rotation=upright)
    q = kin.line(side, q, unpack["pregrasp"] + [0, 0, HOVER], rotation=upright)[-1]
    for k in ("pregrasp", "grasp", "lift", "above", "place", "retreat"):
        q = kin.line(side, q, unpack[k], rotation=upright)[-1]
    kin.line(side, q, tt.READY[side], rotation=upright)


def draw(rng):
    def jitter(xy):
        return [float(v) for v in np.asarray(xy) + rng.uniform(-1, 1, 2) * JITTER]

    side = str(rng.choice(("left", "right")))
    wording = INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))]
    return {
        "basket": {"xy": jitter(BASKET_XY)},
        "items": {n: {"xy": jitter(START[n])} for n in ITEMS},
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
            "task": "robosyn_basket",
            "version": 1,
            "seed": int(seed),
            "draws": attempt,
            **spec,
            "target": target(spec),
            "ready": tt.READY,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def build(spec):
    """Derived scene for spec: build_task's robot, table and lights plus the
    basket, the milk carton and the two counter mats."""
    root, report = build_base()
    root.set("model", "franzi_robosyn_basket")
    asset, world = root.find("asset"), root.find("worldbody")
    for side in tt.SIDES:
        ET.SubElement(
            world,
            "geom",
            name=f"counter_{side}",
            type="box",
            pos=fmt([*counter(side), TABLE_Z + COUNTER_HALF[2]]),
            size=fmt(COUNTER_HALF),
            rgba="0.30 0.45 0.62 1",
            **{k: CONTACT[k] for k in ("friction", "solref", "solimp")},
        )
    placed = {
        "basket": [*spec["basket"]["xy"], TABLE_Z + 0.0005],
        **{n: [*spec["items"][n]["xy"], TABLE_Z + 0.0005] for n in ITEMS},
    }
    for name, pos in placed.items():
        body = pp.free_body(world, name, pos, 0.0, MASS[name], DIMS[name]["size"])
        contact = HANDLE_CONTACT if name == "basket" else CONTACT
        pp.add_geoms(
            body,
            name,
            *pp.mesh_assets(asset, name, PROPS[PROP_OF[name]], DERIVED),
            contact,
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
        ROOT / "outputs/robosyn_basket_task.xml", encoding="unicode"
    )
    (ROOT / "outputs/robosyn_basket_episode.json").write_text(
        json.dumps(spec, indent=2)
    )
    print(json.dumps(spec, indent=2))
