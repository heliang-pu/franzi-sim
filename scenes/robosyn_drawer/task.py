"""Drawer long task with RoboSynChallenge props: episode sampling, key poses
and the scene.

RoboSyn's sliding drawer stands at the back left, facing the robot; two
RoboSyn beakers, blue and amber, stand on the right, with a push button beyond
them; a hand-over pad lies in the middle. The instruction names the beaker to
put away. The right arm can't reach into the drawer and the left can't reach the
objects, so the hands pass it over:

1. the left hand pulls the drawer open while the right hand carries the object
   to the hand-over pad;
2. the left hand takes it from the pad and lowers it into the drawer;
3. the left hand pushes the drawer shut;
4. the right hand presses the button.

The "retrieve" variant runs the other way: the named beaker starts in the
shut drawer, and its slot on the table is empty. The left hand opens the
drawer and sets the beaker on the pad, the right hand stands it back in its
slot while the left hand shuts the drawer, and the right hand presses the
button.

A seed fixes the layout (jitter of every prop, which object stands where,
which one is named) and the wording of the instruction. Every draw is checked
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
ITEMS = ("beaker_blue", "beaker_amber")
NAMES = {"beaker_blue": "blue beaker", "beaker_amber": "amber beaker"}
PROPS = robosyn.prepare(DERIVED, ("drawer", "button", *ITEMS))
DIMS = {name: prop["dims"] for name, prop in PROPS.items()}
# Layout (world, m): x ahead of the robot, y to its left. The cabinet's raw +x
# (the way the drawer slides out) is turned toward the robot.
CABINET_XY = (0.555, 0.20)
CABINET_YAW = np.pi
# How far the drawer opens: enough that a beaker in its middle is wholly in
# front of the cabinet's top (at 12 cm its rim caught the top's edge).
PULL = 0.14
PAD_XY = (0.36, -0.03)  # the hand-over pad: both hands reach it, the drawer's
# open side wall stays 6 cm clear of the left hand's housing there
PAD_RADIUS, PAD_TOP = 0.05, TABLE_Z + 0.003
SLOTS = ((0.37, -0.22), (0.37, -0.36))
BUTTON_XY = (0.38, -0.46)
JITTER = 0.006  # every prop, in x and y (m)
MASS = {"beaker_blue": 0.12, "beaker_amber": 0.12}
# Grasp points above a beaker's base. The right hand carries it to the pad at
# its centre of mass: held 12 mm under the rim, it swung on the pads as the hand
# turned and fell from 6 cm (seeds 1, 7). The left hand takes it into the
# drawer there, 12 mm under the rim, so the hand stays above the drawer's 57 mm
# walls, which the housing behind the grasp point would otherwise hit.
PICK_Z, STORE_Z = 0.040, 0.060
SQUEEZE = 0.007  # ~5 N per finger; the 120 g beakers slid at 4 mm
KNOB_GRASP = 0.12  # raw x along the knob where the fingers close
KNOB_SQUEEZE = 0.006
# Objects off the table and the pad: a carried beaker's base clears the other
# (72 mm) beaker it passes over (lifted 6 cm, it knocked it 3 cm up, seed 0).
LIFT = 0.10
CARRY = 0.12  # an object's base over the drawer floor while carried in
PRESS, PRESSED = 0.004, 0.003  # button press depth and the depth that counts
INSTRUCTIONS = (
    (
        "Put the {item} away: open the drawer, pass the {item} to the left hand, "
        "put it in the drawer, close the drawer and press the button."
    ),
    "Store the {item} in the drawer, close it, then press the button.",
    (
        "Open the drawer, hand the {item} across, lower it into the drawer, shut the "
        "drawer and press the button."
    ),
    (
        "The {item} goes into the drawer: open it, move the {item} in, close it and "
        "press the button."
    ),
)
RETRIEVE_INSTRUCTIONS = (
    (
        "Take the {item} out of the drawer, stand it back in its empty spot, close "
        "the drawer and press the button."
    ),
    (
        "Open the drawer, pass the {item} to the right hand, put it back on the "
        "table, shut the drawer and press the button."
    ),
    (
        "Get the {item} out of the drawer and return it to its place, then close "
        "the drawer and press the button."
    ),
)
VARIANTS = ("store", "retrieve")
SURFACE = {"friction": "0.7 0.005 0.0001", "solref": ".004 1", "solimp": ".95 .99 .001"}
# Torsional friction 0.03 m (the default 0.01 let a beaker turn about the
# fingers' closing axis).
CONTACT = {
    "friction": "1.2 .03 .0001",
    "condim": "4",
    "solref": ".004 1",
    "solimp": ".95 .99 .001",
}
PAD_RGBA = "0.32 0.34 0.38 1"


def width(name):
    """An object's width along y, where the fingers close."""
    return DIMS[name]["size"][1]


def drawer_point(cabinet, q, local):
    """World point of a drawer-frame point (raw x, y, z, m) with the drawer
    out by q."""
    c, s = np.cos(CABINET_YAW), np.sin(CABINET_YAW)
    x, y, z = local[0] + q, local[1], local[2]
    return [cabinet[0] + c * x - s * y, cabinet[1] + s * x + c * y, TABLE_Z + z]


def knob(cabinet, q):
    """The knob's grasp point's base (x, y, table z) with the drawer out by q;
    the grasp point is KNOB_Z above it."""
    x, y, _ = drawer_point(cabinet, q, [KNOB_GRASP, 0, 0])
    return [x, y, TABLE_Z]


KNOB_Z = DIMS["drawer"]["knob_z"]


def in_drawer(cabinet, q=PULL):
    """An object's base on the floor of the drawer out by q, centred."""
    return drawer_point(cabinet, q, [0, 0, DIMS["drawer"]["floor_z"]])


def on_pad(pad):
    return [pad[0], pad[1], PAD_TOP]


def button_top(button):
    return [button[0], button[1], TABLE_Z + DIMS["button"]["cap_z"][1]]


def drawer_moves(cabinet, q0, q1):
    """Key poses for moving the drawer from q0 out to q1 by its knob with the
    left hand: pregrasp, grasp, slid, retreat."""
    knob_w = DIMS["drawer"]["knob_width"]
    at = tt.pick_place(
        "left", knob(cabinet, q0), KNOB_Z, knob_w, knob(cabinet, q1), 0.0
    )
    slid = at["place"] - [0, 0, tt.SET_GAP]
    axis = at["rotation"] @ [0, 0, 1]
    return {
        "rotation": at["rotation"],
        "pregrasp": at["pregrasp"],
        "grasp": at["grasp"],
        "slid": slid,
        "retreat": slid + tt.PRE * axis,
    }


def episode_poses(spec):
    cabinet, item = spec["cabinet"]["xy"], spec["item"]
    start = [*spec["items"][item]["xy"], TABLE_Z]
    pad = on_pad(spec["pad"]["xy"])
    if spec["variant"] == "retrieve":
        home = [*spec["items"][item]["home"], TABLE_Z]
        return {
            "open": drawer_moves(cabinet, 0.0, PULL),
            "to_pad": tt.pick_place(
                "left", in_drawer(cabinet), STORE_Z, width(item), pad, CARRY
            ),
            "to_home": tt.pick_place("right", pad, PICK_Z, width(item), home, LIFT),
            "close": drawer_moves(cabinet, PULL, 0.0),
            "press": tt.press_poses(button_top(spec["button"]["xy"]), "right"),
        }
    return {
        "open": drawer_moves(cabinet, 0.0, PULL),
        "to_pad": tt.pick_place("right", start, PICK_Z, width(item), pad, LIFT),
        "to_drawer": tt.pick_place(
            "left", pad, STORE_Z, width(item), in_drawer(cabinet), LIFT, CARRY
        ),
        "close": drawer_moves(cabinet, PULL, 0.0),
        "press": tt.press_poses(button_top(spec["button"]["xy"]), "right"),
    }


def plan(kin, spec):
    """Raises RuntimeError where a key pose or a 1 cm waypoint between them
    has no IK solution inside the limits."""
    poses = episode_poses(spec)
    slide = ["pregrasp", "grasp", "slid", "retreat"]
    carry = ["pregrasp", "grasp", "lift", "above", "place", "retreat"]
    if spec["variant"] == "retrieve":
        tt.plan(
            kin,
            "left",
            [poses["open"][k] for k in slide]
            + [poses["to_pad"][k] for k in carry]
            + [poses["close"][k] for k in slide],
        )
        tt.plan(
            kin,
            "right",
            [poses["to_home"][k] for k in carry]
            + [poses["press"][k] for k in ("above", "touch", "above")],
        )
        return
    tt.plan(
        kin,
        "left",
        [poses["open"][k] for k in slide]
        + [
            poses["to_drawer"][k]
            for k in ("pregrasp", "grasp", "lift", "above", "place", "rise")
        ]
        + [poses["close"][k] for k in slide],
    )
    tt.plan(
        kin,
        "right",
        [
            poses["to_pad"][k]
            for k in ("pregrasp", "grasp", "lift", "above", "place", "retreat")
        ]
        + [poses["press"][k] for k in ("above", "touch", "above")],
    )


def draw(rng, variant):
    def jitter(xy):
        return [float(v) for v in np.asarray(xy) + rng.uniform(-1, 1, 2) * JITTER]

    slots = rng.permutation(len(SLOTS))
    item = str(rng.choice(ITEMS))
    wordings = RETRIEVE_INSTRUCTIONS if variant == "retrieve" else INSTRUCTIONS
    wording = wordings[int(rng.integers(len(wordings)))]
    spec = {
        "variant": variant,
        "cabinet": {"xy": jitter(CABINET_XY)},
        "pad": {"xy": jitter(PAD_XY)},
        "button": {"xy": jitter(BUTTON_XY)},
        "items": {n: {"xy": jitter(SLOTS[int(k)])} for n, k in zip(ITEMS, slots)},
        "item": item,
        "instruction": wording.format(item=NAMES[item]),
    }
    if variant == "retrieve":
        # The named beaker starts in the shut drawer; its slot waits empty.
        slot = spec["items"][item]["xy"]
        spec["items"][item] = {
            "xy": in_drawer(spec["cabinet"]["xy"], 0.0)[:2],
            "in_drawer": True,
            "home": slot,
        }
    return spec


def sample_episode(seed, kin=None, attempts=100, variant="store"):
    """Episode spec for seed (plain JSON types): the first draw whose every
    key pose and straight move plans."""
    kin = kin or tt.kinematics(tt.SAMPLE_MARGIN)
    rng = np.random.default_rng(seed)
    for attempt in range(1, attempts + 1):
        spec = draw(rng, variant)
        try:
            plan(kin, spec)
        except RuntimeError:
            continue
        return {
            "task": "robosyn_drawer",
            "version": 1,
            "seed": int(seed),
            "draws": attempt,
            **spec,
            "ready": tt.READY,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def build(spec):
    """Derived scene for spec: build_task's robot, table and lights plus the
    drawer cabinet, the hand-over pad, the three objects and the button."""
    root, report = build_base()
    root.set("model", "franzi_robosyn_drawer")
    asset, world = root.find("asset"), root.find("worldbody")
    px, py = spec["pad"]["xy"]
    ET.SubElement(
        world,
        "geom",
        name="handover_pad",
        type="cylinder",
        pos=fmt([px, py, (TABLE_Z + PAD_TOP) / 2]),
        size=fmt([PAD_RADIUS, (PAD_TOP - TABLE_Z) / 2]),
        rgba=PAD_RGBA,
        **SURFACE,
    )
    robosyn.add_drawer(
        root,
        "drawer",
        PROPS["drawer"],
        DERIVED,
        spec["cabinet"]["xy"],
        CABINET_YAW,
        CONTACT,
    )
    for name in ITEMS:
        pos = [*spec["items"][name]["xy"], TABLE_Z + 0.0005]
        if spec["items"][name].get("in_drawer"):
            # On the shut drawer's floor (it moves with the drawer).
            pos[2] = TABLE_Z + DIMS["drawer"]["floor_z"] + 0.0005
        body = pp.free_body(world, name, pos, 0.0, MASS[name], DIMS[name]["size"])
        pp.add_geoms(
            body, name, *pp.mesh_assets(asset, name, PROPS[name], DERIVED), CONTACT
        )
    robosyn.add_button(
        root, "button", PROPS["button"], DERIVED, spec["button"]["xy"], CONTACT
    )
    report["task"] = {k: spec[k] for k in ("task", "version", "seed", "instruction")}
    return root, report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", choices=VARIANTS, default="store")
    args = p.parse_args()
    spec = sample_episode(args.seed, variant=args.variant)
    root, report = build(spec)
    (ROOT / "outputs").mkdir(exist_ok=True)
    ET.ElementTree(root).write(
        ROOT / "outputs/robosyn_drawer_task.xml", encoding="unicode"
    )
    (ROOT / "outputs/robosyn_drawer_episode.json").write_text(
        json.dumps(spec, indent=2)
    )
    print(json.dumps(spec, indent=2))
