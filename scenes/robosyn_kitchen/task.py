"""Kitchen long task with RoboSynChallenge props: episode sampling, key poses
and the scene.

A pot with its lid stands in the middle of the table, a lid rest on its left
and a push button beyond it; on the right are a plate and two ingredients, a
tomato and an apple. The instruction names which ingredient goes into the pot
and which onto the plate. The expert works with both arms at once where the
order allows:

1. the left hand lifts the lid off onto its rest while the right hand picks up
   the pot's ingredient;
2. the right hand lowers that ingredient into the pot;
3. the left hand puts the lid back while the right hand sets the other
   ingredient on the plate;
4. the left hand presses the button.

The "swap" variant starts with one ingredient already in the covered pot: the
left hand takes the lid off, the right hand lifts that ingredient out onto
the plate and puts the other in, the left hand covers the pot and presses the
button.

A seed fixes the layout (jitter of every prop, which ingredient sits in which
slot, which goes where) and the wording of the instruction. Every draw is
checked with the shared IK (each key pose, and each 1 cm waypoint of the
straight moves) and redrawn until it plans.
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
PROPS = robosyn.prepare(DERIVED, ("pot", "lid", "tomato", "apple", "plate", "button"))
DIMS = {name: prop["dims"] for name, prop in PROPS.items()}
# Layout (world, m): x ahead of the robot, y to its left. Every grasp is in
# the band the 45-degree grasp reaches and the head camera sees (x 0.33-0.50).
# The gripper housing spans about 85 mm either side of the grasp point and 65 mm
# behind it, so whatever the hands work next to is at least that far away or
# lower than the grasp.
POT_XY = (0.43, 0.0)
POT_YAW = np.radians(45)  # handle toward the back left, clear of the lid rest
REST_XY = (0.44, 0.28)  # the lid's rest (a cork trivet)
BUTTON_XY = (0.37, 0.47)
PLATE_XY = (0.50, -0.28)
SLOTS = ((0.36, -0.26), (0.36, -0.40))  # the ingredients' starting spots
JITTER = 0.008  # every prop, in x and y (m)
YAW_JITTER = np.radians(8)  # the pot's handle
REST_RADIUS, REST_TOP = 0.12, TABLE_Z + 0.004
FOODS = ("tomato", "apple")
MASS = {"pot": 0.9, "lid": 0.15, "plate": 0.25, "tomato": 0.09, "apple": 0.13}
GRASP_Z = 0.030  # grasp point above an ingredient's base; tips stay 18 mm clear
KNOB_Z = sum(DIMS["lid"]["knob_z"]) / 2  # grasp point above the lid's rim
LIFT = 0.10  # an ingredient's lift: its base clears the pot rim by 2 cm
LID_LIFT = 0.06
# The 150 g lid slid off the knob at the default squeeze (~1 N then); 8 mm past
# contact saturates the finger servos' 6 N.
LID_SQUEEZE = 0.008
PRESS = 0.004  # button press depth (its travel is 5 mm)
PRESSED = 0.003  # a press counts from this depth
INSTRUCTIONS = (
    (
        "Cook the {pot}: take the lid off the pot, put the {pot} in, put the {plate} "
        "on the plate, cover the pot and press the button."
    ),
    (
        "Put the {pot} in the pot and the {plate} on the plate, then put the lid "
        "back on and press the button."
    ),
    (
        "Lift the pot lid, lower the {pot} into the pot, serve the {plate} on the "
        "plate, close the lid and press the button."
    ),
    (
        "The {pot} goes into the pot and the {plate} onto the plate; cover the pot "
        "again and press the button to start cooking."
    ),
)
SURFACE = {"friction": "0.7 0.005 0.0001", "solref": ".004 1", "solimp": ".95 .99 .001"}
CONTACT = {
    "friction": "1.2 .01 .0001",
    "condim": "4",
    "solref": ".004 1",
    "solimp": ".95 .99 .001",
}
SWAP_INSTRUCTIONS = (
    (
        "Swap the ingredients: take the {plate} out of the pot onto the plate, put "
        "the {pot} in, cover the pot and press the button."
    ),
    (
        "Take the lid off, move the {plate} from the pot to the plate, put the {pot} "
        "in the pot, close the lid and press the button."
    ),
    (
        "Serve the {plate} from the pot onto the plate and cook the {pot} instead: "
        "lid off, swap them, lid on, then press the button."
    ),
)
VARIANTS = ("cook", "swap")
OUT_RISE = 0.12  # an ingredient lifted out of the pot: its base 4 cm over the rim
REST_RGBA = "0.66 0.52 0.36 1"


def food_width(name):
    """An ingredient's width along y, where the fingers close."""
    return DIMS[name]["size"][1]


def knob_width():
    return DIMS["lid"]["knob_grip_width"]


def lid_on_pot(pot):
    """Where the lid's base sits on the pot (x, y, z)."""
    return [pot[0], pot[1], TABLE_Z + DIMS["pot"]["rim_z"]]


def lid_on_rest(rest):
    return [rest[0], rest[1], REST_TOP]


def in_pot(pot):
    """An ingredient's base on the pot floor, centred."""
    return [pot[0], pot[1], TABLE_Z + DIMS["pot"]["floor_z"]]


def on_plate(plate):
    return [plate[0], plate[1], TABLE_Z + DIMS["plate"]["floor_z"]]


def button_top(button):
    return [button[0], button[1], TABLE_Z + DIMS["button"]["cap_z"][1]]


def from_pot(side, base, width, target):
    """Key poses for lifting an ingredient out of the pot onto target: in and
    out straight down and up over the pot centre (a 45-degree approach would
    bring the hand in over the rim), then as pick_place."""
    poses = tt.pick_place(side, base, GRASP_Z, width, target, OUT_RISE, 0.06)
    poses["pregrasp"] = poses["grasp"] + [0, 0, OUT_RISE + 0.02]
    poses["lift"] = poses["grasp"] + [0, 0, OUT_RISE]
    return poses


def episode_poses(spec):
    """Every key pose of the episode as the expert visits them, per move."""
    pot, rest = spec["pot"]["xy"], spec["rest"]["xy"]
    first, second = spec["into_pot"], spec["onto_plate"]
    knob, z = knob_width(), KNOB_Z
    foods = {n: [*spec["foods"][n]["xy"], TABLE_Z] for n in FOODS}
    if spec["variant"] == "swap":
        foods[second] = in_pot(pot)
    return {
        "lid_off": tt.pick_place(
            "left", lid_on_pot(pot), z, knob, lid_on_rest(rest), LID_LIFT
        ),
        "to_pot": tt.pick_place(
            "right", foods[first], GRASP_Z, food_width(first), in_pot(pot), LIFT
        ),
        "lid_back": tt.pick_place(
            "left", lid_on_rest(rest), z, knob, lid_on_pot(pot), LID_LIFT
        ),
        "to_plate": tt.pick_place(
            "right",
            foods[second],
            GRASP_Z,
            food_width(second),
            on_plate(spec["plate"]["xy"]),
            LIFT,
            0.06,
        ),
        "from_pot": from_pot(
            "right", in_pot(pot), food_width(second), on_plate(spec["plate"]["xy"])
        ),
        "press": tt.press_poses(button_top(spec["button"]["xy"]), "left"),
    }


def plan(kin, spec):
    """Raises RuntimeError where a key pose or a 1 cm waypoint between them
    has no IK solution inside the limits."""
    poses = episode_poses(spec)
    carry = ["pregrasp", "grasp", "lift", "above", "place", "retreat"]
    tt.plan(
        kin,
        "left",
        [poses["lid_off"][k] for k in carry]
        + [poses["lid_back"][k] for k in carry]
        + [poses["press"][k] for k in ("above", "touch", "above")],
    )
    into = [
        poses["to_pot"][k]
        for k in ("pregrasp", "grasp", "lift", "above", "place", "rise")
    ]
    if spec["variant"] == "swap":
        out = [
            poses["from_pot"][k]
            for k in ("pregrasp", "grasp", "lift", "above", "place", "retreat")
        ]
        tt.plan(kin, "right", out + into)
    else:
        tt.plan(kin, "right", into + [poses["to_plate"][k] for k in carry])


def draw(rng, variant):
    def jitter(xy):
        return [float(v) for v in np.asarray(xy) + rng.uniform(-1, 1, 2) * JITTER]

    slots = rng.permutation(len(SLOTS))
    into_pot = str(rng.permutation(FOODS)[0])
    onto_plate = next(f for f in FOODS if f != into_pot)
    wordings = SWAP_INSTRUCTIONS if variant == "swap" else INSTRUCTIONS
    wording = wordings[int(rng.integers(len(wordings)))]
    spec = {
        "variant": variant,
        "pot": {
            "xy": jitter(POT_XY),
            "yaw": float(POT_YAW + rng.uniform(-1, 1) * YAW_JITTER),
        },
        "rest": {"xy": jitter(REST_XY)},
        "button": {"xy": jitter(BUTTON_XY)},
        "plate": {"xy": jitter(PLATE_XY)},
        "foods": {f: {"xy": jitter(SLOTS[int(k)])} for f, k in zip(FOODS, slots)},
        "into_pot": into_pot,
        "onto_plate": onto_plate,
        "instruction": wording.format(pot=into_pot, plate=onto_plate),
    }
    if variant == "swap":
        spec["foods"][onto_plate] = {"xy": spec["pot"]["xy"], "in_pot": True}
    return spec


def sample_episode(seed, kin=None, attempts=100, variant="cook"):
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
            "task": "robosyn_kitchen",
            "version": 1,
            "seed": int(seed),
            "draws": attempt,
            **spec,
            "ready": tt.READY,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def build(spec):
    """Derived scene for spec: build_task's robot, table and lights plus the
    lid rest, the pot with its lid on, the plate, the ingredients and the
    button, at their sampled poses."""
    root, report = build_base()
    root.set("model", "franzi_robosyn_kitchen")
    asset, world = root.find("asset"), root.find("worldbody")
    rx, ry = spec["rest"]["xy"]
    ET.SubElement(
        world,
        "geom",
        name="lid_rest",
        type="cylinder",
        pos=fmt([rx, ry, (TABLE_Z + REST_TOP) / 2]),
        size=fmt([REST_RADIUS, (REST_TOP - TABLE_Z) / 2]),
        rgba=REST_RGBA,
        **SURFACE,
    )
    pot, lid, plate = DIMS["pot"], DIMS["lid"], DIMS["plate"]
    placed = {
        "pot": (
            [*spec["pot"]["xy"], TABLE_Z],
            spec["pot"]["yaw"],
            [2 * pot["outer_radius"]] * 2 + [pot["rim_z"]],
        ),
        "lid": (
            lid_on_pot(spec["pot"]["xy"]),
            0.0,
            [2 * lid["radius"]] * 2 + [lid["knob_z"][1]],
        ),
        "plate": (
            [*spec["plate"]["xy"], TABLE_Z],
            0.0,
            [2 * plate["radius"]] * 2 + [plate["rim_z"]],
        ),
        **{
            f: ([*spec["foods"][f]["xy"], TABLE_Z + 0.0005], 0.0, DIMS[f]["size"])
            for f in FOODS
        },
    }
    if spec["variant"] == "swap":
        # The served ingredient starts on the pot's floor, under the lid.
        inside = in_pot(spec["pot"]["xy"])
        placed[spec["onto_plate"]] = (
            [inside[0], inside[1], inside[2] + 0.0005],
            0.0,
            DIMS[spec["onto_plate"]]["size"],
        )
    for name, (pos, yaw, size) in placed.items():
        body = pp.free_body(world, name, pos, yaw, MASS[name], size)
        contact = {**CONTACT, "friction": "0.9 .01 .0001"} if name == "pot" else CONTACT
        pp.add_geoms(
            body, name, *pp.mesh_assets(asset, name, PROPS[name], DERIVED), contact
        )
    robosyn.add_button(
        root, "button", PROPS["button"], DERIVED, spec["button"]["xy"], CONTACT
    )
    report["task"] = {k: spec[k] for k in ("task", "version", "seed", "instruction")}
    return root, report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", choices=VARIANTS, default="cook")
    args = p.parse_args()
    spec = sample_episode(args.seed, variant=args.variant)
    root, report = build(spec)
    (ROOT / "outputs").mkdir(exist_ok=True)
    ET.ElementTree(root).write(
        ROOT / "outputs/robosyn_kitchen_task.xml", encoding="unicode"
    )
    (ROOT / "outputs/robosyn_kitchen_episode.json").write_text(
        json.dumps(spec, indent=2)
    )
    print(json.dumps(spec, indent=2))
