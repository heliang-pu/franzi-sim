"""Lab bench task with RoboSynChallenge props: episode sampling, key poses and
the scene.

RoboSyn's magnetic stirrer (mixer_operating) stands in the middle of the table,
its knob panel toward the robot; RoboSyn's beaker stands on the right and a
tray beyond it; RoboSyn's test tube rack with three tubes stands at the back
left as scenery. The instruction names the knob that switches the stirrer:

1. the right hand puts the beaker on the stirrer's plate;
2. the hand on the named knob's side presses it: the stirrer runs (its
   indicator turns green) for 3 s;
3. the same hand presses it again: the stirrer stops;
4. the right hand moves the beaker onto the tray.

A press counts when the knob goes 3 mm in, and the stirrer only starts with the
beaker on its plate. A seed fixes the layout (jitter of every prop), the knob
and the wording of the instruction. Every draw is checked with the shared IK
(each key pose, and each 1 cm waypoint of the straight moves) and redrawn
until it plans.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robots.franzi.scene import build as build_base
from sim import ROOT, TABLE_Z, fmt, robosyn
from sim import props as pp
from sim import tabletop as tt

DERIVED = Path(__file__).resolve().parent / "assets/derived"
PROPS = robosyn.prepare(DERIVED, ("mixer", "beaker_blue", "rack", "tube"))
DIMS = {name: prop["dims"] for name, prop in PROPS.items()}
# Layout (world, m).
MIXER_XY = (0.47, 0.0)
BEAKER_XY = (0.37, -0.30)
TRAY_XY = (0.47, -0.30)
TRAY_HALF = (0.06, 0.06, 0.003)
RACK_XY, RACK_YAW = (0.55, 0.33), np.radians(90)
TUBE_SLOTS = (-0.075, 0.0, 0.075)  # along the rack's length, in its front row
JITTER = 0.006
GRASP_Z = 0.040  # the beaker held at its centre of mass
LIFT = 0.10  # the beaker's base clears the stirrer's 57 mm top by 4 cm
SQUEEZE = 0.007
PRESS, PRESSED, RELEASED = 0.0035, 0.003, 0.001  # knob depths (travel 4 mm)
RUN = 3.0  # seconds the stirrer runs before the second press
KNOB_LIFT = 0.006  # the press point: the knob's top, above its centroid
INSTRUCTIONS = (
    (
        "Stir the solution: put the beaker on the stirrer, switch it on with the "
        "{side} knob, let it run, switch it off and move the beaker to the tray."
    ),
    (
        "Place the beaker on the magnetic stirrer, press the {side} knob to start "
        "it, press it again to stop it, then set the beaker on the tray."
    ),
    (
        "Run the stirrer with the beaker on it using the {side} knob, then take the "
        "beaker off onto the tray."
    ),
)
CONTACT = {
    "friction": "1.2 .03 .0001",
    "condim": "4",
    "solref": ".004 1",
    "solimp": ".95 .99 .001",
}
LED_OFF, LED_ON = "0.25 0.27 0.25 1", "0.15 0.95 0.30 1"


def width():
    return DIMS["beaker_blue"]["size"][1]


def plate(mixer):
    """Where the beaker's base sits on the stirrer's plate (x, y, z)."""
    px, py = DIMS["mixer"]["plate_xy"]
    return [mixer[0] + px, mixer[1] + py, TABLE_Z + DIMS["mixer"]["plate_z"]]


def knob_top(mixer, side):
    """The named knob's press point (world): its centroid lifted along the
    panel's outward normal to its top face."""
    knob = DIMS["mixer"]["knobs"][side]["centre"]
    outward = -np.asarray(DIMS["mixer"]["press_axis"])
    return (
        np.array([mixer[0], mixer[1], TABLE_Z]) + knob + KNOB_LIFT * outward
    ).tolist()


def on_tray(tray):
    return [tray[0], tray[1], TABLE_Z + 2 * TRAY_HALF[2]]


def episode_poses(spec):
    mixer, side = spec["mixer"]["xy"], spec["side"]
    start = [*spec["beaker"]["xy"], TABLE_Z]
    load = tt.pick_place("right", start, GRASP_Z, width(), plate(mixer), LIFT)
    unload = tt.pick_place(
        "right", plate(mixer), GRASP_Z, width(), on_tray(spec["tray"]["xy"]), LIFT
    )
    return load, unload, tt.press_poses(knob_top(mixer, side), side)


def plan(kin, spec):
    load, unload, press = episode_poses(spec)
    steps = ("pregrasp", "grasp", "lift", "above", "place", "retreat")
    knob = [press[k] for k in ("above", "touch", "above", "touch", "above")]
    right = [load[k] for k in steps]
    if spec["side"] == "right":
        right += knob
    right += [unload[k] for k in steps]
    tt.plan(kin, "right", right)
    if spec["side"] == "left":
        tt.plan(kin, "left", knob)


def draw(rng):
    def jitter(xy):
        return [float(v) for v in np.asarray(xy) + rng.uniform(-1, 1, 2) * JITTER]

    side = str(rng.choice(("left", "right")))
    wording = INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))]
    return {
        "mixer": {"xy": jitter(MIXER_XY)},
        "beaker": {"xy": jitter(BEAKER_XY)},
        "tray": {"xy": jitter(TRAY_XY)},
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
            "task": "robosyn_lab",
            "version": 1,
            "seed": int(seed),
            "draws": attempt,
            **spec,
            "ready": tt.READY,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def build(spec):
    """Derived scene for spec: the robot scene (robot, table and lights) plus the
    stirrer with its knobs and indicator, the beaker, the tray and the rack of
    tubes (scenery)."""
    root, report = build_base()
    root.set("model", "franzi_robosyn_lab")
    asset, world = root.find("asset"), root.find("worldbody")
    robosyn.add_mixer(
        root, "mixer", PROPS["mixer"], DERIVED, spec["mixer"]["xy"], CONTACT
    )
    # The indicator: drawn only, recoloured by the scene while the stirrer runs.
    knobs = DIMS["mixer"]["knobs"]
    led = np.mean([knobs["left"]["centre"], knobs["right"]["centre"]], 0)
    outward = -np.asarray(DIMS["mixer"]["press_axis"])
    ET.SubElement(
        world.find("body[@name='mixer_part']"),
        "geom",
        name="mixer_led",
        type="cylinder",
        pos=fmt(led + 0.002 * outward),
        size="0.006 0.0015",
        zaxis=fmt(outward),
        rgba=LED_OFF,
        contype="0",
        conaffinity="0",
    )
    tx, ty = spec["tray"]["xy"]
    ET.SubElement(
        world,
        "geom",
        name="tray",
        type="box",
        pos=fmt([tx, ty, TABLE_Z + TRAY_HALF[2]]),
        size=fmt(TRAY_HALF),
        rgba="0.85 0.86 0.88 1",
        **{k: CONTACT[k] for k in ("friction", "solref", "solimp")},
    )
    body = pp.free_body(
        world,
        "beaker",
        [*spec["beaker"]["xy"], TABLE_Z + 0.0005],
        0.0,
        0.12,
        DIMS["beaker_blue"]["size"],
    )
    pp.add_geoms(
        body,
        "beaker",
        *pp.mesh_assets(asset, "beaker", PROPS["beaker_blue"], DERIVED),
        CONTACT,
    )
    # Scenery: the rack and three tubes in its front row, drawn only.
    rack = ET.SubElement(
        world,
        "body",
        name="rack_part",
        pos=fmt([*RACK_XY, TABLE_Z]),
        quat=fmt([np.cos(RACK_YAW / 2), 0, 0, np.sin(RACK_YAW / 2)]),
    )
    pp.add_geoms(
        rack, "rack", *pp.mesh_assets(asset, "rack", PROPS["rack"], DERIVED), CONTACT
    )
    tube_mesh = pp.mesh_assets(asset, "tube", PROPS["tube"], DERIVED)
    depth = DIMS["rack"]["size"][1] / 4
    for k, along in enumerate(TUBE_SLOTS):
        holder = ET.SubElement(
            rack, "body", name=f"tube{k}_part", pos=fmt([along, depth, 0.004])
        )
        pp.add_geoms(holder, f"tube{k}", *tube_mesh, CONTACT)
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
        ROOT / "outputs/robosyn_lab_task.xml", encoding="unicode"
    )
    (ROOT / "outputs/robosyn_lab_episode.json").write_text(json.dumps(spec, indent=2))
    print(json.dumps(spec, indent=2))
