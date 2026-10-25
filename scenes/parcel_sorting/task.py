"""Parcel sorting: each arm takes the parcels on its half of the belt and drops
each into the bin whose zone letter matches the parcel's label.

The belt conveyor runs along y in front of the robot on a 100 mm plinth, so the
belt top is at 0.90 m; its robot-side edge is at x = 0.45 and the parcels lie
at x = 0.49, which tilted grasps reach at that height. The belt is still in
this version. The three zone bins stand in front of the belt with their rims
at 0.80 m (x 0.20–0.40), A on the left, B in the middle and C on the right;
their zone signs face the robot from the bins' back edges, below the head
camera's line of sight to the belt. Two parcels lie on each half of the belt.
"""

from sim import prop_assets as pa
from sim.build_task import build as robot_scene
from sim.workcell import block, industrial_cell, stand

NAME = "parcel_sorting"
INSTRUCTION = "Sort each parcel into the bin whose zone letter matches its label."
CONVEYOR = (0.6475, 0.0)  # origin; belt centre x 0.525
PLINTH_Z = 0.10
BELT_Z = 0.80 + PLINTH_Z
PARCEL_X = 0.49
BIN_X, BIN_BASE_Z = 0.30, 0.65  # bins are 150 mm deep: rims at 0.80
BINS = {"A": 0.25, "B": 0.0, "C": -0.25}
# (prop, zone, y, height)
PARCELS = [
    ("parcel_box_s", "A", 0.30, 0.06),
    ("parcel_box_m", "B", 0.12, 0.08),
    ("parcel_pouch", "C", -0.12, 0.035),
    ("parcel_box_s", "B", -0.30, 0.06),
]
KEY_POINTS = [
    {
        "name": f"pick_{k}_{zone}",
        "side": "left" if y > 0 else "right",
        "xyz": [PARCEL_X, y, BELT_Z + h / 2],
        "grasp": "any",
    }
    for k, (_, zone, y, h) in enumerate(PARCELS)
] + [
    {"name": "drop_A", "side": "left", "xyz": [BIN_X, BINS["A"], 0.88], "grasp": "top"},
    {"name": "drop_B_left", "side": "left", "xyz": [BIN_X, 0.03, 0.88], "grasp": "top"},
    {
        "name": "drop_B_right",
        "side": "right",
        "xyz": [BIN_X, -0.03, 0.88],
        "grasp": "top",
    },
    {
        "name": "drop_C",
        "side": "right",
        "xyz": [BIN_X, BINS["C"], 0.88],
        "grasp": "top",
    },
]


def build():
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root, bench=False)
    block(root, "plinth", CONVEYOR, (0.52, 0.84), 0.0, PLINTH_Z)
    pa.add(root, "belt_conveyor", "conveyor", pos=(*CONVEYOR, PLINTH_Z))
    for zone, y in BINS.items():
        stand(root, f"bin_stand_{zone}", (BIN_X, y), (0.18, 0.16), top_z=BIN_BASE_Z)
        pa.add(
            root,
            "sort_bin",
            f"bin_{zone}",
            pos=(BIN_X, y, BIN_BASE_Z),
            variant_name=zone,
        )
    for k, (prop, zone, y, _) in enumerate(PARCELS):
        pa.add(
            root,
            prop,
            f"parcel{k}",
            pos=(PARCEL_X, y, BELT_Z + 0.0005),
            free=True,
            variant_name=zone,
        )
    report["task"] = {
        "name": NAME,
        "instruction": INSTRUCTION,
        "key_points": KEY_POINTS,
        "parcels": [
            {"body": f"parcel{k}_part", "zone": z}
            for k, (_, z, _, _) in enumerate(PARCELS)
        ],
    }
    return root, report
