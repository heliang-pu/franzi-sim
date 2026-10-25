"""Lidded tote: lift the tote's lid, take the hex blank out, set it on the
station plate, and close the lid.

The tote stands on the workbench at x = 0.445, y = 0.12, hinged at the back
(x = 0.547, z = 0.952). Its hinge friction holds the lid wherever it is left,
so one arm can open it, pick and close it in turn. The hex blank lies near the
tote's front wall (x = 0.38, the wall's inner face is at 0.356 at the floor),
within a top-down grasp's reach; the station plate lies to the left at
y = 0.40. The left arm does it all.
"""

import numpy as np

from sim import prop_assets as pa
from sim.build_task import TABLE_Z
from sim.build_task import build as robot_scene
from sim.workcell import industrial_cell

NAME = "lidded_tote"
INSTRUCTION = "Open the tote, take the hex blank out and put it on the station plate, then close the lid."
TOTE = np.array([0.445, 0.12, TABLE_Z])
PLATE = (0.36, 0.40)
FLOOR_Z = TABLE_Z + 0.004
HINGE = TOTE + [0.102, 0.0, 0.152]
PULL_TAB = TOTE + [0.334637 - 0.445, 0.0, 0.961937 - TABLE_Z]


def tab_at(angle):
    """The pull tab's position with the lid opened by angle (rad)."""
    rel = PULL_TAB - HINGE
    c, s = np.cos(angle), np.sin(angle)
    return HINGE + [rel[0] * c + rel[2] * s, 0.0, -rel[0] * s + rel[2] * c]


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
        "xyz": [0.38, 0.13, FLOOR_Z + 0.015],
        "grasp": "top",
    },
    {
        "name": "place_on_plate",
        "side": "left",
        "xyz": [PLATE[0], PLATE[1], TABLE_Z + 0.04],
        "grasp": "top",
    },
]


def build():
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    pa.add(root, "lidded_tote", "tote", pos=tuple(TOTE))
    pa.add(root, "station_plate", "plate", pos=(*PLATE, TABLE_Z))
    pa.add(root, "hex_blank", "hex", pos=(0.38, 0.13, FLOOR_Z + 0.0005), free=True)
    pa.add(root, "flange_part", "flange", pos=(0.40, 0.21, FLOOR_Z + 0.0005), free=True)
    report["task"] = {
        "name": NAME,
        "instruction": INSTRUCTION,
        "key_points": KEY_POINTS,
    }
    return root, report
