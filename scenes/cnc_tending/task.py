"""CNC tending: open the machine door, take the finished part out of the vise
into the tray, load a raw blank into the vise, close the door and press CYCLE
START.

The machine front faces the robot at x = 0.36, shifted to y = +0.25 so that its
door handle (y 0.03 when closed) and vise (y 0.25) are the left arm's, while
its control panel (CYCLE START at y −0.155) and the parts tray on the stand to
the right are the right arm's. Moving parts between vise and tray therefore
takes a hand-over. The finished part rests on the vise's parallels with the
jaw open; the tray holds three blanks in its near row, its far row is free.
"""

import numpy as np

from sim import prop_assets as pa
from sim.build_task import TABLE_Z
from sim.build_task import build as robot_scene
from sim.workcell import industrial_cell, stand

NAME = "cnc_tending"
INSTRUCTION = "Swap the finished part in the vise for a raw blank, close the door and press CYCLE START."
MACHINE = np.array([0.36, 0.25, 0.0])
WORKPIECE = MACHINE + [0.10, 0.0, 0.86]
STAND_XY, TRAY_XY = (0.37, -0.345), (0.37, -0.345)
TRAY_TOP = TABLE_Z + 0.03
POCKET_FLOOR = TRAY_TOP - 0.02
NEAR_ROW = [(TRAY_XY[0] - 0.035, TRAY_XY[1] + dy) for dy in (-0.075, 0.0, 0.075)]
FAR_ROW = [(TRAY_XY[0] + 0.035, TRAY_XY[1] + dy) for dy in (-0.075, 0.0, 0.075)]
HANDLE = MACHINE + [-0.06, -0.22, 0.99]
CYCLE_START = MACHINE + [-0.05, -0.405, 1.0]
KEY_POINTS = [
    {
        "name": "door_handle_closed",
        "side": "left",
        "xyz": [round(float(v), 4) for v in HANDLE],
        "grasp": "any",
    },
    {
        "name": "door_handle_open_0.30",
        "side": "left",
        "xyz": [round(float(v), 4) for v in HANDLE + [0, 0.30, 0]],
        "grasp": "any",
    },
    {
        "name": "vise_part",
        "side": "left",
        "xyz": [round(float(v), 4) for v in WORKPIECE + [0, 0, 0.03]],
        "grasp": "forward",
    },
    {
        "name": "tray_far_row",
        "side": "right",
        "xyz": [FAR_ROW[2][0], FAR_ROW[2][1], TRAY_TOP + 0.03],
        "grasp": "forward",
    },
    {
        "name": "tray_blank",
        "side": "right",
        "xyz": [NEAR_ROW[2][0], NEAR_ROW[2][1], TRAY_TOP + 0.03],
        "grasp": "forward",
    },
    {
        "name": "cycle_start",
        "side": "right",
        "xyz": [round(float(v), 4) for v in CYCLE_START],
        "grasp": "any",
    },
]


def build():
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root, bench=False)
    pa.add(root, "cnc_front", "cnc", pos=tuple(MACHINE))
    stand(root, "tray_stand", STAND_XY, (0.22, 0.27), top_z=TABLE_Z)
    pa.add(root, "part_tray_6", "tray", pos=(*TRAY_XY, TABLE_Z), variant_name="IN")
    for k, xy in enumerate(NEAR_ROW):
        pa.add(
            root,
            "round_blank",
            f"blank{k}",
            pos=(*xy, POCKET_FLOOR + 0.0005),
            free=True,
        )
    pa.add(
        root,
        "machined_part",
        "finished",
        pos=tuple(WORKPIECE + [0, 0, 0.0005]),
        free=True,
    )
    report["task"] = {
        "name": NAME,
        "instruction": INSTRUCTION,
        "key_points": KEY_POINTS,
    }
    return root, report
