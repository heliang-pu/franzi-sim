"""Tray push: the left arm pushes the parts tray along its rails until it rests
against the stop, without spilling the parts.

The rails run along y on the workbench at x = 0.37. The tray (200 mm deep,
between rails 206 mm apart) slides on the bench top; its collision geoms get
contact priority so that its own 0.25 friction governs the sliding. It starts
centred at y = 0.12, clear of the rails' cross bar at y = +0.30, and stops at
y = −0.125 against the block at the −y end. The left arm pushes its +y wall
from y 0.275 down to y 0.03, at x = 0.34 where it still reaches y ≈ 0.
"""

from sim import prop_assets as pa
from sim.build_task import TABLE_Z
from sim.build_task import build as robot_scene
from sim.workcell import industrial_cell

NAME = "tray_push"
INSTRUCTION = "Push the parts tray along the rails until it touches the end stop."
RAILS = (0.37, 0.0)
TRAY = (0.37, 0.12)
TRAY_STOP_Y = -0.125
TRAY_FLOOR_Z = TABLE_Z + 0.003
PUSH_X, PUSH_Z = 0.34, TABLE_Z + 0.03
KEY_POINTS = [
    {
        "name": "push_start",
        "side": "left",
        "xyz": [PUSH_X, TRAY[1] + 0.155, PUSH_Z],
        "grasp": "any",
    },
    {
        "name": "push_end",
        "side": "left",
        "xyz": [PUSH_X, TRAY_STOP_Y + 0.155, PUSH_Z],
        "grasp": "any",
    },
]


def build():
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    pa.add(root, "tray_rail_set", "rails", pos=(*RAILS, TABLE_Z))
    pa.add(root, "esd_tray", "tray", pos=(*TRAY, TABLE_Z + 0.0005), free=True)
    pa.set_geoms(root, "tray_body_col", priority="1")
    pa.add(
        root,
        "flange_part",
        "flange",
        pos=(0.345, 0.07, TRAY_FLOOR_Z + 0.001),
        free=True,
    )
    pa.add(
        root,
        "abs_housing",
        "housing",
        pos=(0.39, 0.18, TRAY_FLOOR_Z + 0.001),
        free=True,
    )
    report["task"] = {
        "name": NAME,
        "instruction": INSTRUCTION,
        "key_points": KEY_POINTS,
    }
    return root, report
