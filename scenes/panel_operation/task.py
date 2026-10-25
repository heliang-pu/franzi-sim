"""Panel operation: turn the mode selector to AUTO, then press the green START
button, keeping clear of the emergency stop.

The control console (layout A) stands on the workbench with its sloped panel
facing the robot; its front edge is at x ≈ 0.30, so the devices sit at x
0.36–0.40 and z 0.89–0.92. The selector (y −0.09) and START (y −0.02) are the
right arm's; STOP (y +0.04) and the emergency stop (y +0.11) lie on the left.
Key points are 10 mm out from each device's mount along the panel normal.
"""

import numpy as np

from sim import prop_assets as pa
from sim.build_task import TABLE_Z
from sim.build_task import build as robot_scene
from sim.workcell import industrial_cell

NAME = "panel_operation"
INSTRUCTION = "Turn the selector to AUTO, then press the green START button. Keep clear of the emergency stop."
CONSOLE = np.array([0.41, 0.0, TABLE_Z])
NORMAL = np.array(pa.spec("control_console")["panel_frame"]["outward_normal"])


def device_point(part, lift=0.010):
    return [
        round(float(v), 4)
        for v in CONSOLE + pa.joint_anchor("control_console", part) + lift * NORMAL
    ]


KEY_POINTS = [
    {
        "name": "selector",
        "side": "right",
        "xyz": device_point("mode", 0.012),
        "grasp": "any",
    },
    {
        "name": "start_button",
        "side": "right",
        "xyz": device_point("start"),
        "grasp": "any",
    },
    {
        "name": "stop_button",
        "side": "left",
        "xyz": device_point("stop"),
        "grasp": "any",
    },
    {
        "name": "emergency_stop",
        "side": "left",
        "xyz": device_point("emergency", 0.02),
        "grasp": "any",
    },
]


def build():
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    pa.add(root, "control_console", "console", pos=tuple(CONSOLE), variant_name="A")
    report["task"] = {
        "name": NAME,
        "instruction": INSTRUCTION,
        "key_points": KEY_POINTS,
    }
    return root, report
