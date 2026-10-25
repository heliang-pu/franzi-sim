"""Carton carry: both arms squeeze a carton from its left and right faces, lift
it off the workbench and set it on the raised outbound roller conveyor behind.

The 250 × 200 × 150 mm carton (1 kg) starts on the bench at x = 0.33, y = 0,
its 250 mm side along y. With the waist fixed, the hands cannot cross the
midline, so a two-handed carry keeps the carton near y = 0 and moves it along x
and up. The outbound conveyor stands on a 100 mm riser at x 0.44–0.94 (rollers
at 0.95 m); the carton goes down with its centre at x = 0.50, overhanging the
conveyor's front by 40 mm.
"""

from sim import prop_assets as pa
from sim.build_task import TABLE_Z
from sim.build_task import build as robot_scene
from sim.workcell import block, industrial_cell

NAME = "carton_carry"
INSTRUCTION = (
    "Lift the carton with both hands and set it on the raised conveyor behind it."
)
CARTON = (0.33, 0.0)  # its back face at x 0.43, clear of the riser at 0.44
HALF_W, HEIGHT = 0.125, 0.15
RISER = {"center": (0.69, 0.0), "size": (0.50, 0.46), "top": TABLE_Z + 0.10}
ROLLER_TOP = RISER["top"] + 0.05
TARGET_X = 0.50
KEY_POINTS = [
    {
        "name": f"squeeze_{side}",
        "side": side,
        "xyz": [CARTON[0], s * (HALF_W + 0.005), TABLE_Z + HEIGHT / 2],
        "grasp": "top",
    }
    for side, s in (("left", 1), ("right", -1))
] + [
    {
        "name": f"place_{side}",
        "side": side,
        "xyz": [TARGET_X, s * (HALF_W + 0.005), ROLLER_TOP + HEIGHT / 2],
        "grasp": "top",
    }
    for side, s in (("left", 1), ("right", -1))
]


def build():
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    block(root, "riser", RISER["center"], RISER["size"], TABLE_Z, RISER["top"])
    pa.add(
        root,
        "roller_conveyor",
        "outbound",
        pos=(*RISER["center"], RISER["top"]),
        variant_name="stop_removed",
    )
    pa.add(
        root,
        "carton_250x200x150",
        "carton",
        pos=(*CARTON, TABLE_Z + 0.0005),
        free=True,
        variant_name="mass_1.0kg",
    )
    report["task"] = {
        "name": NAME,
        "instruction": INSTRUCTION,
        "key_points": KEY_POINTS,
    }
    return root, report
