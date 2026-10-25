"""Palletizing: stack the cartons from the two side stands onto the pallet in
the pattern on the sign (layer 1 of plan_01: four 120 × 80 mm cartons in a
2 × 2 block, long side along y).

A scissor lift holds the pallet's top deck at 0.80 m in front of the robot
(pallet centre x = 0.40). Each side stand carries two cartons for the arm on
its side, and each arm fills the pallet's half on its side (y = ±0.06). The
cartons are grasped across their 80 mm width; at this height the grasps reach
x 0.44 only tilted 45 degrees.
"""

from sim import prop_assets as pa
from sim.build_task import TABLE_Z
from sim.build_task import build as robot_scene
from sim.workcell import industrial_cell, stand

NAME = "palletizing"
INSTRUCTION = "Stack the four cartons on the pallet as the sign shows."
LIFT = (0.46, 0.0)  # its base starts at x 0.25, clear of the robot's front wheels
PALLET = (0.40, 0.0, 0.72)
DECK_Z = 0.80
BOX = "pallet_carton_120x80x80"
BOX_H = 0.08
STANDS = {"left": (0.37, 0.37), "right": (0.37, -0.37)}
SUPPLY = {"left": [(0.31, 0.37), (0.43, 0.37)], "right": [(0.31, -0.37), (0.43, -0.37)]}
PLAN = pa.spec("pattern_sign")["stacking_plans"][0]
TARGETS = [
    (PALLET[0] + b["center_xy_mm"][0] / 1000, PALLET[1] + b["center_xy_mm"][1] / 1000)
    for b in PLAN["layers"][0]
]
KEY_POINTS = [
    {
        "name": f"pick_{side}_{k}",
        "side": side,
        "xyz": [xy[0], xy[1], TABLE_Z + BOX_H / 2],
        "grasp": "forward",
    }
    for side, xys in SUPPLY.items()
    for k, xy in enumerate(xys)
] + [
    {
        "name": f"place_{k}",
        "side": "left" if xy[1] > 0 else "right",
        "xyz": [xy[0], xy[1], DECK_Z + BOX_H / 2],
        "grasp": "forward",
    }
    for k, xy in enumerate(TARGETS)
]


def build():
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root, bench=False)
    pa.add(root, "scissor_lift_table", "lift", pos=(*LIFT, 0.0))
    pa.add(root, "mini_pallet", "pallet", pos=PALLET, variant_name="pine")
    for side, xy in STANDS.items():
        stand(root, f"supply_{side}", xy, (0.24, 0.20), top_z=TABLE_Z)
    for side, xys in SUPPLY.items():
        for k, xy in enumerate(xys):
            pa.add(
                root, BOX, f"carton_{side}{k}", pos=(*xy, TABLE_Z + 0.0005), free=True
            )
    stand(root, "sign_stand", (0.72, 0.40), (0.14, 0.26), top_z=TABLE_Z)
    pa.add(
        root,
        "pattern_sign",
        "sign",
        pos=(0.72, 0.40, TABLE_Z),
        quat=(0.7071068, 0, 0, -0.7071068),
        variant_name=PLAN["name"],
    )
    report["task"] = {
        "name": NAME,
        "instruction": INSTRUCTION,
        "key_points": KEY_POINTS,
        "targets_xy": TARGETS,
    }
    return root, report
