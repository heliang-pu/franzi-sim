"""Tote transfer: the left arm moves a part from the blue source tote into the
grey destination tote.

Both totes stand on the workbench at x = 0.37 with their long side along y.
The source tote spans y 0.12–0.42; the destination spans y −0.21–0.09, so the
left arm reaches both (top-down grasps reach x ≤ 0.38 at part height and y down
to about 0.0 at drop height). The aluminium flange and the ABS housing lie
loose on the source tote's floor.
"""

from sim import prop_assets as pa
from sim.build_task import TABLE_Z
from sim.build_task import build as robot_scene
from sim.workcell import industrial_cell

NAME = "tote_transfer"
INSTRUCTION = "Move the aluminium flange from the blue tote into the grey tote."
SOURCE = (0.37, 0.27)
DEST = (0.37, -0.06)
FLOOR_Z = TABLE_Z + 0.004  # the tote's 4 mm floor
PARTS = {
    "flange": ("flange_part", (0.36, 0.22)),
    "housing": ("abs_housing", (0.38, 0.33)),
}
KEY_POINTS = [
    {
        "name": "grasp_flange",
        "side": "left",
        "xyz": [0.36, 0.22, FLOOR_Z + 0.012],
        "grasp": "top",
    },
    {
        "name": "grasp_housing",
        "side": "left",
        "xyz": [0.38, 0.33, FLOOR_Z + 0.018],
        "grasp": "top",
    },
    {
        "name": "drop_into_dest",
        "side": "left",
        "xyz": [0.37, 0.03, TABLE_Z + 0.13],
        "grasp": "top",
    },
]


def build():
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    pa.add(
        root, "small_tote", "source_tote", pos=(*SOURCE, TABLE_Z), variant_name="blue"
    )
    pa.add(root, "small_tote", "dest_tote", pos=(*DEST, TABLE_Z), variant_name="grey")
    for name, (prop, xy) in PARTS.items():
        pa.add(root, prop, name, pos=(*xy, FLOOR_Z + 0.0005), free=True)
    report["task"] = {
        "name": NAME,
        "instruction": INSTRUCTION,
        "key_points": KEY_POINTS,
    }
    return root, report
