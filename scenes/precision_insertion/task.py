"""Precision insertion: the left arm presses the 6204 bearing into its housing
(0.20 mm diametral clearance), and the right arm moves the dowel pin from the
fixture's first hole into its second (0.10 mm clearance).

Both fixtures are fixed to the workbench at x = 0.37: the bearing housing at
y = +0.18 and the pin fixture at y = −0.18, whose holes are at y −0.21 and
−0.15. The bearing lies flat on the bench to the left; the pin stands in hole
one, resting on the bench through the plate. Insertions go top-down, which
reaches x 0.37 at these heights.
"""

from sim import prop_assets as pa
from sim.build_task import TABLE_Z
from sim.build_task import build as robot_scene
from sim.workcell import industrial_cell

NAME = "precision_insertion"
INSTRUCTION = (
    "Press the bearing into its housing and move the dowel pin into the empty hole."
)
HOUSING = (0.37, 0.18)
BEARING = (0.36, 0.36)
FIXTURE = (0.37, -0.18)
HOLES = [(FIXTURE[0], FIXTURE[1] - 0.03), (FIXTURE[0], FIXTURE[1] + 0.03)]
KEY_POINTS = [
    {
        "name": "grasp_bearing",
        "side": "left",
        "xyz": [*BEARING, TABLE_Z + 0.007],
        "grasp": "top",
    },
    {
        "name": "above_bore",
        "side": "left",
        "xyz": [*HOUSING, TABLE_Z + 0.07],
        "grasp": "top",
    },
    {
        "name": "seat_bearing",
        "side": "left",
        "xyz": [*HOUSING, TABLE_Z + 0.033],
        "grasp": "top",
    },
    {
        "name": "grasp_pin",
        "side": "right",
        "xyz": [*HOLES[0], TABLE_Z + 0.035],
        "grasp": "top",
    },
    {
        "name": "above_hole2",
        "side": "right",
        "xyz": [*HOLES[1], TABLE_Z + 0.07],
        "grasp": "top",
    },
]


def build():
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    pa.add(root, "bearing_housing_c020", "housing", pos=(*HOUSING, TABLE_Z))
    pa.add(root, "bearing_6204", "bearing", pos=(*BEARING, TABLE_Z + 0.0005), free=True)
    pa.add(root, "pin_fixture", "fixture", pos=(*FIXTURE, TABLE_Z), variant_name="c010")
    pa.add(root, "dowel_pin", "pin", pos=(*HOLES[0], TABLE_Z + 0.0005), free=True)
    report["task"] = {
        "name": NAME,
        "instruction": INSTRUCTION,
        "key_points": KEY_POINTS,
    }
    return root, report
