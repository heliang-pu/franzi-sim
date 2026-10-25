"""Shaped kitting: put each part into the foam pocket of its shape, in the
orientation the pocket allows (the bushing flange up, the gear hub up, the
hex aligned with its pocket's flats).

The foam kit tray lies on the workbench centred at x = 0.40, y = 0, with its
pockets at x 0.345 and 0.455, y ±0.09. The left arm owns the bushing and gear
pockets (y +0.09) and their parts, which lie loose on the left of the tray; the
right arm owns the hex blank and L-bracket on the right. Grasps reach this far
forward at part height only tilted 45 degrees (sim.tabletop.grasp_rotation).
"""

from sim import prop_assets as pa
from sim.build_task import TABLE_Z
from sim.build_task import build as robot_scene
from sim.workcell import industrial_cell

NAME = "shaped_kitting"
INSTRUCTION = "Put each part into the foam pocket that matches its shape."
TRAY = (0.40, 0.0)
TRAY_TOP = TABLE_Z + 0.05
# part: (prop, xy, height, arm)
PARTS = {
    "bushing": ("flanged_bushing", (0.35, 0.30), 0.030, "left"),
    "gear": ("spur_gear", (0.42, 0.33), 0.018, "left"),
    "hex": ("hex_blank", (0.35, -0.30), 0.030, "right"),
    "bracket": ("l_bracket", (0.42, -0.33), 0.040, "right"),
}
POCKETS = {
    p["part"]: (
        TRAY[0] + p["center_yx_mm"][1] / 1000,
        TRAY[1] + p["center_yx_mm"][0] / 1000,
    )
    for p in pa.spec("foam_kit_tray")["pockets"]
}
KEY_POINTS = [
    {
        "name": f"grasp_{name}",
        "side": arm,
        "xyz": [xy[0], xy[1], TABLE_Z + h / 2],
        "grasp": "forward",
    }
    for name, (_, xy, h, arm) in PARTS.items()
] + [
    {
        "name": f"pocket_{prop}",
        "side": arm,
        "xyz": [POCKETS[prop][0], POCKETS[prop][1], TRAY_TOP + 0.02],
        "grasp": "forward",
    }
    for _, (prop, _, _, arm) in PARTS.items()
]


def build():
    root, report = robot_scene()
    root.set("model", NAME)
    industrial_cell(root)
    # Light-grey foam: the cameras can't make out pockets in the black one.
    pa.add(
        root,
        "foam_kit_tray",
        "kit_tray",
        pos=(*TRAY, TABLE_Z),
        variant_name="light_grey",
    )
    for name, (prop, xy, _, _) in PARTS.items():
        pa.add(root, prop, name, pos=(*xy, TABLE_Z + 0.0005), free=True)
    report["task"] = {
        "name": NAME,
        "instruction": INSTRUCTION,
        "key_points": KEY_POINTS,
        "pockets_world_xy": POCKETS,
    }
    return root, report
