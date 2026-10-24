"""Dual-arm pick and place: the layout and the scene.

Each arm's box starts on the outer table and goes to a wide pad nearer the
middle, all where the head and body cameras see them.
"""

import xml.etree.ElementTree as ET

from sim.build_task import PART_RGBA, ROOT, TABLE_Z
from sim.build_task import build as build_base
from sim.gripper_linkage import contact_drop

# Far enough out for the head and body cameras to see the parts and pads.
SOURCES = {"left": [0.38, 0.34], "right": [0.38, -0.34]}
GOALS = {"left": [0.40, 0.15], "right": [0.40, -0.15]}
# The pads drop as the jaws close (gripper_linkage): the 45 mm parts are grasped
# and placed this much higher, so the pads still close at their centre height.
PAD_DROP = contact_drop(0.045)


def build():
    """Derived scene tree and report: build_task's robot, table and lights plus
    the two parts and their target pads."""
    root, report = build_base()
    # The model name this scene has had from the start.
    root.set("model", "franzi_dual_arm_kitting")
    world = root.find("worldbody")
    parts = [("left", PART_RGBA["blue"]), ("right", PART_RGBA["orange"])]
    for side, color in parts:
        x, y = SOURCES[side]
        body = ET.SubElement(
            world,
            "body",
            name=side + "_part",
            pos=f"{x} {y} {TABLE_Z + 0.0275 + 0.001}",
        )
        ET.SubElement(body, "freejoint", name=side + "_part_joint")
        ET.SubElement(
            body,
            "geom",
            name=side + "_part",
            type="box",
            size=".0225 .0225 .0275",
            mass="0.07",
            rgba=color,
            friction="1.2 .01 .0001",
            condim="4",
            solref=".004 1",
            solimp=".95 .99 .001",
        )
        # Wide, separated destination pads rather than a tight insertion task.
        x, y = GOALS[side]
        ET.SubElement(
            world,
            "geom",
            name=side + "_target",
            type="box",
            pos=f"{x} {y} {TABLE_Z + 0.003}",
            size=".065 .055 .003",
            rgba=color,
            solref=".004 1",
            solimp=".95 .99 .001",
        )
    return root, report


if __name__ == "__main__":
    root, _ = build()
    (ROOT / "outputs").mkdir(exist_ok=True)
    path = ROOT / "outputs/pick_place_task.xml"
    ET.ElementTree(root).write(path, encoding="unicode")
    print(path)
