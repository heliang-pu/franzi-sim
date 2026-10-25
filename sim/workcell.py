"""The industrial work cell around the robot in the benchmark scenes: factory
floor, background fences and racks, the workbench, and plain steel stands.

industrial_cell() takes the robot scene from build_task and swaps its grey floor
plane and plain table for the Blender-built factory floor (with its own
collision plane) and, unless bench=False, the workbench. The bench top is at
TABLE_Z like the table it replaces, and its front edge sits at x = 0.20 (the
table's was 0.18), clear of the hanging hands. The fences and racks stand
behind the bench as camera background and have no colliders.
"""

import xml.etree.ElementTree as ET

from sim import prop_assets as pa
from sim.build_task import TABLE_Z, fmt

BENCH_FRONT_X = 0.20
BENCH_DEPTH = 0.70  # the workbench prop, along x
STEEL_RGBA = "0.33 0.36 0.40 1"


def industrial_cell(root, bench=True, background=True):
    """Replace build_task's floor and table with the factory floor, the
    workbench (optional) and the background."""
    world = root.find("worldbody")
    for geom in list(world.findall("geom")):
        name = geom.get("name") or ""
        if name in ("floor", "table_top") or name.startswith("leg_"):
            world.remove(geom)
    pa.add(root, "factory_floor", "floor")
    if bench:
        pa.add(root, "workbench", "bench", pos=(BENCH_FRONT_X + BENCH_DEPTH / 2, 0, 0))
    if background:
        # Behind the overview camera (x = 1.85), so it never looks through them.
        for k, y in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
            pa.add(root, "safety_fence", f"fence{k}", pos=(2.3, y, 0))
        pa.add(root, "background_rack", "rack0", pos=(3.1, -1.1, 0))
        pa.add(root, "background_rack", "rack1", pos=(3.1, 1.0, 0))


def block(root, name, center, size, base_z, top_z):
    """A solid steel block (a riser or plinth) from base_z to top_z over size
    (x, y) around center (x, y)."""
    body = ET.SubElement(
        root.find("worldbody"),
        "body",
        name=f"{name}_part",
        pos=fmt([center[0], center[1], 0]),
    )
    half = (top_z - base_z) / 2
    ET.SubElement(
        body,
        "geom",
        name=f"{name}_block",
        type="box",
        pos=fmt([0, 0, base_z + half]),
        size=fmt([size[0] / 2, size[1] / 2, half]),
        rgba=STEEL_RGBA,
        friction="0.6 0.005 0.0001",
        solref=pa.SOLREF,
        solimp=pa.SOLIMP,
    )
    return body


def stand(root, name, center, size, top_z=TABLE_Z, legs=True):
    """A plain steel stand or riser: a 20 mm top plate whose top is at top_z,
    covering size (x, y) around center (x, y), on four 40 mm legs. Only the top
    plate collides. Bodies end in _part, so they are not robot bodies."""
    body = ET.SubElement(
        root.find("worldbody"),
        "body",
        name=f"{name}_part",
        pos=fmt([center[0], center[1], 0]),
    )
    sx, sy = size[0] / 2, size[1] / 2
    ET.SubElement(
        body,
        "geom",
        name=f"{name}_top",
        type="box",
        pos=fmt([0, 0, top_z - 0.01]),
        size=fmt([sx, sy, 0.01]),
        rgba=STEEL_RGBA,
        friction="0.6 0.005 0.0001",
        solref=pa.SOLREF,
        solimp=pa.SOLIMP,
    )
    if legs:
        h = (top_z - 0.02) / 2
        for dx in (-sx + 0.03, sx - 0.03):
            for dy in (-sy + 0.03, sy - 0.03):
                ET.SubElement(
                    body,
                    "geom",
                    type="box",
                    pos=fmt([dx, dy, h]),
                    size=fmt([0.02, 0.02, h]),
                    rgba=STEEL_RGBA,
                    contype="0",
                    conaffinity="0",
                    group="2",
                )
    return body
