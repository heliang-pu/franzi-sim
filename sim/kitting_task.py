"""Dual-arm sort-and-kit long task: episode sampling, key poses and the scene.

Each arm moves the three parts on its supply mat into the three cells of its
kit tray, every part into the cell of its colour. Each kind of part has its own
colour, and a kit takes one of each kind. A seed fixes the layout: the order,
jitter and yaw of the parts, the order of the cells and the wording of the
instruction. Every draw is checked with the shared IK (each key
pose, and each 1 cm waypoint of the straight moves) and redrawn until it plans,
so every seed gives a solvable episode.
"""

import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np

from robot_kinematics import Kinematics
from sim.build_task import PART_RGBA, ROOT, TABLE_Z, fmt
from sim.build_task import build as build_base

SIDES = ("left", "right")
# Two rows of three slots per side (y on the left; the right mirrors). At the
# 45-degree grasp the gripper housing spans about 85 mm to either side of the
# grasp point at grasp height, and 65 mm behind it, so slots are 120 mm apart
# and the rows 100 mm. The kit row is where both policy cameras see it; the
# supply row behind it is still inside the reach of the grasp.
SLOT_Y = (0.10, 0.22, 0.34)
KIT_X, SUPPLY_X = 0.38, 0.48
ROW_Y = (SLOT_Y[0] + SLOT_Y[-1]) / 2
ROW_HALF = (0.038, 0.18)  # tray and mat half depth (x) and half length (y)
KIT_TOP = TABLE_Z + 0.006
SUPPLY_TOP = TABLE_Z + 0.004
HEIGHT = 0.05
# Three kinds of part, each its own colour, all 40 mm across and 50 mm tall.
# Every supply mat holds one of each kind and every tray one cell per kind.
# symmetry is the yaw period of a kind's grasp faces (0 for the cylinder).
TYPES = {
    "cube": {
        "color": "blue",
        "noun": "cube",
        "geom": {"type": "box", "size": fmt([0.02, 0.02, HEIGHT / 2])},
        "mass": 0.06,
        "width": 0.04,
        "symmetry": np.pi / 2,
    },
    "cylinder": {
        "color": "orange",
        "noun": "cylinder",
        "geom": {"type": "cylinder", "size": fmt([0.02, HEIGHT / 2])},
        "mass": 0.05,
        "width": 0.04,
        "symmetry": 0.0,
    },
    "hex": {
        "color": "green",
        "noun": "hex prism",
        "geom": {"type": "mesh", "mesh": "kit_hex"},
        "mass": 0.055,
        "width": 0.04,  # across flats
        "symmetry": np.pi / 3,
    },
}
RGBA = {**PART_RGBA, "green": "0.20 0.66 0.32 1"}
JITTER = (0.006, 0.004)  # supply slot x, y (m)
YAW = np.radians(10)  # flat-sided parts; the approach turns with them
FRAME = (0.04, 0.006)  # cell frame half size and line width
CELL_TOL = 0.015  # a part counts as in a cell within this, in x and in y
# Grasp: approach 45 degrees forward and down, fingers closing horizontally.
GRASP_Z = 0.035  # grasp point above a part's base; fingertips stay 18 mm clear
PRE = 0.08  # pre-grasp and retreat distance along the approach axis
LIFT = 0.10  # a carried part clears the others by about 5 cm
JAW_SPAN = 0.095  # jaw gap at closure 0; closure g moves each finger 0.0475 g
CLEAR = 0.012  # finger clearance per side while open
SQUEEZE = 0.0021  # closure past contact, as in the two-part expert (~1.6 N)
# The two-part expert's ready point, now with the kitting hand rotation: from the
# hanging start, a ready point over the tray (0.30, +-0.24, 1.00) clips the
# table edge on the way up.
READY = {"left": [0.15, 0.37, 0.99], "right": [0.15, -0.37, 0.99]}
# The expert keeps 10 mrad inside the joint limits and the sampler 20 mrad, so
# measured part poses that differ a little from the sampled ones still plan.
# The inner slots push the shoulder roll onto its limit; planned on the limit,
# the servo overshot the 2 mrad audit (seed 16).
LIMIT_MARGIN = 0.01
SAMPLE_MARGIN = 0.02
SPEED = 0.6  # peak joint command rate (rad/s); faster moves overshot the limit
SEGMENTS = (
    ("pregrasp", "grasp"),
    ("grasp", "lift"),
    ("lift", "above"),
    ("above", "place"),
    ("place", "retreat"),
)
INSTRUCTIONS = (
    "Kit the parts: put each part in the tray cell of its own color.",
    "Put each side's cube, cylinder and hex prism in the cells of their colors.",
    "Fill both kit trays, matching every part to the cell of the same color.",
    "Use both arms to place each part in the tray cell that matches its color.",
)
TRAY_RGBA = "0.16 0.17 0.19 1"
MAT_RGBA = "0.80 0.79 0.75 1"
SURFACE = {"friction": "0.7 0.005 0.0001", "solref": ".004 1", "solimp": ".95 .99 .001"}
PART_CONTACT = {
    "friction": "1.2 .01 .0001",
    "condim": "4",
    "solref": ".004 1",
    "solimp": ".95 .99 .001",
}


def kinematics(margin=LIMIT_MARGIN):
    """The shared IK with a kitting limit margin."""
    return Kinematics(margin=margin)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def grasp_rotation(side, psi=0.0):
    """Hand frame for the kitting grasp: approach 45 degrees forward and down at
    azimuth psi (rad). The fingers slide along the hand's x axis, so the hand
    rolls 90 degrees to close them horizontally, in the direction that keeps the
    wrist D405 on top of the hand. The opposite roll (left +90, right -90
    degrees) reaches as well but hung the camera underneath, toward the robot."""
    roll = -np.pi / 2 if side == "left" else np.pi / 2
    return rot_z(psi) @ rot_y(-np.pi / 4) @ rot_z(roll)


def grip_open(width):
    """Open closure for a part of width (m). The kitting expert never opens
    fully: open fingers sit on their joint limits, and once the hand tilts,
    gravity along the finger slides (fingers get no feedforward) pushes them
    past the limit."""
    return 1 - (width + 2 * CLEAR) / JAW_SPAN


def grip_closed(width):
    return 1 - (width - SQUEEZE) / JAW_SPAN


def describe(part):
    return f"{part['color']} {TYPES[part['shape']]['noun']}"


def hex_vertices():
    """The hex prism, 40 mm across flats: at yaw 0 two flats face +-y, where the
    kitting grasp closes."""
    r = TYPES["hex"]["width"] / np.sqrt(3)
    a = np.arange(6) * np.pi / 3
    ring = np.c_[r * np.cos(a), r * np.sin(a)]
    return np.r_[
        np.c_[ring, np.full(6, -HEIGHT / 2)], np.c_[ring, np.full(6, HEIGHT / 2)]
    ]


def key_poses(part, cell, xy=None, psi=None):
    """Grasp-point targets (world, m) for moving part into cell, with the hand
    rotation. xy and psi default to the sampled pose; the expert passes the
    measured ones."""
    xy = part["xy"] if xy is None else xy
    psi = part["psi"] if psi is None else psi
    rotation = grasp_rotation(part["side"], psi)
    back = PRE * (rotation @ [0, 0, 1])  # opposite the approach direction
    grasp = np.array([*xy, SUPPLY_TOP + GRASP_Z])
    place = np.array([*cell["xy"], KIT_TOP + GRASP_Z])
    up = np.array([0, 0, LIFT])
    return {
        "rotation": rotation,
        "pregrasp": grasp + back,
        "grasp": grasp,
        "lift": grasp + up,
        "above": place + up,
        "place": place,
        "retreat": place + back,
    }


def plan_part(kin, part, cell, start):
    """Arm pose after moving part into cell from arm pose start, every move a
    straight line with an IK waypoint each 1 cm: to the pre-grasp, then each
    segment. Raises RuntimeError where a waypoint has no solution inside the
    joint limits."""
    poses = key_poses(part, cell)
    side, rotation = part["side"], poses["rotation"]
    q = start
    for b in ["pregrasp"] + [b for _, b in SEGMENTS]:
        q = kin.line(side, q, poses[b], rotation=rotation)[-1]
    return q


def draw(rng):
    parts, cells = [], []
    for side in SIDES:
        sign = 1 if side == "left" else -1
        kinds, cell_kinds = rng.permutation(list(TYPES)), rng.permutation(list(TYPES))
        for k, y in enumerate(SLOT_Y):
            shape = str(kinds[k])
            dx, dy = rng.uniform(-1, 1, 2) * JITTER
            # Flat-sided parts are grasped across two faces, so the approach
            # turns with them; a cylinder is grasped head-on at any angle.
            yaw = float(rng.uniform(-YAW, YAW)) if TYPES[shape]["symmetry"] else 0.0
            parts.append(
                {
                    "name": f"{side[0]}{k}",
                    "side": side,
                    "color": TYPES[shape]["color"],
                    "shape": shape,
                    "xy": [SUPPLY_X + float(dx), sign * (y + float(dy))],
                    "yaw": yaw,
                    "psi": yaw,
                }
            )
            cells.append(
                {
                    "name": f"{side[0]}c{k}",
                    "side": side,
                    "color": TYPES[str(cell_kinds[k])]["color"],
                    "shape": str(cell_kinds[k]),
                    "xy": [KIT_X, sign * y],
                }
            )
    return parts, cells, INSTRUCTIONS[int(rng.integers(len(INSTRUCTIONS)))]


def assign(parts, cells):
    """Each arm takes its parts inner to outer, and each part goes to the free
    cell of its colour nearest in y. Returns {part: cell} and the rounds, one
    part per arm each."""
    assignment, order = {}, {}
    for side in SIDES:
        mine = sorted(
            (p for p in parts if p["side"] == side), key=lambda p: abs(p["xy"][1])
        )
        free = [c for c in cells if c["side"] == side]
        for p in mine:
            cell = min(
                (c for c in free if c["color"] == p["color"]),
                key=lambda c: abs(c["xy"][1] - p["xy"][1]),
            )
            free.remove(cell)
            assignment[p["name"]] = cell["name"]
        order[side] = [p["name"] for p in mine]
    rounds = [{side: order[side][k] for side in SIDES} for k in range(len(SLOT_Y))]
    return assignment, rounds


def sample_episode(seed, kin=None, attempts=100):
    """Episode spec for seed (plain JSON types): the first draw whose every part
    plans from the ready pose, round by round as the expert moves, and back."""
    kin = kin or kinematics(SAMPLE_MARGIN)
    rng = np.random.default_rng(seed)
    start = {
        s: kin.ik(s, READY[s], np.zeros(7), rotation=grasp_rotation(s)) for s in SIDES
    }
    for attempt in range(1, attempts + 1):
        parts, cells, instruction = draw(rng)
        assignment, rounds = assign(parts, cells)
        by_name = {x["name"]: x for x in parts + cells}
        try:
            for side in SIDES:
                q = start[side]
                for pair in rounds:
                    part = by_name[pair[side]]
                    q = plan_part(kin, part, by_name[assignment[part["name"]]], q)
                kin.line(side, q, READY[side], rotation=grasp_rotation(side))
        except RuntimeError:
            continue
        return {
            "task": "dual_arm_sort_and_kit",
            "version": 2,
            "seed": int(seed),
            "draws": attempt,
            "instruction": instruction,
            "parts": parts,
            "cells": cells,
            "assignment": assignment,
            "rounds": rounds,
            "ready": READY,
        }
    raise RuntimeError(f"seed {seed}: no layout plans in {attempts} draws")


def build(spec):
    """Derived scene for spec: build_task's robot, table and lights without its
    two parts, plus the kit trays with their coloured cells, the supply mats and
    the parts at their sampled poses."""
    root, report = build_base(task_objects=False)
    root.set("model", "franzi_dual_arm_sort_and_kit")
    ET.SubElement(
        root.find("asset"), "mesh", name="kit_hex", vertex=fmt(hex_vertices())
    )
    world = root.find("worldbody")
    for side in SIDES:
        y = ROW_Y if side == "left" else -ROW_Y
        for name, x, top, rgba in (
            ("kit_tray", KIT_X, KIT_TOP, TRAY_RGBA),
            ("supply_mat", SUPPLY_X, SUPPLY_TOP, MAT_RGBA),
        ):
            half = (top - TABLE_Z) / 2
            ET.SubElement(
                world,
                "geom",
                name=f"{side}_{name}",
                type="box",
                pos=fmt([x, y, TABLE_Z + half]),
                size=fmt([*ROW_HALF, half]),
                rgba=rgba,
                **SURFACE,
            )
    outer, line = FRAME
    mid = outer - line / 2
    bars = (
        ("near", -mid, 0, line / 2, outer),
        ("far", mid, 0, line / 2, outer),
        ("inner", 0, -mid, outer - line, line / 2),
        ("outer", 0, mid, outer - line, line / 2),
    )
    for cell in spec["cells"]:
        # A coloured square frame, drawn only: no contact, no mass.
        x, y = cell["xy"]
        for name, dx, dy, hx, hy in bars:
            ET.SubElement(
                world,
                "geom",
                name=f"{cell['name']}_frame_{name}",
                type="box",
                pos=fmt([x + dx, y + dy, KIT_TOP + 0.0005]),
                size=fmt([hx, hy, 0.0005]),
                rgba=RGBA[cell["color"]],
                contype="0",
                conaffinity="0",
            )
    for part in spec["parts"]:
        x, y = part["xy"]
        yaw = part["yaw"]
        body = ET.SubElement(
            world,
            "body",
            name=part["name"] + "_part",
            pos=fmt([x, y, SUPPLY_TOP + HEIGHT / 2 + 0.001]),
            quat=fmt([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
        )
        ET.SubElement(body, "freejoint", name=part["name"] + "_part_joint")
        kind = TYPES[part["shape"]]
        ET.SubElement(
            body,
            "geom",
            name=part["name"] + "_part",
            mass=str(kind["mass"]),
            rgba=RGBA[part["color"]],
            **kind["geom"],
            **PART_CONTACT,
        )
    report["task"] = {k: spec[k] for k in ("task", "version", "seed", "instruction")}
    return root, report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    spec = sample_episode(args.seed)
    root, report = build(spec)
    (ROOT / "outputs").mkdir(exist_ok=True)
    ET.ElementTree(root).write(ROOT / "outputs/kitting_task.xml", encoding="unicode")
    (ROOT / "outputs/kitting_episode.json").write_text(json.dumps(spec, indent=2))
    print(json.dumps(spec, indent=2))
