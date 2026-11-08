"""The wrist D405 adapter the simulation mounts (hardware/wrist_d405_mount here).

wheel_robot_26.8.16_3's gripper holds its D405 on a bent strip beside the
housing, near the flange, looking straight down the hand: the housing hides
the jaws and pads. The adapter sits on the strip's top pad, in the D405's
place and on its two screws, and holds the D405 38 mm further down the hand,
tilted 20 degrees toward the jaws. That is where the tilted block of
2026-10-24 (the user's reference model) held it on wheel_robot_4.0's gripper,
the same hardware: the lens 65.4 mm beside the jaw plane and 88 mm behind the
grasp point, so the image shows the whole gripper (user request 2026-10-25).
Its parameters live in robots/franzi/hardware/wrist_d405_mount/params.json. This module
turns them into the D405's frame and the adapter's shape and placement. Pure
NumPy, so the Blender build script, the mount check and the scene build share
it.

Frames: the part frame has its origin on the pad's top face between its two
holes and the axes of the gripper link (x outward from the hand, y along the
jaws, z down the hand toward the fingers); millimetres unless a name ends in
_m. Both gripper links are the same part with the strip in the same place, so
one adapter fits both hands, unturned.
"""

import json
from pathlib import Path

import numpy as np

MOUNT = Path(__file__).resolve().parent / "hardware/wrist_d405_mount"
P = json.loads((MOUNT / "params.json").read_text())
STL = MOUNT / "wrist_d405_mount.stl"
# The part frame's origin in each gripper link frame (the pad's top centre).
PAD = np.array(P["interface_from_cad"]["pad_top_centre_in_gripper_link_mm"])
DEPTH = P["d405"]["housing_mm"][2]  # back face to glass


def sign(side):
    return 1.0 if side == "left" else -1.0


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def camera():
    """D405 glass centre (mm), optical axis, and the axes of its face, along
    the baseline (the jaws) and outward, in the part frame."""
    tilt = np.radians(P["camera_tilt_deg"])
    axis = np.array([-np.sin(tilt), 0.0, np.cos(tilt)])
    out = np.array([np.cos(tilt), 0.0, np.sin(tilt)])
    return np.array(P["glass_centre_mm"], float), axis, np.array([0.0, 1, 0]), out


def layout():
    """The part's construction in the part frame (mm), as build_mount.py
    builds it: the base on the pad, the plate the D405's back screws to, and
    the web between them; holes as (centre, axis)."""
    glass, axis, along, out = camera()
    back = glass - DEPTH * axis  # the D405's back face, the plate's front
    width, height, thick = P["plate_mm"]
    plate = np.array(
        [
            back + along * sy * width / 2 + out * su * height / 2 - axis * dn
            for dn in (0.0, thick)
            for sy in (-1, 1)
            for su in (-1, 1)
        ]
    )
    base = P["base_mm"]
    box = np.array(
        [
            [x, y, z]
            for x in base["x"]
            for y in base["y"]
            for z in (0.0, base["thickness"])
        ]
    )
    half = P["web_width_mm"] / 2
    # The web runs from inside the base to the plate's back face.
    web = np.array(
        [[x, y, base["thickness"] / 2] for x in base["x"] for y in (-half, half)]
        + [
            back + along * y + out * su * height / 2 - axis * thick
            for y in (-half, half)
            for su in (-1, 1)
        ]
    )
    spacing = P["screw_spacing_mm"] / 2
    return {
        "back": back,
        "axis": axis,
        "plate": plate,
        "base": box,
        "web": web,
        # Heat-set inserts in the base's underside, under the pad's holes.
        "inserts": [
            (np.array([0.0, s * spacing, 0.0]), np.array([0, 0, 1.0])) for s in (-1, 1)
        ],
        # Through the plate, for the D405's back screws.
        "screws": [
            (back + along * s * spacing - axis * thick / 2, axis) for s in (-1, 1)
        ],
    }


def block_pieces():
    """The part as three convex point sets in the part frame (mm): base, web
    and plate, without the holes."""
    lay = layout()
    return [lay["base"], lay["web"], lay["plate"]]


def tilt_in_site(side):
    """The adapter's turn of the camera toward the jaws in the D405 site's
    frame, whose x runs along the jaws (model.d405_map)."""
    return rot_x(-sign(side) * np.radians(P["camera_tilt_deg"]))


def sensor_pose(side, site_rot):
    """The D405 sensor frame on the adapter (origin at the glass centre, +z
    out of the glass) in the gripper link, given the rotation of the D405 site
    it replaces there: (position m, rotation)."""
    glass, _, _, _ = camera()
    return (PAD + glass) / 1000, site_rot @ tilt_in_site(side)
