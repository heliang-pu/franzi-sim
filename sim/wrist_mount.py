"""The tilted wrist D405 block the simulation mounts (hardware/wrist_d405_mount).

The block replaces the original camera block in the clevis of the gripper's L
bracket and holds the D405 tilted toward the jaws: the geometry of the user's
reference model (user decision 2026-10-24). Its parameters live in
hardware/wrist_d405_mount/params.json. This module turns them into the D405's
frame on each wrist, the block's shape and its placement. Pure NumPy, so the
Blender build script, the mount check and the scene build share it.

Frames: wrist_roll_Link, millimetres unless a name ends in _m. The part frame
has its origin on the clevis pin axis, x along the pin (the jaw axis), z up
and y outward for the left hand; the right hand uses the same part turned 180
degrees about z.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOUNT = ROOT / "hardware/wrist_d405_mount"
P = json.loads((MOUNT / "params.json").read_text())
STL = MOUNT / "wrist_d405_mount.stl"
HINGE = np.array(P["interface_from_cad"]["clevis_pin_axis_in_left_wrist_roll_link_mm"])
DEPTH = P["d405"]["housing_mm"][2]  # back face to glass
# The original camera block and its two M3 screws in left_wrist_roll_Link.STL
# (bounds, mm); the right wrist mirrors them in y.
OLD_BLOCK_BOXES = (
    ((-45.1, 48.0, -111.2), (-15.1, 68.0, -87.1)),  # camera block
    ((-21.6, 56.5, -117.2), (-18.6, 59.5, -111.2)),  # M3 screw
    ((-41.6, 56.5, -117.2), (-38.6, 59.5, -111.2)),  # M3 screw
)
ROT_Z180 = np.diag([-1.0, -1.0, 1.0])


def sign(side):
    return 1.0 if side == "left" else -1.0


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def tilt_rotation(side, tilt=None):
    """The block's turn of the camera toward the jaws, about the jaw axis."""
    tilt = P["camera_tilt_deg"] if tilt is None else tilt
    return rot_x(-sign(side) * np.radians(tilt))


def camera(side, tilt=None, standoff=None, outward=None):
    """D405 back-face centre (mm), optical axis, and image right and up in the
    side's wrist_roll_Link frame: right along the baseline (the jaw axis), up
    outward. tilt 0, standoff 17, outward 0 is the original block."""
    standoff = P["standoff_mm"] if standoff is None else standoff
    outward = P["outward_mm"] if outward is None else outward
    sy = sign(side)
    R = tilt_rotation(side, tilt)
    back = HINGE * [1, sy, 1] + R @ np.array([0, sy * outward, -standoff])
    axis = R @ np.array([0.0, 0, -1])
    up = R @ np.array([0.0, sy, 0])
    right = np.cross(up, -axis)
    return back, axis, right, up


def sensor_pose(side, site_pos_m, site_rot):
    """The D405 sensor frame on the block (origin at the glass centre, +z out
    of the glass, as the URDF site), from the URDF site it replaces: the
    site's axes turned by the block's tilt. Returns (position m, rotation)."""
    back, axis, _, _ = camera(side)
    return (back + DEPTH * axis) / 1000, tilt_rotation(side) @ site_rot


def layout():
    """The block's construction in the part frame (mm), as build_mount.py
    builds it: the camera back face and its axes, and the y-z outlines that
    are extruded along x into the tongue and the web."""
    R = rot_x(-np.radians(P["camera_tilt_deg"]))
    back = R @ np.array([0.0, P["outward_mm"], -P["standoff_mm"]])
    normal, up = R @ np.array([0.0, 0, 1]), R @ np.array([0.0, 1, 0])
    hx, hu, t = P["plate_mm"][0] / 2, P["plate_mm"][1] / 2, P["plate_mm"][2]
    top = back + normal * t  # plate face on the pin side
    a = 2 * np.pi * np.arange(48) / 48
    circle = np.c_[np.cos(a), np.sin(a)] * P["tongue_radius_mm"]
    edge = np.array([(top + up * s * hu)[1:] for s in (-0.8, 0.8)])
    # Below the clevis prongs the web widens to its full width.
    direction = back[1:] / np.linalg.norm(back[1:])
    across = np.array([-direction[1], direction[0]])
    root = direction * P["web_widens_beyond_mm"]
    web = np.r_[[root + s * 4.0 * across for s in (-1, 1)], edge]
    plate = np.array(
        [
            [sx * hx, *(back + normal * dn + up * su * hu)[1:]]
            for dn in (0.0, t)
            for sx in (-1, 1)
            for su in (-1, 1)
        ]
    )
    return {
        "back": back,
        "normal": normal,
        "up": up,
        "tongue_yz": np.r_[circle, edge],
        "tongue_half_x": P["tongue_thickness_mm"] / 2,
        "web_yz": web,
        "web_half_x": P["web_width_mm"] / 2,
        "plate": plate,
    }


def block_pieces():
    """The block as three convex point sets in the part frame (mm): tongue,
    web and plate, without the pin and screw holes."""
    lay = layout()

    def slab(yz, half):
        return np.array([[x, y, z] for x in (-half, half) for y, z in yz])

    return [
        slab(lay["tongue_yz"], lay["tongue_half_x"]),
        slab(lay["web_yz"], lay["web_half_x"]),
        lay["plate"],
    ]


def part_to_wrist(side):
    """Rotation and translation (mm) from the part frame to the wrist frame."""
    if side == "left":
        return np.eye(3), HINGE.copy()
    return ROT_Z180, HINGE * [1, -1, 1]


def is_old_block(bounds_m, side):
    """Whether a wrist CAD component (bounds in metres) is the original camera
    block or one of its two screws, which the new block replaces."""
    lo, hi = np.asarray(bounds_m) * 1000
    for box in OLD_BLOCK_BOXES:
        blo, bhi = np.array(box)
        if side == "right":
            blo, bhi = blo * [1, -1, 1], bhi * [1, -1, 1]
            blo, bhi = np.minimum(blo, bhi), np.maximum(blo, bhi)
        if np.abs(np.r_[lo - blo, hi - bhi]).max() < 0.6:
            return True
    return False
