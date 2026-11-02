"""Wrist D405 adapter for wheel_robot_26.8.16_3's gripper: Blender script,
writes the STL.

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python hardware/wrist_d405_mount/build_mount.py

The gripper holds its D405 on a bent strip beside the housing, near the
flange, looking straight down the hand, so the housing hides the jaws. The
adapter takes the D405's place on the strip's top pad, on the two screws that
held the D405 (from below, into heat-set inserts), and holds the D405 38 mm
further down the hand, tilted 20 degrees toward the jaws: where the tilted
block of 2026-10-24 held it on wheel_robot_4.0's gripper (check_mount.py).

The shape comes from sim/wrist_mount.py (layout), which the simulation also
mounts; this script only adds the Blender solids, the booleans and the export.

Part frame, millimetres: origin on the pad's top face between its holes; x
outward from the hand, y along the jaws, z down the hand. Both hands take the
same part, unturned. Interfaces (from the CAD, to be checked on the real
gripper): pad holes 20 mm apart along the jaws; D405 back: 2x M3x0.5, 20 mm
apart, at most 4.0 mm thread engagement (RealSense D400 datasheet 337029-017,
figure 10-13).
"""

import math
import sys
from pathlib import Path

import bmesh
import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from sim.wrist_mount import P, layout

OUT = HERE / "wrist_d405_mount.stl"
LAY = layout()


def hull(name, points):
    """Convex hull of 3-D points."""
    bm = bmesh.new()
    for p in points:
        bm.verts.new(tuple(map(float, p)))
    bmesh.ops.convex_hull(bm, input=bm.verts)
    data = bpy.data.meshes.new(name)
    bm.to_mesh(data)
    bm.free()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


def cylinder(name, radius, depth, center, axis, segments=64):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=segments, radius=radius, depth=depth, location=(0, 0, 0)
    )
    obj = bpy.context.active_object
    obj.name = name
    z = (0.0, 0.0, 1.0)
    a = tuple(x / math.sqrt(sum(c * c for c in axis)) for x in axis)
    cross = (
        z[1] * a[2] - z[2] * a[1],
        z[2] * a[0] - z[0] * a[2],
        z[0] * a[1] - z[1] * a[0],
    )
    dot = sum(p * q for p, q in zip(z, a))
    norm = math.sqrt(sum(c * c for c in cross))
    if norm > 1e-9:
        obj.rotation_mode = "AXIS_ANGLE"
        obj.rotation_axis_angle = (math.atan2(norm, dot), *[c / norm for c in cross])
    obj.location = tuple(map(float, center))
    return obj


def boolean(target, tool, op):
    mod = target.modifiers.new(tool.name, "BOOLEAN")
    mod.operation = op
    mod.object = tool
    mod.solver = "EXACT"
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(tool)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    part = hull("base", LAY["base"])
    boolean(part, hull("web", LAY["web"]), "UNION")
    boolean(part, hull("plate", LAY["plate"]), "UNION")
    depth = P["insert_depth_mm"]
    for k, (at, axis) in enumerate(LAY["inserts"]):
        # Blind, from the underside: the cylinder's top ends depth into the base.
        centre = [at[i] + axis[i] * (depth - 5) / 2 for i in range(3)]
        hole = cylinder(f"insert{k}", P["insert_hole_mm"] / 2, depth + 5, centre, axis)
        boolean(part, hole, "DIFFERENCE")
    for k, (at, axis) in enumerate(LAY["screws"]):
        hole = cylinder(f"screw{k}", P["screw_hole_mm"] / 2, 20, at, axis)
        boolean(part, hole, "DIFFERENCE")
    bpy.ops.object.select_all(action="DESELECT")
    part.select_set(True)
    bpy.context.view_layer.objects.active = part
    bpy.ops.wm.stl_export(
        filepath=str(OUT), export_selected_objects=True, global_scale=1.0
    )
    print(f"wrote {OUT} ({len(part.data.polygons)} faces)")


main()
