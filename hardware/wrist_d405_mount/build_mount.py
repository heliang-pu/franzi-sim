"""Tilted wrist D405 block for the Franzi gripper: Blender script, writes the STL.

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python hardware/wrist_d405_mount/build_mount.py

It replaces only the small camera block that hangs in the clevis of the
existing L bracket (CAD: left_wrist_roll_Link.STL components 31 = L bracket,
32 = camera block). The original block holds the camera straight down, 58 mm
beside the gripper axis, so the jaws sit at the edge of the image; tilting it
about the clevis pin collides with the gripper housing beyond about 18 degrees.
This block holds the camera lower, further out and tilted toward the jaws as
params.json sets it: 20 degrees, 34 mm from the pin, 27 mm outward, which puts
the lens where the user's reference model (a D435 wrist camera, 2026-10-24) has
it relative to its gripper: 65 mm beside the jaw plane, 101 mm behind the
fingertips (check_mount.py).

The shape comes from sim/wrist_mount.py (layout), which the simulation also
mounts; this script only adds the Blender solids, the booleans and the export.

Part frame, millimetres: origin on the clevis pin axis, mid-plane of the
clevis; x along the pin (the jaw closing axis), z up, y outward (away from the
gripper) for the left hand. The right hand uses the same part turned 180
degrees about z. Interfaces (from the CAD, to be checked on the real gripper):
clevis gap 5.4 mm (tongue 5.0 mm), pin hole 4.0 mm (tongue hole 4.3 mm);
D405 back: 2x M3x0.5, 20 mm apart, at most 4.0 mm thread engagement
(RealSense D400 datasheet 337029-017, figure 10-13).
"""

import math
import sys
from pathlib import Path

import bmesh
import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from sim.wrist_mount import P, layout

PIN_HOLE = P["pin_hole_mm"]
SCREW_HOLE = P["screw_hole_mm"]
SCREW_SPACING = P["screw_spacing_mm"]
OUT = HERE / "wrist_d405_mount.stl"
LAY = layout()


def mesh_object(name, verts, faces):
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


def hull_slab(name, points_yz, half_x):
    """Convex hull of y-z points, extruded symmetrically along x."""
    bm = bmesh.new()
    for x in (-half_x, half_x):
        for y, z in points_yz:
            bm.verts.new((x, float(y), float(z)))
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
    obj.location = center
    return obj


def plate():
    # Corners in layout order (back face then pin side, x, the camera's up).
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    return mesh_object("plate", [tuple(map(float, v)) for v in LAY["plate"]], faces)


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
    part = hull_slab("tongue", LAY["tongue_yz"], LAY["tongue_half_x"])
    boolean(part, hull_slab("web", LAY["web_yz"], LAY["web_half_x"]), "UNION")
    boolean(part, plate(), "UNION")
    boolean(part, cylinder("pin", PIN_HOLE / 2, 40, (0, 0, 0), (1, 0, 0)), "DIFFERENCE")
    normal = tuple(map(float, LAY["normal"]))
    center = LAY["back"] + LAY["normal"] * P["plate_mm"][2] / 2
    for sx in (-1, 1):
        at = (sx * SCREW_SPACING / 2, float(center[1]), float(center[2]))
        boolean(part, cylinder("screw", SCREW_HOLE / 2, 20, at, normal), "DIFFERENCE")
    bpy.ops.object.select_all(action="DESELECT")
    part.select_set(True)
    bpy.context.view_layer.objects.active = part
    bpy.ops.wm.stl_export(
        filepath=str(OUT), export_selected_objects=True, global_scale=1.0
    )
    print(f"wrote {OUT} ({len(part.data.polygons)} faces)")


main()
