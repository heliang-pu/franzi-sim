"""Shared Blender 5.x prop pipeline; all geometry arguments are millimetres.

Use from a headless build script: blender -b --factory-startup -P script -- args.
Locations are primitive centres (including Z); all exports bake world transforms
into mesh vertices, preserving the shared prop frame. No per-part recentering.
Source image textures should name a separate UV layer (e.g. LabelUV); UVMap is
the nonoverlapping export/bake atlas. Polygonal bores specify an *apothem*;
cylinders specify a circumradius so shafts never exceed their nominal diameter.
"""

import json
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from prop_schema import FICTIONAL_BRANDS, check_schema

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PYTHON = Path("/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python")

# Every builder imports this module, including source/geometry resume modes.
bpy.context.preferences.filepaths.save_version = 0
_DELIVERED_PROPS = set()


def finalize_assets():
    """Finish successful builds with the public-repository file formats.

    Keep this after source saving and the task-specific collision checks: the
    compactor preserves their topology, including precision exceptions, rather
    than rebuilding hulls after rounding. Source .blend files remain untouched.
    """
    if _DELIVERED_PROPS:
        subprocess.run(
            [str(VALIDATOR_PYTHON), "-B", str(Path(__file__).with_name("compact_props.py")),
             "--apply", "--root", str(ROOT / "assets" / "props"),
             "--props", *sorted(_DELIVERED_PROPS)],
            check=True,
        )


def _optimize_png(path):
    subprocess.run(
        [str(VALIDATOR_PYTHON), "-B", str(Path(__file__).with_name("compact_props.py")),
         "--png", str(Path(path).resolve())], check=True,
    )


def locate_blender():
    preferred = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    located = str(preferred) if preferred.is_file() else shutil.which("blender")
    if not located:
        raise FileNotFoundError("Blender is not installed at the macOS path or on PATH")
    return Path(located)


def mm(value):
    return float(value) / 1000.0


def reset_scene():
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    for other in list(bpy.data.scenes):
        if other != scene:
            bpy.data.scenes.remove(other)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.cycles.seed = 0
    scene.cycles.use_adaptive_sampling = False
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.margin = 16
    return scene


def _linear_color(hex_color):
    color = hex_color.lstrip("#")
    if len(color) not in (6, 8):
        raise ValueError("base_color_hex must be #RRGGBB or #RRGGBBAA")
    rgb = [int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    rgb = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return (*rgb, int(color[6:8], 16) / 255 if len(color) == 8 else 1.0)


def _math(tree, operation, *inputs):
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    for index, value in enumerate(inputs):
        if isinstance(value, (int, float)):
            node.inputs[index].default_value = value
        else:
            tree.links.new(value, node.inputs[index])
    return node.outputs[0]


def _surface_group(name):
    group = bpy.data.node_groups.new(name, "ShaderNodeTree")
    group.interface.new_socket(name="Color", in_out="INPUT", socket_type="NodeSocketColor")
    group.interface.new_socket(name="Strength", in_out="INPUT", socket_type="NodeSocketFloat")
    group.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    source = group.nodes.new("NodeGroupInput")
    target = group.nodes.new("NodeGroupOutput")
    return group, source, target


def pbr_material(name, base_color_hex, roughness, metallic, wear=0, grime=0, machining=None):
    if any(not 0 <= value <= 1 for value in (roughness, metallic, wear, grime)):
        raise ValueError("roughness, metallic, wear and grime must be in [0, 1]")
    if machining not in (None, "turned", "milled", "brushed"):
        raise ValueError("machining must be turned, milled, brushed or None")
    mat = bpy.data.materials.new(name)
    tree = mat.node_tree
    tree.nodes.clear()
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.name = "Principled BSDF"
    bsdf.inputs["Base Color"].default_value = _linear_color(base_color_hex)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    mat.diffuse_color = _linear_color(base_color_hex)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    color = tree.nodes.new("ShaderNodeRGB").outputs[0]
    color.default_value = _linear_color(base_color_hex)
    if wear:
        group, source, target = _surface_group("Edge wear: convex curvature")
        geom = group.nodes.new("ShaderNodeNewGeometry")
        # Pointiness above 0.5 identifies convex edges (bevelled actual geometry).
        mask = _math(
            group,
            "MULTIPLY",
            _math(group, "MAXIMUM", _math(group, "SUBTRACT", geom.outputs["Pointiness"], 0.5), 0),
            12,
        )
        factor = _math(group, "MINIMUM", _math(group, "MULTIPLY", mask, source.outputs["Strength"]), 1)
        mix = group.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MIX"
        group.links.new(factor, mix.inputs[0])
        group.links.new(source.outputs["Color"], mix.inputs[1])
        mix.inputs[2].default_value = (0.8, 0.82, 0.84, 1)
        group.links.new(mix.outputs[0], target.inputs["Color"])
        node = tree.nodes.new("ShaderNodeGroup")
        node.node_tree = group
        node.inputs["Strength"].default_value = wear
        tree.links.new(color, node.inputs["Color"])
        color = node.outputs["Color"]
    if grime:
        group, source, target = _surface_group("Grime: AO cavity darkening")
        ao = group.nodes.new("ShaderNodeAmbientOcclusion")
        ao.inputs["Distance"].default_value = mm(8)
        ao.samples = 16
        cavity = _math(group, "SUBTRACT", 1, ao.outputs["AO"])
        dark = _math(group, "SUBTRACT", 1, _math(group, "MULTIPLY", cavity, source.outputs["Strength"]))
        mix = group.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs[0].default_value = 1
        group.links.new(source.outputs["Color"], mix.inputs[1])
        group.links.new(dark, mix.inputs[2])
        group.links.new(mix.outputs[0], target.inputs["Color"])
        node = tree.nodes.new("ShaderNodeGroup")
        node.node_tree = group
        node.inputs["Strength"].default_value = grime
        tree.links.new(color, node.inputs["Color"])
        color = node.outputs["Color"]
    if machining:
        group, source, target = _surface_group("Machining: " + machining)
        coords = group.nodes.new("ShaderNodeTexCoord")
        separate = group.nodes.new("ShaderNodeSeparateXYZ")
        group.links.new(coords.outputs["Object"], separate.inputs[0])
        x, y = separate.outputs["X"], separate.outputs["Y"]
        if machining == "turned":
            phase = _math(
                group,
                "SQRT",
                _math(group, "ADD", _math(group, "MULTIPLY", x, x), _math(group, "MULTIPLY", y, y)),
            )
        elif machining == "milled":
            # Repeated parallel arcs, translated in Y; 20 mm cutter radius.
            arc = _math(
                group,
                "SQRT",
                _math(
                    group,
                    "MAXIMUM",
                    _math(group, "SUBTRACT", mm(20) ** 2, _math(group, "MULTIPLY", x, x)),
                    1e-12,
                ),
            )
            phase = _math(group, "ADD", y, arc)
        else:
            phase = y
        marks = _math(group, "SINE", _math(group, "MULTIPLY", phase, 2 * math.pi / mm(0.18)))
        shade = _math(group, "ADD", 0.96, _math(group, "MULTIPLY", marks, 0.04))
        mix = group.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs[0].default_value = 1
        group.links.new(source.outputs["Color"], mix.inputs[1])
        group.links.new(shade, mix.inputs[2])
        group.links.new(mix.outputs[0], target.inputs["Color"])
        bump_output = group.interface.new_socket(
            name="Height", in_out="OUTPUT", socket_type="NodeSocketFloat"
        )
        del bump_output
        group.links.new(marks, target.inputs["Height"])
        node = tree.nodes.new("ShaderNodeGroup")
        node.node_tree = group
        tree.links.new(color, node.inputs["Color"])
        color = node.outputs["Color"]
        bump = tree.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.12
        bump.inputs["Distance"].default_value = mm(0.015)
        tree.links.new(node.outputs["Height"], bump.inputs["Height"])
        tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    tree.links.new(color, bsdf.inputs["Base Color"])
    return mat


def _select(objects):
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.hide_set(False)
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def smooth_by_angle(obj, angle_degrees=30):
    """Blender 5.x: use sharp_edge attributes, not the removed auto_smooth flag."""
    mesh = obj.data
    for face in mesh.polygons:
        face.use_smooth = True
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for edge in bm.edges:
        edge.smooth = edge.is_manifold and edge.calc_face_angle(0) <= math.radians(angle_degrees)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def _finish(obj, name, location_mm, bevel_mm, segments, material, rotation_deg=(0, 0, 0)):
    if bevel_mm < 0 or segments < 1:
        raise ValueError("bevel width must be nonnegative and segments >= 1")
    obj.name = name
    obj.location = [mm(v) for v in location_mm]
    obj.rotation_euler = [math.radians(v) for v in rotation_deg]
    _select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if bevel_mm:
        bevel = obj.modifiers.new("Edge bevel (mm)", "BEVEL")
        bevel.width = mm(bevel_mm)
        bevel.segments = int(segments)
        bevel.limit_method = "ANGLE"
        bevel.angle_limit = math.radians(20)
        bevel.harden_normals = True
        bpy.ops.object.modifier_apply(modifier=bevel.name)
    smooth_by_angle(obj)
    if material is not None:
        obj.data.materials.append(material)
    return obj


def box(
    name, size_mm, location_mm=(0, 0, 0), bevel_mm=0.5, segments=3, material=None, rotation_deg=(0, 0, 0)
):
    if len(size_mm) != 3 or min(size_mm) <= 0:
        raise ValueError("box dimensions must be positive X, Y, Z")
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.scale = [mm(v) for v in size_mm]
    return _finish(obj, name, location_mm, bevel_mm, segments, material, rotation_deg)


def rounded_box(name, size_mm, radius_mm, location_mm=(0, 0, 0), bevel_mm=None, segments=3, material=None):
    """Round all edges; radius_mm is the bevel radius, inside the stated bbox."""
    return box(name, size_mm, location_mm, radius_mm if bevel_mm is None else bevel_mm, segments, material)


def cylinder(
    name,
    radius_mm,
    depth_mm,
    location_mm=(0, 0, 0),
    bevel_mm=0.5,
    segments=3,
    vertices=64,
    material=None,
    rotation_deg=(0, 0, 0),
):
    if min(radius_mm, depth_mm) <= 0 or vertices < 3:
        raise ValueError("invalid cylinder dimensions/vertices")
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=mm(radius_mm), depth=mm(depth_mm))
    return _finish(bpy.context.object, name, location_mm, bevel_mm, segments, material, rotation_deg)


def _mesh_object(name, vertices_mm, faces, location_mm, bevel_mm, segments, material):
    mesh = bpy.data.meshes.new(name + " mesh")
    mesh.from_pydata([[mm(v) for v in xyz] for xyz in vertices_mm], [], faces)
    mesh.update()
    # Concave end caps are tessellated geometrically, not with a triangle fan.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return _finish(obj, name, location_mm, bevel_mm, segments, material)


def extruded_profile(
    name, profile_mm, depth_mm, location_mm=(0, 0, 0), bevel_mm=0.5, segments=3, material=None
):
    """Extrude a simple XY polygon about Z=0; either polygon winding is accepted."""
    if len(profile_mm) < 3 or depth_mm <= 0:
        raise ValueError("profile needs at least three XY vertices and positive depth")
    count = len(profile_mm)
    vertices = [(x, y, z) for z in (-depth_mm / 2, depth_mm / 2) for x, y in profile_mm]
    faces = [tuple(reversed(range(count))), tuple(range(count, 2 * count))]
    faces.extend((i, (i + 1) % count, (i + 1) % count + count, i + count) for i in range(count))
    return _mesh_object(name, vertices, faces, location_mm, bevel_mm, segments, material)


def hexagonal_prism(
    name, across_flats_mm, depth_mm, location_mm=(0, 0, 0), bevel_mm=0.5, segments=3, material=None
):
    radius = across_flats_mm / math.sqrt(3)
    profile = [(radius * math.cos(i * math.pi / 3), radius * math.sin(i * math.pi / 3)) for i in range(6)]
    return extruded_profile(name, profile, depth_mm, location_mm, bevel_mm, segments, material)


def tube(
    name,
    outer_radius_mm,
    inner_radius_mm,
    depth_mm,
    location_mm=(0, 0, 0),
    bevel_mm=0.5,
    segments=3,
    vertices=64,
    material=None,
):
    """Continuous watertight tube; inner radius is the exact polygon apothem."""
    angle = math.pi / vertices
    inner_vertex_radius = inner_radius_mm / math.cos(angle)
    if depth_mm <= 0 or inner_radius_mm <= 0 or outer_radius_mm <= inner_vertex_radius:
        raise ValueError("tube outer radius must exceed the bore's polygon circumradius")
    verts = []
    for z in (-depth_mm / 2, depth_mm / 2):
        for radius in (outer_radius_mm, inner_vertex_radius):
            verts.extend(
                (radius * math.cos(i * 2 * angle), radius * math.sin(i * 2 * angle), z)
                for i in range(vertices)
            )
    faces = []
    n = vertices
    for i in range(n):
        j = (i + 1) % n
        faces.extend(
            (
                (i, j, j + 2 * n, i + 2 * n),
                (i + n, i + 3 * n, j + 3 * n, j + n),
                (i, i + n, j + n, j),
                (i + 2 * n, j + 2 * n, j + 3 * n, i + 3 * n),
            )
        )
    return _mesh_object(name, verts, faces, location_mm, bevel_mm, segments, material)


def ring_segments(
    name,
    inner_radius_mm,
    outer_radius_mm,
    depth_mm,
    count=32,
    location_mm=(0, 0, 0),
    start_angle_deg=0,
    bevel_mm=0,
    segments=2,
    material=None,
    top_inner_radius_mm=None,
):
    """Separate convex bore-wall sectors, optionally tapered for an entry chamfer.

    Each inner flat has precisely the requested apothem. The outer vertices lie
    on the outer circumcircle. No single hull is ever made around the bore.
    top_inner_radius_mm changes only the top apothem; depth is centred on Z.
    """
    if count < 3 or depth_mm <= 0 or inner_radius_mm <= 0:
        raise ValueError("invalid ring dimensions/count")
    upper = inner_radius_mm if top_inner_radius_mm is None else top_inner_radius_mm
    if (
        min(inner_radius_mm, upper) <= 0
        or max(inner_radius_mm, upper) / math.cos(math.pi / count) >= outer_radius_mm
    ):
        raise ValueError("outer radius must enclose the bore polygon")
    result = []
    for index in range(count):
        a = math.radians(start_angle_deg) + index * 2 * math.pi / count
        b = a + 2 * math.pi / count
        verts = []
        for z, apothem in ((-depth_mm / 2, inner_radius_mm), (depth_mm / 2, upper)):
            inner = apothem / math.cos(math.pi / count)
            verts.extend(
                (r * math.cos(theta), r * math.sin(theta), z)
                for r, theta in ((inner, a), (outer_radius_mm, a), (outer_radius_mm, b), (inner, b))
            )
        faces = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        result.append(
            _mesh_object(f"{name}_{index:02d}", verts, faces, location_mm, bevel_mm, segments, material)
        )
    return result


polygonal_ring_segments = ring_segments
hex_prism = hexagonal_prism
extruded_2d_profile = extruded_profile


def join_objects(objects, name):
    _select(list(objects))
    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = name
    return obj


def _pack_uv(objects):
    _select(objects)
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(
        rotate=True,
        scale=True,
        merge_overlap=False,
        margin_method="FRACTION",
        margin=0.01,
        # AABB packing is deterministic, separates every island and avoids
        # Blender 5.2's very slow concave/fraction search on rounded meshes.
        shape_method="AABB",
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def uv_unwrap(obj, method="smart"):
    if method not in ("smart", "cube"):
        raise ValueError("UV method must be smart or cube")
    _select([obj])
    layer = obj.data.uv_layers.get("UVMap") or obj.data.uv_layers.new(name="UVMap")
    obj.data.uv_layers.active = layer
    layer.active_render = True
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    if method == "smart":
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.01, margin_method="FRACTION")
    else:
        # Cube projection by itself stacks opposite sides. Packing with
        # merge_overlap=False separates those disconnected islands.
        bpy.ops.uv.cube_project(cube_size=1, correct_aspect=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    _pack_uv([obj])
    return layer


def _save_linear_png(image, path, scene):
    settings = scene.render.image_settings
    saved = (settings.file_format, settings.color_mode, settings.color_depth)
    view = scene.view_settings
    saved_view = (view.view_transform, view.look, view.exposure, view.gamma)
    try:
        settings.file_format, settings.color_mode, settings.color_depth = "PNG", "RGB", "8"
        view.view_transform, view.look, view.exposure, view.gamma = "Standard", "None", 0, 1
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save_render(str(Path(path).resolve()), scene=scene)
        _optimize_png(path)
    finally:
        settings.file_format, settings.color_mode, settings.color_depth = saved
        view.view_transform, view.look, view.exposure, view.gamma = saved_view


def bake_albedo(objects, size, out_png, *, ao_strength=0.6):
    """64-sample colour-only diffuse and AO in linear light, then sRGB PNG.

    AO is mixed as (1 - ao_strength) + ao_strength * AO, defaulting to 0.6.
    Multiple input objects are packed into one target atlas. Their original PBR
    graphs are restored, including Metallic=1, after the colour bake. To avoid
    metallic shaders having no diffuse lobe, metallic is temporarily zero.
    """
    objects = list(objects)
    if not 0 <= ao_strength <= 1:
        raise ValueError("ao_strength must be in [0, 1]")
    if not objects or size not in (1024, 2048):
        raise ValueError("bake needs mesh objects and size 1024 or 2048")
    for obj in objects:
        if obj.type != "MESH":
            raise ValueError("only meshes can be baked")
        if not obj.data.uv_layers.get("UVMap"):
            uv_unwrap(obj)
        obj.data.uv_layers.active = obj.data.uv_layers["UVMap"]
        obj.data.uv_layers["UVMap"].active_render = True
        if not obj.data.materials:
            obj.data.materials.append(pbr_material(obj.name + " default", "#808080", 0.6, 0))
    if len(objects) > 1:
        _pack_uv(objects)
    scene = bpy.context.scene
    scene.render.engine, scene.cycles.device, scene.cycles.samples = "CYCLES", "CPU", 64
    scene.cycles.use_adaptive_sampling = False
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    materials = {slot.material for obj in objects for slot in obj.material_slots if slot.material}
    active_nodes, temporary_nodes, metallic_states, images = {}, [], [], []
    try:
        for mat in materials:
            tree = mat.node_tree
            active_nodes[mat] = tree.nodes.active
            node = tree.nodes.new("ShaderNodeTexImage")
            node.name = "Temporary bake target"
            tree.nodes.active = node
            temporary_nodes.append((mat, node))
            for bsdf in [n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"]:
                socket = bsdf.inputs["Metallic"]
                links = [link.from_socket for link in socket.links]
                metallic_states.append((tree, socket, socket.default_value, links))
                for link in list(socket.links):
                    tree.links.remove(link)
                socket.default_value = 0
        passes = []
        for kind in ("DIFFUSE", "AO"):
            image = bpy.data.images.new(
                "Bake " + kind, width=size, height=size, alpha=True, float_buffer=True, is_data=True
            )
            image.colorspace_settings.name = "Non-Color"
            images.append(image)
            for mat, node in temporary_nodes:
                node.image = image
                mat.node_tree.nodes.active = node
            _select(objects)
            kwargs = {
                "type": kind,
                "use_clear": True,
                "use_selected_to_active": False,
                "margin": 16,
                "uv_layer": "UVMap",
            }
            if kind == "DIFFUSE":
                kwargs["pass_filter"] = {"COLOR"}
            bpy.ops.object.bake(**kwargs)
            pixels = np.empty(size * size * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            passes.append(pixels.reshape(size, size, 4))
        color, ao = passes
        color[:, :, :3] *= (1 - ao_strength) + ao_strength * np.clip(ao[:, :, :3], 0, 1)
        color[:, :, 3] = 1
        combined = bpy.data.images.new(
            "Linear combined albedo", width=size, height=size, alpha=True, float_buffer=True
        )
        images.append(combined)
        combined.colorspace_settings.name = "Linear Rec.709"
        combined.pixels.foreach_set(color.ravel())
        combined.update()
        _save_linear_png(combined, out_png, scene)
        result = bpy.data.images.load(str(Path(out_png).resolve()), check_existing=False)
        result.colorspace_settings.name = "sRGB"
        return result
    finally:
        for tree, socket, value, links in metallic_states:
            socket.default_value = value
            for source in links:
                tree.links.new(source, socket)
        for mat, node in temporary_nodes:
            mat.node_tree.nodes.remove(node)
            mat.node_tree.nodes.active = active_nodes[mat]
        for image in images:
            bpy.data.images.remove(image)


def _world_mesh(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(
        obj.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph
    )
    mesh.transform(obj.matrix_world)
    return mesh


def export_obj(obj, path):
    """Export an evaluated copy with identity axes, scale and object transform."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = _world_mesh(obj)
    temporary = bpy.data.objects.new(obj.name + "_export", mesh)
    bpy.context.collection.objects.link(temporary)
    try:
        _select([temporary])
        bpy.ops.wm.obj_export(
            filepath=str(path),
            check_existing=False,
            export_selected_objects=True,
            forward_axis="Y",
            up_axis="Z",
            global_scale=1.0,
            export_uv=True,
            export_normals=True,
            export_materials=False,
            apply_modifiers=True,
            export_triangulated_mesh=True,
        )
    finally:
        bpy.data.objects.remove(temporary, do_unlink=True)
        bpy.data.meshes.remove(mesh)
    return path.name


def _hull(points):
    bm = bmesh.new()
    for point in points:
        bm.verts.new(point)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-9)
    if len(bm.verts) < 4:
        bm.free()
        raise ValueError("collision piece has fewer than four distinct vertices")
    result = bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
    interior = [
        v for v in result["geom_interior"] + result["geom_unused"] if isinstance(v, bmesh.types.BMVert)
    ]
    if interior:
        bmesh.ops.delete(bm, geom=list(set(interior)), context="VERTS")
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    if abs(bm.calc_volume(signed=True)) < 1e-16:
        bm.free()
        raise ValueError("collision piece has no volume (planes need an explicit simulator plane)")
    return bm


def _split_hulls(bm, depth=0):
    """Exact planar subdivision, never decimation that could close a precision bore."""
    if len(bm.verts) <= 64:
        return [bm]
    if depth >= 16:
        bm.free()
        raise ValueError("collision hull could not be split below the 64-vertex limit")
    points = [v.co.copy() for v in bm.verts]
    bounds = np.asarray(points)
    axis = int(np.argmax(np.ptp(bounds, axis=0)))
    middle = (bounds[:, axis].min() + bounds[:, axis].max()) / 2
    intersections = []
    for edge in bm.edges:
        a, b = (v.co for v in edge.verts)
        if (a[axis] - middle) * (b[axis] - middle) < 0:
            intersections.append(a.lerp(b, (middle - a[axis]) / (b[axis] - a[axis])))
    low = [p for p in points if p[axis] <= middle] + intersections
    high = [p for p in points if p[axis] >= middle] + intersections
    bm.free()
    return _split_hulls(_hull(low), depth + 1) + _split_hulls(_hull(high), depth + 1)


def export_collision(part_name, pieces, folder):
    """Caller supplies the concave decomposition; each piece is hull-converted.

    Oversize hulls are split without changing their union. This may emit more
    files than inputs; use the returned names in prop.json. It does not hull
    several pieces together, so a caller's cavities remain open.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for piece in pieces:
        mesh = _world_mesh(piece)
        try:
            hulls = _split_hulls(_hull([v.co for v in mesh.vertices]))
        finally:
            bpy.data.meshes.remove(mesh)
        for bm in hulls:
            hull_mesh = bpy.data.meshes.new("Convex collision")
            bm.to_mesh(hull_mesh)
            bm.free()
            obj = bpy.data.objects.new("Convex collision", hull_mesh)
            bpy.context.collection.objects.link(obj)
            try:
                name = f"{part_name}_col_{len(files):02d}.obj"
                export_obj(obj, folder / name)
                files.append(name)
            finally:
                bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.meshes.remove(hull_mesh)
    return files


def write_prop_json(data, path):
    path = Path(path)

    def texture_sizes(value):
        if isinstance(value, dict):
            if value.get("texture") and "texture_px" in value:
                texture = path.parent / value["texture"]
                if texture.is_file():
                    with texture.open("rb") as stream:
                        width, height = struct.unpack(">II", stream.read(24)[16:24])
                    if width != height:
                        raise ValueError(f"Non-square atlas: {texture}")
                    value["texture_px"] = width
            for child in value.values():
                texture_sizes(child)
        elif isinstance(value, list):
            for child in value:
                texture_sizes(child)

    texture_sizes(data)
    check_schema(data)
    if path.parent.name != data["id"]:
        raise ValueError("prop id must match its folder name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if path.parent.resolve().parent == ROOT / "assets" / "props":
        _DELIVERED_PROPS.add(path.parent.name)


def triangle_count(objects):
    count = 0
    for obj in objects:
        mesh = _world_mesh(obj)
        mesh.calc_loop_triangles()
        count += len(mesh.loop_triangles)
        bpy.data.meshes.remove(mesh)
    return count


def bbox_mm(objects):
    points = []
    for obj in objects:
        mesh = _world_mesh(obj)
        points.extend(tuple(v.co) for v in mesh.vertices)
        bpy.data.meshes.remove(mesh)
    return (np.ptp(np.asarray(points), axis=0) * 1000).tolist()


def render_preview(objects, out_png):
    target = Path(out_png).resolve()
    if target.parent.parent == ROOT / "assets" / "props":
        _DELIVERED_PROPS.add(target.parent.name)
    objects = list(objects)
    if not objects:
        raise ValueError("preview needs at least one object")
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    bounds = np.asarray(points)
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    center = Vector((lower + upper) / 2)
    span = float(max(upper - lower))
    scene = bpy.data.scenes.new("Temporary prop preview")
    created_objects = []
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.device, scene.cycles.samples, scene.cycles.seed = "CPU", 64, 0
        scene.cycles.use_adaptive_sampling = False
        scene.render.resolution_x = scene.render.resolution_y = 512
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode, scene.render.image_settings.color_depth = "RGB", "8"
        scene.view_settings.view_transform, scene.view_settings.look = "AgX", "None"
        scene.world = bpy.data.worlds.new("Neutral grey preview world")
        scene.world.use_nodes = True
        scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.18, 0.18, 0.18, 1)
        scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45
        for obj in objects:
            scene.collection.objects.link(obj)
        camera_data = bpy.data.cameras.new("Front-left preview camera")
        camera = bpy.data.objects.new("Front-left preview camera", camera_data)
        scene.collection.objects.link(camera)
        created_objects.append(camera)
        camera.location = center + Vector((-1.8, 1.35, 1.25)) * span
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera_data.type, camera_data.ortho_scale = "ORTHO", span * 1.95
        camera_data.clip_start, camera_data.clip_end = max(span * 0.001, 0.0001), span * 30
        scene.camera = camera
        for name, offset, power, size in (
            ("Key", (-1.5, 1.2, 2.1), 120, 1.4),
            ("Fill", (-0.8, -1.8, 0.9), 55, 1.6),
            ("Rim", (1.2, 0.7, 1.7), 85, 1.1),
        ):
            data = bpy.data.lights.new(name, "AREA")
            data.energy, data.shape, data.size = power * span * span, "DISK", size * span
            obj = bpy.data.objects.new(name, data)
            scene.collection.objects.link(obj)
            created_objects.append(obj)
            obj.location = center + Vector(offset) * span
            obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
        floor_mesh = bpy.data.meshes.new("Preview floor")
        floor_mesh.from_pydata(
            [
                (-5 * span, -5 * span, lower[2] - span * 0.003),
                (5 * span, -5 * span, lower[2] - span * 0.003),
                (5 * span, 5 * span, lower[2] - span * 0.003),
                (-5 * span, 5 * span, lower[2] - span * 0.003),
            ],
            [],
            [(0, 1, 2, 3)],
        )
        floor = bpy.data.objects.new("Preview floor", floor_mesh)
        scene.collection.objects.link(floor)
        created_objects.append(floor)
        floor.data.materials.append(pbr_material("Preview neutral grey", "#777777", 0.8, 0))
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(Path(out_png).resolve())
        bpy.ops.render.render(write_still=True, scene=scene.name)
        _optimize_png(out_png)
    finally:
        world = scene.world
        for obj in created_objects:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data.users == 0:
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                elif isinstance(data, bpy.types.Camera):
                    bpy.data.cameras.remove(data)
                elif isinstance(data, bpy.types.Light):
                    bpy.data.lights.remove(data)
        bpy.data.scenes.remove(scene)
        if world and world.users == 0:
            bpy.data.worlds.remove(world)


def _label_blender(path, width, height, fields):
    """Pillow-free fallback: text/checkers -> selected-to-active diffuse bake."""
    original_scene = bpy.context.window.scene
    scene = bpy.data.scenes.new("Temporary label bake")
    bpy.context.window.scene = scene
    created = []
    try:
        scene.render.engine, scene.cycles.device, scene.cycles.samples = "CYCLES", "CPU", 64
        scene.cycles.use_adaptive_sampling = False
        white = pbr_material("Label paper", fields.get("background", "#F5F3E9"), 1, 0)
        black = pbr_material("Label ink", fields.get("foreground", "#15191E"), 1, 0)
        aspect = width / height

        def plane(name, x, y, w, h, z, material):
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata([(x, y, z), (x + w, y, z), (x + w, y + h, z), (x, y + h, z)], [], [(0, 1, 2, 3)])
            obj = bpy.data.objects.new(name, mesh)
            scene.collection.objects.link(obj)
            mesh.materials.append(material)
            created.append(obj)
            return obj

        paper = white
        if fields.get("checker"):
            count = int(fields.get("checker_cells", 8))
            if not 1 <= count <= 128:
                raise ValueError("checker_cells must be between 1 and 128")
            paper = pbr_material("Checker paper", "#FFFFFF", 1, 0)
            tree = paper.node_tree
            checker = tree.nodes.new("ShaderNodeTexChecker")
            checker.inputs["Color1"].default_value = _linear_color(fields.get("background", "#F5F3E9"))
            checker.inputs["Color2"].default_value = _linear_color(fields.get("checker_dark", "#285575"))
            checker.inputs["Scale"].default_value = count
            coordinates = tree.nodes.new("ShaderNodeUVMap")
            coordinates.uv_map = "LabelSourceUV"
            tree.links.new(coordinates.outputs["UV"], checker.inputs["Vector"])
            tree.links.new(checker.outputs["Color"], tree.nodes["Principled BSDF"].inputs["Base Color"])
        sources = [plane("Paper source", 0, 0, aspect, 1, 0.001, paper)]

        def text(body, x, y, size, material=black, centered=False):
            curve = bpy.data.curves.new("Label text", "FONT")
            curve.body, curve.size = str(body), size
            curve.align_x = "CENTER" if centered else "LEFT"
            obj = bpy.data.objects.new("Label text", curve)
            scene.collection.objects.link(obj)
            obj.location = (x, y, 0.004)
            curve.materials.append(material)
            _select([obj])
            bpy.ops.object.convert(target="MESH")
            created.append(bpy.context.object)
            return bpy.context.object

        title = fields.get("text", fields.get("title", fields.get("brand", "")))
        if title:
            if fields.get("checker"):
                sources.append(plane("Text strip", aspect * 0.03, 0.34, aspect * 0.94, 0.32, 0.003, white))
            sources.append(
                text(
                    title,
                    aspect / 2,
                    0.4 if fields.get("checker") else 0.78,
                    min(0.25, aspect * 1.4 / max(len(str(title)), 1)),
                    centered=True,
                )
            )
        if fields.get("zone"):
            zone = pbr_material("Zone", fields.get("zone_color", "#D32F2F"), 1, 0)
            sources.append(plane("Zone", 0.025 * aspect, 0.7, 0.22 * aspect, 0.27, 0.003, zone))
            sources.append(text(fields["zone"], 0.13 * aspect, 0.75, 0.18, white, centered=True))
        lines = fields.get("lines", [])
        if isinstance(lines, str):
            lines = lines.splitlines()
        if fields.get("tracking"):
            lines = [*lines, fields["tracking"]]
        for index, line in enumerate(lines):
            sources.append(
                text(
                    line,
                    aspect * 0.035,
                    0.58 - index * 0.08,
                    min(0.06, aspect * 1.5 / max(len(str(line)), 1)),
                )
            )
        if fields.get("barcode"):
            import random

            rng = random.Random(str(fields.get("seed", fields["barcode"])))
            x = 0.03 * aspect
            while x < aspect * 0.96:
                w = rng.randint(1, 3) * aspect / 240
                sources.append(plane("Bar", x, 0.035, w, 0.18, 0.003, black))
                x += w + rng.randint(1, 3) * aspect / 240
        # Use one source mesh so the bake finds the nearest ink/paper surface.
        sources = [join_objects(sources, "Combined label source")]
        source_mesh = sources[0].data
        source_uv = source_mesh.uv_layers.new(name="LabelSourceUV")
        for loop in source_mesh.loops:
            vertex = source_mesh.vertices[loop.vertex_index].co
            source_uv.data[loop.index].uv = (vertex.x / aspect, vertex.y)
        target = plane("Label target", 0, 0, aspect, 1, 0, pbr_material("Bake target", "#FFFFFF", 1, 0))
        layer = target.data.uv_layers.new(name="UVMap")
        for uv, coord in zip(layer.data, ((0, 0), (1, 0), (1, 1), (0, 1)), strict=True):
            uv.uv = coord
        image = bpy.data.images.new("Label bake", width=width, height=height, float_buffer=True)
        image.colorspace_settings.name = "Linear Rec.709"
        node = target.data.materials[0].node_tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        target.data.materials[0].node_tree.nodes.active = node
        _select([target, *sources])
        bpy.ops.object.bake(
            type="DIFFUSE",
            pass_filter={"COLOR"},
            use_selected_to_active=True,
            cage_extrusion=0.01,
            max_ray_distance=0.02,
            use_clear=True,
            margin=0,
        )
        _save_linear_png(image, path, scene)
        bpy.data.images.remove(image)
    finally:
        bpy.context.window.scene = original_scene
        for obj in list(scene.objects):
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        bpy.data.scenes.remove(scene)


def make_label_png(path, width_px, height_px, fields):
    """Use Pillow from an installed external Python; fall back to Blender text.

    fields is a dict; see label_png.py for text, checker, barcode and zone keys.
    Random bars are deterministic for a given seed. Brands are fictional only.
    """
    if not isinstance(fields, dict) or min(width_px, height_px) < 1:
        raise ValueError("fields must be a dict; label dimensions must be positive")
    if fields.get("brand") and fields["brand"] not in FICTIONAL_BRANDS:
        raise ValueError("Only the section 1.10 fictional brands are permitted")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("label_png.py")
    candidates = [shutil.which("python3"), "/usr/bin/python3", str(VALIDATOR_PYTHON)]
    for python in dict.fromkeys(candidate for candidate in candidates if candidate):
        if not Path(python).is_file():
            continue
        probe = subprocess.run(
            [python, "-B", str(worker), "--probe"], capture_output=True, text=True, check=False
        )
        if probe.returncode == 0:
            payload = {"path": str(path), "width": width_px, "height": height_px, "fields": fields}
            result = subprocess.run(
                [python, "-B", str(worker)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise RuntimeError("Pillow label generation failed: " + result.stderr)
            _optimize_png(path)
            print("LABEL_BACKEND", python)
            return path
    print("LABEL_BACKEND Blender text selected-to-active diffuse bake")
    _label_blender(path, width_px, height_px, fields)
    return path
