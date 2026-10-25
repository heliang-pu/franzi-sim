"""Build only the three benchmark_props.md section 3.1 props, in millimetres.

Run: Blender -b --factory-startup -P tools/blender/build_task01_tote_transfer.py
     -- --props small_tote flange_part abs_housing
The shared pipeline supplies UV packing, 64-sample CPU baking, OBJ export,
convex collision export, metadata validation and three-point preview lighting.
"""

import argparse
import math
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bmesh
import bpy
from mathutils import Matrix

import prop_lib as props

OUTPUT = props.ROOT / "assets" / "props"
DRAFT = math.tan(math.radians(2))
WALL = 3 / math.cos(math.radians(2))


def mesh_object(name, vertices_mm, faces, material=None, bevel=0, segments=3):
    return props._mesh_object(name, vertices_mm, faces, (0, 0, 0), bevel, segments, material)


def rounded_rectangle(width, length, radius, segments=8):
    points = []
    for cx, cy, start in (
        (width / 2 - radius, length / 2 - radius, 0),
        (-width / 2 + radius, length / 2 - radius, 90),
        (-width / 2 + radius, -length / 2 + radius, 180),
        (width / 2 - radius, -length / 2 + radius, 270),
    ):
        for step in range(segments + 1):
            theta = math.radians(start + 90 * step / segments)
            points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return points


def loft(name, rings, material=None, cap=True, bevel=0):
    """Each ring is a list of XYZ millimetres, with a common CCW ordering."""
    n = len(rings[0])
    assert all(len(ring) == n for ring in rings)
    vertices = [v for ring in rings for v in ring]
    faces = []
    for row in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            faces.append((row * n + i, row * n + j, (row + 1) * n + j, (row + 1) * n + i))
    if cap:
        faces += [tuple(reversed(range(n))), tuple(range((len(rings) - 1) * n, len(rings) * n))]
    return mesh_object(name, vertices, faces, material, bevel)


def rr_ring(width, length, radius, z, segments=8):
    return [(x, y, z) for x, y in rounded_rectangle(width, length, radius, segments)]


def rr_prism(name, width, length, height, radius, location=(0, 0, 0), material=None, bevel=0):
    return props.extruded_profile(
        name, rounded_rectangle(width, length, radius), height,
        location_mm=location, material=material, bevel_mm=bevel, segments=3,
    )


def revolve(name, profile, sectors=64, material=None):
    """Revolve a closed (radius, Z) section; axis poles are single vertices."""
    vertices, rings, faces = [], [], []
    for radius, z in profile:
        if radius == 0:
            rings.append([len(vertices)])
            vertices.append((0, 0, z))
        else:
            rings.append(list(range(len(vertices), len(vertices) + sectors)))
            vertices.extend(
                (radius * math.cos(2 * math.pi * i / sectors),
                 radius * math.sin(2 * math.pi * i / sectors), z)
                for i in range(sectors)
            )
    for a, b in zip(rings, rings[1:] + rings[:1]):
        if len(a) == len(b) == 1:
            continue
        for i in range(sectors):
            j = (i + 1) % sectors
            if len(a) == 1:
                faces.append((a[0], b[j], b[i]))
            elif len(b) == 1:
                faces.append((a[i], a[j], b[0]))
            else:
                faces.append((a[i], a[j], b[j], b[i]))
    return mesh_object(name, vertices, faces, material)


def boolean(target, cutter, operation="DIFFERENCE"):
    # Boolean-generated cut surfaces must inherit the body's PBR, rather than
    # acquiring an empty material slot that Cycles cannot bake.
    if not cutter.data.materials and target.data.materials:
        cutter.data.materials.append(target.data.materials[0])
    props._select([target])
    modifier = target.modifiers.new(operation + " " + cutter.name, "BOOLEAN")
    modifier.operation, modifier.solver, modifier.object = operation, "EXACT", cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def bevel_edges(obj, width, segments=3, angle=30):
    props._select([obj])
    modifier = obj.modifiers.new("Soft manufactured edges", "BEVEL")
    modifier.width, modifier.segments = props.mm(width), segments
    modifier.limit_method, modifier.angle_limit = "ANGLE", math.radians(angle)
    modifier.harden_normals = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def finish_mesh(obj, weighted_normals=False):
    """Clean at the OBJ writer's 1-micrometre coordinate precision."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # Blender's OBJ writer rounds positions to six decimal places in metres.
    # Exact Boolean/bevel intersections can leave sub-micrometre slivers that
    # otherwise collapse only AFTER export, producing zero normals/triangles.
    for vertex in bm.verts:
        vertex.co = tuple(round(v, 6) for v in vertex.co)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=2e-7)
    bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=2e-7)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    seen, bad = set(), []
    for face in bm.faces:
        key = frozenset(face.verts)
        if key in seen or face.calc_area() <= 2e-18:
            bad.append(face)
        else:
            seen.add(key)
    if bad:
        bmesh.ops.delete(bm, geom=bad, context="FACES_ONLY")
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    props.smooth_by_angle(obj, 30)
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    old_materials = list(obj.data.materials)
    materials = list(dict.fromkeys(m for m in old_materials if m is not None))
    assignments = [materials.index(old_materials[p.material_index])
                   if old_materials[p.material_index] is not None else 0 for p in obj.data.polygons]
    obj.data.materials.clear()
    for material in materials:
        obj.data.materials.append(material)
    for polygon, index in zip(obj.data.polygons, assignments, strict=True):
        polygon.material_index = index
    if weighted_normals:
        # Preserve broad machined/moulded planes next to small bevels. The
        # sharp-edge attribute still supplies the prescribed 30-degree split.
        normal = obj.modifiers.new("Area-weighted manufacturing normals", "WEIGHTED_NORMAL")
        normal.keep_sharp = True
        normal.mode = "FACE_AREA"
        normal.weight = 50
        bpy.ops.object.modifier_apply(modifier=normal.name)
    print("TASK01_CLEAN_MESH", obj.name, "triangles", len(obj.data.polygons), flush=True)


def manufacturing_normals(obj, turned=False):
    """Exact plane/radial corner normals remove Boolean triangulation shading.

    Area weighting alone still tilts normals on small cap triangles near bores.
    The flange's cylindrical/chamfer faces are rotational surfaces about the
    main axis or a bolt-hole axis; the ABS enclosure needs only flat planes.
    """
    mesh = obj.data
    mesh.update()
    normals = [tuple(n.vector) for n in mesh.corner_normals]
    axes = [(0.0, 0.0)] + [(props.mm(x), props.mm(y))
                           for x, y in ((22.5, 0), (0, 22.5), (-22.5, 0), (0, -22.5))]
    for face in mesh.polygons:
        if max(abs(v) for v in face.normal) > 0.999999 and (not turned or abs(face.normal.z) > 0.999999):
            for i in face.loop_indices:
                normals[i] = tuple(face.normal)
        elif turned:
            cx, cy = axes[0]
            for x, y in axes[1:]:
                if math.hypot(face.center.x - x, face.center.y - y) < props.mm(4.1):
                    cx, cy = x, y
                    break
            rx, ry = face.center.x - cx, face.center.y - cy
            sign = 1 if rx * face.normal.x + ry * face.normal.y > 0 else -1
            nz = float(face.normal.z)
            radial = sign * math.sqrt(max(0, 1 - nz * nz))
            for i in face.loop_indices:
                vertex = mesh.vertices[mesh.loops[i].vertex_index].co
                x, y = vertex.x - cx, vertex.y - cy
                length = math.hypot(x, y)
                normals[i] = (radial * x / length, radial * y / length, nz)
    mesh.normals_split_custom_set(normals)
    mesh.update()


def math_node(tree, operation, *inputs):
    return props._math(tree, operation, *inputs)


def multiply_color(tree, color, factor, name):
    node = tree.nodes.new("ShaderNodeMixRGB")
    node.name = name
    node.blend_type = "MULTIPLY"
    node.inputs[0].default_value = 1
    tree.links.new(color, node.inputs[1])
    tree.links.new(factor, node.inputs[2])
    return node.outputs[0]


def plastic_material(name, hex_color, roughness, leather=False, tote=False):
    material = props.pbr_material(name, hex_color, roughness, 0,
                                  wear=0 if leather else 0.045, grime=0.15)
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    color = bsdf.inputs["Base Color"].links[0].from_socket
    coords = tree.nodes.new("ShaderNodeTexCoord")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.name = "Fine moulded leather grain" if leather else "Fine injection-mould texture"
    noise.inputs["Scale"].default_value = 6800 if leather else 3200
    noise.inputs["Detail"].default_value = 3
    noise.inputs["Roughness"].default_value = 0.7
    tree.links.new(coords.outputs["Object"], noise.inputs["Vector"])
    variation = math_node(tree, "ADD", 0.94, math_node(tree, "MULTIPLY", noise.outputs["Fac"], 0.12))
    color = multiply_color(tree, color, variation, "Small grain colour variation")
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.22 if leather else 0.12
    bump.inputs["Distance"].default_value = props.mm(0.035 if leather else 0.015)
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    if leather:
        # Pointiness is vertex-interpolated and can spread a worn edge across
        # an entire triangulated cap. Use the actual R4 enclosure contour for
        # a 1.5 mm wear band, independent of triangulation and vertex valence.
        xyz = tree.nodes.new("ShaderNodeSeparateXYZ")
        tree.links.new(coords.outputs["Object"], xyz.inputs[0])
        qx = math_node(tree, "SUBTRACT", math_node(tree, "ABSOLUTE", xyz.outputs["X"]), props.mm(18.5))
        qy = math_node(tree, "SUBTRACT", math_node(tree, "ABSOLUTE", xyz.outputs["Y"]), props.mm(31))
        ex, ey = math_node(tree, "MAXIMUM", qx, 0), math_node(tree, "MAXIMUM", qy, 0)
        outside = math_node(tree, "SQRT", math_node(tree, "ADD",
                            math_node(tree, "MULTIPLY", ex, ex), math_node(tree, "MULTIPLY", ey, ey)))
        inside = math_node(tree, "MINIMUM", math_node(tree, "MAXIMUM", qx, qy), 0)
        distance = math_node(tree, "SUBTRACT", props.mm(4), math_node(tree, "ADD", outside, inside))
        contour = math_node(tree, "MINIMUM", 1, math_node(tree, "MAXIMUM", 0,
                            math_node(tree, "SUBTRACT", 1,
                            math_node(tree, "DIVIDE", distance, props.mm(1.5)))))
        end_distance = math_node(tree, "MINIMUM", xyz.outputs["Z"],
                                 math_node(tree, "SUBTRACT", props.mm(30), xyz.outputs["Z"]))
        end = math_node(tree, "MINIMUM", 1, math_node(tree, "MAXIMUM", 0,
                        math_node(tree, "SUBTRACT", 1,
                        math_node(tree, "DIVIDE", end_distance, props.mm(1.5)))))
        shade = math_node(tree, "ADD", 1, math_node(tree, "MULTIPLY", 0.18,
                          math_node(tree, "MULTIPLY", contour, end)))
        color = multiply_color(tree, color, shade, "1.5 mm worn perimeter edges")
    if tote:
        separate = tree.nodes.new("ShaderNodeSeparateXYZ")
        tree.links.new(coords.outputs["Object"], separate.inputs[0])
        bottom = math_node(tree, "MAXIMUM", 0, math_node(tree, "SUBTRACT", 1,
                           math_node(tree, "DIVIDE", separate.outputs["Z"], props.mm(20))))
        broad = tree.nodes.new("ShaderNodeTexNoise")
        broad.name = "Handling smudges and accumulated bottom grime"
        broad.inputs["Scale"].default_value = 140
        broad.inputs["Detail"].default_value = 2
        tree.links.new(coords.outputs["Object"], broad.inputs["Vector"])
        grime = math_node(tree, "SUBTRACT", 1, math_node(tree, "MULTIPLY", bottom,
                          math_node(tree, "ADD", 0.10,
                          math_node(tree, "MULTIPLY", broad.outputs["Fac"], 0.12))))
        color = multiply_color(tree, color, grime, "Dirty lower 20 mm")
        stretched = tree.nodes.new("ShaderNodeVectorMath")
        stretched.operation = "MULTIPLY"
        stretched.inputs[1].default_value = (180, 10000, 7000)
        tree.links.new(coords.outputs["Object"], stretched.inputs[0])
        scratches = tree.nodes.new("ShaderNodeTexNoise")
        scratches.name = "Faint directional scuff scratches"
        scratches.inputs["Scale"].default_value = 1
        scratches.inputs["Detail"].default_value = 2
        tree.links.new(stretched.outputs["Vector"], scratches.inputs["Vector"])
        scratch = math_node(tree, "MAXIMUM", 0,
                            math_node(tree, "SUBTRACT", scratches.outputs["Fac"], 0.64))
        color = multiply_color(tree, color, math_node(tree, "ADD", 1,
                               math_node(tree, "MULTIPLY", scratch, 0.75)), "Light surface scratches")
        distance = tree.nodes.new("ShaderNodeVectorMath")
        distance.operation = "DISTANCE"
        tree.links.new(coords.outputs["Object"], distance.inputs[0])
        distance.inputs[1].default_value = (0, props.mm(143), props.mm(85))
        handle = math_node(tree, "MAXIMUM", 0, math_node(tree, "SUBTRACT", 1,
                           math_node(tree, "DIVIDE", distance.outputs["Value"], props.mm(52))))
        fingerprint = math_node(tree, "SUBTRACT", 1, math_node(tree, "MULTIPLY", handle,
                                math_node(tree, "MULTIPLY", broad.outputs["Fac"], 0.16)))
        color = multiply_color(tree, color, fingerprint, "Smudged handle grip")
    tree.links.new(color, bsdf.inputs["Base Color"])
    return material


def move_mesh(obj, location_mm, rotation_deg=(0, 0, 0)):
    obj.location = [props.mm(v) for v in location_mm]
    obj.rotation_euler = [math.radians(v) for v in rotation_deg]
    props._select([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def start_prop(prop_id):
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    folder = OUTPUT / prop_id
    folder.mkdir(parents=True, exist_ok=True)
    print("TASK01_START", prop_id, flush=True)
    return folder


def complete_prop(prop_id, folder, body, collision, size, mass, friction, material, notes,
                  dimensions, variants=None, extra_previews=False):
    collision_files = props.export_collision("body", collision, folder)
    collection = bpy.data.collections.new("Collision proxies (hidden; prop frame)")
    bpy.context.scene.collection.children.link(collection)
    for obj in collision:
        for previous in list(obj.users_collection):
            previous.objects.unlink(obj)
        collection.objects.link(obj)
        obj.hide_render = True
        obj.hide_set(True)
    body.name = "body"
    props.export_obj(body, folder / "body.obj")
    props.render_preview([body], folder / "preview.png")
    if extra_previews:
        back = body.copy()
        back.name = "Temporary reverse view"
        back.matrix_world = Matrix.Rotation(math.pi, 4, "Z") @ body.matrix_world
        bpy.context.collection.objects.link(back)
        props.render_preview([back], folder / "preview_label.png")
        bpy.data.objects.remove(back, do_unlink=True)
    metadata = {
        "id": prop_id, "task": "tote_transfer", "units": "m", "origin": "bottom_center",
        "bbox_mm": [round(v, 5) for v in props.bbox_mm([body])],
        "budget_class": "medium" if prop_id == "small_tote" else "small",
        "triangles": props.triangle_count([body]), "source": "source.blend", "preview": "preview.png",
        "parts": [{
            "name": "body", "visual": "body.obj", "texture": "body_albedo.png", "texture_px": size,
            "collision": collision_files, "mass_kg": mass, "friction": friction,
            "mujoco_material": material, "parent": None, "joint": None,
        }],
        "variants": variants or [], "dimensions_mm": dimensions,
        "recommended_placement_m": {"x": 0.4, "y": [-0.2, 0.2], "support_z": 0.8},
        "notes": notes + " All coordinates are metres with Z up, front -X, common bottom-centre origin. "
        "Visual export is triangulated with normals and a nonoverlapping Smart UV 66-degree atlas. "
        "Albedo: 64-sample CPU Cycles diffuse colour * (0.4 + 0.6 * AO), linear light to 8-bit sRGB; "
        "no direct light or highlights. Full procedural Principled PBR is retained in source.blend. "
        f"Generated with Blender {bpy.app.version_string}.",
    }
    bpy.context.scene["prop_id"] = prop_id
    bpy.context.scene["bake_recipe"] = "64 samples CPU; DIFFUSE COLOR * mix(1, AO, 0.6); 8-bit sRGB"
    bpy.context.scene["dimensions_mm"] = str(dimensions)
    bpy.context.preferences.filepaths.save_version = 0
    props._select([body])
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    props.write_prop_json(metadata, folder / "prop.json")
    print("TASK01_COMPLETE", prop_id, "bbox", metadata["bbox_mm"], "triangles", metadata["triangles"],
          "collision", len(collision_files), flush=True)


def build_small_tote():
    folder = start_prop("small_tote")
    pp = plastic_material("PP blue - lightly scuffed injection moulding", "#1F5FA8", 0.55, tote=True)
    # The 200 x 300 envelope includes a 6 mm OUTWARD rim overhang. Thus the
    # wall's top envelope is 188 x 288, not 200 x 300 (see metadata notes).
    outer = lambda z: (188 - 2 * (100 - z) * DRAFT, 288 - 2 * (100 - z) * DRAFT)
    bottom = outer(0)
    at_floor = outer(4)
    under_rim = outer(96)
    rings = [
        rr_ring(*bottom, 4, 0), rr_ring(*under_rim, 4, 96),
        rr_ring(200, 300, 7, 96), rr_ring(200, 300, 7, 100),
        rr_ring(188 - 2 * WALL, 288 - 2 * WALL, 1, 100),
        rr_ring(at_floor[0] - 2 * WALL, at_floor[1] - 2 * WALL, 1, 4),
    ]
    body = loft("Tapered hollow tote and overhanging rim", rings, pp, bevel=0.65)

    # Four longitudinal ribs per long wall, two per short wall, bonded into
    # the shell. The tops stop below the rim; every rib stays inside its bbox.
    for side in (-1, 1):
        for y in (-105, -35, 35, 105):
            rib_rings = []
            for z in (10, 90):
                face_x = outer(z)[0] / 2
                x0, x1 = sorted((side * (face_x - 0.5), side * (face_x + 5)))
                rib_rings.append([(x0, y - 1.5, z), (x1, y - 1.5, z),
                                  (x1, y + 1.5, z), (x0, y + 1.5, z)])
            boolean(body, loft("Long-side vertical rib", rib_rings, pp, bevel=0.5), "UNION")
        for x in (-60, 60):
            rib_rings = []
            for z in (10, 90):
                face_y = outer(z)[1] / 2
                y0, y1 = sorted((side * (face_y - 0.5), side * (face_y + 5)))
                rib_rings.append([(x - 1.5, y0, z), (x + 1.5, y0, z),
                                  (x + 1.5, y1, z), (x - 1.5, y1, z)])
            boolean(body, loft("Short-side vertical rib", rib_rings, pp, bevel=0.5), "UNION")

    handle = rr_prism("80 x 25 R10 handle cutter", 80, 25, 35, 10)
    move_mesh(handle, (0, 142, 72.5), (90, 0, 0))
    boolean(body, handle)
    bevel_edges(body, 0.32, angle=45)

    # A 4 mm nominal floor, with a 3 mm deep underside waffle pattern.
    for x in (-63, -21, 21, 63):
        for y in (-115, -69, -23, 23, 69, 115):
            pocket = rr_prism("Underside 3 mm waffle recess", 34, 38, 4, 2,
                              location=(x, y, 1), bevel=0.3)
            boolean(body, pocket)

    # The label is on the OTHER short wall (-Y). The recess and card follow
    # the exact 2-degree wall draft instead of floating in front of the shell.
    label_z = 54
    label_y = -outer(label_z)[1] / 2
    slot = props.box("75 x 35, 1 mm recessed card slot", (75, 6, 35),
                     location_mm=(0, label_y - 2, label_z), rotation_deg=(2, 0, 0), bevel_mm=0.35)
    boolean(body, slot)
    for x in (-37.6, 37.6):
        rail = props.box("Label retaining rail", (1.6, 2, 35),
                         location_mm=(x, label_y - 0.2, label_z), rotation_deg=(2, 0, 0),
                         bevel_mm=0.4, material=pp)
        boolean(body, rail, "UNION")
    rail = props.box("Label lower retaining rail", (75, 2, 1.6),
                     location_mm=(0, label_y + 0.4, label_z - 17), rotation_deg=(2, 0, 0),
                     bevel_mm=0.4, material=pp)
    boolean(body, rail, "UNION")

    label_path = props.make_label_png(folder / "label_card_source.png", 1200, 540,
                                     {"text": "BIN-A07", "background": "#FAFAF5",
                                      "lines": ["NOVA Logistics", "COMPONENT STORAGE / A07"],
                                      "barcode": "BIN-A07", "seed": "BIN-A07"})
    paper = props.pbr_material("White BIN-A07 card", "#FAFAF5", 0.8, 0, grime=0.03)
    card = props.box("White card fitted inside recessed slot", (72, 0.25, 32),
                     location_mm=(0, label_y + 0.67, label_z + 0.0234), rotation_deg=(2, 0, 0),
                     bevel_mm=0.1, material=paper)
    label_uv = card.data.uv_layers.new(name="LabelUV")
    for loop in card.data.loops:
        v = card.data.vertices[loop.vertex_index].co
        local_height = (v.z - props.mm(label_z + 0.0234)) * math.cos(math.radians(2))
        local_height -= (v.y - props.mm(label_y + 0.67)) * math.sin(math.radians(2))
        # Looking at -Y, image right is +X.
        label_uv.data[loop.index].uv = (v.x / props.mm(72) + 0.5, local_height / props.mm(32) + 0.5)
    tree = paper.node_tree
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "LabelUV"
    image = tree.nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(label_path))
    image.extension = "EXTEND"
    tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
    rgb = next(n for n in tree.nodes if n.type == "RGB")
    for destination in [link.to_socket for link in rgb.outputs[0].links]:
        tree.links.new(image.outputs["Color"], destination)
    body = props.join_objects([body, card], "body")
    finish_mesh(body)
    props.uv_unwrap(body)
    props.bake_albedo([body], 2048, folder / "body_albedo.png")
    rgb = next(n for n in pp.node_tree.nodes if n.type == "RGB")
    rgb.outputs[0].default_value = props._linear_color("#9AA0A6")
    pp.diffuse_color = props._linear_color("#9AA0A6")
    props.bake_albedo([body], 2048, folder / "body_albedo_v1.png")
    props.render_preview([body], folder / "preview_grey.png")
    rgb.outputs[0].default_value = props._linear_color("#1F5FA8")
    pp.diffuse_color = props._linear_color("#1F5FA8")

    collision = [props.box("Floor collision", (*outer(4), 4), location_mm=(0, 0, 2), bevel_mm=0)]
    for side in (-1, 1):
        for axis in (0, 1):
            wall_rings = []
            for z in (4, 100):
                ox, oy = (v / 2 for v in outer(z))
                if axis == 0:
                    x0, x1 = sorted((side * (ox - WALL), side * ox))
                    wall_rings.append([(x0, -oy, z), (x1, -oy, z), (x1, oy, z), (x0, oy, z)])
                else:
                    y0, y1 = sorted((side * (oy - WALL), side * oy))
                    wall_rings.append([(-ox, y0, z), (ox, y0, z), (ox, y1, z), (-ox, y1, z)])
            collision.append(loft("Drafted wall collision", wall_rings))
    collision.extend([
        props.box("Rim X-", (6.2, 300, 4), (-96.9, 0, 98), bevel_mm=0),
        props.box("Rim X+", (6.2, 300, 4), (96.9, 0, 98), bevel_mm=0),
        props.box("Rim Y-", (200, 6.2, 4), (0, -146.9, 98), bevel_mm=0),
        props.box("Rim Y+", (200, 6.2, 4), (0, 146.9, 98), bevel_mm=0),
    ])
    complete_prop(
        "small_tote", folder, body, collision, 2048, 0.45, 0.40,
        {"specular": 0.25, "shininess": 0.30, "reflectance": 0.0},
        "Spec ambiguity resolved in favour of the exact 200 X x 300 Y x 100 Z mm OUTER rim envelope, "
        "2-degree draft, 3 mm normal wall thickness, and a 6 mm outward rim overhang (4 mm thick). "
        "These jointly imply a 188 x 288 mm wall top and approximately 181.016 x 281.016 mm wall base; "
        "the spec's approximate 193 x 293 mm base cannot coexist with an outward 6 mm rim inside that envelope. "
        "Ribs: four on EACH long side and two on EACH short side, 3 mm wide and 5 mm projection. "
        "Handle is a real 80 x 25 R10 through-cut on +Y; its nominal upper edge is Z85 (15 below top). "
        "Opposite -Y card pocket is 75 x 35, 1 mm deep, with a fitted white BIN-A07 card and retaining rails. "
        "Nominal 4 mm floor has 3 mm underside waffle recesses, leaving a 1 mm skin in each recess. "
        "Task-prescribed collision is nine separate convex pieces: floor, four drafted walls, four rim bars. "
        "It intentionally fills the handle and underside recesses and omits narrow ribs and label details, "
        "so the general 1 mm surface approximation rule cannot apply to these explicitly simplified features. "
        "Blue/grey are texture-only variants on the identical mesh and UV atlas; label colours stay fixed. "
        "preview_label.png shows the opposite short side; preview_grey.png shows the grey variant. "
        "Scene placement is metadata only; the robot should transfer the two small parts, not grasp the 200 mm bin.",
        {"outside_xyz": [200, 300, 100], "wall_top_xy": [188, 288], "wall_bottom_xy": list(bottom),
         "draft_degrees": 2, "wall_normal_thickness": 3, "floor_nominal": 4, "underside_recess": 3,
         "rim_outward_width": 6, "rim_thickness": 4, "rib_width_projection": [3, 5],
         "ribs_long_side_each": 4, "ribs_short_side_each": 2,
         "handle_width_height_radius": [80, 25, 10], "handle_top_z": 85,
         "label_slot_width_height_depth": [75, 35, 1]},
        variants=[{"name": "blue", "textures": {"body": "body_albedo.png"}},
                  {"name": "grey", "textures": {"body": "body_albedo_v1.png"}}],
        extra_previews=True,
    )


def build_flange_part():
    folder = start_prop("flange_part")
    top = props.pbr_material("6061 aluminium - concentric face turning", "#C9CDD1", 0.30, 1,
                             wear=0.015, grime=0.10, machining="turned")
    # A 0.35 mm feed remains legible in the mandated 1024px atlas and preview.
    for node in top.node_tree.nodes:
        if node.type == "GROUP" and node.node_tree.name.startswith("Machining: turned"):
            for operation in node.node_tree.nodes:
                if operation.type != "MATH":
                    continue
                for socket in operation.inputs[:2]:
                    if socket.is_linked:
                        continue
                    if math.isclose(socket.default_value, 2 * math.pi / props.mm(0.18), rel_tol=1e-5):
                        socket.default_value = 2 * math.pi / props.mm(0.35)
    side = props.pbr_material("6061 aluminium - fine cylindrical turning", "#C9CDD1", 0.30, 1, grime=0.10)
    tree = side.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    color = bsdf.inputs["Base Color"].links[0].from_socket
    coords = tree.nodes.new("ShaderNodeTexCoord")
    xyz = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(coords.outputs["Object"], xyz.inputs[0])
    marks = math_node(tree, "SINE", math_node(tree, "MULTIPLY", xyz.outputs["Z"], 2 * math.pi / props.mm(0.12)))
    shade = math_node(tree, "ADD", 0.975, math_node(tree, "MULTIPLY", marks, 0.025))
    tree.links.new(multiply_color(tree, color, shade, "Fine circumferential side tool marks"),
                   bsdf.inputs["Base Color"])
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.10
    bump.inputs["Distance"].default_value = props.mm(0.008)
    tree.links.new(marks, bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # Exact 1 x 45 outer chamfers, 0.5 bore-mouth chamfers, continuous hub.
    body = revolve("Flange with integral hub and through bore", [
        (29, 0), (30, 1), (30, 11), (29, 12), (16, 12), (16, 19.6),
        (15.6, 20), (10.5, 20), (10, 19.5), (10, 0.5), (10.5, 0),
    ], sectors=96, material=top)
    for x, y in ((22.5, 0), (0, 22.5), (-22.5, 0), (0, -22.5)):
        cutter = revolve("6.6 mm bolt hole, 0.5 mm entry chamfers", [
            (0, -1), (3.8, -1), (3.8, 0), (3.3, 0.5), (3.3, 11.5), (3.8, 12), (3.8, 13), (0, 13),
        ], sectors=48)
        move_mesh(cutter, (x, y, 0))
        boolean(body, cutter)
    bevel_edges(body, 0.3, segments=2, angle=60)
    finish_mesh(body, weighted_normals=True)
    body.data.materials.append(side)
    for polygon in body.data.polygons:
        if abs(polygon.normal.z) < 0.25:
            polygon.material_index = 1
    manufacturing_normals(body, turned=True)
    props.uv_unwrap(body)
    props.bake_albedo([body], 1024, folder / "body_albedo.png")
    collision = [
        props.cylinder("Nominal solid disc collision", 30, 12, (0, 0, 6), bevel_mm=0, vertices=32),
        props.cylinder("Nominal solid hub collision", 16, 8, (0, 0, 16), bevel_mm=0, vertices=32),
    ]
    complete_prop(
        "flange_part", folder, body, collision, 1024, 0.090, 0.35,
        {"specular": 0.70, "shininess": 0.70, "reflectance": 0.05},
        "Integral 6061 turned aluminium flange: disc diameter 60 x 12, hub diameter 32 x 8, total height 20. "
        "A real diameter-20 through bore and four real diameter-6.6 through holes on a diameter-45 pitch circle. "
        "Disc outside top and bottom edges have exact 1 x 45-degree chamfers; every bore mouth has a 0.5 mm "
        "chamfer. Hub top has a 0.4 mm chamfer and hub-root corner a 0.3 mm two-segment blend. "
        "Concentric 0.35 mm feed face tool marks and 0.12 mm pitch side turning survive in albedo and source PBR bump. "
        "Task-prescribed collision consists of two 32-sided solid cylinders; all bores are deliberately omitted. "
        "Mass is the specified benchmark mass, not inferred from geometric aluminium density. "
        "The 60 mm grasp diameter is within the 95 mm gripper opening.",
        {"disc_diameter": 60, "disc_height": 12, "hub_diameter": 32, "hub_height": 8, "bore_diameter": 20,
         "bolt_hole_count": 4, "bolt_hole_diameter": 6.6, "bolt_pitch_circle_diameter": 45,
         "outside_chamfer": 1, "outside_chamfer_degrees": 45, "hole_mouth_chamfer": 0.5},
    )


def add_indicator_window(material):
    """Paint a rounded 20 x 8 mm window on -X: no coplanar decal geometry."""
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    color = bsdf.inputs["Base Color"].links[0].from_socket
    coords = tree.nodes.new("ShaderNodeTexCoord")
    xyz = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(coords.outputs["Object"], xyz.inputs[0])
    y = math_node(tree, "ABSOLUTE", xyz.outputs["Y"])
    z = math_node(tree, "ABSOLUTE", math_node(tree, "SUBTRACT", xyz.outputs["Z"], props.mm(16)))
    front = math_node(tree, "LESS_THAN", xyz.outputs["X"], props.mm(-22.3))

    def rounded_mask(half_width, half_height, radius):
        dy = math_node(tree, "MAXIMUM", 0, math_node(tree, "SUBTRACT", y, props.mm(half_width - radius)))
        dz = math_node(tree, "MAXIMUM", 0, math_node(tree, "SUBTRACT", z, props.mm(half_height - radius)))
        distance2 = math_node(tree, "ADD", math_node(tree, "MULTIPLY", dy, dy),
                              math_node(tree, "MULTIPLY", dz, dz))
        return math_node(tree, "MULTIPLY", front,
                         math_node(tree, "LESS_THAN", distance2, props.mm(radius) ** 2))

    for mask, hex_color, name in ((rounded_mask(10, 4, 1), "#11181B", "Window perimeter"),
                                  (rounded_mask(9.2, 3.2, 0.6), "#194650", "Tinted indicator window")):
        node = tree.nodes.new("ShaderNodeMixRGB")
        node.name = name
        tree.links.new(mask, node.inputs[0])
        tree.links.new(color, node.inputs[1])
        node.inputs[2].default_value = props._linear_color(hex_color)
        color = node.outputs[0]
    dot_y = math_node(tree, "SUBTRACT", xyz.outputs["Y"], props.mm(5.5))
    dot_distance2 = math_node(tree, "ADD", math_node(tree, "MULTIPLY", dot_y, dot_y),
                             math_node(tree, "MULTIPLY", z, z))
    dot = math_node(tree, "MULTIPLY", front,
                    math_node(tree, "LESS_THAN", dot_distance2, props.mm(1.1) ** 2))
    node = tree.nodes.new("ShaderNodeMixRGB")
    node.name = "Printed indicator dot"
    tree.links.new(dot, node.inputs[0])
    tree.links.new(color, node.inputs[1])
    node.inputs[2].default_value = props._linear_color("#73C9AD")
    tree.links.new(node.outputs[0], bsdf.inputs["Base Color"])
    # The source PBR carries window gloss; MuJoCo receives the baked colour.
    window = rounded_mask(9.2, 3.2, 0.6)
    roughness = math_node(tree, "SUBTRACT", 0.45, math_node(tree, "MULTIPLY", window, 0.25))
    tree.links.new(roughness, bsdf.inputs["Roughness"])


def build_abs_housing():
    folder = start_prop("abs_housing")
    abs_mat = plastic_material("Dark grey ABS - fine leather grain", "#2B2D30", 0.45, leather=True)
    add_indicator_window(abs_mat)
    body = rr_prism("45 X x 70 Y R4 ABS enclosure", 45, 70, 30, 4,
                    location=(0, 0, 15), material=abs_mat, bevel=0.4)
    # A real circumferential shallow groove, centred exactly 2 mm below top.
    outer = rounded_rectangle(50, 75, 5)
    inner = rounded_rectangle(44.5, 69.5, 3.75)
    n = len(outer)
    rings = [[(x, y, z) for x, y in profile]
             for z, profile in ((27.85, outer), (28.15, outer), (28.15, inner), (27.85, inner))]
    seam = loft("0.3 mm parting-line ring cutter", rings, cap=False)
    # Close the radial ring section without closing its central hole.
    bm = bmesh.new()
    bm.from_mesh(seam.data)
    bm.verts.ensure_lookup_table()
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((bm.verts[3 * n + i], bm.verts[3 * n + j], bm.verts[j], bm.verts[i]))
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(seam.data)
    bm.free()
    boolean(body, seam)
    for x in (-15.5, 15.5):
        for y in (-28, 28):
            hole = revolve("3 mm blind screw hole with chamfer", [
                (0, 21.1), (1.5, 22), (1.5, 29.6), (1.9, 30), (1.9, 31), (0, 31),
            ], sectors=32)
            move_mesh(hole, (x, y, 0))
            boolean(body, hole)
    for side in (-1, 1):
        tab = props.box("8 x 3 x 1.5 snap projection", (1.9, 8, 3),
                        location_mm=(side * 23.05, 0, 10.5), bevel_mm=0.3, material=abs_mat)
        boolean(body, tab, "UNION")
    finish_mesh(body, weighted_normals=True)
    manufacturing_normals(body)
    props.uv_unwrap(body)
    props.bake_albedo([body], 1024, folder / "body_albedo.png")
    collision = [props.box("Nominal housing box collision", (45, 70, 30), (0, 0, 15), bevel_mm=0)]
    complete_prop(
        "abs_housing", folder, body, collision, 1024, 0.040, 0.40,
        {"specular": 0.30, "shininess": 0.40, "reflectance": 0.0},
        "Nominal enclosure body is 45 X x 70 Y x 30 Z mm, with R4 plan-view corners and 0.4 mm edge rounds. "
        "Two opposed side snap tabs are 8 mm along Y x 3 mm high x 1.5 mm outward projection; "
        "their 0.4 mm embedded roots are fused into the enclosure. Therefore overall visual bbox is 48 x 70 x 30, "
        "while the specified 45 x 70 x 30 describes the box itself. "
        "Real parting groove is 0.30 mm wide, 0.25 mm deep at Z28 (2 mm below top). "
        "Four diameter-3 blind screw bores at (+/-15.5, +/-28), 8 mm nominal depth before drill points "
        "and 0.4 mm entry chamfers; blind depth was unspecified and chosen for a faithful moulded housing. "
        "The rounded 20 x 8 indicator window on -X at Z16 is entirely baked colour, with gloss in source PBR. "
        "Fine leather grain, 1.5 mm worn perimeter bands, and cavity grime are procedural and baked. "
        "As prescribed by section 3.1, one nominal box collider ignores R4 corners, screw holes and snap details; "
        "this explicit simplification exceeds the general 1 mm approximation at rounded corners and tabs. "
        "48 mm total grasp width and 0.040 kg are within the task's one-hand workspace limits.",
        {"body_xyz": [45, 70, 30], "corner_radius": 4, "edge_round": 0.4,
         "parting_line_z": 28, "parting_line_width": 0.3, "parting_line_depth": 0.25,
         "screw_count": 4, "screw_diameter": 3, "screw_nominal_depth": 8,
         "screw_cylindrical_depth": 7.6, "screw_depth_to_tip": 8.9,
         "screw_xy": [[x, y] for x in (-15.5, 15.5) for y in (-28, 28)],
         "snap_count": 2, "snap_width_height_projection": [8, 3, 1.5], "indicator_window_yz": [20, 8]},
    )


def main():
    builders = {"small_tote": build_small_tote, "flange_part": build_flange_part,
                "abs_housing": build_abs_housing}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", choices=list(builders), default=list(builders))
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    for prop_id in dict.fromkeys(args.props):
        builders[prop_id]()


if __name__ == "__main__":
    try:
        main()
        props.finalize_assets()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
