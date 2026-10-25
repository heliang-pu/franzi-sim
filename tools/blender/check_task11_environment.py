"""Task 11 geometry, asset reuse, source integrity and MuJoCo support checks.

Run with the prescribed validator Python. --sources runs inside headless Blender.
Only Task 11 assets are read or written; the reused libraries are read-only.
"""

import argparse
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
OUTPUT = ROOT / "assets" / "props"
QA = HERE / "_qa" / "task11"
IDS = ("workbench", "factory_floor", "safety_fence", "background_rack")


def source_checks():
    import bpy

    results = {}
    for prop_id in IDS:
        bpy.ops.wm.open_mainfile(filepath=str(OUTPUT / prop_id / "source.blend"))
        data = json.loads((OUTPUT / prop_id / "prop.json").read_text())
        material_count = 0
        for part in data["parts"]:
            obj = bpy.data.objects[part["source_visual_object"]]
            assert all(abs(v) < 1e-7 for v in obj.location), (prop_id, obj.name, "location")
            assert all(abs(v) < 1e-7 for v in obj.rotation_euler), (prop_id, obj.name, "rotation")
            assert all(abs(v - 1) < 1e-7 for v in obj.scale), (prop_id, obj.name, "scale")
            assert obj.data.uv_layers.get("UVMap"), (prop_id, obj.name, "UV")
            assert not obj.hide_render
            for material in obj.data.materials:
                assert material is not None
                assert any(n.type == "BSDF_PRINCIPLED" for n in material.node_tree.nodes)
                material_count += 1
        files = [image for image in bpy.data.images if image.source == "FILE"]
        assert all(image.packed_file for image in files), (prop_id, "unpacked image")
        results[prop_id] = {"identity_transforms": True, "pbr_material_slots": material_count,
                            "packed_image_count": len(files)}
        if prop_id == "factory_floor":
            obj = bpy.data.objects["floor"]
            layer = obj.data.uv_layers["FloorTileUV"]
            for loop in obj.data.loops:
                v = obj.data.vertices[loop.vertex_index].co
                uv = layer.data[loop.index].uv
                assert abs(uv.x - (v.x + 3)) < 1e-6 and abs(uv.y - (v.y + 3)) < 1e-6
            node = obj.data.materials[0].node_tree.nodes["One metre epoxy tile"]
            assert node.extension == "REPEAT"
            results[prop_id]["one_metre_source_uv_mapping"] = True
    (QA / "source_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def geometry_checks():
    import numpy as np
    from PIL import Image
    from scipy.spatial import cKDTree
    from prop_schema import schema_errors
    from validate_props import read_obj, validate_prop

    results = {}
    metadata = {i: json.loads((OUTPUT / i / "prop.json").read_text()) for i in IDS}
    for prop_id in IDS:
        validated = validate_prop(OUTPUT / prop_id)
        assert not validated.errors, (prop_id, validated.errors)
        for part in metadata[prop_id]["parts"]:
            pixels = np.asarray(Image.open(OUTPUT / prop_id / part["texture"]))
            assert np.count_nonzero(pixels.max(axis=2)) > 10000, (prop_id, part["name"], "empty bake")
        results[prop_id] = {"validator": "PASS"}
    workbench = metadata["workbench"]
    tabletop = next(p for p in workbench["parts"] if p["name"] == "tabletop")
    assert sum(len(p["collision"]) for p in workbench["parts"]) == 1
    collision = read_obj(OUTPUT / "workbench" / tabletop["collision"][0]).vertices * 1000
    assert np.allclose(collision.min(0), [-350, -600, 758], atol=.001)
    assert np.allclose(collision.max(0), [350, 600, 800], atol=.001)
    board = read_obj(OUTPUT / "workbench" / "pegboard_tools.obj").mesh()
    components = board.split(only_watertight=False)
    sheets = [m for m in components if m.euler_number == 2 - 2 * 288 and m.is_watertight]
    assert len(sheets) == 1, [(len(m.faces), m.euler_number, m.is_watertight)
                              for m in sorted(components, key=lambda m: -len(m.faces))[:5]]
    results["workbench"].update({"collision_top_z_mm": float(collision[:, 2].max()),
                                 "pegboard_real_through_holes": 288,
                                 "pegboard_sheet_watertight": True})

    floor = metadata["factory_floor"]
    assert not floor["parts"][0]["collision"]
    primitives = floor["parts"][0]["collision_primitives"]
    assert len(primitives) == 1 and primitives[0]["type"] == "plane"
    assert primitives[0]["pos_m"] == [0, 0, 0] and primitives[0]["normal"] == [0, 0, 1]
    assert floor["dimensions_mm"]["substrate_repeat_xy"] == [1000, 1000]
    # Regression coverage for the new schema path; invalid analytic geometry
    # must never silently pass as an empty collision list.
    for field, value in (("normal", [0, 0, 0]), ("size_m", [3, -3, .01]),
                         ("pos_m", [0, 0, float("nan")]), ("type", "box")):
        bad = copy.deepcopy(floor)
        bad["parts"][0]["collision_primitives"][0][field] = value
        assert schema_errors(bad), field
    bad = copy.deepcopy(floor)
    bad["static"] = False
    assert schema_errors(bad)
    bad = copy.deepcopy(floor)
    bad["parts"][0]["collision_primitives"] = [1]
    assert schema_errors(bad)
    results["factory_floor"].update({"analytic_planes": 1, "plane_z_mm": 0,
                                     "substrate_repeat_mm": [1000, 1000],
                                     "invalid_plane_schema_cases_rejected": 6})

    fence = metadata["safety_fence"]
    mesh = read_obj(OUTPUT / "safety_fence" / "panel.obj").mesh()
    pieces = mesh.split(only_watertight=False)
    vertical = [m for m in pieces if np.allclose(m.extents * 1000, [3, 3, 1540], atol=.01)]
    horizontal = [m for m in pieces if np.allclose(m.extents * 1000, [3, 920, 3], atol=.01)]
    assert len(vertical) == 19 and len(horizontal) == 31, (len(vertical), len(horizontal))
    assert all(m.is_watertight for m in vertical + horizontal)
    ys = sorted(m.bounds.mean(0)[1] * 1000 for m in vertical)
    zs = sorted(m.bounds.mean(0)[2] * 1000 for m in horizontal)
    assert np.allclose(np.diff(ys), 50, atol=.001) and np.allclose(np.diff(zs), 50, atol=.001)
    assert not any(p["collision"] or p.get("collision_primitives") for p in fence["parts"])
    results["safety_fence"].update({"real_vertical_rods": 19, "real_horizontal_rods": 31,
                                    "rod_diameter_mm": 3, "measured_pitch_mm": 50,
                                    "closed_rod_solids": True, "collision": 0})

    rack = metadata["background_rack"]
    instances = [p for p in rack["parts"] if p.get("library_instance")]
    assert len(instances) == 14
    for part in instances:
        lineage = part["library_instance"]
        library = OUTPUT / lineage["prop_id"]
        source = json.loads((library / "prop.json").read_text())
        original = next(p for p in source["parts"] if p["name"] == "body")
        selected = next((v for v in source["variants"] if v["name"] == lineage["variant"]), {})
        texture = selected.get("textures", {}).get("body", original["texture"])
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest(OUTPUT / "background_rack" / part["texture"]) == lineage["texture_sha256"]
        assert digest(library / texture) == lineage["texture_sha256"]
        visual = read_obj(OUTPUT / "background_rack" / part["visual"])
        stock = read_obj(library / original["visual"])
        expected = stock.vertices * lineage["uniform_scale"] + np.array(lineage["position_mm"]) / 1000
        assert np.allclose(visual.vertices.min(0), expected.min(0), atol=3e-6)
        assert np.allclose(visual.vertices.max(0), expected.max(0), atol=3e-6)
        assert len(visual.faces) == len(stock.faces)
        # Blender may reorder almost-identical UV records when transformed
        # vertices are welded. Check their geometric corner associations,
        # rather than requiring the OBJ's arbitrary vt indexing to match.
        restored = (visual.vertices - np.array(lineage["position_mm"]) / 1000) / lineage["uniform_scale"]
        distances, indices = cKDTree(stock.vertices).query(restored)
        assert distances.max() < 2e-6
        source_uv = {}
        for face, corners in zip(stock.faces, stock.uv_faces, strict=True):
            for vertex, uv in zip(face, corners, strict=True):
                source_uv.setdefault(vertex, set()).add(tuple(stock.uv[uv]))
        max_uv_error = 0
        for face, corners in zip(visual.faces, visual.uv_faces, strict=True):
            for vertex, uv in zip(face, corners, strict=True):
                error = min(np.linalg.norm(visual.uv[uv] - candidate)
                            for candidate in source_uv[indices[vertex]])
                max_uv_error = max(max_uv_error, float(error))
        assert max_uv_error < 1e-5, (part["name"], max_uv_error)  # <0.021 atlas pixels
    assert not any(p["collision"] or p.get("collision_primitives") for p in rack["parts"])
    results["background_rack"].update({"reused_stock_instances": 14,
                                       "exact_texture_hashes": True, "preserved_source_uvs": True,
                                       "scale_and_placement_verified": True, "collision": 0})
    (QA / "geometry_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return metadata


def mujoco_check(metadata):
    """Compile all delivered meshes/textures and drop probes onto both supports."""
    import mujoco

    xml = ET.Element("mujoco", model="task11_environment_asset_smoke")
    ET.SubElement(xml, "compiler", angle="radian")
    ET.SubElement(xml, "option", timestep="0.002")
    assets = ET.SubElement(xml, "asset")
    world = ET.SubElement(xml, "worldbody")
    positions = {"workbench": "0 0 0", "factory_floor": "0 0 0",
                 "safety_fence": "1.5 1 0", "background_rack": "2 -1 0"}
    for prop_id, data in metadata.items():
        folder = OUTPUT / prop_id
        body = ET.SubElement(world, "body", name=prop_id, pos=positions[prop_id])
        for part in data["parts"]:
            name = f"{prop_id}__{part['name']}"
            ET.SubElement(assets, "mesh", name=name, file=str(folder / part["visual"]))
            ET.SubElement(assets, "texture", name=name, type="2d", file=str(folder / part["texture"]))
            ET.SubElement(assets, "material", name=name, texture=name, texuniform="false",
                          **{k: str(v) for k, v in part["mujoco_material"].items()})
            ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, material=name,
                          contype="0", conaffinity="0", group="1")
            for i, reference in enumerate(part["collision"]):
                col = f"{name}__col_{i}"
                ET.SubElement(assets, "mesh", name=col, file=str(folder / reference))
                ET.SubElement(body, "geom", name=col, type="mesh", mesh=col,
                              friction=f"{part['friction']} 0.005 0.0001", group="3")
            for i, primitive in enumerate(part.get("collision_primitives", [])):
                assert primitive["normal"] == [0, 0, 1]
                ET.SubElement(body, "geom", name=f"{name}__plane_{i}", type="plane",
                              pos=" ".join(map(str, primitive["pos_m"])),
                              size=" ".join(map(str, primitive["size_m"])), group="3",
                              friction=f"{part['friction']} 0.005 0.0001")
    for name, position in (("bench_probe", "0 0 .95"), ("floor_probe", "-2 0 .15")):
        body = ET.SubElement(world, "body", name=name, pos=position)
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", type="sphere", size=".025", mass=".05")
    path = QA / "environment_smoke.xml"
    ET.ElementTree(xml).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    state = mujoco.MjData(model)
    for _ in range(750):
        mujoco.mj_step(model, state)
    heights = {}
    for name, expected in (("bench_probe", .825), ("floor_probe", .025)):
        actual = float(state.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name), 2])
        assert abs(actual - expected) < .001, (name, actual, expected)
        heights[name] = actual
    assert state.ncon == 2, state.ncon
    result = {"mujoco_version": mujoco.__version__, "meshes_compiled": model.nmesh,
              "textures_compiled": model.ntex, "static_collision_geoms": 2,
              "resting_probe_center_z_m": heights, "contacts": state.ncon,
              "steps": 750, "status": "PASS"}
    (QA / "mujoco_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def contact_sheet():
    from PIL import Image, ImageDraw, ImageFont

    paths = [OUTPUT / name / "preview_export.png" for name in IDS]
    if not all(path.is_file() for path in paths):
        return
    sheet = Image.new("RGB", (1024, 1104), "#343A40")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    for index, (name, path) in enumerate(zip(IDS, paths, strict=True)):
        x, y = index % 2 * 512, index // 2 * 552
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB").resize((512, 512)), (x, y))
        draw.text((x + 18, y + 518), name, font=font, fill="white")
    sheet.save(HERE / "task11_previews.png")


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", action="store_true")
    parsed = parser.parse_args(args)
    if parsed.sources:
        source_checks()
    else:
        metadata = geometry_checks()
        mujoco_check(metadata)
        contact_sheet()


if __name__ == "__main__":
    main()
