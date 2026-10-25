"""Side-by-side close-render inputs for each proposed indexed albedo.

    Reads candidates without changing assets. The actual exported mesh and UVs
    are rendered twice with identical CPU Cycles lighting/camera at 640 x 480.
"""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpy
import numpy as np
from mathutils import Vector
import prop_lib as props

QA = HERE / "_qa" / "task13"
ASSETS = props.ROOT / "assets" / "props"
DESTINATION = QA / "palette_renders"
DESTINATION.mkdir(exist_ok=True)


def texture_parts():
    found = {}
    for path in sorted(ASSETS.glob("*/prop.json")):
        data = json.loads(path.read_text())
        by_name = {part["name"]: part for part in data["parts"]}
        for part in data["parts"] + [p for variant in data["variants"] for p in variant.get("parts", [])]:
            if part.get("texture"):
                found.setdefault((path.parent / part["texture"]).resolve(), (path.parent, part))
        for variant in data["variants"]:
            for name, texture in variant.get("textures", {}).items():
                found.setdefault((path.parent / texture).resolve(), (path.parent, by_name[name]))
    return found


def pair(relative, folder, part):
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    bpy.ops.wm.obj_import(filepath=str(folder / part["visual"]), forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    obj.data.materials.clear()
    material = props.pbr_material("Actual single-atlas material", "#FFFFFF",
                                  1 - part["mujoco_material"]["shininess"], 0)
    tree = material.node_tree
    texture = tree.nodes.new("ShaderNodeTexImage")
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = obj.data.uv_layers.active.name
    tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
    tree.links.new(texture.outputs["Color"], tree.nodes["Principled BSDF"].inputs["Base Color"])
    tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = part["mujoco_material"]["specular"]
    obj.data.materials.append(material)
    bounds = np.asarray([tuple(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box])
    target = Vector((bounds.min(axis=0) + bounds.max(axis=0)) / 2)
    span = float(np.ptp(bounds, axis=0).max())
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.cycles.samples, scene.cycles.seed, scene.cycles.use_denoising = 64, 0, True
    scene.view_settings.view_transform, scene.view_settings.look = "AgX", "None"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (.18, .18, .18, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .5
    camera_data = bpy.data.cameras.new("Close comparison camera")
    camera = bpy.data.objects.new("Close comparison camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = target + Vector((-1.8, 1.35, 1.25)).normalized() * max(.6, span * 2)
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type, camera_data.ortho_scale = "ORTHO", span * 1.45
    camera_data.clip_start, camera_data.clip_end = .0001, max(20, span * 30)
    scene.camera = camera
    for name, offset, power in (("Key", (-1.5, 1.2, 2.1), 120),
                                ("Fill", (-.8, -1.8, .9), 55), ("Rim", (1.2, .7, 1.7), 85)):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy, light_data.shape, light_data.size = power * span * span, "DISK", span * 1.4
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = target + Vector(offset) * span
        light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()
    key = relative.replace("/", "__").removesuffix(".png")
    for version, path in (("rgb", ASSETS / relative), ("palette", QA / "palette_candidates" / relative)):
        texture.image = bpy.data.images.load(str(path), check_existing=False)
        texture.image.colorspace_settings.name = "sRGB"
        scene.render.filepath = str(DESTINATION / f"{key}_{version}.png")
        bpy.ops.render.render(write_still=True)
    print("PALETTE_PAIR", relative, flush=True)
    return {"file": relative, "key": key, "part": part["name"], "mesh": str(folder / part["visual"])}


parts = texture_parts()
selected = json.loads((QA / "palette_selection.json").read_text())
manifest = []
for row in selected:
    relative = row["file"]
    if "albedo" in Path(relative).name:
        folder, part = parts[(ASSETS / relative).resolve()]
        manifest.append(pair(relative, folder, part))
(QA / "palette_render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("PALETTE_RENDER_COMPLETE", len(manifest), flush=True)
