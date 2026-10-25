"""Render original vs compact exported albedos at 640 x 480, at 0.5-0.6 m."""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpy
from mathutils import Vector
import prop_lib as props

QA = HERE / "_qa" / "task13"


def render(prop_id, version):
    folder = QA / "reference" / prop_id if version == "before" else props.ROOT / "assets" / "props" / prop_id
    data = json.loads((folder / "prop.json").read_text())
    props.reset_scene()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    objects = []
    for part in data["parts"]:
        if prop_id == "cnc_front" and part["name"] not in ("panel", "screen"):
            continue
        previous = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(folder / part["visual"]), forward_axis="Y", up_axis="Z")
        for obj in set(bpy.data.objects) - previous:
            obj.data.materials.clear()
            material = props.pbr_material(part["name"] + " exported RGB", "#FFFFFF", .65, 0)
            tree = material.node_tree
            image = tree.nodes.new("ShaderNodeTexImage")
            image.image = bpy.data.images.load(str((folder / part["texture"]).resolve()), check_existing=False)
            image.image.colorspace_settings.name = "sRGB"
            uv = tree.nodes.new("ShaderNodeUVMap")
            uv.uv_map = obj.data.uv_layers.active.name
            tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
            tree.links.new(image.outputs["Color"], tree.nodes["Principled BSDF"].inputs["Base Color"])
            if prop_id == "cnc_front" and part["name"] == "screen":
                tree.links.new(image.outputs["Color"], tree.nodes["Principled BSDF"].inputs["Emission Color"])
                tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = .3
            obj.data.materials.append(material)
            objects.append(obj)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.cycles.samples, scene.cycles.seed = 64, 0
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform, scene.view_settings.look = "AgX", "None"
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (.18, .18, .18, 1)
    background.inputs["Strength"].default_value = .5
    if prop_id == "parcel_box_m":
        target = Vector((0, 0, .075))
        offset = Vector((-.31, .13, .39)).normalized() * .5
        lens = 70
    else:
        target = Vector((-.041, -.365, 1.097))
        offset = Vector((-.5, 0, 0))
        lens = 70
    camera_data = bpy.data.cameras.new("640x480 close camera")
    camera = bpy.data.objects.new("640x480 close camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = target + offset
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.lens, camera_data.sensor_width = lens, 36
    camera_data.clip_start, camera_data.clip_end = .001, 10
    scene.camera = camera
    for name, relative, energy, size in (
        ("Key", (-.35, .3, .6), 30, .5),
        ("Fill", (-.25, -.4, .25), 12, .6),
        ("Rim", (.3, .1, .5), 18, .4),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy, light_data.shape, light_data.size = energy, "DISK", size
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = target + Vector(relative)
        light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(QA / f"{prop_id}_{version}_640.png")
    bpy.ops.render.render(write_still=True)
    print("TASK13_CLOSE", prop_id, version, "640x480", "distance_m", offset.length, flush=True)


for prop_id in ("parcel_box_m", "cnc_front"):
    for version in ("before", "after"):
        render(prop_id, version)
