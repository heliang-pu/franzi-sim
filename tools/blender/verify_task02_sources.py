"""Headless acceptance of packed task-2 PBR sources and applied frames."""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
from mathutils import Matrix

import prop_lib as props


def main():
    results = {}
    for prop_id in ("control_console", "panel_devices"):
        folder = props.ROOT / "assets" / "props" / prop_id
        data = json.loads((folder / "prop.json").read_text())
        bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
        checks = {}
        scene = bpy.context.scene
        checks["metadata_matches_source"] = json.loads(scene["metadata_json"]) == data
        checks["cpu_cycles_64_samples"] = (scene.render.engine == "CYCLES" and scene.cycles.device == "CPU"
                                            and scene.cycles.samples == 64)
        checks["metre_scene_units"] = scene.unit_settings.scale_length == 1.0
        lists = [data["parts"]] if prop_id == "panel_devices" else [
            v["parts"] for v in data["variants"] if "parts" in v]
        for parts in lists:
            for part in parts:
                name = Path(part["visual"]).stem
                obj = bpy.data.objects.get(name)
                checks[name + "_exists"] = obj is not None
                if obj is None:
                    continue
                checks[name + "_applied_transform"] = max(
                    abs(obj.matrix_world[i][j] - Matrix.Identity(4)[i][j])
                    for i in range(4) for j in range(4)) < 1e-8
                checks[name + "_principled_pbr"] = bool(obj.data.materials) and all(
                    material and any(n.type == "BSDF_PRINCIPLED" for n in material.node_tree.nodes)
                    for material in obj.data.materials)
                checks[name + "_packed_prints"] = all(
                    node.image is not None and node.image.packed_file is not None
                    for material in obj.data.materials for node in material.node_tree.nodes
                    if node.type == "TEX_IMAGE")
                if part.get("emission"):
                    checks[name + "_pbr_emission"] = all(
                        abs(node.inputs["Emission Strength"].default_value - 0.8) < 1e-6
                        for material in obj.data.materials for node in material.node_tree.nodes
                        if node.type == "BSDF_PRINCIPLED")
        results[prop_id] = checks
    # Reproduce the earlier sequential-build transition without any bakes.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cleared = bpy.data.filepath == ""
    with bpy.data.libraries.load(str(props.ROOT / "assets/props/panel_devices/source.blend")) as (_, target):
        target.objects = ["bezel"]
    results["sequential_build_transition"] = {
        "saved_filepath_cleared": cleared, "library_append_succeeds": target.objects[0] is not None}
    path = Path(__file__).with_name("task02_source_checks.json")
    path.write_text(json.dumps(results, indent=2) + "\n")
    for group, checks in results.items():
        print(group, sum(checks.values()), "/", len(checks), "source checks passed", flush=True)
        for key, passed in checks.items():
            if not passed:
                print("FAIL", group, key, flush=True)
    return int(any(not value for group in results.values() for value in group.values()))


if __name__ == "__main__":
    sys.exit(main())
