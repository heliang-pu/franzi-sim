"""Exercise the real Task 0 builder in an isolated directory under _qa/task13."""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpy
import prop_lib as props
import build_task00_pipeline as builder

sandbox = HERE / "_qa" / "task13" / "rebuild_sandbox"
props.ROOT = sandbox
sys.argv = ["qa_task13_rebuild.py"]
# Test the downsampling path using an actual 2048 Cycles bake.
original_bake = props.bake_albedo


def large_bake(objects, size, path, **kwargs):
    return original_bake(objects, 2048, path, **kwargs)


props.bake_albedo = large_bake
builder.main()
props.finalize_assets()
source = sandbox / "assets" / "props" / "_test_cube" / "source.blend"
assert bpy.context.preferences.filepaths.save_version == 0
bpy.ops.wm.save_as_mainfile(filepath=str(source), check_existing=False)
assert not list(sandbox.rglob("*.blend1"))
print("TASK13_REBUILD_OK", json.loads(source.with_name("prop.json").read_text())["parts"][0]["texture_px"])
