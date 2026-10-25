"""Read-only runtime API probe; run headless with --factory-startup."""

import sys

sys.dont_write_bytecode = True

import bpy

print("BLENDER_VERSION", bpy.app.version_string)
for name, op in (
    ("obj_export", bpy.ops.wm.obj_export),
    ("smart_project", bpy.ops.uv.smart_project),
    ("pack_islands", bpy.ops.uv.pack_islands),
    ("cube_project", bpy.ops.uv.cube_project),
    ("bake", bpy.ops.object.bake),
):
    print("OPERATOR", name, [(p.identifier, p.type) for p in op.get_rna_type().properties])
print("MESH_SHARP_API", hasattr(bpy.types.Mesh, "set_sharp_from_angle"))
mat = bpy.data.materials.new("probe")
mat.use_nodes = True
for kind in ("ShaderNodeBsdfPrincipled", "ShaderNodeAmbientOcclusion", "ShaderNodeBevel"):
    node = mat.node_tree.nodes.new(kind)
    print("NODE", kind, "INPUTS", [(p.name, p.type) for p in node.inputs])
