"""Reopen compact task-17 PBR sources and verify external artwork references.

blender -b --factory-startup -t 6 -P tools/blender/check_task19_sources.py --
"""
import json
from pathlib import Path
import sys

import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prop_paths import PROPS

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
QA = ROOT / 'tools/blender/_qa/task19'


def main() -> None:
    result = {}
    for name in ('parcel_cube_80', 'parcel_cube_75', 'parcel_cube_85', 'infeed_table'):
        folder = PROPS / name
        bpy.ops.wm.open_mainfile(filepath=str(folder / 'source.blend'))
        manifest = json.loads(bpy.context.scene['task17_manifest'])
        assert manifest == json.loads((folder / 'prop.json').read_text())
        images = []
        for image in bpy.data.images:
            assert not image.packed_file, (name, image.name)
            if image.type in ('RENDER_RESULT', 'COMPOSITING'):
                continue
            assert image.filepath.startswith('//'), (name, image.filepath)
            assert image.filepath.endswith(('_source.png', '_roughness.png'))
            assert Path(bpy.path.abspath(image.filepath)).is_file()
            assert 0 < max(image.size) <= 1024, (name, image.name, list(image.size))
            images.append({'file': image.filepath, 'size': list(image.size)})
        body = bpy.data.objects.get('body')
        assert body and body.type == 'MESH' and len(body.data.materials)
        for material in body.data.materials:
            if material and material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == 'TEX_IMAGE':
                        assert node.image and node.image in bpy.data.images.values()
        result[name] = {'images': images, 'packed_images': 0,
                        'materials': len(body.data.materials), 'vertices': len(body.data.vertices)}
    (QA / 'sources.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: four sources reopen, PBR materials intact, no packed/derived images, all external art resolves.')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
