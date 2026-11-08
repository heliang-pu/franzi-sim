"""Apply review fixes to existing deliveries without rebaking approved exports.

blender -b --factory-startup -t 6 -P tools/blender/refine_task19_props.py -- --props ID ...
Metadata edits and source artwork use PROP_PYTHON; run once per delivery.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prop_paths import PROPS
ROOT = HERE.parents[1]
ASSETS = PROPS
IDS = ('line_feed_conveyor', 'takeaway_conveyor', 'flow_lane',
       'tote_cart', 'parcel_cube_80', 'parcel_cube_75', 'parcel_cube_85', 'infeed_table')
LANE_NOTE = ('Lane friction must win over the goods: MuJoCo geom priority 1; '
             'PhysX friction combine mode min. Friction 0.03 is below '
             'tan(3 degrees) = 0.0524 so gravity moves the tote. ')


def update_delivery(name: str) -> None:
    folder = ASSETS / name
    path = folder / 'prop.json'
    data = json.loads(path.read_text())
    if name.endswith('_conveyor'):
        names = data['conveyor']['roller_parts'] + [
            item['part'] for key in ('buttons', 'lamps') for item in data.get(key, [])]
        data['collision_exclusions'] = [['frame', part] for part in names]
    if name == 'flow_lane':
        lane = next(p for p in data['parts'] if p['name'] == 'lane')
        lane.update(friction=.03, contact_priority=1)
        if LANE_NOTE not in data['notes']:
            data['notes'] = LANE_NOTE + data['notes']
    if name == 'tote_cart':
        for placement in data['recommended_placement_m']['tote_sorting']:
            placement['position'] = [0, .50 if placement['position'][1] > 0 else -.50, 0]
    if name.startswith('parcel_cube_'):
        data['parts'][0]['mass_kg'] = .12
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--props', nargs='+', choices=IDS, default=list(IDS))
    parser.add_argument('--manifest-worker', action='store_true')
    argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if args.manifest_worker:
        for name in args.props:
            update_delivery(name)
        return
    import bpy
    from build_task17_parcels import save_compact_source
    subprocess.run([os.environ['PROP_PYTHON'], '-B', str(Path(__file__)),
                    '--manifest-worker', '--props', *args.props], check=True)
    for name in args.props:
        if name != 'infeed_table' and not name.startswith('parcel_cube_'):
            continue
        folder = ASSETS / name
        bpy.ops.wm.open_mainfile(filepath=str(folder / 'source.blend'))
        bpy.context.preferences.filepaths.save_version = 0
        bpy.context.scene['task17_manifest'] = (folder / 'prop.json').read_text()
        save_compact_source(folder)
        print('TASK19_REFINED', name, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
