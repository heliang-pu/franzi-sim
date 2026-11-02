"""Review regression checks, selected-compaction isolation and SHA-256 evidence."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from validate_props import read_obj

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / 'assets/props'
QA = HERE / '_qa/task19'
IDS = ('induction_conveyor', 'line_feed_conveyor', 'takeaway_conveyor', 'flow_lane',
       'tote_cart', 'parcel_cube_80', 'parcel_cube_75', 'parcel_cube_85', 'infeed_table')


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                     'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file()}


def save(name: str, value: dict) -> None:
    (QA / name).write_text(json.dumps(value, indent=2) + '\n')


def differences(before: dict, after: dict) -> list[str]:
    return sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))


def compactor_regression() -> None:
    results = {}
    for step in ('all', 'downscale', 'deduplicate', 'optimize', 'previews', 'obj', 'palette'):
        with tempfile.TemporaryDirectory(prefix='compact-', dir=QA) as temporary:
            root = Path(temporary)
            selected, other = root / 'selected', root / 'other'
            selected.mkdir(); other.mkdir()
            # An invalid unselected manifest proves no pass reads it.
            (other / 'prop.json').write_text('MUST NOT BE PARSED')
            for folder in (selected, other):
                Image.new('RGB', (2048, 2048), '#537799').save(folder / 'body_albedo.png')
                Image.new('RGB', (800, 800), '#998855').save(folder / 'preview.png')
            (selected / 'copy_albedo.png').write_bytes((selected / 'body_albedo.png').read_bytes())
            (selected / 'prop.json').write_text(json.dumps({'notes': 'Regression fixture.', 'parts': [
                {'texture': 'copy_albedo.png', 'texture_px': 2048}]}))
            (selected / 'body.obj').write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
            before = snapshot(other)
            command = [os.environ['PROP_PYTHON'], '-B', str(HERE / 'compact_props.py'),
                       '--root', str(root), '--props', 'selected', '--apply', '--step', step]
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            assert snapshot(other) == before, step
            if step in ('all', 'downscale'):
                assert Image.open(selected / 'body_albedo.png').size == (1024, 1024)
            if step in ('all', 'deduplicate'):
                manifest = json.loads((selected / 'prop.json').read_text())
                assert manifest['parts'][0]['texture'] == 'body_albedo.png'
                assert (selected / 'copy_albedo.png').exists()  # external references remain safe
            results[step] = {'outside_files_unchanged': len(before), 'stdout': result.stdout}
            inventory = subprocess.run(command[:-3], text=True, capture_output=True, check=True)
            assert set(json.loads(inventory.stdout)['props']) == {'selected'}
    save('compactor_regression.json', results)


def actual_scoped_run() -> None:
    before = snapshot(ASSETS)
    save('scoped_before.json', before)
    command = [os.environ['PROP_PYTHON'], '-B', str(HERE / 'compact_props.py'),
               '--apply', '--props', 'induction_conveyor']
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    (QA / 'scoped_compactor.log').write_text(result.stdout + result.stderr)
    after = snapshot(ASSETS)
    save('scoped_after.json', after)
    changed = differences(before, after)
    assert all(path.startswith('induction_conveyor/') for path in changed), changed
    save('scoped_result.json', {'command': ' '.join(command), 'changed': changed,
         'outside_files_unchanged': sum(not key.startswith('induction_conveyor/') for key in before)})


def audit() -> None:
    evidence = {}
    for name in IDS[:3]:
        data = json.loads((ASSETS / name / 'prop.json').read_text())
        names = data['conveyor']['roller_parts'] + [
            item['part'] for key in ('buttons', 'lamps') for item in data.get(key, [])]
        assert data['collision_exclusions'] == [['frame', part] for part in names]
        evidence[name] = {'exclusions': len(names)}
    folder = ASSETS / 'induction_conveyor'
    data = json.loads((folder / 'prop.json').read_text())
    colliders = []
    for part in data['parts']:
        for ref in part['collision']:
            vertices = read_obj(folder / ref).vertices
            colliders.append((part['name'], ref, vertices, ConvexHull(vertices).equations))
    pairs, narrow, overlaps = 0, 0, []
    for a, b in itertools.combinations(colliders, 2):
        if a[0] == b[0]:
            continue
        pairs += 1
        if np.any(np.minimum(a[2].max(0), b[2].max(0)) < np.maximum(a[2].min(0), b[2].min(0)) - 1e-9):
            continue
        narrow += 1
        equations = np.vstack([a[3], b[3]])
        # Maximize a ball's radius inside BOTH convex hulls. A positive radius
        # proves volume penetration, independently of simulator filtering.
        result = linprog([0, 0, 0, -1], A_ub=np.column_stack([equations[:, :3], np.ones(len(equations))]),
                         b_ub=-equations[:, 3], bounds=[(None, None)]*4, method='highs')
        assert result.success
        if result.x[3] > 1e-9:
            overlaps.append([a[1], b[1], float(result.x[3])])
    assert not overlaps, overlaps
    stop = read_obj(folder / 'frame_col_05.obj').vertices
    gaps = {p['name']: float(stop[:, 0].min()-read_obj(folder / p['collision'][0]).vertices[:, 0].max())
            for p in data['parts'] if p['name'].startswith('roller_')}
    assert min(gaps.values()) >= .002-1e-8
    boss = read_obj(folder / 'frame_col_31.obj').vertices
    lamp = read_obj(folder / 'scan_lamp_col_00.obj').vertices
    evidence['induction_conveyor'].update(cross_part_pairs=pairs, aabb_candidates=narrow,
        overlaps=overlaps, end_stop_clearance_m=gaps, lamp_boss_gap_m=float(lamp[:, 2].min()-boss[:, 2].max()))
    lane = json.loads((ASSETS / 'flow_lane/prop.json').read_text())
    part = next(p for p in lane['parts'] if p['name'] == 'lane')
    assert part['friction'] == .03 and part['contact_priority'] == 1
    assert part['friction'] < math.tan(math.radians(3))
    assert 'PhysX friction combine mode min' in lane['notes']
    evidence['flow_lane'] = {'friction': .03, 'contact_priority': 1, 'tan_3_degrees': math.tan(math.radians(3))}
    cart = json.loads((ASSETS / 'tote_cart/prop.json').read_text())
    assert [p['position'] for p in cart['recommended_placement_m']['tote_sorting']] == [[0, .5, 0], [0, -.5, 0]]
    for name in IDS[5:]:
        folder = ASSETS / name
        total = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        assert total <= (3_000_000 if name == 'infeed_table' else 4_000_000), (name, total)
        if name.startswith('parcel_'):
            assert json.loads((folder / 'prop.json').read_text())['parts'][0]['mass_kg'] == .12
        for path in folder.glob('*.png'):
            if path.name.endswith(('_source.png', '_roughness.png')):
                assert max(Image.open(path).size) <= 1024
        evidence[name] = {'bytes': total}
    before = json.loads((QA / 'before.json').read_text())
    after = snapshot(ASSETS)
    save('after.json', after)
    changed = differences(before, after)
    assert all(Path(key).parts[0] in IDS for key in changed), changed
    protected = {key: row for key, row in before.items() if Path(key).parts[0] not in IDS}
    assert all(after.get(key) == row for key, row in protected.items())
    delivered = {key: row for key, row in before.items() if Path(key).parts[0] in IDS[5:]
                 and (key.endswith('.obj') or 'albedo' in key or Path(key).name.startswith('preview'))}
    assert all(after.get(key) == row for key, row in delivered.items())
    def digest(rows):
        return hashlib.sha256(''.join(f'{row["sha256"]}  {key}\n' for key, row in sorted(rows.items())).encode()).hexdigest()
    evidence['hashes'] = {'before_files': len(before), 'after_files': len(after),
        'changed': changed, 'untouched_props': len({Path(k).parts[0] for k in protected}),
        'untouched_files': len(protected), 'untouched_aggregate_sha256': digest(protected),
        'protected_delivery_files': len(delivered), 'protected_delivery_aggregate_sha256': digest(delivered),
        'whole_tree_before_sha256': digest(before), 'whole_tree_after_sha256': digest(after)}
    save('audit.json', evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compactor-regression', action='store_true')
    parser.add_argument('--scoped-run', action='store_true')
    args = parser.parse_args()
    QA.mkdir(parents=True, exist_ok=True)
    if args.compactor_regression:
        compactor_regression()
    elif args.scoped_run:
        actual_scoped_run()
    else:
        audit()
