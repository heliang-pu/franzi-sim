"""Audit task 14 functional geometry and write machine-readable evidence.

Run with PROP_PYTHON after the build and the public validator.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from validate_props import read_obj, validate_prop
from prop_paths import PROPS

ROOT = Path(__file__).resolve().parents[2]
ASSETS = PROPS
QA = ROOT / 'tools/blender/_qa/task14'
IDS = ('tote_cart', 'line_feed_conveyor', 'takeaway_conveyor', 'induction_conveyor',
       'flow_lane', 'recheck_shelf', 'station_sign', 'small_tote')


def points(folder: Path, references: list[str]) -> np.ndarray:
    return np.concatenate([read_obj(folder / ref).vertices for ref in references])


def repair_collisions(names: list[str]) -> None:
    """Match Task 07: re-hull only unchanged, individually quantized proxies."""
    QA.mkdir(parents=True, exist_ok=True)
    destination = QA / 'convexity_repairs.json'
    report = json.loads(destination.read_text()) if destination.exists() else {}
    for name in names:
        folder = ASSETS / name
        data = json.loads((folder / 'prop.json').read_text())
        repaired = []
        for part in data['parts']:
            for ref in part['collision']:
                path = folder / ref
                original = read_obj(path).mesh()
                if original.is_convex:
                    continue
                hull = original.convex_hull
                ratio = abs(float(hull.volume / original.volume))
                assert original.is_watertight and .998 <= ratio <= 1.002, (name, ref, ratio)
                assert hull.is_convex and len(hull.vertices) <= 64
                existing = {tuple(v) for v in original.vertices}
                assert all(tuple(v) in existing for v in hull.vertices)
                lines = ['# Convex faces of existing quantized vertices; metres, common prop frame']
                lines += ['v ' + ' '.join(f'{v:.9f}' for v in point) for point in hull.vertices]
                lines += ['vn ' + ' '.join(f'{v:.9f}' for v in normal) for normal in hull.face_normals]
                lines += ['f ' + ' '.join(f'{int(v)+1}//{i+1}' for v in face)
                          for i, face in enumerate(hull.faces)]
                path.write_text('\n'.join(lines) + '\n')
                result = read_obj(path).mesh()
                assert result.is_convex and result.is_watertight and result.is_winding_consistent
                repaired.append(dict(file=ref, volume_ratio=ratio, vertices=len(result.vertices)))
        report[name] = repaired or report.get(name, [])
        print(name, len(repaired), 'quantization repairs')
    destination.write_text(json.dumps(report, indent=2)+'\n')


def main() -> None:
    evidence = {}
    for name in IDS:
        folder = ASSETS / name
        data = json.loads((folder / 'prop.json').read_text())
        result = validate_prop(folder)
        assert not result.errors, (name, result.errors)
        assert sum(p['mass_kg'] for p in data['parts']) > 0
        if name != 'small_tote':
            assert data['static'] is True
            assert math.isclose(sum(p['mass_kg'] for p in data['parts']), data['mass_kg'], abs_tol=1e-9)
            assert data['dimensions_mm'] and data['recommended_placement_m']
        evidence[name] = dict(bbox_mm=result.bbox, triangles=result.triangles,
                             collisions=result.collisions, max_collision_vertices=result.max_vertices,
                             texture_px=sorted(result.textures), joints=sum(p['joint'] is not None for p in data['parts']),
                             variants=[v['name'] for v in data['variants']])
    for name, count, pitch, radius, length, top, axis, first in (
            ('line_feed_conveyor',20,.060,.024,.330,1.050,1,.030),
            ('takeaway_conveyor',26,.030,.010,.280,.030,0,-.375),
            ('induction_conveyor',44,.025,.010,.360,.940,1,.015)):
        folder=ASSETS/name
        data=json.loads((folder/'prop.json').read_text())
        conv=data['conveyor']
        rollers=[p for p in data['parts'] if p['name'] in conv['roller_parts']]
        assert len(rollers)==count
        assert len({p['texture'] for p in rollers})==1
        expected_axis=np.eye(3)[axis]
        for i,part in enumerate(rollers):
            joint=part['joint']
            assert joint['type']=='hinge' and joint['continuous'] and joint['limited'] is False
            assert joint['damping']==.001 and joint['frictionloss']==0
            assert joint['axis']==list(expected_axis)
            assert len(part['collision'])==1
            collider=read_obj(folder/part['collision'][0]).vertices
            assert len(collider)==32, (name,part['name'],len(collider))
            center=np.asarray(joint['anchor_m'])
            assert abs(center[2]-(top-radius))<1e-8
            travel_axis=0 if axis==1 else 1
            assert abs(center[travel_axis]-(first+i*pitch))<1e-7
            radial=np.delete(collider-center,axis,axis=1)
            assert np.max(np.abs(np.linalg.norm(radial,axis=1)-radius))<.00001
            assert abs(np.ptp(collider[:,axis])-length)<.00001
            visual=read_obj(folder/part['visual']).vertices
            assert abs(visual[:,2].max()-top)<.00001
        assert conv['roller_top_m']==top and conv['roller_radius_m']==radius
        evidence[name]['roller_audit']='Exact count, pitch, height, nominal radii, 32-vertex / 16-sided colliders, continuous hinges and shared atlas: PASS'
    # Check every collision piece intersecting the robot base's height band.
    # No foot, post or caster may escape the stated footprint.
    placements={
        'tote_cart': ((-.130,.130),(-.260,.260),(.37,0)),
        'line_feed_conveyor': ((0,1.2),(-.240,.240),(.5,0)),
        'induction_conveyor': ((0,1.1),(-.300,.300),(.3,0)),
        'flow_lane': ((-.030,.8),(-.224,.224),(.5,0)),
        'recheck_shelf': ((-.120,.120),(-.200,.200),(.36,0)),
    }
    for name,(xb,yb,offset) in placements.items():
        folder=ASSETS/name
        data=json.loads((folder/'prop.json').read_text())
        n=0
        for part in data['parts']:
            for ref in part['collision']:
                v=read_obj(folder/ref).vertices
                if v[:,2].min()>.410: continue
                n+=1
                assert v[:,0].min()>=xb[0]-.00001 and v[:,0].max()<=xb[1]+.00001,(name,ref,'X footprint')
                assert v[:,1].min()>=yb[0]-.00001 and v[:,1].max()<=yb[1]+.00001,(name,ref,'Y footprint')
                world=v[:,:2]+offset
                lo,hi=world.min(axis=0),world.max(axis=0)
                overlap=(hi[0]>-.33 and lo[0]<.24-1e-6 and hi[1]>-.30 and lo[1]<.30)
                assert not overlap,(name,ref,'robot base overlap',lo.tolist(),hi.tolist())
        evidence[name]['floor_footprint_audit']=dict(pieces_checked=n,result='PASS at nominal placement')
    sign=json.loads((ASSETS/'station_sign/prop.json').read_text())
    for part in sign['parts']:
        for ref in part['collision']:
            v=read_obj(ASSETS/'station_sign'/ref).vertices
            if v[:,2].min() <= .410:
                assert np.max(np.abs(v[:,:2])) <= .12501
    evidence['station_sign']['floor_footprint_audit']='PASS within D250 base; placement unspecified'
    for name, limit, top in (('line_feed_conveyor', .300, 1.050), ('flow_lane', .220, 1.050)):
        data=json.loads((ASSETS/name/'prop.json').read_text())
        frame=next(p for p in data['parts'] if p['name']=='frame')
        for ref in frame['collision']:
            v=read_obj(ASSETS/name/ref).vertices
            # Ignore the intentional flow-lane front stop and central parts.
            if v[:,0].max() <= .001 or np.max(np.abs(v[:,1])) < .155:
                continue
            slope = math.tan(math.radians(3)) if name == 'flow_lane' else 0
            if np.max(v[:,2] - top - slope*v[:,0]) > .001 and v[:,0].min() < limit:
                assert v[:,0].min() >= limit-.001, (name, ref, 'loading area blocked')
        evidence[name]['open_loading_segment']='PASS'
    b=json.loads((ASSETS/'line_feed_conveyor/prop.json').read_text())
    button=next(p for p in b['parts'] if p['name']=='release_button')
    assert button['joint']['axis']==[0,0,-1] and button['joint']['range']==[0,.006]
    assert button['joint']['spring']['stiffness']>0
    b=json.loads((ASSETS/'induction_conveyor/prop.json').read_text())
    lamp=next(p for p in b['parts'] if p['name']=='scan_lamp')
    assert lamp['texture'] is None and lamp['rgba']==[.06,.30,.10,1]
    assert b['lamps'][0]['emission']==.8
    b=json.loads((ASSETS/'flow_lane/prop.json').read_text())
    lane=next(p for p in b['parts'] if p['name']=='lane')
    assert lane['joint'] is None and lane['parent']=='frame' and lane['friction']==.03 and lane['contact_priority']==1
    assert len(lane['collision'])==1
    plane=read_obj(ASSETS/'flow_lane'/lane['collision'][0]).vertices
    normal=np.array([-math.sin(math.radians(3)),0,math.cos(math.radians(3))])
    assert abs((plane@normal).max()-1.05*normal[2])<.00001
    protected=[]
    for line in (QA/'companions_before.sha256').read_text().splitlines():
        digest,path=line.split(maxsplit=1)
        actual=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        assert actual==digest,('protected file changed',path)
        protected.append(path)
    evidence['protected_companions']=dict(files=len(protected),result='All SHA-256 hashes identical')
    original=json.loads((QA/'tote_original/prop.json').read_text())
    current=json.loads((ASSETS/'small_tote/prop.json').read_text())
    assert original['bbox_mm']==current['bbox_mm']
    assert original['parts'][0]['mass_kg']==current['parts'][0]['mass_kg']==.45
    assert original['parts'][0]['friction']==current['parts'][0]['friction']==.4
    tote_files=[]
    for old in sorted((QA/'tote_original').glob('*.obj')):
        new=ASSETS/'small_tote'/old.name
        a,b=read_obj(old),read_obj(new)
        assert np.array_equal(a.vertices,b.vertices), ('tote vertex changes',old.name)
        assert np.array_equal(a.faces,b.faces), ('tote topology changes',old.name)
        if len(a.uv):
            assert np.array_equal(a.uv[a.uv_faces],b.uv[b.uv_faces]), ('tote UV changes',old.name)
        tote_files.append(dict(file=old.name,sha256_identical=hashlib.sha256(old.read_bytes()).digest()==hashlib.sha256(new.read_bytes()).digest()))
    assert [v['name'] for v in current['variants']]==['blue','grey','red','yellow','green']
    evidence['small_tote']['geometry_preservation']=tote_files
    (QA/'audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-collisions',action='store_true')
    parser.add_argument('--props',nargs='+',choices=IDS,default=list(IDS))
    args=parser.parse_args()
    if args.repair_collisions:
        repair_collisions(args.props)
    else:
        main()
