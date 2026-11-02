"""Functional/export audit for section 6.2. Run with PROP_PYTHON; no Blender needed."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from scipy.spatial import ConvexHull
from validate_props import read_obj, validate_prop

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'assets/props'
QA = ROOT / 'tools/blender/_qa/task15'
IDS = ('plc_module','module_tray','test_fixture','ng_bin','product_box','accessory_box','stack_light','line_console')


def repair_collisions(ids):
    """Re-hull only quantization-damaged individual proxies, never assemblies."""
    report_path = QA / "collision_repairs.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for pid in ids:
        folder = OUT / pid
        data = json.loads((folder/'prop.json').read_text())
        repairs = []
        for part in data['parts']:
            for name in part['collision']:
                path = folder / name
                original = read_obj(path).mesh()
                if original.is_convex:
                    continue
                hull = original.convex_hull
                ratio = abs(hull.volume / original.volume)
                if not original.is_watertight or not .998 <= ratio <= 1.002:
                    raise ValueError(f'Refusing nontrivial change: {pid}/{name}: {ratio}')
                assert len(hull.vertices) <= 64 and hull.is_convex
                original_points = {tuple(v) for v in original.vertices}
                assert all(tuple(v) in original_points for v in hull.vertices)
                lines = ['# Convex topology from unchanged quantized Task 15 proxy points; metres']
                lines += ['v '+' '.join(f'{v:.9f}' for v in point) for point in hull.vertices]
                lines += ['vn '+' '.join(f'{v:.9f}' for v in normal) for normal in hull.face_normals]
                lines += ['f '+' '.join(f'{v+1}//{i+1}' for v in face) for i,face in enumerate(hull.faces)]
                path.write_text('\n'.join(lines)+'\n')
                assert read_obj(path).mesh().is_convex
                repairs.append({'file':name,'volume_ratio':float(ratio),'vertices':len(hull.vertices)})
        report[pid] = repairs or report.get(pid, [])
    (QA/'collision_repairs.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Quantized collision repairs:',sum(map(len,report.values())))


def audit():
    data = {pid: json.loads((OUT/pid/'prop.json').read_text()) for pid in IDS}
    checks = []
    def check(condition, message):
        if not condition:
            raise AssertionError(message)
        checks.append(message)
    hulls = {}
    def occupied(pid, point, parts=None):
        key = (pid, tuple(parts or []))
        if key not in hulls:
            files = [f for p in data[pid]['parts'] if parts is None or p['name'] in parts for f in p['collision']]
            hulls[key] = [ConvexHull(read_obj(OUT/pid/f).vertices*1000).equations for f in files]
        p = np.asarray(point)
        return any(np.all(h[:,:3]@p+h[:,3] < -1e-5) for h in hulls[key])
    for pid in IDS:
        result = validate_prop(OUT/pid)
        check(not result.errors, pid+' passes public validator: '+str(result.errors))
        check(data[pid]['budget_class'] in ('small','medium'),pid+' has explicit budget_class')
        for p in data[pid]['parts']:
            j=p['joint']
            if j:
                check(j['parent']==p['parent'],pid+'/'+p['name']+' joint parent matches')
                check(abs(np.linalg.norm(j['axis'])-1)<1e-5,pid+'/'+p['name']+' unit joint axis')
    check([v['name'] for v in data['plc_module']['variants']]==['DI','DO','AI'],'All three PLC variants')
    check([v['name'] for v in data['module_tray']['variants']]==['in','ok'],'Both tray variants')
    check(len(data['plc_module']['parts'][0]['collision'])==1,'PLC single box collision')
    check(len(data['module_tray']['parts'][0]['collision'])<=30,'Tray collision budget <=30')
    for cy in (-56,56):
        for x in (-34,0,34):
            for y in (cy-46,cy,cy+46):
                check(not occupied('module_tray',(x,y,12)),'Tray pocket free '+str((x,y,12)))
                check(occupied('module_tray',(x,y,9)),'Tray pocket supported '+str((x,y,9)))
        for side in (-1,1):
            for y in (cy-14,cy,cy+14):
                for x in (36,45,57):
                    check(not occupied('module_tray',(side*x,y,6)),'Finger slot clear '+str((side*x,y,6)))
                    check(occupied('module_tray',(side*x,y,4)),'Finger slot bottom '+str((side*x,y,4)))
    for x in np.linspace(-32,32,5):
        for y in np.linspace(-44,44,5):
            check(not occupied('test_fixture',(x,y,29),['body']),'Fixture module seat clear '+str((x,y)))
    for side in (-1,1):
        for y in (-24,0,24):
            check(not occupied('test_fixture',(side*43,y,29),['body']),'Fixture finger corridor '+str((side,y)))
    for x in (-84,0,84):
        for y in (-24,0,24):
            for z in (70,110,150):
                check(not occupied('test_fixture',(x,y,z),['body']),
                      'Fixture gripper housing envelope clear '+str((x,y,z)))
    lid = next(p for p in data['product_box']['parts'] if p['name']=='lid')
    check(lid['joint']['axis']==[0,1,0] and lid['joint']['range']==[0,1.75],'Lid opens positively around +Y to 1.75 rad')
    check(lid['joint']['damping']==.01 and lid['joint']['frictionloss']==.001,'Lid exact damping/friction')
    check(len(lid['collision'])==2,'Lid panel + front flange collision')
    for x in (-40,5,45):
        for y in (-78,-52,-22,-18,31,78):
            check(not occupied('product_box',(x,y,4),['body']),'Product interior clear '+str((x,y)))
            check(occupied('product_box',(x,y,2),['body']),'Product floor supports '+str((x,y)))
    for pid in ('test_fixture','line_console'):
        for b in data[pid]['buttons']:
            p=next(p for p in data[pid]['parts'] if p['name']==b['part'])
            check(p['joint']['type']=='slide' and p['joint']['spring']['stiffness']>0,pid+'/'+b['part']+' sprung slide')
            check(b['cap_diameter_mm']==40,pid+'/'+b['part']+' 40 mm cap')
            check(np.allclose(p['joint']['axis'],b['press_direction']),pid+'/'+b['part']+' press direction')
            check(p['joint']['range'][1]==b['travel_m'],pid+'/'+b['part']+' stroke metadata')
    for pid in ('test_fixture','stack_light'):
        for lamp in data[pid]['lamps']:
            p=next(p for p in data[pid]['parts'] if p['name']==lamp['part'])
            check(p['texture'] is None and len(p['rgba'])==4,pid+'/'+lamp['part']+' separate untextured off-state lens')
            check(lamp['emission']==.8 and len(lamp['on_rgba'])==4,pid+'/'+lamp['part']+' on-state metadata')
    check([p['name'] for p in data['stack_light']['parts'] if p['texture'] is None]==['lens_green','lens_amber','lens_red'],'Three separate tower lenses')
    check(not data['line_console']['lamps'],'Console has no obsolete lamps')
    devices=data['line_console']['devices']
    check([p['name'] for p in devices[:4]]==['reset','mode','start','stop'],'Console row order')
    check(all(a['panel_uv_mm'][0]>b['panel_uv_mm'][0] for a,b in zip(devices[:3],devices[1:4])),'Console left-to-right follows +Y to -Y')
    check(all('conveyor' not in d for d in data.values()),'No conveyor metadata: Group B contains no conveyors')
    for line in (QA/'protected_before.sha256').read_text().splitlines():
        expected,path=line.split(None,1)
        actual=hashlib.sha256((ROOT/path.strip()).read_bytes()).hexdigest()
        check(actual==expected,'Protected file unchanged: '+path.strip())
    report={'passed':len(checks),'checks':checks,
            'props':{pid:{'bbox_mm':d['bbox_mm'],'triangles':d['triangles'],
                         'collision_pieces':sum(len(p['collision']) for p in d['parts'])} for pid,d in data.items()}}
    (QA/'functional_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} functional and preservation checks passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-collisions',nargs='+',choices=IDS)
    args=parser.parse_args()
    if args.repair_collisions: repair_collisions(args.repair_collisions)
    else: audit()
