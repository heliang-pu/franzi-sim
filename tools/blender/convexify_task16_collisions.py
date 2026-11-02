"""Restore per-piece convex topology after Blender's micrometre OBJ export.

No vertices move and no two collision pieces are combined. Near-identical
bmesh clipping vertices can quantize to the same point; re-hulling their unique
exported points removes the resulting collapsed triangles before compaction.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from validate_props import read_obj
from convexify_task02_collisions import collision_files

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'assets'/'props'


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--props',nargs='+',required=True)
    args=parser.parse_args()
    report_path=HERE/'task16_convexity_checks.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    for name in args.props:
        folder=ROOT/name; data=json.loads((folder/'prop.json').read_text()); repairs=[]
        for file in collision_files(data):
            path=folder/file; obj=read_obj(path); original=obj.mesh()
            if (original.is_convex and original.is_watertight and original.is_winding_consistent
                    and (original.area_faces>1e-18).all()): continue
            hull=original.convex_hull
            ratio=abs(hull.volume/original.volume)
            if not .998<=ratio<=1.002:
                raise ValueError(f'Unexpected proxy shape change: {name}/{file}: {ratio}')
            assert hull.is_convex and hull.is_watertight and len(hull.vertices)<=64
            existing=set(map(tuple,obj.vertices))
            assert all(tuple(v) in existing for v in hull.vertices)
            lines=['# Convex hull of this individual proxy; original exported positions unchanged.']
            lines+=['v '+' '.join(f'{x:.12f}' for x in v) for v in hull.vertices]
            lines+=['vn '+' '.join(f'{x:.9f}' for x in n) for n in hull.face_normals]
            lines+=['f '+' '.join(f'{int(v)+1}//{i+1}' for v in f) for i,f in enumerate(hull.faces)]
            path.write_text('\n'.join(lines)+'\n')
            check=read_obj(path).mesh()
            assert check.is_convex and check.is_watertight and check.is_winding_consistent
            repairs.append(dict(file=file,hull_volume_ratio=float(ratio),vertices=len(hull.vertices),
                                removed_quantized_duplicates=len(obj.vertices)-len(np.unique(obj.vertices,axis=0))))
        history={r['file']:r for r in report.get(name,{}).get('repaired',[])}
        history.update({r['file']:r for r in repairs})
        report[name]=dict(checked=len(collision_files(data)),repaired=[history[k] for k in sorted(history)])
        print('TASK16_CONVEX',name,len(collision_files(data)),len(repairs),'repaired',flush=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
