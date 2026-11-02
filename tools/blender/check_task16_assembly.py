"""Check exported Task 16 functional geometry, UV contract and dimensions.

Run with PROP_PYTHON after both builders and validate_props.py.
"""
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image
from validate_props import read_obj

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'assets'/'props'
IDS=('parts_bin','relay','terminal_block','cable_gland','fuse_holder','order_tray',
     'order_card_stand','rotor_shaft_fixture','bearing_tray','foam_kit_tray')


def load(name):
    folder=ASSETS/name
    data=json.loads((folder/'prop.json').read_text())
    hulls=[]
    for p in data['parts']:
        for file in p['collision']:
            points=read_obj(folder/file).vertices*1000
            hulls.append((p['name'],file,ConvexHull(points).equations))
    return data,hulls


def solid(hulls,point):
    return any(np.all(e[:,:3]@point+e[:,3]<=1e-5) for _,_,e in hulls)


def main():
    checks={}
    for name in IDS:
        data,hulls=load(name)
        assert all(len(p['collision'])<=64 for p in data['parts'])
        assert Image.open(ASSETS/name/'inspection.png').size==(1024,1024)
        checks[name]={'bbox_mm':data['bbox_mm'],'triangles':data['triangles'],
                      'collisions':len(hulls),'parts':len(data['parts'])}
    data,hulls=load('parts_bin')
    for x in (-55,0,55):
        assert not solid(hulls,[x,0,4])
        assert solid(hulls,[x,0,2])
    assert [v['name'] for v in data['variants']]==['relay','terminal','gland','fuse']
    for name,z in [('relay',40.02),('terminal_block',35.02),('fuse_holder',40.02)]:
        vertices=read_obj(ASSETS/name/'body.obj').vertices*1000
        assert abs(vertices[:,2].max()-z)<.001,(name,'ink offset lost during compaction')
    card,card_hulls=load('order_card_stand')
    obj=read_obj(ASSETS/'order_card_stand'/'card.obj')
    assert Image.open(ASSETS/'order_card_stand'/'card_front.png').size==(1480,1050)
    assert set(map(tuple,obj.uv))=={(0.,0.),(1.,0.),(1.,1.),(0.,1.)}
    uvtri=obj.uv[obj.uv_faces]
    a,b=uvtri[:,1]-uvtri[:,0],uvtri[:,2]-uvtri[:,0]
    uv_area=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]).sum()/2
    assert abs(uv_area-1)<1e-6
    n=np.array(card['card']['front_normal'])
    projection=obj.vertices@n
    front_faces=np.all(np.isclose(projection[obj.faces],projection.max(),atol=1e-5),axis=1)
    assert front_faces.sum()==2
    assert np.all(obj.normals[obj.normal_faces[front_faces]]@n>.999)
    assert obj.mesh().is_watertight and obj.mesh().volume>0
    front_center=obj.vertices.mean(axis=0)*1000
    assert solid(card_hulls,front_center), 'Card collision must cover its front sheet'
    assert not solid(card_hulls,front_center+n), 'Card proxy must remain behind its front'
    assert abs(card['bbox_mm'][0]-60)<.1, 'Acrylic backing must stay within the 60 mm base'
    checks['order_card_stand']['full_front_uv_area']=float(uv_area)
    data,hulls=load('rotor_shaft_fixture')
    for y in (-120,120):
        assert solid(hulls,[0,y,140])
        assert not solid(hulls,[9.02,y,140])
        # At each azimuth the contact shaft stays inside Ø18 and within 0.05 mm.
        for a in np.linspace(0,2*math.pi,257):
            assert solid(hulls,[8.951*math.cos(a),y+8.951*math.sin(a),140])
            assert not solid(hulls,[9.01*math.cos(a),y+9.01*math.sin(a),140])
        assert solid(hulls,[12.5,y,115.9])
        assert not solid(hulls,[12.5,y,116.1])
        assert solid(hulls,[0,y,155.99])
        assert not solid(hulls,[0,y,156.01])
    assert data['collision_exclusions']==[['start_left','body'],['start_right','body']]
    for name in ('start_left','start_right'):
        p=next(p for p in data['parts'] if p['name']==name)
        assert p['joint']['axis']==[0,0,-1] and p['joint']['range']==[0,.006]
        assert p['joint']['spring']['stiffness']>0
    lamp=next(p for p in data['parts'] if p['name']=='station_lamp')
    assert lamp['texture'] is None and data['lamps'][0]['emission']==.8
    data,hulls=load('foam_kit_tray')
    assert {v['name'] for v in data['variants']}=={'black','light_grey'}
    for p in data['pockets']:
        assert p['depth_mm']==20 and p['floor_z_mm']==23
        cx,cy=p['center_xy_mm']
        px=cx-18 if p['part']=='l_bracket' else cx
        assert solid(hulls,[px,cy,22.9]), (p['part'],'floor support')
        assert not solid(hulls,[px,cy,23.1]), (p['part'],'seat blocked')
        for notch in p['finger_notches']:
            x,y=notch['center_xy_mm']
            assert solid(hulls,[x,y,17.9]),(p['part'],'finger floor')
            for z in (18.1,22.9,30,39.9,42.9):
                for dy in (-12.9,0,12.9):
                    assert not solid(hulls,[x,y+dy,z]),(p['part'],'notch blocked',x,y+dy,z)
            # At the outer X end the lead-in expands by exactly 3 mm.
            a,b,_,_=notch['bounds_xy_mm']; edge=a if notch['side']=='-X' else b
            sign=-1 if notch['side']=='-X' else 1
            assert solid(hulls,[edge+sign*1.5,y,40.1])
            assert not solid(hulls,[edge+sign*1.5,y,42.9])
        if p['part']=='flanged_bushing':
            assert solid(hulls,[cx,cy+21,37.9])
            assert not solid(hulls,[cx,cy+21,38.1])
    checks['foam_kit_tray']['functional_probes']='20 mm seats, 25 mm finger slots, 26 mm widths, 3 mm chamfers, stepped counterbore passed'
    (ROOT/'tools'/'blender'/'task16_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
