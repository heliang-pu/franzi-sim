"""Task 17 exported geometry, face labels and preservation audit (PROP_PYTHON)."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
from validate_props import read_obj, validate_prop

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QA = HERE / '_qa/task17'
ASSETS = ROOT / 'assets/props'
PARCELS = {'parcel_cube_80': [80,70,60], 'parcel_cube_75': [75,75,75], 'parcel_cube_85': [85,65,55]}
NORMALS = {'top': [0,0,1], 'px': [1,0,0], 'nx': [-1,0,0], 'py': [0,1,0], 'ny': [0,-1,0]}


def main() -> None:
    findings = {}
    for prop_id, dims in {**PARCELS, 'infeed_table': [300,280,900]}.items():
        folder = ASSETS / prop_id
        data = json.loads((folder/'prop.json').read_text())
        assert not validate_prop(folder).errors
        obj = read_obj(folder/'body.obj')
        bounds = np.array([obj.vertices.min(axis=0),obj.vertices.max(axis=0)])*1000
        np.testing.assert_allclose(bounds[0], [-dims[0]/2,-dims[1]/2,0], atol=.31)
        np.testing.assert_allclose(bounds[1], [dims[0]/2,dims[1]/2,dims[2]], atol=.31)
        assert data['parts'][0]['joint'] is None
        assert data['buttons'] == [] and data['lamps'] == [] and 'conveyor' not in data
        assert (folder/'source.blend').is_file()
        if prop_id in PARCELS:
            assert data['dimensions_mm']['wall'] == 3
            assert data['dimensions_mm']['label_wh'] == [56,46]
            assert data['parts'][0]['mass_kg'] == .12 and data['parts'][0]['friction'] == .6
            assert len(data['parts'][0]['collision']) == 1
            assert [v['name'] for v in data['variants']] == list(NORMALS)
            hashes = []
            centroids = obj.vertices[obj.faces].mean(axis=1)*1000
            triangles = obj.vertices[obj.faces]
            normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            normals /= np.linalg.norm(normals,axis=1)[:,None]
            # Area-weighted pixel sampling detects the white label on its requested outer face.
            areas = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]),axis=1)/2
            face_scores = {}
            for variant in data['variants']:
                face = variant['name']
                assert variant['label_face'] == face and variant['label_normal'] == NORMALS[face]
                texture = folder / variant['textures']['body']
                hashes.append(hashlib.sha256(texture.read_bytes()).hexdigest())
                rgb = np.asarray(Image.open(texture).convert('RGB'))
                # Sample triangle interiors at a grid of barycentric coordinates, not just face centres.
                scores = {}
                for candidate, normal in NORMALS.items():
                    axis = int(np.argmax(np.abs(normal)))
                    boundary = dims[axis] if axis == 2 else normal[axis]*dims[axis]/2
                    selected = (normals @ normal > .95) & (abs(centroids[:,axis]-boundary) < .6)
                    uv = obj.uv[obj.uv_faces[selected]]
                    area = areas[selected]
                    white_area = 0.
                    for i in range(1,13):
                        for j in range(1,14-i):
                            a,b = i/14,j/14
                            coords = uv[:,0]*a + uv[:,1]*b + uv[:,2]*(1-a-b)
                            pixels = rgb[np.clip(((1-coords[:,1])*rgb.shape[0]).astype(int),0,rgb.shape[0]-1),
                                         np.clip((coords[:,0]*rgb.shape[1]).astype(int),0,rgb.shape[1]-1)]
                            is_white = (pixels.min(axis=1)>155) & (np.ptp(pixels.astype(float),axis=1)<45)
                            white_area += float(area[is_white].sum())
                    scores[candidate] = white_area
                assert max(scores,key=scores.get) == face, (prop_id,face,scores)
                face_scores[face] = scores
            assert len(set(hashes)) == 5
            findings[prop_id] = {'bounds_mm': bounds.tolist(), 'label_face_sampling': face_scores,
                                  'distinct_texture_hashes': hashes}
        else:
            assert data['static'] is True and data['parts'][0]['mass_kg'] == 15
            assert len(data['parts'][0]['collision']) == 5
            for filename in data['parts'][0]['collision'][1:]:
                verts = read_obj(folder/filename).vertices
                assert np.max(abs(verts[:,0])) <= .15 and np.max(abs(verts[:,1])) <= .14
                assert .4 + verts[:,0].min() > .24
            findings[prop_id] = {'bounds_mm': bounds.tolist(), 'inset_legs_clear_robot': True}
    baseline = json.loads((QA/'task08_hashes_before.json').read_text())
    after = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in baseline}
    changed = [name for name in baseline if baseline[name] != after[name]]
    assert changed == ['assets/props/pallet_carton_160x80x80/prop.json'], changed
    directories = {Path(name).parent for name in baseline}
    created = sorted(str(p.relative_to(ROOT)) for directory in directories for p in (ROOT/directory).rglob('*')
                     if p.is_file() and str(p.relative_to(ROOT)) not in baseline)
    expected_new = {'body_albedo_imc.png','imc_handling_source.png','imc_sku_source.png','preview_imc.png','source_imc.blend'}
    assert {Path(name).name for name in created} == expected_new, created
    assert all(Path(name).parent == Path('assets/props/pallet_carton_160x80x80') for name in created)
    findings['task08_preservation'] = {'baseline_files':len(baseline),'unchanged_files': len(baseline)-len(changed),
                                      'changed_existing_files':changed, 'created_files':created, 'after_hashes':after}
    carton = json.loads((ASSETS/'pallet_carton_160x80x80/prop.json').read_text())
    original = json.loads((QA/'carton_manifest_before.json').read_text())
    assert {k:v for k,v in carton.items() if k != 'variants'} == {k:v for k,v in original.items() if k != 'variants'}
    assert any(v['name'] == 'imc' and v['sku'] == 'IMC-PLC-IO10' for v in carton['variants'])
    (QA/'verification.json').write_text(json.dumps(findings,indent=2)+'\n')
    print('PASS: dimensions, origin, mass, friction, label-face texture sampling, table leg clearance and Task 08 hashes.')


if __name__ == '__main__':
    main()
