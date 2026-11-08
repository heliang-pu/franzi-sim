"""Write the Task 16 delivery report from validated manifests and OBJ files."""
import json
import numpy as np
from pathlib import Path
from PIL import Image
from validate_props import read_obj, validate_prop
from prop_paths import PROPS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
IDS=('parts_bin','relay','terminal_block','cable_gland','fuse_holder','order_tray',
     'order_card_stand','rotor_shaft_fixture','bearing_tray','foam_kit_tray')


def main() -> None:
    rows=[]; parts=[]; data_by_id={}
    for name in IDS:
        folder=PROPS / name
        assert (folder/'source.blend').is_file()
        result=validate_prop(folder)
        if result.errors: raise ValueError((name,result.errors))
        data=json.loads((folder/'prop.json').read_text()); data_by_id[name]=data
        vertices=np.concatenate([read_obj(folder/p['visual']).vertices for p in data['parts']])
        bbox=' × '.join(f'{v:.3f}' for v in np.ptp(vertices,axis=0)*1000)
        textures=', '.join(str(v) for v in sorted({p['texture_px'] for p in data['parts'] if p['texture_px']}))
        joints=', '.join(p['name']+': '+p['joint']['type'] for p in data['parts'] if p['joint']) or 'none'
        rows.append(f'| `{name}` | {bbox} | {data["triangles"]:,} | {result.collisions} | {textures} | {joints} |')
        for p in data['parts']:
            texture=f'{p["texture_px"]}²' if p['texture'] else 'untextured rgba'
            parts.append(f'| `{name}` / `{p["name"]}` | {len(read_obj(folder/p["visual"]).faces):,} | {len(p["collision"])} | {texture} | {p["mass_kg"]:g} |')
    checks=json.loads((HERE/'task16_checks.json').read_text())
    assert len(checks)==10
    fixture=data_by_id['rotor_shaft_fixture']
    foam=data_by_id['foam_kit_tray']
    lines=['# Task 16 — picking, kitting and bearing assembly','',
        'All ten props pass the unchanged `validate_props.py`. Exported functional geometry also passes '
        '`check_task16_assembly.py`. Every delivered preview was rendered and visually inspected; '
        'the final previews use exported OBJ geometry and albedos, including the light-grey foam variant.','',
        '## Measured delivery','',
        'Bounding boxes are X × Y × Z in millimetres. Export geometry is in metres, Z up, front −X; '
        'all parts retain the common bottom-centre prop frame. Texture sizes are square pixels.','',
        '| Prop | Bbox mm | Triangles | Convex pieces | Texture sizes | Joints |',
        '|---|---:|---:|---:|---|---|',*rows,'',
        '### Per rigid part','',
        '| Prop / part | Triangles | Convex pieces | Texture | Mass kg |',
        '|---|---:|---:|---|---:|',*parts,'',
        '## Functional metadata','',
        '- `parts_bin`: `relay`, `terminal`, `gland`, `fuse` atlas variants, with line icons, '
        '`RLY-24V`, `TB-4P`, `CG-M20`, `FH-10A` and barcodes. `contents` records three centres '
        'along X at −55, 0, +55 mm. Four recommended placements are (0.38, ±0.19/±0.36, 0.90) m.',
        '- `order_tray`: left/right compartment hand assignment; a 12 mm divider above the 3 mm floor; '
        'recommended origin (0.38, 0, 0.90) m.',
        '- `order_card_stand`: `card` is a separate fixed child, with exact full-face [0,1] UVs. '
        '`card_front.png` is the native 1480 × 1050 WORK ORDER template with three blank rows. '
        '`card` metadata records image dimensions, UV bounds, front normal, U/V directions and replacement intent. '
        'Recommended origin (0.56, 0, 0.90) m.',
        '- `rotor_shaft_fixture`: exact button, lamp, interlock and bearing-seat metadata follows. '
        'The two buttons have independent −Z slides, 6 mm travel, 700 N/m return springs, 2 N·s/m '
        'damping, 0.1 N frictionloss and a 4 mm press threshold. The scene implements the 0.5 s '
        'two-hand interlock; the mesh builder does not implement device logic. Lamp is an untextured '
        'fixed part with a dark off-state; green on-state and emission 0.8 are in `lamps`.',
        '- `bearing_tray`: a flat 100 × 100 × 10 mm POM support, printed Ø48 circle and 6204-2RS; '
        'recommended origins (0.38, ±0.30, 0.90) m.',
        '- `foam_kit_tray`: updated `pockets` below includes unchanged centres, 20 mm seat depth, '
        '25 mm finger-notch depth, 26 mm width, 22 mm outward extension and 3 × 45° entries. '
        'The bushing counterbore is Ø48 down to Z38, then Ø38 down to Z23. Both black and light-grey '
        'variants map every fixed section to the same packed 2048 atlas. Recommended origin (0.40, 0, 0.90) m.',
        '- `conveyor`: none; Task 16 contains no conveyors. No other prop has a joint, button or lamp.','',
        '```json',json.dumps({k:fixture[k] for k in ('buttons','lamps','two_hand_start','bearing_seats','collision_exclusions')},indent=2),'```','',
        '```json',json.dumps(foam['pockets'],indent=2),'```','',
        '## Deviations, interpretations and shared-tool conflicts','',
        '1. **Preview resolution:** §1.11 requires 1024 × 1024, but the immutable validator explicitly '
        'requires `preview.png` and every `preview*.png` to be 512 × 512. Each prop therefore includes '
        'a native 1024 × 1024, Cycles 64-sample `inspection.png`, plus its 512 thumbnail as `preview.png`. '
        'The shared validator would need to accept 1024 previews to remove this compatibility exception.',
        '2. **Card texture dimensions:** the shared schema, writer and validator only accept square '
        '1024/2048 atlases. The native 1480 × 1050 template is supplied as `card_front.png`, and the '
        'complete native image maps onto the card front through exact [0,1] UVs. The runtime default '
        'uses a square 2048 bake with the same mapping; its geometric aspect is restored by the A6 '
        'face. Rectangular `texture_px` support in those three shared tools is needed to reference '
        'the native image directly as the primary part texture. Acrylic is untextured rgba as §1.5 requires.',
        '3. **Shaft facets:** 24 sides at radius 9 mm have 0.0770 mm chord error, conflicting with the '
        'binding 0.05 mm precision. The shaft and lead-in use 32 sides: 0.04334 mm error and at most '
        '64 vertices per piece. After generic compaction these coordinates are restored to 12 decimal '
        'metres with a 0.00001 mm inward numerical guard so rounding cannot oversize the shaft.',
        '4. **Fixture collision count:** true concave V seats use three pieces each, rather than one '
        'convex hull per clamp. Two lower spindle supports, two button bezels and the lamp foot have '
        'additional proxies so the actual external geometry is covered. The shaft lead-in means '
        '4 mm axial length at 30° to the shaft axis. The post is 250 mm above the 30 mm base, with a '
        '25 mm lens above it. The 80–84 mm button stem is enclosed by the bezel apart from a 1 mm gap; '
        'the moving cap uses one cylinder proxy as specified. Explicit cap/body collision exclusions '
        'must be honored by the simulator so the simplified housing proxies do not block the 6 mm stroke.',
        '5. **Foam decomposition:** the 216 pieces preserve open seats and the 45° lead-ins; they '
        'are distributed across six fixed parts (5, 31, 40, 50, 31, 59), all below the 64-piece limit. '
        'The bushing cell is split at Z38. Circular collision openings use circumscribed 16-sided '
        'polygons with at most 0.55 mm extra clearance. Each slot overlaps its cavity to open its full '
        '26 mm width while extending 22 mm beyond the rim. For the L pocket the +X slot is next to '
        'the vertical leg’s inner X face, rather than the empty corner of its bounding rectangle. '
        'The 20 mm bushing depth is total depth from the foam top, including its 5 mm counterbore.',
        '6. **Small-part detail:** relay total height includes 3 mm visual pins; its prescribed box '
        'collision bridges the pin gaps. Terminal wells and fixed fuse-cartridge details use the '
        'specified single-box proxies. The gland has decorative concentric M20 thread crests rather '
        'than a helical screw surface (there is no screwing task); its shallow 0.65 mm cable-entry '
        'socket is omitted from the specified convex cap proxy. The four-ring 16-sided cap proxy '
        'stays within 1 mm of the rounded visual profile. Small pin/slot edge breaks may be below '
        '0.3 mm to fit the feature thickness; primary hard edges are bevelled. Thin 2.5 mm tray shells '
        'use 0.6 mm outer / 0.5 mm inner edge breaks instead of the general 1 mm radius to retain '
        'flat wall stock at the rim. The acrylic support uses 0.4–0.5 mm edge breaks.',
        '7. **Printing and unspecified stock:** raised printed sheets use a 0.02 mm offset to avoid '
        'z-fighting after five-decimal compaction, accounting for 40.020/35.020 mm small-part bounding heights. Tray wall/floor '
        'stock defaults to 2.5/3 mm where unspecified. A6 support is a real 1.8 mm acrylic backing; '
        'the 0.3 mm card is a closed slab for simulator mesh loading and shares a collision envelope '
        'with the backing. To keep the entire [0,1] image on the front, hidden back/edge UVs '
        'are pinned to the white lower-left corner (a deliberate zero-area UV exception for those '
        'uniform-colour faces). The front UV area is exactly one and its faces do not overlap.',
        '8. **Numeric export repairs:** Blender’s 1 μm OBJ quantization can collapse clipping vertices '
        'or make planar hull faces slightly nonconvex. `convexify_task16_collisions.py` re-hulls only '
        'each already-convex proxy using its unchanged exported points; no cavity is bridged and '
        'no vertices move. A volume change over 0.2% is rejected. Detailed records are in '
        '`task16_convexity_checks.json`.',
        '9. **Compaction isolation and resolution:** `prop_lib.finalize_assets()` is called through '
        '`support_task16.finalize_selected()`. The shared compactor updates/deduplicates manifests '
        'outside its `--props` selection, and downsizes medium albedos to 1024. An isolated temporary '
        'root prevents changes to other props, while the final output retains the §1.5 2048 atlases '
        'and 1024 native inspections. The protected shared files were not modified.',
        '10. **Mixed fixture materials:** the fixed body uses one MuJoCo material, favouring the steel '
        'shafts with specular/shininess 0.7 and reflectance 0.05. This also applies to the painted '
        'base in a renderer that permits only one material per rigid part. The source retains '
        'separate powder-coat and turned-steel PBR materials; export previews approximate that '
        'uniform simulator material. No highlights are baked into the albedo.','',
        '## Verification and reproducibility','',
        'All Blender calls used factory startup, headless mode and `-t 6`. Artwork workers and '
        'validators used `$PROP_PYTHON`; only the bundled DejaVu Sans fonts and English text were used. '
        'No git commands were run. Changes are confined to `tools/blender/` and `assets/props/`.','',
        '```sh',
        'blender -b --factory-startup -t 6 -P tools/blender/build_task16_assembly.py --',
        'blender -b --factory-startup -t 6 -P tools/blender/build_task06_shaped_kitting.py -- --props foam_kit_tray',
        'blender -b --factory-startup -t 6 -P tools/blender/preview_task16_exports.py --',
        '$PROP_PYTHON tools/blender/validate_props.py --props '+ ' '.join(IDS),
        '$PROP_PYTHON -B tools/blender/check_task16_assembly.py',
        '$PROP_PYTHON -B tools/blender/report_task16.py','```','',
        'Additional runs used `--geometry-only`, `--skip-existing`, and selected `--props` for '
        'targeted visual corrections. See `task16_validation.txt`, `task16_checks.json`, '
        '`task16_build_resumed.log`, `task16_foam_build.log`, `task16_refinement.log` and '
        '`task16_export_previews.log`, `task16_print_offset.log` (successful final compaction), and '
        '`task16_rgba_preview.log`. The checks exercise exported cavity support surfaces, '
        'notch width/depth/chamfer clearance, shaft radius at 257 azimuths per seat, shoulder/tip '
        'heights, full-face card UV area, image dimensions, and button/lamp metadata. '
        'These are asset-level checks; no scene expert or robot rollout was requested or run.','']
    (HERE/'task16_report.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
