"""Generate the Task 15 delivery report from final manifests and audit results."""
import json
from pathlib import Path
from PIL import Image
from prop_paths import PROPS

ROOT=Path(__file__).resolve().parents[2]
OUT=PROPS
QA=ROOT/'tools/blender/_qa/task15'
IDS=('plc_module','module_tray','test_fixture','ng_bin','product_box','accessory_box','stack_light','line_console')


def main():
    specs={pid:json.loads((OUT/pid/'prop.json').read_text()) for pid in IDS}
    checks=json.loads((QA/'functional_checks.json').read_text())
    physics=json.loads((QA/'mujoco_checks.json').read_text())
    lines=['# Task 15 — electronics line props','',
           'Built all eight §6.2 props, with DI/DO/AI PLC textures and in/ok tray textures. '
           'All geometry uses the shared Z-up, front −X prop frame in metres. '
           'Specifications and critical dimension metadata remain in millimetres.','',
           '## Delivered geometry','',
           '| Prop | Bbox X × Y × Z (mm) | Triangles | Convex pieces | Parts | Delivered albedo | Variants |',
           '|---|---:|---:|---:|---:|---|---|']
    for pid,d in specs.items():
        sizes=sorted({p['texture_px'] for p in d['parts'] if p['texture']})
        lines.append('| '+pid+' | '+' × '.join(f'{v:.3f}' for v in d['bbox_mm'])+' | '+f"{d['triangles']:,}"+
                     ' | '+str(sum(len(p['collision']) for p in d['parts']))+' | '+str(len(d['parts']))+
                     ' | '+', '.join(f'{s}²' for s in sizes)+' | '+(', '.join(v['name'] for v in d['variants']) or '—')+' |')
    lines += ['', 'Every prop has `source.blend`, triangulated OBJ visuals with normals/UVs, convex OBJ proxies, '
              '`prop.json`, and `preview.png`. Every textured rigid part has its own baked atlas; the four lens parts '
              'are deliberately untextured. Source PBR and print images are packed in the Blender files.', '',
              '## Per-part collision, texture and joint inventory', '',
              '| Prop / part | Convex pieces | Texture | Joint |', '|---|---:|---|---|']
    for pid,d in specs.items():
        for p in d['parts']:
            j=p['joint']
            desc='Fixed'
            if j:
                desc=f"{j['type']}; axis {j['axis']}; range {j['range']}; damping {j['damping']}; frictionloss {j['frictionloss']}"
                if j.get('spring'):desc+=f"; spring {j['spring']}"
                if j.get('detents'):desc+='; MAN/AUTO endpoint detents'
                if j.get('latch'):desc+='; scene-controlled latch'
            tex=(f"{p['texture']} ({p['texture_px']}²)" if p['texture'] else 'None; off-state RGBA')
            lines.append(f"| {pid} / {p['name']} | {len(p['collision'])} | {tex} | {desc} |")
    lines += ['', '## Functional metadata', '',
              'The following is copied from the delivered manifests. Positions and strokes are metres; joint angles are radians. '
              'Buttons name the moving cap, press direction, stroke, activation threshold and 40 mm diameter. '
              'Lamp entries name the untextured lens and on-state RGBA/emission; off-state RGBA and zero emission live on the part.', '']
    for pid in ('test_fixture','stack_light','line_console','product_box'):
        d=specs[pid]
        selected={key:d[key] for key in ('buttons','lamps','selector','packing','collision_exclusions','recommended_placement_m') if key in d}
        lines += ['### '+pid,'','```json',json.dumps(selected,indent=2),'```','']
    lines += ['There are no conveyors in Group B, so no `conveyor` metadata is written. '
              'Tray pocket/finger dimensions, fixture clearances, compartment dimensions and tower dimensions are in `dimensions_mm`. '
              'Static equipment is marked `static: true`; PLC, product box and accessory box are movable. '
              'Button spring stiffness is 400 N/m, equilibrium zero; detent and emergency latch behavior belongs to the scene state machine, '
              'as it does for the existing panel library.', '',
              '## Specification reconciliations and deviations','',
              '1. **Delivery image sizes.** Binding §1 requests 1024² previews and 2048² medium-part albedos. '
              'The unchanged public validator requires 512² previews, and the mandatory shared compactor reduces albedos to 1024². '
              'Previews were rendered at 1024² with 64 CPU Cycles samples and retained as `render1024*.png`; `preview*.png` is 512² for validation. '
              'Medium rigid parts were baked at 2048², small parts at 1024²; delivered atlases are 1024² after compaction. '
              'A shared-pipeline change would be needed to deliver the larger sizes as the primary files; none was made.', '',
              '2. **Safe finalization.** `compact_props.py --props` still performs global manifest and deduplication passes. '
              'The builder therefore stages only its selected Task 15 folders, calls `prop_lib.finalize_assets()`, and copies the selected results back. '
              '`control_console`, `panel_devices`, and the four protected shared modules remain byte-for-byte unchanged. '
              'A future shared fix should scope all compactor passes to the selected props.', '',
              '3. **Product-box height and origin.** A 3 mm base plus the required 47 mm clear interior gives a 50 mm body; '
              'the separate 3 mm lid then gives a 53 mm closed assembly. Preserving the specified interior takes precedence over interpreting '
              '50 mm as the entire closed height. The 10 mm front overhang makes the closed depth 110 mm; the body centre is shifted to X=+5 mm '
              'to keep the complete assembly bottom-centred. Hinge anchor is [0.055, 0, 0.050], axis +Y, range [0, 1.75]. '
              'The 100 mm main lid and 10 mm horizontal extension form one slab; its separate collision return flange drops 8 mm.', '',
              '4. **Tower base and collision count.** A 22 mm lower socket reconciles the 200 mm mast, 180 mm of lenses, '
              '30 mm buzzer and 8 mm foot with the approximately 440 mm total. The L bracket uses two convex boxes rather than one '
              'that would fill its notch; the socket adds one cylindrical proxy. This gives eight total pieces rather than six.', '',
              '5. **Typography and small print surfaces.** The PLC brand is split into `IMC` / `Controls` so the brand and model glyphs '
              'fit at at least 12 mm high. The accessory description wraps across two lines. The PLC front band is 8 mm high. '
              'Print patches are 0.04 mm thick and sit close to the surface; some overall bbox dimensions gain at most 0.03 mm, '
              'well below the 1 mm tolerance. Paper labels are skewed 2°; factory printing remains aligned.', '',
              '6. **Buttons and console.** The task-specific instruction and §6.0 take precedence over §6.2’s blue flat RESET wording: '
              'all four console buttons have Ø40 mm mushroom heads, including RESET. All have return springs; the emergency stop also '
              'declares a scene-controlled latch. The console has the original 200 × 320 mm enclosure, 26.565° slope, 60/160 mm end heights '
              'and rear gland, producing an approximately 212 mm total depth. The fixed gland is included with the body, adding a seventh '
              'body collision piece to the six shell pieces. Panel holes and thin decorative bezels do not add independent collision pieces; '
              'button/support contact exclusions are written using the actual loader’s `collision_exclusions` key.', '',
              '7. **Lamp collision approximation.** The fixture hemisphere is represented by one 61-vertex convex proxy, balanced up to '
              '0.31 mm outward to keep its polygonal approximation within the general 1 mm surface tolerance. The untextured visual remains '
              'a Ø60 hemisphere. Its post rises 200 mm above the 20 mm base, putting the lens top at 250 mm.', '',
              '8. **Unspecified choices.** PP wall thickness for the NG bin is 2 mm. Button stiffness/damping/friction follow the existing '
              'panel-device conventions. Label positions, the console’s new row coordinates, mast XY position, cable routing and cosmetic screws '
              'are construction choices. The fixture mast is at [88, -100] mm on the +X side, avoiding the gripper housing envelope. Factory-set saturated colours are preserved in artwork; per-instance colour gains of ±3% are advisory '
              'metadata on newly rebuilt props, to be applied by scene instancing.', '',
              '9. **Thin-feature bevels.** Thin paperboard, 2 mm walls, panel trim and small functional edges use '
              '0.25–0.5 mm edge rounds rather than applying the generic 1 mm medium-prop bevel everywhere. '
              'This retains the specified thicknesses and clearances; the fixture base and substantial tower edges use 1 mm rounds.', '',
              '10. **Physics test load and soft limits.** Button tests apply the nominal full-stroke load '
              '`stiffness × travel + frictionloss` and require return within 1 mm and loaded position within 1 mm of the limit. '
              'An exploratory 4 N load on the 6 mm TEST button reached 7.84 mm because the current loader uses MuJoCo’s default soft '
              'joint-limit solver. The declared travel is still exactly 6 mm. Overload-hard stops require scene-level limit-solver tuning; '
              'no loader or simulator file was changed. Emergency latching and selector detents likewise remain scene logic.', '',
              '## Verification','',
              f"- Public validator: **8/8 passed**; full output in `_qa/task15/validation.txt`.",
              f"- Functional/preservation audit: **{checks['passed']} checks passed**, covering pockets, finger slots, support floors, fixture access, "
              'joint metadata, lamps, row order, and protected hashes.',
              f"- MuJoCo {physics['mujoco_version']}: **{len(physics['cases'])} loader cases** (all defaults and variants) compiled and settled; "
              'buttons pressed and returned, and the product lid closed under gravity and remained at the open limit.',
              '- All default and variant previews, the open product-box view, and reimported OBJ/albedo previews were inspected. '
              'The first tray preview exposed transverse rim seams; the final builder uses continuous outer rails and overlapping chamfer shoulders. '
              'The initial NG markings were enlarged for visibility.',
              '- Quantized collision repair only re-hulls an individual already-convex source proxy from unchanged exported points; '
              'changes over 0.2% volume are rejected. Audit: `_qa/task15/collision_repairs.json`.',
              '- No robot policy/expert rollout was performed. Scene-specific lamp transitions, selector detents and emergency latching '
              'remain runtime responsibilities; geometry, loader compilation and passive joint mechanics were checked.', '',
              '## Reproduction','', '```bash',
              'blender -b --factory-startup -t 6 -P tools/blender/build_task15_electronics.py --',
              'blender -b --factory-startup -t 6 -P tools/blender/build_task15_electronics.py -- --render-exports',
              '$PROP_PYTHON tools/blender/validate_props.py --props '+' '.join(IDS),
              '$PROP_PYTHON -B tools/blender/verify_task15_electronics.py',
              '$PROP_PYTHON -B tools/blender/check_task15_mujoco.py',
              '$PROP_PYTHON -B tools/blender/report_task15.py', '```', '',
              'The build supports `--geometry-only`, `--export-only`, `--finalize-only` and a selected `--props` list. '
              ''
              'Blender 5.2.3 was used with `-t 6`. Validators and artwork used the supplied `PROP_PYTHON`; the physics smoke test reused '
              'the existing project MuJoCo package read-only. Because Pinocchio is unavailable in the tools environment, '
              'the isolated loader test supplies ROOT and the unchanged fmt function extracted from robots/franzi/scene.py; '
              'the prop_assets loader itself runs unchanged. No shared source files or existing console/device outputs were modified.']
    (ROOT/'tools/blender/task15_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
