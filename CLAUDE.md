# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a local MuJoCo simulation (macOS) of the Inovance/Franzi wheeled dual-arm robot. It is phase 1 of adapting FineART. A privileged, scripted IK expert performs a bimanual pick-and-place task (`scenes/pick_place/`) and records demonstrations:

- The base is fixed.
- Both arms grasp a box at the same time (left: blue, right: orange), lift it, place it on a wide target pad, release it, and retract.

No FineART weights or inference are loaded yet. The old OMX 6-D policy can't be reused as is: it needs rebuilt feature/normalization configs and fine-tuning.

Three more long tasks use the RoboSynChallenge props (`scenes/robosyn_*`, see "RoboSyn object scenes" below). A second, longer task for VLA training, dual-arm sort-and-kit, lives in `scenes/kitting/` (see "the sort-and-kit task" below). It reuses the robot scene, `Scene`'s audits and recordings, and the 16-D data contract.

The code is in packages run as `python -m <package>.<script>` from this directory: `sim/` holds what every scene shares (the derived robot scene, the policy cameras and the `Scene` episode framework), `scenes/` one package per simulation scene (see "Scenes" below), `sdk/` the stock-SDK tools, `data/` the VLA data tools, and `robot_kinematics.py`, the shared kinematics, stays at the top level. The vendor-SDK bridge is the `simbridge/` package (see "SDK bridge" below) with an engine-free pytest suite in `tests/`. The docs are in Chinese. `README.md` is the short public landing page: demo previews, quick start, layout, current status and links. Keep it that way (user request, 2026-10-24) and put details and new findings into the matching file in `docs/`:
- `robot_model.md`: the source files, the derived scene and colliders, the gripper linkage, kinematics, joint limits and servos, cameras, appearance and lighting; also the planning constraints under `config/`.
- `tasks.md`: each scene's layout, expert, success thresholds and current results, plus the shared rules and outputs.
- `sdk_bridge.md`: simbridge and the vendor SDK under `sdk_review/`. `vla_data.md`: VLA recording, conversion and export. `controller_image.md`: the offline controller-image notes.
- `history.md`: every earlier version with its results and findings. Most of those run folders were deleted.
- `docs/media/`: the README's animated WebP previews, cut from recorded runs; `SOURCE.json` records how. Regenerate them when a scene's look or an expert changes.

The project is a git repository (branch `main`). Its remote is the public GitHub repository `heliang-pu/franzi-sim`, which the user created on 2026-10-24. The local folder keeps its old name. Everything committed is public.
- The repository has no license, on purpose: all rights reserved. The user wants others to view the code but not reuse it; GitHub can't block downloads.
- Commits use a repository-local identity: the name heliang-pu (user decision) and GitHub's noreply address, so no personal email enters the public history. The global git identity is someone else's.
- `.gitignore` keeps out `outputs/`, the rebuildable appearance caches, `assets/derived/` and `sdk_review/`. Never commit `sdk_review/`: it is third-party vendor code, and the controller image holds credential-like fields.
- A fresh clone therefore lacks them. `sdk/stock_sdk.py`, the stock-SDK acceptance, `sdk/check_sdk_kinematics.py` and the tests that read the controller image need `sdk_review/` copied in from this machine.

## Commands

Run from this directory with the project's own conda env, `/Users/puheliang/miniforge3/envs/inovance-fineart` (created 2026-10-24 at the user's request). It has Python 3.12.15, and from PyPI mujoco 3.13.0, numpy 2.2.6, scipy 1.18.1, trimesh 5.1.0, av 15.1.0, pillow 12.3.0, PyYAML 6.0.3, networkx 3.6.1, pytest 8.4.2 and pin 3.4.0 (Pinocchio).
- The other packages match `piper-tactile`, and runs there are bit-identical. `piper-tactile` has no Pinocchio, so `robot_kinematics` (and everything importing `build_task` or `episode`) no longer runs in it.
- `cmeel-urdfdom` must stay at 4.0.1: pin 3.4.0 links urdfdom 4.0, and pip picks 6.0.0 unless pinned.
- To recreate: `mamba create -n inovance-fineart -c conda-forge python=3.12 pip`, then `pip install` the versions above plus `cmeel-urdfdom==4.0.1 cmeel-tinyxml2==10.0.0`.

```bash
# Pick and place: headless episode, a few seconds. --output must not exist yet.
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m scenes.pick_place.run --output outputs/<new_run>
# Same, plus MP4 recording (about a minute)
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m scenes.pick_place.run --output outputs/<new_run> --record
# Acceptance check of a recorded run: writes <run>/verification.json, raises AssertionError on failure
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m scenes.pick_place.check outputs/<run>
# Derive the robot scene only (robot, table, lights): writes outputs/robot_scene.xml and outputs/collision_setup.json
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.build_task
# Derive the pick-and-place scene only: writes outputs/pick_place_task.xml
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m scenes.pick_place.task
# Sort-and-kit: one episode (about 4 s headless, about 3 min with --record)
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m scenes.kitting.run --output outputs/<new_run> --seed 0
# Its acceptance across seeds: expert runs, camera pixel counts, previews -> report.json
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m scenes.kitting.check --output outputs/<new_dir> --seeds 20
# RoboSyn object scenes (kitchen, drawer, two-handed tray): one episode headless ~15 s,
# --record ~4 min; checks take ~5-10 min for 20 seeds. Variants: kitchen cook|swap, drawer store|retrieve.
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m scenes.robosyn_kitchen.run --output outputs/<new_run> --seed 0 [--variant swap] [--record]
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m scenes.robosyn_drawer.check --output outputs/<new_dir> --seeds 20 [--variant retrieve]
# Lint and format the project's own code. No config file, so Ruff 0.16 defaults apply
# (88 columns, isort enabled). Not `.`: sdk_review/ holds vendored third-party Python.
ruff check robot_kinematics.py sim scenes sdk data simbridge tests && ruff format robot_kinematics.py sim scenes sdk data simbridge tests
# SDK bridge: engine-free tests (seconds), then the stock-SDK acceptance per backend
# (a few minutes each with --record; --output may already hold other backends)
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m pytest tests -q
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sdk.verify_backends --output outputs/<run> --backends mujoco,mujoco_owned_loop --task --fault-probe --record
```

- The experts have no unit tests. Validate one in stages: `sim/build_task.py` and the scene's `task.py`, then a headless run, then a `--record` run checked with the scene's `check.py`.
- A failed episode still exits 0. `RuntimeError` and `ValueError` raised inside `expert()` are caught and written to `summary.json` (`success`, `error`), so check that file. Any other exception, and any failure while building `Scene` (including the camera-mount asserts), crashes without writing a summary.
- Run Ruff from this directory. From anywhere else it treats the local modules as third-party and reports false isort errors. The workspace `AGENTS.md` sets a 110-column limit, but that comes from lerobot's config and doesn't apply here.
- Each retained run keeps its stdout next to the run folder as `outputs/<run>.log`.

## Architecture

Imports (as `from sim.<module> import ...` and `from scenes.<scene>.<module> import ...`; `robot_kinematics` is top level):
- `sim.episode` uses `build_task`, `camera_rig`, `gripper_linkage` and `robot_kinematics`.
- `camera_rig` uses `build_task`; `camera_rig`, `appearance` and `build_task` use `wrist_mount` (NumPy only, also used by `hardware/wrist_d405_mount/`).
- `build_task` uses `appearance` (which uses `fittings` and `gripper_linkage`), `gripper_linkage` and `robot_kinematics`. `gripper_linkage` needs only NumPy.
- `scenes.pick_place`: `task` uses `build_task` and `gripper_linkage`; `run` uses `task`, `episode` and `camera_rig`; `check` uses `task` and `build_task`.
- `scenes.kitting`: `task` uses `build_task`, `gripper_linkage` and `robot_kinematics`; `run` uses `task`, `episode` and `camera_rig`; `check` uses `run`.
- Outside the two: simbridge's MuJoCo backend, `sdk/make_sdk_program.py`, `sdk/convert_pybullet_assets.py` and `hardware/wrist_d405_mount/check_mount.py` build `scenes.pick_place.run.PickPlaceScene`, and `sdk/verify_backends.py` scores against `scenes.pick_place.task.GOALS`.

`sim/build_task.py` holds the shared constants: `TABLE_Z`, `PART_RGBA`, `ROOT`, and `fmt`. It also re-exports `ARM_JOINTS`, which fixes the joint order and is defined in `robot_kinematics.py`. `sim/appearance.py` defines its own `ROOT` to avoid a circular import. Each scene's layout lives in its own `task.py`.

**`scenes/`: one package per simulation scene (user request 2026-10-24: many more scenes will follow).** A scene keeps its layout and objects, its planning (the scripted expert and any sampler), its acceptance and its assets together; everything shared stays in `sim/`. `scenes/__init__.py` restates the convention.
- `task.py`: the layout and `build()`, which adds the scene's objects to `build_task.build()` and returns `(root, report)`. Name object bodies `<name>_part`: `Scene.robot_bodies` excludes that suffix.
- `run.py`: the `Scene` subclass, the expert and the episode CLI.
- `check.py`: the scene's acceptance.
- `assets/`: meshes, textures and maps that only this scene uses, referred to by absolute path (`Path(__file__).parent / "assets"`), since the compiler's meshdir is the robot's. Third-party assets that may not be published stay out of git: the repository is public.
- Scenes import `sim` and `robot_kinematics`, never each other.
- The move into `scenes/` (2026-10-24) changed no run: pick and place and kitting seeds 0 and 1 reproduced `scene.xml`, `trace.json`, `trajectory.npz` and `episode.json` bit for bit, and the stock-SDK acceptance passed.

**The industrial benchmark (docs/benchmark.md, user request 2026-10-24).** Ten scenes, three easy, four medium and three hard: `tote_transfer`, `panel_operation`, `tray_push`, `parcel_sorting`, `carton_carry`, `shaped_kitting`, `lidded_tote`, `palletizing`, `cnc_tending` and `precision_insertion`.
- Each has `task.py` (layout, `build()`, `KEY_POINTS`) and `check.py`. Only `tote_transfer` has an expert: `run.py`, a seeded `sample_episode()` that `build(spec)` takes (`build()` alone is the nominal layout, for the scene check) and a success test. The others were stopped half-way when the user moved demonstration collection to Isaac Sim (2026-10-24). docs/benchmark.md records what that work measured: reach, grip widths, clearances, and layout problems in `palletizing` and `parcel_sorting`.
- Their props were built in Blender by Codex from `docs/benchmark_props.md`: `tools/blender/` holds the build scripts, `assets/props/<id>/` the output (OBJ parts, baked textures, convex collision pieces, `prop.json`). Rebuild props there, never by hand-editing the OBJs.
  - Every build script ends with `prop_lib.finalize_assets()`, which runs `compact_props.py`: textures at most 1024², shared atlases deduplicated, OBJ coordinates rounded. `assets/props/` is about 90 MB without the `source.blend` files (54 MB).
  - `shaped_kitting` loads the foam tray's `light_grey` variant (user request 2026-10-24): the cameras can't make out the pockets in the black foam.
- `sim/prop_assets.add()` loads a prop: one `_part` body per part, nested as `prop.json` says, every part in the prop frame. Visual geoms are group 2, massless and contact-free. Convex collision pieces are group 3, with the part's mass spread by volume. Joints follow `prop.json`. A free prop's joint is `<name>_part_joint`, the name `sim.tabletop.ObjectScene` looks up. Detents, latches and the CNC jaw clamp are not modelled, and `recommended_initial_qpos` and the lid hinge's friction-hold settings are not applied yet.
- `sim/workcell.industrial_cell()` swaps build_task's floor and table for the factory floor, the workbench (top at `TABLE_Z`, front edge x = 0.20) and a fence-and-rack background behind the overview camera.
- `sim/bench_check.py` settles a scene for 4 s under the Scene audits, checks free-prop drift and reaches every key point with the shared IK (top-down or the 45° `sim.tabletop.grasp_rotation`), and renders `preview.png`. All ten passed on 2026-10-24. Its key points test reach only, not whether the 95 mm jaws and the housing fit the object there; the instruction comes from the build report.
- Layout limits: a top-down grasp reaches x ≤ 0.38 at table height, the 45° grasp about 0.45–0.51. Each arm stays on its own side of y = 0. The head camera sees the table from x ≈ 0.33.

**`robot_kinematics.py`: the one shared FK/IK ("real follows sim", user decision 2026-10-24).** Cartesian goals are solved here on the host; the simulation and the real robot both receive joint commands only (`MoveJ`, joint teleop). The controller's own IK, `MoveL` and Cartesian teleop are deliberately unused. simbridge's IK/Fk endpoints call this module; `simbridge/sdk_kinematics.py` only converts SDK conventions.
- FK and Jacobians come from Pinocchio (user decision, 2026-10-24). The module needs NumPy, SciPy, PyYAML and pin, all in the `inovance-fineart` env, but no physics engine. Keep MuJoCo and `build_task` out of it.
- Pinocchio's model is the source URDF, with base_link offset as in `franzi.xml` and an operational frame per `{side}_grasp` tool point from `config/robot/franzi_sdk.yaml`. Leg and waist joints come from `body` (`BODY_JOINTS`); every other joint is neutral. FK equals the scene's to 1e-15 m.
- `Kinematics.ik()` is the expert's IK: multi-seed `least_squares` on position (×10) and the rotation vector to a fixed world-aligned grasp orientation, inside the aligned limits.
  - The Jacobian is exact: Pinocchio's frame Jacobian, with `Jlog3` for the rotation rows. It lands within 5e-8 rad of the old finite-difference solutions, and is 6× faster (single target about 6 ms, chained waypoint 1.3 ms).
  - When chaining waypoints (`near=`), a residual `(q - near) * 0.002` keeps the redundant elbow swivel near the previous solution; without it the arm visibly wobbled along straight moves. 0.02 is too strong (the approach IK becomes infeasible).
  - `line()` gives the 1 cm waypoint chains, and `to_sdk()`/`from_sdk()` map URDF ↔ SDK joint values.
  - `Kinematics(margin=...)` keeps solutions that far inside the aligned limits. The default is `PLANNING_MARGIN`, 2° (user decision, 2026-10-24). With the old 0.0001 rad, the pick-and-place approach parked both wrist pitches 0.065° from the controller's −60° limit, where tracking overshoot on the real robot could trip its soft limit. At 2° they stay 2.06° away, and the grasp orientation is unchanged.
  - simbridge's controller-emulating IK endpoints pass 0.0001. The kitting expert uses 10 mrad and its sampler 20 mrad: planned on a limit, the shoulder roll overshot the 2 mrad audit by up to 12 mrad, and its inner cells put the shoulder roll within 2° of its limit.
- `controller_rates()` returns the arm velocity and acceleration limits stored in the controller image, plus its default 30% speed ratio; `Scene.move` keeps the expert within them. They are not verified on hardware.
  - A seed whose rotation error reaches π (no finite `Jlog3`) is skipped instead of raising `ValueError`.
- Joint limits are `controller_ranges()`; `build_task.align_joint_limits()` writes the same numbers into the scene.

**`build_task.build()`: the derived robot scene.** Every run parses `assets/franzi_source/franzi.xml` with ElementTree, edits the tree in memory, and returns `(root, report)`. Never edit `assets/franzi_source/`. It is an unmodified copy of the source project on the 4090 machine, and `summary.json` records the URDF's sha256. The edits:
- The base is fixed (its freejoint and the keyframe are removed). Physics runs at 500 Hz with elliptic friction cones.
- In the source model, every link's collider is a bounding box. The build changes two of them:
  - Each wrist box becomes one convex hull per connected CAD component of the wrist STL, 47 per side. This keeps the jaw opening clear.
  - The original D405 block and its two M3 screws are left out of those hulls (picked as face-connected parts: by shared vertices the block is one component with the L bracket). The tilted block that replaces them (`sim/wrist_mount.py`) adds three convex pieces and a 42 × 42 × 23 mm box for the D405 it holds.
  - The finger boxes become the finger STL meshes (MuJoCo collides with their convex hulls) with grip contact parameters.
  - Every other link keeps its box, so this is not a full CAD collision model.
- The gripper linkage moves as in the CAD (`sim/gripper_linkage.py`, user request 2026-10-24).
  - The source model froze the links and jaw blocks open inside the wrist mesh and slid only the pads, so closed pads hung in the air.
  - In the CAD each jaw block rides a 55 mm parallelogram, so the pads drop as they close: 14.5 mm on the 45 mm boxes, 15.1 mm on the 40 mm kitting parts, at most 16.3 mm.
  - The finger x-slides stay the only actuated gripper joints and still measure jaw travel, so closure, the 16-D contract and the SDK gripper map are unchanged.
  - `add_linkage()` runs before the appearance. It gives each finger a z-slide, and each link and the motor slider their own body and joint: 26 joints in all.
  - Each of those joints follows its finger's x-slide through a quartic joint equality (solref 0.004), fitted to the exact linkage solution (`solve()`) to 0.06 mm and 0.08°.
  - The jaw blocks ride in the finger bodies. `group_of()` picks the moving CAD components by bounding box; the appearance (`"detach"`) and the wrist hulls (23 of the 47 per side) put them in those bodies.
  - `Scene.held` gives the coupled joints gravity feedforward: the real lead screw doesn't back-drive.
  - The finger servos (kp 1500, ±6 N) still yield to a load of over 6 N along the arc, e.g. fingertips pushed into the table, where the real screw would hold. Keep the fingertips off the table.
  - `{side}_grasp` stays the open pads' centre; closed pads sit `contact_drop(width)` below it along the fingers.
- Contacts are filtered only by contype/conaffinity bits. The source model has no `<exclude>`.
  - The source `franzi` class (1/2) collides with the world but never with itself.
  - Arm colliders are re-masked to 5/10 (left) and 9/6 (right). As a result, left-arm vs right-arm contact is on, contact within one arm and between an arm and the torso or head stays off, and robot-world contact stays on.
  - "Arm" means bodies whose names start with `left` or `right` and don't contain `front`, which excludes the front wheels.
  - Only geoms that carry their own `class="collision"` attribute are matched. ElementTree doesn't see a class inherited through `childclass`.
- The build adds the `{side}_grasp` IK sites, the table and the lights; each scene's `task.py` adds its objects. It keeps the source model's name, and each scene sets its own (pick and place keeps `franzi_dual_arm_kitting`, the name its `scene.xml` always had). `PART_RGBA` holds the blue and orange part colours the scenes share.
- `align_joint_limits()` clips joint ranges, and the matching actuator ctrlranges, to the controller limits stored in the controller image. It reads them through `config/robot/franzi_sdk.yaml`, the same file simbridge uses, which also holds the SDK ↔ URDF sign and offset mapping.
  - The range becomes the intersection with the URDF. Sides where the controller is tighter are pulled in by the SDK's 0.001 rad margin, so the expert never plans to a bound the SDK layer rejects.
  - Where the controller is wider than the URDF (elbow upper 15° vs 7.45°, thigh lower −10° vs 0°, waist pitch lower −75° vs −30°), the URDF bound stays. Widen those only after checking the mechanical range on the real robot.
- Shadows come from one directional key light (`studio_key`). MuJoCo centres a directional light's shadow box on the beam through the light's `pos`, so the key's pos is chosen to put that axis on the workspace; its `dir` sets the lighting.
  - `shadowclip` 0.63 (box half-size 1.44 m) and `shadowsize` 8192 give 0.35 mm shadow texels over the robot, the table and the floor they shade. The previous 4096 map over a 4.6 m box made sawtooth shadow edges.
  - If you move the key or the scene, recompute the box. With the old pos (2, −2, 3.5), any shadowclip below about 0.8 cut off up to half of the shadows. Check by comparing renders with `mjRND_SHADOW` on and off.
  - Shadows stay hard-edged. Jittered multi-light soft shadows were tried and rejected: they band visibly and render 2–3× slower.

**`sim/appearance.py` and `sim/fittings.py`: visuals only, matched to the real robot.** They must not affect physics: every geom they add has contype and conaffinity 0 and mass 0. The reference is the lab robot, including the grippers and the RealSense housings.
- Painting. Each visual mesh in `RECIPE` is split into connected CAD components, sorted largest first. Each component gets a `PALETTE` material by index, with `"rest"` covering the others.
- `"parts"` rules repaint whole components chosen by bounding-box size. Mirrored arms share these sizes, though their index order can differ; the gripper's jaw blocks, pins and screws use them.
- `"faces"` rules repaint regions within one component: the thigh crossbar, the upper-arm grille pad, and the shoulder and inner elbow caps. A material of `None` drops faces; the head D435 uses this to drop the URDF camera box that is merged into its bracket.
- `attach_real_cameras()` swaps the URDF camera boxes for the official RealSense meshes in `assets/appearance/cameras/` (see each `SOURCE.json`):
  - D435: Menagerie's coloured parts, rotated 90° about z and offset by `D435_ALIGN`, so the left IR imager is at the sensor origin and the glass is 4.2 mm ahead, as in realsense-ros.
  - D405: realsense-ros `d405.stl` at the same frame, carried onto the tilted wrist block like its camera (`wrist_mount.sensor_pose`).
- The tilted wrist D405 block (`hardware/wrist_d405_mount`, user decision 2026-10-24) replaces the original one: the wrist recipe's `"drop"` rule leaves out the old block and its screws, and `attach_wrist_blocks()` adds the printed part's STL, black, on each clevis pin (the right hand's turned 180° about z).
- The camera origins sit behind the glass, so the policy cameras rely on the 45 mm render `znear` not to see their own housings.
- `attach_chest_slot()` fills the chest camera slot (`CHEST_SLOT`, in the body D435 sensor frame) with a pearl bezel plate and a tinted 82 × 10.5 mm window, 0.4–1.0 mm ahead of the glass. The shell's opening (85.1 × 20.8 mm) otherwise shows the whole D435 front, while the real robots show a narrow dark window. These are inline meshes inside `znear`, so the body camera never sees them.
- The face visor has no camera window. The head D435 sits 16–20 mm behind it, inside the 45 mm `znear`. Removing the whole visor, or the chest decal, from the render changes 0 pixels in all four policy cameras. A cut window with a rim and shroud was built and then dropped at the user's request (2026-10-24), because it changed nothing in the recordings.
- Rendering is deterministic, but recorded MP4s are not a pixel-level comparison. A visual geom that changes `m.stat.extent` moves a few shadow pixels, and H.264 spreads that to roughly 0.5% of pixels. Compare raw `mujoco.Renderer` frames instead.
- Cap rules keep only faces wholly inside a CAD vertex ring and connected to the cap centre.
- The outer elbow cap's triangles cross every face-level cut, so it is drawn by a texture instead: a conformal decal on the pearl elbow mesh with a black disc and the white logo.
- URDF sensor sites are moved to group 4 so their marker spheres don't render; `camera_rig.assert_mounts()` still uses them.
- Each `(mesh, material)` group becomes one visual geom. It keeps the original geom's `pos`/`quat` and has an `inertia="shell"` mesh asset. If the CAD changes, `paint_groups` asserts rather than leaving a component unpainted.
- The wrist recipe's `"detach"` rule (`gripper_linkage.group_of`) splits the moving gripper links and jaw blocks into their own groups, which go into the bodies that move them, offset by those bodies' origins.
- Caches live in `assets/appearance/{painted,wordmark,fittings}/`. The paint key covers only the source STL bytes and `PAINT_VERSION`, so bump `PAINT_VERSION` whenever a face rule changes. Older files directly under `assets/appearance/` (`head_*.stl`, `head_parts.json`, `inovance_wordmark.png`) are unused leftovers.
- The chest decal is a copy of the torso front-shell triangles (component 2) lifted 0.4 mm along MuJoCo's compiled normals. MuJoCo ignores texture alpha, so the texture background is `PEARL_SRGB`, derived from `fa_pearl`, and only the print shows. Don't add face rules on the front shell near the print: the decal's normals must match the pearl geom's.
- The chest print is the factory robot's, from top to bottom: INOVANCE (`WORDMARK`), "Powered by", FMC³ and a tracked ROBOTICS line (`FACTORY_PRINT`). It was measured on frontal factory-video frames, using the INOVANCE width as the ruler.
  - The FMC³ letters are drawn from `FMC3_GLYPHS`, read off a 30-frame stack of the factory video: a floating bar over a Γ, a V-shaped M, a C with two dot eyes, and a small 3 on the cap line.
  - `assets/franzi_source/assets/fmc3_logo.png` is a different logo version (comb M, one square in the C). Only its 3 is used.
- `sim/fittings.py` adds the `fit_*` parts the CAD lacks: the lower-back ribbed panel (a harmonic fill of the opening's rim), the upper-back e-stop, ports and lights, the thigh back cover, and the base light strips. They are placed on surface fits of the shells and reference `PALETTE` material names only.
- Fonts are macOS paths: Avenir Next for the chest and elbow text, and Arial in `Scene.render()` (`sim/episode.py`) and the kitting preview sheets.

**`sim/camera_rig.py` and `sim/camera_rig.json`: the policy cameras.**
- `POLICY_CAMERAS` are the head and body D435 and the left and right wrist D405. Each is attached to the URDF sensor site of the same name.
- Camera axes are the site frame, rotated by the optical roll from the JSON, then 180° about x. The flip is needed because the sensor frame looks along +Z while MuJoCo cameras look along −Z. `assert_mounts()` checks the result after forward kinematics.
- The D435 rolls are −90°. The wrist D405 rolls follow the CAD: left 0°, right 180° (user decision, 2026-10-24).
  - With them, image right runs along the jaw axis and image up points outboard.
  - The reasoning: each D405 is held by its rear M3 pair, which is parallel to its stereo baseline and, in the wrist CAD, runs along the jaw axis. The user confirmed that on the real robot the fingers appear left and right.
  - The old −90° put the fingers at the top and bottom of the image.
- The intrinsics are uncalibrated simulated pinholes.
- Change a camera's view only by moving joints. For example, the head gaze is a `head_pitch_joint` actuator command in `sim/episode.py`. Never re-aim a camera or detach it from its site.
- Exception: the wrist D405s sit on the tilted wrist block (user decision 2026-10-24; `"mount": "wrist_block"` in `camera_rig.json`). Each is its URDF site times a fixed bracket from `sim/wrist_mount.py`, which reads `hardware/wrist_d405_mount/params.json`: 20° toward the jaws, 34 mm below the clevis pin, 27 mm outboard, like the user's reference model. `assert_mounts()` checks camera = site × bracket. The bracket, the D405 mesh, the block's visual and its colliders all come from that module; change them only together.
- The world cameras added in `sim/episode.py` (`overview` and others) are for display only and are never policy inputs.

**`sim/episode.py`: the episode (`Scene`), shared by every scene.**
- `Scene` installs the cameras, compiles the model, sets the render `znear` to 45 mm so each camera sees past its own housing, and writes the result to `scene.xml`. Two overrides are applied only to the compiled model and are missing from `scene.xml`: the finger actuators get kp 1500 and a force range of ±6 N, and the `TORSO_SERVOS` (calf, thigh, waist pitch and yaw) get velocity feedback `kv` for a damping ratio of 0.7 at the start posture. Without `kv` the source servos (kp 6000, damping 5, ζ ≈ 0.01) let the upper body rock at about 1.5 Hz after every arm move (user decision, 2026-10-24). Don't raise it to ζ = 1 without re-testing: before the 2° planning margin, the expert parked the wrist pitch on its bound, and the changed posture tripped the 0.002 rad limit audit.
- Physics runs at 500 Hz and control at 25 Hz. On every 20th physics step the code appends one trace sample (the measured state plus the action about to be applied), renders one frame per video, and updates the stable-frame counter. Joint goals are held constant between control ticks.
- The robot is driven only through position-actuator `ctrl`, plus gravity compensation via `qfrc_applied = qfrc_bias` on the arm DOFs, on the `TORSO_SERVOS` and `HEAD_SERVOS` DOFs, and on the gripper linkage's coupled joints (`Scene.held`).
  - On kp alone the leg and waist servos sagged up to 7.5 mrad each (about 1.3° in all) with the arms forward, and the head pitch 3.8 mrad. That shifted the head and body cameras, and put the hands about 9 mm off when expert joint targets were replayed in PyBullet.
  - With the feedforward the error is 0 at rest, and at most about 1 mrad during moves (user decision, 2026-10-24). Head pitch still lags about 5 mrad while its 1.5 s gaze ramp is moving.
- After every physics step, a penetration over 1 mm or a joint-limit overshoot over 0.002 raises `RuntimeError` and ends the episode.
- `Scene.ik()` delegates to `robot_kinematics.Kinematics.ik()`, passing the live arm pose and leg and waist joints; it never touches the live state. `sdk/make_sdk_program.py` calls it. Change IK behaviour in `robot_kinematics`, not here.
- `move()` follows a quintic minimum-jerk profile. By default it interpolates straight between joint-space IK solutions. With `linear=True` it runs through `Kinematics.line()`: IK waypoints 1 cm apart on the Cartesian line, each chained to its predecessor's branch. The pick-and-place expert uses `linear=True` for align, lift, transfer, place and retract: near the edge of the arm's reach, plain joint interpolation swings the hand about 2 cm forward mid-descent, and a finger lands on the box.
- `move()` then lengthens each move (`stretch()`) until its 25 Hz arm goals stay within `RATE_HEADROOM` (0.9) of `controller_rates()` scaled by the default 30% speed ratio. A replay on the real robot at that ratio is then not clipped (user decision, 2026-10-24).
  - In pick and place only ready and approach stretch: its reference episode went from 24.5 s to 29.9 s, with peaks at 0.90× (velocity) and 0.48× (acceleration) of those limits.
  - A move's `speed=` (used by the kitting task) applies first. Moves already within the limits keep their duration.
- The thigh's straight-leg target of 0 sits on its lower joint limit, so fast arm swings can push it past the 0.002 limit check. That is why pick and place's final retract takes 3 s.
- `Scene` writes `scene.xml`; each scene's `run.py` adds `summary.json` (result, physics stats, data contract, and collision, appearance, and camera metadata), `trace.json`, and `trajectory.npz`.
- `--record` adds `expert.mp4` (overview plus wrist insets), `cameras.mp4` (2×2 mosaic of the policy cameras), and one 640×480 MP4 per policy camera. All videos are 25 FPS with exactly one frame per trace sample.
- The base `Scene` builds the robot scene alone: it samples no objects, tracks nothing and never completes. A scene subclass overrides `build_root()` (scene tree and report), `object_info()`, `sample()` (one trace record), `track_objects()` (per-step grasp evidence, in `lift_evidence`) and `complete()`; `PickPlaceScene` and `KitScene` do. `ik(rotation=)` and `move(rotations=)` take a hand rotation per arm. `move(speed=)` times the waypoints by joint travel (largest joint step) and stretches the move until no joint's peak command rate exceeds `speed`. The defaults keep the pick-and-place expert bit-identical (trajectory, trace and scene.xml checked).

**`scenes/pick_place/`: the pick-and-place task.** `task.py` holds `SOURCES`, `GOALS` and `PAD_DROP` and adds the two 45 mm boxes (`left_part`, `right_part`) and their wide target pads to the robot scene. `run.py` holds `PickPlaceScene` (per-side grasp evidence and the success test) and the expert; `check.py` is the acceptance. simbridge's MuJoCo backend, `sdk/make_sdk_program.py`, `sdk/convert_pybullet_assets.py` and `hardware/wrist_d405_mount/check_mount.py` all build this scene.
- `expert()` runs the scripted stages ready → approach → align → grasp → lift → transfer → place → release → retract → finish. Along the way it checks that both jaws touch each box and that the boxes are lifted high enough. Each stage's name and instruction become the trace's `phase` and `subtask` fields.
- It grasps and places `PAD_DROP` (14.5 mm, the pad drop on the 45 mm boxes) higher than the box centre, so the closed fingertips stay 14.1 mm above the table, as before the linkage model. Without it they pressed about 0.4 mm into the table.
- `check.py` re-checks a recorded run folder:
  - the success thresholds
  - the structure of `scene.xml`: no `<weld>`, exactly 2 freejoints, mesh colliders on the wrists (counted across the wrist's subtree, where the linkage hulls ride) and fingers
  - each video's frame count, FPS, and width
- Video names come from `contract.cameras` in each run's `summary.json`. Runs from before the 4-camera rig, such as the first demo `outputs/expert_local_bright/` (with `front`, `left_wrist`, and `right_wrist` videos; deleted since), therefore still verify.
- The success thresholds are written in four places: `PickPlaceScene.complete()` and `main()` in `scenes/pick_place/run.py`, `scenes/pick_place/check.py`, `sdk/verify_backends.py`, and `docs/tasks.md`. Change them together.

**`scenes/kitting/`: the sort-and-kit task (`task.py`, `run.py`, `check.py`).** Each arm moves the three parts on its supply mat into the three cells of its kit tray, each part into the cell of its colour. Each kind of part has its own colour (blue cube, orange cylinder, green hex prism; user request), and each side gets one of each, so each tray has one cell per kind. Both arms run three rounds in lockstep; an episode takes 63–110 s with the two-step ready of the gripper linkage model (62–93 s before), since Scene.move's time-scaling to the real controller's default 30% speed.
- Layout per side: kit tray at x = 0.38 (both policy cameras see it), supply mat at x = 0.48, slots at |y| = 0.10, 0.22, 0.34. The spacing comes from the gripper housing, which at the kitting grasp spans about 85 mm to either side of the grasp point and 65 mm behind it at grasp height. Four slots per side at 85 mm pitch put the housing into the neighbouring part in round one; hence 120 mm pitch and rows 100 mm apart.
- `grasp_rotation(side, psi) = Rz(psi) Ry(-45°) Rz(±90°)`: the hand approaches 45° forward and down. The fingers slide along the wrist_roll link's local x, so the 90° roll (left −, right +) makes them close horizontally along y and keeps the wrist D405 on top of the hand. The opposite roll reaches as well, but it hung the camera underneath, toward the robot (user correction). psi is the part's yaw folded by its kind's symmetry (90° for the cube, 60° for the hex prism; both start within ±10°), 0 for the cylinder. The grasp point is 35 mm above a part's base; the fingertips reach 17 mm below it. The top-down grasp reaches only x = 0.40 at part height, this one x = 0.44 across y 0.03–0.45.
- `sample_episode(seed)` draws per side the order of the three kinds on the mat and in the tray, slot jitter (±6 mm in x, ±4 mm in y), the yaw of cubes and hex prisms, and one of four instruction wordings. It IK-checks every 1 cm waypoint of the expert's moves at `SAMPLE_MARGIN` and redraws until the episode plans.
- `build(spec)` is `build_task.build()` plus the tray and mat boxes, the coloured cell frames (drawn only: contype and conaffinity 0) and the parts. The hex prism is the vertex-only mesh asset `kit_hex`, whose convex hull MuJoCo builds. Part bodies are named `<name>_part` so that `Scene.robot_bodies`, which excludes by that suffix, stays right.
- The expert reads measured part poses at the start of each round (privileged). Its rules, each learned from a failure:
  - `READY` is the two-part expert's point with the kitting rotation. A ready point over the tray clipped the table edge on the way up.
  - The hands reach it in two steps (`run.go_ready`; the sampler plans from the same pose, `task.ready_pose`): fingers down as the two-part expert, then turned in place. Swung straight into the kitting grasp, a fingertip brushed the table (0.32 mm), which the linkage turned into a jaw driven shut.
  - `key_poses` backs every grasp-site target off along the approach by `contact_drop(width)` (15.1 mm), so the dropping pads still close at the grasp point.
  - Every move after `ready` is a straight line. A joint-space approach swept a hand through a part already in the tray.
  - The grippers never open fully (`grip_open` is the part width + 24 mm). Open fingers sit on their joint limits, and once the hand tilts, gravity along the slides (fingers get no feedforward) pushed them 2.9 mm past.
  - IK at `LIMIT_MARGIN` (10 mrad) and moves at `speed=SPEED` (0.6 rad/s). The inner slots push the shoulder roll onto its limit, and fast joint-space jumps there overshot it.
- Success is `KitScene.placements()`: every part in a cell of its colour (±15 mm in x and y), upright (≤ 15°), seated (±3 mm), still (< 0.025 m/s), clear of the robot, one part per cell, held for at least 15 ticks, and each part with at least 100 airborne bilateral grasp steps. The trace adds `round` and `progress` (parts placed).
- `trajectory.npz` keeps pick and place's keys (`subtask` holds the phase names) plus `instruction`, `round`, `progress` and `task`, so `data.convert_expert_run` reads kitting runs; it takes each run's `instruction` from `summary.json` as the task.
- Reference check: `outputs/kitting_check_v7/`, with the gripper linkage: 20/20 seeds, 120/120 parts, every seed on its first draw, 62.8–110.0 s, max penetration 0.13 mm, max limit error 0.16 mrad. Commanded joint velocity and acceleration peak at 0.80× and 0.90× of the 30%-ratio limits.
- With the tilted wrist block: `outputs/kitting_check_wrist_block/` (20/20, 120/120). All 20 trajectories are bit-identical to v7: the block and D405 colliders never touch anything, and only the wrist images change.
- Earlier kitting outputs were deleted in the outputs cleanup; `docs/history.md` keeps their findings. With the linkage, the one-step ready failed every seed at 3.2 s (the table brush above).

**RoboSyn object scenes: `scenes/robosyn_kitchen/`, `scenes/robosyn_drawer/`, `scenes/robosyn_tray/` (user request 2026-10-24: long tasks built from the RoboSynChallenge props).** `docs/tasks.md` has their layouts, experts, thresholds and results; the shared parts are in `sim/`:
- `sim/props.py`: third-party meshes into scene props: scale, settle (base at z = 0), vertex-clustering decimation, visuals split by colour, convex collision pieces (MuJoCo collides any mesh as its hull, so concave props are cut into pieces: `ring()` angular sectors of a round wall, thickened outward; `box()`), and the MJCF helpers (`mesh_assets`, `add_geoms`, `free_body`, `inertial`). `prepare()` caches each scene's props in `scenes/<name>/assets/derived/` (ignored by git), keyed by the raw bytes and a version: bump `sim.robosyn.VERSION` when a maker changes.
- Raw props live in `assets/robosyn/` (`sim.props.RAW`), copied unmodified from the RoboSynChallenge repo's `assets/` (Apache-2.0) and EmbodiChain's data packs (plate, drawer). A nested `.gitignore` keeps them out of git like `sdk_review/`; a fresh clone needs the folder copied in. Check licences before committing anything derived.
- `sim/robosyn.py`: one maker per prop, with the scale this robot needs and why (pot 10% wider so the housing fits inside, taller lid knob, tomato 0.85, drawer 30% wider with a 90 mm cabinet and 57 mm drawer walls and a 20 mm knob replacing the 6 mm bar), plus `add_button` and `add_drawer`. Both fixed bases need an `<exclude>` with their moving part: MuJoCo's parent filter skips a parent welded to the world (the button cap stopped after 1.2 mm).
- `sim/tabletop.py`: the 45-degree grasp (`grasp_rotation`), `grip_open`/`grip_closed(width, squeeze)`, `pick_place` key poses (backed off by `contact_drop`), `press_poses` and the closed-loop `press` (2 mm steps until the joint reads 4 mm: the pads' lowest corner, not the fingertip axis, meets a button first), `plan` (IK-checks a path's 1 cm waypoints), `ObjectScene` (free objects `<name>_part`, `OBJECTS`/`HELD`/`MILESTONES`, `conditions()` made sticky by `status()`, grasp evidence) and `run_episode`, which writes the usual run files. Success: no error, `complete()` for 15 ticks, 100 airborne bilateral steps per held object.
- `sim/scene_check.py`: the scenes' `check.py` CLI (expert runs, pixel counts per prop in the head and body cameras, previews, `report.json`; `--variant`).
- Grip and grasp rules, each from a failure: 90-130 g objects squeeze 4-7 mm and the 150 g lid and the loaded tray 8 mm (at 2.1 mm the lid slid off its knob); round foods are grasped at their widest, their colliders flattened at the base (the round tomato rolled 4 cm); a 72 mm beaker is carried at its centre of mass (held near the rim it swung and fell) and lifted 10 cm past others; anything lowered into a container goes in at its centre straight from above, the housing (about 85 mm either side) being the limit.
- Reference checks (2026-10-24, 20 seeds each, all first draw): kitchen cook `outputs/kitchen_check_v2/` and swap `outputs/kitchen_swap_check_v1/`, drawer store `outputs/drawer_check_v2/` and retrieve `outputs/drawer_retrieve_check_v1/`, tray `outputs/tray_check_v1/`: all 20/20 with every milestone; max penetration 0.19-0.34 mm, max limit error 0.11-0.90 mrad (the tray lift). Demo videos: `outputs/{kitchen,kitchen_swap,drawer,drawer_retrieve,tray}_demo_seed0/`.

## Experiment rules

- Grasps must hold through contact and friction alone. Don't use any of these:
  - welds or attachment constraints
  - writes to object poses after reset
  - contact-mask changes at runtime
  - writes to the robot's qpos or qvel
- The gripper linkage's joint equalities (`sim/gripper_linkage.py`) couple robot joints to each other only, like URDF mimic joints. They are part of the mechanism, not grasp constraints, so the rule above still holds between the robot and the objects.
- Ground-truth object state is for the expert and for scoring only. It must never become a policy input.
- State and action are 16-D, in this order:
  - left arm: 7 joint angles in radians, in `ARM_JOINTS` order
  - right arm: 7 joint angles
  - left gripper closure, then right (0 open, 1 closed). The two fingers get equal and opposite position targets of 0.0475·g m. Measured closure is (q1 − q2)/0.095.
- The leg stays upright (calf and thigh at 0), matching the lab robot photo. The user decided on 2026-10-24 not to adopt the Z-shaped crouch seen in the factory videos, so don't "fix" the posture unless asked.
- The chest carries the factory print: INOVANCE, "Powered by", FMC³ and ROBOTICS. On 2026-10-24 the user reversed an earlier choice of the lab robot's INOVANCE-only chest, so don't go back to it.
- The workspace layout is set by what the cameras can see. With the head pitched 0.50 rad, the head D435 sees the table only from about x=0.33 m, so pick and place's `SOURCES` are at x=0.38 and `GOALS` at x=0.40; IK fails beyond about x=0.45. If you move them closer, the head camera loses the workspace.
- Trajectories are raw expert captures, not yet a LeRobot v3 dataset. Any export must keep the camera timestamps and the action semantics.
- `scene.xml` refers to meshes by absolute path, so it only loads from this checkout.

## SDK bridge (simbridge)

- Upper-level code talks only to the unmodified V1R1C9 SDK; `sdk.stock_sdk.connect(profile)` picks the target from `config/connections/*.yaml` (RK3588 or a simulator). Never edit anything under `sdk_review/feimi1010/sdk/`; `sdk/verify_backends.py` byte-compares it with `sdk.zip`.
- Layers, each importing only those below: `protocol` → `services` → `control` → `robot` → `runtime` → `backends/<engine>_backend.py`. Only adapters import an engine (a test enforces this), and the runtime runs the engine on the main thread; RPC threads use the command queue and snapshots, never the engine.
- SDK joint values are controller-MDH coordinates, `q_sdk = sign * q_urdf + offset`, from `config/robot/franzi_sdk.yaml` (both arms: sign [-1, 1, -1, -1, -1, -1, 1], offsets 0, from `sdk/check_sdk_kinematics.py` and `config/robot/kinematic_crosscheck.json`; not hardware-verified). Equal numbers in the two conventions are not the same pose.
- `build_task.align_joint_limits` clips the MuJoCo joint ranges to the SDK limits in `config/robot/franzi_sdk.yaml` (URDF ∩ controller, with the 0.001 rad margin on the controller side). The MuJoCo adapter refuses to load if an engine range is wider than the URDF's or narrower than what the SDK layer accepts.
- `simbridge/backends/mujoco_backend.py` wraps `scenes.pick_place.run.PickPlaceScene`, so the bridge serves the pick-and-place scene (`goal`, `grip`, `physics_step`, `object_info`, `robot_bodies`, `maxdepth`); changing those changes the bridge.
- simbridge runs on MuJoCo only: the PyBullet backend was retired on 2026-10-24 (user decision; only the MuJoCo scene is aligned to the real robot). Its earlier results stay in `outputs/sdk_backends_v1`, `v6` and `v8`.
  - `sdk/convert_pybullet_assets.py` still derives `assets/derived/pybullet/`, a URDF scene with MuJoCo's colliders, for the Isaac Sim draft. Rerun it after `sim/build_task.py`, `scenes/pick_place/task.py`, `sim/camera_rig.py`, `franzi.xml` or the URDF change.
    - Exception: it was not rerun for the gripper linkage model. URDF has no polynomial joint couplings (mimic joints are linear), so the coupled links would come out free. Its output keeps the old frozen gripper until the Isaac draft models the linkage.
- `config/sdk_acceptance/program.json` is generated by `sdk/make_sdk_program.py` from the expert run in `outputs/expert_wrist_block/` (its default; before the gripper linkage model, `outputs/expert_rate_limited/`); regenerate both when the task layout or the expert changes. `sdk/sdk_acceptance.py` refuses non-loopback hosts: its motions are simulation-only.
  - The generator slows any phase that exceeds 0.9× the SDK's planning-only rate limits (`config/robot_constraints/joint_limits.yaml`: 1 rad/s and 2 rad/s² per arm joint) and records the factor.
  - The rate-limited expert's velocities fit them. The YAML's acceleration limit is tighter than 30% of the archive's for every arm joint, so retract still slows ×1.005 (elbow 1.82 rad/s²). The v8 program slowed ready ×1.52 and approach ×1.77.
- `simbridge/sdk_kinematics.py` answers Fk, FkTool, IK, FKb/IKb(Tool), Get/SetFlan2tool and GetCurCartesianPose(Dual) from `robot_kinematics`. It only converts SDK conventions:
  - SDK joint values via the active `UnitMap`.
  - Frames: A0 and the flange come from `kinematic_crosscheck.json`, and B is URDF `base_link`, the simulated robot's true pose.
  - Wire format: column-major 4x4 transforms at float32; poses are `[x, y, z, rz, ry, rx]` with `R = Rx Ry Rz`.
  - An unreachable IK target returns -6.
  - The IK passes `Kinematics(margin=0.0001)`: it emulates the controller's IK, which keeps no planning margin. `sdk.sdk_cartesian` uses the planning default.
  - The endpoints that only expose the controller's own model (FkLink, GetMdh, GetMdhCompensation, GetArmEulerAngle, SetMdh*) are unsupported, as are MoveL and Cartesian jog/teleop.
  - Without Pinocchio (the Isaac env) the server still starts and these endpoints report UNSUPPORTED.
  - The vendor manual's hard-coded FKb constants sit (0, ∓0.0725, −0.26355) m from the URDF chain at the upright stance, so the real controller's FKb likely differs from the simulation's. Real Cartesian goals belong on the host.
  - `sdk/check_sdk_kinematics.py` keeps the NumPy MDH-vs-URDF math, offline only. A test checks SDK Fk against the image's MDH chain to 1e-12.
- `sdk.sdk_cartesian.CartesianArm(arm)` is the host-side way to reach Cartesian goals on sim and real alike. It solves with `robot_kinematics` (measured arm and waist joints, current orientation unless given) and sends SDK joints: `move()` as one MoveJ, `line()` through the joint teleop queue at 25 Hz.
  - `line()` slows down to stay within 0.9 of `GetJointVelLimit`.
  - Targets are `{side}_grasp` points in the simulation's world frame.
  - Keep it out of `sdk/stock_sdk.py`: the server's wire layer imports that module, and some server envs lack Pinocchio.
- Teleop playback paces each pushed point by the median of the session's last 50 push gaps (about 2 s), using the 40 ms default until 8 gaps are known. Gaps over 0.2 s count as pauses: they are left out without clearing the history.
  - A slow reply stalls the client, and its absolute schedule then sends the missed points back to back. The old median of 8 gaps, cleared at every pause, collapsed in those bursts, which PyBullet's GIL-holding steps made common.
  - The collapse drove playback into the velocity clamps and bent the hand path by up to 30 mm (the v7 PyBullet task failed).
- The MuJoCo adapter runs under `inovance-fineart` (`config/backends/mujoco*.yaml`). It renders every image size through one shared GL/MuJoCo context: each context carries its own 8192² shadow map, about 256 MB. The pixels match `mujoco.Renderer`'s to within 1 level.
