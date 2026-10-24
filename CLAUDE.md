# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a local MuJoCo simulation (macOS) of the Inovance/Franzi wheeled dual-arm robot. It is phase 1 of adapting FineART. A privileged, scripted IK expert performs a bimanual kitting task and records demonstrations:

- The base is fixed.
- Both arms grasp a box at the same time (left: blue, right: orange), lift it, place it on a wide target pad, release it, and retract.

No FineART weights or inference are loaded yet. The old OMX 6-D policy can't be reused as is: it needs rebuilt feature/normalization configs and fine-tuning.

A second, longer task for VLA training, dual-arm sort-and-kit, lives in `sim/kitting_task.py`, `sim/run_kitting.py` and `sim/check_kitting.py` (see "The sort-and-kit task" below). It reuses the robot scene, `Scene`'s audits and recordings, and the 16-D data contract.

The code is in packages run as `python -m <package>.<script>` from this directory: `sim/` holds the scene and the experts, `sdk/` the stock-SDK tools, `data/` the VLA data tools, and `robot_kinematics.py`, the shared kinematics, stays at the top level. The vendor-SDK bridge is the `simbridge/` package (see "SDK bridge" below) with an engine-free pytest suite in `tests/`. `README.md` (in Chinese) is the experiment log and acceptance spec; its SDK sections document the bridge, the vendor SDK and controller-image notes under `sdk_review/`, and planning constraints under `config/`.

The project is a git repository (branch `main`). Its remote is the public GitHub repository `heliang-pu/franzi-sim`, which the user created on 2026-10-24. The local folder keeps its old name. Everything committed is public.
- The repository has no license, on purpose: all rights reserved. The user wants others to view the code but not reuse it; GitHub can't block downloads.
- Commits use a repository-local identity: the name heliang-pu (user decision) and GitHub's noreply address, so no personal email enters the public history. The global git identity is someone else's.
- `.gitignore` keeps out `outputs/`, the rebuildable appearance caches, `assets/derived/` and `sdk_review/`. Never commit `sdk_review/`: it is third-party vendor code, and the controller image holds credential-like fields.
- A fresh clone therefore lacks them. `sdk/stock_sdk.py`, the stock-SDK acceptance, `sdk/check_sdk_kinematics.py` and the tests that read the controller image need `sdk_review/` copied in from this machine.

## Commands

Run from this directory with the project's own conda env, `/Users/puheliang/miniforge3/envs/inovance-fineart` (created 2026-10-24 at the user's request). It has Python 3.12.15, and from PyPI mujoco 3.13.0, numpy 2.2.6, scipy 1.18.1, trimesh 5.1.0, av 15.1.0, pillow 12.3.0, PyYAML 6.0.3, networkx 3.6.1, pytest 8.4.2 and pin 3.4.0 (Pinocchio).
- The other packages match `piper-tactile`, and runs there are bit-identical. `piper-tactile` has no Pinocchio, so `robot_kinematics` (and everything importing `build_task` or `run_expert`) no longer runs in it.
- `cmeel-urdfdom` must stay at 4.0.1: pin 3.4.0 links urdfdom 4.0, and pip picks 6.0.0 unless pinned.
- To recreate: `mamba create -n inovance-fineart -c conda-forge python=3.12 pip`, then `pip install` the versions above plus `cmeel-urdfdom==4.0.1 cmeel-tinyxml2==10.0.0`.

```bash
# Headless episode, a few seconds. --output must not exist yet.
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.run_expert --output outputs/<new_run>
# Same, plus MP4 recording (about a minute)
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.run_expert --output outputs/<new_run> --record
# Acceptance check of a recorded run: writes <run>/verification.json, raises AssertionError on failure
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.verify_run outputs/<run>
# Derive the scene only: writes outputs/task.xml and outputs/collision_setup.json
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.build_task
# Sort-and-kit: one episode (about 4 s headless, about 3 min with --record)
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.run_kitting --output outputs/<new_run> --seed 0
# Its acceptance across seeds: expert runs, camera pixel counts, previews -> report.json
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.check_kitting --output outputs/<new_dir> --seeds 20
# Lint and format the project's own code. No config file, so Ruff 0.16 defaults apply
# (88 columns, isort enabled). Not `.`: sdk_review/ holds vendored third-party Python.
ruff check robot_kinematics.py sim sdk data simbridge tests && ruff format robot_kinematics.py sim sdk data simbridge tests
# SDK bridge: engine-free tests (seconds), then the stock-SDK acceptance per backend
# (a few minutes each with --record; --output may already hold other backends)
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m pytest tests -q
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sdk.verify_backends --output outputs/<run> --backends mujoco,mujoco_owned_loop --task --fault-probe --record
```

- The expert has no unit tests. Validate it in stages: `sim/build_task.py`, then a headless run, then a `--record` run checked with `sim/verify_run.py`.
- A failed episode still exits 0. `RuntimeError` and `ValueError` raised inside `expert()` are caught and written to `summary.json` (`success`, `error`), so check that file. Any other exception, and any failure while building `Scene` (including the camera-mount asserts), crashes without writing a summary.
- Run Ruff from this directory. From anywhere else it treats the local modules as third-party and reports false isort errors. The workspace `AGENTS.md` sets a 110-column limit, but that comes from lerobot's config and doesn't apply here.
- Each retained run keeps its stdout next to the run folder as `outputs/<run>.log`.

## Architecture

Imports (all in `sim/`, as `from sim.<module> import ...`; `robot_kinematics` is top level):
- `run_expert` uses `build_task`, `camera_rig` and `robot_kinematics`.
- `camera_rig` and `verify_run` use `build_task`.
- `build_task` uses `appearance` (which uses `fittings`) and `robot_kinematics`.
- `kitting_task` uses `build_task` and `robot_kinematics`; `run_kitting` uses `kitting_task`, `run_expert` and `camera_rig`; `check_kitting` uses `run_kitting`.

`sim/build_task.py` holds the shared task constants: `TABLE_Z`, `SOURCES`, `GOALS`, `ROOT`, and `fmt`. It also re-exports `ARM_JOINTS`, which fixes the joint order and is defined in `robot_kinematics.py`. `sim/appearance.py` defines its own `ROOT` to avoid a circular import.

**`robot_kinematics.py`: the one shared FK/IK ("real follows sim", user decision 2026-10-24).** Cartesian goals are solved here on the host; the simulation and the real robot both receive joint commands only (`MoveJ`, joint teleop). The controller's own IK, `MoveL` and Cartesian teleop are deliberately unused. simbridge's IK/Fk endpoints call this module; `simbridge/sdk_kinematics.py` only converts SDK conventions.
- FK and Jacobians come from Pinocchio (user decision, 2026-10-24). The module needs NumPy, SciPy, PyYAML and pin, all in the `inovance-fineart` env, but no physics engine. Keep MuJoCo and `build_task` out of it.
- Pinocchio's model is the source URDF, with base_link offset as in `franzi.xml` and an operational frame per `{side}_grasp` tool point from `config/robot/franzi_sdk.yaml`. Leg and waist joints come from `body` (`BODY_JOINTS`); every other joint is neutral. FK equals the scene's to 1e-15 m.
- `Kinematics.ik()` is the expert's IK: multi-seed `least_squares` on position (×10) and the rotation vector to a fixed world-aligned grasp orientation, inside the aligned limits.
  - The Jacobian is exact: Pinocchio's frame Jacobian, with `Jlog3` for the rotation rows. It lands within 5e-8 rad of the old finite-difference solutions, and is 6× faster (single target about 6 ms, chained waypoint 1.3 ms).
  - When chaining waypoints (`near=`), a residual `(q - near) * 0.002` keeps the redundant elbow swivel near the previous solution; without it the arm visibly wobbled along straight moves. 0.02 is too strong (the approach IK becomes infeasible).
  - `line()` gives the 1 cm waypoint chains, and `to_sdk()`/`from_sdk()` map URDF ↔ SDK joint values.
  - `Kinematics(margin=...)` keeps solutions that far inside the aligned limits (default 0.0001 rad, as before). The kitting expert uses 10 mrad and its sampler 20 mrad: planned on a limit, the shoulder roll overshot the 2 mrad audit by up to 12 mrad.
  - A seed whose rotation error reaches π (no finite `Jlog3`) is skipped instead of raising `ValueError`.
- Joint limits are `controller_ranges()`; `build_task.align_joint_limits()` writes the same numbers into the scene.

**`build_task.build()`: the derived scene.** Every run parses `assets/franzi_source/franzi.xml` with ElementTree, edits the tree in memory, and returns `(root, report)`. Never edit `assets/franzi_source/`. It is an unmodified copy of the source project on the 4090 machine, and `summary.json` records the URDF's sha256. The edits:
- The base is fixed (its freejoint and the keyframe are removed). Physics runs at 500 Hz with elliptic friction cones.
- In the source model, every link's collider is a bounding box. The build changes two of them:
  - Each wrist box becomes one convex hull per connected CAD component of the wrist STL, 49 per side. This keeps the jaw opening clear.
  - The finger boxes become the finger STL meshes (MuJoCo collides with their convex hulls) with grip contact parameters.
  - Every other link keeps its box, so this is not a full CAD collision model.
- Contacts are filtered only by contype/conaffinity bits. The source model has no `<exclude>`.
  - The source `franzi` class (1/2) collides with the world but never with itself.
  - Arm colliders are re-masked to 5/10 (left) and 9/6 (right). As a result, left-arm vs right-arm contact is on, contact within one arm and between an arm and the torso or head stays off, and robot-world contact stays on.
  - "Arm" means bodies whose names start with `left` or `right` and don't contain `front`, which excludes the front wheels.
  - Only geoms that carry their own `class="collision"` attribute are matched. ElementTree doesn't see a class inherited through `childclass`.
- The build adds the `{side}_grasp` IK sites, the table, two free boxes (`left_part`, `right_part`), the target pads, and the lights. `build(task_objects=False)` leaves out the boxes and pads, for a task that brings its own; `PART_RGBA` holds their two colours.
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
  - D405: realsense-ros `d405.stl` at the same frame.
- The camera origins sit behind the glass, so the policy cameras rely on the 45 mm render `znear` not to see their own housings.
- `attach_chest_slot()` fills the chest camera slot (`CHEST_SLOT`, in the body D435 sensor frame) with a pearl bezel plate and a tinted 82 × 10.5 mm window, 0.4–1.0 mm ahead of the glass. The shell's opening (85.1 × 20.8 mm) otherwise shows the whole D435 front, while the real robots show a narrow dark window. These are inline meshes inside `znear`, so the body camera never sees them.
- The face visor has no camera window. The head D435 sits 16–20 mm behind it, inside the 45 mm `znear`. Removing the whole visor, or the chest decal, from the render changes 0 pixels in all four policy cameras. A cut window with a rim and shroud was built and then dropped at the user's request (2026-10-24), because it changed nothing in the recordings.
- Rendering is deterministic, but recorded MP4s are not a pixel-level comparison. A visual geom that changes `m.stat.extent` moves a few shadow pixels, and H.264 spreads that to roughly 0.5% of pixels. Compare raw `mujoco.Renderer` frames instead.
- Cap rules keep only faces wholly inside a CAD vertex ring and connected to the cap centre.
- The outer elbow cap's triangles cross every face-level cut, so it is drawn by a texture instead: a conformal decal on the pearl elbow mesh with a black disc and the white logo.
- URDF sensor sites are moved to group 4 so their marker spheres don't render; `camera_rig.assert_mounts()` still uses them.
- Each `(mesh, material)` group becomes one visual geom. It keeps the original geom's `pos`/`quat` and has an `inertia="shell"` mesh asset. If the CAD changes, `paint_groups` asserts rather than leaving a component unpainted.
- Caches live in `assets/appearance/{painted,wordmark,fittings}/`. The paint key covers only the source STL bytes and `PAINT_VERSION`, so bump `PAINT_VERSION` whenever a face rule changes. Older files directly under `assets/appearance/` (`head_*.stl`, `head_parts.json`, `inovance_wordmark.png`) are unused leftovers.
- The chest decal is a copy of the torso front-shell triangles (component 2) lifted 0.4 mm along MuJoCo's compiled normals. MuJoCo ignores texture alpha, so the texture background is `PEARL_SRGB`, derived from `fa_pearl`, and only the print shows. Don't add face rules on the front shell near the print: the decal's normals must match the pearl geom's.
- The chest print is the factory robot's, from top to bottom: INOVANCE (`WORDMARK`), "Powered by", FMC³ and a tracked ROBOTICS line (`FACTORY_PRINT`). It was measured on frontal factory-video frames, using the INOVANCE width as the ruler.
  - The FMC³ letters are drawn from `FMC3_GLYPHS`, read off a 30-frame stack of the factory video: a floating bar over a Γ, a V-shaped M, a C with two dot eyes, and a small 3 on the cap line.
  - `assets/franzi_source/assets/fmc3_logo.png` is a different logo version (comb M, one square in the C). Only its 3 is used.
- `sim/fittings.py` adds the `fit_*` parts the CAD lacks: the lower-back ribbed panel (a harmonic fill of the opening's rim), the upper-back e-stop, ports and lights, the thigh back cover, and the base light strips. They are placed on surface fits of the shells and reference `PALETTE` material names only.
- Fonts are macOS paths: Avenir Next for the chest and elbow text, and Arial in `run_expert.render()`.

**`sim/camera_rig.py` and `sim/camera_rig.json`: the policy cameras.**
- `POLICY_CAMERAS` are the head and body D435 and the left and right wrist D405. Each is attached to the URDF sensor site of the same name.
- Camera axes are the site frame, rotated by the optical roll from the JSON, then 180° about x. The flip is needed because the sensor frame looks along +Z while MuJoCo cameras look along −Z. `assert_mounts()` checks the result after forward kinematics.
- The intrinsics are uncalibrated simulated pinholes.
- Change a camera's view only by moving joints. For example, the head gaze is a `head_pitch_joint` actuator command in `sim/run_expert.py`. Never re-aim a camera or detach it from its site.
- The world cameras added in `sim/run_expert.py` (`overview` and others) are for display only and are never policy inputs.

**`sim/run_expert.py`: the episode.**
- `Scene` installs the cameras, compiles the model, sets the render `znear` to 45 mm so each camera sees past its own housing, and writes the result to `scene.xml`. Two overrides are applied only to the compiled model and are missing from `scene.xml`: the finger actuators get kp 1500 and a force range of ±6 N, and the `TORSO_SERVOS` (calf, thigh, waist pitch and yaw) get velocity feedback `kv` for a damping ratio of 0.7 at the start posture. Without `kv` the source servos (kp 6000, damping 5, ζ ≈ 0.01) let the upper body rock at about 1.5 Hz after every arm move (user decision, 2026-10-24). Don't raise it to ζ = 1: the expert parks the wrist pitch on its URDF bound, and the changed posture then trips the 0.002 rad limit audit.
- Physics runs at 500 Hz and control at 25 Hz. On every 20th physics step the code appends one trace sample (the measured state plus the action about to be applied), renders one frame per video, and updates the stable-frame counter. Joint goals are held constant between control ticks.
- The robot is driven only through position-actuator `ctrl`, plus gravity compensation via `qfrc_applied = qfrc_bias` on the arm DOFs and on the `TORSO_SERVOS` and `HEAD_SERVOS` DOFs (`Scene.held`).
  - On kp alone the leg and waist servos sagged up to 7.5 mrad each (about 1.3° in all) with the arms forward, and the head pitch 3.8 mrad. That shifted the head and body cameras, and put the hands about 9 mm off when expert joint targets were replayed in PyBullet.
  - With the feedforward the error is 0 at rest, and at most about 1 mrad during moves (user decision, 2026-10-24). Head pitch still lags about 5 mrad while its 1.5 s gaze ramp is moving.
- After every physics step, a penetration over 1 mm or a joint-limit overshoot over 0.002 raises `RuntimeError` and ends the episode.
- `Scene.ik()` delegates to `robot_kinematics.Kinematics.ik()`, passing the live arm pose and leg and waist joints; it never touches the live state. `sdk/make_sdk_program.py` calls it. Change IK behaviour in `robot_kinematics`, not here.
- `move()` follows a quintic minimum-jerk profile. By default it interpolates straight between joint-space IK solutions. With `linear=True` it runs through `Kinematics.line()`: IK waypoints 1 cm apart on the Cartesian line, each chained to its predecessor's branch. The expert uses `linear=True` for align, lift, transfer, place and retract: near the edge of the arm's reach, plain joint interpolation swings the hand about 2 cm forward mid-descent, and a finger lands on the box.
- The thigh's straight-leg target of 0 sits on its lower joint limit, so fast arm swings can push it past the 0.002 limit check. That is why the final retract takes 3 s.
- `expert()` runs the scripted stages ready → approach → align → grasp → lift → transfer → place → release → retract → finish. Along the way it checks that both jaws touch each box and that the boxes are lifted high enough. Each stage's name and instruction become the trace's `phase` and `subtask` fields.
- Every run writes `scene.xml`, `summary.json` (result, physics stats, data contract, and collision, appearance, and camera metadata), `trace.json`, and `trajectory.npz`.
- `--record` adds `expert.mp4` (overview plus wrist insets), `cameras.mp4` (2×2 mosaic of the policy cameras), and one 640×480 MP4 per policy camera. All videos are 25 FPS with exactly one frame per trace sample.
- A task subclass overrides `build_root()` (scene tree and report), `sample()` (one trace record), `track_objects()` (per-step grasp evidence) and `complete()`; `run_kitting.KitScene` does. `ik(rotation=)` and `move(rotations=)` take a hand rotation per arm. `move(speed=)` times the waypoints by joint travel (largest joint step) and stretches the move until no joint's peak command rate exceeds `speed`. The defaults keep the two-part expert bit-identical (trajectory, trace and scene.xml checked).

**The sort-and-kit task: `sim/kitting_task.py`, `sim/run_kitting.py`, `sim/check_kitting.py`.** Each arm moves the three parts on its supply mat into the three cells of its kit tray, each part into the cell of its colour. Each kind of part has its own colour (blue cube, orange cylinder, green hex prism; user request), and each side gets one of each, so each tray has one cell per kind. Both arms run three rounds in lockstep; an episode takes 53–55 s.
- Layout per side: kit tray at x = 0.38 (both policy cameras see it), supply mat at x = 0.48, slots at |y| = 0.10, 0.22, 0.34. The spacing comes from the gripper housing, which at the kitting grasp spans about 85 mm to either side of the grasp point and 65 mm behind it at grasp height. Four slots per side at 85 mm pitch put the housing into the neighbouring part in round one; hence 120 mm pitch and rows 100 mm apart.
- `grasp_rotation(side, psi) = Rz(psi) Ry(-45°) Rz(±90°)`: the hand approaches 45° forward and down. The fingers slide along the wrist_roll link's local x, so the 90° roll (left +, right −, wrist outboard) makes them close horizontally along y. psi is the part's yaw folded by its kind's symmetry (90° for the cube, 60° for the hex prism; both start within ±10°), 0 for the cylinder. The grasp point is 35 mm above a part's base; the fingertips reach 17 mm below it. The top-down grasp reaches only x = 0.40 at part height, this one x = 0.44 across y 0.03–0.45.
- `sample_episode(seed)` draws per side the order of the three kinds on the mat and in the tray, slot jitter (±6 mm in x, ±4 mm in y), the yaw of cubes and hex prisms, and one of four instruction wordings. It IK-checks every 1 cm waypoint of the expert's moves at `SAMPLE_MARGIN` and redraws until the episode plans.
- `build(spec)` is `build_task.build(task_objects=False)` plus the tray and mat boxes, the coloured cell frames (drawn only: contype and conaffinity 0) and the parts. The hex prism is the vertex-only mesh asset `kit_hex`, whose convex hull MuJoCo builds. Part bodies are named `<name>_part` so that `Scene.robot_bodies`, which excludes by that suffix, stays right.
- The expert reads measured part poses at the start of each round (privileged). Its rules, each learned from a failure:
  - `READY` is the two-part expert's point with the kitting rotation. A ready point over the tray clipped the table edge on the way up.
  - Every move after `ready` is a straight line. A joint-space approach swept a hand through a part already in the tray.
  - The grippers never open fully (`grip_open` is the part width + 24 mm). Open fingers sit on their joint limits, and once the hand tilts, gravity along the slides (fingers get no feedforward) pushed them 2.9 mm past.
  - IK at `LIMIT_MARGIN` (10 mrad) and moves at `speed=SPEED` (0.6 rad/s). The inner slots push the shoulder roll onto its limit, and fast joint-space jumps there overshot it.
- Success is `KitScene.placements()`: every part in a cell of its colour (±15 mm in x and y), upright (≤ 15°), seated (±3 mm), still (< 0.025 m/s), clear of the robot, one part per cell, held for at least 15 ticks, and each part with at least 100 airborne bilateral grasp steps. The trace adds `round` and `progress` (parts placed).
- `trajectory.npz` keeps `run_expert`'s keys (`subtask` holds the phase names) plus `instruction`, `round`, `progress` and `task`, so `data.convert_expert_run` reads kitting runs; it takes each run's `instruction` from `summary.json` as the task.
- Reference check: `outputs/kitting_check_v4/` (20/20 seeds, 120/120 parts, max penetration 0.10 mm, max limit error 0.32 mrad). `kitting_check_v1` and `v2` predate the fixes; `v3` is the earlier blue/orange-only version. `outputs/kitting_demo_seed0/` is a recorded episode that converts (`data.convert_expert_run`, 1358 frames) and exports (`data.export_lerobot`).

**`sim/verify_run.py`: acceptance.**
- It re-checks a recorded run folder:
  - the success thresholds
  - the structure of `scene.xml`: no `<weld>`, exactly 2 freejoints, mesh colliders on the wrists and fingers
  - each video's frame count, FPS, and width
- Video names come from `contract.cameras` in each run's `summary.json`. Runs from before the 4-camera rig, such as the README's reference run `outputs/expert_local_bright/` (with `front`, `left_wrist`, and `right_wrist` videos), therefore still verify.
- The success thresholds are written in three places: `Scene.complete()` and `main()` in `sim/run_expert.py`, `sim/verify_run.py`, and the README. Change them together.

## Experiment rules

- Grasps must hold through contact and friction alone. Don't use any of these:
  - welds or attachment constraints
  - writes to object poses after reset
  - contact-mask changes at runtime
  - writes to the robot's qpos or qvel
- Ground-truth object state is for the expert and for scoring only. It must never become a policy input.
- State and action are 16-D, in this order:
  - left arm: 7 joint angles in radians, in `ARM_JOINTS` order
  - right arm: 7 joint angles
  - left gripper closure, then right (0 open, 1 closed). The two fingers get equal and opposite position targets of 0.0475·g m. Measured closure is (q1 − q2)/0.095.
- The leg stays upright (calf and thigh at 0), matching the lab robot photo. The user decided on 2026-10-24 not to adopt the Z-shaped crouch seen in the factory videos, so don't "fix" the posture unless asked.
- The chest carries the factory print: INOVANCE, "Powered by", FMC³ and ROBOTICS. On 2026-10-24 the user reversed an earlier choice of the lab robot's INOVANCE-only chest, so don't go back to it.
- The workspace layout is set by what the cameras can see. With the head pitched 0.50 rad, the head D435 sees the table only from about x=0.33 m, so `SOURCES` are at x=0.38 and `GOALS` at x=0.40; IK fails beyond about x=0.45. If you move them closer, the head camera loses the workspace.
- Trajectories are raw expert captures, not yet a LeRobot v3 dataset. Any export must keep the camera timestamps and the action semantics.
- `scene.xml` refers to meshes by absolute path, so it only loads from this checkout.

## SDK bridge (simbridge)

- Upper-level code talks only to the unmodified V1R1C9 SDK; `sdk.stock_sdk.connect(profile)` picks the target from `config/connections/*.yaml` (RK3588 or a simulator). Never edit anything under `sdk_review/feimi1010/sdk/`; `sdk/verify_backends.py` byte-compares it with `sdk.zip`.
- Layers, each importing only those below: `protocol` → `services` → `control` → `robot` → `runtime` → `backends/<engine>_backend.py`. Only adapters import an engine (a test enforces this), and the runtime runs the engine on the main thread; RPC threads use the command queue and snapshots, never the engine.
- SDK joint values are controller-MDH coordinates, `q_sdk = sign * q_urdf + offset`, from `config/robot/franzi_sdk.yaml` (both arms: sign [-1, 1, -1, -1, -1, -1, 1], offsets 0, from `sdk/check_sdk_kinematics.py` and `config/robot/kinematic_crosscheck.json`; not hardware-verified). Equal numbers in the two conventions are not the same pose.
- `build_task.align_joint_limits` clips the MuJoCo joint ranges to the SDK limits in `config/robot/franzi_sdk.yaml` (URDF ∩ controller, with the 0.001 rad margin on the controller side). The MuJoCo adapter refuses to load if an engine range is wider than the URDF's or narrower than what the SDK layer accepts.
- `simbridge/backends/mujoco_backend.py` wraps `run_expert.Scene` (`goal`, `grip`, `physics_step`, `object_info`, `robot_bodies`, `maxdepth`); changing those changes the bridge.
- simbridge runs on MuJoCo only: the PyBullet backend was retired on 2026-10-24 (user decision; only the MuJoCo scene is aligned to the real robot). Its earlier results stay in `outputs/sdk_backends_v1`, `v6` and `v8`.
  - `sdk/convert_pybullet_assets.py` still derives `assets/derived/pybullet/`, a URDF scene with MuJoCo's colliders, for the Isaac Sim draft. Rerun it after `sim/build_task.py`, `sim/camera_rig.py`, `franzi.xml` or the URDF change.
- `config/sdk_acceptance/program.json` is generated by `sdk/make_sdk_program.py` from the expert run in `outputs/sdk_task_source_expert_v8/`; regenerate both when the task layout or the expert changes. The generator slows any phase that exceeds 0.9× the SDK velocity/acceleration limits (the expert plans without them) and records the factor. `sdk/sdk_acceptance.py` refuses non-loopback hosts: its motions are simulation-only.
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
