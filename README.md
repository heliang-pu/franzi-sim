# Inovance / Franzi 双臂 FineART 实验（本机 MuJoCo）

第一阶段为双臂专家演示：固定底盘，两只手同时抓取蓝色、橙色物体，抬起并放到中央两个宽目标位，松手后撤回。这里没有加载 FineART 权重；先验证本体、动作接口、夹爪接触与数据记录，再进行策略适配。

本目录是新建的本机实验工程，并非 4090 上 `robot_simulation` 的完整副本。已迁移 URDF、完整机器人 STL 网格、源 MuJoCo 模型和原贴图；Isaac Sim、ROS/RViz 工程、历史数据集和录像没有整包迁移。后来提供的 `feimi1010.zip` 中，原厂 SDK 单独解压到 `sdk_review/feimi1010/sdk/`；控制器更新包只离线提取了部分文件。

代码在 GitHub 公开仓库 [heliang-pu/franzi-sim](https://github.com/heliang-pu/franzi-sim)（用户 2026-10-24 建立，本地目录名不变）。仓库故意不带开源许可证，保留所有权利：别人可以看，但无权使用或转载（GitHub 无法禁止下载）。`outputs/`、`sdk_review/`、可自动重建的外观缓存和 `assets/derived/` 不入库；`sdk_review/` 是原厂代码，控制器镜像里还有密码类字段，不得提交。新克隆的副本缺这些目录：用原厂 SDK、跑 SDK 验收或读取控制器镜像的测试前，要先从本机拷入 `sdk_review/`。

## 运行

使用本项目专用的 conda 环境 `inovance-fineart`（2026-10-24 按用户要求新建）。
- 环境内容：conda-forge 的 Python 3.12.15，以及 PyPI 的 mujoco 3.13.0、numpy 2.2.6、scipy 1.18.1、trimesh 5.1.0、av 15.1.0、pillow 12.3.0、PyYAML 6.0.3、networkx 3.6.1、pytest 8.4.2、pin 3.4.0（Pinocchio）。
- 这些模拟相关包的版本与原来借用的 `piper-tactile` 相同，同一份代码在两个环境里跑出的轨迹逐位一致。
- `piper-tactile` 里没有 Pinocchio，现在的代码已不能在其中运行。
- `cmeel-urdfdom` 须固定在 4.0.1：pin 3.4.0 链接的是 urdfdom 4.0，不固定时 pip 会装 6.0.0，导致 Pinocchio 无法导入。

```bash
cd /Users/puheliang/workspace/inovance-fineart
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.run_expert \
  --output outputs/my_fresh_run --record
```

重建环境：

```bash
/Users/puheliang/miniforge3/bin/mamba create -n inovance-fineart -c conda-forge python=3.12 pip
```

```bash
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m pip install mujoco==3.13.0 numpy==2.2.6 scipy==1.18.1 trimesh==5.1.0 av==15.1.0 pillow==12.3.0 PyYAML==6.0.3 networkx==3.6.1 pytest==8.4.2 pin==3.4.0 cmeel-urdfdom==4.0.1 cmeel-tinyxml2==10.0.0
```

输出目录必须不存在。`expert.mp4` 是有阶段提示的总览和双腕视频；`cameras.mp4` 是四路机载相机四宫格。原始 RGB 视频分别为 `head_d435.mp4`、`body_d435.mp4`、`left_wrist_d405.mp4`、`right_wrist_d405.mp4`。`summary.json` 保存验收结果，`trace.json` 保存逐控制拍子任务、动作、状态和物体状态，`trajectory.npz` 是同样的数值轨迹。

已完成的本机演示：`outputs/expert_local_bright/`，23.5 秒，双臂完成，最大接触穿透 0.099 mm，最终稳定 155 个控制拍；使用调亮后的背景与补光。四条视频均为 588 帧、25 FPS。

新版外观及四相机演示：`outputs/expert_appearance_fourcams_final/`。23.5 秒，双臂任务通过，最大接触穿透 0.099 mm；总览、四宫格和四条原始相机视频均通过帧数与解码检查（588 帧、25 FPS）。外观近景位于 `outputs/appearance_v2_preview/appearance.png`。

按真机外观对齐后的演示：`outputs/expert_real_appearance/`，`sim/verify_run.py` 通过；物理结果与上一版逐位一致（23.5 秒、最大穿透 0.0987 mm、稳定 155 拍、物体终点相同），四路相机画面不受影响。仿真与真机对比图在 `outputs/appearance_alignment/`。

工作区前移、末端与相机上色后的演示：`outputs/expert_layout_gripper/`，`sim/verify_run.py` 通过；24.5 秒，最大穿透 0.100 mm，关节超限最大 0.0012 rad，稳定 178 拍，六条视频各 613 帧。各阶段四路相机能看到什么，记录在 `outputs/camera_audit_layout/`；末端与相机近景在 `outputs/appearance_alignment/gripper_sheet.png`。

新面罩开窗版演示：`outputs/expert_visor_window/`（开窗已按用户要求撤掉，仅作历史记录）。

当前最新演示（逆解留 2° 规划余量，动作按真机默认 30% 速度比例限速）：`outputs/expert_rate_limited/`，`sim/verify_run.py` 通过。
- 30.0 秒成功，最大穿透 0.099 mm，关节超限最大 0.0003 rad，稳定 180 拍，六条视频各 749 帧。
- 两只手腕俯仰离控制器限位 2.06°（原来 0.065°）。手臂指令峰值为 30% 速度比例下上限的 0.90 倍（速度）和 0.48 倍（加速度），原来是 1.81 倍和 1.17 倍。

上一版演示（共享运动学改用 Pinocchio）：`outputs/expert_pinocchio/`，`sim/verify_run.py` 通过。
- 24.5 秒成功，最大穿透 0.102 mm，关节超限最大 0.0006 rad，稳定 180 拍，物块最终离目标不到 0.2 mm，六条视频各 613 帧。
- 正解与仿真相差 1.1e-15 m。逆解改用精确解析雅可比，与原有限差分版最多相差 5e-8 rad。
- 速度：单目标逆解 37 → 5.9 ms，链式路点 8.5 → 1.3 ms，正解 43 → 1.9 µs。
- 与上一版相比，指令关节角最多差 5.9e-7 rad，是同一段示教。simbridge 的 27 个测试在新环境里全部通过。

逆解改用共享运动学（NumPy 版）的演示：`outputs/expert_shared_kinematics/`，`sim/verify_run.py` 通过。
- 24.5 秒成功，最大穿透 0.102 mm，关节超限最大 0.0006 rad，稳定 180 拍，物块最终离目标不到 0.2 mm，六条视频各 613 帧。
- 与上一版相比，指令关节角最多差 3.2e-6 rad，实测关节角最多差 1.6e-4 rad（经接触放大），是同一段示教。
- 逆解不再每次调用 MuJoCo 整机前向计算，整次录制从约 138 秒降到 82 秒。

关节限位和头部转向按控制器镜像对齐的演示：`outputs/expert_controller_limits/`，`sim/verify_run.py` 通过。
- 24.5 秒成功，最大穿透 0.102 mm，关节超限最大 0.0006 rad，稳定 180 拍，物块最终离目标不到 0.2 mm，六条视频各 613 帧。
- 两臂 10 个关节的范围收紧到控制器限位以内，见下文。手臂指令峰值在控制器速度上限的 0.54 倍、加速度上限的 0.35 倍以内。
- 头部转向 3.3 秒，满足头部的速度、加速度和加加速度限位。
- 本次也包含 SDK 会话加的逆解平滑项：相邻路点向上一解轻微牵引，消除肘部来回翻转。
- 对齐后的限位记录在 `summary.json` 的 `collision_setup.joint_limits`。

腰腿和头部加重力前馈的演示：`outputs/expert_torso_hold/`，`sim/verify_run.py` 通过。
- 24.5 秒成功，最大穿透 0.099 mm，关节超限最大 0.0014 rad，稳定 180 拍，两个物块最终离目标中心不到 0.3 mm，六条视频各 613 帧。
- 腰腿关节误差从最多 7.5 mrad 降到运动中最多约 1 mrad、静止时 0。原来下沉最严重时，头部相机偏 23.6 mm、转 1.42°，躯干相机偏 14.8 mm、转 1.28°，手腕相机偏约 11 mm，抓取点偏 9.9 mm。
- 轨迹与上一版不同（物理变了）。抓取到撤回各关节只差 0.05–0.1 rad；收尾一步的逆解换了分支，左臂肘部朝向不同，左肩偏航 −1.23 rad，原来是 +0.35 rad。两种都是合法解，收尾逆解的冗余自由度没有固定。

修掉阴影贴图锯齿的演示：`outputs/expert_shadow_fix/`，`sim/verify_run.py` 通过。
- 24.5 秒成功，最大穿透 0.125 mm，关节超限最大 0.0010 rad，稳定 179 拍，六条视频各 613 帧。
- 轨迹与 `expert_chest_slot` 逐位一致：光照只影响画面，不影响物理。
- 四路相机画面按预期变化，主要在阴影边缘：平均 0.6–1.1% 的像素变化，单帧最多 2.4%。
- 整次录制耗时从 125 s 增至 138 s。

胸口相机槽深色窄窗的演示（带 SDK 会话加的腰腿伺服阻尼）：`outputs/expert_chest_slot/`，`sim/verify_run.py` 通过。
- 24.5 秒成功，最大穿透 0.125 mm，关节超限最大 0.0010 rad，稳定 179 拍，六条视频各 613 帧。
- 与 `expert_factory_chest` 的轨迹不再逐位一致，原因是 `sim/run_expert.py` 新加了腰腿伺服速度反馈（阻尼比 0.7，用户 2026-10-24 在 SDK 会话中确定），上身不再晃动。四路相机视频因此有 2–3% 的像素变化（躯干相机最多 12.6%）。
- 相机槽不影响录制：同一份代码下关掉相机槽再录一次，轨迹逐位一致，四路策略相机视频和四宫格 `cameras.mp4` 的 SHA-256 完全相同。只有展示用的 `expert.mp4` 不同，因为总览相机能看到胸口。

工厂胸口印刷、面罩不开窗的演示：`outputs/expert_factory_chest/`，`sim/verify_run.py` 通过。
- 24.5 秒成功，最大穿透 0.100 mm，稳定 178 拍，六条视频各 613 帧。
- 物理轨迹与前两版逐位一致，613 个采样的动作、状态、阶段和时间戳全部相同。
- 面罩和胸口印刷都不进入四路策略相机：在 6 个录制姿态下，把整个面罩或胸口贴花从渲染中去掉，原始画面都是 0 像素变化。
- 各版 MP4 之间有 0.3–1% 的像素差，原因是 H.264 会把极小的渲染差异放大。例如开窗版模型尺寸略大，原始画面每帧只差 0–10 个阴影像素。
- 胸口与工厂视频的对比在 `outputs/appearance_alignment/chest_factory_sheet.png`，头部与相机近景在 `outputs/appearance_alignment/head_sheet.png`。

产物验收：

```bash
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.verify_run outputs/expert_local_bright
```

## 双臂分拣装盘长任务（场景 v1，2026-10-24）

给 VLA 训练用的长程任务。两只手各把本侧供料垫上的 3 个零件（蓝色方块、橙色圆柱、绿色六角柱各一个，每种料一个颜色）放进本侧料盘里同色的格子，共 6 个零件。两臂同步，分 3 个回合完成，一集 62–93 s（`Scene.move` 按真机默认 30% 速度限速之后）。

| 文件 | 作用 |
|---|---|
| `sim/kitting_task.py` | 按种子采样布局：三种料在供料位上的顺序、位置抖动和偏航，料盘三格的顺序，指令措辞。按专家的动作逐段做 IK 检查，可解才接受。生成场景时复用 `build_task.build(task_objects=False)` 的机器人、桌子和灯光 |
| `sim/run_kitting.py` | `KitScene`（`run_expert.Scene` 的子类，沿用全部审计、录制和数据约定）和脚本专家 |
| `sim/check_kitting.py` | 多种子验收：专家成功率、穿透、越限、入格误差、两台相机里的可见像素、预览图，写 `report.json` |

```bash
# 一集：无界面约 4 s，加 --record 约 3 分钟
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.run_kitting --output outputs/<新目录> --seed 0 --record
# 多种子验收（20 个种子约 4 分钟）
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.check_kitting --output outputs/<新目录> --seeds 20
# 只生成场景：写 outputs/kitting_task.xml 和 outputs/kitting_episode.json
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sim.kitting_task --seed 0
```

布局与抓取：
- 每侧两排各 3 个位：料盘在前排 x = 0.38 m，供料垫在后排 x = 0.48 m，三个位在 |y| = 0.10、0.22、0.34 m。
- 抓取改为前倾 45° 的侧夹，姿态 Rz(ψ)·Ry(−45°)·Rz(±90°)，左手 −90°、右手 +90°，这样腕部 D405 在手的上方。反方向转也够得到，但相机会挂在手下面、朝向机器人（用户指出后改正）。手指沿腕部 roll 连杆的局部 x 开合，转 90° 后沿 y 水平夹紧。在零件中心高度，现专家的俯抓只够到 x = 0.40 m，这种侧夹够到 0.44 m（y 0.03–0.45 m）。
- 间距由夹爪壳体决定：45° 侧夹时壳体在抓取高度沿夹持方向左右各伸出约 85 mm、向后约 65 mm。原定每侧 4 个位、间距 85 mm，第一回合壳体就压进了邻位零件，所以改为每侧 3 个位、间距 120 mm、两排相距 100 mm。
- 相机（头部俯仰 0.50 rad，零件中心高度，图像边缘留 24 px）：头部 D435 看到 x ≥ 0.38 m 的整个桌宽，躯干 D435 看到 x ≥ 0.30 m、|y| ≤ 约 0.35 m。
- 每种料一个颜色：蓝色方块（40 × 40 × 50 mm）、橙色圆柱（Ø40 × 50 mm）、绿色六角柱（对边 40 mm、高 50 mm）。每侧三种各一个，料盘三格各对应一种。
- 随机项：三种料在供料位上的顺序、位置抖动（x ±6 mm、y ±4 mm）、方块和六角柱的偏航 ±10°、料盘三格的顺序、4 种指令措辞之一。

专家：
- 准备 → 3 个回合（approach、align、grasp、lift、transfer、place、release、retract）→ 收尾。每只手按 |y| 由内到外取件，放进同色的那一格。每回合开始时读零件的实测位姿（特权信息）。
- 准备点用现任务的 (0.15, ±0.37, 0.99) m。放在料盘上方时，手腕上抬途中会擦到桌沿（穿透 1.25 mm）。
- 准备之后全部是 1 cm 路点的直线运动。关节空间的接近曾扫到已入盘的零件。
- 夹爪从不全开，只张到比零件宽 24 mm。全开时手指正好压在关节限位上，手倾斜后重力沿滑轨把手指推过限位 2.9 mm（手指没有重力前馈）。
- 逆解离关节限位：执行 10 mrad，采样 20 mrad；按关节转角分配路点时间，峰值指令转速 0.6 rad/s。不这样做时，内侧格子的肩部翻滚贴着限位，伺服超调 12 mrad，触发 2 mrad 审计。
- 成功判据：6 个零件都在同色格子里（中心偏差 x、y 各 ≤ 15 mm），竖直（倾斜 ≤ 15°），贴在料盘上，静止，不碰机器人，每格最多一件，保持至少 15 拍；每件还要有至少 100 步双指夹持的离地证据。trace 每拍另记 `round` 和 `progress`（已入格零件数）。

数据：输出与现任务相同（scene.xml、summary.json、trace.json、trajectory.npz，`--record` 时的六条视频），另加 `episode.json`（布局、分配、回合和指令）。`trajectory.npz` 的 `subtask` 与 `sim/run_expert.py` 一样存阶段名，另有 `instruction`、`round`、`progress` 和 `task`。录制的运行可以用 `data.convert_expert_run` 转成原始 episode，它自动取 `summary.json` 里的 `instruction` 作为整段指令。

验收（`outputs/kitting_check_v5/`，日志 `outputs/kitting_check_v5.log`）：
- 20 个种子全部成功，120 个零件全部入格；每个种子第一次采样就可解。
- 一集 61.6–93.0 s；最大接触穿透 0.32 mm（准备时右手指尖擦过桌面）；关节越限最大 0.67 mrad；零件离分配格中心最多 0.47 mm；碰到非本回合零件 0 拍。
- 指令的关节速度峰值是控制器限值的 0.24 倍，加速度 0.27 倍，即真机默认 30% 速度比例下的 0.80 倍和 0.90 倍。限速来自 `Scene.move` 的时间缩放（专家会话按用户决定加入）；之前的 v4 加速度最高超限 1.92 倍。
- 最慢的是第一次接近和收尾：从准备点到工作位，手臂要转动 1.3–1.4 rad（肩部翻滚），各要 5–15 s。
- 静置 3.5 s 后零件漂移不到 0.001 mm。每个零件和格框在两台相机里的可见像素，复位和准备位的最小值是头部 480 px、躯干 1013 px。
- 预览图：`outputs/kitting_check_v5/views/preview_seed000.png` 到 `002`。
- 演示录像 `outputs/kitting_demo_seed0/`：54.3 s，6/6 入格。用 `data.convert_expert_run` 转成原始 episode 得到 1358 帧，再用 `data.export_lerobot` 导出成 LeRobot v3.0 数据集（29 MB），整条链路试通。
- `outputs/kitting_check_v1/`、`v2/` 是修复前的两轮，分别有 1 个和 2 个种子失败；`v3/` 是改成一种料一个颜色之前的蓝橙两色版，`v4/` 是手腕转反、腕部相机在手下面的版本（都是 20/20 成功）。都只作记录。

共享代码的改动不影响现任务：专家重跑后 trajectory、trace 和 scene.xml 与改动前逐位一致，`tests/` 的 34 个测试全部通过。
- `build_task.build(task_objects=False)` 不放两件零件和目标垫；`PART_RGBA` 是两种零件的颜色。
- `run_expert.Scene` 新增可覆盖的 `build_root()`、`sample()`、`track_objects()`，以及 `ik(rotation=)`、`move(rotations=, speed=)`。
- `robot_kinematics.Kinematics(margin=0.0001)`；某个初值的旋转误差到 π 时 `Jlog3` 没有有限值，跳过该初值，不再抛 ValueError。

## 机器人来源与碰撞修正

### 已迁移的规划约束参考

`config/robot_constraints/joint_limits.yaml` 与 `config/robot_constraints/wheel_robot_4.0.srdf` 为 4090 原件，SHA256、源目录和核对结果记录在同目录 `SOURCE.json`。已核对 24 个关节限制、8 个规划组和 37 对禁用碰撞配置；`moveit_root` 是 MoveIt 包装层添加的虚拟根，不在裸 URDF 中。

YAML 原注释明确说明它是**保守的规划限制，尚需对照真机控制器核验**；SRDF 定义规划分组、命名姿态和自碰撞排除关系，不定义真实碰撞形状。本次按要求仅迁移文件，未将这些参数或排除规则自动应用到本机 MuJoCo 控制器。

- 源工程：4090 工作站上的 `robot_simulation` 工程。
- 原始 URDF：`Rviz/src/franzi_description/urdf/wheel_robot_4.0.urdf`，已复制到 `assets/franzi_source/`，原始文件保持不变。
- 源 MuJoCo 模型：`Mujoco/franzi/franzi.xml`，真实 STL 外观，原碰撞体为连杆包围盒。
- `sim/appearance.py` 按实验室真机视频与照片重新上色：各连杆视觉网格先拆成 CAD 连通部件，个别部件再按面区域拆分（大腿横梁、大臂网格护垫、肩部与肘部端盖），分组后用珍珠白、石墨灰、黑色、亮黑面罩等材质。头部为黑色面罩、白色饰带、深灰后壳和银白颈部支架。
- 胸口按工厂机器人的印刷：自上而下是 INOVANCE、Powered by、FMC³、ROBOTICS，下面是躯干相机槽。整块是贴合外壳曲面的贴花，没有底板；源模型的 FMC3 方形贴片已去掉。按用户确认（2026-10-24）采用工厂外观，不再用实验室那台只有 INOVANCE 的胸口。
  - FMC³ 字形按工厂视频 30 帧对齐叠加的图重画：F 是浮在上方的一条顶横加一个 Γ，M 的中间是 V 形，C 里有两个并排的圆点（眼睛），上标 3 很小、立在大写顶线上。
  - `assets/franzi_source/assets/fmc3_logo.png` 是另一版标志（梳子形 M、C 里只有一个方块），只取了它的 3。
  - 尺寸和位置以 INOVANCE 宽度为尺子，在正对胸口的工厂视频帧上量得。INOVANCE 字号与实验室那台相同，位置高约 17 mm，下面才放得下 FMC³。
- 胸口相机槽做成真机的深色窄窗。
  - 外壳开口沿躯干相机光轴看是 85.1 × 20.8 mm 的跑道形，原来能看到整台 D435 的前脸、镜头圈和绿色发射器。
  - 真机在实验室视频里是浅色凹槽中的一条深色窗，宽约 82 mm（几乎和开口一样宽），高约 10.5 mm（只有一半）。
  - 现在在 D435 玻璃前 0.4–1.0 mm 处放一块与外壳同色的盖板，中间是 82 × 10.5 mm 的带色玻璃窄窗，能隐约透出镜头。
  - 盖板和窗在躯干相机 45 mm 近裁剪面以内。在 15 个录制姿态下把它们从渲染中去掉，四路相机原始画面 0 像素变化。
- 末端按实验室照片上色：连杆夹爪的机身、法兰、连杆和相机支架为黑色，夹指座、销轴和夹指为银色铝件，夹指座上的螺钉为红色。肘部外侧黑色端盖带白色 INOVANCE，用贴合曲面的贴花画出，端盖边缘是精确圆形。URDF 传感器坐标系的标记小球已隐藏，相机安装校验仍使用这些坐标系。
- 四路相机改用官方 RealSense 模型，放在 `assets/appearance/cameras/`，来源记录在各自的 `SOURCE.json`：
  - D435：MuJoCo Menagerie 的 D435i 分色模型，由 realsense-ros 的 `d435.dae` 而来，Apache-2.0；转为二进制 STL，几何不变。按镜头位置对齐 URDF 传感器坐标系：左红外在原点、RGB 偏 15 mm、玻璃在原点前 4.2 mm。
  - D405：realsense-ros 的 `d405.stl`，毫米换算为米，与 URDF 方块完全重合。
  - 头部 D435 的黑色支架保留，只替换相机本体。
- 面罩不开窗。头部 D435 在面罩后 16–20 mm，而相机渲染近裁剪平面是 45 mm，面罩本来就不在头部相机画面里；按用户要求（2026-10-24）撤掉了曾做过的开窗版。把整个面罩从渲染中去掉，四路相机原始画面 0 像素变化，所以开不开窗都不影响录制。
- `sim/fittings.py` 补 CAD 缺少的外观件：下背部黑色竖纹背板，上背部红色急停、接口和蓝色指示灯，大腿窗口深色背板，底盘前部与前角绿色灯带。
- 以上生成的网格缓存于 `assets/appearance/{painted,wordmark,fittings}/`，以源 STL 哈希和参数为键。新增 geom 全部无质量、无接触，不改源 URDF、惯量或碰撞体。腿部按用户确认保持直立工位（与实验室照片一致），不采用工厂视频中的 Z 形蹲姿。
- `sim/build_task.py` 只生成派生场景：手腕的大包围盒替换为 CAD 连通零件各自的凸包（每侧 49 个）；手指包围盒替换为手指 STL 凸包，保持实际平面夹持面，螺孔等凹部仍为凸包近似。
- 保留机器人与环境的碰撞，并额外启用左臂与右臂的碰撞；同侧相邻连杆沿用源模型的自碰撞取舍。其他连杆仍是原始包围盒，不能把该版本称为完整 CAD 自碰撞模型。
- 底盘在本任务中固定，用于隔离双臂操作问题；没有移动底盘控制实验。
- 抓取不使用焊接/吸附/对象位姿跟随。初始化后只通过关节目标和手臂重力补偿力矩控制机器人，物体由接触和摩擦运动。运行期间不会改变碰撞 mask。

## 验收与数据约定

- 真机跟仿真对齐（用户 2026-10-24 确定）：逆解和正解只在主机侧用一份共享运动学 `robot_kinematics.py` 来算，仿真和真机都只接收关节指令（`MoveJ`、关节遥操作）。控制器自己的逆解、`MoveL`、笛卡尔遥操作都不使用。simbridge 的 IK/Fk 接口也调用这份实现，`simbridge/sdk_kinematics.py` 只做 SDK 约定的换算（见「运动学接口」）。
  - 正解和雅可比用 Pinocchio 计算（用户 2026-10-24 确定），依赖 NumPy、SciPy、PyYAML 和 pin，都在 `inovance-fineart` 环境里，不依赖物理引擎。模型取自 URDF：从 `base_link` 经腰腿和手臂关节到抓取点；URDF 与控制器镜像的 MDH 一致，用户此前也验证过它与真机一致。
  - 正解和仿真相差不超过 1.1e-15 m。逆解就是示教原来的算法，搬过来后和原 MuJoCo 逆解最多相差 1.7e-7 rad。
  - 关节限位用对齐后的控制器限位。`to_sdk()` 和 `from_sdk()` 负责 URDF 与 SDK 关节值的换算。
  - 逆解默认在对齐后的限位内再留 2° 规划余量（`PLANNING_MARGIN`，用户 2026-10-24 确定）。原来抓放任务的接近阶段，两只手腕俯仰离控制器 −60° 限位只有 0.065°，真机跟踪稍有超调就可能触发软限位；留出余量后离限位 2.06°，抓取姿态不变。simbridge 模拟控制器逆解的接口仍用 0.0001 rad，长任务自己用 10 mrad 和 20 mrad。
  - 示教的 `Scene.ik` 和直线运动都改为调用它；原来示教自带的 MuJoCo 逆解已删除。
- MuJoCo 500 Hz，关节目标 25 Hz 零阶保持，视频 25 FPS。
- 关节限位按控制器镜像对齐（用户 2026-10-24 要求）。`build_task.align_joint_limits()` 写入的数值来自 `robot_kinematics.controller_ranges()`，它读 `config/robot/franzi_sdk.yaml`：镜像 `sysctrl_params.json` 和头、腰配置中的限位，以及 SDK 与 URDF 的正负号映射。
  - 关节范围取 URDF 与控制器的交集。控制器更紧的一侧再留 SDK 的 0.001 rad 余量，执行器指令范围同步修改。
  - 两臂共收紧 10 个关节：肩俯仰上限 149.9° → 140°，肘下限 −150.1° → −148°，肩和腕偏航 ±170.17° → ±170°，腕俯仰 ±60.16° → ±60°。
  - 控制器比 URDF 宽的地方暂时保持 URDF：肘上限 15° 对 7.45°，大腿下限 −10° 对 0°，腰俯仰下限 −75° 对 −30°。要放宽须先在真机上确认机械范围。
  - 速度和加速度：`Scene.move` 把每段动作拉长到手臂的 25 Hz 指令不超过控制器默认 30% 速度比例下上限的 0.9 倍（用户 2026-10-24 确定），这样真机按默认比例回放也不会被限速。只有「准备」和「接近」两段被拉长，参考示教从 24.5 秒变成 29.9 秒，峰值为该上限的 0.90 倍（速度）和 0.48 倍（加速度）。这些上限来自离线镜像，没在真机上验证。
  - 头部转向改为 3.3 秒的最小加加速度曲线，峰值 16 °/s、15 °/s²、48 °/s³，满足控制器头部限位（30 °/s、40 °/s²、50 °/s³）。原来的 1.5 秒 smoothstep 曲线加速度达 76 °/s²。
- 手臂、腰腿（小腿、大腿、腰俯仰和腰偏航）、头部关节都加重力前馈（`qfrc_applied = qfrc_bias`，只是力输入）。
  - 原来腰腿只靠伺服刚度撑，手臂前伸时每个关节下沉最多 7.5 mrad，合计约 1.3°；头部俯仰下沉 3.8 mrad。头部和躯干相机视角跟着偏，经 SDK 回放专家目标时手在 PyBullet 里偏约 9 mm。
  - 现在静止时误差为 0，运动中最多约 1 mrad，和真机伺服一样保持指令姿态（用户 2026-10-24 确定）。头部开场转向的 1.5 秒内仍有约 5 mrad 的跟随滞后。
- 每个物理步检查所有已启用碰撞对的接触深度，超过 1 mm 立即终止；检查关节限位误差。
- 每件物体须有双侧手指接触、抬起的证据，最终进入目标区域、速度低于 0.025 m/s、与机器人脱离接触，稳定至少 15 个控制拍。
- 16 维动作/状态：左臂 7 关节 rad、右臂 7 关节 rad、左右各一维夹爪闭合比例（0 全开 / 1 全闭）。双指机械耦合为等量反向目标。
- 机载相机：头部 D435、腰部/躯干 D435、左腕 D405、右腕 D405。`overview` 只作展示，不作为策略输入。四路相机的原点与挂载父体取自 URDF 固定传感器关节，跟随实际关节状态；光学帧轴转换与模拟视场设置在 `sim/camera_rig.json`。
- 用户尚未提供真机 K/畸变参数，当前为**未标定的模拟针孔内参**，不是完整 RealSense RGB-D 仿真，也没有声称模拟了深度噪声或 RGB/深度基线。头部安装位保留在面罩内；渲染近裁剪平面为 45 mm，避开相机自身壳体遮挡，不改相机安装位置或碰撞。
- 阴影来自一盏方向主光。MuJoCo 方向光的阴影区域以灯的位置为轴心，原来的灯位 (2, −2, 3.5) 让轴心落在工作区外 1 m 多，只能用 4096 贴图覆盖 4.6 m，阴影边成锯齿（胸口槽下方、桌上手爪阴影最明显）。
  - 现在主光方向不变（照明不变），灯位挪到 (1.09, −0.68, 3.48)，让光轴正对工作区。阴影贴图 8192，覆盖半宽 1.44 m（shadowclip 0.63），阴影像素从 1.1 mm 细到 0.35 mm。
  - 与关阴影的渲染对比，四路相机原有阴影丢失不超过 0.8%（只在边缘）。若用旧灯位收紧覆盖范围，会丢掉最多一半阴影。
  - 阴影仍是硬边。多盏抖动光源模拟软阴影会出现分层条纹、渲染慢 2–3 倍，未采用。现在 5 路相机每帧渲染约慢 40%。
  - 修复前后对比在 `outputs/appearance_alignment/shadow_fix_sheet.png`。
- 当前工位将真实头部俯仰关节平滑转至 0.50 rad 以俯视（关节上限 0.52 rad），不人为追踪物体。物块中心放在 x=0.38 m、目标垫在 x=0.40 m；此前 0.28/0.30 m 时头部相机整段任务都看不到工作区。前移后头部和躯干相机从接近到放置都能同时看到两个物块、两个目标垫和两只手。物体真值只用于专家与评分，不是未来 FineART 视觉策略的输入。
- 专家在下探、抬起、搬运、放置和撤回时沿笛卡尔直线运动：每 1 cm 取一个逆解路点，时间曲线不变。工作区接近臂展边缘时，纯关节插值会让手中途前冲约 2 cm、压到物块边上。最后收回准备姿态用 3 s，避免手臂回摆把直立时正压在下限上的大腿关节推过下限。
- 数值轨迹目前为专家原始采集产物，**尚未宣称为 LeRobot v3 数据集**；后续导出应保留相机时间戳与动作语义。
- 原 OMX 6 维策略不能直接作为此双臂任务的已适配权重使用；需要重建特征/归一化配置并微调。

## 仿真与真机共用原厂 SDK（simbridge 多仿真器框架）

上层程序（FineART、遥操作、数据采集、控制脚本）只调用**未修改的原厂 V1R1C9 SDK**（`ArmClient`、`ModuleCommunicationClient`）。连哪台“机器人”只由连接配置决定，上层代码不变：

| 连接配置 | 服务端 | 地址 |
|---|---|---|
| `config/connections/real_rk3588.yaml` | 真实 RK3588 原厂服务端（本仓库未连接过） | 192.168.23.30:8000/8100 |
| `config/connections/sim_mujoco.yaml` | simbridge + MuJoCo | 127.0.0.1:18000/18100 |
| `config/connections/sim_isaacsim.yaml` | simbridge + Isaac Sim（草稿，未运行、未验证） | 127.0.0.1:18400/18500（4090 上运行时经 SSH 隧道） |

```python
from sdk.stock_sdk import connect

arm, sensors = connect("config/connections/sim_mujoco.yaml")  # 真机：real_rk3588.yaml
arm.MoveJ([[...14 个关节角 rad...]], side=15)                     # 原厂 API，原样调用
```

### 分层与目录

```
上层程序 ── 原厂 SDK（sdk_review/feimi1010/sdk/，源文件不改）
               ├─ 真机：RK3588 原厂服务端
               └─ 仿真：python -m simbridge.server --backend config/backends/<引擎>.yaml
simbridge/protocol.py   rest_rpc 20 字节帧头 + MessagePack + 路由；路由名从原厂客户端源码静态提取；不导入任何引擎
simbridge/services.py   每个 SDK 方法的请求校验和返回格式；未提供的方法 → 明确的 UNSUPPORTED 协议错误
simbridge/control.py    统一控制语义（SDK 关节坐标）：模式、队列、限位、使能门控、软急停、故障锁存
simbridge/robot.py      读 config/robot/franzi_sdk.yaml：SDK↔仿真关节映射、限位、夹爪、相机、坐标系、安全阈值
simbridge/sdk_kinematics.py SDK 运动学接口：调用共享运动学 robot_kinematics，只换算 SDK 的关节值、坐标系和位姿格式
simbridge/runtime.py    拥有仿真器的线程（进程主线程）：命令队列、每步一致快照、渲染请求只在该线程执行
simbridge/backends/     适配器，按名称懒加载：mujoco_backend.py、isaacsim_backend.py（草稿）
sdk/check_sdk_kinematics.py 离线验证工具：控制器 MDH 与 URDF 的运动学对照 → config/robot/kinematic_crosscheck.json
sdk/convert_pybullet_assets.py 从 MuJoCo 场景导出带同一套碰撞体的 URDF 场景（Isaac Sim 草稿用；名字沿用已下线的 PyBullet 后端）
sdk/make_sdk_program.py     生成验收用的 SDK 坐标关节程序 → config/sdk_acceptance/program.json
sdk/sdk_acceptance.py       只用原厂 SDK 的验收程序（所有后端同一份）
sdk/verify_backends.py      启动各后端服务、运行验收、按类别打分 → outputs/<run>/report.json
sdk/sdk_cartesian.py        主机侧笛卡尔目标：用 robot_kinematics 求解，以 SDK 关节值发 MoveJ 或关节遥操作（仿真、真机同一套）
tests/test_simbridge.py 不依赖引擎的单元/集成测试
```

引擎模块只在选中对应后端时导入。PyBullet 后端已于 2026-10-24 按用户决定下线（只有 MuJoCo 场景按真机对齐了外观、相机、灯光、限位和躯干保持；IK/Fk 接口也因此能直接用共享运动学），此前的 PyBullet 验收结果留在 `outputs/sdk_backends_v1/`、`v6/`、`v8/`。

### 线程与时钟

- RPC 工作线程不访问仿真器：命令经队列交给物理线程，在两步之间执行；回复在下一次快照发布后才返回，所以 `MoveJ` 返回 0 后立即查询 `GetMotionStatus` 一定看到 1。状态查询读取每个物理步发布一次的不可变快照。相机渲染请求也排队到物理线程（OpenGL/Metal 上下文和 Isaac Sim 都要求在拥有场景的线程渲染）。
- 两种循环：`loop: framework`（框架调用 `step()`）；`loop: simulator`（引擎拥有主循环，每步回调框架的 `pre_step/post_step`；MuJoCo 适配器实现并验证了这一模式，Isaac Sim 草稿用物理回调）。
- 仿真时间 `sim_time`：物理时钟（500 Hz 步长），场景启动为 0。控制周期：设定点每个物理步更新一次，引擎在两步之间保持；状态轨迹以 25 Hz 采样。图像 `timestamp`：渲染完成时的系统 UTC 毫秒，与仿真时间无固定比例；`server/camera_log.json` 记录每帧的 seq、UTC 毫秒、仿真时间和物理步号。服务端按实时节拍运行；录像时渲染开销会使仿真慢于实时（实测见下），客户端一律轮询状态而不是按墙钟睡眠。

### 统一控制语义（公共层，引擎无关）

- **接收 ≠ 完成**：运动类命令返回 0 只表示已校验并入队；到位以 `GetMotionStatus` 为准（0 到位静止，1 运动或未收敛，-1 故障，-3 已停转但长时间未到位）。
- 错误码沿用原厂：参数非法/超限/非有限数/长度不符 `-3`；未就绪（未使能、软急停中）`-5`；软急停中上使能 `-8`；物理故障后 `-9`；遥操作会话与轨迹冲突 `-10`；腰/头运动等存在但不仿真的选项 `-4`。返回字典附 `err_msg` 说明原因（原厂客户端只返回状态码，不会丢失语义）。
- 原厂客户端里存在、但本后端不提供的方法（`MoveL`、笛卡尔点动与遥操作、`FkLink`、MDH 查询与写入、力控/阻抗、夹爪力、标定、导航等）一律返回协议级错误 `UNSUPPORTED: arm_sdk.<方法> ...`，无论 `raise_on_error` 如何设置，原厂客户端都会抛异常，不会得到空列表式的“假成功”。
- `MoveJ`/`Home`：关节空间直线，同步梯形速度曲线（所有关节同时起停），速度/加速度 = 限值 × 速度比例。新 `MoveJ` 在执行中的轨迹后排队（仿真策略）。
- 关节遥操作队列（单臂/双臂）：按客户端的推送节拍回放。每点的时长取本次会话最近 50 次推送间隔（约 2 s）的中位数，凑够 8 个间隔之前用默认 40 ms；超过 0.2 s 的间隔算作停顿，不计入、也不清空历史，在仿真时间里线性插值、延迟一个点，所以仿真慢于实时也保持客户端的节拍，客户端自身的计时抖动也不会变成速度抖动。推送请求只用最新快照校验后立即回复，不等物理线程；物理线程再校验一次，期间状态变化导致的丢点记入事件。速度/加速度限位只作安全钳制：超限的流会被钳制而偏离路径，客户端应发送满足限位的轨迹。队列容量 500。
- 使能门控：Disable 冻结指令并维持仿真伺服保持，不模拟断电掉落。软急停：按加速度限值制动后保持，拒绝运动、上使能和夹爪命令，`SoftStop(False)` 后恢复。
- 物理故障（任一接触穿透 > 1 mm，或关节越限 > 0.002 rad）：冻结物理、锁存故障，`IsReady=false`，`GetMotionStatus=-1`，运动/使能返回 `-9`，`GetCurAlarm` 给出原因（仿真自定义告警码）。故障只能重启仿真会话清除。
- `ReadJointState` 额外返回 `torque_source`：`simulated_actuator_effort`（仿真执行器力，含重力前馈）、`measured_joint_torque_sensor`（只有真机可能有）或 `unavailable`（此时 `torque` 为空，不填假值）。
- 夹爪：原厂 `0=全闭、1000=全开`，线性映射到仿真手指行程；`GripperReadGripState` 由测量值和接触查询得到（0 运动中、1 到位、2 夹持、3 掉落）。`GripperInitialize` 在仿真中是空操作并如实说明；夹爪力度接口不支持。

### 机器人配置与坐标映射

`config/robot/franzi_sdk.yaml` 与引擎代码分离，保存：SDK 关节顺序和仿真关节名、正方向与零位偏移、限位（仿真模型范围 ∩ 镜像控制器范围）、速度/加速度（保守规划 YAML）、夹爪映射、相机名称与安装关系（内参见 `sim/camera_rig.json`，未标定）、基座/法兰/TCP 定义、安全阈值。SDK 关节值统一为 `q_sdk = sign × q_urdf + offset`（rad）。`--joint-convention urdf_passthrough` 可切回旧版“URDF 原值”行为，仅用于调试。

肘关节问题的结论（`sdk/check_sdk_kinematics.py`，结果在 `config/robot/kinematic_crosscheck.json`）：

- 镜像持久化参数 `machine_params.json` 的 `bio_para` 与 `config11/mdh.json` 的 MDH 表完全相同。按 Craig 改进 DH（`θ = q + theta_offset`）解释，它与源 URDF 的手臂链**运动学完全一致**：500 组随机关节角上位置误差 ≤ 1.3e-14 m、姿态误差 ≤ 2.1e-14 rad。同一表按标准 DH 解释最好也有 0.072 m 误差（反证约定）。
- 对 128 种正负号组合拟合偏置和基座/法兰变换，得到 14 个运动学上完全等价的解（腕部等对称性所致），仅靠 FK 无法唯一确定；再用镜像关节范围筛选，只剩一个：**左右臂 sign = [-1, +1, -1, -1, -1, -1, +1]，偏置全为 0**。肘关节：URDF −150°～7.45° 映射为 SDK −7.45°～150°，与镜像 −15°～148° 一致，即屈肘在 SDK 中为正。
- 第 1、7 关节的零位偏置会被基座/法兰坐标吸收，FK 无法辨识；0 是唯一与范围一致的值，标为“范围推断”。腰部：FK 精确，4 个等价解中按范围选 [1,1,1,1]，但腰俯仰范围冲突（URDF ±30°，镜像 −75°～30°）。头部：镜像无 MDH，未标定，按 URDF 原值透传。
- 镜像中 `mdh_repeat_exact*.json` 的关节/笛卡尔样本自相矛盾（起止关节角几乎相同而笛卡尔差几十厘米），不能作为 FK 真值，未使用。
- 限位差异：SDK 层按“仿真模型范围 ∩ 镜像控制器范围”（两侧各留 0.001 rad 余量）拒收越界目标（`-3`）。另一会话的 `build_task.align_joint_limits` 把派生 MuJoCo 场景的关节范围裁到同一交集（双臂共 10 个关节被裁）；MuJoCo 适配器加载时检查模型范围位于 URDF 范围内、且包含 SDK 接受的全部目标。专家因此不再规划到控制器范围之外（此前腕俯仰会超出 0.15°），验收程序也不再需要夹点。

**以上全部来自离线镜像文件与 URDF 的对照，不是在线控制器回读，也没有真机姿态验证。不能宣称仿真与真机运动已经一致**：零位标定、编码器方向、真机生效参数、法兰到 TCP（flan2tool）、真实伺服与轨迹算法都未验证。用户反馈此前实测过 URDF 与真机一致；这是用户的经验，本仓库没有对应的测量记录。

### 运动学接口（共享运动学）

按用户决定（2026-10-24，“真机跟仿真对齐”），逆解和正解只用一份主机侧共享运动学 `robot_kinematics.py`（Pinocchio），仿真和真机都只接收关节命令。simbridge 的运动学接口直接调用它，`simbridge/sdk_kinematics.py` 只做 SDK 约定的换算：

| 接口 | 仿真实现 |
|---|---|
| `Fk` / `FkTool` | `robot_kinematics.fk`，换算成手臂 MDH 基座 A0 → 法兰 A7；工具 = 法兰 × flan2tool |
| `IK`（输入基座→工具） | `robot_kinematics.ik`（示教用的同一求解器），初值取当前测量关节角；无解返回 `-6`（原厂定义为“笛卡尔目标不可达”） |
| `FKb` / `FKbTool` / `IKb` / `IKbTool` | 同上，在基座系 B = URDF `base_link`，即仿真机器人的真实位姿；`torso_joints` 为空时读当前腰部关节（同手册） |
| `GetFlan2tool` / `SetFlan2tool` | 启动时为单位阵（镜像里没有工具坐标系，推断）；设置只在本次会话内有效 |
| `GetCurCartesianPose(Dual)` | 当前测量关节角的 `FkTool`，在 A0 系（推断） |
| `FkLink`、`GetMdh`、`GetMdhCompensation`、`GetArmEulerAngle`、`SetMdh*` | 不提供（UNSUPPORTED）：这些只是控制器自身模型的接口，共享运动学用不到 |

约定（来自原厂手册、Python 客户端和 C++ 头文件）：
- SDK 关节值 `q_sdk = sign × q_urdf + offset`。
- A0 在 torso_Link 中、法兰在 wrist_roll_Link 中的位姿取自 `kinematic_crosscheck.json`。
- 4×4 矩阵按列优先 16 个数传输，平移单位 m，按控制器的 float32 精度收发。
- 位姿 `[x, y, z, rz, ry, rx]`，`R = Rx(rx)·Ry(ry)·Rz(rz)`（外旋 zyx，等价于 SciPy `from_euler("zyx", [rz, ry, rx])`），读回时 `ry ∈ [−π/2, π/2]`。

`MoveL`、笛卡尔点动和遥操作不实现：真机上它们会用控制器自己的逆解。没有 Pinocchio 的服务端（例如 4090 上的 Isaac Sim 环境）照常启动，运动学接口明确返回 UNSUPPORTED。

**真机注意。** 手册写的控制器内 FKb 常量（`WbcTmpKinematics`，镜像里的 SDK 服务端确有这条代码路径）与 URDF 在直立姿态相差 (0, ∓0.0725, −0.26355) m，即 0.273 m：
- 手册把腰 MDH 基座放在 B 原点，实际在小腿轴上，高 0.24785 m。
- 手册把腰末端当作 torso_Link，实际低 0.0157 m。
- 手册把 A0 放在 URDF 肩关节原点，而 MDH 的 A0 在肩部三轴交点，再向外 0.0725 m。

所以真机上的 `FKb`/`IKb` 很可能与仿真不同；按上面的决定，真机的笛卡尔目标应在主机侧用 `robot_kinematics` 求解（见下面的 `sdk/sdk_cartesian.py`）。另外，镜像的 SDK 服务端内嵌了一份 `wheel_robot_4.0.urdf`，36 个关节与 `assets/franzi_source` 完全相同，用于它的质心求解器。

验证（`tests/test_simbridge.py` 和验收打分）：
- 经共享运动学算出的 `Fk` 与镜像 MDH 表在同一 SDK 关节值下一致到 1e-12，说明坐标系、映射和单位都对得上。
- `FKb` 与 `robot_kinematics` 的法兰位姿一致到 1e-12。
- IK/IKb 往返在 1e-5 m 以内；欧拉角与 SciPy 互转误差 2e-13。
- 验收打分时用 trace 中的 SDK 关节角和腰角经共享运动学算出抓取点，再与 MuJoCo 自己的抓取点比较（数值见下方验收结果）。

### 主机侧笛卡尔目标（`sdk/sdk_cartesian.py`）

上层程序要按笛卡尔目标动手臂时，在主机上解逆解、只发关节命令，仿真和真机收到的是同一组 SDK 关节值，控制器自己的逆解、`MoveL`、笛卡尔遥操作都不用：

```python
from sdk.stock_sdk import connect
from sdk.sdk_cartesian import CartesianArm

arm, sensors = connect("config/connections/sim_mujoco.yaml")   # 真机换成 real_rk3588.yaml
cart = CartesianArm(arm)
p, R = cart.pose("left")                         # 当前抓取点（世界系）和姿态
cart.move({"left": p + [0, 0, 0.05]})            # 一次 MoveJ，等到位
cart.line({"left": p + [0.04, 0, 0.05], "right": ...}, speed=0.05)  # 直线，经关节遥操作队列 25 Hz 推送
```

- 目标是抓取点（`{side}_grasp`，即 `frames.tcp`）在仿真世界系中的位置：`base_link` 在 franzi.xml 的偏移处（`cart.kin.base`），坐标轴与世界对齐。
- 姿态默认保持当前姿态，也可以用 `rotations={"left": R}` 指定。
- 每次求解都读入实测的手臂和腰部关节（`GetJointState`）。
- `move` 两臂都给时发一条双臂 `MoveJ`，轮询 `GetMotionStatus` 等到位；运动以故障（-1）或停转（-3）结束、或 IK 无解时抛 `CartesianError`；IK 无解时什么都不发送。
- `line` 用 `robot_kinematics.line` 每 `speed / rate` 取一个逆解路点，两臂重采样成相同点数，经 `TeleJoint(Dual)` 按绝对时刻表推送，结束后 `StopTeleJoint(Dual)`。若某个关节会超过 SDK 当前速度上限的 0.9 倍（`GetJointVelLimit`，已含速度比例，真机也一样），自动放慢并返回放慢倍数。

验证：
- 测试（`tests/test_simbridge.py`，测试服务端）：抓取点到位误差 < 0.1 mm；直线路点都在线段上；桥实际执行的关节经正解落在目标上；IK 无解时不发送任何命令。
- 实际 MuJoCo 服务端（含物理和伺服）：
  - 左手上移 5 cm：到位时误差 0.061 mm，姿态变化 0.19 mrad，用时 0.83 s；
  - 双手直线 4 cm（5 cm/s）：刚到位时误差 0.044 / 0.137 mm，稳定后按 MuJoCo 自己的抓取点为 0.004 / 0.012 mm，没有触发放慢。
- 真机未试。仍要先核对关节方向、零位和 TCP，并显式设置速度比例：真机默认 30%，仿真为 100%。

### 后端支持与验证状态

| 后端 | 实现 | 本机验证 | 解释器 | 循环 | 说明 |
|---|---|---|---|---|---|
| MuJoCo 3.13 | 已实现 | 已验证（两种循环） | inovance-fineart | framework / simulator | 包装 `run_expert.Scene`，沿用全部物理规则和逐步审计；所有图像尺寸共用一个渲染上下文（每个上下文各带一张 8192² 阴影贴图，约 256 MB） |
| Isaac Sim / Isaac Lab | 草稿 | **未运行、未验证** | 4090 工作站的 Isaac Sim 5.1（本机 macOS arm64 不能运行） | 设计为两种 | 按 4.5/5.x API 写成，并对照 4090 上 Isaac Sim 5.1.0 源码修正了导入命令、World/PhysicsContext、Articulation 接口、物理步回调和接触报告的调用；从未启动过 Isaac Sim，不能视为可用 |

### 启动服务与切换后端

```bash
cd /Users/puheliang/workspace/inovance-fineart
# MuJoCo（端口来自 config/connections/sim_mujoco.yaml）
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m simbridge.server \
  --backend config/backends/mujoco.yaml --output outputs/<新目录> [--record]
# MuJoCo，引擎拥有主循环
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m simbridge.server \
  --backend config/backends/mujoco_owned_loop.yaml --output outputs/<新目录>
# Isaac Sim 草稿的场景：sim/build_task.py 等改动后重新从 MuJoCo 场景导出
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sdk.convert_pybullet_assets
```

服务端只监听连接配置中的地址（默认回环），60 秒无请求退出；输出目录须不存在，写入 `ready.json`（后端、能力声明、映射、端口）、`trace.json`（25 Hz 状态/设定点/模式/夹爪/真值物体）、`camera_log.json`、`server_result.json`（故障、RPC 记录、实时率），`--record` 时写 `session.mp4`。

同一份验收程序跑每个后端（现在是 MuJoCo 的两种循环）：

```bash
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sdk.verify_backends \
  --output outputs/<新目录> --backends mujoco,mujoco_owned_loop \
  --task --fault-probe --record
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m pytest tests -q
```

`sdk/sdk_acceptance.py` 拒绝非回环地址：其中的动作只在仿真中检查过，不能直接对真机运行。

### 验收结果

同一份 `sdk/sdk_acceptance.py`（只调用未修改的原厂 SDK）连接服务端，只换连接配置；结果按类别记录，不合成一个通过率。最新一次记录在 `outputs/sdk_backends_v10/`（`inovance-fineart` 环境，共享运动学接口，PyBullet 已下线；任务来源 `outputs/expert_rate_limited/`），汇总为 `report.json`，日志为 `outputs/sdk_backends_v10.log`：

| 类别 | MuJoCo（框架驱动） | MuJoCo（引擎自有循环） |
|---|---|---|
| 通信兼容：协议、返回格式、四路相机、未实现接口明确报错 | 28/28 | 28/28 |
| 控制语义：接收≠完成、-3/-5/-8/-10、单/双臂遥操作、软急停、使能 | 37/37 | 37/37 |
| 运动学接口：Fk/FkTool/IK/FKb/IKb 自洽，位姿约定，工具坐标，-3/-6 | 23/23 | 23/23 |
| 运动学映射：全程 SDK 值 = sign·仿真值 + offset | 偏差 0 | 偏差 0 |
| 共享运动学（SDK 关节角 + 腰角）对 MuJoCo 抓取点 | 1.0e-15 m | 9.0e-16 m |
| 物理表现：双臂到位、重力力矩、夹爪 750/250/1000、软急停制动、Home、IKb + MoveJ 抬高 3 cm | 10/10（抬高误差 0.16 mm） | 10/10（0.16 mm） |
| 整段最大接触穿透 | 0.11 mm | 0.11 mm |
| 故障处理：把左手压进桌面 | 5/5，手指越限审计触发，物理冻结并经 SDK 报告 | 5/5，同左 |
| 完整任务（经 SDK 重放专家关节目标）：SDK 侧 | 4/4，抬起后夹爪报 2（夹持） | 4/4 |
| 完整任务：仿真真值（到位、静止、脱离、≥ 0.4 s 双爪悬空） | 成功，误差 0.3 / 0.1 mm | 成功，0.3 / 0.1 mm |
| 录像时实时率 | 0.28 | 0.28 |

v10 与 v9 的检查项相同，变的是任务来源。示教改为逆解留 2° 余量、按真机默认 30% 速度比例限速（`outputs/expert_rate_limited/`），据此重新生成的 `config/sdk_acceptance/program.json` 只有撤回一段放慢 ×1.005；v9 的程序放慢了准备 ×1.52、接近 ×1.77。IKb 抬高误差从 0.025 mm 变成 0.16 mm：两版的逆解残差都是 0，这是 MoveJ 报完成时手臂离目标还差的一点，起点换成了新示教的准备姿态（肩滚和腕俯仰各差约 2°）。v9 同样全部通过，记录在 `outputs/sdk_backends_v9/`。

v9 起通信类比 v8 多三项：`FkLink`、`GetMdh`、`GetArmEulerAngle` 改为明确不支持。下线前最后一次三后端验收是 `outputs/sdk_backends_v8/`：PyBullet 全部检查通过，任务真值放置误差 0.8 / 1.0 mm。

演示视频：`outputs/sdk_backends_v10/mujoco/server/session.mp4`（1330 帧，25 fps）、`mujoco_owned_loop/server/session.mp4`（1342 帧）。画面为总览相机加四路 SDK 相机小图和状态栏。每个后端目录还有：
- `config/`：本次使用的后端、连接、机器人配置快照；
- `client/sdk_calls.json`：每次 SDK 调用的参数、返回和耗时；
- `client/verification.json`：检查项；
- `client/*.png`：经原厂图像 API 取得的图像；
- `server/ready.json`：能力声明与映射；
- `server/trace.json`：25 Hz 状态、设定点、模式、夹爪、真值物体；
- `server/camera_log.json`；
- `server/server_result.json`：故障、全部 RPC、实时率；
- `report.json`。

原厂 SDK 的 3153 个文件与 `sdk.zip` 逐字节一致。

MuJoCo 两种循环给同一个 SDK 目标后，仿真关节角相差不超过 0.96 mrad，手部与相机坐标相差不超过 0.95 mm。差别来自首条指令落在不同的仿真时刻（开场头部转向还没结束），不是循环方式本身：v7 中两者的首条指令只差 2 ms，结果相差 4.7 µm。

**PyBullet 任务重放失败与节拍修复（2026-10-24）。** `outputs/sdk_backends_v7/`（修复前）中 PyBullet 的客户端检查全部通过，但仿真真值失败：放置误差 24.3 / 26.7 mm，超过 25 mm 阈值（v6 时是 3.5 / 3.8 mm）。

- 现象：放置下降时手先向外摆约 30 mm，再回到目标点，物块被拖离垫子中心。手和设定点一起偏，也就是引擎在照常跟踪一条已经走样的设定点路径。
- 原因在公共层的遥操作回放。PyBullet 步进和渲染时持有 GIL，推送请求的回复会被拖慢（最长 218 ms）；客户端按绝对时刻表随后把落下的点连续补发，服务端看到的推送间隔有 28% 不足 10 ms。
- 原来按“最近 8 个间隔的中位数”定节拍，遇到停顿就清空历史。补发时中位数塌到零点几毫秒，回放撞上速度/加速度钳位，设定点偏离程序路径最多 0.29 rad。v6 也有同样的突发，只是没落在敏感阶段。
- 现在取本次会话最近 50 个间隔的中位数：凑够 8 个间隔之前用默认 40 ms，停顿不计入也不清空历史。用 v6、v7 记录下来的推送时刻回放验证，没有一个点快于 32 ms。
- 结果：v8 中 PyBullet 设定点离程序路径最多 0.017 rad，放置误差 0.8 / 1.0 mm。

**视频晃动的原因与修复（2026-10-24）。** 早先的 `outputs/sdk_backends_v1/` 视频里机器人一直在晃，有三个独立来源，`outputs/sdk_backends_v6/shaking_before_after.png` 是修复前后对比：

1. 源模型腿/腰位置伺服 kp 6000 N·m/rad、阻尼只有 5 N·m·s/rad（阻尼比约 0.01），整个上身约 1.5 Hz 前后摆，每次手臂运动都会重新激起。`run_expert.Scene` 现在给小腿、大腿、腰俯仰和偏航加伺服速度反馈（初始姿态下阻尼比 0.7），专家演示和 SDK 仿真同时受益；阻尼比取 1 时专家把腕俯仰停在 URDF 限位上，会触发 0.002 rad 越限审计，所以用 0.7。
2. 专家 IK 在直线运动的相邻 1 cm 路点之间，7 自由度手臂的冗余自由度（肘部摆角）来回跳，关节加速度达规划限值的 17–32 倍，手在直线上而手臂在抖。`Scene.ik` 现在对串联路点加一个很小的就近正则（权重 0.002），峰值加速度降到限值的 0.9 倍以内。
3. 旧的遥操作队列让每个关节尽快追到每个点，追上 25 Hz 流后刹停、等下一点、再加速，走走停停；改为上面描述的按节拍回放。专家轨迹本身仍有两段关节空间大动作超过保守限速 1 rad/s，`sdk/make_sdk_program.py` 把这两段整体放慢（准备 ×1.52、接近 ×1.77），路径不变。示教改为按真机默认速度比例限速（`outputs/expert_rate_limited/`）后，重新生成的程序只有撤回一段放慢 ×1.005。

同一时段另一会话给腿、腰和头部关节加了重力前馈，消除了约 1.3° 的静态下沉；v6 的结果包含这三处修正、该前馈和上面的关节范围裁剪。修复后 MuJoCo 任务重放中手部路径二阶差分的 95 分位从 11.1 mm 降到 0.95 mm，腰俯仰摆动均方根从 1.37 mrad 降到 0.04 mrad，物体放置误差从 4.6/2.5 mm 降到 0.5/0.2 mm（`outputs/sdk_backends_v6/shaking_before_after.png`）。专家无头运行同样通过（`outputs/sdk_task_source_expert_v6/`）；修改前录制的专家视频没有这两处修正。

### 新增仿真器

1. 在 `simbridge/backends/<引擎>_backend.py` 实现 `SimBackend`（`backends/base.py`）：`load / capabilities / dt / apply / step / read / render / close`，可选 `run_owned_loop`、`task_state`、`tracked_frames`。只说仿真关节名和 SI 单位；`step()` 每步做穿透和越限审计，超阈值抛 `PhysicsFault`；做不到的能力在 `capabilities()` 里声明为不支持，方法本身抛异常。
2. 在 `simbridge/backends/__init__.py` 的 `REGISTRY` 登记模块名和实现/验证状态（懒加载，不影响只用 MuJoCo 的环境）。
3. 新建 `config/backends/<名>.yaml`（解释器、机器人配置、连接配置、运行参数）和 `config/connections/sim_<名>.yaml`。
4. 需要时增加资产转换（参考 `sdk/convert_pybullet_assets.py`：碰撞体从 MuJoCo 派生场景导出，保证各引擎碰撞定义一致）。
5. 运行 `sdk/verify_backends.py --backends <名>`。协议、服务、控制、映射层都不需要修改；业务代码里不应出现按后端名判断的分支。

### 仍未完成或未验证

- 真机：没有连接 RK3588，没有在线参数回读；坐标映射、零位、限位、TCP 都只到“离线文件 + URDF 一致”的程度。
- 未仿真：`MoveL`、笛卡尔点动与遥操作（按“主机侧求解、只发关节命令”的决定不做）、`FkLink` 与 MDH 查询/写入、力控/阻抗/拖动、夹爪力度、标定、RS485/TIO、移动底盘导航、订阅推送、深度图与真实相机内参/畸变、头/腰运动。
- 运动学接口：用共享运动学回答，不复现控制器自己的逆解和分支选择；flan2tool 初值和 `GetCurCartesianPose` 的坐标系是推断；真机控制器的 `FKb` 若按手册常量计算，会与仿真差 0.273 m（见上），需要在真机上核对。
- 轨迹规划、制动和夹爪速度是本仿真的策略（见上），不是原厂控制器算法；新增引擎时动力学结果允许不同，验收只要求接口语义一致。
- Isaac Sim：2026-10-24 按用户决定暂缓验证。适配器是为 GPU 工作站上的 Isaac Sim 5.1.0 安装写的，解释器由环境变量 `ISAAC_PYTHON` 指定（`config/backends/isaacsim.yaml`、`sdk/run_isaacsim_4090.sh`）；无头启动需要接受 NVIDIA Omniverse EULA（`OMNI_KIT_ACCEPT_EULA=YES`），须由用户决定。继续时把本工程同步到工作站，先做适配器文档列出的上机检查，再把工作站上服务端的 18400、18500 两个端口转发到本机（用你本机 SSH 配置里的主机别名），在本机运行同一份 `sdk/sdk_acceptance.py`，结果放进 `outputs/sdk_backends_v10/isaacsim/` 后用 `sdk/verify_backends.py --backends isaacsim --rescore` 打分。工作站上的 Python 没有 Pinocchio，运动学接口会明确报 UNSUPPORTED。
- Isaac Lab：可复用同一份派生 URDF、驱动参数、碰撞分组和相机约定，但它的向量化 GPU 主循环需要另写薄适配层，而且 GPU 管线对凸包有 64 顶点上限（74 个碰撞凸包超出），会破坏与 MuJoCo 的碰撞一致性。

## VLA 训练数据：原厂 SDK 录制与专家数据转换

录制程序只调用未修改的原厂 V1R1C9 SDK（`sdk.stock_sdk.connect`），同一份代码录仿真和真机，只换连接配置。专家运行由转换程序处理。两者写同一种原始 episode，再导出成一个 LeRobot v3.0 数据集。真机还没有连接过，见本节末尾。

| 文件 | 解释器 | 作用 |
|---|---|---|
| `data/vla_episode.py` | 两个环境都能导入 | 原始 episode 格式与数据约定：`episode.json`、`frames.npz`（每个 25 Hz 控制拍一行）、`commands.json`（驱动程序发出的每条 SDK 调用）、`cameras/<相机>.mp4`；LeRobot 特征表 |
| `data/record_sdk_episode.py` | inovance-fineart | 经原厂 SDK 录一条 episode |
| `data/convert_expert_run.py` | inovance-fineart | 把 `sim/run_expert.py` 或 `sim/run_kitting.py` 带 `--record` 的运行转成同一格式，可一次转多条 |
| `data/export_lerobot.py` | lerobot（LeRobot 0.5.2，格式 v3.0） | 把若干原始 episode 写成一个 LeRobot 数据集，写完读回核对 |
| `config/backends/mujoco_sim_clock.yaml` | — | 0.1 倍实时的 MuJoCo 服务端，供按仿真时间录制 |
| `tests/test_vla_recording.py` | inovance-fineart | 不依赖引擎：经原厂客户端对进程内 simbridge 录制；命令→动作；格式往返；转换与 SDK 映射一致；额外列不进策略输入 |

分两步，是为了让录制不依赖 LeRobot（`lerobot` 环境装的是 `~/workspace/lerobot` 的开发版）。原始 episode 保留完整的命令记录和时间戳，可以随时重新导出。

### 数据约定

- 状态 16 维：`GetJointState(15)` 的 14 个手臂关节，SDK（控制器 MDH）约定的 rad，左 j1–j7、右 j1–j7（`q_sdk = sign × q_urdf + offset`，见 `config/robot/franzi_sdk.yaml`）；再加左、右夹爪的 `GripperReadPosition`（0 全闭，1000 全开）。服务端不接受 side 15 时，自动改为左右各读一次，并记在 `episode.json` 的 `reads`。
- 动作 16 维，同单位：
  - 每臂取最近一次被接受的关节目标：`PushJointTeleopQueue(Dual)` 的点，或 `MoveJ`/`Home` 的最后一个路点。每侧夹爪取最近一次 `GripperSetPosition/Open/Close`。
  - 保持到下一条命令；第一条命令之前等于第一拍的测量值。被拒绝的命令只记录，不改动作。
  - 先观测第 k 拍，再决定动作 k，与专家 trace 相同。
- 用 SDK 约定而不是 URDF 约定：仿真数据训练的策略可以原样经 SDK 驱动真机。同一组数在两种约定里不是同一个姿态（例如肘关节符号相反）。
- 相机用 `get_image` 的名字：`head`、`waist`、`wrist_left`、`wrist_right`，即仿真的 `head_d435`、`body_d435`、`left_wrist_d405`、`right_wrist_d405`；640×480，RGBA 去掉 alpha 后存 RGB。
- 语言：`task` 是整段指令，`subtask` 是每拍的阶段指令（专家 `trace.json` 的 subtask 文本）。
  - 分拣装盘任务（`sim/run_kitting.py`）的每个种子自带一句指令，取自其 `summary.json` 的 `instruction`。
  - 原双臂抓放任务默认用 `Pick up the blue and orange parts with both arms at the same time and place them on the target pads.`。
- 其他列不作为策略输入。LeRobot 会把所有 `observation.*` 当状态、`action*` 当动作送进策略，所以它们放在 `command.*` 和 `timing.*` 下（测试会检查）：
  - `command.setpoint`：`GetDesireJointPos(15)`，读状态时伺服正在跟踪的目标；
  - `command.arm_source`：每臂动作的来源，0 尚无命令、1 逐拍关节目标、2 MoveJ 终点；
  - `timing.robot_s`：读状态时的机器人时钟（仿真时间，或主机单调时钟）；`timing.state_utc_ms`：读状态时的主机 UTC 毫秒；
  - `timing.camera_seq`、`timing.camera_utc_ms`：每路图像的 seq 和 SDK 时间戳（UTC 毫秒）。
- LeRobot 的 `timestamp` 是 frame_index / 25，实际时间看 `timing.robot_s`。物体真值不进入数据。
- 专家转换：
  - 关节用 `robot_kinematics.Kinematics().to_sdk`；夹爪用 simbridge 的 `GripperMap.position_from_closure`，与仿真 SDK 报出的值逐位相同（测试会检查）。
  - `command.setpoint` 取上一拍的专家动作。专家没有墙钟，UTC 列填 −1；图像与状态在同一物理步渲染，seq 记为 k+1。视频原样拷贝。
  - 指令优先级：`--task` 参数，其次是运行自带的 `instruction`，最后是上面的默认句。
  - 一次给多条运行时，`--output` 是一个新目录，每条运行转成其中一个同名 episode；失败的运行跳过（`--allow-failed` 才转）。
  - 只能转换四相机机位之后、带 `--record` 的运行（无头运行没有相机视频）。

### 录制时钟：真机与仿真

- 真机：按主机单调时钟走 25 Hz。每路相机一个后台线程取最新帧（每路最多 25 Hz），控制拍不等图像；seq 和时间戳记下重复帧与图像年龄。
- 仿真：MuJoCo 每路 640×480 渲染约 19 ms，在物理线程上执行，四路每拍约 80 ms，所以 25 Hz 四路相机跑不到实时。实测连续并发取图时，仿真降到 0.28 倍实时、12 Hz。
  - 如果 `ReadJointState` 返回 `sim_time`（simbridge 才有，控制器没有），录制改按仿真时间走 25 Hz。
  - 每拍先读状态，再并发请求四路新图；图像与状态在同一物理步或下一步渲染。
  - 服务端要用 `config/backends/mujoco_sim_clock.yaml`（0.1 倍实时），每拍约 0.4 s 墙钟。
- 遥操作节拍：simbridge 按推送之间的墙钟间隔回放（间隔小于 `teleop_max_gap_s` 0.2 s 时取中位数），只在约 1 倍实时下对。
  - 0.1 倍实时下间隔约 0.4 s，超过 0.2 s，于是每点按默认的 0.04 s 仿真时间回放，正好一拍一点。
  - 拍间隔低于 0.25 s 时，录制会给出警告。
- 命令延迟：经 SDK 遥操作时，伺服目标在推送后约两拍到达该点（simbridge 落后一个点并插值）；专家的动作当拍生效。两种数据混合训练时要注意这一差别。真机的延迟未测。

### 用法

仿真服务端（0.1 倍实时，端口来自 `config/connections/sim_mujoco.yaml`）：

```bash
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m simbridge.server --backend config/backends/mujoco_sim_clock.yaml --output outputs/<新目录>/server
```

经原厂 SDK 录一条。`--driver replay`（默认）回放 `config/sdk_acceptance/program.json` 的整段任务，`--until <阶段>` 只录到该阶段，只允许回环地址；`--driver passive --seconds N` 只观测、不发命令：

```bash
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m data.record_sdk_episode --connection config/connections/sim_mujoco.yaml --output outputs/<新目录>/raw/sdk_sim_replay
```

转换专家运行，再导出（导出须用 lerobot 环境）：

```bash
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m data.convert_expert_run outputs/expert_pinocchio --output outputs/<新目录>/raw/expert_pinocchio
```

批量转换（例如一批 `sim/run_kitting.py` 种子）：

```bash
/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m data.convert_expert_run outputs/<运行目录>/runs/seed_* --output outputs/<运行目录>/raw
```

```bash
/Users/puheliang/miniforge3/envs/lerobot/bin/python -m data.export_lerobot --episodes outputs/<新目录>/raw/expert_pinocchio outputs/<新目录>/raw/sdk_sim_replay --output outputs/<新目录>/lerobot
```

遥操作或策略驱动：写一个有 `start/step/stop` 的类，交给 `Recorder(arm, profile, output, task).run(driver)`；命令一律经 `rec.arm` 发出，才会记进动作。

### 样例：`outputs/vla_sample_v1/`

- `server/`：simbridge 服务端输出（0.1 倍实时，带 `--record`，`session.mp4` 是总览）。
- `raw/sdk_sim_replay/`：经原厂 SDK 回放整段任务。
  - 688 拍，仿真时间 27.5 s，墙钟 280 s（0.098 倍实时）。拍间隔全部正好 0.04 s 仿真时间，没有迟到的拍，没有重复图像。
  - 688 拍的状态与服务端自己的 trace 在同一仿真时刻逐位相同，夹爪也相同。每张图都在读状态的那个物理步渲染：服务端开着 `--record`；不开时晚一步，即 2 ms。
  - 动作逐位等于程序点；伺服目标比动作晚两拍，最多差 0.016 rad。
  - 只用于核对、不进数据的服务端真值：任务成功，放置误差 0.38 / 0.17 mm，最大穿透 0.099 mm，无故障。核对脚本和结果是 `check_sdk_episode.py`、`check_sdk_episode.json`。
- `raw/expert_pinocchio/`：由 `outputs/expert_pinocchio` 转换，613 拍。
- `lerobot/`：两条 episode 共 1301 帧，格式 v3.0，AV1 视频，17 MB。导出后读回核对了每条 episode 的首末帧；`meta/inovance_sources.json` 记来源与约定。
- `sources_sheet.png`：从数据集解码，两种来源在同一阶段的四路相机对比。
- 日志：`outputs/vla_sample_v1.{server,record,convert,export}.log`。

### 上真机前要确认（没有连接过真机）

- `GetJointState(15)` 和 `GetDesireJointPos(15)` 控制器是否接受。手册对 `GetJointState` 只写了 0/10/20/30；不接受时录制自动改为左右分读。
- 带宽：四路 640×480 RGBA 每帧共 4.9 MB，25 Hz 合计约 123 MB/s，接近千兆网上限，未必能每路 25 Hz。达不到时 `timing.camera_seq` 会重复，`episode.json` 的 `timing` 和 `warnings` 会报出。ai_com 若不接受四个并发连接，用 `--one-camera-connection`。
- 控制器与主机的时钟是否同步：图像时间戳是控制器的 UTC，不同步时图像年龄的估计不准。
- `replay` 驱动只用于仿真；真机第一次可以用 `passive` 只观测。遥操作或策略驱动还没有写。
- 相机内参仍是未标定的模拟针孔。

## 控制器镜像离线检查

原包 `/Users/puheliang/Downloads/feimi1010.zip` 的 `3588控制器/image_wheel.tar.gz` 含 25,062 个文件，解压总大小 2,901,908,776 字节，属于部署更新目录。里面有 ARM64 Linux 主程序、运控/轨迹规划/遥操作/碰撞检测动态库、ROS/导航依赖和配置。`update.img` 是另一个 RKFW 格式刷机容器，本次只检查容器头，未解出其分区。

在运控发布目录、`rk3588_controller/`、`app/` 的普通文件清单中，未发现核心 C/C++/Python 源文件；其他目录有脚本和 ROS 相关源码。这不等于整个包开源，也不代表已审计所有嵌套归档。`libsdk_server_pkg.so` 已去符号，`ai_com` 仍带符号/调试节，可继续静态逆向，但不能据此宣称恢复了原始 C++ 工程。

`sdk_review/controller_image_probe_v2/catalogue.json` 保存完整文件清单；`summary.json` 和 `additional_config_manifest.json` 保存部分提取文件及 SHA256。参数文件包含关节限制、MDH、惯量和拓扑预设。存在多套备份/机型配置，离线包的默认配置与真实控制器当前生效配置仍需分别核对，不能自动覆盖现有 URDF、零位或保守限制。本次没有执行镜像程序、部署脚本或刷机操作。

进一步检查的结果在 `sdk_review/controller_image_probe_v2/findings.json`：包内默认拓扑为 `whole_body_hard_with_tio`，20 电机、2 个 TIO；配置指定的参数根目录 `/inocfg/inbc_for_humanoid` 也有存储文件，已离线提取 `sysctrl_params.json`、`machine_params.json`、头/腰参数等。`sysctrl_params.json` 的关节限制明确标注 deg、deg/s、deg/s²、deg/s³；与 `app/config` 中的另一套限制数值不同，应优先沿启动配置追踪参数来源，仍不能当作在线硬件回读。报告另附 14 个手臂关节的包内范围与源 MuJoCo 范围对照。

离线参数与 URDF 的运动学对照（MDH 表、正负号与零位、腰部范围冲突、未使用的笛卡尔样本）见上节「机器人配置与坐标映射」和 `config/robot/kinematic_crosscheck.json`。`findings.json` 中“映射未核验”的结论因此收窄为：运动学结构已与离线 MDH 对上，零位标定和在线生效参数仍未核验。
