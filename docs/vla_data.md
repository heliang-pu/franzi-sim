# VLA 训练数据：原厂 SDK 录制、专家数据转换与 LeRobot 导出

录制程序只调用未修改的原厂 V1R1C9 SDK（`sdk.stock_sdk.connect`），同一份代码录仿真和真机，只换连接配置。专家的运行由转换程序处理。两者写同一种原始 episode，再导出成一个 LeRobot 数据集：默认格式 v2.1（openpi 一系读这一版），也可以直接导出 v3.0，两版之间可以无损互转。还没有连接过真机，见文末。

| 文件 | 解释器 | 作用 |
|---|---|---|
| `data/vla_episode.py` | 两个环境都能导入 | 原始 episode 格式与数据约定：`episode.json`、`frames.npz`（每个 25 Hz 控制拍一行）、`commands.json`（驱动程序发出的每条 SDK 调用）、`cameras/<相机>.mp4`；LeRobot 特征表 |
| `data/record_sdk_episode.py` | inovance-fineart | 经原厂 SDK 录一条 episode |
| `data/convert_expert_run.py` | inovance-fineart | 把 `scenes.pick_place.run`、`scenes.kitting.run`、RoboSyn 场景带 `--record` 的运行，或者在 Isaac Sim 上闭环录的运行（[isaac.md](isaac.md)），转成同一格式，可一次转多条。原始数据的 `source_run.simulator` 记下是哪个仿真器 |
| `data/export_lerobot.py` | lerobot；v2.1 也能在 pro6000 的项目环境里跑（装了 pyarrow 和 datasets） | 把若干原始 episode 写成一个 LeRobot 数据集：默认 v2.1；`--format v3.0` 由 lerobot 环境的 LeRobot 写。写完读回核对 |
| `data/lerobot_v21.py` | lerobot（只用 pyarrow、PyAV、NumPy） | 不依赖 LeRobot 的 v2.1 读写：目录与 `info.json`、jsonl 元数据、datasets 3.x 写法的 parquet 特征、统计量 |
| `data/convert_lerobot.py` | lerobot | v2.1 与 v3.0 互转，写到新目录，转完与输入逐项核对 |
| `data/frame_lag.py` | inovance-fineart | 检查录像的每帧是否在自己的控制步上：`frame_offset` 把录像和同一种子的稳定画面重跑（`isaac.episode --settle`）逐帧对比，判得出；`frame_lag` 只看录像本身，要有急动作才判得出。需要完整记录 |
| `data/subtask_check.py` | inovance-fineart | 检查一条运行能不能按 subtask 切开：每段相同的 subtask 文本里只有一只手的指令在变；两只手同时动的段，文本里要写明 "both hands"（两只手作用在同一个物体上） |
| `data/check_lerobot_v21.py` | openpi 的环境，或下文的参照环境 | 用读 v2.1 的 LeRobot（`0cf8648`）按 openpi 的方式加载数据集，与文件逐值核对 |
| `config/backends/mujoco_sim_clock.yaml` | — | 0.1 倍实时的 MuJoCo 服务端，供按仿真时间录制 |
| `tests/test_vla_recording.py` | inovance-fineart | 不依赖引擎：经原厂客户端对进程内 simbridge 录制；命令→动作；格式往返；转换与 SDK 映射一致；额外列不进策略输入 |
| `tests/test_lerobot_formats.py` | lerobot（在 inovance-fineart 下跳过） | 两条合成 episode：v2.1 导出的结构与数值；v2.1 → v3.0 → v2.1 逐字节还原 |

分成录制和导出两步，是为了让录制不依赖 LeRobot（`lerobot` 环境装的是 LeRobot 的开发版）。原始 episode 保留完整的命令记录和时间戳，可以随时重新导出。

## LeRobot 的版本与读取端

- lerobot 环境（`/Users/puheliang/miniforge3/envs/lerobot`）以可编辑方式装着 `/Users/puheliang/workspace/lerobot`。2026-10-24 它是 v0.6.1 之后的 main：`pyproject.toml` 写 0.6.2，`pip show` 仍显示安装时的 0.5.2。它只读写 v3.0；自带的转换脚本只有 v2.1 → v3.0，而且原地改写输入目录、默认推到 Hub。
- 工作区里各训练项目锁定的版本：

| 项目 | LeRobot | 格式 | datasets |
|---|---|---|---|
| openpi、kai0、pistar、Spatial-Forcing | `0cf8648`（2025-05-28） | v2.1 | 3.6.0 |
| openpi_with_subtask | `33cad37`（2025-09-15） | v3.0 | 3.6.0 |
| tau-0-vla | 0.4.1 | v3.0 | — |
| lerobot 环境 | v0.6.2 开发版 | v3.0 | 4.8.5 |

- datasets 4 把定长序列特征记成 `List`，datasets 3.6.0 不认识，读到就报 `Feature type 'List' not found`（实测）。lerobot 环境写的 parquet 都带这种元数据：
  - `0cf8648` 用 `load_dataset("parquet")` 读数据，不给定特征，碰到 `List` 就失败。所以 v2.1 由 `data/lerobot_v21.py` 自己写，特征用 datasets 3.x 的 `Sequence` 写法。
  - `33cad37` 读数据 parquet 时给定特征，不解析这段元数据，所以能读 LeRobot 原生写的 v3.0。但它读 `meta/episodes` 时不给定特征。LeRobot 的 v2.1 → v3.0 转换器经 datasets 4 写这个文件，带着 `List`，`33cad37` 就读不了。`data.convert_lerobot` 转完去掉这段元数据，和 LeRobot 原生写法一样。
- openpi_with_subtask 训练时直接读每拍的 `subtask` 文本（`TokenizeSubtaskTraining` 用 `data["subtask"]`），两种格式都带这一列。

## 数据约定

- 状态 16 维：`GetJointState(15)` 的 14 个手臂关节，SDK（控制器 MDH）约定的 rad，左 j1–j7、右 j1–j7（`q_sdk = sign × q_urdf + offset`，见 `config/robot/franzi_sdk.yaml`）；再加左、右夹爪的 `GripperReadPosition`（0 全闭，1000 全开）。服务端不接受 side 15 时，自动改为左右各读一次，并记在 `episode.json` 的 `reads`。
- 动作 16 维，单位相同：
  - 每臂取最近一次被接受的关节目标：`PushJointTeleopQueue(Dual)` 的点，或 `MoveJ`/`Home` 的最后一个路点。每侧夹爪取最近一次 `GripperSetPosition/Open/Close`。
  - 保持到下一条命令；第一条命令之前等于第一拍的测量值。被拒绝的命令只记录，不改动作。
  - 先观测第 k 拍，再决定动作 k，与专家 trace 相同。
- 用 SDK 约定而不是 URDF 约定：仿真数据训练出的策略可以原样经 SDK 驱动真机。同一组数在两种约定里不是同一个姿态（例如肘关节符号相反）。
- 相机用 `get_image` 的名字：`head`、`waist`、`wrist_left`、`wrist_right`，即仿真的 `head_d435`、`body_d435`、`left_wrist_d405`、`right_wrist_d405`；640×480，RGBA 去掉 alpha 后存 RGB。
- 语言：`task` 是整段指令，`subtask` 是每拍的阶段指令（专家 `trace.json` 的 subtask 文本）。
  - 分拣装盘的每个种子自带一句指令，取自其 `summary.json` 的 `instruction`。
  - 双臂抓放默认用 `Pick up the blue and orange parts with both arms at the same time and place them on the target pads.`。
- 其他列不作为策略输入。LeRobot 会把所有 `observation.*` 当状态、`action*` 当动作送进策略，所以它们放在 `command.*` 和 `timing.*` 下（测试会检查）：
  - `command.setpoint`：`GetDesireJointPos(15)`，读状态时伺服正在跟踪的目标；
  - `command.arm_source`：每臂动作的来源，0 尚无命令、1 逐拍关节目标、2 MoveJ 终点；
  - `timing.robot_s`：读状态时的机器人时钟（仿真时间，或主机单调时钟）；`timing.state_utc_ms`：读状态时的主机 UTC 毫秒；
  - `timing.camera_seq`、`timing.camera_utc_ms`：每路图像的 seq 和 SDK 时间戳（UTC 毫秒）。
- LeRobot 的 `timestamp` 是 frame_index / 25，实际时间看 `timing.robot_s`。物体真值不进入策略输入。
- 全量数据（原始格式 2，2026-10-25，用户要求"尽可能全量采"）：
  - 来源有全身数据时（专家运行；SDK 录制目前没有），每拍再记 22 维 `FULL_NAMES`：上面的 16 维，接着腰的小腿、大腿、腰俯仰、腰转，以及头的转动和俯仰。单位都是 SDK 约定，腰和头按 `franzi_sdk.yaml` 里各自的映射换算，头部的映射标着未标定。
  - 这 22 维分四组：`state_full`、`action_full`，以及带 SDK 符号的 `velocity_full`、`effort_full`（夹爪是位置单位每秒和夹紧力 N）。另有 `eef_pose`（两只手抓取点）和 `camera_pose`（四路相机），都是世界系下的 x、y、z、qw、qx、qy、qz。
  - 训练视图（`--view`）：`arms16`（默认，即上面的 16 维）、`waist17`（再加腰转）、`full22`。`observation.state` 和 `action` 取所选视图。
  - 全量数组导出到 `robot.*` 下（`--full`，默认在每条 episode 都有时加上），策略默认不读。
  - 原始 episode 的 `extras/`：每路相机的深度 `depth_<相机>.mkv`（毫米）和分割 `segmentation_<相机>.mkv`（编号见 `segmentation.json`），FFV1 16 位灰度，每拍一帧；评测用的世界状态 `eval.npz`（自由物体位姿和速度、道具关节）和 `eval_contacts.json`。
  - LeRobot 的视频特征按 8 位 RGB 解码，所以深度不做成特征。导出时把每条 episode 的 `extras/` 拷到数据集的 `extras/episode_<序号>/`，并在 `meta/inovance_extras.json` 里列出。
  - 第 1 版的 episode 照样能读能导出，只是没有全量数据。
- 专家运行的转换：
  - 关节用 `robot_kinematics.Kinematics().to_sdk`；夹爪用 simbridge 的 `GripperMap.position_from_closure`，与仿真 SDK 报出的值逐位相同（测试会检查）。
  - `command.setpoint` 取上一拍的专家动作。专家没有墙钟，UTC 列填 −1；图像与状态在同一物理步渲染，seq 记为 k+1。视频原样拷贝。
  - 指令的优先级：`--task` 参数，其次是运行自带的 `instruction`，最后是上面的默认句。
  - 一次给多条运行时，`--output` 是一个新目录，每条运行转成其中一个同名 episode；失败的运行跳过（加 `--allow-failed` 才转）。
  - 只能转换四相机机位之后、带 `--record` 的运行（无界面运行没有相机视频）。
  - 带全量记录的运行（`sim/episode.py`，2026-10-25 起）转成第 2 版：腰和头经 `simbridge.robot.load_robot()` 的单元映射换算，速度和力矩乘各关节的符号，相机按 `head`、`waist`、`wrist_left`、`wrist_right` 重新排序；录像里的深度、分割视频改用 SDK 的相机名放进 `extras/`，世界状态写成 `extras/eval.npz`。

## 录制时钟：真机与仿真

- 真机：按主机单调时钟走 25 Hz。每路相机一个后台线程取最新帧（每路最多 25 Hz），控制拍不等图像；seq 和时间戳记下重复帧与图像年龄。
- 仿真：MuJoCo 每路 640×480 渲染约 19 ms，在物理线程上执行，四路每拍约 80 ms，所以 25 Hz 四路相机跑不到实时。实测连续并发取图时，仿真降到 0.28 倍实时、12 Hz。
  - 如果 `ReadJointState` 返回 `sim_time`（simbridge 才有，控制器没有），录制改按仿真时间走 25 Hz。
  - 每拍先读状态，再并发请求四路新图；图像与状态在同一物理步或下一步渲染。
  - 服务端要用 `config/backends/mujoco_sim_clock.yaml`（0.1 倍实时），每拍约 0.4 s 墙钟。
- 遥操作节拍：simbridge 按推送之间的墙钟间隔回放（间隔小于 `teleop_max_gap_s` 0.2 s 时取中位数），只在约 1 倍实时下成立。
  - 0.1 倍实时下间隔约 0.4 s，超过 0.2 s，于是每点按默认的 0.04 s 仿真时间回放，正好一拍一点。
  - 拍间隔低于 0.25 s 时，录制会给出警告。
- 命令延迟：经 SDK 遥操作时，伺服目标在推送后约两拍到达该点（simbridge 落后一个点并插值）；专家的动作当拍生效。两种数据混合训练时要注意这一差别。真机的延迟未测。

## 用法

仿真服务端（0.1 倍实时，端口来自 `config/connections/sim_mujoco.yaml`）：

```bash
python -m simbridge.server --backend config/backends/mujoco_sim_clock.yaml --output outputs/<新目录>/server
```

经原厂 SDK 录一条。`--driver replay`（默认）回放 `config/sdk_acceptance/program.json` 的整段任务，`--until <阶段>` 只录到该阶段，只允许回环地址；`--driver passive --seconds N` 只观测、不发命令：

```bash
python -m data.record_sdk_episode --connection config/connections/sim_mujoco.yaml --output outputs/<新目录>/raw/sdk_sim_replay
```

转换专家运行（可以一次给多条，例如一批分拣装盘种子），再在 lerobot 环境里导出。默认写 v2.1；`--format v3.0` 改由 LeRobot 写 v3.0；`--vcodec libx264` 给不支持 AV1 的解码器用：

```bash
python -m data.convert_expert_run outputs/<运行目录> --output outputs/<新目录>/raw/<运行名>
python -m data.convert_expert_run outputs/<运行目录>/runs/seed_* --output outputs/<运行目录>/raw
# lerobot 环境
python -m data.export_lerobot --episodes outputs/<新目录>/raw/<运行名> outputs/<新目录>/raw/sdk_sim_replay --output outputs/<新目录>/lerobot
python -m data.export_lerobot --format v3.0 --episodes outputs/<新目录>/raw/* --output outputs/<新目录>/lerobot_v30
# 状态和动作取 17 维（加腰转）或 22 维；--no-full 不带 robot.*
python -m data.export_lerobot --view waist17 --episodes outputs/<新目录>/raw/* --output outputs/<新目录>/lerobot_waist17
```

在 Isaac Sim 上采集可以一步完成：`isaac.collect` 让每个种子在单独的 Isaac 进程里录一个回合（默认 8 个并行），抽第一个成功的种子检查画面和状态对齐，再把这些回合转成原始数据，导出一个 v2.1 数据集。在 pro6000 的项目环境里运行，见 [isaac.md](isaac.md) 的「录数据」：

```bash
ISAAC_PYTHON=<Isaac Sim 的 python> python -m isaac.collect tote_transfer --seeds 0-19 --output outputs/<新目录>
```

两版互转（lerobot 环境；方向由输入的版本决定，输出是新目录）。Hub 上的数据集先下载到本地（`hf download <repo_id> --repo-type dataset --local-dir <目录>`）再转：

```bash
python -m data.convert_lerobot outputs/<v2.1 或 v3.0 数据集> --output outputs/<新目录>
```

验收 v2.1：在 openpi 的环境（或下文的参照环境）里，从本项目根目录运行。测试只在 lerobot 环境里跑：

```bash
<openpi 的 venv>/bin/python data/check_lerobot_v21.py outputs/<v2.1 数据集>
/Users/puheliang/miniforge3/envs/lerobot/bin/python -m pytest tests/test_lerobot_formats.py -q
```

遥操作或策略驱动：写一个有 `start/step/stop` 的类，交给 `Recorder(arm, profile, output, task).run(driver)`；命令一律经 `rec.arm` 发出，才会记进动作。

## v2.1 导出

- 结构与 LeRobot `0cf8648` 写的相同：每条 episode 一个 parquet，每路相机每条 episode 一个 MP4，`meta/` 下有 `info.json`、`tasks.jsonl`、`episodes.jsonl` 和 `episodes_stats.jsonl`。另外还有两份：聚合的 `stats.json`（LeRobot v2.1 自己聚合，不读它），以及记录来源与约定的 `inovance_sources.json`。
- 统计量按 LeRobot v2.1 的算法，给每个非文本特征算 min、max、mean、std 和 count。图像只取约 n^0.75 帧，每帧再每隔 k = 长边 // 150 个像素取一个（640×480 取 160×120），按通道归一到 [0, 1]。
- 另加当前 LeRobot 保留的 q01、q10、q50、q90、q99 分位数，pi0.5 的分位数归一化要用，这里按全部样本精确计算。v2.1 读取端忽略它们，转成 v3.0 后保留。
- 视频逐 episode、逐相机重新编码，参数取 LeRobot 的默认值：AV1（libsvtav1），每 2 帧一个关键帧，CRF 30，preset 12（PyAV 15 自带的 SVT-AV1 3.0 按 10 跑）。
  - 2 帧的 GOP 让训练时随机取帧只需解码一两帧。原始 episode 的 H.264 关键帧间隔 250，所以不直接拷贝。
  - 不支持 AV1 的解码器（decord 等）用 `--vcodec libx264`，同样 2 帧 GOP、CRF 30。工业场景的头部相机大半是洞洞板的细密纹理，x264 CRF 30 重编码后与原视频平均差 4.7 级，超过导出自检的 4 级（2026-10-25，料箱转运种子 0；AV1 在同一画面上能过），所以这类场景用默认的 AV1。
  - SVT-AV1 每开一个编码器就打印一段横幅，所以导出时默认设 `SVT_LOG=1`，只留错误。
- 画质：与原始视频相比，PSNR 不低于 40.8 dB（中位 42.2 dB），平均差不超过 1.84 级。纯色画面经 H.264 再经 AV1 会整体偏暗 2–4 级，来自 RGB↔YUV 的舍入，LeRobot 自己的编码也一样。
- 速度与体积：5 条 episode（7255 帧 × 4 路）导出 34 秒，视频 143.4 MB。LeRobot 原生导出 v3.0 要 55 秒，视频体积几乎相同。
- 第 2 版 episode 的验证（2026-10-25，料箱转运种子 0 录像，830 帧）：v2.1 用 `waist17`、v3.0 用 `full22` 导出，自检都通过，`robot.*` 六组特征都在，11 个附属文件拷进 `extras/episode_000000/`；v2.1 再转 v3.0 时 `extras/` 一起带过去。

## v2.1 与 v3.0 互转

`data.convert_lerobot` 看输入 `info.json` 的 `codebase_version` 决定方向，写到新目录，不改输入，也不连 Hub。

- v2.1 → v3.0：分步调用 LeRobot 自己的转换器（`lerobot.scripts.convert_dataset_v21_to_v30` 里的 `convert_info` 等函数）。它的 `convert_dataset()` 会把输入目录改名并推到 Hub，所以不调用它。
  - 视频按流拷贝拼接，不重编码。H.264 和 HEVC 例外一点：FFmpeg 的 concat 解复用器（`auto_convert`）在每个关键帧里重复一份参数集（SPS/PPS），关键帧包大几十字节，解出的帧完全相同。AV1 的包原样不变。
  - 转完去掉 `meta/episodes` 里 datasets 4 的元数据（见上节）。
- v3.0 → v2.1（LeRobot 没有这个方向）：
  - 每条 episode 的行写成一个 parquet。
  - 视频按每条 episode 的 `from_timestamp` 流拷贝切出，时间戳从 0 开始，不重编码。这要求每条 episode 从关键帧开始，LeRobot 写的都是。
  - 任务、episode 和逐条统计（含分位数）写回 jsonl，`stats.json` 原样带过去。
- 其他 `meta/` 文件（如 `inovance_sources.json`）照抄。
- 转完的核对：
  - 所有非视频值和统计量逐位相同；
  - 每条 episode 每路视频的包逐字节相同（H.264/HEVC 去掉参数集后再比），显示时间也相同；
  - 首末帧两边都用 PyAV 解，逐像素相同；
  - v3.0 输出再经 LeRobot 自己加载。它用 torchcodec 和系统 FFmpeg 解码，画面比 PyAV 整体亮约 1 级，最大差 3 级，以 3 级为容差。
- 速度：5 条 episode，v2.1 → v3.0 约 5 秒，v3.0 → v2.1 约 1 秒。

## 格式验证 `outputs/lerobot_formats_v1/`（2026-10-24）

- 原始 episode 共 7255 帧：`expert_wrist_block`（双臂抓放，749 帧）、`kitting_demo_seed7`（1571）、`kitchen_demo_seed0`（1875）、`tray_demo_seed0`（1264）、`drawer_demo_seed0`（1796）。
- 导出 v2.1（`v21_all`），转 v3.0 再转回（`v21_all_round_trip`）：与原 v2.1 逐文件相同，包括元数据、统计量、parquet 的 schema 与数值，以及每个视频包。
- 把 v3.0 拆成多个文件再转回 v2.1，同样与原 v2.1 相同。拆法是数据上限 1 MB、视频 10 MB：3 个数据文件，每路最多 5 个视频文件，有 episode 跨文件。
- LeRobot 原生导出的 v3.0（`v30_direct_all`）转 v2.1：通过。`--vcodec libx264` 的 2 条 episode 往返：通过，解出的帧相同。
- 参照环境里的 LeRobot `0cf8648`（datasets 3.6.0）加载上面每份 v2.1：状态、50 步动作块与补齐标记、subtask、task 逐位相同，图像与 PyAV 解码 0 差异（pyav 后端）。
- 同一环境换成 LeRobot `33cad37`：
  - LeRobot 原生 v3.0 和修复后转换出的 v3.0 都能读；后者与 v2.1 源逐值相同，图像 0 差异。
  - 修复前转换出的 v3.0 报 `Feature type 'List' not found`。

### 参照环境

openpi 的 venv 里直接运行 `data/check_lerobot_v21.py` 即可。本机没有 openpi 环境时，按 openpi 锁定的版本搭一个，LeRobot 源码从本机 lerobot 仓库的历史里取。PyAV 14.4 在本机没有可用的轮子，用 openpi 自己锁的 17.0.0：

```bash
uv venv <目录>/v21ref --python 3.11
uv pip install --python <目录>/v21ref/bin/python --only-binary av datasets==3.6.0 torch==2.7.1 \
    torchvision==0.22.1 numpy==1.26.4 pyarrow==20.0.0 av==17.0.0 jsonlines==4.0.0 \
    'huggingface-hub[hf-transfer]>=0.27.1,<0.36' 'packaging>=24.2' draccus==0.10.0 einops termcolor pillow
mkdir <目录>/lerobot_0cf8648 && git -C /Users/puheliang/workspace/lerobot archive 0cf8648 lerobot | tar -x -C <目录>/lerobot_0cf8648
PYTHONPATH=<目录>/lerobot_0cf8648 HF_HUB_OFFLINE=1 <目录>/v21ref/bin/python data/check_lerobot_v21.py outputs/<v2.1 数据集>
```

要换成 `33cad37` 的读取端，就取 `git archive 33cad37 src`，再把 `PYTHONPATH` 指向解出的 `src`。

## 样例 `vla_sample_v1`（已删除）

这份样例在 2026-10-24 清理 `outputs/` 时删除了，结论保留在这里：

- 经原厂 SDK 回放整段双臂抓放：688 拍，仿真时间 27.5 s，墙钟 280 s（0.098 倍实时）。拍间隔全部正好是 0.04 s 仿真时间，没有迟到的拍，没有重复图像。
- 688 拍的状态与服务端自己的 trace 在同一仿真时刻逐位相同，夹爪也相同。每张图都在读状态的那个物理步渲染（服务端开着 `--record`；不开时晚一步，即 2 ms）。
- 动作逐位等于程序点；伺服目标比动作晚两拍，最多差 0.016 rad。
- 只用于核对、不进数据的服务端真值：任务成功，放置误差 0.38 / 0.17 mm，最大穿透 0.099 mm，无故障。
- 同一份数据集里还有一条转换来的专家运行（613 拍）：两条 episode 共 1301 帧，格式 v3.0，AV1 视频，17 MB。导出后读回核对了每条 episode 的首末帧，`meta/inovance_sources.json` 记录来源与约定。

## 上真机前要确认

- `GetJointState(15)` 和 `GetDesireJointPos(15)` 控制器是否接受。手册对 `GetJointState` 只写了 0/10/20/30；不接受时录制自动改为左右分读。
- 带宽：四路 640×480 RGBA 每帧共 4.9 MB，25 Hz 合计约 123 MB/s，接近千兆网上限，未必能每路 25 Hz。达不到时 `timing.camera_seq` 会重复，`episode.json` 的 `timing` 和 `warnings` 会报出。ai_com 若不接受四个并发连接，用 `--one-camera-connection`。
- 控制器与主机的时钟是否同步：图像时间戳是控制器的 UTC，不同步时图像年龄的估计不准。
- `replay` 驱动只用于仿真；真机第一次可以用 `passive` 只观测。遥操作或策略驱动还没有写。
- 相机内参仍是未标定的模拟针孔。
