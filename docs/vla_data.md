# VLA 训练数据：原厂 SDK 录制与专家数据转换

录制程序只调用未修改的原厂 V1R1C9 SDK（`sdk.stock_sdk.connect`），同一份代码录仿真和真机，只换连接配置。专家的运行由转换程序处理。两者写同一种原始 episode，再导出成一个 LeRobot v3.0 数据集。还没有连接过真机，见文末。

| 文件 | 解释器 | 作用 |
|---|---|---|
| `data/vla_episode.py` | 两个环境都能导入 | 原始 episode 格式与数据约定：`episode.json`、`frames.npz`（每个 25 Hz 控制拍一行）、`commands.json`（驱动程序发出的每条 SDK 调用）、`cameras/<相机>.mp4`；LeRobot 特征表 |
| `data/record_sdk_episode.py` | inovance-fineart | 经原厂 SDK 录一条 episode |
| `data/convert_expert_run.py` | inovance-fineart | 把 `scenes.pick_place.run` 或 `scenes.kitting.run` 带 `--record` 的运行转成同一格式，可一次转多条 |
| `data/export_lerobot.py` | lerobot（LeRobot 0.5.2，格式 v3.0） | 把若干原始 episode 写成一个 LeRobot 数据集，写完读回核对 |
| `config/backends/mujoco_sim_clock.yaml` | — | 0.1 倍实时的 MuJoCo 服务端，供按仿真时间录制 |
| `tests/test_vla_recording.py` | inovance-fineart | 不依赖引擎：经原厂客户端对进程内 simbridge 录制；命令→动作；格式往返；转换与 SDK 映射一致；额外列不进策略输入 |

分成录制和导出两步，是为了让录制不依赖 LeRobot（`lerobot` 环境装的是 LeRobot 的开发版）。原始 episode 保留完整的命令记录和时间戳，可以随时重新导出。

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
- LeRobot 的 `timestamp` 是 frame_index / 25，实际时间看 `timing.robot_s`。物体真值不进入数据。
- 专家运行的转换：
  - 关节用 `robot_kinematics.Kinematics().to_sdk`；夹爪用 simbridge 的 `GripperMap.position_from_closure`，与仿真 SDK 报出的值逐位相同（测试会检查）。
  - `command.setpoint` 取上一拍的专家动作。专家没有墙钟，UTC 列填 −1；图像与状态在同一物理步渲染，seq 记为 k+1。视频原样拷贝。
  - 指令的优先级：`--task` 参数，其次是运行自带的 `instruction`，最后是上面的默认句。
  - 一次给多条运行时，`--output` 是一个新目录，每条运行转成其中一个同名 episode；失败的运行跳过（加 `--allow-failed` 才转）。
  - 只能转换四相机机位之后、带 `--record` 的运行（无界面运行没有相机视频）。

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

转换专家运行（可以一次给多条，例如一批分拣装盘种子），再在 lerobot 环境里导出：

```bash
python -m data.convert_expert_run outputs/<运行目录> --output outputs/<新目录>/raw/<运行名>
python -m data.convert_expert_run outputs/<运行目录>/runs/seed_* --output outputs/<运行目录>/raw
# lerobot 环境
python -m data.export_lerobot --episodes outputs/<新目录>/raw/<运行名> outputs/<新目录>/raw/sdk_sim_replay --output outputs/<新目录>/lerobot
```

遥操作或策略驱动：写一个有 `start/step/stop` 的类，交给 `Recorder(arm, profile, output, task).run(driver)`；命令一律经 `rec.arm` 发出，才会记进动作。

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
