# 原厂 SDK 仿真服务端（simbridge）

上层程序（FineART、遥操作、数据采集、控制脚本）只调用**未修改的原厂 V1R1C9 SDK**（`ArmClient`、`ModuleCommunicationClient`）。连哪台「机器人」只由连接配置决定，上层代码不变：

| 连接配置 | 服务端 | 地址 |
|---|---|---|
| `config/connections/real_rk3588.yaml` | 真实 RK3588 原厂服务端（本仓库从未连接过） | 192.168.23.30:8000/8100 |
| `config/connections/sim_mujoco.yaml` | simbridge + MuJoCo | 127.0.0.1:18000/18100 |
| `config/connections/sim_isaacsim.yaml` | simbridge + Isaac Sim（草稿，未运行、未验证） | 127.0.0.1:18400/18500（经 SSH 隧道连到 GPU 工作站） |

```python
from sdk.stock_sdk import connect

arm, sensors = connect("config/connections/sim_mujoco.yaml")  # 真机：real_rk3588.yaml
arm.MoveJ([[...14 个关节角 rad...]], side=15)                     # 原厂 API，原样调用
```

原厂 SDK 在 `sdk_review/feimi1010/sdk/`（不入库），源文件一律不改；`sdk/verify_backends.py` 每次验收前后都把它与 `sdk.zip` 逐字节比较（3153 个文件一致）。

## 分层与目录

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

- 每层只导入它下面的层；只有适配器导入引擎（有测试检查）。
- MuJoCo 适配器包装双臂抓放的场景（`scenes.pick_place.run.PickPlaceScene`），沿用全部物理规则和逐步审计。
- PyBullet 后端已于 2026-10-24 按用户决定下线：只有 MuJoCo 场景按真机对齐了外观、相机、灯光、限位和躯干保持，IK/Fk 接口也因此能直接用共享运动学。

## 线程与时钟

- RPC 工作线程不访问仿真器：命令经队列交给物理线程，在两步之间执行；回复在下一次快照发布后才返回，所以 `MoveJ` 返回 0 后立即查询 `GetMotionStatus` 一定看到 1。状态查询读取每个物理步发布一次的不可变快照。相机渲染请求也排队到物理线程（OpenGL/Metal 上下文和 Isaac Sim 都要求在拥有场景的线程渲染）。
- 两种循环：`loop: framework`（框架调用 `step()`）；`loop: simulator`（引擎拥有主循环，每步回调框架的 `pre_step/post_step`；MuJoCo 适配器实现并验证了这一模式，Isaac Sim 草稿用物理回调）。
- 仿真时间 `sim_time`：物理时钟（500 Hz 步长），场景启动时为 0。设定点每个物理步更新一次，引擎在两步之间保持；状态轨迹按 25 Hz 采样。
- 图像 `timestamp`：渲染完成时的系统 UTC 毫秒，与仿真时间没有固定比例；`server/camera_log.json` 记录每帧的 seq、UTC 毫秒、仿真时间和物理步号。
- 服务端按实时节拍运行；录像时渲染开销会让仿真慢于实时（约 0.3 倍），客户端一律轮询状态，不按墙钟睡眠。

## 统一控制语义（公共层，与引擎无关）

- **接收 ≠ 完成**：运动类命令返回 0 只表示已校验并入队；到位以 `GetMotionStatus` 为准（0 到位静止，1 运动中或未收敛，-1 故障，-3 已停转但长时间未到位）。
- 错误码沿用原厂：参数非法、超限、非有限数、长度不符 `-3`；未就绪（未使能、软急停中）`-5`；软急停中上使能 `-8`；物理故障后 `-9`；遥操作会话与轨迹冲突 `-10`；腰、头运动等存在但不仿真的选项 `-4`。返回字典附 `err_msg` 说明原因（原厂客户端只返回状态码，不会丢失语义）。
- 原厂客户端里有、但本后端不提供的方法（`MoveL`、笛卡尔点动与遥操作、`FkLink`、MDH 查询与写入、力控/阻抗、夹爪力、标定、导航等）一律返回协议级错误 `UNSUPPORTED: arm_sdk.<方法> ...`。无论 `raise_on_error` 怎么设，原厂客户端都会抛异常，不会得到空列表式的「假成功」。
- `MoveJ`/`Home`：关节空间直线，同步梯形速度曲线（所有关节同时起停），速度、加速度 = 限值 × 速度比例。新 `MoveJ` 排在执行中的轨迹之后（仿真的策略）。
- 关节遥操作队列（单臂/双臂）按客户端的推送节拍回放：
  - 每点的时长取本次会话最近 50 次推送间隔（约 2 s）的中位数，凑够 8 个间隔之前用默认 40 ms；超过 0.2 s 的间隔算作停顿，不计入、也不清空历史。在仿真时间里线性插值、延迟一个点，所以仿真慢于实时也保持客户端的节拍，客户端自身的计时抖动也不会变成速度抖动。
  - 推送请求只用最新快照校验后立即回复，不等物理线程；物理线程再校验一次，期间状态变化导致的丢点记入事件。
  - 速度、加速度限位只作安全钳制：超限的流会被钳制而偏离路径，客户端应发送满足限位的轨迹。队列容量 500。
- 使能门控：Disable 冻结指令并保持仿真伺服，不模拟断电掉落。软急停：按加速度限值制动后保持，拒绝运动、上使能和夹爪命令，`SoftStop(False)` 后恢复。
- 物理故障（任一接触穿透 > 1 mm，或关节越限 > 0.002 rad）：冻结物理、锁存故障，`IsReady=false`，`GetMotionStatus=-1`，运动和使能返回 `-9`，`GetCurAlarm` 给出原因（仿真自定义告警码）。只有重启仿真会话才能清除。
- `ReadJointState` 额外返回 `torque_source`：`simulated_actuator_effort`（仿真执行器力，含重力前馈）、`measured_joint_torque_sensor`（只有真机可能有）或 `unavailable`（此时 `torque` 为空，不填假值）。
- 夹爪：原厂 `0 = 全闭、1000 = 全开`，线性映射到仿真手指行程；`GripperReadGripState` 由测量值和接触查询得到（0 运动中、1 到位、2 夹持、3 掉落）。`GripperInitialize` 在仿真中是空操作并如实说明；夹爪力度接口不支持。

## 机器人配置与坐标映射

`config/robot/franzi_sdk.yaml` 与引擎代码分开，保存：SDK 关节顺序和仿真关节名、正方向与零位偏移、限位（仿真模型范围 ∩ 镜像控制器范围）、速度和加速度（保守的规划 YAML）、夹爪映射、相机名称与安装关系（内参见 `sim/camera_rig.json`，未标定）、基座/法兰/TCP 定义、安全阈值。SDK 关节值统一为 `q_sdk = sign × q_urdf + offset`（rad）；同一组数在两种约定里不是同一个姿态。`--joint-convention urdf_passthrough` 可切回旧版「URDF 原值」行为，仅用于调试。

肘关节方向等映射问题的结论（`sdk/check_sdk_kinematics.py`，结果在 `config/robot/kinematic_crosscheck.json`）：

- 镜像持久化参数 `machine_params.json` 的 `bio_para` 与 `config11/mdh.json` 的 MDH 表完全相同。按 Craig 改进 DH（`θ = q + theta_offset`）解释，它与源 URDF 的手臂链**运动学完全一致**：500 组随机关节角上位置误差 ≤ 1.3e-14 m、姿态误差 ≤ 2.1e-14 rad。同一张表按标准 DH 解释，最好也有 0.072 m 误差（反证了约定）。
- 对 128 种正负号组合拟合偏置和基座/法兰变换，得到 14 个运动学上完全等价的解（腕部等对称性所致），只靠正解无法唯一确定；再用镜像关节范围筛选，只剩一个：**左右臂 sign = [-1, +1, -1, -1, -1, -1, +1]，偏置全为 0**。肘关节：URDF −150°～7.45° 映射为 SDK −7.45°～150°，与镜像的 −15°～148° 一致，即屈肘在 SDK 中为正。
- 第 1、7 关节的零位偏置会被基座/法兰坐标吸收，正解无法辨识；0 是唯一与范围一致的值，标为「范围推断」。腰部：正解精确，4 个等价解中按范围选 [1,1,1,1]，但腰俯仰范围冲突（URDF ±30°，镜像 −75°～30°）。头部：镜像里没有 MDH，未标定，按 URDF 原值透传。
- 镜像中 `mdh_repeat_exact*.json` 的关节/笛卡尔样本自相矛盾（起止关节角几乎相同，笛卡尔位置却差几十厘米），不能作为正解真值，未使用。
- 限位：SDK 层按「仿真模型范围 ∩ 镜像控制器范围」（控制器一侧留 0.001 rad 余量）拒收越界目标（`-3`）。`build_task.align_joint_limits` 把派生 MuJoCo 场景的关节范围裁到同一交集；MuJoCo 适配器加载时检查模型范围在 URDF 范围以内、且包含 SDK 接受的全部目标。

**以上全部来自离线镜像文件与 URDF 的对照，不是在线控制器回读，也没有用真机姿态验证。不能宣称仿真与真机运动已经一致**：零位标定、编码器方向、真机生效参数、法兰到 TCP（flan2tool）、真实伺服与轨迹算法都未验证。用户反馈此前实测过 URDF 与真机一致；这是用户的经验，本仓库没有对应的测量记录。

## 运动学接口

按用户决定（2026-10-24，「真机跟仿真对齐」），逆解和正解只用主机侧的共享运动学 `robot_kinematics.py`（Pinocchio），仿真和真机都只接收关节命令。simbridge 的运动学接口直接调用它，`simbridge/sdk_kinematics.py` 只做 SDK 约定的换算：

| 接口 | 仿真实现 |
|---|---|
| `Fk` / `FkTool` | `robot_kinematics.fk`，换算成手臂 MDH 基座 A0 → 法兰 A7；工具 = 法兰 × flan2tool |
| `IK`（输入基座→工具） | `robot_kinematics.ik`（专家用的同一个求解器），初值取当前测量关节角；无解返回 `-6`（原厂定义为「笛卡尔目标不可达」） |
| `FKb` / `FKbTool` / `IKb` / `IKbTool` | 同上，基座系 B = URDF `base_link`，即仿真机器人的真实位姿；`torso_joints` 为空时读当前腰部关节（同手册） |
| `GetFlan2tool` / `SetFlan2tool` | 启动时为单位阵（镜像里没有工具坐标系，推断）；设置只在本次会话内有效 |
| `GetCurCartesianPose(Dual)` | 当前测量关节角的 `FkTool`，在 A0 系（推断） |
| `FkLink`、`GetMdh`、`GetMdhCompensation`、`GetArmEulerAngle`、`SetMdh*` | 不提供（UNSUPPORTED）：这些只是控制器自身模型的接口，共享运动学用不到 |

约定（来自原厂手册、Python 客户端和 C++ 头文件）：

- SDK 关节值 `q_sdk = sign × q_urdf + offset`。
- A0 在 torso_Link 中、法兰在 wrist_roll_Link 中的位姿取自 `kinematic_crosscheck.json`。
- 4×4 矩阵按列优先的 16 个数传输，平移单位 m，按控制器的 float32 精度收发。
- 位姿 `[x, y, z, rz, ry, rx]`，`R = Rx(rx)·Ry(ry)·Rz(rz)`（外旋 zyx，等价于 SciPy `from_euler("zyx", [rz, ry, rx])`），读回时 `ry ∈ [−π/2, π/2]`。
- 逆解用 `Kinematics(margin=0.0001)`：模拟控制器的逆解，不留规划余量。

`MoveL`、笛卡尔点动和遥操作不实现：真机上它们会用控制器自己的逆解。没有 Pinocchio 的服务端（例如 GPU 工作站上的 Isaac Sim 环境）照常启动，运动学接口明确返回 UNSUPPORTED。

**真机注意。** 手册里控制器内 FKb 的常量（`WbcTmpKinematics`，镜像里的 SDK 服务端确有这条代码路径）与 URDF 在直立姿态相差 (0, ∓0.0725, −0.26355) m，即 0.273 m：

- 手册把腰 MDH 基座放在 B 原点，实际在小腿轴上，高 0.24785 m。
- 手册把腰末端当作 torso_Link，实际低 0.0157 m。
- 手册把 A0 放在 URDF 肩关节原点，而 MDH 的 A0 在肩部三轴交点，再向外 0.0725 m。

所以真机上的 `FKb`/`IKb` 很可能与仿真不同；按上面的决定，真机的笛卡尔目标应在主机侧用 `robot_kinematics` 求解（见下一节）。另外，镜像的 SDK 服务端内嵌了一份 `wheel_robot_4.0.urdf`，36 个关节与 `assets/franzi_source` 完全相同，用于它的质心求解器。

验证（`tests/test_simbridge.py` 和验收打分）：

- 经共享运动学算出的 `Fk` 与镜像 MDH 表在同一 SDK 关节值下一致到 1e-12，说明坐标系、映射和单位都对得上。
- `FKb` 与 `robot_kinematics` 的法兰位姿一致到 1e-12。
- IK/IKb 往返在 1e-5 m 以内；欧拉角与 SciPy 互转误差 2e-13。
- 验收打分时用 trace 中的 SDK 关节角和腰角经共享运动学算出抓取点，再与 MuJoCo 自己的抓取点比较（见下面的验收结果）。

## 主机侧笛卡尔目标（`sdk/sdk_cartesian.py`）

上层程序要按笛卡尔目标动手臂时，在主机上解逆解、只发关节命令，仿真和真机收到的是同一组 SDK 关节值：

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
- 姿态默认保持当前姿态，也可以用 `rotations={"left": R}` 指定。逆解用默认的 2° 规划余量。
- 每次求解都读入实测的手臂和腰部关节（`GetJointState`）。
- `move` 两臂都给时发一条双臂 `MoveJ`，轮询 `GetMotionStatus` 等到位；运动以故障（-1）或停转（-3）结束、或逆解无解时抛 `CartesianError`；逆解无解时什么都不发送。
- `line` 用 `robot_kinematics.line` 每 `speed / rate` 取一个逆解路点，两臂重采样成相同点数，经 `TeleJoint(Dual)` 按绝对时刻表推送，结束后 `StopTeleJoint(Dual)`。若某个关节会超过 SDK 当前速度上限的 0.9 倍（`GetJointVelLimit`，已含速度比例，真机也一样），自动放慢并返回放慢倍数。
- 它不放进 `sdk/stock_sdk.py`：服务端的通信层导入那个模块，而有些服务端环境没有 Pinocchio。

验证：

- 测试（`tests/test_simbridge.py`，测试服务端）：抓取点到位误差 < 0.1 mm；直线路点都在线段上；桥实际执行的关节经正解落在目标上；逆解无解时不发送任何命令。
- 实际 MuJoCo 服务端（含物理和伺服）：左手上移 5 cm，到位时误差 0.061 mm，姿态变化 0.19 mrad，用时 0.83 s；双手直线 4 cm（5 cm/s），刚到位时误差 0.044 / 0.137 mm，稳定后按 MuJoCo 自己的抓取点为 0.004 / 0.012 mm，没有触发放慢。
- 真机未试。上真机前仍要核对关节方向、零位和 TCP，并显式设置速度比例：真机默认 30%，仿真为 100%。

## 后端支持与验证状态

| 后端 | 实现 | 本机验证 | 解释器 | 循环 | 说明 |
|---|---|---|---|---|---|
| MuJoCo 3.13 | 已实现 | 已验证（两种循环） | inovance-fineart | framework / simulator | 所有图像尺寸共用一个渲染上下文（每个上下文各带一张 8192² 阴影贴图，约 256 MB）；像素与 `mujoco.Renderer` 相差不超过 1 级 |
| Isaac Sim / Isaac Lab | 草稿 | **未运行、未验证** | GPU 工作站上的 Isaac Sim 5.1（macOS arm64 不能运行） | 设计为两种 | 按 4.5/5.x API 写成，并对照 Isaac Sim 5.1.0 源码修正了导入、World/PhysicsContext、Articulation 接口、物理步回调和接触报告的调用；从未启动过 Isaac Sim，不能视为可用 |

## 启动服务与切换后端

```bash
# MuJoCo（端口来自 config/connections/sim_mujoco.yaml）
python -m simbridge.server --backend config/backends/mujoco.yaml --output outputs/<新目录> [--record]
# MuJoCo，引擎拥有主循环
python -m simbridge.server --backend config/backends/mujoco_owned_loop.yaml --output outputs/<新目录>
# Isaac Sim 草稿的场景：派生场景或相机改动后，从 MuJoCo 场景重新导出
python -m sdk.convert_pybullet_assets
```

- 服务端只监听连接配置中的地址（默认回环），60 秒无请求退出。
- 输出目录须不存在。服务端写 `ready.json`（后端、能力声明、映射、端口）、`trace.json`（25 Hz 状态、设定点、模式、夹爪、物体真值）、`camera_log.json`、`server_result.json`（故障、RPC 记录、实时率），`--record` 时写 `session.mp4`。
- `sdk/convert_pybullet_assets.py` 导出到 `assets/derived/pybullet/`（可重建，不入库）。它没有为夹爪连杆重跑：URDF 没有多项式关节耦合（mimic 关节是线性的），从动连杆会变成自由的。它的输出保留旧的固定夹爪，等 Isaac 草稿建模连杆时再改。

同一份验收程序跑每个后端（现在是 MuJoCo 的两种循环）：

```bash
python -m sdk.verify_backends --output outputs/<新目录> --backends mujoco,mujoco_owned_loop --task --fault-probe --record
python -m pytest tests -q
```

- `sdk/sdk_acceptance.py` 拒绝非回环地址：其中的动作只在仿真中检查过，不能直接对真机运行。
- 验收程序 `config/sdk_acceptance/program.json` 由 `sdk/make_sdk_program.py` 从双臂抓放的参考运行生成（现在是 `outputs/expert_wrist_block/`）；任务布局或专家改了，两者都要重新生成。生成器会把超过 SDK 规划限速（`config/robot_constraints/joint_limits.yaml`，每个手臂关节 1 rad/s、2 rad/s²）0.9 倍的阶段整体放慢，并记录倍数。

## 验收结果

同一份 `sdk/sdk_acceptance.py`（只调用未修改的原厂 SDK）连接服务端，只换连接配置；结果按类别记录，不合成一个通过率。最新一次是 `outputs/sdk_backends_v12/`（2026-10-24；夹爪按 CAD 连杆运动，腕部 D405 在新的倾斜相机块上；任务来源 `outputs/expert_wrist_block/`），汇总在 `report.json`：

| 类别 | MuJoCo（框架驱动） | MuJoCo（引擎自有循环） |
|---|---|---|
| 通信兼容：协议、返回格式、四路相机、未实现接口明确报错 | 28/28 | 28/28 |
| 控制语义：接收≠完成、-3/-5/-8/-10、单/双臂遥操作、软急停、使能 | 37/37 | 37/37 |
| 运动学接口：Fk/FkTool/IK/FKb/IKb 自洽，位姿约定，工具坐标，-3/-6 | 23/23 | 23/23 |
| 运动学映射：全程 SDK 值 = sign·仿真值 + offset | 偏差 0 | 偏差 0 |
| 共享运动学（SDK 关节角 + 腰角）对 MuJoCo 抓取点 | 7.3e-16 m | 7.9e-16 m |
| 物理表现：双臂到位、重力力矩、夹爪 750/250/1000、软急停制动、Home、IKb + MoveJ 抬高 3 cm | 10/10（抬高误差 0.16 mm） | 10/10（0.16 mm） |
| 整段最大接触穿透 | 0.16 mm | 0.16 mm |
| 故障处理：把左手压进桌面 | 5/5，手指越限审计触发，物理冻结并经 SDK 报告 | 5/5，同左 |
| 完整任务（经 SDK 重放专家关节目标）：SDK 侧 | 4/4，抬起后夹爪报 2（夹持） | 4/4 |
| 完整任务：仿真真值（到位、静止、脱离、≥ 0.4 s 双爪悬空） | 成功，误差 0.28 / 0.11 mm | 成功，0.27 / 0.11 mm |
| 录像时实时率 | 0.29 | 0.26 |

- v12 与 v11 的检查项相同，任务程序的动作也相同：重新生成的 `program.json` 只换了来源运行名和机器人配置哈希。变的是腕部相机块：两路腕部画面换成新视角，新相机块和 D405 外壳多了碰撞体；物理结果与 v11 一致。更早各版见 [history.md](history.md#原厂-sdk-验收)。
- MuJoCo 两种循环给同一个 SDK 目标后，仿真关节角相差不超过 0.96 mrad，手部与相机坐标相差不超过 0.95 mm。差别来自首条指令落在不同的仿真时刻（开场头部转向还没结束），不是循环方式本身：首条指令只差 2 ms 时，结果只差 4.7 µm。
- 每个后端目录有 `server/session.mp4`（总览相机加四路 SDK 相机小图和状态栏；v12 分别是 1336 帧和 1332 帧，25 FPS），以及：
  - `config/`：本次使用的后端、连接、机器人配置快照；
  - `client/sdk_calls.json`：每次 SDK 调用的参数、返回和耗时；`client/verification.json`：检查项；`client/*.png`：经原厂图像 API 取得的图像；
  - `server/ready.json`：能力声明与映射；`server/trace.json`：25 Hz 状态、设定点、模式、夹爪、物体真值；`server/camera_log.json`；`server/server_result.json`：故障、全部 RPC、实时率。

## 新增仿真器

1. 在 `simbridge/backends/<引擎>_backend.py` 实现 `SimBackend`（`backends/base.py`）：`load / capabilities / dt / apply / step / read / render / close`，可选 `run_owned_loop`、`task_state`、`tracked_frames`。只说仿真关节名和 SI 单位；`step()` 每步做穿透和越限审计，超阈值抛 `PhysicsFault`；做不到的能力在 `capabilities()` 里声明为不支持，方法本身抛异常。
2. 在 `simbridge/backends/__init__.py` 的 `REGISTRY` 登记模块名和实现/验证状态（懒加载，不影响只用 MuJoCo 的环境）。
3. 新建 `config/backends/<名>.yaml`（解释器、机器人配置、连接配置、运行参数）和 `config/connections/sim_<名>.yaml`。
4. 需要时增加资产转换（参考 `sdk/convert_pybullet_assets.py`：碰撞体从 MuJoCo 派生场景导出，保证各引擎碰撞定义一致）。
5. 运行 `sdk/verify_backends.py --backends <名>`。协议、服务、控制、映射层都不需要修改；业务代码里不应出现按后端名判断的分支。

## 仍未完成或未验证

- 真机：没有连接 RK3588，没有在线参数回读；坐标映射、零位、限位、TCP 都只到「离线文件 + URDF 一致」的程度。
- 未仿真：`MoveL`、笛卡尔点动与遥操作（按「主机侧求解、只发关节命令」的决定不做）、`FkLink` 与 MDH 查询/写入、力控/阻抗/拖动、夹爪力度、标定、RS485/TIO、移动底盘导航、订阅推送、深度图与真实相机内参/畸变、头/腰运动。
- 运动学接口：用共享运动学回答，不复现控制器自己的逆解和分支选择；flan2tool 初值和 `GetCurCartesianPose` 的坐标系是推断；真机控制器的 `FKb` 若按手册常量计算，会与仿真差 0.273 m（见上），需要在真机上核对。
- 轨迹规划、制动和夹爪速度是本仿真的策略（见上），不是原厂控制器的算法；新增引擎时动力学结果允许不同，验收只要求接口语义一致。
- Isaac Sim：2026-10-24 按用户决定暂缓验证。适配器是为 GPU 工作站上的 Isaac Sim 5.1.0 写的，解释器由环境变量 `ISAAC_PYTHON` 指定（`config/backends/isaacsim.yaml`、`sdk/run_isaacsim_4090.sh`）；无头启动需要接受 NVIDIA Omniverse EULA（`OMNI_KIT_ACCEPT_EULA=YES`），须由用户决定。继续时把本工程同步到工作站，先做适配器文档列出的上机检查，再把工作站上服务端的 18400、18500 两个端口转发到本机，在本机运行同一份 `sdk/sdk_acceptance.py`，结果放进 `outputs/<运行目录>/isaacsim/` 后用 `sdk/verify_backends.py --backends isaacsim --rescore` 打分。工作站上的 Python 没有 Pinocchio，运动学接口会明确报 UNSUPPORTED。
- Isaac Lab：可复用同一份派生 URDF、驱动参数、碰撞分组和相机约定，但它的向量化 GPU 主循环需要另写薄适配层，而且 GPU 管线对凸包有 64 顶点上限（74 个碰撞凸包超出），会破坏与 MuJoCo 的碰撞一致性。
