# Franzi 双臂操作仿真 Benchmark

在汇川（Inovance）Franzi 轮式双臂机器人上评测双臂操作策略（VLA、模仿学习）的仿真 benchmark，用 MuJoCo 和 Isaac Sim。

- **按真机对齐**：运动学、关节限位和速度上限来自真机控制器的资料，夹爪按 CAD 连杆运动，四路相机装在真机的位置上。
- **和真机同一套接口**：仿真端实现了原厂 SDK 的服务（`simbridge/`），上层程序换一个连接配置，就能在仿真和真机之间切换。
- **19 个任务**：从双臂轮流抓放，到厨房、抽屉、购物篮、实验台这类多步骤长任务，再到 10 个工业工位场景。
- **示教数据**：脚本专家用特权信息完成任务，录下四路相机画面和 16 维动作，导出为 LeRobot 数据集。
- **策略基线**：ACT、FineART-VLA 和 SLA 在 Isaac Sim 里闭环评测的成功率和成败视频，见[基线结果](#基线结果)。

<p align="center">
  <img src="docs/media/kitting.webp" width="720" alt="分拣装盘：两只手轮流把方块、圆柱和六角柱放进料盘里同色的格子">
</p>
<p align="center">分拣装盘（8 倍速），右侧小图是两路腕部相机</p>

<p align="center">
  <img src="docs/media/scenes.webp" width="100%" alt="15 个场景在躯干相机里的画面">
</p>
<p align="center">15 个场景的躯干 D435 画面。第一行：抓放和分拣装盘（任务进行中）、三个 RoboSyn 场景（初始状态）；下面两行是 10 个工业场景</p>

## 任务

### 有专家和成功判定的任务

| 任务 | 内容 | 一集 | 专家 |
|---|---|---|---|
| `pick_place` | 左手、右手依次把蓝、橙两个物块放到中间的目标垫上 | 43 s | 成功（布局固定） |
| `kitting` | 每只手把方块、圆柱和六角柱放进料盘里同色的格子，两只手轮流，分 3 个回合 | 110–163 s | 20/20 |
| `robosyn_kitchen`（cook） | 揭开锅盖，指定的食材下锅，另一样装盘，再盖回锅盖，按铃 | 84–104 s | 待重做 |
| `robosyn_kitchen`（swap） | 揭开锅盖，锅里的食材换到盘里，桌上的放进锅，盖回锅盖，按铃 | 105–127 s | 待重做 |
| `robosyn_drawer`（store） | 拉开抽屉，两只手经交接垫传递烧杯，放进抽屉，关上，按铃 | 66–72 s | 待重做 |
| `robosyn_drawer`（retrieve） | 从抽屉取出烧杯，经交接垫放回桌上原位，关上抽屉，按铃 | 71–76 s | 待重做 |
| `robosyn_tray` | 两样食材放进托盘，两只手一起端起托盘，平移到指令说的一侧 | 49–53 s | 待重做 |
| `robosyn_basket` | 牛奶放进藤编篮，一只手侧抓提手把篮子提到指令说的一侧，再把牛奶取出放到收银台 | 98–146 s | 待重做 |
| `robosyn_lab` | 烧杯放上磁力搅拌器，按指定旋钮启动、搅拌 3 s 后再按停止，烧杯移到托盘 | 61–70 s | 待重做 |

- 专家一栏是 20 个随机种子里的成功数，每个种子第一次采样就可解。
- 2026-10-25 机器人模型换成 `wheel_robot_26.8.16_3`（夹爪是独立连杆，长 12 mm）。抓放和分拣装盘在新模型上按 subtask 重写：每段只有一只手对一个物体做一个动作，另一只手不动，结果是新模型上的。RoboSyn 五个场景的专家还是旧模型上写的，在新模型上准备阶段指尖就碰到桌面，要重做；表里的时长是旧模型上的。
- 2026-10-25 起检测手臂与躯干之间的碰撞，成功也要求全程零碰撞。旧模型上，厨房、抽屉、篮子的专家在中线附近干活时，前臂或手腕会顶到胸口。
- 除 `pick_place` 外，每个种子随机化物体的位置、朝向和分配，以及指令的措辞。
- RoboSyn 五个场景用的是 RoboSynChallenge 比赛的道具。各任务的布局、专家规则、成功判定和结果见 [docs/tasks.md](docs/tasks.md)。

### 工业场景

10 个工业工位场景，道具用 Blender 建模，工作台高 0.90 m。2026-10-25 按机器人的实际能力改过一轮：布局、间隙和抓法都按 95 mm 的夹爪和手臂的可达范围定。场景都通过了场景检查：静置不漂移，穿透不超过 0.09 mm，关键操作点都够得到，要夹的点还查了夹爪开口和壳体间隙。10 个场景都有脚本专家（2026-10-25），在当前机器人模型 `wheel_robot_26.8.16_3` 上运行。2026-10-27 放大布局随机化后，MuJoCo 里 40 个种子、Isaac 里闭环 3 个种子都成功：物体挪几厘米、转几十度，料箱、托盘和工装本身也挪动和转动。每集都能按 subtask 切开，每段只有一只手对一个物体做一个动作，只有双手搬同一个物体时两只手一起动。示教数据在 Isaac Sim 里闭环采集：每个场景 200 条，布局和光照按种子随机，共 2000 条、约 47 小时。10 个场景都已搭到 Isaac Sim 5.1 上（PhysX 物理、RTX 渲染），见 [docs/isaac.md](docs/isaac.md)。详见 [docs/benchmark.md](docs/benchmark.md)。

| # | 场景 | 难度 | 内容 | 手臂 |
|---|---|---|---|---|
| 1 | `tote_transfer` | 简单 | 从蓝色料箱取法兰盘，放进灰色料箱 | 左 |
| 2 | `panel_operation` | 简单 | 旋钮拨到 AUTO，再按绿色启动按钮，不碰急停 | 右 |
| 3 | `tray_push` | 简单 | 沿导轨把防静电托盘推到挡块，托盘里的零件不能洒 | 左 |
| 4 | `parcel_sorting` | 中等 | 输送线上 4 个包裹按面单分区放进对应格口，每只手管自己一侧的两个格口 | 双手轮流，各管一半 |
| 5 | `tote_carry` | 中等 | 两只手各夹周转箱一端的箱壁，把它搬到后方的辊道上 | 双手一起 |
| 6 | `shaped_kitting` | 中等 | 4 个零件按形状和朝向放进泡棉的凹槽（每边间隙 4 mm） | 双手轮流，各管两个 |
| 7 | `lidded_tote` | 中等 | 掀开带盖周转箱，取出六角坯放到工位板，再合上盖 | 左 |
| 8 | `palletizing` | 困难 | 4 个纸箱按立牌上的垛型码到托盘上 | 双手轮流，各管两个 |
| 9 | `cnc_tending` | 困难 | 开门，换下虎钳里的成品、装上毛坯，关门，按循环启动 | 双手轮流，需要交接 |
| 10 | `precision_insertion` | 困难 | 轴承套到轴上，销钉换到另一个孔（直径间隙都是 1.0 mm，带导向） | 先左手轴承，再右手销钉 |

## 评测约定

<p align="center">
  <img src="docs/media/cameras.webp" width="60%" alt="四路机载相机的画面">
</p>
<p align="center">策略看到的四路相机（5 倍速）：上排头部和躯干 D435，下排左右腕 D405</p>

- **观测**：四路机载相机的 RGB 画面（640×480，25 Hz），16 维本体状态（左右臂各 7 个关节角，左右夹爪闭合度），一句任务指令。物体真值只给专家和评分用，不能作为策略输入。
- **动作**：16 维，25 Hz：14 个手臂关节的目标角和两个夹爪的闭合度。底盘固定，腰腿和头部不动。
- **物理**：MuJoCo 500 Hz；Isaac Sim 上是 PhysX 500 Hz，碰撞体、质量和关节相同，接触走同一套审计。机器人每个连杆用 STL 凸包碰撞，两臂之间、手臂和躯干之间也检测碰撞。任一接触穿透超过 1 mm，或关节越限超过 0.002 rad，这一集立即判失败。仿真本身崩溃的一集作废，换种子重来，不计入成功率；崩溃指状态不是有限值、接触缓冲溢出、物体被弹飞。
- **成功**：每个任务有自己的判定，一般要求物体到位、静止、不碰机器人，并保持至少 15 个控制拍；写在各场景的 `run.py` 和 `check.py` 里。全程不能有碰撞：机器人只能用手碰要操作的东西，用底盘和轮子碰地面；搬运中的物体只能碰机器人。
- **策略接口**：策略服务端和仿真器之间走 websocket，用 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)（RoboDojo、RoboTwin 评测用的那一层）的协议和适配器约定：每个方法一个 `policy/<方法>/`，同一个评测客户端能跑 MuJoCo 和 Isaac Sim，见 [docs/policy.md](docs/policy.md)。
- **接口**：上层程序经原厂 SDK 驱动仿真服务端，调用方式和真机相同。原厂 SDK 验收在 MuJoCo 两种主循环下全部通过：通信 28/28，控制语义 39/39，运动学接口 24/24，物理表现 10/10，故障处理 5/5。

还没有的：

- 做得完全部工业场景的学习策略：SLA 一个模型管 10 个场景，第 3000 步的全量评测是 5.5%，还在训练；ACT 只训了 4 个场景，FineART-VLA 的成功率还是 0，见下面的基线结果。
- 真机：没有连接过，坐标映射、零位和 TCP 只核对到离线资料；相机内参未标定，没有深度图。SDK 仿真服务端的 Isaac Sim 后端还是草稿，冒烟测试停在关节限位核对上。

## 基线结果

用 Isaac Sim 里采集的专家演示训练，再到 Isaac Sim 里闭环评测，种子都是训练没见过的（10000 起）。成功按上面的约定，碰撞或违规都算失败。模型每次预测 50 步动作：“25 步”“50 步”是执行这么多步再重新预测，“时间集成”是每拍都预测，把重叠的预测加权平均。

| 场景 | 模型 | 25 步 | 50 步 | 时间集成 |
|---|---|---|---|---|
| `tray_push` | ACT，第 30 000 步 | 20/20 | – | 5/5 |
| `panel_operation` | ACT，第 60 000 步 | 0/5 | 0/5 | 0/5 |
| `precision_insertion` | ACT，第 100 000 步 | – | 0/5 | 0/5 |
| `tote_transfer` | ACT，第 60 000 步 | 0/5 | – | 0/5 |
| 全部 10 个场景 | FineART-VLA，第 20 000 步 | 0/80 | 0/5 | – |
| 全部 10 个场景 | SLA，第 3000 步 | – | 11/198 | – |

- ACT 不读文字，每个场景单独训一个模型，用这个场景的 200 条演示；其余 6 个场景还没有训。
- FineART-VLA 是一个模型管 10 个场景。25 步那 80 集里，`tote_transfer` 和 `panel_operation` 各 20 集，其余 8 个场景各 5 集；50 步只测了 5 个场景，各 1 集。
- `tote_transfer` 的指令点名拿哪个零件，表里的 ACT 读不到；加上目标输入接着训的版本也是 0/10。
- SLA 也是一个模型管 10 个场景：pi0.5 加 3D 对齐的预训练，输入绝对手位姿、输出相对手位姿；长场景按步骤给粗提示，由单独训练的调度模型切换。第 3000 步每个场景 20 个种子，成功的是 `tote_carry` 6 集、`tote_transfer` 2 集、`tray_push` 2 集、`panel_operation` 1 集；`cnc_tending`、`shaped_kitting` 各有一集仿真出错，不计。
- SLA 之后的检查点每个场景只测 2 个种子：第 16 000 步短场景 50 步 6/10、长场景 25 步 4/10，`lidded_tote`、`palletizing`、`parcel_sorting` 都整条做成过。训完后的全量评测正在跑。详见 [docs/sla.md](docs/sla.md)。

`tote_transfer` 还试了一个简化版，用来查原因，**不是这个场景的成绩**：只拿法兰盘，手的转角和放置点固定（原场景里专家按种子抽这些量，画面里看不出来），换 100 条新演示，从上表的模型接着训。专家的这些改动只在服务器上的冻结快照里，仓库的代码复现不出来。

| 简化版 `tote_transfer`（种子 10000–10019） | 25 步 | 50 步 | 时间集成 |
|---|---|---|---|
| ACT，第 60 000 步 | 15/20 | 15/20 | 17/20 |
| ACT，第 80 000 步 | 14/20 | 18/20 | 19/20 |
| ACT，第 40 000 步后换手下降时带修正的演示，再训 20 000 步 | 14/20 | 19/20 | 18/20 |
| 同上，再训到 40 000 步 | 8/20 | 16/20 | – |

查到的原因（详见 [docs/fineart.md](docs/fineart.md)）：

- **隐藏随机量**：专家有些选择按种子抽，画面和状态里都看不出来。其中两个场景的成功判定还依赖它们，任何策略都有上限（`tote_carry` 约 40%，`cnc_tending` 约 22%），见 [docs/benchmark.md](docs/benchmark.md#隐藏随机量2026-11-04)。
- **对不准，也不修正**：演示每次都对得很准，没有偏了再修正的画面，模型下降时基本不纠偏。换成带修正的演示后它确实会修正，但手的平均偏差越训越大，成功率没有超过只靠多训。简化版合爪时沿垫片方向偏超过约 14 mm 就很容易抓不稳。`precision_insertion` 的轴承到轴上方时离轴线中位 5 mm，配合余量只有约 1 mm。
- **接缝**：执行完一段换下一段时指令会跳变（最大 84 mrad），能把抓着的零件甩掉。简化版第 80 000 步，25 步有 5 集抬起时滑掉，时间集成没有接缝，只掉了 1 集。
- **FineART**：在 `tray_push` 上规划几乎不跟随托盘的位置（跟随比例 0.06，ACT 1.09）。它每个场景看过的数据比 ACT 少至少 10 倍，是不是这个原因，要单场景训练才能确认。

成功和失败的例子（左边躯干 D435，右边干活那只手的腕部 D405）：

<table>
<tr>
<td width="50%"><img src="docs/media/act_tray_push_ok.webp" width="100%" alt="ACT 把托盘推到挡块"><br><b>成功</b>：<code>tray_push</code>，ACT 第 30 000 步，种子 10000，12 s 推到挡块（3 倍速）</td>
<td width="50%"><img src="docs/media/fineart_tray_fail.webp" width="100%" alt="FineART 没把托盘推到挡块"><br><b>失败</b>：同一布局，FineART-VLA 第 20 000 步。托盘差约 4 cm 没到挡块，托盘里的零件被推开约 7 cm（4 倍速）</td>
</tr>
<tr>
<td><img src="docs/media/act_tote_ok.webp" width="100%" alt="ACT 把法兰盘放进灰箱"><br><b>成功</b>：简化版 <code>tote_transfer</code>，ACT 第 80 000 步，时间集成，种子 10000，23 s 放进灰箱（3 倍速）</td>
<td><img src="docs/media/act_tote_drop.webp" width="100%" alt="法兰盘滑回蓝箱"><br><b>失败</b>：同一模型，种子 10016，20 集里唯一失败的。抓偏了，抬起后法兰盘滑回蓝箱，手空着走到灰箱（3 倍速）</td>
</tr>
<tr>
<td><img src="docs/media/act_insertion_fail.webp" width="100%" alt="轴承没套上轴"><br><b>失败</b>：<code>precision_insertion</code>，ACT 第 100 000 步，50 步，种子 10000。轴承离轴线 3.8 mm，顶在轴端上，松手后滚到台面（4 倍速）</td>
<td><img src="docs/media/act_panel_fail.webp" width="100%" alt="START 按空"><br><b>失败</b>：<code>panel_operation</code>，ACT 第 60 000 步，50 步，种子 10001。旋钮拨到了 AUTO，START 按空（4 倍速）</td>
</tr>
</table>

## 快速开始

需要 macOS 和 conda（mamba）。新建环境 `inovance-fineart`：

```bash
mamba create -n inovance-fineart -c conda-forge python=3.12 pip
mamba activate inovance-fineart
pip install mujoco==3.13.0 numpy==2.2.6 scipy==1.18.1 trimesh==5.1.0 av==15.1.0 \
  pillow==12.3.0 PyYAML==6.0.3 networkx==3.6.1 pytest==8.4.2 pin==3.4.0 \
  cmeel-urdfdom==4.0.1 cmeel-tinyxml2==10.0.0
```

`cmeel-urdfdom` 必须固定在 4.0.1：pin 3.4.0 链接的是 urdfdom 4.0，不固定时 pip 会装 6.0.0，Pinocchio 就无法导入。

在仓库根目录运行，`--output` 要指向一个还不存在的目录：

```bash
# 有专家的任务：跑一集（加 --record 另写 6 条视频），再做多种子验收
python -m scenes.kitting.run --output outputs/kitting_seed0 --seed 0 --record
python -m scenes.kitting.check --output outputs/kitting_check --seeds 20
python -m scenes.robosyn_kitchen.run --output outputs/kitchen_seed0 --seed 0 --record   # 加 --variant swap 是换菜
python -m scenes.pick_place.run --output outputs/pick_place --record
python -m scenes.pick_place.check outputs/pick_place

# 工业场景：搭场景、静置，检查可达性、能不能夹和相机，写 preview.png 和 report.json
python -m scenes.tote_transfer.check --output outputs/bench_check/tote_transfer

# 不依赖物理引擎的测试
python -m pytest tests -q
```

- 专家失败时进程仍以 0 退出，结果要看运行目录里 `summary.json` 的 `success` 和 `error`。
- 每次运行写 `scene.xml`、`summary.json`、`trace.json`（每个控制拍的阶段、动作、状态和物体真值）和 `trajectory.npz`；`--record` 另写总览 `expert.mp4`、四宫格 `cameras.mp4` 和四路相机的原始视频。
- 录好的运行可以转成 LeRobot v2.1 数据集（openpi 一系直接可读），也可导出 v3.0，两版可以无损互转，见 [docs/vla_data.md](docs/vla_data.md)。
- 仓库里没有的第三方资料：RoboSynChallenge 道具的原始文件（`assets/robosyn/`，RoboSyn 五个场景要用）；原厂 SDK 和控制器镜像（`sdk_review/`，原厂 SDK 的连接、SDK 验收和读取镜像的测试要用）。

## 目录

| 路径 | 内容 |
|---|---|
| `robots/` | 每台机器人一个包，目前只有 `robots/franzi/`：源模型 `assets/urdf/`（原样，不改），派生的机器人场景 `scene.py`，共享正逆解 `kinematics.py`（专家、SDK 服务端和主机侧笛卡尔控制都用它），相机 `camera_rig.py`，外观、夹爪连杆和腕部相机座，控制器配置 `config/`，腕部 D405 转接座的设计 `hardware/` |
| `scenes/` | 每个场景一个包：`task.py` 布局和物体，`run.py` 专家和命令行，`check.py` 验收，`assets/` 只有这个场景用的道具 |
| `sim/` | 各场景共用：仿真主循环和逐步审计 `episode.py`，桌面搬运 `tabletop.py`，道具和工位 `prop_assets.py`、`props.py`、`robosyn.py`、`workcell.py`，验收 `bench_check.py`、`scene_check.py` |
| `simbridge/` | 原厂 SDK 的仿真服务端 |
| `sdk/` | 原厂 SDK 的连接、SDK 验收、主机侧笛卡尔控制和离线运动学核对 |
| `data/` | 示教数据的录制、专家运行的转换和 LeRobot 导出 |
| `isaac/` | Isaac Sim 5.1 移植：场景导出、专家闭环、数据采集和策略评测 |
| `policy/` | 各策略方法，一个方法一个子包：`model.py` 由策略服务端提供，`deploy.py` 是仿真端的闭环，`deploy.yml` 是配置；现有回放、ACT、FineART-VLA 和 SLA |
| `eval_client/` | 仿真端：一回合闭环和整套评测，MuJoCo 和 Isaac Sim 都能跑 |
| `fineart/` | FineART-VLA 和 ACT 的训练和检查（单独的环境），之后搬进 `policy/` |
| `config/` | 原厂 SDK 的连接和后端配置、SDK 验收程序 |
| `assets/` | 多个场景共用的资产：道具库 `props/`（几个场景共用的道具，以及还没有场景用的道具），RealSense 相机模型 `sensors/` |
| `tools/blender/` | 工业道具的 Blender 构建脚本 |
| `tests/` | 不依赖物理引擎的 pytest |
| `docs/` | 详细文档和 README 的图 |

## 文档

- [工业场景](docs/benchmark.md)：10 个场景的内容、场景检查、布局约束和已知问题；道具建模规格在 [docs/benchmark_props.md](docs/benchmark_props.md)
- [任务与专家](docs/tasks.md)：有专家的任务的布局、专家规则、成功判定、数据约定和当前结果
- [机器人模型](docs/robot_model.md)：源文件、派生场景和碰撞体、夹爪连杆、运动学、关节限位和伺服、相机、外观和灯光
- [原厂 SDK 仿真服务端](docs/sdk_bridge.md)：分层、控制语义、坐标映射、运动学接口和验收
- [VLA 数据](docs/vla_data.md)：经原厂 SDK 录制、转换专家运行、导出 LeRobot 数据集
- [Isaac Sim 移植](docs/isaac.md)：映射规则、检查结果、和 MuJoCo 的差别、数据采集
- [策略接口](docs/policy.md)：`policy/` 和 `eval_client/`，交换的数据、一回合的流程、命令和依赖
- [FineART-VLA 和 ACT](docs/fineart.md)：数据、训练、闭环评测、结果和原因排查
- [SLA](docs/sla.md)：封装、粗 subtask 和调度模型、训练、闭环结果、MEM 视频记忆
- [控制器镜像离线检查](docs/controller_image.md)
- [历次版本](docs/history.md)：每一版改了什么、当时的结果和结论

## 版权

本仓库没有开源许可证，保留所有权利：欢迎查看，未经许可不得使用、修改或转载。第三方素材：RealSense 相机模型（来源和许可见 `assets/sensors/*/SOURCE.json`）；RoboSynChallenge 的道具（来源见 [docs/tasks.md](docs/tasks.md)，原始文件不在仓库里）。
