# 策略接口：policy/ 和 eval_client/

2026-11-08 用户要求：各种模型方法放进 policy 文件夹，策略和 Isaac 或任何仿真器的通信参考 RoboDojo 的 websocket。RoboDojo 的通信层是 XPolicyLab（[github.com/XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)，Apache-2.0，RoboDojo 和 RoboTwin 的排行榜都经它评测）。这里直接用它的 websocket 传输和适配器约定，钉在 commit `6baccd5`。

## 结构

```text
policy/                      策略端，各方法在自己的环境里跑
  contract.py                双方交换的观测和动作字典（只依赖 NumPy）
  session.py                 每回合状态的记账，各模型的基类
  loop.py                    仿真端的闭环，各 deploy.py 共用
  serve.py                   服务入口：python -m policy.serve --config policy/<方法>/deploy.yml
  <方法>/model.py            Model(config)：模型适配器，由服务端提供服务
  <方法>/deploy.py           eval_one_episode(TASK_ENV, model_client)：仿真端的闭环
  <方法>/deploy.yml          policy_name、端口、模型设置和仿真端设置（horizon），都可用 --set 覆盖
eval_client/                 仿真端，对应 RoboDojo 的 src/eval_client
  env.py                     TaskEnv：XPolicyLab 的 TASK_ENV，MuJoCo 和 Isaac Sim 共用
  episode.py                 跑一回合（替代 isaac/policy.py）
  evaluate.py                多场景、多种子，汇总成 report.json（替代 isaac/evaluate.py）
```

现有的方法：`replay`（回放录好的命令，检验评测通路本身）、`act`（ACT）、`fineart`（FineART-VLA）、`sla`（SLA，含长场景的调度器和 MEM 视频记忆，2026-11-08 用户要求一起封装）。训练代码目前还在 `fineart/` 和 `sla/`，各方法都迁完、新旧通道对上之后再搬进 `policy/<方法>/`，同时删掉旧的 multiprocessing 协议（`fineart/protocol.py`、`fineart/serve.py`、`fineart/act_serve.py`、`isaac/policy.py`、`isaac/evaluate.py`）。

## 交换的数据

观测是 XPolicyLab 的标准字典，按数据集的约定填：

- `instruction`：这一回合的任务指令。
- `vision/<相机>/color`：这个控制拍的画面，uint8 [480, 640, 3] RGB，和录像里的是同一帧。相机用 XPolicyLab 的名字：`cam_head`、`cam_left_wrist`、`cam_right_wrist`；躯干 D435 在标准里没有名字，按数据集叫 `cam_waist`。
- `state/<部件>`：测量的 16 维状态，数据集的 SDK 单位：`left_arm_joint_state`、`right_arm_joint_state` 各 7 个关节（SDK 弧度），`left_ee_joint_state`、`right_ee_joint_state` 各 1 个夹爪位置（0 全闭 .. 1000 全开）。
- `additional_info`：`frequency`（25），`episode`（回合编号），`seed` 和 `tick`（模型的随机噪声用它们定种，回放能逐位复现）。

动作是同样四个部件的目标值，一拍一个；一个动作块是一列这样的字典。

## 一回合怎么走

1. 仿真端开头调 `prepare_case`（XPolicyLab 的回合钩子），带上 `episode`、`seed`、`scene`、`instruction`，模型为这一回合建一份新状态。
2. 动作队列空了，就用这一拍的观测调 `query`，返回一个动作块和要记录的信息（子任务、推理耗时），前 `horizon` 个动作一拍一个执行。
3. 模型在 `describe()` 里声明了 `observe_every` 的，仿真端在没有 query 的这些拍也调 `observe` 给它看一眼（视频记忆要固定节奏的帧，和 horizon 无关）。
4. 结束时调 `trial_end`，模型释放这一回合的状态。

一个服务端可以同时服务几个回合（`eval_client.evaluate` 最多同时跑 8 个，一个 VLA 占约 10 GB 显存），状态按 `episode` 分开。XPolicyLab 自己的单回合方法（`reset`、`update_obs`、`get_action`）也实现了，接的是一个默认回合，所以它的部署循环也能驱动这些模型。

FineART 每 75 拍重新生成一次子任务、把上一次的子任务当记忆，这些原来在评测端，现在都在模型里。ACT 的时间集成器、SLA 的调度进度和 MEM 看过的帧也都是每回合一份。服务端出错或断开，这一回合按错误结束（和专家报错一样记进 summary.json）。

几个回合的调用同时进行：XPolicyLab 的服务端本来在每个模型调用外面套一把全局锁，一次只跑一个调用；`policy.serve` 去掉了这把锁，各模型只锁自己的 GPU 推理，和原来的服务端一样。SLA 每块约 0.3 s 的逆解因此在锁外，和别的回合的推理并行。同一个回合的调用来自它自己的同步客户端，一次一个。

## 命令

```bash
# 策略服务（各方法的环境）。只绑 127.0.0.1：协议没有认证；跨机器用时显式给 --host 0.0.0.0。
<fineart python> -m policy.serve --config policy/act/deploy.yml --set checkpoint=<step 文件夹> [--set ensemble=0.01 --set horizon=1] [--port 18780]
<fineart python> -m policy.serve --config policy/fineart/deploy.yml --set checkpoint=<step 文件夹> --set tokenizer=<paligemma_tokenizer>
<SLA python> -m policy.serve --config policy/sla/deploy.yml --set checkpoint=<检查点> --set assets=<assets>/<asset id> [--set scheduler=<调度器>] [--set memory_frames=6]   # SLA 的环境（docs/sla.md）
python -m policy.serve --config policy/replay/deploy.yml --set trajectory=outputs/<run>/trajectory.npz   # 项目环境
# 一回合：Isaac Sim，或 MuJoCo（项目环境）
ISAAC_PYTHON=<Isaac Sim python> isaac/run.sh eval_client.episode <场景> --policy act --output outputs/<new_run> --seed 10000 [--server ws://127.0.0.1:18780] [--set horizon=50] [--lenient]
python -m eval_client.episode <场景> --engine mujoco --policy act --output outputs/<new_run> --seed 10000
# 整套评测 -> report.json（项目环境）
ISAAC_PYTHON=<Isaac Sim python> python -m eval_client.evaluate --policy act --output outputs/<new_dir> [--scenes tray_push] [--seeds 10000-10019] [--parallel 8] [--set horizon=50]
```

## 依赖

XPolicyLab 用 `pip install --no-deps` 装在 commit `6baccd5`（`git+https://github.com/XPolicyLab/XPolicyLab.git@6baccd57ff958570d0db1588370289ee36c8bb70`），它的依赖单独钉版本装：websockets 15.0.1、msgpack 1.2.4、msgpack-numpy 0.4.8、pydantic 2.14.3；服务端还要 h5py 3.16.0 和 OpenCV（XPolicyLab 的服务端会导入它们）。

| 环境 | 装了什么 | 用途 |
|---|---|---|
| 项目环境（本机 conda、pro6000 venv） | 上面全部，OpenCV 用 opencv-python-headless 4.12.0.88 | 回放服务端、MuJoCo 客户端、评测驱动、测试 |
| Isaac Sim 5.1 | websockets、msgpack、msgpack-numpy、xpolicylab；pydantic 用它自带的 2.11.10 | Isaac 客户端 |
| fineart-train | websockets、msgpack、msgpack-numpy、pydantic、h5py、xpolicylab；OpenCV 用它已有的 4.13 | FineART 和 ACT 的服务端 |
| SLA 的环境 | msgpack-numpy、xpolicylab（另一个会话装的）；websockets 14.2、msgpack、pydantic、OpenCV、h5py 原来就有 | SLA 的服务端 |

2026-11-07 装这些包时每个环境都先冻结了包清单，只新增、没有改动已有包的版本（装前装后的清单留在服务器上）。XPolicyLab 的包会装一个顶层的 `utils` 模块（它的 `XPolicyLab.utils` 靠这个找到），装之前这几个环境里都没有可导入的 `utils`。

## 验证

2026-11-07 在 pro6000 的验证副本里，旧通道（`isaac.policy` 加 `--replay`、`fineart.act_serve` 或 `fineart.serve`）和新通道（`eval_client.episode` 加 `policy.serve`）用同样的种子对比：

- **回放**：tray_push 种子 0 的 Isaac 专家录像，新旧两条通道都成功，都是 19.12 s，`trajectory.npz` 的每个数组都逐位相同。MuJoCo 上回放 MuJoCo 专家录像也成功（21.56 s），同一个客户端两种仿真器都通。
- **ACT**（tray_push 第 30000 步，种子 10000）：两条通道都成功，都是 12.16 s、13 次查询。轨迹不逐位相同（动作最多差 7.8 mrad）：同一个状态下第 0 拍的第一块动作就差 2.3e-4，因为 Isaac 的 RTX 画面每次运行都有噪声，而 ACT 看画面。推理本身是对的：同一个进程里把相同的输入分别交给旧的 `act_serve` 和新的 `policy.act` 模型，三块动作逐位相同。
- **FineART**（franzi_v4 第 20000 步）：同一个进程里相同输入，旧的 `fineart.serve` 和新的 `policy.fineart` 模型在 5 次查询上逐位相同，包括第 0 拍生成子任务、第 25 和 50 拍沿用、第 75 拍带记忆重新生成、第 100 拍沿用，生成的子任务文字也一样。新通道的闭环在 tray_push 种子 10000 上跑满 50 s（50 次查询，编译后每次约 0.17 s），和 FineART 一贯的结果一样到时间上限。
- **评测驱动**：`eval_client.evaluate` 用 ACT 跑 tray_push 种子 10001–10002，2/2 成功，`report.json` 格式和原来相同。

对比时发现旧通道有一个问题：`isaac/policy.py` 加 SLA 的视频记忆后，把服务端信息里的 `memory` 当成视频记忆的配置，而 `fineart.serve` 的 `memory` 是“是否用记忆配方”的布尔值，于是所有带记忆配方的 FineART 检查点在旧通道上都报 `TypeError`（已告诉负责它的会话）。新通道里仿真端只读 `observe_every`，没有这个冲突。旧通道的 FineART 基线因此改在验证前的基线副本里跑。
