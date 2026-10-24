# 控制器镜像离线检查

原厂资料包 `feimi1010.zip` 里有原厂 SDK 和 RK3588 控制器的更新包。原厂 SDK 单独解压到 `sdk_review/feimi1010/sdk/`；控制器更新包只离线提取了部分文件。`sdk_review/` 是第三方原厂代码，镜像里还有密码类字段，所以不入库。

- `3588控制器/image_wheel.tar.gz` 含 25,062 个文件，解压后共 2,901,908,776 字节，是部署用的更新目录：ARM64 Linux 主程序、运控/轨迹规划/遥操作/碰撞检测动态库、ROS 与导航依赖和配置。`update.img` 是另一种 RKFW 格式的刷机容器，只检查了容器头，没有解出分区。
- 在运控发布目录、`rk3588_controller/`、`app/` 的普通文件里，没有发现核心 C/C++/Python 源文件；其他目录有脚本和 ROS 相关源码。这不等于整个包开源，也不代表审计了所有嵌套归档。`libsdk_server_pkg.so` 已去符号，`ai_com` 仍带符号和调试节，可以继续静态逆向，但不能据此宣称恢复了原始 C++ 工程。
- 没有执行镜像里的程序、部署脚本，也没有刷机。

## 提取的文件

- `sdk_review/controller_image_probe_v2/catalogue.json` 是完整文件清单；`summary.json` 和 `additional_config_manifest.json` 记录提取出的部分文件及其 SHA-256。
- 参数文件含关节限制、MDH、惯量和拓扑预设。镜像里有多套备份和机型配置，离线包的默认配置与真实控制器当前生效的配置仍需分别核对，不能自动覆盖现有的 URDF、零位或保守限制。
- 进一步检查的结果在 `sdk_review/controller_image_probe_v2/findings.json`：
  - 包内默认拓扑为 `whole_body_hard_with_tio`，20 个电机、2 个 TIO。
  - 配置指定的参数根目录 `/inocfg/inbc_for_humanoid` 也有存储文件，已离线提取 `sysctrl_params.json`、`machine_params.json`、头部和腰部参数等。
  - `sysctrl_params.json` 的关节限制明确标注了 deg、deg/s、deg/s²、deg/s³，与 `app/config` 中另一套限制的数值不同。应沿启动配置追踪参数来源，这仍不能当作在线硬件回读。
  - 报告另附 14 个手臂关节的包内范围与源 MuJoCo 范围的对照。

## 用到了哪里

- 关节限位、手臂速度和加速度上限、默认 30% 速度比例：`config/robot/franzi_sdk.yaml`，经 `robot_kinematics.controller_ranges()` 和 `controller_rates()` 用于仿真场景和专家（见 [robot_model.md](robot_model.md#关节限位速度与伺服)）。
- MDH 表、正负号与零位、腰部范围冲突、未使用的笛卡尔样本：见 [sdk_bridge.md](sdk_bridge.md#机器人配置与坐标映射) 和 `config/robot/kinematic_crosscheck.json`。`findings.json` 里「映射未核验」的结论因此收窄为：运动学结构已与离线 MDH 对上，零位标定和在线生效参数仍未核验。
