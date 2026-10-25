# 工业 benchmark 道具建模规格（Codex + Blender）

这份文档给 Codex 用：Codex 写 Blender Python（bpy）脚本，在无界面模式下建模，导出 MuJoCo 能直接加载的道具。

用法：

1. 先把整份文档交给 Codex。
2. 按顺序发第 4 节的提示词：先发提示词 0 搭好流水线，再发 1–10（每个任务一条），环境道具（提示词 11）可选。
3. 文档里的尺寸单位都是毫米，建模时换算成米。

道具做好后，由仿真这边接进 `scenes/<任务>/`，再调碰撞、关节、质量，写脚本专家和验收。

## 0. 任务与道具一览

| # | 任务 | 难度 | 道具（编号 = 输出文件夹名） |
|---|---|---|---|
| 1 | 料箱间转移零件 `tote_transfer` | 简单 | `small_tote`、`flange_part`、`abs_housing` |
| 2 | 设备控制面板操作 `panel_operation` | 简单 | `control_console`、`panel_devices` |
| 3 | 推物料托盘到工位 `tray_push` | 简单 | `esd_tray`、`tray_rail_set`（托盘里的零件复用任务 1） |
| 4 | 快递包裹分拣 `parcel_sorting` | 中等 | `belt_conveyor`、`sort_bin`、`parcel_box_s`、`parcel_box_m`、`parcel_pouch` |
| 5 | 双臂搬纸箱 `carton_carry` | 中等 | `carton_250x200x150`、`carton_300x200x200`、`carton_220x180x120`、`roller_conveyor` |
| 6 | 零件配盘 `shaped_kitting` | 中等 | `foam_kit_tray`、`hex_blank`、`flanged_bushing`、`l_bracket`、`spur_gear` |
| 7 | 开盖取件 `lidded_tote` | 中等 | `lidded_tote`、`station_plate`、`drawer_cabinet`（可选变体） |
| 8 | 小型码垛 `palletizing` | 困难 | `mini_pallet`、`scissor_lift_table`、`pallet_carton_160x80x80`、`pallet_carton_120x80x80`、`pattern_sign`（供料复用 `roller_conveyor`） |
| 9 | 数控机床上下料 `cnc_tending` | 困难 | `cnc_front`、`part_tray_6`、`round_blank`、`machined_part`（按钮复用 `panel_devices`） |
| 10 | 精密插装 `precision_insertion` | 困难 | `bearing_housing_c010`、`bearing_housing_c020`、`bearing_housing_c030`、`bearing_6204`、`dowel_pin`、`pin_fixture`、`din_rail_kit`（可选） |
| 环境 | 可选 | — | `workbench`、`factory_floor`、`safety_fence`、`background_rack` |

码垛用的是单手能抓的小纸箱（宽 80 mm）：双臂对挤只能搬左右居中的箱子，码 2×2 垛型时两侧的箱子够不着。

## 1. 通用规范（所有道具必须遵守）

### 1.1 软件与运行

- 本机是 Blender 5.2 LTS，路径 `/Applications/Blender.app/Contents/MacOS/Blender`，脚本按 Blender 5.x 的 API 写。统一无界面运行：`<blender> -b --factory-startup -P <脚本> -- <参数>`。
- 烘焙用 Cycles（CPU 即可，64 采样）。
- 脚本放在 `tools/blender/`：公用函数 `prop_lib.py`，每个任务一个 `build_taskNN_<任务名>.py`，检查脚本 `validate_props.py`。检查脚本用 `/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python` 运行（已装 numpy 和 trimesh）；画标签可以用系统 `python3` 里的 Pillow。
- 只在 `tools/blender/` 和 `assets/props/` 下新建、修改文件；不要执行任何会改动仓库状态的 git 命令（commit、push、checkout、reset、stash 等）。

### 1.2 坐标、单位、原点

- Blender 世界坐标就是 MuJoCo 坐标：Z 朝上，单位米。
- 道具坐标系：
  - 原点在包围盒底面中心，除非条目里另有说明；
  - 朝向机器人的“正面”朝 −X，机器人看过去的左右方向是 Y（+Y 在机器人左手边）。
- 导出前对所有物体应用变换（位置、旋转、缩放）。
- 可动部件按“关闭/初始状态”建模，并在道具坐标系下导出：各部件不要各自归零，它们的原点都是道具原点。

### 1.3 文件和目录

每个道具一个文件夹 `assets/props/<编号>/`：

```
source.blend                 Blender 源文件（保留完整 PBR 材质）
<部件>.obj                   视觉网格，每个刚体部件一个
<部件>_albedo.png            该部件烘焙好的颜色贴图（sRGB PNG）
<部件>_albedo_v<k>.png       颜色变体（可选）
<部件>_col_NN.obj            碰撞凸体，NN 从 00 开始
prop.json                    尺寸、质量、摩擦、部件、关节、变体
preview.png                  1024×1024 预览图
```

### 1.4 视觉网格

- 按真实尺寸建模。所有硬边都要倒角：小零件 0.3–0.5 mm，中型道具 1 mm，大型设备 2 mm，每处 2–3 段。
- 三角形上限：小零件 ≤ 1 万，中型道具 ≤ 5 万，大型设备（机床、输送线）≤ 15 万。
- 导出时三角化；平滑着色加 30° 自动平滑，导出法线。
- 不要有 n-gon、重叠面和重复面，尽量保持流形。

### 1.5 材质与贴图（受 MuJoCo 渲染器限制）

- MuJoCo 自带的渲染器每个几何体只用一张颜色贴图，加一组统一的高光参数（specular、shininess、reflectance）。法线、粗糙度、金属度贴图都用不上，贴图的 alpha 也会被忽略（不能靠透明贴图做镂空或网格）。
- 因此：
  - 在 Blender 里按 PBR 做材质（Principled BSDF），保存在 `source.blend` 里，以后换 Isaac Sim 或 Blender 渲染时能直接用。
  - 导出时，把每个部件的全部材质烘焙到一张颜色图集：颜色 = 基础色 × mix(1, AO, 0.6)。不要把直射光和高光烘进去，MuJoCo 会自己打光。
  - 每个部件一套 UV，岛之间不重叠。一般用 Smart UV Project（角度 66°，岛间距 0.01）；箱形件用 Cube Projection。
  - 贴图尺寸：小零件 1024，中型道具 2048，大型设备每个部件 2048（可以拆成多个部件）。PNG，8 位，sRGB。
  - 镂空、网格这类形状必须用真实几何体做出来。
- 金属的光泽靠 `prop.json` 里的 MuJoCo 材质参数来表现。
- 透明件（机床观察窗、亚克力牌）做成单独的部件，不贴图，在 `prop.json` 里给 rgba，例如 `[0.85, 0.90, 0.95, 0.25]`。
- 指示灯等发光件在 `prop.json` 里写 `emission`（0.5–1.0）。

材料参考值：

| 材料 | 基础色 | roughness | metallic | MuJoCo specular / shininess / reflectance | 滑动摩擦 |
|---|---|---|---|---|---|
| PP 塑料 | 按道具 | 0.55 | 0 | 0.25 / 0.30 / 0 | 0.40 |
| ABS | #2B2D30 | 0.45 | 0 | 0.30 / 0.40 / 0 | 0.40 |
| 瓦楞纸 | #B88B5C | 0.85 | 0 | 0.05 / 0.10 / 0 | 0.55 |
| 封箱胶带 | #8B5A2B | 0.30 | 0 | 0.40 / 0.60 / 0 | — |
| 铝（车削、铣削） | #C9CDD1 | 0.30 | 1 | 0.70 / 0.70 / 0.05 | 0.35 |
| 钢（发黑） | #2A2B2F | 0.45 | 0.8 | 0.50 / 0.50 / 0.02 | 0.30 |
| 钢（亮面、镀锌） | #BFC3C7 | 0.30 | 1 | 0.70 / 0.70 / 0.05 | 0.30 |
| 青铜 | #B08D57 | 0.35 | 1 | 0.60 / 0.60 / 0.03 | 0.30 |
| 喷塑钣金 | 按道具 | 0.60 | 0 | 0.30 / 0.30 / 0 | 0.50 |
| 橡胶 | #151515 | 0.90 | 0 | 0.05 / 0.10 / 0 | 0.90 |
| EVA 泡棉 | #161616 | 0.95 | 0 | 0.02 / 0.05 / 0 | 0.80 |
| 松木 | #C49A6C | 0.75 | 0 | 0.10 / 0.15 / 0 | 0.50 |
| PVC 皮带 | #2F7D32 | 0.75 | 0 | 0.15 / 0.20 / 0 | 0.60 |

### 1.6 碰撞代理

- 碰撞形状由若干个凸体拼成，每块单独导出为 `<部件>_col_NN.obj`，坐标系和道具一致。
- 每块必须是凸的：盒子、圆柱（16–32 边，精密件按条目要求），或简化形状的凸包。每块不超过 64 个顶点。
- 凹形一定要拆开，不能用一个凸包包住：料箱 = 底板 + 4 面墙，托盘凹槽 = 槽壁拆块，轴承孔 = 沿圆周拆成扇形块。
- 和视觉外形的误差不超过 1 mm；条目里标了“精密”的地方按条目要求。各块之间可以重叠，但不能伸出视觉表面 1 mm 以上。
- 每个部件最多 64 块；轴承座最多 128 块。
- 精密的圆孔和圆柱用多边形近似：
  - 孔：多边形的内切圆半径（中心到每个平面的距离）等于设计孔半径，这样任何位置的间隙都不会小于设计间隙。
  - 轴、轴承：多边形的顶点落在设计外圆上，这样外形不会比设计尺寸大。

### 1.7 可动部件

- 每个会动的刚体部件单独导出，处于关闭或初始状态。
- 在 `prop.json` 里给出关节：
  - `type`：`hinge` 或 `slide`；
  - `parent`：父部件；
  - `axis`：道具坐标系下的单位向量；
  - `anchor_m`：轴上一点；
  - `range`：弧度或米，0 = 初始状态；
  - `damping`、`frictionloss`；
  - 需要时加 `spring`：刚度和平衡位置，用于按钮回弹和卡扣。
- Blender 只负责几何和参数，关节由仿真实现。

### 1.8 `prop.json` 格式

```json
{
  "id": "lidded_tote",
  "task": "lidded_tote",
  "units": "m",
  "origin": "bottom_center",
  "bbox_mm": [222.0, 304.0, 153.0],
  "triangles": 38214,
  "parts": [
    {
      "name": "body",
      "visual": "body.obj",
      "texture": "body_albedo.png",
      "texture_px": 2048,
      "collision": ["body_col_00.obj", "body_col_01.obj", "body_col_02.obj", "body_col_03.obj", "body_col_04.obj"],
      "mass_kg": 1.0,
      "friction": 0.45,
      "mujoco_material": {"specular": 0.25, "shininess": 0.30, "reflectance": 0.0},
      "parent": null,
      "joint": null
    },
    {
      "name": "lid",
      "visual": "lid.obj",
      "texture": "lid_albedo.png",
      "texture_px": 2048,
      "collision": ["lid_col_00.obj", "lid_col_01.obj"],
      "mass_kg": 0.25,
      "friction": 0.45,
      "mujoco_material": {"specular": 0.25, "shininess": 0.30, "reflectance": 0.0},
      "parent": "body",
      "joint": {"type": "hinge", "axis": [0, -1, 0], "anchor_m": [0.102, 0.0, 0.152],
                "range": [0.0, 1.92], "damping": 0.05, "frictionloss": 0.4}
    }
  ],
  "variants": [{"name": "grey", "textures": {"body": "body_albedo.png"}},
               {"name": "blue", "textures": {"body": "body_albedo_v1.png"}}],
  "notes": "Lid hinge frictionloss holds the lid wherever it is left."
}
```

`bbox_mm` 按 X、Y、Z 顺序，对应所有部件在初始状态下的总包围盒。精密件额外写 `"precision_mm": 0.01` 和关键尺寸。

### 1.9 真实感要求

- 喷涂件和塑料件的棱边要有 1–2 mm 的磨损亮边。
- 凹处和角落要有积灰：按 AO 加深 10–20%。
- 把手和按钮上要有手印和污渍。
- 金属件要有加工纹：车削面是同心纹，铣削面是平行弧线，钣金和型材是直线拉丝。
- 纸箱：
  - 裁切边能看到瓦楞截面，角上有轻微压痕；
  - 胶带有细褶和气泡；
  - 标签歪斜 1–3°。
- 同一种道具的多个实例之间，颜色差约 ±3%。
- 不要有完美锐利的边。

### 1.10 商标与版权

- 只用虚构名称：快递公司 “QF Express”、仓储公司 “NOVA Logistics”、电气件 “IMC Controls”、机床 “IMC-CNC 850”、轴承品牌 “NVK”。
- 不要出现任何真实品牌或商标，比如快递公司、机床厂、轴承厂、汇川。
- 安全标识按通用图形自己画（三角警示、箭头、易碎杯），不要复制官方原图。

### 1.11 预览图和验收

- `preview.png`：1024×1024，Cycles 64 采样，中性灰背景，三点布光。相机放在前左上方 3/4 视角，沿 +X 方向看，和机器人的视角一致。
- `validate_props.py` 必须全部通过：
  - `prop.json` 里引用的文件都存在；
  - 视觉包围盒和 `bbox_mm` 相差不超过 1 mm（精密件 0.1 mm）；
  - 每块碰撞体是凸的（凸包体积 / 自身体积 ≥ 0.99），且不超过 64 个顶点；
  - 贴图尺寸和 `texture_px` 一致；
  - 三角形数在 1.4 节的上限以内；
  - 关节字段完整。

## 2. 机器人工作区约束

机器人底盘原点在世界 (0, 0, 0)，+X 朝前，+Y 朝左，+Z 朝上。目前底盘固定、站立姿态。

| 项 | 数值 |
|---|---|
| 工作台面高度 | 800 mm |
| 手能够到的范围 | x 300–480 mm；左手 y 30–450 mm，右手 y −450 到 −30 mm；高度在台面以上 0–200 mm 最好 |
| 头部相机 | 从 x≈330 mm 开始能看到台面 |
| 夹爪 | 两指平行，最大开口约 95 mm，被抓尺寸建议 20–85 mm |
| 单手负载 | 仿真每指最大 6 N（未标定），单手抓的东西 ≤ 0.5 kg |
| 更大更重的箱子 | 双臂从左右两侧对挤；箱子中心要在 y≈0 附近 |

道具的操作点（被抓、被按、被放的位置）必须落在上面的范围里。大型设备（机床、输送线、托盘）的其余部分可以超出范围。下面各条里的“建议摆放”只是参考，最终位置由仿真按可达性来定。

## 3. 道具规格

### 3.1 任务 1：料箱间转移零件 `tote_transfer`

建议摆放：两个料箱并排放在台面上，中心 x≈400，y 分别 ≈+200 和 −200。

#### `small_tote` 小型周转箱（任务 1、3、7 共用）

- **外形**：顶部外沿 300（Y）× 200（X）× 100（高）。侧壁拔模 2°（底部外形约 293×193），壁厚 3，底厚 4，顶部翻边宽 6、厚 4。
- **细节**：
  - 长边外侧 4 条竖筋，短边外侧 2 条，筋宽 3、凸出 5；
  - 一个短边上开把手孔 80×25（圆角 R10），孔上沿距顶面 15；
  - 另一个短边有标签卡槽 75×35，凹进 1，插一张白卡，写 “BIN-A07”；
  - 底面有 3 mm 深的加强格纹。
- **材质**：PP，蓝 #1F5FA8；变体灰 #9AA0A6。有轻微划痕，底部有污渍。
- **质量 / 摩擦**：0.45 kg / 0.40。
- **碰撞**：底板 1 块、四面墙各 1 块，可以再加 4 块翻边，共 5–9 块盒子。
- **变体**：`body_albedo.png`（蓝）、`body_albedo_v1.png`（灰）。

#### `flange_part` 铝合金法兰盘（任务 1、3、7 共用）

- **外形**：Ø60 × 12 的盘，上面一个 Ø32 × 8 的凸台，总高 20。中心通孔 Ø20；4 个 Ø6.6 通孔，均布在 Ø45 圆周上。外缘倒角 1×45°，孔口倒角 0.5。
- **材质**：6061 车削铝 #C9CDD1，端面有同心车削纹，侧面有细纹。
- **质量 / 摩擦**：0.090 kg / 0.35。
- **碰撞**：两个圆柱凸体：盘 Ø60×12，凸台 Ø32×8。孔不做碰撞。

#### `abs_housing` 注塑外壳（任务 1、3、7 共用）

- **外形**：70（Y）× 45（X）× 30（高）的圆角盒，圆角 R4。顶面下 2 mm 处有一圈分模线，顶面 4 个 Ø3 螺丝孔。两侧各一个卡扣凸起 8×3×1.5。正面一个 20×8 的指示窗，贴图表现即可。
- **材质**：深灰 ABS #2B2D30，带细皮纹。
- **质量 / 摩擦**：0.040 kg / 0.40。
- **碰撞**：一个盒子，圆角忽略。

### 3.2 任务 2：设备控制面板操作 `panel_operation`

建议摆放：控制台放在台面中央，前沿 x≈330。

#### `panel_devices` 22 mm 按钮元件库（任务 2、9 共用）

每种元件是一个独立部件，导出在元件自己的坐标系下：原点在安装面中心，按压方向朝 −Z。由使用它的道具在 `prop.json` 里给出安装位置。

| 部件 | 几何 | 颜色 | 关节 |
|---|---|---|---|
| `bezel` 安装圈 | Ø29.5 × 3 镀铬圈 | 亮银 | 无 |
| `btn_green` / `btn_red` / `btn_yellow` 平头按钮 | 按钮帽 Ø24 × 8，高出安装圈 6 | #2E9E44 / #D32F2F / #F2B705 | slide，沿按压方向，行程 0–4 mm，弹簧回弹 |
| `selector_knob` 两档旋钮 | 底座 Ø27 × 6，拨杆 32 × 9 × 14，黑色，顶面一条白指示线 | #1A1A1A | hinge，绕安装面法线，−45°（MAN）到 +45°（AUTO），两端有定位 |
| `estop` 急停 | 红色蘑菇头 Ø40 × 18，下面是黄色圆牌 Ø60 × 1.5，印 “EMERGENCY STOP” | #D32F2F / #F4C20D | slide，行程 0–5 mm，按下后锁住 |
| `pilot_green` / `pilot_amber` / `pilot_red` 指示灯 | 透镜 Ø24 × 6，平顶 | #2ECC40 / #FFB000 / #E53935，`emission` 0.8 | 无 |
| `legend_plate` 铭牌 | 30 × 10 × 1，白底黑字 | 白 | 无（文字做多张贴图：START、STOP、AUTO、MAN、CYCLE START、FEED HOLD） |

- **质量 / 摩擦**：每个按钮帽 0.01 kg，摩擦 0.8（橡胶感）。
- **碰撞**：按钮帽、急停头各一个圆柱，拨杆一个盒子。

#### `control_console` 台式斜面操作台

- **外形**：底面 320（Y）× 200（X），后高 160，前高 60，顶面是斜面（倾角约 26.6°，朝向机器人）。钣金厚 1.5，四角 R3。
- **面板**：斜面上一块深灰面板 280 × 190（沿斜面方向），开 Ø22.3 安装孔，装 `panel_devices` 的元件。面板坐标：u 沿 Y，v 从斜面前沿沿斜面向上，单位 mm。
  - 布局 A：第一排 v=150，三个指示灯在 u = −60、0、+60；第二排 v=70，旋钮 u=−90，启动 u=−20，停止 u=+40；急停在 u=+110、v=110。每个元件下方 15 mm 贴铭牌。
  - 布局 B：布局 A 左右镜像。
  - 布局 C：只有一排 v=110：旋钮 −110、启动 −50、停止 +10、急停 +100，指示灯移到 v=170。
  - 三种布局的元件中心坐标写进 `prop.json` 的 `variants`。
- **其他细节**：右侧面贴一块铭牌 “IMC Controls · OP-03”，后面有一根电缆接头。
- **材质**：喷塑钣金 RAL 7035 浅灰 #D7D7D7，面板深灰 #3A3D40；变体蓝 #3B6EA5。
- **质量 / 摩擦**：4.0 kg（可以固定在台面上）/ 0.50。
- **碰撞**：底板、斜面、两侧板、前板、后板共 6 块。元件的碰撞跟着各自部件走。

### 3.3 任务 3：推物料托盘到工位 `tray_push`

建议摆放：导轨沿 Y 放，中心 x≈400；托盘从 y≈+250 推到另一端的挡块。

#### `esd_tray` 防静电托盘

- **外形**：300（Y）× 200（X）× 50（高），壁厚 3，底厚 3，四角 R8。两个短边各开一个手扣孔 70×22。
- **细节**：侧面印通用的黄色 ESD 三角标识和 “ESD SAFE”。盘里可以有浅分隔筋（视觉）。
- **材质**：黑色导电 PP #1B1B1B，roughness 0.65，有磨痕。
- **质量 / 摩擦**：0.35 kg / 与导轨 0.25（滑动）。
- **碰撞**：底板加四面墙，共 5 块。
- 托盘里的零件复用 `flange_part`、`abs_housing`。

#### `tray_rail_set` 导轨和挡块

- **导轨**：两根 20×20 铝型材（带 T 型槽外形，槽口 6），长 700，沿 Y 放，两轨内侧间距 206（托盘宽 200，每侧留 3）。两根 20×20×246 的横梁把两轨连起来，分别在两端以内 50。
- **挡块**：放在 −Y 端，40×40×30 铝块，朝托盘的一面贴 5 mm 黑色橡胶缓冲块，顶面贴黄黑警示胶带。
- **材质**：阳极氧化铝 #B9BEC3。
- **碰撞**：每根导轨 1 块、每根横梁 1 块、挡块 1 块。

### 3.4 任务 4：快递包裹分拣 `parcel_sorting`

建议摆放：皮带线沿 Y 放在前方，皮带靠机器人一侧的边沿在 x≈380；三个格口箱放在机器人够得到的台面区域。

#### `belt_conveyor` 皮带输送线

- **坐标**：原点在地面、输送线占地中心。长度方向沿 Y，物料从 +Y 流向 −Y。
- **主要尺寸**：
  - 输送长度 800，两端滚筒 Ø50，滚筒中心距 760；
  - 皮带宽 150（X），厚 3，皮带面高度 800；
  - 侧梁是 40×40 铝型材，顶面在 790；
  - 挡边：不锈钢条，厚 2，高出皮带面 30，两侧挡边内侧间距 160，每 200 mm 一个支架；
  - 4 条 40×40 支腿，底部是 Ø50×20 的调平脚；z=200 处有横撑。
- **附件**：
  - 减速电机在 −Y 端、远离机器人的一侧（+X）：减速箱 100×80×80 加 Ø80×150 电机，灰 #6B7075；
  - −Y 端靠机器人一侧有光电开关，30×20×12，装在支架上，高度 830；对面装一块红色反光板 30×30×3；
  - −Y 端有一块挡料板，不锈钢 160（X）× 40（Z）× 3，顶面高度 840。
- **部件**：皮带面单独做成部件 `belt`（760×150×3 的薄盒），贴绿色 PVC 菱形纹，方便仿真让它“动起来”。其余是 `frame`。
- **碰撞**：皮带面 1 块、两侧挡边 2 块、挡料板 1 块、两侧梁 2 块，不超过 8 块。

#### `sort_bin` 分拣格口箱（做 3 个变体 A、B、C）

- **外形**：200（X）× 160（Y）× 150（高），1 mm 镀锌钢板。朝机器人的一侧（−X）有 30 mm 宽的导入斜唇，向外倾斜 30°。
- **分区牌**：后侧（+X）一角立一根 Ø20 立柱，高出箱底 300，顶端装一块 120×80 的分区牌：A 红 #D32F2F、B 蓝 #1976D2、C 绿 #388E3C，上面一个白色大字母。
- **材质**：镀锌钢 #AEB4BA，有锌花纹理。
- **质量**：0.8 kg（静止）。
- **碰撞**：底板、四面墙、斜唇，共 6 块。
- **变体**：A、B、C 只有分区牌贴图不同。

#### 包裹 `parcel_box_s`、`parcel_box_m`、`parcel_pouch`

| 编号 | 尺寸（Y×X×高） | 形态 | 质量范围 | 摩擦 |
|---|---|---|---|---|
| `parcel_box_s` | 140 × 75 × 60 | 单瓦楞纸盒，壁厚 3，盖缝 + 48 mm 透明胶带 | 0.15–0.35 kg | 0.60 |
| `parcel_box_m` | 180 × 80 × 80 | 同上 | 0.20–0.50 kg | 0.60 |
| `parcel_pouch` | 200 × 80 × 35 | 白色 PE 快递袋外形：鼓起的圆角块，边缘有热封边 | 0.10–0.30 kg | 0.50 |

- **面单**：贴在顶面，100×60（快递袋上 90×60），做 A、B、C 三个贴图变体。每张面单包含：
  - 虚构快递 “QF Express” 的标志；
  - 运单号，例如 “QF2026102400017”；
  - 一组条码（随机黑白条）；
  - 左上角一块大的分区色块和字母：A 红、B 蓝、C 绿，和格口箱对应；
  - 虚构的收寄件信息。
- **碰撞**：每个包裹一个盒子，快递袋用 196×76×33 的盒子。

### 3.5 任务 5：双臂搬纸箱 `carton_carry`

建议摆放：两段辊道沿 X 放在台面上；纸箱在 y≈0，两只手从左右两侧对挤。

#### 纸箱 `carton_250x200x150`、`carton_300x200x200`、`carton_220x180x120`

- **形态**：FEFCO 0201 开槽箱（常见快递纸箱），单瓦楞，壁厚 4。尺寸写法是 Y × X × 高：对挤方向（Y）是第一个数。
- **细节**：
  - 顶面两片摇盖中间有 1–2 mm 的缝；
  - 一条 48 mm 棕色封箱胶带沿中缝贴，两端各下翻 50；
  - 侧面印黑色通用图标：向上箭头、怕湿伞、易碎杯；
  - 贴一张虚构的 SKU 标签 “NOVA Logistics · SKU-B0042”；
  - 角上有轻微压痕，摇盖边能看到瓦楞截面。
- **材质**：牛皮纸 #B88B5C；胶带 #8B5A2B，带光泽。
- **质量 / 摩擦**：每个尺寸都写三档质量变体 0.5、1.0、1.5 kg（箱内看不见的配重）；摩擦 0.55。
- **碰撞**：一个盒子，等于外形尺寸。

#### `roller_conveyor` 台面式无动力辊道（任务 5、8 共用）

- **坐标**：原点在辊道底面中心，长度方向沿 X，物料朝 −X（朝机器人）方向流。
- **尺寸**：
  - 长 500，两侧框之间内宽 320（Y）；
  - 滚筒 Ø30 × 316 镀锌钢管，间距 50，共 10 根；滚筒顶面高出台面 50；
  - 侧框是 40（高）× 20（宽）× 2 的 C 型钢，顶面比滚筒顶面低 10。框比滚筒低，两只手能从左右两侧贴住纸箱，不会撞框。
  - 四个 L 形支脚 40×40 座在台面上。
- **挡块**：下游端（−X）一根可拆的挡杆 320×20×30，比滚筒顶面高 20，作为单独部件 `end_stop`。
- **材质**：侧框蓝 #1E4C9A；滚筒镀锌 #C3C8CC。
- **关节**：每根滚筒是一个部件 `roller_00` 到 `roller_09`，hinge 轴沿 Y，自由转动，damping 0.001。
- **碰撞**：每根滚筒 1 个 16 边圆柱，两侧框各 1 块，挡杆 1 块。

### 3.6 任务 6：零件配盘 `shaped_kitting`

建议摆放：配盘放在 x≈400、y≈0；4 个零件散放在配盘两侧。

#### `foam_kit_tray` 泡棉内衬配盘

- **外壳**：灰色 PP 托盘 380（Y）× 260（X）× 50，壁厚 3。
- **内衬**：黑色 EVA 泡棉 372 × 252 × 40，顶面平整，嵌在外壳里。
- **凹槽**：2×2 排列，槽形按零件俯视轮廓每边外扩 1.5 mm。每个槽朝机器人一侧（−X）有一个 Ø24 的半圆指扣槽，深度和该槽相同。槽中心（相对内衬中心，单位 mm，(Y, X)）：
  - 六角坯 (−90, +55)：正六边形，对边 30，深 25；
  - 法兰轴套 (+90, +55)：上段 Ø43 深 5，下面接 Ø33 深 25，总深 30。只有法兰朝上才能放到底。
  - L 形支架 (−90, −55)：支架侧放时的 L 形轮廓，外形 63 × 43、腿厚 7，深 25；
  - 齿轮 (+90, −55)：Ø51 深 10，轮毂朝上放。
- **材质**：泡棉 #161616，带细泡孔纹理；外壳 #8A9097。
- **质量**：0.6 kg（静止）。
- **碰撞**：内衬按槽拆块。六边形槽用 6 块墙；圆形槽每段用 16 块墙；L 形槽用矩形块拼；槽和槽之间用盒子填充。总数不超过 64 块。

#### 零件

| 编号 | 几何 | 材质 | 质量 | 碰撞 |
|---|---|---|---|---|
| `hex_blank` 六角坯 | 对边 27，高 30，两端倒角 1×30° | 发黑钢 #2A2B2F | 0.149 kg | 一个六棱柱 |
| `flanged_bushing` 法兰轴套 | 外径 30、内径 20，本体长 25，法兰 Ø40 × 5，总高 30，倒角 0.5 | 青铜 #B08D57 | 0.128 kg | 本体、法兰两个圆柱（内孔不做） |
| `l_bracket` L 形支架 | 两腿外长 60 和 40，宽 40，厚 4，内弯 R4；60 腿上两个 6.5×12 腰形孔，40 腿上一个 Ø6.5 孔 | 镀锌钢 #BFC3C7，带锌花 | 0.120 kg | 两条腿各一个盒子 |
| `spur_gear` 直齿轮 | 模数 1.5、30 齿，分度圆 Ø45，齿顶圆 Ø48，齿宽 12；单侧轮毂 Ø24 × 6；中心孔 Ø8，带 3 mm 键槽；用真实渐开线齿形 | POM 白 #ECEAE3；变体为钢 #9CA3AA | 0.029 kg（POM） | 齿顶圆柱、轮毂圆柱两块 |

所有零件的摩擦都是 0.30。

### 3.7 任务 7：开盖取件 `lidded_tote`

建议摆放：箱子放在 x≈400、y≈+150，铰链在远离机器人的一侧；`station_plate` 放在 y≈−200。

#### `lidded_tote` 带盖周转箱

- **箱体**：外形 300（Y）× 200（X）× 150（高），拔模 2°，壁厚 3，顶部翻边 6。
- **盖子**：304 × 204 × 15，四周有 12 mm 下翻裙边。
  - 后侧（+X）顶边装两只铰链，每只长 40、销 Ø5。转轴沿 Y，过点 (x=102, z=152)。
  - 前侧（−X）有一片拉手舌 60 × 18 × 3，向前伸出、略向下倾。
  - 前侧有一个卡扣：盖上的凸起卡进箱体的凹槽。
- **材质**：灰 PP #7A7F85；盖子上贴虚构标签 “PARTS · LINE 3”。
- **关节**：盖子 hinge，范围 0–110°（0 = 关闭）。damping 0.05；frictionloss 0.4 N·m，让盖子开到哪停到哪。卡扣在 0–3° 范围内给 0.3 N·m 的回位力矩，写在 `prop.json` 的 `notes` 里，由仿真实现。
- **质量 / 摩擦**：箱体 1.0 kg（可固定），盖子 0.25 kg / 0.45。
- **碰撞**：
  - 箱体：底板加四面墙；
  - 盖子：顶板 1 块、裙边 4 块、拉手舌 1 块。
- 箱里的零件复用 `flange_part`、`abs_housing`、`hex_blank`。

#### `station_plate` 工位定位板

- 200（Y）× 150（X）× 10 铝板，四角是沉头螺钉。表面丝印黄色定位框（零件轮廓）和十字中心标记，编号 “ST-02”。
- 材质：铝 #C5CACF。静止。碰撞 1 块。

#### `drawer_cabinet` 抽屉式零件柜（可选变体）

- **柜体**：300（Y）× 250（X）× 200（高），1 mm 钢板，蓝灰 #3E6C9C。
- **抽屉**：两层。每层面板 290 × 95，抽屉内腔 270 × 230 × 80。银色拱形把手 100 × 15；面板上有标签卡槽 50 × 20。
- **关节**：每个抽屉 slide，向 −X（朝机器人）拉出，行程 0–180 mm，damping 1.0。
- **质量**：柜体 4 kg（固定），每个抽屉 0.6 kg。
- **碰撞**：柜体外壳 5 块；每个抽屉底板、四壁加面板。

### 3.8 任务 8：小型码垛 `palletizing`

建议摆放：升降台和托盘放在 x≈400、y≈0，托盘顶面高 800；供料辊道（复用 `roller_conveyor`）放在侧面。纸箱单手抓：抓 80 mm 那一边，每个箱子由离它近的那只手抓。

#### `mini_pallet` 缩比木托盘

- **外形**：400（Y）× 300（X）× 80（高）。
- **结构**：
  - 上铺板 5 块，每块 300（X）× 64（Y）× 12，沿 Y 排，板间隙 20；
  - 下面 9 个木块 60 × 60 × 56，排成 3×3：x 在 −120、0、+120，y 在 −170、0、+170；
  - 底板 3 块 64（X）× 400（Y）× 12，位于 x = −118、0、+118。
- **细节**：木纹，深色钉帽，少量污渍，边角磨损，侧面印一个虚构的熏蒸标记。
- **材质**：松木 #C49A6C；变体为蓝色 HDPE 塑料托盘 #1F4E8C，网格面。
- **质量 / 摩擦**：2.5 kg（可固定在升降台上）/ 纸箱对木头 0.50。
- **碰撞**：5 块上铺板各 1 块；木块和底板合成一个整体盒。

#### `scissor_lift_table` 剪叉升降台

- **尺寸**：台面 500（Y）× 400（X）× 40，台面顶高 720，放上托盘后托盘顶面是 800。底座 520 × 420 × 60。两组黑色剪叉臂，30 × 15 方管。
- **材质**：钢，黄 #E3B505，台面边缘有黄黑斜纹；剪叉臂 #222222。
- **其他**：`prop.json` 写可调高度 620–760，供变体使用；本身静止。
- **碰撞**：台面 1 块、底座 1 块。

#### 码垛纸箱 `pallet_carton_160x80x80`、`pallet_carton_120x80x80`

- 形态和细节同 3.5 节的纸箱，SKU 标签分别写 “SKU-A” 和 “SKU-B”，顶面印一个朝向箭头。
- 质量：0.35 kg 和 0.25 kg；摩擦 0.55。
- 碰撞：一个盒子。

#### `pattern_sign` 垛型示意立牌

- 一张 210 × 297 × 3 的透明亚克力板（透明部件，给 rgba），插在 220 × 60 × 20 的底座上。板里夹一张垛型图：俯视图，第 1 层 2×2，第 2 层转 90°，箱子有编号，页脚写虚构的 “NOVA Logistics”。
- 做 3 种垛型，即 3 张贴图变体。
- 静止，碰撞 1 块。

### 3.9 任务 9：数控机床上下料 `cnc_tending`

建议摆放：机床前脸放在 x≈340，正对机器人；进料盘和出料盘放在机床前方左右两侧的台面上。

#### `cnc_front` 机床前段（不是整台机床）

- **坐标**：原点在地面、前脸中心；x=0 是前脸平面，+X 向机床里面。
- **外罩**：
  - 宽 900（y −450 到 +450）× 高 1600 × 深 700，钣金 1.5 mm，圆角；
  - 底部 0–120 高是缩进 50 的踢脚；
  - 前脸开口 y −240 到 +240、z 800 到 1220；
  - 开口下沿内侧是一块向里倾斜的排屑斜板（视觉）。
- **推拉门 `door`**：
  - 门框 500（Y）× 440（Z）× 25（X），关门时覆盖 y −250 到 +250、z 790 到 1230，门前面在 x=−30；
  - 观察窗是 380 × 300 × 5 的聚碳酸酯，作为透明部件 `door_window`，rgba [0.85, 0.90, 0.95, 0.25]；
  - 把手：竖直圆杆 Ø20 × 180，装在门的 −Y 边（机器人看过去的右边），中心 y=−220，z 900 到 1080，离门面 30（即 x≈−60）；
  - 关节：slide，沿 +Y 方向开，行程 0–0.46 m，damping 8，frictionloss 3 N；门质量 4 kg。
- **工作台**：内部 T 型槽工作台，台面在 z=800，范围 x 20 到 420、y −200 到 +200，台面上 3 条 T 型槽（视觉）。
- **平口钳**：
  - 钳身 x 30 到 170、y −45 到 +45、z 800 到 850，铸铁蓝灰 #4A5A6A；
  - 固定钳口在远端：钳口面在 x=120，钳口块 25 厚；
  - 活动钳口 `vise_jaw` 在近端：张开时钳口面在 x=60，夹住 Ø40 工件时在 x=80；
  - 钳口板 90（Y）× 30（Z），高度 z 850 到 880，有斜纹齿；
  - 两块平行垫铁 80 × 8 × 10，把工件底面垫到 z=860；工件中心在 x=100、y=0。
  - 活动钳口关节：slide，沿 +X，0 = 张开，0.02 m = 夹紧。由仿真在“按启动”后自动夹紧，机器人不用去拧手柄。
- **主轴（静止）**：在钳子后上方，x 130 到 260，主轴头 160（Y）× 130（X）× 250（Z），底部是 Ø100 主轴鼻和刀柄，装 Ø10 立铣刀，刀尖在 z=1150。从前上方伸手取件时不会被挡。
- **操作面板**：
  - 前脸 −Y 侧（门右边），y −440 到 −290、z 900 到 1250，凸出 40；
  - 装 `panel_devices` 元件：CYCLE START（绿，z=1000）、FEED HOLD（黄，z=1000，旁边）、急停（z=1180）、两档模式旋钮，按压方向朝 +X；
  - 顶部一块 120 × 90 的显示屏，贴虚构的 “IMC-CNC 850” 界面。
- **外观**：
  - 外罩浅灰 #DADDE0，饰条深灰 #3B3F44，前脸一条蓝色 #1F5FA8 装饰带，印虚构 “IMC” 标识；
  - 通用的黄色三角警示标签；
  - 机床内部（工作台、钳身、排屑斜板）贴铝屑和冷却液痕迹。
- **碰撞**：
  - 开口四周的前脸板 4 块；
  - 工作台 1 块；
  - 钳身 2–3 块，每个钳口 1 块，垫铁 2 块；
  - 门框 4 块、观察窗 1 块、把手 1 块；
  - 主轴 2 块；
  - 面板 1 块，元件按 `panel_devices`。
- **三角形**：不超过 15 万。

#### `part_tray_6` 6 孔工件盘（做进料、出料两个变体）

- **外形**：250（Y）× 170（X）× 30 铝盘。6 个 Ø44 盲孔，深 20，排成 2×3：x 在 −35、+35，y 在 −75、0、+75。孔口倒角 1。
- **标记**：盘边激光刻字，变体分别是 “IN” 和 “OUT”。
- **材质**：拉丝铝 #BFC4C9。
- **质量**：0.9 kg（静止）。
- **碰撞**：底板 1 块；每个孔用 8 块墙围成八边形（内切圆半径 22），其余部分用盒子填充；总数不超过 64 块。

#### `round_blank` 毛坯、`machined_part` 成品

| 编号 | 几何 | 外观 | 质量 | 碰撞 |
|---|---|---|---|---|
| `round_blank` | 6061 铝棒 Ø40 × 60 | 两端是锯切面（粗糙、有锯纹），侧面是轧制面（略暗、纵向纹），#BCC1C6，roughness 0.5 | 0.204 kg | 一个 24 边圆柱 |
| `machined_part` | Ø40 × 60，顶部车成 Ø30 × 20 的台阶，中心孔 Ø12 深 15，所有棱边倒角 1×45° | 车削亮面，roughness 0.25，有同心纹，和毛坯一眼能分开 | 0.170 kg | Ø40×40 圆柱 + Ø30×20 圆柱 |

两者的摩擦都是 0.35。

### 3.10 任务 10：精密插装 `precision_insertion`

建议摆放：轴承座固定在台面 x≈400、y≈0；待装的轴承或销钉放在一侧的料盘上。

#### `bearing_housing_c010`、`bearing_housing_c020`、`bearing_housing_c030` 轴承座（三档间隙）

- **外形**：110（Y）× 70（X）× 40（高）的铝块。
- **孔**：竖直贯通，轴线在块中心：
  - 上段孔径 47 + c，深 14（从顶面算起），c 分别是 0.10、0.20、0.30 mm；
  - 下段 Ø40 通到底，在 z=26 形成 3.5 mm 宽的台肩，让轴承坐住；
  - 孔口导向倒角 1×30°。
- **安装孔**：两个 Ø9 通孔加 Ø15 深 9 的沉孔，位于 y=±42、x=0。
- **材质**：铣削铝 #C4C9CE，表面有平行弧线铣刀纹，棱边去毛刺留亮边。
- **固定**：静止，固定在台面。
- **碰撞（精密，±0.01 mm）**：
  - 上段孔壁：48 块扇形凸体，每块是多边形孔壁的一段，内切圆半径正好是 (47 + c) / 2，外半径 31，高度 z 26 到 39；
  - 孔口倒角：48 块锥面扇形，z 39 到 40；
  - 下段孔壁和台肩：48 块扇形，内半径 20（同样按内切圆规则），外半径 31，z 0 到 26；
  - 外围：两侧 y 方向各 1 块盒子（|y| 31–55），前后各 1 块薄盒子（|x| 31–35）；
  - 总数不超过 150 块，这个道具单独放宽到 150。
- **代码要求**：脚本里写成参数化函数 `bearing_housing(clearance_mm)`，以后改间隙直接重新生成。
- `prop.json` 写 `"precision_mm": 0.01` 和 `"bore_diameter_mm": 47 + c`。

#### `bearing_6204` 深沟球轴承

- **外形**：外径 47、内径 20、宽 14。外圈和内圈之间两侧各有一圈橡胶密封（2RS），比端面凹进 0.5，颜色黑 #111111；变体橙 #E25B12。内外圈倒角 0.6。
- **标记**：侧面激光刻字 “NVK 6204-2RS”。
- **材质**：轴承钢 #A0A6AC，roughness 0.2。
- **质量 / 摩擦**：0.106 kg / 0.25（钢对铝）。
- **碰撞（精密）**：一个 48 边柱体，顶点正好落在 Ø47 圆上，宽 14，两端按 0.6 倒角做成两圈小斜面。内孔不做碰撞。

#### `dowel_pin` 定位销、`pin_fixture` 销孔工装（插装变体）

- **定位销**：Ø10 × 40 淬硬钢，两端 0.5×15° 导向倒角，其中一端是 R1 球头。#8F969D，roughness 0.25，质量 0.025 kg。碰撞：一个 32 边柱体，顶点落在 Ø10 圆上。
- **销孔工装**：发黑钢板 120（Y）× 80（X）× 20 #2E3135，两个孔径 10 + c 的通孔（c = 0.05、0.10、0.20，做 3 个变体），孔距 60，孔口倒角 0.5。碰撞：每个孔 32 块扇形，按内切圆规则，外围用盒子填充。

#### `din_rail_kit` DIN 导轨与端子排（可选）

- **导轨**：TS35 × 7.5 导轨，长 250，厚 1，长腰孔 25 × 5.3、间距 25，彩色钝化镀锌 #C9C08A。
- **端子排**：5.2（Y）× 47（X）× 42.5（高）螺钉式端子，灰 #9E9E9E；端盖和固定件各一个。底部的卡脚按真实形状做。
- **说明**：卡装需要仿真里额外的卡扣模型，所以这一项先只做几何，后面再定要不要用。

### 3.11 环境道具（可选，但对画面真实感影响很大）

#### `workbench` 钢制工作台

- **尺寸**：1200（Y）× 700（X）× 800（高）。台面 40 厚，上面铺 2 mm 灰绿色防静电台垫 #6B7B6E，带纹理；前沿是黑色 PVC 封边。
- **结构**：框架蓝黑 #3B3F44；有下层搁板和调平脚；远离机器人的一侧装一块 1200 × 600 的工具挂板，挂几把工具作背景。
- **碰撞**：台面 1 块，顶面在 800。

#### `factory_floor` 地面

- 6 × 6 m 环氧地坪，灰 #8C9196，有轻微光泽变化和拖痕；75 mm 宽黄色安全线和一段黄黑斜纹警示区。
- 贴图按 1 m 一格平铺；碰撞是一个平面。

#### `safety_fence` 安全围栏

- 面板 1000 × 1800，立柱是黄色 40 × 40 方管，下部 200 高是钣金踢脚板。
- 网格只能用真实细杆几何来做（MuJoCo 不认贴图透明），杆径 3、间距 50。
- 只作背景，不加碰撞。

#### `background_rack` 背景货架

- 横梁式货架：立柱蓝 #1E4C9A、横梁橙 #E86A10，高 2400、宽 1800、深 800，两层摆纸箱和周转箱（复用 3.5 和 3.1 的贴图）。
- 只作背景，不加碰撞。

## 4. Codex 提示词

先把整份文档交给 Codex，再按顺序发下面的提示词。提示词用英文写：Codex 对 bpy 和 Blender 的术语用英文更准确。

### 提示词 0：搭建流水线

```text
You are building simulation-ready 3D props for a MuJoCo robot benchmark, using Blender 5.2 LTS (/Applications/Blender.app/Contents/MacOS/Blender) driven by Python (bpy) in headless mode. Check the Blender 5.x API where it differs from 4.x (for example the OBJ exporter's parameter names). The repository root is the current directory. Read docs/benchmark_props.md in full; its §1 (conventions), §2 (robot workspace) and §3 (prop specs) are binding. All dimensions in the spec are millimetres; build in metres.

Task 0: set up the pipeline only. Build no benchmark props yet.
1. Locate Blender (macOS: /Applications/Blender.app/Contents/MacOS/Blender, otherwise `blender` on PATH). Every Blender script runs as `<blender> -b --factory-startup -P <script> -- <args>`.
2. Create tools/blender/prop_lib.py, imported by every build script, with:
   - reset_scene(): empty scene, metric units, unit scale 1.0, Cycles on CPU.
   - mm(x): millimetres to metres.
   - pbr_material(name, base_color_hex, roughness, metallic, wear=0, grime=0, machining=None): a Principled BSDF material. Optional procedural node groups: edge wear (lighter worn edges), grime (AO-driven darkening of cavities), and machining marks ("turned" = concentric, "milled" = parallel arcs, "brushed" = linear).
   - Primitives in mm with bevel width and segments: box, rounded box, cylinder, tube, hexagonal prism, extruded 2D profile, and polygonal ring segments (for bores; the polygon apothem equals the given radius).
   - uv_unwrap(obj, method): "smart" = Smart UV Project at 66 degrees with island margin 0.01, "cube" = Cube Projection; no overlapping islands.
   - bake_albedo(objects, size, out_png): bake DIFFUSE (colour only) and AO with Cycles at 64 samples into one image per part, combine as colour × mix(1, AO, 0.6), and save an 8-bit sRGB PNG. Never bake lighting or glossy.
   - export_obj(obj, path): bpy.ops.wm.obj_export with export_selected_objects=True, forward_axis='Y', up_axis='Z' (identity, so Blender coordinates equal MuJoCo coordinates), export_uv=True, export_normals=True, export_materials=False, apply_modifiers=True, export_triangulated_mesh=True.
   - export_collision(part_name, pieces, folder): each piece made convex with a convex hull, at most 64 vertices, written as <part>_col_NN.obj in the prop frame.
   - write_prop_json(data, path): checks the schema of §1.8 before writing.
   - render_preview(objects, out_png): 1024×1024 Cycles render at 64 samples, neutral grey background, three-point lighting, 3/4 view from front-left above, looking along +X like the robot.
   - make_label_png(path, width_px, height_px, fields): draws labels, signs and nameplates (text, a simple random barcode, coloured zone blocks). Use Pillow from the system Python if available; otherwise use Blender text objects baked to an image. Fictional brands only (§1.10).
3. Create tools/blender/validate_props.py, run with /Users/puheliang/miniforge3/envs/inovance-fineart/bin/python (it has numpy and trimesh). For every assets/props/<id>/prop.json, or only the ids given with --props, check:
   - every referenced file exists;
   - the visual bbox matches bbox_mm within 1 mm (0.1 mm when precision_mm is set);
   - each collision piece is convex (convex hull volume divided by piece volume is at least 0.99) and has at most 64 vertices;
   - textures are PNGs of the size given in texture_px;
   - triangle counts are within the §1.4 budget;
   - joint fields are complete per §1.7.
   Print one table row per prop and exit non-zero on any failure.
4. Prove the pipeline with a 100 mm test cube at assets/props/_test_cube/: one part, a 1024 texture with a checker pattern and the word TEST, and one collision piece. Then run the validator.
Only create files under tools/blender/ and assets/props/; do not modify anything else in the repository, and never run git commands that change its state (commit, push, checkout, reset, stash). Report the commands you ran and the validator output.
```

### 提示词 1：料箱间转移零件

```text
Task 1 (tote_transfer). Read §1, §2 and §3.1 of docs/benchmark_props.md. Write tools/blender/build_task01_tote_transfer.py using tools/blender/prop_lib.py and run it headless. It must produce assets/props/small_tote, assets/props/flange_part and assets/props/abs_housing exactly as specified.
Key points:
- small_tote: 300 (Y) × 200 (X) × 100 (Z) mm outside, 2° draft, 3 mm walls, 4 mm floor, 6 mm rim, vertical ribs, an 80 × 25 handle cut-out, a label slot with a "BIN-A07" card. Blue and grey are texture variants (body_albedo.png, body_albedo_v1.png). Collision is the floor plus 4 walls (plus the rim), never a single hull.
- flange_part: Ø60 × 12 flange with a Ø32 × 8 hub, a Ø20 bore, 4 × Ø6.6 holes on a 45 mm pitch circle, a 1 × 45° outer chamfer. Turned aluminium with concentric machining marks. 0.090 kg.
- abs_housing: 70 × 45 × 30 mm box with R4 corners, a parting line, 4 screw holes, 2 snap tabs and an indicator window. Textured dark-grey ABS. 0.040 kg.
Then run `/Users/puheliang/miniforge3/envs/inovance-fineart/bin/python tools/blender/validate_props.py --props small_tote flange_part abs_housing` and fix every failure. Report each prop's bbox, triangle count, collision piece count and texture sizes, and show the preview images. Only touch tools/blender/ and assets/props/.
```

### 提示词 2：设备控制面板操作

```text
Task 2 (panel_operation). Read §1, §2 and §3.2 of docs/benchmark_props.md. Write tools/blender/build_task02_panel_operation.py using prop_lib and run it headless. It must produce assets/props/panel_devices and assets/props/control_console.
Key points:
- panel_devices is a library of 22 mm devices, each exported as its own part in its own frame (origin at the mounting-face centre, pressing direction −Z): bezel, btn_green, btn_red, btn_yellow, selector_knob, estop, pilot_green, pilot_amber, pilot_red, and legend_plate with one texture per legend (START, STOP, AUTO, MAN, CYCLE START, FEED HOLD). Joint specs exactly as in the table: buttons slide 4 mm with a return spring, the selector hinges −45° to +45° with detents, the e-stop slides 5 mm and latches. Pilot lights carry emission 0.8.
- control_console: base 320 (Y) × 200 (X), back height 160, front height 60, sloped top facing −X, 1.5 mm sheet with R3 corners, light-grey RAL 7035 with a dark-grey 280 × 190 panel. Ø22.3 mounting holes for layouts A, B and C. Write each layout's device centres (panel coordinates u, v in mm and the resulting 3D pose in the console frame) into prop.json variants. Add an "IMC Controls · OP-03" nameplate.
Then run the validator for these two props, fix every failure, and report as in Task 1. Only touch tools/blender/ and assets/props/.
```

### 提示词 3：推物料托盘到工位

```text
Task 3 (tray_push). Read §1, §2 and §3.3 of docs/benchmark_props.md. Write tools/blender/build_task03_tray_push.py using prop_lib and run it headless. It must produce assets/props/esd_tray and assets/props/tray_rail_set.
Key points:
- esd_tray: 300 (Y) × 200 (X) × 50 mm, 3 mm walls and floor, R8 corners, 70 × 22 finger cut-outs on both short sides, black conductive PP with scuffs, a generic yellow ESD triangle and "ESD SAFE" print. Collision is the floor plus 4 walls. Friction against the rails is 0.25.
- tray_rail_set: two 20 × 20 T-slot aluminium profiles, 700 long along Y, 206 mm apart on the inside, two 246 mm cross bars, and at the −Y end a 40 × 40 × 30 stop block with a 5 mm black rubber bumper and yellow-black hazard tape.
Then run the validator, fix every failure, and report as in Task 1. Only touch tools/blender/ and assets/props/.
```

### 提示词 4：快递包裹分拣

```text
Task 4 (parcel_sorting). Read §1, §2 and §3.4 of docs/benchmark_props.md. Write tools/blender/build_task04_parcel_sorting.py using prop_lib and run it headless. It must produce assets/props/belt_conveyor, sort_bin, parcel_box_s, parcel_box_m and parcel_pouch.
Key points:
- belt_conveyor: the origin is on the floor at the footprint centre, flow from +Y to −Y. 800 long, Ø50 drums at 760 mm centres, a 150 mm wide × 3 mm green PVC belt (diamond pattern) with its top at z = 800, 40 × 40 aluminium side beams, stainless guide rails 30 mm above the belt with a 160 mm inner gap, 4 legs with levelling feet, a gear motor on the +X side of the −Y end, a photo-eye with a reflector, and a stainless end plate. The belt surface is its own part named "belt". At most 8 collision pieces.
- sort_bin: 200 (X) × 160 (Y) × 150 galvanised bin with a 30 mm entry lip sloped 30° toward −X, and a post with a 120 × 80 zone sign. Variants A (red), B (blue) and C (green) differ only in the sign texture.
- Parcels: parcel_box_s 140 × 75 × 60, parcel_box_m 180 × 80 × 80 (single-wall corrugated, tape, label), parcel_pouch 200 × 80 × 35 (puffy white PE mailer with heat seals). Each has label texture variants A, B and C made with make_label_png: fictional "QF Express" logo, a tracking number, a random barcode, a large coloured zone letter, fictional addresses. One box collider each.
Then run the validator, fix every failure, and report as in Task 1. Only touch tools/blender/ and assets/props/.
```

### 提示词 5：双臂搬纸箱

```text
Task 5 (carton_carry). Read §1, §2 and §3.5 of docs/benchmark_props.md. Write tools/blender/build_task05_carton_carry.py using prop_lib and run it headless. It must produce assets/props/carton_250x200x150, carton_300x200x200, carton_220x180x120 and roller_conveyor.
Key points:
- Cartons are FEFCO 0201 single-wall boxes, 4 mm walls; sizes are Y × X × height, and Y is the squeeze direction of the two arms. Each has a 1–2 mm flap gap, a 48 mm brown tape strip folding 50 mm down both ends, black generic handling icons (up arrows, umbrella, fragile glass), a fictional "NOVA Logistics · SKU-B0042" label, dented corners and visible flutes on the flap edges. Write mass variants 0.5, 1.0 and 1.5 kg in prop.json. One box collider each.
- roller_conveyor: the origin is at the bottom centre, flow toward −X. 500 long, 320 inside width, 10 galvanised Ø30 × 316 rollers at 50 mm pitch with their tops 50 mm above the table. Blue 40 × 20 × 2 C-channel side frames whose top sits 10 mm below the roller tops, so the hands can squeeze a carton from both sides. 4 L-feet. A removable end_stop bar at −X, 20 mm above the rollers. Each roller is its own part (roller_00…roller_09) with a free hinge about Y (damping 0.001) and a 16-sided cylinder collider.
Then run the validator, fix every failure, and report as in Task 1. Only touch tools/blender/ and assets/props/.
```

### 提示词 6：零件配盘

```text
Task 6 (shaped_kitting). Read §1, §2 and §3.6 of docs/benchmark_props.md. Write tools/blender/build_task06_shaped_kitting.py using prop_lib and run it headless. It must produce assets/props/foam_kit_tray, hex_blank, flanged_bushing, l_bracket and spur_gear.
Key points:
- Build the parts first, then cut the foam pockets from their top-view outlines offset by 1.5 mm per side, at the positions and depths in §3.6.
  - The bushing pocket is a Ø43 × 5 counterbore over a Ø33 × 25 hole, so it only seats flange-up.
  - The gear pocket is Ø51 × 10, hub up.
  - The L-bracket pocket is the side-lying L profile, 63 × 43 with a 7 mm leg, 25 deep.
  - The hex pocket is 30 mm across flats, 25 deep.
  - Each pocket has a Ø24 half-circle finger notch on its −X side.
- Foam collision is split around the pockets: 6 walls for the hexagon, 16 wall segments per circular step, boxes for the L shape, and boxes filling between pockets. At most 64 pieces.
- spur_gear must use a true involute profile: module 1.5, 30 teeth, 12 mm face width, Ø24 × 6 hub, Ø8 bore with a 3 mm keyway. Generate the involute in Python, or use the Add Mesh: Extra Objects gear with matching parameters.
- Materials and masses are as in the table (blackened steel, bronze, galvanised steel with spangle, white POM with a steel variant).
Then run the validator, fix every failure, and report as in Task 1. Only touch tools/blender/ and assets/props/.
```

### 提示词 7：开盖取件

```text
Task 7 (lidded_tote). Read §1, §2 and §3.7 of docs/benchmark_props.md. Write tools/blender/build_task07_lidded_tote.py using prop_lib and run it headless. It must produce assets/props/lidded_tote, station_plate and drawer_cabinet.
Key points:
- lidded_tote: body 300 (Y) × 200 (X) × 150 with 2° draft. Lid 304 × 204 × 15 with a 12 mm skirt, two 40 mm hinges at the +X top edge (axis along Y through x = 102, z = 152), a 60 × 18 × 3 pull tab at −X and a snap catch. Export the lid as its own part, closed. Hinge joint 0–110°, damping 0.05, frictionloss 0.4 N·m (the lid stays where it is left). The catch's 0.3 N·m return torque within 3° goes in notes. A fictional "PARTS · LINE 3" label.
- station_plate: 200 × 150 × 10 aluminium plate, countersunk screws, a yellow silk-screened part outline, a centre cross and "ST-02".
- drawer_cabinet (optional variant): 300 × 250 × 200 steel cabinet, two drawers sliding 0–180 mm toward −X, arched handles and label slots. Each drawer is its own part with a slide joint.
Then run the validator, fix every failure, and report as in Task 1. Only touch tools/blender/ and assets/props/.
```

### 提示词 8：小型码垛

```text
Task 8 (palletizing). Read §1, §2 and §3.8 of docs/benchmark_props.md. Write tools/blender/build_task08_palletizing.py using prop_lib and run it headless. It must produce assets/props/mini_pallet, scissor_lift_table, pallet_carton_160x80x80, pallet_carton_120x80x80 and pattern_sign. The infeed conveyor reuses roller_conveyor from Task 5; do not rebuild it.
Key points:
- mini_pallet: 400 (Y) × 300 (X) × 80. Five 300 × 64 × 12 top boards along Y with 20 mm gaps, nine 60 × 60 × 56 blocks on a 3 × 3 grid (x −120/0/+120, y −170/0/+170), three 64 × 400 × 12 bottom boards. Pine with grain, nail heads, dirt and a fictional fumigation stamp. A blue HDPE grid-top variant. Collision is the 5 top boards plus one box for the rest.
- scissor_lift_table: 500 × 400 × 40 top at z = 720 (pallet top at 800), 520 × 420 × 60 base, two black scissor pairs, yellow steel with hazard stripes. Adjustable range 620–760 in prop.json.
- Pallet cartons are the §3.5 carton style at 160 × 80 × 80 (0.35 kg, "SKU-A") and 120 × 80 × 80 (0.25 kg, "SKU-B"), with an orientation arrow on top.
- pattern_sign: a 210 × 297 × 3 clear acrylic part (rgba, no texture) on a 220 × 60 × 20 base, holding a stacking-plan sheet. Three plan variants (layer 1 is 2 × 2, layer 2 rotated 90°, numbered boxes, fictional "NOVA Logistics" footer).
Then run the validator, fix every failure, and report as in Task 1. Only touch tools/blender/ and assets/props/.
```

### 提示词 9：数控机床上下料

```text
Task 9 (cnc_tending). Read §1, §2 and §3.9 of docs/benchmark_props.md. Write tools/blender/build_task09_cnc_tending.py using prop_lib and run it headless. It must produce assets/props/cnc_front, part_tray_6, round_blank and machined_part. Panel buttons reuse assets/props/panel_devices from Task 2 (instance its parts; do not rebuild them).
Key points:
- cnc_front is the front section of a compact machining centre. The origin is on the floor at the front-face centre, x = 0 is the front face, +X points into the machine. 900 × 1600 × 700 enclosure; the opening spans y −240…+240 and z 800…1220.
- Separate parts with joints:
  - door: 500 × 440 × 25 frame, front face at x = −30; slides along +Y over 0–0.46 m, damping 8, frictionloss 3 N, 4 kg. Its window is the transparent part door_window (rgba 0.85, 0.90, 0.95, 0.25). Handle: Ø20 × 180 vertical bar at y = −220, z 900–1080, 30 mm proud of the door.
  - vise_jaw: slides along +X over 0–0.02 m (0 = open, jaw face at x = 60; 0.02 = clamped on Ø40, face at x = 80).
- Static parts: T-slot table top at z = 800; vise body x 30–170, y ±45, z 800–850; fixed jaw face at x = 120; serrated jaw plates z 850–880; parallels lifting the part bottom to z = 860; spindle head behind and above the vise with the end-mill tip at z = 1150.
- Operation panel at y −440…−290, z 900–1250, 40 mm proud: CYCLE START (green), FEED HOLD (yellow), e-stop, a mode selector (press direction +X) and a 120 × 90 screen showing a fictional "IMC-CNC 850" UI.
- Look: light-grey and dark-grey sheet metal, a blue accent band with a fictional "IMC" mark, generic warning labels, aluminium chips and coolant stains inside. At most 150k triangles; split into several 2048 textures as needed.
- part_tray_6: 250 × 170 × 30 brushed aluminium tray with 6 × Ø44 pockets 20 deep (2 × 3 grid, x ±35, y −75/0/+75), engraved "IN" and "OUT" variants. Pockets are 8-wall octagons (apothem 22) in collision.
- round_blank: Ø40 × 60 saw-cut, mill-finish bar, 0.204 kg. machined_part: the same bar with a Ø30 × 20 turned step, a Ø12 × 15 hole and 1 × 45° chamfers, bright turned finish, 0.170 kg.
Then run the validator, fix every failure, and report as in Task 1. Only touch tools/blender/ and assets/props/.
```

### 提示词 10：精密插装

```text
Task 10 (precision_insertion). Read §1, §2 and §3.10 of docs/benchmark_props.md. Write tools/blender/build_task10_precision_insertion.py using prop_lib and run it headless. It must produce assets/props/bearing_housing_c010, bearing_housing_c020, bearing_housing_c030, bearing_6204, dowel_pin and pin_fixture, plus din_rail_kit as geometry only.
This task is precision-critical:
- Implement bearing_housing(clearance_mm) as a parametric function. Block 110 (Y) × 70 (X) × 40. Upper bore Ø(47 + c) × 14 deep from the top, lower bore Ø40 through (a 3.5 mm seat at z = 26), 1 × 30° entry chamfer, two Ø9 holes with Ø15 × 9 counterbores at y = ±42.
- Housing collision uses 48 radial convex segments per bore band. The polygon apothem (centre-to-face distance) equals the bore radius exactly, outer radius 31. Bands: upper wall z 26–39, chamfer z 39–40, lower wall plus seat z 0–26 with apothem 20. Fill the rest with 4 boxes. At most 150 pieces. Bore dimensions must be exact to 0.01 mm; put "precision_mm": 0.01 and "bore_diameter_mm" in prop.json.
- bearing_6204: Ø47 / Ø20 / 14, 2RS seals recessed 0.5 mm, 0.6 mm chamfers, "NVK 6204-2RS" marking, 0.106 kg. Collision is one 48-sided prism with its vertices on the Ø47 circle (never larger than nominal), with the chamfers as small bevel rings.
- dowel_pin: Ø10 × 40, chamfers and an R1 ball end, collision a 32-sided prism with vertices on Ø10. pin_fixture: 120 × 80 × 20 blackened plate with two Ø(10 + c) through holes at 60 mm pitch, variants c = 0.05, 0.10 and 0.20, 32-segment hole collision with the apothem rule.
- din_rail_kit: TS35 × 7.5 rail 250 long with 25 × 5.3 slots, generic screw terminal blocks 5.2 × 47 × 42.5 with a realistic clip foot, an end cover and an end clamp. Geometry and textures only; mark in notes that clip-on behaviour is not modelled.
After running the validator, also write a small check that loads each housing's collision pieces with trimesh and measures the minimum distance from the bore axis to any piece face; it must equal (47 + c) / 2 within 0.01 mm. Report it with the usual table. Only touch tools/blender/ and assets/props/.
```

### 提示词 11：环境道具（可选）

```text
Task 11 (environment, optional). Read §1 and §3.11 of docs/benchmark_props.md. Write tools/blender/build_task11_environment.py using prop_lib and run it headless. It must produce assets/props/workbench, factory_floor, safety_fence and background_rack.
Key points:
- workbench: 1200 × 700 × 800 steel bench, grey-green ESD mat on a 40 mm top, black PVC front edge, lower shelf, levelling feet, and a 1200 × 600 pegboard with tools on the +X side. One collision box with its top at z = 800.
- factory_floor: a 6 × 6 m epoxy floor that tiles every 1 m, with gloss variation, scuffs, a 75 mm yellow safety line and a yellow-black hatched zone. Collision is a single plane.
- safety_fence: 1000 × 1800 panels on yellow 40 × 40 posts with a 200 mm sheet-metal kick plate. Build the mesh as real 3 mm rods at 50 mm pitch, because MuJoCo ignores texture alpha. No collision.
- background_rack: a 2400 × 1800 × 800 beam rack with blue uprights and orange beams, holding cartons and totes that reuse the Task 4 and Task 1 textures. No collision.
Then run the validator, fix every failure, and report as in Task 1. Only touch tools/blender/ and assets/props/.
```

### 提示词 12：浅色泡棉变体（2026-10-24 追加）

```text
Task 12 (foam_kit_tray light variant). Read §1 and §3.6 of docs/benchmark_props.md and the existing tools/blender/build_task06_shaped_kitting.py. The black EVA foam (#161616) hides its pockets in the robot's 640 × 480 cameras, which see the tray from 0.5–0.7 m. Add a light-grey foam variant to assets/props/foam_kit_tray without changing any geometry or collision:
- Foam base colour #A9AFB6 (light-grey EVA), keeping the fine foam-cell texture. Pocket walls and floors read clearly darker: AO strength 0.8 inside the pockets, plus a subtle darker floor tint, so every pocket outline and the finger notches are obvious.
- Keep the black foam as it is. Add prop.json variants "black" (the current textures) and "light_grey" (the new foam texture), each naming its textures per part. The top-level parts keep the black textures.
- Render tools/blender/_qa/foam_kit_tray_light_top.png: a 640 × 480 view from 0.6 m above the tray, tilted about 30°, with the four parts beside it, to show the pockets are visible. Also re-render preview.png with the light variant.
- Change build_task06_shaped_kitting.py so a rebuild reproduces both variants.
Run the validator for foam_kit_tray and report.
```

### 提示词 13：压缩道具体积（2026-10-24 追加）

```text
Task 13 (shrink assets/props for the public repository). assets/props is about 350 MB: textures 212 MB (83 baked albedos at 2048 px make 125 MB, previews about 65 MB), OBJ 65 MB, source.blend 52 MB and .blend1 backups 12 MB. tools/blender/_qa is 81 MB. Target: assets/props without the source.blend files at or below 90 MB, with nothing visibly lost in the robot's 640 × 480 cameras.
Do, in this order:
1. Delete every *.blend1 backup under assets/props and tools/blender, and stop the build scripts from writing them (save with no versions, e.g. bpy.context.preferences.filepaths.save_version = 0).
2. Downscale every baked albedo larger than 1024 px to 1024 px with Lanczos filtering, then update texture_px in prop.json. Text that must stay legible (parcel and carton labels, zone signs, legend plates, the CNC screen, the stacking plan, the station plate print) is still readable at 1024 at camera distance; check two of them in a close render. Keep the source artwork PNGs and the factory floor tile.
3. Deduplicate byte-identical textures: inside a prop, point every part at one shared file (for example the ten conveyor rollers) and delete the copies. Across props, a relative path such as ../parcel_box_m/body_albedo.png is allowed; the loader and the validator resolve paths from the prop folder.
4. Re-save all PNGs optimised: Pillow optimize=True, 8-bit RGB without alpha where alpha is unused. Use lossy palette quantisation (256 colours, Floyd–Steinberg) only where a side-by-side close render shows no visible banding.
5. Shrink previews (preview*.png) to 512 × 512 and optimise them.
6. Rewrite OBJ files with 5 decimal places for vertex positions and 4 for normals and UVs, leaving the geometry otherwise identical (max vertex change 0.005 mm). Never touch collision geometry beyond that rounding.
7. Make every build script produce the new sizes, so a rebuild matches.
Then run the validator on all props, make sure every prop.json path still resolves, and report a table of each prop's size before and after, plus the totals for textures, OBJ, previews and source.blend. Only touch tools/blender/ and assets/props/.
```

## 5. 道具交付以后

仿真这边接手后会做这些事：

1. 按任务新建 `scenes/<任务>/`：`task.py` 加载道具、摆放、设关节；`run.py` 写脚本专家；`check.py` 写验收。
2. 用 `prop.json` 里的质量、摩擦和关节参数建 MuJoCo 模型，检查碰撞块是否稳定，有没有穿模和抖动。
3. 精密插装要另外调接触参数：更小的步长、碰撞余量和 solref；必要时加腕部柔顺控制。真机夹爪力标定之前，这个任务的成绩只能作参考。
4. 每个任务定好随机化范围和固定的评测种子，跑通专家，再生成训练数据。
