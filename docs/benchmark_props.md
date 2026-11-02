# 工业 benchmark 道具建模规格（Codex + Blender）

这份文档给 Codex 用：Codex 写 Blender Python（bpy）脚本，在无界面模式下建模，导出 MuJoCo 能直接加载的道具。

用法：

1. 先把整份文档交给 Codex。
2. 按顺序发第 4 节的提示词：先发提示词 0 搭好流水线，再发 1–10（每个任务一条），环境道具（提示词 11）可选。
3. 2026-10-25 重设计的道具见第 6 节，提示词 14–17 可以同时发给四个 Codex。
4. 文档里的尺寸单位都是毫米，建模时换算成米。

道具做好后，由仿真这边接进 `scenes/<任务>/`，再调碰撞、关节、质量，写脚本专家和验收。

## 0. 任务与道具一览

| # | 任务 | 难度 | 道具（编号 = 输出文件夹名） |
|---|---|---|---|
| 1 | 料箱间转移零件 `tote_transfer` | 简单 | `small_tote`、`flange_part`、`abs_housing` |
| 2 | 设备控制面板操作 `panel_operation` | 简单 | `control_console`、`panel_devices` |
| 3 | 推物料托盘到工位 `tray_push` | 简单 | `esd_tray`、`tray_rail_set`（托盘里的零件复用任务 1） |
| 4 | 快递包裹分拣 `parcel_sorting` | 中等 | `belt_conveyor`、`sort_bin`、`parcel_box_s`（`parcel_box_m`、`parcel_pouch` 已建好，场景不再使用） |
| 5 | 双臂搬周转箱 `tote_carry`（原 `carton_carry`） | 中等 | `small_tote`（复用任务 1）、`roller_conveyor`（三种纸箱已建好，场景不再使用） |
| 6 | 零件配盘 `shaped_kitting` | 中等 | `foam_kit_tray`、`hex_blank`、`flanged_bushing`、`l_bracket`、`spur_gear` |
| 7 | 开盖取件 `lidded_tote` | 中等 | `lidded_tote`、`station_plate`、`drawer_cabinet`（可选变体） |
| 8 | 小型码垛 `palletizing` | 困难 | `mini_pallet`、`scissor_lift_table`、`pallet_carton_160x80x80`、`pallet_carton_120x80x80`、`pattern_sign`（供料复用 `roller_conveyor`） |
| 9 | 数控机床上下料 `cnc_tending` | 困难 | `cnc_front`、`part_tray_6`、`round_blank`、`machined_part`（按钮复用 `panel_devices`） |
| 10 | 精密插装 `precision_insertion` | 困难 | `bearing_shaft_fixture`、`bearing_6204`、`dowel_pin`、`pin_fixture`（三档轴承座和 `din_rail_kit` 已建好，场景不再使用） |
| 环境 | 可选 | — | `workbench`、`factory_floor`、`safety_fence`、`background_rack` |

码垛用的是单手能抓的小纸箱（宽 80 mm）：双臂对挤只能搬左右居中的箱子，码 2×2 垛型时两侧的箱子够不着。

2026-10-25 按仿真里量出的可达范围和夹爪尺寸改了一轮（工作台抬高、凹槽和孔的间隙放大、格口和垛型重做、轴承改为套轴），下面各条注明了改动和原因。

## 1. 通用规范（所有道具必须遵守）

### 1.1 软件与运行

- 本机是 Blender 5.2 LTS，路径 `/Applications/Blender.app/Contents/MacOS/Blender`，脚本按 Blender 5.x 的 API 写。统一无界面运行：`<blender> -b --factory-startup -P <脚本> -- <参数>`。
- 烘焙用 Cycles（CPU 即可，64 采样）。
- 脚本放在 `tools/blender/`：公用函数 `prop_lib.py`，每个任务一个 `build_taskNN_<任务名>.py`，检查脚本 `validate_props.py`。检查脚本和画贴图的子进程用环境变量 `PROP_PYTHON` 指定的 Python 运行，它要装 numpy、scipy、trimesh 和 Pillow（比如项目环境）。
- Linux 上用 PATH 里的 `blender`。
- 带字的贴图用 `tools/blender/fonts/` 里的 DejaVu Sans（常规、粗体），只写英文（这个字体没有中文字形）。这样同一个道具在 Mac 和 Linux 上建出来一样。2026-10-25 以前的道具用的是 macOS 字体，只能在 Mac 上重建。
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
| 工作台面高度 | 900 mm（2026-10-25 起；机器人不能弯腰，台面 800 mm 时竖直向下抓只够到 x 370–410 mm） |
| 手能够到的范围 | 台面 900 mm 时竖直向下抓 x ≤ 450–480 mm，45° 前倾抓 x ≤ 570–610 mm；左手 y 30–450 mm，右手 y −450 到 −30 mm；高度在台面以上 0–200 mm 最好 |
| 头部相机 | 从 x≈310–330 mm 开始能看到台面（头部最多低头 30°） |
| 夹爪 | 两指平行，最大开口约 95 mm，被抓尺寸建议 20–85 mm；夹爪壳体沿开合方向伸出约 ±85 mm，45° 抓时在抓取点后方约 65 mm；指尖比夹持中心长 13.4 mm（新模型实测；原写 17 mm），跟不进孔里 |
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

建议摆放：皮带线沿 Y 放在前方，皮带面高 900、中心 x≈525；四个格口箱放在皮带前方 x≈300，每只手各管自己一侧的两个（2026-10-25 起；原来三个格口共用中间的 B，两只手都够不到它的中心）。

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

#### `sort_bin` 分拣格口箱（做 4 个变体 A、B、C、D）

- **外形**：200（X）× 240（Y）× 100（高），1 mm 镀锌钢板。朝机器人的一侧（−X）有 30 mm 宽的导入斜唇，向外倾斜 30°。2026-10-25 之前是 200 × 160 × 150：夹爪壳体放不进去，腕部相机块也过不了箱沿。
- **分区牌**：后侧（+X）一角立一根 Ø20 立柱，高出箱底 190，装一块 120×80 的分区牌，中心高出箱底 150（原来 300，挡在包裹的搬运路线上）：A 红 #D32F2F、B 蓝 #1976D2、C 绿 #388E3C、D 紫 #7B1FA2，上面一个白色大字母。
- **材质**：镀锌钢 #AEB4BA，有锌花纹理。
- **质量**：0.8 kg（静止）。
- **碰撞**：底板、四面墙、斜唇，共 6 块。
- **变体**：A、B、C、D 只有分区牌贴图不同。

#### 包裹 `parcel_box_s`、`parcel_box_m`、`parcel_pouch`

| 编号 | 尺寸（Y×X×高） | 形态 | 质量范围 | 摩擦 |
|---|---|---|---|---|
| `parcel_box_s` | 140 × 75 × 60 | 单瓦楞纸盒，壁厚 3，盖缝 + 48 mm 透明胶带 | 0.15–0.35 kg | 0.60 |
| `parcel_box_m` | 180 × 80 × 80 | 同上 | 0.20–0.50 kg | 0.60 |
| `parcel_pouch` | 200 × 80 × 35 | 白色 PE 快递袋外形：鼓起的圆角块，边缘有热封边 | 0.10–0.30 kg | 0.50 |

- **面单**：贴在顶面，100×60（快递袋上 90×60），做 A、B、C 三个贴图变体（`parcel_box_s` 另有 D）。每张面单包含：
  - 虚构快递 “QF Express” 的标志；
  - 运单号，例如 “QF2026102400017”；
  - 一组条码（随机黑白条）；
  - 左上角一块大的分区色块和字母：A 红、B 蓝、C 绿、D 紫，和格口箱对应；
  - 虚构的收寄件信息。
- **碰撞**：每个包裹一个盒子，快递袋用 196×76×33 的盒子。

### 3.5 任务 5：双臂搬周转箱 `tote_carry`（原 `carton_carry`）

2026-10-25 起，两只手各夹周转箱（复用任务 1 的 `small_tote`）一端的箱壁上沿，把它放到身后的辊道上；辊道转 90° 沿 Y 摆，垫高 50，装挡杆。原来是双臂从左右对挤 250 mm 宽的纸箱：95 mm 的夹爪夹不住，只能用腕部外壳去挤。下面的纸箱规格保留，场景不再使用。

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
- **凹槽**：2×2 排列，槽形按零件俯视轮廓每边外扩 4 mm（2026-10-25 起，原为 1.5 mm：L 形支架只能偏 2°，靠相机的机械手对不准）。每个槽朝机器人一侧（−X）有一个 Ø24 的半圆指扣槽，深度和该槽相同。槽中心（相对内衬中心，单位 mm，(Y, X)）：
  - 六角坯 (−90, +55)：正六边形，对边 35，深 25；
  - 法兰轴套 (+90, +55)：上段 Ø48 深 5，下面接 Ø38 深 25，总深 30。Ø40 的法兰进不了 Ø38 的下段，所以只有法兰朝上才能放到底。
  - L 形支架 (−90, −55)：支架侧放时的 L 形轮廓，外形 68 × 48、腿厚 12（内角的 R4 圆角外扩后成了直角），深 25；
  - 齿轮 (+90, −55)：Ø56 深 10，轮毂朝上放。
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

建议摆放：升降台和托盘放在 x≈400、y≈0，升降台垫高 100，托盘顶面高 900；供料放在两侧的支架上（顶面 900）。纸箱单手抓：长边沿 X 摆，45° 前倾抓 80 mm 那一边，抓取点在箱底以上 65 mm（再低手掌会压到箱顶），每个箱子由离它近的那只手抓。

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
- 做 4 种垛型，即 4 张贴图变体。`plan_04`（2026-10-25）是给机器人的：每层被夹的 80 mm 那两面之间留 40 mm 给手指，其余的面贴紧。
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

建议摆放：轴座和销孔工装固定在台面 x≈370；轴承平放在一侧。2026-10-25 起轴承改为套到轴上：指尖比夹持中心长 13.4 mm（原写 17 mm），跟不进轴承座的孔，轴承只能悬空落进去，落差超过 10 mm 就会触发 1 mm 的穿透审计。下面的三档轴承座保留，场景不再使用。

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
- **碰撞（精密）**：24 块扇形凸体围成的环，内切圆半径正好是 10（Ø20 内孔留空，才能套到轴上），外顶点落在 Ø47 圆上，宽 14，不做两端倒角。2026-10-25 之前是一个填满内孔的 48 边柱体。

#### `bearing_shaft_fixture` 轴承装配轴座（2026-10-25 新增）

- **外形**：110（Y）× 70（X）× 20 的铝底座，两个 Ø9 通孔加 Ø15 沉孔（y=±42）；中心一根磨光钢轴：Ø24 × 14 的轴肩（顶面高 34，轴承内圈坐在上面），上面是 Ø(20 − c) 的轴，高出轴肩 20，顶端 1×30° 导向倒角。c = 1.0 mm（直径间隙）。
- **用法**：手夹着轴承外圈往下套，指尖始终在外圈外面、底座上方 10 mm 以上，轴承落到轴肩上再松手。
- **碰撞**：底座 1 块盒子，轴肩、轴各一个 32 边柱体（顶点在圆上），导向倒角一块 32 边锥台，共 4 块。

#### `dowel_pin` 定位销、`pin_fixture` 销孔工装（插装变体）

- **定位销**：Ø10 × 40 淬硬钢，两端 0.5×15° 导向倒角，其中一端是 R1 球头。#8F969D，roughness 0.25，质量 0.025 kg。碰撞：一个 32 边柱体，顶点落在 Ø10 圆上。
- **销孔工装**：发黑钢板 120（Y）× 80（X）× 20 #2E3135，两个孔径 10 + c 的通孔（c = 0.05、0.10、0.20、1.00，做 4 个变体；场景用 1.00，0.05–0.20 靠位置控制和相机对不准），孔距 60，孔口倒角 0.5。碰撞：每个孔 32 块扇形，按内切圆规则，外围用盒子填充。

#### `din_rail_kit` DIN 导轨与端子排（可选）

- **导轨**：TS35 × 7.5 导轨，长 250，厚 1，长腰孔 25 × 5.3、间距 25，彩色钝化镀锌 #C9C08A。
- **端子排**：5.2（Y）× 47（X）× 42.5（高）螺钉式端子，灰 #9E9E9E；端盖和固定件各一个。底部的卡脚按真实形状做。
- **说明**：卡装需要仿真里额外的卡扣模型，所以这一项先只做几何，后面再定要不要用。

### 3.11 环境道具（可选，但对画面真实感影响很大）

#### `workbench` 钢制工作台

- **尺寸**：1200（Y）× 700（X）× 900（高；2026-10-25 之前是 800）。台面 40 厚，上面铺 2 mm 灰绿色防静电台垫 #6B7B6E，带纹理；前沿是黑色 PVC 封边。
- **结构**：框架蓝黑 #3B3F44；有下层搁板和调平脚；远离机器人的一侧装一块 1200 × 600 的工具挂板，挂几把工具作背景。
- **碰撞**：台面 1 块，顶面在 900。

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
3. Create tools/blender/validate_props.py, run with $PROP_PYTHON (a Python with numpy, scipy, trimesh and Pillow). For every assets/props/<id>/prop.json, or only the ids given with --props, check:
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
Then run `$PROP_PYTHON tools/blender/validate_props.py --props small_tote flange_part abs_housing` and fix every failure. Report each prop's bbox, triangle count, collision piece count and texture sizes, and show the preview images. Only touch tools/blender/ and assets/props/.
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
- sort_bin: 200 (X) × 240 (Y) × 100 galvanised bin with a 30 mm entry lip sloped 30° toward −X, and a post with a 120 × 80 zone sign centred 150 mm above the bin base. Variants A (red), B (blue), C (green) and D (purple) differ only in the sign texture.
- Parcels: parcel_box_s 140 × 75 × 60, parcel_box_m 180 × 80 × 80 (single-wall corrugated, tape, label), parcel_pouch 200 × 80 × 35 (puffy white PE mailer with heat seals). Each has label texture variants A, B and C (parcel_box_s also D) made with make_label_png: fictional "QF Express" logo, a tracking number, a random barcode, a large coloured zone letter, fictional addresses. One box collider each.
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
- Build the parts first, then cut the foam pockets from their top-view outlines offset by 4 mm per side (CLEAR in the script), at the positions and depths in §3.6.
  - The bushing pocket is a Ø48 × 5 counterbore over a Ø38 × 25 hole, so it only seats flange-up.
  - The gear pocket is Ø56 × 10, hub up.
  - The L-bracket pocket is the side-lying L profile, 68 × 48 with a 12 mm leg, 25 deep.
  - The hex pocket is 35 mm across flats, 25 deep.
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
- pattern_sign: a 210 × 297 × 3 clear acrylic part (rgba, no texture) on a 220 × 60 × 20 base, holding a stacking-plan sheet. Four plan variants (layer 1 is 2 × 2, layer 2 rotated 90°, numbered boxes, fictional "NOVA Logistics" footer); plan_04 leaves 40 mm finger gaps between the gripped 80 mm faces.
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
Task 10 (precision_insertion). Read §1, §2 and §3.10 of docs/benchmark_props.md. Write tools/blender/build_task10_precision_insertion.py using prop_lib and run it headless. It must produce assets/props/bearing_housing_c010, bearing_housing_c020, bearing_housing_c030, bearing_6204, bearing_shaft_fixture, dowel_pin and pin_fixture, plus din_rail_kit as geometry only.
This task is precision-critical:
- Implement bearing_housing(clearance_mm) as a parametric function. Block 110 (Y) × 70 (X) × 40. Upper bore Ø(47 + c) × 14 deep from the top, lower bore Ø40 through (a 3.5 mm seat at z = 26), 1 × 30° entry chamfer, two Ø9 holes with Ø15 × 9 counterbores at y = ±42.
- Housing collision uses 48 radial convex segments per bore band. The polygon apothem (centre-to-face distance) equals the bore radius exactly, outer radius 31. Bands: upper wall z 26–39, chamfer z 39–40, lower wall plus seat z 0–26 with apothem 20. Fill the rest with 4 boxes. At most 150 pieces. Bore dimensions must be exact to 0.01 mm; put "precision_mm": 0.01 and "bore_diameter_mm" in prop.json.
- bearing_6204: Ø47 / Ø20 / 14, 2RS seals recessed 0.5 mm, 0.6 mm chamfers, "NVK 6204-2RS" marking, 0.106 kg. Collision is a ring of 24 convex sectors, inner flats exactly at the R10 bore (left open, so the bearing slides onto a shaft) and outer vertices on the Ø47 circle.
- bearing_shaft_fixture: 110 (Y) × 70 (X) × 20 aluminium base with two Ø9/Ø15 counterbored holes at y = ±42, a Ø24 × 14 steel shoulder and a Ø(20 − c) shaft rising 20 mm above it with a 1 × 30° lead-in, c = 1.0. Collision: the base box, 32-sided shoulder and shaft prisms and a 32-sided lead-in frustum.
- dowel_pin: Ø10 × 40, chamfers and an R1 ball end, collision a 32-sided prism with vertices on Ø10. pin_fixture: 120 × 80 × 20 blackened plate with two Ø(10 + c) through holes at 60 mm pitch, variants c = 0.05, 0.10, 0.20 and 1.00, 32-segment hole collision with the apothem rule.
- din_rail_kit: TS35 × 7.5 rail 250 long with 25 × 5.3 slots, generic screw terminal blocks 5.2 × 47 × 42.5 with a realistic clip foot, an end cover and an end clamp. Geometry and textures only; mark in notes that clip-on behaviour is not modelled.
After running the validator, also write a small check that loads each housing's collision pieces with trimesh and measures the minimum distance from the bore axis to any piece face; it must equal (47 + c) / 2 within 0.01 mm. Report it with the usual table. Only touch tools/blender/ and assets/props/.
```

### 提示词 11：环境道具（可选）

```text
Task 11 (environment, optional). Read §1 and §3.11 of docs/benchmark_props.md. Write tools/blender/build_task11_environment.py using prop_lib and run it headless. It must produce assets/props/workbench, factory_floor, safety_fence and background_rack.
Key points:
- workbench: 1200 × 700 × 900 steel bench (BENCH_TOP in the script), grey-green ESD mat on a 40 mm top, black PVC front edge, lower shelf, levelling feet, and a 1200 × 600 pegboard with tools on the +X side. One collision box with its top at z = 900.
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

## 6. 2026-10-25 重设计的道具

用在 [`benchmark_redesign.md`](benchmark_redesign.md) 的 11 个新任务。第 1、2 节照样适用，下面只写这批道具新增的规矩和每个道具的规格。按任务分四组，每组一个构建脚本、一条提示词（6.5 节），四组可以同时建：

| 组 | 构建脚本 | 道具 |
|---|---|---|
| A 物流 | `build_task14_logistics.py`，加改 `build_task01_tote_transfer.py` | `tote_cart`、`line_feed_conveyor`、`takeaway_conveyor`、`induction_conveyor`、`flow_lane`、`recheck_shelf`、`station_sign`；`small_tote` 加三种颜色 |
| A2 周转箱 | `build_task18_handle_tote.py` | `handle_tote`（实验后追加，提示词 18） |
| B 电子装配 | `build_task15_electronics.py` | `plc_module`、`module_tray`、`test_fixture`、`ng_bin`、`product_box`、`accessory_box`、`stack_light`、`line_console` |
| C 拣料、配盘和装配 | `build_task16_assembly.py`，加改 `build_task06_shaped_kitting.py` | `parts_bin`、`relay`、`terminal_block`、`cable_gland`、`fuse_holder`、`order_tray`、`order_card_stand`、`rotor_shaft_fixture`、`bearing_tray`；重建 `foam_kit_tray` |
| D 包裹和码垛 | `build_task17_parcels.py`，加改 `build_task08_palletizing.py` | `parcel_cube_80`、`parcel_cube_75`、`parcel_cube_85`、`infeed_table`；`pallet_carton_160x80x80` 加 IMC 贴图 |

### 6.0 这批道具的通用规矩

- **运行**：`blender -b --factory-startup -t 6 -P <脚本> -- <参数>`。四组同时在一台机器上建，每个 Blender 限 6 个线程。
- **字**：只用 `tools/blender/fonts/` 里的 DejaVu Sans 和 DejaVu Sans Bold，只写英文。
- **品牌**：电气件和产线设备用 “IMC Controls”，快递用 “QF Express”，物流设备用 “NOVA Logistics”（1.10 节）。
- **`prop.json`**：
  - 新道具都写 `budget_class`；
  - `dimensions_mm` 写关键尺寸；
  - 固定不动的道具写 `"static": true`；
  - `recommended_placement_m` 抄下面各条的“摆放”。
- **按钮**：
  - 按钮帽单独一个部件，带 slide 关节，沿按压方向，加弹簧回位；
  - 帽子用 Ø40 的蘑菇头，夹爪合拢后用指尖往下按，小按钮按不准；
  - 在 `prop.json` 的 `buttons` 里写每个按钮的部件名、按压方向和行程。
- **指示灯**：
  - 灯罩单独一个部件，不贴图，`rgba` 写灭灯时的暗色；
  - 在 `prop.json` 的 `lamps` 里写每个灯罩的部件名、亮灯颜色 `on_rgba` 和 `emission`（0.8）；
  - 仿真按设备状态改灯罩颜色。
- **辊筒**：
  - 每根辊筒单独一个部件，带铰链关节：轴沿辊筒，`limited: false`、`continuous: true`、`damping: 0.001`；
  - 碰撞是一个 16 边圆柱，顶点落在名义圆上；
  - 多根辊筒的贴图共用一张（第 4 节提示词 13 第 3 条）；
  - 在 `prop.json` 的 `conveyor` 里写辊筒部件列表、输送方向（道具坐标）、辊筒顶面高度和半径；
  - 仿真给每根辊筒加速度执行器，靠摩擦带走物体。
- **颜色**：策略要靠颜色区分的东西（周转箱、标牌、指示灯、模块色条）用饱和、互相差得远的颜色，按下面给的色号。
- **场地**：机器人底盘占地 x −330 到 240、y ±300（410 高以下），前轮在 x 47–213。落地的道具（推车、输送线、流利道、供件台）的腿和脚轮都不能伸进去。下面的摆放已经避开了，腿只能往道具里面收，不能往外撇。
- **碰撞**：凹的地方一律拆成凸块（1.6 节）；一个部件超过 64 块时，把它拆成几个固定在一起的部件（`parent` 指向主体，`joint` 为 `null`）。

### 6.1 A 组：物流

#### `tote_cart` 料车（补料、分色搬箱共用）

- **外形**：
  - 两层钢制推车。车面 520（Y）× 260（X），顶面高 800；下层板高 300；
  - 四根 25 × 25 方管立柱，在车面边缘以内 15；
  - 四个 Ø75 万向脚轮，连轮叉总高 100，装在立柱正下方；
  - 推把在 +X 一侧，Ø25 弯管从两根 +X 立柱向上，顶高 950；
  - 车面四周 10 高的翻边（1.5 钢板），车面铺灰色防滑垫。
- **材质**：车架粉末喷涂；防滑垫 #6E7378 橡胶。
- **变体**：车架颜色 `blue` #1E4C9A（默认，送料车）、`red` #C62828（隔离车）、`green` #2E7D32（入库车）。
- **质量**：18 kg，static。
- **碰撞**：车面 1、翻边 4、下层板 1、立柱 4、推把 1–3、脚轮 4（盒子）。
- **原点**：地面上，车面中心正下方；长边沿 Y，推把在 +X。
- **摆放**：
  - 补料：在 (0.37, 0)，推把朝 +X；
  - 分色搬箱：(0, ±0.50)，车面长边沿 X，推把朝外。机器人手臂下垂时，手和腕部相机块占着 |y| 0.19–0.33、高 0.63–1.07 m 的地方，车面靠机器人的一边要在 |y| 0.37 以外（2026-10-25 实验：放在 ±0.45 时，手一抬就撞上车面）。

#### `line_feed_conveyor` 供线辊道（补料）

- **坐标**：原点在地面上、入口端（−X 端）中心正下方，+X 是输送方向（离开机器人）。
- **辊筒**：
  - 20 根 Ø48 × 330 镀锌钢管，间距 60，第一根中心在 x = 30；
  - 辊筒顶面高 1050。
- **框架**：
  - 两侧 C 型钢，高 80，顶面比辊筒顶面低 10，内宽 340；
  - 两对 40 × 40 铝型材腿在 x = 100 和 1100，带调平脚和横撑；
  - −Y 侧下方一台减速电机和链条罩（只做外观）。
- **侧挡**：
  - x 300 到末端，两侧各一根 Ø12 圆钢导条，高出辊筒顶面 25，内侧间距 330；
  - 入口处（x 300–360）向外张开 15°；
  - x < 300 不装侧挡：机器人在这里放料箱，手指要从两侧下来。
- **末端挡杆**：+X 端，高出辊筒顶面 40。
- **放行按钮**：
  - 入口端 −Y 侧用一根支臂伸出一个 80 × 80 × 100 的灰色按钮盒，盒中心在 (−50, −280)，盒顶高 1000；
  - 盒顶一个 Ø40 绿色蘑菇头 `release_button`，沿 −Z 滑动，行程 6，弹簧回位；
  - 旁边一块小铭牌写 “RELEASE”。
- **光电开关**：入口端 +Y 侧（只做外观）。
- **质量**：45 kg，static；辊筒摩擦 0.6，其余 0.4。
- **碰撞**：辊筒 20、侧框 2、导条和张口 2–4、挡杆 1、按钮盒 1、按钮 1。
- **摆放**：入口端在 (0.50, 0)。

#### `takeaway_conveyor` 出货辊道（装箱）

- **坐标**：
  - 台面式，放在工作台上；原点在道具底面中心（也就是台面上）；
  - 沿 −Y 输送（从机器人看，从左往右）。
- **装箱台板** `packing_plate`：
  - x −235 到 −65（深 170），y ±200，顶面高 30；
  - 铝板，面上铺灰色防静电垫；
  - 成品盒开局放在这上面，最后被推到辊筒上。
- **辊道**：
  - 辊筒床在 x −65 到 235，y ±400；
  - 26 根 Ø20 × 280 的辊筒，轴沿 X，间距 30，轴心高 20，顶面高 30，和台板齐平；
  - 靠台板一侧的边框顶面高 28，比辊筒低 2，盒子推过去不会卡；
  - 远侧（+X）导板高出辊筒顶面 40，挡住推过头的盒子；
  - −Y 端挡板高出辊筒顶面 40，盒子送到这里停住。
- **质量**：12 kg，static；台板摩擦 0.5，辊筒 0.6。
- **碰撞**：台板 1、辊筒 26、近侧边框 1、远侧导板 1、端挡板 1、辊筒床底板 1。
- **摆放**：原点在 (0.535, 0, 0.90)，台板前沿在 x 0.30。

#### `induction_conveyor` 供件输送线（包裹）

- **坐标**：落地式，原点在地面上、入口端（−X 端）中心正下方，+X 是输送方向。
- **辊筒**：
  - 44 根 Ø20 × 360，间距 25，第一根中心在 x = 15；
  - 辊筒顶面高 940。
- **挡边和端板**：
  - 两侧不锈钢挡边，内侧间距 370，高出辊筒顶面 20，全长都有；
  - 入口端板高出辊筒顶面 30；
  - 末端挡板高出 40，包裹在这里停住。
- **扫码门架**：
  - 在 x = 650，两根 40 × 40 立柱在 y ±280，横梁底面高出辊筒顶面 250；
  - 横梁下装一个扫码头：黑色盒子 120 × 80 × 60，红色窗口朝下；
  - +Y 立柱顶上装一个 Ø40 球形状态灯 `scan_lamp`（亮灯颜色写绿色，仿真另外用红色表示没读到）；
  - 门架上贴一块 “SCAN” 铭牌。
- **腿和驱动**：
  - 腿在 x = 50、500、1050，每处一对，40 × 40 型材，带调平脚；
  - 驱动电机只做外观。
- **质量**：50 kg，static。
- **碰撞**：辊筒 44、挡边 2、端板 2、门架立柱 2、横梁 1。
- **摆放**：入口端在 (0.30, 0)。

#### `flow_lane` 重力流利道（分色搬箱）

- **坐标**：原点在地面上、取箱端（−X 端，靠机器人的低端）中心正下方。
- **滑道**：
  - 单独一个部件 `lane`（固定在架子上）：宽 340，水平投影长 800；
  - 向 +X 上坡 3°，低端顶面高 1050，高端约 1092；
  - 外观上做三条黄色流利条（#F2B705，一排 Ø30 的小滚轮），滚轮顶和板面齐平；
  - 碰撞是一块平板，摩擦 0.03。MuJoCo 默认取两个接触物体中较大的摩擦，仿真里要给滑道设 geom priority 才生效（Isaac 里把滑道材质的摩擦组合方式设成 min）。实验里 3° 坡、摩擦 0.03 时，后面的箱子自己滑到挡块，撞上挡块没有超过 1 mm 的穿透；摩擦 0.08 比 tan 3° 还大，箱子不会动。
- **前挡**：低端一根 30 × 30 方管，跨全宽，顶面高出滑道面 30，内侧面在 x = 0。
- **侧挡**：
  - x 220 到 800，两侧 40 高的导向板，内侧间距 320；
  - 入口（x 220–260）张开导入；
  - x < 220 不装侧挡：这里是取箱的地方，手指从两侧下来。
- **架子**：
  - 40 × 40 型材；
  - 两对立柱，在 x = 60 和 740、y ±200；
  - 带斜撑和调平脚；
  - 高端装一个标牌架（只做外观）。
- **质量**：30 kg，static；架子摩擦 0.4。
- **碰撞**：滑道板 1、前挡 1、侧挡 2–4、立柱 4。
- **摆放**：取箱端在 (0.50, 0)。排在最前面的周转箱靠着前挡，中心在 x 0.60。

#### `recheck_shelf` 复检架（分色搬箱）

- **外形**：
  - 架面 400（Y）× 240（X），顶面高 800，四周 10 高的翻边；
  - 下层板高 300；
  - 四根立柱，在架面边缘以内 20。
- **颜色**：架子黄色 #F9A825，架面灰 #6E7378。
- **质量**：12 kg，static。
- **碰撞**：不超过 12 块。
- **原点**：地面上，架面中心正下方。
- **摆放**：(0.36, 0)，在流利道前面。

#### `station_sign` 工位标牌（分色搬箱）

- **外形**：
  - 底座是 Ø250 × 15 的铸铁圆盘，立杆 Ø30；
  - 顶上一块 300（宽）× 220（高）的标牌：2 mm 铝板加塑料边框；
  - 牌面中心高 1300，朝 −X。
- **变体**：
  - `ok`：绿底 #2E7D32，白色大字 “OK”，下面一行 “STORAGE”，加一个对勾；
  - `recheck`：黄底 #F9A825，黑字 “RECHECK”，加一个放大镜；
  - `ng`：红底 #C62828，白字 “NG”，下面一行 “QUARANTINE”，加一个叉。
- **质量**：5 kg，static。
- **碰撞**：底座 1、立杆 1、牌 1。
- **原点**：地面上，底座中心。

#### `handle_tote` 带把手块的周转箱（补料、分色搬箱，2026-10-25 实验后新增）

- **为什么**：`small_tote` 两端只能夹到端壁加 4 mm 高的外翻箱沿（共 12.4 mm），外侧指垫只贴住这 4 mm。实验里双手抬箱时两只手各有一根手指反复脱开，箱子在手里歪 13°，转腰时掉了。两端改成实心把手块以后，抬起、放平、转腰 90° 放上推车都稳，放置误差 0.6 mm。
- **外形**：和 `small_tote` 一样：顶部外沿 300（Y）× 200（X）× 100（高），侧壁拔模 2°，壁厚 3，底厚 4，顶部翻边宽 6、厚 4，外侧竖筋。
- **把手块**：两个短边（±Y）各一个实心把手块，沿 X 长 120（居中），高 30（z 70–100，顶面和箱沿齐平），沿 Y 厚 14（外侧面在 ±150，和箱沿外沿齐平，内侧面在 ±136）。内外两面都是竖直的平面（这一块不拔模），棱边 1 mm 倒角。把手块下面开一个 80 × 20 的把手孔（外观）。
- **标签**：长边 −X 面中间一张 90 × 40 的白卡：`blue` 和 `grey` 写 “BIN-A07”；`red` 写 “NG”、`yellow` 写 “RECHECK”、`green` 写 “OK”，字高 22。
- **变体**：`blue` #1F5FA8（默认）、`grey` #9AA0A6、`red` #C62828、`yellow` #F9A825、`green` #2E7D32。
- **质量 / 摩擦**：0.45 kg / 0.40（仿真按任务加箱内配重）。
- **碰撞**：底板 1、四面墙各 1、两个把手块各 1、翻边 4，凸块拼成。
- **`prop.json`**：`dimensions_mm` 写把手块的中心（y ±143、z 85）、厚度 14、长 120、高 30；`grasp` 写 `{"points_mm": [[0, 143, 85], [0, -143, 85]], "width_mm": 14}`。
- **原点**：底面中心。

#### `small_tote` 加颜色变体（改 `build_task01_tote_transfer.py`）

- 在现有的 `blue`、`grey` 后面加三个贴图变体：`red` #C62828（`body_albedo_v2.png`）、`yellow` #F9A825（`v3`）、`green` #2E7D32（`v4`）。
- 几何、碰撞、质量都不变。
- 同一个脚本建的 `flange_part` 和 `abs_housing` 要和现在的一致。

### 6.2 B 组：电子装配

#### `plc_module` PLC I/O 模块（测试、装箱）

- **坐标**：
  - 按平放的样子建模，包围盒 65（X）× 90（Y）× 30（Z），原点在底面中心；
  - X 是模块的进深，Y 是模块的高，Z 是模块的宽。
- **正面**（−X，90 × 30，装到导轨上以后朝外的那面）：
  - 8 宽的型号色条，沿 Y；
  - 16 个 LED 小窗（贴图）；
  - 一个可拔端子块的外形，深灰，凸出不超过 2；
  - 一张小翻盖标签。
- **顶面**（+Z，90 × 65）：
  - 大字 “IMC Controls” 和型号（“DI16”、“DO16” 或 “AI4”，字高至少 12）；
  - 一行小字参数和一个二维码。
- **背面**（+X）：DIN 导轨卡槽的外形，凹进不超过 3。
- **侧面**（±Y）：散热槽（贴图）。
- **平整度**：±X 两面中间 50 宽的区域要平，夹爪沿 X 夹这两面。
- **材质**：外壳浅灰 PC #D9DDE0，端子块深灰 #3A3D40。
- **质量 / 摩擦**：0.15 kg / 0.5。
- **碰撞**：1 个盒子 65 × 90 × 30。
- **变体**：色条和型号字不同，`DI` 蓝 #1565C0、`DO` 橙 #EF6C00、`AI` 绿 #2E7D32。

#### `module_tray` 模块托盘（测试的来料盘和良品盘，装箱的模块盘）

- **外形**：
  - 两穴吸塑托盘，150（X）× 230（Y）× 25；
  - 两个穴沿 Y 排，中心在 y ±56；
  - 每个穴 96（Y）× 71（X）× 15 深，模块平放、正面朝 −X，穴口 2 × 45° 倒角。
- **指槽**：每个穴的 ±X 两端各开一道指槽，宽 30（沿 Y），伸出穴边 22（沿 X），从盘面往下深 20。
- **变体**：
  - `in`：黑色导电 PP #1A1A1A，盘边印 “IN”；
  - `ok`：绿色 #2E7D32，印 “OK / PASS”。
- **质量**：0.3 kg，static。
- **碰撞**：凹形拆成凸块，不超过 30 块。

#### `test_fixture` 功能测试治具

- **底板**：240（X）× 300（Y）× 20 黑色阳极氧化铝板，四角螺钉（外观）。原点在底面中心。
- **定位台**：
  - 底板中央一个 100（Y）× 75（X）× 8 高的台子，模块平放在上面，正面朝 −X；
  - 台面下的弹簧顶针只做外观。
- **四角 L 形限位块**：
  - 每块两条臂，高出台面 18，厚 8；
  - 限位内框 96 × 71，模块四周各留 3；
  - 内侧上沿 3 × 45° 倒角；
  - 挡 ±Y 两端的臂长 20，挡 ±X 两面的臂长 18：±X 两面中间 50 空出来，给手指下去。
- **测试按钮**：
  - +Y 侧一个 60 × 60 × 60 的灰色按钮盒，中心在 (−20, +120)，盒顶比底板底面高 80；
  - 盒顶一个 Ø40 蓝色蘑菇头 `test_button`，沿 −Z 滑动，行程 6，弹簧回位；
  - 旁边印 “TEST”。
- **结果灯**：
  - +X 侧一根 Ø20 立柱，顶面高出底板 200；
  - 柱顶一个 Ø60 半球灯罩 `result_lamp`，灭灯色 [0.85, 0.85, 0.82, 1]；
  - 亮灯颜色写绿色，仿真另外用黄色和红色。
- **其他**：线缆从定位台后面接到立柱（只做外观，不做碰撞）；铭牌 “IMC FT-02 FUNCTION TESTER”。
- **质量**：4 kg，static。
- **碰撞**：底板 1、定位台 1、限位块 8（每条臂一块，倒角做进凸块里）、按钮盒 1、按钮 1、立柱 1、灯 1。
- **摆放**：(0.38, 0)，测试按钮在左边。

#### `ng_bin` 不良品框

- **外形**：红色 PP 浅框 #C62828，外形 160 × 160 × 32（壁高 30，底厚 2）。
- **印字**：前壁（−X）外侧和框底印白色大字 “NG”。
- **质量**：0.3 kg，static。
- **碰撞**：底 1、壁 4。

#### `product_box` 成品盒

- **盒子**：
  - 白色纸盒 #F2F2F0，外形 168（Y）× 100（X）× 50（Z），纸板厚 3，原点在底面中心，盖子合着；
  - 内部用一块 2 mm 隔板分成两格：+Y 一格放模块，内尺寸 100（Y）× 94（X）；−Y 一格放附件盒，60（Y）× 94（X）；内高 47。
- **盖子** `lid`：
  - 168 × 100 × 3，铰链在盒子后上沿（+X 边、顶面）；
  - 前沿伸出盒子前壁 10，再向下折 8：手指合拢后能从下面把它挑起来；
  - 铰链关节绕 Y 轴，方向取打开为正，范围 0（合上）到 1.75（100°，打开后靠在限位上），`damping` 0.01，`frictionloss` 0.001。
- **印刷**：盒面和盖子上印蓝色 IMC Controls 标志和 “PLC I/O MODULE”，一侧贴条码标签。
- **质量 / 摩擦**：盒 0.06 kg，盖 0.02 kg / 0.6。
- **碰撞**：底 1、四壁 4、隔板 1、盖 1、翻边 1。

#### `accessory_box` 附件盒

- **外形**：牛皮纸小盒 60（X）× 50（Y）× 25（Z）。
- **标签**：“ACCESSORY KIT”、“TERMINAL BLOCKS + MANUAL”，加一个 IMC Controls 小标志。
- **质量 / 摩擦**：0.05 kg / 0.6。
- **碰撞**：1 个盒子。

#### `stack_light` 三色灯（开机）

- **外形**：
  - L 形安装支架，底面 100 × 60 × 8，放在台面上；
  - Ø25 立杆，高 200；
  - 三节 Ø70 × 60 的灯，从上到下红、黄、绿；
  - 顶上一个 Ø70 × 30 的黑色蜂鸣器帽；
  - 总高约 440。
- **灯罩**：
  - 三节灯罩分别是部件 `lens_red`、`lens_amber`、`lens_green`，不贴图；
  - 灭灯色：红 [0.35, 0.06, 0.06, 1]，黄 [0.40, 0.30, 0.05, 1]，绿 [0.06, 0.30, 0.10, 1]；
  - `lamps` 里写各自的亮灯颜色。
- **质量**：1.5 kg，static。
- **碰撞**：支架 1、立杆 1、灯 3（16 边圆柱）、帽 1。
- **原点**：支架底面中心。

#### `line_console` 产线操作台（开机）

- 照 `control_console` 做：外形、斜面、材质都一样，换掉面板上的元件。
- 面板上一排从左到右：
  - 复位 RESET（蓝色平头按钮）；
  - MAN / AUTO 选择旋钮，MAN −45°、AUTO +45° 两个定位档；
  - 启动 START（绿）；
  - 停止 STOP（红）。
- 右下角一个急停：红色蘑菇头加黄色底牌。
- 去掉原来的三个指示灯，状态改由三色灯显示。
- 关节和行程照 `control_console`：按钮滑动 4 加弹簧，旋钮 ±45° 铰链，急停滑动 5。
- 面板字用 DejaVu Sans Bold。
- 在 `build_task02_panel_operation.py` 的基础上另写函数，放进 `build_task15_electronics.py`；`control_console` 和 `panel_devices` 的输出不能变。

### 6.3 C 组：拣料、配盘和装配

#### `parts_bin` 零件料盒（拣料）

- **外形**：
  - 浅料盒 180（X）× 150（Y）× 38（Z），PP 壁厚 2.5，底厚 3，蓝色 #1F5FA8；
  - 盒壁故意做低：夹爪壳体在夹持中心以上约 23 mm，要从盒壁上方过。
- **标签**：
  - 前壁（−X）外侧有 70 × 30 的标签卡槽；
  - 四个贴图变体 `relay`、`terminal`、`gland`、`fuse`：白卡上画对应零件的线条图标，加料号（“RLY-24V”、“TB-4P”、“CG-M20”、“FH-10A”）和条码。
- **质量**：0.2 kg，static。
- **碰撞**：底 1、壁 4。
- **摆放**：中心 x 0.38，y ±0.19 和 ±0.36。每盒放 3 个同样的零件，沿 X 一排，间距 55。

#### 四种零件（拣料）

都按小零件预算（1 万面以内，贴图 1024），原点在底面中心：

| 编号 | 外形 | 颜色和印字 | 质量 | 碰撞 |
|---|---|---|---|---|
| `relay` 继电器 | 28（X）× 28（Y）× 40（Z），底部 8 个插脚（外观） | 不透明蓝 #1E5AA8，顶面白标签 “IMC RLY 24VDC” | 0.05 kg | 1 个盒子 |
| `terminal_block` 端子模块 | 45（X）× 22（Y）× 35（Z） | 灰 #9AA0A6，顶面 4 个橙色接线螺钉孔，印 “TB-4P” | 0.05 kg | 1 个盒子 |
| `cable_gland` 电缆接头 | 下段 M20 螺纹 Ø20 × 10，中间六角对边 30 × 10，上面圆顶螺帽 Ø28 × 15，总高 35 | 黑色尼龙 | 0.04 kg | 三段各一个凸块 |
| `fuse_holder` 熔断器座 | 25（X）× 60（Y）× 40（Z），前面有熔断器盒外形（不做关节） | 灰 #8C9399，顶面印 “FH-10A” | 0.05 kg | 1 个盒子 |

摩擦都是 0.5。

#### `order_tray` 订单盘（拣料）

- **外形**：灰色防静电盘，200（Y）× 140（X）× 30（Z）；中间一条 12 高的隔筋，把盘分成左右两半。
- **印字**：盘前沿印 “ORDER”。
- **质量**：0.2 kg，static。
- **碰撞**：底 1、壁 4、隔筋 1。
- **摆放**：(0.38, 0)。

#### `order_card_stand` 订单卡座（拣料）

- **外形**：
  - 亚克力 L 形卡座，底座 160 × 60 × 8；
  - 卡片向后倾 15°，148（宽）× 105（高），A6 横放。
- **卡片**：
  - 卡片 `card` 是单独部件，正面朝 −X；
  - UV 把一整张 1480 × 1050 的贴图正好铺满正面：仿真按每集的订单生成这张图换进去；
  - 默认贴图是订单卡模板：“WORK ORDER”，下面三行空栏。
- **质量**：0.3 kg，static。
- **碰撞**：底座 1、卡片 1。
- **摆放**：(0.56, 0)。

#### `foam_kit_tray` 重建（改 `build_task06_shaped_kitting.py`）

- 凹槽深都改成 20（从泡棉顶面算）。法兰衬套的沉台保留：上面 Ø48 深 5，下面 Ø38。
- 每个凹槽沿 X 两侧各开一道取件指槽：宽 26（沿 Y），从凹槽边往外伸 22（沿 X），深 25（从泡棉顶面算）。
- 凹槽和指槽的上沿做 3 × 45° 倒角。
- 每边间隙还是 4，凹槽中心位置不变。
- 碰撞块超过 64 块就按凹槽把泡棉拆成几个固定部件。
- `black` 和 `light_grey` 两个变体都保留；`prop.json` 里更新 `pockets`（深度、指槽位置）。
- 原因：现在的凹槽 25–30 深、没有指槽，指尖跟不进去，零件只能悬空掉进去；有了指槽，手指能跟着零件下到底。

#### `rotor_shaft_fixture` 转子轴夹具（轴承预装，双工位）

- **底板**：260（X）× 440（Y）× 30 钢板，灰色喷塑 #6B7075。原点在底面中心。
- **两个工位**，中心在 y ±120。每个工位一个 V 形夹座（80 × 60 × 30）夹住转子轴下端，转子竖直立在上面，从下往上：
  - 转子铁芯 Ø36 × 50，硅钢片叠层外观（深灰带细纹），底面比底板顶面高 30（底面以上 60）；
  - 轴肩 Ø26 × 6，在铁芯上面，顶面高 116；
  - 轴承位 Ø18 × 40，顶端 4 × 30° 导向倒角，轴顶高 156。这一段是精密尺寸：多边形顶点落在 Ø18 圆上，误差不超过 0.05。
- **双手启动按钮**：
  - 底板前沿两角各一个 60 × 60 × 50 的按钮盒，中心在 (−100, ±190)；
  - 盒顶一个 Ø40 绿色蘑菇头，部件 `start_left`、`start_right`，沿 −Z 滑动，行程 6，弹簧回位。
- **状态灯**：底板后沿中间一根 Ø20 立柱，高 250，柱顶一个 Ø50 灯罩 `station_lamp`（亮灯颜色写绿色）。
- **铭牌**：“IMC BEARING PRE-ASSEMBLY”。
- **质量**：12 kg，static；`precision_mm` 0.05，`dimensions_mm` 写轴径、倒角、轴肩高度和轴顶高度。
- **碰撞**：底板 1、夹座 2、铁芯 2（16 边圆柱）、轴肩 2、轴承位 2（24 边）、倒角段 2、按钮盒 2、按钮 2、立柱 1、灯 1。
- **摆放**：(0.40, 0)。
- **配合**：6204 轴承内孔 Ø20，套在 Ø18 的轴上，直径间隙 2。

#### `bearing_tray` 轴承托盘（轴承预装）

- **外形**：100 × 100 × 10 的灰色 POM 托盘，上面是平的。
- **为什么不做凹窝**：抓轴承时夹持中心在托盘面以上 19，指尖只比托盘面高 2，凹窝的沿会挡住指尖。
- **印刷**：一个 Ø48 的圆圈和 “6204-2RS”。
- **质量**：0.2 kg，static。
- **碰撞**：1 个盒子。
- **摆放**：(0.38, ±0.30)。

### 6.4 D 组：包裹和码垛

#### `parcel_cube_80`、`parcel_cube_75`、`parcel_cube_85` 小包裹（供件）

- **外形**：
  - 三种单瓦楞小纸箱：80（X）× 70（Y）× 60（Z）、75 × 75 × 75、85（X）× 65（Y）× 55（Z）；
  - 壁厚 3；
  - 顶面有摇盖缝，沿 X 贴一条 48 宽的棕色胶带，两端各下翻 30。
- **面单**：
  - 56 × 46 的白色 QF Express 面单，大条码占一半面积，另有运单号；
  - 左上角一个 15 mm 的橙色三角角标（#EF6C00），从远处就能认出面单；
  - 还有虚构的收寄件信息。
- **面单位置**：做 5 个贴图变体：`top`（+Z）、`px`（+X）、`nx`（−X）、`py`（+Y）、`ny`（−Y）。
  - 面单贴在那一面的中央，歪 1–3°；
  - 其余各面只有胶带和纸箱印刷（向上箭头、小标志）。
- **质量 / 摩擦**：0.12 kg（仿真按种子在 0.08–0.15 kg 之间改）/ 0.6。实验里 0.25 kg 的包裹在空中翻到约 60° 就在手里转了，0.10 kg 的能撑住。
- **碰撞**：1 个盒子。
- **原点**：底面中心。
- **`prop.json`**：每个变体写面单所在的面（`label_face`）。

#### `infeed_table` 供件台（供件）

- **外形**：
  - 小钢桌，台面 300（X）× 280（Y），高 900；
  - 台面 30 厚，灰色 HPL；
  - 四条 40 × 40 方管腿，在台面边缘以内，带横撑。
- **质量**：15 kg，static。
- **碰撞**：台面 1、腿 4。
- **原点**：地面上，台面中心正下方。
- **摆放**：(0.40, ±0.36)。

#### `pallet_carton_160x80x80` 加 IMC 贴图（改 `build_task08_palletizing.py`）

- 加一个贴图变体 `imc`。
- 牛皮纸箱上印蓝色 “IMC Controls”、“PLC I/O MODULES × 10”、向上箭头和易碎标识。
- SKU 标签 “IMC-PLC-IO10”。
- 几何和其他输出都不变。

### 6.5 提示词 14–17

四条提示词可以同时发给四个 Codex，各在一个独立的仓库副本里做。共用的开头：

```text
You are adding simulation-ready props to the industrial benchmark of this repository (the current directory is the repository root). Read docs/benchmark_props.md §1, §2 and §6 (binding), and docs/benchmark_redesign.md for how each prop is used. Read tools/blender/prop_lib.py, prop_schema.py, validate_props.py and compact_props.py and one or two existing tools/blender/build_task*.py, and follow their conventions: millimetres in the spec, metres in the props, the prop frame of §1.2, convex collision pieces, a baked albedo per part, prop.json per §1.8, preview.png, and prop_lib.finalize_assets() at the end of the build.
Rules:
- Run Blender headless as `blender -b --factory-startup -t 6 -P <script> -- <args>` (Blender 5.2 on PATH; at most 6 threads, other builds share the machine).
- PROP_PYTHON is set in the environment; prop_lib uses it for validators and artwork workers. Validate with `$PROP_PYTHON tools/blender/validate_props.py --props <ids>`.
- Text artwork: only the DejaVu Sans fonts in tools/blender/fonts/, English text only.
- Only create or modify files under tools/blender/ and assets/props/. Do not modify prop_lib.py, prop_schema.py, validate_props.py or compact_props.py; if you need a change there, explain it in your report instead. Never run git commands.
- Every prop must pass validate_props.py. Render each preview and look at it; fix anything that does not match the spec.
- Finish with a report in tools/blender/task<NN>_report.md: per prop the bbox, triangles, collision pieces, texture sizes, joints, the functional metadata you wrote (buttons, lamps, conveyor), and every deviation from the spec with its reason.
```

#### 提示词 14：A 组，物流

```text
Task 14 (group A, logistics). Build every prop of §6.1: tote_cart, line_feed_conveyor, takeaway_conveyor, induction_conveyor, flow_lane, recheck_shelf and station_sign, in a new tools/blender/build_task14_logistics.py. Add the red, yellow and green small_tote variants by changing tools/blender/build_task01_tote_transfer.py; rebuilding it must leave flange_part and abs_housing as they are (compare file hashes before and after). Conveyor rollers are separate parts with continuous hinges and one 16-sided cylinder collider each; write the conveyor metadata. Keep every floor-standing leg and caster inside its own footprint. Report in tools/blender/task14_report.md.
```

#### 提示词 15：B 组，电子装配

```text
Task 15 (group B, electronics line). Build every prop of §6.2: plc_module (DI, DO and AI variants), module_tray (in and ok variants), test_fixture, ng_bin, product_box (hinged lid with the front flange), accessory_box, stack_light (three separate untextured lens parts) and line_console, in a new tools/blender/build_task15_electronics.py. line_console reuses the control_console design and the panel_devices library; control_console and panel_devices must stay unchanged. Buttons are 40 mm mushroom caps on sprung slide joints; write the buttons and lamps metadata. Report in tools/blender/task15_report.md.
```

#### 提示词 16：C 组，拣料、配盘和装配

```text
Task 16 (group C, picking, kitting and assembly). Build parts_bin (four label variants), relay, terminal_block, cable_gland, fuse_holder, order_tray, order_card_stand (a separate card part whose UVs map one full 1480 × 1050 image onto its front), rotor_shaft_fixture (precision shaft seats, two-hand start buttons, station lamp) and bearing_tray, all of §6.3, in a new tools/blender/build_task16_assembly.py. Rebuild foam_kit_tray with 20 mm pockets, the finger notches and the chamfers of §6.3 by changing tools/blender/build_task06_shaped_kitting.py; keep its black and light_grey variants and update the pocket metadata. Report in tools/blender/task16_report.md.
```

#### 提示词 17：D 组，包裹和码垛

```text
Task 17 (group D, parcels and palletizing). Build parcel_cube_80, parcel_cube_75 and parcel_cube_85, each with the five label-face variants of §6.4 (top, px, nx, py, ny) and label_face metadata, and infeed_table, in a new tools/blender/build_task17_parcels.py. Add the imc texture variant to pallet_carton_160x80x80 by changing tools/blender/build_task08_palletizing.py; nothing else it builds may change (compare file hashes before and after). Report in tools/blender/task17_report.md.
```

#### 提示词 18：带把手块的周转箱（实验后追加）

```text
Task 18 (handle_tote). Build handle_tote of §6.1 in a new tools/blender/build_task18_handle_tote.py: the small_tote body (reuse its geometry approach from tools/blender/build_task01_tote_transfer.py, but do not change that script or small_tote) plus two solid 120 × 14 × 30 mm handle blocks at the short ends, flush with the rim, with flat vertical faces on both sides, the cosmetic handle opening under each block, the −X label card per colour variant (BIN-A07, NG, RECHECK, OK) and the blue, grey, red, yellow and green variants. Write the grasp metadata. Collision: floor, four walls, two handle blocks and the rim, all convex. Report in tools/blender/task18_report.md.
```

### 6.6 交付后的检查

道具拉回本机以后，仿真这边逐个检查：
- `validate_props.py` 全部通过，文件都在、能被 `sim/prop_assets.py` 加载；
- 尺寸和关键部位（辊筒顶面高度、按钮和灯的部件、指槽、轴径、盖子铰链）和规格一致；
- 在 MuJoCo 里静置不漂移、不穿模；按钮按得下去也弹得回来，盖子能开合，辊筒能转；
- 渲染预览和机器人四路相机的画面，看颜色、字和面单是否清楚。

### 6.7 2026-10-25 的交付和检查

提示词 14–18 在 pro6000 上同时跑了五个 Codex，每个用一份自己的仓库副本，做完拉回本机：
- **交付**：29 个新道具，重建了泡棉配盘，码垛纸箱加了 IMC 贴图。`validate_props.py` 全部通过。
- **MuJoCo 检查**（不进仓库的检查脚本）：每个道具单独放在地面上加载、静置 2 s，再推每个关节；输送线给辊筒加速度执行器，放一个测试箱。
  - 全部能加载，放着不漂、不倒，静止时没有超过 0.5 mm 的穿透，只有供件输送线例外：末端挡板和最后一根辊筒的碰撞体重叠 3 mm。
  - 三条动力输送线都能把测试箱按规格的方向送走，3 s 走约 292 mm（表面速度 0.1 m/s）。
  - 按钮都能按下、能弹回。但按钮帽只有约 20 g，MuJoCo 默认的关节限位很软：10 N 时操作台的按钮冲过 4 mm 行程好几倍，用力推成品盒的盖子也会冲过 100° 的限位。搭场景时要把这些关节的 `solreflimit` 调硬，专家也只按到 `activation_m`。
- **没有拉回的**：`small_tote` 的红、黄、绿变体。重新烘焙时，原有的蓝、灰贴图和预览图像素也变了（pro6000 上是 Blender 5.2.3），而 `tote_transfer` 已经用原来的贴图在 Isaac 里录了数据。新任务改用 `handle_tote`，这三个颜色变体用不上了。
- **改名**：Task 16 的辅助模块原名 `task16_support.py`，会被 `.gitignore` 的 `tools/blender/task[0-9]*` 排除，所以改名为 `support_task16.py`，导入它的脚本也一起改了。
- **提示词 19（检查后的修补，已完成）**：改动的 9 个道具通过检查，其余 66 个道具逐字节不变，包裹每个 3.2–3.5 MB、供件台 0.9 MB。
  - 供件输送线的挡板挪开；
  - 三条输送线写 `collision_exclusions`：机架是焊在世界上的，MuJoCo 不过滤它和辊筒、按钮之间的父子接触；
  - 流利道摩擦改成 0.03 并标出要接触优先级；
  - 料车在分色搬箱的推荐位置改成 ±0.50；
  - 包裹质量改成 0.12 kg；
  - 包裹和供件台每个 15–27 MB，主要是 `source.blend` 和每个面的原图，要压到 4 MB 以内；
  - `compact_props.py` 指定 `--props` 时仍会扫描、去重所有道具，B、C 和 18 组都只能自己绕开，这次改成只动指定的道具。
- **加载器**（`sim/prop_assets.py`）跟着改了两处：
  - 有行程的关节，限位用和接触一样硬的参数（`solref` 0.004、`solimp` 0.95–0.99）。像手指那样逐渐加到 10 N 时，按钮只比行程多压 0.2–0.5 mm，旋钮和盒盖正好停在限位上；以前用 MuJoCo 默认值，按钮会冲过行程好几倍。
  - 部件的 `contact_priority` 映射成 geom priority：流利道滑道的 0.03 才压得过周转箱的 0.4。用真道具实验时，后面的箱子滑到挡块，撞上去没有超过 1 mm 的穿透。
