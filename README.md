# 摄像头偏移检测与关键区域自适应

固定安防摄像头一旦被风吹、震动、碰撞而发生**三轴旋转**（pan/tilt/roll），人工画在画面上的「关键区域」多边形就会漂到错误的真实位置，导致 YOLO 误报。

本项目用**图像配准**把这件事统一解决：当前帧与标准位参考帧之间的映射严格是一个**单应矩阵 H**（相机绕光心旋转、与场景深度无关）。估出 H 后，一套机制同时给出三个结果：

1. **偏移检测告警** —— 相机偏离标准位时报警；
2. **角度估计** —— 估出 pan / tilt / roll 三个旋转角；
3. **多边形自适应** —— 自动把多边形校正到当前画面里的真实区域。

> 原理与方案的完整背景见 [`CLAUDE.md`](CLAUDE.md)。

---

## 目录

- [安装](#安装)
- [核心概念](#核心概念)
- [快速开始（端到端 3 步）](#快速开始端到端-3-步)
- [第一步：标定（导入标准位图 + 多边形）](#第一步标定导入标准位图--多边形)
- [第二步：查看校准结果（单帧命令行）](#第二步查看校准结果单帧命令行)
- [视频流：直接标定 + 整段输出](#视频流直接标定--整段输出)
- [第三步：接入运行时（Python API）](#第三步接入运行时python-api)
- [昼夜多参考帧](#昼夜多参考帧)
- [低纹理 / 强光照降级（ECC）](#低纹理--强光照降级ecc)
- [配置参数速查](#配置参数速查)
- [输出与告警解读](#输出与告警解读)
- [缓存文件格式](#缓存文件格式)
- [模块结构](#模块结构)
- [已知约束与注意事项](#已知约束与注意事项)

---

## 安装

项目用 [uv](https://docs.astral.sh/uv/) 管理环境（Python ≥ 3.11）。

```bash
# 在项目根目录
uv sync            # 按 uv.lock 安装依赖（opencv-python, numpy）

# 验证环境（会跑一遍 ORB→匹配→findHomography→投影 的冒烟检查）
uv run python main.py
```

预期输出：

```
camera-drift v0.1.0
OpenCV 4.13.0, NumPy 2.4.6
ORB 特征: ref=2000 cur=2000 | good=688 | inliers=657
多边形投影平均误差: 0.186 px
OK: v1 单应估计工具链跑通。
```

> 之后所有命令都用 `uv run python ...` 运行，确保用的是项目虚拟环境。

---

## 核心概念

| 名词 | 含义 |
|------|------|
| **参考帧 (ref_frame)** | 相机处于**标准位置**时拍的一帧。多边形就画在这帧上。 |
| **多边形 (polygon)** | 人工标的关键区域，坐标是**参考帧像素坐标系**下的点 `[[x,y], ...]`（≥3 个点）。 |
| **标定缓存 (.npz)** | 把参考帧 + 预算好的 ORB 特征 + 多边形打包存盘，装机时生成一次。 |
| **单应 H** | 3×3 矩阵，把「参考帧坐标 → 当前帧坐标」。`当前多边形 = H · 参考多边形`。 |
| **偏移量** | 参考帧四角经 H 投影后移动的像素距离，超阈值即告警。 |

整个流程分两个阶段：

```
① 标定阶段（装机时一次）       ② 运行阶段（每隔 N 秒/定时触发，用缩小图，开销小）
   参考帧 + 多边形                当前帧 → 估 H ┬→ 偏移超阈值？→ 告警        (需求1)
        │  提特征                              ├→ 分解 H → pan/tilt/roll   (需求2)
        ▼                                      └→ 投影多边形 → 自适应       (需求3)
   cache/cam01.npz
```

YOLO 照常每帧跑；偏移检测低频跑，互不影响。

---

## 快速开始（端到端 3 步）

下面用一张你自己的标准位图 `ref.jpg` 演示。假设关键区域是一个矩形。

```bash
# 1) 标定：把标准位图 + 多边形 存成缓存
uv run python calibrate.py ref.jpg \
    --polygon "100,100 540,100 540,380 100,380" \
    -o cache/cam01.npz

# 2) 相机转动后，拿一张当前帧 current.jpg，看多边形是否跟住 + 报偏移/角度
uv run python track.py cache/cam01.npz current.jpg -o result.png

# 3) 打开 result.png：绿色多边形=校正后跟随的区域，灰色细线=原始参考位置
```

> 手上是**视频**？可以全程命令行：`calibrate.py input.mp4 ...`（自动取首帧标定）+ `track_video.py ... input.mp4 -o output.mp4`（逐帧画框输出整段视频）。详见[视频流：直接标定 + 整段输出](#视频流直接标定--整段输出)。

---

## 第一步：标定（导入标准位图 + 多边形）

`calibrate.py` 把「标准位参考帧」和「多边形」固化成一个缓存文件。**多边形有三种导入方式**，任选其一。

> **参考帧可以直接是视频**：`ref_image` 传图片或视频均可。识别为视频（`.mp4 .avi .mov .mkv .m4v .wmv .flv`）时**自动取第一帧**做标定——无需先手动截图。前提是视频首帧就是清晰的标准位（非黑屏/过曝）；若开头有黑屏帧，请改用单独截的标准位图。

### 方式 A：多边形来自 JSON 文件（推荐，适合已有标注系统）

如果你们的标注系统已经存了多边形坐标，导出成 JSON：

`poly.json`：
```json
[[100, 100], [540, 100], [540, 380], [100, 380]]
```

```bash
uv run python calibrate.py ref.jpg --polygon-file poly.json -o cache/cam01.npz
```

### 方式 B：多边形内联在命令行

点之间空格分隔，每个点 `x,y`（逗号分隔，**不要空格**）：

```bash
uv run python calibrate.py ref.jpg --polygon "100,100 540,100 540,380 100,380" -o cache/cam01.npz
```

### 方式 C：交互画（独立测试 / 没有现成坐标时）

```bash
uv run python calibrate.py ref.jpg --draw -o cache/cam01.npz

# 直接对视频画框（自动取首帧）
uv run python calibrate.py input.mp4 --draw -o cache/cam01.npz
```

弹窗操作：
- **左键**：依次添加多边形顶点；
- **回车** 或 **右键**：完成（至少 3 个点）；
- **Esc**：取消。

> **大图看不全 / 窗口太大**：画框窗口会把图等比缩小到最长边 ≤ `--max-display`（默认 1280px）以保证看到全貌，窗口也可拖拽缩放。你点在缩放后的画面上，程序会自动把坐标换算回**原图坐标**，标定精度不受影响。屏幕小可调小（如 `--max-display 960`），想看更清晰可调大。

### 参数说明

| 参数 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `ref_image` | 是 | — | 标准位参考帧：图片，或视频文件（自动取第一帧） |
| `-o, --out` | 否 | `cache/<图名>.npz` | 缓存输出路径 |
| `--polygon-file` | 三选一 | — | 多边形 JSON 文件 `[[x,y],...]` |
| `--polygon` | 三选一 | — | 内联多边形 `"x,y x,y ..."` |
| `--draw` | 三选一 | — | 交互画多边形 |
| `--max-display` | 否 | `1280` | 交互画框窗口最长边像素上限（大图缩放看全貌） |
| `--nfeatures` | 否 | `2000` | ORB 特征点数量 |

成功输出：

```
已保存标定缓存: cache/cam01.npz
  参考帧: (480, 640, 3) | 特征: 2000 | 多边形: 4 点
```

> ⚠️ 如果参考帧是**大白墙/纯地面**这种低纹理画面，ORB 可能提不到特征，会打印 `[警告] 参考帧 ORB 特征为 0` 的警告——这不是错误，缓存仍会生成，运行时会自动改走 ECC 降级（见后文）。

---

## 第二步：查看校准结果（单帧命令行）

`track.py` 载入缓存 + 一张当前帧，**估 H → 报偏移 → 报角度 → 投影多边形**，并可把结果画到图上。

```bash
uv run python track.py cache/cam01.npz current.jpg -o result.png
```

典型输出：

```
good=555 inliers=524 ratio=0.94 ok=True
[偏移] [告警]: 偏离标准位：最大角位移 82.9px > 15.0px
       最大角位移=82.9px 平均=61.2px
[角度] pan=+0.12° tilt=-0.30° roll=-9.98° (f≈610px)
投影后多边形: [[99.4, 139.9], [320.4, 95.4], ...]
已保存标注图: result.png
```

`result.png` 里：
- **绿色实线**：校正后跟随真实区域的多边形；
- **灰色细线**：原始参考位置（对比用，能直观看到相机转了多少）。

### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `cache` | — | 标定缓存 `.npz` |
| `current` | — | 当前帧图片 |
| `-o, --out` | 无 | 标注输出图路径（不给则只打印不画图） |
| `--ratio` | `0.75` | 比率测试阈值（越小匹配越严格） |
| `--ransac` | `3.0` | RANSAC 重投影阈值(px) |
| `--drift-thresh` | `15.0` | 偏移告警阈值(px) |

退出码：`0`=正常，`1`=读不到图，`2`=H 估计失败（画面变化过大/纹理不足，应人工介入）。

> 💡 **如何制造一张「旋转后」的测试帧**：把相机手动转一个小角度再截一帧；或对参考帧做一次已知单应变换（详见 [`CLAUDE.md`](CLAUDE.md) 验证标准）。绿色多边形稳稳贴住真实区域、角度合理，即说明工作正常。

---

## 视频流：直接标定 + 整段输出

如果你手上是一段摄像头视频（首帧为标准位），可以**全程命令行**完成：用视频首帧标定，再对整段视频逐帧校正，输出一段**每一帧都画好多边形**的标注视频。

```bash
# 1) 用视频首帧标定（自动取第一帧；画框或给坐标皆可）
uv run python calibrate.py input.mp4 --draw -o cache/cam01.npz

# 2) 对整段视频逐帧校正并输出标注视频
uv run python track_video.py cache/cam01.npz input.mp4 -o output.mp4
```

`output.mp4` 里每一帧都有：**绿色多边形**（校正后跟随真实区域）+ **灰色细线**（原参考位置）+ 左上角状态文字（帧号、OK/ALARM、方法、偏移量、角度）。

> **每帧都画框**：输出视频始终逐帧绘制，不会有哪一帧漏画。`track_video.py` 内部用有状态的 `PolygonTracker`，自带时序平滑、置信度门控、失败回退（某帧估歪时保持上次好结果不乱跳）、大漂移告警。

典型输出（终端日志为中文，视频内叠加文字为英文——OpenCV 内置字体不支持中文渲染）：

```
[帧 11] 偏离标准位：最大角位移 61.6px > 15.0px | 大漂移 62px，疑似平移/被撞，建议人工确认是否重新基准化
[帧 12] 偏离标准位：最大角位移 67.0px > 15.0px | 大漂移 67px，疑似平移/被撞，建议人工确认是否重新基准化
完成：处理 15 帧，其中 12 次重估触发告警。
已保存标注视频: output.mp4
```

### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `cache` | — | 标定缓存 `.npz`（对首帧/标准位标定得到） |
| `video` | — | 输入视频路径 |
| `-o, --out` | 无 | 输出标注视频 `.mp4`（与 `--show` 至少给一个） |
| `--every` | `1` | 每 N 帧才重估一次 H，中间帧复用上次结果继续画（省算力） |
| `--show` | 关 | 实时预览窗口（按 `q` 退出） |

> **`--every` 怎么选**：持续缓慢旋转的视频建议保持默认 `--every 1`（逐帧重估，框跟得最紧）；固定机位、偶发抖动可调大（如 `--every 5`）大幅省 CPU——契合「偏移检测低频跑」的设计，且因为逐帧绘制，框依然每帧都在。

---

## 第三步：接入运行时（Python API）

实际部署里，把跟踪器嵌进你们的取帧/告警循环。`PolygonTracker` 是**有状态**的：内置置信度门控、时序平滑、失败回退、大漂移告警，并自动选用 ORB 或 ECC。

```python
import cv2
from camera_drift.calibration import load_calibration
from camera_drift.tracker import PolygonTracker, TrackerConfig

calib = load_calibration("cache/cam01.npz")
tracker = PolygonTracker(calib, TrackerConfig(drift_thresh_px=15.0))

# 你的取帧循环（建议每隔 N 秒/定时触发，而非每帧）
def on_new_frame(frame):
    upd = tracker.update(frame)

    # upd.polygon 始终是「当前应使用的多边形」：
    #   - 置信时是平滑校正后的结果；
    #   - 不置信时自动回退上次好结果，绝不乱跳。
    polygon = upd.polygon            # Kx2 float32，喂给 YOLO 判定区域

    for msg in upd.alarms:           # 偏移/被撞/失配等告警
        notify_operator(msg)

    if upd.confident and upd.angles and upd.angles.pan_deg is not None:
        a = upd.angles
        log(f"pan={a.pan_deg:+.1f} tilt={a.tilt_deg:+.1f} roll={a.roll_deg:+.1f}")
```

`update()` 返回的 `TrackUpdate` 字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `polygon` | `np.ndarray (K,2)` | **当前应使用的多边形**（平滑后或回退的上次结果） |
| `confident` | `bool` | 本帧估计是否通过置信度门控 |
| `updated` | `bool` | 多边形是否被本帧更新（= `confident`） |
| `drift` | `DriftReport` | 偏移报告（`max_disp_px` / `alarm` / `reason` 等） |
| `angles` | `AngleResult \| None` | 角度（仅 `confident` 时给） |
| `method` | `str` | 本帧采用的方法：`orb` / `ecc(cc=..)` / `none` |
| `ref_name` | `str` | 多参考帧时命中的参考帧名 |
| `alarms` | `list[str]` | 本帧所有告警文本 |

---

## 昼夜多参考帧

单张参考帧扛不住 7×24 的昼夜光照差异。做法：在**不同时段**（白天/黄昏/夜晚）各采一张标准位参考帧、各自标定，组成参考帧库；运行时对每张 ref 估一次 H，**自动择优**（内点最多者）——夜间帧会命中夜间 ref。

```bash
# 分别标定不同时段的参考帧（多边形相同，都在标准位）
uv run python calibrate.py ref_day.jpg   --polygon-file poly.json -o cache/cam01_day.npz
uv run python calibrate.py ref_dusk.jpg  --polygon-file poly.json -o cache/cam01_dusk.npz
uv run python calibrate.py ref_night.jpg --polygon-file poly.json -o cache/cam01_night.npz
```

```python
from camera_drift.multiref import CalibrationBank
from camera_drift.tracker import PolygonTracker

bank = CalibrationBank.load([
    "cache/cam01_day.npz",
    "cache/cam01_dusk.npz",
    "cache/cam01_night.npz",
])
tracker = PolygonTracker(bank)        # 用法与单张完全一致
upd = tracker.update(frame)
print(upd.ref_name, upd.method)       # 例如 cam01_night orb
```

> `PolygonTracker` 既接受单个 `Calibration`，也接受 `CalibrationBank`；单张时内部会自动包成 1 张的库，上层代码无需区分。

---

## 低纹理 / 强光照降级（ECC）

当 ORB 在大白墙、纯地面或光照剧变下**匹配不足**（内点低于门控）时，跟踪器自动改走 **ECC（增强相关系数）配准**：它不依赖特征点，对整图灰度做多尺度对齐，且判据本身对亮度/对比度变化天然不变。ECC 的相关系数 `cc`（≥ `ecc_min_cc` 才采用）作为该路径的置信度。

- ORB 内点足够 → 用 ORB（精度最高）；
- ORB 不达标但 ECC `cc` 达标 → 用 ECC（保住低纹理/软焦场景）；
- 两者都不达标 → **保持上次好结果 + 告警人工介入**（绝不乱跳）。

ECC 是默认开启的，无需额外配置；可用 `TrackerConfig(use_ecc=False)` 关闭，或调 `ecc_min_cc`。

> 再往上的升级（v2）：夜间/低纹理仍失配时换学习型匹配 SuperPoint+SuperGlue / LoFTR——只需替换估计层，保护逻辑不动。当前版本不含该项（需重模型依赖）。

---

## 配置参数速查

`TrackerConfig`（`camera_drift/tracker.py`）——这些阈值应按现场相机分辨率/抖动情况调，默认值是合理起点：

| 参数 | 默认 | 说明 |
|------|------|------|
| `min_inliers` | `40` | 置信度门控：ORB 内点数下限 |
| `min_inlier_ratio` | `0.5` | 置信度门控：内点/好匹配比率下限 |
| `max_reproj_err` | `3.0` | 置信度门控：内点重投影误差上限(px) |
| `smooth_alpha` | `0.3` | 多边形 EMA 平滑系数（越小越平滑/越滞后） |
| `drift_thresh_px` | `15.0` | 正常偏移告警阈值(px) |
| `big_drift_thresh_px` | `60.0` | 大漂移告警阈值（疑似平移/被撞） |
| `use_ecc` | `True` | ORB 不达标时是否启用 ECC 降级 |
| `ecc_min_cc` | `0.6` | ECC 路径置信度门控：相关系数下限 |

---

## 输出与告警解读

| 告警文本（出现在 `upd.alarms`） | 含义 / 建议 |
|------|------|
| `偏离标准位：最大角位移 X px > Y px` | 相机偏了，多边形已自动校正；按需通知人工 |
| `大漂移 X px，疑似平移/被撞，建议人工确认是否重新基准化` | 位移异常大，可能不是纯旋转（平移/被撞），单 H 可能不准，建议人工确认是否重采参考帧 |
| `H 不可靠（ORB 失配且 ECC 未达标）：保持上次好结果，需人工介入` | 画面变化过大/夜间失配/严重遮挡，已回退上次结果，需人工介入 |

角度部分：
- `roll`（绕光轴/画面内旋转）总能给；
- `pan/tilt`（绕 x/y 轴）需要自标定出焦距 `f`；**纯画面内旋转时 f 无法自标定**（数学本质，非 bug），此时退化为只报 `roll_simple`，pan/tilt 不可分离。真实相机的 pan/tilt 会带来透视畸变，f 才估得出。

---

## 缓存文件格式

`.npz`（`np.savez_compressed`，**不使用 pickle**，`allow_pickle=False` 安全加载），字段：

| 键 | 说明 |
|----|------|
| `ref_frame` | 参考帧图像（原始，BGR 或灰度） |
| `keypoints` | ORB 关键点，打包成 `N×7` 数组（x,y,size,angle,response,octave,class_id） |
| `descriptors` | ORB 描述子 `N×32` uint8（低纹理时为空 `0×32`） |
| `polygon` | 多边形 `K×2` float32（参考帧坐标系） |
| `nfeatures` | 标定时的 ORB 特征数 |

用 `camera_drift.calibration.load_calibration(path)` 读取，得到 `Calibration` 对象。

---

## 模块结构

```
camera_drift/
├─ features.py      ORB 检测器 + 关键点序列化（单点定义，标定/估H 共用同参数）
├─ calibration.py   标定数据生成 / 保存 / 读取（低纹理容错）
├─ homography.py    估 H 主链路 + 多边形投影 + 内点重投影误差
├─ angles.py        自标定焦距 + 旋转分解 pan/tilt/roll
├─ drift.py         偏移量检测与告警
├─ multiref.py      多参考帧库 + 择优（应对昼夜）
├─ ecc.py           金字塔 ECC 降级（低纹理/强光照）
└─ tracker.py       全自动校正 + 估计降级 + 四道保护（总集成）

calibrate.py        命令行：标定（导入参考帧/视频首帧 + 多边形 → 缓存）
track.py            命令行：单帧演示（估 H → 报偏移/角度 → 画多边形）
track_video.py      命令行：整段视频逐帧校正 → 输出标注视频
main.py             环境冒烟检查
```

---

## 已知约束与注意事项

- **角度符号需现场标定一次**：图像 y 轴朝下，估出的角度符号方向需拿真实相机转一个已知方向核对一次，确保告警里「左偏/右偏」对得上。
- **平移是例外**：方案的数学前提是相机**绕光心旋转**，此时单 H 严格成立。发生**平移**会产生视差，单 H 不再严格——幅度小仍是好近似，**大幅平移/被撞应走告警让人复位**，不要硬套 H（`big_drift_thresh_px` 即为此设）。
- **运行频率**：偏移检测不必每帧跑，每隔 N 秒/定时触发即可，建议用缩小图，开销很小。动态前景（人/车走动）不影响——会被 RANSAC 当外点剔除。
- **无法用棋盘格标定相机内参 K**：焦距走自标定/FOV 估算，估不准时 pan/tilt 退化为像素漂移报警。
```
