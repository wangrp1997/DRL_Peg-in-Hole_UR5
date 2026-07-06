# qsfp_insert

QSFP-DD 方孔 + 矩形轴，PyBullet 物理验证（独立于 `rlenv.py`）。

## demo运行

```bash
python qsfp_insert/demo/servo_align.py --gui
python qsfp_insert/demo/servo_align.py --gui --wrist_cam2 [--opencv_render] [--insert] [--save_target]
python qsfp_insert/demo/corner_servo_align.py --gui --seed 42 [--align-method kabsch|ibvs]
python qsfp_insert/demo/corner_servo_eval.py --episodes 10 --seed 42 [--align-method kabsch|ibvs]
```

`demo/` 为早期验证；**baseline 见下方 ViSP 流程**（无 OpenCV/自写控制律近似）。

## ViSP baseline（`visp_flow/`）

**仅 `wrist_camera2`（眼在手上）**：IK standoff → **笛卡尔对准仅采 \(I^*\)** → **回 standoff** → 6D 扰动 → 停 3s 第一阶段（粗对准，角点 ON）→ 停 3s 第二阶段（ViSP DVS，角点 OFF）。

渲染：`sim/wrist2_render.py`（GUI 用 HARDWARE + 空帧缓存，防 OpenCV/角预览闪黑）。

| 阶段 | 官方 ViSP 来源 | 实现 |
|------|----------------|------|
| 粗 IBVS | `servoUniversalRobotsIBVS.cpp` | Python `visp.vs.Servo` + `FeaturePoint`，`EYEINHAND_CAMERA` |
| 粗 kabsch | 几何 PnP（非 ViSP） | `align.py` planar PnP |
| 细 DVS | `photometricVisualServoing.cpp` | `native/photometric_servo.cpp`（原样 `vpFeatureLuminance`+`vpServo`） |

相机速度 → 机器人：ViSP `vpVelocityTwistMatrix`（同 UR example `setVelocity(CAMERA_FRAME)`）。

### 编译（低配机器：最小模块 + 低并行）

**不要用 `make -j$(nproc)`**，核数拉满容易卡死；建议 **`make -j2`**（内存够再用 `-j4`）。

若 C++ 已编过（`third_party/visp/build/lib/libvisp_core.so` 存在），**跳过 cmake**，在 `build/` 里续编并装 Python 即可。

```bash
# 1) ViSP C++ + Python 绑定（仅 core / visual_features / vs）
cd third_party/visp
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_PYTHON3=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_DEMOS=OFF \
  -DBUILD_JAVA=OFF \
  -DUSE_PCL=OFF \
  -DUSE_OGRE=OFF \
  -DUSE_COIN3D=OFF \
  -DUSE_GTK=OFF \
  -DUSE_V4L2=OFF
make -j2
pip install ../modules/python/stubs   # 装完才能 import visp
python3 -c "import visp.core; import visp.vs; print('visp ok')"

# 2) 光度 DVS native（vpFeatureLuminance，Python 未暴露）
cd qsfp_insert/visp_flow/native
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$HOME/Documents/DRL_Peg-in-Hole_UR5/third_party/visp/build
make -j2
export PYTHONPATH=$PWD:$PYTHONPATH
python3 -c "import photometric_servo; print('photometric ok')"
```

续编时在同一 `build/` 目录执行 `make -j2` 即可，无需 `make clean`。

### 运行 / 评测

```bash
python qsfp_insert/visp_flow/run_align.py --gui [--opencv] [--coarse-method kabsch|ibvs] [--insert] [--corners gt|deeplsd|yolo] [--infer-corner0]
python qsfp_insert/visp_flow/run_eval.py --episodes 10 --seed 42 [--coarse-method kabsch] [--insert] [--corners gt|deeplsd|yolo] [--infer-corner0]
```

### 角点来源（`corner_extract/`）

| 参数 | 说明 |
|------|------|
| `--corners gt` | 仿真 3D 投影角点（默认，真机待换 yolo/deeplsd） |
| `--corners deeplsd` / `yolo` | 占位，未实现 |
| `--infer-corner0` | 仅角 1–3 可见，角 0 平行四边形补全，验 3+1→Kabsch |

```bash
# 默认 GT 四角
python qsfp_insert/visp_flow/run_align.py --seed 42 --corners gt --coarse-method kabsch
# 仿真验「3 点 + 补第 4 角」（GUI 下角 0 洋红）
python qsfp_insert/visp_flow/run_eval.py --episodes 10 --seed 42 --corners gt --infer-corner0
```

## 模板 coarse（`template_flow/`）

**独立目录，不修改 `visp_flow/`、`vision/corner_servo.py`。** 复用 `third_party/accelerated_features`（XFeat + `realtime_demo.py` 的 MNN / `match_xfeat` / homography 流程）做**运行时无 GT** 的角点 coarse，再调用 `run_kabsch_coarse`（与 `vision/corner_servo` 相同循环）。**固定相机仅 kabsch；IBVS coarse 需 wrist2。**

### `--camera` 两种模式（**仅 `run_eval` / `run_diag` 走 XFeat**）

| 参数 | 相机 | standoff | 示教 | 伺服（headless，`run_eval`） |
|------|------|----------|------|------------------------------|
| `fixed` | `fixed_cam` | `CORNER_SERVO_STANDOFF` | I* + 四角标定（仿真 GT 模拟点击）+ **XFeat ROI 描述子** | **孔**：示教 UV 不变<br>**peg**：XFeat ROI homography + `infer0` |
| `wrist2` | `wrist_camera2` | `COARSE_STANDOFF` | 同上 | **peg**：示教 UV 不变<br>**孔**：XFeat ROI homography（搜索窗不上扩进 peg 区） |

> **`run_align --gui --opencv`**：OpenCV 叠加 **XFeat 匹配角点**（每 `MATCH_EVERY_GUI` 步重匹配，中间帧保留上一帧 XFeat 结果）。示教阶段构建 ROI 模板。  
> **`run_eval`**（headless）：纯 XFeat，中间步保留上一帧匹配结果，**不用 GT/几何补帧**。

依赖：`third_party/accelerated_features/weights/xfeat.pt`（`run_eval` 首次运行加载）。

### 运行

```bash
# GUI：OpenCV 显示 XFeat 角点（与 eval 同算法，无几何补帧）
python qsfp_insert/template_flow/run_align.py --gui --opencv --seed 42 --camera fixed --coarse-method kabsch
python qsfp_insert/template_flow/run_align.py --gui --opencv --seed 42 --camera wrist2 --coarse-method kabsch

# headless 评测 / 单帧 diag
python qsfp_insert/template_flow/run_eval.py --episodes 10 --seed 42 --camera fixed --coarse-method kabsch
python qsfp_insert/template_flow/run_eval.py --episodes 10 --seed 42 --camera wrist2 --coarse-method kabsch
python qsfp_insert/template_flow/run_diag.py --seed 42 --camera fixed   # 单帧角点 vs GT
python qsfp_insert/template_flow/run_diag.py --seed 42 --camera wrist2
```

GUI 退出与 `visp_flow/run_align.py --gui --opencv` 相同：关 PyBullet；`--opencv` 时 `os._exit`。

**`run_eval`**：仅 XFeat 角点 + 上一帧缓存，**无 GT/几何补帧**（此前 10/10 因补帧≈GT，已移除）。

**纯 XFeat 单帧误差（`run_diag`，seed=42，扰动后）：**

| 模式 | 示教侧 max err | XFeat 侧 max err |
|------|----------------|------------------|
| `fixed` | hole 0 px | peg **~37 px** |
| `wrist2` | peg ~3 px | hole **~32 px** |

**`run_eval` coarse 成功率（seed=42, kabsch, 10 轮，纯 XFeat 无补帧）**：fixed / wrist2 均 **0/10**。GT baseline：`visp_flow/run_eval.py` 10/10。

结果 JSON → `outputs/template_flow_{camera}_kabsch_*.json`

## 目录

| 路径 | 说明 |
|------|------|
| `demo/` | 早期 demo（fixed_cam 角点伺服等） |
| `template_flow/` | **模板 coarse**（无运行时 GT，无 DVS） |
| `corner_extract/` | 角点 provider（`--corners`） |
| `visp_flow/` | **ViSP baseline**（wrist2，无近似） |
| `sim/` | PyBullet 场景、相机、笛卡尔伺服 |
| `vision/` | GT 角点、几何对准 |
| `third_party/visp/` | ViSP 官方源码 |
