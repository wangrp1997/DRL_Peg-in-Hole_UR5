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
python qsfp_insert/visp_flow/run_align.py --gui [--coarse-method kabsch|ibvs]
python qsfp_insert/visp_flow/run_eval.py --episodes 10 --seed 42 [--coarse-method kabsch]
```

## 目录

| 路径 | 说明 |
|------|------|
| `demo/` | 早期 demo（fixed_cam 角点伺服等） |
| `visp_flow/` | **ViSP baseline**（wrist2，无近似） |
| `sim/` | PyBullet 场景、相机、笛卡尔伺服 |
| `vision/` | GT 角点、几何对准 |
| `third_party/visp/` | ViSP 官方源码 |
