# qsfp_insert

QSFP-DD 方孔 + 矩形轴，PyBullet 物理验证（独立于 `rlenv.py`）。

## demo运行

```bash
python qsfp_insert/demo/servo_align.py --gui
python qsfp_insert/demo/servo_align.py --gui --wrist_cam2 [--opencv_render] [--insert] [--save_target]
python qsfp_insert/demo/servo_align.py --gui --wrist_cam [--opencv_render] [--insert]   # 旧 camera_link 腕部相机
python qsfp_insert/demo/servo_align.py --gui --fixed_cam [--opencv_render] [--insert]
python qsfp_insert/demo/servo_align.py --save_target   # 无 GUI，对准后 wrist_camera2 存 I*
python qsfp_insert/demo/ur5_insert.py --gui [--wrist_cam] [--fixed_cam] [--opencv_render]
# 眼在手外demo
python qsfp_insert/demo/fixed_camera_demo.py --gui [--seed 42] [--draw_keypoints] [--align]
# 眼在手demo
python qsfp_insert/demo/wrist_camera_demo.py --gui [--seed 42] [--draw_keypoints] [--move]
python qsfp_insert/demo/servo_align_eval.py --episodes 10 --seed 42 [--insert]
```

参数：`--gui` 仅 3D；`--wrist_cam2` **DVS 眼在手上相机**（挂 `ee_link`，对准位视角 ≈ fixed_cam，随腕运动）；`--wrist_cam` 为 URDF 侧向 `camera_link`；`--fixed_cam` 眼在手外；以上预览须 `--gui`；`--opencv_render` 再开 OpenCV 横排窗；`--insert` 对准后下插；`--save_target` 对准成功后用 **wrist_camera2** 抓 640×480 灰度 PNG（+ JSON）到 `teach/dvs_targets/`。

对准容差见 `constants.py`。

## 伺服对齐

固定外置相机 + GT 角点（暂代检测网络）：**IK 到 standoff 初始位 → 随机 6D 扰动 → 角点几何伺服对准**。

```bash
python qsfp_insert/demo/corner_servo_align.py --gui --seed 42 [--align-method kabsch|ibvs|dvs] [--insert] [--infer-corner0]
python qsfp_insert/demo/corner_servo_eval.py --episodes 10 --seed 42 [--align-method kabsch|ibvs|dvs] [--insert] [--infer-corner0]
```

`--insert` 对准后继续沿 −Z 下插；`--infer-corner0` 仅检测角点 1–3，用平行四边形补 0 号点（overlay 洋红）再参与伺服/插入。

`--align-method kabsch`（默认）：匹配角点 + 已知孔/轴矩形尺寸做 **planar PnP**，得完整 6D 误差。  
`--align-method ibvs`：眼在手外 IBVS（角点像素 + \(J_{img}\)）；远距 PnP 粗调、近距 IBVS 精调。  
`--align-method dvs`：**示教一次**对准位姿 ROI 为目标图 \(I^*\)，用 SSD/ECC 直接图像伺服，**控制环不依赖角点**。参考：Collewet–Marchand–Chaumette, *Photometric Visual Servoing*, IEEE TRO 2011；ViSP [`photometricVisualServoing.cpp`](https://visp-doc.inria.fr/doxygen/visp-3.6.0/photometricVisualServoing_8cpp-example.html) / `vpFeatureLuminance`；本实现用 OpenCV ECC 近似 ViSP template SSD。

初始位姿与 `fixed_camera_demo` 相同，用 **IK 一步到位**（`move_tip_to_standoff`）；扰动与对准阶段才走笛卡尔速度伺服。

## 目录

| 路径 | 说明 |
|------|------|
| `demo/` | 可运行入口脚本 |
| `sim/` | PyBullet 场景、外置相机、笛卡尔伺服 |
| `vision/` | GT 角点、overlay、`align.py` 角点误差、`corner_servo.py` 闭环 |
| `constants.py` / `geometry.py` | 尺寸与判据 |
| `urdf/` | 孔板、治具、轴、UR5、外置相机 |
