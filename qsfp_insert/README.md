# qsfp_insert

QSFP-DD 方孔 + 矩形轴，PyBullet 物理验证（独立于 `rlenv.py`）。

## 运行

```bash
python qsfp_insert/demo/servo_align.py --gui
python qsfp_insert/demo/servo_align.py --gui --wrist_cam [--opencv_render] [--insert]
python qsfp_insert/demo/servo_align.py --gui --fixed_cam [--opencv_render] [--insert]
python qsfp_insert/demo/ur5_insert.py --gui [--wrist_cam] [--fixed_cam] [--opencv_render]
python qsfp_insert/demo/fixed_camera_demo.py --gui [--seed 42] [--draw_keypoints] [--align]
python qsfp_insert/demo/servo_align_eval.py --episodes 10 --seed 42 [--insert]
```

参数：`--gui` 仅 3D；`--wrist_cam` / `--fixed_cam` 开对应 GUI 角预览（须 `--gui`）；`--opencv_render` 再开 OpenCV 横排窗（须指定相机）；`--insert` 对准后下插。

对准容差见 `constants.py`。

## 目录

| 路径 | 说明 |
|------|------|
| `demo/` | 可运行入口脚本 |
| `sim/` | PyBullet 场景、外置相机、笛卡尔伺服 |
| `vision/` | GT 角点、图像 overlay、后续视觉对准 |
| `constants.py` / `geometry.py` | 尺寸与判据 |
| `urdf/` | 孔板、治具、轴、UR5、外置相机 |
