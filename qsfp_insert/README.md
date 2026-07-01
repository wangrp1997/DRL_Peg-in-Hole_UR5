# qsfp_insert

QSFP-DD 方孔 + 矩形轴，PyBullet 物理验证（独立于 `rlenv.py`）。

## 运行

```bash
python qsfp_insert/demo/physics_insert.py --gui   # 仅 peg+孔，测插入物理
python qsfp_insert/demo/ur5_insert.py --gui       # UR5+桌+孔，IK 下插
python qsfp_insert/demo/servo_align.py --gui     # UR5 笛卡尔速度伺服对准孔口（不插入）
```

对准容差见 `constants.py`（文献暂定值，非相机假设）。

## 目录

| 路径 | 说明 |
|------|------|
| `demo/` | 脚本；`cartesian_control.py` 为 6D Jacobian 速度伺服 |
| `constants.py` | 尺寸与对准/插入容差 |
| `geometry.py` | 判据 |
| `urdf/` | 孔板、治具基座、轴、UR5 变体 |
