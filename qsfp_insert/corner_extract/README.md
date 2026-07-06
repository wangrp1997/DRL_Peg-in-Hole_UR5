# corner_extract

角点 provider（`--corners gt|yolo|deeplsd`）+ 仿真自动标注 + YOLO Pose 训练（真机数据另采，不做 sim2real）。

## 角点来源（visp_flow）

| 参数 | 说明 |
|------|------|
| `--corners gt` | 仿真 GT 投影（默认） |
| `--corners yolo` | YOLO Pose 推理（待训） |
| `--infer-corner0` | 仅角 1–3 可见，角 0 平行四边形补全 |

## 仿真采集（wrist2）

眼在手上：**peg + 相机固定**，孔在图像里随轨迹移动。

### 轨迹（接近真机）

参考 `demo/corner_servo_align.py`，**不用**完整 `visp_flow`（无 teach / DVS）：

1. 随机 `hole_xy`
2. IK 到 `COARSE_STANDOFF`（35 mm）
3. 随机 6D 扰动（`COLLECT_PERTURB_*`：xy **±20 mm**、z **±10 mm**、roll/pitch **±0.14 rad**、yaw **±0.20 rad**，大于 demo）
4. **Kabsch / IBVS 角点伺服** 直到对准成功
5. 成功高度随机 **3.0–3.8 mm**，peg 可能挡住 1–2 个孔角 → 训练遮挡

**暂不录「孔出画」负样本**；孔始终在视野内，部分角点可能因 peg 遮挡而 `v=0`。

### 目录结构（Ultralytics YOLO）

```
datasets/
├── sim_wrist2_raw/              # 阶段 1：原始视频（不进 data.yaml）
│   ├── videos/
│   │   ├── align_sXXXX.mp4
│   │   └── align_sXXXX.json
│   └── record_meta.json
└── sim_wrist2/                  # 阶段 2：YOLO 训练集
    ├── data.yaml
    ├── images/{train,val}/
    ├── labels/{train,val}/
    ├── previews/{train,val}/    # 角点叠图（调试用，保留在数据集内）
    └── extract_meta.json
```

### 阶段 1：录视频

```bash
# 试录一条（--overwrite 只清空 raw 目录）
python qsfp_insert/corner_extract/record_sim.py --seed 42 --overwrite

# 批量录 20 条
python qsfp_insert/corner_extract/record_sim.py --count 20 --seed 42 --overwrite

# 追加录制（已有 mp4 则跳过）
python qsfp_insert/corner_extract/record_sim.py --count 20 --seed 42
```

### 阶段 2：抽帧 + YOLO 标注

按 **整条视频** 划分 train/val（默认 80/20），避免同轨迹帧泄漏。

```bash
python qsfp_insert/corner_extract/extract_video.py --stride 10 --clear --headless
```

### 旧脚本（独立随机位姿，已弃用主流程）

`collect_sim.py` — 每帧跳随机位姿，**不等价真机连续轨迹**；保留作调试。

## 训练（后续）

```bash
pip install ultralytics
yolo pose train data=corner_extract/datasets/sim_wrist2/data.yaml model=yolo11n-pose.pt epochs=100
```

仿真只验 pipeline；**真机角点需真机图单独标、单独训/微调**，权重路径接到 `--corners yolo`。
