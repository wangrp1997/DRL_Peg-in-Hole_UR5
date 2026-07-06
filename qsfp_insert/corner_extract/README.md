# corner_extract

角点 provider（`--corners gt|yolo|deeplsd`）+ 仿真自动标注 + YOLO Pose 训练（真机数据另采，不做 sim2real）。

## 角点来源（visp_flow）

| 参数 | 说明 |
|------|------|
| `--corners gt` | 仿真 GT 投影（默认） |
| `--corners yolo` | YOLO Pose 推理（待训） |
| `--infer-corner0` | 仅角 1–3 可见，角 0 平行四边形补全 |

## 仿真采集（计划）

眼在手上：**peg + 相机固定**，动的是孔在图像里的位置。

### 正样本（孔在画内、可对齐）

| 项 | 范围 |
|----|------|
| `hole_xy` | `HOLE_X_RANGE` × `HOLE_Y_RANGE` |
| standoff Z | `COARSE_STANDOFF`(35mm) → `ALIGN_Z_NOMINAL`(6mm) 均匀随机 |
| 6D 扰动 | 略大于 `run_eval` 的 `PERTURB_*`（xy/rpy 稍宽） |
| 标注 | 孔 + 轴各 4 角；不可见角 `v=0` |
| 保留 | peg 至少 3 角可见 |

### 负样本（孔出画 / 不可检）

| 项 | 范围 |
|----|------|
| standoff Z | 高于 `COARSE_STANDOFF`（如 40–55mm） |
| 或 xy | 大偏移，使孔口移出视野 |
| 标注 | 孔 4 角全 `v=0`（或该帧不写 hole 行）；peg 仍标 |

**正负比例：** 建议 **正:负 ≈ 4:1～5:1（负样本约 15–20%）**——纯 positive 也能训，但 coarse 搜孔阶段易在背景/peg 上误检孔角；少量负样本抑制假阳性，不必超过 30%。

### 采集脚本（待实现）

```bash
# headless，输出 YOLO pose 数据集（images/ + labels/ + data.yaml）
python qsfp_insert/corner_extract/collect_sim.py \
  --count 4000 --seed 42 \
  --pos-ratio 0.8 \
  --out qsfp_insert/corner_extract/datasets/sim_wrist2
```

| 参数 | 说明 |
|------|------|
| `--count` | 总帧数（正+负） |
| `--pos-ratio` | 正样本占比，默认 0.8 |
| `--out` | 数据集根目录（建议 gitignore） |

单帧流程：`load_scene` → 随机 hole_xy → IK 到目标 standoff → 6D 扰动 → `capture_teach_gray` 存 PNG → `gt_image_keypoints` 写 label → 校验 → 下一帧。

## 训练（后续）

```bash
pip install ultralytics
yolo pose train data=corner_extract/datasets/sim_wrist2/data.yaml model=yolo11n-pose.pt epochs=100
```

仿真只验 pipeline；**真机角点需真机图单独标、单独训/微调**，权重路径接到 `--corners yolo`。
