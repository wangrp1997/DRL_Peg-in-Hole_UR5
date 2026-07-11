# il_flow

模仿学习管线（仿真）：**无 wrist_cam1**，**wrist_camera2** + **fixed_camera2**（孔轴对侧，绿块）。

## 相机预览 + 笛卡尔对准

```bash
python qsfp_insert/il_flow/run_cartesian_demo.py --gui --standoff-mm -1 
python qsfp_insert/il_flow/run_cartesian_demo.py --gui --opencv   # 额外 OpenCV 双窗（慢）
```

默认：**孔板不透明**；仅 3D 场景（**无** Bullet 右下角相机预览；要预览加 `--opencv`）。

调 fixed2：`il_flow/il_constants.py` → `FIXED_CAM2_EYE_OFFSET` / `FIXED_CAM2_ROLL_DEG`。

## 数据采集（双相机 RGB）

随机孔位 + 6D 扰动 → **角点 Kabsch 专家**（默认，target **3.0–3.8mm** 随机）；失败或 physics/goal 角点校验不通过则不保存。

```bash
python qsfp_insert/il_flow/collect_il.py --episodes 100 --overwrite            # 同上，批量采
# GUI 长连仍崩：每条单独起进程（100 次 × 1 条）
OVERWRITE=1 bash qsfp_insert/il_flow/run_collect_loop.sh 100 42
python qsfp_insert/il_flow/collect_il.py --episodes 2 --expert gt --overwrite
```

无显示器服务器可用虚拟 framebuffer（仍走 GUI 渲染，**不要**加 `--headless`）：

```bash
xvfb-run -a python qsfp_insert/il_flow/collect_il.py --episodes 100 --overwrite
```

`--headless`（EGL）在本场景仍可能有基座/近处发黑，与 `corner_extract/collect_sim.py` 文档一致：**训练用图请用 GUI 路径**。

### 目录格式（`il_flow_v1`，便于转 Robomimic / LeRobot）

```
dataset/raw/
  meta/
    info.json              # 相机名、action 维数、fps
    episodes.jsonl         # 每条 episode 摘要
    collect_summary.json
  episodes/
    episode_000000/
      meta.json
      timesteps.jsonl      # action[6] + qpos + peg_tip pose + peg_in_hole
      wrist_camera2/rgb.mp4
      wrist_camera2/goal_rgb.png
      fixed_camera2/rgb.mp4
      fixed_camera2/goal_rgb.png
```

- **RGB 视频**：每相机独立目录（后续 LeRobot 的 `observation.images.<cam>` 一一对应）
- **`timesteps.jsonl`**：`action` = 笛卡尔 6D twist；`I_goal` = 同 episode 下 `{cam}/goal_rgb.png`
- **框架**：暂未绑定；`info.json` 里 `compatible_with` 列出可转换目标，后续定 Robomimic / LeRobot / Diffusion Policy 再写 converter

### 渲染：GUI vs headless

| 模式 | 渲染器 | 训练数据 |
|------|--------|----------|
| **默认（推荐）** | `p.GUI` + `ER_BULLET_HARDWARE_OPENGL` | 正常，与 `collect_sim` / ViSP teach 同路径 |
| `--headless` | `p.DIRECT` + EGL 插件 | 本场景仍易有黑色伪影，**不推荐**采 IL |
| 无屏服务器 | `xvfb-run -a` + 默认 GUI | 等价于 GUI，无需 `--headless` |

批量采集时 **PyBullet 只连接一次**，episode 之间用 `resetSimulation` 换场景（同 `collect_sim.py`），避免每条都 disconnect/reconnect 导致 GUI 跑飞、机械臂变线。

`--headless` 保留仅供调试；正式 dataset 请去掉该 flag。

## 转 LeRobot

需 **Python ≥3.12** 的 LeRobot 环境（如 `conda activate lerobot312`，`pip install -e ".[dataset]"`）。

全量 100 条：

```bash
python qsfp_insert/il_flow/convert_to_lerobot.py \
  --raw-dir qsfp_insert/il_flow/dataset/raw \
  --out-root qsfp_insert/il_flow/dataset/lerobot \
  --repo-id local/qsfp_il --overwrite
```

转换后 **`images/` 为空是正常的**（临时帧已编码进 `videos/*.mp4`）；训练读 `videos/` + `data/` 即可。

## 训练（LeRobot Diffusion）

```bash
conda activate lerobot312   # pip install -e ".[training]" 若缺训练依赖
# 服务器训练完整指令
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=local/qsfp_il \
  --dataset.root=/mnt/ssd/datasets/qsfp_il/lerobot \
  --policy.type=diffusion \
  --policy.push_to_hub=true \
  --policy.repo_id=rpwang/qsfp_il_dp \
  --wandb.enable=true \
  --wandb.project=qsfp_il \
  --batch_size=32 \
  --output_dir=/mnt/ssd/checkpoints/qsfp_il_ckpt/diffusion
```

默认 checkpoint 写在**当前目录** `outputs/train/日期/时间_diffusion/`；可 `--output_dir=qsfp_insert/il_flow/outputs/train` 指定到项目内。
