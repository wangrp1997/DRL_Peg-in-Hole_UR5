# 几何角点视觉伺服（简版）

> 与 IBVS 的区别：**不算图像交互矩阵 \(\mathbf{L}\)**，直接用 **P 角点 vs H 角点** 算对准误差，再发笛卡尔速度。  
> 前提：每帧能检出 8 个像素点（孔 H0–H3，轴 P0–P3）。

---

## 1. 一句话流程

```
检角点 → 匹配 P↔H → 算几何误差 e → 比例控制 v=K·e → 臂雅可比转关节速度
```

**不需要：** \({}^{f}\mathbf{T}_{p}\)（peg 相对法兰）、手 URDF、图像雅可比 \(\mathbf{L}\)。

---

## 2. 输入：四个角点（各两组）

固定外置相机一帧图像里：

| 名称 | 含义 | 来源 |
|------|------|------|
| \(\mathbf{s}_{H,j}=(u_{H,j},\,v_{H,j})\) | 孔口第 \(j\) 角像素 | 网络 / GT |
| \(\mathbf{s}_{P,i}=(u_{P,i},\,v_{P,i})\) | peg 尖第 \(i\) 角像素 | 网络 / GT |

匹配关系（方孔）：P0↔H0，P1↔H1，P2↔H2，P3↔H3（同序四角）。

---

## 3. 从角点算误差（几何法核心）

### 3.1 图像平面（2D，最常用）

**平移误差** — 两组角点质心之差：

\[
\bar{\mathbf{s}}_H = \frac{1}{4}\sum_{j=0}^{3}\mathbf{s}_{H,j}, \qquad
\bar{\mathbf{s}}_P = \frac{1}{4}\sum_{i=0}^{3}\mathbf{s}_{P,i}
\]

\[
\Delta u = \bar{u}_P - \bar{u}_H, \qquad \Delta v = \bar{v}_P - \bar{v}_H
\]

**旋转误差（绕插入轴 / 图像法向）** — 两条边方向之差，例如：

\[
\theta_H = \mathrm{atan2}(v_{H,2}-v_{H,0},\, u_{H,2}-u_{H,0})
\]
\[
\theta_P = \mathrm{atan2}(v_{P,2}-v_{P,0},\, u_{P,2}-u_{P,0})
\]
\[
\Delta\psi = \mathrm{wrap}(\theta_P - \theta_H)
\]

**高度 / standoff（可选）** — 若已知孔在图像中的尺度或深度：

- 比较 peg 与 hole 边长像素比 → 远近；
- 或用深度图在角点处取 \(Z\)，换算 standoff。

汇总成误差向量（示例，与 `alignment_twist` 接口一致）：

\[
\mathbf{e} =
\begin{bmatrix}
\Delta x \\ \Delta y \\ \Delta z \\ \Delta roll \\ \Delta pitch \\ \Delta\psi
\end{bmatrix}
\]

其中 \(\Delta x,\Delta y\) 由 \((\Delta u,\Delta v)\) 经 **固定相机标定 + 当前高度** 映到世界/插入平面（近似尺度 \(k\) mm/px 即可起步）。

### 3.2 3D（有标定 + 深度时）

把每个角点反投影到 3D，对匹配点对做 **Kabsch / Procrustes**，得 peg 相对 hole 的 6D 位姿差 —— 仍是几何误差，不是 IBVS。

---

## 4. 控制律

**比例笛卡尔伺服**（当前代码 `cartesian_control.alignment_twist`）：

\[
\mathbf{v}_a = \mathbf{K}\,\mathbf{e}
\]

\[
\begin{aligned}
v_x &= -k_{xy}\,\Delta x, & v_y &= -k_{xy}\,\Delta y, & v_z &= -k_z\,(\text{standoff} - z^*) \\
\omega_x &= -k_r\,\Delta roll, & \omega_y &= -k_r\,\Delta pitch, & \omega_z &= -k_r\,\Delta\psi
\end{aligned}
\]

各项单独限幅 → `apply_cartesian_velocity` → **臂几何雅可比** \( \mathbf{J}_{tip} \) → 关节速度 \(\dot{\mathbf{q}}\)。

> 这里的雅可比是 **机器人 URDF 的 tip 雅可比**，不是 IBVS 的 \(\mathbf{L}\)。

**收敛判据**（`geometry.is_aligned`）：\(|\Delta x|,|\Delta y|\)、standoff、\(|\Delta roll|,|\Delta pitch|,|\Delta\psi|\) 进 tol 带。

---

## 5. 数值例子（图像 2D）

孔质心 \(\bar{\mathbf{s}}_H=(400,\,300)\)，peg 质心 \(\bar{\mathbf{s}}_P=(412,\,288)\)：

\[
\Delta u=+12\text{ px},\quad \Delta v=-12\text{ px}
\]

设标定尺度 \(k=0.05\,\text{mm/px}\)：

\[
\Delta x \approx +0.6\,\text{mm},\quad \Delta y \approx -0.6\,\text{mm}
\]

控制：\(v_x = -k_{xy}\cdot 0.6\,\text{mm}\)（往左收），\(v_y = -k_{xy}\cdot(-0.6)\)（往下收）—— **直到下一帧 \(\Delta u,\Delta v \to 0\)**。

若还有 \(\Delta\psi=+2°\)，加 \(\omega_z = -k_r \cdot 2°\)。

---

## 6. 为什么抓取手势不影响控制律？

误差 **只从图像里 P 和 H 的相对关系** 来：

\[
\mathbf{e} = f(\mathbf{s}_P,\,\mathbf{s}_H)
\]

\(\mathbf{s}_P\) 已是 peg **真在相机里哪**；手势不同只改 **初值** \(\mathbf{e}(0)\)，不改公式 \( \mathbf{v}=\mathbf{K}\mathbf{e} \)。

详见 [`visual_servo_grasp_invariance.md`](visual_servo_grasp_invariance.md)。

---

## 7. 与 IBVS 对比

| | **几何角点法（本文）** | **IBVS** |
|---|------------------------|----------|
| 误差 | P 与 H 质心/边向/位姿差 | 像素特征 \(\mathbf{s}_P-\mathbf{s}_H\) + 交互矩阵 |
| 需要 \(\mathbf{L}\) | 否 | 是 |
| 需要 \({}^{f}\mathbf{r}_{p}\) | 否 | 进 \(\mathbf{J}_p\)（可近似） |
| 需要手 URDF | 否（手指锁死） | 否（同上） |
| 直观性 | 高（对准 = 角点重合） | 中 |

---

## 8. 与当前代码

| 步骤 | 文件 | 状态 |
|------|------|------|
| 角点 → 像素 | `vision/corners.py` | GT 投影（仿真） |
| 图像 overlay | `vision/overlay.py` | 已有 |
| 角点 → 误差 | `vision/align.py` | **待接**（现用 `geometry.alignment_metrics` 世界系 GT） |
| 控制律 | `cartesian_control.alignment_twist` | 已有 |
| demo | `demo/fixed_camera_demo.py --align` | 已有（误差来自 FK，视觉仅显示） |

**下一步：** `align.py` 用检出的 \(\mathbf{s}_P,\mathbf{s}_H\) 算 \(\mathbf{e}\)，替换 FK 的 `alignment_metrics`，闭环即 **真·几何视觉伺服**。

---

## 9. 汇报一句话

> **知道四个角点像素就够了：算 peg 相对 hole 偏多少 → 按比例动臂把偏差收掉；不用 IBVS 的 \(\mathbf{L}\)，也不管手怎么抓。**
