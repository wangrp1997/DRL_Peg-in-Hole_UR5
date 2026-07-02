# 固定相机角点视觉伺服：抓取姿态与控制律

> 汇报用简版推导。结论：**只要每帧能从图像里检出 peg / hole 角点，且对准阶段手指锁死、peg 不滑，控制律形式不变**；变的只是每次抓取后的**初始误差**。

---

## 1. 符号

| 符号 | 含义 |
|------|------|
| \(\mathbf{q}_a\) | 臂关节角（对准阶段唯一被控量） |
| \(\mathbf{q}_h\) | 手指关节角（对准阶段**固定**） |
| \({}^{w}\mathbf{T}_{f}(\mathbf{q}_a)\) | 臂末端（法兰/腕）在世界系位姿 |
| \({}^{f}\mathbf{T}_{p}(\mathbf{q}_h)\) | **抓取相关**：peg 坐标系相对法兰的固定位姿（由本次抓取手势决定） |
| \({}^{w}\mathbf{T}_{p}\) | peg 在世界系位姿 |
| \({}^{w}\mathbf{T}_{c}\) | 固定外置相机外参（标定一次） |
| \(\mathbf{P}_i\) | peg 上第 \(i\) 个角点，peg 系下常数 |
| \(\mathbf{H}_j\) | hole 上第 \(j\) 个角点，世界系下常数 |
| \(\pi(\cdot)\) | 相机投影：世界点 → 像素 \((u,v)\) |

运动链（手指不动时）：

\[
{}^{w}\mathbf{T}_{p} = {}^{w}\mathbf{T}_{f}(\mathbf{q}_a)\,{}^{f}\mathbf{T}_{p}(\mathbf{q}_h)
\]

\[
\mathbf{s}_{P,i} = \pi\!\left({}^{w}\mathbf{T}_{c}^{-1}\,{}^{w}\mathbf{T}_{p}\,\mathbf{P}_i\right), \qquad
\mathbf{s}_{H,j} = \pi\!\left({}^{w}\mathbf{T}_{c}^{-1}\,\mathbf{H}_j\right)
\]

---

## 2. 两种抓取：同一控制律，不同初值

设 hole 不动，臂关节初值相同 \(\mathbf{q}_a=\mathbf{q}_{a,0}\)，但两次抓取手指角不同：

| | 抓取 A | 抓取 B |
|---|--------|--------|
| \(\mathbf{q}_h\) | \(\mathbf{q}_h^A\)（三指包左侧） | \(\mathbf{q}_h^B\)（三指包右侧） |
| \({}^{f}\mathbf{T}_{p}\) | peg 尖在法兰下偏移 \((0,\,-5,\,120)\,\text{mm}\) | peg 尖在法兰下偏移 \((0,\,+3,\,118)\,\text{mm}\) |
| 结果 | peg 在世界里偏 **左 5 mm** | peg 在世界里偏 **右 3 mm** |

因此 \(t=0\) 时图像观测不同，例如（示意）：

- 抓取 A：\(\mathbf{s}_{P,0}^A = (412,\,288)\,\text{px}\)，孔角 \(\mathbf{s}_{H,0} = (400,\,300)\,\text{px}\)
- 抓取 B：\(\mathbf{s}_{P,0}^B = (388,\,292)\,\text{px}\)，孔角 \(\mathbf{s}_{H,0} = (400,\,300)\,\text{px}\)（孔不变）

**初值误差不同**，但下面误差定义和控制律 **同一个公式**。

---

## 3. 误差：只比较「看到的 peg」和「看到的 hole」

### 3.1 几何误差（当前仿真 / Haugaard 思路）

对角点做匹配（P0↔H0, …），在图像平面或标定后的 3D 里算 peg 相对 hole 的位姿差：

\[
\mathbf{e}_{\text{geo}} = \mathrm{Align}\!\left(\{\mathbf{s}_{P,i}\}_{i=0}^{3},\,\{\mathbf{s}_{H,j}\}_{j=0}^{3}\right)
= \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta\psi \end{bmatrix}
\]

（方孔插入还可扩展 \(\Delta z\)、roll、pitch；与 `geometry.alignment_metrics` 同类。）

**要点：** \(\mathbf{e}_{\text{geo}}\) 由 **当前帧图像检出的** \(\mathbf{s}_P,\mathbf{s}_H\) 计算，**不查** \(\mathbf{q}_h\) 或手 URDF。

### 3.2 IBVS 误差（可选写法）

堆成特征向量 \(\mathbf{s}=[u_0,v_0,\ldots]^\top\)，目标取 hole 角点：

\[
\mathbf{e}_s = \mathbf{s}_P - \mathbf{s}_H
\]

同样只依赖观测，不依赖 \(\mathbf{q}_h\)。

---

## 4. 控制律（两种抓取完全相同）

### 4.1 几何闭环（推荐，与现有代码一致）

\[
\mathbf{v}_a = \mathbf{K}\,\mathbf{e}_{\text{geo}}
\]

\(\mathbf{v}_a \in \mathbb{R}^6\)：发给臂的笛卡尔速度（或 twist）；\(\mathbf{K}\)：对角增益。

对应代码路径：`alignment_twist(dx, dy, standoff, roll, pitch, yaw)` → `apply_cartesian_velocity`。

### 4.2 IBVS 闭环（固定外置相机）

单点图像交互矩阵（Chaumette，深度 \(Z\)，焦距 \(f\)）：

\[
\mathbf{L}_i =
\begin{bmatrix}
-f/Z & 0 & u/Z & uv/f & -(f+u^2/f) & v \\
0 & -f/Z & v/Z & f+v^2/f & -uv/f & -u
\end{bmatrix}
\]

多角点：\(\mathbf{L}\) 按块对角堆叠。眼在手外组合雅可比：

\[
\dot{\mathbf{s}} = \mathbf{L}(\mathbf{s},Z)\,\,{}^{c}\mathbf{J}_{p}(\mathbf{q}_a,\,{}^{f}\mathbf{r}_{p})\,\mathbf{v}_a
\]

\[
\mathbf{v}_a = -\lambda\,\big(\mathbf{L}\,\,{}^{c}\mathbf{J}_{p}\big)^{\#}\,\mathbf{e}_s
\]

其中 \({}^{f}\mathbf{r}_{p}\) 是 peg 尖相对法兰的杠杆臂。**若 \(\mathbf{r}_{p}\) 标称值有偏**，只要每帧用 **检出的** \((u,v,Z)\) 更新 \(\mathbf{L}\)，且 peg 不滑，闭环仍会把 \(\mathbf{e}_s\to 0\)。

---

## 5. 推导：为什么 \(\mathbf{q}_h\) 不进控制律

### 5.1 观测方程

\[
\mathbf{s}_P = \mathbf{g}\!\left(\mathbf{q}_a,\,\mathbf{q}_h\right)
= \pi\!\left({}^{w}\mathbf{T}_{c}^{-1}\,{}^{w}\mathbf{T}_{f}(\mathbf{q}_a)\,{}^{f}\mathbf{T}_{p}(\mathbf{q}_h)\,\mathbf{P}\right)
\]

\(\mathbf{q}_h\) 只通过 \({}^{f}\mathbf{T}_{p}(\mathbf{q}_h)\) 影响 **peg 在图像里的位置**。

### 5.2 误差对 \(\mathbf{q}_h\) 的依赖

\[
\mathbf{e} = \mathbf{h}(\mathbf{s}_P,\,\mathbf{s}_H)
\]

hole 角点 \(\mathbf{s}_H\) 与抓取无关；\(\mathbf{s}_P\) 虽含 \(\mathbf{q}_h\)，但 **每帧从图像直接测量**，等价于：

\[
\mathbf{e}(t) = \mathbf{h}\!\left(\mathbf{g}(\mathbf{q}_a(t),\,\mathbf{q}_h^{\text{fixed}}),\,\mathbf{s}_H\right)
\]

对准阶段 \(\mathbf{q}_h^{\text{fixed}}\) 为常数 → \(\mathbf{q}_h\) **不出现在控制律公式里**，只决定 \(t=0\) 时 \(\mathbf{e}(0)\) 有多大。

### 5.3 数值走一遍（抓取 A vs B）

**目标：** \(\mathbf{e}\to 0\)（peg 角点与 hole 角点对齐）。

| 步 | 抓取 A | 抓取 B |
|----|--------|--------|
| \(t=0\) 观测 | \(\mathbf{e}(0)=[+12,\,-12,\,+2°]^\top\)（示意） | \(\mathbf{e}(0)=[-12,\,-8,\,-1°]^\top\) |
| 控制律 | \(\mathbf{v}_a = \mathbf{K}\mathbf{e}(0)\) | **同一公式** \(\mathbf{v}_a = \mathbf{K}\mathbf{e}(0)\) |
| 方向 | 臂往「右、下、逆时针」动 | 臂往「左、下、顺时针」动（由 **当前 e** 决定） |
| \(t\to\infty\) | \(\mathbf{e}\to 0\) | \(\mathbf{e}\to 0\) |

**结论：** 手势不同 → 初值 \(\mathbf{e}(0)\) 不同 → 第一步臂运动方向不同；**但律本身不变**，收敛目标都是「图像里 peg 与 hole 对齐」。

---

## 6. 什么时候才需要手 URDF？

| 场景 | 是否需要手 URDF |
|------|-----------------|
| 对准阶段只动臂，手指锁死，用网络/GT 检 peg&hole 角点 | **否** |
| 用 FK 预测 peg 尖位置（不依赖视觉） | **是**（需 \(\mathbf{q}_h\) + 手链） |
| 对准阶段要动手指做 re-grasp | **是** |
| peg 在掌心里滑移 | 视觉仍可用，但需 slip 检测 / 力控；FK 链失效 |

---

## 7. 汇报一句话

> **控制律看的是「peg 角点相对 hole 角点的误差」，不是「手指长什么样」。**  
> 灵巧手抓取只改变 peg 的初始位姿；只要角点可检、peg 不滑，同一套几何伺服 / IBVS 公式对任意 \(\mathbf{q}_h\) 都成立。

---

## 8. 与当前代码对应

```
图像/GT 角点  →  corners.gt_image_keypoints / 未来 U-Net
误差          →  geometry.alignment_metrics（或 IBVS 的 s_P - s_H）
控制律        →  cartesian_control.alignment_twist
执行          →  apply_cartesian_velocity（只动臂）
相机          →  sim/fixed_camera.py（外参固定，孔随 hole_xy 平移）
```

下一步：用检出角点替换 GT，控制律层 **无需因手势改公式**。
