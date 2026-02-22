# 项目数学原理与控制原理说明

本文档详细阐述了本项目中使用的在线微分Koopman模型预测控制（Online Differential Koopman MPC）的数学原理。该项目结合了传统的物理建模、数据驱动的Koopman算子理论以及现代深度学习框架（PyTorch）的可微分特性，实现了对非线性系统的精确控制。

## 1. 物理模型 (Physical Model)

本项目控制的对象是一个经典的质量-弹簧-阻尼系统（Mass-Spring-Damper System）。该系统是一个典型的二阶线性时不变（LTI）系统，但在实际应用中可能受到非线性因素或参数不确定的影响。

### 1.1 微分方程

系统的动力学方程由牛顿第二定律给出：

$$
m\ddot{p} + c\dot{p} + kp = u
$$

其中：
- $p$ 是物体的位置（位移）。
- $\dot{p}$ 是物体的速度。
- $\ddot{p}$ 是物体的加速度。
- $m$ 是质量（Mass）。
- $c$ 是阻尼系数（Damping coefficient）。
- $k$ 是弹簧劲度系数（Spring constant）。
- $u$ 是外部施加的控制力。

### 1.2 状态空间方程

为了便于控制设计，我们将上述二阶微分方程转换为一阶状态空间形式。定义状态向量 $x = [x_1, x_2]^T = [p, \dot{p}]^T$。

系统的连续时间状态空间方程为：

$$
\dot{x} = \begin{bmatrix} \dot{p} \\ \ddot{p} \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -\frac{k}{m} & -\frac{c}{m} \end{bmatrix} \begin{bmatrix} p \\ \dot{p} \end{bmatrix} + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} u
$$

即 $\dot{x} = A_c x + B_c u$。

### 1.3 离散化 (Discretization)

在计算机仿真中，我们需要对连续系统进行离散化。本项目采用了四阶龙格-库塔法（Runge-Kutta 4, RK4）进行高精度的数值积分。对于时间步长 $\Delta t$，状态从 $x_k$ 更新到 $x_{k+1}$ 的过程如下：

$$
\begin{aligned}
k_1 &= f(x_k, u_k) \\
k_2 &= f(x_k + 0.5 \Delta t k_1, u_k) \\
k_3 &= f(x_k + 0.5 \Delta t k_2, u_k) \\
k_4 &= f(x_k + \Delta t k_3, u_k) \\
x_{k+1} &= x_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

其中 $f(x, u)$ 为系统的连续动力学函数。

## 2. Koopman 算子理论 (Koopman Operator Theory)

Koopman算子理论提供了一种通过无限维线性算子来描述非线性动力系统的方法。

### 2.1 定义

对于一个离散时间的非线性动力系统 $x_{k+1} = F(x_k)$，Koopman算子 $\mathcal{K}$ 是一个作用在可观测函数空间（Space of observable functions）上的无限维线性算子。对于任意标量观测函数 $g: \mathbb{R}^n \to \mathbb{R}$，Koopman算子定义为：

$$
\mathcal{K} g(x_k) = g(F(x_k)) = g(x_{k+1})
$$

这表明，虽然状态 $x$ 在状态空间中的演化是非线性的，但观测函数 $g$ 在函数空间中的演化是线性的。

## 3. 扩展动态模态分解 (Extended Dynamic Mode Decomposition, EDMD)

由于Koopman算子是无限维的，我们需要通过有限维近似来在实际中应用它。EDMD是一种常用的数据驱动算法，用于从数据中估计Koopman算子的有限维近似矩阵。

### 3.1 提升函数 (Lifting Functions)

我们选择一组基函数（字典）$\Psi(x) = [\psi_1(x), \psi_2(x), \dots, \psi_N(x)]^T$ 来张成Koopman算子的不变子空间。在本项目中，使用了高达二阶的多项式基函数。对于二维状态 $x = [x_1, x_2]^T$，提升后的状态向量 $z$ 为：

$$
z = \Psi(x) = [x_1, x_2, x_1^2, x_1 x_2, x_2^2, 1]^T
$$

这里 $N=6$。

### 3.2 线性回归问题

我们将非线性系统的演化近似为提升空间中的线性系统：

$$
z_{k+1} \approx A z_k + B u_k
$$

为了求解矩阵 $A$ 和 $B$，我们收集了一组轨迹数据 $\{ (x_k, u_k, x_{k+1}) \}_{k=1}^M$。

定义数据矩阵：
$$
Z = [z_1, z_2, \dots, z_M], \quad Z' = [z_2, z_3, \dots, z_{M+1}], \quad U = [u_1, u_2, \dots, u_M]
$$

构建回归矩阵 $\Omega$：
$$
\Omega = \begin{bmatrix} Z \\ U \end{bmatrix}
$$

我们的目标是找到矩阵 $G = [A, B]$，使得目标函数最小化：

$$
\min_G \| Z' - G \Omega \|_F^2
$$

### 3.3 闭式解

上述最小二乘问题的闭式解为：

$$
G = Z' \Omega^\dagger
$$

其中 $\Omega^\dagger$ 是 $\Omega$ 的摩尔-彭若斯广义逆（Moore-Penrose Pseudoinverse）。解出 $G$ 后，即可提取出系统矩阵 $A$ 和控制矩阵 $B$。此外，还需要求解投影矩阵 $C$，使得 $x \approx C z$。

## 4. 可微分模型预测控制 (Differentiable MPC)

本项目不仅使用离线计算的Koopman模型，还引入了在线微分MPC框架，利用PyTorch的自动微分功能对模型和控制策略进行实时优化。

### 4.1 在线模型自适应

初始模型 $(A_0, B_0, C_0)$ 由离线数据通过EDMD得到。在在线运行过程中，系统通过引入可学习的参数 $\Delta A, \Delta B, \Delta C$ 来补偿模型误差：

$$
\begin{aligned}
A &= A_0 + \Delta A \\
B &= B_0 + \Delta B \\
C &= C_0 + \Delta C
\end{aligned}
$$

这些参数初始化为零，并在每一时刻通过梯度下降进行更新。

### 4.2 MPC 优化问题

MPC的核心是在每个时刻 $t$ 求解一个有限时域内的最优控制序列 $U_{mpc} = \{u_t, u_{t+1}, \dots, u_{t+H-1}\}$。

**目标函数 (Loss Function):**

$$
L = w_m L_{model} + w_p L_{mpc}
$$

其中：
1.  **模型预测损失 ($L_{model}$)**：衡量当前模型对历史数据的拟合程度。
    $$
    L_{model} = \frac{1}{H_{past}} \sum_{k=t-H_{past}}^{t-1} \| z_{k+1} - (A z_k + B u_k) \|^2
    $$

2.  **MPC 控制代价 ($L_{mpc}$)**：衡量预测轨迹的跟踪性能和控制能量。
    $$
    L_{mpc} = \sum_{k=0}^{H-1} \left( \| x_{t+k|t} - x_{ref} \|_Q^2 + \| u_{t+k|t} \|_R^2 \right)
    $$
    其中 $x_{t+k|t}$ 是基于当前模型预测的未来状态，$x_{ref}$ 是参考状态（通常为原点），$Q$ 和 $R$ 分别是状态权重矩阵和控制权重矩阵。

### 4.3 基于梯度的求解

与传统的二次规划（QP）求解器不同，本项目利用PyTorch将控制序列 $U_{mpc}$ 和模型修正参数 $\Delta A, \Delta B$ 视为可优化的张量（Tensors）。

通过计算总损失 $L$ 对这些参数的梯度：

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

其中 $\theta \in \{ U_{mpc}, \Delta A, \Delta B \}$，$\eta$ 是学习率。这种方法允许控制器在优化控制动作的同时，在线修正模型以适应环境变化或初始建模误差。
