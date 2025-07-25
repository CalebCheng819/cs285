# Lecture8

## part1

### Problems with Online Q-learning

![](https://cdn.mathpix.com/snip/images/8crfJqTBY54kT9yyWZ5MnC2jGcjEskS-dkrfAwc8mrs.original.fullsize.png)

**问题 1：非梯度下降**：

- **误解**：尽管步骤 3 的更新看起来像梯度更新，但它实际上并不是任何定义明确的目标函数的梯度。 
- **残差梯度算法**：如果正确应用链式法则，会得到一类称为“残差梯度算法”的方法。然而，这些算法由于严重的数值不稳定性在实践中表现不佳。

**问题 2：样本相关性**：

![](https://cdn.mathpix.com/snip/images/gUjBCccEa_93exM9grqdtST0Zw83UisYONyUGxgEJnA.original.fullsize.png)

- **在线采样**：在线 Q-learning 每次只采样一个过渡。连续的过渡高度相关（例如，时刻 t 的状态 st 与时刻 t+1 的状态 st+1 非常相似）。 
- **违反假设**：在高度相关的样本上进行梯度更新违反了随机梯度方法通常的假设。 
- **局部过拟合**：如果算法一次只看到一小段轨迹中的过渡，它可能会局部过拟合这些过渡。当开始新的轨迹时，函数逼近器可能已经过拟合了之前轨迹的末尾，导致性能不佳。

### Solutions to the Correlation Problem

**方法 1：并行化（Parallelization）**：

![](https://cdn.mathpix.com/snip/images/_2lWa_srV3CoOVgrea6xDLCKVPhqmp36lgzbFdt-uSM.original.fullsize.png)

- **思想**：类似于 Actor-Critic 算法中的解决方案，可以有多个并行workers，每个worker收集不同的过渡。 
- **批量更新**：在一个由来自不同worker的样本组成的批次上进行更新，这样可以缓解样本之间的顺序相关性。
- **异步版本**：worker可以异步地从参数服务器查询最新参数并以自己的速度进行。对于 Q-learning 来说，这在理论上效果更好，因为不必使用最新策略。

**方法 2：回放缓冲区（Replay Buffers）**：

![image-20250725112706272](C:/Users/21934/AppData/Roaming/Typora/typora-user-images/image-20250725112706272.png)

- **核心思想**：回放缓冲区是一种在强化学习中相当古老且非常有效的技术（起源于 1990 年代）。 
- **工作原理**：
  1. 收集一个过渡数据集，并将其添加到回放缓冲区中。 
  2. 从缓冲区中采样一个批次的过渡，并在该批次上进行学习。

## part2

![](https://cdn.mathpix.com/snip/images/pHkuc5Fp5hMlRjqyXSs4LV-r7_L6PbZTNcVmemAM7iw.original.fullsize.png)

| **特性**         | 带经验回放的Q学习     | 完整拟合Q迭代              |
| :--------------- | :-------------------- | :------------------------- |
| **目标值性质**   | 移动目标（依赖当前φ） | **固定目标**（计算后冻结） |
| **优化方式**     | 单步梯度下降          | **完整优化至收敛**         |
| **数据使用**     | 持续增量更新          | 分阶段批量处理             |
| **计算开销**     | 低（单步更新）        | 高（需收敛优化）           |
| **稳定性**       | 较低（目标波动）      | **高（稳定回归）**         |
| **参数更新频率** | 每批次更新后φ立即改变 | 整轮优化完成后才更新φ      |
| **样本时效性**   | 新旧样本混合使用      | 每轮使用当前策略的新数据   |
| **实现复杂度**   | 较简单                | 需解优化问题               |

回顾 Q－learning 面临的问题：

- 样本相关性（Correlated Samples）：在线 Q－learning 连续采样的样本是高度相关的，这违反了随机梯度下降的独立同分布（IID）假设，可能导致局部过拟合。（已解决）

- 移动目标（Moving Target）：Q－learning 的目标值 $Y_i=r_i+\gamma \max _{a^{\prime}} Q_\phi\left(s_i^{\prime}, a^{\prime}\right)$ 本身依赖于当前正在更新的 Q 函数参数 $\phi$ 。这意味着回归的目标值会不断变化，使得学习过程不稳定，难以收敛。

  拓展解释：想象你正在追逐一个目标，但这个目标本身也在不断移动，并且它的移动方式还取决于你的追赶行为。这使得追赶变得异常困难，甚至可能永远无法真正＂追上＂。在 Q－learning 中，神经网络试图拟合的目标值 $Y_i$ 正是这样，它随着 Q 函数参数 $\phi$ 的更新而改变，导致训练不稳定。

### target network

![](https://cdn.mathpix.com/snip/images/yQw8_Z-uqVwEU-rbQuAYYc8Tgi1VBT3goL6pWigU6yY.original.fullsize.png)

使用两个 Q 网络：

1. **当前 Q 网络（Current Q-Network）**，参数为 ϕ。这是我们正在训练的网络。
2. **目标 Q 网络（Target Q-Network）**，参数为 ϕ′。这是一个延迟更新的网络，用于计算目标值。

目标网络提供稳定目标：
$y_i=r_i+\gamma \max _a Q_{\phi^{\prime}}\left(s_i^{\prime}, a_i\right)$（固定周期内不变）

**超参数选择**：

- 内循环更新次数 

  K：通常 K=1 到 4。 

- 目标网络更新频率 

  N：通常 N=10000 步。（这是一个非常慢的进程，通常不频繁地运行，以确保目标值的稳定性）

如果将通用拟合 Q 迭代算法中的 **K 设置为 1**，并加入目标网络和回放缓冲区，就得到了经典的 **DQN 算法**。

![](https://cdn.mathpix.com/snip/images/3gN6nShbzxKmIUPMJkIM5fFxcjKJLxjkaJDbz4KdNGU.original.fullsize.png)

目标网络更新策略的“奇怪之处”（"Strangeness" of Target Network Update） 

- **直观问题**：上述目标网络的硬更新（每 N 步完全复制）会导致目标网络参数的“跳跃式”变化。 

  - 例如，如果 N=4，那么在第 5 步，目标网络参数 ϕ′ 会突然变成与当前网络 ϕ 相同的参数。

  - 这意味着，目标网络的“年龄”或“滞后量”在不同时间点是变化的：

    - 在目标网络刚更新后（例如，第 

      N+1 步），目标网络只比当前网络滞后一步。 

    - 而在目标网络即将更新前（例如，第 

      2N−1 步），目标网络会比当前网络滞后 N−1 步。 

  - 这种不一致的滞后量可能导致训练过程中的不稳定感，尽管在实践中通常不是一个大问题。 

- **替代更新策略：软更新（Soft Updates）/ Polyak Averaging** 

  - 为了解决这种“跳跃式”更新和不一致的滞后量，一种常见的替代方法是

    **软更新（Soft Updates）**，也称为 **Polyak Averaging**。
    $$
    \text { update } \phi^{\prime}: \phi^{\prime} \leftarrow \tau \phi^{\prime}+(1-\tau) \phi \quad \tau=0.999 \text { works well }
    $$

## part3

### Unified Perspective of Q-learning Algorithms

![](https://cdn.mathpix.com/snip/images/tsg1MbAc9HApN4p8OCD3LAueuTBdP0SHbQbDGCJb19g.original.fullsize.png)

**进程 1：数据收集（Data Collection Process）** 

- **功能**：与环境交互，从当前状态 ϕ 构建策略（例如使用 ϵ-greedy 或 Boltzmann 探索），然后将其发送到环境中。环境返回一个或多个转换。
- **运行方式**：这是一个持续运行的进程，不断地从环境中收集数据并发送到回放缓冲区 。

**进程（1a）：数据逐出（Eviction Process）** 

- **功能**：当回放缓冲区达到最大容量时，需要移出旧数据。
- **常见策略**：最简单且合理的方法是使用环形缓冲区（ring buffer），当新数据进入时，最旧的数据被移除 。例如，如果缓冲区有 100 万个转换，当第 100 万零一个转换加入时，最旧的转换会被丢弃 。

**进程 2：目标网络更新（Target Parameter Update Process）** 

- **功能**：周期性地更新目标网络参数 ϕ′。通常是将 ϕ 复制到 ϕ′，或者执行 Polyak Averaging。
- **运行速率**：这是一个非常慢的进程，通常不频繁地运行 。例如，它可能每 10,000 步才更新一次，以确保目标值的稳定性.

**进程 3：学习/ Q 函数更新（Main Learning Process）** 

- **功能**：从回放缓冲区加载一批转换。使用目标网络参数 ϕ′ 计算目标值，然后使用这些目标值更新当前网络参数 ϕ。
- **运行速率**：这是一个相对较快的进程，负责实际的神经网络训练 。

![](https://cdn.mathpix.com/snip/images/_arLzwvIFa43jHtJ_3HS_wyNMWWrZOrfgXRukUslZdM.original.fullsize.png)

为什么不同的运行速率有帮助（Why Different Rates Help）

- **非平稳性（Non-stationarity）**：Q-learning 算法的一个主要挑战是其固有的非平稳性。每个进程（数据收集、目标网络更新、学习）都会为其他进程创建非平稳性 。
  - 例如，如果数据收集策略发生变化，那么回放缓冲区中的数据分布也会变化，导致学习过程变得非平稳。
  - 如果目标网络参数更新，目标值也会变化，导致学习过程的非平稳性。
- **缓解非平稳性**：通过让这些不同的进程以不同的速率运行，我们本质上是在缓解非平稳性的影响 。
  - 如果进程 2（目标网络更新）和进程 1（数据收集）完全停止，那么进程 3（学习）将面临一个标准的、收敛的监督学习问题 。
  - 通过使进程的速率差异很大（例如，进程 3 比进程 2 快得多），对于运行速度快的进程来说，其他进程看起来几乎是平稳的 。
- **深层原因**：这是让 Q-learning 算法更有效收敛的深层原因之一：通过控制各个组件的更新频率，可以减少它们彼此之间的干扰，从而提高整体的稳定性 。

## part4

### double q 

**Q 值过高估计问题（Overestimation of Q-values）** ：

- 在 Q-learning 训练过程中，我们通常会观察到两个趋势 ：

  1. 平均每集回报（Average Reward per Episode）随着训练的进行而增加 。这意味着策略正在变好 。
  2. 平均 Q 值（Average Q-values）的预测值也会越来越大 。

- **问题**：虽然策略变好，回报增加是正常的，但 Q 值预测的增长速度往往比实际回报的增长速度快得多 。

- **原因**：这是因为 `max` 操作（取最大值操作）具有内在的偏置 。

  数学定理：
  $$
  \mathbb{E}\left[\max \left(X_1, X_2\right)\right] \geq \max \left(\mathbb{E}\left[X_1\right], \mathbb{E}\left[X_2\right]\right)
  $$
  - 物理意义：两个随机变量的最大值的期望 $\geq$ 各自期望的最大值
  - 在 Q 学习中的表现：
  $$
  \mathbb{E}\left[\max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}\right)\right] \geq \max _{a^{\prime}} \mathbb{E}\left[Q\left(s^{\prime}, a^{\prime}\right)\right]
  $$

  导致 $Q$ 值系统性高估

  - 当你的 Q 值估计值存在噪声时 ，取最大值总是倾向于选择那些被噪声抬高了的 Q 值 。这导致 Q 值被系统性地过高估计 。

于是，我们考虑引入**双Q网络**
$$
\begin{aligned}
& Q_{\phi_A}(\mathbf{s}, \mathbf{a}) \leftarrow r+\gamma Q_{\phi_B}\left(\mathbf{s}^{\prime}, \arg \max _{\mathbf{a}^{\prime}} Q_{\phi_A}\left(\mathbf{s}^{\prime}, \mathbf{a}^{\prime}\right)\right) \\
& Q_{\phi_B}(\mathbf{s}, \mathbf{a}) \leftarrow r+\gamma Q_{\phi_A}\left(\mathbf{s}^{\prime}, \arg \max _{\mathbf{a}^{\prime}} Q_{\phi_B}\left(\mathbf{s}^{\prime}, \mathbf{a}^{\prime}\right)\right)
\end{aligned}
$$


#### using current network

![](https://cdn.mathpix.com/snip/images/-zGq0E4aLcR69P8Zxrvohcj9O9K-TQYQECJUuezhz8M.original.fullsize.png)

利用目标网络：将现有的目标网络 $Q_{\phi^{\prime}}$ 视为第二个 Q 函数 。
－使用当前网络
$Q_\phi$ 进行动作选择： $\operatorname{argmax}_{a^{\prime}} Q_\phi\left(s_i^{\prime}, a^{\prime}\right)$ 。
－使用目标网络
$Q_{\phi^{\prime}}$ 进行价值评估：$Q_{\phi^{\prime}}\left(s_i^{\prime}, \operatorname{argmax}_{a^{\prime}} Q_\phi\left(s_i^{\prime}, a^{\prime}\right)\right)$ 。
－这个方法就是经典的Double DQN 。它比原始的 DQN 在 Atari 游戏上表现更好。

### Multi-step return

![](https://cdn.mathpix.com/snip/images/s8_y8EVcS3FoVIw3J6rX9VQb8XDk8ggKZrTCy5Ts4Ng.original.fullsize.png)

之前讨论的 Q－learning 目标值 $Y_i=r_i+\gamma \max _{a^{\prime}} Q_{\phi^{\prime}}\left(s_i^{\prime}, a^{\prime}\right)$ 是一种单步返回（1－step return）。它只使用了下一个状态 $s_i^{\prime}$ 的信息。 $N$ 步返回旨在结合蒙特卡洛方法和时序差分（TD）方法的优点。

它累积了未来 N 步的实际奖励。 然后使用 $Q$ 函数来估计第 $N$ 步之后所有未来奖励的期望值。

为了使 N 步返回保持离策略性（即不需要轨迹中的所有中间动作都来自最新策略），必须确保在 t 到 t+N−1 之间的所有动作 at,...,$a_{t+N−1}$ 都是由**当前策略**（或至少是能支持当前策略的策略）生成的 。如果中间的某个动作不是由当前策略生成的，那么这个 N 步返回就不再是最新策略的有效估计 。

#### 解决办法

![](https://cdn.mathpix.com/snip/images/hJMJ9Y4JB7M85x8b8sBaSsgL-YLfRK3y8N4VBiOYxIg.original.fullsize.png)

1. 忽略问题：在实践中，有时可以简单地忽略这个问题，这在许多情况下效果很好 。

2. **动态截断（Dynamically Cut the Trace）**：

   - 动态选择 N 的值，以确保所有中间动作都与当前策略的贪婪动作匹配 。
   - 检查实际执行的动作序列，找到与确定性贪婪策略最长匹配的 N，然后截断该 N 步返回 。
   - **适用性**：当数据大部分是“在线策略”（on-policy）的，并且动作空间很小（容易匹配）时，这种方法效果很好 。

3. 重要性采样：

   做法：通过权重修正行为策略与目标策略的概率差异。 

   定义重要性权重：$\rho_t=\frac{\pi\left(\mathbf{a}_t \mid \mathbf{s}_t\right)}{\beta\left(\mathbf{a}_t \mid \mathbf{s}_t\right)}$ - 修正后的多步目标： $$ y_{j, t}=r_{j, t}+\gamma \rho_{t+1}\left(r_{j, t+1}+\gamma \rho_{t+2}\left(\cdots+\gamma^{N-1} \rho_{t+N-1} r_{j, t+N-1}+\gamma^N \max _{\mathbf{a}} Q_\phi\left(\mathbf{s}_{j, t+N}, \mathbf{a}\right)\right)\right. $$

    - 优点：数学上无偏。 - 缺点： - 权重连乘导致 方差爆炸（尤其当 $N$ 大或 $\pi$ 与 $\beta$ 差异大时）。

## part5

### Q learning in continuous action

### **Stochastic Optimization**

步骤：
1．从动作空间均匀采样 $N$ 个动作 $\left\{a_1, \ldots, a_N\right\}$
2．计算 $Q_\phi\left(s, a_i\right)$ 并取最大值： $\max _i Q_\phi\left(s, a_i\right)$

优点：

- 实现简单，可完全并行化（将不同动作作为batch输入）
- 过估计（overestimation）风险低于精确max（因采样有限）

缺点：

- 精度随动作空间维度指数下降（维度灾难）
- 低维空间（ $\leq 10$ 维）效果较好，高维空间效果差

#### **Cross-Entropy Method, CEM**

- **原理**：迭代式随机优化
  1. 从当前分布（如高斯分布）采样动作集
  2. 评估 Qϕ(s,a) 并保留Top-K样本
  3. 用Top-K样本更新分布（如调整均值和方差）
  4. 重复直到收敛
- **特点**：
  - 比随机采样更精确，适合中低维动作空间（≤40维）
  - 可并行化，但迭代步骤增加计算成本
- **扩展**：CMA-ES（更高级的进化策略）

### **Easy-to-Optimize Function Class**

#### **NAF架构（Normalized Advantage Function）**

**思想**：让 Q 函数对动作是**二次函数**（quadratic），因为二次函数的最大值有封闭解。

**典型代表：NAF（Normalized Advantage Function）**

- Q 函数被建模为：
  $$
  Q(s, a) = V(s) -\frac12(a - \mu(s))^T P(s) (a - \mu(s))
  $$

- 其中：

  - $V(s)$：状态值；
  - $\mu(s)$：最优动作；
  - $P(s)$：正定矩阵，确定曲率；

**优点**：

- `argmax_a Q(s, a)` 有封闭解，速度快；

**缺点**：

- 表达能力有限，只能逼近**动作维度是二次形式的 Q 函数**；
- 若真实 Q 函数不是二次的，就有偏差。

### 学习近似 `argmax` 的神经网络（即 Actor）

代表算法：DDPG（Deep Deterministic Policy Gradient）

**核心思想**：使用一个策略网络学习 `argmax_a Q(s, a)`，即：
$$
\mu_\theta(s) \approx \arg\max_a Q_\phi(s, a)
$$
**这个网络输出的是最优动作**，可视为 actor。

与其一起使用的 Q 网络仍然被训练为 critic。

### 

$$
\frac{d Q_\phi}{d \theta} = \frac{d a}{d \theta} \cdot \frac{d Q_\phi}{d a}
$$

解释：
 用 **链式法则（Chain Rule）** 对 θ 求导：

1. $\mu_\theta(s)$ 是由 θ 参数生成的动作；
2. Q 网络对动作求导；
3. 将两个导数相乘就得到对 θ 的导数：

$$
\nabla_\theta Q_\phi(s, \mu_\theta(s)) = \nabla_a Q_\phi(s, a) \cdot \nabla_\theta \mu_\theta(s)
$$

这其实就是 DDPG 中 actor 更新的核心梯度。

目标值定义（用于更新 Q 网络）
$$
y_j = r_j + \gamma Q'_\phi(s'_j, \mu_\theta(s'_j)) \approx r_j + \gamma Q'_\phi(s'_j, \arg\max_a Q'_\phi(s'_j, a))
$$
解释：

- 这是用于 Q 网络训练的 TD target；
- 注意这里用了目标网络 $Q'$ 和 $\mu'$，提高稳定性；
- 核心是用 $\mu_\theta(s')$ 来代替那个难以优化的 argmax。

![](https://cdn.mathpix.com/snip/images/w_oGIaj_zNYMKXdYtdeLBWrw0aPepraS6fUPf4n2Thg.original.fullsize.png)

| 模块                                     | 功能            | 使用技巧        |
| ---------------------------------------- | --------------- | --------------- |
| Actor 网络 $\mu_\theta(s)$               | 给出确定性动作  | 用策略梯度更新  |
| Critic 网络 $Q_\phi(s, a)$               | 估计状态-动作值 | 用 TD 目标更新  |
| 经验回放池 $\mathcal{B}$                 | 提供非相关样本  | 增加稳定性      |
| Target 网络 $Q_{\phi'}$, $\mu_{\theta'}$ | 计算目标值      | Polyak 平滑更新 |

## part6

Q-learning 比起 Policy Gradient（策略梯度）方法更难调、更容易不稳定   

1. 建议经验回放池 $\mathcal{B}$ 的大小为 **~100万条 transition**。

2. 建议 **高 ε-greedy 探索率** 开始，然后慢慢衰减；

3. Bellman 误差容易爆炸（gradient too large）

   误差函数：
   $$
   L = \left(Q(s, a) - y\right)^2
   $$
   如果某些动作的 Q 值预测极端错误（例如 -1000000），就算策略最终不会选择它，**该动作的误差依然会主导梯度更新**。这会：

   - 造成训练不稳定；
   - 让学习关注到无意义的“极差值动作”。

 解决方法：

- **Clip gradient**：手动限制梯度范围；

- **使用 Huber loss**（平滑损失）：

  Huber Loss 定义如下：
  $$
  L_\delta(a) = 
  \begin{cases}
    \frac{1}{2}a^2 & \text{if } |a| \leq \delta \\
    \delta (|a| - \frac{1}{2} \delta) & \text{otherwise}
  \end{cases}
  $$

  - 小误差时表现为平方误差（smooth）；
  - 大误差时表现为绝对值误差（鲁棒）；
  - **兼具稳定性与可导性**。

#### n-step return 的利与弊

- **优点**：更快传播奖励信号，对 early training 阶段提升大；
- **缺点**：较大 n 会导致目标偏差（bias）积累；
- ✅ 建议：
  - 小 n 值如 3~5 效果通常不错；
  - 训练后期慎用。

![](https://cdn.mathpix.com/snip/images/Pj4YGaNskOD2n0sb_AhCnLma3JgWIg7c4LsFkA3ygMs.original.fullsize.png)