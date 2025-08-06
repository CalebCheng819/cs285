# Lecture7

## part1

### policy iteration

相比与以往的policy gradient，我们是否可以omit policy gradient。

"既然我们能直接找到每个状态的最优动作，为何还要通过梯度更新策略？"

我们直接定义一个策略性的$\pi^\prime$
$$
\pi^{\prime}\left(\mathbf{a}_t \mid \mathbf{s}_t\right)=\left\{\begin{array}{l}
1 \text { if } \mathbf{a}_t=\arg \max _{\mathbf{a}_t} A^\pi\left(\mathbf{s}_t, \mathbf{a}_t\right) \\
0 \text { otherwise }
\end{array}\right.
$$
在原有的算法框架中，我们相当于直接修改蓝色框内的算法，而其他保持不变

![](https://cdn.mathpix.com/snip/images/xnfX8kOncRur1p8hHw7XqdzjOVVPb4P7c1eUcsLm7KU.original.fullsize.png)

这个变动更适合于discreet action，对于continuous action的处理方法我们在后续课程会介绍到

### dynamic programing

Bellman 方程将值函数 $V^\pi(s)$ 分解为：
$$
\underbrace{r(s, a)}_{\text {即时奖励 }}+\gamma \underbrace{\mathbb{E}_{s^{\prime}}\left[V^\pi\left(s^{\prime}\right)\right]}_{\text {未来折扣值 }}
$$

这种递归结构允许通过当前奖励和后续状态值计算当前状态值。

因为在新定义的$\pi^\prime$中动作是唯一确定的，所以可以把上式简化为
$$
V^\pi(\mathbf{s}) \leftarrow r(\mathbf{s}, \pi(\mathbf{s}))+\gamma E_{\mathbf{s}^{\prime} \sim p\left(\mathbf{s}^{\prime} \mid \mathbf{s}, \pi(\mathbf{s})\right)}\left[V^\pi\left(\mathbf{s}^{\prime}\right)\right]
$$
于是可以得到以下步骤：

![](https://cdn.mathpix.com/snip/images/fu1waPWTQFzmA814_rtVc4eWhLcKf2Cy4zMoPM2M9s0.original.fullsize.png)

1．初始化：任意猜测 $V_0(s)$（如全零）
2．迭代更新：对每个状态 $s$ ：
$$
V_{k+1}(s) \leftarrow \sum_a \pi(a \mid s)\left[r(s, a)+\gamma \sum_{s^{\prime}} p\left(s^{\prime} \mid s, a\right) V_k\left(s^{\prime}\right)\right]
$$

3．收敛检测：当 $\left\|V_{k+1}-V_k\right\|<\epsilon$ 时停止

为进一步简化求解，我们用Q去替代A，于是得到
$$
\begin{aligned}
& \arg \max _{\mathbf{a}_t} A^\pi\left(\mathbf{s}_t, \mathbf{a}_t\right)=\arg \max _{\mathbf{a}_t} Q^\pi\left(\mathbf{s}_t, \mathbf{a}_t\right) \\
& Q^\pi(\mathbf{s}, \mathbf{a})=r(\mathbf{s}, \mathbf{a})+\gamma E\left[V^\pi\left(\mathbf{s}^{\prime}\right)\right](\text { a bit simpler })
\end{aligned}
$$
![](https://cdn.mathpix.com/snip/images/r4ae4WpIX225_mfcv3oksQxQQh8II8FYpQ6yKAYI3kc.original.fullsize.png)

## part2

###  fitted value iteration

#### 核心问题：维度灾难（Curse of Dimensionality）

- **传统动态规划的局限**：
  - 表格法需要为每个状态存储 $V(s)$ 值
  - 状态空间指数级增长：例如 $200×200$ RGB 图像有 $(256^3)^{40,000}$ 种状态（远超过宇宙原子数）
  - **无法存储或遍历**所有状态

解决方案：函数逼近（Function Approximation），其中损失函数为
$$
\mathcal{L}(\phi)=\frac{1}{2}\left\|V_\phi(\mathbf{s})-\max _{\mathbf{a}} Q^\pi(\mathbf{s}, \mathbf{a})\right\|^2
$$


![](https://cdn.mathpix.com/snip/images/YSc-1DkaEEkrOzO02vw-Tl8P7B8hWS5xA3p8SYXF-cw.original.fullsize.png)

#### 核心问题：模型未知的挑战

在拟合值迭代中，目标值的计算需要：
$$
y_i \leftarrow \max _{a_i}\left(r\left(s_i, a_i\right)+\gamma \mathbb{E}\left[V_\phi\left(s_i^{\prime}\right)\right]\right)
$$
- 部分需要环境模型：

  奖励函数 $r(s, a)$

  状态转移概率 $p\left(s^{\prime} \mid s, a\right)$（用于计算期望）

于是我们可以考虑policy evaluation去估计Q

![](https://cdn.mathpix.com/snip/images/ju81VtDNTzxTIGH-Kb7LASX5kOyZAUafz0O9TUYl-V0.original.fullsize.png)

**用采样数据估计动作值函数 $Q^\pi$ ：
$$
Q^\pi(s, a) \leftarrow r(s, a)+\gamma \mathbb{E}_{s^{\prime} \sim p\left(s^{\prime} \mid s, a\right)}\left[Q^\pi\left(s^{\prime}, \pi\left(s^{\prime}\right)\right)\right]
$$**

在这里我们会用到一个大数定律
$$
\mathbb{E}_{s^{\prime}}\left[f\left(s^{\prime}\right)\right] \approx \frac{1}{N} \sum_{i=1}^N f\left(s_i^{\prime}\right)
$$
 这种改变使得我们可以在不知道转移动力学的情况下执行策略迭代风格的算法，只需通过采样 `(s, a, s')` 得到样本即可 。这是大多数基于价值的无模型强化学习算法的基础 。

#### **拟合Q迭代（Fitted Q Iteration）**

通过 Q函数 替代 V函数，完全消除对模型的依赖：

![](https://cdn.mathpix.com/snip/images/af2y2ZtV80LX_ErIzB_GfefnT8l8iHHDAzeYXOKM0dQ.original.fullsize.png)

算法步骤
1．目标值计算：
$$
y_i \leftarrow r\left(s_i, a_i\right)+\gamma \max _{a^{\prime}} Q_\phi\left(s_i^{\prime}, a^{\prime}\right)
$$
2．网络更新：
$$
\phi \leftarrow \arg \min _\phi \frac{1}{2} \sum_i\left\|Q_\phi\left(s_i, a_i\right)-y_i\right\|^2
$$
关键创新点
动作最大化延迟（Max Postponement）
原问题：在状态 $s_i$ 需计算 $\max _a$ $\rightarrow$ 需知道所有动作的奖励和转移概率
新方案：在下一状态 $s_i^{\prime}$ 计算 $\max _{a^{\prime}}$ $\rightarrow$ 仅需当前 Q 网络即可计算

> 算法特性分析
>
>  优势
>
> 1. **离策略学习（Off-Policy）**
>    - 可使用**历史数据**（经验回放）
>    - 数据生成策略 ≠ 目标策略
>      *示例：用随机策略数据训练贪婪策略*
> 2. **单网络结构**
>    - 只需维护Q网络，无需策略网络
>    - 避免Actor-Critic的双网络协调问题
> 3. **低方差更新**
>    - 基于Q值更新，无策略梯度的高方差问题
>    - 比REINFORCE等算法更稳定
>
> ⚠️局限性
>
> 1. **收敛性无保证**
>    - 函数逼近（神经网络）下可能发散
>    - 贝尔曼误差传播问题
> 2. **最大化偏差（Maximization Bias）**
>    - max 操作导致Q值系统性高估
>      *解决方案：双Q学习（Double Q-Learning）*
> 3. **连续动作空间挑战**
>    - max⁡a′ 需要全局优化
>      *解决方案：DDPG/SAC等算法*

## part3

### why is off-line

目标值计算公式：
$$
y_i=r_i+\gamma \max _{a_i^{\prime}} Q_\phi\left(s_i^{\prime}, a_i^{\prime}\right)
$$
- $r_i$ ：实际观测的奖励（已记录）
- $s_i^{\prime}$ ：实际到达的状态（已记录）
- $a_i^{\prime}$ ：虚拟动作（非实际执行动作）

> 计算目标值时：
> - 不需要知道数据收集策略
> - 不需要知道实际执行的动作 $a_i^{\prime}$
> - 只需访问 $\left(s_i, a_i, r_i, s_i^{\prime}\right)$ 四元组

### optimizing

算法第3步的损失函数：
$$
\phi \leftarrow \arg \min _\phi \frac{1}{2} \sum_i\left\|Q_\phi\left(s_i, a_i\right)-y_i\right\|^2
$$
本质是在最小化期望贝尔曼误差（Expected Bellman Error）：
$$
\mathcal{E}=\frac{1}{2} \mathbb{E}_{(s, a) \sim \beta}[(Q_\phi(s, a)-\underbrace{\left[r(s, a)+\gamma \max _{a^{\prime}} Q_\phi\left(s^{\prime}, a^{\prime}\right)\right]}_{\text {Bellman最优算子 } \mathcal{T}^* Q_\phi})^2]
$$
内部表达式：
$$
Q_\phi(s, a)-\left[r(s, a)+\gamma \max _{a^{\prime}} Q_\phi\left(s^{\prime}, a^{\prime}\right)\right]
$$
称为时序差分误差（TD Error）

当误差 $\mathcal{E}=0$ 时：
$$
Q_\phi(s, a)=r(s, a)+\gamma \max _{a^{\prime}} Q_\phi\left(s^{\prime}, a^{\prime}\right)
$$
这正是 Bellman 最优方程！这意味着：
1．$Q_\phi$ 是最优 Q 函数 $Q^*$
2．导出策略为最优策略 $\pi^*$ ：
$$
\pi^*(a \mid s)= \begin{cases}1 & \text { if } a=\arg \max _a Q^*(s, a) \\ 0 & \text { otherwise }\end{cases}
$$
| 算法           | 更新公式                                                     | 优化目标     |
| :------------- | :----------------------------------------------------------- | :----------- |
| 值迭代（表格） | $Q_{k+1}(s, a) \leftarrow \mathcal{T}^* Q_k$                 | 直接赋值     |
| 拟合Q迭代      | $Q_{k+1} \leftarrow \arg \min \left\\|Q-\mathcal{T}^* Q_k\right\\|^2$ | 函数空间投影 |

关键区别：
表格法能精确执行 $\mathcal{T}^*$ ，而函数逼近器（如神经网络）只能近似投影

### online Q iteration

![](https://cdn.mathpix.com/snip/images/PVmPO_fSBCSvDq2nQK2EZJJ-XVBQJ7eesg9lvIOUTDc.original.fullsize.png)

1．损失函数：单样本均方误差
$$
\mathcal{L}(\phi)=\frac{1}{2}\left[Q_\phi\left(s_i, a_i\right)-y_i\right]^2
$$
2．梯度计算：
$$
\nabla_\phi \mathcal{L}=\underbrace{\left(Q_\phi-y_i\right)}_\delta \cdot \underbrace{\nabla_\phi Q_\phi\left(s_i, a_i\right)}_{\text {梯度 }}
$$
3．参数更新：
$$
\Delta \phi=-\alpha \delta \nabla_\phi Q_\phi\left(s_i, a_i\right)
$$
物理意义：将Q值向目标值 $y_i$ 方向推动，步长由学习率 $\alpha$ 和TD误差 $\delta$ 共同决定。

在online policy中的第一步，我们并不使用之前说的greedy policy
$$
\pi\left(\mathbf{a}_t \mid \mathbf{s}_t\right)=\left\{\begin{array}{l}
1 \text { if } \mathbf{a}_t=\arg \max _{\mathbf{a}_t} Q_\phi\left(\mathbf{s}_t, \mathbf{a}_t\right) \\
0 \text { otherwise }
\end{array}\right.
$$
**为什么不使用贪婪策略进行探索？** 

- **确定性问题**：`argmax` 策略是确定性的 。
- **初始 Q 函数的问题**：如果初始 Q 函数很差，它不会是随机的，但会是任意的 。
- **陷入局部最优**：它将导致 `argmax` 策略每次进入特定状态时都采取相同的动作 。如果该动作不是一个好动作，我们可能会永远被困在采取那个坏动作，并且可能永远无法发现存在更好的动作 。
- **重要性**：在实践中，运行拟合 Q 迭代或 Q-learning 算法时，非常希望修改步骤 1 中使用的策略，使其不仅仅是 `argmax` 策略，而是注入一些额外的随机性以产生更好的探索 。

#### epsilon greedy
$$
\pi\left(a_t \mid s_t\right)= \begin{cases}1-\epsilon & \text { if } a_t=\arg \max _{a_t} Q_\phi\left(s_t, a_t\right) \\ \epsilon /(|A|-1) & \text { otherwise }\end{cases}
$$
运作机制：
- 以概率 $1-\epsilon$ 选择最优动作（利用）
- 以概率 $\epsilon$ 随机选择动作（探索）

**实践中的调整**：常见的做法是**在训练过程中改变 ϵ 的值** 。

- **初始阶段**：期望 Q 函数最初很差，此时可能希望使用更大的 ϵ 。
- **学习后期**：随着学习的进行，Q 函数变得更好，可以减小 ϵ 。

#### Boltzmann Exploration 
$$
\pi\left(\mathbf{a}_t \mid \mathbf{s}_t\right) \propto \exp \left(Q_\phi\left(\mathbf{s}_t, \mathbf{a}_t\right)\right)
$$
**规则**：动作的选择概率与 Q 值的指数成正比 。

**对比 ϵ-greedy**：当有两个同样好的动作时，ϵ-greedy 会给次优动作分配低得多的概率 ，而 Boltzmann 探索会以几乎相等的次数选择它们 。

**避免探索差的动作**：如果你已经知道某个动作非常糟糕，你可能不想浪费时间探索它 ，而 ϵ-greedy 不会利用这一点 。

## part4

### value iteration theory

#### bellman最优算子

1．向量化表示
- 状态空间： $\mathcal{S}=\{1,2, \ldots, N\}$
- 值函数向量： $\mathbf{V}=[V(1), V(2), \ldots, V(N)]^T$
- 奖励向量： $\mathbf{r}_a=[r(1, a), r(2, a), \ldots, r(N, a)]^T$
- 转移矩阵： $\mathbf{T}_a$ ，其中 $\left(\mathbf{T}_a\right)_{s, s^{\prime}}=p\left(s^{\prime} \mid s, a\right)$

2．Bellman最优算子 $B$
$$
B \mathbf{V}=\max _a\left(\mathbf{r}_a+\gamma \mathbf{T}_a \mathbf{V}\right)
$$

$$
V^{\star} \text { is a fixed point of } \mathcal{B} \quad V^{\star}(\mathbf{s})=\max _{\mathbf{a}} r(\mathbf{s}, \mathbf{a})+\gamma E\left[V^{\star}\left(\mathbf{s}^{\prime}\right)\right] \text {, so } V^{\star}=\mathcal{B} V^{\star}
$$
对任意两个值函数 $V$ 和 $\bar{V}$ ，Bellman最优算子 $B$ （contraction）满足：
$$
\|B V-B \bar{V}\|_{\infty} \leq \gamma\|V-\bar{V}\|_{\infty}
$$
1．迭代步骤：
$$
V_{k+1}=B V_k
$$
2．误差演化：
$$
\left\|V_{k+1}-V^*\right\|_{\infty}=\left\|B V_k-V^*\right\|_{\infty} \leq \gamma\left\|V_k-V^*\right\|_{\infty}
$$
3．递推关系：
$$
\left\|V_k-V^*\right\|_{\infty} \leq \gamma^k\left\|V_0-V^*\right\|_{\infty}
$$
**所以常规价值迭代 (tabular case) 收敛** 。

### Non-tabular value function learning

![](https://cdn.mathpix.com/snip/images/gH-CoWzpFu-Z7CMy0KxE79usl_-n02mmrDs4FddJgFg.original.fullsize.png)

define new operator $\Pi: \Pi V=\arg \min _{V^{\prime} \in \Omega} \frac{1}{2} \sum\left\|V^{\prime}(\mathbf{s})-V(\mathbf{s})\right\|^2$

`B` 是无穷范数下的收缩映射 。

`Π` 是 `L2` 范数下的收缩映射 。

**关键问题**：**`Π \* B` 算子不是任何一种收缩映射** 。

- **原因**：尽管 `B` 和 `Π` 单独都是收缩映射，但它们在不同的范数下是收缩映射 。将它们组合起来，可能导致结果不再是收缩映射 。
- **实践影响**：这不仅仅是理论上的怪癖，它在实践中确实会发生 。
- **图示解释**：最优价值函数是一个“黄星” 。常规价值迭代会逐渐接近这个星。而拟合价值迭代将价值函数限制在一条线上（代表神经网络的可表示空间`Ω`） 。 `B(V)` 会使你更接近星（在无穷范数下），然后投影 `Π` 会把你拉回到线上 。虽然这两个操作单独是收缩的，但组合起来可能会让你离星更远 。每个步骤都可能让你离最优解越来越远 。

![](https://cdn.mathpix.com/snip/images/wUJsi2MojoydE3qEJljY-sbTo0Z48VJpnSqAoD27EiI.original.fullsize.png)

所以**拟合价值迭代通常不收敛，在实践中也常常不收敛** 。

###  fitted Q-iteration

![](https://cdn.mathpix.com/snip/images/k4mT834k2kGcl7J2L20XE27J3mKbUI81-lP0GZ8_KrM.original.fullsize.png)

define an operator $\mathcal{B}: \mathcal{B} Q=r+\gamma \mathcal{T} \max _{\mathbf{a}} Q$
define an operator $\Pi: \Pi Q=\arg \min _{Q^{\prime} \in \Omega} \frac{1}{2} \sum\left\|Q^{\prime}(\mathbf{s}, \mathbf{a})-Q(\mathbf{s}, \mathbf{a})\right\|^2$ 

fitted Q -iteration algorithm (using $\mathcal{B}$ and $\Pi$ ):$Q \leftarrow \Pi \mathcal{B} Q$

$\mathcal{B}$ is a contraction w.r.t. $\infty$-norm ("max" norm)
$\Pi$ is a contraction w.r.t. $\ell_2$-norm (Euclidean distance)
$\Pi \mathcal{B}$ is not a contraction of any kind 

**常见误解**：有人可能会认为 Q-learning 只是对目标值进行回归，而回归是梯度下降，梯度下降会收敛 。
$$
\phi \leftarrow \phi-\alpha \underbrace{\frac{d Q_\phi}{d \phi}\left(s_i, a_i\right)}_{\text {预测值梯度 }} \underbrace{\left(Q_\phi\left(s_i, a_i\right)-\left[r_i+\gamma \max _{a^{\prime}} Q_\phi\left(s_i^{\prime}, a^{\prime}\right)\right]\right)}_{\text {TD 误差 }}
$$
目标值 $y_i$ 本身依赖 $\phi$ ：
$$
y_i=r_i+\gamma \max _{a^{\prime}} Q_\phi\left(s_i^{\prime}, a^{\prime}\right)
$$
这就是为什么它不一定converge

#### Actor-Critic 算法的类似问题（Similar Issues with Actor-Critic Algorithms）

![](https://cdn.mathpix.com/snip/images/NBvJEaF3-p2xVoM1PZDmqpL5EuofgmdQxUtpS5O0J7w.original.fullsize.png)

- **悲观的推论**：之前讨论过的 actor-critic 算法在使用函数逼近时也不保证收敛 。
- **原因**：Actor-critic 也进行了贝尔曼（使用了自举更新），并在更新价值函数时进行了投影 。
- **结果**：这些操作的串联不是一个收敛算子 。因此，拟合自举策略评估（fitted bootstrap policy evaluation）也不收敛 。