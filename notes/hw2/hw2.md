# hw2

## review

### variance reduction

#### reward to go

为了减少式子中的causality，我们使用t到T-1的reward
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta\left(a_{i t} \mid s_{i t}\right)\left(\sum_{t^{\prime}=t}^{T-1} r\left(s_{i t^{\prime}}, a_{i t^{\prime}}\right)\right)
$$

#### discounting

我们有两种方式，一种是一种是把discount加到full trajectory
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta\left(a_{i t} \mid s_{i t}\right)\right)\left(\sum_{t^{\prime}=0}^{T-1} \gamma^{t^{\prime}-1} r\left(s_{i t^{\prime}}, a_{i t^{\prime}}\right)\right)
$$
另一种是把discount加到reward to go（**推荐**）
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta\left(a_{i t} \mid s_{i t}\right)\left(\sum_{t^{\prime}=t}^{T-1} \gamma^{t^{\prime}-t} r\left(s_{i t^{\prime}}, a_{i t^{\prime}}\right)\right)
$$

#### baseline

我们考虑用一个**state-independent**（unbiased）的value function作为baseline
$$
V_\phi^\pi\left(s_t\right) \approx \sum_{t^{\prime}=t}^{T-1} \mathbb{E}_{\pi_\theta}\left[r\left(s_{t^{\prime}}, a_{t^{\prime}}\right) \mid s_t\right]
$$

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta\left(a_{i t} \mid s_{i t}\right)\left(\left(\sum_{t^{\prime}=t}^{T-1} \gamma^{t^{\prime}-t} r\left(s_{i t^{\prime}}, a_{i t^{\prime}}\right)\right)-V_\phi^\pi\left(s_{i t}\right)\right)
$$

#### GAE

我们可以把advantage function这么表示
$$
A^\pi\left(s_t, a_t\right)=Q^\pi\left(s_t, a_t\right)-V^\pi\left(s_t\right)
$$
其中Q是通过Monte Carlo估计的，V是通过已经学习到的$V_\phi^\pi$。我们可以进一步降低方差利用$V_\phi^\pi$去替代Monte Carlo估计的Q。于是得到
$$
A^\pi\left(s_t, a_t\right) \approx \delta_t=r\left(s_t, a_t\right)+\gamma V_\phi^\pi\left(s_{t+1}\right)-V_\phi^\pi\left(s_t\right)
$$
但是这样子会因为$V_\phi^\pi$错误的modeling而引入新的bias，于是我么可以考虑引入n-step
$$
A_n^\pi\left(s_t, a_t\right)=\sum_{t^{\prime}=t}^{t+n} \gamma^{t^{\prime}-t} r\left(s_{t^{\prime}}, a_{t^{\prime}}\right)+\gamma^n V_\phi^\pi\left(s_{t+n+1}\right)-V_\phi^\pi\left(s_t\right)
$$
接下来，我们引入$\lambda$引入指数加权，其中$\lambda$属于[0,1]
$$
A_{G A E}^\pi\left(s_t, a_t\right)=\frac{1-\lambda^{T-t-1}}{1-\lambda} \sum_{n=1}^{T-t-1} \lambda^{n-1} A_n^\pi\left(s_t, a_t\right)
$$
注意求和上限是**T-t-1**，当T趋近于无穷时，有
$$
\begin{aligned}
A_{G A E}^\pi\left(s_t, a_t\right) & =\frac{1}{1-\lambda} \sum_{n=1}^{\infty} \lambda^{n-1} A_n^\pi\left(s_t, a_t\right) \\
& =\sum_{t^{\prime}=t}^{\infty}(\gamma \lambda)^{t^{\prime}-t} \delta_{t^{\prime}}
\end{aligned}
$$
最后，我们对finite的情况化简一下
$$
A_{G A E}^\pi\left(s_t, a_t\right)=\sum_{t^{\prime}=t}^{T-1}(\gamma \lambda)^{t^{\prime}-t} \delta_{t^{\prime}}
$$
于是，接下来我们可以用递归形式（便与写代码）
$$
A_{G A E}^\pi\left(s_t, a_t\right)=\delta_t+\gamma \lambda A_{G A E}^\pi\left(s_{t+1}, a_{t+1}\right)
$$

## code

在配置环境前，会出现

```python
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for box2d-py
  Running setup.py clean for box2d-py
Failed to build box2d-py
ERROR: Failed to build installable wheels for some pyproject.toml based projects (box2d-py)
```

后尝试先`pip install swig`后再`pip install -r requirements.txt`成功解决

在进行编译的时候还出现了

```python
RuntimeError: module compiled against ABI version 0x1000009 but this version of numpy is 0x2000000                                                                                          
RuntimeError: module compiled against ABI version 0x1000009 but this version of numpy is 0x2000000                                                                                          
Traceback (most recent call last):
  File "/home/robot/cjt_cs/cs285/cs285/code/homework_fall2023/hw2/cs285/scripts/run_hw2.py", line 14, in <module>
    from cs285.infrastructure import utils
  File "/home/robot/cjt_cs/cs285/cs285/code/homework_fall2023/hw2/cs285/infrastructure/utils.py", line 6, in <module>
    import cv2
ImportError: numpy.core.multiarray failed to import
```

后先手动删除旧版本`pip uninstall numpy opencv-python opencv-python-headless -y`,再重新安装目前最稳定的兼容版本`pip install numpy==1.24.4 opencv-python==4.7.0.72`得以解决

> 在这里插一个为linux安装clash的快捷blog，实测非常好用[clash-for-linux-install: 优雅地部署基于 clash/mihomo 的代理环境](https://gitee.com/tools-repo/clash-for-linux-install)

### networks

#### policies.py

在MLPPolicy中我们应该写forward和get_action这些函数，**update应该更新到子类中**，因为不同的policy会有不同的算法

##### _distribution

定义了**如何根据输入的观测（observation）生成一个概率分布**，用于从中采样动作。

```python
    def _distribution(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """Returns a distribution over actions given an observation."""
        if self.discrete:
            logits = self.logits_net(obs)
            return distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            return distributions.Normal(mean, std)
```

1. 如果环境的动作是离散的（例如：上下左右，0/1等），我们通常使用 `Categorical` 分布。

   `self.logits_net` 是一个神经网络，它接受观测 `obs` 并输出每个可能动作的**logits（未归一化的对数概率）**。

2. 如果动作是连续的（例如：力的大小、角度等），我们使用正态分布 `Normal`。

   `self.mean_net`: 神经网络，输入观测 `obs`，输出每个动作维度的均值。

   `self.logstd`: 是一个可学习的张量，表示对数标准差（log std），通过 `exp` 还原出标准差。

##### update

```python
   def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        self.optimizer.zero_grad()
        dist=self(obs)
        logp=dist.log_prob(actions)#dist的性质计算对数概率
        if logp.dim()>1:
            logp=logp.sum(-1)
        loss=-(logp*advantages).sum()#含有advantage不建议mean
        loss.backward()
        self.optimizer.step()
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
```

`update()` 是每一步 actor 更新的过程，每一次更新都需要执行：

1. 清空上一次的梯度（`zero_grad()`）；
2. 计算新一批数据的 loss（基于当前 obs, actions, advantages）；
3. 反向传播新 loss 的梯度；
4. 更新网络参数（step）

其中因为 `MLPPolicyPG` 继承自 `MLPPolicy`，而 `MLPPolicy` 又继承自 `nn.Module`，这是 PyTorch 的神经网络模块的基类。在 `nn.Module` 中，已经实现了：

```python
pythonCopyEditdef __call__(self, *input, **kwargs):
    return self.forward(*input, **kwargs)
```

也就是说，**当你调用 `self(obs)` 时，其实内部就是在调用 `self.forward(obs)`**。