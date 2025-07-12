# imitation learning

## behavior cloning（lecture2 part2）

定义：直接使用监督学习的方法，将专家数据看作样本输入

### 目标

我们在$p_{\mathrm{data}}(\mathbf{o}_t)$下训练，但是我们用$p_{\pi_\theta}(\mathbf{o}_t)$测试，但$p_{\mathrm{data}}(\mathbf{o}_t)\neq p_{\pi_\theta}(\mathbf{o}_t)$

所以不考虑以下目标函数
$$
\max_\theta E_{\mathbf{o}_t\sim p_\mathrm{data}(\mathbf{o}_t)}[\log\pi_\theta(\mathbf{a}_t|\mathbf{o}_t)]
$$
首先，定义损失函数为
$$
\left.c(\mathbf{s}_t,\mathbf{a}_t)=\left\{\begin{array}{l}0\mathrm{~if~}\mathbf{a}_t=\pi^\star(\mathbf{s}_t)\\1\mathrm{~otherwise}\end{array}\right.\right.
$$
我们关心的是$\pi_\theta$情况，我们需要做的是
$$
\mathrm{minimize~}E_{\mathbf{s}_t\thicksim p_{\pi_\theta}(\mathbf{s}_t)}[c(\mathbf{s}_t,\mathbf{a}_t)]
$$
假设：$\pi_\theta(\mathbf{a}\neq\pi^\star(\mathbf{s})|\mathbf{s})\leq\epsilon$，对于$\mathrm{s}\in\mathcal{D}_{\mathrm{train}}$

因此有
$$
E\left[\sum_tc(\mathbf{s}_t,\mathbf{a}_t)\right]\leq\underbrace{\epsilon T+(1-\epsilon)(\epsilon(T-1)+(1-\epsilon)(\ldots))}_{T\text{ terms, each }O(\epsilon T)}=O(\epsilon T^2)
$$
这个式子表明BC的cost会随着决策步数的增加而呈现平方次增加

更加泛化的说，对于$\mathbf{s}\sim p_{\mathrm{train}}(\mathbf{s})$，我们假设$E_{p_{\mathrm{train}}(\mathbf{s})}[\pi_{\theta}(\mathbf{a}\neq\pi^{\star}(\mathbf{s})|\mathbf{s})]\leq\epsilon$

对于$p_{\mathrm{train}}(\mathbf{s})\neq p_{\theta}(\mathbf{s})$，我们有
$$
p_\theta(\mathbf{s}_t)=(1-\epsilon)^tp_{\mathrm{train}}(\mathbf{s}_t)+(1-(1-\epsilon)^t))p_{\mathrm{mistake}}(\mathbf{s}_t)
$$
式子前一项表明不犯错（即train data）的概率，后一项表明其他分布

然后有
$$
|p_\theta(\mathbf{s}_t)-p_\mathrm{train}(\mathbf{s}_t)|=(1-(1-\epsilon)^t)|p_\mathrm{mistake}(\mathbf{s}_t)-p_\mathrm{train}(\mathbf{s}_t)|\leq2(1-(1-\epsilon)^t)
$$
其中$(1-\epsilon)^t\geq1-\epsilon t\mathrm，{~for~}\epsilon\in[0,1]$，所以小于$2\epsilon t$

对于我们的目标函数来说
$$
\begin{aligned}\sum_{t}E_{p_{\theta}(\mathbf{s}_{t})}[c_{t}]=\sum_{t}\sum_{\mathbf{s}_{t}}p_{\theta}(\mathbf{s}_{t})c_{t}(\mathbf{s}_{t})&\begin{aligned}\leq\sum_t\sum_{\mathbf{s}_t}p_{\mathrm{train}}(\mathbf{s}_t)c_t(\mathbf{s}_t)+|p_\theta(\mathbf{s}_t)-p_{\mathrm{train}}(\mathbf{s}_t)|c_{\mathrm{max}}\end{aligned}\\&\begin{aligned}\leq\sum_t\epsilon+2\epsilon t\leq\epsilon T+2\epsilon T^2\end{aligned}\\&O(\epsilon T^2)\end{aligned}
$$
