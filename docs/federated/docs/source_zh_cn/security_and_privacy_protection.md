# 模型安全和隐私

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/federated/docs/source_zh_cn/security_and_privacy_protection.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

联邦学习过程中，用户数据仅用于本地设备训练，不需要上传至中心服务器，可以避免用户个人数据的直接泄露。然而传统联邦学习框架中，模型以明文形式上云，仍然存在间接泄露用户隐私的风险。敌手获取到用户上传的明文模型后，可以通过重构、模型逆向等攻击恢复用户的个人训练数据，导致用户隐私泄露。

MindSpore Federated联邦学习框架，提供了基于本地差分隐私（LDP）和基于多方安全计算（MPC）的安全聚合算法，在本地模型上云前对其进行加噪或加扰。在保证模型可用性的前提下，解决横向联邦学习中的隐私泄露问题。

## 基于LDP的安全聚合

### 原理概述

差分隐私（differential privacy）是一种保护用户数据隐私的机制。差分隐私定义为：

$$
Pr[\mathcal{K}(D)\in S] \le e^{\epsilon} Pr[\mathcal{K}(D’) \in S]+\delta​
$$

对于两个差别只有一条记录的数据集$D, D’$，通过随机算法$\mathcal{K}$，输出结果为集合$S$子集的概率满足上面公式。$\epsilon$为差分隐私预算，$\delta$扰动，$\epsilon$和$\delta$越小，说明$\mathcal{K}$在$D$和$D’$上输出的数据分布越接近。

在横向联邦学习中，假设客户端本地训练之后的模型权重矩阵是$W$，由于模型在训练过程中会“记住”训练集的特征，所以敌手可以借助$W$还原出用户的训练数据集[1]。

MindSpore Federated提供基于本地差分隐私的安全聚合算法，防止本地模型上云时泄露隐私数据。

MindSpore Federated客户端会生成一个与本地模型$W$相同维度的差分噪声矩阵$G$，然后将二者相加，得到一个满足差分隐私定义的权重$W_p$:

$$
W_p=W+G
$$

MindSpore Federated客户端将加噪后的模型$W_p$上传至云侧服务器进行联邦聚合。噪声矩阵$G$相当于给原模型加上了一层掩码，在降低模型泄露敏感数据风险的同时，也会影响模型训练的收敛性。如何在模型隐私性和可用性之间取得更好的平衡，仍然是一个值得研究的问题。实验表明，当参与方的数量$n$足够大时（一般指1000以上），大部分噪声能够相互抵消，本地差分机制对聚合模型的精度和收敛性没有明显影响。

### 使用方式

开启差分隐私训练的方式很简单，只需要在启动云侧服务时，使用`context.set_fl_context()`设置`encrypt_type='DP_ENCRYPT'`即可。

此外，为了控制隐私保护的效果，我们还提供了3个参数：`dp_eps`，`dp_delta`以及`dp_norm_clip`，它们也是通过`context.set_fl_context()`设置。

`dp_eps`和`dp_norm_clip`的合法取值范围是大于0，`dp_delta`的合法取值范围是0<`dp_delta`<1。一般来说，`dp_eps`和`dp_delta`越小，隐私保护效果也越好，但是对模型收敛性的影响越大。建议`dp_delta`取成客户端数量的倒数，`dp_eps`大于50。

`dp_norm_clip`是差分隐私机制对模型权重加噪前对权重大小的调整系数，会影响模型的收敛性，一般建议取0.5~2。

## 基于MPC的安全聚合

### 原理概述

尽管差分隐私技术可以适当保护用户数据隐私，但是当参与客户端数量比较少或者高斯噪声幅值较大时，模型精度会受较大影响。为了同时满足模型保护和模型收敛这两个要求，我们提供了基于MPC的安全聚合方案。

在这种训练模式下，假设参与的客户端集合为$U$，对于任意Federated-Client $u$和$v$，
它们会两两协商出一对随机扰动$p_{uv}$、$p_{vu}$，满足

$$
p_{uv}=\begin{cases} -p_{vu}, &u{\neq}v\\\\ 0, &u=v \end{cases}
$$

于是每个Federated-Client $u$ 在上传模型至Server前，会在原模型权重$x_u$加上它与其它用户协商的扰动：

$$
x_{encrypt}=x_u+\sum\limits_{v{\in}U}p_{uv}
$$

从而Federated-Server聚合结果$\overline{x}$为：

$$
\begin{align}
\overline{x}&=\sum\limits_{u{\in}U}(x_{u}+\sum\limits_{v{\in}U}p_{uv})\\\\
&=\sum\limits_{u{\in}U}x_{u}+\sum\limits_{u{\in}U}\sum\limits_{v{\in}U}p_{uv}\\\\
&=\sum\limits_{u{\in}U}x_{u}
\end{align}
$$

上面的过程只是介绍了聚合算法的主要思想，基于MPC的聚合方案是精度无损的，代价是通讯轮次的增加。

如果您对算法的具体步骤感兴趣，可以参考原论文[2]。

### 使用方式

与开启差分隐私训练相似，我们只需要在`context.set_fl_context()`中设置`encrypt_type='PW_ENCRYPT'`即可。

此外，与安全聚合训练相关的云侧环境参数还有`share_secrets_ratio`、`reconstruct_secrets_threshold`和`cipher_time_window`。

`share_client_ratio`指代参与密钥碎片分享的客户端数量与参与联邦学习的客户端数量的比值，取值需要小于等于1。

`reconstruct_secrets_threshold`指代参与密钥碎片恢复的客户端数量，取值需要小于参与密钥碎片分享的客户端数量。

通常为了保证系统安全，当不考虑Server和Client合谋的情况下，`reconstruct_secrets_threshold`需要大于联邦学习客户端数量的一半；当考虑Server和Client合谋，`reconstruct_secrets_threshold`需要大于联邦学习客户端数量的2/3。

`cipher_time_window`指代安全聚合各通讯轮次的时长限制，主要用来保证某些客户端掉线的情况下，Server可以开始新一轮迭代。
需要注意的是，当前版本的安全聚合训练只支持`server_num=1`。

### 参考文献

[1] Ligeng Zhu, Zhijian Liu, and Song Han. [Deep Leakage from Gradients](http://arxiv.org/pdf/1906.08935.pdf). NeurIPS, 2019.

[2] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, et al. [Practical Secure Aggregationfor Privacy-Preserving Machine Learning](https://dl.acm.org/doi/pdf/10.1145/3133956.3133982). NeurIPS, 2016.
