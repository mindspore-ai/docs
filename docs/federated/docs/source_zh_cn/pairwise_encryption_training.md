# 横向联邦-安全聚合训练

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/pairwise_encryption_training.md)

联邦学习过程中，用户数据仅用于本地设备训练，不需要上传至中心服务器，可以避免用户个人数据的直接泄露。然而传统联邦学习框架中，模型以明文形式上云，仍然存在间接泄露用户隐私的风险。攻击者获取到用户上传的明文模型后，可以通过重构、模型逆向等攻击方式恢复用户的个人训练数据，导致用户隐私泄露。

MindSpore Federated联邦学习框架，提供了基于多方安全计算（MPC）的安全聚合算法，在本地模型上云前加上秘密扰动。在保证模型可用性的前提下，解决横向联邦学习中的隐私泄露和模型窃取问题。

## 原理概述

尽管差分隐私技术可以适当保护用户数据隐私，但是当参与客户端数量比较少或者高斯噪声幅值较大时，模型精度会受较大影响。为了同时满足模型保护和模型收敛这两个要求，我们提供了基于MPC的安全聚合方案。

在这种训练模式下，假设参与的客户端集合为$U$，对于任意客户端Client $u$和$v$，
它们会两两协商出一对随机扰动$p_{uv}$、$p_{vu}$，满足

$$
p_{uv}=\begin{cases} -p_{vu}, &u{\neq}v\\\\ 0, &u=v \end{cases}
$$

于是每个客户端Client $u$ 在上传模型至服务端Server前，会在原模型权重$x_u$加上它与其他用户协商的扰动：

$$
x_{encrypt}=x_u+\sum\limits_{v{\in}U}p_{uv}
$$

从而服务端Server聚合结果$\overline{x}$为：

$$
\begin{align}
\overline{x}&=\sum\limits_{u{\in}U}(x_{u}+\sum\limits_{v{\in}U}p_{uv})\\\\
&=\sum\limits_{u{\in}U}x_{u}+\sum\limits_{u{\in}U}\sum\limits_{v{\in}U}p_{uv}\\\\
&=\sum\limits_{u{\in}U}x_{u}
\end{align}
$$

上述过程仅介绍了聚合算法的主要思想，基于MPC的聚合方案是精度无损的，代价是通讯轮次的增加。

如果您对算法的具体步骤感兴趣，可以参考原论文[1]。

## 使用方式

### 端云联邦场景

开启安全聚合训练的方式很简单，只需要在启动云侧服务时，通过yaml文件设置`encrypt_train_type`字段为`PW_ENCRYPT`即可。

此外，由于端云联邦场景下，参与训练的Worker大多是手机等不稳定的边缘计算节点，所以要考虑计算节点的掉线和密钥恢复问题。与之相关的参数有`share_secrets_ratio`、`reconstruct_secrets_threshold`和`cipher_time_window`。

`share_client_ratio`指代公钥分发轮次、秘密分享轮次、秘钥恢复轮次的客户端阈值衰减比例，取值需要小于等于1。

`reconstruct_secrets_threshold`指代恢复秘密需要的碎片数量，取值需要小于参与updateModel的客户端数量(start_fl_job_threshold*update_model_ratio)。

通常为了保证系统安全，当不考虑Server和Client合谋的情况下，`reconstruct_secrets_threshold`需要大于联邦学习客户端数量的一半；当考虑Server和Client合谋，`reconstruct_secrets_threshold`需要大于联邦学习客户端数量的2/3。

`cipher_time_window`指代安全聚合各通讯轮次的时长限制，主要用来保证某些客户端掉线的情况下，Server可以开始新一轮迭代。

### 云云联邦场景

在云云联邦场景下，在云侧启动脚本通过yaml文件设置`encrypt_train_type`字段为`PW_ENCRYPT`即可。

此外，与端云联邦不同的是，在云云联邦场景中，每个Worker都是稳定的服务器，所以不需要考虑掉线问题，因此只需要设置`cipher_time_window`这一超参。

## 参考文献

[1] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, et al. [Practical Secure Aggregationfor Privacy-Preserving Machine Learning](https://dl.acm.org/doi/pdf/10.1145/3133956.3133982). Proceedings of the 2017 ACM SIGSAC Conference on Computer and communications Security. 2017.
