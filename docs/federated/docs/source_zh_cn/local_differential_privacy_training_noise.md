# 局部差分隐私加噪训练

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/local_differential_privacy_training_noise.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

联邦学习过程中，用户数据仅用于客户端设备的本地训练，不需要上传至中心服务器，可以避免泄露用户个人数据。然而，传统联邦学习框架中，模型以明文形式上云，仍然存在间接泄露用户隐私的风险。攻击者获取到客户端上传的明文模型后，可以通过重构、模型逆向等攻击方式，恢复参与学习的用户个人数据，导致用户隐私泄露。

MindSpore Federated联邦学习框架，提供了基于本地差分隐私（LDP）算法，在客户端上传本地模型前对其进行加噪。在保证模型可用性的前提下，解决横向联邦学习中的隐私泄露问题。

## 原理概述

差分隐私（differential privacy）是一种保护用户数据隐私的机制。差分隐私定义为：

$$
Pr[\mathcal{K}(D)\in S] \le e^{\epsilon} Pr[\mathcal{K}(D’) \in S]+\delta​
$$

对于两个差别只有一条记录的数据集$D, D’$，通过随机算法$\mathcal{K}$，输出结果为集合$S$子集的概率满足上述公式。$\epsilon$为差分隐私预算，$\delta$扰动，$\epsilon$和$\delta$越小，说明$\mathcal{K}$在$D$和$D’$上输出的数据分布越接近。

在横向联邦学习中，假设客户端本地训练之后的模型权重矩阵是$W$，由于模型在训练过程中会“记住”训练集的特征，所以攻击者可以借助$W$还原出用户的训练数据集[1]。

MindSpore Federated提供基于本地差分隐私的安全聚合算法，防止客户端上传本地模型时泄露用户隐私数据。

MindSpore Federated客户端会生成一个与本地模型$W$相同维度的差分噪声矩阵$G$，然后将二者相加，得到一个满足差分隐私定义的权重$W_p$:

$$
W_p=W+G
$$

MindSpore Federated客户端将加噪后的模型$W_p$上传至云侧服务器进行联邦聚合。噪声矩阵$G$相当于给原模型加上了一层掩码，在降低模型泄露敏感数据风险的同时，也会影响模型训练的收敛性。如何在模型隐私性和可用性之间取得更好的平衡，仍然是一个值得研究的问题。实验表明，当参与方的数量$n$足够大时（一般指1000以上），大部分噪声能够相互抵消，本地差分机制对聚合模型的精度和收敛性没有明显影响。

## 使用方式

本地差分隐私训练目前只支持端云联邦学习场景。开启差分隐私训练的方式很简单，只需要在启动云侧服务时，使用`set_fl_context()`设置`encrypt_type='DP_ENCRYPT'`即可。

此外，为了控制隐私保护的效果，我们还提供了3个参数：`dp_eps`，`dp_delta`以及`dp_norm_clip`，它们也是通过`set_fl_context()`设置。

`dp_eps`和`dp_norm_clip`的合法取值范围是大于0，`dp_delta`的合法取值范围是0<`dp_delta`<1。一般来说，`dp_eps`和`dp_delta`越小，隐私保护效果也越好，但是对模型收敛性的影响越大。建议`dp_delta`取成客户端数量的倒数，`dp_eps`大于50。

`dp_norm_clip`是差分隐私机制对模型权重加噪前对权重大小的调整系数，会影响模型的收敛性，一般建议取0.5~2。

## 参考文献

[1] Ligeng Zhu, Zhijian Liu, and Song Han. [Deep Leakage from Gradients](http://arxiv.org/pdf/1906.08935.pdf). NeurIPS, 2019.
