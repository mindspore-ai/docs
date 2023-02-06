# 模型压缩

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_zh_cn/infer/model_compression.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## 概述

MindSpore是一个全场景的AI框架。当模型部署到端侧或者其他轻量化设备上时，对于部署的内存、功耗、时延等有各种限制，所以往往在部署前需要对模型进行压缩。

MindSpore的模型压缩能力由 [MindSpore Golden Stick](https://www.mindspore.cn/golden_stick/docs/zh-CN/r0.3.0-alpha/index.html) 提供，MindSpore Golden Stick是华为诺亚团队和华为MindSpore团队联合设计开发的一个模型压缩算法集，为MindSpore提供了一系列模型压缩算法，如量化、剪枝等。详细资料可前往 [MindSpore Golden Stick官方资料](https://www.mindspore.cn/golden_stick/docs/zh-CN/r0.3.0-alpha/index.html) 查看。

## 量化方法

### 概念

量化即以较低的推理精度损失，将网络中的32位有限范围浮点型（FP32）权重或激活近似为有限多个离散值（通常为int8）的过程。换言之，它是以更少位数的数据类型来近似表示FP32数据的过程，而网络的输入输出依然是浮点型，从而达到减少网络尺寸大小、减少网络部署时的内存消耗及加快网络推理速度等目标。

当前业界量化方案主要分为两种：**感知量化训练**（Quantization Aware Training）和**训练后量化**（Post-training Quantization）。

（1）**感知量化训练**需要训练数据，在网络准确率上通常表现更好，适用于对网络压缩率和网络准确率要求较高的场景。目的是减少精度损失，其参与网络训练的前向推理过程令网络获得量化损失的差值，但梯度更新需要在浮点下进行，因而其并不参与反向传播过程。

（2）**训练后量化**简单易用，只需少量校准数据，适用于追求高易用性和缺乏训练资源的场景。

### 量化方法示例

- [SimQAT算法示例](https://www.mindspore.cn/golden_stick/docs/zh-CN/r0.3.0-alpha/quantization/simqat.html)：一种基础的基于伪量化技术的感知量化算法
- [SLB量化算法示例](https://www.mindspore.cn/golden_stick/docs/zh-CN/r0.3.0-alpha/quantization/slb.html)：一种非线性的低比特感知量化算法

## 剪枝方法

### 概念

剪枝是在保证网络准确率下降较小的前提下，通过去除神经网络中部分组件（如权重、特征图、卷积核）降低网络的参数量，从而降低网络部署时的存储和计算代价。

神经网络推理的过程通常可以看作是激活和权重做运算的过程，相应的，剪枝算法也通常分为两大类，权重剪枝和激活剪枝。当前在MindSpore Golden Stick中，我们仅讨论权重剪枝。

对于**权重剪枝**来说，按照剪枝模式的不同，主要分为结构化剪枝和非结构化剪枝。

（1）通常我们称神经元剪枝为**非结构化剪枝**，以单个权值为粒度对权重中任意位置的权值进行裁剪。这种剪枝方式由于其细粒度的特点，对于网络的准确率的影响更小，但会导致权重张量的稀疏化。稀疏化的权重张量对访存不友好，对并行计算不友好，所以非结构化剪枝后的网络难以获得较高的加速比。

（2）而通道剪枝和filter剪枝一般被认为是**结构化剪枝**，以权重的通道或者整个卷积核为粒度对模型的权重进行剪裁。由于是直接剪掉整个通道或者整个卷积核，所以剪枝得到的权重更加规则且规模更小，这也是结构化剪枝的含义所在。相较于非结构化剪枝，结构化剪枝由于得到的权重更加规则，对访存更友好，所以比较适合在CPU、GPU等设备上进行加速推理。

### 剪枝方法示例

- [SCOP剪枝算法示例](https://www.mindspore.cn/golden_stick/docs/zh-CN/r0.3.0-alpha/pruner/scop.html)：一个结构化权重剪枝算法
