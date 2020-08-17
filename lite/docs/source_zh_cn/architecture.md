# 总体架构

<!-- TOC -->

- [总体架构](#总体架构)
    - [MindSpore Lite的架构](#mindspore-lite的架构)
    - [MindSpore Lite的技术特点](#mindspore-lite的技术特点)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/lite/docs/source_zh_cn/architecture.md" target="_blank"><img src="../_static/logo_source.png"></a>

## MindSpore Lite的架构

![architecture](images/architecture.jpg)

MindSpore Lite框架包括如下几部分：

- 前端（Frontend）：负责模型生成，用户可以通过模型构建接口构建模型，将第三方模型和MindSpore训练的模型转换为MindSpore Lite模型，其中第三方模型包括TensorFlow Lite、Caffe 1.0和ONNX模型。

- IR：负责MindSpore的Tensor定义、算子定义和图定义。

- Backend：基于IR进行图优化，包括GHLO、GLLO和量化三部分。其中，GHLO负责和硬件无关的优化，如算子融合、常量折叠等；GLLO负责与硬件相关的优化；量化Quantizer支持权重量化、激活值量化等训练后量化手段。

- Runtime：智能终端的推理运行时，其中session负责会话管理，提供对外接口；线程池和并行原语负责图执行使用的线程池管理，内存分配负责图执行中各个算子的内存复用，算子库提供CPU和GPU算子。 

- Micro：IoT设备的运行时，包括模型生成.c文件、线程池、内存复用和算子库。

其中，Runtime和Micro共享底层的算子库、内存分配、线程池、并行原语等基础设施层。 

## MindSpore Lite的技术特点

MindSpore已经在HMS和华为智能终端的图像分类、目标识别、人脸识别、文字识别等应用中广泛使用。

MindSpore Lite具备如下特征：

1. 超轻量

   - 智能手机部署的MindSpore包大小为800k左右。
   - 智能手表、耳机等IoT设备部署的MindSpore Micro包大小低至几十K级别。

2. 高性能

   - 时延低：通过图优化、算子优化、并行处理、充分利用硬件能力等各种措施，优化推理时延。
   - 功耗低：通过量化、内存复用等，降低功耗。

3. 广覆盖

   - 支持Android和iOS。
   - 支持智能手机、手表、耳机以及各种IoT设备。
   - 支持CV和NLP多种神经网络模型，支持200多CPU算子、60多GPU算子。

4. 端云协同提供一站式训练和推理

   - 提供模型训练，模型转换优化，部署和推理端到端流程。
   - 统一的IR实现端云AI应用一体化。
