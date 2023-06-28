# 推理模型总览

 `Linux` `Ascend` `GPU` `CPU` `推理应用` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/inference/source_zh_cn/multi_platform_inference.md" target="_blank"><img src="./_static/logo_source.png"></a>

基于MindSpore训练后的模型，支持在不同的硬件平台上执行推理。本文介绍各平台上的推理流程。

按照原理不同，推理可以有两种方式：

- 直接使用checkpoint文件进行推理，即在MindSpore训练环境下，使用推理接口加载数据及checkpoint文件进行推理。
- 将checkpoint文件转化为通用的模型格式，如ONNX、AIR格式模型文件进行推理，推理环境不需要依赖MindSpore。这样的好处是可以跨硬件平台，只要支持ONNX/AIR推理的硬件平台即可进行推理。譬如在Ascend 910 AI处理器上训练的模型，可以在GPU/CPU上进行推理。

MindSpore支持的推理场景，按照硬件平台维度可以分为下面几种：

硬件平台 | 模型文件格式 | 说明
--|--|--
Ascend 910 AI处理器 | checkpoint格式 | 与MindSpore训练环境依赖一致
Ascend 310 AI处理器 | ONNX、AIR格式 | 搭载了ACL框架，支持OM格式模型，需要使用工具转化模型为OM格式模型。
GPU | checkpoint格式 | 与MindSpore训练环境依赖一致。
GPU | ONNX格式 | 支持ONNX推理的runtime/SDK，如TensorRT。
CPU | checkpoint格式 | 与MindSpore训练环境依赖一致。
CPU | ONNX格式 | 支持ONNX推理的runtime/SDK，如TensorRT。

> - ONNX，全称Open Neural Network Exchange，是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。它使得不同的人工智能框架（如PyTorch, MXNet）可以采用相同格式存储模型数据并交互。详细了解，请参见ONNX官网<https://onnx.ai/>。
> - AIR，全称Ascend Intermediate Representation，类似ONNX，是华为定义的针对机器学习所设计的开放式的文件格式，能更好地适配Ascend AI处理器。
> - ACL，全称Ascend Computer Language，提供Device管理、Context管理、Stream管理、内存管理、模型加载与执行、算子加载与执行、媒体数据处理等C++ API库，供用户开发深度神经网络应用。它匹配Ascend AI处理器，使能硬件的运行管理、资源管理能力。
> - OM，全称Offline Model，华为Ascend AI处理器支持的离线模型，实现算子调度的优化，权值数据重排、压缩，内存使用优化等可以脱离设备完成的预处理功能。
> - TensorRT，NVIDIA 推出的高性能深度学习推理的SDK，包括深度推理优化器和runtime，提高深度学习模型在边缘设备上的推断速度。详细请参见<https://developer.nvidia.com/tensorrt>。
