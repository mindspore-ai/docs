# 多平台推理

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.5/tutorials/source_zh_cn/use/multi_platform_inference.md)

## 概述

基于MindSpore训练后的模型，支持在不同的硬件平台上执行推理。本文介绍各平台上的推理流程。

按照原理不同，推理可以有两种方式：
- 直接使用checkpiont文件进行推理，即在MindSpore训练环境下，使用推理接口加载数据及checkpoint文件进行推理。
- 将checkpiont文件转化为通用的模型格式，如ONNX、GEIR格式模型文件进行推理，推理环境不需要依赖MindSpore。这样的好处是可以跨硬件平台，只要支持ONNX/GEIR推理的硬件平台即可进行推理。譬如在Ascend 910 AI处理器上训练的模型，可以在GPU/CPU上进行推理。

MindSpore支持的推理场景，按照硬件平台维度可以分为下面几种：

硬件平台 | 模型文件格式 | 说明
--|--|--
Ascend 910 AI处理器 | checkpoint格式 | 与MindSpore训练环境依赖一致
Ascend 310 AI处理器 | ONNX、GEIR格式 | 搭载了ACL框架，支持OM格式模型，需要使用工具转化模型为OM格式模型。
GPU | checkpoint格式 | 与MindSpore训练环境依赖一致。
GPU | ONNX格式 | 支持ONNX推理的runtime/SDK，如TensorRT。
CPU | checkpoint格式 | 与MindSpore训练环境依赖一致。
CPU | ONNX格式 | 支持ONNX推理的runtime/SDK，如TensorRT。

> ONNX，全称Open Neural Network Exchange，是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。它使得不同的人工智能框架（如PyTorch, MXNet）可以采用相同格式存储模型数据并交互。详细了解，请参见ONNX官网<https://onnx.ai/>。

> GEIR，全称Graph Engine Intermediate Representation，类似ONNX，是华为定义的针对机器学习所设计的开放式的文件格式，能更好地适配Ascend AI处理器。

> ACL，全称Ascend Computer Language，提供Device管理、Context管理、Stream管理、内存管理、模型加载与执行、算子加载与执行、媒体数据处理等C++ API库，供用户开发深度神经网络应用。它匹配Ascend AI处理器，使能硬件的运行管理、资源管理能力。

> OM，全称Offline Model，华为Ascend AI处理器支持的离线模型，实现算子调度的优化，权值数据重排、压缩，内存使用优化等可以脱离设备完成的预处理功能。

> TensorRT，NVIDIA 推出的高性能深度学习推理的SDK，包括深度推理优化器和runtime，提高深度学习模型在边缘设备上的推断速度。详细请参见<https://developer.nvidia.com/tensorrt>。

## Ascend 910 AI处理器上推理

### 使用checkpoint格式文件推理

1. 使用`model.eval`接口来进行模型验证，你只需传入验证数据集即可，验证数据集的处理方式与训练数据集相同。
    ```python
    res = model.eval(dataset)
    ```
    其中，  
    `model.eval`为模型验证接口，对应接口说明：<https://www.mindspore.cn/api/zh-CN/r0.5/api/python/mindspore/mindspore.html#mindspore.Model.eval>。
    > 推理样例代码：<https://gitee.com/mindspore/mindspore/blob/r0.5/model_zoo/lenet/eval.py>。

2. 使用`model.predict`接口来进行推理操作。
   ```python
   model.predict(input_data)
   ```
   其中，  
   `model.eval`为推理接口，对应接口说明：<https://www.mindspore.cn/api/zh-CN/r0.5/api/python/mindspore/mindspore.html#mindspore.Model.predict>。

## Ascend 310 AI处理器上推理

### 使用ONNX与GEIR格式文件推理

Ascend 310 AI处理器上搭载了ACL框架，他支持OM格式，而OM格式需要从ONNX或者GEIR模型进行转换。所以在Ascend 310 AI处理器上推理，需要下述两个步骤：

1. 在训练平台上生成ONNX或GEIR格式模型，具体步骤请参考[模型导出-导出GEIR模型和ONNX模型](https://www.mindspore.cn/tutorial/zh-CN/r0.5/use/saving_and_loading_model_parameters.html#geironnx)。

2. 将ONNX/GEIR格式模型文件，转化为OM格式模型，并进行推理。
   - 云上（ModelArt环境），请参考[Ascend910训练和Ascend310推理的样例](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0026.html)完成推理操作。
   - 本地的裸机环境（对比云上环境，即本地有Ascend 310 AI 处理器），请参考Ascend 310 AI处理器配套软件包的说明文档。

## GPU上推理

### 使用checkpoint格式文件推理

与在Ascend 910 AI处理器上推理一样。

### 使用ONNX格式文件推理

1. 在训练平台上生成ONNX格式模型，具体步骤请参考[模型导出-导出GEIR模型和ONNX模型](https://www.mindspore.cn/tutorial/zh-CN/r0.5/use/saving_and_loading_model_parameters.html#geironnx)。

2. 在GPU上进行推理，具体可以参考推理使用runtime/SDK的文档。如在Nvidia GPU上进行推理，使用常用的TensorRT，可参考[TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt)。

## CPU上推理

### 使用checkpoint格式文件推理
与在Ascend 910 AI处理器上推理一样。

### 使用ONNX格式文件推理
与在GPU上进行推理类似，需要以下几个步骤：

1. 在训练平台上生成ONNX格式模型，具体步骤请参考[模型导出-导出GEIR模型和ONNX模型](https://www.mindspore.cn/tutorial/zh-CN/r0.5/use/saving_and_loading_model_parameters.html#geironnx)。

2. 在CPU上进行推理，具体可以参考推理使用runtime/SDK的文档。如使用ONNX Runtime，可以参考[ONNX Runtime说明文档](https://github.com/microsoft/onnxruntime)。

## 端侧推理

端侧推理需使用MindSpore Predict推理引擎，详细操作请参考[端侧推理教程](https://www.mindspore.cn/tutorial/zh-CN/r0.5/advanced_use/on_device_inference.html)。