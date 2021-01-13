# 推理模型总览

 `Linux` `Ascend` `GPU` `CPU` `推理应用` `初级` `中级` `高级`

<!-- TOC -->

- [推理模型总览](#推理模型总览)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/inference/source_zh_cn/multi_platform_inference.md" target="_blank"><img src="./_static/logo_source.png"></a>

MindSpore可以基于训练好的模型，在不同的硬件平台上执行推理任务

MindSpore支持保存两种类型的数据：训练参数和网络模型(模型中包含参数信息)

- 训练参数对应的是Checkpoint文件。
- 网络模型对应的有三种文件格式：MindIR、AIR、ONNX。

下面介绍一下这几种格式的基本概念和相关应用场景：

- Checkpoint
    - 采用了proto buffer格式，存储了网络中所有参数值的二进制文件。
    - 一般用训练任务中断后恢复训练，或训练后Fine-tune(微调)任务。
- MindIR
    - 全称MindSpore IR，是MindSpore的一种基于图表示的函数式IR，定义了可扩展的图结构以及算子的IR表示。
    - 它消除了不同后端的模型差异，一般用于跨硬件平台执行推理任务。
- AIR
    - 全称Ascend Intermediate Representation，是华为定义的针对机器学习所设计的开放式文件格式。
    - 它能更好地适应华为AI处理器，一般用在Ascend 310上进行推理任务。
- ONNX
    - 全称Open Neural Network Exchange，是一种针对机器学习模型的通用表达。
    - 一般用于不同框架间的模型迁移或在推理引擎([TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html))上使用。

按照使用环境的不同，推理有两种方式：

- 本机推理
    - 加载网络训练产生的Checkpoint文件，调用`model.predict`接口进行推理验证，具体操作可查看[本地加载模型](https://www.mindspore.cn/tutorial/training/zh-CN/master/use/load_model_for_inference_and_transfer.html)。
- 跨平台推理
    - 使用网络定义和Checkpoint文件，调用`export`接口导出模型文件，在不同平台执行推理，目前支持导出AIR(仅支持Ascend AI处理器)、MindIR、ONNX模型，具体操作可查看[保存模型](https://www.mindspore.cn/tutorial/training/zh-CN/master/use/save_model.html)。

MindSpore通过统一IR定义了网络的逻辑结构和算子的属性，实现了MindIR格式的模型文件与硬件平台解耦，实现一次训练多次部署。

首先需要使用网络定义和Checkpoint文件导出MindIR模型文件，根据不同需求执行推理任务，常见的应用场景如下：

- 云侧
    - 在Ascend 910上执行推理任务，详见：[Ascend 910 推理](https://www.mindspore.cn/tutorial/inference/zh-CN/master/multi_platform_inference_ascend_910.html)。
    - 在Ascend 310上执行推理任务，详见：[Ascend 310 推理](https://www.mindspore.cn/tutorial/inference/zh-CN/master/multi_platform_inference_ascend_310_mindir.html)。
    - 基于MindSpore Serving部署推理服务，详见：[Serving 部署推理服务](https://www.mindspore.cn/tutorial/inference/zh-CN/master/serving_example.html)。
- 端侧
    - 使用MindSpore Lite提供的离线转换模型功能工具，对模型进行转化优化后用于推理，详见：[端侧推理](https://www.mindspore.cn/lite/docs?master)。
