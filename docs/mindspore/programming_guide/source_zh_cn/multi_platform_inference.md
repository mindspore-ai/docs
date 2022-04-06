# 推理模型总览

`Ascend` `GPU` `CPU` `推理应用`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/multi_platform_inference.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

MindSpore可以基于训练好的模型，在不同的硬件平台上执行推理任务。

## 模型文件

MindSpore支持保存两种类型的数据：训练参数和网络模型（模型中包含参数信息）。

- 训练参数指的是Checkpoint格式文件。
- 网络模型包括MindIR、AIR和ONNX三种格式文件。

下面介绍一下这几种格式的基本概念及其应用场景。

- Checkpoint
    - 采用了Protocol Buffers格式，存储了网络中所有的参数值。
    - 一般用于训练任务中断后恢复训练，或训练后的微调（Fine Tune）任务。
- MindIR
    - 全称MindSpore IR，是MindSpore的一种基于图表示的函数式IR，定义了可扩展的图结构以及算子的IR表示。
    - 它消除了不同后端的模型差异，一般用于跨硬件平台执行推理任务。
- ONNX
    - 全称Open Neural Network Exchange，是一种针对机器学习模型的通用表达。
    - 一般用于不同框架间的模型迁移或在推理引擎([TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html))上使用。
    - 目前MindSpore仅支持ONNX模型的导出，暂不支持加载ONNX模型进行推理。目前支持导出的模型有：Resnet50、YOLOv3_darknet53、YOLOv4、BERT。可以在[ONNX Runtime](https://onnxruntime.ai/)上使用。
- AIR
    - 全称Ascend Intermediate Representation，是华为定义的针对机器学习所设计的开放式文件格式。
    - 它能更好地适应华为AI处理器，一般用于Ascend 310上执行推理任务。

## 执行推理

按照使用环境的不同，推理可以分为以下两种方式。

1. 本机推理

    通过加载网络训练产生的Checkpoint文件，调用`model.predict`接口进行推理验证，具体操作可查看[加载Checkpoint在线推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/online_inference.html)。

2. 跨平台推理

    使用网络定义和Checkpoint文件，调用`export`接口导出模型文件，在不同平台执行推理，目前支持导出MindIR、ONNX和AIR（仅支持Ascend AI处理器）模型，具体操作可查看[保存模型](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html)。

## MindIR介绍

MindSpore通过统一IR定义了网络的逻辑结构和算子的属性，将MindIR格式的模型文件与硬件平台解耦，实现一次训练多次部署。

1. 基本介绍

    MindIR作为MindSpore的统一模型文件，同时存储了网络结构和权重参数值。同时支持部署到云端Serving和端侧Lite平台执行推理任务。

    同一个MindIR文件支持多种硬件形态的部署：

    - 云端Serving部署推理：MindSpore训练生成MindIR模型文件后，可直接发给MindSpore Serving加载，执行推理任务，而无需额外的模型转化，做到Ascend、GPU、CPU等多硬件的模型统一。
    - 端侧Lite推理部署：MindIR可直接供Lite部署使用。同时由于端侧轻量化需求，提供了模型小型化和转换功能，支持将原始MindIR模型文件由Protocol Buffers格式转化为FlatBuffers格式存储，以及网络结构轻量化，以更好的满足端侧性能、内存等要求。

2. 使用场景

    先使用网络定义和Checkpoint文件导出MindIR模型文件，再根据不同需求执行推理任务，如[在Ascend 310上执行推理任务](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_ascend_310_mindir.html)、[基于MindSpore Serving部署推理服务](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_example.html)、[端侧推理](https://www.mindspore.cn/lite/docs/zh-CN/master/index.html)。
