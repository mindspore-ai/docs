# 模型推理总览

`Ascend` `GPU` `CPU` `推理应用`

<a href="https://gitee.com/mindspore/docs/blob/r1.7/tutorials/experts/source_zh_cn/infer/inference.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

MindSpore可以基于训练好的模型，在不同的硬件平台上执行推理任务。

Ascend 310是面向边缘场景的高能效高集成度AI处理器，支持对MindIR格式和AIR格式模型进行推理。

MindIR格式可由MindSpore CPU、GPU、Ascend 910导出，可运行在GPU、Ascend 910、Ascend 310上，推理前不需要手动执行模型转换，推理时需要安装MindSpore，调用MindSpore C++ API进行推理。

AIR格式仅MindSpore Ascend 910可导出，仅Ascend 310可推理，推理前需使用Ascend CANN中atc工具进行模型转换，推理时不依赖MindSpore，仅需Ascend CANN软件包。

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

    通过加载网络训练产生的Checkpoint文件，调用`model.predict`接口进行推理验证。

2. 跨平台推理

    使用网络定义和Checkpoint文件，调用`export`接口导出模型文件，在不同平台执行推理，目前支持导出MindIR、ONNX和AIR（仅支持Ascend AI处理器）模型，具体操作可查看[保存模型](https://www.mindspore.cn/tutorials/zh-CN/r1.7/advanced/train/save.html)。

## MindIR介绍

MindSpore通过统一IR定义了网络的逻辑结构和算子的属性，将MindIR格式的模型文件与硬件平台解耦，实现一次训练多次部署。

1. 基本介绍

    MindIR作为MindSpore的统一模型文件，同时存储了网络结构和权重参数值。同时支持部署到云端Serving和端侧Lite平台执行推理任务。

    同一个MindIR文件支持多种硬件形态的部署：

    - 云端Serving部署推理：MindSpore训练生成MindIR模型文件后，可直接发给MindSpore Serving加载，执行推理任务，而无需额外的模型转化，做到Ascend、GPU、CPU等多硬件的模型统一。
    - 端侧Lite推理部署：MindIR可直接供Lite部署使用。同时由于端侧轻量化需求，提供了模型小型化和转换功能，支持将原始MindIR模型文件由Protocol Buffers格式转化为FlatBuffers格式存储，以及网络结构轻量化，以更好的满足端侧性能、内存等要求。

2. 使用场景

    先使用网络定义和Checkpoint文件导出MindIR模型文件，再根据不同需求执行推理任务，如[在Ascend 310上执行推理任务](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/infer/ascend_310_mindir.html)、[基于MindSpore Serving部署推理服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.7/serving_example.html)、[端侧推理](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/index.html)。

## model.eval模型验证

### 模型已保存在本地

首先构建模型，然后使用`mindspore`模块的`load_checkpoint`和`load_param_into_net`从本地加载模型与参数，传入验证数据集后即可进行模型推理，验证数据集的处理方式与训练数据集相同。

```python
network = LeNet5(cfg.num_classes)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
model = Model(network, net_loss, metrics={"Accuracy": Accuracy()})

print("============== Starting Testing ==============")
param_dict = load_checkpoint(args.ckpt_path)
load_param_into_net(network, param_dict)
dataset = create_dataset(os.path.join(args.data_path, "test"),
                            cfg.batch_size,)
acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
print("============== {} ==============".format(acc))
```

其中，  
`model.eval`为模型验证接口，对应接口说明：<https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.html#mindspore.Model.eval>。

> 推理样例代码：<https://gitee.com/mindspore/models/blob/r1.7/official/cv/lenet/eval.py>。

### 使用MindSpore Hub从华为云加载模型

首先构建模型，然后使用`mindspore_hub.load`从云端加载模型参数，传入验证数据集后即可进行推理，验证数据集的处理方式与训练数据集相同。

```python
model_uid = "mindspore/ascend/0.7/googlenet_v1_cifar10"  # using GoogleNet as an example.
network = mindspore_hub.load(model_uid, num_classes=10)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
model = Model(network, net_loss, metrics={"Accuracy": Accuracy()})

print("============== Starting Testing ==============")
dataset = create_dataset(os.path.join(args.data_path, "test"),
                            cfg.batch_size,)
acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
print("============== {} ==============".format(acc))
```

其中，  
`mindspore_hub.load`为加载模型参数接口，对应接口说明：<https://www.mindspore.cn/hub/docs/zh-CN/r1.7/hub.html#mindspore-hubload>。

## 使用`model.predict`接口进行推理操作

```python
model.predict(input_data)
```

其中，  
`model.predict`为推理接口，对应接口说明：<https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.html#mindspore.Model.predict>。
