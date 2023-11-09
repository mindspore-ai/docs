# 模型推理总览

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_zh_cn/infer/inference.md)

MindSpore可以基于训练好的模型，在不同的硬件平台上执行推理任务。

Ascend 310是面向边缘场景的高能效高集成度AI处理器，支持对MindIR格式模型进行推理。

MindIR格式可由MindSpore CPU、GPU、Ascend 910导出，可运行在GPU、Ascend 910、Ascend 310上，推理前不需要手动执行模型转换，推理时需要安装MindSpore Lite，调用MindSpore Lite C++ API进行推理。

## 模型文件

MindSpore支持保存两种类型的数据：训练参数和网络模型（模型中包含参数信息）。

- 训练参数指的是Checkpoint格式文件。
- 网络模型包括MindIR和ONNX两种格式文件。

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
    - 目前支持导出的模型有：Resnet50、YOLOv3_darknet53、YOLOv4、BERT。可以在[ONNX Runtime](https://onnxruntime.ai/)上使用。
    - 通过昇腾[ACT工具](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_000001.html)实现在昇腾上的推理执行。

## 执行推理

按照使用环境的不同，推理可以分为以下两种方式。

1. 本机推理

    通过加载网络训练产生的Checkpoint文件，调用`model.predict`接口进行推理验证。

2. 跨平台推理

    使用网络定义和Checkpoint文件，调用`export`接口导出模型文件，在不同平台执行推理，目前支持导出MindIR和ONNX（仅支持Ascend AI处理器）模型，具体操作可查看[保存模型](https://www.mindspore.cn/tutorials/zh-CN/r2.3/beginner/save_load.html)。

## MindIR介绍

MindSpore通过统一IR定义了网络的逻辑结构和算子的属性，将MindIR格式的模型文件与硬件平台解耦，实现一次训练多次部署。

1. 基本介绍

    MindIR作为MindSpore的统一模型文件，同时存储了网络结构和权重参数值。同时支持部署到云端Serving和MindSpore Lite平台执行推理任务。

    同一个MindIR文件支持多种硬件形态的部署：

    - 云端Serving部署推理：MindSpore训练生成MindIR模型文件后，可直接发给MindSpore Serving加载，执行推理任务，而无需额外的模型转化，做到Ascend、GPU、CPU等多硬件的模型统一。
    - 使用MindSpore Lite推理部署：MindIR模型可直接使用Lite进行部署。支持在Ascend、英伟达GPU、CPU等云侧服务器上部署，同时也支持在手机等资源受限的端侧硬件上部署。

2. 使用场景

    先使用网络定义和Checkpoint文件导出MindIR模型文件，再根据不同需求执行推理任务，如[基于MindSpore Serving部署推理服务](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_example.html)、[Lite推理](https://www.mindspore.cn/lite/docs/zh-CN/r2.3/index.html)。

## model.eval模型验证

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
`model.eval`为模型验证接口，对应接口说明[mindspore.train.Model.eval](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/train/mindspore.train.Model.html#mindspore.train.Model.eval)。

> 推理样例代码[eval.py](https://gitee.com/mindspore/models/blob/master/research/cv/lenet/eval.py)。

## 使用`model.predict`接口进行推理操作

```python
model.predict(input_data)
```

其中，  
`model.predict`为推理接口，对应接口说明[mindspore.train.Model.predict](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/train/mindspore.train.Model.html#mindspore.train.Model.predict)。
