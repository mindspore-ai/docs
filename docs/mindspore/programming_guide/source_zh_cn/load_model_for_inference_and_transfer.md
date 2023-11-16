# 加载模型用于推理或迁移学习

`Linux` `Ascend` `GPU` `CPU` `模型加载` `初级` `中级` `高级`

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_zh_cn/load_model_for_inference_and_transfer.md)
&nbsp;&nbsp;
[![查看notebook](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_notebook.png)](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/r1.3/notebook/mindspore_load_model_for_inference_and_transfer.ipynb)
&nbsp;&nbsp;
[![在线运行](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_modelarts.png)](https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9sb2FkX21vZGVsX2Zvcl9pbmZlcmVuY2VfYW5kX3RyYW5zZmVyLmlweW5i&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c)

## 概述

在模型训练过程中保存在本地的CheckPoint文件，可以帮助用户进行推理或迁移学习使用。

以下通过示例来介绍如何通过本地加载，用于推理验证和迁移学习。

## 本地加载模型

### 用于推理验证

针对仅推理场景可以使用`load_checkpoint`把参数直接加载到网络中，以便进行后续的推理验证。

示例代码如下：

```python
resnet = ResNet50()
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
dateset_eval = create_dataset(os.path.join(mnist_path, "test"), 32, 1) # define the test dataset
loss = CrossEntropyLoss()
model = Model(resnet, loss, metrics={"accuracy"})
acc = model.eval(dataset_eval)
```

- `load_checkpoint`方法会把参数文件中的网络参数加载到模型中。加载后，网络中的参数就是CheckPoint保存的。
- `eval`方法会验证训练后模型的精度。

### 用于迁移学习

针对任务中断再训练及微调（Fine Tune）场景，可以加载网络参数和优化器参数到模型中。

示例代码如下：

```python
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
resnet = ResNet50()
opt = Momentum(resnet.trainable_params(), 0.01, 0.9)
# load the parameter into net
load_param_into_net(resnet, param_dict)
# load the parameter into optimizer
load_param_into_net(opt, param_dict)
loss = SoftmaxCrossEntropyWithLogits()
model = Model(resnet, loss, opt)
model.train(epoch, dataset)
```

- `load_checkpoint`方法会返回一个参数字典。
- `load_param_into_net`会把参数字典中相应的参数加载到网络或优化器中。
