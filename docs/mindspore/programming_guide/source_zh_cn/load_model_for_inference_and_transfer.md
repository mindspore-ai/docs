# 加载模型用于推理或迁移学习

`Ascend` `GPU` `CPU` `模型加载`

<!-- TOC -->

- [加载模型用于推理或迁移学习](#加载模型用于推理或迁移学习)
    - [概述](#概述)
    - [本地加载模型](#本地加载模型)
        - [用于推理验证](#用于推理验证)
        - [用于迁移学习](#用于迁移学习)
    - [修改checkpoint文件并重新保存](#修改checkpoint文件并重新保存)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/load_model_for_inference_and_transfer.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/master/notebook/mindspore_load_model_for_inference_and_transfer.ipynb"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_notebook.png"></a>
&nbsp;&nbsp;
<a href="https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9sb2FkX21vZGVsX2Zvcl9pbmZlcmVuY2VfYW5kX3RyYW5zZmVyLmlweW5i&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_modelarts.png"></a>

## 概述

在模型训练过程中保存在本地的CheckPoint文件，可以帮助用户进行推理或迁移学习使用。

以下通过示例来介绍如何通过本地加载，用于推理验证和迁移学习。

> 你可以在这里查看网络和数据集的定义：
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/save_model>

## 本地加载模型

### 用于推理验证

针对仅推理场景可以使用`load_checkpoint`把参数直接加载到网络中，以便进行后续的推理验证。

示例代码如下：

```python
from mindspore import Model, load_checkpoint
from mindspore.nn import SoftmaxCrossEntropyWithLogits

resnet = ResNet50()
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
# create eval dataset, mnist_path is the data path
dataset_eval = create_dataset(mnist_path)
loss = SoftmaxCrossEntropyWithLogits()
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
from mindspore import Model, load_checkpoint
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits

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

## 修改checkpoint文件并重新保存

如果想要对checkpoint进行修改，可以使用`load_checkpoint`接口，该接口会返回一个dict。

可以对这个dict进行修改，以便进行后续的操作。

```python
from mindspore import Parameter, Tensor, load_checkpoint, save_checkpoint
# 加载checkpoint文件
param_dict = load_checkpoint("lenet.ckpt")
# 可以通过遍历这个dict，查看key和value
for key, value in param_dict.items():
  # key 为string类型
  print(key)
  # value为parameter类型，使用data.asnumpy()方法可以查看其数值
  print(value.data.asnumpy())

# 拿到param_dict后，就可以对其进行基本的增删操作，以便后续使用

# 1.删除名称为"conv1.weight"的元素
del param_dict["conv1.weight"]
# 2.添加名称为"conv2.weight"的元素，设置它的值为0
param_dict["conv2.weight"] = Parameter(Tensor([0]))
# 3.修改名称为"conv1.bias"的值为1
param_dict["fc1.bias"] = Parameter(Tensor([1]))

# 把修改后的param_dict重新存储成checkpoint文件
save_list = []
# 遍历修改后的dict，把它转化成MindSpore支持的存储格式，存储成checkpoint文件
for key, value in param_dict.items():
  save_list.append({"name": key, "data": value.data})
save_checkpoint(save_list, "new.ckpt")
```
