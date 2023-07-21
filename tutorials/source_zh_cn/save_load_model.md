# 保存及加载模型

`Ascend` `GPU` `CPU` `入门` `模型导出` `模型加载`

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.5/tutorials/source_zh_cn/save_load_model.md)

上一节我们训练完网络，本节将会学习如何保存模型和加载模型，以及如何将保存的模型导出成特定格式到不同平台进行推理。

## 保存模型

保存模型的接口有主要2种方式：

1. 一种是简单的对网络模型进行保存，可以在训练前后进行保存，优点是接口简单易用，但是只保留执行命令时候的网络模型状态；

2. 另外一种是在网络模型训练中进行保存，MindSpore在网络模型训练的过程中，自动保存训练时候设定好的epoch数和step数的参数，也就是把模型训练过程中产生的中间权重参数也保存下来，方便进行网络微调和停止训练。

### 直接保存模型

使用MindSpore提供的save_checkpoint保存模型，传入网络和保存路径：

```python
import mindspore as ms

# 定义的网络模型为net，一般在训练前或者训练后使用
ms.save_checkpoint(net, "./MyNet.ckpt")
```

其中，`net`为训练网络，定义方法可参考[建立神经网络](https://www.mindspore.cn/tutorials/zh-CN/r1.5/model.html)。

### 训练过程中保存模型

在模型训练的过程中，使用`model.train`里面的`callbacks`参数传入保存模型的对象 `ModelCheckpoint`，可以保存模型参数，生成CheckPoint(简称ckpt)文件。

```python
from mindspore.train.callback import ModelCheckpoint

ckpt_cb = ModelCheckpoint()
model.train(epoch_num, dataset, callbacks=ckpt_cb)
```

其中，`epoch_num`为训练轮次，定义方法可参考[训练模型](https://www.mindspore.cn/tutorials/zh-CN/r1.5/optimization.html)。`dataset`为加载的数据集，定义方法可参考[数据加载及处理](https://www.mindspore.cn/tutorials/zh-CN/r1.5/dataset.html)。

用户可以根据具体需求对CheckPoint策略进行配置。具体用法如下：

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
ckpt_cb = ModelCheckpoint(prefix='resnet50', directory=None, config=config_ck)
model.train(epoch_num, dataset, callbacks=ckpt_cb)
```

上述代码中，首先需要初始化一个`CheckpointConfig`类对象，用来设置保存策略。

- `save_checkpoint_steps`表示每隔多少个step保存一次。
- `keep_checkpoint_max`表示最多保留CheckPoint文件的数量。
- `prefix`表示生成CheckPoint文件的前缀名。
- `directory`表示存放文件的目录。

创建一个`ModelCheckpoint`对象把它传递给`model.train`方法，就可以在训练过程中使用CheckPoint功能了。

生成的CheckPoint文件如下：

```text
resnet50-graph.meta # 编译后的计算图
resnet50-1_32.ckpt  # CheckPoint文件后缀名为'.ckpt'
resnet50-2_32.ckpt  # 文件的命名方式表示保存参数所在的epoch和step数
resnet50-3_32.ckpt  # 表示保存的是第3个epoch的第32个step的模型参数
...
```

如果用户使用相同的前缀名，运行多次训练脚本，可能会生成同名CheckPoint文件。MindSpore为方便用户区分每次生成的文件，会在用户定义的前缀后添加"_"和数字加以区分。如果想要删除`.ckpt`文件时，请同步删除`.meta` 文件。

例：`resnet50_3-2_32.ckpt` 表示运行第3次脚本生成的第2个epoch的第32个step的CheckPoint文件。

## 加载模型

要加载模型权重，需要先创建相同模型的实例，然后使用`load_checkpoint`和`load_param_into_net`方法加载参数。

示例代码如下：

```python
from mindspore import load_checkpoint, load_param_into_net

resnet = ResNet50()
# 将模型参数存入parameter的字典中
param_dict = load_checkpoint("resnet50-2_32.ckpt")
# 将参数加载到网络中
load_param_into_net(resnet, param_dict)
model = Model(resnet, loss, metrics={"accuracy"})
```

- `load_checkpoint`方法会把参数文件中的网络参数加载到字典`param_dict`中。
- `load_param_into_net`方法会把字典`param_dict`中的参数加载到网络或者优化器中，加载后，网络中的参数就是CheckPoint保存的。

### 模型验证

针对仅推理场景，把参数直接加载到网络中，以便后续的推理验证。示例代码如下：

```python
# 定义验证数据集
dataset_eval = create_dataset(os.path.join(mnist_path, "test"), 32, 1)

# 调用eval()进行推理
acc = model.eval(dataset_eval)
```

### 用于迁移学习

针对任务中断再训练及微调（Fine-tuning）场景，可以加载网络参数和优化器参数到模型中。示例代码如下：

```python
# 设置训练轮次
epoch = 1
# 定义训练数据集
dataset = create_dataset(os.path.join(mnist_path, "train"), 32, 1)
# 调用train()进行训练
model.train(epoch, dataset)
```

## 导出模型

在模型训练过程中，可以添加检查点（CheckPoint）用于保存模型的参数，以便执行推理及再训练使用。如果想继续在不同硬件平台上做推理，可通过网络和CheckPoint格式文件生成对应的MindIR、AIR或ONNX格式文件。

以下通过示例来介绍保存CheckPoint格式文件和导出MindIR、AIR或ONNX格式文件的方法。

> MindSpore是一个全场景AI框架，使用MindSpore IR统一网络模型中间表达式，因此推荐使用MindIR作为导出格式文件。

### 导出MindIR格式

当有了CheckPoint文件后，如果想跨平台或者硬件执行推理(如昇腾AI处理器、MindSpore端侧、GPU等)，可以通过定义网络和CheckPoint生成MINDIR格式模型文件。当前支持基于静态图，且不包含控制流语义的推理网络导出。导出该格式文件的代码样例如下：

```python
from mindspore import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np

resnet = ResNet50()
# 将模型参数存入parameter的字典中
param_dict = load_checkpoint("resnet50-2_32.ckpt")

# 将参数加载到网络中
load_param_into_net(resnet, param_dict)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='MINDIR')
```

> - `input`用来指定导出模型的输入shape以及数据类型，如果网络有多个输入，需要一同传进`export`方法。 例如：`export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='MINDIR')`
> - 如果`file_name`没有包含".mindir"后缀，系统会为其自动添加".mindir"后缀。

### 其他格式导出

#### 导出AIR格式文件

当有了CheckPoint文件后，如果想继续在昇腾AI处理器上做推理，需要通过网络和CheckPoint生成对应的AIR格式模型文件。导出该格式文件的代码样例如下：

```python
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='AIR')
```

> - `input`用来指定导出模型的输入shape以及数据类型，如果网络有多个输入，需要一同传进`export`方法。 例如：`export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='AIR')`
> - 如果`file_name`没有包含".air"后缀，系统会为其自动添加".air"后缀。

#### 导出ONNX格式文件

当有了CheckPoint文件后，如果想继续在其他三方硬件上进行推理，需要通过网络和CheckPoint生成对应的ONNX格式模型文件。导出该格式文件的代码样例如下：

```python
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='ONNX')
```

> - `input`用来指定导出模型的输入shape以及数据类型，如果网络有多个输入，需要一同传进`export`方法。 例如：`export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='ONNX')`
> - 如果`file_name`没有包含".onnx"后缀，系统会为其自动添加".onnx"后缀。
> - 目前ONNX格式导出仅支持ResNet系列、BERT网络。
