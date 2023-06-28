# 保存模型

`Linux` `Ascend` `GPU` `CPU` `模型导出` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_zh_cn/use/save_model.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

在模型训练过程中，可以添加检查点(CheckPoint)用于保存模型的参数，以便进行推理及再训练使用。如果想继续在不同硬件平台上做推理，可通过网络和CheckPoint格式文件生成对应的MINDIR、AIR和ONNX格式文件。

- MINDIR：MindSpore的一种基于图表示的函数式IR，其最核心的目的是服务于自动微分变换，目前可用于MindSpore Lite端侧推理。
- CheckPoint：MindSpore的存储了所有训练参数值的二进制文件。采用了Google的Protocol Buffers机制，与开发语言、平台无关，具有良好的可扩展性。CheckPoint的protocol格式定义在`mindspore/ccsrc/utils/checkpoint.proto`中。
- AIR：全称Ascend Intermediate Representation，类似ONNX，是华为定义的针对机器学习所设计的开放式的文件格式，能更好地适配Ascend AI处理器。
- ONNX：全称Open Neural Network Exchange，是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。

以下通过示例来介绍保存CheckPoint格式文件和导出MINDIR、AIR和ONNX格式文件的方法。

## 保存CheckPoint格式文件

在模型训练的过程中，使用Callback机制传入回调函数`ModelCheckpoint`对象，可以保存模型参数，生成CheckPoint文件。

通过`CheckpointConfig`对象可以设置CheckPoint的保存策略。保存的参数分为网络参数和优化器参数。

`ModelCheckpoint`提供默认配置策略，方便用户快速上手。具体用法如下：

```python
from mindspore.train.callback import ModelCheckpoint
ckpoint_cb = ModelCheckpoint()
model.train(epoch_num, dataset, callbacks=ckpoint_cb)
```

用户可以根据具体需求对CheckPoint策略进行配置。具体用法如下：

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
ckpoint_cb = ModelCheckpoint(prefix='resnet50', directory=None, config=config_ck)
model.train(epoch_num, dataset, callbacks=ckpoint_cb)
```

上述代码中，首先需要初始化一个`CheckpointConfig`类对象，用来设置保存策略。

- `save_checkpoint_steps`表示每隔多少个step保存一次。
- `keep_checkpoint_max`表示最多保留CheckPoint文件的数量。
- `prefix`表示生成CheckPoint文件的前缀名。
- `directory`表示存放文件的目录。

创建一个`ModelCheckpoint`对象把它传递给model.train方法，就可以在训练过程中使用CheckPoint功能了。

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

> - 当保存的单个模型参数较大时(超过64M)，会因为Protobuf自身对数据大小的限制，导致保存失败。这时可通过设置环境变量`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`解除限制。
> - 当执行分布式并行训练任务时，每个进程需要设置不同`directory`参数，用以保存CheckPoint文件到不同的目录，以防文件发生读写错乱。

### CheckPoint配置策略

MindSpore提供了两种保存CheckPoint策略: 迭代策略和时间策略，可以通过创建`CheckpointConfig`对象设置相应策略。
`CheckpointConfig`中共有四个参数可以设置：

- save_checkpoint_steps: 表示每隔多少个step保存一个CheckPoint文件，默认值为1。
- save_checkpoint_seconds：表示每隔多少秒保存一个CheckPoint文件，默认值为0。
- keep_checkpoint_max：表示最多保存多少个CheckPoint文件，默认值为5。
- keep_checkpoint_per_n_minutes：表示每隔多少分钟保留一个CheckPoint文件，默认值为0。

`save_checkpoint_steps`和`keep_checkpoint_max`为迭代策略，根据训练迭代的次数进行配置。
`save_checkpoint_seconds`和`keep_checkpoint_per_n_minutes`为时间策略，根据训练的时长进行配置。

两种策略不能同时使用，迭代策略优先级高于时间策略，当同时设置时，只有迭代策略可以生效。当参数显示设置为`None`时，表示放弃该策略。在迭代策略脚本正常结束的情况下，会默认保存最后一个step的CheckPoint文件。

## 导出MINDIR格式文件

当有了CheckPoint文件后，如果想继续在MindSpore Lite端侧做推理，需要通过网络和CheckPoint生成对应的MINDIR格式模型文件。当前支持基于静态图，且不包含控制流语义的推理网络导出。导出该格式文件的代码样例如下：

```python
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np
resnet = ResNet50()
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
# load the parameter into net
load_param_into_net(resnet, param_dict)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32.mindir', file_format='MINDIR')
```

建议使用`.mindir`作为MINDIR格式文件的后缀名。

> `input`为`export`方法的入参，代表网络的输入，如果网络有多个输入，需要一同传进`export`方法。
> 例如：`export(network, Tensor(input1), Tensor(input2), file_name='network.mindir', file_format='MINDIR')`

## 导出AIR格式文件

当有了CheckPoint文件后，如果想继续在昇腾AI处理器上做推理，需要通过网络和CheckPoint生成对应的AIR格式模型文件。导出该格式文件的代码样例如下：

```python
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np
resnet = ResNet50()
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
# load the parameter into net
load_param_into_net(resnet, param_dict)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32.air', file_format='AIR')
```

使用`export`接口之前，需要先导入`mindspore.train.serialization`。

`input`用来指定导出模型的输入shape以及数据类型。

建议使用`.air`作为AIR格式文件的后缀名。

> `input`为`export`方法的入参，代表网络的输入，如果网络有多个输入，需要一同传进`export`方法。
> 例如：`export(network, Tensor(input1), Tensor(input2), file_name='network.air', file_format='AIR')`

## 导出ONNX格式文件

当有了CheckPoint文件后，如果想继续在昇腾AI处理器、GPU、CPU等多种硬件上做推理，需要通过网络和CheckPoint生成对应的ONNX格式模型文件。导出该格式文件的代码样例如下：

```python
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np
resnet = ResNet50()
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
# load the parameter into net
load_param_into_net(resnet, param_dict)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32.onnx', file_format='ONNX')
```

建议使用`.onnx`作为ONNX格式文件的后缀名。

> `input`为`export`方法的入参，代表网络的输入，如果网络有多个输入，需要一同传进`export`方法。
> 例如：`export(network, Tensor(input1), Tensor(input2), file_name='network.onnx', file_format='ONNX')`
