# 保存模型

`Ascend` `GPU` `CPU` `模型导出`

<!-- TOC -->

- [保存模型](#保存模型)
    - [概述](#概述)
    - [保存CheckPoint格式文件](#保存checkpoint格式文件)
        - [使用callback机制](#使用callback机制)
            - [CheckPoint配置策略](#checkpoint配置策略)
        - [使用save_checkpoint方法](#使用save_checkpoint方法)
            - [`save_obj`和`ckpt_file_name`参数](#save_obj和ckpt_file_name参数)
            - [`integrated_save`参数](#integrated_save参数)
            - [`async_save`参数](#async_save参数)
            - [`append_dict`参数](#append_dict参数)
    - [导出MindIR格式文件](#导出mindir格式文件)
    - [导出AIR格式文件](#导出air格式文件)
    - [导出ONNX格式文件](#导出onnx格式文件)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/save_model.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.5/notebook/mindspore_save_model.ipynb"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_notebook.png"></a>
&nbsp;&nbsp;
<a href="https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9zYXZlX21vZGVsLmlweW5i&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_modelarts.png"></a>

## 概述

在模型训练过程中，可以添加检查点(CheckPoint)用于保存模型的参数，以便执行推理及再训练使用。如果想继续在不同硬件平台上做推理，可通过网络和CheckPoint格式文件生成对应的MindIR、AIR和ONNX格式文件。

- MindIR：MindSpore的一种基于图表示的函数式IR，定义了可扩展的图结构以及算子的IR表示，它消除了不同后端的模型差异。可以把在Ascend 910训练好的模型，在Ascend 310、GPU以及MindSpore Lite端侧上执行推理。
- CheckPoint：MindSpore的存储了所有训练参数值的二进制文件。采用了Google的Protocol Buffers机制，与开发语言、平台无关，具有良好的可扩展性。CheckPoint的protocol格式定义在`mindspore/ccsrc/utils/checkpoint.proto`中。
- AIR：全称Ascend Intermediate Representation，类似ONNX，是华为定义的针对机器学习所设计的开放式的文件格式，能更好地适配Ascend AI处理器。
- ONNX：全称Open Neural Network Exchange，是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。

以下通过示例来介绍保存CheckPoint格式文件和导出MindIR、AIR和ONNX格式文件的方法。

## 保存CheckPoint格式文件

下面介绍两种保存checkpoint文件的方法。

### 使用callback机制

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

> - 当执行分布式并行训练任务时，每个进程需要设置不同`directory`参数，用以保存CheckPoint文件到不同的目录，以防文件发生读写错乱。

#### CheckPoint配置策略

MindSpore提供了两种保存CheckPoint策略：迭代策略和时间策略，可以通过创建`CheckpointConfig`对象设置相应策略。
`CheckpointConfig`中共有四个参数可以设置：

- save_checkpoint_steps：表示每隔多少个step保存一个CheckPoint文件，默认值为1。
- save_checkpoint_seconds：表示每隔多少秒保存一个CheckPoint文件，默认值为0。
- keep_checkpoint_max：表示最多保存多少个CheckPoint文件，默认值为5。
- keep_checkpoint_per_n_minutes：表示每隔多少分钟保留一个CheckPoint文件，默认值为0。

`save_checkpoint_steps`和`keep_checkpoint_max`为迭代策略，根据训练迭代的次数进行配置。
`save_checkpoint_seconds`和`keep_checkpoint_per_n_minutes`为时间策略，根据训练的时长进行配置。

两种策略不能同时使用，迭代策略优先级高于时间策略，当同时设置时，只有迭代策略可以生效。当参数显示设置为`None`时，表示放弃该策略。在迭代策略脚本正常结束的情况下，会默认保存最后一个step的CheckPoint文件。

### 使用save_checkpoint方法

可以使用`save_checkpoint`函数把自定义信息保存成 checkpoint文件，函数声明如下：

```python
def save_checkpoint(save_obj, ckpt_file_name, integrated_save=True,
                    async_save=False, append_dict=None, enc_key=None, enc_mode="AES-GCM")
```

其中必填的参数有：`save_obj`、`ckpt_file_name`。

下面通过具体示例来说明如何使用每个参数。

#### `save_obj`和`ckpt_file_name`参数

**`save_obj`**：可以传入一个  Cell类对象或一个list。
**`ckpt_file_name`**：string类型，表示保存checkpoint文件的名称。

```python
from mindspore import save_checkpoint, Tensor
from mindspore import dtype as mstype
```

1. 传入Cell对象

    ```python
    net = LeNet()
    save_checkpoint(net, "lenet.ckpt")
    ```

    ​执行后就可以把net中的参数保存成`lenet.ckpt`文件。

2. 传入list对象

    list格式如下：[{"name": param_name, "data": param_data}]，它由一组dict对象组成。

    `param_name`为需要保存对象的名称，`param_data`为需要保存对象的数据，它为Tensor类型。

    ```python
    save_list = [{"name": "lr", "data": Tensor(0.01, mstype.float32)}, {"name": "train_epoch", "data": Tensor(20, mstype.int32)}]
    save_checkpoint(save_list, "hyper_param.ckpt")
    ```

    执行后就可以把`save_list`保存成`hyper_param.ckpt`文件。

#### `integrated_save`参数

**`integrated_save`**：bool类型，表示参数是否合并保存，默认为True。在模型并行场景下，Tensor会被切分到不同卡所运行的程序中。如果`integrated_save`设置为True，则这些被切分的Tensor会被合并保存到每个checkpoint文件中，这样checkpoint文件保存的就是完整的训练参数。

```python
save_checkpoint(net, "lenet.ckpt", integrated_save=True)
```

#### `async_save`参数

**`async_save`**：bool类型，表示是否开启异步保存功能，默认为False。如果设置为True，则会开启多线程执行写checkpoint文件操作，从而可以并行执行训练和保存任务，在训练大规模网络时会节省脚本运行的总时长。

```python
save_checkpoint(net, "lenet.ckpt", async_save=True)
```

#### `append_dict`参数

**`append_dict`**：dict类型，表示需要额外保存的信息，例如：

```python
save_dict = {"epoch_num": 2, "lr": 0.01}
save_checkpoint(net, "lenet.ckpt",append_dict=save_dict)
```

执行后，除了net中的参数，`save_dict`的信息也会保存在`lenet.ckpt`中。
目前只支持基础类型的保存，包括int、float、bool等。

## 导出MindIR格式文件

如果想跨平台或硬件执行推理（如昇腾AI处理器、MindSpore端侧、GPU等），可以通过网络定义和CheckPoint生成MindIR格式模型文件。当前支持基于静态图，且不包含控制流语义的推理网络导出。导出该格式文件的代码样例如下：

```python
import numpy as np
from mindspore import Tensor, export, load_checkpoint, load_param_into_net

resnet = ResNet50()
# load the parameter into net
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='MINDIR')
```

若希望在MindIR中保存模型推理时需要的预处理操作信息，可以将数据集对象传入export接口：

```python
de_dataset = create_dataset_for_renset(mode="eval")
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='MINDIR', dataset=de_dataset)
```

> - `input`为`export`方法的入参，代表网络的输入，如果网络有多个输入，需要一同传进`export`方法。 例如：`export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='MINDIR')`
> - 如果`file_name`没有包含".mindir"后缀，系统会为其自动添加".mindir"后缀。
> - 需要确保数据集对象处于evaluation的状态，即正在使用推理相关的算子，否则可能无法达到预期的结果。

为了避免protobuf的硬件限制，当导出的模型参数大小超过1G时，框架默认会把网络结构和参数分开保存。

- 网络结构文件的名称以用户指定前缀加`_graph.mindir`结尾。
- 同级目录下，会生用户指定前缀加`_variables`的文件夹，里面存放网络的参数。

以上述代码为例，如果模型中参数大小超过1G，生成的目录结构如下：

```text
resnet50-2_32_graph.mindir
resnet50-2_32_variables
    data_0
    data_1
    ...
```

> 加载时，如果传入的文件名以`_graph.mindir`结尾，框架会自动查找同级目录下的参数文件。

## 导出AIR格式文件

如果想在昇腾AI处理器上执行推理，还可以通过网络定义和CheckPoint生成AIR格式模型文件。导出该格式文件的代码样例如下：

```python
import numpy as np
from mindspore import Tensor, export, load_checkpoint, load_param_into_net

resnet = ResNet50()
# load the parameter into net
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='AIR')
```

`input`用来指定导出模型的输入shape以及数据类型。

> - `input`为`export`方法的入参，代表网络的输入，如果网络有多个输入，需要一同传进`export`方法。 例如：`export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='AIR')`
> - 如果`file_name`没有包含".air"后缀，系统会为其自动添加".air"后缀。

## 导出ONNX格式文件

当有了CheckPoint文件后，如果想继续在昇腾AI处理器、GPU、CPU等多种硬件上做推理，需要通过网络和CheckPoint生成对应的ONNX格式模型文件。导出该格式文件的代码样例如下：

```python
import numpy as np
from mindspore import Tensor, export, load_checkpoint, load_param_into_net

resnet = ResNet50()
# load the parameter into net
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='ONNX')
```

> - `input`为`export`方法的入参，代表网络的输入，如果网络有多个输入，需要一同传进`export`方法。 例如：`export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='ONNX')`
> - 如果`file_name`没有包含".onnx"后缀，系统会为其自动添加".onnx"后缀。
> - 目前ONNX格式导出仅支持ResNet系列、YOLOV3、YOLOV4、BERT网络。
