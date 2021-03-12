# 基于Dump功能对网络数据进行分析

`Linux` `Ascend` `GPU` `模型调优` `中级` `高级`

<!-- TOC -->

- [基于Dump功能对网络数据进行分析](#基于dump功能对网络数据进行分析对比)
    - [概述](#概述)
        - [使用场景](#使用场景)
    - [整体流程](#整体流程)
    - [ANF-IR介绍](#ANF-IR介绍)
        - [IR文件获取](#IR文件获取)
        - [使用IR文件进行调试](#使用IR文件进行调试)
    - [Dump功能介绍](#dump功能介绍)
        - [Dump模式](#dump模式)
        - [同步Dump](#同步dump)
            - [同步Dump操作步骤](#同步dump操作步骤)
            - [同步Dump数据对象](#dump数据对象)
            - [同步Dump数据格式介绍](#同步dump数据格式介绍)
            - [同步Dump数据分析样例](#同步dump数据分析样例)
        - [异步Dump](#异步dump)
            - [异步Dump操作步骤](#异步dump操作步骤)
            - [异步Dump数据对象](#dump数据对象)
            - [异步Dump数据格式介绍](#异步dump数据格式介绍)
            - [异步Dump数据格式样例](#异步dump数据格式样例)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/training/source_zh_cn/advanced_use/dump.md" target="_blank"><img src="../_static/logo_source.png"></a>&nbsp;&nbsp;

## 概述

为了对训练过程进行分析，用户需要感知训练过程中算子的输入和输出数据。

- 对于动态图模式，MindSpore提供了Python原生执行能力，用户可以在网络脚本运行过程中查看记录相应的输入输出，详情见[使用PyNative模式调试](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/debug_in_pynative_mode.html) 。

- 对于静态图模式，MindSpore提供了Dump功能，用来将模型训练中的图以及算子的输入输出数据保存到磁盘文件。

本文针对静态图模式，介绍如何基于Dump功能对网络数据进行分析对比。

### 使用场景

1. 静态图算子结果分析。

   通过Dump功能获得的IR图可以了解脚本代码与执行算子的映射关系（详情见[使用IR文件进行调试](#使用ir文件进行调试)）。结合执行算子的输入和输出数据，可以分析训练过程中可能存在的溢出、梯度爆炸与消失等问题，反向跟踪到脚本中可能存在问题的代码。

2. 特征图分析。

   通过获取图层的输出数据，分析特征图的信息。

3. 模型迁移。

   在将模型从第三方框架（TensorFlow、PyTorch）迁移到MindSpore的场景中，通过比对相同位置算子的输出数据，分析第三方框架和MindSpore对于同一模型的训练结果是否足够接近，来定位模型的精度问题。

## 整体流程

1. 从脚本找到对应的算子

    首先利用Dump功能获取相关的IR图文件，参考[IR文件获取](#ir文件获取)。然后通过图文件找到脚本中代码对应的算子，参考[使用IR文件进行调试](#使用ir文件进行调试)。

2. 从算子到Dump数据

    在了解脚本和算子的映射关系后，可以确定想要分析的算子名称，从而找到算子对应的dump文件，参考[同步Dump数据对象](#同步dump数据对象)和[异步Dump数据对象](#异步dump数据对象)。

3. 分析Dump数据

    通过解析Dump数据，可以与其他第三方框架进行对比。同步Dump数据格式参考[同步Dump数据格式介绍](#同步dump数据格式介绍)，异步Dump数据格式参考[异步Dump数据格式介绍](#异步dump数据格式介绍)。

## ANF-IR介绍

中间表示（IR）是程序编译过程中介于源语言和目标语言之间的程序表示，以方便编译器进行程序分析和优化，因此IR的设计需要考虑从源语言到目标语言的转换难度，同时考虑程序分析和优化的易用性和性能。

MindIR是一种基于图表示的函数式IR，其最核心的目的是服务于自动微分变换。自动微分采用的是基于函数式编程框架的变换方法，因此IR采用了接近于ANF函数式的语义。

关于ANF-IR的具体介绍，可以参考[MindSpore IR 文法定义](https://www.mindspore.cn/doc/note/zh-CN/master/design/mindspore/mindir.html#id2) 。

### IR文件获取

使用Dump功能将自动生成最终执行图的IR文件，Dump功能的配置见[同步Dump操作步骤](#同步dump操作步骤)和[异步Dump操作步骤](#异步dump操作步骤)，最终执行图IR文件命名和目录结构见[同步Dump数据对象](#同步dump数据对象)和[异步Dump数据对象](#异步dump数据对象)。

也可以在运行MindSpore脚本时，配置`context.set_context(save_graphs=True, save_graphs_path=“xxx”)`，会在指定路径"xxx"下（默认为脚本执行目录）保存图编译过程中生成的一些中间文件（IR文件），通过这些IR文件可以查看分析整个计算图的变换优化过程。 `set_context`的详情可参考[mindspore.context API](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.context.html#mindspore.context.set_context) 。

IR文件可以用`vi`命令查看。IR文件中包含了算子全名，和算子在计算图中输入和输出的依赖，也包含从算子到相应脚本代码的Trace信息。

### 使用IR文件进行调试

对于Ascend场景，在通过Dump功能将脚本对应的图保存到磁盘上后，会产生最终执行图`ms_output_trace_code_graph_{graph_id}.ir`。该文件中保存了对应的图中每个算子的堆栈信息，记录了算子对应的生成脚本 。

以Alexnet脚本为例：

```python
import mindspore.nn as nn
import mindspore.ops as ops


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     has_bias=has_bias, pad_mode=pad_mode)


def fc_with_initialize(input_channels, out_channels, has_bias=True):
    return nn.Dense(input_channels, out_channels, has_bias=has_bias)


class AlexNet(nn.Cell):
    """
    Alexnet
    """
    def __init__(self, num_classes=10, channel=3, phase='train', include_top=True):
        super(AlexNet, self).__init__()
        self.conv1 = conv(channel, 64, 11, stride=4, pad_mode="same", has_bias=True)
        self.conv2 = conv(64, 128, 5, pad_mode="same", has_bias=True)
        self.conv3 = conv(128, 192, 3, pad_mode="same", has_bias=True)
        self.conv4 = conv(192, 256, 3, pad_mode="same", has_bias=True)
        self.conv5 = conv(256, 256, 3, pad_mode="same", has_bias=True)
        self.relu = ops.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        self.include_top = include_top
        if self.include_top:
            dropout_ratio = 0.65
            if phase == 'test':
                dropout_ratio = 1.0
            self.flatten = nn.Flatten()
            self.fc1 = fc_with_initialize(6 * 6 * 256, 4096)
            self.fc2 = fc_with_initialize(4096, 4096)
            self.fc3 = fc_with_initialize(4096, num_classes)
            self.dropout = nn.Dropout(dropout_ratio)

    def construct(self, x):
        """define network"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
```

若要定位第58行代码`x = self.conv3(x)`的问题，可以在该脚本生成的`ms_output_trace_code_graph_{graph_id}.ir`文件中搜索该行代码，可以找到多个相关算子。如[同步Dump数据格式介绍](#同步dump数据格式介绍)所示，`Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op107`就是其中之一的算子。

后续可以参考[同步Dump数据对象](#同步dump数据对象)，找到算子对应的dump文件。

## Dump功能介绍

### Dump模式

MindSpore提供了同步Dump与异步Dump两种模式。不同模式所需要的配置文件和dump出来的数据格式不同。

- 同步模式较异步模式会占用更多内存，但易用性更好。

- 一般对于中小型网络（如ResNet）等，推荐优先使用同步Dump模式。如果因为模型过大导致需要的内存超过系统限制，再使用异步Dump。

- 同步Dump目前支持Ascend和GPU，暂不支持CPU。

- 异步Dump目前仅支持Ascend。

### 同步Dump

同步Dump的机制是在网络训练过程中每个step执行结束后， Host侧发起Dump动作，从Device上拷贝算子地址里面的数据到Host，并保存文件。同步Dump会默认关闭算子间的内存复用，避免读到脏数据。

#### 同步Dump操作步骤

1. 创建json格式的配置文件

   创建一个json格式的文件作为Dump的配置文件，如`data_dump.json`。
   配置文件的内容规范介绍详见[同步Dump功能介绍](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/custom_debugging_info.html#id4) 。

   若配置文件中`trans_flag`为`True`，则数据会以Host侧的4D格式（NCHW）格式保存，若为`False`，则保留Device侧的数据格式。
   若需要Dump全量或部分算子，则可以详见参考配置文件中的[dump_mode](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/custom_debugging_info.html#id4) 介绍。

2. 设置Dump环境变量

   ```bash
   export MINDSPORE_DUMP_CONFIG=${xxx}
   ```

   其中"xxx"为配置文件的绝对路径，如：

   ```bash
   export MINDSPORE_DUMP_CONFIG=/path/to/data_dump.json
   ```

3. 启动训练

   训练启动后，若正确配置了`MINDSPORE_DUMP_CONFIG`环境变量，则会读取配置文件的内容，并按照Dump配置中指定的路径保存算子数据。
   同步模式下，如果要Dump数据，必须采用非数据下沉模式（设置`model.train`或`DatasetHelper`中的`dataset_sink_mode`参数为`False`），以保证可以获取每个step的Dump数据。
   若脚本中都不调用`model.train`或`DatasetHelper`，则默认为非数据下沉模式。使用Dump功能将自动生成最终执行图的IR文件。

### 同步Dump数据对象

同步Dump数据对象目录结构：

```text
{path}/
    |-- {net_name}/
        |-- {device_id}/
            |-- iteration_{iteration}/
                -- {op_name}_{input_output_index}_{shape}_{data_type}_{format}.bin
                …
            |-- graphs/
                ms_output_trace_code_graph_{graph_id}.pb
                ms_output_trace_code_graph_{graph_id}.ir
            |-- execution_order/
                ms_execution_order_graph_{graph_id}.csv

    |-- .metadata/
        data_dump.json
```

- `path`：`data_dump.json`文件中设置的绝对路径。
- `net_name`：`data_dump.json`文件中设置的网络名称。
- `device_id`：训练的卡号。
- `graph_id`：训练的图标号。
- `iteration`：训练的轮次。
- `operator_name`：算子名称。
- `input_output_index` ：输入或输出标号，例如`output_0`表示该文件是该算子的第1个输出Tensor的数据。
- `shape`: 张量维度信息。
- `data_type`: 数据类型。
- `format`: 数据格式。

Dump功能保存数据的对象就是最终执行图以及图中算子的输入和输出数据。

#### 同步Dump数据格式介绍

同步Dump生成的数据文件是后缀名为`.bin`的二进制文件，文件命名格式为：

```text
{operator_name}_{input_output_index}_{shape}_{data_type}_{format}.bin
```

根据文件名提供的`Tensor`信息，可以用`numpy.fromfile`读取数据，并还原原始数据的`data_type`和`shape`。

同步Dump生成的最终执行图文件后缀名分别为`.pb`和`.ir`，文件命名格式为：

```text
ms_output_trace_code_graph_{graph_id}.pb
ms_output_trace_code_graph_{graph_id}.ir
```

其中以`.ir`为后缀的文件可以通过`vi`命令打开查看

同步Dump生成的节点执行序文件后缀名为`.csv`，文件命名格式为：

```text
ms_execution_order_graph_{graph_id}.csv
```

`.metadata`记录了训练的原信息，其中`data_dump.json`保存了用户设置的dump配置。

#### 同步Dump数据分析样例

以[alexnet脚本为例](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/alexnet/src/alexnet.py) ，如果用户想查看脚本中第58行的代码：

```python
x = self.conv3(x)
```

可以从最终执行图中查找到该行代码所对应的多个算子，如下所示：

```text
  %24(equivoutput) = Conv2D(%23, %21) {instance name: conv2d} primitive_attrs: {compile_info: , pri_format: NC1HWC0, stride: (1, 1, 1, 1), pad: (0, 0, 0, 0), pad_mod: same, out_channel:
192, mode: 1, dilation: (1, 1, 1, 1), output_names: [output], group: 1, format: NCHW, offset_a: 0, kernel_size: (3, 3), groups: 1, input_names: [x, w], pad_list: (1, 1, 1, 1),
IsFeatureMapOutput: true, IsFeatureMapInputList: (0)}
       : (<Tensor[Float32]x[const vector][32, 128, 13, 13]>, <Tensor[Float16]x[const vector][192, 128, 3, 3]>) -> (<Tensor[Float16]x[const vector][32, 192, 13, 13]>)
       : (<Float16xNC1HWC0[const vector][32, 8, 13, 13, 16]>, <Float16xFracZ[const vector][72, 12, 16, 16]>) -> (<Float16xNC1HWC0[const vector][32, 12, 13, 13, 16]>)
       : (Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op107)
       ...
       # In file {Absolute path of model_zoo}/official/cv/alexnet/src/alexnet.py(58)/        x = self.conv3(x)/
       ...
  %25(equivoutput) = BiasAdd(%24, %22) {instance name: bias_add} primitive_attrs: {output_used_num: (1), input_names: [x, b], format: NCHW, compile_info: , output_names: [output],
IsFeatureMapOutput: true, IsFeatureMapInputList: (0), pri_format: NC1HWC0}
       : (<Tensor[Float16]x[const vector][32, 192, 13, 13]>) -> (<Tensor[Float16]x[const vector][192]>) -> (<Tensor[Float16]x[const vector][32, 192, 13, 13]>)
       : (<Float16xNC1HWC0[const vector][32, 12, 13, 13, 16]>) -> (<Float16xDefaultFormat[const vector][192]>) -> (<Float16xNC1HWC0[const vector][32, 12, 13, 13, 16]>)
       : (Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/BiasAdd-op105)
       ...
       # In file {Absolute path of model_zoo}/official/cv/alexnet/src/alexnet.py(58)/        x = self.conv3(x)/
       ...
```

最终执行图中各行所表示的含义如下：

- 算子在Host侧（第一行）和Device侧（第二行，有些算子可能不存在）的输入输出情况。从执行图可知，该算子有两个输入（箭头左侧），一个输出（箭头右侧）。

```text
       : (<Tensor[Float32]x[const vector][32, 128, 13, 13]>, <Tensor[Float16]x[const vector][192, 128, 3, 3]>) -> (<Tensor[Float16]x[const vector][32, 192, 13, 13]>)
       : (<Float16xNC1HWC0[const vector][32, 8, 13, 13, 16]>, <Float16xFracZ[const vector][72, 12, 16, 16]>) -> (<Float16xNC1HWC0[const vector][32, 12, 13, 13, 16]>)
```

- 算子名称。从执行图可知，该算子在最终执行图中的完整名称为`Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op107`。

```text
       : (Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op107)
```

- 算子对应的训练脚本代码。通过搜索要查询的训练脚本代码，可以找到多个匹配的算子。

```text
       # In file {Absolute path of model_zoo}/official/cv/alexnet/src/alexnet.py(58)/        x = self.conv3(x)/
```

通过算子名称和输入输出信息，可以查找到唯一对应的Tensor数据文件。比如，若要查看Conv2D-op107算子的第1个输出数据对应的Dump文件，可获取以下信息：

- `operator_name`：`Default--network-WithLossCell--_backbone-AlexNet--conv3-Conv2d--Conv2D-op107`。基于图中序号2声明的算子名称，将其中的`/`替换为`--`可得。

- `input_output_index` ：`output_0`表示该文件是该算子的第1个输出Tensor的数据。

在Dump目录下搜索到相应的文件名：
`Default--network-WithLossCell--_backbone-AlexNet--conv3-Conv2d--Conv2D-op107_output_0_shape_32_12_13_13_16_Float16_NC1HWC0.bin`。
从文件名中可以得知以下信息：

- `shape`: 张量维度是`32_12_13_13_16`。

- `data_type`: 数据类型为`Float16`。

- `format`: 数据格式为`NC1HWC0`（可通过Dump配置文件修改要保存的数据格式）。

还原数据的时候，首先通过执行：

```python
import numpy
numpy.fromfile("Default--network-WithLossCell--_backbone-AlexNet--conv3-Conv2d--Conv2D-op107_output_0_shape_32_12_13_13_16_Float16_NC1HWC0.bin", numpy.float16)
```

生成一维array数据，再通过执行：

```python
import numpy
numpy.reshape(array, (32,12,13,13,16))
```

还原到原始shape数据。

### 异步Dump

大型网络（如Bert Large）使用同步Dump时会导致内存溢出，MindSpore通过异步Dump提供了大型网络的调试能力。

异步Dump是专门针对Ascend整图下沉而开发的功能，可以一边执行算子一边dump数据，一个算子执行结束后立即dump数据，因此开启内存复用也可以生成正确的数据，但是相应的网络训练的速度会较慢。

#### 异步Dump操作步骤

- 异步Dump的操作步骤与同步Dump的操作步骤一致，只有配置文件的内容和数据解析方式不同。 异步Dump的配置文件内容规范详见[异步Dump功能介绍](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/custom_debugging_info.html#id4)

- 若需要dump全量或部分算子，则可以详见参考配置文件中的[dump_mode](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/custom_debugging_info.html#id5) 介绍。

- 若开启数据下沉功能（设置`model.train`或`DatasetHelper`中的`dataset_sink_mode`参数为`True`），只能dump出配置文件里指定的一个step的数据（此时`iteration 0`表示第0个step），并保存到指定目录下。

- 若不开启数据下沉功能（设置`model.train`或`DatasetHelper`中的`dataset_sink_mode`参数为`False`），配置文档里`iteration`必须指定为0，所有step的数据都保存在一个目录中，无法支持多step的数据管理。此时建议只执行一次step的数据Dump（可以通过修改脚本只训练一个step）。

- 使用Dump功能将自动生成最终执行图的IR文件。

### 异步Dump数据对象

同步Dump数据对象目录结构：

```text
{path}/
    |-- {device_id}/
        |-- {new_name}_graph_{graph_id}/
            |-- {graph_id}/
                |-- {iteration}/
                    |-- {op_type}.{op_name}.{task_id}.{timestamp}
                    …
        |-- graphs/
            ms_output_trace_code_graph_{graph_id}.pb
            ms_output_trace_code_graph_{graph_id}.ir
        |-- execution_order/
            ms_execution_order_graph_{graph_id}.csv

    |-- .metadata/
        data_dump.json
```

- `path`：`data_dump.json`文件中设置的绝对路径。
- `net_name`：`data_dump.json`文件中设置的网络名称。
- `device_id`：训练的卡号。
- `graph_id`：训练的图标号。
- `iteration`：训练的轮次。
- `op_type`：算子类型。
- `op_name`：算子名称。
- `taskid`：任务标号。
- `timestamp`：时间戳。

Dump功能保存数据的对象就是最终执行图以及图中算子的输入和输出数据。

#### 异步Dump数据格式介绍

异步Dump生成的原始数据文件是protobuf格式的文件，需要用到海思Run包中自带的数据解析工具进行解析，详见[使用文档](https://support.huaweicloud.com/tg-Inference-cann/atlasaccuracy_16_0014.html) 。

数据在Device侧的格式可能和Host侧计算图中的定义不同，异步Dump的数据格式为Device侧格式，如果想要转为Host侧格式，可以参考[使用文档](https://support.huaweicloud.com/tg-Inference-cann/atlasaccuracy_16_0013.html) 。

异步Dump生成的数据文件命名规则如下：

```text
{op_type}.{op_name}.{taskid}.{timestamp}
```

如果`op_type`和`op_name`中出现了“.”、“/”、“\”、空格时，会转换为下划线表示。

异步Dump生成的最终执行图文件和节点执行序文件命名规则与同步Dump相同

#### 异步Dump数据格式样例

通过异步Dump的功能，获取到算子的异步Dump的文件，如：

```text
dump_file: "BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491"
```

执行：

```text
python3.7.5 msaccucmp.pyc convert -d BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491 -out ./output -f NCHW -t npy
```

则可以在`./output`下生成该算子的所有输入输出数据。每个数据以`.npy`后缀的文件保存，数据格式为`NCHW`。
在生成结果如下：

```text
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.input.0.30x1024x17x17.npy
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.input.1.1x1024x1x1.npy
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.input.2.1x1024x1x1.npy
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.input.3.1x1024x1x1.npy
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.input.4.1x1024x1x1.npy
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.input.5.1x1024x1x1.npy
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.input.6.1x1024x1x1.npy
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.output.0.30x1024x17x17.npy
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.output.1.1x1024x1x1.npy
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.output.2.1x1024x1x1.npy
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.output.3.1x1024x1x1.npy
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.output.4.1x1024x1x1.npy
```

在文件名的末尾可以看到该文件是算子的第几个输入或输出，以及数据的维度信息。例如，通过第一个文件名

```text
BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.input.0.30x1024x17x17.npy
```

可知该文件是算子的第0个输入，数据的维度信息是`30x1024x17x17`。

通过`numpy.load("file_name")`可以读取到对应数据。例：

```python
import numpy
numpy.load("BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.input.0.30x1024x17x17.npy")
```