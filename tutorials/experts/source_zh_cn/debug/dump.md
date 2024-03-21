# Dump功能调试

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/debug/dump.md)

## 概述

为了对训练过程进行分析，用户需要感知训练过程中算子的输入和输出数据。

- 对于静态图模式，MindSpore提供了Dump功能，用来将模型训练中的图以及算子的输入输出数据保存到磁盘文件。

- 对于动态图模式，Dump功能仅支持Ascend后端的溢出检测能力。要想查看非溢出节点，可以使用Python原生执行能力，用户可以在网络脚本运行过程中查看记录相应的输入输出。

### 调试过程

使用Dump来帮助调试分为两个步骤：1、数据准备；2、数据分析。

#### 数据准备

数据准备阶段使用同步Dump或异步Dump来生成Dump数据。使用方法详见[同步Dump操作步骤](#同步dump操作步骤)和[异步Dump操作步骤](#异步dump操作步骤)。

在准备数据时，您可以参考以下最佳实践：

1. 设置`iteration`参数，仅保存出现问题的迭代和前一个迭代这两个迭代的数据。例如，要分析的问题会在第10个迭代（从1开始数）出现，则可以这样设置：`"iteration": "8|9"`。请注意`iteration`参数从0开始计算迭代数。保存上述两个迭代的数据能够支撑大多数场景的问题分析。
2. 在出现问题的迭代执行完毕后，建议您通过[run_context.request_stop()](https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.RunContext.html#mindspore.train.RunContext.request_stop)等方法提前结束训练。

#### 数据分析

如果用户已经安装了MindSpore Insight, 可以使用MindSpore Insight的离线调试器来分析，目前仅支持分析同步dump保存的数据。离线调试器的使用方法详见[使用离线调试器](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/debugger_offline.html) 。

如果没有安装MindSpore Insight，需要通过以下步骤来分析数据。

1. 从脚本找到对应的算子

    使用Dump功能将自动生成最终执行图的IR文件（IR文件中包含了算子全名，和算子在计算图中输入和输出的依赖，也包含从算子到相应脚本代码的Trace信息），IR文件可以用`vi`命令查看，Dump功能的配置见[同步Dump操作步骤](#同步dump操作步骤)和[异步Dump操作步骤](#异步dump操作步骤)，Dump输出的目录结构见[同步Dump数据对象目录](#同步dump数据对象目录)和[异步Dump数据对象目录](#异步dump数据对象目录)。然后通过图文件找到脚本中代码对应的算子，参考[同步Dump数据分析样例](#同步dump数据分析样例)和[异步Dump数据分析样例](#异步dump数据分析样例)。

2. 从算子到Dump数据

    在了解脚本和算子的映射关系后，可以确定想要分析的算子名称，从而找到算子对应的dump文件，参考[同步Dump数据对象目录](#同步dump数据对象目录)和[异步Dump数据对象目录](#异步dump数据对象目录)。

3. 分析Dump数据

    通过解析Dump数据，可以与其他第三方框架进行对比。同步Dump数据格式参考[同步Dump数据文件介绍](#同步dump数据文件介绍)，异步Dump数据格式参考[异步Dump数据文件介绍](#异步dump数据文件介绍)。

### 适用场景

1. 静态图算子结果分析。

   通过Dump功能获得的IR图（仅同步dump支持保存IR图），可以了解脚本代码与执行算子的映射关系（详情见[MindSpore IR简介](https://www.mindspore.cn/docs/zh-CN/master/design/all_scenarios.html#简介)）。结合执行算子的输入和输出数据，可以分析训练过程中可能存在的溢出、梯度爆炸与消失等问题，反向跟踪到脚本中可能存在问题的代码。

2. 特征图分析。

   通过获取图层的输出数据，分析特征图的信息。

3. 模型迁移。

   在将模型从第三方框架（TensorFlow、PyTorch）迁移到MindSpore的场景中，通过比对相同位置算子的输出数据，分析第三方框架和MindSpore对于同一模型的训练结果是否足够接近，来定位模型的精度问题。

## Dump功能说明

MindSpore提供了同步Dump与异步Dump两种模式：

- 同步Dump的机制是在网络训练过程中每个step执行结束后， Host侧发起Dump动作，从Device上拷贝算子地址里面的数据到Host，并保存文件。同步Dump会默认关闭算子间的内存复用，避免读到脏数据。
- 异步Dump是专门针对Ascend整图下沉而开发的功能，可以一边执行算子一边dump数据，一个算子执行结束后立即dump数据，因此开启内存复用也可以生成正确的数据，但是相应的网络训练的速度会较慢。

不同模式所需要的配置文件和dump出来的数据格式不同：

- 异步Dump全量功能只支持Ascend上的图模式，异步Dump溢出检测功能只支持Ascend上的图模式和PyNative模式。开启异步Dump的时候不会关闭内存复用。
- Ascend仅支持异步Dump模式，GPU仅支持同步Dump模式。
- Dump暂不支持异构训练，如果在异构训练场景启用Dump，生成的Dump数据对象目录可能不符合预期的目录结构。

## 同步Dump

### 同步Dump操作步骤

1. 创建json格式的配置文件，JSON文件的名称和位置可以自定义设置。

    ```json
    {
        "common_dump_settings": {
            "dump_mode": 0,
            "path": "/absolute_path",
            "net_name": "ResNet50",
            "iteration": "0|5-8|100-120",
            "saved_data": "tensor",
            "input_output": 0,
            "kernels": ["Default/Conv-op12"],
            "support_device": [0,1,2,3,4,5,6,7]
        },
        "e2e_dump_settings": {
            "enable": true,
            "trans_flag": true
        }
    }
    ```

    - `dump_mode`：设置成0，表示Dump出该网络中的所有算子数据；设置成1，表示Dump`"kernels"`里面指定的算子数据或算子类型数据。
    - `path`：Dump保存数据的绝对路径。
    - `net_name`：自定义的网络名称，例如："ResNet50"。
    - `iteration`：指定需要Dump数据的迭代。类型为str，用“|”分离要保存的不同区间的step的数据。如"0|5-8|100-120"表示Dump第1个，第6个到第9个， 第101个到第121个step的数据。指定“all”，表示Dump所有迭代的数据。
    - `saved_data`: 指定Dump的数据。类型为str，取值成"tensor"，表示Dump出完整张量数据；取值成"statistic"，表示只Dump张量的统计信息；取值"full"代表两种都要。同步Dump统计信息现只支持GPU场景，CPU或Ascend场景若选"statistic"或"full"便会错误退出。默认取值为"tensor"。
    - `input_output`：设置成0，表示Dump出算子的输入和算子的输出；设置成1，表示Dump出算子的输入；设置成2，表示Dump出算子的输出。
    - `kernels`：该项可以配置两种格式：
        1. 算子的名称列表。开启IR保存开关`set_context(save_graphs=2)`并执行用例，从生成的IR文件`trace_code_graph_{graph_id}`中获取算子名称。详细说明可以参照教程：[如何保存IR](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/error_analysis/mindir.html#如何保存ir)。
        需要注意的是，是否设置`set_context(save_graphs=2)`可能会导致同一个算子的id不同，所以在Dump指定算子时要在获取算子名称之后保持这一项设置不变。或者也可以在Dump保存的`ms_output_trace_code_graph_{graph_id}.ir`文件中获取算子名称，参考[同步Dump数据对象目录](#同步dump数据对象目录)。
        2. 还可以指定算子类型。当字符串中不带算子scope信息和算子id信息时，后台则认为其为算子类型，例如："conv"。算子类型的匹配规则为：当发现算子名中包含算子类型字符串时，则认为匹配成功（不区分大小写），例如："conv" 可以匹配算子 "Conv2D-op1234"、"Conv3D-op1221"。
    - `support_device`：支持的设备，默认设置成0到7即可；在分布式训练场景下，需要dump个别设备上的数据，可以只在`support_device`中指定需要Dump的设备Id。该配置参数在CPU上无效，因为CPU下没有device这个概念，但是在json格式的配置文件中仍需保留该字段。
    - `enable`：设置成true，表示开启同步Dump；设置成false时，在Ascend上会使用异步Dump，在GPU上仍然使用同步Dump。
    - `trans_flag`：开启格式转换。将设备上的数据格式转换成NCHW格式。若为`True`，则数据会以Host侧的4D格式（NCHW）格式保存；若为`False`，则保留Device侧的数据格式。该配置参数在CPU上无效，因为CPU上没有format转换，但是在json格式的配置文件中仍需保留该字段。

2. 设置Dump环境变量。

   指定Dump的json配置文件。

   ```bash
   export MINDSPORE_DUMP_CONFIG=${xxx}
   ```

   其中"xxx"为配置文件的绝对路径，如：

   ```bash
   export MINDSPORE_DUMP_CONFIG=/path/to/data_dump.json
   ```

   如果Dump配置文件没有设置`path`字段或者设置为空字符串，还需要配置环境变量`MS_DIAGNOSTIC_DATA_PATH`。

   ```bash
   export MS_DIAGNOSTIC_DATA_PATH=${yyy}
   ```

   则“$MS_DIAGNOSTIC_DATA_PATH/debug_dump”就会被当做`path`的值。若Dump配置文件中设置了`path`字段，则仍以该字段的实际取值为准。

    注意：

    - 在网络脚本执行前，设置好环境变量；网络脚本执行过程中设置将会不生效。
    - 在分布式场景下，Dump环境变量需要在调用`mindspore.communication.init`之前配置。

3. 启动网络训练脚本。

   训练启动后，若正确配置了`MINDSPORE_DUMP_CONFIG`环境变量，则会读取配置文件的内容，并按照Dump配置中指定的数据保存路径保存算子数据。
   同步模式下，GPU环境如果要Dump数据，必须采用非数据下沉模式（设置`model.train`或`DatasetHelper`中的`dataset_sink_mode`参数为`False`），以保证可以获取每个step的Dump数据。
   若脚本中都不调用`model.train`或`DatasetHelper`，则默认为非数据下沉模式。使用Dump功能将自动生成最终执行图的IR文件。

    可以在训练脚本中设置`set_context(reserve_class_name_in_scope=False)`，避免Dump文件名称过长导致Dump数据文件生成失败。

4. 通过`numpy.load`读取和解析同步Dump数据，参考[同步Dump数据文件介绍](#同步dump数据文件介绍)。

### 同步Dump数据对象目录

启动训练后，同步Dump保存的数据对象包括最终执行图（`ms_output_trace_code_graph_{graph_id}.ir`文件）以及图中算子的输入和输出数据，数据目录结构如下所示：

```text
{path}/
    - rank_{rank_id}/
        - .dump_metadata/
        - {net_name}/
            - {graph_id}/
                - {iteration_id}/
                    statistic.csv
                    {op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy
                - constants/
                    Parameter.data-{data_id}.0.0.{timestamp}.output.0.DefaultFormat.npy
            ...
        - graphs/
            ms_output_trace_code_graph_{graph_id}.pb
            ms_output_trace_code_graph_{graph_id}.ir
        - execution_order/
            ms_execution_order_graph_{graph_id}.csv
            ms_global_execution_order_graph_{graph_id}.csv
```

- `path`：`data_dump.json`配置文件中设置的绝对路径。
- `rank_id`： 逻辑卡号。
- `net_name`：`data_dump.json`配置文件中设置的网络名称。
- `graph_id`：训练的图标号。
- `iteration_id`：训练的轮次。
- `op_type`：算子类型。
- `op_name`：算子名称。
- `task_id`：任务标号。
- `stream_id`：流标号。
- `timestamp`：时间戳。
- `input_output_index`：输入或输出标号，例如`output.0`表示该文件是该算子的第1个输出Tensor的数据。
- `slot`：slot标号。
- `format`: 数据格式。
- `data_id`: 常量数据标号。

对于多图网络，由于存在控制流，某些子图可能不会被执行，Dump只保存执行过的节点，所以graphs目录下`.pb`文件名中的{graph_id}并不一定在{net_name}下存在对应的{graph_id}目录。

只当`saved_data`为"statistic"或者"full"时，才会生成`statistic.csv`，当`saved_data`为"tensor"或者"full"时，才会生成`{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy`命名的完整张量信息。

### 同步Dump数据文件介绍

同步Dump生成的数据文件是后缀名为`.npy`的文件，文件命名格式为：

```text
{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy
```

同步Dump生成的常量数据文件与其他数据文件格式相同，而所有常量数据的{op_type}，{task_id}，{stream_id}，{input_output_index}，{slot}，{format}不变。注意，非Tensor类型数据不会被生成数据文件。

```text
Parameter.data-{data_id}.0.0.{timestamp}.output.0.DefaultFormat.npy
```

可以用Numpy的`numpy.load`接口读取数据。

同步Dump生成的统计数据文件名为`statistic.csv`，此文件存有相同目录下所有落盘张量（文件名为`{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy`）的统计信息。每个张量一行，每行有张量的 Op Type，Op Name，Task ID，Stream ID，Timestamp，IO，Slot，Data Size，Data Type，Shape，Max Value，Min Value，Avg Value，Count，Negative Zero Count，Positive Zero Count，NaN Count，Negative Inf Count，Positive Inf Count，Zero Count，MD5。注意，如果用Excel来打开此文件，数据可能无法正确显示。请用`vi`、`cat`等命令查看，或者使用Excel自文本导入csv查看。

同步Dump生成的最终执行图文件后缀名分别为`.pb`和`.ir`，文件命名格式为：

```text
ms_output_trace_code_graph_{graph_id}.pb
ms_output_trace_code_graph_{graph_id}.ir
```

其中以`.ir`为后缀的文件可以通过`vi`命令打开查看。

同步Dump生成的节点执行序文件后缀名为`.csv`，文件命名格式为：

```text
ms_execution_order_graph_{graph_id}.csv
```

图执行历史文件的后缀为`.csv`，文件名格式为：

```text
ms_global_execution_order_graph_{graph_id}.csv
```

此文件记录该图在训练过程中的执行轮次历史。图编译过程中，一张根图可能产生多张子图，但子图与根图具有相同的执行轮次历史。故与图执行序文件不同，此处仅保存根图的图执行历史文件。

`.dump_metadata`记录了训练的原信息，其中`data_dump.json`保存了用户设置的dump配置。

### 同步Dump数据分析样例

为了更好地展示使用Dump来保存数据并分析数据的流程，我们提供了一套[完整样例脚本](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/dump) ，同步Dump只需要执行 `bash run_sync_dump.sh`。

在通过Dump功能将脚本对应的图保存到磁盘上后，会产生最终执行图文件`ms_output_trace_code_graph_{graph_id}.ir`。该文件中保存了对应的图中每个算子的堆栈信息，记录了算子对应的生成脚本。

以[AlexNet脚本](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/dump/train_alexnet.py)为例 ：

```python
...
def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid"):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode=pad_mode)


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    return TruncatedNormal(0.02)


class AlexNet(nn.Cell):
    """
    Alexnet
    """

    def __init__(self, num_classes=10, channel=3):
        super(AlexNet, self).__init__()
        self.conv1 = conv(channel, 96, 11, stride=4)
        self.conv2 = conv(96, 256, 5, pad_mode="same")
        self.conv3 = conv(256, 384, 3, pad_mode="same")
        self.conv4 = conv(384, 384, 3, pad_mode="same")
        self.conv5 = conv(384, 256, 3, pad_mode="same")
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = fc_with_initialize(6 * 6 * 256, 4096)
        self.fc2 = fc_with_initialize(4096, 4096)
        self.fc3 = fc_with_initialize(4096, num_classes)

    def construct(self, x):
        """
        The construct function.

        Args:
           x(int): Input of the network.

        Returns:
           Tensor, the output of the network.
        """
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
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
...
```

如果用户想查看脚本中第175行的代码：

```python
x = self.conv3(x)
```

执行完训练网络后，可以从最终执行图（`ms_output_trace_code_graph_{graph_id}.ir`文件）中查找到该行代码所对应的多个算子信息，例如Conv2D-op12对应的文件内容如下所示：

```text
  %20(equivoutput) = Conv2D(%17, %19) {instance name: conv2d} primitive_attrs: {IsFeatureMapInputList: (0), kernel_size: (3, 3), mode: 1, out_channel: 384, input_names: [
x, w],    pri_format: NC1HWC0, pad: (0, 0, 0, 0), visited: true, pad_mod: same, format: NCHW,  pad_list: (1, 1, 1, 1), precision_flag: reduce, groups: 1, output_used_num:
(1), stream_id:     0, stride: (1, 1, 1, 1), group: 1, dilation: (1, 1, 1, 1), output_names: [output], IsFeatureMapOutput: true, ms_function_graph: true}
       : (<Tensor[Float32], (32, 256, 13, 13)>, <Tensor[Float32], (384, 256, 3, 3)>) -> (<Tensor[Float32], (32, 384, 13, 13)>)
       : (<Float16xNC1HWC0[const vector][32, 16, 13, 13, 16]>, <Float16xFracZ[const vector][144, 24, 16, 16]>) -> (<Float32xNC1HWC0[const vector][32, 24, 13, 13, 16]>)
       : full_name_with_scope: (Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op12)
       ...
       # In file ./tain_alexnet.py(175)/        x = self.conv3(x)/
       ...
```

以上所示文件内容的各行所表示的含义如下：

- 算子在Host侧（第一行）和Device侧（第二行，有些算子可能不存在）的输入输出情况。从执行图可知，该算子有两个输入（箭头左侧），一个输出（箭头右侧）。

    ```text
       : (<Tensor[Float32], (32, 256, 13, 13)>, <Tensor[Float32], (384, 256, 3, 3)>) -> (<Tensor[Float32], (32, 384, 13, 13)>)
       : (<Float16xNC1HWC0[const vector][32, 16, 13, 13, 16]>, <Float16xFracZ[const vector][144, 24, 16, 16]>) -> (<Float32xNC1HWC0[const vector][32, 24, 13, 13, 16]>)
    ```

- 算子名称。从执行图可知，该算子在最终执行图中的完整名称为`Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op12`。

    ```text
    : (Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op12)
    ```

- 算子对应的训练脚本代码。通过搜索要查询的训练脚本代码，可以找到多个匹配的算子。

    ```text
    # In file {Absolute path of model_zoo}/official/cv/alexnet/src/alexnet.py(175)/        x = self.conv3(x)/
    ```

通过算子名称和输入输出信息，可以查找到唯一对应的Tensor数据文件。比如，若要查看Conv2D-op12算子的第1个输出数据对应的Dump文件，可获取以下信息：

- `operator_name`：`Conv2D-op12`。

- `input_output_index`：`output.0`表示该文件是该算子的第1个输出Tensor的数据。

- `slot`：0，该算子的输出只有一个slot。

在Dump保存的数据对象文件目录下搜索到相应的文件名：
`Conv2D.Conv2D-op12.0.0.1623124369613540.output.0.DefaultFormat.npy`。

还原数据的时候，通过执行：

```python
import numpy
numpy.load("Conv2D.Conv2D-op12.0.0.1623124369613540.output.0.DefaultFormat.npy")
```

生成numpy.array数据。

## 异步Dump

MindSpore通过异步Dump提供了Ascend平台上大型网络的调试能力。

### 异步Dump操作步骤

1. 创建配置文件`data_dump.json`。

    JSON文件的名称和位置可以自定义设置。

    ```json
    {
        "common_dump_settings": {
            "dump_mode": 0,
            "path": "/absolute_path",
            "net_name": "ResNet50",
            "iteration": "0|5-8|100-120",
            "saved_data": "tensor",
            "input_output": 0,
            "kernels": ["Default/Conv-op12"],
            "support_device": [0,1,2,3,4,5,6,7],
            "op_debug_mode": 0,
            "file_format": "npy"
        }
    }
    ```

    - `dump_mode`：设置成0，表示Dump出该网络中的所有算子数据；设置成1，表示Dump`"kernels"`里面指定的算子数据或算子类型数据；设置成2，表示Dump脚本中通过`set_dump`指定的算子数据，`set_dump`的使用详见[mindspore.set_dump](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_dump.html) 。开启溢出检测时，此字段的设置失效，Dump只会保存溢出节点的数据。
    - `path`：Dump保存数据的绝对路径。在[jit_level](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.JitConfig.html?highlight=jit_level)设置为‘O0’时，MindSpore会在path目录下新建每个step的子目录。
    - `net_name`：自定义的网络名称，例如："ResNet50"。
    - `iteration`：指定需要Dump的迭代。类型为str，用“|”分离要保存的不同区间的step的数据。如"0|5-8|100-120"表示Dump第1个，第6个到第9个， 第101个到第121个step的数据。指定“all”，表示Dump所有迭代的数据。PyNative模式开启溢出检测时，必须设置为"all"。
    - `saved_data`: 指定Dump的数据。类型为str，取值成"tensor"，表示Dump出完整张量数据；取值成"statistic"，表示只Dump张量的统计信息；取值"full"代表两种都要。异步Dump统计信息只有在`file_format`设置为`npy`时可以成功，若在`file_format`设置为`bin`时选"statistic"或"full"便会错误退出。默认取值为"tensor"。
    - `input_output`：设置成0，表示Dump出算子的输入和算子的输出；设置成1，表示Dump出算子的输入；设置成2，表示Dump出算子的输出。
    - `kernels`：算子的名称列表。指定算子需要先设置保存图文件的环境变量来保存图，再从保存的图文件中获取算子名称。保存图文件的环境变量请请参考昇腾社区文档[DUMP_GE_GRAPH](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/apiref/envref/envref_07_0011.html) 、[DUMP_GRAPH_LEVEL](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/apiref/envref/envref_07_0012.html) 和[DUMP_GRAPH_PATH](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/apiref/envref/envref_07_0013.html) 。
    - `support_device`：支持的设备，默认设置成0到7即可；在分布式训练场景下，需要dump个别设备上的数据，可以只在`support_device`中指定需要Dump的设备Id。
    - `op_debug_mode`：该属性用于算子溢出调试，设置成0，表示不开启溢出；设置成1，表示开启AiCore溢出检测；设置成2，表示开启Atomic溢出检测；设置成3，表示开启全部溢出检测功能；设置成4，表示开启轻量异常Dump功能。在Dump数据的时候请设置成0，若设置成其他值，则只会Dump溢出算子或异常算子的数据。
    - `file_format`: dump数据的文件类型，只支持`npy`和`bin`两种取值。设置成`npy`，则dump出的算子张量数据将为host侧格式的npy文件；设置成`bin`，则dump出的数据将为device侧格式的protobuf文件，需要借助转换工具进行处理，详细步骤请参考[异步Dump数据分析样例](#异步dump数据分析样例)。默认取值为`bin`。

2. 设置数据Dump的环境变量。

    ```bash
    export MINDSPORE_DUMP_CONFIG=${Absolute path of data_dump.json}
    ```

   如果Dump配置文件没有设置`path`字段或者设置为空字符串，还需要配置环境变量`MS_DIAGNOSTIC_DATA_PATH`。

   ```bash
   export MS_DIAGNOSTIC_DATA_PATH=${yyy}
   ```

   则“$MS_DIAGNOSTIC_DATA_PATH/debug_dump”就会被当做`path`的值。若Dump配置文件中设置了`path`字段，则仍以该字段的实际取值为准。

    - 在网络脚本执行前，设置好环境变量；网络脚本执行过程中设置将会不生效。
    - 在分布式场景下，Dump环境变量需要在调用`mindspore.communication.init`之前配置。

   可通过配置环境变量使能ACL dump。

   ```bash
   export MS_ACL_DUMP_CFG_PATH=${Absolute path of data_dump.json}
   ```

3. 执行用例Dump数据。

   可以在训练脚本中设置`set_context(reserve_class_name_in_scope=False)`，避免Dump文件名称过长导致Dump数据文件生成失败。

4. 参考[异步Dump数据分析样例](#异步dump数据分析样例)解析Dump数据文件。

注意：

- 若需要dump全量或部分算子，则可以修改json配置文件中的`dump_mode`选项为0或1。

### 异步Dump数据对象目录

图模式的异步Dump目录结构如下所示：

```text
{path}/
    - {time}/
        - {device_id}/
            - {model_name}/
                - {model_id}/
                    - {iteration_id}/
                        statistic.csv
                        {op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}
                        Opdebug.Node_OpDebug.{task_id}.{stream_id}.{timestamp}
                        mapping.csv
```

通过MS_ACL_DUMP_CFG_PATH环境变量使能ACL dump，且[jit_level](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.JitConfig.html?highlight=jit_level)不设置为‘O0’时，Dump目录结构如下所示，主要特征为存在{step_id}目录，代表用户侧的训练轮次：

```text
{path}/
    - {step_id}/
        - {time}/
            - {device_id}/
                - {model_name}/
                    - {model_id}/
                        - {iteration_id}/
                            statistic.csv
                            {op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}
                            Opdebug.Node_OpDebug.{task_id}.{stream_id}.{timestamp}
                            mapping.csv
```

通过MS_ACL_DUMP_CFG_PATH环境变量使能ACL dump，且[jit_level](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.JitConfig.html?highlight=jit_level)设置为‘O0’时，Dump目录结构如下所示，主要特征为不存在{model_name}和{model_id}目录，此种场景下的动态shape算子的Dump数据会保存于{iteration_id}目录，静态shape算子的Dump数据会保存在{device_id}目录：

```text
{path}/
    - {step_id}/
        - {time}/
            - {device_id}/
                - {iteration_id}/
                    statistic.csv
                    {op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}
                    Opdebug.Node_OpDebug.{task_id}.{stream_id}.{timestamp}
                    mapping.csv
```

使能ACL dump时，除上述dump数据外，还会在{path}目录生成调用acl接口所需要的json文件，一般情况下无需关注。

- `path`：`data_dump.json`配置文件中设置的绝对路径。
- `time`： dump目录的创建时间。
- `device_id`: 卡号。
- `model_name`：模型名称，由MindSpore生成。
- `model_id`：模型标号。
- `iteration_id`：GE侧训练的轮次。
- `op_type`：算子类型。
- `op_name`：算子名称。
- `task_id`：任务标号。
- `stream_id`：流标号。
- `timestamp`：时间戳。
- `step_id`: 用户侧的训练轮次。

其中，溢出文件（`Opdebug.Node_OpDebug.{task_id}.{stream_id}.{timestamp}`文件）只会在开启溢出Dump且检测到溢出时保存。

若配置文件中`file_format`值设置为`npy`，算子文件会保存成npy格式的文件，溢出文件会被保存成json格式的文件。文件命名格式分别为：

```text
{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy
Opdebug.Node_OpDebug.{task_id}.{stream_id}.{timestamp}.output.0.json
```

如果按命名规则定义的张量文件名称长度超过了OS文件名称长度限制（一般是255个字符），则会将该张量文件重命名为一串随机数字，映射关系会保存在同目录下的“mapping.csv”。

### 异步Dump数据文件介绍

若配置文件中`file_format`值设置为`npy`，可以直接用`numpy.load`加载。

若未配置`file_format`值或`file_format`值为`bin`，启动训练后，异步Dump生成的原始数据文件或溢出检测生成的溢出文件是protobuf格式的文件，需要用到海思Run包中自带的数据解析工具进行解析，详见[如何查看dump数据文件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/devaids/auxiliarydevtool/atlasaccuracy_16_0059.html)。

数据在Device侧的格式可能和Host侧计算图中的定义不同，异步Dump的bin数据格式为Device侧格式，如果想要转为Host侧格式，可以参考[如何进行dump数据文件Format转换](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/devaids/auxiliarydevtool/atlasaccuracy_16_0057.html)。

异步Dump生成的数据文件是`bin`文件时，文件命名格式为：

```text
{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}
```

以AlexNet网络的Conv2D-op12为例：`Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802`，其中`Conv2D`是`{op_type}`，`Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12`是`{op_name}`，`2`是`{task_id}`，`7`是`{stream_id}`，`161243956333802`是`{timestamp}`。

如果`op_type`和`op_name`中出现了“.”、“/”、“\”、空格时，会转换为下划线表示。

Dump生成的原始数据文件也可以使用MindSpore Insight的数据解析工具DumpParser解析，DumpParser的使用方式详见[DumpParser介绍](https://gitee.com/mindspore/mindinsight/tree/master/mindinsight/parser) 。MindSpore Insight解析出来的数据格式与同步dump的数据格式完全相同。

若配置`file_format`值为`npy`，则启用异步dump生成的数据文件命名规则与同步Dump相同，可以参考[同步Dump数据文件介绍](#同步dump数据文件介绍)，溢出检测生成的溢出文件是`json`格式，溢出文件内容解析可参考[解析算子溢出数据文件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/devguide/appdevg/aclpythondevg/aclpythondevg_0078.html#ZH-CN_TOPIC_0000001781325073__section6864050111619) 。

选项`saved_data`只有在`file_format`为"npy"的时候生效。如`saved_data`是"statistic"或者"full"。张量统计数据会落盘到`statistic.csv`。如`saved_data`是"tensor"或者"full"完整张量数据会落盘到`{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy`。`statistic.csv`的格式与同步Dump相同，可以参考[同步Dump数据文件介绍](#同步dump数据文件介绍)。

### 异步Dump数据分析样例

为了更好地展示使用Dump来保存数据并分析数据的流程，我们提供了一套[完整样例脚本](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/dump) ，异步Dump执行 `bash run_async_dump.sh` 即可。用户可以自行下载体验。

通过异步Dump的功能，获取到算子异步Dump生成的数据文件。如果异步Dump配置文件中设置的`file_format`为"npy"，可以跳过以下步骤中的1、2，如果没有设置`file_format`，或者设置为"bin"，需要先转换成`.npy`格式的文件。

1. 使用run包中提供的`msaccucmp.py`解析Dump出来的文件。不同的环境上`msaccucmp.py`文件所在的路径可能不同，可以通过`find`命令进行查找：

    ```bash
    find ${run_path} -name "msaccucmp.py"
    ```

    - `run_path`：run包的安装路径。

2. 找到`msaccucmp.py`后，到`/absolute_path`目录下，运行如下命令解析Dump数据：

    ```bash
    python ${The absolute path of msaccucmp.py} convert -d {file path of dump} -out {file path of output}
    ```

    {file path of dump} 可以是单个`.bin`文件的路径，也可以是包含`.bin`文件的文件夹路径。

    若需要转换数据格式，可参考使用说明链接<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/devaids/auxiliarydevtool/atlasaccuracy_16_0057.html> 。

    如Dump生成的数据文件为：

    ```text
    Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802
    ```

    则执行：

    ```bash
    python3.7.5 msaccucmp.py convert -d /path/to/Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802 -out ./output -f NCHW -t npy
    ```

    则可以在`./output`下生成该算子的所有输入输出数据。每个数据以`.npy`后缀的文件保存，数据格式为`NCHW`。生成结果如下：

    ```text
    Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802.input.0.32x256x13x13.npy
    Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802.input.1.384x256x3x3.npy
    Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802.output.0.32x384x13x13.npy
    ```

    在文件名的末尾可以看到该文件是算子的第几个输入或输出，以及数据的维度信息。例如，通过第一个`.npy`文件名

    ```text
    Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802.input.0.32x256x13x13.npy
    ```

    可知该文件是算子的第0个输入，数据的维度信息是`32x256x13x13`。

3. 通过`numpy.load("file_name")`可以读取到对应数据。例：

    ```python
    import numpy
    numpy.load("Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802.input.0.32x256x13x13.npy")
    ```

## 注意事项

- `bfloat16`类型的算子保存到`npy`文件时，会转换成`float32`类型。
- Dump仅支持bool、int、int8、in16、int32、int64、uint、uint8、uint16、uint32、uint64、float、float16、float32、float64、bfloat16、double类型数据的保存。
- Print算子内部有一个输入参数为string类型，string类型不属于Dump支持的数据类型，所以在脚本中包含Print算子时，会有错误日志，这不会影响其它类型数据的保存。