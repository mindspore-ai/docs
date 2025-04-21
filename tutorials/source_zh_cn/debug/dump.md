# Dump功能调试

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_zh_cn/debug/dump.md)

为了对训练过程进行分析，MindSpore提供了Dump功能，用于保存训练过程中算子的输入和输出数据。

## 功能演进

MindSpore Dump功能已陆续迁移到[msprobe工具](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe)。

> [msprobe](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe) 是 MindStudio Training Tools 工具链下精度调试部分的工具包。主要包括精度预检、溢出检测和精度比对等功能，目前适配 PyTorch 和 MindSpore 框架。

其中动态图、静态图Ascend O2模式Dump已完全迁移到msprobe工具，通过msprobe工具入口使能，详情请查看[《msprobe 工具 MindSpore场景精度数据采集指南》](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md)。

静态图Ascend OO/O1和CPU/GPU模式仍然通过框架入口使能，后续会陆续迁移到msprobe工具。

## 配置指南

MindSpore在不同模式下支持的Dump功能不完全相同，需要的配置文件和以及生成的数据格式也不同，因此需要根据运行的模式选择对应的Dump配置：

- [Ascend下O0/O1模式Dump](#ascend下o0o1模式dump)
- [Ascend下O2模式Dump](#ascend下o2模式dump)
- [CPU/GPU模式Dump](#cpugpu模式dump)

> - Ascend下O0/O1/O2模式的区别请见[set_context的参数jit_level](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mindspore/mindspore.set_context.html)。
>
> - CPU/GPU模式支持dump常量数据，Ascend O0/O1/O2模式不支持Dump常量数据。
>
> - Dump暂不支持异构训练，即不支持CPU/Ascend混合训练或GPU/Ascend混合训练。

MindSpore在不同模式下支持的Dump功能如下表所示：

<table align="center">
  <tr>
   <td colspan="2" align="center">功能</td>
   <td align="center">Ascend O0/O1</td>
   <td align="center">CPU/GPU</td>
  </tr>
  <tr>
   <td align="left">全量dump</td>
   <td align="left">整网数据dump</td>
   <td align="left">支持</td>
   <td align="left">支持</td>
  </tr>
  <tr>
   <td rowspan="2" align="left">部分数据dump</td>
   <td align="left">统计信息dump</td>
   <td align="left">支持host和device模式<sup>1</sup></td>
   <td align="left">CPU不支持，GPU仅支持host模式</td>
  </tr>
  <tr>
   <td align="left">数据采样dump</td>
   <td align="left">支持</td>
   <td align="left">不支持</td>
  </tr>
  <tr>
   <td align="left">溢出dump</td>
   <td align="left">dump溢出算子</td>
   <td align="left">支持</td>
   <td align="left">不支持</td>
  </tr>
  <tr>
   <td rowspan="5" align="left">指定条件dump</td>
   <td align="left">指定算子名称</td>
   <td align="left">支持</td>
   <td align="left">支持</td>
  </tr>
  <tr>
   <td align="left">指定迭代</td>
   <td align="left">支持</td>
   <td align="left">支持</td>
  </tr>
  <tr>
   <td align="left">指定device</td>
   <td align="left">支持</td>
   <td align="left">支持</td>
  </tr>
  <tr>
   <td align="left">指定file_format</td>
   <td align="left">不涉及</td>
   <td align="left">不涉及</td>
  </tr>
  <tr>
   <td align="left">set_dump</td>
   <td align="left">支持</td>
   <td align="left">支持</td>
  </tr>
  <tr>
   <td rowspan="2" align="left">辅助信息dump</td>
   <td align="left">图ir dump</td>
   <td align="left">支持</td>
   <td align="left">支持</td>
  </tr>
  <tr>
   <td align="left">执行序dump</td>
   <td align="left">支持</td>
   <td align="left">支持</td>
  </tr>
</table>

> 在统计信息方面，device计算速度较host快（目前仅支持Ascend后端），但host统计指标比device多，详见`statistic_category`选项。

## Ascend下O0/O1模式Dump

### 操作步骤

1. 创建json格式的配置文件，JSON文件的名称和位置可以自定义设置。

    ```json
    {
        "common_dump_settings": {
            "op_debug_mode": 0,
            "dump_mode": 0,
            "path": "/absolute_path",
            "net_name": "ResNet50",
            "iteration": "0|5-8|100-120",
            "saved_data": "tensor",
            "input_output": 0,
            "kernels": ["Default/Conv-op12"],
            "support_device": [0,1,2,3,4,5,6,7],
            "statistic_category": ["max", "min", "l2norm"]
        },
        "e2e_dump_settings": {
            "enable": true,
            "trans_flag": true,
            "stat_calc_mode": "host"
        }
    }
    ```

    - `common_dump_settings`:

        - `op_debug_mode`：该属性用于算子溢出或算子异常调试，设置成0，表示保存所有算子或指定算子；设置成3，表示只保存溢出算子；设置成4，表示只保存异常算子的输入。在Dump数据的时候请设置成0，若设置成其他值，则只会Dump溢出算子或异常算子的数据。默认值：0。
        - `dump_mode`：设置成0，表示Dump出该网络中的所有算子数据；设置成1，表示Dump`"kernels"`里面指定的算子数据或算子类型数据；设置成2，表示使用[mindspore.set_dump](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mindspore/mindspore.set_dump.html) Dump指定对象。仅在op_debug_mode设置为0时支持指定算子dump。
        - `path`：Dump保存数据的绝对路径。
        - `net_name`：自定义的网络名称，例如："ResNet50"。
        - `iteration`：指定需要Dump数据的迭代。类型为str，用“|”分离要保存的不同区间的step的数据。如"0|5-8|100-120"表示Dump第1个，第6个到第9个，第101个到第121个step的数据。指定“all”，表示Dump所有迭代的数据。仅在op_debug_mode设置为0或3时支持保存指定迭代，op_debug_mode设置为4时不支持指定迭代。
        - `saved_data`: 指定Dump的数据。类型为str，取值成"tensor"，表示Dump出完整张量数据；取值成"statistic"，表示只Dump张量的统计信息；取值"full"代表两种都要。默认取值为"tensor"。保存统计信息仅在op_debug_mode设置为0时生效。
        - `input_output`：设置成0，表示Dump出算子的输入和算子的输出；设置成1，表示Dump出算子的输入；设置成2，表示Dump出算子的输出。在op_debug_mode设置为3时，只能设置`input_output`为同时保存算子输入和算子输出。在op_debug_mode设置为4时，只能保存算子输入。
        - `kernels`：该项可以配置三种格式：
           1. 算子的名称列表。通过设置环境变量`MS_DEV_SAVE_GRAPHS`的值为2开启IR保存开关并执行用例，从生成的IR文件`trace_code_graph_{graph_id}`中获取算子名称。详细说明可以参照教程：[如何保存IR](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/debug/error_analysis/mindir.html#如何保存ir)。
           需要注意的是，是否设置环境变量`MS_DEV_SAVE_GRAPHS`的值为2可能会导致同一个算子的id不同，所以在Dump指定算子时要在获取算子名称之后保持这一项设置不变。或者也可以在Dump保存的`ms_output_trace_code_graph_{graph_id}.ir`文件中获取算子名称，参考[Ascend O0/O1模式下Dump数据对象目录](#数据对象目录和数据文件介绍)。
           2. 还可以指定算子类型。当字符串中不带算子scope信息和算子id信息时，后台则认为其为算子类型，例如："conv"。算子类型的匹配规则为：当发现算子名中包含算子类型字符串时，则认为匹配成功（不区分大小写），例如："conv" 可以匹配算子 "Conv2D-op1234"、"Conv3D-op1221"。
           3. 算子名称的正则表达式。当字符串符合"name-regex(xxx)"格式时，后台则会将其作为正则表达式。例如，"name-regex(Default/.+)"可匹配算子名称以"Default/"开头的所有算子。
        - `support_device`：支持的设备，默认设置成0到7即可；在分布式训练场景下，需要dump个别设备上的数据，可以只在`support_device`中指定需要Dump的设备Id。该配置参数在CPU上无效，因为CPU下没有device这个概念，但是在json格式的配置文件中仍需保留该字段。
        - `statistic_category`: 该属性用于用户配置要保存的统计信息类别，仅在开启了保存统计信息(即`saved_data`设置为"statistic"或"full")时生效。类型为字符串列表，其中的字符串可选值如下：

            - "max": 表示Tensor中元素的最大值，支持在device统计和在host统计；
            - "min": 表示Tensor中元素的最小值，支持在device统计和在host统计；
            - "avg": 表示Tensor中元素的平均值，支持在device统计和在host统计；
            - "count": 表示Tensor中元素的个数；
            - "negative zero count": 表示Tensor中小于0的元素个数；
            - "positive zero count": 表示Tensor中大于0的元素个数；
            - "nan count": 表示Tensor中元素的`Nan`的个数；
            - "negative inf count": 表示Tensor中`-Inf`元素的个数；
            - "positive inf count": 表示Tensor中`+Inf`元素的个数；
            - "zero count": 表示Tensor中元素`0`的个数；
            - "md5": 表示Tensor的MD5值；
            - "l2norm": 表示Tensor的L2Norm值，支持在device统计和在host统计。

            以上除了标记了支持device统计的，其他都仅支持在host统计。
            该字段为可选，默认值为["max", "min", "l2norm"]。

        - `overflow_number`：指定溢出dump的数据个数。该字段仅在`op_debug_mode`设置为3，只保存溢出算子时需要配置，可控制溢出数据按时间序dump，到指定数值后溢出数据不再dump。默认值为0，表示dump全部溢出数据。
        - `initial_iteration`：指定Dump的初始迭代数，需为非负整数。若设置为10，则Dump初始落盘的iteration将从10开始计数。默认值：0。

    - `e2e_dump_settings`:

        - `enable`：设置成true，表示开启同步Dump；设置成false时，采用异步Dump。不设置该字段时默认值为false，开启异步Dump。两者的区别是异步Dump对原本代码执行过程的影响更小。
        - `trans_flag`：开启格式转换，将设备上的数据格式转换成NCHW格式。若为`true`，则数据会以Host侧的4D格式（NCHW）格式保存；若为`false`，则保留Device侧的数据格式。该配置参数在CPU上无效，因为CPU上没有format转换。默认值：true。
        - `stat_calc_mode`：选择统计信息计算后端，可选"host"和"device"。选择"device"后可以使能device计算统计信息，当前只在Ascend生效，只支持`min/max/avg/l2norm`统计量。在op_debug_mode设置为3时，仅支持将`stat_calc_mode`设置为"host"。
        - `device_stat_precision_mode`（可选）：device统计信息精度模式，可选"high"和"low"。选择"high"时，`avg/l2norm`统计量使用float32进行计算，会增加device内存占用，精度更高；为"low"时使用与原始数据相同的类型进行计算，device内存占用较少，但在处理较大数值时可能会导致统计量溢出。默认值为"high"。
        - `sample_mode`（可选）：设置成0，表示不开启切片dump功能；设置成1时，在图编译等级为O0或O1的情况下开启切片dump功能。仅在op_debug_mode设置为0时生效，其他场景不会开启切片dump功能。
        - `sample_num`（可选）：用于控制切片dump中切片的大小。默认值为100。
        - `save_kernel_args`（可选）: 设置成true时，会保存算子的初始化信息。仅当`enable`设置为`true`时生效。

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
   若脚本中都不调用`model.train`或`DatasetHelper`，则默认为非数据下沉模式。使用Dump功能将自动生成最终执行图的IR文件。

   可以在训练脚本中设置`set_context(reserve_class_name_in_scope=False)`，避免Dump文件名称过长导致Dump数据文件生成失败。

4. 通过`numpy.load`读取和解析Dump数据，参考[Ascend O0/O1模式下Dump数据文件介绍](#数据对象目录和数据文件介绍)。

### 数据对象目录和数据文件介绍

启动训练后，Ascend O0/O1模式下Dump保存的数据对象包括最终执行图（`ms_output_trace_code_graph_{graph_id}.ir`文件）以及图中算子的输入和输出数据，数据目录结构如下所示：

```text
{path}/
    - rank_{rank_id}/
        - .dump_metadata/
        - {net_name}/
            - {graph_id}/
                - {iteration_id}/
                    {op_type}.{op_name}.json
                    statistic.csv
                    {op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.{dtype}.npy
                - constants/
                    Parameter.data-{data_id}.0.0.{timestamp}.output.0.DefaultFormat.{dtype}.npy
            ...
        - graphs/
            ms_output_trace_code_graph_{graph_id}.pb
            ms_output_trace_code_graph_{graph_id}.ir
        - execution_order/
            ms_execution_order_graph_{graph_id}.csv
            ms_global_execution_order_graph_{graph_id}.csv
```

- `path`：`data_dump.json`配置文件中设置的绝对路径。
- `rank_id`：逻辑卡号。
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
- `dtype`: 原始的数据类型。如果是`bfloat16`、`int4`或`uint1`类型，保存在`.npy`文件中的数据会分别被转换成`float32`、`int8`或`uint8`类型。
- `data_id`: 常量数据标号。

对于多图网络，由于存在控制流，某些子图可能不会被执行，Dump只保存执行过的节点，所以graphs目录下`.pb`文件名中的{graph_id}并不一定在{net_name}下存在对应的{graph_id}目录。

只当`saved_data`为"statistic"或者"full"时，才会生成`statistic.csv`，当`saved_data`为"tensor"或者"full"时，才会生成`{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.{dtype}.npy`命名的完整张量信息。

只当`save_kernel_args`为`true`时，才会生成`{op_type}.{op_name}.json`，保存算子的初始化信息。该json文件内部格式为算子各初始化参数的对应值，以`Matmul`算子为例，json信息如下：

```json
{
    "transpose_a": "False",
    "transpose_b": "False"
}
```

代表`Matmul`算子的两个初始化参数`transpose_a`和`transpose_b`的值均为`False`。

Ascend O0/O1模式下Dump生成的数据文件是后缀名为`.npy`的文件，文件命名格式为：

```text
{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.{dtype}.npy
```

可以用Numpy的`numpy.load`接口读取数据。

Ascend O0/O1模式下生成的统计数据文件名为`statistic.csv`，此文件存有相同目录下所有落盘张量（文件名为`{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy`）的统计信息。每个张量一行，每行有张量的 Op Type、Op Name、Task ID、Stream ID、Timestamp、IO、Slot、Data Size、Data Type、Shape以及用户配置的统计信息项。注意，如果用Excel来打开此文件，数据可能无法正确显示。请用`vi`、`cat`等命令查看，或者使用Excel自文本导入csv查看。

Ascend O0/O1模式下生成的最终执行图文件后缀名分别为`.pb`和`.ir`，文件命名格式为：

```text
ms_output_trace_code_graph_{graph_id}.pb
ms_output_trace_code_graph_{graph_id}.ir
```

其中以`.ir`为后缀的文件可以通过`vi`命令打开查看。

Ascend O0/O1模式下Dump生成的节点执行序文件后缀名为`.csv`，文件命名格式为：

```text
ms_execution_order_graph_{graph_id}.csv
```

### 数据分析样例

为了更好地展示使用Dump来保存数据并分析数据的流程，我们提供了一套[完整样例脚本](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/dump) ，只需要执行 `bash run_sync_dump.sh`。

在通过Dump功能将脚本对应的图保存到磁盘上后，会产生最终执行图文件`ms_output_trace_code_graph_{graph_id}.ir`。该文件中保存了对应的图中每个算子的堆栈信息，记录了算子对应的生成脚本。

以[AlexNet脚本](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/sample_code/dump/train_alexnet.py)为例：

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
`Conv2D.Conv2D-op12.0.0.1623124369613540.output.0.DefaultFormat.float16.npy`。

还原数据的时候，通过执行以下代码：

```python
import numpy
numpy.load("Conv2D.Conv2D-op12.0.0.1623124369613540.output.0.DefaultFormat.float16.npy")
```

生成numpy.array数据。

## Ascend下O2模式Dump

Ascend下O2模式Dump已迁移到msprobe工具，更多详情请查看[《msprobe 工具 MindSpore场景精度数据采集指南》](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md)。

采集方式请参考示例代码[《msprobe静态图场景采集》](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md#71-%E9%9D%99%E6%80%81%E5%9B%BE%E5%9C%BA%E6%99%AF)；

配置文件示例请参考[《config.json 配置示例》](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/03.config_examples.md#2-mindspore-%E9%9D%99%E6%80%81%E5%9B%BE%E5%9C%BA%E6%99%AF)中的“MindSpore 静态图场景”；

详细配置介绍请参考[《config.json 配置文件介绍》](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/02.config_introduction.md#11-%E9%80%9A%E7%94%A8%E9%85%8D%E7%BD%AE)。

> 迁移到msporbe后部分功能暂不支持：
>
> 1. 数据切片保存，对应原配置中sample_num和sample_mode字段；
>
> 2. set_dump能力，对应原配置中dump_mode为2的场景；
>
> 3. tensor和statistic同时保存，对应原配置中saved_data为full的场景；
>
> 4. MD5和其他统计量无法同时开启，对应原配置中statistic_category字段。

## CPU/GPU模式Dump

### 操作步骤

1. 创建json格式的配置文件，JSON文件的名称和位置可以自定义设置。

    ```json
    {
        "common_dump_settings": {
            "op_debug_mode": 0,
            "dump_mode": 0,
            "path": "/absolute_path",
            "net_name": "ResNet50",
            "iteration": "0|5-8|100-120",
            "saved_data": "tensor",
            "input_output": 0,
            "kernels": ["Default/Conv-op12"],
            "support_device": [0,1,2,3,4,5,6,7],
            "statistic_category": ["max", "min", "l2norm"]
        },
        "e2e_dump_settings": {
            "enable": true,
            "trans_flag": true
        }
    }
    ```

    - `common_dump_settings`:

        - `op_debug_mode`：该属性用于算子溢出或算子异常调试，CPU/GPU Dump只支持设置成0，表示保存所有算子或指定算子。
        - `dump_mode`：设置成0，表示Dump出该网络中的所有算子数据；设置成1，表示Dump`"kernels"`里面指定的算子数据或算子类型数据；设置成2，表示使用[mindspore.set_dump](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mindspore/mindspore.set_dump.html) Dump指定对象。仅在op_debug_mode设置为0时支持指定算子dump。
        - `path`：Dump保存数据的绝对路径。
        - `net_name`：自定义的网络名称，例如："ResNet50"。
        - `iteration`：指定需要Dump数据的迭代。类型为str，用“|”分离要保存的不同区间的step的数据。如"0|5-8|100-120"表示Dump第1个，第6个到第9个，第101个到第121个step的数据。指定“all”，表示Dump所有迭代的数据。仅在op_debug_mode设置为0或3时支持保存指定迭代，op_debug_mode设置为4时不支持指定迭代。
        - `saved_data`: 指定Dump的数据。类型为str，取值成"tensor"，表示Dump出完整张量数据；取值成"statistic"，表示只Dump张量的统计信息；取值"full"代表两种都要。统计信息现只支持GPU场景，CPU场景若选"statistic"或"full"便会错误退出。默认取值为"tensor"。保存统计信息仅支持op_debug_mode设置为0的场景。
        - `input_output`：设置成0，表示Dump出算子的输入和算子的输出；设置成1，表示Dump出算子的输入；设置成2，表示Dump出算子的输出。在op_debug_mode设置为4时，只能保存算子输入。
        - `kernels`：该项可以配置三种格式：
          1. 算子的名称列表。通过设置环境变量`MS_DEV_SAVE_GRAPHS`的值为2开启IR保存开关并执行用例，从生成的IR文件`trace_code_graph_{graph_id}`中获取算子名称。详细说明可以参照教程：[如何保存IR](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/debug/error_analysis/mindir.html#如何保存ir)。
          需要注意的是，是否设置环境变量`MS_DEV_SAVE_GRAPHS`的值为2可能会导致同一个算子的id不同，所以在Dump指定算子时要在获取算子名称之后保持这一项设置不变。或者也可以在Dump保存的`ms_output_trace_code_graph_{graph_id}.ir`文件中获取算子名称，参考[CPU/GPU模式下Dump数据对象目录](#数据对象目录和数据文件介绍-1)。
          2. 还可以指定算子类型。当字符串中不带算子scope信息和算子id信息时，后台则认为其为算子类型，例如："conv"。算子类型的匹配规则为：当发现算子名中包含算子类型字符串时，则认为匹配成功（不区分大小写），例如："conv" 可以匹配算子 "Conv2D-op1234"、"Conv3D-op1221"。
          3. 算子名称的正则表达式。当字符串符合"name-regex(xxx)"格式时，后台则会将其作为正则表达式。例如，"name-regex(Default/.+)"可匹配算子名称以"Default/"开头的所有算子。
        - `support_device`：支持的设备，默认设置成0到7即可；在分布式训练场景下，需要dump个别设备上的数据，可以只在`support_device`中指定需要Dump的设备Id。该配置参数在CPU上无效，因为CPU下没有device这个概念，但是在json格式的配置文件中仍需保留该字段。
        - `statistic_category`: 该属性用于用户配置要保存的统计信息类别，仅在开启了保存统计信息(即`saved_data`设置为"statistic"或"full")时生效。类型为字符串列表，其中的字符串可选值如下：

            - "max": 表示Tensor中元素的最大值；
            - "min": 表示Tensor中元素的最小值；
            - "avg": 表示Tensor中元素的平均值；
            - "count": 表示Tensor中元素的个数；
            - "negative zero count": 表示Tensor中小于0的元素个数；
            - "positive zero count": 表示Tensor中大于0的元素个数；
            - "nan count": 表示Tensor中元素的`Nan`的个数；
            - "negative inf count": 表示Tensor中`-Inf`元素的个数；
            - "positive inf count": 表示Tensor中`+Inf`元素的个数；
            - "zero count": 表示Tensor中元素`0`的个数；
            - "md5": 表示Tensor的MD5值；
            - "l2norm": 表示Tensor的L2Norm值。

        CPU/GPU Dump模式只支持host测统计信息及结算。
        该字段为可选，默认值为["max", "min", "l2norm"]。

    - `e2e_dump_settings`:

        - `enable`：在CPU/GPU Dump模式下，该字段必须设置为`true`。
        - `trans_flag`：开启格式转换。将设备上的数据格式转换成NCHW格式。若为`true`，则数据会以Host侧的4D格式（NCHW）格式保存；若为`false`，则保留Device侧的数据格式。该配置参数在CPU上无效，因为CPU上没有format转换。默认值：true。

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
   GPU环境如果要Dump数据，必须采用非数据下沉模式（设置`model.train`或`DatasetHelper`中的`dataset_sink_mode`参数为`False`），以保证可以获取每个step的Dump数据。
   若脚本中都不调用`model.train`或`DatasetHelper`，则默认为非数据下沉模式。使用Dump功能将自动生成最终执行图的IR文件。

   可以在训练脚本中设置`set_context(reserve_class_name_in_scope=False)`，避免Dump文件名称过长导致Dump数据文件生成失败。

4. 通过`numpy.load`读取和解析CPU/GPU模式下Dump数据，参考[CPU/GPU模式下Dump数据文件介绍](#数据对象目录和数据文件介绍-1)。

### 数据对象目录和数据文件介绍

启动训练后，CPU/GPU模式下Dump保存的数据对象包括最终执行图（`ms_output_trace_code_graph_{graph_id}.ir`文件）以及图中算子的输入和输出数据，数据目录结构如下所示：

```text
{path}/
    - rank_{rank_id}/
        - .dump_metadata/
        - {net_name}/
            - {graph_id}/
                - {iteration_id}/
                    {op_type}.{op_name}.json
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
- `rank_id`：逻辑卡号。
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

CPU/GPU模式下Dump生成的数据文件是后缀名为`.npy`的文件，文件命名格式为：

```text
{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy
```

CPU/GPU模式下Dump生成的常量数据文件与其他数据文件格式相同，而所有常量数据的{op_type}、{task_id}、{stream_id}、{input_output_index}、{slot}、{format}不变。

```text
Parameter.data-{data_id}.0.0.{timestamp}.output.0.DefaultFormat.npy
```

{iteration_id}目录下也可能会保存Parameter开头的文件（weight、bias等参数会保存成Parameter开头的文件。

可以用Numpy的`numpy.load`接口读取数据。

CPU/GPU模式下Dump生成的统计数据文件名为`statistic.csv`，此文件存有相同目录下所有落盘张量（文件名为`{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy`）的统计信息。每个张量一行，每行有张量的 Op Type、Op Name、Task ID、Stream ID、Timestamp、IO，Slot、Data Size、Data Type、Shape以及用户配置的统计信息项。注意，如果用Excel来打开此文件，数据可能无法正确显示。请用`vi`、`cat`等命令查看，或者使用Excel自文本导入csv查看。

CPU/GPU模式下Dump生成的最终执行图文件后缀名分别为`.pb`和`.ir`，文件命名格式为：

```text
ms_output_trace_code_graph_{graph_id}.pb
ms_output_trace_code_graph_{graph_id}.ir
```

其中以`.ir`为后缀的文件可以通过`vi`命令打开查看。

CPU/GPU模式下Dump生成的节点执行序文件后缀名为`.csv`，文件命名格式为：

```text
ms_execution_order_graph_{graph_id}.csv
```

图执行历史文件的后缀为`.csv`，文件名格式为：

```text
ms_global_execution_order_graph_{graph_id}.csv
```

此文件记录该图在训练过程中的执行轮次历史。图编译过程中，一张根图可能产生多张子图，但子图与根图具有相同的执行轮次历史。故与图执行序文件不同，此处仅保存根图的图执行历史文件。

`.dump_metadata`记录了训练的原信息，其中`data_dump.json`保存了用户设置的dump配置。

### 数据分析样例

为了更好地展示使用Dump来保存数据并分析数据的流程，我们提供了一套[完整样例脚本](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/dump) ，CPU/GPU模式下Dump只需要执行 `bash run_sync_dump.sh`。

在通过Dump功能将脚本对应的图保存到磁盘上后，会产生最终执行图文件`ms_output_trace_code_graph_{graph_id}.ir`。该文件中保存了对应的图中每个算子的堆栈信息，记录了算子对应的生成脚本。

以[AlexNet脚本](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/sample_code/dump/train_alexnet.py)为例：

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

## 注意事项

- `bfloat16`类型的算子保存到`npy`文件时，会转换成`float32`类型。
- Dump仅支持bool、int、int8、in16、int32、int64、uint、uint8、uint16、uint32、uint64、float、float16、float32、float64、bfloat16、double、complex64、complex128类型数据的保存。
- complex64和complex128仅支持保存为npy文件，不支持保存为统计值信息。
- Print算子内部有一个输入参数为string类型，string类型不属于Dump支持的数据类型，所以在脚本中包含Print算子时，会有错误日志，这不会影响其他类型数据的保存。
- 使能Ascend O2模式下Dump时，不支持同时使用set_context(ascend_config={"exception_dump": "2"})配置轻量异常dump; 支持同时使用set_context(ascend_config={"exception_dump": "1"})配置全量异常dump。
- 使能Ascend O2模式下Dump时，sink size只能设置为1。用户通常可以使用model.train()或ms.data_sink()接口配置sink size。
- 使能Ascend O2模式下Dump时，**统计值dump**如果是大数据量dump场景（如网络本身规模庞大，连续dump多个step等），可能会导致host侧内存被占满，导致数据流同步失败，建议使用新版[**统计值dump**](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md#51-%E9%9D%99%E6%80%81%E5%9B%BE%E5%9C%BA%E6%99%AF)替代。
