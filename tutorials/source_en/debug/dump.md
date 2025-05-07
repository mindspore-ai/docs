# Using Dump in the Graph Mode

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/debug/dump.md)

To analyze the training process, MindSpore provides the dump function to store the input and output data of operators during the training process.

## Feature Evolution

The MindSpore Dump functionality has been gradually migrated to the [msprobe tool](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe).

> [msprobe](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe) is a toolkit under the MindStudio Training Tools suite, specifically for accuracy debugging. It primarily includes functionalities such as accuracy pre-inspection, overflow detection, and accuracy comparison. Currently, it is compatible with the PyTorch and MindSpore frameworks.

The Dump features for dynamic graphs and static graphs in Ascend O2 mode have been fully migrated to the msprobe tool and are enabled through the msprobe tool entry point. For more details, please refer to the [msprobe Tool MindSpore Scenario Accuracy Data Collection Guide](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md).

For graphs in Ascend OO/O1 modes and CPU/GPU modes, these functionalities are still enabled through the framework entry points but will be gradually migrated to the msprobe tool in subsequent updates.

## Configuration Guide

In different modes, the Dump features supported by MindSpore are not entirely the same, and the required configuration files and the generated data formats vary accordingly. Therefore, you need to select the corresponding Dump configuration based on the running mode:

- [Dump in Ascend O0/O1 Mode](#dump-in-ascend-o0o1-mode)
- [Dump in Ascend O2 Mode](#dump-in-ascend-o2-mode)
- [Dump in CPU/GPU mode](#dump-in-cpugpu-mode)

> - The differences between Ascend O0, O1, and O2 modes can be found in [the parameter jit_level of the set_context method](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_context.html).
>
> - Dumping constant data is only supported in CPU/GPU mode, while not supported in Ascend O0/O1/O2 mode.
>
> - Currently, Dump does not support heterogeneous training, meaning it does not support CPU/Ascend mixed training or GPU/Ascend mixed training.

MindSpore supports different Dump functionalities under various modes, as shown in the following table:

<table align="center">
  <tr>
   <td colspan="2" align="center">Feature</td>
   <td align="center">Ascend O0/O1</td>
   <td align="center">CPU/GPU</td>
  </tr>
  <tr>
   <td align="left">Full Dump</td>
   <td align="left">Full network data dump</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td rowspan="2" align="left">Partial Data Dump</td>
   <td align="left">Statistics Dump</td>
   <td align="left">Supports both host and device modes<sup>1</sup></td>
   <td align="left">Not Supported On CPU, GPU Supports only host mode</td>
  </tr>
  <tr>
   <td align="left">Data Sampling Dump</td>
   <td align="left">Supported</td>
   <td align="left">Not Supported</td>
  </tr>
  <tr>
   <td align="left">Overflow Dump</td>
   <td align="left">Dump overflow operators</td>
   <td align="left">Supported</td>
   <td align="left">Not Supported</td>
  </tr>
  <tr>
   <td rowspan="5" align="left">Conditional Dump</td>
   <td align="left">Specify Operator Name</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td align="left">Specify Iteration</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td align="left">Specify Device</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td align="left">Specify File Format</td>
   <td align="left">Not Applicable</td>
   <td align="left">Not Applicable</td>
  </tr>
  <tr>
   <td align="left">set_dump</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td rowspan="2" align="left">Auxiliary Information Dump</td>
   <td align="left">Graph IR Dump</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td align="left">Execution Sequence Dump</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
  </tr>
</table>

> In terms of statistics, the computing speed of the device is faster than that of the host(currently only supported on Ascend backend), but the host has more statistical indicators than the device. Refer to the `statistic_category` option for details.

## Dump in Ascend O0/O1 Mode

### Dump Step

1. Create a configuration file in json format, and the name and location of the JSON file can be customized.

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

        - `op_debug_mode`: This attribute is used for operator overflow or operator exception debugging. 0: save all operators or specified operators; 3: only save overflow operators; 4: only save input of the exception operator. Set it to 0 when the data is dumped. If it is not set to 0, only the data of the overflow operator or exception operator will be dumped. Default: 0.
        - `dump_mode`: 0: all operator data in the network dumped out; 1: the operator data specified in Dump `"kernels"`; 2: dump target and its contents using [mindspore.set_dump](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_dump.html). Specified data dump is supported only when "dump_mode' is set to `0`.
        - `path`: The absolute path to Dump saved data.
        - `net_name`: The customized net name: "ResNet50".
        - `iteration`: Specify the iterations of data required to be dumped, type is string. Use "|" to separate the step data of different intervals to be saved. For example, "0 | 5-8 | 100-120" represents dump the data of the 1st, 6th to 9th, and 101st to 121st steps. If iteration set to "all", data of every iteration will be dumped. Specified iteration dump is supported only when "op_debug_mode" is set to `0` or `3`, not supported when when "op_debug_mode" is set to `4`.
        - `saved_data`: Specify what data is to be dumped, type is string. Use "tensor" to indicate complete tensor data Dumped, use "statistic" to dump tensor statistics, use "full" to dump both tensor data and statistics. Default setting is "tensor". Statistic dump is only supported when "op_debug_mode" is set to `0`.
        - `input_output`: 0: dump input and output of kernel, 1:dump input of kernel, 2:dump output of kernel. When `op_debug_mode` is set to 3, `input_output` can only be set to save both the operator's inputs and outputs. Only input of kernel can be saved when "op_debug_mode" is set to `4`.
        - `kernels`: This item can be configured in three formats:
           1. List of operator names. Turn on the IR save switch by setting the environment variable `MS_DEV_SAVE_GRAPHS` to 2 and execute the network to obtain the operator name from the generated `trace_code_graph_{graph_id}`IR file. For details, please refer to [Saving IR](https://www.mindspore.cn/tutorials/en/master/debug/error_analysis/mindir.html#saving-ir).
           Note that whether setting the environment variable `MS_DEV_SAVE_GRAPHS` to 2 may cause the different IDs of the same operator, so when dump specified operators, keep this setting unchanged after obtaining the operator name. Or you can obtain the operator names from the file `ms_output_trace_code_graph_{graph_id}.ir` saved by Dump. Refer to [Ascend O0/O1 Dump Data Object Directory](#introduction-to-data-object-directory-and-data-file).
           2. You can also specify an operator type. When there is no operator scope information or operator id information in the string, the background considers it as an operator type, such as "conv". The matching rule of operator type is: when the operator name contains an operator type string, the matching is considered successful (case insensitive). For example, "conv" can match operators "Conv2D-op1234" and "Conv3D-op1221".
           3. Regular expressions are supported. When the string conforms to the format of "name-regex(xxx)", it would be considered a regular expression. For example, "name-regex(Default/.+)" can match all operators with names starting with "Default/".
        - `support_device`: Supported devices, default setting is `[0,1,2,3,4,5,6,7]`. In distributed training scenarios where data on individual devices needs to be dumped, you can specify only the device Id that needs to be dumped in `support_device`. This configuration parameter is invalid on the CPU, because there is no concept of device on the CPU, but it is still need to reserve this parameter in the json file.
        - `statistic_category`: This attribute is used by users to configure the category of statistical information to be saved, and only takes effect when saving statistical information is enabled(i.e.`saved_data` is set to `statistic` or `full`). The type is a string list, where the optional values of the strings are as follows:

            - "max": represents the maximum value of the elements in tensor, supporting both device and host statistics;
            - "min": represents the minimum value of the elements in tensor, supporting both device and host statistics;
            - "avg": represents the average value of elements in tensor, supporting device and host statistics;
            - "count": represents the number of the elements in tensor;
            - "negative zero count": represents the number of the elements which is less then zero in tensor;
            - "positive zero count": represents the number of the elements which is greater then zero in tensor;
            - "nan count": represents the number of `Nan` elements in the tensor;
            - "negative inf count": represents the number of `-Inf` elements in the tensor;
            - "positive inf count": represents the number of `+Inf` elements in the tensor;
            - "zero count": represents the number of zero elements in the tensor;
            - "md5": represents the MD5 value of the tensor;
            - "l2norm": represents L2Norm value of the tensor, supporting both device and host statistics.

            Except for those marked as supporting device statistics, other statistics can be collected only on the host.
            This field is optional, with default values of ["max", "min", "l2norm"].

        - `overflow_number`: Specify the number of data to overflow dump. This field is required only when `op_debug_mode` is set to 3 and only the overflow operator is saved. It can control the overflow data to be dumped in chronological order until the specified value is reached, and the overflow data will no longer be dumped. The default value is 0, which means dumping all overflow data.
        - `initial_iteration`: Specifies the initial iteration number for Dump, which must be a non-negative integer. If set to 10, the iteration count for the initial Dump will start from 10. Default value: 0.

    - `e2e_dump_settings`:

        - `enable`: When set to `true`, enable Synchronous Dump. When set to false or not set, Asynchronous Dump will be used on Ascend. The main difference between the two is that Asynchronous Dump has less impact on the original code execution order.
        - `trans_flag`: Enable trans flag. Transform the device data format into NCHW. If it is `true`, the data will be saved in the 4D format (NCHW) format on the Host side; if it is `false`, the data format on the Device side will be retained. Default: `true`.
        - `stat_calc_mode`: Select the backend for statistical calculations. Options are "host" and "device". Choosing "device" enables device computation of statistics, currently only effective on Ascend, and supports only min/max/avg/l2norm statistics. When `op_debug_mode` is set to 3, only `stat_calc_mode` set to "host" is supported.
        - `device_stat_precision_mode`(Optional): Precision mode of device statistics, and the value can be "high" or "low". When "high" is selected, avg/l2norm statistics will be calculated using float32, which will increase device memory usage and have higher precision; when "low" is selected, the same type as the original data will be used for calculation, which will occupy less device memory, but statistics overflow may be caused when processing large values. The default value is "high".
        - `sample_mode`(Optional): Setting it to 0 means the sample dump function is not enabled. Enable the sample dump function in graph compilation with optimization level O0 or O1. This field is effective only when "op_debug_mode" is set to `0`, sample dump cannot be enabled in other scene.
        - `sample_num`(Optional): Used to control the size of sample in sample dump. The default value is 100.
        - `save_kernel_args`(Optional): When set to true, the initialization information of kernels will be saved. This field is effective only when `enable` is set to `true`.

2. Set Dump environment variable.

   Specify the json configuration file of Dump.

   ```bash
   export MINDSPORE_DUMP_CONFIG=${xxx}
   ```

   "xxx" represents the absolute path to the configuration file.

   ```bash
   export MINDSPORE_DUMP_CONFIG=/path/to/data_dump.json
   ```

   If the `path` field is not set or set to an empty string in the Dump configuration file, you also need to configure the environment variable `MS_DIAGNOSTIC_DATA_PATH`.

   ```bash
   export MS_DIAGNOSTIC_DATA_PATH=${yyy}
   ```

   Then "$MS_DIAGNOSTIC_DATA_PATH/debug_dump" is regarded as `path`. If the `path` field is set in Dump configuration file, the actual value of the field is still the same.

   Note:

   - Set the environment variables before executing the training script. Setting environment variables during training will not take effect.
   - Dump environment variables need to be configured before calling `mindspore.communication.init`.

3. Execute the training script to dump data.

   After the training is started, if the `MINDSPORE_DUMP_CONFIG` environment variable is correctly configured, the content of the configuration file will be read and the operator data will be saved according to the data storage path specified in the Dump configuration.
   If `model.train` or `DatasetHelper` is not called in the script, the default is non-data sinking mode. Using the Dump function will automatically generate the IR file of the final execution graph.

   You can set `set_context(reserve_class_name_in_scope=False)` in your training script to avoid dump failure because of file name is too long.

4. Read and parse dump data through `numpy.load`, refer to [Introduction to Ascend O0/O1 Dump Data File](#introduction-to-data-object-directory-and-data-file).

### Introduction to Data Object Directory and Data File

After starting the training, the data objects saved under the Ascend O0/O1 Dump mode include the final execution graph (`ms_output_trace_code_graph_{graph_id}.ir` file) and the input and output data of the operators in the graph. The data directory structure is as follows:

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

- `path`: the absolute path set in the `data_dump.json` configuration file.
- `rank_id`: the id of the logic device.
- `net_name`: the network name set in the `data_dump.json` configuration file.
- `graph_id`: the id of the training graph.
- `iteration_id`: the iteration of the training.
- `op_type`: the type of the operator.
- `op_name`: the name of the operator.
- `task_id`: the id of the task.
- `stream_id`: the id of the stream.
- `timestamp`: the time stamp.
- `input_output_index` : the index of input or output. For example, `output_0` means that the file is the data of the first output Tensor of the operator.
- `slot`: the id of the slot.
- `format`: the format of the data.
- `dtype`: the original data type. When it is `bfloat16`, `int4` or `uint1`, the saved data in the `.npy` file is converted to `float32`, `int8` or `uint8` respectively.
- `data_id`: the id of constant data.

For multi-graph networks, due to the control flow, some subgraphs may not be executed, but Dump only saves the executed nodes, so the {graph_id} in the `.pb` file name in the graphs directory does not necessarily exist in the {graph_id} directory under {net_name}.

Only when `saved_data` is "statistic" or "full", `statistic.csv` is generated. Only when `saved_data` is "tensor" or "full", `{op_type}. {op_name}. {task_id}. {stream_id}. {timestamp}. {input_output_index}. {slot}. {format}.{dtype}.npy` named complete tensor information is generated.

Only when `save_kernel_args` is `true`, `{op_type}.{op_name}.json` is generated and the params of the corresponding operators is saved. The internal format of this JSON file contains the corresponding values of each initialization parameter of the operator. For example, for the `Matmul` operator, the JSON information would look like this:

```json
{
    "transpose_a": "False",
    "transpose_b": "False"
}
```

This JSON indicates that both initialization parameters `transpose_a` and `transpose_b` of the `Matmul` operator have the value `False`.

The data file generated by the Ascend O0/O1 Dump is a binary file with the suffix `.npy`, and the file naming format is:

```text
{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.{dtype}.npy
```

User can use Numpy interface `numpy.load` to read the data.

The statistics file generated by the Ascend O0/O1 dump is named `statistic.csv`. This file stores key statistics for all tensors dumped under the same directory as itself (with the file names `{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy`). Each row in `statistic.csv` summarizes a single tensor, each row contains the statistics: Op Type, Op Name, Task ID, Stream ID, Timestamp, IO, Slot, Data Size, Data Type, Shape, and statistics items configured by the user. Note that opening this file with Excel may cause data to be displayed incorrectly. Please use commands like `vi` or `cat`, or use Excel to import csv from text for viewing.

The suffixes of the final execution graph files generated by Ascend O0/O1 Dump are `.pb` and `.ir` respectively, and the file naming format is:

```text
ms_output_trace_code_graph_{graph_id}.pb
ms_output_trace_code_graph_{graph_id}.ir
```

The files with the suffix `.ir` can be opened and viewed by the `vi` command.

The suffix of the node execution sequence file generated by the Ascend O0/O1 Dump is `.csv`, and the file naming format is:

```text
ms_execution_order_graph_{graph_id}.csv
```

### Data Analysis Sample

In order to better demonstrate the process of using dump to save and analyze data, we provide a set of [complete sample script](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/dump) , you only need to execute `bash dump_sync_dump.sh` for Ascend O0/O1 dump.

After the graph corresponding to the script is saved to the disk through the Dump function, the final execution graph file `ms_output_trace_code_graph_{graph_id}.ir` will be generated. This file saves the stack information of each operator in the corresponding graph, and records the generation script corresponding to the operator.

Take [AlexNet script](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/dump/train_alexnet.py) as an example:

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

If the user wants to view the code at line 175 in the script:

```python
x = self.conv3(x)
```

After executing the network training, you can find multiple operator information corresponding to the line of code from the final execution graph (`ms_output_trace_code_graph_{graph_id}.ir` file). The content of the file corresponding to Conv2D-op12 is as follows:

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

The meanings of the lines in the file content shown above are as follows:

- The input and output of the operator on the Host side (the first line) and the Device side (the second line, some operators may not exist). It can be seen from the execution graph that the operator has two inputs (left side of the arrow) and one output (right side of the arrow).

    ```text
       : (<Tensor[Float32], (32, 256, 13, 13)>, <Tensor[Float32], (384, 256, 3, 3)>) -> (<Tensor[Float32], (32, 384, 13, 13)>)
       : (<Float16xNC1HWC0[const vector][32, 16, 13, 13, 16]>, <Float16xFracZ[const vector][144, 24, 16, 16]>) -> (<Float32xNC1HWC0[const vector][32, 24, 13, 13, 16]>)
    ```

- Operator name. It can be seen from the execution graph that the full name of the operator in the final execution graph is `Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op12`.

    ```text
    : (Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op12)
    ```

- The training script code corresponding to the operator. By searching the training script code to be queried, multiple matching operators can be found.

    ```text
    # In file {Absolute path of model_zoo}/official/cv/alexnet/src/alexnet.py(175)/        x = self.conv3(x)/
    ```

Through the operator name and input and output information, you can find the only corresponding Tensor data file. For example, if you want to view the dump file corresponding to the first output data of the Conv2D-op12 operator, you can obtain the following information:

- `operator_name`: `Conv2D-op12`.

- `input_output_index`: `output.0` indicates that the file is the data of the first output Tensor of the operator.

- `slot`: 0, this tensor only has one slot.

Search for the corresponding file name in the data object file directory saved by Dump:
`Conv2d.Conv2D-op12.0.0.1623124369613540.output.0.DefaultFormat.float16.npy`.

When restoring data, execute:

```python
import numpy
numpy.load("Conv2D.Conv2D-op12.0.0.1623124369613540.output.0.DefaultFormat.float16.npy")
```

Generate the numpy.array data.

## Dump in Ascend O2 Mode

O2 mode Dump under Ascend has been migrated to the msprobe tool. For more details, please see [msprobe Tool MindSpore Scene Accuracy Data Collection Guide](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md).

For data collection methods, please refer to the example code in [Graph Scenario Data Collection with msprobe](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md#71-%E9%9D%99%E6%80%81%E5%9B%BE%E5%9C%BA%E6%99%AF);

For configuration file examples, please refer to the "MindSpore Graph Scenario" section in [config.json Configuration Examples](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/03.config_examples.md#2-mindspore-%E9%9D%99%E6%80%81%E5%9B%BE%E5%9C%BA%E6%99%AF);

For detailed configuration descriptions, please refer to the [Introduction to config.json Configuration File](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/02.config_introduction.md#11-%E9%80%9A%E7%94%A8%E9%85%8D%E7%BD%AE).

> After migrating to msprobe, some features are temporarily not supported:
>
> 1. Data slicing storage, corresponding to the sample_num and sample_mode fields in the original configuration;
>
> 2. set_dump capability, corresponding to scenarios where dump_mode is set to 2 in the original configuration;
>
> 3. Simultaneous saving of tensor and statistics, corresponding to the saved_data field being set to full in the original configuration;
>
> 4. Simultaneous enabling of MD5 and other statistics, corresponding to the statistic_category field in the original configuration.

## Dump in CPU/GPU Mode

### Dump Step

1. Create a configuration file in json format, and the name and location of the JSON file can be customized.

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

        - `op_debug_mode`: This attribute is used for operator overflow or operator exception debugging. 0 is the only supported mode in CPU/GPU Dump mode, which means saving all operators or specified operators;
        - `dump_mode`: 0: all operator data in the network dumped out; 1: the operator data specified in Dump `"kernels"`; 2: dump target and its contents using [mindspore.set_dump](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_dump.html). Specified data dump is supported only when "dump_mode' is set to `0`.
        - `path`: The absolute path to Dump saved data.
        - `net_name`: The customized net name: "ResNet50".
        - `iteration`: Specify the iterations of data required to be dumped, type is string. Use "|" to separate the step data of different intervals to be saved. For example, "0 | 5-8 | 100-120" represents dump the data of the 1st, 6th to 9th, and 101st to 121st steps. If iteration is set to "all", data of every iteration will be dumped. Specified iteration dump is supported only when "op_debug_mode" is set to `0` or `3`, not supported when when "op_debug_mode" is set to `4`.
        - `saved_data`: Specify what data is to be dumped, type is string. Use "tensor" to indicate complete tensor data Dumped, use "statistic" to dump tensor statistics, use "full" to dump both tensor data and statistics. Using "statistic" or "full" on CPU will result in exception. Default setting is "tensor". Statistic dump is only supported when "op_debug_mode" is set to `0`.
        - `input_output`: 0: dump input and output of kernel, 1: dump input of kernel, 2: dump output of kernel. Only input of kernel can be saved when "op_debug_mode" is set to `4`.
        - `kernels`: This item can be configured in three formats:
             1. List of operator names. Turn on the IR save switch by setting the environment variable `MS_DEV_SAVE_GRAPHS` to 2 and execute the network to obtain the operator name from the generated `trace_code_graph_{graph_id}`IR file. For details, please refer to [Saving IR](https://www.mindspore.cn/tutorials/en/master/debug/error_analysis/mindir.html#saving-ir).
             Note that whether setting the environment variable `MS_DEV_SAVE_GRAPHS` to 2 may cause the different IDs of the same operator, so when dump specified operators, keep this setting unchanged after obtaining the operator name. Or you can obtain the operator names from the file `ms_output_trace_code_graph_{graph_id}.ir` saved by Dump. Refer to [Ascend O0/O1 Dump Data Object Directory](#introduction-to-data-object-directory-and-data-file).
             2. You can also specify an operator type. When there is no operator scope information or operator id information in the string, the background considers it as an operator type, such as "conv". The matching rule of operator type is: when the operator name contains an operator type string, the matching is considered successful (case insensitive). For example, "conv" can match operators "Conv2D-op1234" and "Conv3D-op1221".
             3. Regular expressions are supported. When the string conforms to the format of "name-regex(xxx)", it would be considered a regular expression. For example, "name-regex(Default/.+)" can match all operators with names starting with "Default/".
        - `support_device`: Supported devices, default setting is `[0,1,2,3,4,5,6,7]`. You can specify specific device ids to dump specific device data. This configuration parameter is invalid on the CPU, because there is no concept of device on the CPU, but it is still need to reserve this parameter in the json file.
        - `statistic_category`: This attribute is used by users to configure the category of statistical information to be saved, and only takes effect when saving statistical information is enabled(i.e.`saved_data` is set to `statistic` or `full`). The type is a string list, where the optional values of the strings are as follows:

            - "max": represents the maximum value of the elements in tensor;
            - "min": represents the minimum value of the elements in tensor;
            - "avg": represents the average value of elements in tensor;
            - "count": represents the number of the elements in tensor;
            - "negative zero count": represents the number of the elements which is less then zero in tensor;
            - "positive zero count": represents the number of the elements which is greater then zero in tensor;
            - "nan count": represents the number of `Nan` elements in the tensor;
            - "negative inf count": represents the number of `-Inf` elements in the tensor;
            - "positive inf count": represents the number of `+Inf` elements in the tensor;
            - "zero count": represents the number of zero elements in the tensor;
            - "md5": represents the MD5 value of the tensor;
            - "l2norm": represents L2Norm value of the tensor.

        In CPU/GPU Dump Mode, all statistics are calculated on the host.
        This field is optional, with default values of ["max", "min", "l2norm"].

    - `e2e_dump_settings`:

        - `enable`: In CPU/GPU Dump Mode, this field must be set to `true`.
        - `trans_flag`: Enable trans flag. Transform the device data format into NCHW. If it is `true`, the data will be saved in the 4D format (NCHW) format on the Host side; if it is `false`, the data format on the Device side will be retained. Default: `true`.

2. Set Dump environment variable.

   Specify the json configuration file of Dump.

   ```bash
   export MINDSPORE_DUMP_CONFIG=${xxx}
   ```

   "xxx" represents the absolute path to the configuration file.

   ```bash
   export MINDSPORE_DUMP_CONFIG=/path/to/data_dump.json
   ```

   If the `path` field is not set or set to an empty string in the Dump configuration file, you also need to configure the environment variable `MS_DIAGNOSTIC_DATA_PATH`.

   ```bash
   export MS_DIAGNOSTIC_DATA_PATH=${yyy}
   ```

   Then "$MS_DIAGNOSTIC_DATA_PATH/debug_dump" is regarded as `path`. If the `path` field is set in Dump configuration file, the actual value of the field is still the same.

   Note:

    - Set the environment variables before executing the training script. Setting environment variables during training will not take effect.
    - Dump environment variables need to be configured before calling `mindspore.communication.init`.

3. Execute the training script to dump data.

   After the training is started, if the `MINDSPORE_DUMP_CONFIG` environment variable is correctly configured, the content of the configuration file will be read and the operator data will be saved according to the data storage path specified in the Dump configuration.
   If you want to dump data in GPU environment, you must use the non-data sink mode (set the `dataset_sink_mode` parameter in `model.train` or `DatasetHelper` to `False`) to ensure that you can get the dump data of each step.
   If `model.train` or `DatasetHelper` is not called in the script, the default is non-data sinking mode. Using the Dump function will automatically generate the IR file of the final execution graph.

    You can set `set_context(reserve_class_name_in_scope=False)` in your training script to avoid dump failure because of file name is too long.

4. Read and parse dump data through `numpy.load`, refer to [Introduction to CPU/GPU Dump Data File](#introduction-to-data-object-directory-and-data-file-1).

### Introduction to Data Object Directory and Data File

After starting the training, the data objects saved by the CPU/GPU Dump include the final execution graph (`ms_output_trace_code_graph_{graph_id}.ir` file) and the input and output data of the operators in the graph. The data directory structure is as follows:

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

- `path`: the absolute path set in the `data_dump.json` configuration file.
- `rank_id`: the id of the logic device.
- `net_name`: the network name set in the `data_dump.json` configuration file.
- `graph_id`: the id of the training graph.
- `iteration_id`: the iteration of the training.
- `op_type`: the type of the operator.
- `op_name`: the name of the operator.
- `task_id`: the id of the task.
- `stream_id`: the id of the stream.
- `timestamp`: the time stamp.
- `input_output_index` : the index of input or output. For example, `output_0` means that the file is the data of the first output Tensor of the operator.
- `slot`: the id of the slot.
- `format`: the format of the data.
- `data_id`: the id of constant data.

For multi-graph networks, due to the control flow, some subgraphs may not be executed, but Dump only saves the executed nodes, so the {graph_id} in the `.pb` file name in the graphs directory does not necessarily exist in the {graph_id} directory under {net_name}.

Only when `saved_data` is "statistic" or "full", `statistic.csv` is generated. Only when `saved_data` is "tensor" or "full", `{op_type}. {op_name}. {task_id}. {stream_id}. {timestamp}. {input_output_index}. {slot}. {format}.npy` named complete tensor information is generated.

The data file generated by the CPU/GPU Dump is a binary file with the suffix `.npy`, and the file naming format is:

```text
{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy
```

The constant data file generated by the CPU/GPU Dump is in the same format as data file, whereas {op_type}, {task_id}, {stream_id}, {input_output_index}, {slot}, {format} are unchanged for all constant data. Note, non-Tensor type will not generate data file.

```text
Parameter.data-{data_id}.0.0.{timestamp}.output.0.DefaultFormat.npy
```

The {iteration_id} directory may also save files starting with `Parameter` (parameters such as weight and bias will be saved as files starting with `Parameter`), while `Parameter` files will not be saved on Ascend.

User can use Numpy interface `numpy.load` to read the data.

The statistics file generated by the CPU/GPU dump is named `statistic.csv`. This file stores key statistics for all tensors dumped under the same directory as itself (with the file names `{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy`). Each row in `statistic.csv` summarizes a single tensor, each row contains the statistics: Op Type, Op Name, Task ID, Stream ID, Timestamp, IO, Slot, Data Size, Data Type, Shape, and statistics items configured by the user. Note that opening this file with Excel may cause data to be displayed incorrectly. Please use commands like `vi` or `cat`, or use Excel to import csv from text for viewing.

The suffixes of the final execution graph files generated by CPU/GPU Dump are `.pb` and `.ir` respectively, and the file naming format is:

```text
ms_output_trace_code_graph_{graph_id}.pb
ms_output_trace_code_graph_{graph_id}.ir
```

The files with the suffix `.ir` can be opened and viewed by the `vi` command.

The suffix of the node execution sequence file generated by the CPU/GPU Dump is `.csv`, and the file naming format is:

```text
ms_execution_order_graph_{graph_id}.csv
```

The suffix of the graph execution history file is `.csv`. The file naming format is:

```text
ms_global_execution_order_graph_{graph_id}.csv
```

This file stores the list of iterations in which the graph was executed. After the graph is compiled, it may be split into multiple sub-graphs. Since sub-graphs share the same graph execution history with root graph, only root graph will generate an execution history file. This function is not supported on Ascend.

`.dump_metadata` records the original training information(the directory is not available for Ascend backend), and `data_dump.json` saves the dump configuration set by the user.

### Data Analysis Sample

In order to better demonstrate the process of using dump to save and analyze data, we provide a set of [complete sample script](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/dump) , you only need to execute `bash dump_sync_dump.sh` for CPU/GPU dump.

After the graph corresponding to the script is saved to the disk through the Dump function, the final execution graph file `ms_output_trace_code_graph_{graph_id}.ir` will be generated. This file saves the stack information of each operator in the corresponding graph, and records the generation script corresponding to the operator.

Take [AlexNet script](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/dump/train_alexnet.py) as an example:

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

If the user wants to view the code at line 175 in the script:

```python
x = self.conv3(x)
```

After executing the network training, you can find multiple operator information corresponding to the line of code from the final execution graph (`ms_output_trace_code_graph_{graph_id}.ir` file). The content of the file corresponding to Conv2D-op12 is as follows:

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

The meanings of the lines in the file content shown above are as follows:

- The input and output of the operator on the Host side (the first line) and the Device side (the second line, some operators may not exist). It can be seen from the execution graph that the operator has two inputs (left side of the arrow) and one output (right side of the arrow).

    ```text
       : (<Tensor[Float32], (32, 256, 13, 13)>, <Tensor[Float32], (384, 256, 3, 3)>) -> (<Tensor[Float32], (32, 384, 13, 13)>)
       : (<Float16xNC1HWC0[const vector][32, 16, 13, 13, 16]>, <Float16xFracZ[const vector][144, 24, 16, 16]>) -> (<Float32xNC1HWC0[const vector][32, 24, 13, 13, 16]>)
    ```

- Operator name. It can be seen from the execution graph that the full name of the operator in the final execution graph is `Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op12`.

    ```text
    : (Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op12)
    ```

- The training script code corresponding to the operator. By searching the training script code to be queried, multiple matching operators can be found.

    ```text
    # In file {Absolute path of model_zoo}/official/cv/alexnet/src/alexnet.py(175)/        x = self.conv3(x)/
    ```

Through the operator name and input and output information, you can find the only corresponding Tensor data file. For example, if you want to view the dump file corresponding to the first output data of the Conv2D-op12 operator, you can obtain the following information:

- `operator_name`: `Conv2D-op12`.

- `input_output_index`: `output.0` indicates that the file is the data of the first output Tensor of the operator.

- `slot`: 0, this tensor only has one slot.

Search for the corresponding file name in the data object file directory saved by Dump:
`Conv2d.Conv2D-op12.0.0.1623124369613540.output.0.DefaultFormat.npy`.

When restoring data, execute:

```python
import numpy
numpy.load("Conv2D.Conv2D-op12.0.0.1623124369613540.output.0.DefaultFormat.npy")
```

Generate the numpy.array data.

## Notices

- When an operator of type `bfloat16` is saved to the `npy` file, it will be converted to type `float32`.
- Dump only supports saving data with type of bool, int, int8, in16, int32, int64, uint, uint8, uint16, uint32, uint64, float, float16, float32, float64, bfloat16, double, complex64 and complex128.
- Complex64 and complex128 only support saving as npy files, not as statistics information.
- The Print operator has an input parameter with type of string, which is not a data type supported by Dump. Therefore, when the Print operator is included in the script, there will be an error log, which will not affect the saving data of other types.
- When Ascend O2 dump is enabled, lite exception dump is not supported by using set_context(ascend_config={"exception_dump": "2"}), while full exception dump is supported by using set_context(ascend_config={"exception_dump": "1"}).
- When Ascend O2 dump is enabled, sink size can only be set to 1. User can use model.train() and ms.data_sink() to set up sink size.
- When Ascend O2 dump is enabled, if **statistical value dumping** is performed in scenarios with a large amount of data (such as when the network itself is of a large scale or multiple steps are dumped consecutively), it may cause the host-side memory to become full, leading to a failure in data flow synchronization. It is recommended to replace it with the new version of [**statistical value dumping**](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md#51-%E9%9D%99%E6%80%81%E5%9B%BE%E5%9C%BA%E6%99%AF).
- By default, Dump ignores invalid operator outputs, such as the outputs of the Send/Print operator or the third reserved output of the FlashAttentionScore operator. If you need to retain these invalid outputs, you can set the environment variable `MINDSPORE_DUMP_IGNORE_USELESS_OUTPUT` to `0`. For details, please refer to [Environment Variables - Dump Debugging](https://www.mindspore.cn/docs/en/master/api_python/env_var_list.html#dump-debugging).
