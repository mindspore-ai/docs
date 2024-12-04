# Using Dump in the Graph Mode

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/debug/dump.md)

The input and output of the operator can be saved for debugging through the data dump when the training result deviates from the expectation.

- For dynamic graph mode, the forward process can utilize Python's native execution capabilities, allowing users to view and record the corresponding inputs and outputs during the execution of the network script. The JIT and backward processes, which are part of graph compilation, can use Ascend O0/O1 functionality to save the input and output data of operators to disk files.

- For the static graph mode, MindSpore provides the Dump function to save the graph and the input and output data of the operator during model training to a disk file.

In different modes, the Dump features supported by MindSpore are not entirely the same, and the required configuration files and the generated data formats vary accordingly. Therefore, you need to select the corresponding Dump configuration based on the running mode:

- [Dump in Ascend O0/O1 Mode](#dump-in-ascend-o0o1-mode)
- [Dump in Ascend O2 Mode](#dump-in-ascend-o2-mode)
- [Dump in CPU/GPU mode](#dump-in-cpugpu-mode)

> - The differences between Ascend O0, O1, and O2 modes can be found in [the parameter jit_level of the set_context method](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_context.html).
>
> - Dumping constant data is only supported in CPU/GPU mode, while not supported in Ascend O0/O1/O2 mode.
>
> - In Ascend O2 mode, Dump supports both .npy and .bin file formats for data, while other modes only support the .npy file format for Dump data.
>
> - Currently, Dump does not support heterogeneous training, meaning it does not support CPU/Ascend mixed training or GPU/Ascend mixed training.

MindSpore supports different Dump functionalities under various modes, as shown in the following table:

<table align="center">
  <tr>
   <td colspan="2" align="center">Feature</td>
   <td align="center">Ascend O0/Ascend O1</td>
   <td align="center">Ascend O2</td>
   <td align="center">CPU/GPU</td>
  </tr>
  <tr>
   <td align="left">Full Dump</td>
   <td align="left">Full network data dump</td>
   <td align="left">Supported</td>
   <td align="left">Supported, but without full_name information</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td rowspan="2" align="left">Partial Data Dump</td>
   <td align="left">Statistics Dump</td>
   <td align="left">Supports both host and device modes<sup>1</sup></td>
   <td align="left">Supports only host mode</td>
   <td align="left">Not Supported On CPU, GPU Supports only host mode</td>
  </tr>
  <tr>
   <td align="left">Data Sampling Dump</td>
   <td align="left">Supported</td>
   <td align="left">Not Supported</td>
   <td align="left">Not Supported</td>
  </tr>
  <tr>
   <td align="left">Overflow Dump</td>
   <td align="left">Dump overflow operators</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
   <td align="left">Not Supported</td>
  </tr>
  <tr>
   <td rowspan="5" align="left">Conditional Dump</td>
   <td align="left">Specify Operator Name</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td align="left">Specify Iteration</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td align="left">Specify Device</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td align="left">Specify File Format</td>
   <td align="left">Not Applicable</td>
   <td align="left">Supported</td>
   <td align="left">Not Applicable</td>
  </tr>
  <tr>
   <td align="left">set_dump</td>
   <td align="left">Supported</td>
   <td align="left">Not Supported</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td rowspan="2" align="left">Auxiliary Information Dump</td>
   <td align="left">Graph IR Dump</td>
   <td align="left">Supported</td>
   <td align="left">Not Supported</td>
   <td align="left">Supported</td>
  </tr>
  <tr>
   <td align="left">Execution Sequence Dump</td>
   <td align="left">Supported</td>
   <td align="left">Not Supported</td>
   <td align="left">Supported</td>
  </tr>
</table>

> 1. In terms of statistics, the computing speed of the device is faster than that of the host(currently only supported on Ascend backend), but the host has more statistical indicators than the device. Refer to the `statistic_category` option for details.

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
           1. List of operator names. Turn on the IR save switch `set_context(save_graphs=2)` and execute the network to obtain the operator name from the generated `trace_code_graph_{graph_id}`IR file. For details, please refer to [Saving IR](https://www.mindspore.cn/docs/en/master/model_train/debug/error_analysis/mindir.html#saving-ir).
           Note that whether setting `set_context(save_graphs=2)` may cause the different IDs of the same operator, so when dump specified operators, keep this setting unchanged after obtaining the operator name. Or you can obtain the operator names from the file `ms_output_trace_code_graph_{graph_id}.ir` saved by Dump. Refer to [Ascend O0/O1 Dump Data Object Directory](#introduction-to-data-object-directory-and-data-file).
           2. You can also specify an operator type. When there is no operator scope information or operator id information in the string, the background considers it as an operator type, such as "conv". The matching rule of operator type is: when the operator name contains an operator type string, the matching is considered successful (case insensitive). For example, "conv" can match operators "Conv2D-op1234" and "Conv3D-op1221".
           3. Regular expressions are supported. When the string conforms to the format of "name-regex(xxx)", it would be considered a regular expression. For example, "name-regex(Default/.+)" can match all operators with names starting with "Default/".
        - `support_device`: Supported devices, default setting is `[0,1,2,3,4,5,6,7]`. You can specify specific device ids to dump specific device data. This configuration parameter is invalid on the CPU, because there is no concept of device on the CPU, but it is still need to reserve this parameter in the json file.
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

        - `overflow_number`：Specify the number of data to overflow dump. This field is required only when `op_debug_mode` is set to 3 and only the overflow operator is saved. It can control the overflow data to be dumped in chronological order until the specified value is reached, and the overflow data will no longer be dumped. The default value is 0, which means dumping all overflow data.

    - `e2e_dump_settings`:

        - `enable`: When set to `true`, enable Synchronous Dump. When set to false or not set, Asynchronous Dump will be used on Ascend. The main difference between the two is that Asynchronous Dump has less impact on the original code execution order.
        - `trans_flag`: Enable trans flag. Transform the device data format into NCHW. If it is `true`, the data will be saved in the 4D format (NCHW) format on the Host side; if it is `false`, the data format on the Device side will be retained. Default: `true`.
        - `stat_calc_mode`: Select the backend for statistical calculations. Options are "host" and "device". Choosing "device" enables device computation of statistics, currently only effective on Ascend, and supports only min/max/avg/l2norm statistics. When `op_debug_mode` is set to 3, only `stat_calc_mode` set to "host" is supported.
        - `device_stat_precision_mode`(Optional): Precision mode of device statistics， and the value can be "high" or "low". When "high" is selected, avg/l2norm statistics will be calculated using float32, which will increase device memory usage and have higher precision; when "low" is selected, the same type as the original data will be used for calculation, which will occupy less device memory, but statistics overflow may be caused when processing large values. The default value is "high".
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

Only when `save_kernel_args` is `true`, `{op_type}.{op_name}.json` is generated and the params of the corresponding operators is saved.
When `save_kernel_args` is set to `true`, a JSON file named `{op_type}.{op_name}.json` will be generated, which saves the initialization information of the operator. The internal format of this JSON file contains the corresponding values of each initialization parameter of the operator. For example, for the `Matmul` operator, the JSON information would look like this:

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

The suffix of the graph execution history file is `.csv`. The file naming format is:

```text
ms_global_execution_order_graph_{graph_id}.csv
```

This file stores the list of iterations in which the graph was executed. After the graph is compiled, it may be split into multiple sub-graphs. Since sub-graphs share the same graph execution history with root graph, only root graph will generate an execution history file. This function is not supported on Ascend.

`.dump_metadata` records the original training information(the directory is not available for Ascend backend), and `data_dump.json` saves the dump configuration set by the user.

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

### Dump Step

1. Create configuration file:`data_dump.json`.

    The name and location of the JSON file can be customized.

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
            "statistic_category": ["max", "min", "l2norm"],
            "file_format": "npy"
        }
    }
    ```

    - `common_dump_settings`:

        - `op_debug_mode`: This attribute is used for operator overflow debugging. 0: disable overflow check function; 3: enable overflow check function; 4: enable the lightweight exception dump function. Set it to 0 when Dump data is processed. If it is not set to 0, only the data of the overflow operator or exception operator will be dumped.
        - `dump_mode`: 0: all operator data in the network dumped out; 1: dump kernels data in kernels list. When overflow detection is enabled, the setting of this field becomes invalid, and Dump only saves the data of the overflow node. Specified data dump is supported only when "dump_mode' is set to `0`.
        - `path`: The absolute path to save Dump data.
        - `net_name`: The customized net name: "ResNet50".
        - `iteration`: Specify the iterations to dump, type is string. Use "|" to separate the step data of different intervals to be saved. For example, "0 | 5-8 | 100-120" represents dump the data of the 1st, 6th to 9th, and 101st to 121st steps. If iteration set to "all", data of every iteration will be dumped. Specified iteration dump is supported only when "op_debug_mode" is set to `0`, not supported when when "op_debug_mode" is set to `3` or `4`. When Ascend O2 dump is enabled, sink size can only be set to 1.
        - `saved_data`: Specify what data is to be dumped, type is string. Use "tensor" to dump tensor data, use "statistic" to dump tensor statistics, use "full" to dump both tensor data and statistics. Default setting is "tensor". Dump in Ascend O2 Mode statistics dump is only supported when `file_format` is set to `npy`, using "statistic" or "full" when `file_format` is set to `bin` will result in exception. Statistic dump is only supported when "op_debug_mode" is set to `0`.
        - `input_output`: When set to 0, it means to Dump the operator's input and output; when set to 1, it means to Dump the operator's input; setting it to 2 means to Dump the output of the operator.
        - `kernels`: This item can be configured in two formats:
             1. List of operator names. Specifying operator needs to first set the environment variable for saving the graph file to save the graph, and then obtain the operator name from the saved graph file. Please refer to the documentation on Ascend Developer Zone [DUMP_GE_GRAPH](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/graphdevg/graphdevg_000050.html) , [DUMP_GRAPH_LEVEL](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/graphdevg/graphdevg_000051.html) and [DUMP_GRAPH_PATH](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/graphdevg/graphdevg_000052.html) for details about the environment variable for saving the graph file.
             2. Regular expressions of operator names. When the string conforms to the format of "name-regex(xxx)", it would be considered a regular expression. For example, "name-regex(Default/.+)" can match all operators with names starting with "Default/".
        - `support_device`: Supported devices, default setting is `[0,1,2,3,4,5,6,7]`. You can specify specific device ids to dump specific device data.
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

            This field is optional, with default values of ["max", "min", "l2norm"].

        - `file_format`: Dump file type. It can be either `npy` and `bin`. `npy`: data will be dumped in npy files as host format. `bin`: data will be dumped in protobuf file as device format and need to be transformed to parse using the provided data analysis tool. Please refer to [Ascend O2 Mode Dump Data Analysis Sample](#data-analysis-sample-1) for details. The default value is `bin`.
        - `overflow_number`：Specify the number of data to overflow dump. This field is required only when `op_debug_mode` is set to 3 and `file_format` is set to `npy`. It can control the overflow data to be dumped in chronological order until the specified value is reached, and the overflow data will no longer be dumped. The default value is 0, which means dumping all overflow data.

2. Set Dump environment variable.

    ```bash
    export MINDSPORE_DUMP_CONFIG=${Absolute path of data_dump.json}
    ```

    If the `path` field is not set or set to an empty string in the Dump configuration file, you also need to configure the environment variable `MS_DIAGNOSTIC_DATA_PATH`.

    ```bash
    export MS_DIAGNOSTIC_DATA_PATH=${yyy}
    ```

    Then "$MS_DIAGNOSTIC_DATA_PATH/debug_dump" is regarded as `path`. If the `path` field in configuration file is not empty, it is still used as the path to save Dump data.

    - Set the environment variables before executing the training script. Setting environment variables during training will not take effect.
    - Dump environment variables need to be configured before calling `mindspore.communication.init`.

3. Execute the training script to dump data.

    You can set `set_context(reserve_class_name_in_scope=False)` in your training script to avoid dump failure because of file name is too long.

4. Refer to [Ascend O2 Mode Dump Data Analysis Sample](#data-analysis-sample-1) to analyze the Dump data file.

> - If you need to dump all or part of the operator, you can modify the `dump_mode` option in the json configuration file to 0 or 1.
> - Due to the slow Dump speed, enabling Dump in large model scenarios can extend the communication interval between different cards, leading to communication operator timeouts. This issue can be resolved by adjusting the timeout duration for the communication operators. For the Ascend backend, you can set the HCCL_EXEC_TIMEOUT environment variable. For detailed instructions, please refer to the [Ascend CANN documentation](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/apiref/envvar/envref_07_0072.html).

### Introduction to Data Object Directory and Data File

When the graph compilation level is not O0 or O1, the Dump directory structure is as follows, where the main feature is the {step_id} directory, which represents user side training step id:

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
    acl_dump_{device_id}.json
```

- `path`: the absolute path set in the `data_dump.json` configuration file.
- `device_id`: the id of the device.
- `model_name`: the model name generated by MindSpore.
- `model_id`: the id of the model.
- `graph_id`: the id of the training graph.
- `iteration_id`: the iteration of the training.
- `op_type`: the type of the operator.
- `op_name`: the name of the operator.
- `task_id`: the id of the task, if unable to fetch the value, the default is set to 65535.
- `stream_id`: the id of the stream, if unable to fetch the value, the default is set to 65535.
- `timestamp`: the time stamp.
- `step_id`: user side training step id.

The `acl_damp_{device_id}.json` file in the {path} directory is an intermediate file generated by Ascend O2 Mode dump during interface calls, and generally does not need to be paid attention to.

The overflow file (file `Opdebug.Node_OpDebug.{task_id}.{stream_id}.{timestamp}`) is only saved when overflow dump is enabled and overflow is detected.

If set `file_format` to `npy`, the operator file will be saved as a npy format file, and the overflow file will be saved as a json format file. The file naming formats are:

```text
{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.{dtype}.npy
Opdebug.Node_OpDebug.{task_id}.{stream_id}.{timestamp}.output.0.json
```

If the length of the tensor file name defined according to the naming rules exceeds the OS file name length limit (usually 255 characters), the tensor file will be renamed to a string of random numbers. The mapping relationship will be written to the file 'mapping.csv' in the same directory.

If set `file_format` to `npy`, it can be loaded by `numpy.load`.

If not configured `file_format` or set `file_format` to `bin`, after the training is started, the original data file generated by Ascend O2 Mode Dump or overflow files generated by overflow detection are in protobuf format. They need to be parsed using the data analysis tool that comes with the HiSilicon Run package. For details, please refer to [How to view dump data files](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/developmenttools/devtool/atlasaccuracy_16_0078.html).

The data format on the Device side may be different from the definition in the calculation diagram on the Host side. The bin file data format of the Ascend O2 Mode dump is the Device side format. If you want to convert to the Host side format, you can refer to [How to convert dump data file format](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/developmenttools/devtool/atlasaccuracy_16_0077.html).

If the file is saved in `bin` format, the file naming format is:

```text
{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}
```

Take the Conv2D-op12 of AlexNet network as an example: `Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802`, where `Conv2D` is `{op_type}`, `Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12` is `{op_name}`, and `2` is `{task_id' }`, `7` is `{stream_id' }`, `161243956333802` is `{timestamp}`.

If ".", "/", "\", and spaces appear in `op_type` and `op_name`, they will be converted to underscores.

If setting `file_format` to `npy`, the naming convention of data files generated by Ascend O2 Mode dump is the same as those of Ascend O0/O1 dump. Please refer to [Introduction to Ascend O0/O1 Dump Data File](#introduction-to-data-object-directory-and-data-file). The overflow file generated by overflow detection is in the `json` format, and the content analysis of the overflow file can refer to the [Analyzing the Data File of an Overflow/Underflow Operator](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/infacldevg/aclcppdevg/aclcppdevg_000160.html) .

The `saved_data` option only takes effect when `file_format` is "npy". If `saved_data` is "statistic" or "full", tensor statistics will be dumped in `statistic.csv`. When `saved_data` is "tensor" or "full", full tensor data will be dumped in `{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy`. The format of the statistic file will be the same as that of Ascend O0/O1 dump. Please refer to [Introduction to Ascend O0/O1 Dump Data File](#introduction-to-data-object-directory-and-data-file).

The constant dump file, final execution graph file and execution order file naming rules generated by Ascend O2 Mode Dump are the same as that of Ascend O0/O1 Dump. You can refer to [Introduction to Ascend O0/O1 Dump Data File](#introduction-to-data-object-directory-and-data-file).

### Data Analysis Sample

Ascend O2 Mode dump does not automatically save `.ir` files. To view `.ir` files, you can use MindSpore IR save switch `set_comtext(save_graphs=2)` before executing the use case. After executing the use case, you can view the saved `tracecode_graph_ xxx}` file, which can be opened with `vi`. Please refer to the data analysis example of Ascend O0/O1 dump for the file viewing method. Since the `.ir` file is not the final execution graph, it cannot be guaranteed that the operator names in the operator file correspond one-to-one with those in the `.ir` file. Please refer to the documentation on Ascend Developer Zone [DUMP_GE_GRAPH](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/graphdevg/graphdevg_000050.html) , [DUMP_GRAPH_LEVEL](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/graphdevg/graphdevg_000051.html) and [DUMP_GRAPH_PATH](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/graphdevg/graphdevg_000052.html) to save the final execution graph.

In Ascend O2 mode, dump data files will be generated in the corresponding directory described above. The parsing of these data files can be done through the following three steps: If `file_format` in the Dump configure file is set to "npy", then the step 1, 2 in the follows steps can be skipped. If `file_format` is not set or set to "bin", the tensor files need to be converted to `.npy` format.

1. Parse the dumped file using `msaccucmp.py` provied in the run package, the path where the `msaccucmp.py` file is located may be different on different environments. You can find it through the `find` command:

    ```bash
    find ${run_path} -name "msaccucmp.py"
    ```

    - `run_path`: The installation path of the run package.

2. After finding the `msaccucmp.py`, go to the `/absolute_path` directory and run the following command to parse the Dump data:

    ```bash
    python ${The absolute path of msaccucmp.py} convert -d {file path of dump} -out {file path of output}
    ```

    The {file path of dump} can be path to a single `.bin` file, or the folder that include the `.bin` files.

    If you need to convert the data format, please refer to the user instructions link <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/developmenttools/devtool/atlasaccuracy_16_0077.html>.

    For example, the data file generated by Dump is:

    ```text
    Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802
    ```

    Then execute:

    ```bash
    python3.7.5 msaccucmp.py convert -d /path/to/Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802 -out ./output -f NCHW -t npy
    ```

    All input and output data for this operator can be generated under `./output`. Each data is saved as a file with the `.npy` suffix in the format `NCHW`. The result is as follows:

    ```text
    Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802.input.0.32x256x13x13.npy
    Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802.input.1.384x256x3x3.npy
    Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802.output.0.32x384x13x13.npy
    ```

    At the end of the file name, you can see which input or output the file is the operator, and the dimensional information of the data. For example, by the first `.npy` file name

    ```text
    Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802.input.0.32x256x13x13.npy
    ```

    It can be seen that the file is the 0th input of the operator, and the dimension information of the data is `32x256x13x13`.

3. The corresponding data can be read through `numpy.load("file_name")`. For example:

    ```python
    import numpy
    numpy.load("Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802.input.0.32x256x13x13.npy")
    ```

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
            "trans_flag": true,
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
             1. List of operator names. Turn on the IR save switch `set_context(save_graphs=2)` and execute the network to obtain the operator name from the generated `trace_code_graph_{graph_id}`IR file. For details, please refer to [Saving IR](https://www.mindspore.cn/docs/en/master/model_train/debug/error_analysis/mindir.html#saving-ir).
             Note that whether setting `set_context(save_graphs=2)` may cause the different IDs of the same operator, so when dump specified operators, keep this setting unchanged after obtaining the operator name. Or you can obtain the operator names from the file `ms_output_trace_code_graph_{graph_id}.ir` saved by Dump. Refer to [Ascend O0/O1 Dump Data Object Directory](#introduction-to-data-object-directory-and-data-file).
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

4. Read and parse dump data through `numpy.load`, refer to [Introduction to CPU/GPU Dump Data File](#introduction-to-data-object-directory-and-data-file-2).

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

## Other Description

### Other Dump Function

In some special scenarios, the GE dump mode can be applied under development guidance.

To enable GE dump, set the environment variable MINDSPORE_DUMP_CONFIG and ENABLE_MS_GE_DUMP to 1. This mode applies only to the scenario where the compilation level of the graph is O2. The format of the configuration file is the same as that of the Ascend O2 Dump configuration file. The op_debug_mode field cannot be set to 4. Other parameters are the same as those of the Ascend O2 Dump configuration file.

```bash
export ENABLE_MS_GE_DUMP=1
```

The GE Dump directory structure is as follows:

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

Among them, the meanings of `path`, `time`, `device_id`, `model_name`, `model_id`, `iteration_id`, `op_type`, `op_name`, `task_id`, `stream_id`, and `timestamp` are the same as those of Ascend O2 Dump.

This method will be abandoned in the future and is not recommended for use.

## Notices

- When an operator of type `bfloat16` is saved to the `npy` file, it will be converted to type `float32`.
- Dump only supports saving data with type of bool, int, int8, in16, int32, int64, uint, uint8, uint16, uint32, uint64, float, float16, float32, float64, bfloat16, double, complex64 and complex128.
- Complex64 and complex128 only support saving as npy files, not as statistics information.
- The Print operator has an input parameter with type of string, which is not a data type supported by Dump. Therefore, when the Print operator is included in the script, there will be an error log, which will not affect the saving data of other types.
- When Ascend O2 dump is enabled, lite exception dump is not supported by using set_context(ascend_config={"exception_dump": "2"}), while full exception dump is supported by using set_context(ascend_config={"exception_dump": "1"}).
- When Ascend O2 dump is enabled, sink size can only be set to 1. User can use model.train() and ms.data_sink() to set up sink size. Indtoduction of sink mode can be found at <https://www.mindspore.cn/docs/en/master/model_train/train_process/optimize/sink_mode.html> .