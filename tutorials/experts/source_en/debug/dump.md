# Using Dump in the Graph Mode

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/debug/dump.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

The input and output of the operator can be saved for debugging through the data dump when the training result deviates from the expectation.

- For the dynamic graph mode, MindSpore provides native Python execution capabilities. Users can view and record the corresponding input and output during the running of the network script.

- For the static graph mode, MindSpore provides the Dump function to save the graph and the input and output data of the operator during model training to a disk file.

Aiming at the static graph mode, this tutorial introduces how to analyze and compare network data based on the Dump function.

### Debugging Process

Using dump to help debugging is divided into two steps: 1. Data preparation; 2. Data analysis.

#### Data preparation

The data preparation phase uses synchronous dump or asynchronous dump to generate dump data. See [Synchronous Dump Step](#synchronous-dump-step) and [Asynchronous Dump Step](#asynchronous-dump-step) for details.

When preparing data, you can refer to the following best practices:

1. Set the `iteration` parameter to save only the data of the iteration with the problem and the previous iteration. For example, if the problem to be analyzed will appear in the 10th iteration (counting from 1), you can set it as follows: `"iteration": "8 | 9"`. Note that the `iteration` parameter evaluates iterations from 0. Saving the data of the above two iterations can help problem analysis under most scenarios.
2. After the iteration with problems is completed, it is recommended that you use [run_context.request_stop()](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.RunContext.html#mindspore.RunContext.request_stop) or other methods to stop the training.

#### Data analysis

If you have installed MindInsight, you can use offline debugger of MindInsight to analyze it. See [Using the Offline Debugger](https://www.mindspore.cn/mindinsight/docs/en/master/debugger_offline.html) for the usage of offline debugger.

If MindInsight is not installed, you need to analyze the data through the following steps.

1. Find the corresponding operator from the script.

    The Dump function needs to use the IR file of the final execution graph. The IR file can be viewed with the `vi` command. The IR file contains the full name of the operator, and the dependency of the operator on the input and output of the computational graph, and also contains the trace information from the operator to the corresponding script code. For the configuration of the Dump function, see [Synchronous Dump Step](#synchronous-dump-step) and [Asynchronous Dump Step](#asynchronous-dump-step). For the final implementation of the image IR file naming and directory structure, see [Synchronous Dump Data Object Directory](#synchronous-dump-data-object-directory) and [Asynchronous Dump Data Object Directory](#asynchronous-dump-data-object-directory). Then find the operator corresponding to the code in the script through the graph file, refer to [Synchronous Dump Data Analysis Sample](#synchronous-dump-data-analysis-sample) and [Asynchronous Dump Data Analysis Sample](#asynchronous-dump-data-analysis-sample).

2. From operator to dump data.

    After understanding the mapping relationship between the script and the operator, you can determine the name of the operator you want to analyze and find the dump file corresponding to the operator. Please refer to [Synchronous Dump Data Object Directory](#synchronous-dump-data-object-directory) and [Asynchronous Dump Data Object Directory](#asynchronous-dump-data-object-directory).

3. Analyze Dump data.

    By analyzing Dump data, it can be compared with other third-party frameworks. For the synchronous dump data format, please refer to [Introduction to Synchronous Dump Data File](#introduction-to-synchronous-dump-data-file). For the asynchronous Dump data format, please refer to [Introduction to Asynchronous Dump Data File](#introduction-to-asynchronous-dump-data-file).

### Applicable Scene

1. Analysis of static graph operator results.

   Through the IR diagram obtained by the Dump function, you can understand the mapping relationship between the script code and the execution operator (for details, see [MindSpore IR Introduction](https://www.mindspore.cn/tutorials/experts/en/master/debug/mindir.html#overview)). Combining the input and output data of the execution operator, it is possible to analyze possible overflow, gradient explosion and disappearance during the training process, and backtrack to the code that may have problems in the script.

2. Analysis of the feature map.

   Analyze the information of the feature map by obtaining the output data of the layer.

3. Model migration.

   In the scenario of migrating a model from a third-party framework (TensorFlow, PyTorch) to MindSpore, by comparing the output data of the same position operator, analyzing whether the training results of the third-party framework and MindSpore for the same model are close enough to locate the model Precision issues.

## Dump Introduction

MindSpore provides two modes: synchronous dump and asynchronous dump:

- The mechanism of synchronous dump is that after the execution of each step in the network training process, the host side initiates a dump action, copies the data in the operator address from the device to the host, and saves the file. Synchronous Dump will turn off memory reuse between operators by default to avoid reading dirty data.
- Asynchronous Dump is a function developed specifically for the sinking of the entire Ascend image. It can dump data while executing the operator. The data will be dumped immediately after the execution of an operator. Therefore, the correct data can be generated by turning on the memory reuse, but the corresponding network training speed will be slower.

The configuration files required for different modes and the data format of dump are different:

- When Dump is enabled on Ascend, the operator to Dump will automatically close memory reuse.
- Synchronous Dump supports the graphics mode both on Ascend, GPU and CPU, and currently does not support PyNative mode.
- Asynchronous Dump only supports graph mode on Ascend, not PyNative mode. Memory reuse will not be turned off when asynchronous dump is enabled.
- Default is Asynchronous mode. If synchronous mode is needed, "e2e_dump_settings" should be set in configure file.
- Dump does not support heterogeneous training. If Dump is enabled for heterogeneous training, the saved dump data object directory maybe not in expected directory structure.

## Synchronous Dump

### Synchronous Dump Step

1. Create dump json file:`data_dump.json`, the name and location of the JSON file can be customized.

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

    - `dump_mode`: 0: dump all kernels data in graph, 1: dump kernels data in kernels list.
    - `path`: The absolute path to save dump data.
    - `net_name`: The net name eg:ResNet50.
    - `iteration`: Specify the iterations to dump, type is string. Use "|" to separate the step data of different intervals to be saved. For example, "0 | 5-8 | 100-120" represents dump the data of the 1st, 6th to 9th, and 101st to 121st steps. If iteration set to "all", data of every iteration will be dumped.
    - `saved_data`: Specify what data is to be dumped, type is string. Use "tensor" to dump tensor data, use "statistic" to dump tensor statistics, use "full" to dump both tensor data and statistics. Default setting is "tensor". Synchronous statistics dump is only supported on GPU, using "statistic" or "full" on CPU or Ascend will result in exception.
    - `input_output`: 0: dump input and output of kernel, 1:dump input of kernel, 2:dump output of kernel. This configuration parameter only supports Ascend and CPU, and GPU can only dump the output of operator.
    - `kernels`: List of operator names. Turn on the IR save switch `set_context(save_graphs=True)` and execute the network to obtain the operator name from the generated `trace_code_graph_{graph_id}`IR file. For details, please refer to [Saving IR](https://www.mindspore.cn/tutorials/experts/en/master/debug/mindir.html#saving-ir).
    - `support_device`: Supported devices, default setting is `[0,1,2,3,4,5,6,7]`. You can specify specific device ids to dump specific device data. This configuration parameter is invalid on the CPU, because there is no concept of device on the CPU, but it is still need to reserve this parameter in the json file.
    - `enable`: When set to true, enable Synchronous Dump. When set to false, asynchronous dump will be used on Ascend and synchronous dump will still be used on GPU.
    - `trans_flag`: Enable trans flag. Transform the device data format into NCHW. If it is `True`, the data will be saved in the 4D format (NCHW) format on the Host side; if it is `False`, the data format on the Device side will be retained. This configuration parameter is invalid on the CPU, because there is no format conversion on the CPU, but it is still need to reserve this parameter in the json file.

2. Set Dump environment.

   Specify the json configuration file of Dump.

   ```bash
   export MINDSPORE_DUMP_CONFIG=${xxx}
   ```

   "xxx" represents the absolute path of data_dump.json

   ```bash
   export MINDSPORE_DUMP_CONFIG=/path/to/data_dump.json
   ```

   If the `path` field is not set or set to an empty string in the Dump configuration file, you also need to configure the environment variable `MS_DIAGNOSTIC_DATA_PATH`.

   ```bash
   export MS_DIAGNOSTIC_DATA_PATH=${yyy}
   ```

   Then "$MS_DIAGNOSTIC_DATA_PATH/debug_dump" is regarded as `path`. If the `path` field in configuration file is not empty, it is still used as the path to save Dump data.

    - Set the environment variables before executing the training script. Setting environment variables during training will not take effect.
    - Dump environment variables need to be configured before calling `mindspore.communication.init`.

3. Execute the training script to dump data.

   After the training is started, if the `MINDSPORE_DUMP_CONFIG` environment variable is correctly configured, the content of the configuration file will be read and the operator data will be saved according to the data storage path specified in the Dump configuration.
   In synchronous mode, if you want to dump data in GPU environment, you must use the non-data sink mode (set the `dataset_sink_mode` parameter in `model.train` or `DatasetHelper` to `False`) to ensure that you can get the dump data of each step.
   If `model.train` or `DatasetHelper` is not called in the script, the default is non-data sinking mode. Using the Dump function will automatically generate the IR file of the final execution graph.

    You can set `set_context(reserve_class_name_in_scope=False)` in your training script to avoid dump failure because of file name is too long.

4. Read and parse synchronous dump data through `numpy.load`, refer to [Introduction to Synchronous Dump Data File](#introduction-to-synchronous-dump-data-file).

### Synchronous Dump Data Object Directory

After starting the training, the data objects saved by the synchronous Dump include the final execution graph (`ms_output_trace_code_graph_{graph_id}.ir` file) and the input and output data of the operators in the graph. The data directory structure is as follows:

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

For multi graph networks, due to the control flow, some subgraphs may not be executed, but Dump only saves the executed nodes, so the {graph_id} in the `.pb` file name under the 'graphs' directory may not always have a corresponding {graph_id} directory in {net_name} directory.

Only when `saved_data` is "statistic" or "full", will tensor statistics be dumped in `statistic.csv`. Only when `saved_data` is "tensor" or "full", will full tensor data be dumped in `{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy`.

### Introduction to Synchronous Dump Data File

The data file generated by the synchronous Dump is a binary file with the suffix `.npy`, and the file naming format is:

```text
{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy
```

The constant data file generated by the synchronous Dump is in the same format as data file, whereas {op_type}, {task_id}, {stream_id}, {input_output_index}, {slot}, {format} are unchanged for all constant data. Note, non-Tensor type will not generate dump file.

```text
Parameter.data-{data_id}.0.0.{timestamp}.output.0.DefaultFormat.npy
```

User can use Numpy interface `numpy.load` to read the data.

The statistics file generated by the synchronous dump is named `statistic.csv`. This file stores key statistics for all tensors dumped under the same directory as itself (with the file names `{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy`). Each row in `statistic.csv` summarizes a single tensor, each row contains the statistics: Op Type, Op Name, Task ID, Stream ID, Timestamp, IO, Slot, Data Size, Data Type, Shape, Max Value, Min Value, Avg Value, Count, Negative Zero Count, Positive Zero Count, NaN Count, Negative Inf Count, Positive Inf Count, Zero Count. Note that opening this file with Excel may cause data to be displayed incorrectly. Please use commands like `vi` or `cat`, or use Excel to import csv from text for viewing.

The suffixes of the final execution graph files generated by synchronous Dump are `.pb` and `.ir` respectively, and the file naming format is:

```text
ms_output_trace_code_graph_{graph_id}.pb
ms_output_trace_code_graph_{graph_id}.ir
```

The files with the suffix `.ir` can be opened and viewed by the `vi` command.

The suffix of the node execution sequence file generated by the synchronous Dump is `.csv`, and the file naming format is:

```text
ms_execution_order_graph_{graph_id}.csv
```

The suffix of the graph execution history file is `.csv`. The file naming format is:

```text
ms_global_execution_order_graph_{graph_id}.csv
```

This file stores the list of iterations in which the graph was executed. After the graph is compiled, it may be split into multiple sub-graphs.
Since sub-graphs share the same graph execution history with root graph, only root graph will generate an execution history file.

`.dump_metadata` records the original training information, and `data_dump.json` saves the dump configuration set by the user.

### Synchronous Dump Data Analysis Sample

In order to better demonstrate the process of using dump to save and analyze data, we provide a set of [complete sample script] (https://gitee.com/mindspore/docs/tree/master/docs/sample_code/dump) , you only need to execute `bash dump_sync_dump.sh` for synchronous dump.

After the graph corresponding to the script is saved to the disk through the Dump function, the final execution graph file `ms_output_trace_code_graph_{graph_id}.ir` will be generated. This file saves the stack information of each operator in the corresponding graph, and records the generation script corresponding to the operator.

Take [AlexNet script](https://gitee.com/mindspore/docs/tree/master/docs/sample_code) as an example:

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
        self.max_pool2d = ops.MaxPool(kernel_size=3, strides=2)
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
numpy.load("Conv2d.Conv2D-op12.0.0.1623124369613540.output.0.DefaultFormat.npy")
```

Restore the data as `numpy.array' format.

## Asynchronous Dump

Large networks (such as Bert Large) will cause memory overflow when using synchronous dumps. MindSpore provides debugging capabilities for large networks through asynchronous dumps.

### Asynchronous Dump Step

1. Create dump json file:`data_dump.json`.

    The name and location of the JSON file can be customized.

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

    - `dump_mode`: 0: dump all kernels data in graph, 1: dump kernels data in kernels list, 2: dump the kernels data specified by `set_dump` in the scripts, see [mindspore.dump](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_dump.html) for the usage of `set_dump`.
    - `path`: The absolute path to save dump data. It is not a mandatory option and can be left unset or set to an empty string.
    - `net_name`: The net name eg:ResNet50.
    - `iteration`: Specify the iterations to dump, type is string. Use "|" to separate the step data of different intervals to be saved. For example, "0 | 5-8 | 100-120" represents dump the data of the 1st, 6th to 9th, and 101st to 121st steps. If iteration set to "all", data of every iteration will be dumped.
    - `saved_data`: Specify what data is to be dumped, type is string. Use "tensor" to dump tensor data, use "statistic" to dump tensor statistics, use "full" to dump both tensor data and statistics. Default setting is "tensor". Asynchronous statistics dump is only supported when `file_format` is set to `npy`, using "statistic" or "full" when `file_format` is set to `bin` will result in exception.
    - `input_output`: When set to 0, it means to Dump the operator's input and output; setting it to 1 means to Dump the operator's input; setting it to 2 means to Dump the output of the operator.
    - `kernels`: List of operator names. Turn on the IR save switch `set_context(save_graphs=True)` and execute the network to obtain the operator name from the generated `trace_code_graph_{graph_id}`IR file. `kernels` only supports TBE operator, AiCPU operator and communication operator. The data of communication operation input operator will be dumped if `kernels` is set to the name of communication operator. For details, please refer to [Saving IR](https://www.mindspore.cn/tutorials/experts/en/master/debug/mindir.html#saving-ir).
    - `support_device`: Supported devices, default setting is `[0,1,2,3,4,5,6,7]`. You can specify specific device ids to dump specific device data.
    - `enable`: Enable Asynchronous Dump. If synchronous dump and asynchronous dump are enabled at the same time, only synchronous dump will take effect.
    - `op_debug_mode`: Reserved field, set to 0.
    - `file_format`: Dump file type. It can be either `npy` and `bin`. `npy`: data will be dumped in npy files as host format. `bin`: data will be dumped in protobuf file as device format and need to be transformed to parse using the provided data analysis tool. Please refer to [Asynchronous Dump Data Analysis Sample](#asynchronous-dump-data-analysis-sample) for details. The default value is `bin`.

2. Set Dump environment.

   Specify the json configuration file of Dump.

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

4. Refer to [Asynchronous Dump Data Analysis Sample](#asynchronous-dump-data-analysis-sample) to analyze the Dump data file.

- If you need to dump all or part of the operator, you can modify the `dump_mode` option in the json configuration file to 0 or 1.
- Using the Dump function will automatically generate the IR file of the final execution graph.

### Asynchronous Dump Data Object Directory

If set `file_format` to `npy`, see [Synchronous Dump Data Object Directory](https://www.mindspore.cn/tutorials/experts/en/master/debug/dump.html#synchronous-dump-data-object-directory) for the dump data object directory.

The data objects saved by asynchronous Dump include the final execution graph (`ms_output_trace_code_graph_{graph_id}.ir` file) and the input and output data of the operators in the graph. The directory structure is as follows:

```text
{path}/
    - rank_{rank_id}/
        - .dump_metadata/
        - {net_name}/
            - {graph_id}/
                - {iteration_id}/
                    statistic.csv
                    {op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}
                    mapping.csv
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
- `data_id`: the id of constant data.

Due to the control flow, some sub-graphs may not be executed, but Dump only saves the executed nodes, so the {graph_id} in the `.pb` file name under the 'graphs' directory may not always have a corresponding {graph_id} directory in {net_name} directory.

For multi-graph networks, such as dynamic shape scenario, the iterations of all graphs on each device are counted uniformly.

If the length of the tensor file name defined according to the naming rules exceeds the OS file name length limit (usually 255 characters), the tensor file will be renamed to a string of random numbers. The mapping relationship will be written to the file 'mapping.csv' in the same directory.

### Introduction to Asynchronous Dump Data File

If set `file_format` to `npy`, see [Introduction to Synchronous Dump Data File](https://www.mindspore.cn/tutorials/experts/en/master/debug/dump.html#introduction-to-synchronous-dump-data-file) for the introduction to dump data file.

If not configured `file_format` or set `file_format` to `bin`, after the training is started, the original data file generated by asynchronous Dump is in protobuf format. It needs to be parsed using the data analysis tool that comes with the HiSilicon Run package. For details, please refer to [How to view dump data files](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/640e796d/how-do-i-view-a-dump-file).

The data format on the Device side may be different from the definition in the calculation diagram on the Host side. The data format of the asynchronous dump is the Device side format. If you want to convert to the Host side format, you can refer to [How to convert dump data file format](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/130949fb/how-do-i-convert-the-format-of-a-dump-file).

If the file is saved in `bin' format, the file naming format is:

```text
{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}
```

Take the Conv2D-op12 of AlexNet network as an example: `Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802`, where `Conv2D` is `{op_type}`, `Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12` is `{op_name}`, and `2` is `{task_id' }`, `7` is `{stream_id' }`, `161243956333802` is `{timestamp}`.

If ".", "/", "\", and spaces appear in `op_type` and `op_name`, they will be converted to underscores.

The original data file generated by dump can also be parsed by using the data parsing tool DumpParser of MindInsight. Please refer to [DumpParser Introduction](https://gitee.com/mindspore/mindinsight/blob/master/mindinsight/parser/README.md#) for the usage of DumpParser.
The data format parsed by MindInsight is exactly the same as that of synchronous dump.

If setting `file_format` to `npy`, the naming convention of data files generated by asynchronous dump is the same as those of synchronous dump. Please refer to [Introduction to Synchronous Dump Data File](#introduction-to-synchronous-dump-data-file).

The `saved_data` option only takes effect when `file_format` is "npy". If `saved_data` is "statistic" or "full", tensor statistics will be dumped in `statistic.csv`. When `saved_data` is "tensor" or "full", full tensor data will be dumped in `{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input_output_index}.{slot}.{format}.npy`. The format of the statistic file will be the same as that of synchonous dump. Please refer to [Introduction to Synchronous Dump Data File](#introduction-to-synchronous-dump-data-file).

The constant dump file, final execution graph file and execution order file naming rules generated by asynchronous Dump are the same as that of synchronous Dump. You can refer to [Introduction to Synchronous Dump Data File](#introduction-to-synchronous-dump-data-file).

### Asynchronous Dump Data Analysis Sample

In order to better demonstrate the process of using dump to save and analyze data, we provide a set of [complete sample script] (https://gitee.com/mindspore/docs/tree/master/docs/sample_code/dump) , you only need to execute `bash run_async_dump.sh` for asynchronous dump.

Through the asynchronous Dump function, the data files generated by the operator asynchronous Dump can be obtained. If `file_format` in the Dump configure file is set to "npy", then the step 1, 2 in the follows steps can be skipped. If `file_format` is not set or set to "bin", the tensor files need to be converted to `.npy` format.

1. Parse the dumped file using `msaccucmp.py` provied in the run package, the path where the `msaccucmp.py` file is located may be different on different environments You can find it through the find command:

    ```bash
    find ${run_path} -name "msaccucmp.py"
    ```

    - `run_path`: The installation path of the run package.

2. Change directory to `/absolute_path` after training, execute the following commands to parse Dump data file:

    ```bash
    python ${The  absolute path of msaccucmp.py} convert -d {file path of dump} -out {file path of output}
    ```

    The {file path of dump} can be path to a single `.bin` file, or the folder that include the `.bin` files.

    Or you can use `msaccucmp.py` to convert the format of dump file. Please see <https://support.huawei.com/enterprise/en/doc/EDOC1100206689/130949fb/how-do-i-convert-the-format-of-a-dump-file>.

    For example, the data file generated by Dump is:

    ```text
    Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802
    ```

    Then execute:

    ```bash
    python3.7.5 msaccucmp.py convert -d /path/to/Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3-Conv2d_Conv2D-op12.2.7.161243956333802 -out ./output -f NCHW -t npy
    ```

    Then all input and output data of the operator can be generated under `./output`. Each data is saved as a file with the suffix of `.npy`, and the data format is `NCHW`.

    The generated results are as follows:

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
