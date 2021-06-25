﻿# Using Dump in the Graph Mode

`Linux` `Ascend` `GPU` `CPU` `Model Optimization` `Intermediate` `Expert`

<!-- TOC -->

- [Using Dump in the Graph Mode](#using-dump-in-the-graph-mode)
    - [Overview](#overview)
        - [Debugging Process](#debugging-process)
        - [Applicable Scene](#applicable-scene)
    - [Dump Introduction](#dump-introduction)
    - [Synchronous Dump](#synchronous-dump)
        - [Synchronous Dump Step](#synchronous-dump-step)
        - [Synchronous Dump Data Object Directory](#synchronous-dump-data-object-directory)
        - [Introduction to Synchronous Dump Data File](#introduction-to-synchronous-dump-data-file)
        - [Synchronous Dump Data Analysis Sample](#synchronous-dump-data-analysis-sample)
    - [Asynchronous Dump](#asynchronous-dump)
        - [Asynchronous Dump Step](#asynchronous-dump-step)
        - [Asynchronous Dump Data Object Directory](#asynchronous-dump-data-object-directory)
        - [Introduction to Asynchronous Dump Data File](#introduction-to-asynchronous-dump-data-file)
        - [Asynchronous Dump Data Analysis Sample](#asynchronous-dump-data-analysis-sample)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/training/source_en/advanced_use/dump_in_graph_mode.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.2/resource/_static/logo_source.png"></a>

## Overview

The input and output of the operator can be saved for debugging through the data dump when the training result deviates from the expectation.

- For the dynamic graph mode, MindSpore provides native Python execution capabilities. Users can view and record the corresponding input and output during the running of the network script. For details, see [Use PyNative Mode to Debug](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/debug_in_pynative_mode.html).

- For the static graph mode, MindSpore provides the Dump function to save the graph and the input and output data of the operator during model training to a disk file.

Aiming at the static graph mode, this tutorial introduces how to analyze and compare network data based on the Dump function.

### Debugging Process

1. Find the corresponding operator from the script.

    The Dump function needs to use the IR file of the final execution graph. The IR file can be viewed with the `vi` command. The IR file contains the full name of the operator, and the dependency of the operator on the input and output of the computational graph, and also contains the trace information from the operator to the corresponding script code. For the configuration of the Dump function, see [Synchronous Dump Step](#synchronous-dump-step) and [Asynchronous Dump Step](#asynchronous-dump-step). For the final implementation of the image IR file naming and directory structure, see [Synchronous Dump Data Object Directory](#synchronous-dump-data-object-directory) and [Asynchronous Dump Data Object Directory](#asynchronous-dump-data-object-directory). Then find the operator corresponding to the code in the script through the graph file, refer to [Synchronous Dump Data Analysis Sample](#synchronous-dump-data-analysis-sample) and [Asynchronous Dump Data Analysis Sample](#asynchronous-dump-data-analysis-sample).

2. From operator to dump data.

    After understanding the mapping relationship between the script and the operator, you can determine the name of the operator you want to analyze and find the dump file corresponding to the operator. Please refer to [Synchronous Dump Data Object Directory](#synchronous-dump-data-object-directory) and [Asynchronous Dump Data Object Directory](#asynchronous-dump-data-object-directory).

3. Analyze Dump data.

    By analyzing Dump data, it can be compared with other third-party frameworks. For the synchronous dump data format, please refer to [Introduction to Synchronous Dump Data File](#introduction-to-synchronous-dump-data-file). For the asynchronous Dump data format, please refer to [Introduction to Asynchronous Dump Data File](#introduction-to-asynchronous-dump-data-file).

### Applicable Scene

1. Analysis of static graph operator results.

   Through the IR diagram obtained by the Dump function, you can understand the mapping relationship between the script code and the execution operator (for details, see [MindSpore IR Introduction](https://www.mindspore.cn/doc/note/en/r1.2/design/mindspore/mindir.html#overview)). Combining the input and output data of the execution operator, it is possible to analyze possible overflow, gradient explosion and disappearance during the training process, and backtrack to the code that may have problems in the script.

2. Analysis of the feature map.

   Analyze the information of the feature map by obtaining the output data of the layer.

3. Model migration.

   In the scenario of migrating a model from a third-party framework (TensorFlow, PyTorch) to MindSpore, by comparing the output data of the same position operator, analyzing whether the training results of the third-party framework and MindSpore for the same model are close enough to locate the model Precision issues.

## Dump Introduction

MindSpore provides two modes: synchronous dump and asynchronous dump:

- The mechanism of synchronous dump is that after the execution of each step in the network training process, the host side initiates a dump action, copies the data in the operator address from the device to the host, and saves the file. Synchronous Dump will turn off memory reuse between operators by default to avoid reading dirty data.
- Asynchronous Dump is a function developed specifically for the sinking of the entire Ascend image. It can dump data while executing the operator. The data will be dumped immediately after the execution of an operator. Therefore, the correct data can be generated by turning on the memory reuse, but the corresponding network training speed will be slower.

The configuration files required for different modes and the data format of dump are different:

- Synchronous mode takes up more memory than asynchronous mode, but it is easier to use.
- Generally, for small and medium-sized networks (such as ResNet), it is recommended to use the synchronous dump mode first. When the network does not occupy much memory, please use synchronous dump first.If an error of insufficient device memory occurs after enabling synchronous dump, please use asynchronous dump in the next section.
- When Dump is enabled on Ascend, the operator to Dump will automatically close memory reuse.
- Synchronous Dump supports the graphics mode both on Ascend, GPU and CPU, and currently does not support PyNative mode.
- Asynchronous Dump only supports graph mode on Ascend, not PyNative mode. Memory reuse will not be turned off when asynchronous dump is enabled.

## Synchronous Dump

### Synchronous Dump Step

1. Create dump json file:`data_dump.json`, the name and location of the JSON file can be customized.

    ```json
    {
        "common_dump_settings": {
            "dump_mode": 0,
            "path": "/absolute_path",
            "net_name": "ResNet50",
            "iteration": 0,
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

    - `dump_mode`: 0: dump all kernels in graph, 1: dump kernels in kernels list.
    - `path`: The absolute path to save dump data.
    - `net_name`: The net name eg:ResNet50.
    - `iteration`: Specify the iterations to dump. Iteration should be set to 0 when dataset_sink_mode is False and data of every iteration will be dumped.
    - `input_output`: 0: dump input and output of kernel, 1:dump input of kernel, 2:dump output of kernel. This parameter does not take effect on the GPU and only the output of operator will be dumped.  This configuration parameter only supports Ascend and CPU, and GPU can only dump the output of operator.
    - `kernels`: List of operator names. Turn on the IR save switch `context.set_context(save_graphs=True)` and execute the network to obtain the operator name from the generated `trace_code_graph_{graph_id}`IR file. For details, please refer to [Saving IR](https://www.mindspore.cn/doc/note/en/r1.2/design/mindspore/mindir.html#saving-ir).
    - `support_device`: Supported devices, default setting is `[0,1,2,3,4,5,6,7]`. You can specify specific device ids to dump specific device data.  This configuration parameter is invalid on the CPU, because there is no concept of device on the CPU.
    - `enable`: Enable Asynchronous Dump. If synchronous dump and asynchronous dump are enabled at the same time, only synchronous dump will take effect.
    - `trans_flag`: Enable trans flag. Transform the device data format into NCHW. If it is `True`, the data will be saved in the 4D format (NCHW) format on the Host side; if it is `False`, the data format on the Device side will be retained.  This configuration parameter is invalid on the CPU, because there is no format conversion on the CPU.

2. Specify the json configuration file of Dump.

   ```bash
   export MINDSPORE_DUMP_CONFIG=${xxx}
   ```

   "xxx" represents the absolute path of data_dump.json

   ```bash
   export MINDSPORE_DUMP_CONFIG=/path/to/data_dump.json
   ```

    - Set the environment variables before executing the training script. Setting environment variables during training will not take effect.
    - Dump environment variables need to be configured before calling `mindspore.communication.management.init`.

3. Execute the training script to dump data.

   After the training is started, if the `MINDSPORE_DUMP_CONFIG` environment variable is correctly configured, the content of the configuration file will be read and the operator data will be saved according to the data storage path specified in the Dump configuration.
   In synchronous mode, if you want to dump data, you must use the non-data sink mode (set the `dataset_sink_mode` parameter in `model.train` or `DatasetHelper` to `False`) to ensure that you can get the dump data of each step.
   If `model.train` or `DatasetHelper` is not called in the script, the default is non-data sinking mode. Using the Dump function will automatically generate the IR file of the final execution graph.

    You can set `context.set_context(reserve_class_name_in_scope=False)` in your training script to avoid dump failure because of file name is too long.

4. Read and parse synchronous dump data through `numpy.fromfile`, refer to [Introduction to Synchronous Dump Data File](#introduction-to-synchronous-dump-data-file).

### Synchronous Dump Data Object Directory

After starting the training, the data objects saved by the synchronous Dump include the final execution graph (`ms_output_trace_code_graph_{graph_id}.ir` file) and the input and output data of the operators in the graph. The data directory structure is as follows:

```text
{path}/
    |-- {net_name}/
        |-- device_{device_id}/
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

- `path`: the absolute path set in the `data_dump.json` configuration file.
- `net_name`: the network name set in the `data_dump.json` configuration file.
- `device_id`: the id of the training device.
- `graph_id`: the id of the training graph.
- `iteration`: the iteration of the training.
- `operator_name`: the name of the operator.
- `input_output_index` : the index of input or output. For example, `output_0` means that the file is the data of the first output Tensor of the operator.
- `shape`: Tensor dimension information.
- `data_type`: the type of the data.
- `format`: the format of the data.

When data dump is performed on the CPU, there is no directory level of `device_id`, because there is no concept of device on the CPU, and there are no `graphs`, `execution_order` and `.metadata` directories.

### Introduction to Synchronous Dump Data File

The data file generated by the synchronous Dump is a binary file with the suffix `.bin`, and the file naming format is:

```text
{operator_name}_{input_output_index}_{shape}_{data_type}_{format}.bin
```

According to the `Tensor` information provided by the file name, you can use `numpy.fromfile` to read the data and restore the `data_type` and `shape` of the original data.

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

`.metadata` records the original training information, and `data_dump.json` saves the dump configuration set by the user.

### Synchronous Dump Data Analysis Sample

For the Ascend scene, after the graph corresponding to the script is saved to the disk through the Dump function, the final execution graph file `ms_output_trace_code_graph_{graph_id}.ir` will be generated. This file saves the stack information of each operator in the corresponding graph, and records the generation script corresponding to the operator.

Take [AlexNet script](https://gitee.com/mindspore/mindspore/blob/r1.2/model_zoo/official/cv/alexnet/src/alexnet.py) as an example:

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

If the user wants to view the code at line 58 in the script:

```python
x = self.conv3(x)
```

After executing the network training, you can find multiple operator information corresponding to the line of code from the final execution graph (`ms_output_trace_code_graph_{graph_id}.ir` file). The content of the file is as follows:

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

The meanings of the lines in the file content shown above are as follows:

- The input and output of the operator on the Host side (the first line) and the Device side (the second line, some operators may not exist). It can be seen from the execution graph that the operator has two inputs (left side of the arrow) and one output (right side of the arrow).

    ```text
    : (<Tensor[Float32]x[const vector][32, 128, 13, 13]>, <Tensor[Float16]x[const vector][192, 128, 3, 3]>) -> (<Tensor[Float16]x[const vector][32, 192, 13, 13]>)
    : (<Float16xNC1HWC0[const vector][32, 8, 13, 13, 16]>, <Float16xFracZ[const vector][72, 12, 16, 16]>) -> (<Float16xNC1HWC0[const vector][32, 12, 13, 13, 16]>)
    ```

- Operator name. It can be seen from the execution graph that the full name of the operator in the final execution graph is `Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op107`.

    ```text
    : (Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op107)
    ```

- The training script code corresponding to the operator. By searching the training script code to be queried, multiple matching operators can be found.

    ```text
    # In file {Absolute path of model_zoo}/official/cv/alexnet/src/alexnet.py(58)/        x = self.conv3(x)/
    ```

Through the operator name and input and output information, you can find the only corresponding Tensor data file. For example, if you want to view the dump file corresponding to the first output data of the Conv2D-op107 operator, you can obtain the following information:

- `operator_name`: `Default--network-WithLossCell--_backbone-AlexNet--conv3-Conv2d--Conv2D-op107`. Based on the operator name declared in sequence number 2 in the graph, replace `/` with `--` to get it.

- `input_output_index`: `output_0` indicates that the file is the data of the first output Tensor of the operator.

Search for the corresponding file name in the data object file directory saved by Dump:
`Default--network-WithLossCell--_backbone-AlexNet--conv3-Conv2d--Conv2D-op107_output_0_shape_32_12_13_13_16_Float16_NC1HWC0.bin`.

The following information can be obtained from the file name:

- `shape`: The tensor dimension is `32_12_13_13_16`.

- `data_type`: The data type is `Float16`.

- `format`: The data format is `NC1HWC0` (the data format to be saved can be modified through the Dump configuration file).

When restoring data, first execute:

```python
import numpy
numpy.fromfile("Default--network-WithLossCell--_backbone-AlexNet--conv3-Conv2d--Conv2D-op107_output_0_shape_32_12_13_13_16_Float16_NC1HWC0.bin", numpy.float16)
```

One-dimensional array data is generated, and then execute:

```python
import numpy
numpy.reshape(array, (32,12,13,13,16))
```

Restore to the original shape data.

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
            "iteration": 0,
            "input_output": 0,
            "kernels": ["Default/Conv-op12"],
            "support_device": [0,1,2,3,4,5,6,7]
        },
        "async_dump_settings": {
            "enable": true,
            "op_debug_mode": 0
        }
    }
    ```

    - `dump_mode`: 0: dump all kernels in graph, 1: dump kernels in kernels list.
    - `path`: The absolute path to save dump data.
    - `net_name`: The net name eg:ResNet50.
    - `iteration`: Specify the iterations to dump. Iteration should be set to 0 when dataset_sink_mode is False and data of every iteration will be dumped.
    - `input_output`: When set to 0, it means to Dump the operator's input and output; setting it to 1 means to Dump the operator's input; setting it to 2 means to Dump the output of the operator.
    - `kernels`: List of operator names. Turn on the IR save switch `context.set_context(save_graphs=True)` and execute the network to obtain the operator name from the generated `trace_code_graph_{graph_id}`IR file. `kernels` only supports TBE operator, AiCPU operator and communication operator. The data of communication operation input operator will be dumped if `kernels` is set to the name of communication operator. For details, please refer to [Saving IR](https://www.mindspore.cn/doc/note/en/r1.2/design/mindspore/mindir.html#saving-ir).
    - `support_device`: Supported devices, default setting is `[0,1,2,3,4,5,6,7]`. You can specify specific device ids to dump specific device data.
    - `enable`: Enable Asynchronous Dump. If synchronous dump and asynchronous dump are enabled at the same time, only synchronous dump will take effect.
    - `op_debug_mode`: 0: disable overflow check function; 1: enable AiCore overflow check; 2: enable Atomic overflow check; 3: enable all overflow check function. If it is not set to 0, only the data of the overflow operator will be dumped.

2. Specify the json configuration file of Dump.

    ```bash
    export MINDSPORE_DUMP_CONFIG={Absolute path of data_dump.json}
    ```

    - Set the environment variables before executing the training script. Setting environment variables during training will not take effect.
    - Dump environment variables need to be configured before calling `mindspore.communication.management.init`.

3. Execute the training script to dump data.

    You can set `context.set_context(reserve_class_name_in_scope=False)` in your training script to avoid dump failure because of file name is too long.

4. Refer to [Asynchronous Dump Data Analysis Sample](#asynchronous-dump-data-analysis-sample) to analyze the Dump data file.

- If you need to dump all or part of the operator, you can modify the `dump_mode` option in the json configuration file to 0 or 1.
- If the data sink function is enabled (set the `dataset_sink_mode` parameter in `model.train` or `DatasetHelper` to `True`), only the data of one step specified in the configuration file can be dumped (in this case, `iteration 0` means The 0th step), and save it to the specified directory.
- If the data sink function is not enabled (set the `dataset_sink_mode` parameter in `model.train` or `DatasetHelper` to `False`), `iteration` in the configuration file must be specified as 0, and all step data are stored in a directory In, cannot support multi-step data management. At this time, it is recommended to execute the step data dump only once (you can train only one step by modifying the script).
- Using the Dump function will automatically generate the IR file of the final execution graph.

### Asynchronous Dump Data Object Directory

The data objects saved by asynchronous Dump include the final execution graph (`ms_output_trace_code_graph_{graph_id}.ir` file) and the input and output data of the operators in the graph. The directory structure is as follows:

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

- `path`: the absolute path set in the `data_dump.json` configuration file.
- `net_name`: the network name set in the `data_dump.json` configuration file.
- `device_id`: the id of the training device.
- `graph_id`: the id of the training graph.
- `iteration`: the iteration of the training.
- `op_type`: the type of the operator.
- `op_name`: the name of the operator.
- `taskid`: the id of the task.
- `timestamp`: the time stamp.

### Introduction to Asynchronous Dump Data File

After the training is started, the original data file generated by asynchronous Dump is in protobuf format. It needs to be parsed using the data analysis tool that comes with the HiSilicon Run package. For details, please refer to [How to view dump data files](https://support.huaweicloud.com/intl/en-us/usermanual-mindstudioc73/atlasmindstudioaccuracy_16_0022.html).

The data format on the Device side may be different from the definition in the calculation diagram on the Host side. The data format of the asynchronous dump is the Device side format. If you want to convert to the Host side format, you can refer to [How to convert dump data file format](https://support.huaweicloud.com/intl/en-us/usermanual-mindstudioc73/atlasmindstudioaccuracy_16_0021.html).

The naming rules for data files generated by asynchronous Dump are as follows:

- The naming rule of the dump path is: `{path}/{device_id}/{net_name}_graph_{graph_id}/{graph_id}/{iteration}`.
- The naming rule of Dump file is: `{op_type}.{op_name}.{task_id}.{timestamp}`.

Take the Dump result of a simple network as an example: `Add.Default_Add-op1.2.161243956333802`, where `Add` is `{op_type}`, `Default_Add-op1` is `{op_name}`, and `2` is `{task_id' }`, `161243956333802` is `{timestamp}`.

If ".", "/", "\", and spaces appear in `op_type` and `op_name`, they will be converted to underscores.

The final execution graph file and node execution sequence file naming rules generated by asynchronous Dump are the same as that of synchronous Dump. You can refer to [Introduction to Synchronous Dump Data File](#introduction-to-synchronous-dump-data-file).

### Asynchronous Dump Data Analysis Sample

Through the asynchronous Dump function, the data files generated by the operator asynchronous Dump can be obtained.

1. Parse the dumped file using `msaccucmp.py` provied in the run package, the path where the `msaccucmp.py` file is located may be different on different environments You can find it through the find command:

    ```bash
    find ${run_path} -name "msaccucmp.py"
    ```

    - `run_path`: The installation path of the run package.

2. Change directory to `/absolute_path` after training, execute the following commands to parse Dump data file:

    ```bash
    python ${The  absolute path of msaccucmp.py} convert -d {file path of dump} -out {file path of output}
    ```

    Or you can use `msaccucmp.py` to convert the format of dump file. Please see <https://support.huawei.com/enterprise/zh/doc/EDOC1100191946/fa6aecce>.

    For example, the data file generated by Dump is:

    ```text
    BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491
    ```

    Then execute:

    ```bash
    python3.7.5 msaccucmp.py convert -d BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491 -out ./output -f NCHW -t npy
    ```

    Then all input and output data of the operator can be generated under `./output`. Each data is saved as a file with the suffix of `.npy`, and the data format is `NCHW`.

    The generated results are as follows:

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

    At the end of the file name, you can see which input or output the file is the operator, and the dimensional information of the data. For example, by the first `.npy` file name

    ```text
    BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.input.0.30x1024x17x17.npy
    ```

    It can be seen that the file is the 0th input of the operator, and the dimension information of the data is `30x1024x17x17`.

3. The corresponding data can be read through `numpy.load("file_name")`. For example:

    ```python
    import numpy
    numpy.load("BNTrainingUpdate.Default_network-YoloWithLossCell_yolo_network-YOLOV3DarkNet53_feature_map-YOLOv3_backblock0-YoloBlock_conv3-SequentialCell_1-BatchNorm2d_BNTrainingUpdate-op5489.137.1608983934774491.input.0.30x1024x17x17.npy")
    ```
