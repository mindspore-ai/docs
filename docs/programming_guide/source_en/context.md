# Running Management

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/programming_guide/source_en/context.md)

## Overview

Before initializing the network, configure the context parameter to control the policy executed by the program. For example, you can select an execution mode and backend, and configure distributed parameters. Different context parameter configurations implement different functions, including execution mode management, hardware management, distributed management, and maintenance and test management.

## Execution Mode Management

MindSpore supports two running modes: PyNative and Graph.

- `PYNATIVE_MODE`: dynamic graph mode. In this mode, operators in the neural network are delivered and executed one by one, facilitating the compilation and debugging of the neural network model.

- `GRAPH_MODE`: static graph mode or graph mode. In this mode, the neural network model is compiled into an entire graph, and then the graph is delivered for execution. This mode uses graph optimization to improve the running performance and facilitates large-scale deployment and cross-platform running.

### Mode Selection

You can set and control the running mode of the program. By default, MindSpore is in PyNative mode.

A code example is as follows:

```python
from mindspore import context
context.set_context(mode=context.GRAPH_MODE)
```

### Mode Switching

You can switch between the two modes.

When MindSpore is in PyNative mode, you can switch it to the graph mode using `context.set_context(mode=context.GRAPH_MODE)`. Similarly, when MindSpore is in graph mode, you can switch it to PyNative mode using `context.set_context(mode=context.PYNATIVE_MODE)`.

A code example is as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

conv = nn.Conv2d(3, 4, 3, bias_init='zeros')
input_data = Tensor(np.ones([1, 3, 5, 5]).astype(np.float32))
conv(input_data)
context.set_context(mode=context.PYNATIVE_MODE)

conv(input_data)
```

```text
Tensor(shape=[1, 4, 5, 5], dtype=Float32, value=
[[[[ 1.64782144e-02,  5.31007685e-02,  5.31007685e-02,  5.31007685e-02,  5.11828624e-02],
   [ 3.00714076e-02,  6.57572001e-02,  6.57572001e-02,  6.57572001e-02,  4.35083285e-02],
   [ 3.00714076e-02,  6.57572001e-02,  6.57572001e-02,  6.57572001e-02,  4.35083285e-02]
   [ 3.00714076e-02,  6.57572001e-02,  6.57572001e-02,  6.57572001e-02,  4.35083285e-02],
   [ 1.84759758e-02,  4.71352898e-02,  4.71352898e-02,  4.71352898e-02,  3.72093469e-02]],
  [[-3.36203352e-02, -6.12429380e-02, -6.12429380e-02, -6.12429380e-02, -4.33492810e-02],
   [-2.67659649e-02, -8.04031491e-02, -8.04031491e-02, -8.04031491e-02, -6.84653893e-02],
   [-2.67659649e-02, -8.04031491e-02, -8.04031491e-02, -8.04031491e-02, -6.84653893e-02]
   [-2.67659649e-02, -8.04031491e-02, -8.04031491e-02, -8.04031491e-02, -6.84653893e-02],
   [-5.57974726e-03, -6.80863336e-02, -6.80863336e-02, -6.80863336e-02, -8.38923305e-02]],
  [[-1.60222687e-02,  2.26615220e-02,  2.26615220e-02,  2.26615220e-02,  6.03060052e-02],
   [-6.76476881e-02, -2.96694487e-02, -2.96694487e-02, -2.96694487e-02,  4.86185402e-02],
   [-6.76476881e-02, -2.96694487e-02, -2.96694487e-02, -2.96694487e-02,  4.86185402e-02]
   [-6.76476881e-02, -2.96694487e-02, -2.96694487e-02, -2.96694487e-02,  4.86185402e-02],
   [-6.52819276e-02, -3.50066647e-02, -3.50066647e-02, -3.50066647e-02,  2.85858363e-02]]
  [[-3.10218725e-02, -3.84682454e-02, -3.84682454e-02, -3.84682454e-02, -8.58424231e-03],
   [-4.27014455e-02, -7.07850009e-02, -7.07850009e-02, -7.07850009e-02, -5.36267459e-02],
   [-4.27014455e-02, -7.07850009e-02, -7.07850009e-02, -7.07850009e-02, -5.36267459e-02]
   [-4.27014455e-02, -7.07850009e-02, -7.07850009e-02, -7.07850009e-02, -5.36267459e-02],
   [-1.23060495e-02, -4.99926135e-02, -4.99926135e-02, -4.99926135e-02, -4.71802950e-02]]]])
```

In the preceding example, the running mode is set to `GRAPH_MODE` and then switched to `PYNATIVE_MODE`.

> This code is applicable to GPU environment.

## Hardware Management

Hardware management involves the `device_target` and `device_id` parameters.

- `device_target`: sets the target device. Ascend, GPU, and CPU are supported. Set this parameter based on the actual requirements.

- `device_id`: specifies the physical sequence number of a device, that is, the actual sequence number of the device on the corresponding host. If the target device is Ascend and the specification is N*Ascend (N > 1, for example, 8*Ascend), in non-distributed mode, you can set `device_id` to determine the device ID for program execution to avoid device usage conflicts. The value ranges from 0 to the total number of devices minus 1. The total number of devices cannot exceed 4096. The default value is 0.

> On the GPU and CPU, the `device_id` parameter setting is invalid.

A code example is as follows:

```python
from mindspore import context
context.set_context(device_target="Ascend", device_id=6)
```

## Distributed Management

The context contains the context.set_auto_parallel_context API that is used to configure parallel training parameters. This API must be called before the network is initialized.

> For details about distributed management, see [Parallel Distributed Training](https://www.mindspore.cn/doc/programming_guide/en/r1.2/auto_parallel.html).

## Maintenance and Test Management

To facilitate maintenance and fault locating, the context provides a large number of maintenance and test parameter configurations, such as profiling data collection, asynchronous data dump function, and print operator disk flushing.

### Profiling Data Collection

The system can collect profiling data during training and use the profiling tool for performance analysis. Currently, the following profiling data can be collected:

- `enable_profiling`: indicates whether to enable the profiling function. If this parameter is set to True, the profiling function is enabled, and profiling options are read from `enable_options`. If this parameter is set to False, the profiling function is disabled and only training_trace is collected.

- `profiling_options`: profiling collection options. The values are as follows. Multiple data items can be collected.

    - `result_path`: saving the path of the profiling collection result file. The directory spectified by this parameter needs to be created in advance on the training environment (container or host side) and ensure that the running user configured during installation has read and write permissions. It supports the configuration of absolute or relative paths(relative to the current path when executing the command line). The absolute path configuration starts with '/', for example:/home/data/output. The relative path configuration directly starts with the directory name, for example:output;

    - `training_trace`: collect iterative trajectory data, that is, the training task and software information of the AI software stack, to achieve performance analysis of the training task, focusing on data enhancement, forward and backward calculation, gradient aggregation update and other related data. The value is on/off;

    - `task_trace`: collect task trajectory data, that is, the hardware information of the HWTS/AICore of the Ascend 910 processor, and analyze the information of beginning and ending of the task. The value is on/off;

    - `aicpu_trace`: collect profiling data enhanced by aicpu data. The value is on/off;

    - `fp_point`: specify the start position of the forward operator of the training network iteration trajectory, which is used to record the start timestamp of the forward calculation. The configuration value is the name of the first operator specified in the forward direction. when the value is empty, the system will automatically obtain the forward operator name;

    - `bp_point`: specify the end position of the iteration trajectory reversal operator of the training network, record the end timestamp of the backward calculation. The configuration value is the name of the operator after the specified reverse. when the value is empty, the system will automatically obtain the backward operator name;

    - `ai_core_metrics` the values are as follows:

        - ArithmeticUtilization: percentage statistics of various calculation indicators;

        - PipeUtilization: the time-consuming ratio of calculation unit and handling unit, this item is the default value;

        - Memory: percentage of external memory read and write instructions;

        - MemoryL0: percentage of internal memory read and write instructions;

        - ResourceConflictRatio: proportion of pipline queue instructions.

A code example is as follows:

```python
from mindspore import context
context.set_context(enable_profiling=True, profiling_options= '{"result_path":"/home/data/output","training_trace":"on"}')
```

### Saving MindIR

Saving the intermediate code of each compilation stage through context.set_context(save_graphs=True).

The saved intermediate code has two formats: one is a text format with a suffix of `.ir`, and the other is a graphical format with a suffix of `.dot`.

When the network is large, it is recommended to use a more efficient text format for viewing. When the network is not large, it is recommended to use a more intuitive graphical format for viewing.

A code example is as follows:

```python
from mindspore import context
context.set_context(save_graphs=True)
```

> For a detailed introduction of MindIR, please refer to [MindSpore IR(MindIR)](https://www.mindspore.cn/doc/note/en/r1.2/design/mindspore/mindir.html).

### Print Operator Disk Flushing

By default, the MindSpore self-developed print operator can output the tensor or character string information entered by users. Multiple character string inputs, multiple tensor inputs, and hybrid inputs of character strings and tensors are supported. The input parameters are separated by commas (,).

> For details about the print function, see [MindSpore Print Operator](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/custom_debugging_info.html#mindspore-print-operator).

- `print_file_path`: saves the print operator data to a file and disables the screen printing function. If the file to be saved exists, a timestamp suffix is added to the file. Saving data to a file can solve the problem that the data displayed on the screen is lost when the data volume is large.

A code example is as follows:

```python
from mindspore import context
context.set_context(print_file_path="print.pb")
```

> For details about the context API, see [mindspore.context](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/mindspore.context.html).
