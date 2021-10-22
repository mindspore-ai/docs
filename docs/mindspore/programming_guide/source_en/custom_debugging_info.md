# Custom Debugging Information

`Ascend` `Model Optimization`

<!-- TOC -->

- [Custom Debugging Information](#custom-debugging-information)
    - [Overview](#overview)
    - [Introduction to Callback](#introduction-to-callback)
        - [Callback Capabilities of MindSpore](#callback-capabilities-of-mindspore)
        - [Custom Callback](#custom-callback)
    - [MindSpore Metrics](#mindspore-metrics)
    - [MindSpore Print Operator](#mindspore-print-operator)
    - [Data Dump Introduction](#data-dump-introduction)
        - [Synchronous Dump](#synchronous-dump)
        - [Asynchronous Dump](#asynchronous-dump)
    - [Running Data Recorder](#running-data-recorder)
        - [Usage](#usage)
    - [Log-related Environment Variables and Configurations](#log-related-environment-variables-and-configurations)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_en/custom_debugging_info.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Overview

This section describes how to use the customized capabilities provided by MindSpore, such as `callback`, `metrics`, `Print` operators and log printing, to help you quickly debug the training network.

## Introduction to Callback

Here, callback is not a function but a class. You can use callback to observe the internal status and related information of the network during training or perform specific actions in a specific period.
For example, you can monitor the loss, save model parameters, dynamically adjust parameters, and terminate training tasks in advance.

### Callback Capabilities of MindSpore

MindSpore provides the callback capabilities to allow users to insert customized operations in a specific phase of training or inference, including:

- Callback classes such as `ModelCheckpoint`, `LossMonitor`, and `SummaryCollector` provided by the MindSpore framework.
- Custom callback classes.

Usage: Transfer the callback object in the `model.train` method. The callback object can be a list, for example:

```python
from mindspore.train.callback import ModelCheckpoint, LossMonitor, SummaryCollector

ckpt_cb = ModelCheckpoint()
loss_cb = LossMonitor()
summary_cb = SummaryCollector(summary_dir='./summary_dir')
model.train(epoch, dataset, callbacks=[ckpt_cb, loss_cb, summary_cb])
```

`ModelCheckpoint` can save model parameters for retraining or inference.
`LossMonitor` can output loss information in logs for users to view. In addition, `LossMonitor` monitors the loss value change during training. When the loss value is `Nan` or `Inf`, the training terminates.
`SummaryCollector` can save the training information to files for later use.
During the training process, the callback list will execute the callback function in the defined order. Therefore, in the definition process, the dependency between callbacks needs to be considered.

### Custom Callback

You can customize callback based on the `callback` base class as required.

The callback base class is defined as follows:

```python
class Callback():
    """Callback base class"""
    def begin(self, run_context):
        """Called once before the network executing."""
        pass

    def epoch_begin(self, run_context):
        """Called before each epoch beginning."""
        pass

    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        pass

    def step_begin(self, run_context):
        """Called before each step beginning."""
        pass

    def step_end(self, run_context):
        """Called after each step finished."""
        pass

    def end(self, run_context):
        """Called once after network training."""
        pass
```

The callback can record important information during training and transfer the information to the callback object through a dictionary variable `cb_params`,
You can obtain related attributes from each custom callback and perform customized operations. You can also customize other variables and transfer them to the `cb_params` object.

The main attributes of `cb_params` are as follows:

- loss_fn: Loss function
- optimizer: Optimizer
- train_dataset: Training dataset
- cur_epoch_num: Number of current epochs
- cur_step_num: Number of current steps
- batch_num: Number of batches in an epoch
- epoch_num: Number of training epochs
- batch_num: Number of training batch
- train_network: Training network
- parallel_mode: Parallel mode
- list_callback: All callback functions
- net_outputs: Network output results
- ...

You can inherit the callback base class to customize a callback object.

Here are two examples to further explain the usage of custom Callback.

> custom `Callback` sample code：
>
> <https://gitee.com/mindspore/docs/blob/r1.5/docs/sample_code/debugging_info/custom_callback.py>

- Terminate training within the specified time.

    ```python
    from mindspore.train.callback import Callback

    class StopAtTime(Callback):
        def __init__(self, run_time):
            super(StopAtTime, self).__init__()
            self.run_time = run_time*60

        def begin(self, run_context):
            cb_params = run_context.original_args()
            cb_params.init_time = time.time()

        def step_end(self, run_context):
            cb_params = run_context.original_args()
            epoch_num = cb_params.cur_epoch_num
            step_num = cb_params.cur_step_num
            loss = cb_params.net_outputs
            cur_time = time.time()
            if (cur_time - cb_params.init_time) > self.run_time:
                print("epoch: ", epoch_num, " step: ", step_num, " loss: ", loss)
                run_context.request_stop()
    ```

    The output is as follows:

    ```text
    epoch: 20 step: 32 loss: 2.298344373703003
    ```

    The implementation principle is: You can use the `run_context.original_args` method to obtain the `cb_params` dictionary, which contains the main attribute information described above.
    In addition, you can modify and add values in the dictionary. In the preceding example, an `init_time` object is defined in `begin` and transferred to the `cb_params` dictionary.
    A decision is made at each `step_end`. When the training time is longer than the configured time threshold, a training termination signal will be sent to the `run_context` to terminate the training in advance and the current values of epoch, step, and loss will be printed.

- Save the checkpoint file with the highest accuracy during training.

    ```python
    from mindspore.train.callback import Callback

    class SaveCallback(Callback):
        def __init__(self, eval_model, ds_eval):
            super(SaveCallback, self).__init__()
            self.model = eval_model
            self.ds_eval = ds_eval
            self.acc = 0

        def step_end(self, run_context):
            cb_params = run_context.original_args()
            result = self.model.eval(self.ds_eval)
            if result['accuracy'] > self.acc:
                self.acc = result['accuracy']
                file_name = str(self.acc) + ".ckpt"
                save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
                print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)
    ```

    The specific implementation principle is: define a callback object, and initialize the object to receive the model object and the ds_eval (verification dataset). Verify the accuracy of the model in the step_end phase. When the accuracy is the current highest, automatically trigger the save checkpoint method to save the current parameters.

## MindSpore Metrics

After the training is complete, you can use metrics to evaluate the training result.

MindSpore provides multiple metrics, such as `accuracy`, `loss`, `tolerance`, `recall`, and `F1`.

You can define a metrics dictionary object that contains multiple metrics and transfer them to the `model` object and use the `model.eval` function to verify the training result.

> `metrics` sample code：
>
> <https://gitee.com/mindspore/docs/blob/r1.5/docs/sample_code/debugging_info/custom_metrics.py>

```python
from mindspore import Model
import mindspore.nn as nn

metrics = {
    'accuracy': nn.Accuracy(),
    'loss': nn.Loss(),
    'precision': nn.Precision(),
    'recall': nn.Recall(),
    'f1_score': nn.F1()
}
model = Model(network=net, loss_fn=net_loss, optimizer=net_opt, metrics=metrics)
result = model.eval(ds_eval)
```

The `model.eval` method returns a dictionary that contains the metrics and results transferred to the metrics.

The callback function can also be used in the eval process, and the user can call the related API or customize the callback method to achieve the desired function.

You can also define your own metrics class by inheriting the `Metric` base class and rewriting the `clear`, `update`, and `eval` methods.

The `Accuracy` operator is used as an example to describe the internal implementation principle.

The `Accuracy` inherits the `EvaluationBase` base class and rewrites the preceding three methods.

- The `clear` method initializes related calculation parameters in the class.
- The `update` method accepts the predicted value and tag value and updates the internal variables of Accuracy.
- The `eval` method calculates related indicators and returns the calculation result.

By invoking the `eval` method of `Accuracy`, you will obtain the calculation result.

You can understand how `Accuracy` runs by using the following code:

```python
from mindspore import Tensor
from mindspore.nn import Accuracy
import numpy as np

x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
y = Tensor(np.array([1, 0, 1]))
metric = Accuracy()
metric.clear()
metric.update(x, y)
accuracy = metric.eval()
print('Accuracy is ', accuracy)
```

The output is as follows:

```text
Accuracy is 0.6667
```

## MindSpore Print Operator

MindSpore-developed `Print` operator is used to print the tensors or character strings input by users. Multiple strings, multiple tensors, and a combination of tensors and strings are supported, which are separated by comma (,). The `Print` operator is only supported in Ascend environment.
The method of using the MindSpore `Print` operator is the same as using other operators. You need to assert MindSpore `Print` operator in `__init__` and invoke it using `construct`. The following is an example.

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE)

class PrintDemo(nn.Cell):
    def __init__(self):
        super(PrintDemo, self).__init__()
        self.print = ops.Print()

    def construct(self, x, y):
        self.print('print Tensor x and Tensor y:', x, y)
        return x

x = Tensor(np.ones([2, 1]).astype(np.int32))
y = Tensor(np.ones([2, 2]).astype(np.int32))
net = PrintDemo()
output = net(x, y)
```

The output is as follows:

```text
print Tensor x and Tensor y:
Tensor(shape=[2, 1], dtype=Int32, value=
[[1]
 [1]])
Tensor(shape=[2, 2], dtype=Int32, value=
[[1 1]
 [1 1]])
```

## Data Dump Introduction

When training the network, if the training result deviates from the expectation, the input and output of the operator can be saved for debugging through the data dump function. For detailed Dump function introduction, please refer to [Dump Mode](https://www.mindspore.cn/docs/programming_guide/en/r1.5/dump_in_graph_mode.html#dump-introduction).

### Synchronous Dump

Synchronous Dump function usage reference [Synchronous Dump Step](https://www.mindspore.cn/docs/programming_guide/en/r1.5/dump_in_graph_mode.html#synchronous-dump-step).

### Asynchronous Dump

Asynchronous Dump function usage reference [Asynchronous Dump Step](https://www.mindspore.cn/docs/programming_guide/en/r1.5/dump_in_graph_mode.html#asynchronous-dump-step)。

## Running Data Recorder

Running Data Recorder(RDR) is the feature MindSpore provides to record data while training program is running. If a running exception occurs in MindSpore, the pre-recorded data in MindSpore is automatically exported to assist in locating the cause of the running exception. Different exceptions will export different data, for instance, the occurrence of `Run task error` exception, the computational graph, execution sequence of the graph, memory allocation and other information will be exported to assist in locating the cause of the exception.

> Not all run exceptions export data, and only partial exception exports are currently supported.
>
> Only supports the data collection of CPU/Ascend/GPU in the training scenario with the graph mode.

### Usage

#### Set RDR By Configuration File

1. Create the configuration file `mindspore_config.json`.

    ```json
    {
        "rdr": {
            "enable": true,
            "path": "/path/to/rdr/dir"
        }
    }
    ```

    > enable: Controls whether the RDR is enabled.
    >
    > path: Set the path to which RDR stores data. The current path must be absolute.

2. Configure RDR via `context`.

    ```python
    context.set_context(env_config_path="./mindspore_config.json")
    ```

#### Set RDR By Environment Variables

Set `export MS_RDR_ENABLE=1` to enable RDR, and set the root directory by `export MS_RDR_PATH=/path/to/root/dir` for recording data. The final directory for recording data is `/path/to/root/dir/rank_{RANK_ID}/rdr/`. `{RANK_ID}` is the unique ID for multi-cards training, the single card scenario defaults to `RANK_ID=0`.

> The configuration file set by the user takes precedence over the environment variables.

#### Exception Handling

If MindSpore is used for training on Ascend 910, there is an exception `Run task error` in training.

When we go to the directory for recording data, we can see several files appear in this directory, each file represents a kind of data. For example, `hwopt_d_before_graph_0.ir` is a computational graph file. You can use a text tool to open this file to view the calculational graph and analyze whether the calculational graph meets your expectations.

## Log-related Environment Variables and Configurations

MindSpore uses glog to output logs. The following environment variables are commonly used:

- `GLOG_v`

    The environment variable specifies the log level.  
    The default value is 2, indicating the WARNING level. The values are as follows: 0: DEBUG; 1: INFO; 2: WARNING; 3: ERROR.

- `GLOG_logtostderr`

    The environment variable specifies the log output mode.  
    When `GLOG_logtostderr` is set to 1, logs are output to the screen. If the value is set to 0, logs are output to a file. The default value is 1.

- `GLOG_log_dir`

    The environment variable specifies the log output path. Log files will be saved to the path of `the_specified_directory/rank_${rank_id}/logs/`. During the distributed training, `rank_id` is the ID of the current device in the cluster. Otherwise, `rank_id` is `0`.  
    If `GLOG_logtostderr` is set to 0, value of this variable must be specified.  
    If `GLOG_log_dir` is specified and the value of `GLOG_logtostderr` is 1, logs are output to the screen but not to a file.  
    Logs of C++ and Python will be output to different files. The file name of C++ log complies with the naming rule of `GLOG` log file. Here, the name is `mindspore.MachineName.UserName.log.LogLevel.Timestamp`. The file name of Python log is `mindspore.log`.  
    `GLOG_log_dir` can only contains characters such as uppercase letters, lowercase letters, digits, "-", "_" and "/".

- `GLOG_log_max`
    Each log file's max size is 50 MB by default. But we can change it by set this environment variable. When the log file reaches the max size, the next logs will be written to the new log file.

- `MS_SUBMODULE_LOG_v`

    The environment variable specifies log levels of C++ sub modules of MindSpore.  
    The environment variable is assigned as: `MS_SUBMODULE_LOG_v="{SubModule1:LogLevel1,SubModule2:LogLevel2,...}"`.  
    The specified sub module log level will overwrite the global log level. The meaning of sub module log level is the same as `GLOG_v`, the sub modules of MindSpore are categorized by source directory is shown in the below table.  
    E.g. when set `GLOG_v=1 MS_SUBMODULE_LOG_v="{PARSER:2,ANALYZER:2}"` then log levels of `PARSER` and `ANALYZER` are WARNING, other modules' log levels are INFO.

- `GLOG_stderrthreshold`

    The log module will print logs to the screen when these logs are output to a file. This environment variable is used to control the log level printed to the screen in this scenario.
    The default value is 2, indicating the WARNING level. The values are as follows: 0: DEBUG; 1: INFO; 2: WARNING; 3: ERROR.

Sub modules of MindSpore grouped by source directory:

| Source Files | Sub Module Name |
| ------------ | --------------- |
| mindspore/ccsrc/backend/kernel_compiler | KERNEL |
| mindspore/ccsrc/backend/optimizer | PRE_ACT |
| mindspore/ccsrc/backend/session | SESSION |
| mindspore/ccsrc/common | COMMON |
| mindspore/ccsrc/debug | DEBUG |
| mindspore/ccsrc/frontend/operator | ANALYZER |
| mindspore/ccsrc/frontend/optimizer | OPTIMIZER |
| mindspore/ccsrc/frontend/parallel | PARALLEL |
| mindspore/ccsrc/minddata/dataset | MD |
| mindspore/ccsrc/minddata/mindrecord | MD |
| mindspore/ccsrc/pipeline/jit/*.cc | PIPELINE |
| mindspore/ccsrc/pipeline/jit/parse | PARSER |
| mindspore/ccsrc/pipeline/jit/static_analysis | ANALYZER |
| mindspore/ccsrc/pipeline/pynative | PYNATIVE |
| mindspore/ccsrc/profiler | PROFILER |
| mindspore/ccsrc/pybind_api | COMMON |
| mindspore/ccsrc/runtime/device | DEVICE |
| mindspore/ccsrc/transform/graph_ir | GE_ADPT |
| mindspore/ccsrc/transform/express_ir | EXPRESS |
| mindspore/ccsrc/utils | UTILS |
| mindspore/ccsrc/vm | VM |
| mindspore/ccsrc | ME |
| mindspore/core/gvar | COMMON |
| mindspore/core/ | CORE |

> The glog does not support log rotate. To control the disk space occupied by log files, use the log file management tool provided by the operating system, such as: logrotate of Linux.
