# Custom Debugging Information

`Linux` `Ascend` `GPU` `CPU` `Model Optimization` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_en/advanced_use/custom_debugging_info.md)

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
- ...

You can inherit the callback base class to customize a callback object.

Here are two examples to further explain the usage of custom Callback.

- Terminate training within the specified time.

    ```python
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

    stop_cb = StopAtTime(run_time=10)
    model.train(100, dataset, callbacks=stop_cb)
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
    from mindspore import save_checkpoint

    class SaveCallback(Callback):
        def __init__(self, model, eval_dataset):
            super(SaveCallback, self).__init__()
            self.model = model
            self.eval_dataset = eval_dataset
            self.acc = 0.5

        def step_end(self, run_context):
            cb_params = run_context.original_args()
            epoch_num = cb_params.cur_epoch_num

            result = self.model.eval(self.eval_dataset)
            if result['accuracy'] > self.acc:
                self.acc = result['accuracy']
                file_name = str(self.acc) + ".ckpt"
                save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
                print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)


    network = Lenet()
    loss = nn.SoftmaxCrossEntryWithLogits(sparse=True, reduction='mean')
    oprimizer = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    model = Model(network, loss_fn=loss, optimizer=optimizer, metrics={"accuracy"})
    model.train(epoch_size, train_dataset=ds_train, callbacks=SaveCallback(model, ds_eval))
    ```

    The specific implementation principle is: define a callback object, and initialize the object to receive the model object and the ds_eval (verification dataset). Verify the accuracy of the model in the step_end phase. When the accuracy is the current highest, manually trigger the save checkpoint method to save the current parameters.

## MindSpore Metrics

After the training is complete, you can use metrics to evaluate the training result.

MindSpore provides multiple metrics, such as `accuracy`, `loss`, `tolerance`, `recall`, and `F1`.

You can define a metrics dictionary object that contains multiple metrics and transfer them to the `model.eval` interface to verify the training precision.

```python
metrics = {
    'accuracy': nn.Accuracy(),
    'loss': nn.Loss(),
    'precision': nn.Precision(),
    'recall': nn.Recall(),
    'f1_score': nn.F1()
}
net = ResNet()
loss = CrossEntropyLoss()
opt = Momentum()
model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics, callbacks=TimeMonitor())
ds_eval = create_dataset()
output = model.eval(ds_eval)
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

The input and output of the operator can be saved for debugging through the data dump when the training result deviates from the expectation.

### Synchronous Dump

Synchronous Dump supports the graphics mode both on GPU and Ascend, and currently does not support PyNative mode. When Dump is enabled on Ascend, the operator to Dump will automatically close memory reuse. When the network does not occupy much memory, please use synchronous dump first. If an error of insufficient device memory occurs after enabling synchronous dump, please use asynchronous dump in the next section.

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
        "e2e_dump_settings": {
            "enable": true,
            "trans_flag": false
        }
    }
    ```

    - `dump_mode`: 0: dump all kernels in graph, 1: dump kernels in kernels list.
    - `path`: The absolute path to save dump data.
    - `net_name`: The net name, e.g. ResNet50.
    - `iteration`: Specify the iterations to dump. All kernels in graph will be dumped.
    - `input_output`: 0:dump input and output of kernel, 1:dump input of kernel, 2:dump output of kernel.
    - `kernels`: Full name of kernel. Enable `context.set_context(save_graphs=True)` and get full name of kernel from `ir` file. You can get it from `hwopt_d_end_graph_{graph_id}.ir` when `device_target` is `Ascend` and you can get it from `hwopt_pm_7_getitem_tuple.ir` when `device_target` is `GPU`.
    - `support_device`: Supported devices, default setting is `[0,1,2,3,4,5,6,7]`. You can specify specific device ids to dump specific device data.
    - `enable`: Enable synchronous dump.
    - `trans_flag`: Enable trans flag. Transform the device data format into NCHW.

2. Specify the location of the JSON file.

    ```bash
    export MINDSPORE_DUMP_CONFIG={Absolute path of data_dump.json}
    ```

    - Set the environment variables before executing the training script. Settings will not take effect during training.
    - Dump environment variables need to be configured before calling `mindspore.communication.management.init`.

3. Execute the training script to dump data.

    You can set `context.set_context(reserve_class_name_in_scope=False)` in your training script to avoid dump failure because of file name is too long.

4. Parse the Dump file.

    Call `numpy.fromfile` to parse dump data file.

### Asynchronous Dump

Asynchronous Dump only supports graph mode on Ascend, not PyNative mode. Memory reuse will not be turned off when asynchronous dump is enabled.

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
    - `input_output`: 0: dump input and output of kernel, 1:dump input of kernel, 2:dump output of kernel. This parameter does not take effect on the GPU and only the output of operator will be dumped.
    - `kernels`: Full name of kernel. Enable `context.set_context(save_graphs=True)` and get full name of kernel from `hwopt_d_end_graph_{graph_id}.ir`. `kernels` only support TBE operator, AiCPU operator and communication operator. Data of communication operation input operator will be dumped if `kernels` is set to the name of communication operator.
    - `support_device`: Supported devices, default setting is `[0,1,2,3,4,5,6,7]`. You can specify specific device ids to dump specific device data.
    - `enable`: Enable Asynchronous Dump.
    - `op_debug_mode`: 0: disable overflow check function; 1: enable AiCore overflow check; 2: enable Atomic overflow check; 3: enable all overflow check function.

2. Specify the json configuration file of Dump.

    ```bash
    export MINDSPORE_DUMP_CONFIG={Absolute path of data_dump.json}
    ```

    - Set the environment variables before executing the training script. Setting environment variables during training will not take effect.
    - Dump environment variables need to be configured before calling `mindspore.communication.management.init`.

3. Execute the training script to dump data.

    You can set `context.set_context(reserve_class_name_in_scope=False)` in your training script to avoid dump failure because of file name is too long.

4. Parse the dumped file.

    Parse the dumped file using `dump_data_conversion.pyc` provied in the run package. You can find it through the find command:

    ```bash
    find ${Installation path of run package} -name "dump_data_conversion.pyc"
    ```

    Change directory to `/absolute_path` after training, execute the following commands to parse Dump data file:

    ```bash
    python ${Absolute path of dump_data_conversion.pyc} -type offline -target numpy -i ./{Dump file path}} -o ./{output file path}
    ```

    Or you can use `msaccucmp.pyc` to convert the format of dump file. Please see <https://support.huaweicloud.com/tg-Inference-cann/atlasaccuracy_16_0013.html>.

## Log-related Environment Variables and Configurations

MindSpore uses glog to output logs. The following environment variables are commonly used:

- `GLOG_v`

    The environment variable specifies the log level.  
    The default value is 2, indicating the WARNING level. The values are as follows: 0: DEBUG; 1: INFO; 2: WARNING; 3: ERROR.

- `GLOG_logtostderr`

    The environment variable specifies the log output mode.  
    When `GLOG_logtostderr` is set to 1, logs are output to the screen. If the value is set to 0, logs are output to a file. The default value is 1.

- `GLOG_log_dir`

    The environment variable specifies the log output path.  
    If `GLOG_logtostderr` is set to 0, value of this variable must be specified.  
    If `GLOG_log_dir` is specified and the value of `GLOG_logtostderr` is 1, logs are output to the screen but not to a file.  
    Logs of C++ and Python will be output to different files. The file name of C++ log complies with the naming rule of `GLOG` log file. Here, the name is `mindspore.MachineName.UserName.log.LogLevel.Timestamp`. The file name of Python log is `mindspore.log`.

- `MS_SUBMODULE_LOG_v`

    The environment variable specifies log levels of C++ sub modules of MindSpore.  
    The environment variable is assigned as: `MS_SUBMODULE_LOG_v="{SubModule1:LogLevel1,SubModule2:LogLevel2,...}"`.  
    The specified sub module log level will overwrite the global log level. The meaning of sub module log level is the same as `GLOG_v`, the sub modules of MindSpore are categorized by source directory is shown in the below table.  
    E.g. when set `GLOG_v=1 MS_SUBMODULE_LOG_v="{PARSER:2,ANALYZER:2}"` then log levels of `PARSER` and `ANALYZER` are WARNING, other modules' log levels are INFO.

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
