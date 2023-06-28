# Customized Debugging Information


<a href="https://gitee.com/mindspore/docs/blob/r0.6/tutorials/source_en/advanced_use/customized_debugging_information.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

This section describes how to use the customized capabilities provided by MindSpore, such as `callback`, `metrics`,`Print` operator and log printing, to help you quickly debug the training network.

## Introduction to Callback 

Callback here is not a function but a class. You can use callback to observe the internal status and related information of the network during training or perform specific actions in a specific period.
For example, you can monitor the loss, save model parameters, dynamically adjust parameters, and terminate training tasks in advance.

### Callback Capabilities of MindSpore

MindSpore provides the callback capabilities to allow users to insert customized operations in a specific phase of training or inference, including:

- Callback functions such as `ModelCheckpoint`, `LossMonitor`, and `SummaryStep` provided by the MindSpore framework
- Custom callback functions

Usage: Transfer the callback object in the `model.train` method. The callback object can be a list, for example:

```python
ckpt_cb = ModelCheckpoint()                                                            
loss_cb = LossMonitor()
summary_cb = SummaryStep()
model.train(epoch, dataset, callbacks=[ckpt_cb, loss_cb, summary_cb])
```

`ModelCheckpoint` can save model parameters for retraining or inference.
`LossMonitor` can output loss information in logs for users to view. In addition, `LossMonitor` monitors the loss value change during training. When the loss value is `Nan` or `Inf`, the training terminates.
SummaryStep can save the training information to a file for later use.
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
        """Called before each epoch beginning.""" 
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
- batch_num: Number of steps in an epoch
- ...

You can inherit the callback base class to customize a callback object.

The following example describes how to use a custom callback function.

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

```
epoch: 20 step: 32 loss: 2.298344373703003
```

This callback function is used to terminate the training within a specified period. You can use the `run_context.original_args` method to obtain the `cb_params` dictionary, which contains the main attribute information described above.
In addition, you can modify and add values in the dictionary. In the preceding example, an `init_time` object is defined in `begin` and transferred to the `cb_params` dictionary.
A decision is made at each `step_end`. When the training time is greater than the configured time threshold, a training termination signal will be sent to the `run_context` to terminate the training in advance and the current values of epoch, step, and loss will be printed.

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
model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics)
ds_eval = create_dataset()
output = model.eval(ds_eval)
```

The `model.eval` method returns a dictionary that contains the metrics and results transferred to the metrics.

You can also define your own metrics class by inheriting the `Metric` base class and rewriting the `clear`, `update`, and `eval` methods.

The `accuracy` operator is used as an example to describe the internal implementation principle.

The `accuracy` inherits the `EvaluationBase` base class and rewrites the preceding three methods.
The `clear` method initializes related calculation parameters in the class.
The `update` method accepts the predicted value and tag value and updates the internal variables of accuracy.
The `eval` method calculates related indicators and returns the calculation result.
By invoking the `eval` method of `accuracy`, you will obtain the calculation result.

You can understand how `accuracy` runs by using the following code:

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
```
Accuracy is 0.6667
```
## MindSpore Print Operator
MindSpore-developed `Print` operator is used to print the tensors or character strings input by users. Multiple strings, multiple tensors, and a combination of tensors and strings are supported, which are separated by comma (,). 
The use method of MindSpore `Print` operator is the same that of other operators. You need to assert MindSpore `Print` operator in `__init__` and invoke using `construct`. The following is an example. 
```python
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE)

class PrintDemo(nn.Cell):
    def __init__(self):
        super(PrintDemo, self).__init__()
        self.print = P.Print()

    def construct(self, x, y):
        self.print('print Tensor x and Tensor y:', x, y)
        return x

x = Tensor(np.ones([2, 1]).astype(np.int32))
y = Tensor(np.ones([2, 2]).astype(np.int32))
net = PrintDemo()
output = net(x, y)
```
The output is as follows：
```
print Tensor x and Tensor y:
Tensor shape:[[const vector][2, 1]]Int32
val:[[1]
[1]]
Tensor shape:[[const vector][2, 2]]Int32
val:[[1 1]
[1 1]]
```

## Asynchronous Data Dump

When the training result deviates from the expectation on Ascend, the input and output of the operator can be dumped for debugging through Asynchronous Data Dump.

> `comm_ops` operators are not supported by Asynchronous Data Dump. `comm_ops` can be found in [Operator List](https://www.mindspore.cn/docs/en/r0.6/operator_list.html).

1. Turn on the switch to save graph IR: `context.set_context(save_graphs=True)`.
2. Execute training script.
3. Open `hwopt_d_end_graph_{graph id}.ir` in the directory you execute the script and find the name of the operators you want to Dump.
4. Configure json file: `data_dump.json`.

    ```json
    {
        "DumpSettings": {
            "net_name": "ResNet50",
            "mode": 1,
            "iteration": 0,
            "kernels": ["Default/Conv2D-op2", "Default/TensorAdd-op10"]
        },

        "DumpSettingsSpec": {
            "net_name": "net name eg:ResNet50",
            "mode": "0: dump all kernels, 1: dump kernels in kernels list",
            "iteration": "specified iteration",
            "kernels": "op's full scope name which need to be dump"
        }
    }
    ```

    > - Iteration should be set to 0 when `dataset_sink_mode` is False and data of every iteration will be dumped.
    > - Iteration should increase by 1 when `dataset_sink_mode` is True. For example, data of GetNext will be dumped in iteration 0 and data of compute graph will be dumped in iteration 1.

5. Set environment variables.

    ```bash
    export ENABLE_DATA_DUMP=1
    export DATA_DUMP_PATH=/test
    export DATA_DUMP_CONFIG_PATH=data_dump.json
    ```

    > - Set the environment variables before executing the training script. Setting environment variables during training will not take effect.
    > - Dump environment variables need to be configured before calling `mindspore.communication.management.init`.

6. Execute the training script again.
7. Parse the Dump file.

    Change directory to `/var/log/npu/ide_daemon/dump/` after training and execute the following commands to parse Dump data file:

    ```bash
    python /usr/local/Ascend/toolkit/tools/operator_cmp/compare/dump_data_conversion.pyc -type offline -target numpy -i ./{Dump file path}} -o ./{output file path}
    ```

## Log-related Environment Variables and Configurations
MindSpore uses glog to output logs. The following environment variables are commonly used:

1. `GLOG_v` specifies the log level. The default value is 2, indicating the WARNING level. The values are as follows: 0: DEBUG; 1: INFO; 2: WARNING; 3: ERROR.
2. When `GLOG_logtostderr` is set to 1, logs are output to the screen. If the value is set to 0, logs are output to a file. Default value: 1
3. GLOG_log_dir=*YourPath* specifies the log output path. If `GLOG_logtostderr` is set to 0, value of this variable must be specified. If `GLOG_log_dir is` specified and the value of `GLOG_logtostderr` is 1, logs are output to the screen but not to a file. Logs of C++ and Python will be output to different files. The file name of C++ log complies with the naming rule of GLOG log file. Here, the name is `mindspore.MachineName.UserName.log.LogLevel.Timestamp`. The file name of Python log is `mindspore.log`.
4. `MS_SUBMODULE_LOG_v="{SubModule1:LogLevel1,SubModule2:LogLevel2,...}"` specifies log levels of C++ sub modules of MindSpore. The specified sub module log level will overwrite the global log level. The meaning of submodule log level is same as `GLOG_v`, the sub modules of MindSpore grouped by source directory is as the bellow table. E.g. when set `GLOG_v=1 MS_SUBMODULE_LOG_v="{PARSER:2,ANALYZER:2}"` then log levels of `PARSER` and `ANALYZER` are WARNING, other modules' log levels are INFO.

Sub moudles of MindSpore grouped by source directory:

| Source Files | Sub Module Name |
| ------------ | --------------- |
| mindspore/ccsrc/common | COMMON |
| mindspore/ccsrc/dataset | MD |
| mindspore/ccsrc/debug | DEBUG |
| mindspore/ccsrc/device | DEVICE |
| mindspore/ccsrc/gvar | COMMON |
| mindspore/ccsrc/ir | IR |
| mindspore/ccsrc/kernel | KERNEL |
| mindspore/ccsrc/mindrecord | MD |
| mindspore/ccsrc | ME |
| mindspore/ccsrc/onnx | ONNX |
| mindspore/ccsrc/operator | ANALYZER |
| mindspore/ccsrc/optimizer | OPTIMIZER |
| mindspore/ccsrc/parallel | PARALLEL |
| mindspore/ccsrc/pipeline/*.cc | PIPELINE |
| mindspore/ccsrc/pipeline/parse | PARSER |
| mindspore/ccsrc/pipeline/static_analysis | ANALYZER |
| mindspore/ccsrc/pre_activate | PRE_ACT |
| mindspore/ccsrc/pybind_api | COMMON |
| mindspore/ccsrc/pynative | PYNATIVE |
| mindspore/ccsrc/session | SESSION |
| mindspore/ccsrc/transform | GE_ADPT |
| mindspore/ccsrc/utils | UTILS |
| mindspore/ccsrc/vm | VM |

