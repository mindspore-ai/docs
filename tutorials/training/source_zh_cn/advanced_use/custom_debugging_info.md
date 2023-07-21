# 自定义调试信息

`Linux` `Ascend` `GPU` `CPU` `模型调优` `中级` `高级`

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_zh_cn/advanced_use/custom_debugging_info.md)
&nbsp;&nbsp;
[![查看Notebook](../_static/logo_notebook.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/notebook/customized_debugging_information.ipynb)

## 概述

本文介绍如何使用MindSpore提供的`Callback`、`metrics`、`Print`算子、日志打印等自定义能力，帮助用户快速调试训练网络。

## Callback介绍

`Callback`是回调函数的意思，但它其实不是一个函数而是一个类，用户可以使用回调函数来观察训练过程中网络内部的状态和相关信息，或在特定时期执行特定动作。
例如监控loss、保存模型参数、动态调整参数、提前终止训练任务等。

### MindSpore的Callback能力

MindSpore提供`Callback`能力，支持用户在训练/推理的特定阶段，插入自定义的操作。包括：

- MindSpore框架提供的`ModelCheckpoint`、`LossMonitor`、`SummaryCollector`等`Callback`类。
- MindSpore支持用户自定义`Callback`。

使用方法：在`model.train`方法中传入`Callback`对象，它可以是一个`Callback`列表，例：

```python
ckpt_cb = ModelCheckpoint()
loss_cb = LossMonitor()
summary_cb = SummaryCollector(summary_dir='./summary_dir')
model.train(epoch, dataset, callbacks=[ckpt_cb, loss_cb, summary_cb])
```

`ModelCheckpoint`可以保存模型参数，以便进行再训练或推理。  
`LossMonitor`可以在日志中输出loss，方便用户查看，同时它还会监控训练过程中的loss值变化情况，当loss值为`Nan`或`Inf`时终止训练。  
`SummaryCollector` 可以把训练过程中的信息存储到文件中，以便后续可视化展示。  
在训练过程中，`Callback`列表会按照定义的顺序执行`Callback`函数。因此在定义过程中，需考虑`Callback`之间的依赖关系。

### 自定义Callback

用户可以基于`Callback`基类，根据自身的需求，实现自定义`Callback`。

`Callback`基类定义如下所示：

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
        """Called before step epoch beginning."""
        pass

    def step_end(self, run_context):
        """Called after each step finished."""
        pass

    def end(self, run_context):
        """Called once after network training."""
        pass
```

`Callback`可以把训练过程中的重要信息记录下来，通过一个字典类型变量`cb_params`传递给`Callback`对象，
用户可以在各个自定义的`Callback`中获取到相关属性，执行自定义操作。也可以自定义其他变量传递给`cb_params`对象。

`cb_params`中的主要属性包括：

- `loss_fn`：损失函数
- `optimizer`：优化器
- `train_dataset`：训练的数据集
- `cur_epoch_num`：当前的epoch数
- `cur_step_num`：当前的step数
- `batch_num`：一个epoch中batch的数量
- ...

用户可以继承`Callback`基类自定义`Callback`对象。

下面通过两个例子，进一步了解自定义`Callback`的用法。

- 在规定时间内终止训练。

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

    输出：

    ```text
    epoch: 20 step: 32 loss: 2.298344373703003
    ```

    实现逻辑为：通过`run_context.original_args`方法可以获取到`cb_params`字典，字典里会包含前文描述的主要属性信息。
    同时可以对字典内的值进行修改和添加，上述用例中，在`begin`中定义一个`init_time`对象传递给`cb_params`字典。
    在每次`step_end`会做出判断，当训练时间大于设置的时间阈值时，会向`run_context`传递终止训练的信号，提前终止训练，并打印当前的`epoch`、`step`、`loss`的值。

- 保存训练过程中精度最高的checkpoint文件。

    ```python
    from mindspore.train.serialization import save_checkpoint

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

    具体实现逻辑为：定义一个`Callback`对象，初始化对象接收`model`对象和`ds_eval`(验证数据集)。在`step_end`阶段验证模型的精度，当精度为当前最高时，手动触发保存checkpoint方法，保存当前的参数。

## MindSpore metrics功能介绍

当训练结束后，可以使用metrics评估训练结果的好坏。

MindSpore提供了多种metrics评估指标，如：`accuracy`、`loss`、`precision`、`recall`、`F1`。

用户可以定义一个metrics字典对象，里面包含多种指标，传递给`model.eval`接口用来验证训练精度。

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

`model.eval`方法会返回一个字典，里面是传入metrics的指标和结果。

在eval过程中也可以使用`Callback`功能，用户可以调用相关API或自定义`Callback`方法实现想要的功能。

用户也可以定义自己的`metrics`类，通过继承`Metric`基类，并重写`clear`、`update`、`eval`三个方法即可实现。

以`Accuracy`算子举例说明其内部实现原理：

`Accuracy`继承了`EvaluationBase`基类，重写了上述三个方法。  

- `clear`方法会把类中相关计算参数初始化。  
- `update`方法接受预测值和标签值，更新`Accuracy`内部变量。  
- `eval`方法会计算相关指标，返回计算结果。  

调用`Accuracy`的`eval`方法，即可得到计算结果。  

通过如下代码可以更清楚了解到`Accuracy`是如何运行的：

```python
x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
y = Tensor(np.array([1, 0, 1]))
metric = Accuracy()
metric.clear()
metric.update(x, y)
accuracy = metric.eval()
print('Accuracy is ', accuracy)
```

输出：

```text
Accuracy is 0.6667
```

## Print算子功能介绍

MindSpore的自研`Print`算子可以将用户输入的Tensor或字符串信息打印出来，支持多字符串输入，多Tensor输入和字符串与Tensor的混合输入，输入参数以逗号隔开。

`Print`算子使用方法与其他算子相同，在网络中的`__init__`声明算子并在`construct`进行调用，具体使用实例及输出结果如下：

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

输出：

```text
print Tensor x and Tensor y:
Tensor shape:[[const vector][2, 1]]Int32
val:[[1]
[1]]
Tensor shape:[[const vector][2, 2]]Int32
val:[[1 1]
[1 1]]
```

## 数据Dump功能介绍

训练网络时，当训练结果和预期有偏差时，可以通过数据Dump功能保存算子的输入输出进行调试。Dump功能分为同步Dump和异步Dump，同步Dump同时支持GPU和Ascend，而异步Dump只支持Ascend。

### 同步Dump功能介绍

1. 创建配置文件`data_dump.json`。

    JSON文件的名称和位置可以自定义设置。

    ```json
    {
        "common_dump_settings": {
            "dump_mode": 0,
            "path": "/tmp/net/",
            "net_name": "ResNet50",
            "iteration": 0,
            "input_output": 0,
            "kernels": ["Default/Conv-op12"],
            "support_device": [0,1,2,3,4,5,6,7]
        },
        "e2e_dump_settings": {
            "enable": false,
            "trans_flag": false
        }
    }
    ```

    - `dump_mode`：设置成0，表示Dump出改网络中的所有算子；设置成1，表示Dump`"kernels"`里面制定的算子。
    - `path`：Dump保存数据的绝对路径。
    - `net_name`：自定义的网络名称，例如："ResNet50"。
    - `iteration`：指定需要Dump的迭代，若设置成0，表示Dump所有的迭代。
    - `input_output`：设置成0，表示Dump出算子的输入和算子的输出；设置成1，表示Dump出算子的输入；设置成2，表示Dump出算子的输出。该参数仅支持Ascend，GPU只能Dump算子的输出。
    - `kernels`：算子的全称，可以通过开启IR保持开关`context.set_context(save_graphs=True)`执行用例，从生成的`ir`文件获取。例如，`device_target`为`Ascend`时，可以从`hwopt_d_end_graph_{graph_id}.ir`中获取算子全称，`device_target`为`GPU`时，可以从`hwopt_pm_7_getitem_tuple.ir`中获取算子全称。
    - `support_device`：支持的设备，默认设置成0到7即可；在分布式训练场景下，需要dump个别设备上的数据，可以只在`support_device`中指定需要Dump的设备Id。
    - `enable`：开启E2E Dump。
    - `trans_flag`：开启格式转换。将设备上的数据格式转换成NCHW格式。

2. 指定Dump的json配置文件。

    ```bash
    export MINDSPORE_DUMP_CONFIG={Absolute path of data_dump.json}
    ```

    - 在网络脚本执行前，设置好环境变量；网络脚本执行过程中设置将会不生效。
    - 在分布式场景下，Dump环境变量需要调用`mindspore.communication.management.init`之前配置。

3. 执行用例Dump数据。

    可以在训练脚本中设置`context.set_context(reserve_class_name_in_scope=False)`，避免Dump文件名称过长导致Dump数据文件生成失败。
    
4. 解析Dump数据。

    通过`numpy.fromfile`读取Dump数据文件即可解析。

### 异步Dump功能介绍

1. 创建配置文件`data_dump.json`。

    JSON文件的名称和位置可以自定义设置。

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
            "enable": false,
            "op_debug_mode": 0
        }
    }
    ```

    - `dump_mode`：设置成0，表示Dump出改网络中的所有算子；设置成1，表示Dump`"kernels"`里面指定的算子。
    - `path`：Dump保存数据的绝对路径。
    - `net_name`：自定义的网络名称，例如："ResNet50"。
    - `iteration`：指定需要Dump的迭代。非数据下沉模式下，`iteration`需要设置成0，并且会Dump出每个迭代的数据。
    - `input_output`：设置成0，表示Dump出算子的输入和算子的输出；设置成1，表示Dump出算子的输入；设置成2，表示Dump出算子的输出。
    - `kernels`：算子的全称。开启IR保持开关`context.set_context(save_graphs=True)`并执行用例，从生成的`hwopt_d_end_graph_{graph_id}.ir`文件获取。`kernels`仅支持TBE算子、AiCPU算子、通信算子，若设置成通信算子的名称，将会Dump出通信算子的输入算子的数据。
    - `support_device`：支持的设备，默认设置成0到7即可；在分布式训练场景下，需要dump个别设备上的数据，可以只在`support_device`中指定需要Dump的设备Id。
    - `enable`：开启异步Dump。
    - `op_debug_mode`：该属性用于算子溢出调试，在使用Dump功能的时候，请设置成0。

2. 设置数据Dump的环境变量。

    ```bash
    export MINDSPORE_DUMP_CONFIG={Absolute path of data_dump.json}
    ```

    - 在网络脚本执行前，设置好环境变量；网络脚本执行过程中设置将会不生效。
    - 在分布式场景下，Dump环境变量需要调用`mindspore.communication.management.init`之前配置。

3. 执行用例Dump数据。

    可以在训练脚本中设置`context.set_context(reserve_class_name_in_scope=False)`，避免Dump文件名称过长导致Dump数据文件生成失败。
    
4. 解析文件。

    执行完用例后去`/var/log/npu/ide_daemon/dump/`目录下，运行如下命令解析Dump数据：

    ```bash
    python /usr/local/Ascend/toolkit/tools/operator_cmp/compare/dump_data_conversion.pyc -type offline -target numpy -i ./{Dump出来的文件} -o ./{解析的文件路径}
    ```

## 日志相关的环境变量和配置

MindSpore采用glog来输出日志，常用的几个环境变量如下：

- `GLOG_v`

    该环境变量控制日志的级别。  
    该环境变量默认值为2，即WARNING级别，对应关系如下：0-DEBUG、1-INFO、2-WARNING、3-ERROR。

- `GLOG_logtostderr`

    该环境变量控制日志的输出方式。
    该环境变量的值设置为1时，日志输出到屏幕；值设置为0时，日志输出到文件。默认值为1。

- `GLOG_log_dir`

    该环境变量指定日志输出的路径。  
    若`GLOG_logtostderr`的值为0，则必须设置此变量。  
    若指定了`GLOG_log_dir`且`GLOG_logtostderr`的值为1时，则日志输出到屏幕，不输出到文件。  
    C++和Python的日志会被输出到不同的文件中，C++日志的文件名遵从`GLOG`日志文件的命名规则，这里是`mindspore.机器名.用户名.log.日志级别.时间戳`，Python日志的文件名为`mindspore.log`。

- `MS_SUBMODULE_LOG_v`

    该环境变量指定MindSpore C++各子模块的日志级别。  
    该环境变量赋值方式为：`MS_SUBMODULE_LOG_v="{SubModule1:LogLevel1,SubModule2:LogLevel2,...}"`。  
    其中被指定子模块的日志级别将覆盖`GLOG_v`在此模块内的设置，此处子模块的日志级别`LogLevel`与`GLOG_v`的日志级别含义相同，MindSpore子模块的划分如下表。  
    例如可以通过`GLOG_v=1 MS_SUBMODULE_LOG_v="{PARSER:2,ANALYZER:2}"`把`PARSER`和`ANALYZER`模块的日志级别设为WARNING，其他模块的日志级别设为INFO。

MindSpore子模块按照目录划分如下：

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
| mindspore/ccsrc/transform/onnx | ONNX |
| mindspore/ccsrc/utils | UTILS |
| mindspore/ccsrc/vm | VM |
| mindspore/ccsrc | ME |
| mindspore/core/gvar | COMMON |
| mindspore/core/ | CORE |

> glog不支持日志文件的绕接，如果需要控制日志文件对磁盘空间的占用，可选用操作系统提供的日志文件管理工具，例如：Linux的logrotate。  
