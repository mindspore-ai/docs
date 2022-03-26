# 自定义调试信息

`Ascend` `GPU` `CPU` `模型调优`

<a href="https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbWFzdGVyL25vdGVib29rL21pbmRzcG9yZV9jdXN0b21fZGVidWdnaW5nX2luZm8uaXB5bmI=&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_modelarts.png"></a>
&nbsp;&nbsp;
<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/custom_debugging_info.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

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
from mindspore.train.callback import ModelCheckpoint, LossMonitor, SummaryCollector

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
        """Called before each step beginning."""
        pass

    def step_end(self, run_context):
        """Called after each step finished."""
        pass

    def end(self, run_context):
        """Called once after network training."""
        pass
```

`Callback`可以把训练过程中的重要信息记录下来，通过把一个字典类型变量`RunContext.original_args()`传递给`Callback`对象，
用户可以在各个自定义的`Callback`中获取到相关属性，执行自定义操作。也可以自定义其他变量传递给`RunContext.original_args()`对象。

`RunContext.original_args()`中的主要属性包括：

- `loss_fn`：损失函数
- `optimizer`：优化器
- `train_dataset`：训练的数据集
- `epoch_num`：训练的epoch的数量
- `batch_num`：一个epoch中step的数量
- `train_network`：训练的网络
- `cur_epoch_num`：当前的epoch数
- `cur_step_num`：当前的step数
- `parallel_mode`：并行模式
- `list_callback`：所有的callback函数
- `net_outputs`：网络的输出结果
- ...

用户可以继承`Callback`基类自定义`Callback`对象。

下面通过两个例子，进一步了解自定义`Callback`的用法。

> 自定义`Callback`样例代码：
>
> <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/debugging_info/custom_callback.py>

- 在规定时间内终止训练。

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

    实现逻辑为：通过`run_context.original_args`方法可以获取到`cb_params`字典，字典里会包含前文描述的主要属性信息。
    同时可以对字典内的值进行修改和添加，上述用例中，在`begin`中定义一个`init_time`对象传递给`cb_params`字典。
    在每次`step_end`会做出判断，当训练时间大于设置的时间阈值时，会向`run_context`传递终止训练的信号，提前终止训练，并打印当前的`epoch`、`step`、`loss`的值。

- 保存训练过程中精度最高的checkpoint文件。

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

    具体实现逻辑为：定义一个`Callback`对象，初始化对象接收`model`对象和`ds_eval`(验证数据集)。在`step_end`阶段验证模型的精度，当精度为当前最高时，自动触发保存checkpoint方法，保存当前的参数。

## MindSpore metrics功能介绍

当训练结束后，可以使用metrics评估训练结果的好坏。

MindSpore提供了多种metrics评估指标，如：`accuracy`、`loss`、`precision`、`recall`、`F1`。

用户可以定义一个metrics字典对象，里面包含多种指标，传递给`model`对象，通过`model.eval`来验证训练的效果。

> `metrics`使用样例代码：
>
> <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/debugging_info/custom_metrics.py>

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

输出：

```text
Accuracy is 0.6667
```

## Print算子功能介绍

MindSpore的自研`Print`算子可以将用户输入的Tensor或字符串信息打印出来，支持多字符串输入，多Tensor输入和字符串与Tensor的混合输入，输入参数以逗号隔开。目前`Print`算子仅支持在Ascend环境下使用。

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
Tensor(shape=[2, 1], dtype=Int32, value=
[[1]
 [1]])
Tensor(shape=[2, 2], dtype=Int32, value=
[[1 1]
 [1 1]])
```

## 数据Dump功能介绍

训练网络时，若训练结果和预期有偏差，可以通过数据Dump功能保存算子的输入输出进行调试。详细Dump功能介绍参考[Dump功能说明](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dump_in_graph_mode.html#dump)。

### 同步Dump功能使用方法

同步Dump功能使用参考[同步Dump操作步骤](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dump_in_graph_mode.html#同步dump)。

### 异步Dump功能使用方法

异步Dump功能使用参考[异步Dump操作步骤](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dump_in_graph_mode.html#异步dump)。

## Running Data Recorder

Running Data Recorder(RDR)是MindSpore提供训练程序运行时记录数据的功能。要记录的数据将会在MindSpore中进行预设，运行训练脚本时，如果MindSpore出现了运行异常，则会自动地导出MindSpore中预先记录的数据以辅助定位运行异常的原因。不同的运行异常将会导出不同的数据，比如出现`Run task error`异常，将会导出计算图、图执行顺序、内存分配等信息以辅助定位异常的原因。

> 并非所有运行异常都会导出数据，目前仅支持部分异常导出数据。
>
> 目前仅支持图模式训练场景下，收集CPU/Ascend/GPU的相关数据。

### 使用方法

#### 通过配置文件配置RDR

1. 创建配置文件`mindspore_config.json`。

    ```json
    {
        "rdr": {
            "enable": true,
            "mode": 1,
            "path": "/path/to/rdr/dir"
        }
    }
    ```

    > enable: 控制RDR功能是否开启。
    >
    > mode: 控制RDR数据导出模式，设置为1表示仅在训练异常终止时导出数据，设置为2表示训练异常终止或正常结束时导出数据。
    >
    > path: 设置RDR保存数据的路径，仅支持绝对路径。

2. 通过 `context` 配置RDR。

    ```python
    context.set_context(env_config_path="./mindspore_config.json")
    ```

#### 通过环境变量配置RDR

通过`export MS_RDR_ENABLE=1`来开启RDR，通过`export MS_RDR_MODE=1`或`export MS_RDR_MODE=2`来设置导出数据模式，然后通过`export MS_RDR_PATH=/path/to/root/dir`设置RDR文件导出的根目录路径，最终RDR文件将保存在`/path/to/root/dir/rank_{RANK_ID}/rdr/`目录下。其中`RANK_ID`为多卡训练场景中的卡号，单卡场景默认`RANK_ID=0`。

> 用户设置的配置文件优先级高于环境变量。

#### 异常处理

假如在Ascend 910上使用MindSpore进行训练，训练出现了`Run task error`异常。

这时我们到RDR文件的导出目录中，可以看到有几个文件，每一个文件都代表着一种数据。比如 `hwopt_d_before_graph_0.ir` 该文件为计算图文件。可以使用文本工具打开该文件，用以查看计算图，分析计算图是否符合预期。

#### 诊断处理

当开启RDR并设置环境变量`export MS_RDR_MODE=2`，进入诊断模式。在图编译结束后，我们同样可以在RDR文件的导出目录中看到保存的与异常处理相同的文件。

## 内存复用

内存复用功能(Mem Reuse)是让不同的Tensor共用同样的一部分内存，以降低内存开销，支撑更大的网络，关闭后每个Tensor有自己独立的内存空间，Tensor间无共享内存。
MindSpore内存复用功能默认开启，可以通过以下方式手动控制该功能的关闭和开启。

### 使用方法

1. 创建配置文件`mindspore_config.json`。

    ```json
    {
        "sys": {
            "mem_reuse": true
        }
    }
    ```

    > mem_reuse: 控制内存复用功能是否开启，当设置为true时，控制内存复用功能开启，为false时，内存复用功能关闭。

2. 通过 `context` 配置内存复用功能。

    ```python
    context.set_context(env_config_path="./mindspore_config.json")
    ```

## 日志相关的环境变量和配置

MindSpore采用glog来输出日志，常用的几个环境变量如下：

- `GLOG_v`

    该环境变量控制日志的级别。指定日志级别后，将会输出大于或等于该级别的日志信息，对应关系如下：0-DEBUG、1-INFO、2-WARNING、3-ERROR、4-CRITICAL。
    该环境变量默认值为2，即WARNING级别。ERROR级别表示程序执行出现报错，输出错误日志，程序可能不会终止。CRITICAL级别表示程序执行出现异常，将会终止执行程序。

- `GLOG_logtostderr`

    该环境变量控制日志的输出方式。  
    该环境变量的值设置为1时，日志输出到屏幕；值设置为0时，日志输出到文件。默认值为1。

- `GLOG_log_dir`

    该环境变量指定日志输出的路径，日志保存路径为：`指定的路径/rank_${rank_id}/logs/`。非分布式训练场景下，`rank_id`为0；分布式训练场景下，`rank_id`为当前设备在集群中的ID。  
    若`GLOG_logtostderr`的值为0，则必须设置此变量。  
    若指定了`GLOG_log_dir`且`GLOG_logtostderr`的值为1时，则日志输出到屏幕，不输出到文件。  
    C++和Python的日志会被输出到不同的文件中，C++日志的文件名遵从`GLOG`日志文件的命名规则，这里是`mindspore.机器名.用户名.log.日志级别.时间戳.进程ID`，Python日志的文件名为`mindspore.log.进程ID`。  
    `GLOG_log_dir`只能包含大小写字母、数字、"-"、"_"、"/"等字符。

- `GLOG_stderrthreshold`

    日志模块在将日志输出到文件的同时也会将日志打印到屏幕，该环境变量用于控制此种场景下打印到屏幕的日志级别。
    该环境变量默认值为2，即WARNING级别，对应关系如下：0-DEBUG、1-INFO、2-WARNING、3-ERROR、4-CRITICAL。

- `MS_SUBMODULE_LOG_v`

    该环境变量指定MindSpore C++各子模块的日志级别。  
    该环境变量赋值方式为：`MS_SUBMODULE_LOG_v="{SubModule1:LogLevel1,SubModule2:LogLevel2,...}"`。  
    其中被指定子模块的日志级别将覆盖`GLOG_v`在此模块内的设置，此处子模块的日志级别`LogLevel`与`GLOG_v`的日志级别含义相同，MindSpore子模块的划分如下表。  
    例如可以通过`GLOG_v=1 MS_SUBMODULE_LOG_v="{PARSER:2,ANALYZER:2}"`把`PARSER`和`ANALYZER`模块的日志级别设为WARNING，其他模块的日志级别设为INFO。

MindSpore子模块按照目录划分如下：

| Source Files                                 | Sub Module Name |
| -------------------------------------------- | --------------- |
| mindspore/ccsrc/backend/kernel_compiler      | KERNEL          |
| mindspore/ccsrc/backend/optimizer            | PRE_ACT         |
| mindspore/ccsrc/backend/session              | SESSION         |
| mindspore/ccsrc/common                       | COMMON          |
| mindspore/ccsrc/debug                        | DEBUG           |
| mindspore/ccsrc/frontend/operator            | ANALYZER        |
| mindspore/ccsrc/frontend/optimizer           | OPTIMIZER       |
| mindspore/ccsrc/frontend/parallel            | PARALLEL        |
| mindspore/ccsrc/minddata/dataset             | MD              |
| mindspore/ccsrc/minddata/mindrecord          | MD              |
| mindspore/ccsrc/pipeline/jit/*.cc            | PIPELINE        |
| mindspore/ccsrc/pipeline/jit/parse           | PARSER          |
| mindspore/ccsrc/pipeline/jit/static_analysis | ANALYZER        |
| mindspore/ccsrc/pipeline/pynative            | PYNATIVE        |
| mindspore/ccsrc/profiler                     | PROFILER        |
| mindspore/ccsrc/pybind_api                   | COMMON          |
| mindspore/ccsrc/runtime/device               | DEVICE          |
| mindspore/ccsrc/transform/graph_ir           | GE_ADPT         |
| mindspore/ccsrc/transform/express_ir         | EXPRESS         |
| mindspore/ccsrc/utils                        | UTILS           |
| mindspore/ccsrc/vm                           | VM              |
| mindspore/ccsrc                              | ME              |
| mindspore/core/gvar                          | COMMON          |
| mindspore/core/                              | CORE            |

- `GLOG_log_max`

    用于控制MindSpore C++模块日志单文件大小，默认最大为50MB，可以通过该环境变量更改日志文件默认的最大值。如果当前写入的日志文件超过最大值，则新输出的日志内容会写入到新的日志文件中。

- `logger_maxBytes`

    用于控制MindSpore Python模块日志单文件大小，默认是52428800 bytes。

- `logger_backupCount`

    用于控制MindSpore Python模块日志文件数量，默认是30个。

> glog不支持日志文件的绕接，如果需要控制日志文件对磁盘空间的占用，可选用操作系统提供的日志文件管理工具，例如：Linux的logrotate。  
