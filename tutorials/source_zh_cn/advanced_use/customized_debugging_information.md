# 自定义调试信息

<a href="https://gitee.com/mindspore/docs/blob/r0.5/tutorials/source_zh_cn/advanced_use/customized_debugging_information.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

本文介绍如何使用MindSpore提供的`Callback`、`metrics`、`Print`算子、日志打印等自定义能力，帮助用户快速调试训练网络。

## Callback介绍

Callback是回调函数的意思，但它其实不是一个函数而是一个类，用户可以使用回调函数来观察训练过程中网络内部的状态和相关信息，或在特定时期执行特定动作。
例如监控loss、保存模型参数、动态调整参数、提前终止训练任务等。

### MindSpore的Callback能力

MindSpore提供Callback能力，支持用户在训练/推理的特定阶段，插入自定义的操作。包括：

- MindSpore框架提供的`ModelCheckpoint`、`LossMonitor`、`SummaryStep`等Callback函数。
- MindSpore支持用户自定义Callback。

使用方法：在`model.train`方法中传入Callback对象，它可以是一个Callback列表，例：

```python
ckpt_cb = ModelCheckpoint()                                                            
loss_cb = LossMonitor()
summary_cb = SummaryStep()
model.train(epoch, dataset, callbacks=[ckpt_cb, loss_cb, summary_cb])
```

`ModelCheckpoint`可以保存模型参数，以便进行再训练或推理。
`LossMonitor`可以在日志中输出loss，方便用户查看，同时它还会监控训练过程中的loss值变化情况，当loss值为`Nan`或`Inf`时终止训练。
SummaryStep可以把训练过程中的信息存储到文件中，以便后续进行查看或可视化展示。
在训练过程中，Callback列表会按照定义的顺序执行Callback函数。因此在定义过程中，需考虑Callback之间的依赖关系。

### 自定义Callback

用户可以基于`Callback`基类，根据自身的需求，实现自定义`Callback`。

Callback基类定义如下所示：

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

Callback可以把训练过程中的重要信息记录下来，通过一个字典类型变量`cb_params`传递给Callback对象，
用户可以在各个自定义的Callback中获取到相关属性，执行自定义操作。也可以自定义其他变量传递给`cb_params`对象。

`cb_params`中的主要属性包括：

- loss_fn：损失函数
- optimizer：优化器
- train_dataset：训练的数据集
- cur_epoch_num：当前的epoch数
- cur_step_num：当前的step数
- batch_num：一个epoch中step的数量
- ...

用户可以继承Callback基类自定义Callback对象。

下面通过介绍一个例子，更深一步地了解自定义Callback的用法。

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

```
epoch: 20 step: 32 loss: 2.298344373703003
```

此Callback的功能为：在规定时间内终止训练。通过`run_context.original_args`方法可以获取到`cb_params`字典，字典里会包含前文描述的主要属性信息。
同时可以对字典内的值进行修改和添加，上述用例中，在`begin`中定义一个`init_time`对象传递给`cb_params`字典。
在每次`step_end`会做出判断，当训练时间大于设置的时间阈值时，会向`run_context`传递终止训练的信号，提前终止训练，并打印当前的`epoch`、`step`、`loss`的值。

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
model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics)
ds_eval = create_dataset()
output = model.eval(ds_eval)
```

`model.eval`方法会返回一个字典，里面是传入metrics的指标和结果。

用户也可以定义自己的`metrics`类，通过继承`Metric`基类，并重写`clear`、`update`、`eval`三个方法即可实现。

以`accuracy`算子举例说明其内部实现原理：

`accuracy`继承了`EvaluationBase`基类，重写了上述三个方法。
`clear`方法会把类中相关计算参数初始化。
`update`方法接受预测值和标签值，更新accuracy内部变量。
`eval`方法会计算相关指标，返回计算结果。
调用`accuracy`的`eval`方法，即可得到计算结果。

通过如下代码可以更清楚了解到`accuracy`是如何运行的：

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
```
Accuracy is 0.6667
```
## Print算子功能介绍
MindSpore的自研`Print`算子可以将用户输入的Tensor或字符串信息打印出来，支持多字符串输入，多Tensor输入和字符串与Tensor的混合输入，输入参数以逗号隔开。

`Print`算子使用方法与其他算子相同，在网络中的`__init__`声明算子并在`construct`进行调用，具体使用实例及输出结果如下：
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
输出：
```
print Tensor x and Tensor y:
Tensor shape:[[const vector][2, 1]]Int32
val:[[1]
[1]]
Tensor shape:[[const vector][2, 2]]Int32
val:[[1 1]
[1 1]]
```

## 日志相关的环境变量和配置
MindSpore采用glog来输出日志，常用的几个环境变量如下：

1. `GLOG_v` 控制日志的级别，默认值为2，即WARNING级别，对应关系如下：0-DEBUG、1-INFO、2-WARNING、3-ERROR。
2. `GLOG_logtostderr` 值设置为1时，日志输出到屏幕；值设置为0时，日志输出到文件。默认值为1。
3. GLOG_log_dir=*YourPath* 指定日志输出的路径。若`GLOG_logtostderr`的值为0，则必须设置此变量。若指定了`GLOG_log_dir`且`GLOG_logtostderr`的值为1时，则日志输出到屏幕，不输出到文件。C++和Python的日志会被输出到不同的文件中，C++日志的文件名遵从GLOG日志文件的命名规则，这里是`mindspore.机器名.用户名.log.日志级别.时间戳`，Python日志的文件名为`mindspore.log`。
4. `MS_SUBMODULE_LOG_v="{SubModule1:LogLevel1,SubModule2:LogLevel2,...}"` 指定MindSpore C++各子模块的日志级别，被指定的子模块的日志级别将覆盖GLOG_v在此模块内的设置，此处子模块的日志级别LogLevel与`GLOG_v`的日志级别含义相同，MindSpore子模块的划分如下表。如可以通过`GLOG_v=1 MS_SUBMODULE_LOG_v="{PARSER:2,ANALYZER:2}"`把`PARSER`和`ANALYZER`模块的日志级别设为WARNING，其他模块的日志级别设为INFO。

> glog不支持日志文件的绕接，如果需要控制日志文件对磁盘空间的占用，可选用操作系统提供的日志文件管理工具，例如：Linux的logrotate。  

MindSpore子模块按照目录划分如下：

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

