# Summary数据收集

`Ascend` `GPU` `CPU` `模型调优` `中级` `高级`

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.7/tutorials/source_zh_cn/advanced_use/summary_record.md)&nbsp;&nbsp;
[![查看Notebook](../_static/logo_notebook.png)](https://gitee.com/mindspore/docs/tree/r0.7/tutorials/notebook/mindinsight)

## 概述

训练过程中的标量、图像、计算图以及模型超参等信息记录到文件中，通过可视化界面供用户查看。

## 操作流程

- 准备训练脚本，并在训练脚本中指定标量、图像、计算图、模型超参等信息记录到summary日志文件，接着运行训练脚本。
- 启动MindInsight，并通过启动参数指定summary日志文件目录，启动成功后，根据IP和端口访问可视化界面，默认访问地址为 `http://127.0.0.1:8080`。
- 在训练过程中，有数据写入summary日志文件时，即可在页面中查看可视的数据。

## 准备训练脚本

当前MindSpore支持将标量、图像、计算图、模型超参等信息保存到summary日志文件中，并通过可视化界面进行展示。

MindSpore目前支持三种方式将数据记录到summary日志文件中。

### 方式一：通过SummaryCollector自动收集

在MindSpore中通过 `Callback` 机制提供支持快速简易地收集一些常见的信息，包括计算图，损失值，学习率，参数权重等信息的 `Callback`, 叫做 `SummaryCollector`。

在编写训练脚本时，仅需要实例化 `SummaryCollector`，并将其应用到 `model.train` 或者 `model.eval` 中，
即可自动收集一些常见信息。`SummaryCollector` 详细的用法可以参考 `API` 文档中 `mindspore.train.callback.SummaryCollector`。

样例代码如下：
```python
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.train.callback import SummaryCollector

"""AlexNet initial."""
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
    return TruncatedNormal(0.02)  # 0.02


class AlexNet(nn.Cell):
    def __init__(self, num_classes=10, channel=3):
        super(AlexNet, self).__init__()
        self.conv1 = conv(channel, 96, 11, stride=4)
        self.conv2 = conv(96, 256, 5, pad_mode="same")
        self.conv3 = conv(256, 384, 3, pad_mode="same")
        self.conv4 = conv(384, 384, 3, pad_mode="same")
        self.conv5 = conv(384, 256, 3, pad_mode="same")
        self.relu = nn.ReLU()
        self.max_pool2d = P.MaxPool(ksize=3, strides=2)
        self.flatten = nn.Flatten()
        self.fc1 = fc_with_initialize(6*6*256, 4096)
        self.fc2 = fc_with_initialize(4096, 4096)
        self.fc3 = fc_with_initialize(4096, num_classes)

    def construct(self, x):
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

context.set_context(mode=context.GRAPH_MODE)

network = AlexNet(num_classes=10)
loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
lr = Tensor(0.1)
opt = nn.Momentum(network.trainable_params(), lr, momentum=0.9)
model = Model(network, loss, opt)
ds_train = create_dataset('./dataset_path')

# Init a SummaryCollector callback instance, and use it in model.train or model.eval
summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=1)

# Note: dataset_sink_mode should be set to False, else you should modify collect freq in SummaryCollector
model.train(epoch=1, ds_train, callbacks=[summary_collector], dataset_sink_mode=False)

ds_eval = create_dataset('./dataset_path')
model.eval(ds_eval, callbacks=[summary_collector])
```

### 方式二：结合Summary算子和SummaryCollector，自定义收集网络中的数据

MindSpore除了提供 `SummaryCollector` 能够自动收集一些常见数据，还提供了Summary算子，支持在网络中自定义收集其他的数据，比如每一个卷积层的输入，或在损失函数中的损失值等。

当前支持的Summary算子:
- [ScalarSummary](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html?highlight=scalarsummary#mindspore.ops.operations.ScalarSummary): 记录标量数据
- [TensorSummary](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html?highlight=tensorsummary#mindspore.ops.operations.TensorSummary): 记录张量数据
- [ImageSummary](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html?highlight=imagesummary#mindspore.ops.operations.ImageSummary): 记录图片数据
- [HistogramSummary](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html?highlight=histogramsummar#mindspore.ops.operations.HistogramSummary): 将张量数据转为直方图数据记录

记录方式如下面的步骤所示。

步骤一：在继承 `nn.Cell` 的衍生类的 `construct` 函数中调用Summary算子来采集图像或标量数据或者其他数据。

比如，定义网络时，在网络的 `construct` 中记录图像数据；定义损失函数时，在损失函数的 `construct`中记录损失值。

如果要记录动态学习率，可以定义优化器时，在优化器的 `construct` 中记录学习率。

样例代码如下：

```python
from mindspore import context, Tensor, nn
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn import Optimizer


class CrossEntropyLoss(nn.Cell):
    """Loss function definition."""
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

        # Init ScalarSummary
        self.scalar_summary = P.ScalarSummary()

    def construct(self, logits, label):
        label = self.one_hot(label, F.shape(logits)[1], self.on_value, self.off_value)
        loss = self.cross_entropy(logits, label)[0]
        loss = self.mean(loss, (-1,))

        # Record loss
        self.scalar_summary("loss", loss)
        return loss


class MyOptimizer(Optimizer):
    """Optimizer definition."""
    def __init__(self, learning_rate, params, ......):
        ......
        # Initialize ScalarSummary
        self.scalar_summary = P.ScalarSummary()
        self.histogram_summary = P.HistogramSummary()
        self.weight_names = [param.name for param in self.parameters]

    def construct(self, grads):
        ......
        # Record learning rate here
        self.scalar_summary("learning_rate", learning_rate)

        # Record weight
        self.histogram_summary(self.weight_names[0], self.paramters[0])
        # Record gradient
        self.histogram_summary(self.weight_names[0] + ".gradient", grads[0])

        ......

class Net(nn.Cell):
    """Net definition."""
    def __init__(self):
        super(Net, self).__init__()
        ......

        # Init ImageSummary
        self.image_summary = P.ImageSummary()
        # Init TensorSummary
        self.tensor_summary = P.TensorSummary()

    def construct(self, data):
        # Record image by Summary operator
        self.image_summary("image", data)
        # Record tensor by Summary operator
        self.tensor_summary("tensor", data)
        ......
        return out

```

> 同一种Summary算子中，给数据设置的名字不能重复，否则数据收集和展示都会出现非预期行为。
> 比如使用两个 `ScalarSummary` 算子收集标量数据，给两个标量设置的名字不能是相同的。

步骤二：在训练脚本中，实例化 `SummaryCollector`，并将其应用到 `model.train`。

样例代码如下：

```python
from mindspore import Model, nn, context
from mindspore.train.callback import SummaryCollector

context.set_context(mode=context.GRAPH_MODE)
net = Net()
loss_fn = CrossEntropyLoss()
optim = MyOptimizer(learning_rate=0.01, params=network.trainable_params())
model = Model(net, loss_fn=loss_fn, optimizer=optim, metrics=None)

train_ds = create_mindrecord_dataset_for_training()

summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=1)
model.train(epoch=2, train_ds, callbacks=[summary_collector])
```

### 方式三：自定义Callback记录数据

MindSpore支持自定义Callback, 并允许在自定义Callback中将数据记录到summary日志文件中，
并通过可视化页面进行查看。

下面的伪代码则展示在CNN网络中，开发者可以利用带有原始标签和预测标签的网络输出，生成混淆矩阵的图片,
然后通过 `SummaryRecord` 模块记录到summary日志文件中。
`SummaryRecord` 详细的用法可以参考 `API` 文档中 `mindspore.train.summary.SummaryRecord`。

样例代码如下：

```
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord

class ConfusionMatrixCallback(Callback):
    def __init__(self, summary_dir):
        self._summary_dir = summary_dir
    
    def __enter__(self):
        # init you summary record in here, when the train script run, it will be inited before training
        self.summary_record = SummaryRecord(summary_dir)
    
    def __exit__(self, *exc_args):
        # Note: you must close the summary record, it will release the process pool resource
        # else your training script will not exit from training.
        self.summary_record.close()
        return self

    def step_end(self, run_context):
        cb_params = run_context.run_context.original_args()

        # create a confusion matric image, and record it to summary file
        confusion_martrix = create_confusion_matrix(cb_params)        
        self.summary_record.add_value('image', 'confusion_matrix', confusion_matric)
        self.summary_record.record(cb_params.cur_step)

# init you train script
...

confusion_martrix = ConfusionMartrixCallback(summary_dir='./summary_dir')
model.train(cnn_network, callbacks=[confusion_martrix])
```

上面的三种方式，支持记录计算图, 损失值等多种数据。除此以外，MindSpore还支持保存训练中其他阶段的计算图，通过
将训练脚本中 `context.set_context` 的 `save_graphs` 选项设置为 `True`, 可以记录其他阶段的计算图，其中包括算子融合后的计算图。 

在保存的文件中，`ms_output_after_hwopt.pb` 即为算子融合后的计算图，可以使用可视化页面对其进行查看。

## 运行MindInsight
按照上面教程完成数据收集后，启动MindInsight，即可可视化收集到的数据。启动MindInsight时，
需要通过 `--summary-base-dir` 参数指定summary日志文件目录。

其中指定的summary日志文件目录可以是一次训练的输出目录，也可以是多次训练输出目录的父目录。


一次训练的输出目录结构如下：
```
└─summary_dir
    events.out.events.summary.1596869898.hostname_MS
    events.out.events.summary.1596869898.hostname_lineage
```

启动命令：
```Bash
mindinsight start --summary-base-dir ./summary_dir
```

多次训练的输出目录结构如下：
```
└─summary
    ├─summary_dir1
    │      events.out.events.summary.1596869898.hostname_MS
    │      events.out.events.summary.1596869898.hostname_lineage
    │
    └─summary_dir2
            events.out.events.summary.1596869998.hostname_MS
            events.out.events.summary.1596869998.hostname_lineage
```

启动命令:
```Bash
mindinsight start --summary-base-dir ./summary
```

启动成功后，通过浏览器访问 `http://127.0.0.1:8080` 地址，即可查看可视化页面。

停止MindInsight命令：
```Bash
mindinsight stop
```

更多参数设置，请点击查看[MindInsight相关命令](https://www.mindspore.cn/tutorial/zh-CN/r0.7/advanced_use/mindinsight_commands.html)页面。


## 注意事项

1. 为了控制列出summary文件目录的用时，MindInsight最多支持发现999个summary文件目录。

2. 不能同时使用多个 `SummaryRecord` 实例 （`SummaryCollector` 中使用了 `SummaryRecord`）。

    如果在 `model.train` 或者 `model.eval` 的callback列表中使用两个及以上的 `SummaryCollector` 实例，则视为同时使用 `SummaryRecord`，导致记录数据失败。

    自定义callback中如果使用 `SummaryRecord`，则其不能和 `SummaryCollector` 同时使用。

    正确代码:
    ```python3
    ...
    summary_collector = SummaryCollector('./summary_dir')
    model.train(epoch=2, train_dataset, callbacks=[summary_collector])

    ...
    model.eval(dataset， callbacks=[summary_collector])
    ```

    错误代码：
    ```python3
    ...
    summary_collector1 = SummaryCollector('./summary_dir1')
    summary_collector2 = SummaryCollector('./summary_dir2')
    model.train(epoch=2, train_dataset, callbacks=[summary_collector1, summary_collector2])
    ```

    错误代码：
    ```python3
    ...
    # Note: the 'ConfusionMatrixCallback' is user-defined, and it uses SummaryRecord to record data.
    confusion_callback = ConfusionMatrixCallback('./summary_dir1')
    summary_collector = SummaryCollector('./summary_dir2')
    model.train(epoch=2, train_dataset, callbacks=[confusion_callback, summary_collector])
    ```

3. 每个summary日志文件目录中，应该只放置一次训练的数据。一个summary日志目录中如果存放了多次训练的summary数据，MindInsight在可视化数据时会将这些训练的summary数据进行叠加展示，可能会与预期可视化效果不相符。