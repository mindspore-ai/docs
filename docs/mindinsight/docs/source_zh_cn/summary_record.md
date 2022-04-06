# 收集Summary数据

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindinsight/docs/source_zh_cn/summary_record.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

训练过程中的标量、图像、计算图、训练优化过程以及模型超参等信息记录到文件中，通过可视化界面供用户查看。

## 操作流程

- 准备训练脚本，并在训练脚本中指定标量、图像、计算图、训练优化过程、模型超参等信息记录到summary日志文件，接着运行训练脚本。
- 启动MindInsight，并通过启动参数指定summary日志文件目录，启动成功后，根据IP和端口访问可视化界面，默认访问地址为 `http://127.0.0.1:8080`。
- 在训练过程中，有数据写入summary日志文件时，即可在页面中[查看训练看板中可视的数据](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/dashboard.html)。

> 在ModelArts中查看可视数据，可参考[ModelArts上管理可视化作业](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0050.html)。

## 准备训练脚本

当前MindSpore支持将标量、图像、计算图、训练优化过程、模型超参等信息保存到summary日志文件中，并通过可视化界面进行展示。计算图数据仅能在图模式下记录，训练优化过程数据收集及地形图绘制的详细流程可参考[训练优化过程可视化](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/landscape.html)。

MindSpore目前支持多种方式将数据记录到summary日志文件中。

### 方式一：通过SummaryCollector自动收集

在MindSpore中通过 `Callback` 机制提供支持快速简易地收集一些常见的信息，包括计算图，损失值，学习率，参数权重等信息的 `Callback`, 叫做 `SummaryCollector`。

在编写训练脚本时，仅需要实例化 `SummaryCollector`，并将其应用到 `model.train` 或者 `model.eval` 中，
即可自动收集一些常见信息。`SummaryCollector` 详细的用法可以参考 `API` 文档中 [mindspore.train.callback.SummaryCollector](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.train.html#mindspore.train.callback.SummaryCollector) 。

下面仅展示使用SummaryCollector自动收集数据的关键代码，[完整样例代码](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/mindinsight/summary_record/summary_1.py) 可以到gitee下载。

```python

def train(ds_train):
    ...
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    # Init a SummaryCollector callback instance, and use it in model.train or model.eval
    specified = {"collect_metric": True, "histogram_regular": "^conv1.*|^conv2.*", "collect_graph": True,
                 "collect_dataset_graph": True}

    summary_collector = SummaryCollector(summary_dir="./summary_dir/summary_01", collect_specified_data=specified,
                                         collect_freq=1, keep_default_action=False, collect_tensor_freq=200)

    print("============== Starting Training ==============")
    model.train(epoch=1, train_dataset=ds_train, callbacks=[time_cb, LossMonitor(), summary_collector],
                dataset_sink_mode=False)

```

> 1. 使用summary功能时，建议将`model.train`的`dataset_sink_mode`参数设置为`False`。请参考文末的注意事项。
> 2. 使用summary功能时，需要将代码放置到`if __name__ == "__main__"`中运行。详情请[参考Python官网介绍](https://docs.python.org/zh-cn/3.7/library/multiprocessing.html#multiprocessing-programming)。
> 3. dataset_path为用户本地的训练数据集路径。

### 方式二：结合Summary算子和SummaryCollector，自定义收集网络中的数据

MindSpore除了提供 `SummaryCollector` 能够自动收集一些常见数据，还提供了Summary算子，支持在网络中自定义收集其他的数据，比如每一个卷积层的输入，或在损失函数中的损失值等。

当前支持的Summary算子:

- [ScalarSummary](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScalarSummary.html)：记录标量数据
- [TensorSummary](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.TensorSummary.html)：记录张量数据
- [ImageSummary](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ImageSummary.html)：记录图片数据
- [HistogramSummary](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.HistogramSummary.html)：将张量数据转为直方图数据记录

记录方式如下面的步骤所示，[完整样例代码](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/mindinsight/summary_record/summary_2.py) 可以到gitee下载。

步骤一：在继承 `nn.Cell` 的衍生类的 `construct` 函数中调用Summary算子来采集图像或标量数据或者其他数据。

比如，定义网络时，在网络的 `construct` 中记录图像数据；定义损失函数时，在损失函数的 `construct`中记录损失值。

如果要记录动态学习率，可以定义优化器时，在优化器的 `construct` 中记录学习率。

样例代码如下：

```python
class AlexNet(nn.Cell):
    """
    Alexnet
    """
    def __init__(self, num_classes=10, channel=3):
        super(AlexNet, self).__init__()
        self.conv1 = conv(channel, 96, 11, stride=4)
        ...
        # Init TensorSummary
        self.tensor_summary = ops.TensorSummary()
        # Init ImageSummary
        self.image_summary = ops.ImageSummary()

    def construct(self, x):
        # Record image by Summary operator
        self.image_summary("Image", x)
        x = self.conv1(x)
        # Record tensor by Summary operator
        self.tensor_summary("Tensor", x)
        ...
        return x
```

> 1. 同一种Summary算子中，给数据设置的名字不能重复，否则数据收集和展示都会出现非预期行为。比如使用两个 `ScalarSummary` 算子收集标量数据，给两个标量设置的名字不能是相同的。
> 2. summary算子仅支持图模式，需要在`nn.Cell`的`construct`中使用。暂不支持PyNative模式。

步骤二：在训练脚本中，实例化 `SummaryCollector`，并将其应用到 `model.train`。

样例代码如下：

```python

def train(ds_train):
    ...
    # Init a SummaryCollector callback instance, and use it in model.train or model.eval
    specified = {"collect_metric": True, "histogram_regular": "^conv1.*|^conv2.*", "collect_graph": True,
                 "collect_dataset_graph": True}

    summary_collector = SummaryCollector(summary_dir="./summary_dir/summary_02", collect_specified_data=specified,
                                         collect_freq=1, keep_default_action=False, collect_tensor_freq=200)

    print("============== Starting Training ==============")
    model.train(epoch=1, train_dataset=ds_train, callbacks=[time_cb, LossMonitor(), summary_collector],
                dataset_sink_mode=False)

```

### 方式三：自定义Callback记录数据

MindSpore支持自定义Callback, 并允许在自定义Callback中将数据记录到summary日志文件中，
并通过可视化页面进行查看。

下面的样例代码则展示在自定义Callback函数中通过 `SummaryRecord` 模块记录网络输入到summary日志文件中。
`SummaryRecord` 详细的用法可以参考 `API` 文档中 [mindspore.train.summary.SummaryRecord](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.train.html#mindspore.train.summary.SummaryRecord) 。

下面展示了自定义Callback记录数据的关键样例代码，[完整样例代码](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/mindinsight/summary_record/summary_3.py) 可以到gitee下载。

```python
class MyCallback(Callback):
    def __init__(self, summary_dir):
        self._summary_dir = summary_dir

    def __enter__(self):
        # init you summary record in here, when the train script run, it will be inited before training
        self.summary_record = SummaryRecord(self._summary_dir)
        return self

    def __exit__(self, *exc_args):
        # Note: you must close the summary record, it will release the process pool resource
        # else your training script will not exit from training.
        self.summary_record.close()

    def step_end(self, run_context):
        cb_params = run_context.original_args()

        # create a confusion matric image, and record it to summary file
        self.summary_record.add_value('image', 'image0', cb_params.train_dataset_element[0])
        self.summary_record.add_value('scalar', 'loss', cb_params.net_outputs)
        self.summary_record.record(cb_params.cur_step_num)


def train(ds_train):
    ...
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    # Init a specified callback instance, and use it in model.train or model.eval
    specified_callback = MyCallback(summary_dir='./summary_dir/summary_03')

    print("============== Starting Training ==============")
    model.train(epoch=1, train_dataset=ds_train, callbacks=[time_cb, LossMonitor(), specified_callback],
                dataset_sink_mode=False)

```

上面的三种方式，支持记录计算图, 损失值等多种数据。除此以外，MindSpore还支持保存训练中其他阶段的计算图，通过
将训练脚本中 `context.set_context` 的 `save_graphs` 选项设置为 `True`, 可以记录其他阶段的计算图，其中包括算子融合后的计算图。

在保存的文件中，`ms_output_after_hwopt.pb` 即为算子融合后的计算图，可以使用可视化页面对其进行查看。

### 方式四：进阶用法，自定义训练循环

如果训练时不是使用MindSpore提供的 `Model` 接口，而是模仿 `Model` 的 `train` 接口自由控制循环的迭代次数。则可以模拟 `SummaryCollector`，使用下面的方式记录summary算子数据。详细的自定义训练循环教程，请参考[构建训练与评估网络](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/train_and_eval.html)。

下面的例子，将演示如何使用summary算子以及 `SummaryRecord` 的 `add_value` 接口在自定义训练循环中记录数据。[完整样例代码](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/mindinsight/summary_record/summary_4.py) 可以到gitee下载。
更多 `SummaryRecord` 的教程，请[参考Python API文档](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.train.html#mindspore.train.summary.SummaryRecord)。需要说明的是，`SummaryRecord`不会自动记录计算图，您需要手动传入继承了`Cell`的网络实例以记录计算图。此外，生成计算图的内容仅包含您在`construct`方法中使用到的代码和函数。

```python
def train(ds_train):
    ...
    summary_collect_frequency = 200
    # Note1: An instance of the network should be passed to SummaryRecord if you want to record
    # computational graph.
    with SummaryRecord('./summary_dir/summary_04', network=train_net) as summary_record:
        for epoch in range(epochs):
            step = 1
            for inputs in ds_train:
                output = train_net(*inputs)
                current_step = epoch * ds_train.get_dataset_size() + step
                print("step: {0}, losses: {1}".format(current_step, output.asnumpy()))

                # Note2: The output should be a scalar, and use 'add_value' method to record loss.
                # Note3: You must use the 'record(step)' method to record the data of this step.
                if current_step % summary_collect_frequency == 0:
                    summary_record.add_value('scalar', 'loss', output)
                    summary_record.record(current_step)

                step += 1

```

### 分布式训练场景

由于`SummaryCollector`和`SummaryRecord`写数据是非进程安全的。所以在单机多卡的场景中，需要确保每张卡保存数据的目录不一样。在分布式场景下，我们通过`get_rank`函数设置summary目录。

```python
summary_dir = "summary_dir" + str(get_rank())
```

示例代码如下：

```python
from mindspore.communication import get_rank

...

network = ResNet50(num_classes=10)

# Init a SummaryCollector callback instance, and use it in model.train or model.eval
summary_dir = "summary_dir" + str(get_rank())
summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=1)

# Note: dataset_sink_mode should be set to False, else you should modify collect freq in SummaryCollector
model.train(epoch=1, train_dataset=ds_train, callbacks=[summary_collector], dataset_sink_mode=False)

model.eval(ds_eval, callbacks=[summary_collector])
```

### 使用技巧：记录梯度信息

除了上述使用方式外，使用summary算子时还有一个记录梯度信息的技巧。请注意此技巧需要和上述的某一种使用方式同时使用。

通过继承原有优化器类的方法可以插入summary算子读取梯度信息。[完整样例代码](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/mindinsight/summary_record/summary_gradients.py) 可以到gitee下载。样例代码片段如下：

```python
import mindspore.nn as nn
import mindspore.ops as ops
...

# Define a new optimizer class by inheriting your original optimizer.
class MyOptimizer(nn.Momentum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_construct = super().construct
        self.histogram_summary = ops.HistogramSummary()
        self.gradient_names = [param.name + ".gradient" for param in self.parameters]

    def construct(self, grads):
        # Record gradient.
        l = len(self.gradient_names)
        for i in range(l):
            self.histogram_summary(self.gradient_names[i], grads[i])
        return self._original_construct(grads)

...

def train(ds_train):
    device_target = "GPU"
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    network = AlexNet(num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    lr = Tensor(get_lr(0, 0.002, 10, ds_train.get_dataset_size()))
    net_opt = MyOptimizer(network.trainable_params(), learning_rate=lr, momentum=0.9)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    # Init a SummaryCollector callback instance, and use it in model.train or model.eval
    summary_collector = SummaryCollector(summary_dir="./summary_dir/summary_gradients",
                                         collect_freq=200, keep_default_action=False, collect_tensor_freq=200)

    print("============== Starting Training ==============")
    model.train(epoch=1, train_dataset=ds_train, callbacks=[time_cb, LossMonitor(), summary_collector],
                dataset_sink_mode=False)
```

## 运行MindInsight

按照上面教程完成数据收集后，启动MindInsight，即可可视化收集到的数据。启动MindInsight时，
需要通过 `--summary-base-dir` 参数指定summary日志文件目录。

其中指定的summary日志文件目录可以是一次训练的输出目录，也可以是多次训练输出目录的父目录。

一次训练的输出目录结构如下：

```text
└─summary_dir
    events.out.events.summary.1596869898.hostname_MS
    events.out.events.summary.1596869898.hostname_lineage
```

启动命令：

```Bash
mindinsight start --summary-base-dir ./summary_dir
```

多次训练的输出目录结构如下：

```text
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

更多参数设置，请点击查看[MindInsight相关命令](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/mindinsight_commands.html)页面。

## 注意事项

1. 为了控制列出summary文件目录的用时，MindInsight最多支持发现999个summary文件目录。

2. 不能同时使用多个 `SummaryRecord` 实例 （`SummaryCollector` 中使用了 `SummaryRecord`）。

    如果在 `model.train` 或者 `model.eval` 的callback列表中使用两个及以上的 `SummaryCollector` 实例，则视为同时使用 `SummaryRecord`，可能导致记录数据失败。

    自定义callback中如果使用 `SummaryRecord`，则其不能和 `SummaryCollector` 同时使用。

    正确代码:

    ```python
    ...
    summary_collector = SummaryCollector('./summary_dir')
    model.train(2, train_dataset, callbacks=[summary_collector])

    ...
    model.eval(dataset, callbacks=[summary_collector])
    ```

    错误代码：

    ```python
    ...
    summary_collector1 = SummaryCollector('./summary_dir1')
    summary_collector2 = SummaryCollector('./summary_dir2')
    model.train(2, train_dataset, callbacks=[summary_collector1, summary_collector2])
    ```

    错误代码：

    ```python
    ...
    # Note: the 'ConfusionMatrixCallback' is user-defined, and it uses SummaryRecord to record data.
    confusion_callback = ConfusionMatrixCallback('./summary_dir1')
    summary_collector = SummaryCollector('./summary_dir2')
    model.train(2, train_dataset, callbacks=[confusion_callback, summary_collector])
    ```

3. 每个summary日志文件目录中，应该只放置一次训练的数据。一个summary日志目录中如果存放了多次训练的summary数据，MindInsight在可视化数据时会将这些训练的summary数据进行叠加展示，可能会与预期可视化效果不相符。

4. 使用summary功能时，建议将`model.train`方法的`dataset_sink_mode`参数设置为`False`，从而以`step`作为`collect_freq`参数的单位收集数据。当`dataset_sink_mode`为`True`时，将以`epoch`作为`collect_freq`的单位，此时建议手动设置`collect_freq`参数。`collect_freq`参数默认值为`10`。

5. 每个step保存的数据量，最大限制为2147483647Bytes。如果超出该限制，则无法记录该step的数据，并出现错误。

6. PyNative模式下，`SummaryCollector` 能够正常使用，但不支持记录计算图以及不支持使用Summary算子。
