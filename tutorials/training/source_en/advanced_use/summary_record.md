# Collecting Summary Record

`Linux` `Ascend` `GPU` `CPU` `Model Optimization` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/advanced_use/summary_record.md)

## Overview

Scalars, images, computational graphs, and model hyperparameters during training are recorded in files and can be viewed on the web page.

## Operation Process

- Prepare a training script, specify scalars, images, computational graphs, and model hyperparameters in the training script, record them in the summary log file, and run the training script.
- Start MindInsight and specify the summary log file directory using startup parameters. After MindInsight is started, access the visualization page based on the IP address and port number. The default access IP address is `http://127.0.0.1:8080`.
- During the training, when data is written into the summary log file, you can view the data on the web page.

## Preparing The Training Script

Currently, MindSpore supports to save scalars, images, computational graph, and model hyperparameters to summary log file and display them on the web page.

MindSpore currently supports three ways to record data into summary log file.

### Method one: Automatically collected through SummaryCollector

The `Callback` mechanism in MindSpore provides a quick and easy way to collect common information, including the calculational graph, loss value, learning rate, parameter weights, etc. It is named 'SummaryCollector'.

When you write a training script, you just instantiate the `SummaryCollector` and apply it to either `model.train` or `model.eval`. You can automatically collect some common summary data. `SummaryCollector` detailed usage can reference `API` document `mindspore.train.callback.SummaryCollector`.

The sample code is as follows:

```python
import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.common.initializer import TruncatedNormal
import mindspore.ops as ops
from mindspore.train.callback import SummaryCollector
from mindspore.nn.metrics import Accuracy

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
    return TruncatedNormal(0.02)


class AlexNet(nn.Cell):
    def __init__(self, num_classes=10, channel=3):
        super(AlexNet, self).__init__()
        self.conv1 = conv(channel, 96, 11, stride=4)
        self.conv2 = conv(96, 256, 5, pad_mode="same")
        self.conv3 = conv(256, 384, 3, pad_mode="same")
        self.conv4 = conv(384, 384, 3, pad_mode="same")
        self.conv5 = conv(384, 256, 3, pad_mode="same")
        self.relu = nn.ReLU()
        self.max_pool2d = ops.MaxPool(ksize=3, strides=2)
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
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
lr = Tensor(0.5, mindspore.float32)
opt = nn.Momentum(network.trainable_params(), lr, momentum=0.9)
model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})
ds_train = create_dataset('./dataset_path')

# Init a SummaryCollector callback instance, and use it in model.train or model.eval
summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=1)

# Note: dataset_sink_mode should be set to False, else you should modify collect freq in SummaryCollector
model.train(epoch=1, train_dataset=ds_train, callbacks=[summary_collector], dataset_sink_mode=False)

ds_eval = create_dataset('./dataset_path')
model.eval(ds_eval, callbacks=[summary_collector])
```

### Method two: Custom collection of network data with summary operators and SummaryCollector

In addition to providing the `SummaryCollector` that automatically collects some summary data, MindSpore provides summary operators that enable custom collection other data on the network, such as the input of each convolutional layer, or the loss value in the loss function, etc.

Summary operators currently supported:

- [ScalarSummary](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScalarSummary): Record a scalar data.
- [TensorSummary](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TensorSummary): Record a tensor data.
- [ImageSummary](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ImageSummary): Record a image data.
- [HistogramSummary](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.HistogramSummary): Convert tensor data into histogram data records.

The recording method is shown in the following steps.

Step 1: Call the summary operator in the `construct` function of the derived class that inherits `nn.Cell` to collect image or scalar data.

For example, when a network is defined, image data is recorded in `construct` of the network. When the loss function is defined, the loss value is recorded in `construct` of the loss function.

Record the dynamic learning rate in `construct` of the optimizer when defining the optimizer.

The sample code is as follows:

```python
from mindspore import context, Tensor, nn
from mindspore.common import dtype as mstype
import mindspore.ops as ops
from mindspore.nn import Optimizer


class CrossEntropyLoss(nn.Cell):
    """Loss function definition."""
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()
        self.mean = ops.ReduceMean()
        self.one_hot = ops.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

        # Init ScalarSummary
        self.scalar_summary = ops.ScalarSummary()

    def construct(self, logits, label):
        label = self.one_hot(label, ops.shape(logits)[1], self.on_value, self.off_value)
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
        self.scalar_summary = ops.ScalarSummary()
        self.histogram_summary = ops.HistogramSummary()
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
        self.image_summary = ops.ImageSummary()
        # Init TensorSummary
        self.tensor_summary = ops.TensorSummary()

    def construct(self, data):
        # Record image by Summary operator
        self.image_summary("image", data)
        # Record tensor by Summary operator
        self.tensor_summary("tensor", data)
        ......
        return out
```

> In the same Summary operator, the name given to the data must not be repeated, otherwise the data collection and presentation will have unexpected behavior.
> For example, if two `ScalarSummary` operators are used to collect scalar data, two scalars cannot be given the same name.

Step 2: In the training script, instantiate the `SummaryCollector` and apply it to `model.train`.

The sample code is as follows:

```python
from mindspore import Model, nn, context
from mindspore.train.callback import SummaryCollector
......

context.set_context(mode=context.GRAPH_MODE)
net = Net()
loss_fn = CrossEntropyLoss()
optim = MyOptimizer(learning_rate=0.01, params=network.trainable_params())
model = Model(net, loss_fn=loss_fn, optimizer=optim, metrics={"Accuracy": Accuracy()})

train_ds = create_mindrecord_dataset_for_training()

summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=1)
model.train(epoch=2, train_dataset=train_ds, callbacks=[summary_collector])
```

### Method three: Custom callback recording data

MindSpore supports custom callback and support to record data into summary log file
in custom callback, and display the data by the web page.

The following pseudocode is shown in the CNN network, where developers can use the network output with the original tag and the prediction tag to generate the image of the confusion matrix.
It is then recorded into the summary log file through the `SummaryRecord` module.
`SummaryRecord` detailed usage can reference `API` document `mindspore.train.summary.SummaryRecord`.

The sample code is as follows:

```python
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

The above three ways, support the record computational graph, loss value and other data. In addition, MindSpore also supports the saving of computational graph for other phases of training, through
the `save_graphs` option of `context.set_context` in the training script is set to `True` to record computational graphs of other phases, including the computational graph after operator fusion.

In the saved files, `ms_output_after_hwopt.pb` is the computational graph after operator fusion, which can be viewed on the web page.

## Run MindInsight

After completing the data collection in the tutorial above, you can start MindInsight to visualize the collected data. When start MindInsight, you need to specify the summary log file directory with the `--summary-base-dir` parameter.

The specified summary log file directory can be the output directory of a training or the parent directory of the output directory of multiple training.

The output directory structure for a training is as follows

```text
└─summary_dir
    events.out.events.summary.1596869898.hostname_MS
    events.out.events.summary.1596869898.hostname_lineage
```

Start command:

```Bash
mindinsight start --summary-base-dir ./summary_dir
```

The output directory structure of multiple training is as follows:

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

Start command:

```Bash
mindinsight start --summary-base-dir ./summary
```

After successful startup, the visual page can be viewed by visiting the `http://127.0.0.1:8080` address through the browser.

Stop MindInsight command:

```Bash
mindinsight stop
```

For more parameter Settings, see the [MindInsight related commands](https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/mindinsight_commands.html) page.

## Notices

1. To limit time of listing summaries, MindInsight lists at most 999 summary items.

2. Multiple `SummaryRecord` instances can not be used at the same time. (`SummaryRecord` is used in `SummaryCollector`)

    If you use two or more instances of `SummaryCollector` in the callback list of 'model.train' or 'model.eval', it is seen as using multiple `SummaryRecord` instances at the same time, and it will cause recoding data fail.

    If the custom callback use `SummaryRecord`, it can not be used with `SummaryCollector` at the same time.

    Right code:

    ```python
    ...
    summary_collector = SummaryCollector('./summary_dir')
    model.train(2, train_dataset, callbacks=[summary_collector])
    ...
    model.eval(dataset， callbacks=[summary_collector])
    ```

    Wrong code:

    ```python
    ...
    summary_collector1 = SummaryCollector('./summary_dir1')
    summary_collector2 = SummaryCollector('./summary_dir2')
    model.train(2, train_dataset, callbacks=[summary_collector1, summary_collector2])
    ```

    Wrong code:

    ```python
    ...
    # Note: the 'ConfusionMatrixCallback' is user-defined, and it uses SummaryRecord to record data.
    confusion_callback = ConfusionMatrixCallback('./summary_dir1')
    summary_collector = SummaryCollector('./summary_dir2')
    model.train(2, train_dataset, callbacks=[confusion_callback, summary_collector])
    ```

3. In each Summary log file directory, only one training data should be placed. If a summary log directory contains summary data from multiple training, MindInsight will overlay the summary data from these training when visualizing the data, which may not be consistent with the expected visualizations.

4. Currently, `SummaryCollector` and `SummaryRecord` do not support scenarios with GPU multi-card running.
