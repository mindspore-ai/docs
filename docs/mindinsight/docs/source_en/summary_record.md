# Collecting Summary Record

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindinsight/docs/source_en/summary_record.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

Scalars, images, computational graphs, and model hyperparameters during training are recorded in files and can be viewed on the web page.

## Operation Process

- Prepare a training script, specify scalars, images, computational graphs, and model hyperparameters in the training script, record them in the summary log file, and run the training script.
- Start MindInsight and specify the summary log file directory using startup parameters. After MindInsight is started, access the visualization page based on the IP address and port number. The default access IP address is `http://127.0.0.1:8080`.
- During the training, when data is written into the summary log file, you can view the data on the web page.

## Preparing The Training Script

Currently, MindSpore supports to save scalars, images, computational graph, and model hyperparameters to summary log file and display them on the web page. The computational graph can only be recorded in the graph mode.

MindSpore currently supports multiple ways to record data into summary log files.

### Method one: Automatically collected through SummaryCollector

The `Callback` mechanism in MindSpore provides a quick and easy way to collect common information, including the calculational graph, loss value, learning rate, parameter weights, etc. It is named 'SummaryCollector'.

When you write a training script, you just instantiate the `SummaryCollector` and apply it to either `model.train` or `model.eval`. You can automatically collect some common summary data. The detailed usage of `SummaryCollector` can refer to the `API` document `mindspore.train.callback.SummaryCollector`.

The sample code is as follows:

```python
import mindspore
import mindspore.nn as nn
from mindspore import ops
from mindspore import context, Tensor, Model
from mindspore.nn import Accuracy
from mindspore.train.callback import SummaryCollector


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     has_bias=has_bias, pad_mode=pad_mode)


def fc_with_initialize(input_channels, out_channels, has_bias=True):
    return nn.Dense(input_channels, out_channels, has_bias=has_bias)


class AlexNet(nn.Cell):
    """AlexNet"""
    def __init__(self, num_classes=10, channel=3, phase='train', include_top=True):
        super(AlexNet, self).__init__()
        self.conv1 = conv(channel, 64, 11, stride=4, pad_mode="same", has_bias=True)
        self.conv2 = conv(64, 128, 5, pad_mode="same", has_bias=True)
        self.conv3 = conv(128, 192, 3, pad_mode="same", has_bias=True)
        self.conv4 = conv(192, 256, 3, pad_mode="same", has_bias=True)
        self.conv5 = conv(256, 256, 3, pad_mode="same", has_bias=True)
        self.relu = ops.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.include_top = include_top
        if self.include_top:
            dropout_ratio = 0.65
            if phase == 'test':
                dropout_ratio = 1.0
            self.flatten = nn.Flatten()
            self.fc1 = fc_with_initialize(6 * 6 * 256, 4096)
            self.fc2 = fc_with_initialize(4096, 4096)
            self.fc3 = fc_with_initialize(4096, num_classes)
            self.dropout = nn.Dropout(dropout_ratio)

    def construct(self, x):
        """define network"""
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
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train():
    context.set_context(mode=context.GRAPH_MODE)

    network = AlexNet(num_classes=10)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    lr = Tensor(0.5, mindspore.float32)
    opt = nn.Momentum(network.trainable_params(), lr, momentum=0.9)
    model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})

    # How to create a valid dataset instance,
    # for details, see the https://www.mindspore.cn/tutorials/en/master/quick_start.html document.
    ds_train = create_dataset('./dataset_path')

    # Initialize a SummaryCollector callback instance, and use it in model.train or model.eval
    summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=1)

    # Note: dataset_sink_mode should be set to False, else you should modify collect_freq in SummaryCollector
    model.train(epoch=1, train_dataset=ds_train, callbacks=[summary_collector], dataset_sink_mode=False)

    ds_eval = create_dataset('./dataset_path')
    model.eval(ds_eval, callbacks=[summary_collector])

if __name__ == '__main__':
    train()
```

> 1. When using summary, it is recommended that you set `dataset_sink_mode` argument of `model.train` to `False`. Please see notices for more information.
> 2. When using summary, you need to run the code in `if __name__ == "__main__"`. For more detail, refer to [Python tutorial](https://docs.python.org/3.7/library/multiprocessing.html#multiprocessing-programming).
> 3. dataset_path is the path to the user's local training dataset.

### Method two: Custom collection of network data with summary operators and SummaryCollector

In addition to providing the `SummaryCollector` that automatically collects some summary data, MindSpore provides summary operators that enable customized collection of other data on the network, such as the input of each convolutional layer, or the loss value in the loss function, etc.

The following summary operators are currently supported:

- [ScalarSummary](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.ScalarSummary.html): Record a scalar data.
- [TensorSummary](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.TensorSummary.html): Record a tensor data.
- [ImageSummary](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.ImageSummary.html): Record a image data.
- [HistogramSummary](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.HistogramSummary.html): Convert tensor data into histogram data records.

The recording method is shown in the following steps.

Step 1: Call the summary operator in the `construct` function of the derived class that inherits `nn.Cell` to collect image or scalar data.

For example, when a network is defined, image data is recorded in `construct` of the network. When the loss function is defined, the loss value is recorded in `construct` of the loss function.

Record the dynamic learning rate in `construct` of the optimizer when defining the optimizer.

The sample code is as follows:

```python
import mindspore
import mindspore.ops as ops
from mindspore import Tensor, nn
from mindspore.nn import Optimizer


class CrossEntropyLoss(nn.Cell):
    """Loss function definition."""
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()
        self.mean = ops.ReduceMean()
        self.one_hot = ops.OneHot()
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)

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
    def __init__(self, learning_rate, params, ...):
        ...
        # Initialize ScalarSummary
        self.scalar_summary = ops.ScalarSummary()
        self.histogram_summary = ops.HistogramSummary()
        self.weight_names = [param.name for param in self.parameters]

    def construct(self, grads):
        ...
        # Record learning rate here
        self.scalar_summary("learning_rate", learning_rate)

        # Record weight
        self.histogram_summary(self.weight_names[0], self.parameters[0])
        # Record gradient
        self.histogram_summary(self.weight_names[0] + ".gradient", grads[0])

        ...

class Net(nn.Cell):
    """Net definition."""
    def __init__(self):
        super(Net, self).__init__()
        ...

        # Init ImageSummary
        self.image_summary = ops.ImageSummary()
        # Init TensorSummary
        self.tensor_summary = ops.TensorSummary()

    def construct(self, data):
        # Record image by Summary operator
        self.image_summary("image", data)
        # Record tensor by Summary operator
        self.tensor_summary("tensor", data)
        ...
        return out
```

> 1. In the same Summary operator, the name given to the data must not be repeated, otherwise the data collection and presentation will have unexpected behavior.
> For example, if two `ScalarSummary` operators are used to collect scalar data, two scalars cannot be given the same name.
> 2. Summary operator only supports Graph mode and needs to be used in `construct` of `nn.Cell`. The PyNative mode is not supported yet.

Step 2: In the training script, instantiate the `SummaryCollector` and apply it to `model.train`.

The sample code is as follows:

```python
from mindspore import Model, nn, context
from mindspore.train.callback import SummaryCollector
...

def train():
    context.set_context(mode=context.GRAPH_MODE)
    network = Net()
    loss_fn = CrossEntropyLoss()
    optim = MyOptimizer(learning_rate=0.01, params=network.trainable_params())
    model = Model(network, loss_fn=loss_fn, optimizer=optim, metrics={"Accuracy": Accuracy()})

    ds_train = create_dataset('./dataset_path')

    summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=1)
    model.train(epoch=2, train_dataset=ds_train, callbacks=[summary_collector])

if __name__ == '__main__':
    train()
```

### Method three: Custom callback recording data

MindSpore supports customized callback and supports to record data into summary log file
in custom callback, and display the data by the web page.

The following pseudocode is shown in the CNN network, where developers can use the network output with the original tag and the prediction tag to generate the image of the confusion matrix.
It is then recorded into the summary log file through the `SummaryRecord` module.
The detailed usage of `SummaryRecord` can refer to the `API` document `mindspore.train.summary.SummaryRecord`.

The sample code is as follows:

```python
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord

class ConfusionMatrixCallback(Callback):
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
        confusion_matrix = create_confusion_matrix(cb_params)
        self.summary_record.add_value('image', 'confusion_matrix', confusion_matrix)
        self.summary_record.record(cb_params.cur_step_num)

# init you train script
...

confusion_matrix = ConfusionMatrixCallback(summary_dir='./summary_dir')
model.train(epoch=2, train_dataset=ds_train, callbacks=[confusion_matrix])
```

The above three ways support the record computational graph, loss value and other data. In addition, MindSpore also supports the saving of computational graph for other phases of training, through
the `save_graphs` option of `context.set_context` in the training script is set to `True` to record computational graphs of other phases, including the computational graph after operator fusion.

In the saved files, `ms_output_after_hwopt.pb` is the computational graph after operator fusion, which can be viewed on the web page.

### Method four: Advanced usage, custom training cycle

If you are not using the `Model` interface provided by MindSpore, you can implement a method by imitating `train` method of `Model` interface to control the number of iterations. You can imitate the `SummaryCollector` and record the summary operator data in the following manner.

The following example demonstrates how to record data in a custom training cycle using the summary operator and the `add_value` interface of `SummaryRecord`. For more tutorials about `SummaryRecord`, [refer to the Python API documentation](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.train.html#mindspore.train.summary.SummaryRecord). Please note that `SummaryRecord` will not record computational graph automatically. If you need to record the computational graph, please manually pass the instance of network that inherits from Cell. The recorded computational graph only includes the code and functions used in the construct method.

```python
from mindspore import nn
from mindspore.train.summary import SummaryRecord
import mindspore.ops as ops

class LeNet5(nn.Cell):
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        ...

        self.image_summary = ops.ImageSummary()
        self.tensor_summary = ops.TensorSummary()

    def construct(self, x):
        self.image_summary('x1', x)
        x = self.conv1(x)
        self.tensor_summary('after_conv1', x)
        x = self.relu(x)
        ...
        return x

...

def train():
    epochs = 10
    net = LeNet5()
    # Note1: An instance of the network should be passed to SummaryRecord if you want to record
    # computational graph.
    with SummaryRecord('./summary_dir', network=net) as summary_record:
        for epoch in range(epochs):
            step = 1
            for inputs in dataset_helper:
                output = net(*inputs)
                current_step = epoch * len(dataset_helper) + step
                print("step: {0}, losses: {1}".format(current_step, output.asnumpy()))

                # Note2: The output should be a scalar, and use 'add_value' method to record loss.
                # Note3: You must use the 'record(step)' method to record the data of this step.
                summary_record.add_value('scalar', 'loss', output)
                summary_record.record(current_step)

                step += 1

if __name__ == '__main__':
    train()

```

### Distributed Training Scene

The `SummaryCollector` and the `SummaryRecord` are not multi-process safe when writing data, so in a single-machine multi-card scenario, you need to make sure that each card stores data in a different directory. In a distributed scenario, we set the summary directory with the 'get_rank' function.

```python
from mindspore.communication import get_rank
summary_dir = "summary_dir" + str(get_rank())
```

The sample code is as follows:

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

### Tip: Recording gradients

There is a tip for recording gradients with summary in addition to the above methods. Please note that the tip should be used with one of the above methods.

Recording gradients is possible by inheriting your original optimizer and inserting calls to summary operator. An example of code is as follows:

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
        self.histogram_summary(self.gradient_names[0], grads[0])
        return self._original_construct(grads)

...

# Initialize your model with the newly defined optimizer.
model = Model(network, loss_fn=loss_fn, optimizer=MyOptimizer(arg1=arg1value))
```

## Run MindInsight

After completing the data collection in the tutorial above, you can start MindInsight to visualize the collected data. When start MindInsight, you need to specify the summary log file directory with the `--summary-base-dir` parameter.

The specified summary log file directory can be the output directory of a training or the parent directory of the output directory of multiple training.

The output directory structure for a training is as follows

```text
└─summary_dir
    events.out.events.summary.1596869898.hostname_MS
    events.out.events.summary.1596869898.hostname_lineage
```

Execute command:

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

Execute command:

```Bash
mindinsight start --summary-base-dir ./summary
```

After successful startup, the visual page can be viewed by visiting the `http://127.0.0.1:8080` address through the browser.

Stop MindInsight command:

```Bash
mindinsight stop
```

For more parameter Settings, see the [MindInsight related commands](https://www.mindspore.cn/mindinsight/docs/en/master/mindinsight_commands.html) page.

## Notices

1. To limit time of listing summaries, MindInsight lists at most 999 summary items.

2. Multiple `SummaryRecord` instances can not be used at the same time. (`SummaryRecord` is used in `SummaryCollector`)

    If you use two or more instances of `SummaryCollector` in the callback list of 'model.train' or 'model.eval', it is seen as using multiple `SummaryRecord` instances at the same time, and it may cause recoding data failure.

    If the customized callback uses `SummaryRecord`, it can not be used with `SummaryCollector` at the same time.

    Correct code:

    ```python
    ...
    summary_collector = SummaryCollector('./summary_dir')
    model.train(2, train_dataset, callbacks=[summary_collector])
    ...
    model.eval(dataset, callbacks=[summary_collector])
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

4. When using summary, it is recommended that you set `dataset_sink_mode` argument of `model.train` to `False`, so that the unit of `collect_freq` is `step`. When `dataset_sink_mode` was `True`, the unit of `collect_freq` would be `epoch` and it is recommended that you set `collect_freq` manually.

5. The maximum amount of data saved per step is 2147483647 Bytes. If this limit is exceeded, data for the step cannot be recorded and an error occurs.

6. In PyNative mode, the `SummaryCollector` can be used properly, but the computational graph can not be recorded and the summary operator can not be used.
