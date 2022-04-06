# Collecting Summary Record

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindinsight/docs/source_en/summary_record.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Scalars, images, computational graphs, training optimization process, and model hyperparameters during training are recorded in files and can be viewed on the web page.

## Operation Process

- Prepare a training script, specify scalars, images, computational graphs, training optimization process, and model hyperparameters in the training script, record them in the summary log file, and run the training script.
- Start MindInsight and specify the summary log file directory using startup parameters. After MindInsight is started, access the visualization page based on the IP address and port number. The default access IP address is `http://127.0.0.1:8080`.
- During the training, when data is written into the summary log file, you can view the data on the web page.

## Preparing The Training Script

Currently, MindSpore supports to save scalars, images, computational graph, training optimization process, and model hyperparameters to summary log file and display them on the web page. The computational graph can only be recorded in the graph mode. The detailed process of data collection and landscape drawing in the training optimization process can be referred to  [Training Optimization Process Visualization](https://www.mindspore.cn/mindinsight/docs/en/master/landscape.html).

MindSpore currently supports multiple ways to record data into summary log files.

### Method one: Automatically collected through SummaryCollector

The `Callback` mechanism in MindSpore provides a quick and easy way to collect common information, including the calculational graph, loss value, learning rate, parameter weights, etc. It is named 'SummaryCollector'.

When you write a training script, you just instantiate the `SummaryCollector` and apply it to either `model.train` or `model.eval`. You can automatically collect some common summary data. The detailed usage of `SummaryCollector` can refer to the `API` document [mindspore.train.callback.SummaryCollector](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.train.html#mindspore.train.callback.SummaryCollector) .

The sample code snippet is shown as follows. The [whole script](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/mindinsight/summary_record/summary_1.py) is put on gitee.

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

The recording method is shown in the following steps. The [whole script](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/mindinsight/summary_record/summary_2.py) is put on gitee.

Step 1: Call the summary operator in the `construct` function of the derived class that inherits `nn.Cell` to collect image or scalar data.

For example, when a network is defined, image data is recorded in `construct` of the network. When the loss function is defined, the loss value is recorded in `construct` of the loss function.

Record the dynamic learning rate in `construct` of the optimizer when defining the optimizer.

The sample code is as follows:

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

> 1. In the same Summary operator, the name given to the data must not be repeated, otherwise the data collection and presentation will have unexpected behavior.
> For example, if two `ScalarSummary` operators are used to collect scalar data, two scalars cannot be given the same name.
> 2. Summary operator only supports Graph mode and needs to be used in `construct` of `nn.Cell`. The PyNative mode is not supported yet.

Step 2: In the training script, instantiate the `SummaryCollector` and apply it to `model.train`.

The sample code is as follows:

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

### Method three: Custom callback recording data

MindSpore supports customized callback and supports to record data into summary log file
in custom callback, and display the data by the web page.

The following pseudocode is shown in the CNN network, where developers can use the network output with the original tag and the prediction tag to generate the image of the confusion matrix.
It is then recorded into the summary log file through the `SummaryRecord` module.
The detailed usage of `SummaryRecord` can refer to the `API` document [mindspore.train.summary.SummaryRecord](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.train.html#mindspore.train.summary.SummaryRecord) .

The sample code snippet is as follows. The [whole script](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/mindinsight/summary_record/summary_3.py) is put on gitee.

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

The above three ways support the record computational graph, loss value and other data. In addition, MindSpore also supports the saving of computational graph for other phases of training, through
the `save_graphs` option of `context.set_context` in the training script is set to `True` to record computational graphs of other phases, including the computational graph after operator fusion.

In the saved files, `ms_output_after_hwopt.pb` is the computational graph after operator fusion, which can be viewed on the web page.

### Method four: Advanced usage, custom training cycle

If you are not using the `Model` interface provided by MindSpore, you can implement a method by imitating `train` method of `Model` interface to control the number of iterations. You can imitate the `SummaryCollector` and record the summary operator data in the following manner.

The following code snippet demonstrates how to record data in a custom training cycle using the summary operator and the `add_value` interface of `SummaryRecord`. The [whole script](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/mindinsight/summary_record/summary_4.py) is put on gitee.

For more tutorials about `SummaryRecord`, [refer to the Python API documentation](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.train.html#mindspore.train.summary.SummaryRecord). Please note that `SummaryRecord` will not record computational graph automatically. If you need to record the computational graph, please manually pass the instance of network that inherits from Cell. The recorded computational graph only includes the code and functions used in the construct method.

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

Recording gradients is possible by inheriting your original optimizer and inserting calls to summary operator. An example of code snippet is as follows. The [whole script](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/mindinsight/summary_record/summary_gradients.py) is put on gitee.

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
