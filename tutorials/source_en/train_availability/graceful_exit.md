# Training Process Exit Gracefully

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/train_availability/graceful_exit.md)

## Overview

When there are suboptimal devices in the training cluster, saving checkpoint and exiting the cluster training process before the failure occurs can effectively prevent the loss of weight data when the cluster is damaged. This also avoids issues such as training data rollback and loading checkpoint rollback when training recovery, effectively preventing the waste of training resources.

> This document describes how to use the process graceful exit. In order to illustrate the specific usage, the example of detecting the exit configuration message at the first training step and terminating the training process early is used. You can get the full sample code here: [process_graceful_exit](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/graceful_exit/) .

`graceful_exit.py` is the source code, `train.sh` is the start training script, and `graceful_exit.json` is the graceful exit config json file.

## Dataset And Training Model

### Data Preparation

Download the MNIST dataset and unzip the dataset to the project directory.

```bash
wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
unzip MNIST_Data.zip
```

### Model Definition

```python
import os
import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
import mindspore.nn as nn
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import Inter
from mindspore.train import Accuracy
from mindspore.train import Model, LossMonitor
from mindspore.train.callback import OnRequestExit
from mindspore.common.initializer import TruncatedNormal
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.nn.utils import no_init_parameters

context.set_context(mode=context.GRAPH_MODE)

# dataset
DATASET_PATH = "./MNIST_Data"


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


# define the training model
def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, channel=1):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(channel, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
```

## Environment Variable And Callback Function

### Environment Variable

Using Training process Graceful Exit requires setting the environment variable `MS_ENABLE_GRACEFUL_EXIT` to `1`. This environment variable can control the synchronization operator into the graph to ensure that all training processes can exit synchronously.

```bash
export MS_ENABLE_GRACEFUL_EXIT=1
```

### Callback Function

In addition to the above of environment variable, it also needs to configure the callback function `OnRequestExit` , and specify the path to the graceful exit configuration file with the parameter `config_file`. This callback function will check if there is a graceful exit json file in the specified path at every training step begin. If the file exists, and the `GracefulExit` is `1` , it will save checkpoint and exit training process at current step end.

The `GracefulExit` in the configuration file is dynamically configured during training. Generally, the keyword is modified when suboptimal devices exist in the training cluster and the training process needs to exit.

```python
# key in json file: ‘{“GracefulExit”: 1}’
config_json = r"./graceful_exit.json"

# callback function
cb = OnRequestExit(file_name="LeNet", config_file=config_json)
```

When configuring the `OnRequestExit` callback function, you can configure saving mindir, saving checkpoint, and other configuration parameters as required. For more details, please refer to the documentation [OnRequestExit](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.OnRequestExit.html) .

```python
def graceful_exit_case():
    # init
    device_num = 8
    context.set_context(mode=context.GRAPH_MODE)
    ms.set_device("Ascend")

    init()

    # build
    with no_init_parameters():
        network = LeNet5(10)
        net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    ds_train = create_dataset(os.path.join(DATASET_PATH, "train"), 32, 1)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    parallel_net = AutoParallel(network, parallel_mode='semi_auto')
    model = Model(parallel_net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    # graceful exit json file：{"GracefulExit": 1}
    reset_json = r"./graceful_exit.json"

    # callback
    cb = OnRequestExit(file_name="LeNet", config_file=reset_json)
    # train
    model.train(1, ds_train, callbacks=[cb, LossMonitor()], dataset_sink_mode=False)
```

## Starting Training

Using `msrun` to start training.

```bash
msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10970 --join=True --log_dir=./comm_subgraph_logs graceful_exit_case.py
```

## Analyzing The Results

After training ends, the following WARNING log will be printed: `Graceful exit is triggered, stop training`. Eight directories named `rank_0` to `rank_7` will be generated in the current execution directory, each containing a `LeNet_train.ckpt` file (if saving checkpoints is set in `OnRequestExit` ).

```text
./rank_0
├── LeNet_train.ckpt
└── LeNet_train.mindir
./rank_1
├── LeNet_train.ckpt
└── LeNet_train.mindir
./rank_2
├── LeNet_train.ckpt
└── LeNet_train.mindir
./rank_3
├── LeNet_train.ckpt
└── LeNet_train.mindir
./rank_4
├── LeNet_train.ckpt
└── LeNet_train.mindir
./rank_5
├── LeNet_train.ckpt
└── LeNet_train.mindir
./rank_6
├── LeNet_train.ckpt
└── LeNet_train.mindir
./rank_7
├── LeNet_train.ckpt
└── LeNet_train.mindir
```

## Notes

If `TrainOneStepCell` is not overridden, you only need to configure the `MS_ENABLE_GRACEFUL_EXIT` environment variable, the `OnRequestExit` callback function, and modify the graceful exit json file as needed at a certain point during training.

If the network model requires overriding `TrainOneStepCell`:

1. The new method inherits from `TrainOneStepCell` , and the following `if` conditional branch code is added in the `construct` method to ensure the graceful exit feature works properly.

    ```python
    class TrainOneStepCellWithABC(TrainOneStepCell):
        def __init__(self, ...):
            ...

        def construct(self, *inputs):
            ...
            grads = self.grad(self.network, self.weights)(*inputs, sens)
            if self.use_graceful_exit:
                grads = self.graceful_exit.exit_by_request(grads, self.init_param, self.exit_param)

            loss = F.depend(loss, self.optimizer(grads))
            ...
    ```

2. The new method is not inherits from `TrainOneStepCell` , you need add the following code in `__init__` method(don't change parameter's name), and using in the `construct` method.

    ```python
    from mindspore.utils import ExitByRequest

    class TrainOneStepCellWithABC(Cell):
        def __init__(self, ...):
            ...
            self.use_graceful_exit = os.environ.get("MS_ENABLE_GRACEFUL_EXIT") == "1"
            if self.use_graceful_exit:
                self.graceful_exit = ExitByRequest()
                self.exit_param = Parameter(Tensor(False, mstype.bool_), name="graceful_exit")  # update by reduce value
                self.init_param = Parameter(Tensor([0], mstype.int32), name="graceful_init")  # update by config file

        def construct(self, *inputs):
            ...
            if self.use_graceful_exit:
                grads = self.graceful_exit.exit_by_request(grads, self.init_param, self.exit_param)
            loss = F.depend(loss, self.optimizer(grads))
            ...
   ```
