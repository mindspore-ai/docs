# Operator-level Parallelism

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/parallel/operator_parallel.md)

## Overview

With the development of deep learning, network models are becoming larger and larger, such as trillions of parametric models have emerged in the field of NLP, and the model capacity far exceeds the memory capacity of a single device, making it impossible to train on a single card or data parallel. Operator-level parallelism is achieved by slicing the tensor involved in each operator in the network model and distributing the operators to multiple devices, reducing memory consumption on a single device, thus enabling the training of large models.

MindSpore provides two levels of granularity: operator-level parallelism and higher-order operator-level parallelism. Operator-level parallelism describes the tensor dimensionality distribution through a simple slicing strategy, which meets the requirements of most scenarios. Higher-order operator parallelism supports complex slicing scenarios through open device scheduling descriptions. The Operator-level Parallelism capabilities at two granularities both support ops and mint operators simultaneously. This chapter will introduce the practices of operator-level parallelism and high-order operator-level parallelism based on ops and mint operators respectively.

## Operator-Level Parallel Practice

### ops Operator Parallel Practice

The illustration of the ops operator parallel operation is based on the Ascend single-machine 8-card example.

#### Sample Code Description

> Download the complete sample code here: [distributed_operator_parallel](https://gitee.com/mindspore/docs/tree/br_base/docs/sample_code/distributed_operator_parallel).

The directory structure is as follows:

```text
└─ sample_code
    ├─ distributed_operator_parallel
       ├── distributed_operator_parallel.py
       ├── run.sh
       └── ...
    ...
```

Among them, `distributed_operator_parallel.py` is the script that defines the network structure and the training process. `run.sh` is the execution script.

#### Configuring the Distributed Environment

Unlike single card scripts, parallel scripts also need to initialize the communication domain through the `init` interface. In addition, limiting the model's maximum available device memory via the `max_size` of the `set_memory` interface leaves enough device memory for communication on the Ascend hardware platform.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.runtime.set_memory(max_size="28GB")
init()
ms.set_seed(1)
```

#### Loading the Dataset

In the operator-level parallel scenario, the dataset is loaded in the same way as single-card is loaded, with the following code:

```python
import os
import mindspore.dataset as ds

def create_dataset(batch_size):
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

data_set = create_dataset(32)
```

#### Defining the Network

In the current semi-automatic parallel mode, the network needs to be defined with ops operators(Primitive). Users can manually configure the slicing strategy for some operators based on a single-card network, e.g., the network structure after configuring the strategy is:

```python
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import initializer

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ops.MatMul().shard(((2, 4), (4, 1)))
        self.relu1 = ops.ReLU().shard(((4, 1),))
        self.matmul2 = ops.MatMul().shard(((1, 8), (8, 1)))
        self.relu2 = ops.ReLU().shard(((8, 1),))
        self.matmul3 = ops.MatMul()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

```

The `ops.MatMul()` and `ops.ReLU()` operators for the above networks are configured with slicing strategy, in the case of `ops.MatMul().shard(((2, 4), (4, 1)))`, which has a slicing strategy of: rows of the first input are sliced in 2 parts and columns in 4 parts; rows of the second input are sliced in 4 parts. For `ops.ReLU().shard(((8, 1),))`, its slicing strategy is: the row of the first input is sliced in 8 parts. Note that since the two `ops.ReLU()` here have different slicing strategies, i.e., `ops.ReLU().shard(((4, 1),))` and `ops.ReLU().shard(((8, 1),))` have to be defined twice separately.

#### Training Network Definition

In this step, we need to define the loss function, the optimizer, and the training process. Note that due to the huge number of parameters of the large model, the graphics memory will be far from sufficient if parameter initialization is performed when defining the network on a single card. Therefore, delayed initialization is required when defining the network in conjunction with the `no_init_parameters` interface to delay parameter initialization until the parallel multicard phase. Here both network and optimizer definitions need to be delayed initialized.

```python
from mindspore.nn.utils import no_init_parameters

with no_init_parameters():
    net = Network()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)

loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

def train_step(inputs, targets):
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

```

#### Parallel Configuration

We need to further set up the parallelism-related configuration by specifying the parallel mode `semi_auto` as semi-automatic parallel.

```python
from mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(train_step, parallel_mode="semi_auto")
```

#### Training Loop

This step performs a training loop, the outer loop is the number of epochs to train and the inner loop traverses the dataset, calling `parallel_net` to train and obtain the loss values.

```python
for epoch in range(10):
    i = 0
    for image, label in data_set:
        loss_output = parallel_net(image, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
        i += 1
```

#### Running the Single-machine Eight-card Script

Next, the corresponding scripts are invoked by commands, using the `msrun` startup method and the 8-card distributed training script as an example of distributed training:

```bash
bash run.sh
```

After training, the log files are saved to the `log_output` directory, where part of the file directory structure is as follows:

```text
└─ log_output
    ├─ scheduler.log
    ├─ worker_0.log
    ├─ worker_1.log
...
```

The results on the Loss section are saved in `log_output/worker_*.log`, and example is as follows:

```text
epoch: 0 step: 0, loss is 2.3016002
epoch: 0 step: 10, loss is 2.2889402
epoch: 0 step: 20, loss is 2.2843816
epoch: 0 step: 30, loss is 2.248126
epoch: 0 step: 40, loss is 2.1581488
epoch: 0 step: 50, loss is 1.8051043
...
```

Other startup methods such as `mpirun` and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/en/br_base/parallel/startup_method.html).

### mint Operator Parallel Practice

The illustration of the mint operator parallel operation is based on the Ascend single-machine 8-card example.

#### Sample Code Description

> Download the complete sample code here: [distributed_operator_parallel](https://gitee.com/mindspore/docs/tree/br_base/docs/sample_code/distributed_operator_parallel).

The directory structure is as follows:

```text
└─ sample_code
    ├─ distributed_operator_parallel
       ├── distributed_mint_operator_parallel.py
       ├── run_mint.sh
       └── ...
    ...
```

Among them, `distributed_mint_operator_parallel.py` is the script that defines the network structure and the training process. `run_mint.sh` is the execution script.

#### Configuring the Distributed Environment

Unlike single card scripts, parallel scripts also need to initialize the communication domain through the `init` interface. In addition, limiting the model's maximum available device memory via the `max_size` of the `set_memory` interface leaves enough device memory for communication on the Ascend hardware platform.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.runtime.set_memory(max_size="28GB")
init()
ms.set_seed(1)
```

#### Loading the Dataset

In the mint operator parallel scenario, the dataset is loaded in the same way as ops operator parallel practice is loaded, with the following code:

```python
import os
import mindspore.dataset as ds

def create_dataset(batch_size):
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

data_set = create_dataset(32)
```

#### Defining the Network

In the current mint operator parallel mode, the network needs to be defined with mint operators. Since the mint operators, as a functional interface, does not directly expose its operator type (Primitive), it is impossible to directly configure the slicing strategy for the operator. Instead, users need to manually configure the slicing strategy for mint operators by using `mindspore.parallel.shard` interface based on a single-card network, e.g., the network structure after configuring the strategy is:

```python
import mindspore as ms
from mindspore import nn, mint

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = mint.flatten
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ms.parallel.shard(mint.matmul, in_strategy=((2, 4), (4, 1)))
        self.relu1 = ms.parallel.shard(mint.nn.functional.relu, in_strategy=((4, 1),))
        self.matmul2 = ms.parallel.shard(mint.matmul, in_strategy=((1, 8), (8, 1)))
        self.relu2 = ms.parallel.shard(mint.nn.functional.relu, in_strategy=((8, 1),))
        self.matmul3 = mint.matmul

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x, dim=0, keepdims=True)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x, dim=0, keepdims=True)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

net = Network()
```

The `mint.matmul` and `mint.nn.functional.relu` operators for the above networks are configured with slicing strategy, in the case of `ms.parallel.shard(mint.matmul, in_strategy=((2, 4), (4, 1)))`, which has a slicing strategy of: rows of the first input are sliced in 2 parts and columns in 4 parts; rows of the second input are sliced in 4 parts. For `ms.parallel.shard(mint.mean, in_strategy=((8, 1),))`, its slicing strategy is: the row of the first input is sliced in 8 parts. Note that since the two `mint.nn.functional.relu` here have different slicing strategies, i.e., `ms.parallel.shard(mint.nn.functional.relu, in_strategy=((4, 1),))` and `ms.parallel.shard(mint.nn.functional.relu, in_strategy=((8, 1),))` have to be defined twice separately.

#### Parallel Configuration

We need to further set up the parallelism-related configuration by specifying the parallel mode `semi_auto` as semi-automatic parallel.

```python
from mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(train_step, parallel_mode="semi_auto")
```

#### Executing the Network

This step executes a forward calculation of the network in a loop, the outer loop is the number of epochs to be executed and the inner loop traverses the dataset, calling `parallel_net` to execute distributed calculation and obtain the forward output.

```python
for epoch in range(10):
    i = 0
    for image, _ in data_set:
        forward_logits = parallel_net(image)
        if i % 10 == 0:
            forward_sum = mint.sum(forward_logits).asnumpy()
            print("epoch: %s, step: %s, forward_sum is %s" % (epoch, i, forward_sum))
        i += 1
```

#### Running the Single-machine Eight-card Script

Next, the corresponding scripts are invoked by commands, using the `msrun` startup method and the 8-card distributed training script as an example of distributed training:

```bash
bash run_mint.sh
```

After training, the log files are saved to the `mint_log_output` directory, where part of the file directory structure is as follows:

```text
└─ mint_log_output
    ├─ scheduler.log
    ├─ worker_0.log
    ├─ worker_1.log
...
```

The results are saved in `mint_log_output/worker_*.log`, and example is as follows:

```text
epoch: 0 step: 0, forward_sum is 0.90023
epoch: 0 step: 10, forward_sum is 1.07679
epoch: 0 step: 20, forward_sum is 1.02521
epoch: 0 step: 30, forward_sum is 0.96682
epoch: 0 step: 40, forward_sum is 0.93158
epoch: 0 step: 50, forward_sum is 0.96655
...
```

Other startup methods such as `mpirun` and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/en/br_base/parallel/startup_method.html).

## Higher-Order Operator-Level Parallel Practice

### Higher-Order ops Operator Parallel Practice

An illustration of higher-order ops operator parallel operations follows, using the Ascend single 8-card as an example.

#### Sample Code Description

> Download the complete sample code here: [distributed_operator_parallel](https://gitee.com/mindspore/docs/tree/br_base/docs/sample_code/distributed_operator_parallel).

The directory structure is as follows:

```text
└─ sample_code
    ├─ distributed_operator_parallel
       ├── advanced_distributed_operator_parallel.py
       ├── run_advanced.sh
       └── ...
    ...
```

Among them, `advanced_distributed_operator_parallel.py` is the script that defines the network structure and the training process. `run_advanced.sh` is the execution script.

#### Environment Configuration

Before performing higher-order operator-level parallelism, the environment is first configured, and the process is consistent with operator-level parallelism, which can be found in [Configure Distributed Environment](#configuring-the-distributed-environment) and [Dataset Load](#loading-the-dataset).

#### Defining the Network

Higher-order operator-level parallelism extends the functionality of the `shard` interface by additionally accepting the new quantity type `tuple(Layout)` type for both the `in_strategy`/`out_strategy` in-parameters of the `shard` interface.

Layout is initialized using the device matrix, and requires an alias for each axis of the device matrix, such as "layout = Layout((2, 2, 2), name = ("dp", "sp", "mp"))", which describes a total of 8 cards arranged in the shape of (2, 2, 2), and each axis is aliased to "dp", "sp", "mp".

The call to Layout passes in these axes, and each tensor picks which axis of the device each dimension is expected to map to according to its shape, and also determines the number of parts to be sliced, e.g., here "dp" means 2 parts in 2 devices in the highest dimension of the device layout; "sp" means 2 parts in 2 devices in the middle dimension of the device layout; "mp" means 2 parts in 2 devices in the lowest dimension of the device layout. In particular, one dimension of the tensor may be mapped to multiple dimensions of the device to express multiple slices in one dimension.

```python

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import initializer
from mindspore.parallel import Layout

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
        layout2 = Layout((8,), ("tp",))
        self.matmul1 = ops.MatMul().shard((layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None")))
        self.relu1 = ops.ReLU().shard(((4, 1),))
        self.matmul2 = ops.MatMul().shard((layout2("None", "tp"), layout2("tp", "None")))
        self.relu2 = ops.ReLU().shard(((8, 1),))
        self.matmul3 = ops.MatMul()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

```

In the network defined above, `self.matmul1 = ops.MatMul().shard((layout("mp", ("sp", "dp")), layout(("sp", "dp")), "None"))` The layout for slicing the input tensor x is `layout("mp", ("sp ", "dp"))`, i.e., the first dimension is sliced into 2 parts by mp, and the second dimension combines sp and dp for a total of 2*2=4 parts.

The layout for slicing the weight self.fc1_weight is `layout(("sp", "dp"), "None")`, i.e., the first dimension merges sp and dp and slices it into 4 parts, and the second dimension is not sliced.

Similarly, `self.matmul2 = ops.MatMul().shard((layout2("None", "tp"), layout2("tp", "None")))` When slicing the input tensor x first dimension by rows not sliced and columns sliced into 8 parts by tp, and when slicing the weight self.fc2_weight, the rows are sliced into 8 parts by tp and the columns are not sliced.

Taking `self.matmul1 = ops.MatMul().shard((layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None"))` as an example, the slicing will produce the following table of device and data slice mappings:

| device coordinates (dp, sp, mp) | input x slice         | weight fc1_weight slice     |
|-----------------------|----------------------|---------------------------|
| (0, 0, 0)             | `x[0:16, 0:196]`     | `fc1_weight[0:196, 0:512]` |
| (0, 0, 1)             | `x[16:32, 0:196]`    | `fc1_weight[0:196, 0:512]` |
| (0, 1, 0)             | `x[0:16, 196:392]`   | `fc1_weight[196:392, 0:512]` |
| (0, 1, 1)             | `x[16:32, 196:392]`  | `fc1_weight[196:392, 0:512]` |
| (1, 0, 0)             | `x[0:16, 392:588]`   | `fc1_weight[392:588, 0:512]` |
| (1, 0, 1)             | `x[16:32, 392:588]`  | `fc1_weight[392:588, 0:512]` |
| (1, 1, 0)             | `x[0:16, 588:784]`   | `fc1_weight[588:784, 0:512]` |
| (1, 1, 1)             | `x[16:32, 588:784]`  | `fc1_weight[588:784, 0:512]` |

#### Training Process

The training flow for higher-order operator-level parallelism is identical to operator-level parallelism, and can be found in [Training Network Definition](#training-network-definition), [Parallel Configuration](#parallel-configuration), and [Training Loop](#training-loop).

#### Running the Single-machine Eight-card Script

Next, the corresponding scripts are invoked by commands, using the `msrun` startup method and the 8-card distributed training script as an example of distributed training:

```bash
bash run_advanced.sh
```

After training, the log files are saved to the `advanced_log_output` directory, where part of the file directory structure is as follows:

```text
└─ advanced_log_output
    ├─ scheduler.log
    ├─ worker_0.log
    ├─ worker_1.log
...
```

The results are saved in `advanced_log_output/worker_*.log`, and example is as follows:

```text
epoch: 0 step: 0, loss is 2.3016002
epoch: 0 step: 10, loss is 2.2889402
epoch: 0 step: 20, loss is 2.2843816
epoch: 0 step: 30, loss is 2.248126
epoch: 0 step: 40, loss is 2.1581488
epoch: 0 step: 50, loss is 1.8051043
...
```

Other startup methods such as `mpirun` and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/en/br_base/parallel/startup_method.html).

### Higher-Order mint Operator Parallel Practice

An illustration of higher-order mint operator parallel operations follows, using the Ascend single 8-card as an example.

#### Sample Code Description

> Download the complete sample code here: [distributed_operator_parallel](https://gitee.com/mindspore/docs/tree/br_base/docs/sample_code/distributed_operator_parallel).

The directory structure is as follows:

```text
└─ sample_code
    ├─ distributed_operator_parallel
       ├── advanced_distributed_mint_operator_parallel.py
       ├── run_advanced_mint.sh
       └── ...
    ...
```

Among them, `advanced_distributed_mint_operator_parallel.py` is the script that defines the network structure and the training process. `run_advanced_mint.sh` is the execution script.

#### Environment Configuration

Before performing higher-order mint operator parallelism, the environment is first configured, and the process is consistent with operator-level parallelism, which can be found in [Configure Distributed Environment](#configuring-the-distributed-environment) and [Dataset Load](#loading-the-dataset).

#### Defining the Network

The configuration method for the slicing strategy of high-order mint operator parallelism is similar to that of mint operator parallelism. You only need to pass an input of type tuple(Layout) to the parameter `in_strategy` in the `mindspore.parallel.shard` interface.

Layout is initialized using the device matrix, and requires an alias for each axis of the device matrix, such as "layout = Layout((2, 2, 2), name = ("dp", "sp", "mp"))", which describes a total of 8 cards arranged in the shape of (2, 2, 2), and each axis is aliased to "dp", "sp", "mp".

The call to Layout passes in these axes, and each tensor picks which axis of the device each dimension is expected to map to according to its shape, and also determines the number of parts to be sliced, e.g., here "dp" means 2 parts in 2 devices in the highest dimension of the device layout; "sp" means 2 parts in 2 devices in the middle dimension of the device layout; "mp" means 2 parts in 2 devices in the lowest dimension of the device layout. In particular, one dimension of the tensor may be mapped to multiple dimensions of the device to express multiple slices in one dimension.

```python

import mindspore as ms
from mindspore import nn, mint

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = mint.flatten
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
        layout2 = Layout((8,), ("tp",))
        self.matmul1 = ms.parallel.shard(mint.matmul, in_strategy=(layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None")))
        self.relu1 = ms.parallel.shard(mint.nn.functional.relu, in_strategy=((4, 1),))
        self.matmul2 = ms.parallel.shard(mint.matmul, in_strategy=(layout2("None", "tp"), layout2("tp", "None")))
        self.relu2 = ms.parallel.shard(mint.nn.functional.relu, in_strategy=((8, 1),))
        self.matmul3 = mint.matmul

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x, dim=0, keepdims=True)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x, dim=0, keepdims=True)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

net = Network()
```

In the network defined above, `self.matmul1 = ms.parallel.shard(mint.matmul, in_strategy=(layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None")))` The layout for slicing the input tensor x is `layout("mp", ("sp ", "dp"))`, i.e., the first dimension is sliced into 2 parts by mp, and the second dimension combines sp and dp for a total of 2*2=4 parts.

The layout for slicing the weight self.fc1_weight is `layout(("sp", "dp"), "None")`, i.e., the first dimension merges sp and dp and slices it into 4 parts, and the second dimension is not sliced.

Similarly, `self.matmul2 = ms.parallel.shard(mint.matmul, in_strategy=(layout2("None", "tp"), layout2("tp", "None")))` When slicing the input tensor x first dimension by rows not sliced and columns sliced into 8 parts by tp, and when slicing the weight self.fc2_weight, the rows are sliced into 8 parts by tp and the columns are not sliced.

Taking `self.matmul1 = ms.parallel.shard(mint.matmul, in_strategy=(layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None")))` as an example, the slicing will produce the following table of device and data slice mappings:

| device coordinates (dp, sp, mp) | input x slice         | weight fc1_weight slice     |
|-----------------------|----------------------|---------------------------|
| (0, 0, 0)             | `x[0:16, 0:196]`     | `fc1_weight[0:196, 0:512]` |
| (0, 0, 1)             | `x[16:32, 0:196]`    | `fc1_weight[0:196, 0:512]` |
| (0, 1, 0)             | `x[0:16, 196:392]`   | `fc1_weight[196:392, 0:512]` |
| (0, 1, 1)             | `x[16:32, 196:392]`  | `fc1_weight[196:392, 0:512]` |
| (1, 0, 0)             | `x[0:16, 392:588]`   | `fc1_weight[392:588, 0:512]` |
| (1, 0, 1)             | `x[16:32, 392:588]`  | `fc1_weight[392:588, 0:512]` |
| (1, 1, 0)             | `x[0:16, 588:784]`   | `fc1_weight[588:784, 0:512]` |
| (1, 1, 1)             | `x[16:32, 588:784]`  | `fc1_weight[588:784, 0:512]` |

#### Parallel Configuration

We need to further set up the parallelism-related configuration by specifying the parallel mode `semi_auto` as semi-automatic parallel.

```python
from mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(train_step, parallel_mode="semi_auto")
```

#### Executing the Network

This step executes a forward calculation of the network in a loop, the outer loop is the number of epochs to be executed and the inner loop traverses the dataset, calling `parallel_net` to execute distributed calculation and obtain the forward output.

```python
for epoch in range(10):
    i = 0
    for image, _ in data_set:
        forward_logits = parallel_net(image)
        if i % 10 == 0:
            forward_sum = mint.sum(forward_logits).asnumpy()
            print("epoch: %s, step: %s, forward_sum is %s" % (epoch, i, forward_sum))
        i += 1
```

#### Running the Single-machine Eight-card Script

Next, the corresponding scripts are invoked by commands, using the `msrun` startup method and the 8-card distributed training script as an example of distributed training:

```bash
bash run_advanced_mint.sh
```

After training, the log files are saved to the `advanced_mint_log_output` directory, where part of the file directory structure is as follows:

```text
└─ advanced_mint_log_output
    ├─ scheduler.log
    ├─ worker_0.log
    ├─ worker_1.log
...
```

The results are saved in `advanced_mint_log_output/worker_*.log`, and example is as follows:

```text
epoch: 0 step: 0, forward_sum is 0.90023
epoch: 0 step: 10, forward_sum is 1.07679
epoch: 0 step: 20, forward_sum is 1.02521
epoch: 0 step: 30, forward_sum is 0.96682
epoch: 0 step: 40, forward_sum is 0.93158
epoch: 0 step: 50, forward_sum is 0.96655
...
```

Other startup methods such as `mpirun` and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/en/br_base/parallel/startup_method.html).
