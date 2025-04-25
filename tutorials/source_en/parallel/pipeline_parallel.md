# Pipeline Parallel

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/parallel/pipeline_parallel.md)

## Overview

In recent years, the scale of neural networks has increased exponentially. Limited by the memory on a single device, the number of devices used for training large models is also increasing. Due to the low communication bandwidth between servers, the performance of the conventional hybrid parallelism (data parallel + model parallel) is poor. Therefore, pipeline parallelism needs to be introduced. Pipeline parallel can divide a model in space based on stage. Each stage needs to execute only a part of the network, which greatly reduces memory overheads, shrinks the communication domain, and shortens the communication time. MindSpore can automatically convert a standalone model to the pipeline parallel mode based on user configurations.

## Training Operation Practices

The following is an illustration of pipeline parallel operation using Ascend or GPU single-machine 8-card as an example:

### Sample Code Description

> Download the complete sample code: [distributed_pipeline_parallel](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_pipeline_parallel).

The directory structure is as follows:

```text
└─ sample_code
    ├─ distributed_pipeline_parallel
       ├── distributed_pipeline_parallel.py
       └── run.sh
    ...
```

`distributed_pipeline_parallel.py` is the script that defines the network structure and training process. `run.sh` is the execution script.

### Configuring the Distributed Environment

Specify the run mode. Unlike single-card scripts, parallel scripts also need to initialize HCCL or NCCL communication via init.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
init()
ms.set_seed(1)
```

### Loading the Dataset

In the pipeline parallel scenario, the dataset is loaded in the same way as a single card is loaded, with the following code:

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

### Defining the Network

The pipeline parallel network structure is basically the same as the single-card network structure. It should be noted that:

> - Under pipeline parallelism, when enabling Print/Summary/TensorDump related operators, the operator needs to be used in a Cell with the pipeline_stage attribute. Otherwise, there is a possibility that the operator will not take effect due to pipeline parallel split.
> - Under pipeline parallelism, the output of the network does not support dynamic shapes.
> - Under pipeline parallelism, suggests to use lazy_inline decorator to reduce compile time, and only support to set the lazy_inline decorator to the outermost cell.

```python
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, HeUniform

import math

class MatMulCell(nn.Cell):
    """
    MatMulCell definition.
    """
    def __init__(self, param=None, shape=None):
        super().__init__()
        if shape is None:
            shape = [28 * 28, 512]
        weight_init = HeUniform(math.sqrt(5))
        self.param = Parameter(initializer(weight_init, shape), name="param")
        if param is not None:
            self.param = param
        self.print = ops.Print()
        self.matmul = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.param)
        self.print("out is:", out)
        return out


class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = MatMulCell()
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

```

### Training Network Definition

In this step, we need to define the loss function, the optimizer, and the training process. It should be noted that the definitions of both the network and the optimizer here require deferred initialization. Besides, the interface `PipelineGradReducer` is needed to handle gradient of pipeline parallelism, the first parameter of this interface is the network parameter to be updated, and the second one is whether to use optimizer parallelism.

Unlike the single-card model, two interfaces need to be called in this section to configure the pipeline parallel:

- First define the LossCell. In this case the `nn.WithLossCell` interface is called to encapsulate the network and loss functions.
- Finally, wrap the LossCell with `Pipeline`, and specify the size of MicroBatch. Configure the `pipeline_stage` for each `Cell` containing training parameters via `stage_config`.

```python
import mindspore as ms
from mindspore import nn, ops
from mindspore.parallel.nn import Pipeline, PipelineGradReducer
from mindspore.nn.utils import no_init_parameters

with no_init_parameters():
    net = Network()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)
    pp_grad_reducer = PipelineGradReducer(optimizer.parameters, opt_shard=False)

loss_fn = nn.CrossEntropyLoss()
net_with_loss = Pipeline(nn.WithLossCell(net, loss_fn), 4, stage_config={"_backbone.flatten":0,
                            "_backbone.layer1": 0, "_backbone.relu1": 0, "_backbone.layer2": 1, "_backbone.relu2": 1, "_backbone.layer3": 1})
net_with_loss.set_train()

def forward_fn(inputs, target):
    loss = net_with_loss(inputs, target)
    return loss

grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

@ms.jit
def train_one_step(inputs, target):
    loss, grads = grad_fn(inputs, target)
    grads = pp_grad_reducer(grads)
    optimizer(grads)
    return loss, grads

```

To enable interleaved pipeline scheduling, the `stage_config` in `Pipeline` needs to be interleaved for the discontinuous model layers, configured as follows:

```python
net_with_loss = Pipeline(nn.WithLossCell(net, loss_fn), 4, stage_config={"_backbone.flatten":0,
                            "_backbone.layer1": 1, "_backbone.relu1": 0, "_backbone.layer2": 1, "_backbone.relu2": 0, "_backbone.layer3": 1})
```

## Parallel Configuration

We need to further set up the parallelism-related configuration by specifying the parallelism mode `semi_auto` as semi-automatic parallelism. It is also necessary to turn on pipeline parallelism, configure `pipeline`, and specify the total number of stages by configuring the `stages` count.

```python
import mindspore as ms
from mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(train_one_step, parallel_mode="semi_auto")
parallel_net.pipeline(stages=2)

```

If you need to run interleaved pipeline scheduling, you also need to configure:`parallel_net.pipeline(stages=2, interleave=True)`. It should be noted that MindSpore interleaved pipeline scheduling is still in the refinement stage and currently performs better in O0 or O1 mode.

```python
import mindspore as ms
import mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(train_one_step, parallel_mode="semi_auto")
parallel_net.pipeline(stages=2, interleave=True)
```

## Training Loop

This step performs the training loop, the outer loop is the number of epochs to train and the inner loop traverses the dataset and calls parallel_net to train and get the loss values.

```python
for epoch in range(10):
    i = 0
    for data, label in data_set:
        loss, grads = parallel_net(data, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss))
        i += 1
```

> Currently pipeline parallel does not support the automatic mixed precision.
>
> Pipeline parallel training is more suitable to use `model.train` approach, because the TrainOneStep logic under pipeline parallelism is complex, while `model.train` internally encapsulates the TrainOneStepCell for pipeline parallel, which is much easier to use.

### Running the Single-host with 8 Devices Script

Next, the corresponding scripts are called by commands, using the `msrun` startup method and the 8-card distributed training script as an example of distributed training:

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

The results are saved in `log_output/worker_*.log`, and the example is as below:

```text
epoch: 0 step: 0, loss is 9.137518
epoch: 0 step: 10, loss is 8.826559
epoch: 0 step: 20, loss is 8.675843
epoch: 0 step: 30, loss is 8.307994
epoch: 0 step: 40, loss is 7.856993
epoch: 0 step: 50, loss is 7.0662785
...
```

The results of operator `Print` is:

```text
out is:
Tensor(shape=[8, 512], dtype=Float32, value=
[[ 4.61914062e-01 5.78613281e-01 1.34995094e-01 ... 8.54492188e-02 7.91992188e-01 2.13378906e-01]
...
[  4.89746094e-01 3.56689453e-01 -4.90966797e-01 ... -3.30078125e-e01 -2.38525391e-01 7.33398438e-01]])
```

Other startup methods such as dynamic cluster and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/en/master/parallel/startup_method.html).

## Inference Operation Practices

The following is an illustration of pipeline parallel inference operation using Ascend single-machine 8-card as an example:

### Sample Code Description

> Download the complete sample code: [distributed_pipeline_parallel](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_pipeline_parallel).

The directory structure is as follows:

```text

└─ sample_code
    ├─ distributed_pipeline_parallel
       ├── distributed_pipeline_parallel_inference.py
       └── run_inference.sh
    ...

```

`distributed_pipeline_parallel_inference.py` is the script that defines the network structure and inference process. `run_inference.sh` is the execution script.

### Configuring the Distributed Environment

Specify the run mode, run device, run card number, etc. via the context interface. Unlike single-card scripts, parallel scripts also need to initialize HCCL or NCCL communication via init.

```python

import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
init()
ms.set_seed(1)

```

### Defining the Network

Pipeline parallel requires the user to define the parallel strategy by calling the `pipeline_stage` interface to specify the stage on which each layer is to be executed. The granularity of the `pipeline_stage` interface is `Cell`. All `Cells` containing training parameters need to be configured with `pipeline_stage`, and `pipeline_stage` should be configured in the order of network execution, from smallest to largest. Configuration after adding `pipeline_stage` based on the single-card model is as follows:

```python

import numpy as np
from mindspore import lazy_inline, nn, ops, Tensor, Parameter, sync_pipeline_shared_parameters

class VocabEmbedding(nn.Cell):
    """Vocab Embedding"""
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding_table = Parameter(Tensor(np.ones([vocab_size, embedding_size]), ms.float32),
                                         name='embedding_table')
        self.gather = ops.Gather()

    def construct(self, x):
        output = self.gather(self.embedding_table, x, 0)
        output = output.squeeze(1)
        return output, self.embedding_table.value()


class Head(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul(transpose_b=True)

    def construct(self, state, embed):
        return self.matmul(state, embed)


class Network(nn.Cell):
    """Network"""
    @lazy_inline
    def __init__(self):
        super().__init__()
        self.word_embedding = VocabEmbedding(vocab_size=32, embedding_size=32)
        self.layer1 = nn.Dense(32, 32)
        self.layer2 = nn.Dense(32, 32)
        self.head = Head()

    def construct(self, x):
        x, embed = self.word_embedding(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.head(x, embed)
        return x

# Define network and set pipeline stage
net = Network()
net.word_embedding.pipeline_stage = 0
net.layer1.pipeline_stage = 1
net.layer2.pipeline_stage = 2
net.head.pipeline_stage = 3

```

### Inferring the Network

Wrap the netork with `PipelineCellInference`, and specify the size of MicroBatch. `PipelineCellInference` splits input into several micro batch, then executes the network, and finally concats the results along the batch axis through `ops.Concat` operator.

In the previous step, the parameter `embed` is shared by `self.word_embedding` and `self.head` layer, and these two layers are split into different stages.

We need to further set up the parallelism-related configuration by wrapping the network again with `AutoParallel`, specifying the parallelism mode `semi-auto` as semi-automatic parallelism, in addition to turning on pipeline parallelism, configuring `pipeline`, and specifying the total number of stages by configuring the number of `stages`. If `device_target` is not set here, it will be automatically specified as the backend hardware device corresponding to the MindSpore package (default is Ascend). `output_broadcast=True` indicates that the result of the last stage will be broadcast to the remaining stages when pipelined parallel inference is performed, which can be used in autoregressive inference scenarios.

Before inference, executing `parallel_net.compile()` and `sync_pipeline_shared_parameters(parallel_net)`, the framework will synchronize the shared parameter between stages automatically.

```python

from mindspore import nn, ops

class PipelineCellInference(nn.Cell):
    """Pipeline Cell Inference wrapper"""
    def __init__(self, network, micro_batch_num):
        super().__init__()
        self.network = network
        self.micro_batch_num = micro_batch_num
        self.concat = ops.Concat()

    def construct(self, x):
        """Apply the pipeline inference"""
        ret = ()
        for i in range(self.micro_batch_num):
            micro_batch_size = x.shape[0] // self.micro_batch_num
            start = micro_batch_size * i
            end = micro_batch_size * (i + 1)

            micro_input = x[start:end]
            micro_output = self.network(micro_input)
            ret = ret + (micro_output,)

        ret = self.concat(ret)
        return ret

inference_network = PipelineCellInference(network=net, micro_batch_num=4)
inference_network.set_train(False)

parallel_net = AutoParallel(inference_network, parallel_mode="semi_auto")
parallel_net.dataset_strategy("full_batch")
parallel_net.pipeline(stages=4, output_broadcast=True)

# Compile and synchronize shared parameter.
input_ids = Tensor(np.random.randint(low=0, high=32, size=(8, 1)), ms.int32)
parallel_net.compile(input_ids)
sync_pipeline_shared_parameters(parallel_net)

# Execute the inference network
logits = parallel_net(input_ids)
print(logits.asnumpy())

```

### Running the Single-host with 8 Devices Script

Next, the corresponding scripts are called by commands, using the `msrun` startup method and the 8-card distributed inference script as an example of distributed inference:

```bash

bash run_inference.sh

```

After training, the log files are saved to the `pipeline_inference_logs` directory, where part of the file directory structure is as follows:

```text

└─ pipeline_inference_logs
   ├── scheduler.log
   ├── worker_0.log
   ├── worker_1.log
   ├── worker_2.log
...

```

The results are saved in `pipeline_inference_logs/worker_*.log`, and the example is as below:

```text

[[0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556]
  ...]

```
