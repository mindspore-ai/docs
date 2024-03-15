# Pipeline Parallel

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/pipeline_parallel.md)

## Overview

In recent years, the scale of neural networks has increased exponentially. Limited by the memory on a single device, the number of devices used for training large models is also increasing. Due to the low communication bandwidth between servers, the performance of the conventional hybrid parallelism (data parallel + model parallel) is poor. Therefore, pipeline parallelism needs to be introduced. Pipeline parallel can divide a model in space based on stage. Each stage needs to execute only a part of the network, which greatly reduces memory overheads, shrinks the communication domain, and shortens the communication time. MindSpore can automatically convert a standalone model to the pipeline parallel mode based on user configurations.

> Hardware platforms supported by the pipeline parallel model include Ascend, GPU, and need to be run in Graph mode.

Related interfaces:

1. `mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=NUM, pipeline_result_broadcast=True)`: Set semi-automatic parallel mode and set `pipeline_stages` to indicate that the total number of stages is NUM and call it before initializing the network. `pipeline_result_broadcast`: A switch that broadcast the last stage result to all other stage in pipeline parallel inference.

2. `nn.PipelineCell(loss_cell, micro_size)`: pipeline parallelism requires wrapping a layer of `PipelineCell` around the LossCell and specifying the size of the MicroBatch. In order to improve machine utilization, MindSpore slices the MiniBatch into finer-grained MicroBatches, and the final loss is the sum of the loss values computed by all MicroBatches, where the size of the MicroBatch must be greater than or equal to the number of stages.

3. `nn.PipelineGradReducer(parameters)`: pipeline parallelism requires using `PipelineGradReducer` for gradient reduction. Because the output of pipeline parallelism is derived by the addition of several micro-batch outputs, as the gradient do.

4. `mindspore.parallel.sync_pipeline_shared_parameters(net)`: Synchronize pipeline parallel stage shared parameters.

## Basic Principle

Pipeline parallel is the splitting of operators in a neural network into multiple stages, and then mapping the stages to different devices, so that different devices can compute different parts of the neural network. Pipeline parallel is suitable for graph structures where the model is linear. As shown in Figure 1, the network of 4 layers of MatMul is split into 4 stages and distributed to 4 devices. In forward calculations, each machine sends the result to the next machine through the communication operator after calculating the MatMul on the machine, and at the same time, the next machine receives (Receive) the MatMul result of the previous machine through the communication operator, and starts to calculate the MatMul on the machine; In reverse calculation, after the gradient of the last machine is calculated, the result is sent to the previous machine, and at the same time, the previous machine receives the gradient result of the last machine and begins to calculate the reverse of the current machine.

![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/experts/source_zh_cn/parallel/images/pipeline_parallel_image_0_zh.png)

*Figure 1: Schematic diagram of graph splitting in pipeline parallel*

Simply splitting the model onto multiple devices does not bring about a performance gain, because the linear structure of the model has only one device at work at a time, while other devices are waiting, resulting in a waste of resources. In order to improve efficiency, the pipeline parallel further divides the small batch (MiniBatch) into more fine-grained micro batches (MicroBatch), and adopts a pipeline execution sequence in the micro batch, so as to achieve the purpose of improving efficiency, as shown in Figure 2. The small batches are cut into 4 micro-batches, and the 4 micro-batches are executed on 4 groups to form a pipeline. The gradient aggregation of the micro-batch is used to update the parameters, where each device only stores and updates the parameters of the corresponding group. where the white ordinal number represents the index of the micro-batch.

![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/experts/source_zh_cn/parallel/images/pipeline_parallel_image_1_zh.png)

*Figure 2: Schematic diagram of a pipeline parallel execution timeline with MicroBatch*

In MindSpore's pipeline parallel implementation, the execution order has been adjusted for better memory management. As shown in Figure 3, the reverse of the MicroBatch numbered 0 is performed immediately after its forward execution, so that the memory of the intermediate result of the numbered 0 MicroBatch is freed earlier (compared to Figure 2), thus ensuring that the peak memory usage is lower than in the way of Figure 2.

![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/experts/source_zh_cn/parallel/images/pipeline_parallel_image_2_zh.png)

*Figure 3: MindSpore Pipeline Parallel Execution Timeline Diagram*

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

Specify the run mode, run device, run card number, etc. via the context interface. Unlike single-card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` to be semi-automatic parallel mode and initialize HCCL or NCCL communication via init. In addition, `pipeline_stages=2` should be configured to specify the total number of stages. Not setting `device_target` here automatically specifies the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=2)
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

The pipeline parallel network structure is basically the same as the single-card network structure, and the difference is the addition of pipeline parallel strategy configuration. Pipeline parallel requires the user to define the parallel strategy by calling the `pipeline_stage` interface to specify the stage on which each layer is to be executed. The granularity of the `pipeline_stage` interface is `Cell`. All `Cells` containing training parameters need to be configured with `pipeline_stage`, and `pipeline_stage` should be configured in the order of network execution, from smallest to largest. After adding `pipeline_stage` configuration based on the single-card model is as follows:

```python
from mindspore import nn

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.relu1= nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2= nn.ReLU()
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()
net.layer1.pipeline_stage = 0
net.relu1.pipeline_stage = 0
net.layer2.pipeline_stage = 0
net.relu2.pipeline_stage = 1
net.layer3.pipeline_stage = 1
```

### Training the Network

In this step, we need to define the loss function, the optimizer, and the training process, and unlike the single-card model, two interfaces need to be called in this section to configure the pipeline parallel:

- First define the LossCell. In this case the `nn.WithLossCell` interface is called to encapsulate the network and loss functions.
- Finally, wrap the LossCell with `nn.PipelineCell`, and specify the size of MicroBatch. For detailed information, refer to the related interfaces in the overview.

Besides, the interface `nn.PipelineGradReducer` is needed to handle gradient of pipeline parallelism, the first parameter of this interface is the network parameter to be updated.

```python
import mindspore as ms
from mindspore import nn, ops

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
net_with_loss = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 4)
net_with_loss.set_train()

def forward_fn(inputs, target):
    loss = net_with_loss(inputs, target)
    return loss

grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
pp_grad_reducer = nn.PipelineGradReducer(optimizer.parameters)

@ms.jit
def train_one_step(inputs, target):
    loss, grads = grad_fn(inputs, target)
    grads = pp_grad_reducer(grads)
    optimizer(grads)
    return loss, grads

for epoch in range(10):
    i = 0
    for data, label in data_set:
        loss, grads = train_one_step(data, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss))
        i += 1
```

> Currently pipeline parallel does not support the automatic mixed precision.
>
> Pipeline parallel training is more suitable to use `model.train` approach, because the TrainOneStep logic under pipeline parallelism is complex, while `model.train` internally encapsulates the TrainOneStepCell for pipeline parallel, which is much easier to use.

### Running the Single-host with 8 Devices Script

Next, the corresponding scripts are called by commands, using the `mpirun` startup method and the 8-card distributed training script as an example of distributed training:

```bash
bash run.sh
```

After training, the log files are saved to the `log_output` directory, where part of the file directory structure is as follows:

```text
└─ log_output
    └─ 1
        ├─ rank.0
        |   └─ stdout
        ├─ rank.1
        |   └─ stdout
...
```

The results are saved in `log_output/1/rank.*/stdout`, and the example is as below:

```text
epoch: 0 step: 0, loss is 9.087993
epoch: 0 step: 10, loss is 8.575434
epoch: 0 step: 20, loss is 8.185939
epoch: 0 step: 30, loss is 6.7301626
epoch: 0 step: 40, loss is 5.2246842
epoch: 0 step: 50, loss is 3.8342278
...
```

Other startup methods such as dynamic cluster and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/experts/en/master/parallel/startup_method.html).

## Inference Operation Practices

The following is an illustration of pipeline parallel inference operation using Ascend or GPU single-machine 8-card as an example:

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

Specify the run mode, run device, run card number, etc. via the context interface. Unlike single-card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` to be semi-automatic parallel mode and initialize HCCL or NCCL communication via init. In addition, `pipeline_stages=2` should be configured to specify the total number of stages. Not setting `device_target` here automatically specifies the backend hardware device corresponding to the MindSpore package. `pipeline_result_broadcast=True` specifies broadcast last stage inference to other stages. It is useful during auto-regression inference.

```python

import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True,
                             pipeline_stages=4, pipeline_result_broadcast=True)
init()
ms.set_seed(1)

```

### Defining the Network

The pipeline parallel network structure is basically the same as the single-card network structure, and the difference is the addition of pipeline parallel strategy configuration. Pipeline parallel requires the user to define the parallel strategy by calling the `pipeline_stage` interface to specify the stage on which each layer is to be executed. The granularity of the `pipeline_stage` interface is `Cell`. All `Cells` containing training parameters need to be configured with `pipeline_stage`, and `pipeline_stage` should be configured in the order of network execution, from smallest to largest. Configuration after adding `pipeline_stage` based on the single-card model is as follows:

```python

import numpy as np
from mindspore import lazy_inline, nn, ops, Tensor, Parameter
from mindspore.parallel.checkpoint_transform import sync_pipeline_shared_parameters

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

wrap the netork with `PipelineCellInference`, and specify the size of MicroBatch. `PipelineCellInference` splits input into several micro batch, then executes the network, and finally concats the results along the batch axis through `ops.Concat` operator.

In the previous step, the parameter `embed` is shared by `self.word_embedding` and `self.head` layer, and these two layers are split into different stages. Before inference, executing `inference_network.compile()` and `sync_pipeline_shared_parameters(inference_network)`, the framework will synchronize the shared parameter automatically.

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

# Compile and synchronize shared parameter.
input_ids = Tensor(np.random.randint(low=0, high=32, size=(8, 1)), ms.int32)
inference_network.compile(input_ids)
sync_pipeline_shared_parameters(inference_network)

# Execute the inference network
logits = inference_network(input_ids)
print(logits.asnumpy())

```

### Running the Single-host with 8 Devices Script

Next, the corresponding scripts are called by commands, using the `msrun` startup method and the 8-card distributed inference script as an example of distributed inference:

```bash

bash run_inference.sh

```

After training, the log files are saved to the `log_output` directory, where part of the file directory structure is as follows:

```text

└─ pipeline_inference_logs
   ├── scheduler.log
   ├── worker_0.log
   ├── worker_1.log
   ├── worker_2.log
...

```

The results are saved in `pipeline_inference_logs/worker_0.log`, and the example is as below:

```text

[[0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556]
  ...]

```
