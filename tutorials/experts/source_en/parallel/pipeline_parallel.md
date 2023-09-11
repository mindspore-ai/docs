# Pipeline Parallel

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/pipeline_parallel.md)

## Overview

In recent years, the scale of neural networks has increased exponentially. Limited by the memory on a single device, the number of devices used for training large models is also increasing. Due to the low communication bandwidth between servers, the performance of the conventional hybrid parallelism (data parallel + model parallel) is poor. Therefore, pipeline parallelism needs to be introduced. Pipeline parallel can divide a model in space based on `stage`. Each `stage` needs to execute only a part of the network, which greatly reduces memory overheads, shrinks the communication domain, and shortens the communication time. MindSpore can automatically convert a standalone model to the pipeline parallel mode based on user configurations.

> Hardware platforms supported by the pipeline parallel model include Ascend, GPU, and need to be run in Graph mode.

Related interfaces:

1. `mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=NUM)`: Set semi-automatic parallel mode and set `pipeline_stages` to indicate that the total number of `stages` is NUM and call it before initializing the network.

2. `nn.WithLossCell(backbone, loss_fn)`: pipeline parallelism requires first defining the Cell of the loss function, i.e., LossCell, for encapsulating the backbone network and the loss function, through this interface.

3. `nn.PipelineCell(loss_cell, micro_size)`: pipeline parallelism requires wrapping a layer of `PipelineCell` around the LossCell and specifying the size of the MicroBatch. In order to improve machine utilization, MindSpore slices the MiniBatch into finer-grained MicroBatches, and the final loss is the sum of the loss values computed by all MicroBatches, where the size of the MicroBatch must be greater than or equal to the number of `stages`.

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

## Operation Practices

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

Specify the run mode, run device, run card number, etc. via the context interface. Unlike single-card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` to be semi-automatic parallel mode and initialize HCCL or NCCL communication via init. In addition, `pipeline_stages=2` should be configured to specify the total number of `stages`. Not setting `device_target` here automatically specifies the backend hardware device corresponding to the MindSpore package.

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

```python
import mindspore as ms
from mindspore import nn, train

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
loss_cb = train.LossMonitor()
net_with_grads = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 4)
model = ms.Model(net_with_grads, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb], dataset_sink_mode=True)
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
epoch: 1 step: 1875, loss is 1.9490933418273926
epoch: 2 step: 1875, loss is 0.44548869132995605
epoch: 3 step: 1875, loss is 0.034527599811553955
epoch: 4 step: 1875, loss is 1.0163589715957642
epoch: 5 step: 1875, loss is 0.02109396457672119
epoch: 6 step: 1875, loss is 0.012739777565002441
epoch: 7 step: 1875, loss is 0.004988193511962891
epoch: 8 step: 1875, loss is 0.10372555255889893
epoch: 9 step: 1875, loss is 0.019182920455932617
epoch: 10 step: 1875, loss is 0.021012544631958008
```

Other startup methods such as dynamic networking and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/experts/en/master/parallel/startup_method.html).
