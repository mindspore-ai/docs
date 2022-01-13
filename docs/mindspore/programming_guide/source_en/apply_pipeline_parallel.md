# Pipeline Parallelism

`Ascend` `GPU` `Distributed Parallel` `Whole Process`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_en/apply_pipeline_parallel.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## Overview

In recent years, the scale of neural networks has increased exponentially. Limited by the memory on a single device, the
number of devices used for training large models is also increasing. Due to the low communication bandwidth between
servers, the performance of the conventional hybrid parallelism (data parallel + model parallel) is poor. Therefore,
pipeline parallelism needs to be introduced. Pipeline parallelism can divide a model in space based on `stage`.
Each `stage` needs to execute only a part of the network, which greatly reduces memory overheads, shrinks the
communication domain, and shortens the communication time. MindSpore can automatically convert a standalone model to the
pipeline parallel mode based on user configurations.

> Download address of the complete sample code:
>
> <https://gitee.com/mindspore/docs/tree/r1.6/docs/sample_code/distributed_training>.

The directory structure is as follows:

```text
└─sample_code
    ├─distributed_training
    │      rank_table_16pcs.json
    │      rank_table_8pcs.json
    │      rank_table_2pcs.json
    │      resnet.py
    │      resnet50_distributed_training_pipeline.py
    │      run_pipeline.sh
    ...
```

`rank_table_16pcs.json`, `rank_table_8pcs.json` and `rank_table_2pcs.json` are the networking information files. `resnet.py` and `resnet50_distributed_training_pipeline.py` are the network structure files. `run_pipeline.sh` are the execute scripts.

## Preparations

### Downloading the Dataset

This example uses the `CIFAR-10` dataset. For details about how to download and load the dataset,
visit <https://www.mindspore.cn/docs/programming_guide/en/r1.6/distributed_training_ascend.html>.

### Configuring the Distributed Environment

> Pipeline parallelism supports Ascend and GPU.

For details about how to configure the distributed environment and call the HCCL,
visit <https://www.mindspore.cn/docs/programming_guide/en/r1.6/distributed_training_ascend.html>.

## Defining the Network

The network definition is the same as that in the Parallel Distributed Training Example.

For details about the definitions of the network, optimizer, and loss function,
visit <https://www.mindspore.cn/docs/programming_guide/en/r1.6/distributed_training_ascend.html>.

> To implement pipeline parallelism, you need to define the parallel strategy and call the `pipeline_stage` API to specify the stage on which each layer is to be executed. The granularity of the `pipeline_stage` API is `Cell`. `pipeline_stage` must be configured for all `Cells` that contain training parameters.

```python
class ResNet(nn.Cell):
    """ResNet"""

    def __init__(self, block, num_classes=100, batch_size=32):
        """init"""
        super(ResNet, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.head = Head()
        self.layer1 = MakeLayer0(block, in_channels=64, out_channels=256, stride=1)
        self.layer2 = MakeLayer1(block, in_channels=256, out_channels=512, stride=2)
        self.layer3 = MakeLayer2(block, in_channels=512, out_channels=1024, stride=2)
        self.layer4 = MakeLayer3(block, in_channels=1024, out_channels=2048, stride=2)

        self.pool = ops.ReduceMean(keep_dims=True)
        self.squeeze = ops.Squeeze(axis=(2, 3))
        self.fc = fc_with_initialize(512 * block.expansion, num_classes)

        # pipeline parallel config
        self.head.pipeline_stage = 0
        self.layer1.pipeline_stage = 0
        self.layer2.pipeline_stage = 0
        self.layer3.pipeline_stage = 1
        self.layer4.pipeline_stage = 1
        self.fc.pipeline_stage = 1

    def construct(self, x):
        """construct"""
        x = self.head(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x, (2, 3))
        x = self.squeeze(x)
        x = self.fc(x)
        return x
```

## Training the Network

To enable pipeline parallelism, you need to add the following configurations to the training script:

- Set `pipeline_stages` in `set_auto_parallel_context` to specify the total number of `stages`.
- Set the `SEMI_AUTO_PARALLEL` mode. Currently, the pipeline parallelism supports only this mode.
- Define the LossCell. In this example, the `nn.WithLossCell` API is called.
- Pass the `parameters` used in this `stage` to the optimizer. Call the `add_pipeline_stage` method of `Parameter` to
  pass all `stage` information to `Parameter` if multiple `stages` share a parameter. Then, call
  the `infer_param_pipeline_stage` API of the `Cell` to obtain the training parameters of the `stage`.
- Finally, wrap the LossCell with `PipelineCell`, and specify the Micro_batch size. To improve machine utilization,
  MindSpore divides Mini_batch into finer-grained Micro_batch to streamline the entire cluster. The final loss value is
  the sum of the loss values computed by all Micro_batch. The size of Micro_batch must be greater than or equal to the
  number of `stages`.

```python
from mindspore import context, Model, nn
from mindspore.nn import Momentum
from mindspore.train.callback import LossMonitor
from mindspore.context import ParallelMode
from resnet import resnet50


def test_train_cifar(epoch_size=10):
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=True)
    context.set_auto_parallel_context(pipeline_stages=2, save_graphs=True)
    loss_cb = LossMonitor()
    data_path = os.getenv('DATA_PATH')
    dataset = create_dataset(data_path)
    batch_size = 32
    num_classes = 10
    net = resnet50(batch_size, num_classes)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    net_with_loss = nn.WithLossCell(net, loss)
    net_pipeline = nn.PipelineCell(net_with_loss, 2)
    opt = Momentum(net.infer_param_pipeline_stage(), 0.01, 0.9)
    model = Model(net_pipeline, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=True)
```

## Running the Single-host with 8 devices Script

Using the sample code, you can run a 2-stage pipeline on 8 Ascend devices using below scripts:

```bash
bash run_pipeline.sh [DATA_PATH] Ascend
```

You can run a 2-stage pipeline on 8 GPU devices using below scripts:

```bash
bash run_pipeline.sh [DATA_PATH] GPU
```