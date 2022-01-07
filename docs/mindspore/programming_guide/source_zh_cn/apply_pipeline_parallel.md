# 流水线并行

`Ascend` `GPU` `分布式并行` `全流程`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/apply_pipeline_parallel.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

近年来，神经网络的规模几乎是呈指数型增长。受单卡内存的限制，训练这些大模型用到的设备数量也在不断增加。受server间通信带宽低的影响，传统数据并行叠加模型并行的这种混合并行模式的性能表现欠佳，需要引入流水线并行。流水线并行能够将模型在空间上按`stage`
进行切分，每个`stage`只需执行网络的一部分，大大节省了内存开销，同时缩小了通信域，缩短了通信时间。MindSpore能够根据用户的配置，将单机模型自动地转换成流水线并行模式去执行。

> 你可以在这里下载完整的样例代码：
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training>。

## 准备环节

### 下载数据集

本样例采用`CIFAR-10`
数据集，数据集的下载和加载方式可参考：<https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_ascend.html>。

### 配置分布式环境

> 流水线并行支持Ascend和GPU。

分布式环境的配置以及集合通信库的调用可参考：<https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_ascend.html>。

## 定义网络

网络的定义和Ascend的分布式并行训练基础样例中一致。

网络、优化器、损失函数的定义可参考：<https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_ascend.html>。

> 流水线并行需要用户去定义并行的策略，通过调用`pipeline_stage`接口来指定每个layer要在哪个stage上去执行。`pipeline_stage`接口的粒度为`Cell`。所有包含训练参数的`Cell`都需要配置`pipeline_stage`，并且`pipeline_stage`要按照网络执行的先后顺序，从小到大进行配置。

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

## 训练网络

为了使能流水线并行，需要在训练脚本中加一些必要的配置：

- 在`set_auto_parallel_context`中设置`pipeline_stages`，`pipeline_stages`用来表明`stage`的总数。
- 目前流水线并行只支持`SEMI_AUTO_PARALLEL`模式，数据集要以`full_batch`模式导入。
- 需要定义LossCell，本例中调用了`nn.WithLossCell`接口。
- 目前流水线并行不支持自动混合精度特性。
- 优化器需要传入本`stage`用到的`parameters`。若有多个`stage`共用了一个参数，则需要调用`Parameter`的`add_pipeline_stage`方法，将所有`stage`信息传给`Parameter`
  。随后，可以调用`Cell`的`infer_param_pipeline_stage`接口来获取本`stage`的训练参数。
- 最后，需要在LossCell外包一层`PipelineCell`
  ，并指定Micro_batch的size。为了提升机器的利用率，MindSpore将Mini_batch切分成了更细粒度的Micro_batch，从而能够使整个集群流水线起来，最终的loss则是所有Micro_batch计算的loss值的加和。其中，Micro_batch的size必须大于等于`stage`
  的数量。

```python
from mindspore import context, Model, nn
from mindspore.nn import Momentum
from mindspore.train.callback import LossMonitor
from mindspore.context import ParallelMode
from resnet import resnet50


def test_train_cifar(epoch_size=10):
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=True)
    context.set_auto_parallel_context(pipeline_stages=2, full_batch=True)
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

## 运行单机八卡脚本

利用样例代码，

Ascend可以用以下命令运行8卡，2个stage的流水线训练：

```bash
bash run_pipeline.sh [DATA_PATH] Ascend
```

GPU可以用以下命令运行8卡，2个stage的流水线训练：

```bash
bash run_pipeline.sh [DATA_PATH] GPU
```
