# 模型加载

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/model_loading.md)

## 概述

分布式下的模型加载主要是指分布式推理，即推理阶段采用多卡进行推理。如果训练时采用数据并行或者模型参数是合并保存，那么每张卡均持有完整的权重，每张卡推理自身的输入数据，推理方式与[单卡推理](https://www.mindspore.cn/tutorials/experts/zh-CN/master/infer/inference.html#modeleval模型验证)一致，只需要注意每卡加载同样的CheckPoint文件进行推理。
本篇教程主要介绍在多卡训练过程中，每张卡上保存模型的切片，在推理阶段采用多卡形式，按照推理策略重新加载模型进行推理的过程。针对超大规模神经网络模型的参数个数过多，模型无法完全加载至单卡中进行推理的问题，可利用多卡进行分布式推理。

> - 当模型非常大，本教程中使用[load_distributed_checkpoint](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.load_distributed_checkpoint.html)接口主机内存不足情况下，可以参考[模型转换](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_transformation.html) 章节，采用每张卡加载自身对应的切片Checkpoint的方式。
> - 若采用流水线分布式推理，则训练也必须采用流水线并行训练，并且流水线并行训练和推理所用的`device_num`以及`pipeline_stages`必须相同。流水线并行推理时，`micro_batch`为1，不需要调用`PipelineCell`，每个`stage`只需要加载本`stage`的Checkpoint文件。参考[流水线并行](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/pipeline_parallel.html)训练教程。

相关接口：

1. `mindspore.set_auto_parallel_context(full_batch=True, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)`：设置并行配置，其中`full_batch`表示是否全量导入数据集，为`True`时表明全量导入，每卡的数据相同，该场景中必须设置为`True`。`parallel_mode`为并行模式，该场景中必须设置为自动并行或者半自动并行模式。

2. `mindspore.set_auto_parallel_context(strategy_ckpt_config=strategy_ckpt_dict)`：用于设置并行策略文件的配置。`strategy_ckpt_dict`是用于设置并行策略文件的配置，是字典类型。strategy_ckpt_dict = {"load_file": "./stra0.ckpt", "save_file": "./stra1.ckpt", "only_trainable_params": False}，其中：
    - `load_file(str)`：加载并行切分策略的路径，训练阶段生成的策略文件的文件地址，分布式推理场景中该参数必须设置。默认值：""。
    - `save_file(str)`：保存并行切分策略的路径。默认值：""。
    - `only_trainable_params(bool)`：仅保存/加载可训练参数的策略信息。默认值：`True`。

3. `model.infer_predict_layout(predict_data)`：根据推理数据生成推理策略。

4. `model.infer_train_layout(data_set)`：根据训练数据生成训练策略。

5. `mindspore.load_distributed_checkpoint(network, checkpoint_filenames, predict_strategy)`：该接口对模型切片进行合并，再根据推理策略进行切分，加载至网络中。其中`network`为网络结构，`checkpoint_filenames`是Checkpoint文件的名称构成的list，按rank id顺序排列。`predict_strategy`是`model.infer_predict_layout`导出的推理策略。

## 操作实践

下面以单机8卡为例，进行分布式推理的操作说明。

### 样例代码说明

> 下载完整的样例代码：[model_saving_loading](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/model_saving_loading)。

目录结构如下：

```text
└─ sample_code
    ├─ model_saving_loading
       ├── test_loading.py
       ├── run_loading.sh
       ...
    ...
```

其中，`test_loading.py`是定义网络结构和推理的脚本。`run_loading.sh`是执行脚本。

用户首先需要按照[模型保存](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_saving.html)教程执行8卡分布式训练，训练结束后将会在当前路径生成Checkpoint文件目录以及切分策略文件：

```text
src_checkpoints/
src_strategy.ckpt
```

### 配置分布式环境

通过context接口指定运行模式、运行设备、运行卡号等，与单卡脚本不同，并行脚本还需指定并行模式`parallel_mode`为半自动并行模式，通过`strategy_ckpt_config`配置需要加载的分布式策略文件路径，并通过init初始化HCCL或NCCL通信。`device_target`会自动指定为MindSpore包对应的后端硬件设备。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(strategy_ckpt_config={"load_file": "./src_strategy.ckpt"})
init()
ms.set_seed(1)
```

### 网络定义

推理网络定义需要与训练网络相同：

```python
from mindspore import nn, ops
from mindspore.common.initializer import initializer

class Dense(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = ms.Parameter(initializer("normal", [in_channels, out_channels], ms.float32))
        self.bias = ms.Parameter(initializer("normal", [out_channels], ms.float32))
        self.matmul = ops.MatMul()
        self.add = ops.Add()

    def construct(self, x):
        x = self.matmul(x, self.weight)
        x = self.add(x, self.bias)
        return x

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.layer1 = Dense(28*28, 512)
        self.relu1 = ops.ReLU()
        self.layer2 = Dense(512, 512)
        self.relu2 = ops.ReLU()
        self.layer3 = Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()
net.layer1.matmul.shard(((2, 1), (1, 2)))
net.layer3.matmul.shard(((2, 2), (2, 1)))
```

### 推理

分布式推理与单机模式推理的区别在于，分布式推理需要调用`model.infer_predict_layout`接口和`ms.load_distributed_checkpoint`接口加载分布式参数，且输入类型必需为Tensor。代码如下：

```python
import numpy as np
import mindspore as ms

# 分布式推理场景输入类型必须为Tensor
predict_data = ms.Tensor(np.random.randn(1, 28, 28).astype(np.float32))
model = ms.Model(net)
# 得到推理策略文件
predict_layout = model.infer_predict_layout(predict_data)
# 创建checkpoint list
ckpt_file_list = ["./src_checkpoints/rank_{}/checkpoint-10_1875.ckpt".format(i) for i in range(0, get_group_size())]
# 加载分布式参数
ms.load_distributed_checkpoint(net, ckpt_file_list, predict_layout)
predict_result = model.predict(predict_data)
print(predict_result)
```

### 分布式场景导出MindIR文件

在超大规模神经网络模型的场景中，针对因为参数量过大，导致模型无法进行单卡推理的问题，可以采用分布式推理方案。此时在运行推理任务前，需要导出多个MindIR文件。核心代码如下：

```python
import mindspore as ms
from mindspore.communication import get_rank

# 导出分布式MindIR文件
ms.export(net, predict_data, file_name='./mindir/net_rank_' + str(get_rank()), file_format='MINDIR')
```

多卡训练、单卡推理的情况，导出MindIR的用法与单机相同。

### 运行单机八卡脚本

接下来通过命令调用对应的脚本，以`mpirun`启动方式，8卡的分布式推理脚本为例，进行分布式推理：

```bash
bash run_loading.sh
```

推理结果保存在`log_output/1/rank.*/stdout`中，示例如下：

```text
[[ 0.0504479  -0.94413316  0.84689146 -0.28818333  0.66444737  1.0564338
  -0.04191194  0.25590336 -0.69010115 -0.6532427 ]]
```

其中MindIR文件保存在`mindir`目录中，目录结构如下：

```text
├─ mindir
|   ├─ net_rank_0.mindir
|   ├─ net_rank_1.mindir
|   ├─ net_rank_2.mindir
|   ├─ net_rank_3.mindir
|   ├─ net_rank_4.mindir
|   ├─ net_rank_5.mindir
|   ├─ net_rank_6.mindir
|   └─ net_rank_7.mindir
...
```
