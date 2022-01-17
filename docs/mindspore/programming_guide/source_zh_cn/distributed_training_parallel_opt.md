# 优化器并行

`Ascend` `分布式并行`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/distributed_training_parallel_opt.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

在进行数据并行训练时，模型的参数更新部分在各卡间存在冗余计算，优化器并行通过将优化器的计算量分散到数据并行维度的卡上，在大规模网络上（比如Bert、GPT）可以有效减少内存消耗并提升网络性能。

在数据并行模式下使能优化器并行，框架会将需要更新的参数进行分散到不同卡上，各自更新后再通过Broadcast算子在集群间做权重共享。需要注意的是参数量应当大于机器数，当前只支持Lamb和AdamWeightDecay优化器。

在auto_parallel或者semi_auto_parallel模式下使能优化器并行，如果经过策略切分后的参数在机器间存在重复切片，并且shape的最高维可以被重复切片的卡数整除，框架会以最小切片的方式保存参数并在优化器中更新。该模式下支持所有优化器。

| 并行模式      | 参数更新方式                                        | 优化器支持            |
| ------------- | --------------------------------------------------- | --------------------- |
| 数据并行      | 参数分组更新，然后广播到所有卡                      | Lamb和AdamWeightDecay |
| 全/半自动并行 | 参数按数据并行度切分成N份，每张卡更新当前卡上的参数 | 所有优化器            |

无论是哪种模式，优化器并行不会影响原有正反向网络的计算图，只会影响参数更新的计算量和计算逻辑。

>你可以在这里下载完整的样例代码：
>
><https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_optimizer_parallel>。

目录结构如下：

```text
└─sample_code
    ├─distributed_optimizer_parallel
        ├── fusion_example.py
        ├── rank_table_2pcs.json
        ├── rank_table_8pcs.json
        └── run_fusion_example.sh
```

其中每个文件的作用如下：

- fusion_example.py：优化器融合的示例代码，阐述了如何配置优化器的融合标记。
- rank_table_2pcs.json：RANK_TABLE_FILE的2卡配置文件。
- rank_table_8pcs.json：RANK_TABLE_FILE的8卡配置文件。
- run_fusion_example.sh：优化器融合代码的启动脚本。

## 开启优化器并行

在`mindspore.context.set_auto_parallel_context`中提供了`enable_parallel_optimizer`选项，将其配置为True后，即可使能优化器并行，默认对所有参数进行优化器切分。

```python
from mindspore import context
context.set_auto_parallel_context(enable_parallel_optimizer=True)
```

## 配置参数优化器并行

此外，用户还可以自定义某些参数是否优化器切分。Parameter提供了一个`parallel_optimizer`的参数，用来配置当前的参数是否进行优化器切分。因此用户单独针对每个参数配置是否开启优化器并行，如下所示

```python
import numpy as np
from mindspore import Parameter, Tensor
param = Parameter(Tensor(np.ones((10, 2))), name='weight1', parallel_optimizer=True)

# 或者通过下述的方式配置
param2 = Parameter(Tensor(np.ones((10, 2))), name='weight2')
param2.parallel_optimizer = False
```

## 配置通信融合

在设置参数优化器并行一节中，我们阐述了如何配置每个参数的优化器并行属性。在全/半自动模式下，每个参数都会产生一个对应的AllGather操作和ReduceScatter操作。这些通信算子是自动并行框架自动插入的。然而，随着参数量增多，对应的通信算子也会增多，通信操作产生的算子调度和启动都会产生更多的开销。因此，可以通过`cell`提供的`set_comm_fusion`方法，对每个`cell`内的参数对应的AllGather和ReduceScatter操作配置融合标记。

`Transformer`接口提供了`lambda_func`参数来自定义每层的分组融合标记。如下代码所示，通过此接口将Transformer中每一层的参数配置了fusion值，融合的通信算子个数设置为2个。

```python
"""Parallel Optimizer Fusion Example"""
from mindspore.communication import init
from mindspore import context, ParallelMode
from mindspore.nn.transformer import Transformer, TransformerOpParallelConfig

init()
context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, enable_parallel_optimizer=True)
def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):

    # 将transformer的融合层数设置为4个
    gradient_aggregation_group = 4
    dis = max(int((layers + offset) / gradient_aggregation_group), 1)
    # 此处的network是一个cell，用户可以针对自己的网络层调用set_comm_fusion方法
    network.set_comm_fusion(int((layer_id + offset) / dis) + 1)

model_parallel_config = TransformerOpParallelConfig()
net = Transformer(encoder_layers=1, decoder_layers=1,
                  batch_size=4, src_seq_length=10,
                  tgt_seq_length=10, hidden_size=24,
                  num_heads=8, attention_dropout_rate=0.0,
                  hidden_dropout_rate=0.0, lambda_func=set_parallel_configure_for_layer,
                  ffn_hidden_size=24, parallel_config=model_parallel_config)
for item in net.trainable_params():
    if 'dense1' in item.name:
        print(f"The parameter {item.name}'s fusion id is {item.comm_fusion}")
```

对应的输出如下，表示了每层特定dense的funsion值：

```text
The parameter encoder.blocks.0.attention.dense1.weight's fusion id is 1
The parameter encoder.blocks.0.attention.dense1.bias's fusion id is 1
The parameter decoder.blocks.0.attention.dense1.weight's fusion id is 2
The parameter decoder.blocks.0.attention.dense1.bias's fusion id is 2
The parameter decoder.blocks.0.cross_attention.dense1.weight's fusion id is 2
The parameter decoder.blocks.0.cross_attention.dense1.bias's fusion id is 2
```

>在编译图的流程中，相同融合标记并且是相同的通信操作，会被融合成一个通信操作。从而减少通信操作的数量。对于融合标记为0的通信算子时，优化流程中不会对它们进行融合。

开启优化器切分时，网络中每个参数都会产生一个相应的通信算子，然而频繁地调用通信算子将造成较多的算子启动消耗。将这些通信算子融合成一个通信算子，是最有效减少通信算子个数的办法。MindSpore提供了但这样会导致计算资源的浪费。例如，将所有的通信算子融合成一个算子后，在当前训练迭代中，NPU需要等待切分的参数汇聚完成后才能进行网络的前向计算。这样会造成设备的等待。

为了避免上述问题，可以将网络参数进行分组融合：在上一组参数进行的计算的同时，进行下组参数的通信，使得计算和通信能够互相隐藏。这就是上述代码将通信算子个数融合成2个而不是1个的原因。

## 运行代码

上述代码需要在配置分布式变量后才可以运行。Ascend环境需要配置RANK_TABLE_FILE、RANK_ID和DEVICE_ID。配置的过程请参考[此处](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_ascend.html#id4)，GPU环境需要配置[OpenMPI](https://www.mindspore.cn/tutorials/zh-CN/master/intermediate/distributed_training/distributed_training_gpu.html?highlight=openmpi)、NCCL和[HOST_FILE](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_gpu.html#id14)，配置的过程请参考[此处](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_gpu.html#id3)。

Ascend分布式相关的环境变量有:

- RANK_TABLE_FILE：组网信息文件的路径。rank_table_file文件可以使用models代码仓中的hccl_tools.py生成，可以从[此处](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)获取。
- DEVICE_ID：当前卡在机器上的实际序号。
- RANK_ID：当前卡的逻辑序号。

GPU分布式相关的环境变量：

- HOST_FILE: 描述多卡训练时的设备IP和个数。文件每一行格式为[hostname] slots=[slotnum]，hostname可以是ip或者主机名。需要注意的是，不同机器上的用户名需要相同，但是hostname不可以相同。

用户可以通过[此处](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_optimizer_parallel)获取上述的此文档中的脚本。执行下述的`bash`脚本即可运行程序，输出日志在device0/train.log0文件。

```bash
#!/bin/bash
set -e
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_fusion_example.sh DATA_PATH RANK_SIZE"
echo "For example: bash run_fusion_example.sh 8"
echo "It is better to use the absolute path."
echo "This example is expected to run on the Ascend environment."
echo "=============================================================================================================="
RANK_SIZE=$1

EXEC_PATH=$(pwd)

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./fusion_example.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    pytest -s -v ./fusion_example.py > train.log$i 2>&1 &
    cd ../
done
echo "The program launch succeed, the log is under device0/train.log0."
```

在当前目录下配置完RANK_TABLE_FILE之后，下述的命令要求用户拥有8张Ascend 910设备。运行命令如下：

```bash
bash run_fusion_example.sh 8
```
