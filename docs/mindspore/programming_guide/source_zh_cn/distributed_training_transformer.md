# 分布式并行训练Transformer模型

`Ascend` `分布式并行`

<!-- TOC -->

- [分布式并行训练Transformer模型](#分布式并行训练transformer模型)
    - [概述](#概述)
    - [并行配置定义](#并行配置定义)
    - [模型定义](#模型定义)
        - [Embedding层](#embedding层)
        - [Transformer层](#transformer层)
        - [定义损失函数](#定义损失函数)
    - [端到端流程](#端到端流程)
    - [准备环节](#准备环节)
        - [下载数据集](#下载数据集)
        - [预处理流程](#预处理流程)
        - [配置分布式环境变量](#配置分布式环境变量)
        - [调用集合通信库](#调用集合通信库)
    - [运行脚本](#运行脚本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/distributed_training_transformer.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

近年来，基于Transformer的预训练模型参数量越来越大，而Ascend 910、GPU等设备内存的增长显著小于模型大小增长的速度。因此，将Transformer模型进行并行训练已经一个非常迫切的需求。MindSpore提供了一个分布式的Transformer接口`mindspore.nn.transformer.transformer`，将Transformer内部用到的每个算子都配置了并行策略，而用户只需要配置全局的`data_parallel`和`model_parallel`属性，即可完成分布式并行策略的配置。可以极大地方便用户应用Transformer进行分布式训练。目前分布式训练支持Ascend 910和GPU环境，总结如下：

- `Transformer`提供了简单的并行配置，即可实现算子级别并行和流水线并行。

> 你可以在这里下载完整的样例代码：
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training_transformer>

目录结构如下：

```text
└─sample_code
    ├─distribute_training_transformer
        ├── dataset.py
        ├── model.py
        ├── parallel_recover_train.py
        ├── parallel_save_ckpt_train.py
        ├── preprocess.py
        ├── rank_table_16pcs.json
        ├── rank_table_2pcs.json
        ├── rank_table_8pcs.json
        ├── run_cluster.sh
        ├── run_parallel_recover_ckpt.sh
        ├── run_parallel_save_ckpt.sh
        ├── run.sh
        └── train.py
```

其中，`rank_table_8pcs.json`和`rank_table_2pcs.json`是配置当前多卡环境的组网信息文件。`model.py`、`dataset.py`和`train.py`三个文件是定义数据导入，网络结构的脚本和训练文件。`run.sh`是执行脚本。

使用`mindspore.parallel`中的`Transformer`库，用户需要决定并行配置和模型这两个部分的入参，即可完成分布式配置。**分布式配置仅在半自动和自动并行模式下生效**。

## 并行配置定义

针对`Transformer`中网络的定义和实现，我们为每个算子设置了对应的切分策略。用户根据自己的需求，设置全局的并行配置可以实现`Transformer`网络的并行配置。`Transformer`目前定义的并行配置主要有三个类别`TransformerOpParallelConfig`、`OpParallelConfig`和`EmbeddingOpParallelConfig`。`TransformerOpParallelConfig`的导入路径为`mindspore.nn.transformer`，它可以配置的属性如下所示：

- data_parallel (int): # 设置数据并行数，默认值为1。
- model_parallel (int): # 设置模型并行数，默认值为1。
- pipeline_stage (int): # 设置Pipeline Stage数目，默认值为 1。
- micro_batch_num (int): # 设置输入Batch的切分个数，即将一个Batch切分成多个小batch，默认值为1。
- optimizer_shard (bool): # 是否开启优化器并行，默认值为False。
- gradient_aggregation_group (int): # 优化器并行对应梯度聚合个数，默认值为4。
- recompute (bool): # 是否开启重计算，默认值为False。
- vocab_emb_dp (bool): # 是否配置Embedding为数据并行，默认值为True。

我们会在接下来讨论他们的区别。现在以单机八卡训练一个`Transformer`模型为例，我们根据目前的卡数8设置`Transformer`模型的并行配置。我们可以设置`data_parallel`=1，`model_parallel`=8作为并行的基本配置。注意并行配置的情况下，`data_parallel`\*`model_parallel`\*`pipeline_stages`<=总卡数。对应的代码中的**并行配置**如下。

```python
from mindspore import context
from mindspore.nn.transformer import TransformerOpParalllelConfig
context.set_auto_parallel_context(parallel_mode=context.ParallelMode.SEMI_AUTO_PARALLEL)
parallel_config = TransformerOpParalllelConfig(data_parallel=1, model_parallel=8)
```

## 模型定义

在定义好配置之后，我们可以开始构造一个网络。由于MindSpore已经提供了`Transformer`的使用，用户只需要额外增加`Embedding`层，输出层和损失函数即可。下面依次介绍各个模块的配置。

### Embedding层

Tranformer中的Embeding层主要由词向量嵌入和位置向量嵌入两部分组成。我们提供了`VocabEmbedding`作为并行的Embedding层，需要传入`EmbeddingOpParallelConfig`进行初始化。和`OpParallelConfig`不同的是，`EmbeddingOpParallelConfig`拥有的属性如下

- `data_parallel`: 设置数据并行数，默认值为1。
- `model_parallel`: 设置模型并行数，默认值为1。
- `vocab_emb_dp`: 是否配置Embedding为数据并行，默认值为True。

`vocab_emb_dp`用来区分`embedding_lookup`操作的两种并行模式`数据并行`和`行切分并行`。当`vocab_emb_dp`为`True`时，embedding查找的过程将会被设置为并行度为`data_parallel`的数据并行。当`vocab_emb_dp`为`False`时，embedding的权重将会在第0维度按`model_parallel`进行均分，可以减少变量的存储。

在此我们定义了一个`EmbeddingLayer`，将查询的词向量和位置向量进行相加求和。注意，我们在此设置了`add`和`dorpout`操作。由于输入的tensor大小为`[batch_size, seq_length, hidden_szie]`，并且词向量的查找过程为数据并行，所以我们根据`OpParallelConfig`中的数据并行值`data_parallel`，调用算子的`shard`方法分别设置这两个算子的并行策略。如果用户不进行设置`shard`方法，那么默认的算子并行策略为**并行度为卡数的数据并行**。那么完成对应的代码如下所示:

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.transformer import VocabEmbedding
class EmbeddingLayer(nn.Cell):
    def __init__(self, vocab_size, position_size, embedding_size,
                 parallel_config, dropout_rate=0.1):
        super(EmbeddingLayer, self).__init__()
        self.word_embedding = VocabEmbedding(vocab_size=vocab_size,
                                             embedding_size=embedding_size,
                                             parallel_config=parallel_config)
        self.position_embedding = VocabEmbedding(vocab_size=position_size,
                                                 embedding_size=embedding_size,
                                                 parallel_config=parallel_config)
        self.add = ops.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        self.dropout = nn.Dropout(1 - dropout_rate)
        self.dropout.dropout.shard(((parallel_config.data_parallel, 1, 1),))

    def construct(self, input_ids, input_position):
        word_embedding, word_table = self.word_embedding(input_ids)
        position_embedding, _ = self.position_embedding(input_position)
        embed = self.add(word_embedding, position_embedding)
        embed = self.dropout(embed)
        return embed, word_table
```

注意我们还将词嵌入的embedding_table作为返回值返回了。

### Transformer层

用户可以调用三个接口作为主要的构建API:`Transformer`、`TransformerEncoder`和`TransformerDecoder`。它们都需要传入`TransformerOpParallelConfig`作为并行设置的配置。我们根据`TransformerOpParallelConfig`中配置的并行配置，对`Transformer`内部使用的算子设置对应的并行策略。

> `pipeline_func`这个方法可以设置transformer中每个`block`属于的`stage`、是否开启重计算和优化器切分的融合标记。例如下面的例子中，我们根据传入的`layer_id`和`offset`(在`Transformer`接口中，在实例化`Encoder`时传入的`offset`为0， `Decoder`中传入的`offset`的值为`Encoder`的层数), `Encoder_layer`和`Decoder_layer`的总层数，和指定的`pipeline_stage`数目，按照均分的配置计算出当前的`block`对应的`stage`。在默认情况下，即用户不传入`lambda_func`的情况下，也是按照层数进行均分的设置。

```python
def pipeline_func(network, layer_id, offset, parallel_config, layers):
    layers_per_stage = 2
    pp_id = max(int(layer_id + offset) / layers_per_stage, 1)
    network.pipeline_stage = int(pp_id)
    print(f"pipeline id is:{pp_id}", flush=True)
```

在下面的代码中，我们实例化了上述定义的`EmbeddingLayer`，并且调用`set_comm_fusion`将其对应的反向梯度融合标记为第0组，调用`pipeline_stage`方法设置对应embedding的权重为第0个`stage`。将最后的`Head`类，一个简单的`Linear`层，放置于最后一个`stage`。在用户不设置Linear中的算子并行策略的情况下，默认是当前`stage`内的数据并行。

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.transformer import Transformer, AttentionMask, CrossEntropyLoss
from mindspore.nn import Dense as Linear
class Net(nn.Cell):
    """
      Single Transformer Model
    """
    def __init__(self, batch, src_len, tgt_len, hidden_size, vocab_size,
                 en_layer, de_layer, parallel_config, return_loss=False):
        super(Net, self).__init__()
        self.src_embedding = EmbeddingLayer(vocab_size=vocab_size, embedding_size=hidden_size,
                                            position_size=src_len,
                                            parallel_config=parallel_config.embedding_dp_mp_config)
        self.tgt_embedding = EmbeddingLayer(vocab_size=vocab_size, embedding_size=hidden_size,
                                            position_size=tgt_len,
                                            parallel_config=parallel_config.embedding_dp_mp_config)
        total_layers = en_layer + de_layer + 2
        layers_per_stage = total_layers // parallel_config.pipeline_stage
        self.src_embedding.pipeline_stage = 0
        self.tgt_embedding.pipeline_stage = 0
        self.return_loss = return_loss

        def pipeline_func(network, layer_id, offset, parallel_config, layers):
            pp_id = max(int(layer_id + offset) / layers_per_stage, 1)
            network.pipeline_stage = int(pp_id)
            gradient_aggregation_group = 4
            dis = max(int((layer_id + offset) / gradient_aggregation_group), 1)
            network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
            print(f"pipeline id is:{pp_id}", flush=True)

        self.base1 = Transformer(encoder_layers=en_layer,
                                 decoder_layers=de_layer,
                                 batch_size=batch,
                                 src_seq_length=src_len,
                                 tgt_seq_length=tgt_len,
                                 hidden_size=hidden_size,
                                 num_heads=8,
                                 attention_dropout_rate=0.0,
                                 hidden_dropout_rate=0.0,
                                 lambda_func=pipeline_func,
                                 ffn_hidden_size=hidden_size,
                                 parallel_config=parallel_config)

        self.attention_mask = AttentionMask(seq_length=tgt_len)
        self.head = Linear(in_channels=hidden_size, out_channels=vocab_size, has_bias=False)
        self.head.matmul.shard(((1, 1), (1, 1)))
        self.head.pipeline_stage = parallel_config.pipeline_stage - 1
        self.loss = CrossEntropyLoss(parallel_config=parallel_config.dp_mp_config)
        self.no_equal = ops.NotEqual().shard(((1, 1), ()))

```

### 定义损失函数

MindSpore还提供了一个支持并行的交叉商损失函数`mindspore.nn.transformer.CrossEntroyLoss`。这个函数接收一个`OpParallelConfig`来配置并行属性。`OpParallelConfig`实际包含了两个属性`data_parallel`和`model_parallel`。通过将模型的输出和真实标签输入损失函数，我们即可计算当前数据对应的损失值。

```python
from mindspore.nn.transformer import CrossEntropyLoss
self.loss = CrossEntropyLoss(parallel_config=parallel_config.dp_mp_config)
```

## 端到端流程

在定义并行配置、模型和损失函数之后，我们可以将上述代码整合完成训练过程。在启动训练之前，我们调用`auto_parallel_context`设置并行选项，设置并行模式为`SEMI_AUTO_PARALLEL`。在流水线并行的情况下，MindSpore提供了额外的配置可以通信为代价额外节省内存。其过程如下：在含有数据并行维度的并且开启优化器切分的情况下(`enable_parallel_optimizer=True`)，
通过设置`parallel_optimizer_config= {"gradient_accumulation_shard":True}`可以将流水线并行训练时的累积变量进一步切分，以达到节省内存的目的，同时会在每个`micro_step`之间引入通信以保证每卡梯度的一致性。

```python
from mindspore import context
from mindspore.context import ParallelMode
context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False, full_batch=True, loss_repeated_mean=True, device_num=device_num, enable_parallel_optimizer=True, parallel_optimizer_config = {"gradient_accumulation_shard": gradient_accumulation_shard})
```

关于`stage_num`的说明如下，MindSpore通过`stage_num`来判断是否进入流水线并行训练。

- 在设置`stage_num=1`的情况下，进行算子级别的并行。用户可以通过设置`TransformerOpParallelConfig`中的`model_parallel`和`data_parallel`属性进行配置并行训练。
- 在设置`stage_num>1`的情况下，会进入流水线并行模式。流水线的配置就是设置每个`cell`对应的`pipeline_stage`属性，另外，在实例化网络中后，我们需要再调用`PipelineCell`来封装定义好的网络。这个`Cell`的作用是将输入切分成`mirco_batch_num`个数的小数据，以最大利用计算资源。值得注意的是，我们需要调用`net.infer_param_pipeline_stage()`而不是`net.trainable_params()`来获取当前`stage`对应的训练权重。注意，pipeline的stage内的卡数至少为8。pipeline的详细教程可以参考[这里](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/apply_pipeline_parallel.html)。

整合后的主文件代码如下。

```python
from mindspore.nn.transformer import TransformerOpParallelConfig
from mindspore import Model
import mindspore.communication as D
from mindspore.context import ParallelMode
from mindspore.nn import PipelineCell
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.nn import AdamWeightDecay
from mindspore import context
from dataset import ToyDataset, Tokenzier
from model import Net


def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    return group_params


def main():
    # Run the total forward model
    ...
    args_opt = parser.parse_args()
    ...

    if args_opt.distribute == 'true':
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        dp = device_num // args_opt.mp // args_opt.pipeline_stage
        print("rank_id is {}, device_num is {}, dp is {}".format(rank_id, device_num, dp))
        gradient_accumulation_shard = dp > 1 and args_opt.pipeline_stage > 1
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
            full_batch=True, loss_repeated_mean=True,
            device_num=device_num, enable_parallel_optimizer=True,
            parallel_optimizer_config={"gradient_accumulation_shard": gradient_accumulation_shard})
    else:
        dp = 1

    parallel_config = TransformerOpParallelConfig(pipeline_stage=args_opt.pipeline_stage,
                                                  micro_batch_num=args_opt.micro_batch_num,
                                                  model_parallel=args_opt.mp,
                                                  data_parallel=dp)

    net = Net(batch=args_opt.batch_size // args_opt.micro_batch_num if args_opt.pipeline_stage else args_opt.batch_size,
              src_len=args_opt.src_len, tgt_len=args_opt.tgt_len,
              vocab_size=args_opt.vocab_size,
              hidden_size=args_opt.d_model,
              en_layer=args_opt.encoder_layer,
              de_layer=args_opt.decoder_layer,
              parallel_config=parallel_config, return_loss=args_opt.train)

    tokenizer = Tokenzier()
    task = ToyDataset(file_path=args_opt.file_path,
                      tokenizer=tokenizer,
                      seq_length=(args_opt.src_len, args_opt.tgt_len))
    dataset = task.get_dataset(batch_size=args_opt.batch_size)

    if args_opt.pipeline_stage > 1:
        net = PipelineCell(net, args_opt.micro_batch_num)
        param = net.infer_param_pipeline_stage()
        print(f"params is:{param}", flush=True)
        group_params = set_weight_decay(param)
        opt = AdamWeightDecay(group_params, learning_rate=args_opt.lr)
    else:
        group_params = set_weight_decay(net.trainable_params())
        opt = AdamWeightDecay(group_params, learning_rate=args_opt.lr)

    if not args_opt.train:
        model = Model(net)
    else:
        model = Model(net, optimizer=opt)

    callback_size = 1
    # single vs pipeline (save a slice of the model)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=callback_size, keep_checkpoint_max=4,
                                   integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix="test",
                                 config=ckpt_config)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size), ckpoint_cb]
    model.train(1, dataset, callbacks=callback, dataset_sink_mode=False)

if __name__ == "__main__":
    main()
```

## 准备环节

### 下载数据集

- [WMT14 En-Fr数据集下载](http://statmt.org/wmt14/test-full.tgz)  

使用`newstest2014-fren-ref.en.sgm`作为该任务的训练集合，合并且清洗该数据集。将数据集解压至`docs/sample_code/distributed_training_transformer`目录下。

### 预处理流程

执行下述代码进行数据的预处理过程，将会在当前目录下产生`output`目录，目录下将会生成`wmt14.en_ft.txt`和`wmt14.fr_en.txt`两个文件，文件中每行是一个法语和英语的句子对。我们将采用`wmt14.fr_en.txt`作为训练数据。

```python
python preprocess.py
```

### 配置分布式环境变量

在裸机环境（对比云上环境，即本地有Ascend 910 AI 处理器）进行分布式训练时，需要配置当前多卡环境的组网信息文件。如果使用华为云环境，因为云服务本身已经做好了配置，可以跳过本小节。

以Ascend 910 AI处理器为例，1个8卡环境的json配置文件示例如下，本样例将该配置文件命名为`rank_table_8pcs.json`。2卡环境配置可以参考样例代码中的`rank_table_2pcs.json`文件。

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.155.111.140",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

其中需要根据实际训练环境修改的参数项有：

- `server_count`表示参与训练的机器数量。
- `server_id`表示当前机器的IP地址。
- `device_id`表示卡物理序号，即卡所在机器中的实际序号。
- `device_ip`表示集成网卡的IP地址，可以在当前机器执行指令`cat /etc/hccn.conf`，`address_x`的键值就是网卡IP地址。
- `rank_id`表示卡逻辑序号，固定从0开始编号。

### 调用集合通信库

MindSpore分布式并行训练的通信使用了华为集合通信库`Huawei Collective Communication Library`（以下简称HCCL），可以在Ascend AI处理器配套的软件包中找到。同时`mindspore.communication.management`中封装了HCCL提供的集合通信接口，方便用户配置分布式信息。
> HCCL实现了基于Ascend AI处理器的多机多卡通信，有一些使用限制，我们列出使用分布式服务常见的，详细的可以查看HCCL对应的使用文档。
>
> - 单机场景下支持1、2、4、8卡设备集群，多机场景下支持8*n卡设备集群。
> - 每台机器的0-3卡和4-7卡各为1个组网，2卡和4卡训练时卡必须相连且不支持跨组网创建集群。
> - 组建多机集群时需要保证各台机器使用同一交换机。
> - 服务器硬件架构及操作系统需要是SMP（Symmetrical Multi-Processing，对称多处理器）处理模式。

下面是调用集合通信库样例代码：

```python
import os
from mindspore import context
from mindspore.communication import init

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
    init()
    ...
```

其中，

- `mode=context.GRAPH_MODE`：使用分布式训练需要指定运行模式为图模式（PyNative模式不支持并行）。
- `device_id`：卡的物理序号，即卡所在机器中的实际序号。
- `init`：使能HCCL通信，并完成分布式训练初始化操作。

## 运行脚本

上述已将训练所需的脚本编辑好了，接下来通过命令调用对应的脚本。

目前MindSpore分布式执行采用单卡单进程运行方式，即每张卡上运行1个进程，进程数量与使用的卡的数量一致。其中，0卡在前台执行，其他卡放在后台执行。每个进程创建1个目录，用来保存日志信息以及算子编译信息。下面以使用8张卡的分布式训练脚本为例，演示如何运行脚本：

```bash
#!/bin/bash
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATA_PATH RANK_SIZE"
echo "For example: bash run.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}
RANK_SIZE=$2

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

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./train.py ./model.py ./dataset.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python ./train.py --distribute=true --file_path=${DATA_PATH} --mp=${RANK_SIZE} > train.log$i 2>&1 &
    cd ../
done
rm -rf device0
mkdir device0
cp ./train.py ./model.py ./dataset.py ./device0
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
python ./train.py --distribute=true --file_path=${DATA_PATH} --mp=${RANK_SIZE} > train.log0 2>&1 &
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
cd ../
```

脚本需要传入变量`DATA_PATH`和`RANK_SIZE`，分别表示`wmt14.fr_en.txt`数据集绝对路径和卡的数量。

分布式相关的环境变量有，

- `RANK_TABLE_FILE`：组网信息文件的路径。
- `DEVICE_ID`：当前卡在机器上的实际序号。
- `RANK_ID`：当前卡的逻辑序号。

其余环境变量请参考安装教程中的配置项。

运行时间大约在5分钟内，主要时间是用于算子的编译，实际训练时间在20秒内。用户可以通过`ps -ef | grep python`来监控任务进程。

日志文件保存到`rank`所对应的`device0`、 `device1`......目录下，`env.log`中记录了环境变量的相关信息，关于Loss部分结果保存在`train.log`中，示例如下：

```text
epoch: 1 step: 1, loss is 9.9034
epoch: 1 step: 2, loss is 9.9033
epoch: 1 step: 3, loss is 9.9031
epoch: 1 step: 4, loss is 9.9025
epoch: 1 step: 5, loss is 9.9022
```
