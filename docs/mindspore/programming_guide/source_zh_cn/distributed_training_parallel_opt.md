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
><https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training_transformer>。

目录结构如下：

```text
└─sample_code
    ├─distribute_training_transformer
        ├── dataset.py
        ├── model.py
        ├── rank_table_16pcs.json
        ├── rank_table_2pcs.json
        ├── rank_table_8pcs.json
        ├── run_cluster.sh
        ├── run.sh
        └── train.py
```

## 设置参数优化器并行

在`mindspore.context.set_auto_parallel_context`中提供了`enable_parallel_optimizer`选项，将其配置为True后，即可使能优化器并行。

此外，用户还可以自定义某些参数是否优化器切分。Parameter提供了一个`parallel_optimizer`的参数，用来配置当前的参数是否进行优化器切分。如下述的代码进行了一个embedding_table的查表操作，我们对其配置了优化器并行属性为`True`。例如在8卡机器上进行数据并行训练时，用户定义了一个查表操作，其中embedding_table参数shape为[80, 200]，那么每卡在参数初始化时得到的参数分片大小为[10, 200]。然后在正向计算开始时，MindSpore会执行一个AllGather通信操作，将每卡切片参数汇聚为一个完整shape[80, 200]的全量参数参与到网络的计算中。这个AllGather操作是用户不感知的操作。在计算反向梯度时，根据自动微分，AllGather的反向对应的是ReduceScatter操作，将梯度求和之后取对应设备的切片，shape即为[10, 200]的梯度分片到每张卡上。

```python
from mindspore import Parameter
from mindspore.common import initializer
from mindspore.nn import Cell
from mindspore import ops
from mindspore.nn.transformer import EmbeddingOpParallelConfig
default_embedding_parallel_config = EmbeddingOpParallelConfig()
class VocabEmbedding(Cell):
    def __init__(self, vocab_size, embedding_size, parallel_config=default_embedding_parallel_config,
                 param_init='normal'):
        super(VocabEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # 通过parallel_optimizer=True 配置embedding_table的进行优化器切分
        self.embedding_table = Parameter(initializer(param_init, [self.vocab_size, self.embedding_size]),
                                         name='embedding_table', parallel_optimizer=True)
        self.gather = ops.Gather()

    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table

# 配置优化器全局配置，设置其生效
context.set_auto_parallel_context(enable_parallel_optimizer=True)
```

## 配置通信融合

在[设置参数优化器并行](#设置参数优化器并行)一节中，我们阐述了如何配置每个参数的优化器并行属性。在全/半自动模式下，每个参数都会产生一个对应的AllGather操作和ReduceScatter操作。然而，随着参数量增多，对应的通信算子也会增多，通信操作产生的算子调度和启动都会产生更多的开销。因此，可以通过cell提供的`set_comm_fusion`方法，对每个cell内的参数对应的AllGather和ReduceScatter操作配置融合标记。在编译图的流程中，相同融合标记并且是相同的通信操作，会被融合成一个通信操作。从而减少通信操作的数量。例如下述的代码中，通过设置`src_embedding`和`tgt_embedding`的`set_comm_fusion`方法，将Encoder和Decoder的embedding层由于优化器产生的通信算子设置融合标记为0，将`Transformer`设置的融合标记为1。

```python
from mindspore.nn.transformer import Transformer, AttentionMask, CrossEntropyLoss
from mindspore import nn, ops
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

class TransformerModel(nn.Cell):
    def __init__(self, batch, src_len, tgt_len, hidden_size, vocab_size,
                 en_layer, de_layer, parallel_config, return_loss=False):
        super(TransformerModel, self).__init__()
        self.src_embedding = EmbeddingLayer(vocab_size=vocab_size, embedding_size=hidden_size,
                                            position_size=src_len,
                                            parallel_config=parallel_config.embedding_dp_mp_config)

        self.tgt_embedding = EmbeddingLayer(vocab_size=vocab_size, embedding_size=hidden_size,
                                            position_size=tgt_len, parallel_config=parallel_config.embedding_dp_mp_config)

        self.src_embedding.set_comm_fusion(0)
        self.tgt_embedding.set_comm_fusion(0)

        total_layers = en_layer + de_layer + 2
        layers_per_stage = total_layers // parallel_config.pipeline_stage
        self.return_loss = return_loss
        self.base1 = Transformer(encoder_layers=en_layer, decoder_layers=de_layer,
                                 batch_size=batch, src_seq_length=src_len,
                                 tgt_seq_length=tgt_len, hidden_size=hidden_size,
                                 num_heads=8, attention_dropout_rate=0.0,
                                 hidden_dropout_rate=0.0, lambda_func=pipeline_func,
                                 ffn_hidden_size=hidden_size, parallel_config=parallel_config)

        self.base1.set_comm_fusion(1)

        self.attention_mask = AttentionMask(seq_length=tgt_len)
        self.head = nn.Dense(in_channels=hidden_size, out_channels=vocab_size, has_bias=False)
        self.head.matmul.shard(((1, 1), (1, 1)))
        self.head.pipeline_stage = parallel_config.pipeline_stage - 1
        self.loss = CrossEntropyLoss(parallel_config=parallel_config.dp_mp_config)
        self.no_equal = ops.NotEqual().shard(((1, 1), ()))
```

### 通信优化

正常情况下，将整个网络因为优化器切分产生的通信算子融合成一个，是最有效减少通信算子个数的办法。然而这样会导致计算资源的浪费。例如，参数规模巨大的模型会导致对应的通信耗时也很大，并且网络的前向计算需要等待通信完成后才能获取完整shape的参数进行前向计算。因此可以将网络参数进行分组融合，在上一组参数进行的计算的同时，进行下组参数的融合。`Transformer`接口提供了`lambda_func`参数来自定义每层的分组融合标记，如下代码所示，将融合的算子个数定义为4个，然后去计算每层对应的融合标记。

```python
from mindspore.nn.transformer import Transformer, TransformerOpParallelConfig
layers_per_stage = 2
en_layer = 2
de_layer = 2
src_len = 10
batch = 4
tgt_len = 10
hidden_size = 20
def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
    pp_id = max(int(layer_id + offset) / layers_per_stage, 1)
    network.pipeline_stage = int(pp_id)
    print(f"pipeline stage id is {pp_id}", flush=True)

    # 将transformer的融合层数设置为4个
    gradient_aggregation_group = 4
    dis = max(int((layers + offset) / gradient_aggregation_group), 1)
    network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    # Used for enabling recomputation of the block
    if parallel_config.recompute:
        network.recompute()
parallel_config = TransformerOpParallelConfig()
base1 = Transformer(encoder_layers=en_layer, decoder_layers=de_layer,
                    batch_size=batch, src_seq_length=src_len,
                    tgt_seq_length=tgt_len, hidden_size=hidden_size,
                    num_heads=8, attention_dropout_rate=0.0,
                    hidden_dropout_rate=0.0, lambda_func=set_parallel_configure_for_layer,
                    ffn_hidden_size=hidden_size, parallel_config=parallel_config)
```

## 运行代码

上述流程的数据，代码和执行过程，可以参考：<https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_transformer.html>。
