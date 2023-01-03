# PengCheng·PanGu Model Network Multi-dimension Hydrid Parallel Analysis

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_en/parallel/pangu_alpha.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

In the PengCheng·PanGu model [1] published by MindSpore, we see that distributed training of very large Transformer networks can be achieved with the help of multi-dimensional automatic hybrid parallelism. This article will explain the sharding method of each component in the model in detail, starting from the network script.

> For the complete code, refer to [pangu_alpha](https://gitee.com/mindspore/models/tree/r2.0.0-alpha/official/nlp/Pangu_alpha)

In the training entry script train.py, the semi-automatic parallel mode `SEMI_AUTO_PARALLEL` is enabled by the `set_auto_parallel_context` interface, indicating that users can automatically complete the sharding with the help of the framework by configuring the sharding strategy for the operator. According to the features of operation volume and calculation methods in different network layers, choosing the appropriate sharding strategy is the focus of this paper. In addition, you can configure the optimizer parallelism and pipeline parallelism through the `enable_parallel_optimizer` and `pipeline_stages` parameters.

## Embedding Layer

In language model training, the input data are sentences composed of words, and we usually use the embedding algorithm to implement word vectorization, which maps the words and their location information into word vectors of size dimension `config.hidden_size`. The Embedding layer in the PanGu model consists of two parts, location encoding and word embedding, and implements basic data parallelism and model parallelism logic through `mindspore.nn.transformer.VocabEmbedding`.

The following code shows that the `Gather` operator takes two inputs and finds the corresponding vectors in the lookup table `embedding_table` according to the index `input_ids`. The lookup table is a parameter to be learned during training and statically occupies memory resources on the card. We can decide to use a data parallel strategy for the `Gather` operator to slice the index batch dimension or a model parallel strategy to row slice the lookup table depending on the size of the lookup table. When the word list range `config.vocab_size` is large, it is recommended to choose a model parallel strategy for `word_embedding`, and the framework will automatically introduce computation and communication operators to handle out-of-bounds lookup cases.

- Data parallel strategy `gather.shard(((1, 1), (parallel_config.data_parallel, 1)))`

- Model parallel strategy `gather.shard(((parallel_config.model_parallel, 1), (1, 1)))`

> The scripts and articles use config.data_parallel and config.model_parallel to refer to the data parallel slice dimension size and the model parallel slice dimension size.

```python
import mindspore as ms
from mindspore.common.initializer import initializer
import mindspore.ops as ops
from mindspore.nn import Cell
from mindspore.nn.transformer import EmbeddingOpParallelConfig
default_embedding_parallel_config = EmbeddingOpParallelConfig()
class VocabEmbedding(Cell):
    def __init__(self, vocab_size, hidden_size, parallel_config=default_embedding_parallel_config,
                 param_init='normal'):
        super(VocabEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_table = ms.Parameter(initializer(param_init, [self.vocab_size, self.hidden_size]),
                                            name='embedding_table', parallel_optimizer=False)
        if parallel_config.vocab_emb_dp:
            self.gather = ops.GatherV2().shard(((1, 1), (parallel_config.data_parallel, 1)))
        else:
            self.gather = ops.GatherV2().shard(((parallel_config.model_parallel, 1), (1, 1)))
    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table
```

Based on `mindspore.nn.transformer.VocabEmbedding`, we can implement the summation of word embedding vectors and location embedding vectors. We define the `Add` and `Dropout` operators and set the strategy corresponding to these two operators to be data parallelism.

```python
from mindspore.common.initializer import initializer
import mindspore.ops as ops
from mindspore import nn
from mindspore.nn.transformer import VocabEmbedding
class EmbeddingLayer(nn.Cell):
    """Embedding layer of the PanGUAlpha Model"""
    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.word_embedding = VocabEmbedding(vocab_size=config.vocab_size,
                                             hidden_size=config.hidden_size,
                                             param_init=initializer("normal", [config.vocab_size, config.hidden_size],
                                                                    dtype=config.param_init_type),
                                             parallel_config=config.parallel_config.embedding_dp_mp_config)
        self.position_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                 hidden_size=config.hidden_size,
                                                 param_init=initializer("normal",
                                                                        [config.seq_length, config.hidden_size],
                                                                        dtype=config.param_init_type),
                                                 parallel_config=config.parallel_config.embedding_dp_mp_config)
        self.add = ops.Add().shard(
            ((config.parallel_config.data_parallel, 1, 1), (config.parallel_config.data_parallel, 1, 1)))
        self.dropout = nn.Dropout(1 - config.dropout_rate)
        self.dropout.dropout.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.batch_size = config.batch_size

    def construct(self, input_ids, input_position, init_reset, batch_valid_length):
        word_embedding, word_table = self.word_embedding(input_ids)
        if self.use_past and not self.is_first_iteration:
            _, seq_length = ops.shape(input_ids)
            input_position = batch_valid_length.view(self.batch_size, seq_length)
        position_embedding, _ = self.position_embedding(input_position)
        embed = self.add(word_embedding, position_embedding)
        embed = self.dropout(embed)
        return embed, word_table
```

## Decoder Layer

The key difficulty in training large-scale Transformer networks is how to solve the computational and memory bottlenecks caused by the increasing number of layers, and it is especially important to choose a reasonable slicing. The main network of the PengCheng-PanGu model consists of multiple Decoders with the same structure but do not share weights, and the Decoder is composed of two parts, Self-Attention and FeedForward. The principle of slicing is to minimize the communication, and their slicing can be referred to the following figure [1]:

![image](https://gitee.com/mindspore/docs/raw/r2.0.0-alpha/tutorials/experts/source_zh_cn/parallel/images/pangu_strategy.png)

### Self-Attention

Self-Attention can be implemented directly via `mindspore.nn.transformer.MultiHeadAttention`. In the process of computing Attention, the input vector needs to be projected to the Query, Key, and Value vectors, and then the output of attention needs to be passed through the Dense layer again after the calculation of attention is completed. The following describes the strategy configuration of these three sections respectively.

- Three Dense Matrix Multiplication

  Here project the input tensor with shape `[batch*sequence_length, hidden_size]` into three vectors as the Query, Key, and Value vectors for the Attention calculation.

  Hybrid parallel slicing of the input batch dimension and the output_channel dimension of the weight:

  `matmul.shard(((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)))`.

  Output matrix rows and sliced columns, plus the sliced bias term.

  `bias_add.shard(((parallel_config.data_parallel, parallel_config.model_parallel), (parallel_config.model_parallel,)))`.

  ```python
  self.dense1 = nn.Dense(hidden_size,
                         hidden_size).to_float(compute_dtype)
  self.dense1.matmul.shard(((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)))
  self.dense1.bias_add.shard(((parallel_config.data_parallel, parallel_config.model_parallel), (parallel_config.model_parallel,)))
  ```

- `Softmax` and `BatchMatMul`

  The matrix multiplication of Query and Key vectors is implemented by `BatchMatMul` in the process of computing Attention. Here the input shape of `softmax` is `[batch, sequence_length, num_heads, size_per_head]`. Because each `head` is independent from each other in computing the attention score, the `softmax` operator can be sliced in the `batch` dimension and the `heads` dimension.

  ```python
  self.softmax = nn.Softmax()
  self.softmax.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1),))
  self.batch_matmul = ops.BatchMatMul().shard(
                      ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                      (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
  ```

- Projection Layer

  Projection projects the output of attention once. The relevant dimension in the `MatMul` operator is sliced.

  ```python
  self.projection = nn.Dense(hidden_size,
                             hidden_size).to_float(compute_dtype)
  self.projection.matmul.shard(((parallel_config.data_parallel, 1), (1, parallel_config.model_parallel)))
  ```

### FeedForward

FeedForward can be implemented by calling `mindspore.nn.transformer.FeedForward` directly. The FeedForward network layer consists of two matrix multiplications. The first matrix multiplication slices in the same way as attention, outputting matrix rows and sliced columns, i.e., in the `batch` dimension and the `output dimension`. In order to avoid introducing redistribution communication between operators, the second matrix multiplication slices the input_channel dimension of the weights, i.e. `matmul.shard(((parallel_config.data_parallel, parallel_config.model_parallel), ( parallel_config.model_parallel, 1)))`. The framework automatically inserts the `AllReduce` operator when the relevant dimension is sliced, and accumulates the slicing results in the model parallel dimension. The output matrix is sliced in the `batch` dimension only, plus the bias term `add.shard(((parallel_config.data_parallel, 1), (1,)))`.

```python
from mindspore.common.initializer import initializer
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn
from mindspore.nn import get_activation
from mindspore.nn.transformer import OpParallelConfig

default_dpmp_config = OpParallelConfig()
class Linear(nn.Cell):
    """
    The dense connected layer. Once the parallel mode is enabled, the input shape should be
    a 3-D tensor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 transpose_b=True,
                 expert_num=1,
                 param_init_type=ms.float32,
                 compute_dtype=ms.float16):
        super(Linear, self).__init__()
        if transpose_b:
            weight_shape = [out_channels, in_channels]
        else:
            weight_shape = [in_channels, out_channels]
        self.expert_num = expert_num
        if self.expert_num > 1:
            self.expert_flag = True
            self.weight = ms.Parameter(initializer(weight_init, [self.expert_num] + weight_shape, param_init_type),
                                       name="weight")
            self.matmul = ops.BatchMatMul(transpose_b=transpose_b)
        else:
            self.expert_flag = False
            self.weight = ms.Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
            self.matmul = ops.MatMul(transpose_b=transpose_b)
        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            if isinstance(bias_init, ms.Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("Bias init shape error.")
            self.bias = ms.Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias_add = ops.Add()
        self.act_name = activation
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        self.activation_flag = self.activation is not None
        self.dtype = compute_dtype
        self.cast = ops.Cast()

    def construct(self, x):
        out_shape = ops.Shape()(x)[:-1] + (self.out_channels,)
        x = ops.Reshape()(x, (-1, self.in_channels))
        if self.expert_flag is True:
            x = ops.Reshape()(x, (self.expert_num, -1, self.in_channels))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        output = ops.Reshape()(x, out_shape)
        if self.activation_flag:
            output = self.activation(output)
        return output

    def shard(self, strategy_matmul, strategy_bias=None, strategy_activation=None):
        """
         Set the shard for the linear. the strategy size should be equal to the inputs.
        """
        self.matmul.shard(strategy_matmul)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        if self.activation_flag:
            getattr(self.activation, self.act_name).shard(strategy_activation)
        return self

class FeedForward(nn.Cell):
    """
    The multilayer perceptron with two linear layers with dropout applied at final output. The first linear
    will project the input dimension from hidden_size to ffn_hidden_size, the second linear will project the
    dimension from ffn_hidden_size to hidden_size. The first linear is sharded on the relative dimension,
    the second linear is sharded on the output dimension.
    """
    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act='gelu',
                 expert_num=1,
                 param_init_type=ms.float32,
                 parallel_config=default_dpmp_config):
        super(FeedForward, self).__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        input_size = hidden_size
        output_size = ffn_hidden_size
        # Here, 'ep' stands for expert parallel number, which is equal to data parallel number.
        ep = dp
        # Project to ffn_hidden_size
        self.mapping = Linear(in_channels=input_size,
                               out_channels=output_size,
                               activation=hidden_act,
                               transpose_b=False,
                               expert_num=expert_num,
                               param_init_type=param_init_type)
        self.mapping.shard(strategy_matmul=((dp, 1), (1, mp)),
                           strategy_bias=((dp, mp), (mp,)),
                           strategy_activation=((dp, mp),))
        # Project back to hidden_size
        self.projection = Linear(in_channels=output_size,
                                  out_channels=input_size,
                                  transpose_b=False,
                                  expert_num=expert_num,
                                  param_init_type=param_init_type)
        self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)),
                              strategy_bias=((dp, 1), (1,)))
        self.projection.bias.parallel_optimizer = False
        self.dropout = nn.Dropout(1 - dropout_rate)
        self.dropout.dropout.shard(((dp, 1),))
        self.cast = ops.Cast()

    def construct(self, x):
        x = self.cast(x, ms.float16)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        hidden = self.mapping(x)
        output = self.projection(hidden)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        output = self.dropout(output)
        return output
```

## Residual Layer

A detail of the Transformer structure that should be noted is that each sublayer is connected with residuals and follows the layernorm operation. Although the layernorm also contains weights, it is only a one-dimensional vector of size `hidden_size`, which accounts for a very small proportion of the network weights, so data parallel slicing is directly used here.

```python
from mindspore import nn

layernorm1 = nn.LayerNorm((hidden_size,))
layernorm1.shard(((parallel_config.data_parallel, 1),))
```

## Prediction Layer

A fully-connected layer is needed to map the output features from `config.hidden_size` back to the `config.vocab_size` dimension to get logits before calculating the loss. Here the fully-connected layer and the `word_embedding` operation share weights, so the slicing of the fully connected layer weights is required to be consistent with that of the embedding layer.

```python
import mindspore.ops as ops
from mindspore import nn
class PanguAlpha_Head(nn.Cell):
    """
    Head for PanguAlpha to get the logits of each token in the vocab
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self, config):
        super(PanguAlpha_Head, self).__init__()
        if config.word_emb_dp:
            self.matmul = ops.MatMul(transpose_b=True).shard(((parallel_config.dp, 1), (1, 1)))
        else:
            self.matmul = ops.MatMul(transpose_b=True).shard(((parallel_config.dp, 1), (parallel_config.model_parallel, 1)))
        self.hidden_size = config.hidden_size
        self.log_softmax = ops.LogSoftmax(axis=-1)
        self.dtype = config.compute_dtype
        self.cast = ops.Cast()

    def construct(self, state, embedding_table):
        state = ops.Reshape()(state, (-1, self.hidden_size))
        # output logits over vocabulary [bs*seq_length, vocab_size]
        logits = self.matmul(state, self.cast(embedding_table, self.dtype))
        return logits
```

In this article, we learn how to quickly implement distributed training of Transformer-like networks on the basis of a stand-alone script by configuring an operator sharding strategy. When specific to the network structure, embedding layer, decoder layer, residual layer and linear layer all have their own slicing features, and users can improve the distributed training and tuning efficiency by mastering the operator strategy configuration method.

## References

[1] Zeng W, Ren X, Su T, et al. PanGu-$\\alpha$: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation. 2021.
