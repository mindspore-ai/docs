# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Begin large language model parallel inference development."""

import math

import numpy as np
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.communication import create_group, get_group_size, get_rank, init
import mindspore.communication as comm


class ConfigHelper:
    """ Helper class to store model configuration parameters """

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_layers,
                 batch_size,
                 seq_length, dtype,
                 num_heads,
                 has_bias=False
                 ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.dtype = dtype
        self.num_heads = num_heads
        self.has_bias = has_bias


class CommunicationHelper:
    """ Helper class to manage communication across parallel groups """

    def __init__(self, group_name, size):
        self.group_name = group_name
        self.size = size
        self.rank_list = [i for i in range(size)]

    def create_tensor_model_parallel_group(self):
        create_group(group=self.group_name, rank_ids=self.rank_list)

    def get_tensor_model_paralell_group_size(self):
        return get_group_size(group=self.group_name)

    def get_tensor_model_paralell_group_rank(self):
        return get_rank(group=self.group_name)

    def get_tensor_model_parallel_group(self):
        return self.group_name


class ColumnParallelLinear(nn.Cell):
    """ Linear layer for column parallelism in distributed models """

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init=None,
                 has_bias=True,
                 dtype=mstype.float32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bias = has_bias
        self.tensor_parallel_group_size = COMMUN_HELPER.get_tensor_model_paralell_group_size()
        self.out_channels_per_partition = out_channels // self.tensor_parallel_group_size
        self.dtype = dtype
        weight_shape = (self.out_channels_per_partition, self.in_channels)
        self.weight = Parameter(initializer(weight_init, weight_shape, self.dtype), name="weight")
        if self.has_bias:
            self.bias = Parameter(initializer(bias_init, (self.out_channels_per_partition), self.dtype), name="bias")
            self.bias_add = ops.Add()
        self.matmul = ops.BatchMatMul(transpose_b=True)
        self.cast = ops.Cast()

    def construct(self, x):
        origin_dtype = x.dtype
        x = self.cast(x, self.dtype)
        output = self.matmul(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.cast(self.bias, self.dtype))
        output = self.cast(output, origin_dtype)
        return output

    def sharded_state_dict(self):
        w_shard = (self.tensor_parallel_group_size, 1)
        state_dict = {}
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (self.tensor_parallel_group_size,)}
        return state_dict



class GatherLastDim(nn.Cell):
    """ Gather the last dimension across all parallel ranks """

    def __init__(self):
        super().__init__()
        self.world_size = COMMUN_HELPER.get_tensor_model_paralell_group_size()
        self.split = ops.Split(axis=0, output_num=self.world_size)

    def construct(self, input_):
        output = comm.comm_func.all_gather_into_tensor(input_, group=COMMUN_HELPER.get_tensor_model_parallel_group())[0]
        tensor_list = self.split(output)
        output = ops.cat(tensor_list, axis=-1)
        return output


class RowParallelLinear(nn.Cell):
    """ Linear layer for row parallelism in distributed models """

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init=None,
                 has_bias=True,
                 dtype=mstype.float32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bias = has_bias
        self.tensor_parallel_group_size = COMMUN_HELPER.get_tensor_model_paralell_group_size()
        self.in_channels_per_partition = in_channels // self.tensor_parallel_group_size
        self.dtype = dtype
        weight_shape = (self.out_channels, self.in_channels_per_partition)
        self.weight = Parameter(initializer(weight_init, weight_shape, self.dtype), name="weight")
        if self.has_bias:
            self.bias = Parameter(initializer(bias_init, (self.in_channels_per_partition), self.dtype), name="bias")
            self.bias_add = ops.Add()
        self.bmm = ops.BatchMatMul(transpose_b=True)
        self.cast = ops.Cast()

    def construct(self, x):
        origin_dtype = x.dtype
        x = self.cast(x, self.dtype)
        output_parallel = self.bmm(x, self.weight)
        if self.has_bias:
            output_parallel = self.bias_add(output_parallel, self.cast(self.bias, self.dtype))
        output = comm.comm_func.all_reduce(output_parallel, group=COMMUN_HELPER.get_tensor_model_parallel_group())[0]
        output = self.cast(output, origin_dtype)
        return output

    def sharded_state_dict(self):
        w_shard = (1, self.tensor_parallel_group_size)
        state_dict = {}
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (1,)}
        return state_dict


class VocabParallelEmbedding(nn.Cell):
    """ Embedding layer with parallelism across vocabulary partitions """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 init_method="normal",
                 init_type=mstype.float32,
                 ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tensor_parallel_group_size = COMMUN_HELPER.get_tensor_model_paralell_group_size()
        per_partition_vocab_size = self.num_embeddings // self.tensor_parallel_group_size
        self.vocab_start_index = COMMUN_HELPER.get_tensor_model_paralell_group_rank() * per_partition_vocab_size
        self.vocab_end_index = self.vocab_start_index + per_partition_vocab_size
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )
        self.embedding_weight = Parameter(
            initializer(
                init=init_method,
                shape=(self.num_embeddings_per_partition, self.embedding_dim),
                dtype=init_type,
            ),
            name="embedding_weight",
        )
        self.max_index_per_partition = Tensor(self.num_embeddings_per_partition - 1, dtype=mstype.int32)
        self.expand_dims = ops.ExpandDims()
        self.gather = ops.Gather()
        self.sub = ops.Sub()
        self.relu = ops.ReLU()
        self.minimum = ops.Minimum()
        self.eq = ops.Equal()
        self.mul = ops.Mul()

    def construct(self, x):
        displaced_x = self.sub(x, self.vocab_start_index)
        down_truncated_x = self.relu(displaced_x)
        truncated_x = self.minimum(down_truncated_x, self.max_index_per_partition)
        input_mask = self.eq(displaced_x, truncated_x)
        input_mask = self.expand_dims(input_mask, -1)
        output_parallel = self.gather(self.embedding_weight, truncated_x, 0)
        output_parallel = self.mul(output_parallel, input_mask)
        output = comm.comm_func.all_reduce(output_parallel, group=COMMUN_HELPER.get_tensor_model_parallel_group())[0]
        return output

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (self.tensor_parallel_group_size, 1)
        state_dict = {}
        state_dict[self.embedding_weight.name] = {'shape': self.embedding_weight.shape,
                                                  'shard': w_shard}

        return state_dict


class ParallelAttention(nn.Cell):
    """ Multi-head attention layer with tensor parallelism """

    def __init__(self, config):
        super().__init__()
        self.tensor_parallel_group_size = COMMUN_HELPER.get_tensor_model_paralell_group_size()
        self.num_heads_per_partition = config.num_heads // self.tensor_parallel_group_size
        self.head_dim = config.hidden_size // config.num_heads
        self.norm_factor = math.sqrt(self.head_dim)
        self.q = ColumnParallelLinear(in_channels=config.hidden_size,
                                      out_channels=config.hidden_size,
                                      weight_init='normal',
                                      has_bias=config.has_bias)
        self.k = ColumnParallelLinear(in_channels=config.hidden_size,
                                      out_channels=config.hidden_size,
                                      weight_init='normal',
                                      dtype=config.dtype,
                                      has_bias=config.has_bias)
        self.v = ColumnParallelLinear(in_channels=config.hidden_size,
                                      out_channels=config.hidden_size,
                                      weight_init='normal',
                                      dtype=config.dtype,
                                      has_bias=config.has_bias)
        self.flash_attention = ops.operations.nn_ops.FlashAttentionScore(head_num=self.num_heads_per_partition,
                                                                         scale_value=1.0/self.norm_factor,
                                                                         next_tokens=0)
        self.out = RowParallelLinear(in_channels=config.hidden_size,
                                     out_channels=config.hidden_size,
                                     weight_init='normal',
                                     dtype=config.dtype,
                                     has_bias=config.has_bias)

    def construct(self, x, mask):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        _, _, _, context_layer = self.flash_attention(query, key, value, attn_mask=mask)
        output = self.out(context_layer)
        return output


class ParallelMLP(nn.Cell):
    """ MLP (feedforward) layer with parallelism """

    def __init__(self, config):
        super().__init__()
        self.w1 = ColumnParallelLinear(in_channels=config.hidden_size,
                                       out_channels=config.ffn_hidden_size,
                                       weight_init='normal',
                                       dtype=config.dtype,
                                       has_bias=config.has_bias)
        self.w2 = RowParallelLinear(in_channels=config.ffn_hidden_size,
                                    out_channels=config.hidden_size,
                                    weight_init='normal',
                                    dtype=config.dtype,
                                    has_bias=config.has_bias)
        self.act_func = nn.SiLU()
        self.mul = ops.Mul()

    def construct(self, x):
        x = self.w1(x)
        x = self.act_func(x)
        output = self.w2(x)
        return output


class RMSNorm(nn.Cell):
    """ Root mean square (RMS) normalization layer """

    def __init__(self, dim, eps=1e-6, dtype=mstype.float32):
        super().__init__()
        self.dtype = dtype
        self.weight = Parameter(initializer('ones', (dim,), dtype=self.dtype))
        self.norm = ops.RmsNorm(eps)
        self.cast = ops.Cast()
        self.rcast = ops.Cast()

    def construct(self, x):
        original_type = x.dtype
        output = self.norm(self.cast(x, self.dtype), self.weight)[0]
        return self.rcast(output, original_type)


class ParallelTransformerLayer(nn.Cell):
    """ Single transformer layer with parallel attention and MLP """

    def __init__(self, config):
        super().__init__()
        self.attention = ParallelAttention(config=config)
        self.feed_forward = ParallelMLP(config=config)
        self.attention_norm = RMSNorm(dim=config.hidden_size, dtype=config.dtype)
        self.ffn_norm = RMSNorm(dim=config.hidden_size, dtype=config.dtype)
        self.add = ops.Add()

    def construct(self, x, mask):
        norm_output = self.attention_norm(x)
        attention_output_ = self.attention(norm_output, mask)
        norm_input = self.add(x, attention_output_)
        norm_output = self.ffn_norm(norm_input)
        mlp_output_ = self.feed_forward(norm_output)
        output = self.add(norm_input, mlp_output_)
        return output


class ParallelTransformer(nn.Cell):
    """ Full transformer model with multiple parallel transformer layers """

    def __init__(self, config):
        super().__init__()
        self.embedding = VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                embedding_dim=config.hidden_size,
                                                init_method='normal',
                                                init_type=config.dtype)
        self.layers = nn.CellList()
        self.num_layers = config.num_layers
        for _ in range(config.num_layers):
            layer = ParallelTransformerLayer(config=model_config)
            self.layers.append(layer)
        self.norm_out = RMSNorm(dim=config.hidden_size, dtype=config.dtype)

    def construct(self, x, mask):
        hidden_state = self.embedding(x)
        for i in range(self.num_layers):
            hidden_state = self.layers[i](hidden_state, mask)
        hidden_state = self.norm_out(hidden_state)
        return hidden_state


if __name__ == "__main__":
    COMMUN_HELPER = CommunicationHelper(group_name='tp', size=2)
    model_config = ConfigHelper(batch_size=64,
                                vocab_size=32000,
                                num_layers=4,
                                seq_length=2048,
                                hidden_size=1024,
                                ffn_hidden_size=4096,
                                dtype=mstype.float16,
                                num_heads=8,
                                has_bias=False)
    init()
    COMMUN_HELPER.create_tensor_model_parallel_group()

    column_parallel_linear = ColumnParallelLinear(in_channels=model_config.hidden_size,
                                                  out_channels=model_config.hidden_size,
                                                  weight_init='normal',
                                                  dtype=model_config.dtype,
                                                  has_bias=False)
    input_x = Tensor(np.random.randn(model_config.batch_size, model_config.seq_length,
                                     model_config.hidden_size).astype(np.float32))
    out_parallel = column_parallel_linear(input_x)
    print(out_parallel.shape)

    gather_last_dim = GatherLastDim()
    out = gather_last_dim(out_parallel)
    print(out.shape)

    row_parallel_linear = RowParallelLinear(in_channels=model_config.hidden_size,
                                            out_channels=model_config.hidden_size,
                                            weight_init='normal',
                                            dtype=model_config.dtype,
                                            has_bias=False)
    out = row_parallel_linear(out_parallel)
    print(out.shape)

    vocab_parallel_embedding = VocabParallelEmbedding(num_embeddings=model_config.vocab_size,
                                                      embedding_dim=model_config.hidden_size,
                                                      init_method='normal',
                                                      init_type=model_config.dtype)
    input_ids = np.random.randint(0, model_config.vocab_size, size=(
        model_config.batch_size, model_config.seq_length), dtype=np.int32)
    input_ids = Tensor(input_ids)

    embedding_output = vocab_parallel_embedding(input_ids)
    print(embedding_output.shape)

    attn_mask = np.ones(shape=(model_config.seq_length, model_config.seq_length), dtype=np.uint8)
    attn_mask = np.triu(attn_mask, 1)
    attn_mask = Tensor(attn_mask)
    attention = ParallelAttention(config=model_config)
    attention_output = attention(embedding_output, attn_mask)
    print(attention_output.shape)

    mlp = ParallelMLP(config=model_config)
    mlp_output = mlp(attention_output)
    print(mlp_output.shape)

    parallel_transformer = ParallelTransformer(config=model_config)
    parallel_transformer_output = parallel_transformer(input_ids, attn_mask)
    print(parallel_transformer_output.shape)

    transformerlayer = ParallelTransformerLayer(config=model_config)

    transformerlayer_output = transformerlayer(embedding_output, attn_mask)
    print(transformerlayer_output.shape)
