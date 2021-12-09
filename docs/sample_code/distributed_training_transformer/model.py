# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Model for training transformers
"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.parallel.nn import Transformer, VocabEmbedding, AttentionMask, CrossEntropyLoss
from mindspore.nn import Dense as Linear

class EmbeddingLayer(nn.Cell):
    """Embedding layer including position embedding and word embedding"""
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
        total_layers = en_layer + de_layer
        layers_per_stage = total_layers // parallel_config.pipeline_stage
        self.src_embedding.pipeline_stage = 0
        self.tgt_embedding.pipeline_stage = 0
        self.return_loss = return_loss

        def pipeline_func(network, layer_id, offset, parallel_config, layers):
            pp_id = min(int((layer_id + offset) / layers_per_stage), 1)
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

    def construct(self, encoder_input, encoder_position, encoder_mask,
                  decoder_input, decoder_position, memory_mask_input, y):
        """Construct function"""
        encoder_embed, _ = self.src_embedding(encoder_input, encoder_position)
        decoder_embed, _ = self.tgt_embedding(decoder_input, decoder_position)
        input_mask_value = self.no_equal(decoder_input, 1)
        input_mask_value = ops.Cast()(input_mask_value, mstype.float32)
        decoder_mask = self.attention_mask(input_mask_value)
        decoder_output, _, _ = self.base1(encoder_embed,
                                          encoder_mask,
                                          decoder_embed,
                                          decoder_mask,
                                          memory_mask_input)
        predict = self.head(decoder_output)
        predict = ops.Reshape()(predict, (-1, ops.shape(predict)[-1]))
        if self.return_loss:
            input_mask_value = ops.Reshape()(input_mask_value, (-1,))
            y = ops.Reshape()(y, (-1,))
            return self.loss(predict, y, input_mask_value)

        return predict
