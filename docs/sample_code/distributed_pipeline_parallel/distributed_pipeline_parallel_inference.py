# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Distributed Pipeline Parallel Inference Example"""

import numpy as np
import mindspore as ms
from mindspore import lazy_inline, nn, ops, Tensor, Parameter
from mindspore.communication import init
from mindspore.parallel.checkpoint_transform import sync_pipeline_shared_parameters


ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True,
                             pipeline_stages=4, pipeline_result_broadcast=True)
init()
ms.set_seed(1)

class VocabEmbedding(nn.Cell):
    """Vocab Embedding"""
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding_table = Parameter(Tensor(np.ones([vocab_size, embedding_size]), ms.float32),
                                         name='embedding_table')
        self.gather = ops.Gather()

    def construct(self, x):
        output = self.gather(self.embedding_table, x, 0)
        output = output.squeeze(1)
        return output, self.embedding_table.value()


class Head(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul(transpose_b=True)

    def construct(self, state, embed):
        return self.matmul(state, embed)


class Network(nn.Cell):
    """Network"""
    @lazy_inline
    def __init__(self):
        super().__init__()
        self.word_embedding = VocabEmbedding(vocab_size=32, embedding_size=32)
        self.layer1 = nn.Dense(32, 32)
        self.layer2 = nn.Dense(32, 32)
        self.head = Head()

    def construct(self, x):
        x, embed = self.word_embedding(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.head(x, embed)
        return x

# Define network and set pipeline stage
net = Network()
net.word_embedding.pipeline_stage = 0
net.layer1.pipeline_stage = 1
net.layer2.pipeline_stage = 2
net.head.pipeline_stage = 3


class PipelineCellInference(nn.Cell):
    """Pipeline Cell Inference wrapper"""
    def __init__(self, network, micro_batch_num):
        super().__init__()
        self.network = network
        self.micro_batch_num = micro_batch_num
        self.concat = ops.Concat()

    def construct(self, x):
        """Apply the pipeline inference"""
        ret = ()
        for i in range(self.micro_batch_num):
            micro_batch_size = x.shape[0] // self.micro_batch_num
            start = micro_batch_size * i
            end = micro_batch_size * (i + 1)

            micro_input = x[start:end]
            micro_output = self.network(micro_input)
            ret = ret + (micro_output,)

        ret = self.concat(ret)
        return ret

inference_network = PipelineCellInference(network=net, micro_batch_num=4)
inference_network.set_train(False)

# Compile and synchronize shared parameter.
input_ids = Tensor(np.random.randint(low=0, high=32, size=(8, 1)), ms.int32)
inference_network.compile(input_ids)
sync_pipeline_shared_parameters(inference_network)

# Execute the inference network
logits = inference_network(input_ids)
print(logits.asnumpy())
