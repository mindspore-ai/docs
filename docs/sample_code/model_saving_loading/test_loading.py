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
"""Loading Programming Guide"""

import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.communication import init, get_rank, get_group_size
from mindspore.common.initializer import initializer

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(strategy_ckpt_config={"load_file": "./src_strategy.ckpt"})
init()
ms.set_seed(1)

class Dense(nn.Cell):
    """Dense layer"""
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
    """Network"""
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

predict_data = ms.Tensor(np.random.randn(1, 28, 28).astype(np.float32))
model = ms.Model(net)
predict_layout = model.infer_predict_layout(predict_data)
ckpt_file_list = ["./src_checkpoints/rank_{}/checkpoint-10_1875.ckpt".format(i) for i in range(0, get_group_size())]
ms.load_distributed_checkpoint(net, ckpt_file_list, predict_layout)
predict_result = model.predict(predict_data)
print(predict_result)

ms.export(net, predict_data, file_name='./mindir/net_rank_' + str(get_rank()), file_format='MINDIR')
