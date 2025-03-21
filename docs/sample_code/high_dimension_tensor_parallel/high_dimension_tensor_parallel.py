# Copyright 2025 Huawei Technologies Co., Ltd
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

"""High Dimension Tensor Parallel Example"""

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.parallel import Layout
from mindspore.common.initializer import initializer
from mindspore.communication import init
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.nn.utils import no_init_parameters

ms.set_context(mode=ms.GRAPH_MODE)
init()
ms.set_seed(1)


class Network(nn.Cell):
    """ construct network for 2D Tensor Parallel """
    def __init__(self):
        super().__init__()
        self.fc1_weight = ms.Parameter(initializer("normal", [256, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 256], ms.float32))
        self.matmul1 = ops.MatMul()
        self.relu = ops.ReLU()
        self.matmul2 = ops.MatMul()

    def construct(self, x):
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu(x)
        return self.matmul2(x, self.fc2_weight)

with no_init_parameters():
    net = Network()

in_layout = Layout((2, 4), ("x", "y"))
net.matmul1.add_prim_attr("enable_nd_tp", True)
net.matmul1.shard(in_strategy=(in_layout("None", ("x", "y")), in_layout("x", "y")))
net.relu.shard(in_strategy=(in_layout("None", ("y", "x")),))
net.matmul2.add_prim_attr("enable_nd_tp", True)
net.matmul2.shard(in_strategy=(in_layout("None", ("y", "x")), in_layout("y", "x")))

input_data = Tensor(np.ones((1024, 256)), dtype=ms.float32)
net = AutoParallel(net)
output = net(input_data)
print("The output is:", output)
