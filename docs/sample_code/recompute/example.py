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

"""Recompute Example"""

import numpy as np
from mindspore.nn import Cell
from mindspore.common import Tensor, Parameter
from mindspore import context, ops, nn

context.set_context(mode=context.GRAPH_MODE)


class Block(Cell):
    """Block"""
    def __init__(self):
        super(Block, self).__init__()
        self.transpose1 = ops.Transpose()
        self.transpose2 = ops.Transpose()
        self.transpose3 = ops.Transpose()
        self.transpose4 = ops.Transpose()
        self.real_div1 = ops.RealDiv()
        self.real_div2 = ops.RealDiv()
        self.batch_matmul1 = ops.BatchMatMul()
        self.batch_matmul2 = ops.BatchMatMul()
        self.add = ops.Add()
        self.softmax = ops.Softmax(-1)
        self.dropout = ops.Dropout(0.9)
        self.expand_dims = ops.ExpandDims()
        self.sub = ops.Sub()
        self.mul = ops.Mul()
        self.y = Parameter(Tensor(np.ones((8, 128, 128)).astype(np.float32)))

    def construct(self, x):
        """Network definition"""
        transpose1 = self.transpose1(x, (0, 2, 1, 3))
        real_div1 = self.real_div1(transpose1, Tensor(2.37891))
        transpose2 = self.transpose2(x, (0, 2, 3, 1))
        real_div2 = self.real_div2(transpose2, Tensor(2.37891))
        batch_matmul1 = self.batch_matmul1(real_div1, real_div2)
        expand_dims = self.expand_dims(self.y, 1)
        sub = self.sub(Tensor([1.0]), expand_dims)
        mul = self.mul(sub, Tensor([-0.0001]))
        add = self.add(mul, batch_matmul1)
        soft_max = self.softmax(add)
        dropout = self.dropout(soft_max)
        transpose3 = self.transpose3(x, (0, 2, 1, 3))
        batch_matmul2 = self.batch_matmul2(dropout[0], transpose3)
        transpose4 = self.transpose4(batch_matmul2, (0, 2, 1, 3))
        return transpose4


class Net(Cell):
    """Network"""
    def __init__(self):
        super(Net, self).__init__()
        self.blocks = nn.CellList()
        for _ in range(10):
            b = Block()
            self.blocks.append(b)

    def construct(self, x):
        out = x
        for i in range(10):
            out = self.blocks[i](out)
        return out


class Grad(Cell):
    def __init__(self, net):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation()
        self.net = net

    def construct(self, x):
        grad_net = self.grad(self.net)
        return grad_net(x)


input_x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
network = Net()
grad_network = Grad(network)
output = grad_network(input_x)
print(output)
