# Copyright 2020 Huawei Technologies Co., Ltd
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
# ==========================================================

"""multiple in multiple out tutorial
This sample code is applicable to GPU and Ascend.
"""
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, set_context, GRAPH_MODE

set_context(mode=GRAPH_MODE, device_target="GPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        x_square = self.mul(x, x)
        x_square_y = self.mul(x_square, y)
        return x_square_y

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=False)
        self.network = network
    def construct(self, x, y):
        gout = self.grad(self.network)(x, y) # return dx dy
        return gout

class GradSec(nn.Cell):
    """construct secend grad"""
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.network = network
        self.sens1 = Tensor(np.array([1]).astype('float32'))
        self.sens2 = Tensor(np.array([0]).astype('float32'))
    def construct(self, x, y):
        dxdx, dxdy = self.grad(self.network)(x, y, (self.sens1, self.sens2))
        dydx, dydy = self.grad(self.network)(x, y, (self.sens2, self.sens1))
        return dxdx, dxdy, dydx, dydy

net = Net()
firstgrad = Grad(net) # first order
secondgrad = GradSec(firstgrad) # second order
x_train = Tensor(np.array([4], dtype=np.float32))
y_train = Tensor(np.array([5], dtype=np.float32))
input_dxdx, input_dxdy, input_dydx, input_dydy = secondgrad(x_train, y_train)
print(input_dxdx, input_dxdy, input_dydx, input_dydy)
