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

"""Single in single out tutorial
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
        self.sin = ops.Sin()
    def construct(self, x):
        out = self.sin(x)
        return out

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network
    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout
class GradSec(nn.Cell):
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network
    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout

net = Net()
firstgrad = Grad(net) # first order
secondgrad = GradSec(firstgrad) # second order
x_train = Tensor(np.array([1.0], dtype=np.float32))
output = secondgrad(x_train)
print(output)
