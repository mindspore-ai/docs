"""Improving performance in PyNative mode: Method 1
This sample code is applicable to Ascend.
"""
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, PYNATIVE_MODE, set_context
import mindspore.ops as ops
from mindspore import ms_function

set_context(mode=PYNATIVE_MODE, device_target="Ascend")

class TensorAddNet(nn.Cell):
    def __init__(self):
        super(TensorAddNet, self).__init__()
        self.add = ops.Add()

    @ms_function
    def construct(self, x, y):
        res = self.add(x, y)
        return res

input_x = Tensor(np.ones([4, 4]).astype(np.float32))
input_y = Tensor(np.ones([4, 4]).astype(np.float32))
net = TensorAddNet()

z = net(input_x, input_y) # Staging mode
tensor_add = ops.Add()
result = tensor_add(input_x, z) # PyNative mode
print(result.asnumpy())
