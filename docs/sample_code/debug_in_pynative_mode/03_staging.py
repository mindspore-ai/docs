"""Improving performance in PyNative mode: Method 1
This sample code is applicable to Ascend.
"""
import numpy as np
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
from mindspore import ms_function

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

class TensorAddNet(nn.Cell):
    def __init__(self):
        super(TensorAddNet, self).__init__()
        self.add = ops.Add()

    @ms_function
    def construct(self, x, y):
        res = self.add(x, y)
        return res

input_x = ms.Tensor(np.ones([4, 4]).astype(np.float32))
input_y = ms.Tensor(np.ones([4, 4]).astype(np.float32))
net = TensorAddNet()

z = net(input_x, input_y) # Staging mode
tensor_add = ops.Add()
result = tensor_add(input_x, z) # PyNative mode
print(result.asnumpy())
