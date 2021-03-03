import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

class TensorAddNet(nn.Cell):
    def __init__(self):
        super(TensorAddNet, self).__init__()
        self.add = ops.Add()

    @ms_function
    def construct(self, x, y):
        res = self.add(x, y)
        return res

x = Tensor(np.ones([4, 4]).astype(np.float32))
y = Tensor(np.ones([4, 4]).astype(np.float32))
net = TensorAddNet()

z = net(x, y) # Staging mode
tensor_add = ops.Add()
res = tensor_add(x, z) # PyNative mode
print(res.asnumpy())