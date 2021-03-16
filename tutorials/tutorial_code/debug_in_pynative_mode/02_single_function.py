"""single function
This sample code is applicable to Ascend.
"""
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

def tensor_add_func(x, y):
    z = ops.tensor_add(x, y)
    z = ops.tensor_add(z, x)
    return z

x = Tensor(np.ones([3, 3], dtype=np.float32))
y = Tensor(np.ones([3, 3], dtype=np.float32))
output = tensor_add_func(x, y)
print(output.asnumpy())