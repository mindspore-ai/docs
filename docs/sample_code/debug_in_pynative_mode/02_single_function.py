"""single function
This sample code is applicable to Ascend.
"""
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

def tensor_add_func(x, y):
    z = ops.tensor_add(x, y)
    z = ops.tensor_add(z, x)
    return z

input_x = ms.Tensor(np.ones([3, 3], dtype=np.float32))
input_y = ms.Tensor(np.ones([3, 3], dtype=np.float32))
output = tensor_add_func(input_x, input_y)
print(output.asnumpy())
