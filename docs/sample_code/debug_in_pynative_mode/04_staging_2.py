"""Improving performance in PyNative mode: Method 2
This sample code is applicable to Ascend.
"""
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import ms_function

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

tensor_add = ops.Add()

@ms_function
def tensor_add_fn(x, y):
    res = tensor_add(x, y)
    return res

input_x = ms.Tensor(np.ones([4, 4]).astype(np.float32))
input_y = ms.Tensor(np.ones([4, 4]).astype(np.float32))
z = tensor_add_fn(input_x, input_y)
print(z.asnumpy())
