"""Improving performance in PyNative mode: Method 2
This sample code is applicable to Ascend.
"""
import numpy as np
from mindspore import Tensor, set_context, PYNATIVE_MODE
import mindspore.ops as ops
from mindspore import ms_function

set_context(mode=PYNATIVE_MODE, device_target="Ascend")

tensor_add = ops.Add()

@ms_function
def tensor_add_fn(x, y):
    res = tensor_add(x, y)
    return res

input_x = Tensor(np.ones([4, 4]).astype(np.float32))
input_y = Tensor(np.ones([4, 4]).astype(np.float32))
z = tensor_add_fn(input_x, input_y)
print(z.asnumpy())
