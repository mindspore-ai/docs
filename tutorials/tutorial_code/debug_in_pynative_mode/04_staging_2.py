"""Improving performance in PyNative mode: Method 2
This sample code is applicable to Ascend.
"""
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

tensor_add = ops.Add()

@ms_function
def tensor_add_fn(x, y):
    res = tensor_add(x, y)
    return res

x = Tensor(np.ones([4, 4]).astype(np.float32))
y = Tensor(np.ones([4, 4]).astype(np.float32))
z = tensor_add_fn(x, y)
print(z.asnumpy())
