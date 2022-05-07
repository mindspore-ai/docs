"""Improving performance in PyNative mode: Method 3
This sample code is applicable to Ascend.
"""
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, PYNATIVE_MODE, set_context
from mindspore import ms_function

set_context(mode=PYNATIVE_MODE, device_target="Ascend")

conv_obj = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)
conv_obj.init_parameters_data()
@ms_function
def conv_fn(x):
    res = conv_obj(x)
    return res

input_data = np.random.randn(2, 3, 6, 6).astype(np.float32)
z = conv_fn(Tensor(input_data))
print(z.asnumpy())
