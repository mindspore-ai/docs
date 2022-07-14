"""Improving performance in PyNative mode: Method 3
This sample code is applicable to Ascend.
"""
import numpy as np
from mindspore import ms_function
import mindspore.nn as nn
import mindspore as ms


ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

conv_obj = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)
conv_obj.init_parameters_data()
@ms_function
def conv_fn(x):
    res = conv_obj(x)
    return res

input_data = np.random.randn(2, 3, 6, 6).astype(np.float32)
z = conv_fn(ms.Tensor(input_data))
print(z.asnumpy())
