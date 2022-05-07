"""single op tutorial
This sample code is applicable to Ascend.
"""
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, PYNATIVE_MODE, set_context

set_context(mode=PYNATIVE_MODE, device_target="Ascend")

conv = nn.Conv2d(3, 4, 3, bias_init='zeros')
input_data = Tensor(np.ones([1, 3, 5, 5]).astype(np.float32))
output = conv(input_data)
print(output.asnumpy())
