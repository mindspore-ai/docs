"""grad tutorial
This sample code is applicable to Ascend.
"""
import mindspore.ops as ops
from mindspore import dtype as mstype, set_context, PYNATIVE_MODE
from mindspore import Tensor

set_context(mode=PYNATIVE_MODE, device_target="Ascend")

def mul(x, y):
    return x * y

def mainf(x, y):
    return ops.GradOperation(get_all=True)(mul)(x, y)

print(mainf(Tensor(1, mstype.int32), Tensor(2, mstype.int32)))
