"""grad tutorial
This sample code is applicable to Ascend.
"""
import mindspore.ops as ops
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

def mul(x, y):
    return x * y

def mainf(x, y):
    return ops.GradOperation(get_all=True)(mul)(x, y)

print(mainf(ms.Tensor(1, ms.int32), ms.Tensor(2, ms.int32)))
