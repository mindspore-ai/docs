import mindspore.ops as ops
import mindspore.context as context
from mindspore import dtype as mstype
from mindspore import Tensor

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

def mul(x, y):
    return x * y

def mainf(x, y):
    return ops.GradOperation(get_all=True)(mul)(x, y)

print(mainf(Tensor(1, mstype.int32), Tensor(2, mstype.int32)))