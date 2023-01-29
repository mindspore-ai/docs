# Dependency Control

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_en/network/dependency_control.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

If the result of a function depends on or affects an external state, we consider that the function has side effects, such as a function changing an external global variable, and the result of a function depends on the value of a global variable. If the operator changes the value of the input parameter or the output of the operator depends on the value of the global parameter, we think this is an operator with side effects.

Side effects are classified as memory side effects and IO side effects based on memory properties and IO status. At present, memory side effects are mainly Assign, optimizer operators and so on, while IO side effects are mainly Print operators. You can view the operator definition in detail. The memory side effect operator has side_effect_mem properties in the definition, and the IO side effect operator has side_effect_io properties in the definition.

Depend is used for processing dependency operations. In most cases, if the operators have IO or memory side effects, they will be executed according to the user's semantics, and there is no need to use the Depend operator to guarantee the execution order. In some cases, if the two operators A and B do not have sequential dependencies, and A must be executed before B, we recommend that you use Depend to specify the order in which they are executed. Here's how to use it:

```python
a = A(x)
b = B(y)
```

After inserting the Depend operator, as follows:

```python
a = A(x)
y = Depend(y, a)
b = B(y)
```

Please note that a special set of operators for detecting floating point overflow state have hidden side effects, which are not IO side effects or memory side effects. In addition, there are strict sequencing requirements for use, i.e., before using the NPUClearFloatStatus operator, you need to ensure that the NPU AllocFloatStatus has been executed, and before using the NPUGetFloatStatus operator, you need to ensure that the NPUClearFlotStatus has been executed. Because these operators are used less, the current scenario is to keep their definition as in the form of side-effect-free and to ensure execution order by Depend. Examples are as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import ops, set_context, Tensor
from mindspore import dtype as mstype

set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.alloc_status = ops.NPUAllocFloatStatus()
        self.get_status = ops.NPUGetFloatStatus()
        self.clear_status = ops.NPUClearFloatStatus()
        self.sub = ops.Sub()
        self.neg = ops.Neg()

    def construct(self, x):
        init = self.alloc_status()
        clear_status = self.clear_status(init)
        x = ops.depend(x, clear_status)
        res = self.sub(x, self.neg(x))
        init = ops.depend(init, res)
        get_status = self.get_status(init)
        res = ops.depend(res, get_status)
        return res

value = 5
data = np.full((2, 3), value, dtype=np.float16)
x = Tensor(data, dtype=mstype.float16)
net = Net()
res = net(x)
print(res)
```

Running the above script, you can get:

```text
 [[10. 10. 10.]
  [10. 10. 10.]]
```