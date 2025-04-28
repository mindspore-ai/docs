# Advance Usage of Custom Operators

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_en/custom_program/operation/op_custom_adv.md)

## Registering the Operator Information

The operator information describes the supported inputs and outputs data type, the supported inputs and outputs format, attributes, and target (platform information) of the operator implementation. It is used to select and map operators by the backend. The operator information can be defined by using the [CustomRegOp](https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.CustomRegOp.html#mindspore-ops-customregop) API, then you can use the [custom_info_register](https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.custom_info_register.html#mindspore-ops-custom-info-register) decorator or just pass it to the `reg_info` parameter of [Custom](https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom) primitive to bind the information to the operator implementation. The operator information will be registered to the operator information library on the MindSpore C++ side at last. The `reg_info` parameter takes higher priority than the `custom_info_register` decorator.

The target value in operator information can be "Ascend", "GPU" or "CPU", which describes the operator information on a specific target. For the same operator implementation, it may have different supported data types on different targets. The operator information on a specific target will be registered only once, so you can use the target value in operator information to differ this.

> - The numbers and sequences of the input and output information defined in the operator information must be the same as those in the parameters of the operator implementation.
> - For the custom operator of akg type, if the operator has attributes, you need to register operator information. The attribute name in the operator information must be consistent with the attribute name used in the operator implementation. For the custom operator of tbe type, you need to register operator information. For the custom operator of aot type, since the operator implementation needs to be compiled into a dynamic library in advance, it is not possible to bind operator information by means of decorators, and the operator information can only be passed in through the `reg_info` parameter.
> - If the custom operator only supports a specific input and output data type or data format, the operator information needs to be registered so that the data type and data format can be checked when the operator is selected in the backend. For the case where the operator information is not provided, the information will be derived from the inputs of the current operator.

## Defining the bprop Function for Operators

If an operator needs to support automatic differentiation, the backpropagation (bprop) function needs to be defined first and then passed to the `bprop` parameter of `Custom` primitive. In the bprop function, you need to describe the backward computation logic that uses the forward input, forward output, and output gradients to obtain the input gradients. The backward computation logic can be composed of built-in operators or custom operators.

Note the following points when defining the bprop function for operators:

- The input parameter sequence of the bprop function is the forward input, forward output, and output gradients. For a multi-output operator, the forward output and output gradients are provided in the form of tuples.
- The return value of the bprop function is tuples consisting of input gradients. The sequence of elements in a tuple is the same as that of the forward input parameters. Even if there is only one input gradient, the return value must be a tuple.

Take test_grad.py as an example to show the usage of the backpropagation function:

```python
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_device(device_target="CPU")

# Forward computation of custom operator
def square(x):
    return x * x

# Backward computation of custom operator
def square_grad(x, dout):
    dx = 2.0 * x * dout
    return dx

# Backpropagation function
def bprop():
    op = ops.Custom(square_grad, lambda x, _: x, lambda x, _: x, func_type="pyfunc")

    def custom_bprop(x, out, dout):
        dx = op(x, dout)
        return (dx,)

    return custom_bprop

class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        # Define a custom operator of pyfunc type and provide a backpropagation function
        self.op = ops.Custom(square, lambda x: x, lambda x: x, bprop=bprop(), func_type="pyfunc")

    def construct(self, x):
        return self.op(x)

if __name__ == "__main__":
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    dx = ms.grad(Net())(ms.Tensor(x))
    print(dx)
```

```text
[ 2.  8. 18.]
```

The following points need to be explained in this example:

- The backpropagation function uses a custom operator of akg type, and the operator definition and use need to be separated, that is, the custom operator is defined outside the `custom_bprop` function and used inside the `custom_bprop` function.

Execute case:

```bash
python test_grad.py
```

The execution result is as follows:

```text
[ 2.  8. 18.]
```

> More examples can be found in the MindSpore [source code](https://gitee.com/mindspore/mindspore/tree/v2.6.0/tests/st/graph_kernel/custom).
