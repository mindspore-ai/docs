# Custom Operators (Ascend)

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/operation/op_ascend.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

When built-in operators cannot meet requirements during network development, you can call the Python API of MindSpore to quickly extend custom operators of the Ascend AI processor.

To add a custom operator, you need to register the operator primitive, implement the operator, and register the operator information.

The related concepts are as follows:  

- Operator primitive: defines the frontend API prototype of an operator on the network. It is the basic unit for forming a network model and includes the operator name, attribute (optional), input and output names, output shape inference method, and output dtype inference method.
- Operator implementation: describes the implementation of the internal computation logic for an operator through the DSL API provided by the Tensor Boost Engine (TBE). The TBE supports the development of custom operators based on the Ascend AI chip.
- Operator information: describes basic information about a TBE operator, such as the operator name and supported input and output types. It is the basis for the backend to select and map operators.

This section takes a Square operator as an example to describe how to customize an operator.

> For details, see cases in [tests/st/ops/custom_ops_tbe](https://gitee.com/mindspore/mindspore/tree/master/tests/st/ops/custom_ops_tbe) in the MindSpore source code.

## Registering the Operator Primitive

The primitive of an operator is a subclass inherited from `PrimitiveWithInfer`. The type name of the subclass is the operator name.

The definition of the custom operator primitive is the same as that of the built-in operator primitive.  

- The attribute is defined by the input parameter of the constructor function `__init__`. The operator in this test case has no attribute. Therefore, `__init__` has only one input parameter. For details about test cases in which operators have attributes, see [custom add3](https://gitee.com/mindspore/mindspore/blob/master/tests/st/ops/custom_ops_tbe/cus_add3.py) in the MindSpore source code.
- The input and output names are defined by the `init_prim_io_names` function.
- The shape inference method of the output tensor is defined in the `infer_shape` function, and the dtype inference method of the output tensor is defined in the `infer_dtype` function.

The only difference between a custom operator and a built-in operator is that the operator implementation function (`from square_impl import CusSquareImpl`) needs to be imported to the `__init__` function to register the operator implementation with the backend for the custom operator. In this test case, the operator implementation and information are defined in `square_impl.py`, and the definition will be described in the following parts.

The following code takes the Square operator primitive `cus_square.py` as an example:

```python
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
import mindspore.ops as ops
# y = x^2
class CusSquare(PrimitiveWithInfer):
    """
    The definition of the CusSquare primitive.
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        from square_impl import CusSquareImpl # Import the entry function of the kernel implementation from relative path or PYTHONPATH.

    def infer_shape(self, data_shape):
        return data_shape

    def infer_dtype(self, data_dtype):
        return data_dtype
```

## Implementing a TBE Operator and Registering the Operator Information

### Implementing a TBE Operator

To compile an operator implementation, you need to compile a computable function and an entry function first.

The computable function of an operator is mainly used to encapsulate the computation logic of the operator for the main function to call. The computation logic is implemented by calling the combined API of the TBE.

The entry function of an operator describes the internal process of compiling the operator. The process is as follows:  

1. Prepare placeholders to be input. A placeholder will return a tensor object that represents a group of input data.
2. Call the computable function. The computable function uses the API provided by the TBE to describe the computation logic of the operator.
3. Call the scheduling module. The model tiles the operator data based on the scheduling description and specifies the data transfer process to ensure optimal hardware execution. By default, the automatic scheduling module (`auto_schedule`) can be used.
4. Call `cce_build_code` to compile and generate an operator binary file.

> The input parameters of the entry function require the input information of each operator, output information of each operator, operator attributes (optional), and `kernel_name` (name of the generated operator binary file). The input and output information is encapsulated in dictionaries, including the input and output shape and dtype when the operator is called on the network.

For details about TBE operator development, visit the [TBE website](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_10_0063.html). For details about how to debug and optimize the TBE operator, visit the [Mind Studio website](https://support.huaweicloud.com/usermanual-mindstudioc73/atlasmindstudio_02_0043.html).

### Registering the Operator Information

The operator information is key for the backend to select the operator implementation and guides the backend to insert appropriate type and format conversion operators. It uses the `TBERegOp` API for definition and uses the `op_info_register` decorator to bind the operator information to the entry function of the operator implementation. When the .py operator implementation file is imported, the `op_info_register` decorator registers the operator information to the operator information library at the backend. For details about how to use the operator information, see comments for the member method of `TBERegOp`.

> The numbers and sequences of the input and output information defined in the operator information must be the same as those in the parameters of the entry function of the operator implementation and those listed in the operator primitive.
>
> If an operator has attributes, use `attr` to describe the attribute information in the operator information. The attribute names must be the same as those in the operator primitive definition.

### Example

The following takes the TBE implementation `square_impl.py` of the `Square` operator as an example. `square_compute` is a computable function of the operator implementation. It describes the computation logic of `x * x` by calling the API provided by `te.lang.cce`. `cus_square_op_info` is the operator information, which is defined by `TBERegOp`. For the specific field meaning of the operator information, visit the [TBE website](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_10_0096.html).

Note the following parameters when setting `TBERegOp`:

- `OPAQUE` in `fusion_type("OPAQUE")` indicates that the custom operator uses the non-fusion strategy.
- `CusSquareImpl` in `kernel_name("CusSquareImpl")` must be the same as the name of the operator entry function.
- `dtype_format` is used to describe data types supported by the operator. In the following example, two types are registered, indicating that the operator supports two data types. Each type describes the supported format in order of input and output. The first `dtype_format` indicates that the data type input0 is in F32_Default format and the data type output0 is in F32_Default format. The second `dtype_format` indicates that the data type input0 is in F16_Default format and the data type output0 is in F16_Default format.
- About the interfaces `auto_schedule` and `cce_build_code`, please see the TBE documents [auto_schedule](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_07_0071.html) and [cce_build_code](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_07_0072.html) for details.

```python
from __future__ import absolute_import
from te import tvm
from topi import generic
import te.lang.cce
from topi.cce import util
from mindspore.ops import op_info_register, TBERegOp, DataType

def square_compute(input_x):
    """
    The compute function of the CusSquare implementation.
    """
    res = te.lang.cce.vmul(input_x, input_x)
    return res

# Define the kernel info of CusSquare.
cus_square_op_info = TBERegOp("CusSquare") \
    .fusion_type("OPAQUE") \
    .partial_flag(True) \
    .async_flag(False) \
    .binfile_name("square.so") \
    .compute_cost(10) \
    .kernel_name("CusSquareImpl") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()

# Binding kernel info with the kernel implementation.
@op_info_register(cus_square_op_info)
def CusSquareImpl(input_x, output_y, kernel_name="CusSquareImpl"):
    """
    The entry function of the CusSquare implementation.
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()

    shape = util.shape_refine(shape)
    data = tvm.placeholder(shape, name="data", dtype=dtype.lower())

    with tvm.target.cce():
        res = square_compute(data)
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(sch, config)
```

## Using Custom Operators

The usage of custom operators is the same as that of built-in operators in the network. The operators can be directly used by importing primitives. The following takes the single-operator network test of `CusSquare` as an example.

Define the network in the `test_square.py` file.

```python
import numpy as np
import mindspore.nn as nn
from mindspore import set_context, GRAPH_MODE
from mindspore import Tensor
# Import the definition of the CusSquare primitive.
from cus_square import CusSquare
set_context(mode=GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.square = CusSquare()

    def construct(self, data):
        return self.square(data)

def test_net():
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    square = Net()
    output = square(Tensor(x))
    print("x: ", x)
    print("output: ", output)
```

Execute the test case.

```bash
pytest -s tests/st/ops/custom_ops_tbe/test_square.py::test_net
```

The execution result is as follows:

```text
x: [1. 4. 9.]
output: [1. 16. 81.]
```

## Defining the bprop Function for an Operator

If an operator needs to support automatic differentiation, the bprop function needs to be defined in the primitive of the operator. In the bprop function, you need to describe the backward computation logic that uses the forward input, forward output, and output gradients to obtain the input gradients. The backward computation logic can be composed of built-in operators or custom backward operators.

Note the following points when defining the bprop function:

- The input parameter sequence of the bprop function is the forward input, forward output, and output gradients. For a multi-output operator, the forward output and output gradients are provided in the form of tuples.
- The return value of the bprop function is tuples consisting of input gradients. The sequence of elements in a tuple is the same as that of the forward input parameters. Even if there is only one input gradient, the return value must be a tuple.

For example, the `CusSquare` primitive after the bprop function is added is as follows:

```python
class CusSquare(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """init CusSquare"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        from square_impl import CusSquareImpl

    def infer_shape(self, data_shape):
        return data_shape

    def infer_dtype(self, data_dtype):
        return data_dtype

    def get_bprop(self):
        def bprop(data, out, dout):
            twos_like = ops.OnesLike()(data) * 2.0
            gradient = ops.Mul()(data, twos_like)
            dx = ops.Mul()(gradient, dout)
            return (dx,)
        return bprop
```

Define backward cases in the `test_square.py` file.

```python
import mindspore.ops as ops
def test_grad_net():
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    sens = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    square = Net()
    grad = ops.GradOperation(sens_param=True)
    dx = grad(square)(Tensor(x), Tensor(sens))
    print("x: ", x)
    print("dx: ", dx)
```

Execute the test case.

```bash
pytest -s tests/st/ops/custom_ops_tbe/test_square.py::test_grad_net
```

The execution result is as follows:

```text
x: [1. 4. 9.]
dx: [2. 8. 18.]
```
