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
3. Call the Schedule scheduling module. The model tiles the operator data based on the scheduling description and specifies the data transfer process to ensure optimal hardware execution. By default, the automatic scheduling module (`auto_schedule`) can be used.
4. Call `cce_build_code` to compile and generate an operator binary file.

> The input parameters of the entry function require the input information of each operator, output information of each operator, operator attributes (optional), and `kernel_name` (name of the generated operator binary file). The input and output information is encapsulated in dictionaries, including the input and output shape and dtype when the operator is called on the network.

For details about TBE operator development, visit the [TBE website](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_10_0063.html). For details about how to debug and optimize the TBE operator, visit the [Mind Studio website](https://support.huaweicloud.com/usermanual-mindstudioc73/atlasmindstudio_02_0043.html).

### Registering the Operator Information

The operator information is key for the backend to select the operator implementation and guides the backend to insert appropriate type and format conversion operators. It uses the `TBERegOp` API for definition and uses the `op_info_register` decorator to bind the operator information to the entry function of the operator implementation. When the .py operator implementation file is imported, the `op_info_register` decorator registers the operator information to the operator information library at the backend. For details about how to use the operator information, see comments for the member method of `TBERegOp`. For the specific field meaning of the operator information, visit the [TBE website](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_10_0096.html).

> - The numbers and sequences of the input and output information defined in the operator information must be the same as those in the parameters of the entry function of the operator implementation and those listed in the operator primitive.
>
> - If an operator has attributes, use `attr` to describe the attribute information in the operator information. The attribute names must be the same as those in the operator primitive definition.

### Example

The following takes the TBE implementation `square_impl.py` of the `Square` operator as an example. `square_compute` is a computable function of the operator implementation. It describes the computation logic of `x * x` by calling the API provided by `te.lang.cce`. `cus_square_op_info` is the operator information, which is defined by `TBERegOp`.

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

## Implementing an AICPU Operator and Registering the Operator Information

### Implementing an AICPU Operator

The AICPU operator based on CANN includes operator prototype definition, operator code implementation, operator repository definition and other steps, for specific development steps, please refer to [CANN AICPU Custom Operator Development](https://support.huaweicloud.com/usermanual-mindstudio303/atlasms_02_0194.html).

After the development is completed, a file with a specified name will be compiled, such as `libmindspore_aicpu_kernels.so`, `libcust_reshape.so` files. These dynamic libraries can contain one or more AICPU operator implementations, put the file into the lib directory under the MindSpore installation or compilation directory. MindSpore can load the file through subsequent custom operator registration information.

> The dynamic library file implemented by the operator needs to be placed in the lib directory of MindSpore. For example, MindSpore is installed in the virtual environment `/home/conda/envs/aicpu/lib/python3.7/site-packages/mindspore`, so file of the aicpu needs to be placed in the `/home/conda/envs/aicpu/lib/python3.7/site-packages/mindspore/lib/` directory, This allows the file to be loaded normally.

For more information on debugging and performance optimization of AICPU operators, see [MindStudio Documentation](https://support.huaweicloud.com/usermanual-mindstudioc73/atlasmindstudio_02_0043.html).

### Registering the AICPU Custom Operator Information

After completing the previous step, consistent with the TBE operator, we need to supplement the operator information. The AICPU operator is defined through the `AiCPURegOp` interface, and the operator information is bound to the operator implementation entry function through the `op_info_register` decorator. When the operator implements importing the py file, the `op_info_register` decorator registers the operator information in the operator database on the back end. For more information about the use of operator information, please refer to the annotation of the member method of `AiCPURegOp`, and the field meaning of operator information can be found in [AICPU Documentation](https://support.huaweicloud.com/usermanual-mindstudio303/atlasms_02_0194.html).

> - The number and order of input and output information defined in the operator information, the number and order of the input and output information in the parameters of the operator implementation entry function, and the number and order of the input and output name list in the operator primitive should be completely consistent.
> - If the operator has attributes, the property information needs to be described with `attr` in the operator information, and the attribute name is consistent with the attribute name in the operator primitive definition.

It should be noted that in addition to the basic registration information, we need to add an additional `attr("cust_aicpu", "str")` attribute, which is the so name used to get the operator implementation. Taking the `RandomChoiceWithMask` operator as an example, assuming that we have defined the operator primitive and the operator implementation has been compiled to `librandom_choice_with_mask.so`, then we only need to add `attr("cust_aicpu", "str")` to the operator database, and then set the attribute value to  `"random_choice_with_ mask"` to complete registering the operator in the custom AICPU operator list when the operator is defined.

> The value of "cust_aicpu" is a string, and the name of the operator so is denoted by the `lib` prefix and the `.so` suffix, such as `libmindspore_aicpu_kernels.so` is set to `"mindspore_aicpu_kernels"`.

```python
from mindspore.ops import op_info_register, AiCPURegOp, DataType

random_choice_with_mask_op_info = AiCPURegOp("RandomChoiceWithMask") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .output(0, "y", "required") \
    .output(1, "mask", "required") \
    .attr("count", "int") \
    .attr("seed", "int") \
    .attr("seed2", "int") \
    .attr("cust_aicpu", "str") \
    .dtype_format(DataType.BOOL_Default, DataType.I32_Default, DataType.BOOL_Default) \
    .get_op_info()

@op_info_register(random_choice_with_mask_op_info)
def _random_choice_with_mask_aicpu():
    """RandomChoiceWithMask AiCPU register"""
    return
```

### Example

The following is an example of the AICPU call implementation of the `Dropout2D` operator, and we will go through four steps: operator implementation, operator primitive registration, operator information database, and operator call:

1. Operator implementation: Referring to [Implementing an AICPU Operator](#implementing-an-aicpu-operator), we compile the operator to `libmindspore_aicpu_kernels.so`.
2. Operator primitive registration: Referring to [Register the Operator Primitives](#registering-the-operator-primitive), we define a Dropout2D operator.
3. Operator information database: Referring to [Registering the AICPU Custom Operator Information](#registering-the-aicpu-custom-operator-information), we implement the Dropout2D database and add the attribute of `"cust_aicpu"`.
4. Operator call: We can call the Dropout2D operator normally in the form of a single operator network, and at the same time, we can configure the attribute value of `"cust_aicpu"` to be `mindspore_aicpu_kernels`.

```python
import numpy as np
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore import dtype as mstype
from mindspore.ops import op_info_register, AiCPURegOp, DataType
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import set_context, GRAPH_MODE
from mindspore import Tensor
set_context(mode=GRAPH_MODE, device_target="Ascend")

class Dropout2D(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, keep_prob=0.5):
        """Initialize Dropout2D."""
        pass

    def infer_shape(self, x_shape):
        return x_shape, x_shape

    def infer_dtype(self, x_dtype):
        mask_dtype = mstype.tensor_type(mstype.bool_)
        return x_dtype, mask_dtype

dropout2d_op_info = AiCPURegOp("Dropout2D") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .output(0, "y", "required") \
    .output(1, "mask", "required") \
    .attr("keep_prob", "float") \
    .attr("cust_aicpu", "str") \
    .dtype_format(DataType.BOOL_Default, DataType.BOOL_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I8_Default, DataType.I8_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I16_Default, DataType.I16_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U16_Default, DataType.U16_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U32_Default, DataType.U32_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U64_Default, DataType.U64_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.BOOL_Default) \
    .get_op_info()

@op_info_register(dropout2d_op_info)
def _dropout2d_aicpu():
    """Dropout2D AiCPU register"""
    return

class NetDropout2D(nn.Cell):
    def __init__(self, keep_prob=0.5):
        super(NetDropout2D, self).__init__()
        self.op = Dropout2D(keep_prob)
        self.op.add_prim_attr("cust_aicpu", "mindspore_aicpu_kernels")

    def construct(self, inputs):
        return self.op(inputs)

if __name__ == "__main__":
    input_tensor = Tensor(np.ones([1, 1, 2, 3]), mstype.float32)
    dropout2d_nn = NetDropout2D(0.5)
    output, mask = dropout2d_nn(input_tensor)
    print("output: ", output)
    print("mask: ", mask)
```

The execution result is as follows:

```text
output: [[[[0.0.0.]
  [0.0.0.]]]]
mask: [[[[False False False]
  [False False False]]]]
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
