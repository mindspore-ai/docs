# Custom Operators (Custom-based)

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/operation/op_custom.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

When built-in operators cannot meet requirements during network development, you can call the Python API [Custom](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom) primitive defined in MindSpore to quickly create different types of custom operators for use.

Traditional methods to add a custom operator need three steps: registering the operator primitive, implementing the operator, and registering the operator information.

The related concepts are as follows:

- Operator primitive: defines the frontend API prototype of an operator on the network. It is the basic unit for forming a network model and includes the operator name, attribute (optional), input and output names, output shape inference method, and output data type inference method.
- Operator implementation: defines a Python function (Ascend custom operators) or a C++ class (GPU and CPU custom operators), which describes the implementation of the internal computation logic of an operator.
- Operator information: describes basic information about an operator, such as the operator name, supported input and output data types, supported input and output data formats, and attributes. It is the basis for the backend to select and map operators.

Compared with traditional custom operator creating methods, creating custom operators based on `Custom` primitive has several advantages:

- Different custom operators use the same `Custom` primitive, there is no need to define a primitive for every operator. The above three parts of work can be implemented in a network script in a unified way and used as part of the network expression, there is no need to modify and recompile the source codes of MindSpore.
- It unifies the interface and usage for different kinds of custom operators, which is convenient for network developers to flexibly choose which kind of custom operator to use according to their needs.
- Supports defining custom operators with hybrid expression, which can be used across platforms.

## Basic Usage

The operator development methods supported by custom operator based on the [Custom](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom) primitive include: hybrid, tbe, aot, pyfunc, julia, and akg.

The difference between these operator development methods are as follows:

| Defining Methods | Development Language | Compilation Method | Supported Platforms | Recommended Scenarios                                                         |
|:----------------:|:--------------------:| :------: | ------ |-------------------------------------------------------------------------------|
| hybrid           | MindSpore HYBRID DSL | JIT | `Ascend` `GPU` `CPU` | Ascend/GPU platform common development and rapid validation |
|       tbe        |       TBE DSL        | JIT | `Ascend` | Ascend AICORE custom the operator scenarios         |
|      aicpu       |        C/C++         |        AOT         | `Ascend`             | Ascend AICORE custom the operator scenarios                  |
|       aot        |      C/C++/CUDA      |        AOT         | `GPU` `CPU`          | high-performance scenarios / use third-party operators scenarios |
|      pyfunc      |        Python        |        JIT         | `CPU`                | Fast algorithm verification, need to interact with Python and other scenarios |
|      julia       |        Julia         |        JIT         | `CPU`                | Science compute scenarios / use Julia scenarios              |
| akg | MindSpore AKG DSL | JIT | `Ascend` `GPU` | Ascend/GPU platform general scenarios |

> - The full name of DSL is Domain Specific Language.
> - AOT(Ahead Of Time) compiling means the operator implementation needs to be compiled into a dynamic library in advance and then automatically called by the framework when the network is running. JIT(Just In Time) compiling does not need to compile the operator implementation in advance, the operator implementation will be directly called by the framework during network compilation or runtime.

Different custom operator defining methods use different development languages to implement the operator, but the development process is the same, including operator implementation, operator output shape, data type inference, and operator information registration (optional). You can choose which one to use based on needs. The defining methods of these custom operators will be introduced here, and examples are provided for each method.

> More examples can be found in the MindSpore source code [tests/st/ops/graph_kernel/custom](https://gitee.com/mindspore/mindspore/tree/master/tests/st/ops/graph_kernel/custom).

### Defining Custom Operator of Hybrid Type

A custom operator of Hybrid type is the default defined type of a custom operator. By using a custom operator of the Hybrid type, users can describe the operator calculation logic in Python-like syntax without paying attention to the engineering details defined by the operator for the MindSpore framework, allowing the user to focus on the algorithm itself.

Custom operators of Hybrid type use [MindSpore Hybrid DSL](#mindspore-hybrid syntax specification) to describe the implementation of the calculation logic inside the operator. Functions defined with MindSpore Hybrid DSL can be parsed by the [AKG Operator Compiler](https://gitee.com/mindspore/akg) for JIT compilation to generate efficient operators for use in training reasoning for large-scale models.

Custom operators of Hybrid type use [MindSpore Hybrid DSL](#mindspore-hybrid syntax specification) to describe the implementation of the calculation logic inside the operator. Functions defined with MindSpore Hybrid DSL can be parsed by the [AKG Operator Compiler](https://gitee.com/mindspore/akg) for JIT compilation to generate efficient operators for use in training reasoning for large-scale models. At the same time, the function defined by MindSpore Hybrid DSL can be called directly as a `numpy` function, which is convenient for users to debug and flexibly switch to [pyfunc type custom operator](#pyfunc type custom operator), so that develop once, custom operator expressions are reused for multiple modes, multiple platforms and multiple scenes.

The following example (test_custom_hybrid.py) shows how to write a custom operator of the hybrid type. The operator computes the sum of two tensors.

```python
import numpy as np
from mindspore import Tensor, ops, set_context
from mindspore.ops import ms_hybrid

set_context(device_target="GPU")

# Operator implementation, Hybrid DSL
@ms_hybrid
def add(a, b):
    c = output_tensor(a.shape, a.dtype)
    for i0 in range(a.shape[0]):
        for i1 in range(a.shape[1]):
            c[i0, i1] = a[i0, i1] + b[i0, i1]
    return c

if __name__ == "__main__":
    # Define custom operators of the hybrid type (Custom's default mode)
    op = ops.Custom(add)

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

In this case,

- The Hybrid type is the default type for Custom.
- The input of custom operators with Hybrid type must be a function with [`@ms_hybrid`](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ms_hybrid.html).
- When defining a custom operator for the Hybrid type, you can use the built-in automatic shape/dtype derivation function, or you can manually enter the shape/dtype deduction function.

Execute case:

```bash
python test_custom_hybrid.py
```

The execution result is as follows:

```text
[[2. 2.]
 [4. 4.]]
```

### Defining Custom Operator of tbe Type

The custom operator of tbe type uses the TBE(Tensor Boost Engine) operator DSL to describe the internal calculation logic of the operator. You can refer to the [TBE document](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_10_0063.html) for the implementation details.

Operator output shape and data type inference can be realized by defining Python functions to describe the inference logic.

Operator information needs to be registered. For the creation of operator information, please refer to [Registering the Operator Information](#registering-the-operator-information).

Takes test_custom_tbe.py as an example to introduce how to define a custom operator of tbe type, where the custom operator implements the function of adding two input tensors.

Here is the content of test_custom_tbe.py:

```python
import numpy as np
from mindspore import Tensor, set_context
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp, custom_info_register

set_context(device_target="Ascend")

# Operator implementation, and operator information registration
@custom_info_register(CustomRegOp() \
                      .input(0, "a") \
                      .input(1, "b") \
                      .output(0, "output") \
                      .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
                      .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
                      .target("Ascend") \
                      .get_op_info())
def add(a, b, output, kernel_name="add"):
    import te.lang.cce
    from te import tvm
    data0 = tvm.placeholder(a.get("shape"), name="data0", dtype=a.get("dtype").lower())
    data1 = tvm.placeholder(b.get("shape"), name="data1", dtype=b.get("dtype").lower())
    res = te.lang.cce.vadd(data0, data1)
    with tvm.target.cce():
        sch = te.lang.cce.auto_schedule(res)
    config = {"print_ir": False, "name": kernel_name, "tensor_list": [data0, data1, res]}
    te.lang.cce.cce_build_code(sch, config)

if __name__ == "__main__":
    # Define a custom operator of tbe type
    op = ops.Custom(add, out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="tbe")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

The following points need to be explained in this example:

- Use Python lambda functions to infer the output shape and data type, and pass them to the `out_shape` and `out_dtype` parameters of the `Custom` primitive. In this example, the lambda function indicates that the output shape and data type are the same as the information of the first input tensor.
- Use `CustomRegOp` to create the operator information and use `custom_info_register` decorator to register it.

Execute case:

```bash
python test_custom_tbe.py
```

The execution result is as follows:

```text
[[2. 2.]
 [4. 4.]]
```

### Defining Custom Operator of aicpu Type

The custom operator of the aicpu type adopts the AOT compilation method, which requires the operator developer to implement the corresponding source code file of the function based on the specific interface provided, and compiles the source code file into a dynamic link library in advance. The framework will find the corresponding dynamic link library and load the operator according to the name of the dynamic link library configured by the developer in the operator properties. Reference for specific operator implementation [CANN AICPU Custom Operator Development](https://support.huaweicloud.com/usermanual-mindstudio303/atlasms_02_0193.html).

Operator output shape and data type inference can be implemented by defining Python functions that describe the derivation logic of operator output shape and data type.

This type of custom operator needs to register operator information, operator information generation method, please refer to [Registering the Operator Information](#Registering the Operator Information), a custom operator of type aicpu, you need to specify the attributes of `attr("cust_aicpu", "required", "str", "mindspore_aicpu_kernels")` for MindSpore to find the dynamic link library corresponding to the operator implementation.

> - It should be noted that the dynamic link library compiled after the development of a custom operator of aicpu type needs to be stored in the lib directory of MindSpore. For example, If MindSpore is installed in the virtual environment `/home/conda/envs/aicpu/lib/python3.7/site-packages/mindspore`, the aicpu so file needs to be placed in `/home/conda/envs/aicpu/lib/python3.7/site-packages/mindspore/lib/` directory.
>
> - The value of "cust_aicpu" is a string, which is represented by the `lib` prefix and the `.so` suffix removed from the name of the operator dynamic link library. If the name of `libmindspore_aicpu_kernels.so` is removed, it can be set to `mindspore_aicpu_kernels`.

The following takes test_dropout_aicpu.py as an example to introduce the development process of custom operators of type aicpu, in which the custom operator implements the function of dropout, and the compiled operator dynamic link library, we named it libmindspore_aicpu_kernels.so, and have put the dynamic link library under the lib of the mindspore root directory.

The contents of test_dropout_aicpu.py are as follows:

```python
import numpy as np
from mindspore import Tensor, set_context, GRAPH_MODE
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.ops import CustomRegOp, custom_info_register, DataType

set_context(mode=GRAPH_MODE, device_target="Ascend")

# Operator implementation, registering operator information
dropout2d_op_info = CustomRegOp("Dropout2D") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .output(0, "y", "required") \
    .output(1, "mask", "required") \
    .attr("keep_prob", "required", "float") \
    .attr("cust_aicpu", "required", "str", "mindspore_aicpu_kernels") \
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
    .target("Ascend") \
    .get_op_info()

@custom_info_register(dropout2d_op_info)
def dropout2d_aicpu():
    """Dropout2D AiCPU register"""
    return

# Define a custom operator network
class NetDropout2D(nn.Cell):
    def __init__(self, keep_prob=0.5):
        super(NetDropout2D, self).__init__()
        self.op = ops.Custom(dropout2d_aicpu, out_shape=lambda x, _, cust_attr: (x, x), \
                              out_dtype=lambda x, _, cust_attr: (x, mstype.bool_), func_type="aicpu")
        self.keep_prob = keep_prob
        self.cust_aicpu_so_path = "mindspore_aicpu_kernels"

    def construct(self, inputs):
        return self.op(inputs, self.keep_prob,  self.cust_aicpu_so_path)

if __name__ == "__main__":
    # Defines a custom operator of type aicpu
    input_tensor = Tensor(np.ones([1, 1, 2, 3]), mstype.float32)
    dropout2d_nn = NetDropout2D(0.5)
    output, mask = dropout2d_nn(input_tensor)
    print("output: ", output)
    print("mask: ", mask)
```

In this example, there are the following points to explain:

- The `out_shape` and `out_dtype` parameters of the `Custom` primitive can be specified in a variety of ways, either given a type or set with a Python lambda function. In this example, the lambda function indicates that the two shapes of the output are the same as the input, the data type of the first output and the information of the input tensor are the same, and the data type of the second output is the bool type.
- Operator information is generated via `CustomRegOp` and operator information is registered with the `custom_info_register` decorator.

Executing cash:

```bash
python test_dropout_aicpu.py
```

The execution result is as follows (due to the random nature of the dropout operator, there is a difference in the result of multiple runs):

```text
output : [[[[2.  2.  2.] [2.  2.  2.]]]]
mask: [[[[True  True  True]  [True  True  True]]]]
```

### Defining Custom Operator of aot Type

The custom operator of aot type adopts the AOT compilation method, which requires network developers to hand-write the source code file of the operator implementation based on a specific interface and compiles the source code file into a dynamic library in advance, and then the framework will automatically call and run the function defined in the dynamic library. In terms of the development language of the operator implementation, the GPU platform supports CUDA, and the CPU platform supports C and C++. The interface specification of the operator implementation in the source file is as follows:

```cpp
extern "C" int func_name(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra);
```

where the function name `func_name` can be replaced with any valid function name. The return value is of type int. 0 means normal exit, and non-zero means an exception occurs. The meaning of the parameter list is as follows:

- nparam (int): The number of inputs and outputs. For example, if an operator has 2 inputs and 1 output, then the value of nparam is 3.
- params (void \*\*): An array of pointers, with each pointer pointing to the input or output data. For example, if an operator has 2 inputs and 1 output, then params[0] points to the first input data, params[1] points to the second input data, params[2] points to the output data.
- ndims (int \*): An array of integers, each integer represents the dimensions of the shape of input or output. For example, if params[i] is a tensor with shape [1024, 1024], then ndims[i] is 2.
- shapes (int64_t \*\*): An array of shapes, each element in array represents for the shape of input or output. For example, if params[i] is a tensor with shape [1024, 1024], then shapes[i][0] is 1024, shapes[i][1] is 1024.
- dtypes (const char \*\*): Array of data types, each element in array represents for the data type of input or output. The value of data type can be "float32", "float16", "float", "float64", "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", "bool".
- stream (void \*): Stream pointer, only used in Cuda file.
- extra (void \*): Used for further extension.

Operator output shape and data type inference can be realized by defining Python functions to describe the inference logic.

If the operator only supports some specific input and output data types, then the operator information needs to be registered. For the creation of operator information, please refer to [Registering the Operator Information](#registering-the-operator-information).

The following examples introduce the development process of aot type custom operator on GPU platform and CPU platform, where the custom operator implements the function of adding two input tensors.

#### A GPU Example

Use the CUDA language to write the source file add.cu for the operator implementation:

```cpp
#define THREADS 1024
__global__ void CustomAddKernel(float *input1, float *input2, float *output, size_t size) {
  auto idx = blockIdx.x * THREADS + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[idx] + input2[idx];
  }
}

extern "C" int CustomAdd(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  if (nparam != 3) return 1;
  void *input1 = params[0];
  void *input2 = params[1];
  void *output = params[2];
  size_t size = 1;

  for (int i = 0; i < ndims[2]; i++) {
    size *= shapes[2][i];
  }
  int n = size / THREADS;
  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }
  CustomAddKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<float *>(input1), static_cast<float *>(input2),
                                                   static_cast<float *>(output), size);
  return 0;
}
```

Compile add.cu into a dynamic library add.so:

```bash
nvcc --shared -Xcompiler -fPIC -o add.so add.cu
```

Write the test case test_custom_aot.py:

```python
import numpy as np
from mindspore import set_context, Tensor
import mindspore.ops as ops

set_context(device_target="GPU")

if __name__ == "__main__":
    # Define a custom operator of aot type
    op = ops.Custom("./add.so:CustomAdd", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="aot")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

The following points need to be explained in this example:

- In this example, you need to place test_custom_aot.py and add.so in the same directory. If add.so is in another directory, you need to replace the value of the first parameter of `Custom` primitive with the absolute path of add.so.
- Use Python lambda functions to infer the output shape and data type, and pass them to the `out_shape` and `out_dtype` parameters of the `Custom` primitive. In this example, the lambda function indicates that the output shape and data type are the same as the information of the first input tensor.
- The operator information is not registered, so the operator information of the custom operator will be inferred from the inputs.

Execute case:

```bash
python test_custom_aot.py
```

The execution result is as follows:

```text
[[2. 2.]
 [4. 4.]]
```

#### A CPU Example

Use C/C++ language to write the source file add.cc for the operator implementation:

```cpp
#include <string.h>
using size_t = decltype(sizeof(int));
using int64_t = decltype(sizeof(long));

extern "C" int CustomAdd(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra) {
  if (nparam != 3) return 1;
  float *input1 = static_cast<float *>(params[0]);
  float *input2 = static_cast<float *>(params[1]);
  float *output = static_cast<float *>(params[2]);
  size_t size = 1;
  for (int i = 0; i < nparam; i++) {
    size *= shapes[2][i];
  }
  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }
  for (int i = 0; i < size; i++) {
    output[i] = input1[i] + input2[i];
  }
  return 0;
}
```

Compile add.cc into a dynamic library add.so:

```bash
g++ --shared -fPIC -o add.so add.cc
```

Write the test case test_custom_aot.py:

```python
import numpy as np
from mindspore import set_context, Tensor
import mindspore.ops as ops

set_context(device_target="CPU")

if __name__ == "__main__":
    # Define a custom operator of aot type
    op = ops.Custom("./add.so:CustomAdd", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="aot")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

The following points need to be explained in this example:

- In this example, you need to place test_custom_aot.py and add.so in the same directory. If add.so is in another directory, you need to replace the value of the first parameter of `Custom` primitive with the absolute path of add.so.
- Use Python lambda functions to infer the output shape and data type, and pass them to the `out_shape` and `out_dtype` parameters of the `Custom` primitive. In this example, the lambda function indicates that the output shape and data type are the same as the information of the first input tensor.
- The operator information is not registered, so the operator information of the custom operator will be inferred from the inputs.

Execute case:

```bash
python test_custom_aot.py
```

The execution result is as follows:

```text
[[2. 2.]
 [4. 4.]]
```

### Defining Custom Operator of pyfunc Type

The custom operator of pyfunc type uses native Python syntax to define the operator implementation, which describes the internal calculation logic of the operator. The framework will automatically call this function during the network runtime.

Operator output shape and data type inference can be realized by defining Python functions to describe the inference logic.

If the operator only supports some specific input and output data types, then the operator information needs to be registered. For the creation of operator information, please refer to [Registering the Operator Information](#registering-the-operator-information).

Takes test_custom_pyfunc.py as an example to introduce how to define a custom operator of pyfunc type, where the custom operator implements the function of adding two input tensors.

Here is the content of test_custom_pyfunc.py:

```python
import numpy as np
from mindspore import set_context, Tensor
import mindspore.ops as ops

set_context(device_target="CPU")

def add(a, b):
    return a + b

if __name__ == "__main__":
    # Define a custom operator of pyfunc type
    op = ops.Custom(add, out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="pyfunc")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

The following points need to be explained in this example:

- Use Python lambda functions to infer the output shape and data type, and pass them to the `out_shape` and `out_dtype` parameters of the `Custom` primitive. In this example, the lambda function indicates that the output shape and data type are the same as the information of the first input tensor.
- The operator information is not registered, so the operator information of the custom operator will be inferred from the inputs.

Execute case:

```bash
python test_custom_pyfunc.py
```

The execution result is as follows:

```text
[[2. 2.]
 [4. 4.]]
```

### Defining Custom Operator of julia Type

The custom operator of julia type uses Julia to describe the internal calculation logic of the operator. The framework will automatically call this function during the network runtime.

Operator output shape and data type inference can be realized by defining Python functions to describe the inference logic of the operator output shape and the data type.

If the custom operator only supports specific input and output data types, you need to define the operator information. For the creation of operator information, please refer to [Registering the Operator Information](#registering-the-operator-information).

Takes the function of adding two input tensors as an example to introduce how to define a custom operator of julia type.

Firstly, users need to implement Julia functions via separate files, such as (add.jl):

```julia
# add.jl
module Add
# inputs: x, y, output: z, output should use .= to inplace assign
function add(x, y, z)
    z .= x + y
end
end
```

Secondly, refer to the Julia function written above in a custom operator in the network script, taking test_custom_julia.py as an example:

```python
import numpy as np
from mindspore import set_context, Tensor
import mindspore.ops as ops

set_context(device_target="CPU")

if __name__ == "__main__":
    op = ops.Custom("./add.jl:Add:add", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="julia")
    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

The following points need to be explained in this example:

- Use Python lambda functions to infer the output shape and data type, and pass them to the `out_shape` and `out_dtype` parameters of the `Custom` primitive. In this example, the lambda function indicates that the output shape and data type are the same as the information of the first input tensor.
- The operator information is not registered, so the operator information of the custom operator will be inferred from the inputs.

Execute case:

```bash
python test_custom_julia.py
```

The execution result is as follows:

```text
[[2. 2.]
 [4. 4.]]
```

Matters need attention:

1. User should make sure to download the correct version of Julia, that is, version >= 1.6.0.
2. User is required to set `julia/lib` to `LD_LIBRARY_PATH`, because the Julia C API called at runtime is obtained from `libjulia.so`, taking julia-1.6.5 as an example:

   ```bash
   # download julia-1.6.5
   wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.5-linux-x86_64.tar.gz
   # extract file
   tar xvf julia-1.6.5-linux-x86_64.tar.gz
   # if $JULIA_DIR not exist
   export LD_LIBRARY_PATH=$PWD/julia-1.6.5/lib:$LD_LIBRARY_PATH
   # else
   export LD_LIBRARY_PATH=$JULIA_DIR/lib:$LD_LIBRARY_PATH
   ```

3. `Custom` operator's first arg `func` should keep format like `file_name:module_name:func_name`, `file_name` should include path, suggest using absolute path.
4. Julia file should include `module`, `module` include `function`, both ends with `end`.
5. The input and output order of the Julia function needs to be consistent with the input and output order of the operator.
6. The final output of the Julia function, i.e. assignment of kernel output, needs to use `.=`, otherwise the result cannot be written to memory.
7. Julia code supports Julia's common syntax, and referring to the <https://docs.julialang.org/en/v1/>, users need to ensure that the syntax is correct and the function can be executed correctly.
8. Users who want to use Julia's third-party software packages in Julia files need to download the corresponding software to ensure that they can call it correctly, which can be called through `import pkg; pkg.add ("somepkg")` to install.
9. `julia array` is `column major` arranged in memory, while `numpy array` is `row major`. If Julia and numpy are compared, non-elemwise calculations need to consider memory arrangement. In the Julia function, the conversion of `numpy array` and `julia array` can be performed by following the following code example:An example of MatMul:

     ```julia
    function change_input_to_row_major(x)
        return permutedims(reshape(x, reverse(size(x))), length(size(x)):-1:1)
    end

    function change_output_to_row_major(x)
        return reshape(permutedims(x, length(size(x)):-1:1), size(x))
    end
    ```

Taking matrix multiplication as an example:

```julia
# julia array is column-major, numpy aray is row-major
# user should change julia or numpy's layout to keep same behavior
#= EXAMPLE
A[2,3]               B[3,4]               C[2,4]
NUMPY:
[[1, 2, 3]       [[1, 2, 3, 4]         [[38, 44, 50,  56]
 [4, 5, 6]]       [5, 6, 7, 8]          [83, 98, 113,128]]
                  [9,10,11,12]]
JULIA:
change_input_to_row_major:
1.inputs read numpy data from memory:
[[1, 3, 5]       [[1, 4, 7,10]
 [2, 4, 6]]       [2, 5, 8,11]
                  [3, 6, 9,12]]
2.inputs after reshape(reverse(shape)):
[[1, 4]          [[1, 5, 9]
 [2, 5]           [2, 6,10]
 [3, 6]]          [3, 7,11]
                  [4, 8,12]]
3.inputs after transpose/permutedims:
[[1, 2, 3]       [[1, 2, 3, 4]         [[38, 44, 50,  56]
 [4, 5, 6]]       [5, 6, 7, 8]          [83, 98, 113,128]]
                  [9,10,11,12]]
change_output_to_row_major:
1.output after transpose/permutedims:
                                       [[38, 83]
                                        [44, 98]
                                        [50,113]
                                        [56,128]
2.output after reshape:
                                       [[38, 50, 83, 113]
                                        [44, 56, 98, 128]]
3.output read numpy data from memory:
                                       [[38, 44, 50,  56]
                                        [83, 98,113, 128]]
=#
function foo!(x, y, z)
    x = change_input_to_row_major(x)
    y = change_input_to_row_major(y)
    z .= gemm(x, y, z)
    z .= change_output_to_row_major(z)
end
```

### Defining Custom Operator of akg Type

The custom operator of akg type uses the [MindSpore AKG](https://gitee.com/mindspore/akg) operator DSL to describe the internal calculation logic of the operator. MindSpore AKG is an operator development and compilation framework based on TVM(Tensor Virtual Machine) and Polyhedral technology, and it supports multiple types of operator DSL, such as Hybrid, IR builder and TVM compute.

Operator output shape and data type inference can be realized by defining Python functions to describe the inference logic of operator output shape and data type.

If the operator contains attributes or only supports specific input and output data types or data formats, operator information needs to be registered, and for how to generate operator information, see [Registering the Operator Information](#registering-the-operator-information). If the operator information is not registered, when operator selection and mapping are made in the backend, the operator information is derived from the input of the current operator.

Takes test_custom_akg.py as an example of how to define a custom operator of akg type, where the operator computes the sum of two tensors.

Here is the content of test_custom_akg.py:

```python
import numpy as np
from mindspore import set_context, Tensor
import mindspore.ops as ops

set_context(device_target="GPU")

# Operator implementation, Hybrid DSL
def add(a, b):
    c = output_tensor(a.shape, a.dtype)
    for i0 in range(a.shape[0]):
        for i1 in range(a.shape[1]):
            c[i0, i1] = a[i0, i1] + b[i0, i1]
    return c

if __name__ == "__main__":
    # Define a custom operator of akg type
    op = ops.Custom(add, out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="akg")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

The following points need to be explained in this example:

- `set_context(device_target="GPU")` indicates that the operator runs on the GPU platform. To run on the Ascend platform, please compile an Ascend version of MindSpore and set the value of device_target to "Ascend".
- Use Python lambda functions to infer the output shape and data type, and pass them to the `out_shape` and `out_dtype` parameters of the `Custom` primitive. In this example, the lambda function indicates that the output shape and data type are the same as the information of the first input tensor.
- The operator information is not registered, so the operator information of the custom operator will be inferred from the inputs.

Execute case:

```bash
python test_custom_akg.py
```

The execution result is as follows:

```text
[[2. 2.]
 [4. 4.]]
```

## Advanced Usage

### Registering the Operator Information

The operator information describes the supported inputs and outputs data type, the supported inputs and outputs format, attributes, and target (platform information) of the operator implementation. It is used to select and map operators by the backend. The operator information can be defined by using the [CustomRegOp](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.CustomRegOp.html#mindspore-ops-customregop) API, then you can use the [custom_info_register](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.custom_info_register.html#mindspore-ops-custom-info-register) decorator or just pass it to the `reg_info` parameter of [Custom](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom) primitive to bind the information to the operator implementation. The operator information will be registered to the operator information library on the MindSpore C++ side at last. The `reg_info` parameter takes higher priority than the `custom_info_register` decorator.

The target value in operator information can be "Ascend", "GPU" or "CPU", which describes the operator information on a specific target. For the same operator implementation, it may have different supported data types on different targets, so you can use the target value in operator information to differ this. The operator information on a specific target will be registered only once.

> - The numbers and sequences of the input and output information defined in the operator information must be the same as those in the parameters of the operator implementation.
> - For the custom operator of akg type, if the operator has attributes, you need to register operator information, The attribute name in the operator information must be consistent with the attribute name used in the operator implementation. For the custom operator of tbe type, you need to register operator information. For the custom operator of aot type, since the operator implementation needs to be compiled into a dynamic library in advance, the decorator will not work, and the operator information can only be passed in through the `reg_info` parameter.
> - If the custom operator only supports a specific input and output data type or data format, the operator information needs to be registered so that the data type and data format can be checked when the operator is selected in the backend. For the case where the operator information is not provided, the information will be derived from the inputs of the current operator.

### Defining the bprop Function for Operators

If an operator needs to support automatic differentiation, the backpropagation(bprop) function needs to be defined first and then passed to the `bprop` parameter of `Custom` primitive. In the bprop function, you need to describe the backward computation logic that uses the forward input, forward output, and output gradients to obtain the input gradients. The backward computation logic can be composed of built-in operators or custom backward operators.

Note the following points when defining the bprop function for operators:

- The input parameter sequence of the bprop function is the forward input, forward output, and output gradients. For a multi-output operator, the forward output and output gradients are provided in the form of tuples.
- The return value of the bprop function is tuples consisting of input gradients. The sequence of elements in a tuple is the same as that of the forward input parameters. Even if there is only one input gradient, the return value must be a tuple.

Take test_grad.py as an example to show the usage of the backpropagation function:

```python
import numpy as np
from mindspore import set_context, Tensor, GRAPH_MODE
from mindspore.nn import Cell
import mindspore.ops as ops

set_context(mode=GRAPH_MODE, device_target="GPU")

# Forward computation of custom operator
def square(x):
    y = output_tensor(x.shape, x.dtype)
    for i0 in range(x.shape[0]):
        y[i0] = y[i0] * y[i0]
    return y

# Backward computation of custom operator
def square_grad(x, dout):
    dx = output_tensor(x.shape, x.dtype)
    for i0 in range(x.shape[0]):
        dx[i0] = 2.0 * x[i0]
    for i0 in range(x.shape[0]):
        dx[i0] = dx[i0] * dout[i0]
    return dx

# Backpropagation function
def bprop():
    op = ops.Custom(square_grad, lambda x, _: x, lambda x, _: x, func_type="akg")

    def custom_bprop(x, out, dout):
        dx = op(x, dout)
        return (dx,)

    return custom_bprop

class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        # Define a custom operator of akg type and provide a backpropagation function
        self.op = ops.Custom(square, lambda x: x, lambda x: x, bprop=bprop(), func_type="akg")

    def construct(self, x):
        return self.op(x)

if __name__ == "__main__":
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    sens = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    dx = ops.GradOperation(sens_param=True)(Net())(Tensor(x), Tensor(sens))
    print(dx)
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

> More examples can be found in the MindSpore source code [tests/st/ops/graph_kernel/custom](https://gitee.com/mindspore/mindspore/tree/master/tests/st/ops/graph_kernel/custom).

### MindSpore Hybrid Syntax Specification

The syntax of MindSpore Hybrid DSL is similar to Python syntax, such as function definitions, indentation, and annotations. Functions written in MindSpore Hybrid DSL can be used as ordinary `numpy` functions after adding ms_hybrid decorators, or they can be custom operators used for Custom.

```python
import numpy as np
from mindspore import ops, Tensor
from mindspore.ops import ms_hybrid

@ms_hybrid
def outer_product(a, b):
    d = allocate(a.shape, a.dtype)
    c = output_tensor(a.shape, a.dtype)

    for i0 in range(a.shape[0]):
        for i1 in range(b.shape[1]):
            c[i0, i1] = 0.0
            for i2 in range(a.shape[1]):
                d[i0, i2] = 2 * a[i0, i2]
                c[i0, i1] = c[i0, i1] + sin(d[i0, i2] * b[i2, i1])
    return c

np_x = np.random.normal(0, 1, [4, 4]).astype(np.float32)
np_y = np.random.normal(0, 1, [4, 4]).astype(np.float32)

print(outer_product(np_x, np_y))

input_x = Tensor(np_x)
input_y = Tensor(np_y)

test_op_akg = ops.Custom(outer_product)
out = test_op_akg(input_x, input_y)
print(out)
```

The detailed syntax specification of MindSpore Hybrid DSL is as follows.

#### Variables

Variables in MindSpore Hybrid DSL includes two formats Tensor and Scalar.

Tensor variables, besides those in the inputs of the function, must be declared with `shape`和 `dtype` before use.

- declare a output tensor by `output_tensor`, such as `output_tensor(shape, dtype)`.
- declare an intermediate tensor by `allocate`, such as `allocate(shape, dtype)`.

Example of Tensor allocation:

```python
@ms_hybrid
def kernel_func(a, b):
    # We can use a and b directly as input tensors

    # d is a tensor with dtype fp16 and shape (2,), and will be used as an intermediate variable in the following code
    d = allocate((2,), "float16")
    # c is a tensor with same dtype and shape as a, and will be used as a output function in the following code
    c = output_tensor(a.shape, b.dtype)

    # assign value to c by d as the intermediate variable
    d[0] = b[0, 0]
    for i in range(4):
        for j in range(4):
            c[i, j] = d[0]

    # c as output
    return c
```

Scalar variables will regard its first assignment as the declaration. The assignment can be either a number or an expression. The place of the first assignment of a scalar variable defines its scope, such as inside a certain level of for loop. Using the variable outside its scope will lead to error.

Example of using Scalar variable:

```python
@ms_hybrid
def kernel_func(a, b):
    c = output_tensor(a.shape, a.dtype)

    for i in range(10): # i loop
        for j in range(5): # j loop
            # assign a number to Scalar d
            d = 2.0
            # assign an expression to Scalar e
            e = a[i, j]
            # use scalars
            c[i, j] = d + e

    # Wrong: c[0, 0] = d
    # Can't use Scalar d outside its scope (j loop)
    return c
```

Unlike native Python language, once a variable is defined, we can't change its `shape`和 `dtype`.

#### Expressions

MindSpore Hybrid DSL supports basic math operators, including `+, -, *, /`, as well as self-assign operators, including `=, +=, -=, *=, /=`.
Users can write codes like writing Python expressions.

**All the expressions must be based on scalars. Computation for the tensors must include all indices, such as `C[i, j] = A[i, j] + B[i, j]`. Currently, tensorized codes such as `C = A + B` are not supported.**

When writing assignment expressions, users must take care of the dtype of the expression and make them consistent on both sides of the equality. Otherwise, the error might be thrown on the stage of **operator compilation**. Any integer numbers in the expression will be treated as int32, while float numbers will be treated as float32. There is no implicit dtype casting in MindSpore Hybrid DSL, and all dtype casting must be written with dtype names as casting functions, including:

- int32
- float16
- float32
- (only on gpu backend) int8, int16, int64, float64

Example of dtype casting:

```python
@ms_hybrid
def kernel_func(a):
    c = output_tensor((2,), "float16")

    # Wrong: c[0] = 0.1 c's dtype is fp16, while 0.1's dtype is fp32
    c[0] = float16(0.1) # float16(0.1) cast the number 0.1 to dtype fp16
    c[1] = float16(a[0, 0]) # float16(a[0, 0]) cast the number 0.1 to dtype fp16
    return c
```

#### Loop

Currently, only the `for` loop is supported. `while`, `break`, and `continue` are illegal in MindSpore Hybrid DSL.

Loops are the same as those in Python. `range` and `grid` are supported to express extents of loops. `range` is for one-dimensional loops and accept a number as the upper bound of the loop, such as:

```python
@ms_hybrid
def kernel_func(a, b):
    c = output_tensor((3, 4, 5), "float16")

    for i in range(3):
        for j in range(4):
            for k in range(5):
                out[i, j, k] = a[i, j, k] + b[i, j, k]
    return  c
```

The iteration space of the above loops is `0 <= i < 3, 0 <= j < 4, 0 <= k < 5`.

`grid` is for multi-dimensional loops and accepts `tuple` as its input. For example, the above code can be also written as follows in `grid`:

```python
@ms_hybrid
def kernel_func(a, b):
    c = output_tensor((3, 4, 5), "float16")

    for arg in grid((4,5,6)):
        out[arg] = a[arg] + b[arg]
    return  c
```

Right now `arg` is equivalent to a three dimensional index `(i,j,k)`, with upper bound 4, 5, 6 respectively. We also have access to each element in `arg`, such as:

```python
@ms_hybrid
def kernel_func(a, b):
    c = output_tensor(a.shape, "float16")

    for arg in grid(a.shape):
        out[arg] = a[arg] + b[arg[0]]
    return  c
```

Then the expression inside loops is equivalent to `out[i, j, k] = a[i, j, k] + b[i]`.

#### Attribute

Current we support only tensor's `shape` and `dtype` attributes, such as `a.shape`, and `c.dtype`.

The shape attribute of a Tensor is a `tuple`. We have access to its element with a **fixed** index, such as `a.shape[0]`.

Once `grid` accepts one Tensor's `shape` attribute as its input, then the dimension of the loops is the same as the dimension of the Tensor. For example:

```python
@ms_hybrid
def kernel_func(a, b):
    c = output_tensor(a.shape, "float16")

    for arg in grid(a.shape):
        out[arg] = a[arg] + b[arg[0]]
    return  c
```

If a is a two dimensional tensor, then the expression inside loops is equivalent to `out[i, j] = a[i, j] + b[i]`, while if a is a three dimensional tensor, then the expression inside loops is equivalent to `out[i, j, k] = a[i, j, k] + b[i]`.

#### Keywords

Currently, we support keywords including:

- Math keywords (all platform): `log`, `exp`, `sqrt`, `tanh`, `power`, `floor`
- Memory allocation: `allocate`, `output_tensor`
- Datatype keywords: `int32`, `float16`, `float32`, `float64`
- For keywords: `for`, `range`, `grid`
- In current version, some CPU/GPU platform only keywords:
    - Math keywords: `rsqrt`, `erf`, `isnan`, `sin`, `cos`, `isinf`, `isfinite`, `atan`, `atan2` (only on GPU), `expm1` (only on GPU), `floor`, `ceil`, `trunc`, `round`, `ceil_div`
    - Datatype keywords: `int8`, `int16`, `int64`

#### Frequent Error Messages and Error Attributions

To help users effectively develop and locate bugs, MindSpore Hybrid DSL provides the following error messages, including:

- TypeError: There are Python keywords such as `while`, `break` and `continue` which are not supported by MindSpore Hybrid DSL.
- ValueError:
    - There are built-in function names which are not in the above support list;
    - Take properties that are not `shape` or `dtype` for tensors.
- Other frequent error message:
    - "SyntaxError": DSL does not conform to the Python syntax(not the syntax defined by MindSpore Hybrid DSL), and is reported by the Python interpreter itself;
    - "ValueError: Compile error" and "The pointer\[kernel_mod\] is null": the kernel compiler fails in compiling the DSL. Check error messages from AKG for further information;
    - "Launch graph failed": The compiled kernel fails in running. Check the error message from the hardware. For example, when the kernel fails in Ascend, there will be an "Ascend error occurred" message and corresponding hareware error messages.
