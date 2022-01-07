# Custom Operators (Custom based)

`Ascend` `GPU` `CPU` `Model Development`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/custom_operator_custom.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

When built-in operators cannot meet requirements during network development, you can call the Python API [Custom](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom) primitive defined in MindSpore to quickly create different types of custom operators for use.

Traditional methods to add a custom operator need three steps: defining the operator primitive, implementing the operator, and registering the operator information.

The related concepts are as follows:

- Operator primitive: defines the frontend API prototype of an operator on the network. It is the basic unit for forming a network model and includes the operator name, attribute (optional), input and output names, output shape inference method, and output data type inference method.
- Operator implementation: defines a Python function(Ascend custom operators) or a C++ class(GPU and CPU custom operators), which describes the implementation of the internal computation logic of an operator.
- Operator information: describes basic information about an operator, such as the operator name, supported input and output data types, supported input and output data formats, and attributes. It is the basis for the backend to select and map operators.

Compared with traditional custom operator creating methods, creating custom operators based on `Custom` primitive has several advantages:

- Different custom operators use the same `Custom` primitive, there is no need to define a primitive for every operator. The above three parts of work can be implemented in a network script with a unified way, and used as part of the network expression, there is no need to modify and recompile the source codes of MindSpore.
- It unifies the interface and usage for different kinds of custom operators, which is convenient for network developers to flexibly choose which kind of custom operator to use according to their needs.
- Supports defining custom operators with hybrid expression, which can be used across platforms.

## Basic Usage

The supported custom operator defining methods based on the [Custom](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom) primitive include: akg, tbe, aot and pyfunc.

The difference of these operator defining methods are as follows:

| Defining Methods | Development Language | Compilation Method | Supported Platforms | Recommended Scenarios |
| :------: | :------: | :------: | ------ | ------ |
| akg | MindSpore AKG DSL | JIT | `Ascend` `GPU` | Ascend/GPU platform general scenarios |
| tbe | TBE DSL | JIT | `Ascend` | Ascend platform scenarios |
| aot | C/C++/CUDA | AOT | `GPU` `CPU` | GPU/CPU platform high-performance scenarios |
| pyfunc | Python | JIT | `CPU` | Fast algorithm verification, need to interact with Python and other scenarios |

> - The full name of DSL is Domain Specific Language.
> - AOT(Ahead Of Time) compiling means the operator implementation needs to be compiled into a dynamic library in advance, and then automatically called by the framework when the network is running. JIT(Just In Time) compiling does not need to compile the operator implementation in advance, the operator implementation will be directly called by the framework during network compilation or runtime.

Different custom operator defining methods use different development language to implement the operator, but the development process is the same, including operator implementation, operator output shape and data type inference, and operator information registration (optional). You can choose which one to use base on needs. The defining methods of these custom operators will be introduced here, and examples are provided for each method.

> More examples can be found in the MindSpore source code [tests/st/ops/graph_kernel/custom](https://gitee.com/mindspore/mindspore/tree/master/tests/st/ops/graph_kernel/custom).

### Defining Custom Operator of akg Type

The custom operator of akg type uses the [MindSpore AKG](https://gitee.com/mindspore/akg) operator DSL to describe the internal calculation logic of the operator. MindSpore AKG is an operator development and compilation framework based on TVM(Tensor Virtual Machine) and Polyhedral technology, it supports multiple types of operator DSL, such as Hybrid, IR builder and TVM compute.

Operator output shape and data type inference can be realized by defining Python functions to describe the inference logic.

If the operator has attributes or only supports specific input and output data types or data formats, the operator information needs to be registered. For the creation of operator information, please refer to [Registering the Operator Information](#registering-the-operator-information). If the operator information is not registered, then the operator information will be derived from the inputs of the current operator during operator selection process.

Takes test_custom_akg.py as an example to introduce how to define a custom operator of akg type, where the custom operator implements the function of adding two input tensors.

Here is the content of test_custom_akg.py:

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(device_target="GPU")

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

- `context.set_context(device_target="GPU")` indicates that the operator runs on the GPU platform. To run on the Ascend platform, please compile an Ascend version of MindSpore and set the value of device_target to "Ascend".
- Use Python lambda functions to infer the output shape and data type, and pass them to the `out_shape` and `out_dtype` parameters of the `Custom` primitive. In this example, the lambda function indicates that the output shape and data type are the same as the information of the first input tensor.
- The operator information is not registered, so the operator information of the custom operator will be inferred from the inputs.

Running case:

```bash
python test_custom_akg.py
```

Running results:

```text
[[2. 2.]
 [4. 4.]]
```

### Defining Custom Operator of tbe Type

The custom operator of tbe type uses the TBE(Tensor Boost Engine) operator DSL to describe the internal calculation logic of the operator. You can refer the [TBE document](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_10_0063.html) for the implementation details.

Operator output shape and data type inference can be realized by defining Python functions to describe the inference logic.

Operator information needs to be registered. For the creation of operator information, please refer to [Registering the Operator Information](#registering-the-operator-information).

Takes test_custom_tbe.py as an example to introduce how to define a custom operator of tbe type, where the custom operator implements the function of adding two input tensors.

Here is the content of test_custom_tbe.py:

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp, custom_info_register

context.set_context(device_target="Ascend")

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
- Use `CustomRegOp` to create the the operator information and use `custom_info_register` decorator to register it.

Running case:

```bash
python test_custom_tbe.py
```

Running results:

```text
[[2. 2.]
 [4. 4.]]
```

### Defining Custom Operator of aot Type

The custom operator of aot type adopts the AOT compilation method, which requires network developers to hand-write the source code file of the operator implementation based on a specific interface, and compile the source code file into a dynamic library in advance, and then the framework will automatically call and run the function defined in the dynamic library. In terms of the development language of the operator implementation, the GPU platform supports CUDA, and the CPU platform supports C and C++. The interface specification of the operator implementation in the source file is as follows:

```cpp
extern "C" int func_name(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra);
```

Where, the function name `func_name` can be replaced with any valid function name. The return value is of type int, and 0 means normal exit, non-zero means an exception occurs. The meaning of the parameter list is as follows:

- nparam (int): The number of inputs and outputs. For example, if an operator has 2 inputs and 1 output, then the value of nparam is 3.
- params (void \*\*): An array of pointers, with each pointer pointing to the input or output data. For example, if an operator has 2 inputs and 1 output, then params[0] points to the first input data, params[1] points to the second input data, params[2] points to the output data.
- ndims (int \*): An array of integers, each integer represents for the dimensions of shape of input or output. For example, if params[i] is a tensor with shape [1024, 1024], then ndims[i] is 2.
- shapes (int64_t \*\*): An array of shapes, each element in array represents for the shape of input or output. For example, if params[i] is a tensor with shape [1024, 1024], then shapes[i][0] is 1024, shapes[i][1] is 1024.
- dtypes (const char \*\*): Array of data types, each element in array represents for the data type of input or output. The value of data type can be "float32", "float16", "float", "float64", "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", "bool".
- stream (void \*): Stream pointer, only used in cuda file.
- extra (void \*): Used for further extension.

Operator output shape and data type inference can be realized by defining Python functions to describe the inference logic.

If the operator only supports some specific input and output data types, then the operator information needs to be registered. For the creation of operator information, please refer to [Registering the Operator Information](#registering-the-operator-information).

The following examples introduces the development process of aot type custom operator on GPU platform and CPU platform, where the custom operator implements the function of adding two input tensors.

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
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(device_target="GPU")

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

Running case:

```bash
python test_custom_aot.py
```

Running results:

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
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(device_target="CPU")

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

Running case:

```bash
python test_custom_aot.py
```

Running results:

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
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(device_target="CPU")

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

Running case:

```bash
python test_custom_pyfunc.py
```

Running results:

```text
[[2. 2.]
 [4. 4.]]
```

## Advanced Usage

### Registering the Operator Information

The operator information describes the supported inputs and outputs data type, the supported inputs and outputs format, attributes, and target(platform information) of the operator implementation. It is used to select and map operators at later. The operator information can be defined by using the [CustomRegOp](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.CustomRegOp.html#mindspore-ops-customregop) API, then you can use the [custom_info_register](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.custom_info_register.html#mindspore-ops-custom-info-register) decorator or just pass it to the `reg_info` parameter of [Custom](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom) primitive to bind the information to the operator implementation. The operator information will be registered to the operator information library on the MindSpore C++ side at last. The `reg_info` parameter takes higher priority than `custom_info_register` decorator.

The target value in operator information can be "Ascend", "GPU" or "CPU". Which describes the operator information on a specific target. For the same operator implementation, it may have different supported data types on different targets, so you can use the target value in operator information to differ this. The operator information on a specific target will be registered only once.

> - The numbers and sequences of the input and output information defined in the operator information must be the same as those in the parameters of the operator implementation.
> - For the custom operator of akg type, if the operator has attributes, you need to register operator information, The attribute name in the operator information must be consistent with the attribute name used in the operator implementation. For the custom operator of tbe type, you need to register operator information. For the custom operator of aot type, since the operator implementation needs to be compiled into a dynamic library in advance, the decorator will not work, and the operator information can only be passed in through the `reg_info` parameter.
> - If the custom operator only supports a specific input and output data type or data format, the operator information needs to be registered so that the data type and data format can be checked when the operator is selected in the backend. For the case where the operator information is not provided, the information will be derived from the inputs of the current operator.

### Defining the bprop Function for Operators

If an operator needs to support automatic differentiation, the backpropagation(bprop) function needs to be defined first and then passed to the `bprop` parameter of `Custom` primitive. In the bprop function, you need to describe the backward computation logic that uses the forward input, forward output, and output gradients to obtain the input gradients. The backward computation logic can be composed of built-in operators or custom backward operators.

Note the following points when defining the bprop function:

- The input parameter sequence of the bprop function is the forward input, forward output, and output gradients. For a multi-output operator, the forward output and output gradients are provided in the form of tuples.
- The return value of the bprop function is tuples consisting of input gradients. The sequence of elements in a tuple is the same as that of the forward input parameters. Even if there is only one input gradient, the return value must be a tuple.

Take test_grad.py as an example to show the usage of backpropagation function:

```python
import numpy as np
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

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

Running case:

```bash
python test_grad.py
```

Running results:

```text
[ 2.  8. 18.]
```

> More examples can be found in the MindSpore source code [tests/st/ops/graph_kernel/custom](https://gitee.com/mindspore/mindspore/tree/master/tests/st/ops/graph_kernel/custom).
