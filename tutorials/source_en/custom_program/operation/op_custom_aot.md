# AOT-Type Custom Operators(CPU/GPU)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/custom_program/operation/op_custom_aot.md)

## Overview

AOT (Ahead-Of-Time) type of custom operators employ a pre-compilation approach, which requires network developers to manually write the source code files corresponding to the operator implementation functions based on specific interfaces. These source code files need to be compiled into dynamic link libraries (DLLs) in advance. During network runtime, the framework will automatically invoke and execute the functions contained within these dynamic link libraries. AOT-type custom operators support the CUDA language for GPU platforms and the C and C++ languages for CPU platforms. For the development of custom operators specifically on the Ascend platform, please refer to [AOT-Type Custom Operators(Ascend)](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/custom_program/operation/op_custom_ascendc.html).

In this tutorial, we provide several simple use cases of AOT-type custom operators on both CPU and GPU platforms as demonstrations. For more comprehensive examples of AOT-type custom operators, please refer to the [examples](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/tests/st/graph_kernel/custom/test_custom_aot.py) section in the MindSpore source code.

## The Introduction to the General Usage Features of AOT-type Custom Operators

The custom operator of AOT-type adopts the AOT compilation method, which requires network developers to hand-write the source code file of the operator implementation based on a specific interface and compiles the source code file into a dynamic library in advance, and then the framework will automatically call and run the function defined in the dynamic library. In terms of the development language of the operator implementation, the GPU platform supports CUDA, and the CPU platform supports C and C++. The interface specification of the operator implementation in the source file is as follows:

```text
extern "C" int CustomFunc(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra);
```

where the function name `func_name` can be replaced with any valid function name. The return value is of type int. 0 means normal exit, and non-zero means an exception occurs. The meaning of the parameter list is as follows:

- nparam (int): The number of inputs and outputs. For example, if an operator has 2 inputs and 1 output, then the value of nparam is 3.
- params (void \*\*): An array of pointers, with each pointer pointing to the input or output data. For example, if an operator has 2 inputs and 1 output, then params[0] points to the first input data, params[1] points to the second input data, params[2] points to the output data.
- ndims (int \*): An array of integers, each integer represents the dimensions of the shape of input or output. For example, if params[i] is a tensor with shape [1024, 1024], then ndims[i] is 2.
- shapes (int64_t \*\*): An array of shapes, each element in array represents for the shape of input or output. For example, if params[i] is a tensor with shape [1024, 1024], then shapes[i][0] is 1024, shapes[i][1] is 1024.
- dtypes (const char \*\*): Array of data types, each element in array represents for the data type of input or output. The value of data type can be "float32", "float16", "float", "float64", "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", "bool".
- stream (void \*): Stream pointer, only used in Cuda file.
- extra (void \*): Used for further extension.

In the Python script, the format for the `func` input in `Custom` is `Path_To_Func:CustomFunc`, where `CustomFunc` is the name of the function above and `Path_To_Func` is the path to the corresponding function source file or binary library.

> - MindSpore recognizes the automatic compilation through the file name suffix. To use the automatic compilation feature, please use source files with suffixes such as `cpp`, `cc`, or `cu`. Otherwise, MindSpore will treat it as a binary library path;
> - To prevent malicious third-party libraries from tampering, please set the environment variable `MS_CUSTOM_AOT_WHITE_LIST` to the path of legitimate third-party libraries. Only files in the directories specified by `MS_CUSTOM_AOT_WHITE_LIST` and its subdirectories will be called by custom operators.

Operator output shape and data type inference can be realized by defining Python functions to describe the inference logic.

If the operator only supports some specific input and output data types, the operator information needs to be registered. For the creation of operator information, please refer to [Registering the Operator Information](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/custom_program/operation/op_custom_adv.html#registering-the-operator-information).

The following examples introduce the development process of AOT-type custom operator on GPU platform and CPU platform, where the custom operator implements the function of adding two input tensors.

### A GPU Example

Use the CUDA language to write the source file add.cu for the operator implementation:

```c++
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

```shell
nvcc --shared -Xcompiler -fPIC -o add.so add.cu
```

Write the test case test_custom_aot.py:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device(device_target="GPU")

if __name__ == "__main__":
    # Define a custom operator of aot type
    op = ops.Custom("./add.so:CustomAdd", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="aot")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(ms.Tensor(x0), ms.Tensor(x1))
    print(output)
```

The following points need to be explained in this example:

- In this example, you need to place test_custom_aot.py and add.so in the same directory. If add.so is in another directory, you need to replace the value of the first parameter of `Custom` primitive with the absolute path of add.so.
- Use Python lambda functions to infer the output shape and data type, and pass them to the `out_shape` and `out_dtype` parameters of the `Custom` primitive. In this example, the lambda function indicates that the output shape and data type are the same as the information of the first input tensor.
- The operator information is not registered, so the operator information of the custom operator will be inferred from the inputs.

Execute case:

```shell
python test_custom_aot.py
```

The execution result is as follows:

```text
[[2. 2.]
 [4. 4.]]
```

### A CPU Example

Use C/C++ language to write the source file add.cc for the operator implementation:

```c++
#include <string.h>
using size_t = decltype(sizeof(int));
using int64_t = decltype(sizeof(long));

extern "C" int CustomAdd(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra) {
  if (nparam != 3) return 1;
  float *input1 = static_cast<float *>(params[0]);
  float *input2 = static_cast<float *>(params[1]);
  float *output = static_cast<float *>(params[2]);
  size_t size = 1;
  for (int i = 0; i < ndims[2]; i++) {
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

```shell
g++ --shared -fPIC -o add.so add.cc
```

Write the test case test_custom_aot.py:

```text
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device(device_target="CPU")

if __name__ == "__main__":
    # Define a custom operator of aot type
    op = ops.Custom("./add.so:CustomAdd", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="aot")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(ms.Tensor(x0), ms.Tensor(x1))
    print(output)
```

The following points need to be explained in this example:

- In this example, you need to place test_custom_aot.py and add.so in the same directory. If add.so is in another directory, you need to replace the value of the first parameter of `Custom` primitive with the absolute path of add.so.
- Use Python lambda functions to infer the output shape and data type, and pass them to the `out_shape` and `out_dtype` parameters of the `Custom` primitive. In this example, the lambda function indicates that the output shape and data type are the same as the information of the first input tensor.
- The operator information is not registered, so the operator information of the custom operator will be inferred from the inputs.

Execute case:

```shell
python test_custom_aot.py
```

The execution result is as follows:

```text
[[2. 2.]
 [4. 4.]]
```

## The Introduction to the Advanced Usage Features of AOT-type Custom Operators

In the rest of tutorial, we will demonstrate advanced features of AOT-type custom operators, including:

- Auto-compilation of AOT-type custom operators;
- Attributes and intermediate variables of AOT-type custom operators;
- Dynamic shape support for AOT-type custom operators.

For the complete source code of the example, check [here](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/tests/st/graph_kernel/custom/test_custom_aot_fused.py) in the MindSpore source code.

### Auto-compilation of AOT-type Custom Operators

When the user's AOT-type custom operator file is a single file and does not require custom compilation options during compilation, users can use the automatic compilation feature.
In this way, users will provide the source file for the implementation of the custom operator, and MindSpore will automatically compile the source file into a binary library.
Currently, this function supports C++ file compilation based on GCC and CUDA file compilation based on NVCC. When using the automatic compilation function, there are several points to note:

- MindSpore recognizes the method of automatic compilation as a file name suffix. In order to use the auto compilation feature, please use a source file with a suffix of `cpp`, `cc`, or `cu`. In other cases, MindSpore will process as a binary library path.
- The result of automatic compilation is in the folder akg_kernel_meta.
- The default compilation options are:
    - C++: `g++ -std=c++17 --shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I./ -o $object_path, $source_path`
    - CUDA 10: `nvcc --shared -Xcompiler -fPIC -O3 -gencode arch=compute_70, code=sm_70 --use_fast_math --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=0 -I./ -o $object_path, $source_path`
    - CUDA 11(or higher version): `nvcc --shared -Xcompiler -fPIC -O3 -gencode arch=compute_80, code=sm_80 --use_fast_math --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=0 -I./ -o $object_path, $source_path`
- MindSpore requires the compilation option of `-D_ GLIBCXX_ USE_ CXX11_ ABI = 0`, so please avoid using a CUDA software stack with a version lower than 10.1.168 on GPU platforms.

### Attributes and Intermediate Variables of AOT-type Custom Operators

Many commonly used operators have attributes, such as the kernel size, padding, and strides of the convlution operator.
Operators with different attribute values have the same computational logic, with the only difference being the values of the attributes during initialization.
In addition, during the calculation process of the operator, some additional memory spaces may be needed to store the intermediate variables.
The following formula is an example. Use `input_1` and `input_2` to calculate `output`:

```python
tmp = Add(input_1, input_2)
output = ReduceSum(tmp, axis, keep_dims)
```

Here, we need to add the following intermediate variables and attributes to the operator in the computation function, including:

- `tmp` as an intermediate variable to record the intermediate result of addition;
- `axis` as an attribute of type `int`, and `keep_dims` as an attribute of type `bool`.

AOT-type custom operators provide functionality to add attributes, and then we can define a class of custom operators with a single source code.
These operators have the same computational logic but achieve different computational effects by assigning values to the attributes during operator initialization.
Additionally, to allow MindSpore to manage memory allocation and release, AOT-type custom operators provide interfaces to specify the size of intermediate variables, allowing MindSpore to allocate memory for computation.

### Dynamic Shape Support for AOT-type Custom Operators

Dynamic Shape refers to that the shapes of inputs or outputs of an operator depends on the specific operation and cannot be calculated in advance at compile time.
Specifically, there are two cases: the shapes of the operator's inputs are unknown at compile time, and the shapes of the operator's outputs depend on the specific input values.
The case that the shapes of the operator's inputs are unknown at compile time is more common.
Any operator, regardless of their own calculation logic, needs to support this case if it is used in a network that supports dynamic shape inputs.

Currently, the AOT-type custom operators support the dynamic shape scenario when the shape of the operator's input is unknown at compile time.
This is achieved by defining a C++ version of the shape derivation function to support type derivation for custom operators in this scenario.

It should be noted that custom operators do not yet support dynamic shape scenarios where the shape of the operator output depends on the value of a specific input.

## The Introduction AOT-type Custom Operator Advanced Usage Interface

### Main Function

In the source code file, the main function of the operator implementation function must follow the following specifications:

```cpp
extern "C" int FuncName(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra);
```

The function name `FuncName` can be replaced with any valid function name. The return value is of type int, with 0 indicating normal exit and non-zero indicating an exception. The meaning of the parameter list is as follows:

- nparam (int): The total number of inputs, outputs, and intermediate variables. For example, if the operator has 2 inputs, 1 output, and 1 intermediate variable, then `nparam` is 4.
- params (void \*\*): An array of pointers to inputs, outputs, and intermediate variables. For example, if the operator has 2 inputs, 1 output, and 1 intermediate variable, then `params[0]` points to the memory of the first input data, `params[1]` points to the memory of the second input data, `params[2]` points to the memory of the output data, and `params[3]` points to the memory of the intermediate variable.
- ndims (int \*): An array of dimensions for inputs, output,s and intermediate variables. For example, if `params[i]` is a tensor with shape [1024, 1024], then `ndims[i]` is 2.
- shapes (int64_t \*\*): An array of shapes for inputs, outputs, and intermediate variables. For example, if `params[i]` is a tensor with shape [1024, 1024], then `shapes[i][0]` is 1024 and `shapes[i][1]` is 1024.
- dtypes (const char \*\*): An array of data types for inputs, outputs, and intermediate variables. The elements in `dtypes` can take values among the list "float32", "float16", "float", "float64", "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", and "bool".
- stream (void \*): The pointer to a CUDA stream, only required for GPU operator implementation.
- extra_void (void \*): The pointer to a data structure related to attributes.

### Initialization Function

To support operator attributes and intermediate variables, we need to define an operator initialization function. The definition of the operator initialization function must follow the following specifications:

```cpp
extern "C" int FuncNameInit(int *ndims, int64_t **shapes, const char **dtypes, AotExtra *extra);
```

The function name `FuncName` is the name of the operator main function. The return value is of type int, with 0 indicating normal exit and non-zero indicating an exception. The meaning of the parameter list is as follows:

- ndims (int \*): Array of dimensions for input and output shapes.
- shapes (int64_t \*\*): Array of shapes for inputs and outputs.
- dtypes (const char \*\*): Array of data types for inputs and outputs.
- extra (AotExtra \*): Custom operator extensions with attributes. The `AotExtra` type is defined in the header file [custom_aot_extra.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/tests/st/graph_kernel/custom/aot_test_files/custom_aot_extra.h) provided by MindSpore.

### Shape Inference Function

To support dynamic shape, a C++ version of the shape inference function needs to be added to the custom operator of Aot type. The definition of the operator shape inference function must meet the following specifications:

```cpp
extern "C" std::vector<int64_t> FuncNameInferShape(int *ndims, int64_t **shapes, AotExtra *extra)
```

The function name `FuncName` is the name of the operator main function.
The return value is of type `std::vector<int64_t>` and represents the output shape.
The meaning of the parameter list is as follows:

- `ndims` (int \*): Array of dimensions for input shapes.
- `shapes` (int64_t \*\*): Array of shapes for inputs.
- `extra` (AotExtra \*): Pointer to an extension for attribute-bearing custom operators. The `AotExtra` type is defined in the header file [custom_aot_extra.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/tests/st/graph_kernel/custom/aot_test_files/custom_aot_extra.h) provided by MindSpore.

### Type Inference Function

The AOT custom operators also support type inference functions in C++ version. The definition of the operator's type inference function must meet the following specifications:

```cpp
extern "C" TypeId FuncNameInferType(std::vector<TypeId> type_ids, AotExtra *extra)
```

In this function, the name FuncName is the name of the main operator function. The return type is TypeId, which represents the output type. The meanings of the parameters are as follows:

- `type_ids` (std::vector<TypeId>): An array of input types.
- `extra` (AotExtra *): Used for extending custom operators with attributes, consistent with the input parameters of the shape inference function.

### Operator Attribute Registration (Python)

The initialization of operator attributes is implemented through the operator registration function. For each attribute, we create an `attr` for the operator registration file, setting the attribute name and value. The registration function is as follows:

```python
def attr(self, name=None, param_type=None, value_type=None, default_value=None, **kwargs)
```

Please refer to the [CustomRegOp](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.CustomRegOp.html#mindspore-ops-customregop) interface documentation for the meaning of each parameter. When registering a custom operator of Aot type, we set the following four parameters:

- `name`: the name of the attribute of the AOT-type custom operator;
- `param_type`: the parameter type of the attribute. For attributes of AOT-type custom operators, this input is fixed to be "required", which means it is a required parameter;
- `value_type`: the numerical type of the attribute. For attributes of AOT-type custom operators, this input can be a specific numerical type or "all", which means no restrictions on the type;
- The last input needs to specify the input name as `value=`, and the input value is the value of the attribute.

## Advanced Usage Example of AOT-type Custom Operator

Now we introduce the advanced usage of custom Aot operators using an example of a fused Add and ReduceSum operator. The operator first adds two inputs, and then performs sum operation along a certain axis. The basic calculation logic is as follows:

```python
tmp = Add(input_1, input_2)
output = ReduceSum(tmp, axis, keep_dims)
```

Here, we need to add the following intermediate variables and attributes in the computation function, including:

- `tmp` is an intermediate variable that records the intermediate result of the addition;
- `axis` is a property of type `int`, and `keep_dims` is a property of type `bool`.

### Operator Implementation File (C++/CUDA): kernel.cc

To implement the operator, we create a source file named `kernel.cc`, which includes an operator attribute class `add_reduce_kernel_attr` and three functions: `CustomKernelInit`, `CustomKernelInferShape`, and `CustomKernel`.

#### Operator Attribute Class

First, we define a data structure to store operator attributes, which inherits from `AotKernelData`.
`AotKernelData` is the base class for custom operator attribute data structures.
By downloading the header file [custom_aot_extra.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/tests/st/graph_kernel/custom/aot_test_files/custom_aot_extra.h) provided by MindSpore and placing it in the same directory as the source file, we can use the related interfaces by including it with `#include "custom_aot_extra.h"` at the beginning of the file.

```c++
#include <vector>
#include "custom_aot_extra.h"
class add_reduce_kernel_attr : public AotKernelData {
 public:
  int64_t axis;
  bool keep_dim;
};
```

Here, we define the following variables in the attribute class `add_kernel`:

- `axis` : member variable, type is `int64_t`;
- `keep_dim` : member variable, type is `bool`.

#### Operator Initialization Function

After defining the operator attribute class, we define the operator initialization function. Notice that the initialization function name here is `CustomKernelInit`, and the corresponding prefix for the following functions should be `CustomKernel`.

```c++

extern "C" int CustomKernelInit(int *ndims, int64_t **shapes, const char **dtypes, AotExtra *extra) {
  size_t workspace_size = 1;
  for (size_t i = 0; i < ndims[0]; i++) {
    workspace_size *= shapes[0][i];
  }

  std::vector<size_t> workspace = {workspace_size * sizeof(float)};
  extra->SetWorkSpace(workspace);

  add_reduce_kernel_attr *kernel_data_ptr = new add_reduce_kernel_attr;
  kernel_data_ptr->axis = extra->Attr<int64_t>("axis");
  kernel_data_ptr->keep_dim = extra->Attr<bool>("keep_dim");
  extra->SetKernelData(kernel_data_ptr);
  return 0;
}
```

Here, we need a intermediate variable `workspace` to record the intermediate result of addition. The method is as follows:

1. Calculate the memory size required for `workspace`: Since the size of `workspace` is the same as that of the first input, we multiply the size of each dimension of `shapes[0]` to calculate the number of elements in `workspace`, and then multiply it by `sizeof(float)` to get the memory size (assuming the element type is float by default).
2. Store all the memory sizes of intermediate variables in a `std::vector<size_t>` object: `std::vector<size_t> workspace = {workspace_size * sizeof(float)};`. Here, since there is only one intermediate variable, the vector has only one element.
3. Set the memory size of the intermediate variable using the `SetWorkSpace` function of `AotExtra *extra`: `extra->SetWorkSpace(workspace)`.

In addition, we need to obtain the values of two attributes, `axis` and `keep_dim`, as follows:

1. Create a pointer to an `add_reduce_kernel_attr` object: `add_reduce_kernel_attr *kernel_ptr = new add_reduce_kernel_attr`.
2. Retrieve the attribute values from `extra` and store them in the member variables of `kernel_ptr`: `kernel_data_ptr->axis = extra->Attr<int64_t>("axis"); kernel_data_ptr->keep_dim = extra->Attr<bool>("keep_dim");`. Here, `reduce_axis` and `keep_dim` are of type `int` and `bool` respectively. We use the corresponding template function of `extra->Attr<T>(std::string name)` to obtain the value of the attribute with the given type.
    - The supported types for `T` in step 2 are `bool`, `string`, `int64_t`, `float`, `std::vector<int64_t>`, `std::vector<float>`, `std::vector<std::vector<int64_t>>`, and `std::vector<std::vector<float>>`.
3. Store `kernel_ptr` in `extra` for use during operator calculation: `extra->SetKernelData(kernel_ptr)`.

#### Operator Shape Inference Function

To define a dynamic shape scene, we define a C++ version of the operator shape inference function as follows. Notice that the operator shape inference function name is `CustomKernelInferShape`, and shares the same prefix `CustomKernel` with the initialization function name `CustomKernelInit`.

```c++
#include <vector>
#include "custom_aot_extra.h"

extern "C" std::vector<int64_t> CustomKernelInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
  const int64_t kDynRankSize = -2;

  if (shapes[0][0] == kDynRankSize) {
    return std::vector<int64_t>{shapes[0][0]};
  }
  int64_t axis = extra->Attr<int64_t>("axis");
  bool keep_dim = extra->Attr<bool>("keep_dim");
  if (keep_dim) {
    if (axis == 0) {
      return std::vector<int64_t>{1, shapes[0][1]};
    } else {
      return std::vector<int64_t>{shapes[0][0], 1};
    }
  } else {
    return std::vector<int64_t>{shapes[0][1 - axis]};
  }
}
```

In the above example, we need to note the following:

- According to the MindSpore specifications, dynamic shape inputs includes two cases: the dynamic shape case and the dynamic rank case, with corresponding shape inputs as follows:
    - the dynamic shape case: If the size of a certain dimension of the input is unknown, it is represented by -1. For example, the shape of the input is [1024, -1, 1024], which indicates that the input is a three-dimensional tensor with dimensions of 1024 and -1 for the second dimension;
    - the dynamic rank case: The number of dimensions of the input is unknown, and the shape of the input is fixed as [-2, ].
- To support C++ shape inference functions, we need to handle cases when inputs are either dynamic shape or dynamic rank. For example, in the above example, if the input is of dynamic rank, the output will also be of dynamic rank. Therefore, when we find that the input is [-2, ], we directly return [-2, ].
- For scenarios where the output shape depends on attributes, you can use the `extra->Attr<T>(std::string name)` template interface to obtain attributes.

#### Operator Computation Function (Main Function)

The interface specification of the operator computation function is the same as that of a custom operator without attributes.
It is worth noting that the operator main function name `CustomKernel` needs to be the same as the prefix of the initialization function name `CustomKernelInit` and the operator shape inference function name `CustomKernelInferShape` mentioned above.
The main function, together with the above two functions, forms the source file `kernel.cc`.

```c++
extern "C" int CustomKernel(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra_void) {
  constexpr int OUTPUT_INDEX = 2;

  float *input_1 = static_cast<float *>(params[0]);
  float *input_2 = static_cast<float *>(params[1]);
  float *output = static_cast<float *>(params[2]);
  float *tmp = static_cast<float *>(params[3]);

  // Add
  int in_size = 1;
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++) {
    in_size *= shapes[OUTPUT_INDEX][i];
  }

  for (int i = 0; i < in_size; i++) {
    tmp[i] = input_1[i] + input_2[i];
  }

  // ReduceSum
  AotExtra *extra = static_cast<AotExtra *>(extra_void);
  auto kernel_ptr = static_cast<add_reduce_kernel_attr *>(extra->KernelData());
  bool keep_dim = kernel_ptr->keep_dim;
  int64_t axis = kernel_ptr->axis;
  int64_t input_dim_1 = shapes[0][1];
  int size;
  if (keep_dim) {
    size = shapes[1][0] * shapes[1][1];
  } else {
    size = shapes[1][0];
  }

  int ext = shapes[0][axis];
  for (int i = 0; i < size; i++) {
    output[i] = 0;
    for (int j = 0; j < ext; j++) {
      int idx = input_dim_1 * (i * axis + j * (1 - axis)) + i * (1 - axis) + j * axis;
      output[i] = output[i] + tmp[idx];
    }
  }
  return 0;
}
```

In the computation of Add, we used the intermediate variable of the operator, and the method is as follows:

1. Convert the pointers in the `params` array to `float *` one by one. According to the introduction of the interface above, the elements in the array are: two input address pointers (`input_1` and `input_2`), an output address pointer (`output`), and an intermediate variable address pointer (`tmp`);
2. Store the result of adding the two inputs into the intermediate variable: `tmp[i] = input_1[i] + input_2[i]`.

In the computation of ReduceSum, we used the attribute value of the operator, and the method is as follows:

1. Convert the `extra_void` type to a `AotExtra` type pointer: `AotExtra *extra = static_cast<AotExtra *>(extra_void)`.
2. Get the `kernel_ptr` object pointer created in the initialization function from `extra`: `auto kernel_ptr = static_cast<add_reduce_kernel_attr *>(extra->KernelData())`. Here, `extra->KernelData()` obtains a void object pointer, which needs to be further converted to the `kernel_ptr` object pointer.
3. Use the attribute values stored in `kernel_ptr` for calculation: `bool keep_dim = kernel_ptr->keep_dim; int64_t axis = kernel_ptr->axis;`. Here, we obtain the variables `keep_dim` and `axis` from `kernel_ptr` for computation.

### Operator Definition File: test_custom_aot.py

To add AOT-type custom operator to a MindSpore network using the above functions, we create the file `test_custom_aot.py`.

```python
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.nn import Cell
import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp

class ReduceDynNet(Cell):
    def __init__(self, out_types, axis, keep_dim):
        super(ReduceDynNet, self).__init__()
        reduce_cpu_info = CustomRegOp("reduce_kernel_cpu") \
            .input(0, "x1") \
            .input(0, "x2") \
            .output(0, "y") \
            .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None) \
            .attr("axis", "required", "all", value=axis) \
            .attr("keep_dim", "required", "all", value=keep_dim) \
            .target("CPU") \
            .get_op_info()
        # As the shape inference function of C++ version is defined above, the ouptut_shape can be 'None'
        self.program = ops.Custom("./kernel.cc:CustomKernel", None, out_types, "aot", reg_info=reduce_cpu_info)

    def construct(self, x, y):
        return self.program(x, y)
```

The `ReduceDynNet` in this file includes two parts: the operator registration function and the operator definition class.

#### Operator Registration

The assignment of operator attributes during initialization is implemented through the operator registration function.
For the function of custom operator registration, please refer to the relevant documentation of [CustomRegOp](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.CustomRegOp.html#mindspore-ops-customregop).
For each attribute, we create an `attr` for the operator registration file `reduce_cpu_info`, setting the attribute name and value.

Each `attr` item here has four inputs: the first is the name, such as `"axis"` or `"keep_dim"`; the middle two are `"required"` and `"all"`; the last input needs to specify the input name as `value=`, and the input value is the value of the attribute, for example, `value=axis` and `value=keep_dim` here.
We determine the values of these two parameters from the network input, and these values should match the types used in the `extra->Attr<T>` template interface in the initialization function and shape inference function above.

In addition, if we need to define multiple operator registration files, we need to use different operator file names, which is the argument of `CustomRegOp`, here it is `"add_with_attr_kernel_cpu"`. If we want to define another operator with the same prototype but different attribute values, the name cannot be duplicated.

#### Operator Definition

In the Python file above, a custom operator of type `aot` is defined using the interface `Custom` of MindSpore: `self.program = ops.Custom("./kernel.cc:CustomKernel", None, out_types, "aot", reg_info=reduce_cpu_info)`. Since we defined the C++ version of the shape inference function earlier, `ouptut_shape` can be set to `None`.

Notice that in the operator definition, we directly use the source file name `./kernel.cc`, so we are utilizing the automatic compilation feature provided by MindSpore. Make sure that the corresponding compiler (g++ in this case, and nvcc for GPU environment) is available in the environment.

### Operator Call

As a test, we add the `__main__` function to the `test_custom_aot.py` file:

```python
if __name__ == "__main__":
    shape = (4, 5)
    axis = 1
    keep_dim = False
    ms.set_device(device_target="CPU")

    input_x = np.ones(shape).astype(np.float32)
    input_y = np.ones(shape).astype(np.float32)

    test = ReduceDynNet(mstype.float32, axis, keep_dim)
    dyn_x = Tensor(shape=[4, None], dtype=mstype.float32)
    # set the net to dynamic shape
    test.set_inputs(dyn_x, dyn_x)
    output = test(Tensor(input_x),Tensor(input_y))
    print(output)
```

Execute the file to call the operator:

```bash
python test_custom_aot.py
```

Execution result is as follows:

```text
[10. 10. 10. 10.]
```

## Introduction to Multi-Output Custom Operators of the AOT Type

Custom operators of the AOT-type support multiple outputs (outputs as tuples). The definition of the operator file for a custom operator with multiple outputs is similar to that of a single-output operator, but corresponding modifications need to be made based on the multi-output scenario, including:

- Operator inference function: The output of the `infer` function needs to be written in the form of a tuple;
- Operator registration file: The names and data type information of multiple outputs need to be listed;
- Operator computation function: It needs to identify the pointers corresponding to multiple outputs.

Below, we demonstrate the method of defining a custom operator of the AOT-type with multiple outputs using an example. For specific file usage, please refer to [here](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/tests/st/graph_kernel/custom/test_custom_aot.py#L405).

### Operator Inference Function

In the case of multiple outputs, the operator inference function should be written in the form of a tuple.
Taking the case where the output shapes are constants as an example, the `out_shapes` in the custom operator below is `([3], [3], [3])`,
and `out_dtypes` is `(mstype.float32, mstype.float32, mstype.float32)`, which correspond to the shapes and data types of the three outputs, respectively.

```python
self.program = ops.Custom(func, ([3], [3], [3]), (mstype.float32, mstype.float32, mstype.float32), "aot", bprop, reg)
```

### Operator Registering Function

When defining multiple outputs, we need to clearly specify the names of the inputs and outputs in sequence, and indicate the corresponding data formats for both inputs and outputs in the `dtype_format` section. For example:

```python
multioutput_gpu_info = CustomRegOp() \
    .input(0, "x1") \
    .input(1, "x2") \
    .output(0, "y1") \
    .output(1, "y2") \
    .output(2, "y3") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()
```

Here, we define a registration file for an operator with two inputs and three outputs. Therefore, we add two `input` items and three `output` items in the registration file. Additionally, the five data formats defined in `dtype_format` correspond to the data format requirements for the two inputs and three outputs in sequence.

### Operator Computation Function

The following `CustomAddMulDiv` function is the computation function of the custom op.

```c++
constexpr int THREADS = 1024;

__global__ void CustomAddMulDivKernel(float *input1, float *input2, float *output1, float *output2, float *output3,
                                      size_t size) {
  auto idx = blockIdx.x * THREADS + threadIdx.x;
  if (idx < size) {
    output1[idx] = input1[idx] + input2[idx];
    output2[idx] = input1[idx] * input2[idx];
    output3[idx] = input1[idx] / input2[idx];
  }
}

extern "C" int CustomAddMulDiv(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                               void *stream, void *extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);

  constexpr int OUTPUT_INDEX = 2;
  constexpr int TOTAL_PARAM_NUM = 5;

  // There are two inputs and three outputs, so the nparam should be 5.
  if (nparam != TOTAL_PARAM_NUM) {
    return 1;
  }

  // This is to check if the type of parameters the same as what the user wants.
  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }

  // input1's index is 0, input2's index is 1, output1's index is 2, output2's index is 3 and output3's index is 4
  void *input1 = params[0];
  void *input2 = params[1];
  void *output1 = params[2];
  void *output2 = params[3];
  void *output3 = params[4];
  size_t size = 1;

  // Cumprod of output's shape to compute elements' num
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++) {
    size *= shapes[OUTPUT_INDEX][i];
  }
  int n = size / THREADS;

  // Do the computation
  CustomAddMulDivKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<float *>(input1), static_cast<float *>(input2),
                                                         static_cast<float *>(output1), static_cast<float *>(output2),
                                                         static_cast<float *>(output3), size);
  // When return 0, MindSpore will continue to run if this kernel could launch successfully.
  return 0;
}
```

Please note that since the operator has two inputs and three outputs, `nparam` should be 5, and the five pointers in the `params` array should correspond to the two inputs and three outputs in sequence.
Therefore, in the above code, we obtain the inputs and outputs as follows:

```c++
void *input1 = params[0];
void *input2 = params[1];
void *output1 = params[2];
void *output2 = params[3];
void *output3 = params[4];
```

For the complete operator computation file, please refer to [here](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/tests/st/graph_kernel/custom/aot_test_files/add_mul_div.cu).

### Operator in Scripts

When a custom operator with multiple outputs is involved in a net, the results can be used as a normal tuple, for example:

```python
class AOTMultiOutputNet(Cell):
    def __init__(self, func, out_shapes, out_types, bprop=None, reg=None):
        super(AOTMultiOutputNet, self).__init__()

        self.program = ops.Custom(func, out_shapes, out_types, "aot", bprop, reg)
        self.add = ops.Add()
        self.mul = ops.Mul()

    def construct(self, x, y):
        aot = self.program(x, y)
        add_res = self.add(aot[0], aot[1])
        mul_res = self.mul(add_res, aot[2])
        return mul_res

if __name__ == "__main__":
  x = np.array([1.0, 1.0, 1.0]).astype(np.float32)
  y = np.array([1.0, 1.0, 1.0]).astype(np.float32)
  net = AOTMultiOutputNet("./add_mul_div.cu:CustomAddMulDiv", ([3], [3], [3]),
                          (mstype.float32, mstype.float32, mstype.float32), reg=multioutput_gpu_info)
  output = test(Tensor(input_x),Tensor(input_y))
  print(output)
```

Here `aot` as the output of the multi-output custom operator can be used as a tuple. Execution result is as follows:

```text
[3. 3. 3.]
```
