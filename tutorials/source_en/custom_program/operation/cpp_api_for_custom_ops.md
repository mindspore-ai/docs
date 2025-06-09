# C++ API Description for Custom Operators

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/custom_program/operation/cpp_api_for_custom_ops.md)

## Overview

The C++ API for MindSpore custom operators is divided into two categories:

1. **API Interfaces**:  
   Interfaces marked as 【API】 are stable public interfaces intended for direct use by users. These interfaces have been thoroughly tested, have clear functionality, and are highly backward compatible.

2. **Experimental Interfaces**:  
   Interfaces not marked as 【API】 are experimental. These interfaces may change or be removed in future versions and should be used with caution.

When developing custom operators, you can include the header files referenced by the following interfaces via `#include "ms_extension/api.h"`, without worrying about the specific location of each interface.

## namespace ms

### enum TypeId

The `TypeId` enumeration type is defined in the [type_id.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/include/mindapi/base/type_id.h) header file and specifies the tensor data types supported in MindSpore, including boolean, integer, floating-point, and complex types.

This interface is also included in the `namespace ms` namespace and can be accessed via `ms::TypeId`.

```cpp
kNumberTypeBegin,       // Start value for the Number type
kNumberTypeBool,        // Boolean type
kNumberTypeInt,         // Default integer type
kNumberTypeInt8,        // 8-bit signed integer
kNumberTypeInt16,       // 16-bit signed integer
kNumberTypeInt32,       // 32-bit signed integer
kNumberTypeInt64,       // 64-bit signed integer
kNumberTypeUInt,        // Default unsigned integer type
kNumberTypeUInt8,       // 8-bit unsigned integer
kNumberTypeUInt16,      // 16-bit unsigned integer
kNumberTypeUInt32,      // 32-bit unsigned integer
kNumberTypeUInt64,      // 64-bit unsigned integer
kNumberTypeFloat,       // Default floating-point type
kNumberTypeFloat16,     // 16-bit half-precision floating-point
kNumberTypeFloat32,     // 32-bit single-precision floating-point
kNumberTypeFloat64,     // 64-bit double-precision floating-point
kNumberTypeBFloat16,    // 16-bit brain floating-point
kNumberTypeDouble,      // Double-precision floating-point (equivalent to kNumberTypeFloat64)
kNumberTypeComplex,     // Default complex number type
kNumberTypeComplex64,   // 64-bit complex number (composed of two 32-bit floating-point numbers)
kNumberTypeComplex128,  // 128-bit complex number (composed of two 64-bit floating-point numbers)
kNumberTypeInt4,        // 4-bit signed integer
kNumberTypeGLUInt,      // OpenGL unsigned integer type
kNumberTypeEnd,         // End value for the Number type
```

### class Tensor

The `Tensor` class is defined in the [tensor.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/ms_extension/common/tensor.h) header file, representing the tensor object in MindSpore. It provides methods for operating on and querying tensor properties.

#### Constructors

- **Tensor()**
    - **Description**: 【API】 Constructs an undefined placeholder tensor.

- **Tensor(TypeId, const ShapeVector &)**

  ```cpp
  Tensor(TypeId type_id, const ShapeVector &shape)
  ```

    - **Description**: 【API】 Constructs a tensor with the specified data type and shape.
    - **Parameters**:
        - `type_id`: The data type of the tensor.
        - `shape`: The shape of the tensor, represented as a vector of integers.

- **Tensor(const mindspore::ValuePtr &)**

  ```cpp
  Tensor(const mindspore::ValuePtr &value)
  ```

    - **Description**: Constructs a tensor object from the given `ValuePtr`.
    - **Parameters**:
        - `value`: A smart pointer to a MindSpore `Value` object. If the value is `nullptr`, an undefined tensor is constructed.

#### Public Methods (Attributes and Configurations)

- **is_defined()**

  ```cpp
  bool is_defined() const
  ```

    - **Description**: 【API】 Checks whether the tensor is defined.
    - **Return Value**: Returns `true` if the tensor is defined, otherwise returns `false`.

- **data_type()**

  ```cpp
  TypeId data_type() const
  ```

    - **Description**: 【API】 Retrieves the data type of the tensor.
    - **Return Value**: The data type of the tensor.

- **shape()**

  ```cpp
  const ShapeVector &shape() const
  ```

    - **Description**: 【API】 Retrieves the shape of the tensor.
    - **Return Value**: A reference to the shape of the tensor (`ShapeVector`, i.e., `std::vector<int64_t>`).

- **numel()**

  ```cpp
  size_t numel() const
  ```

    - **Description**: 【API】 Returns the total number of elements in the tensor.
    - **Return Value**: The total number of elements.

- **stride()**

  ```cpp
  std::vector<int64_t> stride() const
  ```

    - **Description**: 【API】 Computes the strides of the tensor.
    - **Return Value**: A vector representing the strides of the tensor for each dimension.

- **storage_offset()**

  ```cpp
  int64_t storage_offset() const
  ```

    - **Description**: 【API】 Retrieves the storage offset of the tensor.
    - **Return Value**: The offset from the start of storage (in terms of elements).

- **is_contiguous()**

  ```cpp
  bool is_contiguous() const
  ```

    - **Description**: 【API】 Checks whether the tensor is stored contiguously in memory.
    - **Return Value**: Returns `true` if the tensor is stored contiguously, otherwise returns `false`.

- **SetNeedContiguous(bool)**

  ```cpp
  void SetNeedContiguous(bool flag) const
  ```

    - **Description**: 【API】 Sets whether the tensor requires contiguous storage space.
    - **Parameters**:
        - `flag`: A boolean value indicating whether the tensor needs contiguous storage.

- **GetDataPtr()**

  ```cpp
  void *GetDataPtr() const
  ```

    - **Description**: 【API】 Retrieves a pointer to the tensor data.
    - **Return Value**: A `void` pointer pointing to the tensor data.
    - **Note**: The returned pointer already includes the offset indicated by the `storage_offset()` interface.

#### Public Methods (Operator Calls)

- **cast(TypeId)**

  ```cpp
  Tensor cast(TypeId dtype) const
  ```

    - **Description**: 【API】 Converts the tensor to the specified data type.
    - **Parameters**:
        - `dtype`: The target data type, such as `float`, `int`, etc.
    - **Return Value**: A new tensor with the specified data type.

- **chunk(int64_t, int64_t)**

  ```cpp
  std::vector<Tensor> chunk(int64_t chunks, int64_t dim = 0) const
  ```

    - **Description**: 【API】 Splits the tensor into multiple smaller tensors along the specified dimension.
    - **Parameters**:
        - `chunks`: The number of chunks to split into, must be a positive number.
        - `dim`: The dimension along which to split, default is 0.
    - **Return Value**: A vector containing multiple smaller tensors. All chunks have equal size; if the dimension size cannot be evenly divided by the number of chunks, the last chunk may be smaller.

- **contiguous()**

  ```cpp
  Tensor contiguous() const
  ```

    - **Description**: 【API】 Returns a tensor that is stored contiguously in memory.
    - **Return Value**: A new tensor with contiguous storage.

- **flatten(int64_t, int64_t)**

  ```cpp
  Tensor flatten(int64_t start_dim = 0, int64_t end_dim = -1) const
  ```

    - **Description**: 【API】 Flattens multiple dimensions of the tensor into one dimension.
    - **Parameters**:
        - `start_dim`: The starting dimension to flatten, default is 0.
        - `end_dim`: The ending dimension to flatten, default is -1 (the last dimension).
    - **Return Value**: A flattened tensor.

- **index_select(int64_t, const Tensor &)**

  ```cpp
  Tensor index_select(int64_t dim, const Tensor &index) const
  ```

    - **Description**: 【API】 Selects elements along a specified dimension based on an index tensor.
    - **Parameters**:
        - `dim`: The dimension along which to select elements.
        - `index`: A tensor containing the indices. The values in the tensor must be within the range `[0, shape(dim)-1]`.
    - **Return Value**: A tensor containing the selected elements.

- **reshape(const std::vector<int64_t> &)**

  ```cpp
  Tensor reshape(const std::vector<int64_t> &shape) const
  ```

    - **Description**: 【API】 Reshapes the tensor to the specified shape.
    - **Parameters**:
        - `shape`: A vector specifying the new shape. The total number of elements in the new shape must match the original tensor. A `-1` can be used in one dimension to infer its size automatically.
    - **Return Value**: A tensor with the new shape.

- **repeat(const std::vector<int64_t> &)**

  ```cpp
  Tensor repeat(const std::vector<int64_t> &repeats) const
  ```

    - **Description**: 【API】 Repeats the tensor along each dimension.
    - **Parameters**:
        - `repeats`: A vector specifying the number of times to repeat along each dimension. Its size must match the number of dimensions of the tensor.
    - **Return Value**: A new tensor with repeated elements.

- **repeat_interleave**

  ```cpp
  Tensor repeat_interleave(const Tensor &repeats, std::optional<int64_t> dim = std::nullopt,
                           std::optional<int64_t> output_size = std::nullopt) const;
  Tensor repeat_interleave(int64_t repeats, std::optional<int64_t> dim = std::nullopt,
                           std::optional<int64_t> output_size = std::nullopt) const;
  ```

    - **Description**: 【API】 Repeats the elements of the tensor along a specified dimension.
    - **Parameters**:
        - `repeats`: A scalar or tensor specifying the number of times each element is repeated. If a tensor is provided, its size must match the size of the specified dimension.
        - `dim`: The dimension along which to repeat elements, default is `std::nullopt`.
        - `output_size`: (Optional) The size of the output tensor along the specified dimension.
    - **Return Value**: A new tensor with repeated elements.

#### Public Methods (Internal Processes)

The following methods are not part of the API and are used only in internal module processes. Due to syntax constraints, they are set as public methods but are not recommended for direct use by users.

- **need_contiguous()**

  ```cpp
  bool need_contiguous() const
  ```

    - **Description**: Checks whether the tensor requires contiguous storage space.
    - **Return Value**: Returns `true` if the tensor requires contiguous storage, otherwise `false`.

- **stub_node()**

  ```cpp
  const mindspore::ValuePtr &stub_node() const
  ```

    - **Description**: Retrieves the stub node associated with the tensor.
    - **Return Value**: A smart pointer to the stub node (`ValuePtr`).

- **tensor()**

  ```cpp
  const mindspore::tensor::TensorPtr &tensor() const
  ```

    - **Description**: Retrieves the underlying tensor object.
    - **Return Value**: A smart pointer to the `TensorPtr` object.

- **ConvertStubNodeToTensor()**

  ```cpp
  void ConvertStubNodeToTensor() const
  ```

    - **Description**: Converts the stub node into a tensor object.
    - **Behavior**: Ensures that the tensor is fully realized from its stub representation. After the conversion, the stub node is released.

### function tensor

Factory methods for constructing constant tensors, defined in the [tensor_utils.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/ms_extension/common/tensor_utils.h) header file.

```cpp
Tensor tensor(int64_t value, TypeId dtype = TypeId::kNumberTypeInt64)
Tensor tensor(const std::vector<int64_t> &value, TypeId dtype = TypeId::kNumberTypeInt64)
Tensor tensor(double value, TypeId dtype = TypeId::kNumberTypeFloat64)
Tensor tensor(const std::vector<double> &value, TypeId dtype = TypeId::kNumberTypeFloat64)
```

- **Description**: 【API】 Creates a tensor with the given initial value.
- **Parameters**:
    - `value`: The value used to initialize the tensor. Supports integers, floating-point numbers, integer vectors, and floating-point vectors.
    - `dtype`: The data type of the tensor. For integers, the default is `ms::TypeId::kNumberTypeInt64`. For floating-point numbers, the default is `ms::TypeId::kNumberTypeFloat64`.
- **Return Value**: A tensor containing the specified value.

### function ones

Factory method for constructing a tensor filled with ones, defined in the [tensor_utils.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/ms_extension/common/tensor_utils.h) header file.

```cpp
Tensor ones(const ShapeVector &shape, TypeId dtype = TypeId::kNumberTypeFloat32)
```

- **Description**: 【API】 Creates a tensor with the specified shape and initializes all elements to `1`.
- **Parameters**:
    - `shape`: The shape of the tensor, represented as a vector of integers.
    - `dtype`: The data type of the tensor, default is `TypeId::kNumberTypeFloat32`.
- **Return Value**: A tensor where all elements are `1`.

### function zeros

Factory method for constructing a tensor filled with zeros, defined in the [tensor_utils.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/ms_extension/common/tensor_utils.h) header file.

```cpp
Tensor zeros(const ShapeVector &shape, TypeId dtype = TypeId::kNumberTypeFloat32)
```

- **Description**: 【API】 Creates a tensor with the specified shape and initializes all elements to `0`.
- **Parameters**:
    - `shape`: The shape of the tensor, represented as a vector of integers.
    - `dtype`: The data type of the tensor, default is `TypeId::kNumberTypeFloat32`.
- **Return Value**: A tensor where all elements are `0`.

## namespace ms::pynative

### class PyboostRunner

The `PyboostRunner` class for PyNative processes is defined in the [pyboost_extension.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/ms_extension/pynative/pyboost_extension.h) header file. It provides methods for managing execution, memory allocation, and kernel launching.

`PyboostRunner` is a subclass of `std::enable_shared_from_this` and requires the use of the smart pointer `std::shared_ptr` to manage its objects.

#### Constructor

- **PyboostRunner(const std::string &)**

  ```cpp
  PyboostRunner(const std::string &op_name)
  ```

    - **Description**: 【API】 Constructs a `PyboostRunner`.
    - **Parameters**:
        - `op_name`: The name of the operator.

#### Static Public Methods

- **static Call(FuncType, Args &&...)**

  ```cpp
  template <int OUT_NUM, typename FuncType, typename... Args>
  static py::object Call(FuncType func, Args &&... args)
  ```

    - **Description**: 【API】 Executes the given function and converts its output to a Python object.
    - **Template Parameters**:
        - `OUT_NUM`: The number of outputs from the operator, which must match the length of the tensor list returned by `func`. Currently, scenarios with variable output numbers are not supported.
        - `FuncType`: The prototype of the operator entry function, which can be automatically recognized from the function arguments.
        - `Args`: The types of operator input arguments, which can also be automatically recognized from the function arguments. The order of arguments must match the parameter order of `func`.
    - **Parameters**:
        - `func`: The function to execute.
        - `args`: The arguments required to execute the function.
    - **Return Value**: A Python object representing the operator's output.

#### Public Methods

- **Run(const std::vector<Tensor> &, const std::vector<Tensor> &)**

  ```cpp
  void Run(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs)
  ```

    - **Description**: 【API】 Runs the operator with the specified inputs and outputs.
    - **Parameters**:
        - `inputs`: A list of input tensors.
        - `outputs`: A list of output tensors.

- **CalcWorkspace()**

  ```cpp
  virtual size_t CalcWorkspace()
  ```

    - **Description**: 【API】 Calculates the workspace size required by the operator.
    - **Return Value**: The workspace size (in bytes). The default value is 0.

- **LaunchKernel()**

  ```cpp
  virtual void LaunchKernel() = 0;
  ```

    - **Description**: 【API】 Launches the kernel function of the operator.

- **op_name()**

  ```cpp
  const std::string &op_name() const
  ```

    - **Description**: 【API】 Retrieves the name of the operator associated with the runner.
    - **Return Value**: A string containing the operator's name.

- **inputs()**

  ```cpp
  const std::vector<ms::Tensor> &inputs() const
  ```

    - **Description**: 【API】 Retrieves the list of input tensors.
    - **Return Value**: A reference to the list of input tensors.

- **outputs()**

  ```cpp
  const std::vector<ms::Tensor> &outputs() const
  ```

    - **Description**: 【API】 Retrieves the list of output tensors.
    - **Return Value**: A reference to the list of output tensors.

- **stream_id()**

  ```cpp
  uint32_t stream_id() const
  ```

    - **Description**: 【API】 Retrieves the stream ID associated with the runner.
    - **Return Value**: The stream ID.

- **stream()**

  ```cpp
  void *stream()
  ```

    - **Description**: 【API】 Retrieves the stream pointer associated with the runner.
    - **Return Value**: A pointer to the stream.

- **workspace_ptr()**

  ```cpp
  void *workspace_ptr()
  ```

    - **Description**: 【API】 Retrieves the workspace pointer of the operator.
    - **Return Value**: A pointer to the workspace memory.

### class AtbOpRunner

The `AtbOpRunner` class is a runner for executing Ascend Transformer Boost (ATB) operators, defined in the [atb_common.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/ms_extension/ascend/atb/atb_common.h) header file.

This class inherits from `PyboostRunner` and encapsulates the process of invoking ATB operators, including initialization, running the ATB operator, managing input/output tensors, memory allocation, and kernel scheduling.

Refer to the tutorial [CustomOpBuilder Using AtbOpRunner to Integrate ATB Operators](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/op_customopbuilder_atb.html) for usage methods.

#### Constructor

- **AtbOpRunner**

  ```cpp
  using PyboostRunner::PyboostRunner;
  ```

  Constructor inherited from `PyboostRunner`.

#### Public Methods

- **Init(const ParamType&)**

  ```cpp
  template <typename ParamType>
  void Init(const ParamType &param)
  ```

    - **Description**: 【API】 Initializes the ATB operator with the given parameters. This method creates a corresponding `atb::Operation` instance for the operator via `atb::CreateOperation` and places it in the cache. Only one `atb::Operation` instance is created for operators with the same `param` hash value.
    - **Parameters**:
        - `param`: Parameters used to configure the ATB operator.
    - **Note**: For the `ParamType` type passed in, you need to specialize the `template <> struct HashOpParam<ParamType>::operator()` function in advance.

### function RunAtbOp

The interface for executing ATB operators in dynamic graphs, defined in the [atb_common.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/ms_extension/ascend/atb/atb_common.h) header file.

```cpp
template <typename ParamType>
void RunAtbOp(const std::string &op_name, const ParamType &param, const std::vector<Tensor> &inputs,
              const std::vector<Tensor> &outputs)
```

【API】 Executes an ATB operator using the provided parameters, inputs, and outputs. This function is a wrapper around `AtbOpRunner`.

- **Parameters**:
    - `op_name`: The name of the ATB operator to execute.
    - `param`: Parameters required to initialize the ATB operator.
    - `inputs`: A list of input tensors for the operator.
    - `outputs`: A list of output tensors for the operator.
