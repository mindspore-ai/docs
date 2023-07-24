# mindspore

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/api_cpp/source_en/mindspore.md)

## Context

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

The Context class is used to store environment variables during execution.

### Public Member Functions

#### SetThreadNum

```cpp
void SetThreadNum(int32_t thread_num);
```

Set the number of threads at runtime. This option is only valid for MindSpore Lite.

- Parameters

    - `thread_num`: the number of threads at runtime.

#### GetThreadNum

```cpp
int32_t GetThreadNum() const;
```

Get the current thread number setting.

- Returns

  The current thread number setting.

#### SetAllocator

```cpp
void SetAllocator(const std::shared_ptr<Allocator> &allocator);
```

Set Allocator, which defines a memory pool for dynamic memory malloc and memory free. This option is only valid for MindSpore Lite.

- Parameters

    - `allocator`: A pointer to an Allocator.

#### GetAllocator

```cpp
std::shared_ptr<Allocator> GetAllocator() const;
```

Get the current Allocator setting.

- Returns

  The current Allocator setting.

#### MutableDeviceInfo

```cpp
std::vector<std::shared_ptr<DeviceInfoContext>> &MutableDeviceInfo();
```

Get a mutable reference of [DeviceInfoContext](#deviceinfocontext) vector in this context. Only MindSpore Lite supports heterogeneous scenarios with multiple members in the vector.

- Returns

  Mutable reference of DeviceInfoContext vector in this context.

## DeviceInfoContext

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

DeviceInfoContext defines different device contexts.

### Public Member Functions

#### GetDeviceType

```cpp
virtual enum DeviceType GetDeviceType() const = 0
```

Get the type of this DeviceInfoContext.

- Returns

  Type of this DeviceInfoContext.

  ```cpp
  enum DeviceType {
    kCPU = 0,
    kMaliGPU,
    kNvidiaGPU,
    kKirinNPU,
    kAscend910,
    kAscend310,
    // add new type here
    kInvalidDeviceType = 100,
  };
  ```

#### Cast

```cpp
template <class T> std::shared_ptr<T> Cast();
```

A similar function to RTTI is provided when the `-fno-rtti` compilation option is turned on, which converts DeviceInfoContext to a shared pointer of type `T`, and returns `nullptr` if the conversion fails.

- Returns

  A pointer of type `T` after conversion. If the conversion fails, it will be `nullptr`.

## CPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

Derived from [DeviceInfoContext](#deviceinfocontext), The configuration of the model running on the CPU. This option is only valid for MindSpore Lite.

### Public Member Functions

| Functions                                                                                                        | Notes                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetThreadAffinity(int mode)`          | Set thread affinity mode<br><br> - `mode`: 0: no affinity,  1: big cores first,  2: little cores first. |
| `int GetThreadAffinity() const`                                | - Returns: The thread affinity mode                                                                                                                                                    |
| `void SetEnableFP16(bool is_fp16)`                   | Enables to perform the float16 inference<br><br> - `is_fp16`: Enable float16 inference or not.                                                                                                                                                            |
| `bool GetEnableFP16() const`                                       | - Returns: whether enable float16 inference.                                                                                                                                                                                                                               |

## MaliGPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

Derived from [DeviceInfoContext](#deviceinfocontext), The configuration of the model running on the GPU. This option is only valid for MindSpore Lite.

### Public Member Functions

| Functions                                                                                                        | Notes                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetEnableFP16(bool is_fp16)`                   | Enables to perform the float16 inference<br><br> - `is_fp16`: Enable float16 inference or not.                                                                                                                                                            |
| `bool GetEnableFP16() const`                                       | - Returns: whether enable float16 inference.                                                                                                                                                                                                                               |

## KirinNPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

Derived from [DeviceInfoContext](#deviceinfocontext), The configuration of the model running on the NPU. This option is only valid for MindSpore Lite.

### Public Member Functions

| Functions                                                                                                        | Notes                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetFrequency(int frequency)`                   | Used to set the NPU frequency<br><br> - `frequency`: can be set to 1 (low power consumption), 2 (balanced), 3 (high performance), 4 (extreme performance), default as 3.                                                                                                                                                            |
| `int GetFrequency() const`                                       | - Returns: NPU frequency                                                                                                                                                                                                                               |

## NvidiaGPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

Derived from [DeviceInfoContext](#deviceinfocontext), The configuration of the model running on the GPU. This option is invalid for MindSpore Lite.

### Public Member Functions

| Functions                                                                                                        | Notes                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetDeviceID(uint32_t device_id)`                   | Used to set device id<br><br> - `device_id`: The device id.                                                                                                                                                            |
| `uint32_t GetDeviceID() const`                                       | - Returns: The device id.                                                                                                                                                                                                                               |

## Ascend910DeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

Derived from [DeviceInfoContext](#deviceinfocontext), The configuration of the model running on the Ascend910. This option is invalid for MindSpore Lite.

### Public Member Functions

| Functions                                                                                                        | Notes                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetDeviceID(uint32_t device_id)`                   | Used to set device id<br><br> - `device_id`: The device id.                                                                                                                                                            |
| `uint32_t GetDeviceID() const`                                       | - Returns: The device id.                                                                                                                                                                                                                               |

## Ascend310DeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

Derived from [DeviceInfoContext](#deviceinfocontext), The configuration of the model running on the Ascend310. This option is invalid for MindSpore Lite.

### Public Member Functions

| Functions                                                                                                        | Notes                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetDeviceID(uint32_t device_id)`                   | Used to set device id<br><br> - `device_id`: The device id.                                                                                                                                                            |
| `uint32_t GetDeviceID() const`                                       | - Returns: The device id.                                                                                                                                                                                                                             |
| `void SetInsertOpConfigPath(const std::string &cfg_path)`          | Set [AIPP](https://support.huaweicloud.com/intl/en-us/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html) configuration file path |
| `std::string GetInsertOpConfigPath()`                                | - Returns: [AIPP](https://support.huaweicloud.com/intl/en-us/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html) configuration file path                                                                                                                                                    |
| `void SetInputFormat(const std::string &format)`                   | Set format of model inputs<br><br> - `format`: Optional `"NCHW"`, `"NHWC"`, etc.                                                                                                                                                            |
| `std::string GetInputFormat()`                                       | - Returns: The set format of model inputs                                                                                                                                                                                                                               |
| `void SetInputShape(const std::string &shape)`                     | Set shape of model inputs<br><br> - `shape`: e.g., `"input_op_name1: 1,2,3,4;input_op_name2: 4,3,2,1"`                                                                                                                           |
| `std::string GetInputShape()`                                        | - Returns: The set shape of model inputs                                                                                                                                                                                                                                |
| `void SetOutputType(enum DataType output_type)`                    | Set type of model outputs<br><br> - `output_type`: Only uint8, fp16 and fp32 are supported                                                                                                                                                            |
| `enum DataType GetOutputType()`                                      | - Returns: The set type of model outputs                                                                                                                                                                                                                                 |
| `void SetPrecisionMode(const std::string &precision_mode)`         | Set precision mode of model<br><br> - `precision_mode`: Optional `"force_fp16"`, `"allow_fp32_to_fp16"`, `"must_keep_origin_dtype"` and `"allow_mix_precision"`, `"force_fp16"` is set as default                                                       |
| `std::string GetPrecisionMode(t)`                                     | - Returns: The set precision mode                                                                                                                                                                                                                                 |
| `void SetOpSelectImplMode(const std::string &op_select_impl_mode)` | Set op select implementation mode<br><br> - `op_select_impl_mode`: Optional `"high_performance"` and `"high_precision"`, `"high_performance"` is set as default                                                                                                 |
| `std::string GetOpSelectImplMode()`                                  | - Returns: The set op select implementation mode                                                                                                                                                                                                                                 |

## Serialization

\#include &lt;[serialization.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/serialization.h)&gt;

The Serialization class is used to summarize methods for reading and writing model files.

### Static Public Member Function

#### Load

Loads a model file from path, is not supported on MindSpore Lite.

```cpp
Status Load(const std::string &file, ModelType model_type, Graph *graph);
```

- Parameters

    - `file`: the path of model file.
    - `model_type`：the Type of model file, options are `ModelType::kMindIR`, `ModelType::kOM`.
    - `graph`：the output parameter, a object saves graph data.

- Returns

  Status code.

#### Load

Loads a model file from memory buffer.

```cpp
Status Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph);
```

- Parameters

    - `model_data`：a buffer filled by model file.
    - `data_size`：the size of the buffer.
    - `model_type`：the Type of model file, options are `ModelType::kMindIR`, `ModelType::kOM`.
    - `graph`：the output parameter, a object saves graph data.

- Returns

  Status code.

## Model

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/model.h)&gt;

The Model class is used to define a MindSpore model, facilitating computational graph management.

### Constructor and Destructor

```cpp
Model();
~Model();
```

### Public Member Function

#### Build

```cpp
Status Build(GraphCell graph, const std::shared_ptr<Context> &model_context);
```

Builds a model so that it can run on a device.

- Parameters

    - `graph`: `GraphCell` is a derivative of `Cell`. `Cell` is not available currently. `GraphCell` can be constructed from `Graph`, for example, `model.Build(GraphCell(graph), context)`.
    - `model_context`: a [context](#context) used to store options during execution.

- Returns

  Status code.

> Modifications to `model_context` after `Build` will no longer take effect.

#### Predict

```cpp
Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);
```

Inference model.

- Parameters

    - `inputs`: a `vector` where model inputs are arranged in sequence.
    - `outputs`: output parameter, which is a pointer to a `vector`. The model outputs are filled in the container in sequence.

- Returns

  Status code.

#### GetInputs

```cpp
std::vector<MSTensor> GetInputs();
```

Obtains all input tensors of the model.

- Returns

  The vector that includes all input tensors.

#### GetInputByTensorName

```cpp
MSTensor GetInputByTensorName(const std::string &tensor_name);
```

Obtains the input tensor of the model by name.

- Returns

  The input tensor with the given name, if the name is not found, an invalid tensor is returned.

#### GetOutputs

```cpp
std::vector<MSTensor> GetOutputs();
```

Obtains all output tensors of the model.

- Returns

  A `vector` that includes all output tensors.

#### GetOutputTensorNames

```cpp
std::vector<std::string> GetOutputTensorNames();
```

Obtains names of all output tensors of the model.

- Returns

  A `vector` that includes names of all output tensors.

#### GetOutputByTensorName

```cpp
MSTensor GetOutputByTensorName(const std::string &tensor_name);
```

Obtains the output tensor of the model by name.

- Returns

  The output tensor with the given name, if the name is not found, an invalid tensor is returned.

#### Resize

```cpp
Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims);
```

Resizes the shapes of inputs.

- Parameters

    - `inputs`: a `vector` that includes all input tensors in order.
    - `dims`: defines the new shapes of inputs, should be consistent with `inputs`.

- Returns

  Status code.

#### CheckModelSupport

```cpp
static bool CheckModelSupport(enum DeviceType device_type, ModelType model_type);
```

Checks whether the type of device supports the type of model.

- Parameters

    - `device_type`: device type，options are `kMaliGPU`, `kAscend910`, etc.
    - `model_type`: the Type of model file, options are `ModelType::kMindIR`, `ModelType::kOM`.

- Returns

  A bool value.

## MSTensor

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/types.h)&gt;

The MSTensor class defines a tensor in MindSpore.

### Constructor and Destructor

```cpp
MSTensor();
explicit MSTensor(const std::shared_ptr<Impl> &impl);
MSTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data, size_t data_len);
~MSTensor();
```

### Static Public Member Function

#### CreateTensor

```cpp
MSTensor *CreateTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                       const void *data, size_t data_len) noexcept;
```

Creates a MSTensor object, whose data need to be copied before accessed by `Model`, must be used in pairs with `DestroyTensorPtr`.

- Parameters

    - `name`: the name of the `MSTensor`.
    - `type`: the data type of the `MSTensor`.
    - `shape`: the shape of the `MSTensor`.
    - `data`: the data pointer that points to allocated memory.
    - `data`: the length of the memory, in bytes.

- Returns

  An pointer of `MStensor`.

#### CreateRefTensor

```cpp
MSTensor *CreateRefTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, void *data,
                          size_t data_len) noexcept;
```

Creates a MSTensor object, whose data can be directly accessed by `Model`, must be used in pairs with `DestroyTensorPtr`.

- Parameters

    - `name`: the name of the `MSTensor`.
    - `type`: the data type of the `MSTensor`.
    - `shape`: the shape of the `MSTensor`.
    - `data`: the data pointer that points to allocated memory.
    - `data`: the length of the memory, in bytes.

- Returns

  An pointer of `MStensor`.

#### StringsToTensor

```cpp
MSTensor *StringsToTensor(const std::string &name, const std::vector<std::string> &str);
```

Create a string type `MSTensor` object whose data can be accessed by `Model` only after being copied, must be used in pair with `DestroyTensorPtr`.

- Parameters

    - `name`: the name of the `MSTensor`.
    - `str`：a `vector` container containing several strings.

- Returns

  An pointer of `MStensor`.

#### TensorToStrings

```cpp
std::vector<std::string> TensorToStrings(const MSTensor &tensor);
```

Parse the string type `MSTensor` object into strings.

- Parameters

    - `tensor`: a `MSTensor` object.

- Returns

  A `vector` container containing several strings.

#### DestroyTensorPtr

```cpp
void DestroyTensorPtr(MSTensor *tensor) noexcept;
```

Destroy an object created by `Clone`, `StringsToTensor`, `CreateRefTensor` or `CreateTensor`. Do not use it to destroy `MSTensor` from other sources.

- Parameters

    - `tensor`: a pointer returned by `Clone`, `StringsToTensor`, `CreateRefTensor` or `CreateTensor`.

### Public Member Functions

#### Name

```cpp
std::string Name() const;
```

Obtains the name of the `MSTensor`.

- Returns

  The name of the `MSTensor`.

#### DataType

```cpp
enum DataType DataType() const;
```

Obtains the data type of the `MSTensor`.

- Returns

  The data type of the `MSTensor`.

#### Shape

```cpp
const std::vector<int64_t> &Shape() const;
```

Obtains the shape of the `MSTensor`.

- Returns

  A `vector` that contains the shape of the `MSTensor`.

#### ElementNum

```cpp
int64_t ElementNum() const;
```

Obtains the number of elements of the `MSTensor`.

- Returns

  The number of elements of the `MSTensor`.

#### Data

```cpp
std::shared_ptr<const void> Data() const;
```

Obtains a shared pointer to the copy of data of the `MSTensor`.

- Returns

  A shared pointer to the copy of data of the `MSTensor`.

#### MutableData

```cpp
void *MutableData();
```

Obtains the pointer to the data of the `MSTensor`.

- Returns

  The pointer to the data of the `MSTensor`.

#### DataSize

```cpp
size_t DataSize() const;
```

Obtains the length of the data of the `MSTensor`, in bytes.

- Returns

  The length of the data of the `MSTensor`, in bytes.

#### IsDevice

```cpp
bool IsDevice() const;
```

Gets the boolean value that indicates whether the memory of `MSTensor` is on device.

- Returns

  The boolean value that indicates whether the memory of `MSTensor` is on device.

#### Clone

```cpp
MSTensor *Clone() const;
```

Gets a deep copy of the `MSTensor`, must be used in pair with `DestroyTensorPtr`.

- Returns

  A pointer points to a deep copy of the `MSTensor`.

#### operator==(std::nullptr_t)

```cpp
bool operator==(std::nullptr_t) const;
```

Gets the boolean value that indicates whether the `MSTensor` is valid.

- Returns

  The boolean value that indicates whether the `MSTensor` is valid.

## KernelCallBack

\#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/include/ms_tensor.h)&gt;

```cpp
using KernelCallBack = std::function<bool(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs, const CallBackParam &opInfo)>
```

A function wrapper. KernelCallBack defines the pointer for callback function.

## CallBackParam

\#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/include/ms_tensor.h)&gt;

A **struct**. CallBackParam defines input arguments for callback function.

### Public Attributes

#### node_name

```cpp
node_name
```

A **string** variable. Node name argument.

#### node_type

```cpp
node_type
```

A **string** variable. Node type argument.
