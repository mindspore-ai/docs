# mindspore

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_en/mindspore.md)

## Context

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/context.h)&gt;

The Context class is used to store environment variables during execution, which has two derived classes: `GlobalContext` and `ModelContext`.

## GlobalContext : Context

GlobalContext is used to store global environment variables during execution.

### Static Public Member Function

#### GetGlobalContext

```cpp
static std::shared_ptr<Context> GetGlobalContext();
```

Obtains the single instance of GlobalContext.

- Returns

  The single instance of GlobalContext.

#### SetGlobalDeviceTarget

```cpp
static void SetGlobalDeviceTarget(const std::string &device_target);
```

Configures the target device.

- Parameters

    - `device_target`: target device to be configured, options are `kDeviceTypeAscend310`, `kDeviceTypeAscend910`.

#### GetGlobalDeviceTarget

```cpp
static std::string GetGlobalDeviceTarget();
```

Obtains the configured target device.

- Returns

  The configured target device.

#### SetGlobalDeviceID

```cpp
static void SetGlobalDeviceID(const unit32_t &device_id);
```

Configures the device ID.

- Parameters

    - `device_id`: the device ID to configure.

#### GetGlobalDeviceID

```cpp
static uint32_t GetGlobalDeviceID();
```

Obtains the configured device ID.

- Returns

  The configured device ID.

## ModelContext : Context

### Static Public Member Function

| Function                                                                                                    | Notes                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `void SetInsertOpConfigPath(const std::shared_ptr<Context> &context, const std::string &cfg_path)`          | Set [AIPP](https://support.huaweicloud.com/intl/en-us/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html) configuration file path<br><br> - `context`: context to be set<br><br> - `cfg_path`: [AIPP](https://support.huaweicloud.com/intl/en-us/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html) configuration file path |
| `std::string GetInsertOpConfigPath(const std::shared_ptr<Context> &context)`                                | - Returns: The set [AIPP](https://support.huaweicloud.com/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html) configuration file path                                                                                                                                                                              |
| `void SetInputFormat(const std::shared_ptr<Context> &context, const std::string &format)`                   | Set format of model inputs<br><br> - `context`: context to be set<br><br> - `format`: Optional `"NCHW"`, `"NHWC"`, etc.                                                                                                                                                                                               |
| `std::string GetInputFormat(const std::shared_ptr<Context> &context)`                                       | - Returns: The set format of model inputs                                                                                                                                                                                                                                                                       |
| `void SetInputShape(const std::shared_ptr<Context> &context, const std::string &shape)`                     | Set shape of model inputs<br><br> - `context`: context to be set<br><br> - `shape`: e.g., `"input_op_name1: 1,2,3,4;input_op_name2: 4,3,2,1"`                                                                                                                                                                 |
| `std::string GetInputShape(const std::shared_ptr<Context> &context)`                                        | - Returns: The set shape of model inputs                                                                                                                                                                                                                                                                        |
| `void SetOutputType(const std::shared_ptr<Context> &context, enum DataType output_type)`                    | Set type of model outputs<br><br> - `context`: context to be set<br><br> - `output_type`: Only uint8, fp16 and fp32 are supported                                                                                                                                                                                     |
| `enum DataType GetOutputType(const std::shared_ptr<Context> &context)`                                      | - Returns: The set type of model outputs                                                                                                                                                                                                                                                                        |
| `void SetPrecisionMode(const std::shared_ptr<Context> &context, const std::string &precision_mode)`         | Set precision mode of model<br><br> - `context`: context to be set<br><br> - `precision_mode`: Optional `"force_fp16"`, `"allow_fp32_to_fp16"`, `"must_keep_origin_dtype"` and `"allow_mix_precision"`, `"force_fp16"` is set as default                                                                              |
| `std::string GetPrecisionMode(const std::shared_ptr<Context> &context)`                                     | - Returns: The set precision mode                                                                                                                                                                                                                                                                               |
| `void SetOpSelectImplMode(const std::shared_ptr<Context> &context, const std::string &op_select_impl_mode)` | Set op select implementation mode<br><br> - `context`: context to be set<br><br> - `op_select_impl_mode`: Optional `"high_performance"` and `"high_precision"`, `"high_performance"` is set as default                                                                                                                |
| `std::string GetOpSelectImplMode(const std::shared_ptr<Context> &context)`                                  | - Returns: The set op select implementation mode                                                                                                                                                                                                                                                                |

## Serialization

\#include &lt;[serialization.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/serialization.h)&gt;

The Serialization class is used to summarize methods for reading and writing model files.

### Static Public Member Function

#### LoadModel

```cpp
static Graph LoadModel(const std::string &file, ModelType model_type);
```

Loads a model file from path.

- Parameters

    - `file`: the path of model file.
    - `model_type`: the Type of model file, options are `ModelType::kMindIR`, `ModelType::kOM`.

- Returns

  An instance of `Graph`, used for storing graph data.

```cpp
static Graph LoadModel(const void *model_data, size_t data_size, ModelType model_type);
```

Loads a model file from memory buffer.

- Parameters

    - `model_data`: a buffer filled by model file.
    - `data_size`: the size of the buffer.
    - `model_type`: the Type of model file, options are `ModelType::kMindIR`, `ModelType::kOM`.

- Returns

  An instance of `Graph`, used for storing graph data.

## Model

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/model.h)&gt;

The Model class is used to define a MindSpore model, facilitating computational graph management.

### Constructor and Destructor

```cpp
explicit Model(const GraphCell &graph, const std::shared_ptr<Context> &model_context);
explicit Model(const std::vector<Output> &network, const std::shared_ptr<Context> &model_context);
~Model();
```

`GraphCell` is a derivative of `Cell`. `Cell` is not available currently. `GraphCell` can be constructed from `Graph`, for example, `Model model(GraphCell(graph))`。

`Context` is used to store the [model options](#modelcontext-contextfor-mindspore) during execution.

### Public Member Functions

#### Build

```cpp
Status Build();
```

Builds a model so that it can run on a device.

- Returns

  Status code.

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

#### GetOutputs

```cpp
std::vector<MSTensor> GetOutputs();
```

Obtains all output tensors of the model.

- Returns

  A `vector` that includes all output tensors.

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
static bool CheckModelSupport(const std::string &device_type, ModelType model_type);
```

Checks whether the type of device supports the type of model.

- Parameters

    - `device_type`: device type，options are `Ascend310`, `Ascend910`.
    - `model_type`: the Type of model file, options are `ModelType::kMindIR`, `ModelType::kOM`.

- Returns

  Status code.

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
static MSTensor CreateTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                             const void *data, size_t data_len) noexcept;
```

Creates a MSTensor object, whose data need to be copied before accessed by `Model`.

- Parameters

    - `name`: the name of the `MSTensor`.
    - `type`: the data type of the `MSTensor`.
    - `shape`: the shape of the `MSTensor`.
    - `data`: the data pointer that points to allocated memory.
    - `data`: the length of the memory, in bytes.

- Returns

  An instance of `MStensor`.

#### CreateRefTensor

```cpp
static MSTensor CreateRefTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, void *data,
                                size_t data_len) noexcept;
```

Creates a MSTensor object, whose data can be directly accessed by `Model`.

- Parameters

    - `name`: the name of the `MSTensor`.
    - `type`: the data type of the `MSTensor`.
    - `shape`: the shape of the `MSTensor`.
    - `data`: the data pointer that points to allocated memory.
    - `data`: the length of the memory, in bytes.

- Returns

  An instance of `MStensor`.

### Public Member Functions

#### Name

```cpp
const std::string &Name() const;
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
MSTensor Clone() const;
```

Gets a deep copy of the `MSTensor`.

- Returns

  A deep copy of the `MSTensor`.

#### operator==(std::nullptr_t)

```cpp
bool operator==(std::nullptr_t) const;
```

Gets the boolean value that indicates whether the `MSTensor` is valid.

- Returns

  The boolean value that indicates whether the `MSTensor` is valid.

## CallBack

\#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/ms_tensor.h)&gt;

The CallBack struct defines the call back function in MindSpore Lite.

### KernelCallBack

```cpp
using KernelCallBack = std::function<bool(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs, const CallBackParam &opInfo)>
```

A function wrapper. KernelCallBack defines the pointer for callback function.

### CallBackParam

A **struct**. CallBackParam defines input arguments for callback function.

#### Public Attributes

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
