# mindspore::api

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_en/api.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Context

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/context.h)&gt;

The Context class is used to store environment variables during execution.

### Static Public Member Function

#### Instance

```cpp
static Context &Instance();
```

Obtains the MindSpore Context instance object.

### Public Member Functions

#### GetDeviceTarget

```cpp
const std::string &GetDeviceTarget() const;
```

Obtains the target device type.

- Returns

  Current DeviceTarget type.

#### GetDeviceID

```cpp
uint32_t GetDeviceID() const;
```

Obtains the device ID.

- Returns

  Current device ID.

#### SetDeviceTarget

```cpp
Context &SetDeviceTarget(const std::string &device_target);
```

Configures the target device.

- Parameters

    - `device_target`: target device to be configured. The options are `kDeviceTypeAscend310` and `kDeviceTypeAscend910`.

- Returns

  MindSpore Context instance object.

#### SetDeviceID

```cpp
Context &SetDeviceID(uint32_t device_id);
```

Obtains the device ID.

- Parameters

    - `device_id`: device ID to be configured.

- Returns

  MindSpore Context instance object.

## Serialization

\#include &lt;[serialization.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/serialization.h)&gt;

The Serialization class is used to summarize methods for reading and writing model files.

### Static Public Member Function

#### LoadModel

- Parameters

    - `file`: model file path.
    - `model_type`: model file type. The options are `ModelType::kMindIR` and `ModelType::kOM`.

- Returns

  Object for storing graph data.

## Model

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/model.h)&gt;

A Model class is used to define a MindSpore model, facilitating computational graph management.

### Constructor and Destructor

```cpp
Model(const GraphCell &graph);
~Model();
```

`GraphCell` is a derivative of `Cell`. `Cell` is not open for use currently. `GraphCell` can be constructed from `Graph`, for example, `Model model(GraphCell(graph))`.

### Public Member Functions

#### Build

```cpp
Status Build(const std::map<std::string, std::string> &options);
```

Builds a model so that it can run on a device.

- Parameters

    - `options`: model build options. In the following table, Key indicates the option name, and Value indicates the corresponding option.

| Key | Value |
| --- | --- |
| kModelOptionInsertOpCfgPath | [AIPP](https://support.huaweicloud.com/intl/en-us/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html) configuration file path. |
| kModelOptionInputFormat | Manually specifies the model input format. The options are `"NCHW"` and `"NHWC"`. |
| kModelOptionInputShape | Manually specifies the model input shape, for example, `"input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1"` |
| kModelOptionOutputType | Manually specifies the model output type, for example, `"FP16"` or `"UINT8"`. The default value is `"FP32"`. |
| kModelOptionPrecisionMode | Model precision mode. The options are `"force_fp16"`, `"allow_fp32_to_fp16"`, `"must_keep_origin_dtype"`, and `"allow_mix_precision"`. The default value is `"force_fp16"`. |
| kModelOptionOpSelectImplMode | Operator selection mode. The options are `"high_performance"` and `"high_precision"`. The default value is `"high_performance"`. |

- Returns

  Status code.

#### Predict

```cpp
Status Predict(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs);
```

Inference model.

- Parameters

    - `inputs`: a `vector` where model inputs are arranged in sequence.
    - `outputs`: output parameter, which is the pointer to a `vector`. The model outputs are filled in the container in sequence.

- Returns

  Status code.

#### GetInputsInfo

```cpp
Status GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes, std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const;
```

Obtains the model input information.

- Parameters

    - `names`: optional output parameter, which is the pointer to a `vector` where model inputs are arranged in sequence. The input names are filled in the container in sequence. If `nullptr` is input, the attribute is not obtained.
    - `shapes`: optional output parameter, which is the pointer to a `vector` where model inputs are arranged in sequence. The input shapes are filled in the container in sequence. If `nullptr` is input, the attribute is not obtained.
    - `data_types`: optional output parameter, which is the pointer to a `vector` where model inputs are arranged in sequence. The input data types are filled in the container in sequence. If `nullptr` is input, the attribute is not obtained.
    - `mem_sizes`: optional output parameter, which is the pointer to a `vector` where model inputs are arranged in sequence. The input memory lengths (in bytes) are filled in the container in sequence. If `nullptr` is input, the attribute is not obtained.

- Returns

  Status code.

#### GetOutputsInfo

```cpp
Status GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes, std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const;
```

Obtains the model output information.

- Parameters

    - `names`: optional output parameter, which is the pointer to a `vector` where model outputs are arranged in sequence. The output names are filled in the container in sequence. If `nullptr` is input, the attribute is not obtained.
    - `shapes`: optional output parameter, which is the pointer to a `vector` where model outputs are arranged in sequence. The output shapes are filled in the container in sequence. If `nullptr` is input, the attribute is not obtained.
    - `data_types`: optional output parameter, which is the pointer to a `vector` where model outputs are arranged in sequence. The output data types are filled in the container in sequence. If `nullptr` is input, the attribute is not obtained.
    - `mem_sizes`: optional output parameter, which is the pointer to a `vector` where model outputs are arranged in sequence. The output memory lengths (in bytes) are filled in the container in sequence. If `nullptr` is input, the attribute is not obtained.

- Returns

  Status code.

## Tensor

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/types.h)&gt;

### Constructor and Destructor

```cpp
Tensor();
Tensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data, size_t data_len);
~Tensor();
```

### Static Public Member Function

#### GetTypeSize

```cpp
static int GetTypeSize(api::DataType type);
```

Obtains the memory length of a data type, in bytes.

- Parameters

    - `type`: data type.

- Returns

  Memory length, in bytes.

### Public Member Functions

#### Name

```cpp
const std::string &Name() const;
```

Obtains the name of a tensor.

- Returns

  Tensor name.

#### DataType

```cpp
api::DataType DataType() const;
```

Obtains the data type of a tensor.

- Returns

  Tensor data type.

#### Shape

```cpp
const std::vector<int64_t> &Shape() const;
```

Obtains the shape of a tensor.

- Returns

  Tensor shape.

#### SetName

```cpp
void SetName(const std::string &name);
```

Sets the name of a tensor.

- Parameters

    - `name`: name to be set.

#### SetDataType

```cpp
void SetDataType(api::DataType type);
```

Sets the data type of a tensor.

- Parameters

    - `type`: type to be set.

#### SetShape

```cpp
void SetShape(const std::vector<int64_t> &shape);
```

Sets the shape of a tensor.

- Parameters

    - `shape`: shape to be set.

#### Data

```cpp
const void *Data() const;
```

Obtains the constant pointer to the tensor data.

- Returns

  Constant pointer to the tensor data.

#### MutableData

```cpp
void *MutableData();
```

Obtains the pointer to the tensor data.

- Returns

  Pointer to the tensor data.

#### DataSize

```cpp
size_t DataSize() const;
```

Obtains the memory length (in bytes) of the tensor data.

- Returns

  Memory length of the tensor data, in bytes.

#### ResizeData

```cpp
bool ResizeData(size_t data_len);
```

Adjusts the memory size of the tensor.

- Parameters

    - `data_len`: number of bytes in the memory after adjustment.

- Returns

  A value of bool indicates whether the operation is successful.

#### SetData

```cpp
bool SetData(const void *data, size_t data_len);
```

Adjusts the memory data of the tensor.

- Parameters

    - `data`: memory address of the source data.
    - `data_len`: length of the source data memory.

- Returns

  A value of bool indicates whether the operation is successful.

#### ElementNum

```cpp
int64_t ElementNum() const;
```

Obtains the number of elements in a tensor.

- Returns

    Number of elements in a tensor.

#### Clone

```cpp
Tensor Clone() const;
```

Performs a self copy.

- Returns

  A deep copy.