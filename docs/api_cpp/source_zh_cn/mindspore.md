# mindspore

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/api_cpp/source_zh_cn/mindspore.md)

## Context

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

Context类用于保存执行中的环境变量。

### 公有成员函数

#### SetThreadNum

```cpp
void SetThreadNum(int32_t thread_num);
```

设置运行时的线程数，该选项仅MindSpore Lite有效。

- 参数

    - `thread_num`: 运行时的线程数。

#### GetThreadNum

```cpp
int32_t GetThreadNum() const;
```

获取当前线程数设置。

- 返回值

  当前线程数设置。

#### SetAllocator

```cpp
void SetAllocator(const std::shared_ptr<Allocator> &allocator);
```

设置Allocator，Allocator定义了用于动态内存分配和释放的内存池，该选项仅MindSpore lite有效。

- 参数

    - `allocator`: Allocator指针。

#### GetAllocator

```cpp
std::shared_ptr<Allocator> GetAllocator() const;
```

获取当前Allocator设置。

- 返回值

  当前Allocator的指针。

#### MutableDeviceInfo

```cpp
std::vector<std::shared_ptr<DeviceInfoContext>> &MutableDeviceInfo();
```

修改该context下的[DeviceInfoContext](#deviceinfocontext)数组，仅mindspore lite支持数组中有多个成员是异构场景。

- 返回值

  存储DeviceInfoContext的vector的引用。

## DeviceInfoContext

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

DeviceInfoContext类定义不同硬件设备的环境信息。

### 公有成员函数

#### GetDeviceType

```cpp
virtual enum DeviceType GetDeviceType() const = 0
```

获取该DeviceInfoContext的类型。

- 返回值

  该DeviceInfoContext的类型。

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

在打开`-fno-rtti`编译选项的情况下提供类似RTTI的功能，将DeviceInfoContext转换为`T`类型的指针，若转换失败返回`nullptr`。

- 返回值

  转换后`T`类型的指针，若转换失败则为`nullptr`。

## CPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

派生自[DeviceInfoContext](#deviceinfocontext)，模型运行在CPU上的配置，仅mindspore lite支持该选项。

### 公有成员函数

| 函数                                                                                                        | 说明                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetThreadAffinity(int mode)`          | 设置线程亲和性模式<br><br> - `mode`: 0：无亲和性， 1：大核优先， 2：小核优先。 |
| `int GetThreadAffinity() const`                                | - 返回值: 已配置的线程亲和性模式                                                                                                                                                    |
| `void SetEnableFP16(bool is_fp16)`                   | 用于指定是否以FP16精度进行推理<br><br> - `is_fp16`: 是否以FP16精度进行推理                                                                                                                                                            |
| `bool GetEnableFP16() const`                                       | - 返回值: 已配置的精度模式                                                                                                                                                                                                                               |

## MaliGPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

派生自[DeviceInfoContext](#deviceinfocontext)，模型运行在GPU上的配置，仅mindspore lite支持该选项。

### 公有成员函数

| 函数                                                                                                        | 说明                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetEnableFP16(bool is_fp16)`                   | 用于指定是否以FP16精度进行推理<br><br> - `is_fp16`: 是否以FP16精度进行推理                                                                                                                                                            |
| `bool GetEnableFP16() const`                                       | - 返回值: 已配置的精度模式                                                                                                                                                                                                                               |

## KirinNPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

派生自[DeviceInfoContext](#deviceinfocontext)，模型运行在NPU上的配置，仅mindspore lite支持该选项。

### 公有成员函数

| 函数                                                                                                        | 说明                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetFrequency(int frequency)`                   | 用于指定NPU频率<br><br> - `frequency`: 设置为1（低功耗）、2（均衡）、3（高性能）、4（极致性能），默认为3                                                                                                                                                            |
| `int GetFrequency() const`                                       | - 返回值: 已配置的NPU频率模式                                                                                                                                                                                                                               |

## NvidiaGPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

派生自[DeviceInfoContext](#deviceinfocontext)，模型运行在GPU上的配置，mindspore lite不支持该选项。

### 公有成员函数

| 函数                                                                                                        | 说明                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetDeviceID(uint32_t device_id)`                   | 用于指定设备ID<br><br> - `device_id`: 设备ID                                                                                                                                                            |
| `uint32_t GetDeviceID() const`                                       | - 返回值: 已配置的设备ID                                                                                                                                                                                                                               |

## Ascend910DeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

派生自[DeviceInfoContext](#deviceinfocontext)，模型运行在Ascend910上的配置，mindspore lite不支持该选项。

### 公有成员函数

| 函数                                                                                                        | 说明                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetDeviceID(uint32_t device_id)`                   | 用于指定设备ID<br><br> - `device_id`: 设备ID                                                                                                                                                            |
| `uint32_t GetDeviceID() const`                                       | - 返回值: 已配置的设备ID                                                                                                                                                                                                                               |

## Ascend310DeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/context.h)&gt;

派生自[DeviceInfoContext](#deviceinfocontext)，模型运行在Ascend310上的配置，mindspore lite不支持该选项。

### 公有成员函数

| 函数                                                                                                        | 说明                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetDeviceID(uint32_t device_id)`                   | 用于指定设备ID<br><br> - `device_id`: 设备ID                                                                                                                                                            |
| `uint32_t GetDeviceID() const`                                       | - 返回值: 已配置的设备ID                                                                                                                                                                                                                               |
| `void SetInsertOpConfigPath(const std::string &cfg_path)`          | 模型插入[AIPP](https://support.huaweicloud.com/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html)算子<br><br> - `cfg_path`: [AIPP](https://support.huaweicloud.com/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html)配置文件路径 |
| `std::string GetInsertOpConfigPath()`                                | - 返回值: 已配置的[AIPP](https://support.huaweicloud.com/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html)                                                                                                                                                    |
| `void SetInputFormat(const std::string &format)`                   | 指定模型输入formatt<br><br> - `format`: 可选有`"NCHW"`，`"NHWC"`等                                                                                                                                                            |
| `std::string GetInputFormat()`                                       | - 返回值: 已配置模型输入format                                                                                                                                                                                                                               |
| `void SetInputShape(const std::string &shape)`                     | 指定模型输入shape<br><br> - `shape`: 如`"input_op_name1:1,2,3,4;input_op_name2:4,3,2,1"`                                                                                                                           |
| `std::string GetInputShape()`                                        | - 返回值: 已配置模型输入shape                                                                                                                                                                                                                                |
| `void SetOutputType(enum DataType output_type)`                    | 指定模型输出type<br><br> - `output_type`: 仅支持uint8、fp16和fp32                                                                                                                                                            |
| `enum DataType GetOutputType()`                                      | - 返回值: 已配置模型输出type                                                                                                                                                                                                                                 |
| `void SetPrecisionMode(const std::string &precision_mode)`         | 配置模型精度模式<br><br> - `precision_mode`: 可选有`"force_fp16"`，`"allow_fp32_to_fp16"`，`"must_keep_origin_dtype"`或者`"allow_mix_precision"`，默认为`"force_fp16"`                                                       |
| `std::string GetPrecisionMode(t)`                                     | - 返回值: 已配置模型精度模式                                                                                                                                                                                                                                 |
| `void SetOpSelectImplMode(const std::string &op_select_impl_mode)` | 配置算子选择模式<br><br> - `op_select_impl_mode`: 可选有`"high_performance"`和`"high_precision"`，默认为`"high_performance"`                                                                                                 |
| `std::string GetOpSelectImplMode()`                                  | - 返回值: 已配置算子选择模式                                                                                                                                                                                                                                 |

## Serialization

\#include &lt;[serialization.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/serialization.h)&gt;

Serialization类汇总了模型文件读写的方法。

### 静态公有成员函数

#### Load

从文件加载模型，MindSpore Lite未提供此功能。

```cpp
Status Load(const std::string &file, ModelType model_type, Graph *graph);
```

- 参数

    - `file`: 模型文件路径。
    - `model_type`：模型文件类型，可选有`ModelType::kMindIR`、`ModelType::kOM`。
    - `graph`：输出参数，保存图数据的对象。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Load

从内存缓冲区加载模型。

```cpp
Status Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph);
```

- 参数

    - `model_data`：模型数据指针。
    - `data_size`：模型数据字节数。
    - `model_type`：模型文件类型，可选有`ModelType::kMindIR`、`ModelType::kOM`。
    - `graph`：输出参数，保存图数据的对象。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

## Model

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.2/include/api/model.h)&gt;

Model定义了MindSpore中的模型，便于计算图管理。

### 构造函数和析构函数

```cpp
Model();
~Model();
```

### 公有成员函数

#### Build

```cpp
Status Build(GraphCell graph, const std::shared_ptr<Context> &model_context);
```

将模型编译至可在Device上运行的状态。

- 参数

    - `graph`: `GraphCell`是`Cell`的一个派生，`Cell`目前没有开放使用。`GraphCell`可以由`Graph`构造，如`model.Build(GraphCell(graph), context)`。
    - `model_context`: 模型[Context](#context)。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

> `Build`之后对`model_context`的其他修改不再生效。

#### Predict

```cpp
Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);
```

推理模型。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `outputs`: 输出参数，按顺序排列的`vector`的指针，模型输出会按顺序填入该容器。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### GetInputs

```cpp
std::vector<MSTensor> GetInputs();
```

获取模型所有输入张量。

- 返回值

  包含模型所有输入张量的容器类型变量。

#### GetInputByTensorName

```cpp
MSTensor GetInputByTensorName(const std::string &tensor_name);
```

获取模型指定名字的输入张量。

- 返回值

  指定名字的输入张量，如果该名字不存在则返回非法张量。

#### GetOutputs

```cpp
std::vector<MSTensor> GetOutputs();
```

获取模型所有输出张量。

- 返回值

  包含模型所有输出张量的容器类型变量。

#### GetOutputTensorNames

```cpp
std::vector<std::string> GetOutputTensorNames();
```

获取模型所有输出张量的名字。

- 返回值

  包含模型所有输出张量名字的容器类型变量。

#### GetOutputByTensorName

```cpp
MSTensor GetOutputByTensorName(const std::string &tensor_name);
```

获取模型指定名字的输出张量。

- 返回值

  指定名字的输出张量，如果该名字不存在则返回非法张量。

#### Resize

```cpp
Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims);
```

调整已编译模型的输入形状。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `dims`: 输入形状，按输入顺序排列的由形状组成的`vector`，模型会按顺序依次调整张量形状。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### CheckModelSupport

```cpp
static bool CheckModelSupport(enum DeviceType device_type, ModelType model_type);
```

检查设备是否支持该模型。

- 参数

    - `device_type`: 设备类型，例如`kMaliGPU`。
    - `model_type`: 模型类型，例如`MindIR`。

- 返回值

    状态码。

## MSTensor

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/types.h)&gt;

`MSTensor`定义了MindSpore中的张量。

### 构造函数和析构函数

```cpp
MSTensor();
explicit MSTensor(const std::shared_ptr<Impl> &impl);
MSTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data, size_t data_len);
~MSTensor();
```

### 静态公有成员函数

#### CreateTensor

```cpp
MSTensor *CreateTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                       const void *data, size_t data_len) noexcept;
```

创建一个`MSTensor`对象，其数据需复制后才能由`Model`访问，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `type`：数据类型。
    - `shape`：形状。
    - `data`：数据指针，指向一段已开辟的内存。
    - `data`：数据长度，以字节为单位。

- 返回值

  `MStensor`指针。

#### CreateRefTensor

```cpp
MSTensor *CreateRefTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, void *data,
                          size_t data_len) noexcept;
```

创建一个`MSTensor`对象，其数据可以直接由`Model`访问，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `type`：数据类型。
    - `shape`：形状。
    - `data`：数据指针，指向一段已开辟的内存。
    - `data`：数据长度，以字节为单位。

- 返回值

  `MStensor`指针。

#### StringsToTensor

```cpp
MSTensor *StringsToTensor(const std::string &name, const std::vector<std::string> &str);
```

创建一个字符串类型的`MSTensor`对象，其数据需复制后才能由`Model`访问，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `str`：装有若干个字符串的`vector`容器。

- 返回值

  `MStensor`指针。

#### TensorToStrings

```cpp
std::vector<std::string> TensorToStrings(const MSTensor &tensor);
```

将字符串类型的`MSTensor`对象解析为字符串。

- 参数

    - `tensor`: 张量对象。

- 返回值

  装有若干个字符串的`vector`容器。

#### DestroyTensorPtr

```cpp
void DestroyTensorPtr(MSTensor *tensor) noexcept;
```

销毁一个由`Clone`、`StringsToTensor`、`CreateRefTensor`或`CreateTensor`所创建的对象，请勿用于销毁其他来源的`MSTensor`。

- 参数

    - `tensor`: 由`Clone`、`StringsToTensor`、`CreateRefTensor`或`CreateTensor`返回的指针。

### 公有成员函数

#### Name

```cpp
std::string Name() const;
```

获取`MSTensor`的名字。

- 返回值

  `MSTensor`的名字。

#### DataType

```cpp
enum DataType DataType() const;
```

获取`MSTensor`的数据类型。

- 返回值

  `MSTensor`的数据类型。

#### Shape

```cpp
const std::vector<int64_t> &Shape() const;
```

获取`MSTensor`的Shape。

- 返回值

  `MSTensor`的Shape。

#### ElementNum

```cpp
int64_t ElementNum() const;
```

获取`MSTensor`的元素个数。

- 返回值

  `MSTensor`的元素个数。

#### Data

```cpp
std::shared_ptr<const void> Data() const;
```

获取指向`MSTensor`中的数据拷贝的智能指针。

- 返回值

  指向`MSTensor`中的数据拷贝的智能指针。

#### MutableData

```cpp
void *MutableData();
```

获取`MSTensor`中的数据的指针。

- 返回值

  指向`MSTensor`中的数据的指针。

#### DataSize

```cpp
size_t DataSize() const;
```

获取`MSTensor`中的数据的以字节为单位的内存长度。

- 返回值

  `MSTensor`中的数据的以字节为单位的内存长度。

#### IsDevice

```cpp
bool IsDevice() const;
```

判断`MSTensor`中是否在设备上。

- 返回值

  `MSTensor`中是否在设备上。

#### Clone

```cpp
MSTensor *Clone() const;
```

拷贝一份自身的副本。

- 返回值

  指向深拷贝副本的指针，必须与`DestroyTensorPtr`成对使用。

#### operator==(std::nullptr_t)

```cpp
bool operator==(std::nullptr_t) const;
```

判断`MSTensor`是否合法。

- 返回值

  `MSTensor`是否合法。

## KernelCallBack

\#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/include/ms_tensor.h)&gt;

```cpp
using KernelCallBack = std::function<bool(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs, const CallBackParam &opInfo)>
```

一个函数包装器。KernelCallBack 定义了指向回调函数的指针。

## CallBackParam

\#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/include/ms_tensor.h)&gt;

一个结构体。CallBackParam定义了回调函数的输入参数。

### 公有属性

#### node_name

```cpp
node_name
```

**string** 类型变量。节点名参数。

#### node_type

```cpp
node_type
```

**string** 类型变量。节点类型参数。
