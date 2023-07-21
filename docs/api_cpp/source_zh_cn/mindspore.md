# mindspore

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_zh_cn/mindspore.md)

## Context

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/context.h)&gt;

Context类用于保存执行中的环境变量。包含GlobalContext与ModelContext两个派生类。

## GlobalContext : Context

GlobalContext定义了执行时的全局变量。

### 静态公有成员函数

#### GetGlobalContext

```cpp
static std::shared_ptr<Context> GetGlobalContext();
```

返回GlobalContext单例。

- 返回值

  指向GlobalContext单例的智能指针。

#### SetGlobalDeviceTarget

```cpp
static void SetGlobalDeviceTarget(const std::string &device_target);
```

配置目标Device。

- 参数
    `device_target`: 将要配置的目标Device，可选有`kDeviceTypeAscend310`、`kDeviceTypeAscend910`。

#### GetGlobalDeviceTarget

```cpp
static std::string GetGlobalDeviceTarget();
```

获取已配置的Device。

- 返回值
  已配置的目标Device。

#### SetGlobalDeviceID

```cpp
static void SetGlobalDeviceID(const unit32_t &device_id);
```

配置Device ID。

- 参数
    `device_id`: 将要配置的Device ID。

#### GetGlobalDeviceID

```cpp
static uint32_t GetGlobalDeviceID();
```

获取已配置的Device ID。

- 返回值
  已配置的Device ID。

## ModelContext : Context

### 静态公有成员函数

| 函数                                                                                                        | 说明                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `void SetInsertOpConfigPath(const std::shared_ptr<Context> &context, const std::string &cfg_path)`          | 模型插入[AIPP](https://support.huaweicloud.com/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html)算子<br><br> - `context`: 将要修改的context<br><br> - `cfg_path`: [AIPP](https://support.huaweicloud.com/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html)配置文件路径 |
| `std::string GetInsertOpConfigPath(const std::shared_ptr<Context> &context)`                                | - 返回值: 已配置的[AIPP](https://support.huaweicloud.com/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html)                                                                                                                                                    |
| `void SetInputFormat(const std::shared_ptr<Context> &context, const std::string &format)`                   | 指定模型输入format<br><br> - `context`: 将要修改的context<br><br> - `format`: 可选有`"NCHW"`，`"NHWC"`等                                                                                                                                                            |
| `std::string GetInputFormat(const std::shared_ptr<Context> &context)`                                       | - 返回值: 已配置模型输入format                                                                                                                                                                                                                               |
| `void SetInputShape(const std::shared_ptr<Context> &context, const std::string &shape)`                     | 指定模型输入shape<br><br> - `context`: 将要修改的context<br><br> - `shape`: 如`"input_op_name1:1,2,3,4;input_op_name2:4,3,2,1"`                                                                                                                           |
| `std::string GetInputShape(const std::shared_ptr<Context> &context)`                                        | - 返回值: 已配置模型输入shape                                                                                                                                                                                                                                |
| `void SetOutputType(const std::shared_ptr<Context> &context, enum DataType output_type)`                    | 指定模型输出type<br><br> - `context`: 将要修改的context<br><br> - `output_type`: 仅支持uint8、fp16和fp32                                                                                                                                                            |
| `enum DataType GetOutputType(const std::shared_ptr<Context> &context)`                                      | - 返回值: 已配置模型输出type                                                                                                                                                                                                                                 |
| `void SetPrecisionMode(const std::shared_ptr<Context> &context, const std::string &precision_mode)`         | 配置模型精度模式<br><br> - `context`: 将要修改的context<br><br> - `precision_mode`: 可选有`"force_fp16"`，`"allow_fp32_to_fp16"`，`"must_keep_origin_dtype"`或者`"allow_mix_precision"`，默认为`"force_fp16"`                                                       |
| `std::string GetPrecisionMode(const std::shared_ptr<Context> &context)`                                     | - 返回值: 已配置模型精度模式                                                                                                                                                                                                                                 |
| `void SetOpSelectImplMode(const std::shared_ptr<Context> &context, const std::string &op_select_impl_mode)` | 配置算子选择模式<br><br> - `context`: 将要修改的context<br><br> - `op_select_impl_mode`: 可选有`"high_performance"`和`"high_precision"`，默认为`"high_performance"`                                                                                                 |
| `std::string GetOpSelectImplMode(const std::shared_ptr<Context> &context)`                                  | - 返回值: 已配置算子选择模式                                                                                                                                                                                                                                 |

## Serialization

\#include &lt;[serialization.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/serialization.h)&gt;

Serialization类汇总了模型文件读写的方法。

### 静态公有成员函数

#### LoadModel

```cpp
static Graph LoadModel(const std::string &file, ModelType model_type);
```

从文件加载模型。MindSpore Lite未提供此功能。

- 参数

    - `file`：模型文件路径。
    - `model_type`：模型文件类型，可选有`ModelType::kMindIR`，`ModelType::kOM`。

- 返回值

  保存图数据的`Graph`实例。

```cpp
static Graph LoadModel(const void *model_data, size_t data_size, ModelType model_type);
```

从内存缓冲区加载模型。

- 参数

    - `model_data`：已读取模型文件的缓存区。
    - `data_size`：缓存区大小。
    - `model_type`：模型文件类型，可选有`ModelType::kMindIR`、`ModelType::kOM`。

- 返回值

  保存图数据的`Graph`实例。

## Model

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/model.h)&gt;

Model定义了MindSpore中的模型，便于计算图管理。

### 构造函数和析构函数

```cpp
explicit Model(const GraphCell &graph, const std::shared_ptr<Context> &model_context);
explicit Model(const std::vector<Output> &network, const std::shared_ptr<Context> &model_context);
~Model();
```

`GraphCell`是`Cell`的一个派生，`Cell`目前没有开放使用。`GraphCell`可以由`Graph`构造，如`Model model(GraphCell(graph))`。

`Context`表示运行时的[模型配置](#modelcontext-contextfor-mindspore)。

### 公有成员函数

#### Build

```cpp
Status Build();
```

将模型编译至可在Device上运行的状态。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Predict

```cpp
Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);
```

执行推理。

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

#### GetOutputs

```cpp
std::vector<MSTensor> GetOutputs();
```

获取模型所有输出张量。

- 返回值

  包含模型所有输出张量的容器类型变量。

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
static bool CheckModelSupport(const std::string &device_type, ModelType model_type);
```

检查设备是否支持该模型。

- 参数

    - `device_type`: 设备名称，例如`Ascend310`。
    - `model_type`: 模型类型，例如`MindIR`。

- 返回值

  状态码。

## MSTensor

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/types.h)&gt;

`MSTensor`定义了MindSpore中的Tensor。

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
static MSTensor CreateTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                             const void *data, size_t data_len) noexcept;
```

创建一个`MSTensor`对象，其数据需复制后才能由`Model`访问。

- 参数

    - `name`: 名称。
    - `type`：数据类型。
    - `shape`：形状。
    - `data`：数据指针，指向一段已开辟的内存。
    - `data`：数据长度，以字节为单位。

- 返回值

  `MStensor`实例。

#### CreateRefTensor

```cpp
static MSTensor CreateRefTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, void *data,
                                size_t data_len) noexcept;
```

创建一个`MSTensor`对象，其数据可以直接由`Model`访问。

- 参数

    - `name`: 名称。
    - `type`：数据类型。
    - `shape`：形状。
    - `data`：数据指针，指向一段已开辟的内存。
    - `data`：数据长度，以字节为单位。

- 返回值

  `MStensor`实例。

### 公有成员函数

#### Name

```cpp
const std::string &Name() const;
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
MSTensor Clone() const;
```

拷贝一份自身的副本。

- 返回值

  深拷贝的副本。

#### operator==(std::nullptr_t)

```cpp
bool operator==(std::nullptr_t) const;
```

判断`MSTensor`是否合法。

- 返回值

  `MSTensor`是否合法。

## CallBack

\#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/ms_tensor.h)&gt;

CallBack定义了MindSpore Lite中的回调函数。

### KernelCallBack

```cpp
using KernelCallBack = std::function<bool(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs, const CallBackParam &opInfo)>
```

一个函数包装器。KernelCallBack 定义了指向回调函数的指针。

### CallBackParam

一个结构体。CallBackParam定义了回调函数的输入参数。

#### 公有属性

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
