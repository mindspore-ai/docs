# mindspore::api

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_zh_cn/api.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Context

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/context.h)&gt;

Context类用于保存执行中的环境变量。

### 静态公有成员函数

#### Instance

```cpp
static Context &Instance();
```

获取MindSpore Context实例对象。

### 公有成员函数

#### GetDeviceTarget

```cpp
const std::string &GetDeviceTarget() const;
```

获取当前目标Device类型。

- 返回值

  当前DeviceTarget的类型。

#### GetDeviceID

```cpp
uint32_t GetDeviceID() const;
```

获取当前Device ID。

- 返回值

  当前Device ID。

#### SetDeviceTarget

```cpp
Context &SetDeviceTarget(const std::string &device_target);
```

配置目标Device。

- 参数

    - `device_target`: 将要配置的目标Device，可选有`kDeviceTypeAscend310`、`kDeviceTypeAscend910`。

- 返回值

  该MindSpore Context实例对象。

#### SetDeviceID

```cpp
Context &SetDeviceID(uint32_t device_id);
```

获取当前Device ID。

- 参数

    - `device_id`: 将要配置的Device ID。

- 返回值

  该MindSpore Context实例对象。

## Serialization

\#include &lt;[serialization.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/serialization.h)&gt;

Serialization类汇总了模型文件读写的方法。

### 静态公有成员函数

#### LoadModel

- 参数

    - `file`: 模型文件路径。
    - `model_type`：模型文件类型，可选有`ModelType::kMindIR`、`ModelType::kOM`。

- 返回值

  保存图数据的对象。

## Model

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/model.h)&gt;

Model定义了MindSpore中的模型，便于计算图管理。

### 构造函数和析构函数

```cpp
Model(const GraphCell &graph);
~Model();
```

`GraphCell`是`Cell`的一个派生，`Cell`目前没有开放使用。`GraphCell`可以由`Graph`构造，如`Model model(GraphCell(graph))`。

### 公有成员函数

#### Build

```cpp
Status Build(const std::map<std::string, std::string> &options);
```

将模型编译至可在Device上运行的状态。

- 参数

    - `options`: 模型编译选项，key为选项名，value为对应选项，支持的options有：

| Key | Value |
| --- | --- |
| kModelOptionInsertOpCfgPath | [AIPP](https://support.huaweicloud.com/adevg-ms-atlas200dkappc32/atlasadm_01_0023.html)配置文件路径 |
| kModelOptionInputFormat | 手动指定模型输入format，可选有`"NCHW"`，`"NHWC"`等 |
| kModelOptionInputShape | 手动指定模型输入shape，如`"input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1"` |
| kModelOptionOutputType | 手动指定模型输出type，如`"FP16"`，`"UINT8"`等，默认为`"FP32"` |
| kModelOptionPrecisionMode | 模型精度模式，可选有`"force_fp16"`，`"allow_fp32_to_fp16"`，`"must_keep_origin_dtype"`或者`"allow_mix_precision"`，默认为`"force_fp16"` |
| kModelOptionOpSelectImplMode | 算子选择模式，可选有`"high_performance"`和`"high_precision"`，默认为`"high_performance"` |

- 返回值

  状态码。

#### Predict

```cpp
Status Predict(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs);
```

推理模型。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `outputs`: 输出参数，按顺序排列的`vector`的指针，模型输出会按顺序填入该容器。

- 返回值

  状态码。

#### GetInputsInfo

```cpp
Status GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes, std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const;
```

获取模型输入信息。

- 参数

    - `names`: 可选输出参数，模型输入按顺序排列的`vector`的指针，模型输入的name会按顺序填入该容器，传入`nullptr`则表示不获取该属性。
    - `shapes`: 可选输出参数，模型输入按顺序排列的`vector`的指针，模型输入的shape会按顺序填入该容器，传入`nullptr`则表示不获取该属性。
    - `data_types`: 可选输出参数，模型输入按顺序排列的`vector`的指针，模型输入的数据类型会按顺序填入该容器，传入`nullptr`则表示不获取该属性。
    - `mem_sizes`: 可选输出参数，模型输入按顺序排列的`vector`的指针，模型输入的以字节为单位的内存长度会按顺序填入该容器，传入`nullptr`则表示不获取该属性。

- 返回值

  状态码。

#### GetOutputsInfo

```cpp
Status GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes, std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const;
```

获取模型输出信息。

- 参数

    - `names`: 可选输出参数，模型输出按顺序排列的`vector`的指针，模型输出的name会按顺序填入该容器，传入`nullptr`则表示不获取该属性。
    - `shapes`: 可选输出参数，模型输出按顺序排列的`vector`的指针，模型输出的shape会按顺序填入该容器，传入`nullptr`则表示不获取该属性。
    - `data_types`: 可选输出参数，模型输出按顺序排列的`vector`的指针，模型输出的数据类型会按顺序填入该容器，传入`nullptr`则表示不获取该属性。
    - `mem_sizes`: 可选输出参数，模型输出按顺序排列的`vector`的指针，模型输出的以字节为单位的内存长度会按顺序填入该容器，传入`nullptr`则表示不获取该属性。

- 返回值

  状态码。

## Tensor

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore/blob/r1.1/include/api/types.h)&gt;

### 构造函数和析构函数

```cpp
Tensor();
Tensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data, size_t data_len);
~Tensor();
```

### 静态公有成员函数

#### GetTypeSize

```cpp
static int GetTypeSize(api::DataType type);
```

获取数据类型的内存长度，以字节为单位。

- 参数

    - `type`: 数据类型。

- 返回值

  内存长度，单位是字节。

### 公有成员函数

#### Name

```cpp
const std::string &Name() const;
```

获取Tensor的名字。

- 返回值

  Tensor的名字。

#### DataType

```cpp
api::DataType DataType() const;
```

获取Tensor的数据类型。

- 返回值

  Tensor的数据类型。

#### Shape

```cpp
const std::vector<int64_t> &Shape() const;
```

获取Tensor的Shape。

- 返回值

  Tensor的Shape。

#### SetName

```cpp
void SetName(const std::string &name);
```

设置Tensor的名字。

- 参数

    - `name`: 将要设置的name。

#### SetDataType

```cpp
void SetDataType(api::DataType type);
```

设置Tensor的数据类型。

- 参数

    - `type`: 将要设置的type。

#### SetShape

```cpp
void SetShape(const std::vector<int64_t> &shape);
```

设置Tensor的Shape。

- 参数

    - `shape`: 将要设置的shape。

#### Data

```cpp
const void *Data() const;
```

获取Tensor中的数据的const指针。

- 返回值

  指向Tensor中的数据的const指针。

#### MutableData

```cpp
void *MutableData();
```

获取Tensor中的数据的指针。

- 返回值

  指向Tensor中的数据的指针。

#### DataSize

```cpp
size_t DataSize() const;
```

获取Tensor中的数据的以字节为单位的内存长度。

- 返回值

  Tensor中的数据的以字节为单位的内存长度。

#### ResizeData

```cpp
bool ResizeData(size_t data_len);
```

重新调整Tensor的内存大小。

- 参数

    - `data_len`: 调整后的内存字节数。

- 返回值

  bool值表示是否成功。

#### SetData

```cpp
bool SetData(const void *data, size_t data_len);
```

重新调整Tensor的内存数据。

- 参数

    - `data`: 源数据内存地址。
    - `data_len`: 源数据内存长度。

- 返回值

  bool值表示是否成功。

#### ElementNum

```cpp
int64_t ElementNum() const;
```

获取Tensor中元素的个数。

- 返回值

    Tensor中的元素个数

#### Clone

```cpp
Tensor Clone() const;
```

拷贝一份自身的副本。

- 返回值

  深拷贝的副本。
