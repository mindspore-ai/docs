# mindspore::lite

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_zh_cn/lite.md)

## Allocator

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

Allocator类定义了一个内存池，用于动态地分配和释放内存。

## Context

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

Context类用于保存执行中的环境变量。

### 构造函数和析构函数

#### Context

```cpp
Context()
```

用默认参数构造MindSpore Lite Context 对象。

#### ~Context

```cpp
~Context()
```

MindSpore Lite Context 的析构函数。

### 公有属性

#### vendor_name_

```cpp
vendor_name_
```

**string**值，芯片厂商名字，用于区别不同的芯片厂商。

#### thread_num_

```cpp
thread_num_
```

**int**值，默认为**2**，设置线程数。

#### allocator

```cpp
allocator
```

**pointer**类型，指向内存分配器 [**Allocator**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#allocator) 的指针。

#### device_list_

```cpp
device_list_
```

[**DeviceContextVector**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#devicecontextvector) 类型, 元素为 [**DeviceContext**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#devicecontext) 的**vector**.

> 现在支持CPU、GPU和NPU。如果设置了GPU设备环境变量并且设备支持GPU，优先使用GPU设备，否则优先使用CPU设备。如果设置了NPU设备环境变量并且设备支持NPU，优先使用NPU设备，否则优先使用CPU设备。

## PrimitiveC

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/model.h)&gt;

PrimitiveC定义为算子的原型。

## Model

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/model.h)&gt;

Model定义了MindSpore Lite中的模型，便于计算图管理。

### 析构函数

#### ~Model

```cpp
~Model()
```

MindSpore Lite Model的析构函数。

### 公有成员函数

#### Destroy

```cpp
void Destroy()
```

释放Model内的所有过程中动态分配的内存。

#### Free

```cpp
void Free()
```

释放MindSpore Lite Model中的MetaGraph，用于减小运行时的内存。

### 静态公有成员函数

#### Import

```cpp
static Model *Import(const char *model_buf, size_t size)
```

创建Model指针的静态方法。

- 参数

    - `model_buf`: 定义了读取模型文件的缓存区。

    - `size`: 定义了模型缓存区的字节数。

- 返回值  

  指向MindSpore Lite的Model的指针。

## CpuBindMode

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

枚举类型，设置cpu绑定策略。

### 公有属性

#### MID_CPU

```cpp
MID_CPU = 2
```

优先中等CPU绑定策略。

#### HIGHER_CPU

```cpp
HIGHER_CPU = 1
```

优先高级CPU绑定策略。

#### NO_BIND

```cpp
NO_BIND = 0
```

不绑定。

## DeviceType

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

枚举类型，设置设备类型。

### 公有属性

#### DT_CPU

```cpp
DT_CPU = 0
```

设备为CPU。

#### DT_GPU

```cpp
DT_GPU = 1
```

设备为GPU。

#### DT_NPU

```cpp
DT_NPU = 2
```

设备为NPU。

## Version

\#include &lt;[version.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/version.h)&gt;

```cpp
std::string Version()
```

全局方法，用于获取版本的字符串。

- 返回值

    MindSpore Lite版本的字符串。

## StringsToMSTensor

```cpp
int StringsToMSTensor(const std::vector<std::string> &inputs, tensor::MSTensor *tensor)
```

全局方法，用于将字符串存入MSTensor。

- 返回值

    STATUS，STATUS在errorcode.h中定义。

## MSTensorToStrings

```cpp
std::vector<std::string> MSTensorToStrings(const tensor::MSTensor *tensor)
```

全局方法，用于从MSTensor获取字符串。

- 返回值

    字符串的vector。

## DeviceContextVector

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

元素为[**DeviceContext**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#devicecontext) 的**vector**。

## DeviceContext

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

DeviceContext类定义不同硬件设备的环境信息。

### 公有属性

#### device_type

```cpp
device_type
```

[**DeviceType**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#devicetype) 枚举类型。默认为**DT_CPU**，标明设备信息。

#### device_info_

```cpp
device_info_
```

**union**类型，包含 [**CpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#cpudeviceinfo) 、 [**GpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#gpudeviceinfo)  和 [**NpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#npudeviceinfo) 。

## DeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

**union**类型，设置不同硬件的环境变量。

### 公有属性

#### cpu_device_info_

```cpp
cpu_device_info_
```

[**CpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#cpudeviceinfo) 类型，配置CPU的环境变量。

#### gpu_device_info_

```cpp
gpu_device_info_
```

[**GpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#gpudeviceinfo) 类型，配置GPU的环境变量。

#### npu_device_info_

```cpp
npu_device_info_
```

[**NpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#npudeviceinfo) 类型，配置NPU的环境变量。

## CpuDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

CpuDeviceInfo类，配置CPU的环境变量。

### Public Attributes

#### enable_float16_

```cpp
enable_float16_
```

**bool**值，默认为**false**，用于使能float16 推理。

> 使能float16推理可能会导致模型推理精度下降，因为在模型推理的中间过程中，有些变量可能会超出float16的数值范围。

#### cpu_bind_mode_

```cpp
cpu_bind_mode_
```

[**CpuBindMode**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/lite.html#cpubindmode) 枚举类型，默认为**MID_CPU**。

## GpuDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

GpuDeviceInfo类，用来配置GPU的环境变量。

### 公有属性

#### enable_float16_

```cpp
enable_float16_
```

**bool**值，默认为**false**，用于使能float16 推理。

> 使能float16推理可能会导致模型推理精度下降，因为在模型推理的中间过程中，有些变量可能会超出float16的数值范围。

## NpuDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

NpuDeviceInfo类，用来配置NPU的环境变量。

### 公有属性

#### frequency

```cpp
frequency_
```

**int**值，默认为**3**，用来设置NPU频率，可设置为1（低功耗）、2（均衡）、3（高性能）、4（极致性能）。

## TrainModel

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/model.h)&gt;

继承于结构体Model，用于导入或导出训练模型。

### 析构函数

#### ~TrainModel

```cpp
virtual ~TrainModel();
```

虚析构函数。

### 公有成员函数

#### Import

```cpp
static TrainModel *Import(const char *model_buf, size_t size);
```

导入模型。

- 参数

    - `model_buf`: 指向存储读入MindSpore模型缓冲区的常量字符型指针。

    - `size`: 缓冲区大小。

- 返回值  

    返回一个指向MindSpore Lite训练模型(TrainModel)的指针。

#### ExportBuf

```cpp
char* ExportBuf(char *buf, size_t *len) const;
```

导出模型缓冲区。

- 参数

    - `buf`: 指向模型导出的目标缓冲区的指针，如果指针为空则自动分配一块内存。

    - `len`: 指向预分配缓冲区大小的指针。

- 返回值  

    返回一个指向存储导出模型缓冲区的字符指针。

#### Free

```cpp
void Free() override;
```

释放计算-图的元数据。

### 公有属性

#### buf_size_

```cpp
size_t buf_size_;
```

缓冲区大小。
