# mindspore::lite

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/context.h)&gt;

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/model.h)&gt;

\#include &lt;[version.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/version.h)&gt;

## Allocator

Allocator类定义了一个内存池，用于动态地分配和释放内存。

## Context

Context类用于保存执行中的环境变量。

### 构造函数和析构函数

```cpp
Context()
```

用默认参数构造MindSpore Lite Context 对象。

```cpp
~Context()
```

MindSpore Lite Context 的析构函数。

### 公有属性

```cpp
vendor_name_
```

**string**值，芯片厂商名字，用于区别不同的芯片厂商。

```cpp
thread_num_
```

**int**值，默认为**2**，设置线程数。

```cpp
allocator
```

**pointer**类型，指向内存分配器 [**Allocator**](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#allocator) 的指针。

```cpp
device_list_
```

[**DeviceContextVector**](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#devicecontextvector) 类型, 元素为 [**DeviceContext**](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#devicecontext) 的**vector**. 

> 现在只支持CPU和GPU。如果设置了GPU设备环境变量，优先使用GPU设备，否则优先使用CPU设备。

## PrimitiveC

PrimitiveC定义为算子的原型。

## Model

Model定义了MindSpore Lite中的模型，便于计算图管理。

### 析构函数

```cpp
~Model()
```

MindSpore Lite Model的析构函数。

### 公有成员函数

```cpp
void Destroy()
```

释放Model内的所有过程中动态分配的内存。

```cpp
void Free()
```

释放MindSpore Lite Model中的MetaGraph，用于减小运行时的内存。

### 静态公有成员函数

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

枚举类型，设置cpu绑定策略。

### 属性

```cpp
MID_CPU = 2
```

优先中等CPU绑定策略。

```cpp
HIGHER_CPU = 1
```

优先高级CPU绑定策略。

```cpp
NO_BIND = 0
```

不绑定。

## DeviceType

枚举类型，设置设备类型。

### 属性

```cpp
DT_CPU = 0
```

设备为CPU。

```cpp
DT_GPU = 1
```

设备为GPU。

```cpp
DT_NPU = 2
```

设备为NPU，暂不支持。

## Version

```cpp
std::string Version()
```

全局方法，用于获取版本的字符串。

- 返回值

    MindSpore Lite版本的字符串。

   
## DeviceContextVector

元素为[**DeviceContext**](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#devicecontext) 的**vector**。

## DeviceContext

DeviceContext类定义不同硬件设备的环境信息。

### Public Attributes

```cpp
device_type
```

[**DeviceType**](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#devicetype) 枚举类型。默认为**DT_CPU**，标明设备信息。

```cpp
device_info_
```

**union**类型，包含[**CpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#cpudeviceinfo) 和[**GpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#gpudeviceinfo) 。

## DeviceInfo

**union**类型，设置不同硬件的环境变量。

### Public Attributes

```cpp
cpu_device_info_
```
[**CpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#cpudeviceinfo) 类型，配置CPU的环境变量。
```cpp
gpu_device_info_
```

[**GpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#gpudeviceinfo) 类型，配置GPU的环境变量。

## CpuDeviceInfo

CpuDeviceInfo类，配置CPU的环境变量。

### Public Attributes

```cpp
enable_float16_
```

**bool**值，默认为**false**，用于使能float16 推理。


> 使能float16推理可能会导致模型推理精度下降，因为在模型推理的中间过程中，有些变量可能会超出float16的数值范围。

```cpp
cpu_bind_mode_
```

[**CpuBindMode**](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#cpubindmode) 枚举类型，默认为**MID_CPU**。


## GpuDeviceInfo

GpuDeviceInfo类，用来配置GPU的环境变量。

### Public Attributes

```cpp
enable_float16_
```

**bool**值，默认为**false**，用于使能float16 推理。


> 使能float16推理可能会导致模型推理精度下降，因为在模型推理的中间过程中，有些变量可能会超出float16的数值范围。
