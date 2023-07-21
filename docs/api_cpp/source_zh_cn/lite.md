# mindspore::lite

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/api_cpp/source_zh_cn/lite.md)

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/lite/include/context.h)&gt;

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/lite/include/model.h)&gt;

\#include &lt;[version.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/lite/include/version.h)&gt;

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
float16_priority
```

**bool**值，默认为**false**，用于使能float16 推理。

> 使能float16推理可能会导致模型推理精度下降，因为在模型推理的中间过程中，有些变量可能会超出float16的数值范围。

```cpp
device_type
```

[**DeviceType**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.0/lite.html#devicetype)枚举类型。默认为**DT_CPU**，用于设置设备信息。

```cpp
thread_num_
```

**int** 值，默认为**2**，设置线程数。

```cpp
allocator
```

指针类型，指向内存分配器[**Allocator**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.0/lite.html#allocator)的指针。

```cpp
cpu_bind_mode_
```

[**CpuBindMode**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.0/lite.html#cpubindmode)枚举类型，默认为**MID_CPU**。

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
MID_CPU = -1
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
DT_CPU = -1
```

设备为CPU。

```cpp
DT_GPU = 1
```

设备为GPU。

```cpp
DT_NPU = 0
```

设备为NPU，暂不支持。

## Version

```cpp
std::string Version()
```

全局方法，用于获取版本的字符串。

- 返回值

    MindSpore Lite版本的字符串。
