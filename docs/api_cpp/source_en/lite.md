# mindspore::lite

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/api_cpp/source_en/lite.md)

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/lite/include/context.h)&gt;

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/lite/include/model.h)&gt;

\#include &lt;[version.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/lite/include/version.h)&gt;

## Allocator

Allocator defines a memory pool for dynamic memory malloc and memory free.

## Context

Context is defined for holding environment variables during runtime.

### Constructors & Destructors

```cpp
Context()
```

Constructor of MindSpore Lite Context using default value for parameters.

```cpp
 ~Context()
```

Destructor of MindSpore Lite Context.

### Public Attributes

```cpp
float16_priority
```

A **bool** value. Defaults to **false**. Prior enable float16 inference.

> Enabling float16 inference may cause low precision inference，because some variables may exceed the range of float16 during forwarding.

```cpp
device_type
```

A [**DeviceType**](https://www.mindspore.cn/doc/api_cpp/en/r1.0/lite.html#devicetype) **enum** type. Defaults to **DT_CPU**. Using to specify the device.

```cpp
thread_num_
```

An **int** value. Defaults to **2**. Thread number config for thread pool.

```cpp
allocator
```

A **pointer** pointing to [**Allocator**](https://www.mindspore.cn/doc/api_cpp/en/r1.0/lite.html#allocator).

```cpp
cpu_bind_mode_
```

A [**CpuBindMode**](https://www.mindspore.cn/doc/api_cpp/en/r1.0/lite.html#cpubindmode) **enum** variable. Defaults to **MID_CPU**.

## PrimitiveC

Primitive is defined as prototype of operator.

## Model

Model defines model in MindSpore Lite for managing graph.

### Destructors

```cpp
virtual ~Model()
```

Destructor of MindSpore Lite Model.

### Public Member Functions

```cpp
void Free()
```

Free MetaGraph in MindSpore Lite Model to reduce memory usage during inference.

```cpp
void Destroy()
```

Destroy all temporary memory in MindSpore Lite Model.

### Static Public Member Functions

```cpp
static Model *Import(const char *model_buf, size_t size)
```

Static method to create a Model pointer.

- Parameters

    - `model_buf`: Define the buffer read from a model file.

    - `size`: variable. Define bytes number of model buffer.

- Returns

    Pointer of MindSpore Lite Model.

## CpuBindMode

An **enum** type. CpuBindMode defined for holding bind cpu strategy argument.

### Attributes

```cpp
MID_CPU = -1
```

Bind middle cpu first.

```cpp
HIGHER_CPU = 1
```

Bind higher cpu first.

```cpp
NO_BIND = 0
```

No bind.

## DeviceType

An **enum** type. DeviceType defined for holding user's preferred backend.

### Attributes

```cpp
DT_CPU = -1
```

CPU device type.

```cpp
DT_GPU = 1
```

GPU device type.

```cpp
DT_NPU = 0
```

NPU device type, not supported yet.

## Version

```cpp
std::string Version()
```

Global method to get a version string.

- Returns

    The version string of MindSpore Lite.
