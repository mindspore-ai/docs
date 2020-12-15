# mindspore::lite

<a href="https://gitee.com/mindspore/docs/blob/master/docs/api_cpp/source_en/lite.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Allocator

Allocator defines a memory pool for dynamic memory malloc and memory free.

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/context.h)&gt;

## Context

Context is defined for holding environment variables during runtime.

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/context.h)&gt;

### Constructors & Destructors

#### Context

```cpp
Context()
```

Constructor of MindSpore Lite Context using default value for parameters.

#### ~Context

```cpp
~Context()
```

Destructor of MindSpore Lite Context.

### Public Attributes

#### vendor_name_

```cpp
vendor_name_
```

A **string** value. Describes the vendor information.

#### thread_num_

```cpp
thread_num_
```

An **int** value. Defaults to **2**. Thread number config for thread pool.

#### allocator

```cpp
allocator
```

A **pointer** pointing to [**Allocator**](https://www.mindspore.cn/doc/api_cpp/en/master/lite.html#allocator).

#### device_list_

```cpp
device_list_
```

A [**DeviceContextVector**](https://www.mindspore.cn/doc/api_cpp/en/master/lite.html#devicecontextvector) contains [**DeviceContext**](https://www.mindspore.cn/doc/api_cpp/en/master/lite.html#devicecontext) variables.

> Only CPU and GPU are supported now. If GPU device context is set, use GPU device first, otherwise use CPU device first.

## PrimitiveC

Primitive is defined as prototype of operator.

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/model.h)&gt;

## Model

Model defines model in MindSpore Lite for managing graph.

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/model.h)&gt;

### Destructors

#### ~Model

```cpp
virtual ~Model()
```

Destructor of MindSpore Lite Model.

### Public Member Functions

#### Free

```cpp
void Free()
```

Free MetaGraph in MindSpore Lite Model to reduce memory usage during inference.

#### Destroy

```cpp
void Destroy()
```

Destroy all temporary memory in MindSpore Lite Model.

### Static Public Member Functions

#### Import

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

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/context.h)&gt;

### Public Attributes

#### MID_CPU

```cpp
MID_CPU = 2
```

Bind middle cpu first.

#### HIGHER_CPU

```cpp
HIGHER_CPU = 1
```

Bind higher cpu first.

#### NO_BIND

```cpp
NO_BIND = 0
```

No bind.

## DeviceType

An **enum** type. DeviceType defined for holding user's preferred backend.

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/context.h)&gt;

### Public Attributes

#### DT_CPU

```cpp
DT_CPU = 0
```

CPU device type.

#### DT_GPU

```cpp
DT_GPU = 1
```

GPU device type.

#### DT_NPU

```cpp
DT_NPU = 2
```

NPU device type, not supported yet.

## Version

```cpp
std::string Version()
```

\#include &lt;[version.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/version.h)&gt;

Global method to get a version string.

- Returns

    The version string of MindSpore Lite.

## StringsToMSTensor

```cpp
int StringsToMSTensor(const std::vector<std::string> &inputs, tensor::MSTensor *tensor)
```

Global method to store strings into MSTensor.

- Returns

    STATUS, STATUS is defined in errorcode.h.

## MSTensorToStrings

```cpp
std::vector<std::string> MSTensorToStrings(const tensor::MSTensor *tensor)
```

Global method to get strings from MSTensor.

- Returns

    The vector of strings.

## DeviceContextVector

A **vector** contains [**DeviceContext**](https://www.mindspore.cn/doc/api_cpp/en/master/lite.html#devicecontext) variable.

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/context.h)&gt;

## DeviceContext

DeviceContext defines different device contexts.

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/context.h)&gt;

### Public Attributes

#### device_type_

```cpp
device_type_
```

Defaults to **DT_CPU**. **enum** type. DeviceType is defined for holding user's cpu backend.

#### device_info_

```cpp
device_info_
```

 An **union** value, contains [**CpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/en/master/lite.html#cpudeviceinfo) and [**GpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/en/master/lite.html#gpudeviceinfo)

## DeviceInfo

An **union** value. DeviceInfo is defined for backend's configuration information.

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/context.h)&gt;

### Public Attributes

#### cpu_device_info_

```cpp
cpu_device_info_
```

[**CpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/en/master/lite.html#cpudeviceinfo) defined for CPU's configuration information.

#### gpu_device_info_

```cpp
gpu_device_info_
```

[**GpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/en/master/lite.html#gpudeviceinfo) defined for GPU's configuration information.

## CpuDeviceInfo

CpuDeviceInfo is defined for CPU's configuration information.

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/context.h)&gt;

### Public Attributes

#### enable_float16_

```cpp
enable_float16_
```

A **bool** value. Defaults to **false**. Prior enable GPU float16 inference.

> Enabling float16 inference may cause low precision inference，because some variables may exceed the range of float16 during forwarding.

#### cpu_bind_mode_

```cpp
cpu_bind_mode_
```

A [**CpuBindMode**](https://www.mindspore.cn/doc/api_cpp/en/master/lite.html#cpubindmode) **enum** variable. Defaults to **MID_CPU**.

## GpuDeviceInfo

GpuDeviceInfo is defined for GPU's configuration information.

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/context.h)&gt;

### Public Attributes

#### enable_float16_

```cpp
enable_float16_
```

A **bool** value. Defaults to **false**. Prior enable GPU float16 inference.

> Enabling float16 inference may cause low precision inference，because some variables may exceed the range of float16 during forwarding.

## TrainModel

TrainModel defines a class that allows to import and export the MindSpore trainable model.

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/model.h)&gt;

### Constructors & Destructors

#### ~TrainModel

```cpp
virtual ~TrainModel();
```

Class destructor, free all memory.

### Public Member Functions

#### Import

```cpp
static TrainModel *Import(const char *model_buf, size_t size);
```

Static method to create a TrainModel object.

- Parameters

    - `model_buf`: A buffer that was read from a MS model file.

    - `size`: Length of the buffer.

- Returns  

    Pointer to MindSpore Lite TrainModel.

#### Free

```cpp
void Free() override;
```

Free meta graph related data.

#### ExportBuf

```cpp
char *ExportBuf(char *buf, size_t *len) const;
```

Export Model into a buffer.

- Parameters

    - `buf`: The buffer to be exported into. If it is equal to nullptr, `buf` will be allocated.

    - `len`: Size of the pre-allocated buffer and the returned size of the exported buffer.

- Returns  

    Pointer to buffer with exported model.

### Public Attributes

#### buf_size_

```cpp
size_t buf_size_;
```

The length of the buffer with exported model.
