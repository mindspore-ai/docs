# mindspore::lite

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_en/lite.md)

## Allocator

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

Allocator defines a memory pool for dynamic memory malloc and memory free.

## Context

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

Context is defined for holding environment variables during runtime.

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

A **string** value. Describes the vendor information. This attribute is used to distinguish from different vendors.

#### thread_num_

```cpp
thread_num_
```

An **int** value. Defaults to **2**. Thread number config for thread pool.

#### allocator

```cpp
allocator
```

A **pointer** pointing to [**Allocator**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/lite.html#allocator).

#### device_list_

```cpp
device_list_
```

A [**DeviceContextVector**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/lite.html#devicecontextvector) contains [**DeviceContext**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/lite.html#devicecontext) variables.

> CPU, GPU and NPU are supported now. If GPU device context is set and GPU is supported in the current device, use GPU device first, otherwise use CPU device first. If NPU device context is set and GPU is supported in the current device, use NPU device first, otherwise use CPU device first.

## PrimitiveC

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/model.h)&gt;

Primitive is defined as prototype of operator.

## Model

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/model.h)&gt;

Model defines model in MindSpore Lite for managing graph.

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

Free all temporary memory in MindSpore Lite Model.

### Static Public Member Functions

#### Import

```cpp
static Model *Import(const char *model_buf, size_t size)
```

Static method to create a Model pointer.

- Parameters

    - `model_buf`: Defines the buffer read from a model file.

    - `size`: variable. Defines the byte number of model buffer.

- Returns  

    Pointer that points to the MindSpore Lite Model.

## CpuBindMode

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

An **enum** type. CpuBindMode is defined for holding arguments of the bind CPU strategy.

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

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

An **enum** type. DeviceType is defined for holding user's preferred backend.

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

NPU device type.

## Version

\#include &lt;[version.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/version.h)&gt;

```cpp
std::string Version()
```

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

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

A **vector** contains [**DeviceContext**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/lite.html#devicecontext) variable.

## DeviceContext

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

DeviceContext defines different device contexts.

### Public Attributes

#### device_type_

```cpp
device_type_
```

An **enum** type. Defaults to **DT_CPU**. DeviceType is defined for holding user’s CPU backend.

#### device_info_

```cpp
device_info_
```

An **union** value, contains [**CpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/lite.html#cpudeviceinfo) ,  [**GpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/lite.html#gpudeviceinfo) and [**NpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/lite.html#npudeviceinfo) .

## DeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

An **union** value. DeviceInfo is defined for backend's configuration information.

### Public Attributes

#### cpu_device_info_

```cpp
cpu_device_info_
```

[**CpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/lite.html#cpudeviceinfo) is defined for CPU's configuration information.

#### gpu_device_info_

```cpp
gpu_device_info_
```

[**GpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/lite.html#gpudeviceinfo) is defined for GPU's configuration information.

```cpp
npu_device_info_
```

[**GpuDeviceInfo**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/lite.html#gpudeviceinfo) is defined for NPU's configuration information.

## CpuDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

CpuDeviceInfo is defined for CPU's configuration information.

### Public Attributes

#### enable_float16_

```cpp
enable_float16_
```

A **bool** value. Defaults to **false**. This attribute enables to perform the GPU float16 inference.

> Enabling float16 inference may cause low precision inference，because some variables may exceed the range of float16 during forwarding.

#### cpu_bind_mode_

```cpp
cpu_bind_mode_
```

A [**CpuBindMode**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/lite.html#cpubindmode) **enum** variable. Defaults to **MID_CPU**.

## GpuDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

GpuDeviceInfo is defined for GPU's configuration information.

### Public Attributes

#### enable_float16_

```cpp
enable_float16_
```

A **bool** value. Defaults to **false**. This attribute enables to perform the GPU float16 inference.

> Enabling float16 inference may cause low inference precision, because some variables may exceed the range of float16 during forwarding.

## NpuDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/context.h)&gt;

NpuDeviceInfo is defined for NPU's configuration information.

```cpp
frequency_
```

A **int** value. Defaults to **3**. This attribute is used to set the NPU frequency, which can be set to 1 (low power consumption), 2 (balanced), 3 (high performance), 4 (extreme performance).

## TrainModel

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/model.h)&gt;

Inherited from Model, TrainModel defines a class that allows to import and export the MindSpore trainable model.

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
