# mindspore

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/docs/lite/api/source_zh_cn/api_cpp/mindspore.md)

## 接口汇总

### 推理

| 类名                                               | 描述                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------|---------------------------------------------------|--------|--------|
| [Model](#model)                                  | MindSpore中的模型，便于计算图管理。                            | √      | √      |

### 运行环境配置

| 类名                                               | 描述                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------|---------------------------------------------------|--------|--------|
| [Context](#context)                              | 保存执行中的环境变量。                                       | √      | √      |
| [DeviceInfoContext](#deviceinfocontext)          | 不同硬件设备的环境信息。                                      | √      | √      |
| [CPUDeviceInfo](#cpudeviceinfo)                  | 模型运行在CPU上的配置。                   | √      | √      |
| [GPUDeviceInfo](#gpudeviceinfo)                  | 模型运行在GPU上的配置。                                     | √      | √      |
| [KirinNPUDeviceInfo](#kirinnpudeviceinfo)        | 模型运行在NPU上的配置。                   | ✕      | √      |
| [AscendDeviceInfo](#ascenddeviceinfo)            | 模型运行在Atlas 200/300/500推理产品、Atlas推理系列产品上的配置。                               | √      | √      |

### 并发推理

| 类名                                               | 描述                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------|---------------------------------------------------|--------|--------|
| [RunnerConfig](#runnerconfig)                    | 模型并发推理配置参数。                                       | √      | ✕      |
| [ModelParallelRunner](#modelparallelrunner)      | 模型并发推理类。                                          | √      | ✕      |

### 张量Tensor相关

| 类名                                               | 描述                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------|---------------------------------------------------|--------|--------|
| [MSTensor](#mstensor)                            | MindSpore中的张量。                                    | √      | √      |
| [QuantParam](#quantparam)                        | MSTensor中的一组量化参数。                                 | √      | √      |
| [mindspore::DataType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_datatype.html) | MindSpore MSTensor保存的数据支持的类型。 | √      | √      |
| [mindspore::Format](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_format.html) | MindSpore MSTensor保存的数据支持的排列格式。 | √      | √      |
| [Allocator](#allocator-1)                          | 内存管理基类。                                           | √      | √      |

### 模型分组

| 类名                                       | 描述                  | 云侧推理是否支持 | 端侧推理是否支持 |
|-------------------------------------------|-----------------------|--------|--------|
| [ModelGroup](#modelgroup)                 | 模型分组。              |    √   |  ✕     |

### 状态

| 类名                                               | 描述                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------|---------------------------------------------------|--------|--------|
| [Status](#status)                                | 返回状态类。                                            | √      | √      |

### 序列化保存与加载

| 类名                                               | 描述                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------|---------------------------------------------------|--------|--------|
| [Serialization](#serialization)                  | 汇总了模型文件读写的方法。                                     | √      | √      |
| [Buffer](#buffer)                                | Buff数据类。                                          | √      | √      |

### 版本查询

| 类名                                               | 描述                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------|---------------------------------------------------|--------|--------|
| [SchemaVersion](#schemaversion)                  | MindSpore Lite 执行推理时，模型文件的版本。    | ✕      | √      |
| [CharVersion](#charversion)                      | 获取字符vector形式的当前版本号。                        | ✕      | √      |
| [Version](#version)                              | 获取字符串形式的当前版本号。                        | ✕      | √      |

### 回调函数

| 类名                                               | 描述                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------|---------------------------------------------------|--------|--------|
| [MSKernelCallBack](#mskernelcallback)            | MindSpore回调函数包装器。               | √      | √      |
| [MSCallBackParam](#mscallbackparam)              | MindSpore回调函数的参数。               | √      | √      |

### MindSpore Lite 训练配置

| 类名                                               | 描述                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------|---------------------------------------------------|--------|--------|
| [TrainCfg](#traincfg)                            | MindSpore Lite训练配置类。            | ✕      | √      |
| [MixPrecisionCfg](#mixprecisioncfg)              | MindSpore Lite训练混合精度配置类。        | ✕      | √      |
| [AccuracyMetrics](#accuracymetrics)              | MindSpore Lite训练精度类。            | ✕      | √      |
| [Metrics](#metrics)                              | MindSpore Lite训练指标类。            | ✕      | √      |
| [TrainCallBack](#traincallback)                  | MindSpore Lite训练回调类。            | ✕      | √      |
| [TrainCallBackData](#traincallbackdata)          | 定义了训练回调的一组参数。                   | ✕      | √      |
| [CkptSaver](#ckptsaver)                          | MindSpore Lite训练模型文件保存类。        | ✕      | √      |
| [LossMonitor](#lossmonitor)                      | MindSpore Lite训练学习率调度类。         | ✕      | √      |
| [LRScheduler](#lrscheduler)                      | MindSpore Lite训练配置类。            | ✕      | √      |
| [StepLRLambda](#steplrlambda)                    | MindSpore Lite训练学习率的一组参数。       | ✕      | √      |
| [MultiplicativeLRLambda](#multiplicativelrlambda) | 每个epoch将学习率乘以一个因子。              | ✕      | √      |
| [TimeMonitor](#timemonitor)                      | MindSpore Lite训练时间监测类。          | ✕      | √      |
| [TrainAccuracy](#trainaccuracy)                  | MindSpore Lite训练学习率调度类。         | ✕      | √      |

### Delegate三方框架接入机制

| 类名                                               | 描述                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------|---------------------------------------------------|--------|--------|
| [Delegate](#delegate)                            | MindSpore Lite接入第三方AI框架的代理。     | ✕      | √      |
| [DelegateModel](#delegatemodel)                  | MindSpore Lite Delegate机制封装的模型。 | ✕      | √      |
| [KernelIter](#kerneliter)                        | MindSpore Lite 算子列表的迭代器。        | ✕      | √      |
| [CoreMLDelegate](#coremldelegate)               | MindSpore Lite接入CoreML框架的代理。    | ✕      | √      |

### 图容器

| 类名                                               | 描述                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------|---------------------------------------------------|--------|--------|
| [CellBase](#cellbase)                            | 容器基类。                                             | ✕      | √      |
| [Cell](#cell)                                    | 容器类。                                              | ✕      | √      |
| [GraphCell](#graphcell)                          | 图容器类。                                             | ✕      | √      |
| [Graph](#graph)                                  | 图类。                                               | ✕      | √      |

## Context

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/context.h)&gt;

Context类用于保存执行中的环境变量。

### 构造函数和析构函数

```cpp
Context()
~Context() = default;
```

### 公有成员函数

| 函数                                                                            | 云侧推理是否支持 | 端侧推理是否支持 |
|-------------------------------------------------------------------------------|--------|--------|
| [void SetThreadNum(int32_t thread_num)](#setthreadnum)     |    √    |    √    |
| [int32_t GetThreadNum() const](#getthreadnum)     |    √    |    √    |
| [void SetInterOpParallelNum(int32_t parallel_num)](#setinteropparallelnum)     |    √    |    √    |
| [int32_t GetInterOpParallelNum() const](#getinteropparallelnum)     |    √    |    √    |
| [void SetThreadAffinity(int mode)](#setthreadaffinity)     |    √    |    √    |
| [int GetThreadAffinityMode() const](#getthreadaffinitymode)     |    √    |    √    |
| [void SetThreadAffinity(const std::vector<int> &core_list)](#setthreadaffinity)     |    √    |    √    |
| [std::vector<int32_t> GetThreadAffinityCoreList() const](#getthreadaffinitycorelist)     |    √    |    √    |
| [void SetEnableParallel(bool is_parallel)](#setenableparallel)     |    ✕    |    √    |
| [bool GetEnableParallel() const](#getenableparallel)     |    ✕    |    √    |
| [void SetBuiltInDelegate(DelegateMode mode)](#setbuiltindelegate)     |    ✕    |    √    |
| [DelegateMode GetBuiltInDelegate() const](#getbuiltindelegate)     |    ✕    |    √    |
| [void SetDelegate(const std::shared_ptr<Delegate> &delegate)](#setdelegate)     |    ✕    |    √    |
| [std::shared_ptr<Delegate> GetDelegate() const](#getdelegate)     |    ✕    |    √    |
| [void set_delegate(const std::shared_ptr<AbstractDelegate> &delegate)](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#set-delegate)         | ✕      | √      |
| [std::shared_ptr<AbstractDelegate> get_delegate() const](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#get-delegate)     |    ✕    |    √    |
| [void SetMultiModalHW(bool float_mode)](#setmultimodalhw)     |    ✕    |    √    |
| [bool GetMultiModalHW() const](#getmultimodalhw)     |    ✕    |    √    |
| [std::vector<std::shared_ptr<DeviceInfoContext>> &MutableDeviceInfo()](#mutabledeviceinfo)     |    √    |    √    |

#### SetThreadNum

```cpp
void SetThreadNum(int32_t thread_num)
```

设置运行时的线程数。

- 参数

    - `thread_num`: 运行时的线程数。

#### GetThreadNum

```cpp
int32_t GetThreadNum() const;
```

获取当前线程数设置。

- 返回值

  当前线程数设置。

#### SetInterOpParallelNum

```cpp
void SetInterOpParallelNum(int32_t parallel_num)
```

设置运行时的算子并行推理数目。

- 参数

    - `parallel_num`: 运行时的算子并行数。

#### GetInterOpParallelNum

```cpp
int32_t GetInterOpParallelNum() const;
```

获取当前算子并行数设置。

- 返回值

  当前算子并行数设置。

#### SetThreadAffinity

```cpp
void SetThreadAffinity(int mode)
```

设置运行时的CPU绑核策略。

- 参数

    - `mode`: 绑核的模式，有效值为0-2，0为默认不绑核，1为绑大核，2为绑中核。

#### GetThreadAffinityMode

```cpp
int GetThreadAffinityMode() const;
```

获取当前CPU绑核策略。

- 返回值

  当前CPU绑核策略，有效值为0-2，0为默认不绑核，1为绑大核，2为绑中核。

#### SetThreadAffinity

```cpp
void SetThreadAffinity(const std::vector<int> &core_list)
```

设置运行时的CPU绑核列表。如果SetThreadAffinity和SetThreadAffinity同时设置，core_list生效，mode不生效。

- 参数

    - `core_list`: CPU绑核的列表。

#### GetThreadAffinityCoreList

```cpp
std::vector<int32_t> GetThreadAffinityCoreList() const;
```

获取当前CPU绑核列表。

- 返回值

  当前CPU绑核列表。

#### SetEnableParallel

```cpp
void SetEnableParallel(bool is_parallel)
```

设置运行时是否支持并行。

- 参数

    - `is_parallel`: bool量，为true则支持并行。

#### GetEnableParallel

```cpp
bool GetEnableParallel() const;
```

获取当前是否支持并行。

- 返回值

  返回值为为true，代表支持并行。

#### SetBuiltInDelegate

```cpp
void SetBuiltInDelegate(DelegateMode mode);
```

设置内置Delegate模式，以使用第三方AI框架辅助推理。

- 参数

    - `mode`: 内置Delegate模式，可选配置选项`kNoDelegate`、`kCoreML`、`kNNAPI`。`kNoDelegate`表示不使用第三方AI框架辅助推理，`kCoreML`表示使用CoreMI进行推理（在iOS上可选），`kNNAPI`表示使用NNAPI进行推理（在Android上可选）。

#### GetBuiltInDelegate

```cpp
DelegateMode GetBuiltInDelegate() const;
```

获取当前内置Delegate模式。

- 返回值

  返回当前内置Delegate模式。

#### SetDelegate

```cpp
void SetDelegate(const std::shared_ptr<Delegate> &delegate)
```

设置Delegate，Delegate定义了用于支持第三方AI框架接入的代理。

- 参数

    - `delegate`: Delegate指针。

#### GetDelegate

```cpp
std::shared_ptr<Delegate> GetDelegate() const;
```

获取当前Delegate。

- 返回值

  当前Delegate的指针。

#### set_delegate

```cpp
void set_delegate(const std::shared_ptr<AbstractDelegate> &delegate)
```

设置Delegate，Delegate定义了用于支持第三方AI框架接入的代理。

- 参数

    - `delegate`: Delegate指针。

#### get_delegate

```cpp
std::shared_ptr<AbstractDelegate> get_delegate() const;
```

获取当前Delegate。

- 返回值

  当前Delegate的指针。

#### SetMultiModalHW

```cpp
void SetMultiModalHW(bool float_mode);
```

在多设备中，配置量化模型是否以浮点模式运行。

- 参数

    - `float_mode`: 是否以浮点模式运行。

#### GetMultiModalHW

```cpp
bool GetMultiModalHW() const;
```

获取当前配置中，量化模型的运行模式。

- 返回值

  当前配置中，量化模型是否以浮点模式运行。

#### MutableDeviceInfo

```cpp
std::vector<std::shared_ptr<DeviceInfoContext>> &MutableDeviceInfo()
```

修改该context下的[DeviceInfoContext](#deviceinfocontext)数组，仅端侧推理支持数组中有多个成员是异构场景。

- 返回值

  存储DeviceInfoContext的vector的引用。

## DeviceInfoContext

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/context.h)&gt;

DeviceInfoContext类定义不同硬件设备的环境信息。

### 构造函数和析构函数

```cpp
DeviceInfoContext()
virtual ~DeviceInfoContext() = default;
```

### 公有成员函数

| 函数                                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|-------------------------------------------------------------------|---------|---------|
| [enum DeviceType GetDeviceType() const](#getdevicetype)     |    √    |    √    |
| [std::shared_ptr<T> Cast()](#cast)     |    √    |    √    |
| [void SetProvider(const std::string &provider)](#setprovider)     |    √    |    √    |
| [std::string GetProvider() const](#getprovider)     |    √    |    √    |
| [void SetProviderDevice(const std::string &device)](#setproviderdevice)     |    √    |    √    |
| [std::string GetProviderDevice() const](#getproviderdevice)     |    √    |    √    |
| [void SetAllocator(const std::shared_ptr<Allocator> &allocator)](#setallocator)     |    ✕    |    ✕    |
| [std::shared_ptr<Allocator> GetAllocator() const](#getallocator)     |    ✕    |    ✕    |

#### GetDeviceType

```cpp
virtual enum DeviceType GetDeviceType() const = 0;
```

获取该DeviceInfoContext的类型。

- 返回值

  该DeviceInfoContext的类型。

  ```cpp
  enum DeviceType {
    kCPU = 0,
    kGPU,
    kKirinNPU,
    kAscend910,
    kAscend310,
    // add new type here
    kInvalidDeviceType = 100,
  };
  ```

#### Cast

```cpp
template <class T> std::shared_ptr<T> Cast()
```

在打开`-fno-rtti`编译选项的情况下提供类似RTTI的功能，将DeviceInfoContext转换为`T`类型的指针，若转换失败返回`nullptr`。

- 返回值

  转换后`T`类型的指针，若转换失败则为`nullptr`。

#### GetProvider

```cpp
std::string GetProvider() const;
```

获取设备的生产商名。

#### SetProvider

```cpp
void SetProvider(const std::string &provider)
```

设置设备生产商名。

- 参数

    - `provider`: 生产商名。

#### GetProviderDevice

```cpp
std::string GetProviderDevice() const;
```

获取生产商设备名。

#### SetProviderDevice

```cpp
void SetProviderDevice(const std::string &device)
```

设备生产商设备名。

- 参数

    - `device`: 设备名。

#### SetAllocator

```cpp
void SetAllocator(const std::shared_ptr<Allocator> &allocator)
```

设置内存管理器。

- 参数

    - `allocator`: 内存管理器。

#### GetAllocator

```cpp
std::shared_ptr<Allocator> GetAllocator() const;
```

获取内存管理器。

## CPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/context.h)&gt;

派生自[DeviceInfoContext](#deviceinfocontext)，模型运行在CPU上的配置。

### 公有成员函数

|     函数     |     说明      | 云侧推理是否支持 | 端侧推理是否支持 |
| ------------ | ------------ |---------|---------|
| `enum DeviceType GetDeviceType() const` | - 返回值: DeviceType::kCPU |        √ |        √ |
| `void SetEnableFP16(bool is_fp16)`      | 用于指定是否以FP16精度进行推理<br><br> - `is_fp16`: 是否以FP16精度进行推理 |        √ |        √ |
| `bool GetEnableFP16() const`            | - 返回值: 已配置的精度模式 |        √ |        √ |

## GPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/context.h)&gt;

派生自[DeviceInfoContext](#deviceinfocontext)，模型运行在GPU上的配置。

### 公有成员函数

| 函数                                                         | 说明                                                                                                   | 云侧推理是否支持 | 端侧推理是否支持 |
|------------------------------------------------------------|------------------------------------------------------------------------------------------------------|----------|----------|
| `enum DeviceType GetDeviceType() const`                    | - 返回值: DeviceType::kGPU                                                                              | √        | √        |
| `void SetDeviceID(uint32_t device_id)`                     | 用于指定设备ID<br><br> - `device_id`: 设备ID                                                                 | √        | √        |
| `uint32_t GetDeviceID() const`                             | - 返回值: 已配置的设备ID                                                                                      | √        | √        |
| `void SetPrecisionMode(const std::string &precision_mode)` | 用于指定推理时算子精度<br><br> - `precision_mode`: 可选值`origin`(以模型中指定精度进行推理), `fp16`(以FP16精度进行推理)，默认值: `origin` | √        | √        |
| `std::string GetPrecisionMode() const`                     | - 返回值: 已配置的精度模式                                                                                      | √        | √        |
| `int GetRankID() const`                                    | - 返回值: 当前运行的RANK ID                                                                                  | √        | √        |
| `int GetGroupSize() const`                                 | - 返回值: 当前运行的GROUP SIZE                                                                               | √        | √        |
| `void SetEnableFP16(bool is_fp16)`                         | 用于指定是否以FP16精度进行推理<br><br> - `is_fp16`: 是否以FP16精度进行推理                                                 | √        | √        |
| `bool GetEnableFP16() const`                               | - 返回值: 已配置的精度模式                                                                                      | √        | √        |
| `void SetGLContext(void *gl_context)`                      | 用于指定OpenGL EGLContext<br><br> - `*gl_context`: OpenGL的当前运行时的EGLContext值                              | ✕        | √        |
| `void *GetGLContext() const`                               | - 返回值: 已配置的指向OpenGL EGLContext的指针                                                                    | ✕        | √        |
| `void SetGLDisplay(void *gl_display)`                      | 用于指定OpenGL EGLDisplay<br><br> - `*gl_display`: OpenGL的当前运行时的EGLDisplay值                              | ✕        | √        |
| `void *GetGLDisplay() const`                               | - 返回值: 已配置的指向OpenGL EGLDisplay的指针                                                                    | ✕        | √        |
| `void SetEnableGLTexture(bool is_enable_gl_texture)`       | 用于指定是否绑定OpenGL纹理数据<br><br> - `is_enable_gl_texture`: 是否在推理时绑定OpenGL纹理数据                              | ✕        | √        |
| `bool GetEnableGLTexture() const`                          | - 返回值: 已配置的绑定OpenGL纹理数据模式                                                                            | ✕        | √        |

## KirinNPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/context.h)&gt;

派生自[DeviceInfoContext](#deviceinfocontext)，模型运行在NPU上的配置。

### 公有成员函数

|     函数     |     说明      | 云侧推理是否支持 | 端侧推理是否支持 |
| ------------ | ------------ |-------|-------|
| `enum DeviceType GetDeviceType() const` | - 返回值: DeviceType::kKirinNPU |    √   |    √   |
| `void SetFrequency(int frequency)`      | 用于指定NPU频率<br><br> - `frequency`: 设置为1（低功耗）、2（均衡）、3（高性能）、4（极致性能），默认为3 | ✕     |    √   |
| `int GetFrequency() const`              | - 返回值: 已配置的NPU频率模式 | ✕     |    √   |
| `void SetEnableFP16(bool is_fp16)`                          | 用于指定是否以FP16精度进行推理<br><br> - `is_fp16`: 是否以FP16精度进行推理 |        ✕ |    √   |
| `bool GetEnableFP16() const`                                | - 返回值: 已配置的精度模式 |       ✕ |    √   |

## AscendDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/context.h)&gt;

派生自[DeviceInfoContext](#deviceinfocontext)，模型运行在Atlas 200/300/500推理产品、Atlas推理系列产品上的配置。

### 公有成员函数

| 函数                                                                 | 说明                                                                                                                                                                                                                                                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|--------|
| `enum DeviceType GetDeviceType() const`                            | - 返回值: DeviceType::kAscend                                                                                                                                                                                                                                                        |    √     |    √     |
| `void SetDeviceID(uint32_t device_id)`                             | 用于指定设备ID<br><br> - `device_id`: 设备ID                                                                                                                                                                                                                                              |        √ |    √     |
| `uint32_t GetDeviceID() const`                                     | - 返回值: 已配置的设备ID                                                                                                                                                                                                                                                                   |        √ |    √     |
| `void SetInsertOpConfigPath(const std::string &cfg_path)`          | 模型插入[AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/atc/atlasatc_16_0016.html)算子<br><br> - `cfg_path`: [AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/atc/atlasatc_16_0016.html)配置文件路径   |        √ |    √     |
| `std::string GetInsertOpConfigPath()`                              | - 返回值: 已配置的[AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/atc/atlasatc_16_0016.html)                                                                                                                                                 |        √ |    √     |
| `void SetInputFormat(const std::string &format)`                   | 指定模型输入format<br><br> - `format`: 可选有`"NCHW"`，`"NHWC"`，`"ND"`                                                                                                                                                                                                                           |        √ |    √     |
| `std::string GetInputFormat()`                                     | - 返回值: 已配置模型输入format                                                                                                                                                                                                                                                              |        √ |    √     |
| `void SetInputShape(const std::string &shape)`                     | 指定模型输入shape，为字符串形式，需指定输入名称，每个shape值由`,`隔开，不同输入由`;`隔开<br><br> - `shape`: 如`"input_op_name1:1,2,3,4;input_op_name2:4,3,2,1"`                                                                                                                                                                                                 |        √ |    √     |
| `std::string GetInputShape()`                                      | - 返回值: 已配置模型输入shape                                                                                                                                                                                                                                                               |        √ |    √     |
| `void SetInputShapeMap(const std::map<int, std::vector<int>> &shape)`                     | 指定模型输入shape<br><br> - `shape`: map的key对应输入的下标，例如第一个输入对应下标0，第二个对应下标1。value对应输入shape，为数组形式。                                                                                                                                                                                                 |        √ |    √     |
| `std::map<int, std::vector<int>> GetInputShapeMap()`                                      | - 返回值: 已配置模型输入shape                                                                                                                                                                                                                                                               |        √ |    √     |
| `void SetDynamicBatchSize(const std::vector<size_t> &dynamic_batch_size)`                     | 指定模型动态batch的挡位，支持个数范围[2, 100]，为数组形式。<br><br> - `dynamic_batch_size`: 如`{1, 2}`。                                                                                                                                                                                                 |        √ |    √     |
| `std::string GetDynamicBatchSize()`                                      | - 返回值: 已配置模型的动态batch                                                                                                                                                                                                                                                               |        √ |    √     |
| `void SetDynamicImageSize(const std::string &dynamic_image_size)`                     | 指定模型动态分辨率的挡位，支持个数范围[2, 100]，为字符串形式，每个shape值由`,`隔开，不同输入由`;`隔开。<br><br> - `dynamic_image_size`: 如`"64,64;128,128"`。                                                                                                                                                                                                 |        √ |    √     |
| `std::string GetDynamicImageSize()`                                      | - 返回值: 已配置模型的动态分辨率                                                                                                                                                                                                                                                               |        √ |    √     |
| `void SetOutputType(enum DataType output_type)`                    | 指定模型输出type<br><br> - `output_type`: 仅支持uint8、fp16和fp32                                                                                                                                                                                                                            |        √ |    √     |
| `enum DataType GetOutputType()`                                    | - 返回值: 已配置模型的输出类型                                                                                                                                                                                                                                                                |        √ |    √     |
| `void SetPrecisionMode(const std::string &precision_mode)`         | 配置模型精度模式<br><br> - `precision_mode`: 可选有`"enforce_fp16"`，`"preferred_fp32"`，`"enforce_origin"`，`"enforce_fp32"`或者`"preferred_optimal"`，默认为`"enforce_fp16"`                                                                                                                               |        √ |    √     |
| `std::string GetPrecisionMode()`                                  | - 返回值: 已配置模型的精度模式                                                                                                                                                                                                                                                                  |        √ |    √     |
| `void SetOpSelectImplMode(const std::string &op_select_impl_mode)` | 配置算子实现方式<br><br> - `op_select_impl_mode`: 可选有`"high_performance"`和`"high_precision"`，默认为`"high_performance"`                                                                                                                                                                      |        √ |    √     |
| `std::string GetOpSelectImplMode()`                                | - 返回值: 已配置的算子选择模式                                                                                                                                                                                                                                                                  |        √ |    √     |
| `void SetFusionSwitchConfigPath(const std::string &cfg_path)` | 配置融合开关<br><br> - `cfg_path`: 融合开关配置文件，可指定关闭特定融合规则                                                                                                                                                                      |        √ |    √     |
| `std::string GetFusionSwitchConfigPath()`                                | - 返回值: 已配置的融合开关文件路径                                                                                                                                                                                                                                                                  |        √ |    √     |
| `void SetBufferOptimizeMode(const std::string &buffer_optimize_mode)` | 配置缓存优化模式<br><br> - `buffer_optimize_mode`: 可选有`l1_optimize`，`l2_optimize`或`off_optimize`，默认为`l2_optimize`                                                                                                                                                                      |        √ |    √     |
| `std::string GetBufferOptimizeMode()`                                | - 返回值: 已配置的缓存优化模式                                                                                                                                                                                                                                                                  |        √ |    √     |

## Serialization

\#include &lt;[serialization.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/serialization.h)&gt;

Serialization类汇总了模型文件读写的方法。

### 静态公有成员函数

| 函数                                                                 | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------------------------|--------|--------|
| [Status Load(const std::string &file, ModelType model_type, Graph *graph, const Key &dec_key = {}, const std::string &dec_mode = kDecModeAesGcm)](#load)     |    ✕    |    √    |
| [Status Load(const std::vector<std::string> &files, ModelType model_type, std::vector<Graph> *graphs, const Key &dec_key = {}, const std::string &dec_mode = kDecModeAesGcm)](#load-1)     |    ✕    |    ✕    |
| [Status Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph, const Key &dec_key = {}, const std::string &dec_mode = kDecModeAesGcm)](#load-2)     |    ✕    |    √    |
| [static Status SetParameters(const std::map<std::string, Buffer> &parameters, Model *model)](#setparameters)     |    ✕    |    ✕    |
| [static Status ExportModel(const Model &model, ModelType model_type, Buffer *model_data)](#exportmodel)     |    ✕    |    √    |
| [static Status ExportModel(const Model &model, ModelType model_type, const std::string &model_file, QuantizationType quantization_type = kNoQuant, bool export_inference_only = true, std::vector<std::string> output_tensor_name = {})](#exportmodel)     |    ✕    |    √    |
| [static Status ExportWeightsCollaborateWithMicro(const Model &model, ModelType model_type, const std::string &weight_file, bool is_inference = true, bool enable_fp16 = false, const std::vector<std::string> &changeable_weights_name = {})](#exportweightscollaboratewithmicro)     |    ✕    |    √    |

#### Load

从文件加载模型。

```cpp
Status Load(const std::string &file, ModelType model_type, Graph *graph, const Key &dec_key = {},
            const std::string &dec_mode = kDecModeAesGcm)
```

- 参数

    - `file`: 模型文件路径。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR`、`ModelType::kMindIR_Lite`、`ModelType::kOM`。
    - `graph`: 输出参数，保存图数据的对象。
    - `dec_key`: 解密密钥，用于解密密文模型，密钥长度为16、24或32。
    - `dec_mode`: 解密模式，可选有`AES-GCM`、`AES-CBC`。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Load

从多个文件加载多个模型。

```cpp
Status Load(const std::vector<std::string> &files, ModelType model_type, std::vector<Graph> *graphs,
            const Key &dec_key = {}, const std::string &dec_mode = kDecModeAesGcm)
```

- 参数

    - `files`: 多个模型文件路径，用vector存储。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR`、`ModelType::kMindIR_Lite`、`ModelType::kOM`。
    - `graphs`: 输出参数，依次保存图数据的对象。
    - `dec_key`: 解密密钥，用于解密密文模型，密钥长度为16、24或32。
    - `dec_mode`: 解密模式，可选有`AES-GCM`、`AES-CBC`。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Load

从内存缓冲区加载模型。

```cpp
Status Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph,
            const Key &dec_key = {}, const std::string &dec_mode = kDecModeAesGcm)
```

- 参数

    - `model_data`：模型数据指针。
    - `data_size`：模型数据字节数。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR`、`ModelType::kMindIR_Lite`、`ModelType::kOM`。
    - `graph`：输出参数，保存图数据的对象。
    - `dec_key`: 解密密钥，用于解密密文模型，密钥长度为16、24或32。
    - `dec_mode`: 解密模式，可选有`AES-GCM`、`AES-CBC`。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### SetParameters

配置模型参数。

```cpp
static Status SetParameters(const std::map<std::string, Buffer> &parameters, Model *model)
```

- 参数

    - `parameters`：参数。
    - `model`：模型。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### ExportModel

导出训练模型，MindSpore Lite训练使用。

```cpp
static Status ExportModel(const Model &model, ModelType model_type, Buffer *model_data)
```

- 参数

    - `model`：模型数据。
    - `model_type`：模型文件类型。
    - `model_data`：模型参数数据。

 - 返回值

    状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### ExportModel

导出训练模型，MindSpore Lite训练使用。

```cpp
static Status ExportModel(const Model &model, ModelType model_type, const std::string &model_file,
                        QuantizationType quantization_type = kNoQuant, bool export_inference_only = true,
                        std::vector<std::string> output_tensor_name = {})
```

- 参数

    - `model`：模型数据。
    - `model_type`：模型文件类型。
    - `model_file`：保存的模型文件。
    - `quantization_type`: 量化类型。
    - `export_inference_only`: 是否导出只做推理的模型。
    - `output_tensor_name`: 设置导出的推理模型的输出张量的名称，默认为空，导出完整的推理模型。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### ExportWeightsCollaborateWithMicro

试验接口，导出供micro推理使用的模型权重，MindSpore Lite训练使用。

```cpp
static Status ExportWeightsCollaborateWithMicro(const Model &model, ModelType model_type,
                                                const std::string &weight_file, bool is_inference = true,
                                                bool enable_fp16 = false,
                                                const std::vector<std::string> &changeable_weights_name = {})
```

- 参数

    - `model`：模型数据。
    - `model_type`：模型文件类型。
    - `weight_file`：保存的权重文件。
    - `is_inference`: 是否是对推理模型的导出，当前仅支持推理模型。
    - `enable_fp16`: 权重保存类型。
    - `changeable_weights_name`: 设置shape会变化的权重名称。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

## Buffer

\#include &lt;types.h&gt;

Buffer定义了MindSpore中Buffer数据的结构。

### 构造函数和析构函数

```cpp
  Buffer()
  Buffer(const void *data, size_t data_len)
  ~Buffer()
```

### 公有成员函数

| 函数                                | 云侧推理是否支持 | 端侧推理是否支持 |
|-----------------------------|--------|--------|
| [const void *Data() const](#data)     |    √    |    √    |
| [void *MutableData()](#mutabledata)     |    √    |    √    |
| [size_t DataSize() const](#datasize)     |    √    |    √    |
| [bool ResizeData(size_t data_len)](#resizedata)     |    √    |    √    |
| [bool SetData(const void *data, size_t data_len)](#setdata)     |    √    |    √    |
| [Buffer Clone() const](#clone)     |    √    |    √    |

#### Data

```cpp
const void *Data() const;
```

获取只读的数据地址。

- 返回值

  const void指针。

#### MutableData

```cpp
void *MutableData()
```

获取可写的数据地址。

- 返回值

  void指针。

#### DataSize

```cpp
size_t DataSize() const;
```

获取data大小。

- 返回值

  当前data大小。

#### ResizeData

```cpp
bool ResizeData(size_t data_len)
```

重置data大小。

- 参数

    - `data_len`: data大小

- 返回值

  是否配置成功。

#### SetData

```cpp
bool SetData(const void *data, size_t data_len)
```

配置Data和大小。

- 参数

    - `data`: data地址
    - `data_len`: data大小

- 返回值

  是否配置成功。

#### Clone

```cpp
Buffer Clone() const;
```

拷贝一份自身的副本。

- 返回值

  指向副本的指针。

## Model

\#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/model.h)&gt;

Model定义了MindSpore中的模型，便于计算图管理。

### 构造函数和析构函数

```cpp
Model()
~Model()
```

### 公有成员函数

| 函数                                                                                                                                                                                                                 | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|---------|
| [Status Build(const void *model_data, size_t data_size, ModelType model_type, const std::shared_ptr<Context> &model_context = nullptr)](#build)     |    √    |    √    |
| [Status Build(const void *model_data, size_t data_size, ModelType model_type, const std::shared_ptr<Context> &model_context, const Key &dec_key, const std::string &dec_mode, const std::string &cropto_lib_path)](#build-1)     |    √    |    √    |
| [Status Build(const std::string &model_path, ModelType model_type, const std::shared_ptr<Context> &model_context = nullptr)](#build-2)     |    √    |    √    |
| [Status Build(const std::string &model_path, ModelType model_type, const std::shared_ptr<Context> &model_context, const Key &dec_key, const std::string &dec_mode, const std::string &cropto_lib_path)](#build-3)     |    √    |    √    |
| [Status Build(GraphCell graph, const std::shared_ptr<Context> &model_context = nullptr, const std::shared_ptr<TrainCfg> &train_cfg = nullptr)](#build-4)     |    ✕    |    √    |
| [Status Build(const std::string &model_path, ModelType model_type, const std::shared_ptr<Context> &model_context, const CryptoInfo &cryptoInfo);](#build-5)     |    √    |    ✕    |
| [Status BuildTransferLearning(GraphCell backbone, GraphCell head, const std::shared_ptr<Context> &context, const std::shared_ptr<TrainCfg> &train_cfg = nullptr)](#buildtransferlearning)     |    ✕    |    √    |
| [Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims)](#resize)     |    √    |    √    |
| [Status UpdateWeights(const std::vector<MSTensor> &new_weights)](#updateweights)     |    ✕    |    √    |
| [Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs, const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)](#predict)     |    √    |    √    |
| [Status Predict(const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)](#predict-1)     |    ✕    |    √    |
| [Status RunStep(const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)](#runstep)     |    ✕    |    √    |
| [Status PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs, const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)](#predictwithpreprocess)     |    ✕    |    ✕    |
| [Status Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs)](#preprocess)     |    ✕    |    ✕    |
| [bool HasPreprocess()](#haspreprocess)     |    ✕    |    ✕    |
| [Status LoadConfig(const std::string &config_path)](#loadconfig)     |    √    |    √    |
| [Status UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config)](#updateconfig)     |    √    |    √    |
| [std::vector<MSTensor> GetInputs()](#getinputs)     |    √    |    √    |
| [MSTensor GetInputByTensorName(const std::string &tensor_name)](#getinputbytensorname)     |    √    |    √    |
| [std::vector<MSTensor> GetOutputs()](#getoutputs)     |    √    |    √    |
| [std::vector< std::string> GetOutputTensorNames()](#getoutputtensornames)     |    √    |    √    |
| [MSTensor GetOutputByTensorName(const std::string &tensor_name)](#getoutputbytensorname)     |    √    |    √    |
| [std::vector<MSTensor> GetOutputsByNodeName(const std::string &node_name)](#getoutputsbynodename)     |    ✕    |    √    |
| [static bool CheckModelSupport(enum DeviceType device_type, ModelType model_type)](#checkmodelsupport)     |    √    |    √    |
| [std::vector<MSTensor> GetGradients() const](#getgradients)     |    ✕    |    √    |
| [Status ApplyGradients(const std::vector<MSTensor> &gradients)](#applygradients)     |    ✕    |    √    |
| [std::vector<MSTensor> GetFeatureMaps() const](#getfeaturemaps)     |    ✕    |    √    |
| [std::vector<MSTensor> GetTrainableParams() const](#gettrainableparams)     |    ✕    |    √    |
| [Status UpdateFeatureMaps(const std::vector<MSTensor> &new_weights)](#updatefeaturemaps)     |    ✕    |    √    |
| [std::vector<MSTensor> GetOptimizerParams() const](#getoptimizerparams)     |    ✕    |    √    |
| [Status SetOptimizerParams(const std::vector<MSTensor> &params)](#setoptimizerparams)     |    ✕    |    √    |
| [Status SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f)](#setupvirtualbatch)     |    ✕    |    √    |
| [Status SetLearningRate(float learning_rate)](#setlearningrate)     |    ✕    |    √    |
| [float GetLearningRate()](#getlearningrate)     |    ✕    |    √    |
| [Status InitMetrics(std::vector<Metrics *> metrics)](#initmetrics)     |    ✕    |    √    |
| [std::vector<Metrics *> GetMetrics()](#getmetrics)     |    ✕    |    √    |
| [Status BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture, std::map<std::string, unsigned int> *outputGLTexture)](#bindgltexture2dmemory)     |    ✕    |    √    |
| [Status SetTrainMode(bool train)](#settrainmode)     |    ✕    |    √    |
| [bool GetTrainMode() const](#gettrainmode)     |    ✕    |    √    |
| [Status Train(int epochs, std::shared_ptr< dataset::Dataset> ds, std::vector<TrainCallBack *> cbs)](#train)     |    ✕    |    √    |
| [Status Evaluate(std::shared_ptr< dataset::Dataset> ds, std::vector<TrainCallBack *> cbs)](#evaluate)     |    ✕    |    √    |

#### Build

```cpp
Status Build(const void *model_data, size_t data_size, ModelType model_type,
             const std::shared_ptr<Context> &model_context = nullptr)
```

从内存缓冲区加载模型，并将模型编译至可在Device上运行的状态。

- 参数

    - `model_data`: 指向存储读入模型文件缓冲区的指针。
    - `data_size`: 缓冲区大小。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR_Lite`、`ModelType::kMindIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。在端侧和云侧推理包中，端侧推理只支持`ms`模型推理，该入参值被忽略。云端推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `model_context`: 模型[Context](#context)。

- 返回值

    状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Build

```cpp
Status Build(const void *model_data, size_t data_size, ModelType model_type,
             const std::shared_ptr<Context> &model_context = nullptr, const Key &dec_key = {},
             const std::string &dec_mode = kDecModeAesGcm, const std::string &cropto_lib_path)
```

从内存缓冲区加载模型，并将模型编译至可在Device上运行的状态。

- 参数

    - `model_data`: 指向存储读入模型文件缓冲区的指针。
    - `data_size`: 缓冲区大小。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR_Lite`、`ModelType::kMindIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。在端侧和云侧推理包中，端侧推理只支持`ms`模型推理，该入参值被忽略。云端推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `model_context`: 模型[Context](#context)。
    - `dec_key`: 解密密钥，用于解密密文模型，密钥长度为16。
    - `dec_mode`: 解密模式，可选有`AES-GCM`。
    - `cropto_lib_path`: OpenSSL Crypto解密库路径。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Build

```cpp
Status Build(const std::string &model_path, ModelType model_type,
             const std::shared_ptr<Context> &model_context = nullptr)
```

根据路径读取加载模型，并将模型编译至可在Device上运行的状态。

- 参数

    - `model_path`: 模型文件路径。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR_Lite`、`ModelType::kMindIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。在端侧和云侧推理包中，端侧推理只支持`ms`模型推理，该入参值被忽略。云端推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `model_context`: 模型[Context](#context)。

- 返回值

    状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Build

```cpp
Status Build(const std::string &model_path, ModelType model_type,
             const std::shared_ptr<Context> &model_context = nullptr, const Key &dec_key = {},
             const std::string &dec_mode = kDecModeAesGcm, const std::string &cropto_lib_path)
```

根据路径读取加载模型，并将模型编译至可在Device上运行的状态。

- 参数

    - `model_path`: 模型文件路径。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR_Lite`、`ModelType::kMindIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。在端侧和云侧推理包中，端侧推理只支持`ms`模型推理，该入参值被忽略。云端推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `model_context`: 模型[Context](#context)。
    - `dec_key`: 解密密钥，用于解密密文模型，密钥长度为16。
    - `dec_mode`: 解密模式，可选有`AES-GCM`。
    - `cropto_lib_path`: OpenSSL Crypto解密库路径。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

> `Build`之后对`model_context`的其他修改不再生效。

#### Build

```cpp
Status Build(GraphCell graph, const std::shared_ptr<Context> &model_context = nullptr,
             const std::shared_ptr<TrainCfg> &train_cfg = nullptr)
```

将GraphCell存储的模型编译至可在Device上运行的状态。

- 参数

    - `graph`: `GraphCell`是`Cell`的一个派生，`Cell`目前没有开放使用。`GraphCell`可以由`Graph`构造，如`model.Build(GraphCell(graph), context)`。
    - `model_context`: 模型[Context](#context)。
    - `train_cfg`: train配置文件[TrainCfg](#traincfg)。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Build

```cpp
Status Build(const std::string &model_path, ModelType model_type,
             const std::shared_ptr<Context> &model_context, const CryptoInfo &cryptoInfo);
```

根据路径读取加载密文模型，并将模型解密而后编译至可在Device上运行的状态。

- 参数

    - `model_path`: 密文模型文件路径。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR_Lite`、`ModelType::kMindIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。MindIR推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。MindIR推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行推理。
    - `model_context`: 模型[Context](#context)。
    - `cryptoInfo`: 解密相关配置信息。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Predict

```cpp
Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs, const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)
```

推理模型。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `outputs`: 输出参数，按顺序排列的`vector`的指针，模型输出会按顺序填入该容器。
    - `before`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。
    - `after`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Predict

```cpp
Status Predict(const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)
```

推理模型。

- 参数

    - `before`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。
    - `after`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### LoadConfig

```cpp
Status LoadConfig(const std::string &config_path)
```

根据路径读取配置文件。

- 参数

    - `config_path`: 配置文件路径。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

> 用户可以调用`LoadConfig`接口进行混合精度推理的设置，配置文件举例如下：
>
> [execution_plan]
>
> op_name1=data_type:float16
>
> op_name2=data_type:float32
>
> 在使用GPU推理时，用户可以使用`LoadConfig`接口进行TensorRT设置，配置文件内容及说明如下：
>
> [ms_cache]
>
> serialize_path=（序列化模型的存储路径）
>
> [gpu_context]
>
> input_shape=input_name:[input_dim]（模型输入维度，用于动态维度张量输入）
>
> dynamic_dims=[min_dim~max_dim]（模型输入的动态维度范围，用于动态维度张量输入）
>
> opt_dims=[opt_dim]（模型最优输入维度，用于动态维度张量输入）

#### UpdateConfig

```cpp
Status UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config)
```

刷新配置，读文件相对比较费时，如果少部分配置发生变化可以通过该接口更新部分配置。

- 参数

    - `section`: 配置的章节名。
    - `config`: 要更新的配置对。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### GetInputs

```cpp
std::vector<MSTensor> GetInputs()
```

获取模型所有输入张量。

- 返回值

  包含模型所有输入张量的容器类型变量。

#### GetInputByTensorName

```cpp
MSTensor GetInputByTensorName(const std::string &tensor_name)
```

获取模型指定名字的输入张量。

- 返回值

  指定名字的输入张量，如果该名字不存在则返回非法张量。

#### GetGradients

```cpp
std::vector<MSTensor> GetGradients() const;
```

获取所有Tensor的梯度。

- 返回值

  获取所有Tensor的梯度。

#### ApplyGradients

```cpp
Status ApplyGradients(const std::vector<MSTensor> &gradients)
```

应用所有Tensor的梯度。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### GetOptimizerParams

```cpp
std::vector<MSTensor> GetOptimizerParams() const;
```

获取optimizer参数MSTensor。

- 返回值

  所有optimizer参数MSTensor。

#### SetOptimizerParams

```cpp
Status SetOptimizerParams(const std::vector<MSTensor> &params)
```

更新optimizer参数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### GetTrainableParams

```cpp
std::vector<MSTensor> GetTrainableParams() const;
```

获取optimizer中所有参与权重更新的MSTensor。

- 返回值

  optimizer中所有参与权重更新的MSTensor。

#### GetOutputs

```cpp
std::vector<MSTensor> GetOutputs()
```

获取模型所有输出张量。

- 返回值

  包含模型所有输出张量的容器类型变量。

#### GetOutputTensorNames

```cpp
std::vector<std::string> GetOutputTensorNames()
```

获取模型所有输出张量的名字。

- 返回值

  包含模型所有输出张量名字的容器类型变量。

#### GetOutputByTensorName

```cpp
MSTensor GetOutputByTensorName(const std::string &tensor_name)
```

获取模型指定名字的输出张量。

- 返回值

  指定名字的输出张量，如果该名字不存在则返回非法张量。

#### GetOutputsByNodeName

```cpp
std::vector<MSTensor> GetOutputsByNodeName(const std::string &node_name)
```

通过节点名获取模型的MSTensors输出张量。不建议使用，将在2.0版本废弃。

- 参数

    - `node_name`: 节点名称。

- 返回值

    包含在模型输出Tensor中的该节点输出Tensor的vector。

#### BindGLTexture2DMemory

```cpp
  Status BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture,
                               std::map<std::string, unsigned int> *outputGLTexture)
```

将OpenGL纹理数据与模型的输入和输出进行绑定。

- 参数

    - `inputGLTexture`: 模型输入的OpenGL纹理数据, key为输入Tensor的名称，value为OpenGL纹理。
    - `outputGLTexture`: 模型输出的OpenGL纹理数据，key为输出Tensor的名称，value为OpenGL纹理。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### InitMetrics

```cpp
Status InitMetrics(std::vector<Metrics *> metrics)
```

训练指标参数初始化。

- 参数

    - `metrics`: 训练指标参数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### GetMetrics

```cpp
std::vector<Metrics *> GetMetrics()
```

获取训练指标参数。

- 返回值

  训练指标参数。

#### SetTrainMode

```cpp
Status SetTrainMode(bool train)
```

session设置训练模式。

- 参数

    - `train`: 是否为训练模式。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### GetTrainMode

```cpp
bool GetTrainMode() const;
```

获取session是否是训练模式。

- 返回值

  bool类型，表示是否是训练模式。

#### Train

```cpp
Status Train(int epochs, std::shared_ptr<dataset::Dataset> ds, std::vector<TrainCallBack *> cbs)
```

模型训练。

- 参数

    - `epochs`: 迭代轮数。
    - `ds`: 训练数据。
    - `cbs`: 包含训练回调类对象的`vector`。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Evaluate

```cpp
Status Evaluate(std::shared_ptr<dataset::Dataset> ds, std::vector<TrainCallBack *> cbs)
```

模型验证。

- 参数

    - `ds`: 训练数据。
    - `cbs`: 包含训练回调类对象的`vector`。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Resize

```cpp
Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims)
```

调整已编译模型的输入张量形状。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `dims`: 输入张量形状，按输入顺序排列的由形状组成的`vector`，模型会按顺序依次调整对应输入顺序的`inputs`张量形状。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### CheckModelSupport

```cpp
static bool CheckModelSupport(enum DeviceType device_type, ModelType model_type)
```

检查设备是否支持该模型。

- 参数

    - `device_type`: 设备类型，例如`kMaliGPU`。
    - `model_type`: 模型类型，例如`MindIR`。

- 返回值

    状态码。

#### BuildTransferLearning

```cpp
Status BuildTransferLearning(GraphCell backbone, GraphCell head, const std::shared_ptr<Context> &context,
                      const std::shared_ptr<TrainCfg> &train_cfg = nullptr)
```

构建一个迁移学习模型，其中主干权重是固定的，头部权重是可训练的。

- 参数

    - `backbone`: 静态、不可学习部分。
    - `head`: 可训练部分。
    - `model_context`: 模型[Context](#context)。
    - `train_cfg`: train配置文件[TrainCfg](#traincfg)。

- 返回值

    状态码。

#### GetLearningRate

```cpp
float GetLearningRate()
```

获取学习率。

- 返回值

    float类型，获取学习率。如果为0.0，表示没有找到优化器。

#### SetLearningRate

```cpp
Status SetLearningRate(float learning_rate)
```

设置学习率。

- 参数

    - `learning_rate`: 指定的学习率。

- 返回值

    状态码。

#### SetupVirtualBatch

```cpp
Status SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f)
```

设置虚拟batch用于训练。

- 参数

    - `virtual_batch_multiplier`: 虚拟batch乘法器，当设置值小于1时，表示禁用虚拟batch。
    - `lr`: 学习率，默认为-1.0f。
    - `momentum`: 动量，默认为-1.0f。

- 返回值

    状态码。

#### RunStep

```cpp
Status RunStep(const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)
```

单步训练模型。

- 参数

    - `before`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。
    - `after`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

    状态码。

#### PredictWithPreprocess

```cpp
Status PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs,
                             const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)
```

进行推理模型，并在推理前进行数据预处理。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `outputs`: 输出参数，按顺序排列的`vector`的指针，模型输出会按顺序填入该容器。
    - `before`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。
    - `after`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

    状态码。

#### Preprocess

```cpp
Status Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs)
```

若模型配置了数据预处理，对模型输入数据进行数据预处理。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `outputs`: 输出参数，按顺序排列的`vector`的指针，模型输出会按顺序填入该容器。

- 返回值

    状态码。

#### HasPreprocess

```cpp
bool HasPreprocess()
```

模型是否配置了数据预处理。

- 返回值

    模型是否配置了数据预处理。

#### GetFeatureMaps

```cpp
std::vector<MSTensor> GetFeatureMaps() const
```

获取模型的所有权重Tensors。

- 返回值

    获取模型的所有权重Tensor。

#### UpdateFeatureMaps

```cpp
Status UpdateFeatureMaps(const std::vector<MSTensor> &new_weights)
```

更新模型的权重Tensor内容。

- 参数

    - `new_weights`: 要更新的权重Tensor。

- 返回值

    状态码。

#### UpdateWeights

```cpp
Status UpdateWeights(const std::vector<MSTensor> &new_weights)
```

更新模型的权重Tensor的大小和内容。

- 参数

    - `new_weights`: 要更新的权重Tensor，可同时更新大小和内容。

- 返回值

    状态码。

## MSTensor

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/types.h)&gt;

`MSTensor`定义了MindSpore中的张量。

### 构造函数和析构函数

```cpp
MSTensor()
explicit MSTensor(const std::shared_ptr<Impl> &impl)
MSTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data, size_t data_len)
explicit MSTensor(std::nullptr_t)
~MSTensor()
```

注意：MSTensor构造时，若data指针通过malloc生成，用户在构造完成MSTensor后，需自行释放free，否则存在内存泄露。

### 静态公有成员函数

| 函数                                                                                                                                                                                                                 | 云侧推理是否支持 | 端侧推理是否支持 |
|------------------------------------------------------------------------------------------------------------------|---------|---------|
| [MSTensor *CreateTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data, size_t data_len, const std::string &device = "", int device_id = -1) noexcept](#createtensor)     |    √    |    √    |
| [MSTensor *CreateTensor(const std::string &name, const MSTensor &tensor, const std::string &device = "", int device_id = -1) noexcept](#createtensor)     |    √    |    √    |
| [MSTensor *CreateRefTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, void *data, size_t data_len) noexcept](#createreftensor)     |    √    |    √    |
| [static inline MSTensor CreateDeviceTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, void *data, size_t data_len) noexcept](#createdevicetensor)     |    √    |    ✕    |
| [static inline MSTensor *CreateTensorFromFile(const std::string &file, DataType type = DataType::kNumberTypeUInt8, const std::vector<int64_t> &shape = {}) noexcept](#createtensorfromfile)     |    √    |    ✕    |
| [MSTensor *StringsToTensor(const std::string &name, const std::vector<std::string> &str)](#stringstotensor)     |    √    |    √    |
| [std::vector<std::string> TensorToStrings(const MSTensor &tensor)](#tensortostrings)     |    √    |    √    |
| [void DestroyTensorPtr(MSTensor *tensor) noexcept](#destroytensorptr)     |    √    |    √    |

#### CreateTensor

```cpp
MSTensor *CreateTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                       const void *data, size_t data_len, const std::string &device = "",
                       int device_id = -1) noexcept;
```

创建一个`MSTensor`对象，其数据需复制后才能由`Model`访问，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `type`：数据类型。
    - `shape`：张量的形状。
    - `data`：数据指针，指向一段已开辟的内存。
    - `data_len`：数据长度，以字节为单位。
    - `device`：设备类型，表明Tensor的内存存放的位置位于设备侧。
    - `device_id`：设备编号。

- 返回值

  `MStensor`指针。

#### CreateTensor

```cpp
MSTensor *CreateTensor(const std::string &name, const MSTensor &tensor, const std::string &device = "",
                       int device_id = -1) noexcept;
```

创建一个`MSTensor`对象，其数据需复制后才能由`Model`访问，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `tensor`：用于作为拷贝的源MSTensor。
    - `device`：设备类型，表明Tensor的内存存放的位置位于设备侧。
    - `device_id`：设备编号。

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
    - `shape`：张量的形状。
    - `data`：数据指针，指向一段已开辟的内存。
    - `data_len`：数据长度，以字节为单位。

- 返回值

  `MStensor`指针。

#### CreateDeviceTensor

```cpp
static inline MSTensor CreateDeviceTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                                          void *data, size_t data_len) noexcept;
```

创建一个`MSTensor`对象，其device数据可以直接由`Model`访问，不需要与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `type`：数据类型。
    - `shape`：张量的形状。
    - `data`：数据指针，指向一段已开辟的device内存。
    - `data_len`：数据长度，以字节为单位。

- 返回值

  `MStensor`对象。

#### CreateTensorFromFile

```cpp
static inline MSTensor *CreateTensorFromFile(const std::string &file, DataType type = DataType::kNumberTypeUInt8,
                                             const std::vector<int64_t> &shape = {}) noexcept;
```

创建一个`MSTensor`对象，其数据由文件路径`file`所指定，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `file`: 文件路径，指向存放数据的二进制格式文件，可以是相对路径或者绝对路径。
    - `type`：`file`文件保存的数据类型，也是创建后`MSTensor`对象的数据类型。
    - `shape`：张量的形状，`shape`的乘积代表了`file`文件内数据的个数。

- 返回值

  `MStensor`指针。

#### StringsToTensor

```cpp
MSTensor *StringsToTensor(const std::string &name, const std::vector<std::string> &str)
```

创建一个字符串类型的`MSTensor`对象，其数据需复制后才能由`Model`访问，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `str`：装有若干个字符串的`vector`容器。

- 返回值

  `MStensor`指针。

#### TensorToStrings

```cpp
std::vector<std::string> TensorToStrings(const MSTensor &tensor)
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

| 函数                            | 云侧推理是否支持 | 端侧推理是否支持 |
|---------------------------------------------------------------------------------------------|---------|---------|
| [std::string Name() const](#name)     |    √    |    √    |
| [enum DataType DataType() const](#datatype)     |    √    |    √    |
| [const std::vector<int64_t> &Shape() const](#shape)     |    √    |    √    |
| [int64_t ElementNum() const](#elementnum)     |    √    |    √    |
| [std::shared_ptr<const void> Data() const](#data)     |    √    |    √    |
| [void *MutableData()](#mutabledata)     |    √    |    √    |
| [size_t DataSize() const](#datasize)     |    √    |    √    |
| [int GetDevice() const](#getdevice)     |    √    |    ✕    |
| [int GetDeviceId() const](#getdeviceid)     |    √    |    ✕    |
| [bool IsConst() const](#isconst)     |    √    |    √    |
| [bool IsDevice() const](#isdevice)     |    √    |    ✕    |
| [MSTensor *Clone() const](#clone)     |    √    |    √    |
| [bool operator==(std::nullptr_t) const](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#operator==std-nullptr-t)     |    √    |    √    |
| [bool operator!=(std::nullptr_t) const](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#operator!=std-nullptr-t)     |    √    |    √    |
| [bool operator==(const MSTensor &tensor) const](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#operator==const-mstensor-tensor)     |    √    |    √    |
| [void SetShape(const std::vector<int64_t> &shape)](#setshape)     |    √    |    √    |
| [void SetDataType(enum DataType data_type)](#setdatatype)     |    √    |    √    |
| [void SetTensorName(const std::string &name)](#settensorname)     |    √    |    √    |
| [void SetAllocator(std::shared_ptr<Allocator> allocator)](#setallocator)     |    √    |    √    |
| [std::shared_ptr<Allocator> allocator() const](#allocator)     |    √    |    √    |
| [void SetFormat(mindspore::Format format)](#setformat)     |    √    |    √    |
| [mindspore::Format format() const](#format)     |    √    |    √    |
| [void SetData(void *data, bool own_data = true)](#setdata)     |    √    |    √    |
| [void SetDeviceData(void *data)](#setdevicedata)     |    √    |    √    |
| [void *GetDeviceData()](#getdevicedata)     |    √    |    √    |
| [std::vector<QuantParam> QuantParams() const](#quantparams)     |    √    |    √    |
| [void SetQuantParams(std::vector<QuantParam> quant_params)](#setquantparams)     |    √    |    √    |
| [const std::shared_ptr<Impl> impl()](#impl)     |    √    |    √    |

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
void *MutableData()
```

获取`MSTensor`中的数据的指针。如果为空指针，为`MSTensor`的数据申请内存，并返回申请内存的地址，如果不为空，返回数据的指针。

- 返回值

  指向`MSTensor`中的数据的指针。

#### DataSize

```cpp
size_t DataSize() const;
```

获取`MSTensor`中的数据的以字节为单位的内存长度。

- 返回值

  `MSTensor`中的数据的以字节为单位的内存长度。

#### GetDevice

```cpp
int GetDevice() const;
```

获取`MSTensor`所处的设备类型。

- 返回值

  `MSTensor`所处的设备类型。

#### GetDeviceId

```cpp
int GetDeviceId() const;
```

获取`MSTensor`所处的设备编号。

- 返回值

  `MSTensor`所处的设备编号。

#### IsConst

```cpp
bool IsConst() const;
```

判断`MSTensor`中的数据是否是常量数据。

- 返回值

  `MSTensor`中的数据是否是常量数据。

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

#### operator!=(std::nullptr_t)

```cpp
bool operator!=(std::nullptr_t) const;
```

判断`MSTensor`是否合法。

- 返回值

  `MSTensor`是否合法。

#### operator==(const MSTensor &tensor)

```cpp
bool operator==(const MSTensor &tensor) const;
```

判断`MSTensor`是否与另一个MSTensor相等。

- 返回值

  `MSTensor`是否与另一个MSTensor相等。

#### SetShape

```cpp
void SetShape(const std::vector<int64_t> &shape)
```

设置`MSTensor`的Shape，目前在[Delegate](#delegate)机制使用。

#### SetDataType

```cpp
void SetDataType(enum DataType data_type)
```

设置`MSTensor`的DataType，目前在[Delegate](#delegate)机制使用。

#### SetTensorName

```cpp
void SetTensorName(const std::string &name)
```

设置`MSTensor`的名字，目前在[Delegate](#delegate)机制使用。

#### SetAllocator

```cpp
void SetAllocator(std::shared_ptr<Allocator> allocator)
```

设置`MSTensor`数据所属的内存池。

- 参数

    - `model`: 指向Allocator的指针。

#### allocator

```cpp
std::shared_ptr<Allocator> allocator() const;
```

获取`MSTensor`数据所属的内存池。

- 返回值

    - 指向Allocator的指针。

#### SetFormat

```cpp
void SetFormat(mindspore::Format format)
```

设置`MSTensor`数据的format，目前在[Delegate](#delegate)机制使用。

#### format

```cpp
mindspore::Format format() const;
```

获取`MSTensor`数据的format，目前在[Delegate](#delegate)机制使用。

#### SetData

```cpp
void SetData(void *data, bool own_data = true)
```

设置指向`MSTensor`数据的指针。

- 参数

    - `data`: 新的数据的地址。
    - `own_data`: 是否在`MSTensor`析构时释放数据内存。如果为true，将在`MSTensor`析构时释放数据内存，如果重复调用`SetData`，将仅释放新的数据内存，老的数据内存需要用户释放；如果为false，需要用户释放数据内存。由于向前兼容，默认为true，建议用户设置为false。

#### SetDeviceData

```cpp
void SetDeviceData(void *data)
```

设置数据的设备地址，由用户负责设备内存的申请和释放。仅适用于Ascend和GPU硬件后端。

#### GetDeviceData

```cpp
void *GetDeviceData()
```

获取由`SetDeviceData`接口设置的`MSTensor`数据的设备地址。

#### QuantParams

```cpp
std::vector<QuantParam> QuantParams() const;
```

获取`MSTensor`的量化参数，目前在[Delegate](#delegate)机制使用。

#### SetQuantParams

```cpp
void SetQuantParams(std::vector<QuantParam> quant_params)
```

设置`MSTensor`的量化参数，目前在[Delegate](#delegate)机制使用。

#### impl

```cpp
const std::shared_ptr<Impl> impl()
```

获取实现类的指针。

## QuantParam

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/types.h)&gt;

一个结构体。QuantParam定义了MSTensor的一组量化参数。

### 公有属性

#### bit_num

```cpp
bit_num
```

**int** 类型变量。量化的bit数。

#### scale

```cpp
scale
```

**double** 类型变量。

#### zero_point

```cpp
zero_point
```

**int32_t** 类型变量。

#### min

```cpp
min
```

**double** 类型变量。量化的最小值。

#### max

```cpp
max
```

**double** 类型变量。量化的最大值。

## MSKernelCallBack

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/types.h)&gt;

```cpp
using MSKernelCallBack = std::function<bool(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs, const MSCallBackParam &opInfo)>
```

一个函数包装器。MSKernelCallBack 定义了指向回调函数的指针。

## MSCallBackParam

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/types.h)&gt;

一个结构体。MSCallBackParam定义了回调函数的输入参数。

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

#### execute_time

```cpp
execute_time
```

**double** 类型变量。GPU执行时间。

## Delegate

\#include &lt;[delegate.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/delegate.h)&gt;

`Delegate`定义了第三方AI框架接入MindSpore Lite的代理接口。

### 构造函数和析构函数

```cpp
Delegate() = default;
virtual ~Delegate() = default;
```

### 公有成员函数

#### Init

```cpp
virtual Status Init() = 0;
```

初始化Delegate资源。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Build

```cpp
virtual Status Build(DelegateModel *model) = 0;
```

Delegate在线构图。

- 参数

    - `model`: 指向存储[DelegateModel](#delegatemodel)实例的指针。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

## CoreMLDelegate

\#include &lt;[delegate.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/delegate.h)&gt;

`CoreMLDelegate`继承自`Delegate`类，定义了CoreML框架接入MindSpore Lite的代理接口。

### 构造函数

```cpp
CoreMLDelegate()
```

### 公有成员函数

#### Init

```cpp
Status Init() overirde;
```

初始化CoreMLDelegate资源，仅在内部图编译阶段调用。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Build

```cpp
Status Build(DelegateModel *model) override;
```

CoreMLDelegate在线构图，仅在内部图编译阶段调用。

- 参数

    - `model`: 指向存储[DelegateModel](#delegatemodel)实例的指针。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

## SchemaVersion

\#include &lt;[delegate.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/delegate.h)&gt;

定义了MindSpore Lite执行在线推理时模型文件的版本。

```cpp
typedef enum {
  SCHEMA_INVALID = -1, /**< invalid version */
  SCHEMA_CUR,          /**< current version for ms model defined in model.fbs*/
  SCHEMA_V0,           /**< previous version for ms model defined in model_v0.fbs*/
} SchemaVersion;
```

## KernelIter

\#include &lt;[delegate.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/delegate.h)&gt;

定义了MindSpore Lite [Kernel](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_kernel.html#mindspore-kernel)列表的迭代器。

```cpp
using KernelIter = std::vector<kernel::Kernel *>::iterator;
```

## DelegateModel

\#include &lt;[delegate.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/delegate.h)&gt;

`DelegateModel`定义了MindSpore Lite Delegate机制操作的的模型对象。

### 构造函数

```cpp
DelegateModel(std::vector<kernel::Kernel *> *kernels, const std::vector<MSTensor> &inputs,
              const std::vector<MSTensor> &outputs,
              const std::map<kernel::Kernel *, const schema::Primitive *> &primitives, SchemaVersion version)
```

### 析构函数

```cpp
~DelegateModel() = default;
```

### 保护成员

#### kernels_

```cpp
std::vector<kernel::Kernel *> *kernels_;
```

[**Kernel**](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_kernel.html#kernel)的列表，保存模型的所有算子。

#### inputs_

```cpp
const std::vector<mindspore::MSTensor> &inputs_;
```

[**MSTensor**](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#mstensor)的列表，保存这个算子的输入tensor。

#### outputs_

```cpp
const std::vector<mindspore::MSTensor> &outputs;
```

[**MSTensor**](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#mstensor)的列表，保存这个算子的输出tensor。

#### primitives_

```cpp
const std::map<kernel::Kernel *, const schema::Primitive *> &primitives_;
```

[**Kernel**](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_kernel.html#kernel)和**schema::Primitive**的Map，保存所有算子的属性。

#### version_

```cpp
SchemaVersion version_;
```

**enum**值，当前执行推理的模型的版本[SchemaVersion](#schemaversion)。

### 公有成员函数

#### GetPrimitive

```cpp
const schema::Primitive *GetPrimitive(kernel::Kernel *kernel) const;
```

获取一个Kernel的属性值。

- 参数

    - `kernel`: 指向Kernel的指针。

- 返回值

  const schema::Primitive *，输入参数Kernel对应的该算子的属性值。

#### BeginKernelIterator

```cpp
KernelIter BeginKernelIterator()
```

返回DelegateModel Kernel列表起始元素的迭代器。

- 返回值

  **KernelIter**，指向DelegateModel Kernel列表起始元素的迭代器。

#### EndKernelIterator

```cpp
KernelIter EndKernelIterator()
```

返回DelegateModel Kernel列表末尾元素的迭代器。

- 返回值

  **KernelIter**，指向DelegateModel Kernel列表末尾元素的迭代器。

#### Replace

```cpp
KernelIter Replace(KernelIter from, KernelIter end, kernel::Kernel *graph_kernel)
```

用Delegate子图Kernel替换Delegate支持的连续Kernel列表。

- 参数

    - `from`: Delegate支持的连续Kernel列表的起始元素迭代器。
    - `end`: Delegate支持的连续Kernel列表的末尾元素迭代器。
    - `graph_kernel`: 指向Delegate子图Kernel实例的指针。

- 返回值

  **KernelIter**，用Delegate子图Kernel替换之后，子图Kernel下一个元素的迭代器，指向下一个未被访问的Kernel。

#### inputs

```cpp
const std::vector<mindspore::MSTensor> &inputs()
```

返回DelegateModel输入tensor列表。

- 返回值

  [**MSTensor**](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#mstensor)的列表。

#### outputs

```cpp
const std::vector<mindspore::MSTensor> &outputs()
```

返回DelegateModel输出tensor列表。

- 返回值

  [**MSTensor**](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#mstensor)的列表。

#### GetVersion

```cpp
const SchemaVersion GetVersion() { return version_; }
```

返回当前执行推理的模型文件的版本。

- 返回值

  **enum**值，0: r1.2及r1.2之后的版本，1: r1.1及r1.1之前的版本，-1: 无效版本。

## TrainCfg

\#include &lt;[cfg.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/cfg.h)&gt;

`TrainCfg`MindSpore Lite训练的相关配置参数。

### 构造函数

```cpp
TrainCfg() { this->loss_name_ = "_loss_fn"; }
```

### 公有成员变量

```cpp
OptimizationLevel optimization_level_ = kO0;
```

优化的数据类型。

```cpp
enum OptimizationLevel : uint32_t {
  kO0 = 0,
  kO2 = 2,
  kO3 = 3,
  kAuto = 4,
  kOptimizationType = 0xFFFFFFFF
};
```

```cpp
std::string loss_name_;
```

损失节点的名称。

```cpp
MixPrecisionCfg mix_precision_cfg_;
```

混合精度配置。

```cpp
bool accumulate_gradients_;
```

是否累加梯度。

## MixPrecisionCfg

\#include &lt;[cfg.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/cfg.h)&gt;

`MixPrecisionCfg`MindSpore Lite训练混合精度配置类。

### 构造函数

```cpp
  MixPrecisionCfg() {
    dynamic_loss_scale_ = false;
    loss_scale_ = 128.0f;
    num_of_not_nan_iter_th_ = 1000;
  }
```

### 共有成员变量

```cpp
bool dynamic_loss_scale_ = false;
```

混合精度训练中是否启用动态损失比例。

```cpp
float loss_scale_;
```

初始损失比例。

```cpp
uint32_t num_of_not_nan_iter_th_;
```

动态损失阈值。

```cpp
bool is_raw_mix_precision_;
```

原始模型是否是原生混合精度模型。

## AccuracyMetrics

\#include &lt;[accuracy.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/metrics/accuracy.h)&gt;

`AccuracyMetrics`MindSpore Lite训练精度类。

### 构造函数和析构函数

```cpp
explicit AccuracyMetrics(int accuracy_metrics = METRICS_CLASSIFICATION, const std::vector<int> &input_indexes = {1}, const std::vector<int> &output_indexes = {0})
virtual ~AccuracyMetrics()
```

### 公有成员函数

#### Clear

```cpp
void Clear() override;
```

精度清零。

#### Eval

```cpp
float Eval() override;
```

模型验证。

- 返回值

  float，模型验证精度。

## Metrics

\#include &lt;[metrics.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/metrics/metrics.h)&gt;

`Metrics`MindSpore Lite训练指标类。

### 析构函数

```cpp
virtual ~Metrics() = default;
```

### 公有成员函数

#### Clear

```cpp
virtual void Clear() {}
```

训练指标清零。

#### Eval

```cpp
virtual float Eval() { return 0.0; }
```

模型验证。

- 返回值

  float，模型验证精度。

#### Update

```cpp
virtual void Update(std::vector<MSTensor *> inputs, std::vector<MSTensor *> outputs) {}
```

模型输入输出数据更新。

- 参数

    - `inputs`: 模型输入MSTensor的`vector`。
    - `outputs`: 模型输输出MSTensor的`vector`。

## TrainCallBack

\#include &lt;[callback.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/callback/callback.h)&gt;

`Metrics`MindSpore Lite训练回调类。

### 析构函数

```cpp
virtual ~TrainCallBack() = default;
```

### 公有成员函数

#### Begin

```cpp
virtual void Begin(const TrainCallBackData &cb_data) {}
```

网络执行前调用。

- 参数

    - `cb_data`: 回调参数。

#### End

```cpp
  virtual void End(const TrainCallBackData &cb_data) {}
```

网络执行后调用。

- 参数

    - `cb_data`: 回调参数。

#### EpochBegin

```cpp
  virtual void EpochBegin(const TrainCallBackData &cb_data) {}
```

每轮迭代前回调。

- 参数

    - `cb_data`: 回调参数。

#### EpochEnd

```cpp
  virtual CallbackRetValue EpochEnd(const TrainCallBackData &cb_data) { return kContinue; }
```

每轮迭代后回调。

- 参数

    - `cb_data`: 回调参数。

- 返回值

  `CallbackRetValue`，表示是否在训练中继续循环。

    ```cpp
    enum CallbackRetValue : uint32_t {
      kContinue = 0,
      kStopTraining = 1,
      kExit = 2,
      kUnknownRetValue = 0xFFFFFFFF
    };
    ```

#### StepBegin

```cpp
  virtual void StepBegin(const TrainCallBackData &cb_data) {}
```

每步迭代前回调。

- 参数

    - `cb_data`: 回调参数。

#### StepEnd

```cpp
  virtual void StepEnd(const TrainCallBackData &cb_data) {}
```

每步迭代后回调。

- 参数

    - `cb_data`: 回调参数。

## TrainCallBackData

\#include &lt;[callback.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/callback/callback.h)&gt;

一个结构体。TrainCallBackData定义了训练回调的一组参数。

### 公有属性

#### train_mode_

```cpp
train_mode_
```

**bool** 类型变量。训练模式。

#### epoch_

```cpp
epoch_
```

**unsigned int** 类型变量。训练迭代的epoch次数。

#### step_

```cpp
step_
```

**unsigned int** 类型变量。训练迭代的step次数。

#### model_

```cpp
model_
```

**Model** 类型指针。训练模型对象。

## CkptSaver

\#include &lt;[ckpt_saver.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/callback/ckpt_saver.h)&gt;

`Metrics`MindSpore Lite训练模型文件保存类。

### 构造函数和析构函数

```cpp
  explicit CkptSaver(int save_every_n, const std::string &filename_prefix)
  virtual ~CkptSaver()
```

## LossMonitor

\#include &lt;[loss_monitor.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/callback/loss_monitor.h)&gt;

`Metrics`MindSpore Lite训练损失函数类。

### 构造函数和析构函数

```cpp
  explicit LossMonitor(int print_every_n_steps = INT_MAX)
  virtual ~LossMonitor()
```

### 公有成员函数

#### GetLossPoints

```cpp
  const std::vector<GraphPoint> &GetLossPoints()
```

获取训练损失数据。

- 返回值

  包含`GraphPoint`数据的`vector`，训练的损失数据。

## LRScheduler

\#include &lt;[lr_scheduler.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/callback/lr_scheduler.h)&gt;

`Metrics`MindSpore Lite训练学习率调度类。

### 构造函数和析构函数

```cpp
  explicit LRScheduler(LR_Lambda lambda_func, void *lr_cb_data = nullptr, int step = 1)
  virtual ~LRScheduler()
```

## StepLRLambda

\#include &lt;[lr_scheduler.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/callback/lr_scheduler.h)&gt;

一个结构体。StepLRLambda定义了训练学习率的一组参数。

### 公有属性

#### step_size

```cpp
step_size
```

**int** 类型变量。学习率衰减步长。

#### gamma

```cpp
gamma
```

**float** 类型变量。学习率衰减因子。

## MultiplicativeLRLambda

\#include &lt;[lr_scheduler.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/callback/lr_scheduler.h)&gt;

每个epoch将学习率乘以一个因子。

```cpp
using LR_Lambda = std::function<int(float *lr, int epoch, void *cb_data)>;
int MultiplicativeLRLambda(float *lr, int epoch, void *multiplication)
```

学习率更新。

- 参数

    - `lr`: 学习率。
    - `epoch`: 迭代轮数。
    - `multiplication`: 更新方式。

- 返回值

  int类型返回值，表示是否更新，DONT_UPDATE_LR为0表示不更新，UPDATE_LR为1表示更新。

  ```cpp
  constexpr int DONT_UPDATE_LR = 0;
  constexpr int UPDATE_LR = 1;
  ```

## TimeMonitor

\#include &lt;[time_monitor.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/callback/time_monitor.h)&gt;

`Metrics`MindSpore Lite训练时间监测类。

### 析构函数

```cpp
  virtual ~TimeMonitor() = default;
```

### 公有成员函数

#### EpochBegin

```cpp
  void EpochBegin(const TrainCallBackData &cb_data) override;
```

每轮迭代前调用。

- 参数

    - `cb_data`: 回调参数。

- 返回值

  `CallbackRetValue`，表示是否在训练中继续循环。

#### EpochEnd

```cpp
  CallbackRetValue EpochEnd(const TrainCallBackData &cb_data) override;
```

每轮迭代后调用。

- 参数

    - `cb_data`: 回调参数。

- 返回值

  `CallbackRetValue`，表示是否在训练中继续循环。

## TrainAccuracy

\#include &lt;[train_accuracy.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/callback/train_accuracy.h)&gt;

`Metrics`MindSpore Lite训练学习率调度类。

### 构造函数和析构函数

```cpp
explicit TrainAccuracy(int print_every_n = INT_MAX, int accuracy_metrics = METRICS_CLASSIFICATION, const std::vector<int> &input_indexes = {1}, const std::vector<int> &output_indexes = {0})
virtual ~TrainAccuracy()
```

- 参数

    - `print_every_n`: 间隔print_every_n步打印一次。
    - `accuracy_metrics`: 精度指标，默认值为METRICS_CLASSIFICATION表示0，METRICS_MULTILABEL表示1。
    - `input_indexes`: 输入索引。
    - `output_indexes`: 输出索引。

```cpp
constexpr int METRICS_CLASSIFICATION = 0;
constexpr int METRICS_MULTILABEL = 1;
```

#### GetAccuracyPoints

```cpp
  const std::vector<GraphPoint> &GetAccuracyPoints()
```

获取训练精度。

- 返回值

  包含`GraphPoint`的`vector`，训练精度数据。

## CharVersion

| 函数                | 云侧推理是否支持 | 端侧推理是否支持 |
|-----------------------|--------|--------|
| [std::vector<char> CharVersion()](#charversion)     |    ✕    |    √    |

\#include &lt;types.h&gt;

```cpp
std::vector<char> CharVersion()
```

全局方法，用于获取版本的字符vector。

- 返回值

  MindSpore Lite版本的字符vector。

## Version

| 函数                | 云侧推理是否支持 | 端侧推理是否支持 |
|-----------------------|--------|--------|
| [std::string Version()](#version)     |    ✕    |    √    |

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/types.h)&gt;

```cpp
std::string Version()
```

全局方法，用于获取版本的字符串。

- 返回值

    MindSpore Lite版本的字符串。

## Allocator

\#include &lt;[allocator.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/allocator.h)&gt;

内存管理基类。

### 析构函数

```cpp
virtual ~Allocator()
```

析构函数。

### 公有成员函数

#### Malloc

```cpp
virtual void *Malloc(size_t size)
```

内存分配。

- 参数

    - `size`: 要分配的内存大小，单位为Byte。

```cpp
virtual void *Malloc(size_t weight, size_t height, DataType type)
```

Image格式内存分配。

- 参数

    - `weight`: 要分配的Image格式内存的宽度。
    - `height`: 要分配的Image格式内存的高度。
    - `type`: 要分配的Image格式内存的数据类型。

#### Free

```cpp
virtual void *Free(void *ptr)
```

内存释放。

- 参数

    - `ptr`: 要释放的内存地址，该值由[Malloc](#malloc)分配。

#### RefCount

```cpp
virtual int RefCount(void *ptr)
```

返回分配内存的引用计数。

- 参数

    - `ptr`: 要操作的内存地址，该值由[Malloc](#malloc)分配。

#### SetRefCount

```cpp
virtual int SetRefCount(void *ptr, int ref_count)
```

设置分配内存的引用计数。

- 参数

    - `ptr`: 要操作的内存地址，该值由[Malloc](#malloc)分配。

    - `ref_count`: 引用计数值。

#### DecRefCount

```cpp
virtual int DecRefCount(void *ptr, int ref_count)
```

分配的内存引用计数减一。

- 参数

    - `ptr`: 要操作的内存地址，该值由[Malloc](#malloc)分配。

    - `ref_count`: 引用计数值。

#### IncRefCount

```cpp
virtual int IncRefCount(void *ptr, int ref_count)
```

分配的内存引用计数加一。

- 参数

    - `ptr`: 要操作的内存地址，该值由[Malloc](#malloc)分配。

    - `ref_count`: 引用计数值。

#### Create

```cpp
static std::shared_ptr<Allocator> Create()
```

创建默认的内存分配器。

#### Prepare

```cpp
virtual void *Prepare(void *ptr)
```

对分配的内存进行预处理。

- 参数

    - `ptr`: 要操作的内存地址，该值由[Malloc](#malloc)分配。

### 保护的数据成员

#### aligned_size_

内存对齐的字节数。默认值: `32` 。

## Status

\#include &lt;status.h&gt;

### 构造函数和析构函数

```cpp
Status()
inline Status(enum StatusCode status_code, const std::string &status_msg = "")
inline Status(const StatusCode code, int line_of_code, const char *file_name, const std::string &extra = "")
~Status() = default;
```

### 公有成员函数

| 函数                                | 云侧推理是否支持 | 端侧推理是否支持 |
|-----------------------------|--------|--------|
| [enum StatusCode StatusCode() const](#statuscode)     |    √    |    √    |
| [inline std::string ToString() const](#tostring)     |    √    |    √    |
| [int GetLineOfCode() const](#getlineofcode)     |    √    |    √    |
| [inline std::string GetFileName() const](#getfilename)     |    √    |    √    |
| [inline std::string GetErrDescription() const](#geterrdescription)     |    √    |    √    |
| [inline std::string SetErrDescription(const std::string &err_description)](#seterrdescription)     |    √    |    √    |
| [inline void SetStatusMsg(const std::string &status_msg)](#setstatusmsg)     |    √    |    √    |
| [friend std::ostream &operator<<(std::ostream &os, const Status &s)](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#operator<<std-ostream-os,-const-status-s)     |    √    |    √    |
| [bool operator==(const Status &other) const](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#operator==const-status-other)     |    √    |    √    |
| [bool operator==(enum StatusCode other_code) const](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#operator==enum-statuscode-other-code)     |    √    |    √    |
| [bool operator!=(const Status &other) const](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#operator!=const-status-other)     |    √    |    √    |
| [bool operator!=(enum StatusCode other_code) const](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore.html#operator!=enum-statuscode-other-code)     |    √    |    √    |
| [explicit operator bool() const](#operator-bool)     |    √    |    √    |
| [explicit operator int() const](#explicit-operator-int-const)     |    √    |    √    |
| [static Status OK()](#ok)     |    √    |    √    |
| [bool IsOk() const](#isok)     |    √    |    √    |
| [bool IsError() const](#iserror)     |    √    |    √    |
| [static inline std::string CodeAsString(enum StatusCode c)](#codeasstring)     |    √    |    √    |

#### StatusCode

```cpp
enum StatusCode StatusCode() const;
```

获取状态码。

- 返回值

    状态码。

#### ToString

```cpp
inline std::string ToString() const;
```

状态码转成字符串。

- 返回值

    状态码的字符串。

#### GetLineOfCode

```cpp
int GetLineOfCode() const;
```

获取代码行数。

- 返回值

    代码行数。

#### GetFileName

```cpp
inline std::string GetFileName() const;
```

获取文件名。

- 返回值

    文件名。

#### GetErrDescription

```cpp
inline std::string GetErrDescription() const;
```

获取错误描述字符串。

- 返回值

    错误描述字符串。

#### SetErrDescription

```cpp
inline std::string SetErrDescription(const std::string &err_description)
```

配置错误描述字符串。

- 参数

    - `err_description`: 错误描述字符串。

- 返回值

    状态信息字符串。

#### SetStatusMsg

```cpp
inline void SetStatusMsg(const std::string &status_msg)
```

配置状态描述字符串。

- 参数

    - `status_msg`: 状态描述字符串。

#### operator<<(std::ostream &os, const Status &s)

```cpp
friend std::ostream &operator<<(std::ostream &os, const Status &s)
```

状态信息写到输出流。

- 参数

    - `os`: 输出流。
    - `s`: 状态类。

- 返回值

    输出流。

#### operator==(const Status &other)

```cpp
bool operator==(const Status &other) const;
```

判断是否与另一个Status相等。

- 参数

    - `other`: 另一个Status。

- 返回值

    是否与另一个Status相等。

#### operator==(enum StatusCode other_code)

```cpp
bool operator==(enum StatusCode other_code) const;
```

判断是否与一个StatusCode相等。

- 参数

    - `other_code`: 一个StatusCode。

- 返回值

    是否与一个StatusCode相等。

#### operator!=(const Status &other)

```cpp
bool operator!=(const Status &other) const;
```

判断是否与另一个Status不相等。

- 参数

    - `other`: 另一个Status。

- 返回值

    是否与另一个Status不相等。

#### operator!=(enum StatusCode other_code)

```cpp
bool operator!=(enum StatusCode other_code) const;
```

判断是否与一个StatusCode不等。

- 参数

    - `other_code`: 一个StatusCode。

- 返回值

    是否与一个StatusCode不等。

#### operator bool()

```cpp
explicit operator bool() const;
```

重载bool操作，判断是否当前状态为kSuccess。

- 返回值

    是否当前状态为kSuccess。

#### explicit operator int() const

```cpp
explicit operator int() const;
```

重载int操作。当`Status`对象被作为整型表达式使用时，返回整型表示的当前状态值。

- 返回值

    当前状态值。

#### OK

```cpp
static Status OK()
```

获取kSuccess的状态码。

- 返回值

    StatusCode::kSuccess。

#### IsOk

```cpp
bool IsOk() const;
```

判断是否是kSuccess的状态码。

- 返回值

    是否是kSuccess。

#### IsError

```cpp
bool IsError() const;
```

判断是否不是kSuccess的状态码。

- 返回值

    是否不是kSuccess。
#### CodeAsString

```cpp
static inline std::string CodeAsString(enum StatusCode c)
```

获取StatusCode对应的字符串。

- 参数

    - `c`: 状态码枚举值。

- 返回值

    状态码对应的字符串。

## Graph

\#include &lt;[graph.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/graph.h)&gt;

### 构造函数和析构函数

```cpp
  Graph()
  explicit Graph(const std::shared_ptr<GraphData> &graph_data)
  explicit Graph(std::shared_ptr<GraphData> &&graph_data)
  explicit Graph(std::nullptr_t)
  ~Graph()
```

- 参数

    - `graph_data`: 输出通道数。

### 公有成员函数

#### ModelType

```cpp
  enum ModelType ModelType() const;
```

获取模型类型。

- 返回值

  模型类型。

#### operator==(std::nullptr_t)

```cpp
  bool operator==(std::nullptr_t) const;
```

判断是否为空指针。

- 返回值

  是否为空指针。

#### operator!=(std::nullptr_t)

```cpp
  bool operator!=(std::nullptr_t) const;
```

判断是否为非空指针。

- 返回值

  是否为非空指针。

## CellBase

\#include &lt;[cell.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/cell.h)&gt;

### 构造函数和析构函数

```cpp
  CellBase() = default;
  virtual ~CellBase() = default;
```

### 公有成员函数

#### Clone

```cpp
  virtual std::shared_ptr<CellBase> Clone() const = 0;
```

拷贝一份自身的副本。

- 返回值

  指向副本的指针。

## Cell

\#include &lt;[cell.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/cell.h)&gt;

### 析构函数

```cpp
  virtual ~Cell() = default;
```

### 公有成员函数

#### Clone

```cpp
  std::shared_ptr<CellBase> Clone() const;
```

拷贝一份自身的副本。

- 返回值

  指向副本的指针。

## GraphCell

\#include &lt;[cell.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/cell.h)&gt;

### 构造函数和析构函数

```cpp
  GraphCell() = default;
  ~GraphCell() override = default;
  explicit GraphCell(const Graph &)
  explicit GraphCell(Graph &&)
  explicit GraphCell(const std::shared_ptr<Graph> &)
```

### 公有成员函数

#### GetGraph

```cpp
  const std::shared_ptr<Graph> &GetGraph() const { return graph_; }
```

获取Graph指针。

- 返回值

  指向Graph的指针。

## RunnerConfig

\#include &lt;[model_parallel_runner.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/model_parallel_runner.h)&gt;

RunnerConfig定义了ModelParallelRunner中使用的配置选项参数。

### 构造函数和析构函数

```cpp
RunnerConfig()
~RunnerConfig()
```

### 公有成员函数

| 函数                   | 云侧推理是否支持 | 端侧推理是否支持 |
|-------------------------------------------------------------|---------|---------|
| [void SetWorkersNum(int32_t workers_num)](#setworkersnum)     |    √    |    ✕    |
| [int32_t GetWorkersNum() const](#getworkersnum)     |    √    |    ✕    |
| [void SetContext(const std::shared_ptr<Context> &context)](#setcontext)     |    √    |    ✕    |
| [std::shared_ptr<Context> GetContext() const](#getcontext)     |    √    |    ✕    |
| [inline void SetConfigInfo(const std::string &section, const std::map<std::string, std::string> &config)](#setconfiginfo)     |    √    |    ✕    |
| [inline std::map<std::string, std::map<std::string, std::string>> GetConfigInfo() const](#getconfiginfo)     |    √    |    ✕    |
| [inline void SetConfigPath(const std::string &config_path)](#setconfigpath)     |    √    |    ✕    |
| [inline std::string GetConfigPath() const](#getconfigpath)     |    √    |    ✕    |
| [void SetDeviceIds(const std::vector<uint32_t\> &device_ids)](#setdeviceids)     |    √    |    ✕    |
| [std::vector<uint32_t\> GetDeviceIds() const](#getdeviceids)     |    √    |    ✕    |

#### SetWorkersNum

```cpp
void SetWorkersNum(int32_t workers_num)
```

设置RunnerConfig的worker的个数。

- 参数

    - `workers_num`: worker的数量。

#### GetWorkersNum

```cpp
int32_t GetWorkersNum() const
```

获取RunnerConfig的worker的个数。

- 返回值

  RunnerConfig类中配置的worker数量。

#### SetContext

```cpp
void SetContext(const std::shared_ptr<Context> &context)
```

设置RunnerConfig的context参数。

- 参数

    - `context`: worker上下文配置。

#### GetContext

```cpp
std::shared_ptr<Context> GetContext() const
```

获取RunnerConfig配置的上下文参数。

- 返回值

  上下文配置类`Context`对象。

#### SetConfigInfo

```cpp
void SetConfigInfo(const std::string &key, const std::map<std::string, std::string> &config)
```

设置RunnerConfig的配置参数。

- 参数

    - `key`: string类型关键字。
    - `config`: map类型的配置参数。

#### GetConfigInfo

```cpp
std::map<std::string, std::map<std::string, std::string>> GetConfigInfo() const
```

获取RunnerConfig配置参数信息。

- 返回值

  `map`类型的配置信息。

#### SetConfigPath

```cpp
void SetConfigPath(const std::string &config_path)
```

设置RunnerConfig中的配置文件路径。

- 参数

    - `config_path`: 配置文件路径。

#### GetConfigPath

```cpp
std::string GetConfigPath() const
```

获取RunnerConfig中的配置文件的路径。

- 返回值

  RunnerConfig类中的配置文件路径。

#### SetDeviceIds

```cpp
void SetDeviceIds(const std::vector<uint32_t> &device_ids)
```

设置RunnerConfig中的设备ID列表。

- 参数

    - `device_ids`: 设备ID列表。

#### GetDeviceIds

```cpp
std::vector<uint32_t> GetDeviceIds() const
```

获取RunnerConfig中的设备ID列表。

- 返回值

  RunnerConfig类中的设备ID列表。

## ModelParallelRunner

\#include &lt;[model_parallel_runner.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/model_parallel_runner.h)&gt;

ModelParallelRunner定义了MindSpore的多个Model以及并发策略，便于多个Model的调度与管理。

### 构造函数和析构函数

```cpp
ModelParallelRunner()
~ModelParallelRunner()
```

### 公有成员函数

| 函数                   | 云侧推理是否支持 | 端侧推理是否支持 |
|-------------------------------------------------------------|---------|---------|
| [inline Status Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config = nullptr)](#init)     |    √    |    ✕    |
| [Status Init(const void *model_data, const size_t data_size, const std::shared_ptr<RunnerConfig> &runner_config = nullptr)](#init-1)     |    √    |    ✕    |
| [std::vector<MSTensor> GetInputs()](#getinputs)     |    √    |    ✕    |
| [std::vector<MSTensor> GetOutputs()](#getoutputs)     |    √    |    ✕    |
| [Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs, const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)](#predict)     |    √    |    ✕    |

#### Init

```cpp
inline Status Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config = nullptr)
```

根据路径读取加载模型，生成一个或者多个模型，并将所有模型编译至可在Device上运行的状态。该接口支持传入`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出），但对`ms`模型的支持，将在未来的迭代中删除，推荐使用`mindir`模型进行推理。当使用`ms`模型进行推理时，请保持模型的后缀名为`.ms`，否则无法识别。

- 参数

    - `model_path`: 模型文件路径。
    - `runner_config`: 一个[RunnerConfig](#runnerconfig)类。定义了并发推理模型的配置参数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Init

```cpp
Status Init(const void *model_data, const size_t data_size, const std::shared_ptr<RunnerConfig> &runner_config = nullptr)
```

根据模型文件数据，生成一个或者多个模型，并将所有模型编译至可在Device上运行的状态。该接口仅支持传入`mindir`模型文件数据。

- 参数

    - `model_data`: 模型文件数据。
    - `data_size`: 模型文件数据大小。
    - `runner_config`: 一个[RunnerConfig](#runnerconfig)类。定义了并发推理模型的配置参数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### Predict

```cpp
Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)
```

并发推理模型。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `outputs`: 输出参数，按顺序排列的`vector`的指针，模型输出会按顺序填入该容器。
    - `before`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。
    - `after`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### GetInputs

```cpp
std::vector<MSTensor> GetInputs()
```

获取模型所有输入张量。

- 返回值

  包含模型所有输入张量的容器类型变量。

#### GetOutputs

```cpp
std::vector<MSTensor> GetOutputs()
```

获取模型所有输出张量。

- 返回值

  包含模型所有输出张量的容器类型变量。

## ModelGroup

\#include &lt;[model_group.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0/include/api/model_group.h)&gt;

ModelGroup 类定义MindSpore Lite模型分组信息，用于共享工作空间（Workspace）内存或者权重（包括常量和变量）内存。

### 构造函数和析构函数

```cpp
ModelGroup(ModelGroupFlag flags = ModelGroupFlag::kShareWorkspace)
~ModelGroup()
```

- 参数

    - `flags`: 指示 ModelGroup 的类型，取值 ``ModelGroupFlag::kShareWorkspace`` ， ``ModelGroupFlag::kShareWorkspace`` 。默认 ``ModelGroupFlag::kShareWorkspace`` 。

### 公有成员函数

| 函数                   | 云侧推理是否支持 | 端侧推理是否支持 |
|-------------------------------------------------------------|---------|---------|
| [Status AddModel(const std::vector<std::string> &model_path_list)](#addmodel)     |    √    |    ✕    |
| [Status AddModel(const std::vector<std::pair<const void *, size_t>> &model_buff_list)](#addmodel-1)     |    √    |    ✕    |
| [Status AddModel(const std::vector<Model> &model_list)](#addmodel-2)     |    √    |    ✕    |
| [Status CalMaxSizeOfWorkspace(ModelType model_type, const std::shared_ptr<Context> &ms_context)](#calmaxsizeofworkspace)     |    √    |    ✕    |

#### AddModel

```cpp
Status AddModel(const std::vector<std::string> &model_path_list)
```

共享工作空间内存时，添加需要共享工作空间内存的模型路径。

- 参数

    - `model_path_list`: 需要共享工作空间内存的模型路径。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### AddModel

```cpp
Status AddModel(const std::vector<std::pair<const void *, size_t>> &model_buff_list)
```

共享工作空间内存时，添加需要共享工作空间内存的模型缓存。

- 参数

    - `model_buff_list`: 需要共享工作空间内存的模型缓存。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### AddModel

```cpp
Status AddModel(const std::vector<Model> &model_list)
```

共享权重内存时，添加需要共享权重内存的模型对象。

- 参数

    - `model_list`: 需要共享权重内存的模型对象[Model](#model)列表。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

#### CalMaxSizeOfWorkspace

```cpp
Status CalMaxSizeOfWorkspace(ModelType model_type, const std::shared_ptr<Context> &ms_context)
```

共享工作空间内存时，计算最大的工作空间内存大小。

- 参数

    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR_Lite`、`ModelType::kMindIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。在端侧和云侧推理包中，端侧推理只支持`ms`模型推理，该入参值被忽略。云端推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `model_context`: 模型[Context](#context)。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。
