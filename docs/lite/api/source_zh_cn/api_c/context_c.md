# context_c

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_c/context_c.md)

```c
#include<context_c.h>
```

context_c.h提供了操作Context的接口，Context对象用于保存执行中的环境变量。

## 公有函数

| function                                                                                                                                           |
| -------------------------------------------------------------------------------------------------------------------------------------------------- |
| [MSContextHandle MSContextCreate()](#mscontextcreate)                                                                                              |
| [void MSContextDestroy(MSContextHandle* context)](#mscontextdestroy)                                                                               |
| [void MSContextSetThreadNum(MSContextHandle context, int32_t thread_num)](#mscontextsetthreadnum)                                                  |
| [int32_t MSContextGetThreadNum(const MSContextHandle context)](#mscontextgetthreadnum)                                                             |
| [void MSContextSetThreadAffinityMode(MSContextHandle context, int mode)](#mscontextsetthreadaffinitymode)                                          |
| [int MSContextGetThreadAffinityMode(const MSContextHandle context)](#mscontextgetthreadaffinitymode)                                               |
| [void MSContextSetThreadAffinityCoreList(MSContextHandle context, const int32_t* core_list, size_t core_num)](#mscontextsetthreadaffinitycorelist) |
| [int32_t* MSContextGetThreadAffinityCoreList(const MSContextHandle context, size_t* core_num)](#mscontextgetthreadaffinitycorelist)                |
| [void MSContextSetEnableParallel(MSContextHandle context, bool is_parallel)](#mscontextsetenableparallel)                                          |
| [bool MSContextGetEnableParallel(const MSContextHandle context)](#mscontextgetenableparallel)                                                      |
| [void MSContextAddDeviceInfo(MSContextHandle context, MSDeviceInfoHandle device_info)](#mscontextadddeviceinfo)                                    |
| [MSDeviceInfoHandle MSDeviceInfoCreate(MSDeviceType device_type)](#msdeviceinfocreate)                                                             |
| [void MSDeviceInfoDestroy(MSDeviceInfoHandle* device_info)](#msdeviceinfodestroy)                                                                  |
| [void MSDeviceInfoSetProvider(MSDeviceInfoHandle device_info, const char* provider)](#msdeviceinfosetprovider)                                     |
| [const char* MSDeviceInfoGetProvider(const MSDeviceInfoHandle device_info)](#msdeviceinfogetprovider)                                              |
| [void MSDeviceInfoSetProviderDevice(MSDeviceInfoHandle device_info, const char* device)](#msdeviceinfosetproviderdevice)                           |
| [const char* MSDeviceInfoGetProviderDevice(const MSDeviceInfoHandle device_info)](#msdeviceinfogetproviderdevice)                                  |
| [const char* MSDeviceType MSDeviceInfoGetDeviceType(const MSDeviceInfoHandle device_info)](#msdeviceinfogetdevicetype)                             |
| [void MSDeviceInfoSetEnableFP16(MSDeviceInfoHandle device_info, bool is_fp16)](#msdeviceinfosetenablefp16)                                         |
| [bool MSDeviceInfoGetEnableFP16(const MSDeviceInfoHandle device_info)](#msdeviceinfogetenablefp16)                                                 |
| [void MSDeviceInfoSetFrequency(MSDeviceInfoHandle device_info, int frequency)](#msdeviceinfosetfrequency)                                          |
| [int MSDeviceInfoGetFrequency(const MSDeviceInfoHandle device_info)](#msdeviceinfogetfrequency)                                                    |

### MSContextCreate

```C
MSContextHandle MSContextCreate()
```

创建一个MSContext。

- 返回值

  指向创建的MSContext的指针。

### MSContextDestroy

```C
void MSContextDestroy(MSContextHandle* context)
```

销毁一个MSContext。

- 参数

    - `context`: 指向MSContext的指针。

### MSContextSetThreadNum

```C
void MSContextSetThreadNum(MSContextHandle context, int32_t thread_num);
```

设置运行时的线程数量，该选项仅MindSpore Lite有效。
若参数context为空则不会做任何操作，并在日志中输出空指针信息。

- 参数
    - `context`: 指向MSContext的指针。
    - `thread_num`: 运行时的线程数。

### MSContextGetThreadNum

```C
int32_t MSContextGetThreadNum(const MSContextHandle context)
```

获取MSContext的线程数量，该选项仅MindSpore Lite有效。
若参数context为空则不会做任何操作，并在日志中输出空指针信息。

- 参数
    - `context`: 指向MSContext的指针。

- 返回值

  线程数量。

### MSContextSetThreadAffinityMode

```C
void MSContextSetThreadAffinityMode(MSContextHandle context, int mode)
```

设置运行时的CPU绑核策略，该选项仅MindSpore Lite有效。
若参数context为空则不会做任何操作，并在日志中输出空指针信息。

- 参数
    - `context`: 指向MSContext的指针。
    - `mode`: 绑核的模式，有效值为0-2，0为默认不绑核，1为绑大核，2为绑中核。

### MSContextGetThreadAffinityMode

```C
int MSContextGetThreadAffinityMode(const MSContextHandle context)
```

获取当前CPU绑核策略，该选项仅MindSpore Lite有效。
若参数context为空则返回-1，并在日志中输出空指针信息。

- 参数
    - `context`: 指向MSContext的指针。

- 返回值

  当前CPU绑核策略，有效值为0-2，0为默认不绑核，1为绑大核，2为绑中核。

### MSContextSetThreadAffinityCoreList

```C
void MSContextSetThreadAffinityCoreList(MSContextHandle context, const int32_t* core_list, size_t core_num)
```

设置运行时的CPU绑核列表，如果同时调用了两个不同的`SetThreadAffinity`函数来设置同一个MSContext，仅`core_list`生效，而`mode`不生效。该选项仅MindSpore Lite有效。
若参数context为空则不会做任何操作，并在日志中输出空指针信息。

- 参数
    - `context`: 指向MSContext的指针。
    - `core_list`: CPU绑核的列表。
    - `core_num`: 核的数量。

### MSContextGetThreadAffinityCoreList

```C
int32_t* MSContextGetThreadAffinityCoreList(const MSContextHandle context, size_t* core_num)
```

获取当前CPU绑核列表，该选项仅MindSpore Lite有效。
若参数context为空则会返回`nullptr`，并在日志中输出空指针信息。  

- 参数
    - `context`: 指向MSContext的指针。
    - `core_num`: 输出参数，表示返回值对应的数组的长度。

- 返回值

  当前CPU绑核列表。

### MSContextSetEnableParallel

```C
void MSContextSetEnableParallel(MSContextHandle context, bool is_parallel)
```

设置运行时是否支持并行，该选项仅MindSpore Lite有效。
若参数context为空则不会做任何操作，并在日志中输出空指针信息。

- 参数
    - `context`: 指向MSContext的指针。
    - `is_parallel`: 为true则支持并行。

### MSContextGetEnableParallel

```C
bool MSContextGetEnableParallel(const MSContextHandle context)
```

获取当前是否支持并行，该选项仅MindSpore Lite有效。
若参数context为空则会返回false，并在日志中输出空指针信息。

- 参数
    - `context`: 指向MSContext的指针。

- 返回值

  返回值为为true，代表支持并行。

### MSContextAddDeviceInfo

```C
void MSContextAddDeviceInfo(MSContextHandle context, MSDeviceInfoHandle device_info)
```

添加运行设备信息。

- 参数
    - `context`: 指向MSContext的指针。
    - `device_info`: 指向设备类型信息的指针。

### MSDeviceInfoCreate

```C
MSDeviceInfoHandle MSDeviceInfoCreate(MSDeviceType device_type)
```

新建运行设备信息，若创建失败则会返回`nullptr`，并日志中输出信息。

- 参数
    - `device_type`: 设备类型，具体见[MSDeviceType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_c/types_c.html#msdevicetype)。

- 返回值

  指向创建的运行设备信息的指针。

### MSDeviceInfoDestroy

```C
void MSDeviceInfoDestroy(MSDeviceInfoHandle* device_info)
```

销毁一个运行设备信息对象。

- 参数
    - `device_info`: 指向设备类型信息的指针。

### MSDeviceInfoSetProvider

```C
void MSDeviceInfoSetProvider(MSDeviceInfoHandle device_info, const char* provider)
```

设置设备生产商名称。若参数device_info为空则不会做任何操作，并在日志中输出空指针信息。

- 参数
    - `device_info`: 指向设备类型信息的指针。
    - `provider`: 生产商名称。

### MSDeviceInfoGetProvider

```C
const char* MSDeviceInfoGetProvider(const MSDeviceInfoHandle device_info)
```

获取生产商设备名称。若参数device_info为空则输出`nullptr`，并在日志中输出空指针信息。

- 参数
    - `device_info`: 指向设备类型信息的指针。

- 返回值

  生产商名称。

### MSDeviceInfoSetProviderDevice

```C
void MSDeviceInfoSetProviderDevice(MSDeviceInfoHandle device_info, const char* device)
```

设置供应商设备名称。若参数device_info为空则不会做任何操作，并在日志中输出空指针信息。

- 参数
    - `device_info`: 指向设备类型信息的指针。
    - `device`: 供应商设备类型，例如"CPU"。

### MSDeviceInfoGetProviderDevice

```C
const char* MSDeviceInfoGetProviderDevice(const MSDeviceInfoHandle device_info)
```

获取生产商设备名。若参数device_info为空则不会做任何操作，并在日志中输出空指针信息。

- 参数
    - `device_info`: 指向设备类型信息的指针。

- 返回值

  生产商设备名。

### MSDeviceInfoGetDeviceType

```C
const char* MSDeviceType MSDeviceInfoGetDeviceType(const MSDeviceInfoHandle device_info)
```

获得生产商设备类型。若参数device_info为空则返回`nullptr`，并在日志中输出空指针信息。

- 参数
    - `device_info`: 指向设备类型信息的指针。

- 返回值

  生产商设备类型。

### MSDeviceInfoGetEnableFP16

```C
bool MSDeviceInfoGetEnableFP16(const MSDeviceInfoHandle device_info)
```

获取是否开启float16推理模式，仅CPU/GPU设备可用。若参数device_info为空则返回flase，并在日志中输出空指针信息。

- 参数
    - `device_info`: 指向设备类型信息的指针。

- 返回值

  是否开启FP16。

### MSDeviceInfoSetEnableFP16

```C
void MSDeviceInfoSetEnableFP16(MSDeviceInfoHandle device_info, bool is_fp16)
```

设置是否开启float16推理模式，仅CPU/GPU设备可用。若参数device_info为空则不会做任何操作，并在日志中输出空指针信息。

- 参数
    - `device_info`: 指向设备类型信息的指针。
    - `is_fp16`: 是否基于float16进行推理。

### MSDeviceInfoSetFrequency

```C
void MSDeviceInfoSetFrequency(MSDeviceInfoHandle device_info, int frequency)
```

设置NPU的频率类型，仅NPU设备可用。若参数device_info为空或者当前设备不为NPU则不会做任何操作，并在日志中输出具体信息。

- 参数
    - `device_info`: 指向设备类型信息的指针。
    - `frequency`: 频率类型，取值范围为1-4，默认是3。1表示低功耗，2表示平衡，3表示高性能，4表示超高性能。

### MSDeviceInfoGetFrequency

```C
int MSDeviceInfoGetFrequency(const MSDeviceInfoHandle device_info)
```

获取NPU的频率类型，仅NPU设备可用。若参数device_info为空或者当前设备不为NPU则返回-1，并在日志中输出具体信息。

- 参数
    - `device_info`: 指向设备类型信息的指针。

- 返回值

  NPU的频率类型。

