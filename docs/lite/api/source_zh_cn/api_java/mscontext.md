# MSContext

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_java/mscontext.md)

```java
import com.mindspore.config.MSContext;
```

MSContext类用于配置运行时的上下文配置。

## 公有成员函数

| function                                                                                      | 云侧推理是否支持 | 端侧推理是否支持 |
| --------------------------------------------------------------------------------------------- |--------|--------|
| [boolean init()](#init)                                                                       | √      | √      |
| [boolean init(int threadNum, int cpuBindMode)](#init)                                         | √      | √      |
| [boolean init(int threadNum, int cpuBindMode, boolean isEnableParallel)](#init)               | ✕      | √      |
| [boolean addDeviceInfo(int deviceType, boolean isEnableFloat16)](#adddeviceinfo)              | √      | √      |
| [boolean addDeviceInfo(int deviceType, boolean isEnableFloat16, int npuFreq)](#adddeviceinfo) | ✕      | √      |
| [void free()](#free)                                                                          | √      | √      |
| [long getMSContextPtr()](#getmscontextptr)                                                    | √      | √      |
| [void setThreadNum(int threadNum)](#setenableparallel)                                        | √      | √      |
| [int getThreadNum()](#getenableparallel)                                                      | √      | √      |
| [void setInterOpParallelNum(int parallelNum)](#setinteropparallelnum)                         | √      | √      |
| [int getInterOpParallelNum()](#getinteropparallelnum)                                         | √      | √      |
| [void setThreadAffinity(int mode)](#setthreadaffinity)                                        | √      | √      |
| [int getThreadAffinityMode()](#getthreadaffinitycorelist)                                     | √      | √      |
| [void setThreadAffinity(ArrayList<Integer\> coreList)](#setthreadaffinity-1)                 | √      | √      |
| [ArrayList<Integer\> getThreadAffinityCoreList()](#getthreadaffinitycorelist)                | √      | √      |
| [void setEnableParallel(boolean isParallel)](#setenableparallel)                              | ✕      | √      |
| [boolean getEnableParallel()](#getenableparallel)                                             | ✕      | √      |
| [DeviceType](#devicetype)                                                                     | √      | √      |
| [CpuBindMode](#cpubindmode)                                                                   | √      | √      |

## init

```java
public boolean init()
```

初始化MSContext，采用默认配置：2线程，不绑核，不开启异构并行。

- 返回值

  初始化是否成功。

```java
public boolean init(int threadNum, int cpuBindMode)
```

初始化MSContext，设置CPU线程数、CPU绑定模式。不开启异构并行。

- 参数

    - `threadNum`: 线程数。
    - `cpuBindMode`: CPU绑定模式，`cpuBindMode`在[com.mindspore.config.CpuBindMode](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/CpuBindMode.java)中定义。

- 返回值

  初始化是否成功。

```java
public boolean init(int threadNum, int cpuBindMode, boolean isEnableParallel)
```

初始化MSContext，设置CPU线程数、CPU绑定模式、是否开启异构并行。

- 参数

    - `threadNum`: 线程数。
    - `cpuBindMode`: CPU绑定模式，`cpuBindMode`在[com.mindspore.config.CpuBindMode](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/CpuBindMode.java)中定义。
    - `isEnableParallel`: 是否开启异构并行。

- 返回值

  初始化是否成功。

## addDeviceInfo

```java
boolean addDeviceInfo(int deviceType, boolean isEnableFloat16)
```

添加运行设备信息。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/DeviceType.java)中定义。
    - `isEnableFloat16`: 是否开启fp16。

- 返回值

  设备添加是否成功。

```java
boolean addDeviceInfo(int deviceType, boolean isEnableFloat16, int npuFreq)
```

添加运行设备信息。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/DeviceType.java)中定义。
    - `isEnableFloat16`: 是否开启fp16。
    - `npuFreq`: NPU运行频率，仅当deviceType为npu才需要。

- 返回值

  设备添加是否成功。

## getMSContextPtr

```java
public long getMSContextPtr()
```

获取MSContext底层运行指针。

- 返回值

  MSContext底层运行指针。

## free

```java
public void free()
```

释放MSContext运行过程中动态分配的内存。

## setThreadNum

```java
void setThreadNum(int threadNum)
```

设置运行时的线程数量。
若未初始化 MSContext 则不会做任何操作，并在日志中输出空指针信息。

- 参数

    - `threadNum`: 运行时的线程数。

## getThreadNum

```java
int getThreadNum()
```

获取当MSContext的线程数量设置，该选项仅MindSpore Lite有效。
若未初始化 MSContext 则会返回-1，并在日志中输出空指针信息。

- 返回值

  线程数量

## setInterOpParallelNum

```java
void setInterOpParallelNum(int parallelNum)
```

设置运行时的算子并行推理数目。
若未初始化 MSContext 则不会做任何操作，并在日志中输出空指针信息。

- 参数

    - `parallelNum`: 运行时的算子并行数。

## getInterOpParallelNum

```java
int getInterOpParallelNum()
```

获取当前算子并行数设置。
若未初始化 MSContext 则会返回-1，并在日志中输出空指针信息。

- 返回值

  当前算子并行数设置。

## setThreadAffinity

```java
void setThreadAffinity(int mode)
```

设置运行时的CPU绑核策略。
若未初始化 MSContext 则不会做任何操作，并在日志中输出空指针信息。

- 参数

    - `mode`: 绑核的模式，有效值为0-2，0为默认不绑核，1为绑大核，2为绑小核。

## getThreadAffinityMode

```java
 int getThreadAffinityMode()
```

获取当前CPU绑核策略。
若未初始化 MSContext 则返回-1，并在日志中输出空指针信息。

- 返回值

  当前CPU绑核策略，有效值为0-2，0为默认不绑核，1为绑大核，2为绑小核。

## setThreadAffinity

```java
void setThreadAffinity(ArrayList<Integer> coreList)
```

设置运行时的CPU绑核列表，如果同时调用了两个不同的`SetThreadAffinity`函数来设置同一个的MSContext，仅`coreList`生效，而`mode`不生效。该选项仅MindSpore Lite有效。
若未初始化 MSContext 则不会做任何操作，并在日志中输出空指针信息。

- 参数

    - `coreList`: CPU绑核的列表。

## getThreadAffinityCoreList

```java
ArrayList<Integer> getThreadAffinityCoreList()
```

获取当前CPU绑核列表。
若未初始化 MSContext 则会返回长度为0的`ArrayList`，并在日志中输出空指针信息。  

- 返回值

  当前CPU绑核列表

## setEnableParallel

```java
void setEnableParallel(boolean isParallel)
```

设置运行时是否使能异构并行。
若未初始化 MSContext 则不会做任何操作，并在日志中输出空指针信息。

- 参数

    - `isParallel`: 为true则使能异构并行。

## getEnableParallel

```java
boolean getEnableParallel()
```

获取当前是否使能异构并行。
若未初始化 MSContext 则会返回false，并在日志中输出空指针信息。

- 返回值

  返回值为为true，代表使能异构并行。

## DeviceType

```java
import com.mindspore.config.DeviceType;
```

设备类型。

### 公有成员变量

```java
public static final int DT_CPU = 0;
public static final int DT_GPU = 1;
public static final int DT_NPU = 2;
```

DeviceType的值为0，指定设备类型为CPU。

DeviceType的值为1，指定设备类型为GPU。

DeviceType的值为2，指定设备类型为NPU。

## CpuBindMode

```java
import com.mindspore.config.CpuBindMode;
```

绑核策略。

### 公有成员变量

```java
public static final int MID_CPU = 2;
public static final int HIGHER_CPU = 1;
public static final int NO_BIND = 0;
```

CpuBindMode的值为2，优先绑定中核。

CpuBindMode的值为1，优先绑定大核。

CpuBindMode的值为0，不绑核。
