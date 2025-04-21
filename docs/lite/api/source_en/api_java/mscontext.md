# MSContext

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_en/api_java/mscontext.md)

```java
import com.mindspore.config.MSContext;
```

MSContext is defined for holding environment variables during runtime.

## Public Member Functions

| function                                                                                      | Supported At Cloud-side Inference | Supported At Device-side Inference |
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

Use default parameters initialize MSContext, use two thread, no bind, no parallel.

- Returns

  Whether the initialization is successful.

```java
public boolean init(int threadNum, int cpuBindMode)
```

Initialize MSContext for cpu.

- Parameters

    - `threadNum`: Thread number config for thread pool.
    - `cpuBindMode`: A **[CpuBindMode](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/CpuBindMode.java)** **enum** variable.

- Returns

  Whether the initialization is successful.

```java
public boolean init(int threadNum, int cpuBindMode, boolean isEnableParallel)
```

Initialize MSContext.

- Parameters

    - `threadNum`: Thread number config for thread pool.
    - `cpuBindMode`: A **[CpuBindMode](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/CpuBindMode.java)** **enum** variable.
    - `isEnableParallel`: Is enable parallel in different device.

- Returns

  Whether the initialization is successful.

## addDeviceInfo

```java
public boolean addDeviceInfo(int deviceType, boolean isEnableFloat16)
```

Add device info for mscontext.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/DeviceType.java)** **enum** type.
    - `isEnableFloat16`: Is enable fp16.

- Returns

  Whether the device info add is successful.

```java
boolean addDeviceInfo(int deviceType, boolean isEnableFloat16, int npuFreq)
```

Add device info for mscontext.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/DeviceType.java)** **enum** type.
    - `isEnableFloat16`: is enable fp16.
    - `npuFreq`: Npu frequency.

- Returns

  Whether the device info add is successful.

## getMSContextPtr

```java
public long getMSContextPtr()
```

Get MSContext pointer.

- Returns

  The MSContext pointer.

## free

```java
public void free()
```

Free the memory allocated for MSContext.

## setThreadNum

```java
void setThreadNum(int threadNum)
```

Sets the number of threads at runtime.
If MSContext is not initialized, this function will do nothing and output null pointer information in the log.

- Parameters

    - `threadNum`: Number of threads at runtime.

## getThreadNum

```java
void int getThreadNum()
```

Get the current thread number setting.
If MSContext is not initialized, this function will do nothing and output null pointer information in the log.

- Returns

    The current thread number setting.

## setInterOpParallelNum

```java
void setInterOpParallelNum(int parallelNum)
```

Set the parallel number of operators at runtime.
If MSContext is not initialized, this function will do nothing and output null pointer information in the log.

- Parameters

    - `parallelNum`: the parallel number of operators at runtime.

## getInterOpParallelNum

```java
int getInterOpParallelNum()
```

et the current operators parallel number setting.
If MSContext is not initialized, this function will return -1 and output null pointer information in the log.

- Returns

    The current operators parallel number setting.

## setThreadAffinity

```java
void setThreadAffinity(int mode)
```

Set the thread affinity to CPU cores.
If MSContext is not initialized, this function will do nothing and output null pointer information in the log.

- Parameters

    - `mode`: 0, no affinities; 1, big cores first; 2, little cores first.

## getThreadAffinityMode

```java
 int getThreadAffinityMode()
```

Get the thread affinity of CPU cores.
If MSContext is not initialized, this function will return -1 and output null pointer information in the log.

- Returns

      Thread affinity to CPU cores. 0, no affinities; 1, big cores first; 2, little cores first.

## setThreadAffinity

```java
void setThreadAffinity(ArrayList<Integer> coreList)
```

Set the thread lists to CPU cores, if two different `setThreadAffinity` are set for a single MSContext at the same time, only `coreList` will take effect and `mode` will not.
If MSContext is not initialized, this function will do nothing and output null pointer information in the log.

- Parameters

    - `coreList`:  An Arraylist of thread core list.

## getThreadAffinityCoreList

```java
ArrayList<Integer> getThreadAffinityCoreList()
```

 Get the thread lists of CPU cores.
If MSContext is not initialized, this function will retutn an empty Arraylist and output null pointer information in the log.

- Returns

    An Arraylist of thread core list.

## setEnableParallel

```java
void setEnableParallel(boolean isParallel)
```

Set the status whether to perform model inference or training in parallel.
If MSContext is not initialized, this function will do nothing and output null pointer information in the log.

- Parameters

    - `isParallel`: true, parallel; false, not in parallel.

## getEnableParallel

```java
boolean getEnableParallel()
```

Get the status whether to perform model inference or training in parallel.
If MSContext is not initialized, this function will return `false` and output null pointer information in the log.

- Returns

    Bool value that indicates whether in parallel. true, parallel; false, not in parallel.

## DeviceType

```java
import com.mindspore.config.DeviceType;
```

Define device type.

### Public member variable

```java
public static final int DT_CPU = 0;
public static final int DT_GPU = 1;
public static final int DT_NPU = 2;
```

The value of DeviceType is 0, and the specified device type is CPU.

The value of DeviceType is 1, and the specified device type is GPU.

The value of DeviceType is 2, and the specified device type is NPU.

## CpuBindMode

```java
import com.mindspore.config.CpuBindMode;
```

Define CPU core bind mode.

### Public member variable

```java
public static final int MID_CPU = 2;
public static final int HIGHER_CPU = 1;
public static final int NO_BIND = 0;
```

The value of CpuBindMode is 2, and the middle core is bound first.

The value of CpuBindMode is 1, and higher cores are bound first.

The value of CpuBindMode is 0, and no core is bound.
