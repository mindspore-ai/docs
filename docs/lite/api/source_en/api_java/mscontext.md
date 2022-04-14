# MSContext

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/lite/api/source_en/api_java/mscontext.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

```java
import com.mindspore.config.MSContext;
```

MSContext is defined for holding environment variables during runtime.

## Public Member Functions

| function                                                     |
| ------------------------------------------------------------ |
| [boolean init()](#init) |
| [boolean init(int threadNum, int cpuBindMode)](#init) |
| [boolean init(int threadNum, int cpuBindMode, boolean isEnableParallel)](#init)         |
| [boolean addDeviceInfo(int deviceType, boolean isEnableFloat16)](#adddeviceinfo)                        |
| [boolean addDeviceInfo(int deviceType, boolean isEnableFloat16, int npuFreq)](#adddeviceinfo)                                      |
| [void free()](#free)                                         |
| [long getMSContextPtr()](#getmscontextptr)                                         |
| [DeviceType](#devicetype)                                         |
| [CpuBindMode](#cpubindmode)                                         |

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
    - `cpuBindMode`: A **[CpuBindMode](https://gitee.com/mindspore/mindspore/blob/r1.7/mindspore/lite/java/src/main/java/com/mindspore/config/CpuBindMode.java)** **enum** variable.

- Returns

  Whether the initialization is successful.

```java
public boolean init(int threadNum, int cpuBindMode, boolean isEnableParallel)
```

Initialize MSContext.

- Parameters

    - `threadNum`: Thread number config for thread pool.
    - `cpuBindMode`: A **[CpuBindMode](https://gitee.com/mindspore/mindspore/blob/r1.7/mindspore/lite/java/src/main/java/com/mindspore/config/CpuBindMode.java)** **enum** variable.
    - `isEnableParallel`: Is enable parallel in different device.

- Returns

  Whether the initialization is successful.

## addDeviceInfo

```java
public boolean addDeviceInfo(int deviceType, boolean isEnableFloat16)
```

Add device info for mscontext.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.7/mindspore/lite/java/src/main/java/com/mindspore/config/DeviceType.java)** **enum** type.
    - `isEnableFloat16`: Is enable fp16.

- Returns

  Whether the device info add is successful.

```java
boolean addDeviceInfo(int deviceType, boolean isEnableFloat16, int npuFreq)
```

Add device info for mscontext.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.7/mindspore/lite/java/src/main/java/com/mindspore/config/DeviceType.java)** **enum** type.
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
