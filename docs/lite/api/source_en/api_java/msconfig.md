# MSConfig

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_en/api_java/msconfig.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

```java
import com.mindspore.lite.config.MSConfig;
```

MSConfig is defined for holding environment variables during runtime.

## Public Member Functions

| function                                                     |
| ------------------------------------------------------------ |
| [public boolean init(int deviceType, int threadNum, int cpuBindMode)](#init) |
| [boolean init(int deviceType, int threadNum, int cpuBindMode)](#init) |
| [boolean init(int deviceType, int threadNum)](#init)         |
| [boolean init(int deviceType)](#init)                        |
| [boolean init()](#init)                                      |
| [void free()](#free)                                         |
| [long getMSConfigPtr()](#getmsconfigptr)                     |
| [DeviceType](#devicetype)                                    |
| [CpuBindMode](#cpubindmode)                                  |

## init

```java
public boolean init(int deviceType, int threadNum, int cpuBindMode, boolean enable_float16)
```

Initialize MSConfig.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)** **enum** type.- `threadNum`: Thread number config for thread pool.
    - `cpuBindMode`: A **[CpuBindMode](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/CpuBindMode.java)** **enum** variable.
    - `enable_float16`：Whether to use float16 operator for priority.

- Returns

  Whether the initialization is successful.

```java
public boolean init(int deviceType, int threadNum, int cpuBindMode)
```

Initialize MSConfig.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)** **enum** type.
    - `threadNum`: Thread number config for thread pool.
    - `cpuBindMode`: A **[CpuBindMode](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/CpuBindMode.java)** **enum** variable.

- Returns

  Whether the initialization is successful.

```java
public boolean init(int deviceType, int threadNum)
```

Initialize MSConfig, `cpuBindMode` defaults to `CpuBindMode.MID_CPU`.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)** **enum** type.
    - `threadNum`: Thread number config for thread pool.

- Returns

  Whether the initialization is successful.

```java
public boolean init(int deviceType)
```

Initialize MSConfig，`cpuBindMode` defaults to `CpuBindMode.MID_CPU`, `threadNum` defaults to `2`.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)** **enum** type.

- Returns

  Whether the initialization is successful.

```java
public boolean init()
```

Initialize MSConfig，`deviceType` defaults to `DeviceType.DT_CPU`，`cpuBindMode` defaults to`CpuBindMode.MID_CPU`，`threadNum` defaults to `2`.

- Returns

  Whether the initialization is successful.

## free

```java
public void free()
```

Free all temporary memory in MindSpore Lite MSConfig.

## getMSConfigPtr

```java
public long getMSConfigPtr()
```

Get msconfig pointer.

- Returns

  Return msconfig pointer.

## DeviceType

```java
import com.mindspore.lite.config.DeviceType;
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
import com.mindspore.lite.config.CpuBindMode;
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
