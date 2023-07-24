# MSConfig

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/api_java/source_en/msconfig.md)

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

## init

```java
public boolean init(int deviceType, int threadNum, int cpuBindMode, boolean enable_float16)
```

Initialize MSConfig.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)** **enum** type.- `threadNum`: Thread number config for thread pool.
    - `cpuBindMode`: A **[CpuBindMode](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/CpuBindMode.java)** **enum** variable.
    - `enable_float16`：Whether to use float16 operator for priority.

- Returns

  Whether the initialization is successful.

```java
public boolean init(int deviceType, int threadNum, int cpuBindMode)
```

Initialize MSConfig.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)** **enum** type.
    - `threadNum`: Thread number config for thread pool.
    - `cpuBindMode`: A **[CpuBindMode](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/CpuBindMode.java)** **enum** variable.

- Returns

  Whether the initialization is successful.

```java
public boolean init(int deviceType, int threadNum)
```

Initialize MSConfig, `cpuBindMode` defaults to `CpuBindMode.MID_CPU`.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)** **enum** type.
    - `threadNum`: Thread number config for thread pool.

- Returns

  Whether the initialization is successful.

```java
public boolean init(int deviceType)
```

Initialize MSConfig，`cpuBindMode` defaults to `CpuBindMode.MID_CPU`, `threadNum` defaults to `2`.

- Parameters

    - `deviceType`: A **[DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)** **enum** type.

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
