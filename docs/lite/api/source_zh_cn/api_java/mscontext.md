# MSContext

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_java/mscontext.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

```java
import com.mindspore.config.MSContext;
```

MSContext类用于配置运行时的上下文配置。

## 公有成员函数

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

使用默认参数初始化MSContext，2线程，不绑核，不开启异构并行。

- 返回值

  初始化是否成功。

```java
public boolean init(int threadNum, int cpuBindMode)
```

使用线程数和绑＆模式初始化MSContext。

- 参数

    - `threadNum`: 线程数。
    - `cpuBindMode`: CPU绑定模式，`cpuBindMode`在[com.mindspore.config.CpuBindMode](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/src/main/java/com/mindspore/config/CpuBindMode.java)中定义。

- 返回值

  初始化是否成功。

```java
public boolean init(int threadNum, int cpuBindMode, boolean isEnableParallel)
```

初始化MSContext。

- 参数

    - `threadNum`: 线程数。
    - `cpuBindMode`: CPU绑定模式，`cpuBindMode`在[com.mindspore.config.CpuBindMode](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/src/main/java/com/mindspore/config/CpuBindMode.java)中定义。
    - `isEnableParallel`: 是否开启异构并行。

- 返回值

  初始化是否成功。

## addDeviceInfo

```java
boolean addDeviceInfo(int deviceType, boolean isEnableFloat16)
```

添加运行设备信息。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/src/main/java/com/mindspore/config/DeviceType.java)中定义。
    - `isEnableFloat16`: 是否开启fp16。

- 返回值

  设备添加是否成功。

```java
boolean addDeviceInfo(int deviceType, boolean isEnableFloat16, int npuFreq)
```

添加运行设备信息。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/src/main/java/com/mindspore/config/DeviceType.java)中定义。
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
