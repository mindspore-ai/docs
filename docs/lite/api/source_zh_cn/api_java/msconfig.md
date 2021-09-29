# MSConfig

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/lite/api/source_zh_cn/api_java/msconfig.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

```java
import com.mindspore.lite.config.MSConfig;
```

MSConfig类用于保存执行中的配置变量。

## 公有成员函数

| function                                                     |
| ------------------------------------------------------------ |
| [boolean init(int deviceType, int threadNum, int cpuBindMode, boolean enable_float16)](#init) |
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

初始化MSConfig。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.lite.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)中定义。
    - `threadNum`: 线程数。
    - `cpuBindMode`: CPU绑定模式，`cpuBindMode`在[com.mindspore.lite.config.CpuBindMode](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/CpuBindMode.java)中定义。
    - `enable_float16`：是否优先使用float16算子。

- 返回值

  初始化是否成功。

```java
public boolean init(int deviceType, int threadNum, int cpuBindMode)
```

初始化MSConfig，`enable_float16`默认为false。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.lite.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)中定义。
    - `threadNum`: 线程数。
    - `cpuBindMode`: CPU绑定模式，`cpuBindMode`在[com.mindspore.lite.config.CpuBindMode](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/CpuBindMode.java)中定义。

- 返回值

  初始化是否成功。

```java
public boolean init(int deviceType, int threadNum)
```

初始化MSConfig，`cpuBindMode`默认为`CpuBindMode.MID_CPU`，`enable_float16`默认为false。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.lite.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)中定义。
    - `threadNum`: 线程数。

- 返回值

  初始化是否成功。

```java
public boolean init(int deviceType)
```

初始化MSConfig，`cpuBindMode`默认为`CpuBindMode.MID_CPU`，`threadNum`默认为`2`，`enable_float16`默认为false。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.lite.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/config/DeviceType.java)中定义。

- 返回值

  初始化是否成功。

```java
public boolean init()
```

初始化MSConfig，`deviceType`默认为`DeviceType.DT_CPU`，`cpuBindMode`默认为`CpuBindMode.MID_CPU`，`threadNum`默认为`2`，`enable_float16`默认为false。

- 返回值

  初始化是否成功。

## free

```java
public void free()
```

释放MSConfig运行过程中动态分配的内存。LiteSession init之后即可释放。

## getMSConfigPtr

```java
public long getMSConfigPtr()
```

获取MSConfig指针。

- 返回值

  MSConfig指针。

## DeviceType

```java
import com.mindspore.lite.config.DeviceType;
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
import com.mindspore.lite.config.CpuBindMode;
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
