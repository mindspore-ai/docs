# MSConfig

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/api_java/source_zh_cn/msconfig.md)

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

## init

```java
public boolean init(int deviceType, int threadNum, int cpuBindMode, boolean enable_float16)
```

初始化MSConfig。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.lite.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/java/java/app/src/main/java/com/mindspore/lite/config/DeviceType.java)中定义。
    - `threadNum`: 线程数。
    - `cpuBindMode`: CPU绑定模式，`cpuBindMode`在[com.mindspore.lite.config.CpuBindMode](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/java/java/app/src/main/java/com/mindspore/lite/config/CpuBindMode.java)中定义。
    - `enable_float16`：是否优先使用float16算子。

- 返回值

  初始化是否成功。

```java
public boolean init(int deviceType, int threadNum, int cpuBindMode)
```

初始化MSConfig，`enable_float16`默认为false。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.lite.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/java/java/app/src/main/java/com/mindspore/lite/config/DeviceType.java)中定义。
    - `threadNum`: 线程数。
    - `cpuBindMode`: CPU绑定模式，`cpuBindMode`在[com.mindspore.lite.config.CpuBindMode](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/java/java/app/src/main/java/com/mindspore/lite/config/CpuBindMode.java)中定义。

- 返回值

  初始化是否成功。

```java
public boolean init(int deviceType, int threadNum)
```

初始化MSConfig，`cpuBindMode`默认为`CpuBindMode.MID_CPU`，`enable_float16`默认为false。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.lite.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/java/java/app/src/main/java/com/mindspore/lite/config/DeviceType.java)中定义。
    - `threadNum`: 线程数。

- 返回值

  初始化是否成功。

```java
public boolean init(int deviceType)
```

初始化MSConfig，`cpuBindMode`默认为`CpuBindMode.MID_CPU`，`threadNum`默认为`2`，`enable_float16`默认为false。

- 参数

    - `deviceType`: 设备类型，`deviceType`在[com.mindspore.lite.config.DeviceType](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/java/java/app/src/main/java/com/mindspore/lite/config/DeviceType.java)中定义。

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
