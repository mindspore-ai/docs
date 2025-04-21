# RunnerConfig

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_java/runner_config.md)

RunnerConfig定义了MindSpore Lite并发推理的配置参数。

## 公有成员函数

| function                                                       | 云侧推理是否支持 | 端侧推理是否支持 |
| ------------------------------------------------------------   |--------|--------|
| [boolean init()](#init)                            | √      | ✕      |
| [public boolean init(MSContext msContext)](#init)  | √      | ✕      |
| [public void setWorkerNum(int workerNum)](#setworkernum)                           | √      | ✕      |
| [public void setConfigInfo(String section, HashMap<String, String> config)](#setconfiginfo)               | √      | ✕      |
| [public void setConfigPath(String configPath)](#setconfigpath)                         | √      | ✕      |
| [void getConfigPath()](#getconfigpath)                         | √      | ✕      |
| [long getRunnerConfigPtr()](#getrunnerconfigptr)               | √      | ✕      |
| [public void setDeviceIds(ArrayList<Integer\> deviceIds)](#setdeviceids)               | √      | ✕      |
| [public ArrayList<Integer\> getDeviceIds()](#getdeviceids)               | √      | ✕      |
| [public void free()](#free)    | √      | ✕      |

## init

```java
public boolean init()
```

并发推理的配置参数初始化。

- 返回值

  是否初始化成功。

```java
public boolean init(MSContext msContext)
```

并发推理的配置参数初始化。

- 参数

    - `msContext`: 并发推理运行时的上下文配置。

- 返回值

  是否初始化成功。

## setWorkerNum

```java
public void setWorkerNum(int workerNum)
```

并发推理中模型个数参数设置。

- 参数

    - `workerNum`: 模型个数。

## setConfigInfo

```java
public void setConfigInfo(String section, HashMap<String, String> config)
```

并发推理中模型配置参数设置。

- 参数

    - `section`: 配置的章节名。
    - `config`: 要更新的配置对。

## setConfigPath

```java
public void setConfigPath(String configPath)
```

并发推理中模型配置文件路径参数设置。

- 参数

    - `configPath`: 配置文件路径。

## getConfigPath

```java
public String getConfigPath()
```

获取RunnerConfig中设置的配置文件的路径。

- 返回值

  配置文件路径。

## getRunnerConfigPtr

```java
public long getRunnerConfigPtr()
```

获取底层并发推理配置参数指针。

- 返回值

  底层并发推理配置参数指针。

## setDeviceIds

```java
public void setDeviceIds(ArrayList<Integer> deviceIds)
```

并发推理中设置设备ID列表。

- 参数

    - `deviceIds`: 设备ID列表。

## getDeviceIds

```java
public ArrayList<Integer> getDeviceIds()
```

获取RunnerConfig中设置的设备ID列表。

- 返回值

  设备ID列表。

## free

```java
public void free()
```

释放runnerConfig。

