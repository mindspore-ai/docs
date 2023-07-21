# RunnerConfig

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.8/docs/lite/api/source_zh_cn/api_java/runner_config.md)

RunnerConfig定义了MindSpore Lite并发推理的配置参数。

## 公有成员函数

| function                                                       |
| ------------------------------------------------------------   |
| [boolean init()](#init)                                        |
| [boolean setWorkerNum()](#setworkernum)                          |
| [long getRunnerConfigPtr()](#getrunnerconfigptr)               |

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

    - `workerNum`: 配置文件中设置模型个数。

## getRunnerConfigPtr

```java
public long getRunnerConfigPtr()
```

获取底层并发推理配置参数指针。

- 返回值

  底层并发推理配置参数指针。

## free

```java
public void free()
```

释放runnerConfig。

