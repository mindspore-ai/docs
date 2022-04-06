# RunnerConfig

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_en/api_java/runner_config.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

RunnerConfig定义了MindSpore Lite并发推理的配置参数。

## Public Member Functions

| function                                                       |
| ------------------------------------------------------------   |
| [boolean init()](#init)                                        |
| [boolean setWorkerNum()](#setworkernum)                          |
| [long getRunnerConfigPtr()](#getrunnerconfigptr)               |

## init

```java
public boolean init(MSContext msContext)
```

Configuration parameter initialization for parallel inference.

- Parameters

    - `msContext`: Context configuration for concurrent inference runtime.

- Returns

  Whether the initialization is successful.

## setWorkerNum

```java
public void setWorkerNum(int workerNum)
```

The parameter setting of the number of models in parallel inference.

- Parameters

    - `workerNum`: Set the number of models in the configuration.

## getRunnerConfigPtr

```java
public long getRunnerConfigPtr()
```

Get a pointer to the underlying concurrent inference configuration parameters.

- Returns

  Low-level concurrent inference configuration parameter pointer.
