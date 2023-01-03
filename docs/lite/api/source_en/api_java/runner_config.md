# RunnerConfig

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/lite/api/source_en/api_java/runner_config.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

RunnerConfig defines the configuration parameters of MindSpore Lite concurrent inference.

## Public Member Functions

| function                                                       |
| ------------------------------------------------------------   |
| [boolean init()](#init)                                        |
| [void setWorkerNum()](#setworkernum)                           |
| [void setConfigInfo()](#setconfiginfo)                         |
| [void setConfigPath()](#setconfigpath)                         |
| [void getConfigPath()](#getconfigpath)                         |
| [long getRunnerConfigPtr()](#getrunnerconfigptr)               |

## init

```java
public boolean init()
```

Configuration parameter initialization for parallel inference.

- Returns

  Whether the initialization is successful.

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

## setConfigInfo

```java
public void setConfigInfo(String section, HashMap<String, String> config)
```

Model configuration parameter settings in parallel inference.

- Parameters

    - `section`: Configured chapter name.
    - `config`: config pair to update.

## setConfigPath

```java
public void setConfigPath(String configPath)
```

Set the configuration file path in concurrent inference.

- Parameters

    - `configPath`: config path.

## getConfigPath

```java
public String getConfigPath()
```

Get the path to the configuration file set in RunnerConfig.

- Returns

  config path.

## getRunnerConfigPtr

```java
public long getRunnerConfigPtr()
```

Get a pointer to the underlying concurrent inference configuration parameters.

- Returns

  Low-level concurrent inference configuration parameter pointer.
