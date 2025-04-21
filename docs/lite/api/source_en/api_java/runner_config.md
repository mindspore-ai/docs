# RunnerConfig

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_en/api_java/runner_config.md)

RunnerConfig defines the configuration parameters of MindSpore Lite concurrent inference.

## Public Member Functions

| function                                                       | Supported At Cloud-side Inference | Supported At Device-side Inference |
| ------------------------------------------------------------   |--------|--------|
| [boolean init()](#init)                            | √      | ✕      |
| [public boolean init(MSContext msContext)](#init)  | √      | ✕      |
| [public void setWorkerNum(int workerNum)](#setworkernum)                           | √      | ✕      |
| [public void setConfigInfo(String section, HashMap<String, String> config)](#setconfiginfo)               | √      | ✕      |
| [public void setConfigPath(String configPath)](#setconfigpath)                         | √      | ✕      |
| [void getConfigPath()](#getconfigpath)                         | √      | ✕      |
| [long getRunnerConfigPtr()](#getrunnerconfigptr)               | √      | ✕      |
| [public void setDeviceIds(ArrayList<Integer\> deviceIds)](#setdeviceids)                         | √      | ✕      |
| [public ArrayList<Integer\> getDeviceIds()](#getdeviceids)                         | √      | ✕      |

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

## setDeviceIds

```java
public void setDeviceIds(ArrayList<Integer> deviceIds)
```

Set the list of device id in concurrent inference.

- Parameters

    - `deviceIds`: list of device id.

## getDeviceIds

```java
public ArrayList<Integer> getDeviceIds()
```

Get the list of device id set in RunnerConfig.

- Returns

  list of device id.
