# ModelParallelRunner

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/lite/api/source_zh_cn/api_java/model_parallel_runner.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

```java
import com.mindspore.config.RunnerConfig;
```

ModelParallelRunner定义了MindSpore Lite并发推理。

## 公有成员函数

| function                                                       |
| ------------------------------------------------------------   |
| [long getModelParallelRunnerPtr()](#getmodelparallelrunnerptr) |
| [boolean init()](#init)                                        |
| [boolean predict()](#predict)                                  |
| [boolean getInputs()](#getinputs)                              |
| [boolean getOutputs()](#getoutputs)                            |
| [void free()](#free)                                           |

## getModelParallelRunnerPtr

```java
public long getModelParallelRunnerPtr()
```

获取底层并发推理类指针。

- 返回值

  底层并发推理类指针。

## init

```java
public boolean init(String modelPath, RunnerConfig runnerConfig)
```

根据路径读取加载模型，生成一个或者多个模型，并将所有模型编译至可在Device上运行的状态。

- 参数

    - `modelPath`: 模型文件路径。

    - `runnerConfig`: 一个RunnerConfig结构体。定义了并发推理模型的配置参数。

- 返回值

  是否初始化成功。

```java
public boolean init(String modelPath)
```

根据路径读取加载模型，生成一个或者多个模型，并将所有模型编译至可在Device上运行的状态。

- 参数

    - `modelPath`: 模型文件路径。

- 返回值

  是否初始化成功。

## predict

```java
public boolean predict(List<MSTensor> inputs, List<MSTensor> outputs)
```

并发推理模型。

- 参数

    - `inputs`: 模型输入。

    - `outputs`: 模型输出。

- 返回值

  是否推理成功。

## getInputs

```java
public List<MSTensor> getInputs()
```

获取模型所有输入张量。

- 返回值

  模型的输入张量列表。

## getOutputs

```java
public List<MSTensor> getOutputs()
```

获取模型所有输入张量。

- 返回值

  模型的输出张量列表。

## free

```java
public void free()
```

释放并发推理类内存。
