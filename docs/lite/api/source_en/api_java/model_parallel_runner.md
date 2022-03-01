# ModelParallelRunner

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_en/api_java/model_parallel_runner.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

```java
import com.mindspore.config.RunnerConfig;
```

ModelParallelRunner defines MindSpore Lite concurrent inference.

## Public Member Functions

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

Get the underlying concurrent inference class pointer.

- Returns

  Low-level concurrent inference class pointer.

## init

```java
public boolean init(String modelPath, RunnerConfig runnerConfig)
```

Read and load models according to the path, generate one or more models, and compile all models to a state that can be run on the Device.

- Parameters

    - `modelPath`: model file path.

    - `runnerConfig`: A RunnerConfig structure. Defines configuration parameters for the concurrent inference model.

- Returns

  Whether the initialization is successful.

```java
public boolean init(String modelPath)
```

Read and load models according to the path, generate one or more models, and compile all models to a state that can be run on the Device.

- Parameters

    - `modelPath`: model file path.

- Returns

  Whether the initialization is successful.

## predict

```java
public boolean predict(List<MSTensor> inputs, List<MSTensor> outputs)
```

Concurrent inference model.

- Parameters

    - `inputs`: model input.

    - `outputs`: model output.

- Returns

  Whether the inference is successful.

## getInputs

```java
public List<MSTensor> getInputs()
```

Get all input tensors of the model.

- Returns

  A list of input tensors for the model.

## getOutputs

```java
public List<MSTensor> getOutputs()
```

Get all input tensors of the model.

- Returns

  A list of output tensors for the model.

## free

```java
public void free()
```

Free concurrent inference class memory.
