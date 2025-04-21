# ModelParallelRunner

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_en/api_java/model_parallel_runner.md)

```java
import com.mindspore.config.RunnerConfig;
```

ModelParallelRunner defines MindSpore Lite concurrent inference.

## Public Member Functions

| function                                                       | Supported At Cloud-side Inference | Supported At Device-side Inference |
| ------------------------------------------------------------   |--------|--------|
| [long getModelParallelRunnerPtr()](#getmodelparallelrunnerptr) | √      | ✕      |
| [boolean init(String modelPath, RunnerConfig runnerConfig)](#init)         | √      | ✕      |
| [public boolean init(String modelPath)](#init) | √      | ✕      |
| [public boolean predict(List<MSTensor> inputs, List<MSTensor> outputs)](#predict)   | √      | ✕      |
| [boolean getInputs()](#getinputs)                              | √      | ✕      |
| [boolean getOutputs()](#getoutputs)                            | √      | ✕      |
| [void free()](#free)                                           | √      | ✕      |

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

Read and load models according to the path, generate one or more models, and compile all models to a state that can be run on the Device. Supports importing the `ms` model (exported by the `converter_lite` tool) and the `mindir` model (exported by MindSpore or exported by the `converter_lite` tool). The support for the `ms` model will be removed in future iterations, and it is recommended to use the `mindir` model for inference. When using the `ms` model for inference, please keep the suffix name of the model as `.ms`, otherwise it will not be recognized.

- Parameters

    - `modelPath`: model file path.

    - `runnerConfig`: A RunnerConfig structure. Defines configuration parameters for the concurrent inference model.

- Returns

  Whether the initialization is successful.

```java
public boolean init(String modelPath)
```

Read and load models according to the path, generate one or more models, and compile all models to a state that can be run on the Device. Supports importing the `ms` model (exported by the `converter_lite` tool) and the `mindir` model (exported by MindSpore or exported by the `converter_lite` tool). The support for the `ms` model will be removed in future iterations, and it is recommended to use the `mindir` model for inference. When using the `ms` model for inference, please keep the suffix name of the model as `.ms`, otherwise it will not be recognized.

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
