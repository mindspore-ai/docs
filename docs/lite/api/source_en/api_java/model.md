# Model

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_en/api_java/model.md)

```java
import com.mindspore.model;
```

Model defines model in MindSpore for compiling and running.

## Public Member Functions

| function                                                     | Supported At Cloud-side Inference | Supported At Device-side Inference |
| ------------------------------------------------------------ |--------|--------|
| [boolean build(MappedByteBuffer buffer, int modelType, MSContext context, char[] dec_key, String dec_mode)](#build)           | ✕      | √      |
| [boolean build(Graph graph, MSContext context, TrainCfg cfg)](#build) | ✕      | √      |
| [boolean build(MappedByteBuffer buffer, MSContext context)](#build)                            | √      | √      |
| [boolean build(String modelPath, MSContext context, char[] dec_key, String dec_mode)](#build)  | ✕      | √      |
| [boolean build(String modelPath, MSContext context)](#build)                                         | √      | √      |
| [boolean predict()](#predict)                                         | √      | √      |
| [boolean runStep()](#runstep)                                         | ✕      | √      |
| [boolean resize(List<MSTensor\> inputs, int[][] dims)](#resize)                                         | √      | √      |
| [List<MSTensor\> getInputs()](#getinputs)                                         | √      | √      |
| [List<MSTensor\> getOutputs()](#getoutputs)                                         | √      | √      |
| [MSTensor getInputsByTensorName(String tensorName)](#getinputsbytensorname)                                         | √      | √      |
| [MSTensor getOutputByTensorName(String tensorName)](#getoutputbytensorname)                                         | √      | √      |
| [List<MSTensor\> getOutputsByNodeName(String nodeName)](#getoutputsbynodename)                                         | ✕      | √      |
| [List<String\> getOutputTensorNames()](#getoutputtensornames)                                         | √      | √      |
| [boolean export(String fileName, int quantizationType, boolean isOnlyExportInfer,List<String\> outputTensorNames)](#export)             | ✕      | √      |
| [boolean exportWeightsCollaborateWithMicro(String weightFile, boolean isInference,boolean enableFp16, List<String> changeableWeightNames)](#exportweightscollaboratewithmicro)             | ✕      | √      |
| [List<MSTensor\> getFeatureMaps()](#getfeaturemaps)                                         | ✕      | √      |
| [boolean updateFeatureMaps(List<MSTensor\> features)](#updatefeaturemaps)                                         | ✕      | √      |
| [boolean setTrainMode(boolean isTrain)](#settrainmode)                                         | ✕      | √      |
| [boolean getTrainMode()](#gettrainmode)                               | ✕      | √      |
| [boolean setLearningRate(float learning_rate)](#setlearningrate)                                         | ✕      | √      |
| [boolean setupVirtualBatch(int virtualBatchMultiplier, float learningRate, float momentum)](#setupvirtualbatch)                           | ✕      | √      |
| [void free()](#free)                                    | √      | √      |
| [ModelType](#modeltype)                                    | √      | √      |

## build

```java
public boolean build(Graph graph, MSContext context, TrainCfg cfg)
```

Compile MindSpore model by computational graph.

- Parameters

    - `graph`: computational graph.
    - `context`: compile context.
    - `cfg`: train config.

- Returns

  Whether the build is successful.

```java
public boolean build(MappedByteBuffer buffer, int modelType, MSContext context, char[] dec_key, String dec_mode)
```

Compile MindSpore model by computational graph buffer.

- Parameters

    - `buffer`: computational graph buffer.
    - `modelType`: computational graph type. Optionally, there are `MT_MINDIR_LITE`, `MT_MINDIR`, corresponding to the `ms` model (exported by the `converter_lite` tool) and the `mindir` model (exported by MindSpore or exported by the `converter_lite` tool), respectively. Only MT_MINDIR_LITE is valid for Device-side Inference and the parameter value is ignored. Cloud-side inference supports `ms` and `mindir` model inference, which requires setting the parameter to the option value corresponding to the model. Cloud-side inference support for `ms` model will be removed in future iterations, and cloud-side inference via `mindir` model is recommended.
    - `context`: compile context.
    - `dec_key`: define the key used to decrypt the ciphertext model. The key length is 16, 24, or 32.
    - `dec_mode`: define the decryption mode. Options: AES-GCM, AES-CBC.

- Returns

  Whether the build is successful.

```java
public boolean build(final MappedByteBuffer buffer, int modelType, MSContext context)
```

Compile MindSpore model by computational graph buffer, the default is MindIR model type.

- Parameters

    - `buffer`: computational graph buffer.
    - `modelType`: computational graph type. Optionally, there are `MT_MINDIR_LITE`, `MT_MINDIR`, corresponding to the `ms` model (exported by the `converter_lite` tool) and the `mindir` model (exported by MindSpore or exported by the `converter_lite` tool), respectively. Only MT_MINDIR_LITE is valid for Device-side Inference and the parameter value is ignored. Cloud-side inference supports `ms` and `mindir` model inference, which requires setting the parameter to the option value corresponding to the model. Cloud-side inference support for `ms` model will be removed in future iterations, and cloud-side inference via `mindir` model is recommended.
    - `context`: compile context.

- Returns

  Whether the build is successful.

```java
public boolean build(String modelPath, int modelType, MSContext context, char[] dec_key, String dec_mode)
```

Compile MindSpore model by computational graph file.

- Parameters

    - `modelPath`: computational graph file.
    - `modelType`: computational graph type. Optionally, there are `MT_MINDIR_LITE`, `MT_MINDIR`, corresponding to the `ms` model (exported by the `converter_lite` tool) and the `mindir` model (exported by MindSpore or exported by the `converter_lite` tool), respectively. Only MT_MINDIR_LITE is valid for Device-side Inference and the parameter value is ignored. Cloud-side inference supports `ms` and `mindir` model inference, which requires setting the parameter to the option value corresponding to the model. Cloud-side inference support for `ms` model will be removed in future iterations, and cloud-side inference via `mindir` model is recommended.
    - `context`: compile context.
    - `dec_key`: define the key used to decrypt the ciphertext model. The key length is 16, 24, or 32.
    - `dec_mode`: define the decryption mode. Options: AES-GCM, AES-CBC.

- Returns

  Whether the build is successful.

```java
public boolean build(String modelPath, int modelType, MSContext context)
```

Compile MindSpore model by computational graph file,no decrypt.

- Parameters

    - `modelPath`: computational graph file.
    - `modelType`: computational graph type. Optionally, there are `MT_MINDIR_LITE`, `MT_MINDIR`, corresponding to the `ms` model (exported by the `converter_lite` tool) and the `mindir` model (exported by MindSpore or exported by the `converter_lite` tool), respectively. Only MT_MINDIR_LITE is valid for Device-side Inference and the parameter value is ignored. Cloud-side inference supports `ms` and `mindir` model inference, which requires setting the parameter to the option value corresponding to the model. Cloud-side inference support for `ms` model will be removed in future iterations, and cloud-side inference via `mindir` model is recommended.
    - `context`: compile context.

- Returns

  Whether the build is successful.

## predict

```java
public boolean predict()
```

Run predict.

- Returns

Whether the predict is successful.

## runStep

```java
public boolean runStep()
```

Run train by step.

- Returns

Whether the run is successful.

## resize

```java
public boolean resize(List<MSTensor> inputs, int[][] dims)
```

Resize inputs shape.

- Parameters

    - `inputs`: Model inputs.
    - `dims`: Define the new inputs shape.

- Returns

  Whether the resize is successful.

## getInputs

```java
public List<MSTensor> getInputs()
```

Get the input MSTensors of MindSpore model.

- Returns

  The MindSpore MSTensor list.

## getOutputs

```java
public List<MSTensor> getOutputs()
```

Get the output MSTensors of MindSpore model.

- Returns

  The MindSpore MSTensor list.

## getInputsByTensorName

```java
public MSTensor getInputsByTensorName(String tensorName)
```

Get the input MSTensors of MindSpore model by the node name.

- Parameters

    - `tensorName`: Define the tensor name.

- Returns

  MindSpore MSTensor.

## getOutputsByNodeName

```java
public List<MSTensor> getOutputsByNodeName(String nodeName)
```

Get the output MSTensors of MindSpore model by the node name.

- Parameters

    - `nodeName`: Define the node name.

- Returns

  The MindSpore MSTensor list.

## getOutputTensorNames

```java
public List<String> getOutputTensorNames()
```

Get output tensors names of the model compiled by this session.

- Returns

  The vector of string as output tensor names in order.

## getOutputByTensorName

```java
public MSTensor getOutputByTensorName(String tensorName)
```

Get the MSTensors output of MindSpore model by the tensor name.

- Parameters

    - `tensorName`: Define the tensor name.

- Returns

  MindSpore MSTensor.

## export

```java
public boolean export(String fileName, int quantizationType, boolean isOnlyExportInfer,List<String> outputTensorNames)
```

Export the model.

- Parameters

    - `fileName`: Model file name.
    - `quantization_type`: The quant type.
    - `isOnlyExportInfer`: Is only export infer.
    - `outputTensorNames`: The output tensor names for export.

- Returns

   Whether the export is successful.

## exportweightscollaboratewithmicro

```java
public boolean exportWeightsCollaborateWithMicro(String weightFile, boolean isInference,boolean enableFp16, List<String> changeableWeightNames)
```

Export the training model weights.

- Parameters

    - `weightFile`: Model weight file name.
    - `isInference`: Whether to export the weights of inference graph.
    - `enableFp16`:  Whether to save as float16 dtype.
    - `changeableWeightNames`: The weight tensor names whose shape is changeable.

- Returns

   Whether the export is successful.

## getFeatureMaps

```java
public List<MSTensor> getFeatureMaps()
```

Get the FeatureMap.

- Returns

    FeatureMaps tensor list.

## updatefeaturemaps

```java
public boolean updateFeatureMaps(List<MSTensor> features)
```

Update model Features.

- Parameters

    - `features`: New featureMaps tensor List.

- Returns

    Whether the model features is successfully update.

## settrainMode

```java
public boolean setTrainMode(boolean isTrain)
```

Set train mode.

- Parameters

    - `isTrain`: Is train mode.

## gettrainmode

```java
public boolean getTrainMode()
```

Get train mode.

- Returns

    Whether the model work in train mode.

## setLearningRate

```java
public boolean setLearningRate(float learning_rate)
```

set learning rate.

- Parameters

    - `learning_rate`: learning rate.

- Returns

    Whether the set learning rate is successful.

## setupVirtualBatch

```java
public boolean setupVirtualBatch(int virtualBatchMultiplier, float learningRate, float momentum)
```

Set the virtual batch.

- Parameters

    - `virtualBatchMultiplier`: virtual batch multuplier.
    - `learningRate`: learning rate.
    - `momentum`: monentum.

- Returns

    Whether the virtual batch is successfully set.

## free

```java
public void free()
```

Free Model.

## ModelType

```java
import com.mindspore.config.ModelType;
```

Model file type.

```java
public static final int MT_MINDIR = 0;
public static final int MT_AIR = 1;
public static final int MT_OM = 2;
public static final int MT_ONNX = 3;
public static final int MT_MINDIR_LITE = 4;
```
