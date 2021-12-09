# LiteSession

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_en/api_java/lite_session.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

```java
import com.mindspore.lite.LiteSession;
```

LiteSession defines session in MindSpore Lite for compiling Model and forwarding model.

## Public Member Functions

| function                                                     |
| ------------------------------------------------------------ |
| [boolean init(MSConfig config)](#init)                       |
| [static LiteSession createSession(final MappedByteBuffer buffer, final MSConfig config)](#createsession)  |
| [static LiteSession createSession(final MSConfig config)](#createsession)                                 |
| [long getSessionPtr()](#getsessionptr)                       |
| [void setSessionPtr(long sessionPtr)](#setsessionptr)        |
| [void bindThread(boolean if_bind)](#bindthread)              |
| [boolean compileGraph(Model model)](#compilegraph)           |
| [boolean runGraph()](#rungraph)                              |
| [List<MSTensor\> getInputs()](#getinputs)                    |
| [MSTensor getInputsByTensorName(String tensorName)](#getinputsbytensorname) |
| [List<MSTensor\> getOutputsByNodeName(String nodeName)](#getoutputsbynodename) |
| [Map<String, MSTensor\> getOutputMapByTensor()](#getoutputmapbytensor) |
| [List<String\> getOutputTensorNames()](#getoutputtensornames) |
| [MSTensor getOutputByTensorName(String tensorName)](#getoutputbytensorname) |
| [boolean resize(List<MSTensor\> inputs, int[][] dims)](#resize) |
| [void free()](#free)                                         |
| [boolean export(String modelFilename, int model_type, int quantization_type)](#export) |
| [boolean train()](#train) |
| [boolean eval()](#eval) |
| [boolean isTrain()](#isTrain) |
| [boolean isEval()](#isEval) |
| [boolean setLearningRate(float learning_rate)](#setLearningRate) |
| [boolean setupVirtualBatch(int virtualBatchMultiplier, float learningRate, float momentum)](#setupVirtualBatch)   |
| [List<MSTensor> getFeaturesMap()](#getFeaturesMap) |
| [boolean updateFeatures(List<MSTensor> features)](#updateFeatures) |

## init

```java
public boolean init(MSConfig config)
```

Initialize LiteSession.

- Parameters

    - `config`: MSConfig class.

- Returns

  Whether the initialization is successful.

## createSession

```java
public static LiteSession createSession(final MSConfig config)
```

Use MSConfig to create Litessesion.

- Parameters

    - `config`: MSConfig class.

- Returns

  Return LiteSession object.

```java
public static LiteSession createSession(final MappedByteBuffer buffer, final MSConfig config)
```

Use Model buffer and MSConfig to create Litessesion.

- Parameters

    - `buffer`: MappedByteBuffer class.
    - `config`: MSConfig class.

- Returns

  Return LiteSession object.

## getSessionPtr

```java
public long getSessionPtr()
```

- Returns

  Return session pointer.

## setSessionPtr

```java
public void setSessionPtr(long sessionPtr)
```

- Parameters

    - `sessionPtr`: session pointer.

## bindThread

```java
public void bindThread(boolean isBind)
```

Attempt to bind or unbind threads in the thread pool to or from the specified cpu core.

- Parameters

    - `isBind`: Define whether to bind or unbind threads.

## compileGraph

```java
public boolean compileGraph(Model model)
```

Compile MindSpore Lite model.

- Parameters

    - `Model`: Define the model to be compiled.

- Returns

  Whether the compilation is successful.

## runGraph

```java
public boolean runGraph()
```

Run the session for inference.

- Returns

  Whether the inference is successful.

## getInputs

```java
public List<MSTensor> getInputs()
```

Get the MSTensors input of MindSpore Lite model.

- Returns

  The vector of MindSpore Lite MSTensor.

## getInputsByTensorName

```java
public MSTensor getInputsByTensorName(String tensorName)
```

Get the MSTensors input of MindSpore Lite model by the node name.

- Parameters

    - `tensorName`: Define the tensor name.

- Returns

  MindSpore Lite MSTensor.

## getOutputsByNodeName

```java
public List<MSTensor> getOutputsByNodeName(String nodeName)
```

Get the MSTensors output of MindSpore Lite model by the node name.

- Parameters

    - `nodeName`: Define the node name.

- Returns

  The vector of MindSpore Lite MSTensor.

## getOutputMapByTensor

```java
public Map<String, MSTensor> getOutputMapByTensor()
```

Get the MSTensors output of the MindSpore Lite model associated with the tensor name.

- Returns

  The map of output tensor name and MindSpore Lite MSTensor.

## getOutputTensorNames

```java
public List<String> getOutputTensorNames()
```

Get the name of output tensors of the model compiled by this session.

- Returns

  The vector of string as output tensor names in order.

## getOutputByTensorName

```java
public MSTensor getOutputByTensorName(String tensorName)
```

Get the MSTensors output of MindSpore Lite model by the tensor name.

- Parameters

    - `tensorName`: Define the tensor name.

- Returns

  Pointer of MindSpore Lite MSTensor.

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

## free

```java
public void free()
```

Free LiteSession.

## export

```java
public boolean export(String modelFilename, int model_type, int quantization_type)
```

Export the model.

- Parameters

    - `modelFilename`: Model file name.
    - `model_type`: Train or Inference type.
    - `quantization_type`: The quant type.

- Returns

   Whether the export is successful.

## train

```java
public void train()
```

Switch to the train mode

## eval

```java
public void eval()
```

Switch to the eval mode.

## istrain

```java
public void isTrain()
```

It is Train mode.

## iseval

```java
public void isEval()
```

It is Eval mode.

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

## getFeaturesMap

```java
public List<MSTensor> getFeaturesMap()
```

Get the FeatureMap.

- Returns

    FeaturesMap Tensor list.

## updateFeatures

```java
public boolean updateFeatures(List<MSTensor> features)
```

Update model Features.

- Parameters

    - `features`: new FeatureMap Tensor List.

- Returns

    Whether the model features is successfully update.
