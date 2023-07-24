# LiteSession

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/api_java/source_en/lite_session.md)

```java
import com.mindspore.lite.LiteSession;
```

LiteSession defines session in MindSpore Lite for compiling Model and forwarding model.

## Public Member Functions

| function                                                     |
| ------------------------------------------------------------ |
| [boolean init(MSConfig config)](#init)                       |
| [void bindThread(boolean if_bind)](#bindthread)              |
| [boolean compileGraph(Model model)](#compilegraph)           |
| [boolean runGraph()](#rungraph)                              |
| [List<MSTensor\> getInputs()](#getinputs)                    |
| [MSTensor getInputsByTensorName(String tensorName)](#getinputsbytensorname) |
| [List<MSTensor\> getOutputsByNodeName(String nodeName)](#getoutputsbynodename) |
| [Map<String, MSTensor\> getOutputMapByTensor()](#getoutputmapbytensor) |
| [List<String\> getOutputTensorNames()](#getoutputtensornames) |
| [MSTensor getOutputByTensorName(String tensorName)](#getoutputbytensorname) |
| [boolean resize(List<MSTensor\> inputs, int[][] dims](#resize) |
| [void free()](#free)                                         |

## init

```java
public boolean init(MSConfig config)
```

Initialize LiteSession.

- Parameters

    - `MSConfig`: MSConfig class.

- Returns

  Whether the initialization is successful.

## bindThread

```java
public void bindThread(boolean if_bind)
```

Attempt to bind or unbind threads in the thread pool to or from the specified cpu core.

- Parameters
    - `if_bind`: Define whether to bind or unbind threads.

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
