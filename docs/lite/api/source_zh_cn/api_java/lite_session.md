# LiteSession

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/lite/api/source_zh_cn/api_java/lite_session.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

```java
import com.mindspore.lite.LiteSession;
```

LiteSession定义了MindSpore Lite中的会话，用于进行Model的编译和前向推理。

## 公有成员函数

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

初始化LiteSession。

- 参数

    - `config`: MSConfig类。

- 返回值

  初始化是否成功。

## createSession

```java
public static LiteSession createSession(final MSConfig config)
```

创建LiteSession。

- 参数

    - `config`: MSConfig类。

- 返回值

  返回创建的LiteSession。

```java
public static LiteSession createSession(final MappedByteBuffer buffer, final MSConfig config)
```

创建LiteSession。

- 参数

    - `buffer`: MappedByteBuffer类。
    - `config`: MSConfig类。

- 返回值

  返回创建的LiteSession。

## getSessionPtr

```java
public long getSessionPtr()
```

- 返回值

  返回session指针。

## setSessionPtr

```java
public void setSessionPtr(long sessionPtr)
```

- 参数

    - `sessionPtr`: session指针。

## bindThread

```java
public void bindThread(boolean isBind)
```

尝试将线程池中的线程绑定到指定的CPU内核，或从指定的CPU内核进行解绑。

- 参数

    - `isBind`: 是否对线程进行绑定或解绑。

## compileGraph

```java
public boolean compileGraph(Model model)
```

编译MindSpore Lite模型。

- 参数

    - `Model`: 需要被编译的模型。

- 返回值

  编译是否成功。

## runGraph

```java
public boolean runGraph()
```

运行图进行推理。

- 返回值

  推理是否成功。

## getInputs

```java
public List<MSTensor> getInputs()
```

获取MindSpore Lite模型的MSTensors输入。

- 返回值

  所有输入MSTensor组成的List。

## getInputsByTensorName

```java
public MSTensor getInputByTensorName(String tensorName)
```

通过节点名获取MindSpore Lite模型的MSTensors输入。

- 参数

    - `tensorName`: 张量名。

- 返回值

  tensorName所对应的输入MSTensor。

## getOutputsByNodeName

```java
public List<MSTensor> getOutputsByNodeName(String nodeName)
```

通过节点名获取MindSpore Lite模型的MSTensors输出。

- 参数

    - `nodeName`: 节点名。

- 返回值

  该节点所有输出MSTensor组成的List。

## getOutputMapByTensor

```java
public Map<String, MSTensor> getOutputMapByTensor()
```

获取与张量名相关联的MindSpore Lite模型的MSTensors输出。

- 返回值

  输出张量名和MSTensor的组成的Map。

## getOutputTensorNames

```java
public List<String> getOutputTensorNames()
```

获取由当前会话所编译的模型的输出张量名。

- 返回值

  按顺序排列的输出张量名组成的List。

## getOutputByTensorName

```java
public MSTensor getOutputByTensorName(String tensorName)
```

通过张量名获取MindSpore Lite模型的MSTensors输出。

- 参数

    - `tensorName`: 张量名。

- 返回值

  该张量所对应的MSTensor。

## resize

```java
public boolean resize(List<MSTensor> inputs, int[][] dims)
```

调整输入的形状。

- 参数

    - `inputs`: 模型对应的所有输入。
    - `dims`: 输入对应的新的shape，顺序注意要与inputs一致。

- 返回值

  调整输入形状是否成功。

## free

```java
public void free()
```

释放LiteSession。

## export

```java
public boolean export(String modelFilename, int model_type, int quantization_type)
```

导出模型。

- 参数

    - `modelFilename`: 模型文件名称。
    - `model_type`: 训练或者推理类型。
    - `quantization_type`: 量化类型。

- 返回值

  导出模型是否成功。

## train

```java
public void train()
```

切换训练模式。

## eval

```java
public void eval()
```

切换推理模式。

## istrain

```java
public void isTrain()
```

是否训练模式。

## iseval

```java
public void isEval()
```

是否推理模式。

## setLearningRate

```java
public boolean setLearningRate(float learning_rate)
```

设置学习率。

- 参数

    - `learning_rate`: 学习率。

- 返回值

  学习率设置是否成功。

## setupVirtualBatch

```java
public boolean setupVirtualBatch(int virtualBatchMultiplier, float learningRate, float momentum)
```

设置虚批次系数。

- 参数

    - `virtualBatchMultiplier`: 虚批次系数。
    - `learningRate`: 学习率。
    - `momentum`: 动量系数。

- 返回值

  虚批次系数设置是否成功。  

## getFeaturesMap

```java
public List<MSTensor> getFeaturesMap()
```

获取权重参数。

- 返回值

  权重参数列表。

## updateFeatures

```java
public boolean updateFeatures(List<MSTensor> features)
```

更新权重参数。

- 参数

    - `features`: 新的权重参数列表。

- 返回值

  权重是否更新成功。
