# LiteSession

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/api_java/source_zh_cn/lite_session.md)

```java
import com.mindspore.lite.LiteSession;
```

LiteSession定义了MindSpore Lite中的会话，用于进行Model的编译和前向推理。

## 公有成员函数

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

初始化LiteSession。

- 参数

    - `MSConfig`: MSConfig类。

- 返回值

  初始化是否成功。

## bindThread

```java
public void bindThread(boolean if_bind)
```

尝试将线程池中的线程绑定到指定的CPU内核，或从指定的CPU内核进行解绑。

- 参数

    - `if_bind`: 是否对线程进行绑定或解绑。

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
