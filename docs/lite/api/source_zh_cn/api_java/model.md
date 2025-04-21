# Model

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_java/model.md)

```java
import com.mindspore.Model;
```

Model定义了MindSpore中编译和运行的模型。

## 公有成员函数

| function                                                     | 云侧推理是否支持 | 端侧推理是否支持 |
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

通过模型计算图编译MindSpore模型。

- 参数

    - `graph`: 模型计算图。
    - `context`: 编译运行上下文。
    - `cfg`: 训练配置。

- 返回值

  是否编译成功。

```java
public boolean build(MappedByteBuffer buffer, int modelType, MSContext context, char[] dec_key, String dec_mode)
```

通过模型计算图内存块编译MindSpore模型。

- 参数

    - `buffer`: 模型计算图内存块。
    - `modelType`: 模型计算图类型，可选有`MT_MINDIR_LITE`、`MT_MINDIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。端侧推理只支持`ms`模型推理，该入参值被忽略。云端推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `context`: 运行时Context上下文。
    - `dec_key`: 模型解密秘钥。
    - `dec_mode`: 模型解密算法，可选AES-GCM、AES-CBC。

- 返回值

  是否编译成功。

```java
public boolean build(final MappedByteBuffer buffer, int modelType, MSContext context)
```

通过模型计算图内存块编译MindSpore模型。

- 参数

    - `buffer`: 模型计算图内存块。
    - `modelType`: 模型计算图类型，可选有`MT_MINDIR_LITE`、`MT_MINDIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。端侧推理只支持`ms`模型推理，该入参值被忽略。云端推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `context`: 运行时Context上下文。

- 返回值

  是否编译成功。

```java
public boolean build(String modelPath, int modelType, MSContext context, char[] dec_key, String dec_mode)
```

通过模型计算图文件编译MindSpore MindIR模型。

- 参数

    - `modelPath`: 模型计算图文件。
    - `modelType`: 模型计算图类型，可选有`MT_MINDIR_LITE`、`MT_MINDIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。端侧推理只支持`ms`模型推理，该入参值被忽略。云端推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `context`: 运行时Context上下文。
    - `dec_key`: 模型解密秘钥。
    - `dec_mode`: 模型解密算法，可选AES-GCM、AES-CBC。

- 返回值

  是否编译成功。

```java
public boolean build(String modelPath, int modelType, MSContext context)
```

通过模型计算图文件编译MindSpore MindIR模型。

- 参数

    - `modelPath`: 模型计算图文件。
    - `modelType`: 模型计算图类型，可选有`MT_MINDIR_LITE`、`MT_MINDIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。端侧推理只支持`ms`模型推理，该入参值被忽略。云端推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `context`: 运行时Context上下文。

- 返回值

  是否编译成功。

## predict

```java
public boolean predict()
```

执行推理。

- 返回值

  是否推理成功。

## runStep

```java
public boolean runStep()
```

执行单步训练。

- 返回值

  是否单步训练成功。

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

## getInputs

```java
public List<MSTensor> getInputs()
```

获取MindSpore模型的输入tensor列表。

- 返回值

  所有输入MSTensor组成的List。

## getOutputs

```java
public List<MSTensor> getOutputs()
```

获取MindSpore模型的输出tensor列表。

- 返回值

  所有输出MSTensor组成的List。

## getInputsByTensorName

```java
public MSTensor getInputsByTensorName(String tensorName)
```

通过张量名获取MindSpore模型的输入张量。

- 参数

    - `tensorName`: 张量名。

- 返回值

  tensorName所对应的输入MSTensor。

## getOutputByTensorName

```java
public MSTensor getOutputByTensorName(String tensorName)
```

通过张量名获取MindSpore模型的输出张量。

- 参数

    - `tensorName`: 张量名。

- 返回值

  该张量所对应的MSTensor。

## getOutputsByNodeName

```java
public List<MSTensor> getOutputsByNodeName(String nodeName)
```

通过节点名获取MindSpore模型的MSTensors输出。

- 参数

    - `nodeName`: 节点名。

- 返回值

  该节点所有输出MSTensor组成的List。

## getOutputTensorNames

```java
public List<String> getOutputTensorNames()
```

获取由当前会话所编译的模型的输出张量名。

- 返回值

  按顺序排列的输出张量名组成的List。

## export

```java
public boolean export(String fileName, int quantizationType, boolean isOnlyExportInfer,List<String> outputTensorNames)
```

导出模型。

- 参数

    - `fileName`: 模型文件名称。
    - `quantizationType`: 量化类型。可选不量化，权重量化。
    - `isOnlyExportInfer`: 是否只导推理图。
    - `outputTensorNames`: 指定导出图结尾的tensor名称。

- 返回值

  导出模型是否成功。

## exportweightscollaboratewithmicro

```java
public boolean exportWeightsCollaborateWithMicro(String weightFile, boolean isInference,boolean enableFp16, List<String> changeableWeightNames)
```

导出训练模型权重。

- 参数

    - `weightFile`: 模型权重文件路径名称。
    - `isInference`: 是否导出推理图权重。当前只支持true。
    - `enableFp16`: 是否权重保存fp16。
    - `changeableWeightNames`: 可变shape的权重tensor名称。

- 返回值

  导出模型权重是否成功。

## getFeatureMaps

```java
public List<MSTensor> getFeatureMaps()
```

获取权重参数。

- 返回值

  权重参数列表。

## updateFeatureMaps

```java
public boolean updateFeatureMaps(List<MSTensor> features)
```

更新权重参数。

- 参数

    - `features`: 新的权重参数列表。

- 返回值

  权重是否更新成功。

## setTrainMode

```java
public boolean setTrainMode(boolean isTrain)
```

设置训练或推理模式。

- 参数

    - `isTrain`: 是否训练。

- 返回值

  运行模式是否设置成功。

## getTrainMode

```java
public boolean getTrainMode()
```

获取训练模式。

- 返回值

  是否是训练模式。

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

    - `virtualBatchMultiplier`: 虚批次系数，实际批次数需要乘以此系数。
    - `learningRate`: 学习率。
    - `momentum`: 动量系数。

- 返回值

  虚批次系数设置是否成功。

## free

```java
public void free()
```

释放Model内存。

## ModelType

```java
import com.mindspore.config.ModelType;
```

模型文件类型。

```java
public static final int MT_MINDIR = 0;
public static final int MT_AIR = 1;
public static final int MT_OM = 2;
public static final int MT_ONNX = 3;
public static final int MT_MINDIR_LITE = 4;
```
