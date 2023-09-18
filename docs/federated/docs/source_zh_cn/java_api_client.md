# Client

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/java_api_client.md)

```java
import com.mindspore.flclient.model.Client
```

Client定义了端侧联邦学习算法执行流程对象。

## 公有成员函数

| function                   |
| -------------------------------- |
| [abstract List<Callback\> initCallbacks(RunType runType, DataSet dataSet)](#initcallbacks) |
| [abstract Map<RunType, Integer\> initDataSets(Map<RunType, List<String\>\> files)](#initdatasets)    |
| [abstract float getEvalAccuracy(List<Callback\> evalCallbacks)](#getevalaccuracy) |
| [abstract List<Object\> getInferResult(List<Callback\> inferCallbacks)](#getinferresult) |
| [Status trainModel(int epochs)](#trainmodel) |
| [float evalModel()](#evalmodel) |
| [Map<String, float[]\> genUnsupervisedEvalData(List<Callback\> evalCallbacks)](#genunsupervisedevaldata) |
| [List<Object\> inferModel()](#infermodel) |
| [Status setLearningRate(float lr)](#setlearningrate) |
| [void setBatchSize(int batchSize)](#setbatchsize) |

## initCallbacks

```java
public abstract List<Callback> initCallbacks(RunType runType, DataSet dataSet)
```

初始化callback列表。

- 参数

    - `runType`: RunType类，标识训练、评估还是预测阶段。
    - `dataSet`: DataSet类，训练、评估还是预测阶段数据集。

- 返回值

  初始化的callback列表。

## initDataSets

```java
public abstract Map<RunType, Integer> initDataSets(Map<RunType, List<String>> files)
```

初始化dataset列表。

- 参数

    - `files`: 训练、评估和预测阶段使用的数据文件。

- 返回值

  训练、评估和预测阶段数据集样本量。

## getEvalAccuracy

```java
public abstract float getEvalAccuracy(List<Callback> evalCallbacks)
```

获取评估阶段的精度。

- 参数

    - `evalCallbacks`: 评估阶段使用的callback列表。

- 返回值

  评估阶段精度。

## getInferResult

```java
public abstract List<Object> getInferResult(List<Callback> inferCallbacks)
```

获取预测结果。

- 参数

    - `inferCallbacks`: 预测阶段使用的callback列表。

- 返回值

  预测结果。

## trainModel

```java
public Status trainModel(int epochs)
```

开启模型训练。

- 参数

    - `epochs`: 训练的epoch数。

- 返回值

  模型训练结果。

## evalModel

```java
public float evalModel()
```

执行模型评估过程。

- 返回值

  模型评估精度。

## genUnsupervisedEvalData

```java
public Map<String, float[]> genUnsupervisedEvalData(List<Callback> evalCallbacks)
```

生成无监督训练评估数据，子类需要覆写该函数。

- 参数

    - `evalCallbacks`: 推理回调类，该类生成数据。

- 返回值

  无监督训练评估数据。

## inferModel

```java
public List<Object> inferModel()
```

执行模型预测过程。

- 返回值

  模型预测结果。

## setLearningRate

```java
public Status setLearningRate(float lr)
```

设置学习率。

- 参数

    - `lr`: 学习率。

- 返回值

  设置结果。

## setBatchSize

```java
public void setBatchSize(int batchSize)
```

设置执行批次数。

- 参数

    - `batchSize`: 批次数。