# Client

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/java_api_client.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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
| [abstract List<Integer\> getInferResult(List<Callback\> inferCallbacks)](#getinferresult) |
| [Status initSessionAndInputs(String modelPath, MSConfig config)](#initsessionandinputs) |
| [Status trainModel(int epochs)](#trainmodel) |
| [evalModel()](#evalmodel) |
| [List<Integer\> inferModel()](#infermodel) |
| [Status saveModel(String modelPath)](#savemodel) |
| [List<MSTensor\> getFeatures()](#getfeatures) |
| [Status updateFeatures(String modelName, List<FeatureMap\> featureMaps)](#updatefeatures) |
| [void free()](#free) |
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
public abstract List<Integer> getInferResult(List<Callback> inferCallbacks)
```

获取预测结果。

- 参数

    - `inferCallbacks`: 预测阶段使用的callback列表。

- 返回值

  预测结果。

## initSessionAndInputs

```java
public Status initSessionAndInputs(String modelPath, MSConfig config)
```

初始化client底层会话和输入。

- 参数

    - `modelPath`: 模型文件。
    - `config`: 会话配置。

- 返回值

  初始化状态结果。

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

## inferModel

```java
public List<Integer> inferModel()
```

执行模型预测过程。

- 返回值

  模型预测结果。

## saveModel

```java
public Status saveModel(String modelPath)
```

保存模型。

- 返回值

  模型保存结果。

## getFeatures

```java
public List<MSTensor> getFeatures()
```

获取端侧权重。

- 返回值

  模型权重。

## updateFeatures

```java
public Status updateFeatures(String modelName, List<FeatureMap> featureMaps)
```

更新端侧权重。

- 参数

    - `modelName`: 待更新的模型文件。
    - `featureMaps`: 待更新的模型权重。

- 返回值

  模型权重。

## free

```java
public void free()
```

释放模型。

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
