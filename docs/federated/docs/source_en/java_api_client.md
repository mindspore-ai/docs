# Client

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_en/java_api_client.md)

```java
import com.mindspore.flclient.model.Client
```

Client defines the execution process object of the end-side federated learning algorithm.

## Public Member Functions

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

Initialize the callback list.

- Parameters

    - `runType`: RunType class, identify whether the training, evaluation or prediction phase.
    - `dataSet`: DataSet class, identify whether the training, evaluation or prediction phase datasets.

- Returns

  The initialized callback list.

## initDataSets

```java
public abstract Map<RunType, Integer> initDataSets(Map<RunType, List<String>> files)
```

Initialize dataset list.

- Parameters

    - `files`: Data files used in the training, evaluation or prediction phase.

- Returns

  Data counts in different run type.

## getEvalAccuracy

```java
public abstract float getEvalAccuracy(List<Callback> evalCallbacks)
```

Get eval model accuracy.

- Parameters

    - `evalCallbacks`: Callback used in eval phase.

- Returns

   The accuracy in eval phase.

## getInferResult

```java
public abstract List<Object> getInferResult(List<Callback> inferCallbacks)
```

Get infer phase result.

- Parameters

    - `inferCallbacks`: Callback used in prediction phase.

- Returns

  predict results.

## trainModel

```java
public Status trainModel(int epochs)
```

Execute train model process.

- Parameters

    - `epochs`: Epoch num used in train process.

- Returns

  Whether the train model is successful.

## evalModel

```java
public float evalModel()
```

Execute eval model process.

- Returns

  The accuracy in eval process.

## genUnsupervisedEvalData

```java
public Map<String, float[]> genUnsupervisedEvalData(List<Callback> evalCallbacks)
```

Generate unsupervised training evaluation data, and the subclass needs to rewrite this function.

- Parameters

    - `evalCallbacks`: the eval Callback that generates data.

- Returns

  unsupervised training evaluation data

## inferModel

```java
public List<Object> inferModel()
```

Execute model prediction process.

- Returns

  The prediction result.

## setLearningRate

```java
public Status setLearningRate(float lr)
```

Set learning rate.

- Parameters

    - `lr`: Learning rate.

- Returns

  Whether the set is successful.

## setBatchSize

```java
public void setBatchSize(int batchSize)
```

Set batch size.

- Parameters

    - `batchSize`: batch size.
