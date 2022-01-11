# Client

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/federated/docs/source_en/java_api_client.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

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
| [void setBatchSize(int batchSize)](#setbatchSize) |

## initCallbacks

```java
public abstract List<Callback> initCallbacks(RunType runType, DataSet dataSet)
```

Initialize the callback list.

- Parameters

    - `runType`: Define run phase.
    - `dataSet`: DataSet.

- Returns

  The initialized callback list.

## initDataSets

```java
public abstract Map<RunType, Integer> initDataSets(Map<RunType, List<String>> files)
```

Initialize dataset list.

- Parameters

    - `files`: Data files.

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
public abstract List<Integer> getInferResult(List<Callback> inferCallbacks)
```

Get infer phase result.

- Parameters

    - `inferCallbacks`: Callback used in infer phase.

- Returns

  predict results.

## initSessionAndInputs

```java
public Status initSessionAndInputs(String modelPath, MSConfig config)
```

Initialize client runtime session and input buffer.

- Parameters

    - `modelPath`: Model file path.
    - `config`: session config.

- Returns

    Whether the Initialization is successful.

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

## inferModel

```java
public List<Integer> inferModel()
```

Execute infer model process.

- Returns

  The infer result in infer process.

## saveModel

```java
public Status saveModel(String modelPath)
```

Save model.

- Returns

  Whether the inference is successful.

## getFeatures

```java
public List<MSTensor> getFeatures()
```

Get feature weights.

- Returns

  The feature weights of model.

## updateFeatures

```java
public Status updateFeatures(String modelName, List<FeatureMap> featureMaps)
```

Update model feature weights.

- Parameters

    - `modelName`: Model file name.
    - `featureMaps`: New model weights.

- Returns

  Whether the update is successful.

## free

```java
public void free()
```

free model.

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
