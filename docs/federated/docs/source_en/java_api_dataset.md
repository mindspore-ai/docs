# DataSet

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/federated/docs/source_en/java_api_dataset.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

```java
import com.mindspore.flclient.model.DataSet
```

DataSet defines end-side federated learning data set object.

## Public Member Functions

| function                    |
| -------------------------------- |
| [abstract void fillInputBuffer(List<ByteBuffer\> var1, int var2)](#fillinputbuffer) |
| [abstract void shuffle()](#shuffle)    |
| [abstract void padding()](#padding) |
| [abstract Status dataPreprocess(List<String\> var1)](#datapreprocess) |

## fillInputBuffer

```java
public abstract void fillInputBuffer(List<ByteBuffer> var1, int var2)
```

Fill input buffer data.

- Parameters

    - `var1`: Need fill buffer.
    - `var2`: Need fill batch index.

## shuffle

```java
 public abstract void shuffle()
```

Shuffle data.

## padding

```java
 public abstract void padding()
```

Pad data.

## dataPreprocess

```java
public abstract Status dataPreprocess(List<String> var1)
```

Data preprocess.

- Parameters

    - `var1`: Data files.

- Returns

  Whether the execution is successful.
