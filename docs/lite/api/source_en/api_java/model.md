# Model

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.5/docs/lite/api/source_en/api_java/model.md)

```java
import com.mindspore.lite.Model;
```

Model defines model in MindSpore Lite for managing graph.

## Public Member Functions

| function                                                     |
| ------------------------------------------------------------ |
| [boolean loadModel(Context context, String modelName)](#loadmodel) |
| [boolean loadModel(String modelPath)](#loadmodel)           |
| [void freeBuffer()](#freebuffer)                            |
| [void free()](#free)                                         |

## loadModel

```java
public boolean loadModel(Context context, String modelName)
```

Load the MindSpore Lite model from Assets.

- Parameters

    - `context`: Context in Android.
    - `modelName`: Model file name.

- Returns

  Whether the load is successful.

```java
public boolean loadModel(String modelPath)
```

Load the MindSpore Lite model from path.

- Parameters

    - `modelPath`: Model file path.

- Returns

  Whether the load is successful.

## freeBuffer

```java
public void freeBuffer()
```

Free MetaGraph in MindSpore Lite Model to reduce memory usage during inference.

## free

```java
public void free()
```

Free all temporary memory in MindSpore Lite Model.
