# Graph

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_en/api_java/graph.md)

```java
import com.mindspore.Graph;
```

Model defines computational graph in MindSpore.

## Public Member Functions

| function                                                     | Supported At Cloud-side Inference | Supported At Device-side Inference |
| ------------------------------------------------------------ |--------|--------|
| [boolean load(String file)](#load) | √      | √      |
| [long getGraphPtr()](#getgraphptr)                            | √      | √      |
| [void free()](#free)                                         | √      | √      |

## load

```java
 boolean load(String file)
```

Load the MindSpore model from file.

- Parameters

    - `File`: Model File.

- Returns

  Whether the load is successful.

## getGraphPtr

```java
public long getGraphPtr()
```

Get the MindSpore computational graph pointer.

- Returns

  The graph pointer.

## free

```java
public void free()
```

Free the computational graph memory.
