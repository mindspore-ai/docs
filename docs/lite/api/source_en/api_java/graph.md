# Graph

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.6/docs/lite/api/source_en/api_java/graph.md)

```java
import com.mindspore.Graph;
```

Model defines computational graph in MindSpore.

## Public Member Functions

| function                                                     |
| ------------------------------------------------------------ |
| [boolean load(String file)](#load) |
| [long getGraphPtr()](#getgraphptr)                           |
| [void free()](#free)                                         |

## load

```java
 boolean load(String file)
```

Load the MindSpore model from file.

- Parameters

    - `File`: Model File.

- Returns

  Whether the load is successful.

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
