# Graph

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/lite/api/source_en/api_java/graph.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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
