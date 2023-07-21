# Graph

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.7/docs/lite/api/source_zh_cn/api_java/graph.md)

```java
import com.mindspore.Graph;
```

Graph定义了MindSpore的计算图。

## 公有成员函数

| function                                                     |
| ------------------------------------------------------------ |
| [boolean load(String file)](#load) |
| [long getGraphPtr()](#getgraphptr)                            |
| [void free()](#free)                                         |

## load

```java
 boolean load(String file)
```

从指定文件加载MindSpore模型。

- 参数

    - `file`: 模型文件名。

- 返回值

  是否加载成功。

## getGraphPtr

```java
public long getGraphPtr()
```

获取底层计算图指针。

- 返回值

  底层计算图指针。

## free

```java
public void free()
```

释放计算图内存。
