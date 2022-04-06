# DataSet

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/java_api_dataset.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

```java
import com.mindspore.flclient.model.DataSet
```

DataSet定义了端侧联邦学习数据集对象。

## 公有成员函数

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

填充输入buffer数据。

- 参数

    - `var1`: 需要填充的buffer内存。
    - `var2`: 需要填充的batch索引。

## shuffle

```java
 public abstract void shuffle()
```

打乱数据。

## padding

```java
 public abstract void padding()
```

补齐数据。

## dataPreprocess

```java
public abstract Status dataPreprocess(List<String> var1)
```

数据前处理。

- 参数

    - `var1`: 使用的训练、评估或推理数据集。

- 返回值

  数据处理结果。
